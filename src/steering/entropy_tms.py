from __future__ import annotations

import math
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from src.core.contracts import SimulationConfig, TMSSpec, normalize_item_id


class EntropyTMSRouter:
    """
    M4: Semantic-Offset Entropy-Triggered Transactive Memory System (TMS)
    & Bayesian Conjugate Social Routing Engine.
    """

    def __init__(
        self,
        config: Any = None,
        db_path: str = "data/mvp_world_state.sqlite",
        **kwargs: Any,
    ) -> None:
        if isinstance(config, str) and isinstance(db_path, (SimulationConfig, TMSSpec)):
            config, db_path = db_path, config
        elif isinstance(config, str):
            db_path = config
            config = kwargs.get("config", SimulationConfig())

        if config is None:
            config = SimulationConfig()

        self.db_path = str(db_path)
        if isinstance(config, SimulationConfig):
            self.config = config
            self.tms_spec = config.tms_spec
        elif isinstance(config, TMSSpec):
            self.config = SimulationConfig(tms_spec=config)
            self.tms_spec = config
        else:
            self.config = SimulationConfig()
            self.tms_spec = self.config.tms_spec

    def compute_normalized_top_p_entropy(
        self,
        logits: torch.Tensor,
        semantic_start_idx: int = 0,
    ) -> float:
        if logits is None or logits.numel() == 0:
            return 0.0

        if logits.dim() == 3:
            logits_2d = logits[0]
        elif logits.dim() == 2:
            logits_2d = logits
        else:
            logits_2d = logits.unsqueeze(0)

        seq_len = logits_2d.size(0)
        start = min(max(0, semantic_start_idx), max(0, seq_len - 1))
        end = min(seq_len, start + self.tms_spec.probe_tokens_k)
        probe_slice = logits_2d[start:end].detach().float()

        probs = F.softmax(probe_slice, dim=-1)
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        top_p = self.tms_spec.top_p_truncation
        token_entropies: List[float] = []

        for row_idx in range(sorted_probs.size(0)):
            row_p = sorted_probs[row_idx]
            row_cum = cum_probs[row_idx]
            keep_mask = row_cum <= top_p
            cutoff_idx = int(keep_mask.sum().item())
            cutoff_idx = max(8, min(row_p.size(0), cutoff_idx + 1))

            active_p = row_p[:cutoff_idx]
            active_p = active_p / active_p.sum().clamp(min=1e-9)
            ent = -torch.sum(active_p * torch.log(active_p.clamp(min=1e-12))).item()
            max_ent = math.log(float(cutoff_idx))
            norm_ent = ent / max(max_ent, 1e-6)
            token_entropies.append(max(0.0, min(1.0, norm_ent)))

        if not token_entropies:
            return 0.0
        return round(float(sum(token_entropies) / len(token_entropies)), 4)

    @staticmethod
    def find_semantic_value_start_index(
        generated_token_strings: List[str],
    ) -> int:
        joined = ""
        for idx, tok in enumerate(generated_token_strings):
            joined += tok
            if "internal_thought" in joined and ":" in joined.split("internal_thought")[-1]:
                return min(len(generated_token_strings) - 1, idx + 2)
        return 0

    def evaluate_and_route(
        self,
        owner_id: str,
        domain_id: str,
        mean_entropy: float,
        effective_trust_map: Dict[str, float],
    ) -> Dict[str, Any]:
        norm_domain = normalize_item_id(domain_id)
        if not self.tms_spec.enabled or mean_entropy <= self.tms_spec.entropy_threshold_tau:
            return {
                "halt_and_route": False,
                "mean_entropy": mean_entropy,
                "selected_expert_id": None,
                "domain_id": norm_domain,
                "expected_competence": 0.0,
            }

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT target_id, alpha_prior, beta_prior
                FROM tms_directory
                WHERE owner_id = ? AND domain_id = ?;
                """,
                (owner_id, norm_domain),
            ).fetchall()

        best_expert: Optional[str] = None
        best_score = -1e9
        best_comp = 0.0

        for r in rows:
            tid = r["target_id"]
            if tid == owner_id:
                continue
            alpha_p = float(r["alpha_prior"])
            beta_p = float(r["beta_prior"])
            expected_d = alpha_p / max(alpha_p + beta_p, 1e-6)
            t_eff = float(effective_trust_map.get(tid, 0.50))
            score = expected_d * t_eff - self.tms_spec.communication_cost_xi
            if score > best_score:
                best_score = score
                best_expert = tid
                best_comp = expected_d

        return {
            "halt_and_route": best_expert is not None,
            "mean_entropy": round(mean_entropy, 4),
            "selected_expert_id": best_expert,
            "domain_id": norm_domain,
            "expected_competence": round(best_comp, 4),
        }

    def bayesian_conjugate_update(
        self,
        owner_id: str,
        expert_id: str,
        domain_id: str,
        accuracy_realization_y: float,
        tick: int = 0,
    ) -> Tuple[float, float, float]:
        norm_domain = normalize_item_id(domain_id)
        y = max(0.0, min(1.0, float(accuracy_realization_y)))

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT alpha_prior, beta_prior
                FROM tms_directory
                WHERE owner_id = ? AND target_id = ? AND domain_id = ?;
                """,
                (owner_id, expert_id, norm_domain),
            ).fetchone()
            if not row:
                new_alpha = 2.0 + y
                new_beta = 2.0 + (1.0 - y)
            else:
                new_alpha = round(float(row[0]) + y, 4)
                new_beta = round(float(row[1]) + (1.0 - y), 4)

            conn.execute(
                """
                INSERT OR REPLACE INTO tms_directory VALUES (?, ?, ?, ?, ?, ?);
                """,
                (owner_id, expert_id, norm_domain, new_alpha, new_beta, tick),
            )
            conn.commit()

        expected_d = new_alpha / max(new_alpha + new_beta, 1e-6)
        return (new_alpha, new_beta, round(expected_d, 4))