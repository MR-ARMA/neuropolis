from __future__ import annotations

import math
import random
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from src.core.contracts import SimulationConfig, TMSSpec, normalize_item_id


class EntropyTMSRouter:
    """
    M4: Semantic-Offset Entropy-Triggered Transactive Memory System (TMS)
    & Value-of-Information Experimental Comparator (Hypothesis H3).
    Supports all 5 Reviewer #2 C12 conditions:
      - `OFF`: Never queries an expert.
      - `ENTROPY_TRIGGER`: Queries when generated semantic token entropy H_bar > tau_TMS.
      - `RANDOM_TRIGGER`: Queries with matched probability p = 0.35.
      - `ALWAYS_ASK`: Queries on 100% of turns.
      - `ORACLE_TRIGGER`: Queries if and only if agent has high signal noise (is_ignorant == True).
    Also tracks turn-by-turn `(entropy, is_ignorant)` pairs to compute the exact
    Mann-Whitney U **AUROC** of the entropy probe against ground-truth epistemic ignorance.
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

        self.rng = random.Random(self.config.random_seed + 77)
        self.entropy_ignorance_records: List[Tuple[float, int]] = []

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

        clean_slice = torch.nan_to_num(
            probe_slice, nan=0.0, posinf=20.0, neginf=-20.0
        )
        k_top = min(32, int(clean_slice.size(-1)))
        top_logits, _ = torch.topk(clean_slice, k=k_top, dim=-1)
        probs = F.softmax(top_logits, dim=-1)
        max_ent = math.log(float(max(2, k_top)))

        token_entropies: List[float] = []
        for row_idx in range(probs.size(0)):
            row_p = probs[row_idx]
            ent = -torch.sum(row_p * torch.log(row_p.clamp(min=1e-12))).item()
            norm_ent = ent / max(max_ent, 1e-6)
            token_entropies.append(max(0.0, min(1.0, norm_ent)))

        if not token_entropies:
            return 0.0
        return round(float(sum(token_entropies) / len(token_entropies)), 4)

    @staticmethod
    def find_semantic_value_start_index(
        generated_token_strings: List[str],
    ) -> int:
        """
        Finds the first token index following a JSON value delimiter (`"thought": "`
        or `"internal_thought": "` or `"decision": "`) so entropy is measured on
        semantic tokens rather than opening braces.
        """
        joined = ""
        for idx, tok in enumerate(generated_token_strings):
            joined += tok
            for key_marker in ("thought", "decision", "action"):
                if key_marker in joined and ":" in joined.split(key_marker)[-1]:
                    return min(len(generated_token_strings) - 1, idx + 1)
        return min(len(generated_token_strings) - 1, 2) if len(generated_token_strings) > 2 else 0

    def evaluate_and_route(
        self,
        owner_id: str,
        domain_id: str,
        mean_entropy: float,
        effective_trust_map: Dict[str, float],
        is_ground_truth_ignorant: bool = False,
        tms_mode_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates whether to trigger `ASK_EXPERT` under the active `TMSMode`,
        logs `(mean_entropy, int(is_ground_truth_ignorant))` for AUROC calculation,
        and selects expert j* maximizing expected competence weighted by effective trust.
        """
        norm_domain = normalize_item_id(domain_id)
        self.entropy_ignorance_records.append(
            (float(mean_entropy), int(is_ground_truth_ignorant))
        )

        active_mode = (
            tms_mode_override or self.config.experimental_design.tms_mode
        )
        if not self.tms_spec.enabled or active_mode == "OFF":
            should_trigger = False
        elif active_mode == "ALWAYS_ASK":
            should_trigger = True
        elif active_mode == "RANDOM_TRIGGER":
            should_trigger = self.rng.random() < 0.35
        elif active_mode == "ORACLE_TRIGGER":
            should_trigger = bool(is_ground_truth_ignorant)
        else:
            # ENTROPY_TRIGGER (default)
            should_trigger = mean_entropy > self.tms_spec.entropy_threshold_tau

        if not should_trigger:
            return {
                "halt_and_route": False,
                "mean_entropy": round(mean_entropy, 4),
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
        """
        Updates Beta-Bernoulli priors (alpha, beta) based on whether the expert's
        private signal theta_tilde_{j*,t} matched the realized macro state theta_t.
        """
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

    def compute_entropy_ignorance_auroc(self) -> float:
        """
        Computes exact Mann-Whitney U AUROC measuring how well semantic token entropy
        discriminates true high-noise ignorance (label=1) from specialist competence (label=0).
        """
        pos = [e for e, lbl in self.entropy_ignorance_records if lbl == 1]
        neg = [e for e, lbl in self.entropy_ignorance_records if lbl == 0]
        if not pos or not neg:
            return 0.50

        concordant = 0.0
        total_pairs = float(len(pos) * len(neg))
        for p_val in pos:
            for n_val in neg:
                if p_val > n_val:
                    concordant += 1.0
                elif abs(p_val - n_val) <= 1e-6:
                    concordant += 0.5
        return round(concordant / max(1.0, total_pairs), 4)