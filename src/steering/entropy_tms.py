"""
src/steering/entropy_tms.py
MODULE 4 (M4): Semantic-Offset Entropy Probe & Bayesian Transactive Memory System (TMS) Router.
Implements Audit Flaw 4 Resolution:
  1. Probes unconstrained vocabulary logits strictly AFTER the deterministic JSON prefix
     `{"internal_thought": "` for K_probe = 5 semantic tokens.
  2. Computes top-p truncated Normalized Shannon Entropy H_bar_i(c) in [0, 1].
  3. If H_bar_i(c) > tau_TMS, halts generation (rewinding KV-cache) and routes an
     `ASK_EXPERT` query to peer j* maximizing expected Bayesian competence weighted by trust.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from src.core.contracts import (
    AgentTurnOutput,
    ActionCategory,
    ConsentDecision,
    EconomicActionIntent,
    EntropyTMSSpec,
    ExpertiseDomain,
    RelationalEdge,
    TMSPeerCompetenceRecord,
    TMSQueryIntent,
    TMSRoutingDecision,
)
from src.memory.scoped_memory import ScopedAgentMemoryView


class EntropyTMSRouter:
    """
    Monitors epistemic uncertainty on semantic reasoning tokens and routes queries
    across the Bayesian Beta-Bernoulli Transactive Memory directory.
    """

    def __init__(self, spec: EntropyTMSSpec) -> None:
        self.spec = spec

    def compute_top_p_normalized_shannon_entropy(
        self,
        vocab_logits_list: List[torch.Tensor],
    ) -> float:
        """
        Given K_probe raw vocabulary logit tensors `z_m in R^{|V|}`:
          1. Computes full softmax distribution p_m(v).
          2. Truncates to the top-p cumulative probability support V_p (or top-64 tokens)
             and renormalizes p_tilde_m(v).
          3. Computes normalized Shannon entropy:
               H_bar = (1 / K_probe) * sum_{m=1}^{K_probe} [ - sum_{v in V_p} p_tilde_m(v) ln p_tilde_m(v) / ln |V_p| ]
        """
        if not vocab_logits_list:
            return 0.0

        entropies: List[float] = []
        top_p = max(0.50, min(0.999, self.spec.top_p_truncation))

        for logits in vocab_logits_list[: self.spec.probe_tokens_k]:
            flat = logits.detach().float().view(-1)
            probs = torch.softmax(flat, dim=-1)

            # Sort descending for top-p nucleus support
            sorted_probs, _ = torch.sort(probs, descending=True)
            # Cap support window at 128 tokens for numerical stability across large vocabularies
            k_cap = min(128, sorted_probs.numel())
            sorted_probs = sorted_probs[:k_cap]

            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            keep_mask = cum_probs <= top_p
            # Always keep at least the top 8 tokens so ln(|V_p|) is well-defined
            keep_mask[: min(8, k_cap)] = True

            support_probs = sorted_probs[keep_mask]
            support_size = int(support_probs.numel())
            if support_size <= 1:
                entropies.append(0.0)
                continue

            p_tilde = support_probs / torch.clamp(support_probs.sum(), min=1e-9)
            p_safe = torch.clamp(p_tilde, min=1e-12)
            shannon_h = -torch.sum(p_safe * torch.log(p_safe)).item()
            norm_h = shannon_h / math.log(float(support_size))
            entropies.append(max(0.0, min(1.0, norm_h)))

        return round(float(sum(entropies) / len(entropies)), 4)

    def select_optimal_expert(
        self,
        scoped_memory: ScopedAgentMemoryView,
        domain: ExpertiseDomain,
    ) -> Tuple[Optional[str], float]:
        """
        Selects peer j* maximizing:
            Score(i -> j, c) = D_i(j, c) * ((1 + tau_ij(t)) / 2) - xi_comm
        where D_i(j, c) = alpha_{ij,c} / (alpha_{ij,c} + beta_{ij,c}).
        """
        records: List[TMSPeerCompetenceRecord] = scoped_memory.get_tms_competence_directory(domain)
        edges: Dict[str, RelationalEdge] = scoped_memory.get_all_outgoing_edges()

        best_peer: Optional[str] = None
        best_score = -1e9

        for rec in records:
            if rec.target_id == scoped_memory.owner_id:
                continue
            edge = edges.get(
                rec.target_id,
                RelationalEdge(owner_id=scoped_memory.owner_id, target_id=rec.target_id),
            )
            trust_weight = 0.5 * (1.0 + edge.trust_tau)
            score = rec.expected_competence * trust_weight - self.spec.communication_cost_xi
            if score > best_score:
                best_score = score
                best_peer = rec.target_id

        return best_peer, round(best_score, 4)

    def evaluate_and_route(
        self,
        vocab_logits_list: List[torch.Tensor],
        scoped_memory: ScopedAgentMemoryView,
        domain: ExpertiseDomain,
    ) -> TMSRoutingDecision:
        """
        Evaluates semantic-offset token entropy and triggers `ASK_EXPERT` if H_bar > tau_TMS.
        """
        if not self.spec.enabled:
            return TMSRoutingDecision(halted_generation=False, normalized_entropy=0.0)

        h_bar = self.compute_top_p_normalized_shannon_entropy(vocab_logits_list)
        if h_bar <= self.spec.entropy_threshold_tau:
            return TMSRoutingDecision(
                halted_generation=False,
                normalized_entropy=h_bar,
                domain=domain,
            )

        expert_id, _ = self.select_optimal_expert(scoped_memory, domain)
        if expert_id is None:
            return TMSRoutingDecision(
                halted_generation=False,
                normalized_entropy=h_bar,
                domain=domain,
            )

        advice = (
            f"[Expert {expert_id} Advisory on {domain.value}]: Current fair clearing price is "
            f"stable near reference levels; avoid overpaying above 1.2x reference."
        )
        return TMSRoutingDecision(
            halted_generation=True,
            normalized_entropy=h_bar,
            selected_expert_id=expert_id,
            domain=domain,
            expert_advice_injected=advice,
        )

    @staticmethod
    def build_tms_query_turn_output(
        decision: TMSRoutingDecision,
        counterparty_id: Optional[str],
    ) -> AgentTurnOutput:
        """
        Constructs the deterministic `AgentTurnOutput` emitted when the KV-cache is
        rewound at semantic offset `m_0` due to high epistemic entropy.
        """
        expert_id = decision.selected_expert_id or (counterparty_id or "Agent_1")
        return AgentTurnOutput(
            internal_thought=(
                f"Epistemic uncertainty high (H_bar={decision.normalized_entropy:.3f} > tau); "
                f"halting speculative guess and routing TMS query to domain expert {expert_id}."
            ),
            epistemic_confidence=round(max(0.05, 1.0 - decision.normalized_entropy), 3),
            consent_decision=ConsentDecision.INITIATE,
            verbal_message=(
                f"@{expert_id}, I need your domain expertise on {decision.domain.value} "
                f"before locking in a trade price."
            ),
            perceived_honesty_delta=0.05,
            economic_intent=EconomicActionIntent(
                action_category=ActionCategory.TMS_QUERY,
                target_peer_id=expert_id,
                item_id="NONE",
                quantity=0,
                total_price_cash=0.0,
                is_buying=True,
            ),
            tms_query=TMSQueryIntent(
                queried_expert_id=expert_id,
                domain=decision.domain,
                question_text=f"What is your assessment of {decision.domain.value} supply and fair price?",
                measured_entropy_bar=decision.normalized_entropy,
            ),
        )