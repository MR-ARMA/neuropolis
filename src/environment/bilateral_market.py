"""
src/environment/bilateral_market.py
MODULE 9 (M9): Environment Institutions:
  1. FULLY IMPLEMENTED FOR MVP:
     - `BilateralTrustMarket`: Relationship-modulated reservation prices,
       Multinomial Logit (MNL) partner selection over trust/complementarity,
       and Slovic Asymmetric Trust Updating (eta_- = 2.5 * eta_+).
  2. PRESERVED POST-MVP ARCHITECTURAL STUBS:
     - `CentralAndCommercialBankModule` (Diamond-Dybvig fractional reserves & bank runs).
     - `PublicGoodsAndLawCommonsModule` (Ostrom public goods multiplier & costly punishment).
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

from src.core.contracts import (
    AgentTurnOutput,
    ConsentDecision,
    EconomyInstitutionsSpec,
    ExternalAgentState,
    GameMasterVerdict,
    InternalAgentState,
    MemoryCognitionSpec,
    RelationalEdge,
    RelationshipType,
    VerdictStatus,
)
from src.memory.scoped_memory import ScopedMemoryEngine


class BilateralTrustMarket:
    """
    Governs partner matching, relationship-modulated reservation prices,
    and post-interaction Slovic asymmetric trust & ToM updating.
    """

    def __init__(
        self,
        memory_engine: ScopedMemoryEngine,
        econ_spec: EconomyInstitutionsSpec,
        cog_spec: MemoryCognitionSpec,
        rng: random.Random,
    ) -> None:
        self.memory_engine = memory_engine
        self.econ_spec = econ_spec
        self.cog_spec = cog_spec
        self.rng = rng

    def compute_reservation_price(
        self,
        item_id: str,
        edge: RelationalEdge,
        is_seller: bool,
    ) -> float:
        """
        Relationship-Modulated Reservation Price:
          - Friends (intimacy_alpha > 0, trust_tau > 0) grant a relational discount.
          - Distrusted peers (trust_tau < 0) incur a default/lemons risk premium.
        """
        base = (
            self.econ_spec.reference_food_price
            if item_id == "FOOD"
            else self.econ_spec.reference_tech_price
        )
        trust_term = edge.trust_tau
        affinity_term = edge.intimacy_alpha

        if is_seller:
            # Seller demands higher price from distrusted buyer, offers small discount to friends
            multiplier = 1.0 - 0.12 * affinity_term - 0.20 * min(0.0, trust_term)
        else:
            # Buyer is willing to pay less when distrusting seller's quality
            multiplier = 1.0 + 0.10 * affinity_term + 0.25 * min(0.0, trust_term)

        return round(max(2.0, base * multiplier), 2)

    def select_interaction_pairs_mnl(
        self,
        agent_ids: List[str],
    ) -> List[Tuple[str, str]]:
        """
        Multinomial Logit (MNL) Partner Selection based on:
          U_attract(i -> j) = 1.6 * tau_ij + 0.9 * alpha_ij + 1.2 * Surplus_complementarity(i, j)
        Produces a list of directed (initiator_id, responder_id) requests for the tick.
        """
        external_map = self.memory_engine.get_all_external_states()
        requests: List[Tuple[str, str]] = []

        shuffled = list(agent_ids)
        self.rng.shuffle(shuffled)

        for aid in shuffled:
            scoped = self.memory_engine.get_scoped_view(aid)
            edges = scoped.get_all_outgoing_edges()
            my_ext = external_map[aid]

            candidates: List[str] = [pid for pid in agent_ids if pid != aid]
            if not candidates:
                continue

            logits: List[float] = []
            for pid in candidates:
                edge = edges.get(pid, RelationalEdge(owner_id=aid, target_id=pid))
                peer_ext = external_map[pid]

                # Economic resource complementarity: high if one has FOOD surplus and the other has TECH/cash
                my_food = my_ext.inventory.get("FOOD", 0)
                peer_food = peer_ext.inventory.get("FOOD", 0)
                complementarity = 0.45 if abs(my_food - peer_food) >= 1 else 0.10

                u_ij = (
                    1.6 * edge.trust_tau
                    + 0.9 * edge.intimacy_alpha
                    + 1.2 * complementarity
                )
                logits.append(u_ij)

            max_l = max(logits)
            exps = [math.exp(l - max_l) for l in logits]
            total_exp = sum(exps)
            probs = [e / total_exp for e in exps]

            chosen_peer = self.rng.choices(candidates, weights=probs, k=1)[0]
            requests.append((aid, chosen_peer))

        return requests

    def update_dyadic_trust_and_memories(
        self,
        tick: int,
        initiator_id: str,
        responder_id: str,
        initiator_output: AgentTurnOutput,
        responder_output: AgentTurnOutput,
        verdict: GameMasterVerdict,
    ) -> None:
        """
        Applies Slovic's Asymmetric Trust Principle:
          - Cooperative settlement: tau_ij(t+1) = tau_ij(t) + eta_+ * (1 - tau_ij(t)) * coop_signal
          - Betrayal / unfair rejection: tau_ij(t+1) = tau_ij(t) - eta_- * (1 + tau_ij(t)) * betrayal_signal
            where eta_- = 2.5 * eta_+ (trust is slow to build and rapid to collapse).
        Also writes scoped episodic memories into both agents' SQLite partitions.
        """
        eta_pos = self.cog_spec.slovic_pos_trust_rate
        eta_neg = self.cog_spec.slovic_neg_trust_rate

        approved = verdict.status == VerdictStatus.APPROVED
        fair_ref = (
            self.econ_spec.reference_food_price
            if verdict.item_id == "FOOD"
            else self.econ_spec.reference_tech_price
        ) * max(1, verdict.quantity_transferred)

        offered_price = initiator_output.economic_intent.total_price_cash
        price_gouge_ratio = (offered_price - fair_ref) / max(1.0, fair_ref)

        for owner_id, peer_id, my_out, peer_out in [
            (initiator_id, responder_id, initiator_output, responder_output),
            (responder_id, initiator_id, responder_output, initiator_output),
        ]:
            scoped = self.memory_engine.get_scoped_view(owner_id)
            ist = scoped.get_internal_state()
            edge = scoped.get_relational_edge(peer_id)
            pmodel = scoped.get_peer_perceived_model(peer_id)

            edge.total_interactions += 1

            if approved:
                edge.successful_trades += 1
                coop_signal = max(0.25, min(1.0, 0.65 + 0.35 * my_out.perceived_honesty_delta))
                edge.trust_tau = min(
                    1.0,
                    round(edge.trust_tau + eta_pos * (1.0 - edge.trust_tau) * coop_signal, 4),
                )
                edge.intimacy_alpha = min(1.0, round(edge.intimacy_alpha + 0.05, 4))
                edge.tie_strength_w = min(1.0, round(edge.tie_strength_w + 0.06, 4))
                if edge.trust_tau > 0.55:
                    edge.rel_type = RelationshipType.CLOSE_FRIEND
                rpe = 0.35
                ist.mood_pad.valence = min(1.0, round(ist.mood_pad.valence + 0.12, 4))
            else:
                # Determine whether rejection or exploitative offer constitutes betrayal
                betrayal_signal = max(
                    0.15,
                    min(1.0, abs(min(0.0, my_out.perceived_honesty_delta)) + max(0.0, price_gouge_ratio)),
                )
                if peer_out.consent_decision == ConsentDecision.REJECT or price_gouge_ratio > 0.35:
                    edge.betrayals_experienced += 1
                    edge.trust_tau = max(
                        -1.0,
                        round(edge.trust_tau - eta_neg * (1.0 + edge.trust_tau) * betrayal_signal, 4),
                    )
                    if edge.trust_tau < -0.25:
                        edge.rel_type = RelationshipType.RIVAL
                rpe = 0.55 if betrayal_signal > 0.3 else 0.20
                ist.mood_pad.valence = max(-1.0, round(ist.mood_pad.valence - 0.15, 4))
                ist.mood_pad.arousal = min(1.0, round(ist.mood_pad.arousal + 0.12, 4))

            # Update subjective ToM model of counterparty
            pmodel.est_honesty = max(
                0.0, min(1.0, round(0.8 * pmodel.est_honesty + 0.2 * (0.5 + 0.5 * edge.trust_tau), 4))
            )
            pmodel.narrative_summary = (
                f"At Tick {tick}, {peer_id} said: '{peer_out.verbal_message[:80]}' "
                f"-> Outcome: {verdict.status.value} (Trust={edge.trust_tau:+.2f})."
            )

            scoped.save_relational_edge(edge)
            scoped.save_peer_perceived_model(pmodel)
            scoped.save_internal_state(ist)

            narrative = (
                f"Tick {tick} interaction with {peer_id}: Proposal was "
                f"{initiator_output.economic_intent.quantity} {initiator_output.economic_intent.item_id} "
                f"at ${offered_price:.2f}. {peer_id} said '{peer_out.verbal_message}'. "
                f"Verdict: {verdict.diagnostic_feedback}"
            )
            scoped.record_episodic_memory(
                tick=tick,
                narrative_text=narrative,
                involved_peers=[peer_id],
                arousal=ist.mood_pad.arousal,
                valence=ist.mood_pad.valence,
                rpe_magnitude=rpe,
            )


# ==============================================================================
# POST-MVP INSTITUTIONAL STUBS (Preserved in Architecture with Exact Equations)
# ==============================================================================
class CentralAndCommercialBankModule:
    """
    # TODO(post-mvp): [Institutions - Diamond-Dybvig Fractional Reserve & Minsky Credit Engine]
    Implements:
      1. Risk-adjusted loan interest rate:
         r_loan(t, CS_i) = r_base(t) + delta_spread * exp( - (CS_i - 300) / 250 )
      2. Fractional reserve liquidity tracking:
         R_bank(t) = rho_reserve * sum_i B_i(t) - PanicWithdrawals(t)
         Triggering `ILLIQUID_FREEZE` when R_bank(t) <= 0.
    """

    def step_monetary_cycle(self, tick: int) -> None:
        # TODO(post-mvp): Accrue deposit/loan interest and evaluate Diamond-Dybvig run threshold.
        pass


class PublicGoodsAndLawCommonsModule:
    """
    # TODO(post-mvp): [Institutions - Ostrom Commons & Herrmann-Thoni-Gachter Punishment]
    Implements:
      1. Public Pool Multiplication & Dividend:
         G_t = sum_{i=1}^N (T_i + g_i),   dividend d_t = (r_pub * G_t) / N
      2. Costly Peer Punishment & Tax Audit:
         Agent i pays c_punish to deduct m_punish * c_punish from free-rider j.
    """

    def step_commons_and_audits(self, tick: int) -> None:
        # TODO(post-mvp): Collect voluntary public goods contributions and run stochastic tax audits.
        pass