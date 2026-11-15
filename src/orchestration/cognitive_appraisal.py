"""
src/orchestration/cognitive_appraisal.py
MODULE 6 (M6): Pre-Action Cognitive Appraisal, Prospect Theory Valuation,
Two-Tier Habit Router Gate, and 2nd-Order Theory-of-Mind (ToM) Forecaster.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from src.core.contracts import (
    ActionCategory,
    AgentTurnOutput,
    ConsentDecision,
    EconomicActionIntent,
    EconomyInstitutionsSpec,
    EndogenousStatePacket,
    ExternalAgentState,
    InternalAgentState,
    MemoryCognitionSpec,
    PeerPerceivedModel,
    RelationalEdge,
)


class CognitiveAppraisalAndToMEngine:
    """
    Computes the continuous psychological scalars (`EndogenousStatePacket`) that
    parametrize the Mechanistic Attention Master Equation (`M3: AttentionSteeringLab`)
    and evaluates whether the turn can be handled by the fast System-1 Habit Automaton.
    """

    def __init__(
        self,
        cog_spec: MemoryCognitionSpec,
        econ_spec: EconomyInstitutionsSpec,
    ) -> None:
        self.cog_spec = cog_spec
        self.econ_spec = econ_spec

    @staticmethod
    def compute_prospect_theory_valuation(
        delta_wealth: float,
        loss_aversion_lambda: float,
        risk_curvature_gamma: float,
    ) -> float:
        """
        Kahneman-Tversky Cumulative Prospect Theory Value Function:
            v(Delta_x) = (Delta_x)^gamma                     if Delta_x >= 0
            v(Delta_x) = -lambda * (-Delta_x)^gamma          if Delta_x < 0
        """
        gamma = max(0.2, min(1.5, float(risk_curvature_gamma)))
        lam = max(0.5, min(6.0, float(loss_aversion_lambda)))
        if delta_wealth >= 0.0:
            return math.pow(delta_wealth, gamma)
        return -lam * math.pow(abs(delta_wealth), gamma)

    @staticmethod
    def compute_semantic_dissonance(
        core_belief: str,
        incoming_message: str,
        identity_rigidity: float,
    ) -> float:
        """
        Computes Cognitive Dissonance Score `D_t(i) in [0, 1]` when an incoming
        message contains adversarial, coercive, or belief-contradicting cues.
        """
        if not incoming_message.strip():
            return 0.0

        msg_low = incoming_message.lower()
        hostile_markers = [
            "betray",
            "exploit",
            "scam",
            "impossible",
            "refuse",
            "greed",
            "unfair",
            "threat",
            "lie",
            "monopoly",
            "extort",
            "worthless",
        ]
        hits = sum(1 for w in hostile_markers if w in msg_low)

        # Lexical overlap vs. contradiction heuristic for fast local execution
        belief_words = {w.strip(".,!?;:\"'").lower() for w in core_belief.split() if len(w) > 4}
        msg_words = {w.strip(".,!?;:\"'").lower() for w in msg_low.split() if len(w) > 4}
        overlap = len(belief_words.intersection(msg_words))

        contradiction_signal = min(1.0, 0.32 * hits + (0.25 if overlap == 0 and len(msg_words) > 4 else 0.0))
        d_score = float(identity_rigidity) * contradiction_signal
        return round(max(0.0, min(1.0, d_score)), 4)

    def forecast_2nd_order_tom_acceptance(
        self,
        item_id: str,
        proposed_price: float,
        quantity: int,
        edge: Optional[RelationalEdge],
        peer_model: Optional[PeerPerceivedModel],
    ) -> float:
        """
        2nd-Order Quantal Response Theory-of-Mind Forecaster:
        Estimates the probability `P_hat(j accepts offer)` using agent i's subjective
        estimate of j's Fehr-Schmidt disadvantageous inequity aversion `alpha_hat_{i->j}`:
            U_hat_j(offer) = Surplus_j - alpha_hat_{i->j} * max(0, Surplus_i - Surplus_j)
            P_hat(accept)  = 1 / (1 + exp(-lambda_QRE * U_hat_j))
        """
        ref_unit = (
            self.econ_spec.reference_food_price
            if item_id == "FOOD"
            else self.econ_spec.reference_tech_price
        )
        fair_total = ref_unit * max(1, quantity)
        price_diff = fair_total - proposed_price  # Positive if buyer gets a bargain

        est_alpha = peer_model.est_fehr_alpha if peer_model else 1.2
        trust_bonus = 2.5 * edge.trust_tau if edge else 0.0

        inequity_disutility = est_alpha * max(0.0, abs(price_diff) - 2.0)
        net_perceived_utility = (0.25 * fair_total) - inequity_disutility + trust_bonus

        logit = max(-10.0, min(10.0, 0.35 * net_perceived_utility))
        prob = 1.0 / (1.0 + math.exp(-logit))
        return round(prob, 4)

    def appraise_turn_state(
        self,
        internal_state: InternalAgentState,
        external_state: ExternalAgentState,
        counterparty_id: Optional[str],
        scoped_edges: Dict[str, RelationalEdge],
        scoped_models: Dict[str, PeerPerceivedModel],
        incoming_message: str,
        proposed_price_hint: Optional[float] = None,
    ) -> EndogenousStatePacket:
        """
        Smith-Ellsworth Multi-Dimensional Appraisal Engine:
        Synthesizes financial stress, peer distrust salience, cynicism (`gamma`),
        anxiety (`eta`), cognitive dissonance (`D_t`), arousal (`A_t`), fatigue (`F_t`),
        cognitive load (`L_t`), and construal closeness (`S_closeness`).
        """
        # 1. Financial Stress from Cash Ratio + Survival Deficit
        cash_ratio = external_state.liquid_wealth_cash / max(1.0, self.econ_spec.initial_agent_cash)
        survival_deficit = 1.0 - internal_state.maslow_needs.survival_satiety
        fin_stress = max(
            0.0,
            min(1.0, 0.65 * max(0.0, 1.0 - cash_ratio) + 0.55 * survival_deficit),
        )

        # 2. Dyadic Distrust, Cynicism & Anxiety
        direct_edge = scoped_edges.get(counterparty_id) if counterparty_id else None
        direct_model = scoped_models.get(counterparty_id) if counterparty_id else None

        if direct_edge is not None:
            distrust_salience = max(0.05, min(1.0, 0.5 * (1.0 - direct_edge.trust_tau) + 0.15 * direct_edge.betrayals_experienced))
            cynicism = max(0.0, min(1.0, max(0.0, -direct_edge.trust_tau) + 0.4 * (1.0 - (direct_model.est_honesty if direct_model else 0.5))))
        else:
            distrust_salience = 0.15
            cynicism = 0.10

        anxiety = max(
            0.0,
            min(
                1.0,
                0.45 * fin_stress
                + 0.35 * internal_state.big_five.neuroticism
                + 0.20 * internal_state.mood_pad.arousal,
            ),
        )

        # 3. Cognitive Dissonance Score D_t(i)
        dissonance = self.compute_semantic_dissonance(
            core_belief=internal_state.core_belief_statement,
            incoming_message=incoming_message,
            identity_rigidity=internal_state.identity_rigidity,
        )

        # 4. Construal Psychological Closeness Map S_closeness(i, j) in [-1, 1]
        closeness_map: Dict[str, float] = {}
        for pid, edge in scoped_edges.items():
            closeness_map[pid] = round(
                max(-1.0, min(1.0, 0.6 * edge.trust_tau + 0.4 * edge.intimacy_alpha)), 4
            )

        # 5. 2nd-Order ToM Acceptance Forecast
        price_eval = proposed_price_hint if proposed_price_hint is not None else self.econ_spec.reference_food_price
        tom_prob = self.forecast_2nd_order_tom_acceptance(
            item_id="FOOD",
            proposed_price=price_eval,
            quantity=1,
            edge=direct_edge,
            peer_model=direct_model,
        )

        # 6. Reward Prediction Error & Two-Tier Habit Router Gate (Audit Flaw 6 Resolution)
        rpe_delta = abs(price_eval - self.econ_spec.reference_food_price) / max(
            1.0, self.econ_spec.reference_food_price
        )
        arousal = max(
            0.05,
            min(1.0, 0.6 * internal_state.mood_pad.arousal + 0.25 * fin_stress + 0.25 * dissonance),
        )
        cog_load = max(
            0.1,
            min(1.0, 0.4 * fin_stress + 0.3 * internal_state.fatigue_level + 0.1 * len(scoped_edges)),
        )

        route_habit = (
            not incoming_message.strip()
            and counterparty_id is None
            and arousal < self.cog_spec.habit_router_arousal_threshold
            and rpe_delta < self.cog_spec.habit_router_rpe_threshold
        )

        return EndogenousStatePacket(
            agent_id=internal_state.agent_id,
            counterparty_id=counterparty_id,
            financial_stress=round(fin_stress, 4),
            peer_distrust_salience=round(distrust_salience, 4),
            cynicism_gamma=round(cynicism, 4),
            anxiety_eta=round(anxiety, 4),
            dissonance_score_d=round(dissonance, 4),
            arousal_a=round(arousal, 4),
            fatigue_f=round(internal_state.fatigue_level, 4),
            cognitive_load_l=round(cog_load, 4),
            construal_closeness_map=closeness_map,
            rpe_delta=round(rpe_delta, 4),
            tom_acceptance_forecast=tom_prob,
            route_to_fast_habit=route_habit,
        )

    @staticmethod
    def execute_fast_habit_automaton(agent_id: str) -> AgentTurnOutput:
        """Executes in <0.05ms when an unperturbed agent is routed to Tier-1 Habit."""
        return AgentTurnOutput(
            internal_thought="Routine unperturbed state; executing habitual work and consumption step.",
            epistemic_confidence=0.95,
            consent_decision=ConsentDecision.INITIATE,
            verbal_message="Continuing routine market production.",
            perceived_honesty_delta=0.0,
            economic_intent=EconomicActionIntent(
                action_category=ActionCategory.WORK_AND_CONSUME,
                target_peer_id=None,
                item_id="NONE",
                quantity=0,
                total_price_cash=0.0,
                is_buying=False,
            ),
        )