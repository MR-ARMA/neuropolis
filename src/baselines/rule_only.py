from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.contracts import (
    EconomicActionIntent,
    SimulationConfig,
    normalize_item_id,
)


class RuleOnlyBaselineAgent:
    """
    Isolated Closed-Form Stone-Geary / Fehr-Schmidt Rule-Only Baseline Agent.
    Used ONLY when `decision_mode == "RULE_ONLY"`. Never invoked as a fallback
    inside `LLM_AUTONOMOUS` runs (Resolves Defect D5).
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.omega = config.orthogonal_clamping

    def propose_initiator_intent(
        self,
        initiator_id: str,
        responder_id: str,
        trade_rec: Dict[str, Any],
        scarcity_f: float,
        effective_trust: float,
        acquired_expert_signal: Optional[float] = None,
    ) -> EconomicActionIntent:
        action = str(trade_rec["recommended_action"])
        item_id = normalize_item_id(str(trade_rec["item_id"]))
        is_fire_sale = bool(trade_rec["is_distress_fire_sale"])
        p_min_s = float(trade_rec["p_min_seller"])
        p_max_b = float(trade_rec["p_max_buyer"])

        if (
            acquired_expert_signal is not None
            and item_id == "PRODUCTIVE_CAPITAL_ASSET"
            and acquired_expert_signal < -0.40
        ):
            p_min_s *= 0.88
            p_max_b *= 0.88

        init_cfg = next(
            (a for a in self.config.agents if a.agent_id == initiator_id),
            self.config.agents[0],
        )
        beta_fs = (
            init_cfg.beta_advantageous
            if self.omega.omega_d2_trust_and_parochialism
            else 0.0
        )

        if p_max_b >= p_min_s:
            if action == "SELL":
                bargain_weight = (
                    0.55
                    + 0.30 * (1.0 - beta_fs)
                    + 0.35 * max(0.0, 0.55 - effective_trust)
                )
                if is_fire_sale:
                    bargain_weight = max(0.15, 0.40 - 0.30 * scarcity_f)
            else:
                bargain_weight = 0.45 + 0.25 * beta_fs + 0.25 * scarcity_f
            strategic_p = round(p_min_s + bargain_weight * (p_max_b - p_min_s), 2)
        else:
            strategic_p = round(0.50 * (p_min_s + p_max_b), 2)

        return EconomicActionIntent(
            action_type=action,  # type: ignore[arg-type]
            counterparty_id=responder_id,
            item_id=item_id,
            quantity=1.0,
            unit_price=max(1.0, strategic_p),
            is_distress_fire_sale=is_fire_sale,
        )

    def evaluate_responder_consent(
        self,
        resp_is_buyer: bool,
        offered_price: float,
        p_min_seller: float,
        p_max_buyer: float,
        kkt_gouging_index: float,
        effective_trust_ba: float,
        responder_ctx: Dict[str, Any],
    ) -> str:
        fs = responder_ctx["internal_state"]["fehr_schmidt"]
        alpha_i = (
            float(fs.get("alpha", 0.5))
            if self.omega.omega_d2_trust_and_parochialism
            else 0.0
        )

        if resp_is_buyer:
            surplus = p_max_buyer - offered_price
            opponent_surplus = max(0.0, offered_price - p_min_seller)
        else:
            surplus = offered_price - p_min_seller
            opponent_surplus = max(0.0, p_max_buyer - offered_price)

        ineq_disutility = alpha_i * max(0.0, opponent_surplus - surplus) * 0.35
        kkt_disutility = (
            self.config.trust_and_market_spec.kkt_weight_lambda
            * kkt_gouging_index
            * 2.5
            if resp_is_buyer
            else 0.0
        )
        net_utility = surplus - ineq_disutility - kkt_disutility

        if net_utility < -0.75 or (effective_trust_ba < 0.22 and net_utility < 0.50):
            return "REJECT"
        return "ACCEPT"