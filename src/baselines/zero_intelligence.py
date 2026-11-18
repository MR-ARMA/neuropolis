from __future__ import annotations

from typing import Any, Dict
import numpy as np

from src.core.contracts import EconomicActionIntent, SimulationConfig
from src.core.seed_bundle import SeedBundle


class ZeroIntelligenceBaselineAgent:
    """
    Budget-Constrained Zero-Intelligence (ZI-C) Baseline Trader
    (Gode & Sunder 1993 [VERIFY]; Section 5 & Section 7).
    Establishes the empirical null floor (`D_null`): chooses uniformly at random
    among budget- and inventory-feasible actions and prices without strategic
    appraisal, fairness preferences, or LLM inference.
    """

    def __init__(self, config: SimulationConfig, seed_bundle: SeedBundle) -> None:
        self.config = config
        self.rng: np.random.Generator = seed_bundle.get_np_rng("env_rng")

    def propose_initiator_intent(
        self,
        initiator_ledger: Dict[str, Any],
        responder_id: str,
        ref_consumable_price: float,
        ref_capital_price: float,
    ) -> EconomicActionIntent:
        cash = float(initiator_ledger["liquid_wealth_cash"])
        b_bar = float(initiator_ledger["credit_limit_b_bar"])
        spendable = max(0.0, cash - b_bar)
        c_inv = float(initiator_ledger["inventory"]["CONSUMABLE_NECESSITY"])
        k_inv = float(initiator_ledger["inventory"]["PRODUCTIVE_CAPITAL_ASSET"])

        feasible_choices = ["HOLD_OR_WORK"]
        if spendable >= 6.0:
            feasible_choices.append("BUY_CONSUMABLE")
        if spendable >= 30.0:
            feasible_choices.append("BUY_CAPITAL")
        if c_inv >= 1.0:
            feasible_choices.append("SELL_CONSUMABLE")
        if k_inv >= 1.0:
            feasible_choices.append("SELL_CAPITAL")

        choice = str(self.rng.choice(feasible_choices))
        if choice == "HOLD_OR_WORK":
            return EconomicActionIntent(
                action_type="HOLD_OR_WORK",
                counterparty_id=responder_id,
                item_id="CONSUMABLE_NECESSITY",
                quantity=0.0,
                unit_price=ref_consumable_price,
                public_goods_contribution=round(float(self.rng.uniform(0.0, 3.0)), 2),
            )

        if "CONSUMABLE" in choice:
            item_id = "CONSUMABLE_NECESSITY"
            p_low = 0.60 * ref_consumable_price
            p_high = min(max(p_low + 1.0, spendable), 1.60 * ref_consumable_price)
        else:
            item_id = "PRODUCTIVE_CAPITAL_ASSET"
            p_low = 0.55 * ref_capital_price
            p_high = min(max(p_low + 5.0, spendable), 1.35 * ref_capital_price)

        act = "BUY" if choice.startswith("BUY") else "SELL"
        unit_p = round(float(self.rng.uniform(p_low, max(p_low + 0.5, p_high))), 2)
        return EconomicActionIntent(
            action_type=act,  # type: ignore[arg-type]
            counterparty_id=responder_id,
            item_id=item_id,
            quantity=1.0,
            unit_price=unit_p,
            public_goods_contribution=round(float(self.rng.uniform(0.0, 3.0)), 2),
            is_distress_fire_sale=(act == "SELL" and item_id == "PRODUCTIVE_CAPITAL_ASSET"),
        )

    def evaluate_responder_consent(
        self,
        responder_ledger: Dict[str, Any],
        intent: EconomicActionIntent,
    ) -> str:
        if intent.action_type == "HOLD_OR_WORK":
            return "IGNORE"
        # Budget-constrained uniform random acceptance (p = 0.50 if feasible)
        if intent.action_type == "SELL":
            # Responder is buyer
            spendable = float(responder_ledger["liquid_wealth_cash"]) - float(
                responder_ledger["credit_limit_b_bar"]
            )
            if spendable < intent.quantity * intent.unit_price:
                return "REJECT"
        else:
            # Responder is seller
            avail = float(responder_ledger["inventory"].get(intent.item_id, 0.0))
            if avail < intent.quantity:
                return "REJECT"
        return "ACCEPT" if float(self.rng.random()) < 0.50 else "REJECT"