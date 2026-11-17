from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from src.core.contracts import (
    SimulationConfig,
    normalize_item_id,
)
from src.environment.game_master import AccountingAndLawEngine


class BilateralMarketEngine:
    """
    Implements Dimension 1 & 2 Market Physics:
    - Computes KKT Dual-Entitlement Price Gouging Index (with TFP supply-cost pass-through).
    - Computes reference reservation bounds for surplus accounting and the `RULE_ONLY`
      (Rung R1) comparator baseline—WITHOUT injecting pre-computed actions or reservation
      formulas into the LLM's prompt in `LLM_AUTONOMOUS` mode (Fixing Reviewer #2 Loophole L1).
    """

    def __init__(
        self, config: SimulationConfig, game_master: AccountingAndLawEngine
    ) -> None:
        self.config = config
        self.omega = config.orthogonal_clamping
        self.hank_spec = config.hank_and_crisis_spec
        self.trust_spec = config.trust_and_market_spec
        self.game_master = game_master

    def compute_kkt_gouging_index(
        self,
        buyer_scarcity_f: float,
        offered_unit_price: float,
        item_id: str,
        seller_cost_shift: float = 0.0,
    ) -> float:
        if not self.omega.omega_d2_trust_and_parochialism:
            return 0.0

        norm_item = normalize_item_id(item_id)
        p_ref = (
            self.game_master.macro_state.reference_consumable_price_pc
            if norm_item == "CONSUMABLE_NECESSITY"
            else self.game_master.macro_state.reference_capital_price_qk
        )
        if seller_cost_shift == 0.0 and norm_item == "CONSUMABLE_NECESSITY":
            tfp_ac = max(0.35, self.game_master.macro_state.consumable_tfp_ac)
            seller_cost_shift = round(
                p_ref * max(0.0, (1.0 / tfp_ac - 1.0) * 0.20), 4
            )
        fair_entitlement_price = p_ref + seller_cost_shift
        excess_markup = max(
            0.0, (offered_unit_price - fair_entitlement_price) / max(p_ref, 1e-6)
        )
        desperation_mult = (
            1.0 + self.trust_spec.kkt_desperation_kappa * buyer_scarcity_f
            if self.omega.omega_d1_scarcity_and_hank
            else 1.0
        )
        return round(float(desperation_mult * excess_markup), 4)

    def compute_reservation_prices(
        self,
        seller_id: str,
        buyer_id: str,
        item_id: str,
        effective_trust_sb: float,
        effective_trust_bs: float,
        intimacy_sb: float = 0.0,
    ) -> Dict[str, float]:
        norm_item = normalize_item_id(item_id)
        buyer_ledger = self.game_master.get_agent_ledger(buyer_id)

        f_seller, _, delta_seller = (
            self.game_master.compute_agent_scarcity_and_present_bias(seller_id)
        )
        f_buyer, _, delta_buyer = (
            self.game_master.compute_agent_scarcity_and_present_bias(buyer_id)
        )

        if self.omega.omega_d2_trust_and_parochialism:
            distrust_sb = max(0.0, 0.55 - effective_trust_sb) * 1.6
            distrust_bs = max(0.0, 0.55 - effective_trust_bs) * 1.6
            friend_discount = max(0.0, intimacy_sb) * 0.35
        else:
            distrust_sb = 0.0
            distrust_bs = 0.0
            friend_discount = 0.0

        mu_margin = self.trust_spec.seller_base_margin_mu
        phi_risk = self.trust_spec.distrust_risk_premium_phi

        if norm_item == "CONSUMABLE_NECESSITY":
            p_ref = self.game_master.macro_state.reference_consumable_price_pc
            base_cost = 0.78 * p_ref
            p_min_seller = base_cost * (
                1.0 + mu_margin * (1.0 - friend_discount) + phi_risk * distrust_sb
            )
            buyer_c = float(buyer_ledger["inventory"]["CONSUMABLE_NECESSITY"])
            urgency_boost = 1.0 + 0.45 * f_buyer + (0.20 if buyer_c <= 1.0 else 0.0)
            base_val = p_ref * 1.15 * urgency_boost
            p_max_buyer = base_val * max(0.65, 1.0 - 0.50 * phi_risk * distrust_bs)
            is_fire_sale = False
        else:
            q_ref = self.game_master.macro_state.reference_capital_price_qk
            if self.omega.omega_d1_scarcity_and_hank and f_seller > 0.42:
                fire_sale_factor = max(0.52, delta_seller * (1.0 - 0.38 * f_seller))
                p_min_seller = q_ref * fire_sale_factor * (1.0 + 0.25 * distrust_sb)
                is_fire_sale = True
            else:
                p_min_seller = q_ref * (
                    0.92
                    + 0.50 * mu_margin * (1.0 - friend_discount)
                    + 0.50 * phi_risk * distrust_sb
                )
                is_fire_sale = False

            buyer_cash = float(buyer_ledger["liquid_wealth_cash"])
            liquidity_premium = 1.12 if buyer_cash >= 100.0 else 0.75
            p_max_buyer = (
                q_ref
                * liquidity_premium
                * delta_buyer
                * max(0.65, 1.0 - 0.50 * phi_risk * distrust_bs)
            )

        mid_clearing_price = round(0.50 * (p_min_seller + p_max_buyer), 2)
        return {
            "item_id": norm_item,
            "p_min_seller": round(float(p_min_seller), 2),
            "p_max_buyer": round(float(p_max_buyer), 2),
            "zopa_width": round(float(p_max_buyer - p_min_seller), 2),
            "suggested_clearing_price": mid_clearing_price,
            "is_fire_sale": float(is_fire_sale),
        }

    def recommend_dyadic_trade_direction(
        self, initiator_id: str, partner_id: str, effective_trust_ij: float
    ) -> Dict[str, Any]:
        """
        Computes the closed-form Stone-Geary / KMV benchmark trade direction used
        by the `RULE_ONLY` baseline (Rung R1) and for post-trade surplus accounting.
        """
        init_ledger = self.game_master.get_agent_ledger(initiator_id)
        part_ledger = self.game_master.get_agent_ledger(partner_id)

        f_init, _, _ = self.game_master.compute_agent_scarcity_and_present_bias(
            initiator_id
        )
        f_part, _, _ = self.game_master.compute_agent_scarcity_and_present_bias(
            partner_id
        )

        init_c = float(init_ledger["inventory"]["CONSUMABLE_NECESSITY"])
        init_k = float(init_ledger["inventory"]["PRODUCTIVE_CAPITAL_ASSET"])
        init_cash = float(init_ledger["liquid_wealth_cash"])

        part_c = float(part_ledger["inventory"]["CONSUMABLE_NECESSITY"])
        part_k = float(part_ledger["inventory"]["PRODUCTIVE_CAPITAL_ASSET"])
        part_cash = float(part_ledger["liquid_wealth_cash"])

        if (
            self.omega.omega_d1_scarcity_and_hank
            and f_init > 0.42
            and init_k >= 1.0
            and part_cash >= 65.0
        ):
            bounds = self.compute_reservation_prices(
                seller_id=initiator_id,
                buyer_id=partner_id,
                item_id="PRODUCTIVE_CAPITAL_ASSET",
                effective_trust_sb=effective_trust_ij,
                effective_trust_bs=effective_trust_ij,
            )
            return {
                "recommended_action": "SELL",
                "item_id": "PRODUCTIVE_CAPITAL_ASSET",
                "recommended_qty": 1.0,
                "is_distress_fire_sale": True,
                **bounds,
            }

        if (
            self.omega.omega_d1_scarcity_and_hank
            and f_part > 0.42
            and part_k >= 1.0
            and init_cash >= 65.0
        ):
            bounds = self.compute_reservation_prices(
                seller_id=partner_id,
                buyer_id=initiator_id,
                item_id="PRODUCTIVE_CAPITAL_ASSET",
                effective_trust_sb=effective_trust_ij,
                effective_trust_bs=effective_trust_ij,
            )
            return {
                "recommended_action": "BUY",
                "item_id": "PRODUCTIVE_CAPITAL_ASSET",
                "recommended_qty": 1.0,
                "is_distress_fire_sale": True,
                **bounds,
            }

        if init_c <= 2.5 and part_c >= 2.0:
            bounds = self.compute_reservation_prices(
                seller_id=partner_id,
                buyer_id=initiator_id,
                item_id="CONSUMABLE_NECESSITY",
                effective_trust_sb=effective_trust_ij,
                effective_trust_bs=effective_trust_ij,
            )
            return {
                "recommended_action": "BUY",
                "item_id": "CONSUMABLE_NECESSITY",
                "recommended_qty": 1.0,
                "is_distress_fire_sale": False,
                **bounds,
            }

        if init_c >= 2.0:
            bounds = self.compute_reservation_prices(
                seller_id=initiator_id,
                buyer_id=partner_id,
                item_id="CONSUMABLE_NECESSITY",
                effective_trust_sb=effective_trust_ij,
                effective_trust_bs=effective_trust_ij,
            )
            return {
                "recommended_action": "SELL",
                "item_id": "CONSUMABLE_NECESSITY",
                "recommended_qty": 1.0,
                "is_distress_fire_sale": False,
                **bounds,
            }

        bounds = self.compute_reservation_prices(
            seller_id=partner_id,
            buyer_id=initiator_id,
            item_id="CONSUMABLE_NECESSITY",
            effective_trust_sb=effective_trust_ij,
            effective_trust_bs=effective_trust_ij,
        )
        return {
            "recommended_action": "BUY",
            "item_id": "CONSUMABLE_NECESSITY",
            "recommended_qty": 1.0,
            "is_distress_fire_sale": False,
            **bounds,
        }

    def match_conflict_free_pairs(
        self,
        tick: int,
        agent_ids: List[str],
        effective_trust_matrix: Dict[Tuple[str, str], float],
    ) -> List[Tuple[str, str]]:
        ledgers = {
            aid: self.game_master.get_agent_ledger(aid) for aid in agent_ids
        }
        scarcities = {
            aid: self.game_master.compute_agent_scarcity_and_present_bias(aid)[0]
            for aid in agent_ids
        }

        candidate_edges: List[Tuple[float, str, str]] = []
        for i_idx, aid_i in enumerate(agent_ids):
            for j_idx, aid_j in enumerate(agent_ids):
                if i_idx == j_idx:
                    continue
                t_eff = effective_trust_matrix.get((aid_i, aid_j), 0.50)
                if not self.omega.omega_d2_trust_and_parochialism:
                    t_eff = 0.50

                c_i = float(ledgers[aid_i]["inventory"]["CONSUMABLE_NECESSITY"])
                c_j = float(ledgers[aid_j]["inventory"]["CONSUMABLE_NECESSITY"])
                k_i = float(ledgers[aid_i]["inventory"]["PRODUCTIVE_CAPITAL_ASSET"])
                cash_j = float(ledgers[aid_j]["liquid_wealth_cash"])

                food_comp = (
                    1.0
                    if (c_i <= 2.0 and c_j >= 2.5) or (c_i >= 2.5 and c_j <= 2.0)
                    else 0.1
                )
                fire_sale_comp = (
                    1.2
                    if (scarcities[aid_i] > 0.42 and k_i >= 1.0 and cash_j >= 70.0)
                    else 0.0
                )
                surplus_score = food_comp + fire_sale_comp
                rot_bonus = 0.15 * math.sin(0.9 * tick + 1.7 * i_idx - 1.1 * j_idx)

                total_score = (
                    self.trust_spec.partner_match_trust_beta * t_eff
                    + self.trust_spec.partner_match_surplus_beta * surplus_score
                    + rot_bonus
                )
                candidate_edges.append((total_score, aid_i, aid_j))

        candidate_edges.sort(key=lambda x: x[0], reverse=True)
        used = set()
        matched_pairs: List[Tuple[str, str]] = []
        for _, u, v in candidate_edges:
            if u not in used and v not in used:
                matched_pairs.append((u, v))
                used.add(u)
                used.add(v)
        return matched_pairs