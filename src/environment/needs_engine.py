from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from src.core.contracts import HouseholdBalanceSheet, TwoAssetEconomyConfig


@dataclass
class MetabolicStepOutcome:
    """Outcome of one tick of biological subsistence consumption and health evolution."""
    agent_id: str
    was_alive_before: bool
    is_alive_after: bool
    died_this_tick: bool
    consumed_units: float
    subsistence_shortfall: float
    health_before: float
    health_after: float
    composite_stress_after: float


class NeedsEngine:
    """
    Biological Survival & Physiological Stress Physics.
      - Every living household must consume `c_min` units of CONSUMABLE per tick.
      - Sub-subsistence consumption degrades biological health H_{i,t} in [0, 1].
      - Health == 0.0 represents permanent economic death: dead agents earn zero wages,
        produce zero goods, and cannot submit CDA orders.
      - Computes objective `composite_stress` S_{i,t} in [0, 1] used exclusively to
        parameterize domain-general cognitive bottlenecks (Deliberation Budget & Memory
        Friction), NEVER to gate Representation Engineering.
    """

    @staticmethod
    def is_alive(sheet: HouseholdBalanceSheet) -> bool:
        return sheet.needs.health > 0.0

    @staticmethod
    def compute_productivity_multiplier(
        sheet: HouseholdBalanceSheet, cfg: TwoAssetEconomyConfig
    ) -> float:
        """
        Maps biological health H_{i,t} to labor wage and agricultural output capacity.
        Dead agents (H == 0) have strictly 0.0 productivity.
        """
        h = sheet.needs.health
        if h <= 1e-9:
            return 0.0
        floor = cfg.health_productivity_floor
        return round(floor + (1.0 - floor) * h, 6)

    @staticmethod
    def compute_objective_stress(
        sheet: HouseholdBalanceSheet, cfg: TwoAssetEconomyConfig
    ) -> float:
        """
        Computes objective physiological and liquidity stress S_{i,t} in [0, 1]
        from biological health deficit, consumable reserve depletion, and proximity
        to the hard borrowing floor b_bar_i.
        """
        if sheet.needs.health <= 1e-9:
            return 1.0

        health_deficit = 1.0 - sheet.needs.health

        c_safe = 2.0 * cfg.subsistence_min_c
        food_deficit = max(0.0, (c_safe - sheet.consumable_inventory) / max(c_safe, 1e-6))

        b_comfort = max(30.0, 4.0 * sheet.periodic_obligation_r + 2.0 * sheet.base_wage)
        b_floor = sheet.borrowing_limit_b_bar
        denom = max(1.0, b_comfort - b_floor)
        cash_deficit = max(0.0, min(1.0, (b_comfort - sheet.liquid_cash) / denom))

        stress = 0.45 * health_deficit + 0.30 * food_deficit + 0.25 * cash_deficit
        return round(max(0.0, min(1.0, stress)), 6)

    @classmethod
    def step_metabolic_needs(
        cls, sheet: HouseholdBalanceSheet, cfg: TwoAssetEconomyConfig
    ) -> MetabolicStepOutcome:
        """
        Deducts up to `subsistence_min_c` from the household's available consumable
        inventory, updates biological health H_{i,t}, checks for economic death,
        and recomputes objective stress S_{i,t}.
        """
        was_alive = cls.is_alive(sheet)
        h_before = sheet.needs.health

        if not was_alive:
            sheet.needs.last_consumed_units = 0.0
            sheet.needs.composite_stress = 1.0
            return MetabolicStepOutcome(
                agent_id=sheet.agent_id,
                was_alive_before=False,
                is_alive_after=False,
                died_this_tick=False,
                consumed_units=0.0,
                subsistence_shortfall=cfg.subsistence_min_c,
                health_before=0.0,
                health_after=0.0,
                composite_stress_after=1.0,
            )

        c_min = cfg.subsistence_min_c
        eaten = round(min(sheet.consumable_inventory, c_min), 6)
        sheet.consumable_inventory = round(max(0.0, sheet.consumable_inventory - eaten), 6)
        shortfall = round(max(0.0, c_min - eaten), 6)

        if shortfall > 1e-9:
            h_after = round(
                max(0.0, h_before - cfg.health_decay_per_missing_unit * shortfall), 6
            )
            sheet.needs.subsistence_deficit_streak += 1
        else:
            h_after = round(
                min(1.0, h_before + cfg.health_recovery_per_surplus_tick), 6
            )
            sheet.needs.subsistence_deficit_streak = 0

        if h_after <= 1e-9:
            h_after = 0.0

        sheet.needs.health = h_after
        sheet.needs.last_consumed_units = eaten
        sheet.needs.composite_stress = cls.compute_objective_stress(sheet, cfg)

        is_alive_after = h_after > 0.0
        died_now = was_alive and (not is_alive_after)

        return MetabolicStepOutcome(
            agent_id=sheet.agent_id,
            was_alive_before=True,
            is_alive_after=is_alive_after,
            died_this_tick=died_now,
            consumed_units=eaten,
            subsistence_shortfall=shortfall,
            health_before=h_before,
            health_after=h_after,
            composite_stress_after=sheet.needs.composite_stress,
        )


class TwoAssetAccounting:
    """
    Exact Two-Asset Portfolio Mathematics (Kaplan, Violante, & Weidner 2014;
    Kaplan, Moll, & Violante 2018):
      - Liquid Cash b_{i,t} earns r_b when positive and pays r_b + kappa when negative.
      - Illiquid Capital a_{i,t} earns r_k > r_b and incurs convex adjustment cost chi(d, a)
        on any deposit or liquidation d_{i,t}.
    """

    @staticmethod
    def compute_kmv_adjustment_cost_cash(
        delta_capital_units: float,
        current_capital_units: float,
        cfg: TwoAssetEconomyConfig,
    ) -> float:
        """
        Kaplan-Moll-Violante convex adjustment/liquidation cost in currency units:
          chi_units(d, a) = chi_0 |d| + [chi_1 / (1 + chi_2)] * (|d| / max(a, a_min))^(1 + chi_2) * max(a, a_min)
          chi_cash = chi_units * capital_unit_par_value
        Capped strictly below 85% of gross par value so non-zero liquidation always yields net cash.
        """
        abs_d = abs(float(delta_capital_units))
        if abs_d <= 1e-9:
            return 0.0

        a_denom = max(float(current_capital_units), cfg.kmv_a_min)
        chi_units = cfg.kmv_chi_0 * abs_d + (
            cfg.kmv_chi_1 / (1.0 + cfg.kmv_chi_2)
        ) * ((abs_d / a_denom) ** (1.0 + cfg.kmv_chi_2)) * a_denom

        gross_par = abs_d * cfg.capital_unit_par_value
        raw_cash_fee = chi_units * cfg.capital_unit_par_value
        capped_fee = min(raw_cash_fee, 0.85 * gross_par)
        return round(float(capped_fee), 8)

    @staticmethod
    def compute_liquid_interest_flow(
        liquid_cash: float,
        cfg: TwoAssetEconomyConfig,
        credit_wedge_shock_addon: float = 0.0,
    ) -> float:
        """
        Returns signed interest flow for the household:
          - Positive if b_{i,t} >= 0 (interest earned from FinancialIntermediary).
          - Negative if b_{i,t} < 0 (borrowing cost paid to FinancialIntermediary).
        """
        b = float(liquid_cash)
        if b >= 0.0:
            return round(b * cfg.liquid_return_rate_rb, 8)
        borrowing_rate = (
            cfg.liquid_return_rate_rb
            + cfg.borrowing_wedge_kappa
            + max(0.0, credit_wedge_shock_addon)
        )
        return round(b * borrowing_rate, 8)

    @staticmethod
    def compute_capital_dividend_flow(
        total_capital_units: float,
        cfg: TwoAssetEconomyConfig,
        yield_multiplier: float = 1.0,
    ) -> float:
        """Returns dividend payout earned by illiquid productive capital holdings a_{i,t}."""
        a = max(0.0, float(total_capital_units))
        if a <= 1e-9:
            return 0.0
        eff_rk = cfg.capital_return_rate_rk * max(0.0, float(yield_multiplier))
        return round(a * cfg.capital_unit_par_value * eff_rk, 8)

    @staticmethod
    def classify_htm_status(
        sheet: HouseholdBalanceSheet,
        cfg: TwoAssetEconomyConfig,
        consumable_market_price: float,
    ) -> Literal["POOR_HTM", "WEALTHY_HTM", "NON_HTM"]:
        """
        Kaplan, Violante, & Weidner (2014) empirical Hand-to-Mouth classification:
        A household is Hand-to-Mouth if its total liquid cash (b_{i,t} + escrow_cash)
        is at or below half of its periodic wage income (or within half a period's
        wage of its borrowing limit b_bar_i).
          - Illiquid capital > 0 -> WEALTHY_HTM (W-HtM)
          - Illiquid capital == 0 -> POOR_HTM (P-HtM)
        """
        total_liquid = sheet.liquid_cash + sheet.escrow_cash
        total_illiquid = sheet.illiquid_capital + sheet.escrow_capital
        half_wage = 0.50 * max(sheet.base_wage, consumable_market_price)

        is_htm = (total_liquid <= half_wage) or (
            total_liquid <= sheet.borrowing_limit_b_bar + half_wage
        )
        if not is_htm:
            return "NON_HTM"
        if total_illiquid > 1e-6:
            return "WEALTHY_HTM"
        return "POOR_HTM"

    @staticmethod
    def compute_net_worth(
        sheet: HouseholdBalanceSheet,
        cfg: TwoAssetEconomyConfig,
        consumable_price: float,
        capital_price: Optional[float] = None,
    ) -> float:
        k_val = (
            capital_price if capital_price is not None else cfg.capital_unit_par_value
        )
        total_cash = sheet.liquid_cash + sheet.escrow_cash
        total_k = sheet.illiquid_capital + sheet.escrow_capital
        total_c = sheet.consumable_inventory + sheet.escrow_consumable
        return round(total_cash + total_k * k_val + total_c * consumable_price, 6)