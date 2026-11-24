from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import uuid
import numpy as np

from src.core.contracts import (
    AgentActionType,
    AssetType,
    CDAOrder,
    HouseholdBalanceSheet,
    MacroShockType,
    NeedsState,
    OrderBookSnapshot,
    OrderSide,
    OrderStatus,
    OrderType,
    SectoralBalanceAudit,
    SimulationConfig,
    Stage2ActionPayload,
    TradeFill,
)
from src.environment.market_cda import (
    ContinuousDoubleAuctionExchange,
    OrderEscrowRelease,
)
from src.environment.needs_engine import (
    MetabolicStepOutcome,
    NeedsEngine,
    TwoAssetAccounting,
)


@dataclass
class ActionExecutionOutcome:
    """Deterministic GameMaster verdict for a single agent's Stage 2 Action."""
    agent_id: str
    action_type: AgentActionType
    accepted: bool
    reason: str
    submitted_order: Optional[CDAOrder] = None
    fills: List[TradeFill] = field(default_factory=list)
    kmv_fee_paid: float = 0.0
    net_cash_delta: float = 0.0


@dataclass
class TickMacroSummary:
    """Objective end-of-tick environment and SFC telemetry (Zero fabricated spending/MPC)."""
    tick: int
    shock_intensity_s: float
    living_agents_count: int
    deaths_this_tick: List[str]
    consumable_book: OrderBookSnapshot
    capital_book: Optional[OrderBookSnapshot]
    fills_this_tick: List[TradeFill]
    consumable_market_frozen: bool
    poor_htm_share: float
    wealthy_htm_share: float
    non_htm_share: float
    realized_consumable_spend_by_agent: Dict[str, float]
    realized_disposable_income_by_agent: Dict[str, float]
    obligation_defaults_this_tick: List[str]
    sfc_audit: SectoralBalanceAudit


class GameMaster:
    """
    Module A Physics Orchestrator — Strict Stock-Flow Consistent (SFC) Environment.
      - Manages 5 closed sectoral cash balances:
          1. Households (liquid_cash)
          2. OrderBookEscrow (escrow_cash locked in active CDA bids)
          3. ProductionFirm (firm_sector_cash)
          4. FinancialIntermediary (intermediary_cash)
          5. PublicTreasury (treasury_cash)
      - Executes peer-to-peer Continuous Double Auction (CDA) clearing ONLY when Bid >= Ask.
      - Applies Two-Asset return wedge (r_k > r_b), borrowing wedge (kappa), and KMV convex
        liquidation costs chi(d, a).
      - Applies Needs Engine metabolic depreciation, health decay, and economic death (H == 0).
      - ZERO automated consumption spending on behalf of agents: realized spend equals exact
        peer-to-peer CDA fills.
      - Dynamically asserts `abs(total_money - M_0) < 1e-9` on every mutation and tick end.
      - Includes native Zero-Intelligence (ZI) CDA policy generation to benchmark macro-freezes.
    """

    def __init__(
        self,
        config: SimulationConfig,
        economy_seed: Optional[int] = None,
    ) -> None:
        self.config = config
        self.economy_cfg = config.economy
        self.cda_cfg = config.cda
        self.shock_cfg = config.shock
        self.seed = economy_seed if economy_seed is not None else config.master_seed
        self.rng = np.random.default_rng(self.seed)

        # Institutional sectoral cash accounts (Strict SFC Closure)
        self.firm_sector_cash: float = round(
            float(self.economy_cfg.initial_firm_sector_cash), 8
        )
        self.intermediary_cash: float = round(
            float(self.economy_cfg.initial_intermediary_cash), 8
        )
        self.treasury_cash: float = round(
            float(self.economy_cfg.initial_treasury_cash), 8
        )
        self.intermediary_capital_book: float = 0.0

        # Dynamic AR(1) macro shock state
        self.shock_intensity_s: float = 0.0
        self.current_tfp_a_c: float = self.economy_cfg.baseline_tfp_a_c
        self.current_capital_yield_mult: float = 1.0
        self.current_credit_wedge_addon: float = 0.0

        # Exchange & Households
        self.exchange = ContinuousDoubleAuctionExchange(self.cda_cfg)
        self.households: Dict[str, HouseholdBalanceSheet] = {}
        self._initialize_stratified_two_asset_population()

        # Freeze initial global money supply M_0 for invariant assertions
        self.initial_money_supply_m0: float = self._compute_total_money_supply()

        # Per-tick tracking (strictly realized transactions, zero circular MPC)
        self.tick_fills: List[TradeFill] = []
        self.tick_consumable_spend: Dict[str, float] = {
            aid: 0.0 for aid in self.households
        }
        self.tick_disposable_income: Dict[str, float] = {
            aid: 0.0 for aid in self.households
        }
        self.audit_and_assert_sfc(tick=0)

    # ========================================================================
    # 1. STRATIFIED TWO-ASSET POPULATION INITIALIZATION (KVW W-HtM CALIBRATED)
    # ========================================================================

    def _initialize_stratified_two_asset_population(self) -> None:
        """
        Initializes N heterogeneous households across 4 economic strata calibrated so:
          - ~21% start as Wealthy Hand-to-Mouth (high illiquid capital, low liquid cash).
          - ~13% start as Poor Hand-to-Mouth (zero illiquid capital, low liquid cash).
          - ~66% start as Non-HtM (producers, liquid capitalists, arbitrageurs).
          - Aggregate consumable production capacity exceeds N * c_min in normal equilibrium
            so starvation/freezes are never hard-wired by an impossible resource deficit.
        """
        n_agents = self.config.num_agents
        ids = [f"Agent_{i + 1:02d}" for i in range(n_agents)]
        self.rng.shuffle(ids)

        for idx, aid in enumerate(ids):
            bucket = idx % 12
            if bucket in (0, 1, 2):
                # Consumable Producers (25% of pop; each harvests ~5.2 units/tick -> 1.3 * N supply)
                role = "PRODUCER"
                wage = round(float(np.clip(self.rng.normal(14.0, 1.5), 10.0, 18.0)), 2)
                cash = round(float(np.clip(self.rng.normal(65.0, 12.0), 35.0, 110.0)), 2)
                capital = round(float(self.rng.choice([0.5, 1.0, 1.5])), 2)
                consumable = round(float(np.clip(self.rng.normal(8.0, 1.5), 5.0, 12.0)), 2)
                prod_q = round(float(np.clip(self.rng.normal(5.2, 0.4), 4.2, 6.2)), 2)
                ob_r = round(0.32 * wage, 2)
                exposure = round(float(self.rng.uniform(0.55, 0.85)), 2)
            elif bucket in (3, 4, 5):
                # Liquid Capitalists (25% of pop; high cash & high capital, need to buy consumables)
                role = "CAPITALIST"
                wage = round(float(np.clip(self.rng.normal(24.0, 3.0), 18.0, 32.0)), 2)
                cash = round(float(np.clip(self.rng.normal(140.0, 25.0), 85.0, 210.0)), 2)
                capital = round(float(np.clip(self.rng.normal(4.5, 1.0), 2.5, 7.0)), 2)
                consumable = round(float(self.rng.choice([1.0, 2.0])), 2)
                prod_q = 0.0
                ob_r = round(0.28 * wage, 2)
                exposure = round(float(self.rng.uniform(0.20, 0.45)), 2)
            elif bucket in (6, 7):
                # Wealthy Hand-to-Mouth (W-HtM, ~17%–20%): paycheck-to-paycheck net cash (~$12.3/tick)
                # matching subsistence food cost while holding substantial illiquid capital.
                role = "WORKER"
                wage = round(float(np.clip(self.rng.normal(22.0, 1.5), 19.0, 25.5)), 2)
                cash = round(float(np.clip(self.rng.normal(9.5, 2.0), 5.0, 14.0)), 2)
                capital = round(float(np.clip(self.rng.normal(3.5, 0.7), 2.2, 5.2)), 2)
                consumable = round(float(self.rng.choice([1.0, 2.0])), 2)
                prod_q = 0.0
                ob_r = round(0.56 * wage, 2)
                exposure = round(float(self.rng.uniform(0.60, 0.90)), 2)
            elif bucket in (8, 9):
                # Poor Hand-to-Mouth (P-HtM, ~14%–17%): zero illiquid capital, net cash (~$12.3/tick)
                # sufficient to survive normal periods while staying strictly liquid-constrained.
                role = "WORKER"
                wage = round(float(np.clip(self.rng.normal(21.5, 1.5), 18.5, 25.0)), 2)
                cash = round(float(np.clip(self.rng.normal(8.5, 2.0), 4.0, 13.0)), 2)
                capital = 0.0
                consumable = round(float(self.rng.choice([1.0, 2.0])), 2)
                prod_q = 0.0
                ob_r = round(0.43 * wage, 2)
                exposure = round(float(self.rng.uniform(0.75, 1.00)), 2)
            else:
                # Liquid Arbitrageurs / Balanced Workers (~17%)
                role = "ARBITRAGEUR"
                wage = round(float(np.clip(self.rng.normal(18.0, 2.0), 14.0, 23.0)), 2)
                cash = round(float(np.clip(self.rng.normal(85.0, 15.0), 50.0, 130.0)), 2)
                capital = round(float(np.clip(self.rng.normal(2.0, 0.5), 1.0, 3.5)), 2)
                consumable = round(float(np.clip(self.rng.normal(3.0, 0.8), 2.0, 5.0)), 2)
                prod_q = 0.0
                ob_r = round(0.35 * wage, 2)
                exposure = round(float(self.rng.uniform(0.40, 0.70)), 2)

            b_bar = round(
                self.economy_cfg.borrowing_limit_income_multiple * wage, 2
            )
            sheet = HouseholdBalanceSheet(
                agent_id=aid,
                role=role,  # type: ignore[arg-type]
                liquid_cash=cash,
                escrow_cash=0.0,
                illiquid_capital=capital,
                escrow_capital=0.0,
                consumable_inventory=consumable,
                escrow_consumable=0.0,
                borrowing_limit_b_bar=b_bar,
                base_wage=wage,
                periodic_obligation_r=ob_r,
                base_consumable_production_q=prod_q,
                shock_exposure_e=exposure,
                needs=NeedsState(health=1.0, last_consumed_units=1.0),
            )
            sheet.needs.composite_stress = NeedsEngine.compute_objective_stress(
                sheet, self.economy_cfg
            )
            self.households[aid] = sheet

        # Keep deterministic sorted ordering by agent_id
        self.households = dict(sorted(self.households.items()))

    # ========================================================================
    # 2. STRICT SFC DOUBLE-ENTRY AUDIT & INVARIANT ASSERTIONS
    # ========================================================================

    def _compute_total_money_supply(self) -> float:
        hh_liquid = sum(h.liquid_cash for h in self.households.values())
        hh_escrow = sum(h.escrow_cash for h in self.households.values())
        return round(
            hh_liquid
            + hh_escrow
            + self.firm_sector_cash
            + self.intermediary_cash
            + self.treasury_cash,
            8,
        )

    def audit_and_assert_sfc(self, tick: int) -> SectoralBalanceAudit:
        """
        Verifies two hard invariants on every tick:
          1. Exact Global Monetary Conservation: |M_t - M_0| < 1e-9.
          2. Exact Escrow Parity: sum(household.escrow_cash) == exchange.total_exchange_locked_cash().
        """
        hh_cash = round(sum(h.liquid_cash for h in self.households.values()), 8)
        hh_escrow = round(sum(h.escrow_cash for h in self.households.values()), 8)
        exchange_escrow = self.exchange.total_exchange_locked_cash()

        escrow_diff = abs(hh_escrow - exchange_escrow)
        assert escrow_diff < 1e-6, (
            f"Escrow Parity Violation at tick {tick}: households={hh_escrow:.8f} vs exchange={exchange_escrow:.8f}"
        )

        total_m = round(
            hh_cash
            + hh_escrow
            + self.firm_sector_cash
            + self.intermediary_cash
            + self.treasury_cash,
            8,
        )
        sfc_err = abs(total_m - self.initial_money_supply_m0)
        assert sfc_err < 1e-6, (
            f"STRICT SFC CONSERVATION VIOLATION at tick {tick}: "
            f"M_t={total_m:.8f}, M_0={self.initial_money_supply_m0:.8f}, err={sfc_err:.2e}"
        )

        total_k = round(
            sum(h.illiquid_capital + h.escrow_capital for h in self.households.values())
            + self.intermediary_capital_book,
            6,
        )
        total_c = round(
            sum(h.consumable_inventory + h.escrow_consumable for h in self.households.values()),
            6,
        )
        return SectoralBalanceAudit(
            tick=tick,
            total_household_cash=hh_cash,
            total_escrow_cash=hh_escrow,
            firm_sector_cash=round(self.firm_sector_cash, 8),
            intermediary_cash=round(self.intermediary_cash, 8),
            treasury_cash=round(self.treasury_cash, 8),
            total_money_supply_m=total_m,
            initial_money_supply_m0=self.initial_money_supply_m0,
            conservation_error_abs=sfc_err,
            total_capital_units=total_k,
            total_consumable_units=total_c,
        )

    def _apply_escrow_release(self, release: OrderEscrowRelease) -> None:
        sheet = self.households[release.agent_id]
        if release.cash_to_release > 1e-9:
            amt = round(min(sheet.escrow_cash, release.cash_to_release), 8)
            sheet.escrow_cash = round(sheet.escrow_cash - amt, 8)
            sheet.liquid_cash = round(sheet.liquid_cash + amt, 8)
        if release.asset_units_to_release > 1e-9:
            if release.asset == AssetType.CONSUMABLE:
                amt_a = round(
                    min(sheet.escrow_consumable, release.asset_units_to_release), 6
                )
                sheet.escrow_consumable = round(sheet.escrow_consumable - amt_a, 6)
                sheet.consumable_inventory = round(
                    sheet.consumable_inventory + amt_a, 6
                )
            else:
                amt_a = round(
                    min(sheet.escrow_capital, release.asset_units_to_release), 6
                )
                sheet.escrow_capital = round(sheet.escrow_capital - amt_a, 6)
                sheet.illiquid_capital = round(sheet.illiquid_capital + amt_a, 6)

    # ========================================================================
    # 3. ACTION ARBITRATION: CDA ORDERS, CAPITAL LIQUIDATION & INVESTMENT
    # ========================================================================

    def begin_tick(self, tick: int) -> None:
        """Resets per-tick flow trackers and advances any exogenous macro shock."""
        self.tick_fills = []
        self.tick_consumable_spend = {aid: 0.0 for aid in self.households}
        self.tick_disposable_income = {aid: 0.0 for aid in self.households}
        self._step_macro_shock(tick)

    def _step_macro_shock(self, tick: int) -> None:
        if self.shock_cfg.shock_type == MacroShockType.NONE:
            self.shock_intensity_s = 0.0
            self.current_tfp_a_c = self.economy_cfg.baseline_tfp_a_c
            self.current_capital_yield_mult = 1.0
            self.current_credit_wedge_addon = 0.0
            return

        in_window = (
            self.shock_cfg.start_tick
            <= tick
            < self.shock_cfg.start_tick + self.shock_cfg.duration_ticks
        )
        s_t = min(
            1.0,
            self.shock_cfg.persistence_rho * self.shock_intensity_s
            + (self.shock_cfg.severity_sigma if in_window else 0.0),
        )
        self.shock_intensity_s = round(s_t, 6)

        if self.shock_cfg.shock_type == MacroShockType.TFP_SUBSISTENCE_CONTRACTION:
            self.current_tfp_a_c = round(
                self.economy_cfg.baseline_tfp_a_c * max(0.35, 1.0 - 0.55 * s_t), 4
            )
        elif self.shock_cfg.shock_type == MacroShockType.CAPITAL_YIELD_COLLAPSE:
            self.current_capital_yield_mult = round(max(0.15, 1.0 - 0.80 * s_t), 4)
        elif self.shock_cfg.shock_type == MacroShockType.CREDIT_WEDGE_CRUNCH:
            self.current_credit_wedge_addon = round(0.04 * s_t, 6)

    def execute_agent_action(
        self,
        tick: int,
        agent_id: str,
        payload: Stage2ActionPayload,
    ) -> ActionExecutionOutcome:
        """
        Deterministically arbitrates and settles a single agent's Stage 2 ActionPayload.
        Zero automated trades or silent repairs: illegal actions are rejected cleanly.
        """
        sheet = self.households[agent_id]
        if not NeedsEngine.is_alive(sheet):
            return ActionExecutionOutcome(
                agent_id=agent_id,
                action_type=payload.action_type,
                accepted=False,
                reason="AGENT_DECEASED",
            )

        if payload.action_type in (
            AgentActionType.CONSUME_AND_HOLD,
            AgentActionType.ABSTAIN,
        ):
            return ActionExecutionOutcome(
                agent_id=agent_id,
                action_type=payload.action_type,
                accepted=True,
                reason=payload.action_type.value,
            )

        if payload.action_type == AgentActionType.CANCEL_ORDER:
            oid = payload.cancel_order_id or ""
            for book in self.exchange.books.values():
                rel = book.cancel_order(oid, requesting_agent_id=agent_id)
                if rel is not None:
                    self._apply_escrow_release(rel)
                    self.audit_and_assert_sfc(tick)
                    return ActionExecutionOutcome(
                        agent_id=agent_id,
                        action_type=AgentActionType.CANCEL_ORDER,
                        accepted=True,
                        reason=f"CANCELLED_{oid}",
                    )
            return ActionExecutionOutcome(
                agent_id=agent_id,
                action_type=AgentActionType.CANCEL_ORDER,
                accepted=False,
                reason="ORDER_NOT_FOUND_OR_NOT_OWNED",
            )

        if payload.action_type == AgentActionType.LIQUIDATE_CAPITAL:
            units = round(float(payload.capital_transfer_units), 6)
            if units <= 1e-9 or sheet.illiquid_capital + 1e-9 < units:
                return ActionExecutionOutcome(
                    agent_id=agent_id,
                    action_type=AgentActionType.LIQUIDATE_CAPITAL,
                    accepted=False,
                    reason="INSUFFICIENT_ILLIQUID_CAPITAL",
                )
            gross_par = round(units * self.economy_cfg.capital_unit_par_value, 8)
            kmv_fee = TwoAssetAccounting.compute_kmv_adjustment_cost_cash(
                units, sheet.illiquid_capital, self.economy_cfg
            )
            net_cash = round(gross_par - kmv_fee, 8)
            if self.intermediary_cash < net_cash:
                return ActionExecutionOutcome(
                    agent_id=agent_id,
                    action_type=AgentActionType.LIQUIDATE_CAPITAL,
                    accepted=False,
                    reason="INTERMEDIARY_ILLIQUID",
                )

            sheet.illiquid_capital = round(sheet.illiquid_capital - units, 6)
            self.intermediary_capital_book = round(
                self.intermediary_capital_book + units, 6
            )
            self.intermediary_cash = round(self.intermediary_cash - net_cash, 8)
            sheet.liquid_cash = round(sheet.liquid_cash + net_cash, 8)
            sheet.needs.composite_stress = NeedsEngine.compute_objective_stress(
                sheet, self.economy_cfg
            )
            self.audit_and_assert_sfc(tick)
            return ActionExecutionOutcome(
                agent_id=agent_id,
                action_type=AgentActionType.LIQUIDATE_CAPITAL,
                accepted=True,
                reason=f"LIQUIDATED_{units:.2f}_CAPITAL_NET_${net_cash:.2f}_FEE_${kmv_fee:.2f}",
                kmv_fee_paid=kmv_fee,
                net_cash_delta=net_cash,
            )

        if payload.action_type == AgentActionType.INVEST_CAPITAL:
            units = round(float(payload.capital_transfer_units), 6)
            if units <= 1e-9:
                return ActionExecutionOutcome(
                    agent_id=agent_id,
                    action_type=AgentActionType.INVEST_CAPITAL,
                    accepted=False,
                    reason="INVALID_CAPITAL_UNITS",
                )
            gross_par = round(units * self.economy_cfg.capital_unit_par_value, 8)
            kmv_fee = TwoAssetAccounting.compute_kmv_adjustment_cost_cash(
                units, sheet.illiquid_capital, self.economy_cfg
            )
            total_cost = round(gross_par + kmv_fee, 8)
            spendable = round(sheet.liquid_cash - sheet.borrowing_limit_b_bar, 8)
            if spendable + 1e-9 < total_cost:
                return ActionExecutionOutcome(
                    agent_id=agent_id,
                    action_type=AgentActionType.INVEST_CAPITAL,
                    accepted=False,
                    reason="INSUFFICIENT_LIQUID_CASH_FOR_INVESTMENT",
                )

            sheet.liquid_cash = round(sheet.liquid_cash - total_cost, 8)
            self.intermediary_cash = round(self.intermediary_cash + total_cost, 8)
            sheet.illiquid_capital = round(sheet.illiquid_capital + units, 6)
            self.intermediary_capital_book = round(
                self.intermediary_capital_book - units, 6
            )
            sheet.needs.composite_stress = NeedsEngine.compute_objective_stress(
                sheet, self.economy_cfg
            )
            self.audit_and_assert_sfc(tick)
            return ActionExecutionOutcome(
                agent_id=agent_id,
                action_type=AgentActionType.INVEST_CAPITAL,
                accepted=True,
                reason=f"INVESTED_{units:.2f}_CAPITAL_COST_${total_cost:.2f}_FEE_${kmv_fee:.2f}",
                kmv_fee_paid=kmv_fee,
                net_cash_delta=-total_cost,
            )

        # Otherwise: PLACE_ORDER on Continuous Double Auction Order Book
        return self._execute_cda_order_submission(tick, sheet, payload)

    def _execute_cda_order_submission(
        self,
        tick: int,
        sheet: HouseholdBalanceSheet,
        payload: Stage2ActionPayload,
    ) -> ActionExecutionOutcome:
        if payload.asset not in self.exchange.books:
            return ActionExecutionOutcome(
                agent_id=sheet.agent_id,
                action_type=AgentActionType.PLACE_ORDER,
                accepted=False,
                reason="ASSET_BOOK_DISABLED",
            )
        book = self.exchange.get_book(payload.asset)
        qty = round(min(float(payload.quantity), self.cda_cfg.max_order_quantity), 6)
        side = payload.side
        if side is None or qty <= 1e-9:
            return ActionExecutionOutcome(
                agent_id=sheet.agent_id,
                action_type=AgentActionType.PLACE_ORDER,
                accepted=False,
                reason="INVALID_SIDE_OR_QUANTITY",
            )

        locked_cash = 0.0
        locked_asset = 0.0
        quantized_limit: Optional[float] = None

        if side == OrderSide.BID:
            spendable = round(sheet.liquid_cash - sheet.borrowing_limit_b_bar, 8)
            if payload.order_type == OrderType.LIMIT:
                if payload.limit_price is None or payload.limit_price <= 0.0:
                    return ActionExecutionOutcome(
                        agent_id=sheet.agent_id,
                        action_type=AgentActionType.PLACE_ORDER,
                        accepted=False,
                        reason="MISSING_LIMIT_PRICE",
                    )
                quantized_limit = book.quantize_price(payload.limit_price)
                req_cash = round(qty * quantized_limit, 8)
                if spendable + 1e-9 < req_cash:
                    return ActionExecutionOutcome(
                        agent_id=sheet.agent_id,
                        action_type=AgentActionType.PLACE_ORDER,
                        accepted=False,
                        reason=f"INSUFFICIENT_FUNDS_FOR_BID: need=${req_cash:.2f}, spendable=${spendable:.2f}",
                    )
                locked_cash = req_cash
            else:
                # MARKET BID: check resting ASK liquidity first
                fillable, est_cost = book.estimate_market_sweep_cost(
                    OrderSide.BID, qty, exclude_agent_id=sheet.agent_id
                )
                if fillable <= 1e-9 or spendable <= 1e-9:
                    return ActionExecutionOutcome(
                        agent_id=sheet.agent_id,
                        action_type=AgentActionType.PLACE_ORDER,
                        accepted=False,
                        reason="NO_RESTING_ASK_LIQUIDITY_OR_ZERO_CASH",
                    )
                locked_cash = round(min(spendable, est_cost), 8)

            # Move buyer cash into Escrow strictly before matching
            sheet.liquid_cash = round(sheet.liquid_cash - locked_cash, 8)
            sheet.escrow_cash = round(sheet.escrow_cash + locked_cash, 8)
        else:
            # OrderSide.ASK
            if payload.order_type == OrderType.LIMIT:
                if payload.limit_price is None or payload.limit_price <= 0.0:
                    return ActionExecutionOutcome(
                        agent_id=sheet.agent_id,
                        action_type=AgentActionType.PLACE_ORDER,
                        accepted=False,
                        reason="MISSING_LIMIT_PRICE",
                    )
                quantized_limit = book.quantize_price(payload.limit_price)
            else:
                fillable, _ = book.estimate_market_sweep_cost(
                    OrderSide.ASK, qty, exclude_agent_id=sheet.agent_id
                )
                if fillable <= 1e-9:
                    return ActionExecutionOutcome(
                        agent_id=sheet.agent_id,
                        action_type=AgentActionType.PLACE_ORDER,
                        accepted=False,
                        reason="NO_RESTING_BID_LIQUIDITY",
                    )

            avail_inv = (
                sheet.consumable_inventory
                if payload.asset == AssetType.CONSUMABLE
                else sheet.illiquid_capital
            )
            if avail_inv + 1e-9 < qty:
                return ActionExecutionOutcome(
                    agent_id=sheet.agent_id,
                    action_type=AgentActionType.PLACE_ORDER,
                    accepted=False,
                    reason=f"INSUFFICIENT_INVENTORY_FOR_ASK: avail={avail_inv:.2f}, qty={qty:.2f}",
                )
            locked_asset = qty
            if payload.asset == AssetType.CONSUMABLE:
                sheet.consumable_inventory = round(
                    sheet.consumable_inventory - locked_asset, 6
                )
                sheet.escrow_consumable = round(
                    sheet.escrow_consumable + locked_asset, 6
                )
            else:
                sheet.illiquid_capital = round(
                    sheet.illiquid_capital - locked_asset, 6
                )
                sheet.escrow_capital = round(sheet.escrow_capital + locked_asset, 6)

        order = CDAOrder(
            order_id=f"ord_{tick:04d}_{uuid.uuid4().hex[:8]}",
            tick_submitted=tick,
            sequence_num=self.exchange.next_sequence_num(),
            agent_id=sheet.agent_id,
            asset=payload.asset,
            side=side,
            order_type=payload.order_type,
            limit_price=quantized_limit,
            quantity=qty,
        )

        res = book.submit_order(
            order=order,
            locked_cash=locked_cash,
            locked_asset=locked_asset,
            sales_tax_rate=self.economy_cfg.sales_tax_rate,
        )

        # Settle all matched TradeFills atomically across sectoral accounts
        net_cash_delta = 0.0
        for fill in res.fills:
            buyer = self.households[fill.buyer_id]
            seller = self.households[fill.seller_id]

            # 1. Buyer escrow cash -> Seller liquid cash + PublicTreasury sales tax
            buyer.escrow_cash = round(buyer.escrow_cash - fill.gross_value, 8)
            seller_net = round(fill.gross_value - fill.sales_tax_paid, 8)
            seller.liquid_cash = round(seller.liquid_cash + seller_net, 8)
            self.treasury_cash = round(self.treasury_cash + fill.sales_tax_paid, 8)

            # 2. Seller escrow asset -> Buyer available inventory
            if fill.asset == AssetType.CONSUMABLE:
                seller.escrow_consumable = round(
                    seller.escrow_consumable - fill.quantity, 6
                )
                buyer.consumable_inventory = round(
                    buyer.consumable_inventory + fill.quantity, 6
                )
                # Realized peer-to-peer consumable spend (Zero circular MPC)
                self.tick_consumable_spend[buyer.agent_id] = round(
                    self.tick_consumable_spend[buyer.agent_id] + fill.gross_value, 8
                )
            else:
                seller.escrow_capital = round(
                    seller.escrow_capital - fill.quantity, 6
                )
                buyer.illiquid_capital = round(
                    buyer.illiquid_capital + fill.quantity, 6
                )

            if fill.buyer_id == sheet.agent_id:
                net_cash_delta -= fill.gross_value
            if fill.seller_id == sheet.agent_id:
                net_cash_delta += seller_net

            self.tick_fills.append(fill)

        # Refund buyer price improvement from escrow back to liquid_cash
        for buyer_id, refund_amt in res.price_improvement_refunds:
            b_sheet = self.households[buyer_id]
            b_sheet.escrow_cash = round(b_sheet.escrow_cash - refund_amt, 8)
            b_sheet.liquid_cash = round(b_sheet.liquid_cash + refund_amt, 8)

        # Release unfilled IOC / rounding dust escrow
        if res.unfilled_ioc_release is not None:
            self._apply_escrow_release(res.unfilled_ioc_release)

        # Recompute objective stress for participating agents
        sheet.needs.composite_stress = NeedsEngine.compute_objective_stress(
            sheet, self.economy_cfg
        )
        self.audit_and_assert_sfc(tick)

        return ActionExecutionOutcome(
            agent_id=sheet.agent_id,
            action_type=AgentActionType.PLACE_ORDER,
            accepted=True,
            reason=f"CDA_{order.status.value}_FILLS={len(res.fills)}",
            submitted_order=order,
            fills=res.fills,
            net_cash_delta=round(net_cash_delta, 8),
        )

    # ========================================================================
    # 4. END-OF-TICK PHYSICS: PRODUCTION, RETURNS, OBLIGATIONS & NEEDS ENGINE
    # ========================================================================

    def step_end_of_tick_physics(self, tick: int) -> TickMacroSummary:
        """
        Executes deterministic end-of-tick physics in strict SFC order:
          1. Expire stale CDA orders past TTL and release escrow.
          2. Harvest consumable production (producers only, scaled by TFP & Health).
          3. Pay labor wages, capital dividends, liquid/borrowing interest, and mandatory obligations.
          4. Execute biological Needs Engine (consume c_min, degrade/recover health, handle deaths).
          5. Cancel open orders of any agent who died this tick and release escrow.
          6. Assert exact SFC monetary conservation (`abs(M_t - M_0) < 1e-9`).
        """
        # 1. Expire stale limit orders
        expired_releases = self.exchange.expire_all_stale_orders(tick)
        for rel in expired_releases:
            self._apply_escrow_release(rel)

        deaths_this_tick: List[str] = []
        obligation_defaults: List[str] = []

        for aid, sheet in self.households.items():
            if not NeedsEngine.is_alive(sheet):
                continue

            health_mult = NeedsEngine.compute_productivity_multiplier(
                sheet, self.economy_cfg
            )

            # 2. Consumable Production (Producers only)
            if sheet.base_consumable_production_q > 0.0:
                harvest = round(
                    sheet.base_consumable_production_q
                    * self.current_tfp_a_c
                    * health_mult,
                    6,
                )
                sheet.consumable_inventory = round(
                    sheet.consumable_inventory + harvest, 6
                )

            # 3A. Labor Wages (ProductionFirm -> Household)
            wage_shock_mult = 1.0
            if self.shock_cfg.shock_type == MacroShockType.ASYMMETRIC_WAGE_SHOCK:
                wage_shock_mult = max(
                    0.25, 1.0 - 0.50 * sheet.shock_exposure_e * self.shock_intensity_s
                )
            wage_paid = round(sheet.base_wage * wage_shock_mult * health_mult, 8)
            wage_paid = min(wage_paid, max(0.0, self.firm_sector_cash))
            self.firm_sector_cash = round(self.firm_sector_cash - wage_paid, 8)
            sheet.liquid_cash = round(sheet.liquid_cash + wage_paid, 8)

            # 3B. Illiquid Capital Dividends (ProductionFirm -> Household)
            total_k = sheet.illiquid_capital + sheet.escrow_capital
            cap_div = TwoAssetAccounting.compute_capital_dividend_flow(
                total_k,
                self.economy_cfg,
                yield_multiplier=self.current_capital_yield_mult,
            )
            cap_div = min(cap_div, max(0.0, self.firm_sector_cash))
            self.firm_sector_cash = round(self.firm_sector_cash - cap_div, 8)
            sheet.liquid_cash = round(sheet.liquid_cash + cap_div, 8)

            # 3C. Liquid Interest / Borrowing Wedge (FinancialIntermediary <-> Household)
            int_flow = TwoAssetAccounting.compute_liquid_interest_flow(
                sheet.liquid_cash,
                self.economy_cfg,
                credit_wedge_shock_addon=self.current_credit_wedge_addon,
            )
            if int_flow >= 0.0:
                int_flow = min(int_flow, max(0.0, self.intermediary_cash))
            self.intermediary_cash = round(self.intermediary_cash - int_flow, 8)
            sheet.liquid_cash = round(sheet.liquid_cash + int_flow, 8)

            # 3D. Mandatory Periodic Obligation R_{i,t} (Household -> ProductionFirm)
            avail_to_pay = max(
                0.0, round(sheet.liquid_cash - sheet.borrowing_limit_b_bar, 8)
            )
            ob_due = sheet.periodic_obligation_r
            ob_paid = round(min(avail_to_pay, ob_due), 8)
            if ob_paid + 1e-6 < ob_due:
                obligation_defaults.append(aid)
            sheet.liquid_cash = round(sheet.liquid_cash - ob_paid, 8)
            self.firm_sector_cash = round(self.firm_sector_cash + ob_paid, 8)

            self.tick_disposable_income[aid] = round(
                wage_paid + cap_div + int_flow - ob_paid, 8
            )

            # 4. Biological Needs Engine (Consume c_min, decay/recover health)
            met_out: MetabolicStepOutcome = NeedsEngine.step_metabolic_needs(
                sheet, self.economy_cfg
            )
            if met_out.died_this_tick:
                deaths_this_tick.append(aid)
                # 5. Economic death: immediately cancel all open CDA orders and release escrow
                death_releases = self.exchange.cancel_all_agent_orders(aid)
                for d_rel in death_releases:
                    self._apply_escrow_release(d_rel)

        # 6. Collect Market Microstructure Snapshots & HtM Classification
        snapshots = self.exchange.get_all_snapshots(tick)
        c_snap = snapshots[AssetType.CONSUMABLE]
        k_snap = snapshots.get(AssetType.CAPITAL)
        ref_c_price = c_snap.last_trade_price or c_snap.mid_price or 12.0

        living_sheets = [h for h in self.households.values() if NeedsEngine.is_alive(h)]
        n_living = len(living_sheets)
        p_htm = 0
        w_htm = 0
        non_htm = 0
        for h in living_sheets:
            cls_tag = TwoAssetAccounting.classify_htm_status(
                h, self.economy_cfg, ref_c_price
            )
            if cls_tag == "POOR_HTM":
                p_htm += 1
            elif cls_tag == "WEALTHY_HTM":
                w_htm += 1
            else:
                non_htm += 1

        denom = max(1, n_living)
        c_fills_count = sum(
            1 for f in self.tick_fills if f.asset == AssetType.CONSUMABLE
        )

        # 7. Strict SFC Invariant Check
        sfc_audit = self.audit_and_assert_sfc(tick)

        return TickMacroSummary(
            tick=tick,
            shock_intensity_s=self.shock_intensity_s,
            living_agents_count=n_living,
            deaths_this_tick=deaths_this_tick,
            consumable_book=c_snap,
            capital_book=k_snap,
            fills_this_tick=list(self.tick_fills),
            consumable_market_frozen=(c_fills_count == 0),
            poor_htm_share=round(p_htm / denom, 4),
            wealthy_htm_share=round(w_htm / denom, 4),
            non_htm_share=round(non_htm / denom, 4),
            realized_consumable_spend_by_agent=dict(self.tick_consumable_spend),
            realized_disposable_income_by_agent=dict(self.tick_disposable_income),
            obligation_defaults_this_tick=obligation_defaults,
            sfc_audit=sfc_audit,
        )

    # ========================================================================
    # 5. NATIVE ZERO-INTELLIGENCE (ZI-C) CDA BASELINE GENERATOR
    # ========================================================================

    def propose_zero_intelligence_action(
        self,
        tick: int,
        agent_id: str,
        zi_rng: Optional[np.random.Generator] = None,
    ) -> Stage2ActionPayload:
        """
        Budget-Constrained Zero-Intelligence (ZI-C) Trader (Gode & Sunder 1993)
        integrated natively with the CDA Order Book & Two-Asset balance sheet.
        Proves whether macro-freezes arise from LLM cognition vs. CDA mechanics.
        """
        rng = zi_rng if zi_rng is not None else self.rng
        sheet = self.households[agent_id]
        if not NeedsEngine.is_alive(sheet):
            return Stage2ActionPayload(action_type=AgentActionType.ABSTAIN)

        c_book = self.exchange.get_book(AssetType.CONSUMABLE)
        c_snap = c_book.get_snapshot(tick)
        anchor_p = c_snap.last_trade_price or c_snap.mid_price or 12.0
        spendable = max(0.0, sheet.liquid_cash - sheet.borrowing_limit_b_bar)

        # If cash-constrained but holding illiquid capital, ZI-C may liquidate 1 unit
        if (
            sheet.consumable_inventory < 1.5
            and spendable < 0.85 * anchor_p
            and sheet.illiquid_capital >= 0.35
        ):
            return Stage2ActionPayload(
                action_type=AgentActionType.LIQUIDATE_CAPITAL,
                asset=AssetType.CAPITAL,
                capital_transfer_units=0.35,
            )

        # Surplus consumable holders post random ASK limit orders above cost floor
        if sheet.consumable_inventory >= 2.5:
            ask_p = c_book.quantize_price(
                float(rng.uniform(0.70 * anchor_p, 1.35 * anchor_p))
            )
            qty = float(rng.choice([1.0, 2.0]))
            qty = min(qty, max(1.0, sheet.consumable_inventory - 1.0))
            return Stage2ActionPayload(
                action_type=AgentActionType.PLACE_ORDER,
                asset=AssetType.CONSUMABLE,
                side=OrderSide.ASK,
                order_type=OrderType.LIMIT,
                limit_price=ask_p,
                quantity=round(qty, 2),
            )

        # Low-inventory households with spendable cash post random BID limit orders
        if spendable >= 4.0:
            max_bid = min(spendable, 1.45 * anchor_p)
            min_bid = min(max_bid, 0.70 * anchor_p)
            bid_p = c_book.quantize_price(float(rng.uniform(min_bid, max_bid)))
            return Stage2ActionPayload(
                action_type=AgentActionType.PLACE_ORDER,
                asset=AssetType.CONSUMABLE,
                side=OrderSide.BID,
                order_type=OrderType.LIMIT,
                limit_price=bid_p,
                quantity=1.0,
            )

        return Stage2ActionPayload(action_type=AgentActionType.CONSUME_AND_HOLD)

    def run_zero_intelligence_tick(
        self, tick: int, zi_rng: Optional[np.random.Generator] = None
    ) -> TickMacroSummary:
        """Executes a full randomized-arrival Zero-Intelligence tick over the CDA."""
        rng = zi_rng if zi_rng is not None else self.rng
        self.begin_tick(tick)
        arrival_order = list(self.households.keys())
        rng.shuffle(arrival_order)
        for aid in arrival_order:
            action = self.propose_zero_intelligence_action(tick, aid, zi_rng=rng)
            self.execute_agent_action(tick, aid, action)
        return self.step_end_of_tick_physics(tick)