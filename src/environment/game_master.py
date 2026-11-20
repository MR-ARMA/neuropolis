from __future__ import annotations

import json
import math
import random
import sqlite3
import uuid
from typing import Any, Dict, List, Optional, Tuple

from src.core.contracts import (
    EconomicActionIntent,
    GameMasterVerdict,
    LedgerTransactionReceipt,
    MacroRegimeState,
    SimulationConfig,
    normalize_item_id,
)


class AccountingAndLawEngine:
    """
    M8 & M9: Remediated Stock-Flow Consistent (SFC) Game Master.
    Implements Reviewer #2 Required Mechanics M1 through M13:
      - M1: Explicit CRRA Period Utility u_{i,t} & Cumulative Discounted Payoff U_i.
      - M2: Idiosyncratic shock exposure e_i in [0, 1] per agent.
      - M3: Contractual periodic obligations R_{i,t} & default penalty phi(default).
      - M4: Two-Asset HANK budget constraint with borrowing limit b_bar and wedge kappa.
      - M5: Endogenous market-clearing price adjustment p_{t+1} = p_t * [1 + phi*(D_t - S_t)/(D_t + S_t)].
      - M7: Latent macro state theta_t in {-1, +1} & agent-specific noisy private signals
            theta_tilde_{i,t} = theta_t + N(0, sigma_i^2) so TMS expert advice carries real value of information.
      - M13: Optional seed-based permutation of agent identities across endowments.
    """

    def __init__(self, db_path: str, config: SimulationConfig) -> None:
        self.db_path = db_path
        self.config = config
        self.omega = config.orthogonal_clamping
        self.hank_spec = config.hank_and_crisis_spec
        self.trust_spec = config.trust_and_market_spec
        self.fiscal_spec = config.fiscal_and_legal_spec
        self.rng = random.Random(config.random_seed)

        self.initial_money_supply_m0: float = 0.0
        self.private_signals_theta_tilde: Dict[str, float] = {}
        self.tick_order_flow: Dict[str, Dict[str, float]] = {
            "CONSUMABLE_NECESSITY": {"buy": 0.0, "sell": 0.0},
            "PRODUCTIVE_CAPITAL_ASSET": {"buy": 0.0, "sell": 0.0},
        }

        self.macro_state = MacroRegimeState(
            tick=0,
            active_crisis=self.hank_spec.active_crisis,
            crisis_intensity_s=0.0,
            latent_macro_state_theta=1.0,
            consumable_tfp_ac=self.hank_spec.prod_tfp_baseline_ac,
            reference_consumable_price_pc=self.hank_spec.reference_consumable_price_pc,
            reference_capital_price_qk=self.hank_spec.reference_capital_price_qk,
            capital_yield_rk=self.hank_spec.capital_yield_rate_rk,
            liquid_rate_rb=self.hank_spec.liquid_return_rate_rb,
            borrowing_wedge_kappa=self.hank_spec.borrowing_wedge_kappa,
            borrowing_limit_scale=1.0,
            kmv_chi_1_current=(
                self.hank_spec.kmv_chi_1
                if self.omega.omega_d1_scarcity_and_hank
                else 0.0
            ),
            generalized_social_trust=(
                self.trust_spec.initial_generalized_trust
                if self.omega.omega_d2_trust_and_parochialism
                else 0.50
            ),
            institutional_trust=(
                self.trust_spec.initial_institutional_trust
                if self.omega.omega_d2_trust_and_parochialism
                else 0.50
            ),
            public_treasury_cash=self.fiscal_spec.initial_public_treasury_cash,
            firm_sector_cash=self.fiscal_spec.initial_firm_sector_cash,
        )
        self._init_tables_and_conservation_anchor()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_tables_and_conservation_anchor(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS agents_external_ledger (
                    agent_id TEXT PRIMARY KEY,
                    occupation_role TEXT NOT NULL,
                    wage_tier TEXT NOT NULL DEFAULT 'MID',
                    liquid_wealth_cash REAL NOT NULL,
                    credit_limit_b_bar REAL NOT NULL DEFAULT -12.0,
                    credit_score REAL NOT NULL DEFAULT 720.0,
                    inventory_json TEXT NOT NULL,
                    base_wage REAL NOT NULL,
                    periodic_obligation_r REAL NOT NULL DEFAULT 6.0,
                    shock_exposure_e REAL NOT NULL DEFAULT 0.50,
                    base_consumable_prod_q REAL NOT NULL DEFAULT 0.0,
                    last_disposable_income REAL NOT NULL DEFAULT 0.0,
                    last_consumption_exp REAL NOT NULL DEFAULT 0.0,
                    last_period_utility_u REAL NOT NULL DEFAULT 0.0,
                    cumulative_utility_u REAL NOT NULL DEFAULT 0.0,
                    prior_violations_count INTEGER NOT NULL DEFAULT 0,
                    last_sanction_points INTEGER NOT NULL DEFAULT 0,
                    legal_status TEXT NOT NULL DEFAULT 'LAW_ABIDING'
                );

                CREATE TABLE IF NOT EXISTS macro_institutions (
                    institution_id TEXT PRIMARY KEY,
                    cash_balance REAL NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS acid_transaction_ledger (
                    tx_id TEXT PRIMARY KEY,
                    tick INTEGER NOT NULL,
                    initiator_id TEXT NOT NULL,
                    responder_id TEXT NOT NULL,
                    buyer_id TEXT NOT NULL,
                    seller_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    unit_price REAL NOT NULL,
                    gross_amount REAL NOT NULL,
                    tax_withheld REAL NOT NULL,
                    kmv_adjustment_cost REAL NOT NULL DEFAULT 0.0,
                    seller_net_received REAL NOT NULL,
                    kkt_gouging_index REAL NOT NULL DEFAULT 0.0,
                    is_fire_sale INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    error_code TEXT,
                    diagnostic_message TEXT NOT NULL,
                    global_money_supply_error REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS fiscal_and_commons_log (
                    tick INTEGER NOT NULL,
                    agent_id TEXT NOT NULL,
                    gross_income REAL NOT NULL,
                    obligation_paid_r REAL NOT NULL,
                    saez_income_tax REAL NOT NULL,
                    wealth_tax REAL NOT NULL,
                    ubi_and_targeted_transfer REAL NOT NULL,
                    pg_contribution REAL NOT NULL,
                    pg_dividend_received REAL NOT NULL,
                    audited INTEGER NOT NULL,
                    ostrom_sanction_points INTEGER NOT NULL,
                    sanction_penalty_cash REAL NOT NULL,
                    period_utility_u REAL NOT NULL,
                    PRIMARY KEY (tick, agent_id)
                );

                CREATE TABLE IF NOT EXISTS macro_timeseries_log (
                    tick INTEGER PRIMARY KEY,
                    active_crisis TEXT NOT NULL,
                    decision_mode TEXT NOT NULL,
                    state_channel TEXT NOT NULL,
                    steering_method TEXT NOT NULL,
                    llm_valid_rate REAL NOT NULL,
                    action_diversity_entropy REAL NOT NULL,
                    mean_realized_utility_u REAL NOT NULL,
                    trust_index REAL NOT NULL,
                    generalized_trust REAL NOT NULL,
                    institutional_trust REAL NOT NULL,
                    wealth_gini REAL NOT NULL,
                    income_gini REAL NOT NULL,
                    htm_share REAL NOT NULL,
                    realized_mpc REAL NOT NULL,
                    mpc_low_liq REAL NOT NULL,
                    mpc_high_liq REAL NOT NULL,
                    distress_sale_ratio REAL NOT NULL,
                    avg_food_price REAL NOT NULL,
                    avg_capital_price REAL NOT NULL,
                    trade_modularity_q REAL NOT NULL,
                    price_dispersion_r REAL NOT NULL,
                    unfairness_rejection_rate REAL NOT NULL,
                    cumul_volume REAL NOT NULL,
                    opinion_dispersion_std REAL NOT NULL,
                    cascade_share REAL NOT NULL,
                    pg_provision_rate REAL NOT NULL,
                    free_rider_share REAL NOT NULL,
                    public_treasury REAL NOT NULL,
                    total_sanction_points INTEGER NOT NULL,
                    tms_queries INTEGER NOT NULL,
                    tms_auroc REAL NOT NULL,
                    conservation_error REAL NOT NULL
                );
                """
            )

            cur = conn.execute("SELECT COUNT(*) AS cnt FROM agents_external_ledger;")
            if cur.fetchone()["cnt"] == 0:
                for ag in self.config.agents:
                    b_bar = (
                        self.hank_spec.borrowing_limit_ratio_income * ag.base_income_wage
                    )
                    init_htm_cutoff = (
                        1.65
                        * self.hank_spec.reference_consumable_price_pc
                        * max(1.0, self.hank_spec.subsistence_min_c)
                        + float(ag.periodic_obligation_r)
                        + 32.0
                    )
                    init_cash = float(ag.liquid_wealth_cash)
                    if ag.wage_tier == "LOW" and init_cash >= init_htm_cutoff - 34.0:
                        init_cash = round(max(18.0, init_htm_cutoff - 34.0), 2)
                    inv_map = {
                        "CONSUMABLE_NECESSITY": round(ag.get_consumable_inv(), 4),
                        "PRODUCTIVE_CAPITAL_ASSET": round(ag.get_capital_inv(), 4),
                    }
                    conn.execute(
                        """
                        INSERT INTO agents_external_ledger (
                            agent_id, occupation_role, wage_tier, liquid_wealth_cash,
                            credit_limit_b_bar, credit_score, inventory_json,
                            base_wage, periodic_obligation_r, shock_exposure_e,
                            base_consumable_prod_q, last_disposable_income,
                            last_consumption_exp, last_period_utility_u,
                            cumulative_utility_u, prior_violations_count,
                            last_sanction_points, legal_status
                        ) VALUES (?, ?, ?, ?, ?, 720.0, ?, ?, ?, ?, ?, ?, ?, 0.0, 0.0, 0, 0, 'LAW_ABIDING');
                        """,
                        (
                            ag.agent_id,
                            ag.occupation_role,
                            ag.wage_tier,
                            init_cash,
                            float(b_bar),
                            json.dumps(inv_map),
                            float(ag.base_income_wage),
                            float(ag.periodic_obligation_r),
                            float(ag.shock_exposure_e),
                            float(ag.base_consumable_prod_q),
                            float(ag.base_income_wage - ag.periodic_obligation_r),
                            float(self.hank_spec.reference_consumable_price_pc),
                        ),
                    )

            cur_inst = conn.execute("SELECT COUNT(*) AS cnt FROM macro_institutions;")
            if cur_inst.fetchone()["cnt"] == 0:
                conn.execute(
                    "INSERT INTO macro_institutions VALUES ('PublicTreasury', ?, ?);",
                    (
                        float(self.fiscal_spec.initial_public_treasury_cash),
                        json.dumps({"role": "Fiscal_State_and_Commons"}),
                    ),
                )
                conn.execute(
                    "INSERT INTO macro_institutions VALUES ('AggregateFirmSector', ?, ?);",
                    (
                        float(self.fiscal_spec.initial_firm_sector_cash),
                        json.dumps({"role": "Production_Wages_Obligations_and_Capital_Book"}),
                    ),
                )
            conn.commit()

        self.initial_money_supply_m0 = self._compute_current_total_money()
        self._generate_latent_signals_for_tick(tick=0)

    # ========================================================================
    # STOCK-FLOW CONSISTENT (SFC) CONSERVATION & LEDGER QUERIES
    # ========================================================================

    def _compute_current_total_money(self, conn: Optional[sqlite3.Connection] = None) -> float:
        close_after = False
        if conn is None:
            conn = self._connect()
            close_after = True
        try:
            row_ag = conn.execute(
                "SELECT COALESCE(SUM(liquid_wealth_cash), 0.0) AS s FROM agents_external_ledger;"
            ).fetchone()
            row_in = conn.execute(
                "SELECT COALESCE(SUM(cash_balance), 0.0) AS s FROM macro_institutions;"
            ).fetchone()
            return float(row_ag["s"]) + float(row_in["s"])
        finally:
            if close_after:
                conn.close()

    def get_conservation_error(self, conn: Optional[sqlite3.Connection] = None) -> float:
        return abs(self._compute_current_total_money(conn) - self.initial_money_supply_m0)

    def get_public_treasury_balance(self) -> float:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cash_balance FROM macro_institutions WHERE institution_id = 'PublicTreasury';"
            ).fetchone()
            return float(row["cash_balance"]) if row else 0.0

    def get_agent_ledger(self, agent_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM agents_external_ledger WHERE agent_id = ?;",
                (agent_id,),
            ).fetchone()
            if not row:
                raise KeyError(f"Unknown agent_id {agent_id}")
            data = dict(row)
            inv = json.loads(data["inventory_json"])
            c_qty = float(inv.get("CONSUMABLE_NECESSITY", inv.get("FOOD", 0.0)))
            k_qty = float(inv.get("PRODUCTIVE_CAPITAL_ASSET", inv.get("TECH", 0.0)))
            data["inventory"] = {
                "CONSUMABLE_NECESSITY": c_qty,
                "PRODUCTIVE_CAPITAL_ASSET": k_qty,
                "FOOD": c_qty,
                "TECH": k_qty,
            }
            return data

    def get_all_agent_ledgers(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT agent_id FROM agents_external_ledger ORDER BY agent_id;"
            ).fetchall()
        return [self.get_agent_ledger(r["agent_id"]) for r in rows]

    # ========================================================================
    # M7: LATENT MACRO STATE (theta_t) & AGENT-SPECIFIC NOISY SIGNALS
    # ========================================================================

    def _generate_latent_signals_for_tick(self, tick: int) -> None:
        """
        Generates true latent state theta_t in {-1.0, +1.0} (representing whether
        upcoming capital returns/supply conditions are favorable or adverse) and
        draws each agent's private noisy observation theta_tilde_{i,t} = theta_t + N(0, sigma_i^2).
        """
        in_crisis = (
            self.hank_spec.active_crisis != "NONE"
            and self.hank_spec.crisis_start_tick
            <= tick
            < self.hank_spec.crisis_start_tick + self.hank_spec.crisis_duration_ticks
        )
        theta_true = -1.0 if in_crisis else 1.0
        self.macro_state.latent_macro_state_theta = theta_true

        self.private_signals_theta_tilde.clear()
        for ag in self.config.agents:
            is_cap_specialist = ag.tms_specialty == "PRODUCTIVE_CAPITAL_ASSET"
            sigma_i = (
                self.hank_spec.latent_signal_specialist_sigma
                if is_cap_specialist
                else self.hank_spec.latent_signal_novice_sigma
            )
            noisy_obs = theta_true + self.rng.gauss(0.0, sigma_i)
            self.private_signals_theta_tilde[ag.agent_id] = round(noisy_obs, 3)

    def get_agent_private_signal(self, agent_id: str) -> Tuple[float, bool]:
        """
        Returns `(theta_tilde_{i,t}, is_high_noise_novice)`.
        """
        sig = float(self.private_signals_theta_tilde.get(agent_id, 0.0))
        ag_cfg = next(
            (a for a in self.config.agents if a.agent_id == agent_id), None
        )
        is_novice = (
            ag_cfg.tms_specialty != "PRODUCTIVE_CAPITAL_ASSET"
            if ag_cfg is not None
            else True
        )
        return (sig, is_novice)

    # ========================================================================
    # KAPLAN-MOLL-VIOLANTE (2018) CONVEX ADJUSTMENT COST & SCARCITY
    # ========================================================================

    def compute_kmv_adjustment_cost(
        self, delta_a_units: float, current_a_units: float
    ) -> float:
        if not self.omega.omega_d1_scarcity_and_hank:
            return 0.0
        abs_d = abs(delta_a_units)
        if abs_d <= 1e-9:
            return 0.0
        a_denom = max(float(current_a_units), self.hank_spec.kmv_a_min)
        chi_0 = self.hank_spec.kmv_chi_0
        chi_1 = self.macro_state.kmv_chi_1_current
        chi_2 = self.hank_spec.kmv_chi_2

        unit_cost = (
            chi_0 * abs_d
            + (chi_1 / (1.0 + chi_2)) * ((abs_d / a_denom) ** (1.0 + chi_2)) * a_denom
        )
        cash_fee = unit_cost * (0.15 * self.macro_state.reference_capital_price_qk)
        return round(float(cash_fee), 4)

    def compute_agent_scarcity_and_present_bias(
        self, agent_id: str
    ) -> Tuple[float, float, float]:
        if not self.omega.omega_d1_scarcity_and_hank:
            return (0.0, 1.0, self.hank_spec.long_run_discount_delta)

        ledger = self.get_agent_ledger(agent_id)
        b_it = float(ledger["liquid_wealth_cash"])
        obligation_r = float(ledger["periodic_obligation_r"])
        b_bar = float(ledger["credit_limit_b_bar"]) * self.macro_state.borrowing_limit_scale
        c_it = float(ledger["inventory"]["CONSUMABLE_NECESSITY"])
        c_min = self.hank_spec.subsistence_min_c
        b_safe = self.hank_spec.safe_cash_buffer_b + obligation_r

        c_safe = 2.0 * c_min
        consumable_gap = max(0.0, c_safe - c_it) / max(c_safe, 1e-6)
        cash_gap = max(0.0, (b_safe - b_it) / max(b_safe - b_bar, 1e-6))
        f_it = max(0.0, min(1.0, 0.55 * consumable_gap + 0.45 * cash_gap))

        subsistence_cost = max(
            1.0,
            self.macro_state.reference_consumable_price_pc * c_min + obligation_r,
        )
        liquidity_ratio = max(-4.0, min(8.0, b_it / subsistence_cost))
        beta_min = self.hank_spec.present_bias_beta_min
        eta_beta = self.hank_spec.present_bias_eta
        beta_i = beta_min + (1.0 - beta_min) / (
            1.0 + math.exp(-eta_beta * (liquidity_ratio - 1.0))
        )
        tilde_delta = beta_i * self.hank_spec.long_run_discount_delta
        return (float(f_it), float(beta_i), float(tilde_delta))

    # ========================================================================
    # PARAMETERIZED 4-CRISIS SHOCK ENGINE & ENDOGENOUS PRICE CLEARING
    # ========================================================================

    def step_crisis_regime(self, tick: int) -> MacroRegimeState:
        self.macro_state.tick = tick
        self._generate_latent_signals_for_tick(tick=tick)

        # Reset intra-tick order flow counters
        self.tick_order_flow = {
            "CONSUMABLE_NECESSITY": {"buy": 0.0, "sell": 0.0},
            "PRODUCTIVE_CAPITAL_ASSET": {"buy": 0.0, "sell": 0.0},
        }

        crisis_type = self.hank_spec.active_crisis
        if not self.omega.omega_d1_scarcity_and_hank or crisis_type == "NONE":
            self.macro_state.crisis_intensity_s = 0.0
            return self.macro_state

        in_window = (
            self.hank_spec.crisis_start_tick
            <= tick
            < self.hank_spec.crisis_start_tick + self.hank_spec.crisis_duration_ticks
        )
        rho = self.hank_spec.crisis_persistence_rho
        sigma = self.hank_spec.crisis_severity_sigma
        prev_s = self.macro_state.crisis_intensity_s
        s_t = min(1.0, rho * prev_s + (sigma if in_window else 0.0))
        self.macro_state.crisis_intensity_s = round(s_t, 4)

        if crisis_type == "STAGFLATION_SUBSISTENCE_SHOCK":
            self.macro_state.consumable_tfp_ac = round(
                self.hank_spec.prod_tfp_baseline_ac * max(0.45, 1.0 - 0.55 * s_t), 4
            )
        elif crisis_type == "ASYMMETRIC_UNEMPLOYMENT_SHOCK":
            pass
        elif crisis_type == "ASSET_VALUATION_CRASH":
            self.macro_state.capital_yield_rk = round(
                self.hank_spec.capital_yield_rate_rk * max(0.15, 1.0 - 0.80 * s_t), 4
            )
        elif crisis_type == "LIQUIDITY_CREDIT_CRUNCH":
            self.macro_state.borrowing_limit_scale = round(max(0.0, 1.0 - 0.95 * s_t), 4)
            self.macro_state.borrowing_wedge_kappa = round(
                self.hank_spec.borrowing_wedge_kappa + 0.05 * s_t, 4
            )
            self.macro_state.kmv_chi_1_current = round(
                self.hank_spec.kmv_chi_1 * (1.0 + 1.0 * s_t), 4
            )
        return self.macro_state

    # ========================================================================
    # 5-GATE BILATERAL TRADE ARBITRATION & DOUBLE-ENTRY SETTLEMENT
    # ========================================================================

    def arbitrate_and_settle(
        self,
        tick: int,
        initiator_id: str,
        responder_id: str,
        intent: EconomicActionIntent,
        responder_consent: str,
        kkt_gouging_index: float = 0.0,
    ) -> GameMasterVerdict:
        tx_id = f"tx_{tick:03d}_{uuid.uuid4().hex[:8]}"
        item_id = normalize_item_id(intent.item_id)
        qty = float(intent.quantity)
        unit_price = float(intent.unit_price)
        gross = round(qty * unit_price, 4)

        # Record order-flow demand/supply intent for endogenous price formation (M5)
        if intent.action_type == "BUY" and qty > 0:
            self.tick_order_flow[item_id]["buy"] += qty
        elif intent.action_type == "SELL" and qty > 0:
            self.tick_order_flow[item_id]["sell"] += qty

        if intent.action_type == "HOLD_OR_WORK":
            return self._record_rejected_tx(
                tx_id,
                tick,
                initiator_id,
                responder_id,
                initiator_id,
                responder_id,
                intent.action_type,
                item_id,
                qty,
                unit_price,
                gross,
                "NO_TRADE_REQUESTED",
                "Initiator selected HOLD_OR_WORK.",
                kkt_gouging_index,
                intent.is_distress_fire_sale,
            )

        buyer_id = initiator_id if intent.action_type == "BUY" else responder_id
        seller_id = responder_id if intent.action_type == "BUY" else initiator_id

        if responder_consent != "ACCEPT":
            return self._record_rejected_tx(
                tx_id,
                tick,
                initiator_id,
                responder_id,
                buyer_id,
                seller_id,
                intent.action_type,
                item_id,
                qty,
                unit_price,
                gross,
                "COUNTERPARTY_REJECTED",
                f"Responder {responder_id} emitted {responder_consent}.",
                kkt_gouging_index,
                intent.is_distress_fire_sale,
            )

        if (
            not math.isfinite(qty)
            or not math.isfinite(unit_price)
            or qty <= 0.0
            or unit_price <= 0.0
        ):
            return self._record_rejected_tx(
                tx_id,
                tick,
                initiator_id,
                responder_id,
                buyer_id,
                seller_id,
                intent.action_type,
                item_id,
                qty,
                unit_price,
                gross,
                "INVALID_AMOUNT",
                f"Invalid quantity ({qty}) or unit_price ({unit_price}).",
                kkt_gouging_index,
                intent.is_distress_fire_sale,
            )

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE TRANSACTION;")
            try:
                buyer_row = conn.execute(
                    "SELECT * FROM agents_external_ledger WHERE agent_id = ?;",
                    (buyer_id,),
                ).fetchone()
                seller_row = conn.execute(
                    "SELECT * FROM agents_external_ledger WHERE agent_id = ?;",
                    (seller_id,),
                ).fetchone()

                if (
                    buyer_row["legal_status"] == "IMPRISONED"
                    or seller_row["legal_status"] == "IMPRISONED"
                ):
                    conn.rollback()
                    return self._record_rejected_tx(
                        tx_id,
                        tick,
                        initiator_id,
                        responder_id,
                        buyer_id,
                        seller_id,
                        intent.action_type,
                        item_id,
                        qty,
                        unit_price,
                        gross,
                        "LEGAL_RESTRICTION",
                        "Party legally restricted.",
                        kkt_gouging_index,
                        intent.is_distress_fire_sale,
                    )

                buyer_inv = json.loads(buyer_row["inventory_json"])
                seller_inv = json.loads(seller_row["inventory_json"])
                seller_avail = float(seller_inv.get(item_id, 0.0))

                if seller_avail + 1e-9 < qty:
                    conn.rollback()
                    return self._record_rejected_tx(
                        tx_id,
                        tick,
                        initiator_id,
                        responder_id,
                        buyer_id,
                        seller_id,
                        intent.action_type,
                        item_id,
                        qty,
                        unit_price,
                        gross,
                        "INSUFFICIENT_INVENTORY",
                        f"Seller {seller_id} has {seller_avail:.2f} {item_id}, needs {qty:.2f}.",
                        kkt_gouging_index,
                        intent.is_distress_fire_sale,
                    )

                seller_kmv_cost = 0.0
                buyer_kmv_cost = 0.0
                if item_id == "PRODUCTIVE_CAPITAL_ASSET":
                    buyer_avail_k = float(buyer_inv.get(item_id, 0.0))
                    seller_kmv_cost = round(
                        0.50 * self.compute_kmv_adjustment_cost(qty, seller_avail), 4
                    )
                    buyer_kmv_cost = round(
                        0.50 * self.compute_kmv_adjustment_cost(qty, buyer_avail_k), 4
                    )
                kmv_cost = round(seller_kmv_cost + buyer_kmv_cost, 4)

                buyer_cash = float(buyer_row["liquid_wealth_cash"])
                b_bar = (
                    float(buyer_row["credit_limit_b_bar"])
                    * self.macro_state.borrowing_limit_scale
                    if self.omega.omega_d1_scarcity_and_hank
                    else 0.0
                )
                max_spendable = buyer_cash - b_bar
                if max_spendable + 1e-9 < (gross + buyer_kmv_cost):
                    conn.rollback()
                    return self._record_rejected_tx(
                        tx_id,
                        tick,
                        initiator_id,
                        responder_id,
                        buyer_id,
                        seller_id,
                        intent.action_type,
                        item_id,
                        qty,
                        unit_price,
                        gross,
                        "INSUFFICIENT_FUNDS",
                        f"Buyer {buyer_id} spendable=${max_spendable:.2f} < cost=${gross + buyer_kmv_cost:.2f}.",
                        kkt_gouging_index,
                        intent.is_distress_fire_sale,
                    )

                sales_tax_rate = (
                    self.fiscal_spec.sales_tax_rate
                    if self.omega.omega_d4_fiscal_and_legal_state
                    else 0.0
                )
                tax_withheld = round(gross * sales_tax_rate, 4)
                seller_kmv_cost = min(seller_kmv_cost, round(gross - tax_withheld, 4))
                kmv_cost = round(seller_kmv_cost + buyer_kmv_cost, 4)
                seller_net = round(gross - tax_withheld - seller_kmv_cost, 4)

                new_buyer_cash = round(buyer_cash - gross - buyer_kmv_cost, 4)
                new_seller_cash = round(
                    float(seller_row["liquid_wealth_cash"]) + seller_net, 4
                )

                buyer_inv[item_id] = round(float(buyer_inv.get(item_id, 0.0)) + qty, 4)
                seller_inv[item_id] = round(seller_avail - qty, 4)

                new_buyer_c_exp = float(buyer_row["last_consumption_exp"]) + (
                    gross if item_id == "CONSUMABLE_NECESSITY" else 0.0
                )

                conn.execute(
                    """
                    UPDATE agents_external_ledger
                    SET liquid_wealth_cash = ?, inventory_json = ?, last_consumption_exp = ?
                    WHERE agent_id = ?;
                    """,
                    (new_buyer_cash, json.dumps(buyer_inv), new_buyer_c_exp, buyer_id),
                )
                conn.execute(
                    """
                    UPDATE agents_external_ledger
                    SET liquid_wealth_cash = ?, inventory_json = ?
                    WHERE agent_id = ?;
                    """,
                    (new_seller_cash, json.dumps(seller_inv), seller_id),
                )

                conn.execute(
                    """
                    UPDATE macro_institutions
                    SET cash_balance = ROUND(cash_balance + ?, 4)
                    WHERE institution_id = 'PublicTreasury';
                    """,
                    (tax_withheld,),
                )
                conn.execute(
                    """
                    UPDATE macro_institutions
                    SET cash_balance = ROUND(cash_balance + ?, 4)
                    WHERE institution_id = 'AggregateFirmSector';
                    """,
                    (kmv_cost,),
                )

                sfc_err = self.get_conservation_error(conn)
                assert sfc_err < 1e-6, f"SFC violation in trade {tx_id}: {sfc_err}"

                conn.execute(
                    """
                    INSERT INTO acid_transaction_ledger VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        'APPROVED', NULL, 'Settled via ACID Double-Entry', ?
                    );
                    """,
                    (
                        tx_id,
                        tick,
                        initiator_id,
                        responder_id,
                        buyer_id,
                        seller_id,
                        intent.action_type,
                        item_id,
                        qty,
                        unit_price,
                        gross,
                        tax_withheld,
                        kmv_cost,
                        seller_net,
                        kkt_gouging_index,
                        int(intent.is_distress_fire_sale),
                        sfc_err,
                    ),
                )
                conn.commit()

                receipt = LedgerTransactionReceipt(
                    tx_id=tx_id,
                    tick=tick,
                    initiator_id=initiator_id,
                    responder_id=responder_id,
                    buyer_id=buyer_id,
                    seller_id=seller_id,
                    action_type=intent.action_type,
                    item_id=item_id,
                    quantity=qty,
                    unit_price=unit_price,
                    gross_amount=gross,
                    tax_withheld=tax_withheld,
                    kmv_adjustment_cost=kmv_cost,
                    seller_net_received=seller_net,
                    kkt_gouging_index=kkt_gouging_index,
                    is_fire_sale=intent.is_distress_fire_sale,
                    global_money_supply_error=sfc_err,
                )
                return GameMasterVerdict(
                    status="APPROVED",
                    error_code=None,
                    diagnostic_message=f"APPROVED: {buyer_id} bought {qty:.1f} {item_id} from {seller_id} @ ${unit_price:.2f}.",
                    ledger_receipt=receipt,
                )
            except Exception:
                conn.rollback()
                raise

    def _record_rejected_tx(
        self,
        tx_id: str,
        tick: int,
        initiator_id: str,
        responder_id: str,
        buyer_id: str,
        seller_id: str,
        action_type: str,
        item_id: str,
        qty: float,
        unit_price: float,
        gross: float,
        error_code: str,
        diag_msg: str,
        kkt_gouging_index: float,
        is_fire_sale: bool,
    ) -> GameMasterVerdict:
        sfc_err = self.get_conservation_error()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO acid_transaction_ledger VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, 0.0, 0.0, ?, ?,
                    'REJECTED', ?, ?, ?
                );
                """,
                (
                    tx_id,
                    tick,
                    initiator_id,
                    responder_id,
                    buyer_id,
                    seller_id,
                    action_type,
                    item_id,
                    qty,
                    unit_price,
                    gross,
                    kkt_gouging_index,
                    int(is_fire_sale),
                    error_code,
                    diag_msg,
                    sfc_err,
                ),
            )
            conn.commit()
        return GameMasterVerdict(
            status="REJECTED",
            error_code=error_code,
            diagnostic_message=diag_msg,
            ledger_receipt=None,
        )

    # ========================================================================
    # END-OF-TICK HANK PRODUCTION, OBLIGATIONS R_{i,t}, UTILITY & OSTROM
    # ========================================================================

    def execute_end_of_tick_upkeep_and_fiscal_state(
        self,
        tick: int,
        pg_contributions: Optional[Dict[str, float]] = None,
        tax_evasion_intents: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        pg_contributions = pg_contributions or {}
        tax_evasion_intents = tax_evasion_intents or {}

        s_t = self.macro_state.crisis_intensity_s
        crisis_type = self.hank_spec.active_crisis

        saez_top_rate = (
            self.fiscal_spec.compute_saez_optimal_top_rate()
            if self.omega.omega_d4_fiscal_and_legal_state
            else 0.0
        )
        mid_tax_rate = (
            self.fiscal_spec.saez_mid_income_tax_rate
            if self.omega.omega_d4_fiscal_and_legal_state
            else 0.0
        )
        wealth_tax_rate = (
            self.fiscal_spec.wealth_tax_rate
            if self.omega.omega_d4_fiscal_and_legal_state
            else 0.0
        )
        subsistence_floor_y = (
            self.macro_state.reference_consumable_price_pc
            * self.hank_spec.subsistence_min_c
        )

        upkeep_report: Dict[str, Any] = {
            "starved_agents": [],
            "defaulted_agents": [],
            "harvested_units": {},
            "sanctioned_agents": {},
            "pg_provision_rate": 1.0,
            "free_rider_share": 0.0,
            "total_sanction_points": 0,
            "saez_top_rate": saez_top_rate,
            "mean_period_utility_u": 0.0,
        }

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE TRANSACTION;")
            try:
                rows = conn.execute(
                    "SELECT * FROM agents_external_ledger ORDER BY agent_id;"
                ).fetchall()
                n_agents = len(rows)

                firm_cash = float(
                    conn.execute(
                        "SELECT cash_balance FROM macro_institutions WHERE institution_id = 'AggregateFirmSector';"
                    ).fetchone()["cash_balance"]
                )
                treasury_cash = float(
                    conn.execute(
                        "SELECT cash_balance FROM macro_institutions WHERE institution_id = 'PublicTreasury';"
                    ).fetchone()["cash_balance"]
                )

                agent_state_work: Dict[str, Dict[str, Any]] = {}
                total_supply_harvested = 0.0
                total_demand_eaten = 0.0

                for r in rows:
                    aid = r["agent_id"]
                    inv = json.loads(r["inventory_json"])
                    c_qty = float(inv.get("CONSUMABLE_NECESSITY", 0.0))
                    a_qty = float(inv.get("PRODUCTIVE_CAPITAL_ASSET", 0.0))
                    b_cash = float(r["liquid_wealth_cash"])
                    base_w = float(r["base_wage"])
                    obligation_r = float(r["periodic_obligation_r"])
                    exposure_e = float(r["shock_exposure_e"])
                    base_q = float(r["base_consumable_prod_q"])

                    # 1. Consumable Production Q_{i,t}
                    if base_q > 0.0:
                        q_harvest = self.macro_state.consumable_tfp_ac * (
                            base_q + self.hank_spec.prod_capital_elasticity_eta * a_qty
                        )
                        c_qty = round(c_qty + q_harvest, 4)
                        total_supply_harvested += q_harvest
                        upkeep_report["harvested_units"][aid] = round(q_harvest, 2)

                    # 2. Metabolic Depreciation (1.0 unit of CONSUMABLE_NECESSITY)
                    c_min = (
                        self.hank_spec.subsistence_min_c
                        if self.omega.omega_d1_scarcity_and_hank
                        else 0.0
                    )
                    ate_qty = 0.0
                    ate_full = True
                    if c_min > 0.0:
                        total_demand_eaten += c_min
                        if c_qty >= c_min:
                            c_qty = round(c_qty - c_min, 4)
                            ate_qty = c_min
                        else:
                            ate_qty = c_qty
                            c_qty = 0.0
                            ate_full = False
                            upkeep_report["starved_agents"].append(aid)
                    else:
                        ate_qty = 1.0

                    # 3. Wage Income (with heterogeneous shock exposure e_i * s_t) + Capital Dividends
                    wage_mult = 1.0
                    if (
                        self.omega.omega_d1_scarcity_and_hank
                        and crisis_type in ("ASYMMETRIC_UNEMPLOYMENT_SHOCK", "STAGFLATION_SUBSISTENCE_SHOCK")
                    ):
                        wage_mult = max(0.30, 1.0 - 0.32 * exposure_e * s_t)
                    realized_wage = round(base_w * wage_mult, 4)

                    # Latent state theta_t modulates capital dividend yield
                    theta_mult = (
                        1.15
                        if self.macro_state.latent_macro_state_theta > 0
                        else 0.65
                    )
                    capital_div = round(
                        a_qty
                        * self.macro_state.capital_yield_rk
                        * theta_mult
                        * self.macro_state.reference_capital_price_qk,
                        4,
                    )
                    if b_cash >= 0.0:
                        interest_flow = round(b_cash * self.macro_state.liquid_rate_rb, 4)
                    else:
                        borr_rate = (
                            self.macro_state.liquid_rate_rb
                            + self.macro_state.borrowing_wedge_kappa
                        )
                        interest_flow = round(b_cash * borr_rate, 4)

                    gross_income = round(realized_wage + capital_div + interest_flow, 4)
                    b_cash = round(b_cash + gross_income, 4)
                    firm_cash = round(firm_cash - gross_income, 4)

                    # 4. Contractual Periodic Obligation R_{i,t} (M3: housing/debt service)
                    ob_paid = round(obligation_r, 4) if self.omega.omega_d1_scarcity_and_hank else 0.0
                    b_cash = round(b_cash - ob_paid, 4)
                    firm_cash = round(firm_cash + ob_paid, 4)

                    # 5. Saez (2001) Progressive Income & Wealth Tax
                    saez_tax = 0.0
                    w_tax = 0.0
                    evaded_amount = 0.0
                    if self.omega.omega_d4_fiscal_and_legal_state:
                        y_high_cutoff = 2.2 * subsistence_floor_y
                        if gross_income <= subsistence_floor_y:
                            theoretical_y_tax = 0.0
                        elif gross_income <= y_high_cutoff:
                            theoretical_y_tax = mid_tax_rate * (
                                gross_income - subsistence_floor_y
                            )
                        else:
                            theoretical_y_tax = mid_tax_rate * (
                                y_high_cutoff - subsistence_floor_y
                            ) + saez_top_rate * (gross_income - y_high_cutoff)

                        net_worth = (
                            b_cash + a_qty * self.macro_state.reference_capital_price_qk
                        )
                        if net_worth > self.fiscal_spec.wealth_tax_threshold:
                            w_tax = wealth_tax_rate * (
                                net_worth - self.fiscal_spec.wealth_tax_threshold
                            )

                        evade_frac = max(
                            0.0, min(0.80, float(tax_evasion_intents.get(aid, 0.0)))
                        )
                        saez_tax = round(theoretical_y_tax * (1.0 - evade_frac), 4)
                        evaded_amount = round(theoretical_y_tax * evade_frac, 4)
                        w_tax = round(w_tax, 4)

                        total_agent_tax = round(saez_tax + w_tax, 4)
                        b_cash = round(b_cash - total_agent_tax, 4)
                        treasury_cash = round(treasury_cash + total_agent_tax, 4)

                    b_bar = (
                        float(r["credit_limit_b_bar"])
                        * self.macro_state.borrowing_limit_scale
                    )
                    c_gap = (
                        max(0.0, self.hank_spec.subsistence_min_c - c_qty)
                        if self.omega.omega_d1_scarcity_and_hank
                        else 0.0
                    )
                    b_gap = (
                        max(
                            0.0,
                            (self.hank_spec.safe_cash_buffer_b + ob_paid - b_cash)
                            / max(self.hank_spec.safe_cash_buffer_b + ob_paid - b_bar, 1.0),
                        )
                        if self.omega.omega_d1_scarcity_and_hank
                        else 0.0
                    )
                    f_it = max(0.0, min(1.0, 0.55 * c_gap + 0.45 * b_gap))

                    agent_state_work[aid] = {
                        "row": r,
                        "c_qty": c_qty,
                        "a_qty": a_qty,
                        "b_cash": b_cash,
                        "b_bar": b_bar,
                        "ate_qty": ate_qty,
                        "ate_full": ate_full,
                        "gross_income": gross_income,
                        "ob_paid": ob_paid,
                        "saez_tax": saez_tax,
                        "w_tax": w_tax,
                        "evaded_amount": evaded_amount,
                        "f_it": f_it,
                    }

                # 6. Redistributive Transfers d_{i,t} (UBI + Targeted Scarcity Relief)
                distributable_pool = (
                    round(0.45 * max(0.0, treasury_cash - 40.0), 4)
                    if self.omega.omega_d4_fiscal_and_legal_state
                    else 0.0
                )
                phi_univ = self.fiscal_spec.universal_vs_targeted_phi
                sum_f = sum(v["f_it"] for v in agent_state_work.values())

                for aid, st in agent_state_work.items():
                    if distributable_pool > 0.0:
                        ubi_part = phi_univ * (distributable_pool / n_agents)
                        if sum_f > 1e-6:
                            targ_part = (
                                (1.0 - phi_univ)
                                * distributable_pool
                                * (st["f_it"] / sum_f)
                            )
                        else:
                            targ_part = (1.0 - phi_univ) * (
                                distributable_pool / n_agents
                            )
                        d_it = round(ubi_part + targ_part, 4)
                    else:
                        d_it = 0.0
                    st["b_cash"] = round(st["b_cash"] + d_it, 4)
                    treasury_cash = round(treasury_cash - d_it, 4)
                    st["transfer_d"] = d_it

                # 7. Fehr & Gächter (2000) Public Goods & Ostrom Graduated Sanctions
                total_pg_contrib = 0.0
                total_ideal_endowment = 0.0
                free_riders_count = 0

                for aid, st in agent_state_work.items():
                    endowment_e = max(
                        2.0,
                        round(
                            self.fiscal_spec.public_goods_endowment_share
                            * max(st["gross_income"], 10.0),
                            4,
                        ),
                    )
                    total_ideal_endowment += endowment_e
                    if not self.omega.omega_d4_fiscal_and_legal_state:
                        g_it = 0.0
                    else:
                        if aid in pg_contributions and pg_contributions[aid] > 0.0:
                            g_it = max(
                                0.0, min(endowment_e, float(pg_contributions[aid]))
                            )
                        else:
                            ag_cfg = next(
                                a for a in self.config.agents if a.agent_id == aid
                            )
                            prior_sanctions = int(
                                st["row"]["prior_violations_count"]
                            )
                            coop_propensity = min(
                                1.0,
                                0.55 * self.macro_state.institutional_trust
                                + 0.55 * (ag_cfg.beta_advantageous + 0.50)
                                + 0.16 * min(3, prior_sanctions)
                                - 0.18 * st["f_it"],
                            )
                            g_it = round(endowment_e * max(0.0, coop_propensity), 4)

                    st["pg_endowment_e"] = endowment_e
                    st["pg_contrib_g"] = g_it
                    st["b_cash"] = round(st["b_cash"] - g_it, 4)
                    treasury_cash = round(treasury_cash + g_it, 4)
                    total_pg_contrib = round(total_pg_contrib + g_it, 4)

                mean_g_rate = sum(
                    v["pg_contrib_g"] / max(v["pg_endowment_e"], 1e-6)
                    for v in agent_state_work.values()
                ) / max(1, n_agents)
                mpcr_a = self.fiscal_spec.public_goods_mpcr_a
                total_pg_payout = (
                    round(mpcr_a * n_agents * total_pg_contrib, 4)
                    if self.omega.omega_d4_fiscal_and_legal_state
                    else 0.0
                )
                per_agent_pg_div = round(total_pg_payout / max(1, n_agents), 4)
                actual_pg_paid = round(per_agent_pg_div * n_agents, 4)

                surplus_from_firm = round(actual_pg_paid - total_pg_contrib, 4)
                treasury_cash = round(treasury_cash - total_pg_contrib, 4)
                firm_cash = round(firm_cash - surplus_from_firm, 4)

                utilities_this_tick: List[float] = []
                sigma_crra = self.hank_spec.crra_risk_aversion_sigma

                for aid, st in agent_state_work.items():
                    st["b_cash"] = round(st["b_cash"] + per_agent_pg_div, 4)
                    st["pg_dividend"] = per_agent_pg_div

                    prior_v = int(st["row"]["prior_violations_count"])
                    g_it = st["pg_contrib_g"]
                    own_g_rate = g_it / max(st["pg_endowment_e"], 1e-6)
                    rate_dev = max(0.0, mean_g_rate - own_g_rate)
                    is_free_rider = own_g_rate < 0.72
                    if is_free_rider:
                        free_riders_count += 1

                    audited = 0
                    p_points = 0
                    penalty_cash = 0.0

                    if self.omega.omega_d4_fiscal_and_legal_state and (
                        rate_dev > 0.06 or st["evaded_amount"] > 0.25
                    ):
                        p_audit = min(
                            0.95,
                            self.fiscal_spec.base_audit_probability
                            + 0.15 * prior_v,
                        )
                        if self.rng.random() < p_audit:
                            audited = 1
                            raw_pts = (
                                8.0 * rate_dev
                                + 2.5 * (st["evaded_amount"] / 5.0)
                                + 1.2 * prior_v
                            )
                            if prior_v == 0:
                                raw_pts *= max(
                                    0.25,
                                    1.0
                                    - self.fiscal_spec.ostrom_scarcity_mercy_gamma
                                    * st["f_it"],
                                )
                            p_points = max(1, min(10, int(round(raw_pts))))
                            stage1_payoff = max(
                                5.0,
                                st["gross_income"]
                                - st["saez_tax"]
                                + st["pg_dividend"],
                            )
                            penalty_cash = round(
                                stage1_payoff
                                * self.fiscal_spec.ostrom_sanction_penalty_per_point
                                * min(p_points, 10),
                                4,
                            )
                            st["b_cash"] = round(st["b_cash"] - penalty_cash, 4)
                            treasury_cash = round(treasury_cash + penalty_cash, 4)
                            prior_v += 1
                            upkeep_report["sanctioned_agents"][aid] = {
                                "points": p_points,
                                "penalty": penalty_cash,
                            }
                            upkeep_report["total_sanction_points"] += p_points

                    # M1 & M3: Compute Realized CRRA Period Utility u_{i,t} & Default Penalty
                    defaulted = (not st["ate_full"]) or (st["b_cash"] < st["b_bar"])
                    if defaulted:
                        upkeep_report["defaulted_agents"].append(aid)
                    eff_c = max(
                        0.20,
                        st["ate_qty"]
                        + 0.10
                        * max(0.0, st["b_cash"])
                        / max(1.0, self.macro_state.reference_consumable_price_pc),
                    )
                    if abs(sigma_crra - 1.0) < 1e-4:
                        crra_u = math.log(eff_c)
                    else:
                        crra_u = ((eff_c ** (1.0 - sigma_crra)) - 1.0) / (
                            1.0 - sigma_crra
                        )
                    period_u = round(
                        crra_u
                        - (
                            self.hank_spec.default_utility_penalty_phi
                            if defaulted
                            else 0.0
                        ),
                        4,
                    )
                    utilities_this_tick.append(period_u)
                    prev_cum_u = float(st["row"]["cumulative_utility_u"])
                    disc_factor = self.hank_spec.long_run_discount_delta ** tick
                    new_cum_u = round(prev_cum_u + disc_factor * period_u, 4)

                    old_cs = float(st["row"]["credit_score"])
                    new_cs = max(
                        300.0,
                        min(
                            850.0,
                            old_cs
                            + (4.0 if not defaulted else -22.0)
                            - 4.0 * p_points,
                        ),
                    )

                    disposable_y = round(
                        st["gross_income"]
                        - st["ob_paid"]
                        - st["saez_tax"]
                        - st["w_tax"]
                        + st["transfer_d"]
                        - st["pg_contrib_g"]
                        + st["pg_dividend"]
                        - penalty_cash,
                        4,
                    )
                    new_inv_json = json.dumps(
                        {
                            "CONSUMABLE_NECESSITY": st["c_qty"],
                            "PRODUCTIVE_CAPITAL_ASSET": st["a_qty"],
                        }
                    )
                    conn.execute(
                        """
                        UPDATE agents_external_ledger
                        SET liquid_wealth_cash = ?,
                            credit_score = ?,
                            inventory_json = ?,
                            last_disposable_income = ?,
                            last_period_utility_u = ?,
                            cumulative_utility_u = ?,
                            prior_violations_count = ?,
                            last_sanction_points = ?
                        WHERE agent_id = ?;
                        """,
                        (
                            st["b_cash"],
                            round(new_cs, 1),
                            new_inv_json,
                            disposable_y,
                            period_u,
                            new_cum_u,
                            prior_v,
                            p_points,
                            aid,
                        ),
                    )
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO fiscal_and_commons_log VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        );
                        """,
                        (
                            tick,
                            aid,
                            st["gross_income"],
                            st["ob_paid"],
                            st["saez_tax"],
                            st["w_tax"],
                            st["transfer_d"],
                            st["pg_contrib_g"],
                            st["pg_dividend"],
                            audited,
                            p_points,
                            penalty_cash,
                            period_u,
                        ),
                    )

                conn.execute(
                    "UPDATE macro_institutions SET cash_balance = ? WHERE institution_id = 'PublicTreasury';",
                    (round(treasury_cash, 4),),
                )
                conn.execute(
                    "UPDATE macro_institutions SET cash_balance = ? WHERE institution_id = 'AggregateFirmSector';",
                    (round(firm_cash, 4),),
                )

                sfc_err = self.get_conservation_error(conn)
                assert sfc_err < 1e-6, f"SFC violation in upkeep tick {tick}: {sfc_err}"
                conn.commit()

                # M5: Endogenous Market-Clearing Price Formation from Order Flow & Supply/Demand
                phi_clear = self.hank_spec.market_clearing_elasticity_phi
                c_buy = self.tick_order_flow["CONSUMABLE_NECESSITY"]["buy"] + total_demand_eaten
                c_sell = self.tick_order_flow["CONSUMABLE_NECESSITY"]["sell"] + max(0.5, total_supply_harvested)
                excess_c = (c_buy - c_sell) / max(1.0, c_buy + c_sell)
                self.macro_state.reference_consumable_price_pc = round(
                    self.macro_state.reference_consumable_price_pc
                    * max(0.80, min(1.30, 1.0 + phi_clear * excess_c)),
                    2,
                )

                k_buy = self.tick_order_flow["PRODUCTIVE_CAPITAL_ASSET"]["buy"]
                k_sell = self.tick_order_flow["PRODUCTIVE_CAPITAL_ASSET"]["sell"]
                if k_buy + k_sell > 0:
                    excess_k = (k_buy - k_sell) / max(1.0, k_buy + k_sell)
                    self.macro_state.reference_capital_price_qk = round(
                        self.macro_state.reference_capital_price_qk
                        * max(0.75, min(1.25, 1.0 + phi_clear * excess_k)),
                        2,
                    )

                self.macro_state.public_treasury_cash = round(treasury_cash, 4)
                self.macro_state.firm_sector_cash = round(firm_cash, 4)
                upkeep_report["pg_provision_rate"] = round(
                    total_pg_contrib / max(total_ideal_endowment, 1e-6), 4
                )
                upkeep_report["free_rider_share"] = round(
                    free_riders_count / max(1, n_agents), 4
                )
                upkeep_report["mean_period_utility_u"] = round(
                    sum(utilities_this_tick) / max(1, len(utilities_this_tick)), 4
                )
                return upkeep_report
            except Exception:
                conn.rollback()
                raise