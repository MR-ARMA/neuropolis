from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ============================================================================
# 0. STRICT BASE MODEL & CORE ENUMS
# ============================================================================

class StrictContract(BaseModel):
    """Strict Pydantic v2 base contract (`extra='forbid'`) preventing silent config or schema drift."""
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        protected_namespaces=(),
    )


class AssetType(str, Enum):
    CONSUMABLE = "CONSUMABLE"   # Non-durable subsistence necessity (c_min drained per tick)
    CAPITAL = "CAPITAL"         # Illiquid productive asset (return r_k > r_b; KMV convex cost chi)


class OrderSide(str, Enum):
    BID = "BID"                 # Limit/Market order to BUY an asset with liquid cash
    ASK = "ASK"                 # Limit/Market order to SELL an asset from inventory


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class OrderStatus(str, Enum):
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class AgentActionType(str, Enum):
    PLACE_ORDER = "PLACE_ORDER"             # Submit BID/ASK order to the CDA Order Book
    CANCEL_ORDER = "CANCEL_ORDER"           # Cancel an open resting order and release escrow
    LIQUIDATE_CAPITAL = "LIQUIDATE_CAPITAL" # Convert illiquid capital a_{i,t} -> cash b_{i,t} (incurs chi)
    INVEST_CAPITAL = "INVEST_CAPITAL"       # Convert cash b_{i,t} -> illiquid capital a_{i,t} (incurs chi)
    CONSUME_AND_HOLD = "CONSUME_AND_HOLD"   # No market or capital transfer action this tick
    ABSTAIN = "ABSTAIN"                     # Explicit non-execution after exhausting JSON retries


class RepEMode(str, Enum):
    OFF = "OFF"
    ENDOGENOUS_PROBE_GATED = "ENDOGENOUS_PROBE_GATED"
    PLACEBO_RANDOM_VECTOR = "PLACEBO_RANDOM_VECTOR"
    SHUFFLED_LAMBDA = "SHUFFLED_LAMBDA"


class PolicyMode(str, Enum):
    LLM_AGENT = "LLM_AGENT"
    ZERO_INTELLIGENCE = "ZERO_INTELLIGENCE"
    CLOSED_FORM_BENCHMARK = "CLOSED_FORM_BENCHMARK"


class MacroShockType(str, Enum):
    NONE = "NONE"
    TFP_SUBSISTENCE_CONTRACTION = "TFP_SUBSISTENCE_CONTRACTION"
    ASYMMETRIC_WAGE_SHOCK = "ASYMMETRIC_WAGE_SHOCK"
    CAPITAL_YIELD_COLLAPSE = "CAPITAL_YIELD_COLLAPSE"
    CREDIT_WEDGE_CRUNCH = "CREDIT_WEDGE_CRUNCH"


# ============================================================================
# 1. TWO-ASSET SFC SYSTEM & NEEDS ENGINE SCHEMAS
# ============================================================================

class TwoAssetEconomyConfig(StrictContract):
    """
    Strict Stock-Flow Consistent (SFC) Two-Asset Household Economy parameters
    (Kaplan, Violante, & Weidner 2014; Kaplan, Moll, & Violante 2018).
    """
    # Liquid Asset (Cash b_{i,t}) return and credit constraint wedge
    liquid_return_rate_rb: float = Field(
        default=0.002,
        ge=0.0,
        description="Per-tick return rate r_b paid on positive liquid cash balances b_{i,t} >= 0.",
    )
    borrowing_wedge_kappa: float = Field(
        default=0.008,
        ge=0.0,
        description="Per-tick borrowing spread kappa charged above r_b when b_{i,t} < 0.",
    )
    borrowing_limit_income_multiple: float = Field(
        default=-0.50,
        le=0.0,
        description="Hard credit floor b_bar_i = multiple * base_wage_i (currency units).",
    )
    # Illiquid Asset (Productive Capital a_{i,t}) return wedge & convex adjustment cost chi(d, a)
    capital_return_rate_rk: float = Field(
        default=0.015,
        gt=0.0,
        description="Per-tick yield r_k on illiquid capital a_{i,t} (strictly greater than r_b).",
    )
    capital_unit_par_value: float = Field(
        default=50.0,
        gt=0.0,
        description="Accounting par conversion value of 1.0 unit of illiquid capital (currency units).",
    )
    kmv_chi_0: float = Field(
        default=0.075,
        ge=0.0,
        description="Linear liquidation/adjustment cost coefficient chi_0 in chi(d, a).",
    )
    kmv_chi_1: float = Field(
        default=0.526,
        ge=0.0,
        description="Convex adjustment cost scale chi_1 in chi(d, a).",
    )
    kmv_chi_2: float = Field(
        default=0.736,
        gt=0.0,
        description="Convexity exponent chi_2 in chi(d, a).",
    )
    kmv_a_min: float = Field(
        default=0.50,
        gt=0.0,
        description="Regularization denominator floor a_min in chi(d, a).",
    )
    # Needs Engine (Biological Subsistence & Health Decay)
    subsistence_min_c: float = Field(
        default=1.0,
        gt=0.0,
        description="Consumable units required per tick to maintain biological health.",
    )
    health_decay_per_missing_unit: float = Field(
        default=0.18,
        ge=0.0,
        le=1.0,
        description="Health H_{i,t} drop per 1.0 unit of unmet subsistence consumption.",
    )
    health_recovery_per_surplus_tick: float = Field(
        default=0.06,
        ge=0.0,
        le=1.0,
        description="Health H_{i,t} recovery per tick when full subsistence is met.",
    )
    health_productivity_floor: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum wage/output multiplier as living health approaches 0+.",
    )
    # Production & Initial Sectoral Cash Balances (SFC Closure)
    baseline_tfp_a_c: float = Field(default=1.0, gt=0.0)
    sales_tax_rate: float = Field(default=0.05, ge=0.0, le=0.50)
    initial_firm_sector_cash: float = Field(default=50000.0, ge=0.0)
    initial_intermediary_cash: float = Field(default=25000.0, ge=0.0)
    initial_treasury_cash: float = Field(default=5000.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_return_wedge(self) -> "TwoAssetEconomyConfig":
        if self.capital_return_rate_rk <= self.liquid_return_rate_rb:
            raise ValueError(
                "Two-Asset return wedge violated: capital_return_rate_rk must strictly exceed liquid_return_rate_rb."
            )
        return self


class NeedsState(StrictContract):
    """Biological vitality and objective stress state managed by the Needs Engine."""
    health: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Biological health H_{i,t} in [0, 1]; 0.0 represents permanent economic death.",
    )
    last_consumed_units: float = Field(default=1.0, ge=0.0)
    subsistence_deficit_streak: int = Field(default=0, ge=0)
    composite_stress: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Objective physiological/liquidity stress index S_{i,t} driving cognitive bottlenecks ONLY.",
    )


class HouseholdBalanceSheet(StrictContract):
    """Double-entry household balance sheet separating available holdings from CDA escrow locks."""
    agent_id: str
    role: Literal["PRODUCER", "CAPITALIST", "WORKER", "ARBITRAGEUR"] = "WORKER"
    liquid_cash: float = Field(
        description="Unencumbered liquid cash b_{i,t} (may go negative down to borrowing_limit_b_bar)."
    )
    escrow_cash: float = Field(
        default=0.0, ge=0.0, description="Cash locked in active CDA BID limit orders."
    )
    illiquid_capital: float = Field(
        default=0.0, ge=0.0, description="Unencumbered illiquid productive capital a_{i,t} (units)."
    )
    escrow_capital: float = Field(
        default=0.0, ge=0.0, description="Capital units locked in active CDA ASK limit orders."
    )
    consumable_inventory: float = Field(
        default=2.0, ge=0.0, description="Unencumbered consumable inventory c_{i,t} (units)."
    )
    escrow_consumable: float = Field(
        default=0.0, ge=0.0, description="Consumable units locked in active CDA ASK limit orders."
    )
    borrowing_limit_b_bar: float = Field(
        default=-10.0, le=0.0, description="Hard credit floor b_bar_i (currency units)."
    )
    base_wage: float = Field(default=15.0, ge=0.0)
    periodic_obligation_r: float = Field(
        default=6.0, ge=0.0, description="Mandatory fixed per-tick cash outflow R_{i,t}."
    )
    base_consumable_production_q: float = Field(default=0.0, ge=0.0)
    shock_exposure_e: float = Field(default=0.5, ge=0.0, le=1.0)
    needs: NeedsState = Field(default_factory=NeedsState)


class SectoralBalanceAudit(StrictContract):
    """Immutable per-tick Stock-Flow Consistency (SFC) certificate across all 5 sectors."""
    tick: int
    total_household_cash: float
    total_escrow_cash: float
    firm_sector_cash: float
    intermediary_cash: float
    treasury_cash: float
    total_money_supply_m: float
    initial_money_supply_m0: float
    conservation_error_abs: float
    total_capital_units: float
    total_consumable_units: float


# ============================================================================
# 2. CONTINUOUS DOUBLE AUCTION (CDA) ORDER BOOK SCHEMAS
# ============================================================================

class CDAMarketConfig(StrictContract):
    tick_size: float = Field(
        default=0.05, gt=0.0, description="Minimum price increment on the CDA order book."
    )
    min_order_price: float = Field(default=0.50, gt=0.0)
    max_order_price: float = Field(default=500.00, gt=1.0)
    max_order_quantity: float = Field(default=25.0, gt=0.0)
    order_ttl_ticks: int = Field(
        default=3, ge=1, description="Ticks before unfilled resting limit orders expire."
    )
    enable_capital_secondary_book: bool = Field(
        default=True, description="Enable secondary CDA order book for illiquid capital."
    )


class CDAOrder(StrictContract):
    """Single BID or ASK order submitted to the Price-Time Priority Limit Order Book."""
    order_id: str
    tick_submitted: int
    sequence_num: int = Field(
        ge=0, description="Monotonic intra-tick arrival sequence enforcing time priority."
    )
    agent_id: str
    asset: AssetType
    side: OrderSide
    order_type: OrderType = OrderType.LIMIT
    limit_price: Optional[float] = Field(
        default=None, gt=0.0, description="Required for LIMIT orders; None for MARKET orders."
    )
    quantity: float = Field(gt=0.0)
    filled_quantity: float = Field(default=0.0, ge=0.0)
    status: OrderStatus = OrderStatus.OPEN
    rejection_reason: Optional[str] = None

    @property
    def remaining_quantity(self) -> float:
        return round(max(0.0, self.quantity - self.filled_quantity), 6)

    @model_validator(mode="after")
    def _validate_limit_price(self) -> "CDAOrder":
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("LIMIT orders must specify a positive limit_price.")
        return self


class TradeFill(StrictContract):
    """Executed peer-to-peer CDA trade match settled at the resting maker's price."""
    fill_id: str
    tick: int
    asset: AssetType
    buy_order_id: str
    sell_order_id: str
    buyer_id: str
    seller_id: str
    execution_price: float = Field(gt=0.0)
    quantity: float = Field(gt=0.0)
    gross_value: float = Field(gt=0.0)
    sales_tax_paid: float = Field(default=0.0, ge=0.0)
    maker_side: OrderSide


class PriceLevelDepth(StrictContract):
    price: float
    total_quantity: float
    order_count: int


class OrderBookSnapshot(StrictContract):
    """Public Level-2 Top-5 order book state visible to agents."""
    tick: int
    asset: AssetType
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    mid_price: Optional[float] = None
    spread: Optional[float] = None
    last_trade_price: Optional[float] = None
    tick_volume: float = 0.0
    bids_top5: List[PriceLevelDepth] = Field(default_factory=list)
    asks_top5: List[PriceLevelDepth] = Field(default_factory=list)


# ============================================================================
# 3. DOMAIN-GENERAL COGNITIVE BOTTLENECK SCHEMAS (FACTORS A & B)
# ============================================================================

class DeliberationBottleneckConfig(StrictContract):
    """
    Factor A: Bounded Deliberation Budget.
    Truncates `max_tokens` STRICTLY for Stage 1 hidden CoT reasoning under high stress,
    while preserving the full, stress-invariant budget for Stage 2 JSON Action generation.
    """
    enabled: bool = True
    stress_activation_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Stress threshold above which CoT token truncation engages (0.0 = smooth continuous decay).",
    )
    cot_max_tokens_calm: int = Field(
        default=256, ge=16, description="Stage 1 CoT token budget when stress S_{i,t} = 0."
    )
    cot_min_tokens_stressed: int = Field(
        default=24, ge=8, description="Minimum Stage 1 CoT token budget when stress S_{i,t} = 1."
    )
    stress_decay_steepness: float = Field(
        default=3.5, ge=0.0, description="Exponential decay constant k_delib in B_CoT(S)."
    )
    action_json_max_tokens: int = Field(
        default=128,
        ge=32,
        description="Stress-invariant token budget reserved strictly for Stage 2 JSON Action generation.",
    )

    @model_validator(mode="after")
    def _validate_cot_bounds(self) -> "DeliberationBottleneckConfig":
        if self.cot_min_tokens_stressed > self.cot_max_tokens_calm:
            raise ValueError("cot_min_tokens_stressed cannot exceed cot_max_tokens_calm.")
        return self


class MemoryFrictionConfig(StrictContract):
    """
    Factor B: Episodic Memory Friction (Salience Bias & Recall Degradation).
    Contracts RAG Top-K and injects Gaussian retrieval noise under high stress S_{i,t}.
    """
    enabled: bool = True
    stress_activation_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Stress threshold above which Top-K contraction and retrieval noise engage.",
    )
    top_k_calm: int = Field(default=8, ge=1)
    top_k_stressed: int = Field(default=2, ge=1)
    similarity_noise_sigma_max: float = Field(
        default=0.35,
        ge=0.0,
        description="Max Gaussian std dev added to retrieval similarities/embeddings at S_{i,t} = 1.",
    )
    recency_salience_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    max_episodic_capacity: int = Field(default=256, ge=16)
    embedding_model_name: str = Field(default="BAAI/bge-small-en-v1.5")
    embedding_dim: int = Field(default=384, ge=16)

    @model_validator(mode="after")
    def _validate_top_k_bounds(self) -> "MemoryFrictionConfig":
        if self.top_k_stressed > self.top_k_calm:
            raise ValueError("top_k_stressed cannot exceed top_k_calm.")
        return self


class EpisodicMemoryRecord(StrictContract):
    mem_id: str
    agent_id: str
    tick: int
    event_type: Literal[
        "TRADE_FILL",
        "ORDER_EXPIRED",
        "SUBSISTENCE_DEFICIT",
        "CAPITAL_TRANSFER",
        "SHOCK_OBSERVED",
    ]
    summary_text: str
    salience_score: float = Field(default=1.0, ge=0.0)
    embedding: Optional[List[float]] = None


# ============================================================================
# 4. TWO-STAGE PROBE-GATED RepE SCHEMAS (FACTOR C — ENDOGENOUS STEERING)
# ============================================================================

class TwoStageRepEConfig(StrictContract):
    """
    Factor C: Endogenous Two-Stage Probe-Gated Representation Engineering.
      - Stage 1 (Unsteered Appraisal): Linear probe (w, b) reads Stage 1 CoT residuals
        at `target_layer` l and computes lambda = sigmoid((w^T h + b - theta) / T).
      - Stage 2 (Controlled Action): Injects lambda * gamma * v_hat strictly into
        Stage 2 Action tokens at `target_layer` l.
      - ZERO external environment if/else triggers. ZERO post-hoc logit overrides.
    """
    enabled: bool = True
    mode: RepEMode = RepEMode.ENDOGENOUS_PROBE_GATED
    concept_name: str = Field(
        default="perceived_inequity",
        description="Disjoint out-of-domain concept tracked by Linear Probe and RepE vector.",
    )
    target_layer: int = Field(
        default=18,
        ge=0,
        description="Transformer residual-stream layer index l for Stage 1 probing and Stage 2 injection.",
    )
    steering_strength_gamma: float = Field(
        default=1.50,
        ge=-10.0,
        le=10.0,
        description="Global latent steering multiplier gamma in h' = h + lambda * gamma * v_hat.",
    )
    probe_pooling: Literal["early_appraisal_cot", "mean_cot", "last_cot_token"] = "early_appraisal_cot"
    appraisal_window_tokens: int = Field(
        default=20,
        ge=4,
        description="Opening Primary Appraisal CoT token window (<23 decode steps at budget=24) pooled under `early_appraisal_cot`.",
    )
    gate_activation: Literal["sigmoid_centered", "relu_clipped"] = "sigmoid_centered"
    probe_threshold_theta: float = Field(
        default=0.0,
        description="Probe decision boundary threshold tau/theta before gate activation.",
    )
    probe_temperature: float = Field(
        default=1.0,
        gt=0.0,
        description="Sigmoid temperature T scaling (w^T h + b - theta) / T.",
    )
    max_relative_norm_ratio: float = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
        description="Hard fluency cap: ||lambda * gamma * v_hat||_2 / ||h^(l)||_2 <= max_relative_norm_ratio.",
    )
    probe_artifact_path: str = Field(
        default="artifacts/probes/inequity_probe_l18.pt",
        description="File path for serialized disjoint probe weights (w, b), v_hat, and placebo vector.",
    )
    min_contrastive_pairs: int = Field(
        default=240,
        ge=200,
        description="Minimum total disjoint out-of-domain contrastive pairs (split across probe & vector).",
    )


class ProbeGateMeasurement(StrictContract):
    """Per-turn audit telemetry of Stage 1 linear probing and Stage 2 latent injection."""
    layer_idx: int
    cot_tokens_inspected: int
    raw_probe_score: float
    endogenous_lambda: float = Field(ge=0.0, le=1.0)
    steering_gamma: float
    injected_vector_l2_norm: float
    relative_residual_norm_ratio: float
    repe_mode: RepEMode


class Stage2ActionPayload(StrictContract):
    """Strict structured action payload decoded from Stage 2 JSON emission."""
    action_type: AgentActionType
    asset: AssetType = AssetType.CONSUMABLE
    side: Optional[OrderSide] = None
    order_type: OrderType = OrderType.LIMIT
    limit_price: Optional[float] = Field(default=None, gt=0.0)
    quantity: float = Field(default=0.0, ge=0.0)
    cancel_order_id: Optional[str] = None
    capital_transfer_units: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_payload(self) -> "Stage2ActionPayload":
        if self.action_type == AgentActionType.PLACE_ORDER:
            if self.side is None:
                raise ValueError("PLACE_ORDER requires side ('BID' or 'ASK').")
            if self.quantity <= 0.0:
                raise ValueError("PLACE_ORDER requires quantity > 0.")
            if self.order_type == OrderType.LIMIT and self.limit_price is None:
                raise ValueError("LIMIT PLACE_ORDER requires a positive limit_price.")
        elif self.action_type == AgentActionType.CANCEL_ORDER and not self.cancel_order_id:
            raise ValueError("CANCEL_ORDER requires cancel_order_id.")
        elif self.action_type in (
            AgentActionType.LIQUIDATE_CAPITAL,
            AgentActionType.INVEST_CAPITAL,
        ) and self.capital_transfer_units <= 0.0:
            raise ValueError(f"{self.action_type.value} requires capital_transfer_units > 0.")
        return self


class AgentTurnRecord(StrictContract):
    """Immutable provenance record capturing both Stage 1 and Stage 2 for a single agent turn."""
    run_id: str
    economy_seed: int
    tick: int
    agent_id: str
    policy_mode: PolicyMode
    factorial_cell_id: str
    pre_turn_cash: float
    pre_turn_capital: float
    pre_turn_consumable: float
    pre_turn_health: float
    pre_turn_stress: float
    rag_top_k_used: int
    rag_noise_sigma_used: float
    cot_token_budget_allocated: int
    cot_tokens_generated: int
    cot_truncated_by_bottleneck: bool
    stage1_cot_text: str
    probe_measurement: Optional[ProbeGateMeasurement] = None
    stage2_raw_json: str
    stage2_action: Stage2ActionPayload
    json_valid: bool
    abstain: bool
    retries_used: int
    action_tokens_generated: int
    latency_ms: float
    execution_status: str = "SUBMITTED"


# ============================================================================
# 5. MODEL RUNTIME, 2^3 FACTORIAL DESIGN & STATISTICAL PROTOCOL SCHEMAS
# ============================================================================

class ModelRuntimeConfig(StrictContract):
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    revision: str = "main"
    local_dir: str = "./models/Qwen--Qwen2.5-3B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True
    compute_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    attn_implementation: Literal["eager", "sdpa"] = "eager"
    temperature: float = Field(default=0.70, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    max_json_retries: int = Field(default=3, ge=1, le=5)


class MacroShockConfig(StrictContract):
    shock_type: MacroShockType = MacroShockType.NONE
    start_tick: int = Field(default=20, ge=1)
    duration_ticks: int = Field(default=15, ge=1)
    severity_sigma: float = Field(default=0.35, ge=0.0, le=0.90)
    persistence_rho: float = Field(default=0.85, ge=0.0, le=0.99)


class FactorialCellSpec(StrictContract):
    """Specification of a single cell in the 2^3 Factorial Ablation Matrix or matched control."""
    cell_id: str
    deliberation_bottleneck: bool
    memory_friction: bool
    endogenous_repe: bool
    repe_mode: RepEMode = RepEMode.ENDOGENOUS_PROBE_GATED
    policy_mode: PolicyMode = PolicyMode.LLM_AGENT


class StatisticalProtocolConfig(StrictContract):
    """
    Preregistered statistical rigor parameters:
      - Clustered Bootstrapping over M >= 30 independent economy seeds.
      - Per-Bin TOST (Intersection-Union Test) with a priori SESOI equivalence margins.
    """
    num_economy_seeds_m: int = Field(
        default=30,
        ge=30,
        description="Minimum M >= 30 independent economy seeds for clustered inference.",
    )
    bootstrap_resamples_b: int = Field(default=10000, ge=1000)
    alpha_level: float = Field(default=0.05, gt=0.0, lt=0.25)
    min_observations_per_bin: int = Field(default=30, ge=10)
    multiplicity_correction: Literal["holm_bonferroni", "benjamini_hochberg"] = "holm_bonferroni"
    sesoi_ultimatum_bin_diff: float = Field(default=0.08, gt=0.0)
    sesoi_whtm_share: float = Field(default=0.05, gt=0.0)
    sesoi_mpc_bin_diff: float = Field(default=0.06, gt=0.0)


class PerBinTOSTResult(StrictContract):
    """Statistical verdict for a single empirical curve bin under Clustered Per-Bin TOST."""
    benchmark_name: str
    bin_label: str
    n_clusters_m: int
    n_observations_in_bin: int
    bin_coverage_valid: bool
    sim_cluster_mean: float
    human_reference_mean: float
    mean_difference: float
    clustered_se: float
    ci_90_lower: float
    ci_90_upper: float
    ci_95_lower: float
    ci_95_upper: float
    sesoi_margin_delta: float
    tost_p_lower: float
    tost_p_upper: float
    tost_p_max: float
    holm_adjusted_p: Optional[float] = None
    statistically_equivalent: bool


class SimulationConfig(StrictContract):
    """Master configuration schema for Neuropolis v2 (strict `extra='forbid'` validation)."""
    experiment_id: str = "neuropolis-v2-endogenous-repe-cda"
    master_seed: int = 42
    num_agents: int = Field(default=24, ge=4)
    burn_in_ticks: int = Field(default=10, ge=0)
    active_ticks: int = Field(default=50, ge=1)
    policy_mode: PolicyMode = PolicyMode.LLM_AGENT
    factorial_cell: FactorialCellSpec = Field(
        default_factory=lambda: FactorialCellSpec(
            cell_id="C111",
            deliberation_bottleneck=True,
            memory_friction=True,
            endogenous_repe=True,
            repe_mode=RepEMode.ENDOGENOUS_PROBE_GATED,
            policy_mode=PolicyMode.LLM_AGENT,
        )
    )
    model: ModelRuntimeConfig = Field(default_factory=ModelRuntimeConfig)
    economy: TwoAssetEconomyConfig = Field(default_factory=TwoAssetEconomyConfig)
    cda: CDAMarketConfig = Field(default_factory=CDAMarketConfig)
    deliberation: DeliberationBottleneckConfig = Field(
        default_factory=DeliberationBottleneckConfig
    )
    memory_friction: MemoryFrictionConfig = Field(
        default_factory=MemoryFrictionConfig
    )
    repe: TwoStageRepEConfig = Field(default_factory=TwoStageRepEConfig)
    shock: MacroShockConfig = Field(default_factory=MacroShockConfig)
    stats: StatisticalProtocolConfig = Field(
        default_factory=StatisticalProtocolConfig
    )

    @model_validator(mode="after")
    def _sync_factorial_cell_switches(self) -> "SimulationConfig":
        self.deliberation.enabled = self.factorial_cell.deliberation_bottleneck
        self.memory_friction.enabled = self.factorial_cell.memory_friction
        self.repe.enabled = self.factorial_cell.endogenous_repe
        self.repe.mode = (
            self.factorial_cell.repe_mode
            if self.factorial_cell.endogenous_repe
            else RepEMode.OFF
        )
        self.policy_mode = self.factorial_cell.policy_mode
        return self