from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator
import torch


# ============================================================================
# 0. CANONICAL SPAN CONSTANTS & STRICT EXPERIMENTAL ENUMS
# ============================================================================

SPAN_SINK: str = "SPAN_SINK"
SPAN_SELF_IDENTITY: str = "SPAN_SELF_IDENTITY"
SPAN_DRIVES_AND_FINANCE: str = "SPAN_DRIVES_AND_FINANCE"
SPAN_CORE_BELIEF: str = "SPAN_CORE_BELIEF"
SPAN_PEER_MEMORY: str = "SPAN_PEER_MEMORY"
SPAN_INCOMING_MSG: str = "SPAN_INCOMING_MSG"
SPAN_APPRAISAL_AND_TOM: str = "SPAN_APPRAISAL_AND_TOM"

ALL_SEMANTIC_SPANS: List[str] = [
    SPAN_SINK,
    SPAN_SELF_IDENTITY,
    SPAN_DRIVES_AND_FINANCE,
    SPAN_CORE_BELIEF,
    SPAN_PEER_MEMORY,
    SPAN_INCOMING_MSG,
    SPAN_APPRAISAL_AND_TOM,
]
CANONICAL_SPAN_NAMES: List[str] = ALL_SEMANTIC_SPANS
SEMANTIC_SPAN_ORDER: List[str] = ALL_SEMANTIC_SPANS
TokenSpanMap = Dict[str, Tuple[int, int]]


class DecisionMode(str, Enum):
    LLM_AUTONOMOUS = "LLM_AUTONOMOUS"
    RULE_ONLY = "RULE_ONLY"


class StateChannel(str, Enum):
    NONE = "NONE"                     # C0: Neutral baseline text, no steering hooks
    TEXT_ONLY = "TEXT_ONLY"           # C1: True state in prompt text, no steering hooks
    STEER_ONLY = "STEER_ONLY"         # C2: Neutral baseline text, true state in steering hooks
    TEXT_AND_STEER = "TEXT_AND_STEER" # C3: True state in both prompt text and steering hooks
    CUE_CONFLICT = "CUE_CONFLICT"     # C4: Text says liquid/calm; steering injects high scarcity (and vice versa)


class SteeringMethod(str, Enum):
    NONE = "NONE"
    PLACEBO_RANDOM = "PLACEBO_RANDOM"       # C5: Matched-norm random unit vector at identical hook sites
    SHUFFLED_STATE = "SHUFFLED_STATE"       # C6: State packet permuted from a different agent
    PASTA = "PASTA"                         # Zhang et al. (2024) post-hoc attention scaling baseline
    ACTADD_CAA = "ACTADD_CAA"               # Turner et al. / Rimsky et al. residual activation addition
    OUR_COUPLED_STEER = "OUR_COUPLED_STEER" # Mechanisms A (Attn Bias) + B (Value CAA) + C (Easterbrook Beta)
    MASTER_EQUATION = "OUR_COUPLED_STEER"   # Backward-compatible alias


class TMSMode(str, Enum):
    OFF = "OFF"
    ENTROPY_TRIGGER = "ENTROPY_TRIGGER"
    RANDOM_TRIGGER = "RANDOM_TRIGGER"
    ALWAYS_ASK = "ALWAYS_ASK"
    ORACLE_TRIGGER = "ORACLE_TRIGGER"


class ClaimStatus(str, Enum):
    IMPLEMENTED = "IMPLEMENTED"
    PARTIAL = "PARTIAL"
    DEFERRED = "DEFERRED"


class BackendType(str, Enum):
    WHITE_BOX = "WHITE_BOX"
    BLACK_BOX = "BLACK_BOX"


SteeringMode = SteeringMethod


class ActionCategory(str, Enum):
    COMMERCIAL_TRADE = "COMMERCIAL_TRADE"
    PERSONAL_WORK_CONSUME = "PERSONAL_WORK_CONSUME"
    TMS_QUERY = "TMS_QUERY"
    PUBLIC_COMMONS = "PUBLIC_COMMONS"


class ConsentDecision(str, Enum):
    INITIATE = "INITIATE"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    COUNTER_OFFER = "COUNTER_OFFER"
    IGNORE = "IGNORE"


class ActionType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD_OR_WORK = "HOLD_OR_WORK"


class LegalStatus(str, Enum):
    LAW_ABIDING = "LAW_ABIDING"
    SANCTIONED = "SANCTIONED"
    IMPRISONED = "IMPRISONED"


CanonicalItemType = Literal["CONSUMABLE_NECESSITY", "PRODUCTIVE_CAPITAL_ASSET"]
CrisisScenarioType = Literal[
    "NONE",
    "STAGFLATION_SUBSISTENCE_SHOCK",
    "ASYMMETRIC_UNEMPLOYMENT_SHOCK",
    "ASSET_VALUATION_CRASH",
    "LIQUIDITY_CREDIT_CRUNCH",
]


def normalize_item_id(raw_item: str) -> str:
    """Maps item names onto the strict Two-Asset HANK taxonomy."""
    token = (raw_item or "").strip().upper()
    if token in ("FOOD", "CONSUMABLE", "CONSUMABLE_NECESSITY", "NECESSITY", "GRAIN"):
        return "CONSUMABLE_NECESSITY"
    if token in ("TECH", "CAPITAL", "PRODUCTIVE_CAPITAL_ASSET", "CAPITAL_ASSET", "ASSET"):
        return "PRODUCTIVE_CAPITAL_ASSET"
    return "CONSUMABLE_NECESSITY"


class StrictConfigModel(BaseModel):
    """Strict validation model (`extra='forbid'`) so YAML config typos fail loudly."""
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        protected_namespaces=(),
    )


class RuntimePacketModel(BaseModel):
    """Typed runtime packet model allowing tensor attachments."""
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        protected_namespaces=(),
    )


# ============================================================================
# 1. EXPERIMENTAL DESIGN & CONFIGURATION SCHEMAS (`extra="forbid"`)
# ============================================================================

class ExperimentalDesignSpec(StrictConfigModel):
    decision_mode: Literal["LLM_AUTONOMOUS", "RULE_ONLY"] = "LLM_AUTONOMOUS"
    state_channel: Literal[
        "NONE", "TEXT_ONLY", "STEER_ONLY", "TEXT_AND_STEER", "CUE_CONFLICT"
    ] = "TEXT_AND_STEER"
    steering_method: Literal[
        "NONE",
        "PLACEBO_RANDOM",
        "SHUFFLED_STATE",
        "PASTA",
        "ACTADD_CAA",
        "OUR_COUPLED_STEER",
        "MASTER_EQUATION",
    ] = "OUR_COUPLED_STEER"
    tms_mode: Literal[
        "OFF", "ENTROPY_TRIGGER", "RANDOM_TRIGGER", "ALWAYS_ASK", "ORACLE_TRIGGER"
    ] = "ENTROPY_TRIGGER"
    randomize_agent_identities: bool = True


class OrthogonalClampingSpec(StrictConfigModel):
    omega_d1_scarcity_and_hank: bool = True
    omega_d2_trust_and_parochialism: bool = True
    omega_d3_contagion_and_polarization: bool = True
    omega_d4_fiscal_and_legal_state: bool = True

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (
            int(self.omega_d1_scarcity_and_hank),
            int(self.omega_d2_trust_and_parochialism),
            int(self.omega_d3_contagion_and_polarization),
            int(self.omega_d4_fiscal_and_legal_state),
        )


class ModelSpec(StrictConfigModel):
    backend_type: Literal["WHITE_BOX", "BLACK_BOX"] = "WHITE_BOX"
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    local_dir: str = "./models/Qwen--Qwen2.5-1.5B-Instruct"
    auto_download: bool = True
    precision: Literal["float16", "bfloat16", "float32"] = "float16"
    attn_implementation: Literal["eager", "sdpa"] = "eager"
    api_base_url: Optional[str] = "http://localhost:1234/v1"
    api_key: Optional[str] = "lm-studio"
    max_new_tokens: int = 96
    temperature: float = 0.0


class SteeringLabConfig(StrictConfigModel):
    enabled: bool = True
    mode: str = "OUR_COUPLED_STEER"
    enable_attn_bias: bool = True
    enable_value_steer: bool = True
    enable_easterbrook_beta: bool = True
    omega_1_drives_finance: float = 2.20
    omega_2_peer_memory: float = 1.60
    gamma_periph_suppress: float = 0.85
    gamma_cynicism_scale: float = 1.20
    eta_threat_scale: float = 0.85
    easterbrook_beta_min: float = 0.85
    easterbrook_beta_max: float = 1.40
    easterbrook_u_opt: float = 0.45
    easterbrook_sigma: float = 0.30
    logit_clamp_max: float = 6.00
    pasta_alpha: float = 0.25
    actadd_coeff: float = 1.15
    hebbian_decay_lambda: float = 0.35
    hebbian_learning_eta: float = 0.40


SteeringSpec = SteeringLabConfig


class TMSSpec(StrictConfigModel):
    enabled: bool = True
    entropy_threshold_tau: float = 0.46
    probe_tokens_k: int = 5
    top_p_truncation: float = 0.90
    communication_cost_xi: float = 0.05


EntropyTMSSpec = TMSSpec


class HANKAndCrisisSpec(StrictConfigModel):
    subsistence_min_c: float = 1.0
    safe_cash_buffer_b: float = 50.0
    borrowing_limit_ratio_income: float = -0.42
    liquid_return_rate_rb: float = 0.002
    borrowing_wedge_kappa: float = 0.0064
    capital_yield_rate_rk: float = 0.075
    reference_consumable_price_pc: float = 12.00
    reference_capital_price_qk: float = 60.00
    market_clearing_elasticity_phi: float = 0.18
    kmv_chi_0: float = 0.075
    kmv_chi_1: float = 0.526
    kmv_chi_2: float = 0.736
    kmv_a_min: float = 0.50
    prod_tfp_baseline_ac: float = 1.00
    prod_capital_elasticity_eta: float = 0.25
    present_bias_beta_min: float = 0.45
    present_bias_eta: float = 1.35
    long_run_discount_delta: float = 0.96
    crra_risk_aversion_sigma: float = 1.50
    default_utility_penalty_phi: float = 4.00
    latent_signal_specialist_sigma: float = 0.15
    latent_signal_novice_sigma: float = 1.80
    active_crisis: CrisisScenarioType = "NONE"
    crisis_start_tick: int = 3
    crisis_duration_ticks: int = 4
    crisis_severity_sigma: float = 0.35
    crisis_persistence_rho: float = 0.85


class TrustAndMarketSpec(StrictConfigModel):
    initial_generalized_trust: float = 0.58
    initial_institutional_trust: float = 0.65
    slovic_eta_pos: float = 0.14
    slovic_eta_neg: float = 0.35
    kkt_desperation_kappa: float = 1.20
    kkt_weight_lambda: float = 1.50
    trade_fairness_baseline_gamma: float = 0.35
    gini_trust_sensitivity_beta: float = 0.70
    theil_trust_sensitivity_beta: float = 0.35
    pg_compliance_trust_beta: float = 0.25
    scarcity_trust_beta: float = 0.30
    inst_trust_spillover_beta: float = 0.20
    familiarity_decay_lambda_n: float = 0.45
    seller_base_margin_mu: float = 0.20
    distrust_risk_premium_phi: float = 0.35
    partner_match_trust_beta: float = 2.50
    partner_match_surplus_beta: float = 1.80


class ContagionAndOpinionSpec(StrictConfigModel):
    hk_initial_confidence_eps: float = 0.50
    hk_trust_inc_gamma: float = 0.12
    hk_distrust_dec_gamma: float = 0.18
    fj_susceptibility_default: float = 0.45
    complex_contagion_min_exposures: int = 2
    complex_contagion_min_tie_trust: float = 0.20


class FiscalAndLegalSpec(StrictConfigModel):
    sales_tax_rate: float = 0.05
    saez_mid_income_tax_rate: float = 0.18
    saez_pareto_alpha: float = 2.00
    saez_elasticity_uncomp: float = 0.25
    saez_elasticity_comp: float = 0.30
    saez_top_social_weight_g: float = 0.15
    wealth_tax_rate: float = 0.015
    wealth_tax_threshold: float = 220.0
    universal_vs_targeted_phi: float = 0.60
    public_goods_endowment_share: float = 0.15
    public_goods_mpcr_a: float = 0.40
    base_audit_probability: float = 0.35
    ostrom_scarcity_mercy_gamma: float = 0.50
    ostrom_sanction_penalty_per_point: float = 0.10
    initial_public_treasury_cash: float = 100.0
    initial_firm_sector_cash: float = 2000.0

    def compute_saez_optimal_top_rate(self) -> float:
        num = 1.0 - self.saez_top_social_weight_g
        denom = (
            num
            + self.saez_pareto_alpha * self.saez_elasticity_uncomp
            + (self.saez_pareto_alpha - 1.0)
            * max(0.0, self.saez_elasticity_comp - self.saez_elasticity_uncomp)
        )
        return float(num / max(denom, 1e-6))


class BigFiveTraits(StrictConfigModel):
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5


class AgentSeedConfig(StrictConfigModel):
    agent_id: str = "Agent_01"
    occupation_role: str = "Worker"
    wage_tier: Literal["LOW", "MID", "HIGH"] = "MID"
    liquid_wealth_cash: float = 100.0
    consumable_necessity_inv: float = 2.0
    productive_capital_inv: float = 1.0
    base_income_wage: float = 15.0
    periodic_obligation_r: float = 6.0
    shock_exposure_e: float = 0.50
    base_consumable_prod_q: float = 0.0
    fehr_schmidt_tier: Literal[
        "SELFISH", "MODERATE", "STRONG", "HYPER_EGALITARIAN"
    ] = "MODERATE"
    alpha_disadvantageous: float = 0.50
    beta_advantageous: float = 0.25
    loss_aversion_lambda: float = 2.25
    risk_tolerance_gamma: float = 0.88
    identity_rigidity: float = 0.50
    innate_opinion_anchor_s: float = 0.0
    tms_specialty: str = "CONSUMABLE_NECESSITY"
    big_five: BigFiveTraits = Field(default_factory=BigFiveTraits)

    def get_consumable_inv(self) -> float:
        return float(self.consumable_necessity_inv)

    def get_capital_inv(self) -> float:
        return float(self.productive_capital_inv)


class SimulationConfig(StrictConfigModel):
    experiment_id: str = "neuropolis-remediated-causal-v3"
    random_seed: int = 42
    total_ticks: int = 8
    footprint_profile: Literal[
        "PROTOTYPE_LAPTOP", "WORKSTATION_RESEARCH", "CLUSTER_PRODUCTION"
    ] = "PROTOTYPE_LAPTOP"
    experimental_design: ExperimentalDesignSpec = Field(
        default_factory=ExperimentalDesignSpec
    )
    orthogonal_clamping: OrthogonalClampingSpec = Field(
        default_factory=OrthogonalClampingSpec
    )
    model_spec: ModelSpec = Field(default_factory=ModelSpec)
    steering_spec: SteeringLabConfig = Field(default_factory=SteeringLabConfig)
    tms_spec: TMSSpec = Field(default_factory=TMSSpec)
    hank_and_crisis_spec: HANKAndCrisisSpec = Field(
        default_factory=HANKAndCrisisSpec
    )
    trust_and_market_spec: TrustAndMarketSpec = Field(
        default_factory=TrustAndMarketSpec
    )
    contagion_and_opinion_spec: ContagionAndOpinionSpec = Field(
        default_factory=ContagionAndOpinionSpec
    )
    fiscal_and_legal_spec: FiscalAndLegalSpec = Field(
        default_factory=FiscalAndLegalSpec
    )
    agents: List[AgentSeedConfig] = Field(default_factory=list)


MVPConfig = SimulationConfig


# ============================================================================
# 2. CALIBRATION, STEERING & PROMPT PACKETS
# ============================================================================

class MechanisticCalibrationBundle(RuntimePacketModel):
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    num_layers: int = 28
    num_heads: int = 12
    num_kv_heads: int = 2
    head_dim: int = 128
    hidden_size: int = 1536
    u_trust: Optional[torch.Tensor] = None             # Shape [L, d_kv] real CAA unit vectors
    u_threat: Optional[torch.Tensor] = None            # Shape [L, d_kv] real CAA unit vectors
    random_placebo_u: Optional[torch.Tensor] = None    # Shape [L, d_kv] matched-norm random control (C5)
    residual_caa_vector: Optional[torch.Tensor] = None # Shape [L, hidden_size] for ACTADD_CAA baseline
    probe_accuracy_trust: float = 0.0
    probe_accuracy_threat: float = 0.0
    calibrated_from_model: bool = False
    sink_token_ids: List[int] = Field(default_factory=lambda: [0])


class EndogenousStatePacket(RuntimePacketModel):
    """
    Separates the true environmental state from the Text-Delivered State and
    the Steering-Delivered State so C0..C4 (including Cue Conflict C4) are unconfounded.
    """
    agent_id: str = ""
    partner_id: Optional[str] = None
    # True underlying environmental variables
    true_scarcity_f: float = 0.0
    true_effective_trust_01: float = 0.50
    # Active Steering-Channel Variables (fed into AttentionSteeringLab hooks)
    financial_stress: float = 0.0
    survival_deficit: float = 0.0
    scarcity_deficit_f: float = 0.0
    present_bias_beta: float = 1.0
    effective_discount_delta: float = 0.96
    arousal_u: float = 0.45
    peer_trust_tau: float = 0.0
    effective_trust_01: float = 0.50
    # Active Text-Channel Variables (rendered into prompt string)
    text_scarcity_f: float = 0.0
    text_cash_display: float = 100.0
    text_consumable_display: float = 3.0
    text_obligation_display: float = 6.0
    # Span & Hebbian metadata
    hebbian_trace_h: List[float] = Field(default_factory=lambda: [0.0] * 7)
    span_map: Dict[str, Tuple[int, int]] = Field(default_factory=dict)


SteeringContextPacket = EndogenousStatePacket


class SteeredForwardTelemetry(RuntimePacketModel):
    updated_hebbian_trace: List[float] = Field(default_factory=lambda: [0.0] * 7)
    span_attention_distribution: Dict[str, float] = Field(default_factory=dict)
    easterbrook_beta: float = 1.0
    attention_measured_from_model: bool = False
    applied_steering_method: str = "NONE"


SteeringTelemetryPacket = SteeredForwardTelemetry


class CompiledPromptPacket(RuntimePacketModel):
    agent_id: str = ""
    raw_prompt: str = ""
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    span_map: Dict[str, Tuple[int, int]] = Field(default_factory=dict)
    semantic_probe_offset: int = 0


# ============================================================================
# 3. AGENT TURN OUTPUT, VERDICTS & EMPIRICAL EVALUATION CONTRACTS
# ============================================================================

class EconomicActionIntent(RuntimePacketModel):
    action_type: Literal["BUY", "SELL", "HOLD_OR_WORK"] = "HOLD_OR_WORK"
    counterparty_id: str = ""
    item_id: str = "CONSUMABLE_NECESSITY"
    quantity: float = 1.0
    unit_price: float = 12.0
    public_goods_contribution: float = 0.0
    tax_evasion_fraction: float = 0.0
    ask_expert: bool = False
    is_distress_fire_sale: bool = False

    @field_validator("item_id", mode="before")
    @classmethod
    def _validate_item(cls, v: Any) -> str:
        return normalize_item_id(str(v))


EconomicIntent = EconomicActionIntent


class AgentTurnOutput(RuntimePacketModel):
    internal_thought: str = Field(
        default="Evaluating household liquidity, obligations, and market prices."
    )
    consent_decision: Literal[
        "INITIATE", "ACCEPT", "REJECT", "COUNTER_OFFER", "IGNORE"
    ] = "INITIATE"
    action_category: Literal[
        "COMMERCIAL_TRADE", "PERSONAL_WORK_CONSUME", "TMS_QUERY", "PUBLIC_COMMONS"
    ] = "COMMERCIAL_TRADE"
    economic_intent: EconomicActionIntent = Field(
        default_factory=EconomicActionIntent
    )
    verbal_message: str = Field(default="Proposing trade.")
    expressed_opinion_z: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    epistemic_confidence: float = Field(default=0.80, ge=0.0, le=1.0)


class RawGenerationResult(RuntimePacketModel):
    parsed_output: AgentTurnOutput = Field(default_factory=AgentTurnOutput)
    raw_text: str = ""
    llm_valid: bool = False
    parse_error: Optional[str] = None
    attention_measured_from_model: bool = False
    tms_halt_triggered: bool = False
    mean_shannon_entropy: float = 0.25
    attention_telemetry: Dict[str, float] = Field(default_factory=dict)
    updated_hebbian_trace: List[float] = Field(default_factory=lambda: [0.0] * 7)


class LedgerTransactionReceipt(RuntimePacketModel):
    tx_id: str = ""
    tick: int = 0
    initiator_id: str = ""
    responder_id: str = ""
    buyer_id: str = ""
    seller_id: str = ""
    action_type: str = "BUY"
    item_id: str = "CONSUMABLE_NECESSITY"
    quantity: float = 1.0
    unit_price: float = 12.0
    gross_amount: float = 12.0
    tax_withheld: float = 0.60
    kmv_adjustment_cost: float = 0.0
    seller_net_received: float = 11.40
    kkt_gouging_index: float = 0.0
    is_fire_sale: bool = False
    global_money_supply_error: float = 0.0


class GameMasterVerdict(RuntimePacketModel):
    status: Literal["APPROVED", "REJECTED"] = "APPROVED"
    error_code: Optional[str] = None
    diagnostic_message: str = ""
    ledger_receipt: Optional[LedgerTransactionReceipt] = None


class MacroRegimeState(RuntimePacketModel):
    tick: int = 0
    active_crisis: CrisisScenarioType = "NONE"
    crisis_intensity_s: float = 0.0
    latent_macro_state_theta: float = 1.0
    consumable_tfp_ac: float = 1.0
    reference_consumable_price_pc: float = 12.00
    reference_capital_price_qk: float = 60.00
    capital_yield_rk: float = 0.075
    liquid_rate_rb: float = 0.002
    borrowing_wedge_kappa: float = 0.0064
    borrowing_limit_scale: float = 1.0
    kmv_chi_1_current: float = 0.526
    generalized_social_trust: float = 0.58
    institutional_trust: float = 0.65
    public_treasury_cash: float = 100.0
    firm_sector_cash: float = 2000.0


class FourDimTickTelemetry(RuntimePacketModel):
    tick: int
    active_crisis: str
    decision_mode: str = "LLM_AUTONOMOUS"
    state_channel: str = "TEXT_AND_STEER"
    steering_method: str = "OUR_COUPLED_STEER"
    llm_valid_rate: float = 1.0
    action_diversity_entropy: float = 0.0
    mean_realized_utility_u: float = 0.0
    # Dimension 1: Scarcity & Two-Asset HANK
    wealth_gini: float
    disposable_income_gini: float
    htm_total_share: float
    poor_htm_share: float
    wealthy_htm_share: float
    realized_mpc: float
    mpc_low_liquidity: float = 0.0
    mpc_high_liquidity: float = 0.0
    distress_sale_ratio_dsr: float
    avg_consumable_price: float
    avg_capital_price: float
    mean_scarcity_f: float
    # Dimension 2: Trust & Bargaining Response
    generalized_trust_tsoc: float
    mean_relational_trust_tau: float
    institutional_trust_tinst: float
    mean_effective_trust_teff: float
    trade_network_modularity_q: float
    price_dispersion_ratio_r: float
    unfairness_rejection_rate: float
    cumulative_trade_volume: float
    # Dimension 3: Opinions & Epistemic TMS
    opinion_dispersion_std: float
    complex_contagion_cascade_share: float
    cumulative_tms_queries: int
    tms_routing_accuracy: float
    tms_entropy_auroc: float = 0.50
    # Dimension 4: Fiscal, Commons & SFC Conservation
    public_goods_provision_rate_pgpr: float
    free_rider_share_fr: float
    public_treasury_balance: float
    total_sanction_points_p: int
    saez_top_marginal_rate: float
    godley_lavoie_error: float

    @property
    def scarcity_bandwidth_cohen_d(self) -> float:
        return round(self.mpc_low_liquidity - self.mpc_high_liquidity, 4)

    @property
    def affective_sync_index_asi(self) -> float:
        return round(max(0.0, 1.0 - self.opinion_dispersion_std), 4)

    @property
    def opinion_bimodality_bc(self) -> float:
        return self.opinion_dispersion_std


TickMacroTelemetry = FourDimTickTelemetry