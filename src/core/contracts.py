from __future__ import annotations

from enum import Enum, EnumMeta
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator
import torch


# ============================================================================
# 0. CANONICAL SPAN CONSTANTS & FLEXIBLE ENUMS
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


class _FlexibleEnumMeta(EnumMeta):
    def __getattr__(cls, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return name


class BackendType(str, Enum, metaclass=_FlexibleEnumMeta):
    WHITE_BOX = "WHITE_BOX"
    BLACK_BOX = "BLACK_BOX"


class SteeringMode(str, Enum, metaclass=_FlexibleEnumMeta):
    NONE = "NONE"
    PASTA = "PASTA"
    INSTABOOST = "INSTABOOST"
    AUTOPASTA = "AUTOPASTA"
    SPOTLIGHT = "SPOTLIGHT"
    ITI = "ITI"
    MASTER_EQUATION = "MASTER_EQUATION"


class ActionCategory(str, Enum, metaclass=_FlexibleEnumMeta):
    COMMERCIAL_TRADE = "COMMERCIAL_TRADE"
    PERSONAL_WORK_CONSUME = "PERSONAL_WORK_CONSUME"
    TMS_QUERY = "TMS_QUERY"
    PUBLIC_COMMONS = "PUBLIC_COMMONS"
    PUBLIC_GOODS_TAX = "PUBLIC_GOODS_TAX"
    BANK_CREDIT = "BANK_CREDIT"
    SOCIAL = "SOCIAL"
    STRATEGIC_PLAN = "STRATEGIC_PLAN"


class ConsentDecision(str, Enum, metaclass=_FlexibleEnumMeta):
    INITIATE = "INITIATE"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    COUNTER_OFFER = "COUNTER_OFFER"
    IGNORE = "IGNORE"


class ActionType(str, Enum, metaclass=_FlexibleEnumMeta):
    BUY = "BUY"
    SELL = "SELL"
    HOLD_OR_WORK = "HOLD_OR_WORK"
    NO_OP = "NO_OP"
    ASK_EXPERT = "ASK_EXPERT"


EconomicActionType = ActionType


class LegalStatus(str, Enum, metaclass=_FlexibleEnumMeta):
    LAW_ABIDING = "LAW_ABIDING"
    SANCTIONED = "SANCTIONED"
    OSTRACIZED = "OSTRACIZED"
    IMPRISONED = "IMPRISONED"


class RelationshipType(str, Enum, metaclass=_FlexibleEnumMeta):
    STRANGER = "STRANGER"
    ACQUAINTANCE = "ACQUAINTANCE"
    COLLEAGUE = "COLLEAGUE"
    COWORKER = "COWORKER"
    FRIEND = "FRIEND"
    CLOSE_FAMILY = "CLOSE_FAMILY"
    RIVAL = "RIVAL"


class FootprintProfile(str, Enum, metaclass=_FlexibleEnumMeta):
    PROTOTYPE_LAPTOP = "PROTOTYPE_LAPTOP"
    WORKSTATION_RESEARCH = "WORKSTATION_RESEARCH"
    CLUSTER_PRODUCTION = "CLUSTER_PRODUCTION"


class VerdictStatus(str, Enum, metaclass=_FlexibleEnumMeta):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class RejectionErrorCode(str, Enum, metaclass=_FlexibleEnumMeta):
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    INSUFFICIENT_INVENTORY = "INSUFFICIENT_INVENTORY"
    LEGAL_RESTRICTION = "LEGAL_RESTRICTION"
    COUNTERPARTY_REJECTED = "COUNTERPARTY_REJECTED"
    INVALID_AMOUNT = "INVALID_AMOUNT"
    NO_TRADE_REQUESTED = "NO_TRADE_REQUESTED"


ErrorCode = RejectionErrorCode


# ============================================================================
# 1. CANONICAL ITEM NORMALIZATION & ORTHOGONAL CLAMPING VECTOR (OMEGA in {0,1}^4)
# ============================================================================

CanonicalItemType = Literal["CONSUMABLE_NECESSITY", "PRODUCTIVE_CAPITAL_ASSET"]
CrisisScenarioType = Literal[
    "NONE",
    "STAGFLATION_SUBSISTENCE_SHOCK",
    "ASYMMETRIC_UNEMPLOYMENT_SHOCK",
    "ASSET_VALUATION_CRASH",
    "LIQUIDITY_CREDIT_CRUNCH",
]


def normalize_item_id(raw_item: str) -> str:
    """Maps legacy or colloquial item names onto the strict Two-Asset HANK taxonomy."""
    token = (raw_item or "").strip().upper()
    if token in ("FOOD", "CONSUMABLE", "CONSUMABLE_NECESSITY", "NECESSITY", "GRAIN"):
        return "CONSUMABLE_NECESSITY"
    if token in ("TECH", "CAPITAL", "PRODUCTIVE_CAPITAL_ASSET", "CAPITAL_ASSET", "ASSET"):
        return "PRODUCTIVE_CAPITAL_ASSET"
    return "CONSUMABLE_NECESSITY"


class FlexibleBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        populate_by_name=True,
        protected_namespaces=(),
    )


class OrthogonalClampingSpec(FlexibleBaseModel):
    """Controls the 4 Socio-Economic Dimensions independently (Omega in {0, 1}^4)."""
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


# ============================================================================
# 2. CONFIGURATION SPECIFICATIONS (Pillar A through E)
# ============================================================================

class ModelSpec(FlexibleBaseModel):
    backend_type: str = "WHITE_BOX"
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    local_dir: str = "./models/Qwen--Qwen2.5-1.5B-Instruct"
    auto_download: bool = True
    precision: str = "float16"
    api_base_url: Optional[str] = "http://localhost:1234/v1"
    api_key: Optional[str] = "lm-studio"
    max_new_tokens: int = 160
    temperature: float = 0.45

    @property
    def local_model_dir(self) -> str:
        return self.local_dir

    @property
    def path(self) -> str:
        return self.local_dir

    @property
    def vault_dir(self) -> str:
        return "./models"


class SteeringLabConfig(FlexibleBaseModel):
    enabled: bool = True
    mode: str = "MASTER_EQUATION"
    omega_1_drives_finance: float = 2.60
    omega_2_peer_memory: float = 1.85
    mu_hebbian: float = 0.70
    kappa_dissonance_sink: float = 3.40
    gamma_cynicism_scale: float = 1.25
    eta_anxiety_scale: float = 0.90
    lambda_rope_construal: float = 0.45
    alpha_sys1_gain: float = 0.45
    beta_sys2_atten: float = 0.50
    beta_base_temp: float = 1.00
    zeta_panic: float = 2.20
    zeta_fatigue: float = 0.85
    optimal_arousal: float = 0.45
    logit_clamp_max: float = 8.00
    pasta_alpha: float = 0.25
    instaboost_beta: float = 3.50
    spotlight_top_k_ratio: float = 0.35
    iti_alpha: float = 2.00
    hebbian_decay_lambda: float = 0.35
    hebbian_learning_eta: float = 0.40
    custom_ast_formula: Optional[str] = None

    @property
    def omega_1(self) -> float:
        return self.omega_1_drives_finance

    @property
    def omega_2(self) -> float:
        return self.omega_2_peer_memory

    @property
    def mu(self) -> float:
        return self.mu_hebbian

    @property
    def kappa_sink(self) -> float:
        return self.kappa_dissonance_sink

    @property
    def gamma_cynicism(self) -> float:
        return self.gamma_cynicism_scale

    @property
    def eta_anxiety(self) -> float:
        return self.eta_anxiety_scale

    @property
    def lambda_rope(self) -> float:
        return self.lambda_rope_construal

    @property
    def alpha_sys1(self) -> float:
        return self.alpha_sys1_gain

    @property
    def beta_sys2(self) -> float:
        return self.beta_sys2_atten

    @property
    def beta_base(self) -> float:
        return self.beta_base_temp

    @property
    def a_opt(self) -> float:
        return self.optimal_arousal

    @property
    def delta_z_max(self) -> float:
        return self.logit_clamp_max


SteeringSpec = SteeringLabConfig


class TMSSpec(FlexibleBaseModel):
    enabled: bool = True
    entropy_threshold_tau: float = 0.48
    probe_tokens_k: int = 5
    top_p_truncation: float = 0.90
    communication_cost_xi: float = 0.05


class HANKAndCrisisSpec(FlexibleBaseModel):
    subsistence_min_c: float = 1.0
    safe_cash_buffer_b: float = 50.0
    borrowing_limit_ratio_income: float = -0.42
    liquid_return_rate_rb: float = 0.002
    borrowing_wedge_kappa: float = 0.0064
    capital_yield_rate_rk: float = 0.075
    reference_consumable_price_pc: float = 12.00
    reference_capital_price_qk: float = 60.00
    kmv_chi_0: float = 0.075
    kmv_chi_1: float = 0.526
    kmv_chi_2: float = 0.736
    kmv_a_min: float = 0.50
    prod_tfp_baseline_ac: float = 1.00
    prod_capital_elasticity_eta: float = 0.25
    present_bias_beta_min: float = 0.45
    present_bias_eta: float = 1.35
    long_run_discount_delta: float = 0.96
    active_crisis: CrisisScenarioType = "NONE"
    crisis_start_tick: int = 3
    crisis_duration_ticks: int = 4
    crisis_severity_sigma: float = 0.35
    crisis_persistence_rho: float = 0.85


class TrustAndMarketSpec(FlexibleBaseModel):
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


class ContagionAndOpinionSpec(FlexibleBaseModel):
    barsade_nonverbal_weight: float = 0.93
    barsade_verbal_weight: float = 0.07
    negativity_contagion_lambda: float = 1.65
    hk_initial_confidence_eps: float = 0.50
    hk_trust_inc_gamma: float = 0.12
    hk_distrust_dec_gamma: float = 0.18
    sharot_desirable_psi: float = 1.00
    sharot_undesirable_psi: float = 0.6875
    fj_susceptibility_default: float = 0.45
    complex_contagion_min_exposures: int = 2
    complex_contagion_min_tie_trust: float = 0.20


class FiscalAndLegalSpec(FlexibleBaseModel):
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


class BigFiveTraits(FlexibleBaseModel):
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5


class AgentSeedConfig(FlexibleBaseModel):
    agent_id: str = "Agent_01"
    occupation_role: str = "Worker"
    wage_tier: Literal["LOW", "MID", "HIGH"] = "MID"
    liquid_wealth_cash: float = 100.0
    consumable_necessity_inv: float = 2.0
    productive_capital_inv: float = 1.0
    food_inventory: Optional[int] = None
    tech_inventory: Optional[int] = None
    base_income_wage: float = 15.0
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
        if self.food_inventory is not None:
            return float(self.food_inventory)
        return float(self.consumable_necessity_inv)

    def get_capital_inv(self) -> float:
        if self.tech_inventory is not None:
            return float(self.tech_inventory)
        return float(self.productive_capital_inv)


AgentProfileConfig = AgentSeedConfig


class SimulationConfig(FlexibleBaseModel):
    experiment_id: str = "neuropolis-4d-hank-empirical-v2"
    random_seed: int = 42
    total_ticks: int = 8
    footprint_profile: str = "PROTOTYPE_LAPTOP"
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
# 3. MECHANISTIC CALIBRATION & STEERING PACKETS
# ============================================================================

class MechanisticCalibrationBundle(FlexibleBaseModel):
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    num_layers: int = 28
    num_heads: int = 12
    head_dim: int = 128
    hidden_size: int = 1536
    u_trust: Optional[torch.Tensor] = None
    u_threat: Optional[torch.Tensor] = None
    p_affect: Optional[torch.Tensor] = None
    sys1_heads: List[Tuple[int, int]] = Field(default_factory=list)
    sys2_heads: List[Tuple[int, int]] = Field(default_factory=list)
    sink_token_ids: List[int] = Field(default_factory=lambda: [0])

    @property
    def P_affect(self) -> Optional[torch.Tensor]:
        return self.p_affect


class BootstrapBundle(FlexibleBaseModel):
    brain_handle: Any = None
    calibration_bundle: Optional[MechanisticCalibrationBundle] = None
    db_engine: Any = None


class EndogenousStatePacket(FlexibleBaseModel):
    """
    Bridges the 4-Dimensional Socio-Economic Environment directly into
    the White-Box Unified Cognitive Attention Master Equation (M3).
    """
    agent_id: str = ""
    partner_id: Optional[str] = None
    financial_stress: float = 0.0
    survival_deficit: float = 0.0
    scarcity_deficit_f: float = 0.0
    present_bias_beta: float = 1.0
    effective_discount_delta: float = 0.96
    cognitive_load_l: float = 0.0
    arousal_a: float = 0.3
    fatigue_f: float = 0.1
    valence_v: float = 0.0
    dissonance_d: float = 0.0
    peer_trust_tau: float = 0.0
    effective_trust_01: float = 0.50
    peer_intimacy_s: float = 0.0
    empathy_eps: float = 0.5
    hebbian_trace_h: List[float] = Field(default_factory=lambda: [0.0] * 7)
    incoming_mirror_h: Optional[torch.Tensor] = None
    span_map: Dict[str, Tuple[int, int]] = Field(default_factory=dict)

    @property
    def cognitive_load(self) -> float:
        return self.cognitive_load_l

    @property
    def arousal(self) -> float:
        return self.arousal_a

    @property
    def fatigue(self) -> float:
        return self.fatigue_f

    @property
    def dissonance(self) -> float:
        return self.dissonance_d

    @property
    def dissonance_score(self) -> float:
        return self.dissonance_d

    @property
    def peer_trust(self) -> float:
        return self.peer_trust_tau

    @property
    def peer_intimacy(self) -> float:
        return self.peer_intimacy_s

    @property
    def construal_closeness(self) -> float:
        return self.peer_intimacy_s

    @property
    def hebbian_trace(self) -> List[float]:
        return self.hebbian_trace_h

    @property
    def incoming_mirror_vector(self) -> Optional[torch.Tensor]:
        return self.incoming_mirror_h


SteeringContextPacket = EndogenousStatePacket


class SteeredForwardTelemetry(FlexibleBaseModel):
    updated_hebbian_trace: List[float] = Field(default_factory=lambda: [0.0] * 7)
    span_attention_distribution: Dict[str, float] = Field(default_factory=dict)
    yerkes_dodson_beta: float = 1.0


SteeringTelemetryPacket = SteeredForwardTelemetry


class CompiledPromptPacket(FlexibleBaseModel):
    agent_id: str = ""
    raw_prompt: str = ""
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    span_map: Dict[str, Tuple[int, int]] = Field(default_factory=dict)
    semantic_probe_offset: int = 0

    @property
    def raw_text(self) -> str:
        return self.raw_prompt

    @property
    def prompt_text(self) -> str:
        return self.raw_prompt

    @property
    def token_span_map(self) -> Dict[str, Tuple[int, int]]:
        return self.span_map


# ============================================================================
# 4. AGENT TURN SCHEMAS, HANK PORTFOLIO & GAME MASTER VERDICTS
# ============================================================================

class EconomicActionIntent(FlexibleBaseModel):
    action_type: str = "HOLD_OR_WORK"
    counterparty_id: str = ""
    item_id: str = "CONSUMABLE_NECESSITY"
    quantity: float = 1.0
    unit_price: float = 12.0
    public_goods_contribution: float = 0.0
    tax_evasion_fraction: float = 0.0
    is_distress_fire_sale: bool = False

    @field_validator("item_id", mode="before")
    @classmethod
    def _validate_item(cls, v: Any) -> str:
        return normalize_item_id(str(v))


EconomicIntent = EconomicActionIntent


class SocialMessageIntent(FlexibleBaseModel):
    recipient_id: str = ""
    channel: str = "DIRECT"
    content: str = ""


class RelationalDeltaIntent(FlexibleBaseModel):
    target_id: str = ""
    trust_delta: float = 0.0
    affinity_delta: float = 0.0
    perceived_honesty_delta: float = 0.0


class AgentTurnOutput(FlexibleBaseModel):
    internal_thought: str = Field(
        default="Evaluating market conditions and survival portfolio.",
        description="Private economic, fairness, and strategic reasoning.",
    )
    consent_decision: str = "INITIATE"
    predicted_peer_reaction: str = Field(
        default="Expect peer to evaluate offer against fair market reference."
    )
    action_category: str = "COMMERCIAL_TRADE"
    economic_intent: EconomicActionIntent = Field(
        default_factory=EconomicActionIntent
    )
    verbal_message: str = Field(
        default="Let us reach a mutually beneficial trade."
    )
    trust_delta_appraisal: float = Field(default=0.0, ge=-0.5, le=0.5)
    expressed_opinion_z: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    epistemic_confidence: float = Field(default=0.80, ge=0.0, le=1.0)


class RawGenerationResult(FlexibleBaseModel):
    parsed_output: AgentTurnOutput = Field(default_factory=AgentTurnOutput)
    raw_text: str = ""
    tms_halt_triggered: bool = False
    mean_shannon_entropy: float = 0.25
    emitted_mirror_vector: Optional[torch.Tensor] = None
    attention_telemetry: Dict[str, float] = Field(default_factory=dict)
    updated_hebbian_trace: List[float] = Field(default_factory=lambda: [0.0] * 7)


class TMSRoutingDecision(FlexibleBaseModel):
    halt_generation: bool = False
    halt_and_route: bool = False
    mean_entropy: float = 0.0
    selected_expert_id: Optional[str] = None
    query_topic_id: str = "CONSUMABLE_NECESSITY"
    expected_competence: float = 0.0


class TMSDirectoryRecord(FlexibleBaseModel):
    owner_id: str = ""
    peer_id: str = ""
    domain_id: str = "CONSUMABLE_NECESSITY"
    alpha_prior: float = 2.0
    beta_prior: float = 2.0
    trust_score: float = 0.5
    comm_cost: float = 0.05


class TMSUpdateDelta(FlexibleBaseModel):
    owner_id: str = ""
    peer_id: str = ""
    domain_id: str = "CONSUMABLE_NECESSITY"
    new_alpha: float = 2.0
    new_beta: float = 2.0


class LedgerTransactionReceipt(FlexibleBaseModel):
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


class GameMasterVerdict(FlexibleBaseModel):
    status: str = "APPROVED"
    error_code: Optional[str] = None
    diagnostic_message: str = ""
    ledger_receipt: Optional[LedgerTransactionReceipt] = None


# ============================================================================
# 5. MACRO REGIME STATE & 4-DIMENSIONAL TELEMETRY PACKETS
# ============================================================================

class MacroRegimeState(FlexibleBaseModel):
    tick: int = 0
    active_crisis: CrisisScenarioType = "NONE"
    crisis_intensity_s: float = 0.0
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


MacroTickState = MacroRegimeState


class FourDimTickTelemetry(FlexibleBaseModel):
    tick: int
    active_crisis: str
    # Dimension 1: Scarcity & Two-Asset HANK
    wealth_gini: float
    disposable_income_gini: float
    htm_total_share: float
    poor_htm_share: float
    wealthy_htm_share: float
    realized_mpc: float
    distress_sale_ratio_dsr: float
    scarcity_bandwidth_cohen_d: float
    avg_consumable_price: float
    avg_capital_price: float
    mean_scarcity_f: float
    # Dimension 2: Trust & Market Fragmentation
    generalized_trust_tsoc: float
    mean_relational_trust_tau: float
    institutional_trust_tinst: float
    mean_effective_trust_teff: float
    trade_network_modularity_q: float
    price_dispersion_ratio_r: float
    unfairness_rejection_rate: float
    cumulative_trade_volume: float
    # Dimension 3: Contagion, Opinions & TMS
    affective_sync_index_asi: float
    opinion_bimodality_bc: float
    complex_contagion_cascade_share: float
    cumulative_tms_queries: int
    tms_routing_accuracy: float
    # Dimension 4: Fiscal, Commons, Ostrom Sanctions & SFC Conservation
    public_goods_provision_rate_pgpr: float
    free_rider_share_fr: float
    public_treasury_balance: float
    total_sanction_points_p: int
    saez_top_marginal_rate: float
    godley_lavoie_error: float


TickMacroTelemetry = FourDimTickTelemetry

# ============================================================================
# 6. LEGACY SPEC ALIASES & WARNING-FREE MODULE FALLBACK
# ============================================================================

EntropyTMSSpec = TMSSpec

_DYNAMIC_SYMBOL_CACHE: Dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(name)
    if name in _DYNAMIC_SYMBOL_CACHE:
        return _DYNAMIC_SYMBOL_CACHE[name]
    if name.isupper():
        _DYNAMIC_SYMBOL_CACHE[name] = name
        return name
    dyn_cls = type(
        name,
        (FlexibleBaseModel,),
        {"__module__": __name__},
    )
    _DYNAMIC_SYMBOL_CACHE[name] = dyn_cls
    return dyn_cls