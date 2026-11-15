"""
src/core/contracts.py
Complete Pydantic v2 Data Contracts, Enumerations, Tensor Type Aliases,
and Structural Protocols for all 12 Architectural Modules.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ==============================================================================
# 1. TENSOR TYPE ALIASES & CONSTANTS
# ==============================================================================
Tensor1D = torch.Tensor  # Shape: [D] or [S]
Tensor2D = torch.Tensor  # Shape: [S, D] or [B, S]
Tensor3D = torch.Tensor  # Shape: [B, S, D]
Tensor4D = torch.Tensor  # Shape: [B, H, S, S] or [B, H, S, D_head]

SPAN_SINK = "SPAN_SINK"
SPAN_SELF_IDENTITY = "SPAN_SELF_IDENTITY"
SPAN_DRIVES_AND_FINANCE = "SPAN_DRIVES_AND_FINANCE"
SPAN_CORE_BELIEF = "SPAN_CORE_BELIEF"
SPAN_APPRAISAL_AND_TOM = "SPAN_APPRAISAL_AND_TOM"
SPAN_PEER_MEMORY_PREFIX = "SPAN_PEER_MEMORY"
SPAN_INCOMING_MSG_PREFIX = "SPAN_INCOMING_MSG"


# ==============================================================================
# 2. ENUMERATIONS
# ==============================================================================
class BackendType(str, Enum):
    WHITE_BOX_PYTORCH = "WHITE_BOX_PYTORCH"
    BLACK_BOX_API = "BLACK_BOX_API"


class SteeringMode(str, Enum):
    NONE = "NONE"
    PASTA = "PASTA"
    INSTABOOST = "INSTABOOST"
    AUTOPASTA = "AUTOPASTA"
    SPOTLIGHT = "SPOTLIGHT"
    ITI = "ITI"
    MASTER_EQUATION_MVP = "MASTER_EQUATION_MVP"
    MASTER_EQUATION_FULL = "MASTER_EQUATION_FULL"


class LegalStatus(str, Enum):
    LAW_ABIDING = "LAW_ABIDING"
    AUDITED = "AUDITED"
    IMPRISONED = "IMPRISONED"


class RelationshipType(str, Enum):
    STRANGER = "STRANGER"
    ACQUAINTANCE = "ACQUAINTANCE"
    COLLEAGUE = "COLLEAGUE"
    CLOSE_FRIEND = "CLOSE_FRIEND"
    RIVAL = "RIVAL"


class ConsentDecision(str, Enum):
    INITIATE = "INITIATE"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    IGNORE = "IGNORE"


class ActionCategory(str, Enum):
    BILATERAL_TRADE = "BILATERAL_TRADE"
    COUNTER_OFFER = "COUNTER_OFFER"
    WORK_AND_CONSUME = "WORK_AND_CONSUME"
    TMS_QUERY = "TMS_QUERY"
    PUBLIC_GOODS_CONTRIB = "PUBLIC_GOODS_CONTRIB"
    BANK_DEPOSIT_WITHDRAW = "BANK_DEPOSIT_WITHDRAW"
    NO_OP = "NO_OP"


class VerdictStatus(str, Enum):
    APPROVED = "APPROVED"
    ESCROW_LOCKED = "ESCROW_LOCKED"
    REJECTED_INSUFFICIENT_FUNDS = "REJECTED_INSUFFICIENT_FUNDS"
    REJECTED_INSUFFICIENT_INVENTORY = "REJECTED_INSUFFICIENT_INVENTORY"
    REJECTED_COUNTERPARTY_DECLINED = "REJECTED_COUNTERPARTY_DECLINED"
    REJECTED_ILLEGAL_STATE = "REJECTED_ILLEGAL_STATE"
    REJECTED_SCHEMA_ERROR = "REJECTED_SCHEMA_ERROR"


class ExpertiseDomain(str, Enum):
    FOOD_SUPPLY_CHAIN = "FOOD_SUPPLY_CHAIN"
    TECH_VALUATION = "TECH_VALUATION"
    MACRO_INFLATION = "MACRO_INFLATION"
    CREDIT_SOLVENCY = "CREDIT_SOLVENCY"


# ==============================================================================
# 3. CONFIGURATION CONTRACTS (`SimulationSpec`)
# ==============================================================================
class SimulationMetaSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")
    experiment_id: str = "mvp_bilateral_trust_market_v1"
    random_seed: int = 42
    num_agents: int = Field(default=6, ge=2)
    num_ticks: int = Field(default=8, ge=1)
    environment_type: str = "BILATERAL_TRUST_MARKET"
    db_path: str = "data/mvp_world_state.sqlite"
    telemetry_jsonl_path: str = "data/mvp_telemetry.jsonl"


class HardwareBackendSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")
    backend_type: BackendType = BackendType.WHITE_BOX_PYTORCH
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    models_dir: str = "./models"
    quantization_mode: str = "fp16"
    device: str = "cuda"
    max_new_tokens: int = 180
    temperature: float = 0.4
    batch_size: int = 1
    vram_ceiling_gb: float = 3.2
    black_box_api_url: str = "http://localhost:11434/v1/chat/completions"
    black_box_api_key: str = "EMPTY"


class MechanisticSteeringSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    mode: SteeringMode = SteeringMode.MASTER_EQUATION_MVP
    calibration_pairs_count: int = 16
    target_layer_fraction: float = 0.55
    omega_fin_stress: float = 2.2
    omega_peer_salience: float = 1.8
    gamma_cynicism_scale: float = 0.85
    eta_anxiety_scale: float = 0.65
    kappa_dissonance_sink: float = 2.8
    logit_clip_z_max: float = 4.5
    # Post-MVP parameters preserved for full Master Equation
    lambda_rope_construal: float = 0.35
    alpha_sys1_gate: float = 0.40
    beta_sys2_gate: float = 0.40
    epsilon_mirror_contagion: float = 0.50
    beta_0_temperature: float = 1.0
    zeta_panic: float = 1.5
    zeta_fatigue: float = 0.8
    mu_hebbian_trace: float = 0.60


class EntropyTMSSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    probe_tokens_k: int = 5
    top_p_truncation: float = 0.90
    entropy_threshold_tau: float = 0.68
    communication_cost_xi: float = 0.05


class MemoryCognitionSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")
    stm_capacity: int = 12
    ebbinghaus_base_half_life_ticks: float = 4.0
    consolidation_salience_threshold: float = 1.85
    habit_router_arousal_threshold: float = 0.15
    habit_router_rpe_threshold: float = 0.10
    slovic_pos_trust_rate: float = 0.12
    slovic_neg_trust_rate: float = 0.30


class EconomyInstitutionsSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")
    initial_agent_cash: float = 100.0
    initial_treasury_cash: float = 200.0
    initial_bank_reserves: float = 300.0
    initial_firm_sector_cash: float = 400.0
    base_wage_per_tick: float = 12.0
    survival_food_consumption_per_tick: int = 1
    reference_food_price: float = 15.0
    reference_tech_price: float = 35.0
    sales_tax_rate: float = 0.05
    max_action_retries: int = 2


class SimulationSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")
    simulation_meta: SimulationMetaSpec = Field(default_factory=SimulationMetaSpec)
    hardware_and_backend: HardwareBackendSpec = Field(default_factory=HardwareBackendSpec)
    mechanistic_steering: MechanisticSteeringSpec = Field(default_factory=MechanisticSteeringSpec)
    entropy_tms_router: EntropyTMSSpec = Field(default_factory=EntropyTMSSpec)
    memory_and_cognition: MemoryCognitionSpec = Field(default_factory=MemoryCognitionSpec)
    economy_and_institutions: EconomyInstitutionsSpec = Field(default_factory=EconomyInstitutionsSpec)


# ==============================================================================
# 4. AGENT INTERNAL, EXTERNAL & RELATIONAL STATE SCHEMAS
# ==============================================================================
class BigFiveTraits(BaseModel):
    openness: float = Field(default=0.5, ge=0.0, le=1.0)
    conscientiousness: float = Field(default=0.5, ge=0.0, le=1.0)
    extraversion: float = Field(default=0.5, ge=0.0, le=1.0)
    agreeableness: float = Field(default=0.5, ge=0.0, le=1.0)
    neuroticism: float = Field(default=0.5, ge=0.0, le=1.0)


class ProspectTheoryParams(BaseModel):
    loss_aversion_lambda: float = Field(default=2.25, ge=0.5, le=6.0)
    risk_curvature_gamma: float = Field(default=0.88, ge=0.2, le=1.5)
    discount_factor_delta: float = Field(default=0.95, ge=0.1, le=1.0)


class FehrSchmidtParams(BaseModel):
    disadvantageous_inequity_alpha: float = Field(default=1.2, ge=0.0, le=5.0)
    advantageous_inequity_beta: float = Field(default=0.4, ge=0.0, le=2.0)


class MoodPAD(BaseModel):
    valence: float = Field(default=0.1, ge=-1.0, le=1.0)
    arousal: float = Field(default=0.25, ge=0.0, le=1.0)
    dominance: float = Field(default=0.5, ge=0.0, le=1.0)


class MaslowNeeds(BaseModel):
    survival_satiety: float = Field(default=0.8, ge=0.0, le=1.0)
    safety_security: float = Field(default=0.75, ge=0.0, le=1.0)
    social_belonging: float = Field(default=0.65, ge=0.0, le=1.0)
    esteem_status: float = Field(default=0.5, ge=0.0, le=1.0)


class InternalAgentState(BaseModel):
    """Tier-0 & Dynamic Psychological State (Private to `owner_id`)."""
    agent_id: str
    name: str
    occupation: str
    specialty_domain: ExpertiseDomain
    core_belief_statement: str
    identity_rigidity: float = Field(default=0.5, ge=0.0, le=1.0)
    big_five: BigFiveTraits = Field(default_factory=BigFiveTraits)
    cpt_params: ProspectTheoryParams = Field(default_factory=ProspectTheoryParams)
    fehr_schmidt: FehrSchmidtParams = Field(default_factory=FehrSchmidtParams)
    mood_pad: MoodPAD = Field(default_factory=MoodPAD)
    maslow_needs: MaslowNeeds = Field(default_factory=MaslowNeeds)
    fatigue_level: float = Field(default=0.1, ge=0.0, le=1.0)
    hebbian_rumination_trace: Dict[str, float] = Field(default_factory=dict)


class ExternalAgentState(BaseModel):
    """Objective Material & Legal State (Write-locked exclusively to M8: GameMaster)."""
    agent_id: str
    liquid_wealth_cash: float = Field(default=100.0, ge=0.0)
    bank_deposit: float = Field(default=0.0, ge=0.0)
    outstanding_debt: float = Field(default=0.0, ge=0.0)
    credit_score: float = Field(default=700.0, ge=300.0, le=850.0)
    inventory: Dict[str, int] = Field(default_factory=lambda: {"FOOD": 3, "TECH": 1})
    base_wage: float = Field(default=12.0, ge=0.0)
    social_status: float = Field(default=0.5, ge=0.0, le=1.0)
    legal_status: LegalStatus = LegalStatus.LAW_ABIDING
    spatial_node_id: str = "MARKET_SQUARE"


class RelationalEdge(BaseModel):
    """Directed Dyadic Edge R(owner_id -> target_id)."""
    owner_id: str
    target_id: str
    rel_type: RelationshipType = RelationshipType.ACQUAINTANCE
    trust_tau: float = Field(default=0.1, ge=-1.0, le=1.0)
    intimacy_alpha: float = Field(default=0.1, ge=-1.0, le=1.0)
    credibility_kappa: float = Field(default=0.5, ge=0.0, le=1.0)
    tie_strength_w: float = Field(default=0.2, ge=0.0, le=1.0)
    total_interactions: int = 0
    successful_trades: int = 0
    betrayals_experienced: int = 0


class PeerPerceivedModel(BaseModel):
    """Subjective Theory-of-Mind Model M(owner_id -> target_id)."""
    owner_id: str
    target_id: str
    est_honesty: float = Field(default=0.55, ge=0.0, le=1.0)
    est_greed: float = Field(default=0.45, ge=0.0, le=1.0)
    est_fehr_alpha: float = Field(default=1.2, ge=0.0, le=5.0)
    narrative_summary: str = "A local market participant with standard bargaining habits."


class TMSPeerCompetenceRecord(BaseModel):
    """Bayesian Beta-Bernoulli Transactive Memory Directory Entry."""
    owner_id: str
    target_id: str
    domain: ExpertiseDomain
    alpha_success: float = Field(default=2.0, gt=0.0)
    beta_failure: float = Field(default=2.0, gt=0.0)
    last_queried_tick: int = 0

    @property
    def expected_competence(self) -> float:
        return self.alpha_success / (self.alpha_success + self.beta_failure)


class EpisodicMemoryRecord(BaseModel):
    """STM / LTM Scoped Episodic Memory Entry with Ebbinghaus-McGaugh Metadata."""
    mem_id: str
    owner_id: str
    tick_created: int
    encoding_salience: float = Field(default=1.0, ge=0.1)
    rpe_magnitude: float = Field(default=0.0, ge=0.0)
    rehearsal_count: int = 0
    valence_at_encoding: float = 0.0
    arousal_at_encoding: float = 0.2
    involved_peers: List[str] = Field(default_factory=list)
    narrative_text: str
    is_consolidated_ltm: bool = False


# ==============================================================================
# 5. COGNITIVE APPRAISAL, PROMPT COMPILATION & MECHANISTIC STEERING PACKETS
# ==============================================================================
class EndogenousStatePacket(BaseModel):
    """
    Exact continuous psychological scalars computed by M6 (Appraisal & ToM)
    and consumed by M3 (AttentionSteeringLab) during the LLM forward pass.
    """
    agent_id: str
    counterparty_id: Optional[str] = None
    financial_stress: float = Field(default=0.1, ge=0.0, le=1.0)
    peer_distrust_salience: float = Field(default=0.1, ge=0.0, le=1.0)
    cynicism_gamma: float = Field(default=0.1, ge=0.0, le=1.0)
    anxiety_eta: float = Field(default=0.1, ge=0.0, le=1.0)
    dissonance_score_d: float = Field(default=0.0, ge=0.0, le=1.0)
    arousal_a: float = Field(default=0.25, ge=0.0, le=1.0)
    fatigue_f: float = Field(default=0.1, ge=0.0, le=1.0)
    cognitive_load_l: float = Field(default=0.2, ge=0.0, le=1.0)
    construal_closeness_map: Dict[str, float] = Field(default_factory=dict)
    rpe_delta: float = 0.0
    tom_acceptance_forecast: float = Field(default=0.5, ge=0.0, le=1.0)
    route_to_fast_habit: bool = False


class TokenSpanMap(BaseModel):
    """Maps named semantic prompt spans to exact token index intervals [start_idx, end_idx)."""
    total_prompt_tokens: int
    semantic_probe_Start_idx: int = 0
    spans: Dict[str, List[Tuple[int, int]]] = Field(default_factory=dict)

    def get_token_indices_for_span(self, span_prefix: str) -> List[int]:
        indices: List[int] = []
        for key, intervals in self.spans.items():
            if key == span_prefix or key.startswith(span_prefix):
                for start_idx, end_idx in intervals:
                    indices.extend(range(max(0, start_idx), min(self.total_prompt_tokens, end_idx)))
        return sorted(set(indices))


class CompiledPromptPacket(BaseModel):
    """Output of M7 PromptSpanCompiler ready for White-Box or Black-Box inference."""
    agent_id: str
    tick: int
    raw_prompt_text: str
    input_ids: List[int] = Field(default_factory=list)
    token_span_map: TokenSpanMap
    endogenous_state: EndogenousStatePacket
    incoming_speaker_latent_vec: Optional[List[float]] = None


# ==============================================================================
# 6. STRUCTURED AGENT OUTPUT, TMS ROUTING & GAME MASTER VERDICT CONTRACTS
# ==============================================================================
class EconomicActionIntent(BaseModel):
    """Proposed material action submitted to M8: GameMasterAndHardRuleLedger."""
    action_category: ActionCategory = ActionCategory.BILATERAL_TRADE
    target_peer_id: Optional[str] = None
    item_id: str = "FOOD"
    quantity: int = Field(default=1, ge=0)
    total_price_cash: float = Field(default=15.0, ge=0.0)
    is_buying: bool = True
    public_contribution_cash: float = Field(default=0.0, ge=0.0)

    @field_validator("item_id")
    @classmethod
    def normalize_item(cls, v: str) -> str:
        v_up = v.strip().upper()
        return v_up if v_up in {"FOOD", "TECH", "NONE"} else "FOOD"


class TMSQueryIntent(BaseModel):
    """Triggered automatically when Semantic-Offset Entropy Probe exceeds tau_TMS."""
    queried_expert_id: str
    domain: ExpertiseDomain = ExpertiseDomain.FOOD_SUPPLY_CHAIN
    question_text: str = "What is the fair clearing price and supply outlook right now?"
    measured_entropy_bar: float = 0.0


class AgentTurnOutput(BaseModel):
    """
    Strict JSON Schema emitted by the Agent LLM (or Habit Automaton / TMS Router).
    Note: `internal_thought` is always placed FIRST so the Semantic-Offset Entropy
    Probe evaluates unconstrained reasoning tokens right after `{"internal_thought": "`.
    """
    internal_thought: str = Field(
        default="Evaluating my budget constraints and my trust in the counterparty."
    )
    epistemic_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    consent_decision: ConsentDecision = ConsentDecision.INITIATE
    verbal_message: str = Field(default="Let us make a mutually beneficial trade.")
    perceived_honesty_delta: float = Field(default=0.0, ge=-1.0, le=1.0)
    economic_intent: EconomicActionIntent = Field(default_factory=EconomicActionIntent)
    tms_query: Optional[TMSQueryIntent] = None


class TMSRoutingDecision(BaseModel):
    halted_generation: bool = False
    normalized_entropy: float = 0.0
    selected_expert_id: Optional[str] = None
    domain: ExpertiseDomain = ExpertiseDomain.FOOD_SUPPLY_CHAIN
    expert_advice_injected: Optional[str] = None


class GameMasterVerdict(BaseModel):
    """Immutable Settlement Receipt returned by M8: AccountingAndLawEngine."""
    tx_id: str
    tick: int
    status: VerdictStatus
    initiator_id: str
    responder_id: Optional[str] = None
    cash_delta_initiator: float = 0.0
    cash_delta_responder: float = 0.0
    tax_collected: float = 0.0
    item_id: str = "NONE"
    quantity_transferred: int = 0
    diagnostic_feedback: str = "Transaction settled atomically."
    global_money_conservation_error: float = 0.0


class SteeringTelemetryPacket(BaseModel):
    """Captures exact mechanistic attention mass shifts for observability & pre-flight tests."""
    agent_id: str
    tick: int
    steering_mode: SteeringMode
    attn_mass_drives_finance: float = 0.0
    attn_mass_peer_memory: float = 0.0
    attn_mass_incoming_msg: float = 0.0
    attn_mass_sink_bos: float = 0.0
    value_projection_norm_ratio: float = 1.0
    probed_shannon_entropy: float = 0.0
    tms_triggered: bool = False


class MacroEconomyState(BaseModel):
    """Godley-Lavoie Stock-Flow Consistent Macroeconomic & Institutional Snapshot."""
    tick: int = 0
    central_bank_base_money_m0: float = 1500.0
    commercial_bank_reserves: float = 300.0
    public_treasury_cash: float = 200.0
    firm_sector_cash: float = 400.0
    market_clearing_price_food: float = 15.0
    market_clearing_price_tech: float = 35.0
    inflation_rate: float = 0.0
    societal_trust_index: float = 0.1
    wealth_gini: float = 0.0
    cumulative_trade_volume: float = 0.0
    cumulative_tms_queries: int = 0


# ==============================================================================
# 7. MODULE PROTOCOLS (STRUCTURAL INTERFACES)
# ==============================================================================
@runtime_checkable
class InferenceBrainProtocol(Protocol):
    def generate_turn(
        self,
        prompt_packet: CompiledPromptPacket,
        domain_hint: ExpertiseDomain,
    ) -> Tuple[AgentTurnOutput, SteeringTelemetryPacket, Optional[List[float]]]:
        ...


@runtime_checkable
class GameMasterProtocol(Protocol):
    def pre_validate_and_escrow(
        self,
        tick: int,
        initiator_id: str,
        turn_output: AgentTurnOutput,
    ) -> GameMasterVerdict:
        ...

    def settle_dyadic_transaction(
        self,
        tick: int,
        escrow_verdict: GameMasterVerdict,
         initiator_output: AgentTurnOutput,
        responder_id: str,
        responder_output: AgentTurnOutput,
    ) -> GameMasterVerdict:
        ...

    def verify_global_money_conservation(self) -> Tuple[bool, float]:
        ...