from __future__ import annotations

import argparse
import inspect
import os
from typing import Any, Dict, List
import yaml
import torch

from src.brain.inference_brain import DualBackendInferenceBrain
from src.brain.model_vault import ModelVaultManager
from src.core.contracts import (
    AgentTurnOutput,
    CompiledPromptPacket,
    EconomicActionIntent,
    EndogenousStatePacket,
    FourDimTickTelemetry,
    RawGenerationResult,
    SimulationConfig,
)
from src.environment.bilateral_market import BilateralMarketEngine
from src.environment.game_master import AccountingAndLawEngine
from src.evaluation.live_monitor import RealTimeLiveMonitor
from src.evaluation.metrics import FourDimensionalMetricsEngine
from src.memory.scoped_memory import MultiTierScopedMemory
from src.orchestration.cognitive_appraisal import CognitiveAppraisalAndToM
from src.orchestration.turn_orchestrator import TurnOrchestrator
from src.steering.attention_lab import AttentionSteeringLab
from src.steering.entropy_tms import EntropyTMSRouter


class FastDeterministicBrain:
    """
    Lightweight, deterministic behavioral-economics brain used when `--mock-brain`
    is passed for instant pre-flight/ablation verification without GPU overhead.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.tokenizer = None

    def generate_turn(
        self,
        prompt_packet: CompiledPromptPacket,
        steering_packet: EndogenousStatePacket,
    ) -> RawGenerationResult:
        f_it = steering_packet.scarcity_deficit_f
        t_eff = steering_packet.effective_trust_01
        d_it = steering_packet.dissonance_d
        arousal = steering_packet.arousal_a
        fatigue = steering_packet.fatigue_f

        beta_lc = self.config.steering_spec.beta_base_temp * (
            (
                1.0
                + self.config.steering_spec.zeta_panic
                * max(0.0, arousal - self.config.steering_spec.optimal_arousal) ** 2
            )
            / (1.0 + self.config.steering_spec.zeta_fatigue * fatigue)
        )
        attn_fin = min(0.75, 0.22 + 0.45 * f_it)
        attn_sink = min(0.65, 0.08 + 0.48 * d_it)
        attn_peer = max(0.08, 0.35 - 0.30 * attn_fin)
        attn_msg = max(0.08, 1.0 - (attn_fin + attn_sink + attn_peer))

        thought = (
            f"Evaluating HANK portfolio under Scarcity F={f_it:.2f} and Effective Trust T_eff={t_eff:.2f}."
        )
        msg = (
            "I need an urgent survival trade within fair bounds."
            if f_it > 0.45
            else "Proposing a mutually beneficial bilateral exchange."
        )
        out = AgentTurnOutput(
            internal_thought=thought,
            consent_decision="ACCEPT" if t_eff >= 0.25 else "REJECT",
            action_category="COMMERCIAL_TRADE",
            economic_intent=EconomicActionIntent(
                action_type="BUY",
                counterparty_id=steering_packet.partner_id or "",
                item_id="CONSUMABLE_NECESSITY",
                quantity=1.0,
                unit_price=12.0,
            ),
            verbal_message=msg,
        )
        mirror_vec = torch.full(
            (1, 64), fill_value=-0.45 * f_it + 0.25 * (2.0 * t_eff - 1.0)
        )
        return RawGenerationResult(
            parsed_output=out,
            tms_halt_triggered=False,
            mean_shannon_entropy=0.32 + 0.22 * f_it,
            emitted_mirror_vector=mirror_vec,
            attention_telemetry={
                "yerkes_dodson_beta": round(beta_lc, 4),
                "SPAN_DRIVES_AND_FINANCE": round(attn_fin, 4),
                "SPAN_PEER_MEMORY": round(attn_peer, 4),
                "SPAN_INCOMING_MSG": round(attn_msg, 4),
                "SPAN_SINK": round(attn_sink, 4),
            },
            updated_hebbian_trace=[
                round(0.5 * x + 0.2 * f_it, 4)
                for x in steering_packet.hebbian_trace_h
            ],
        )


def _bootstrap_whitebox_brain(
    config: SimulationConfig, tms_router: EntropyTMSRouter
) -> Any:
    """
    Signature-adaptive bootstrapper that mounts `ModelVaultManager`,
    `AttentionSteeringLab`, and `DualBackendInferenceBrain` using their exact
    constructor and method signatures on disk.
    """
    # 1. Instantiate ModelVaultManager
    v_sig = inspect.signature(ModelVaultManager.__init__)
    v_params = [p for p in v_sig.parameters.keys() if p != "self"]
    if not v_params:
        vault = ModelVaultManager()  # type: ignore[call-arg]
    elif "model_spec" in v_params:
        vault = ModelVaultManager(config.model_spec)  # type: ignore[call-arg]
    else:
        vault = ModelVaultManager(config)  # type: ignore[call-arg]

    # Call mount/bootstrap method on vault
    mount_fn = (
        getattr(vault, "mount_model_and_calibrate", None)
        or getattr(vault, "load_and_calibrate", None)
        or getattr(vault, "bootstrap", None)
        or getattr(vault, "initialize", None)
        or getattr(vault, "load_model", None)
    )
    mount_res = mount_fn() if mount_fn is not None else None

    model = getattr(vault, "model", None)
    tokenizer = getattr(vault, "tokenizer", None)
    calibration_bundle = getattr(vault, "calibration_bundle", None)

    if isinstance(mount_res, tuple):
        if len(mount_res) == 3:
            model, tokenizer, calibration_bundle = mount_res
        elif len(mount_res) == 2:
            model, tokenizer = mount_res
    elif mount_res is not None:
        model = getattr(mount_res, "model", model)
        tokenizer = getattr(mount_res, "tokenizer", tokenizer)
        calibration_bundle = getattr(
            mount_res, "calibration_bundle", calibration_bundle or mount_res
        )

    # 2. Instantiate AttentionSteeringLab
    s_sig = inspect.signature(AttentionSteeringLab.__init__)
    s_kwargs: Dict[str, Any] = {}
    for p in s_sig.parameters.keys():
        if p == "self":
            continue
        pl = p.lower()
        if "calib" in pl or "bundle" in pl:
            s_kwargs[p] = calibration_bundle
        elif "steer" in pl or "spec" in pl:
            s_kwargs[p] = config.steering_spec
        elif "config" in pl:
            annot = str(s_sig.parameters[p].annotation)
            s_kwargs[p] = (
                config if "SimulationConfig" in annot else config.steering_spec
            )
    steering_lab = (
        AttentionSteeringLab(**s_kwargs)
        if s_kwargs
        else AttentionSteeringLab(config.steering_spec, calibration_bundle)  # type: ignore[call-arg]
    )

    # 3. Instantiate DualBackendInferenceBrain
    b_sig = inspect.signature(DualBackendInferenceBrain.__init__)
    b_kwargs: Dict[str, Any] = {}
    for p in b_sig.parameters.keys():
        if p == "self":
            continue
        pl = p.lower()
        if "vault" in pl:
            b_kwargs[p] = vault
        elif "model_spec" in pl:
            b_kwargs[p] = config.model_spec
        elif "config" in pl:
            b_kwargs[p] = config
        elif "model" in pl and "spec" not in pl:
            b_kwargs[p] = model
        elif "tok" in pl:
            b_kwargs[p] = tokenizer
        elif "steer" in pl or "lab" in pl:
            b_kwargs[p] = steering_lab
        elif "tms" in pl or "router" in pl:
            b_kwargs[p] = tms_router
        elif "calib" in pl:
            b_kwargs[p] = calibration_bundle

    return DualBackendInferenceBrain(**b_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Neuropolis 4-Dimensional HANK & Mechanistic Social Simulation Runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mvp_config.yaml",
        help="Path to YAML experiment configuration",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=None,
        help="Override total simulation ticks",
    )
    parser.add_argument(
        "--omega",
        type=str,
        default=None,
        help="Comma-separated Orthogonal Clamping Vector Omega in {0,1}^4, e.g. '1,1,1,1' or '1,0,1,0'",
    )
    parser.add_argument(
        "--crisis",
        type=str,
        default=None,
        choices=[
            "NONE",
            "STAGFLATION_SUBSISTENCE_SHOCK",
            "ASYMMETRIC_UNEMPLOYMENT_SHOCK",
            "ASSET_VALUATION_CRASH",
            "LIQUIDITY_CREDIT_CRUNCH",
        ],
        help="Override active macroeconomic crisis shock scenario",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/mvp_world_state.sqlite",
        help="Path to SQLite world state database",
    )
    parser.add_argument(
        "--mock-brain",
        action="store_true",
        help="Use fast deterministic brain for instant verification without loading LLM weights",
    )
    return parser.parse_args()


def run_simulation() -> List[FourDimTickTelemetry]:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    config = SimulationConfig(**raw_cfg)

    if args.ticks is not None:
        config.total_ticks = int(args.ticks)
    if args.omega is not None:
        parts = [int(x.strip()) for x in args.omega.split(",")]
        if len(parts) != 4:
            raise ValueError("--omega must specify 4 comma-separated bits: e.g. 1,1,1,1")
        config.orthogonal_clamping.omega_d1_scarcity_and_hank = bool(parts[0])
        config.orthogonal_clamping.omega_d2_trust_and_parochialism = bool(parts[1])
        config.orthogonal_clamping.omega_d3_contagion_and_polarization = bool(parts[2])
        config.orthogonal_clamping.omega_d4_fiscal_and_legal_state = bool(parts[3])
    if args.crisis is not None:
        config.hank_and_crisis_spec.active_crisis = args.crisis  # type: ignore[assignment]

    os.makedirs(os.path.dirname(args.db_path) or ".", exist_ok=True)
    if os.path.exists(args.db_path):
        os.remove(args.db_path)

    # 1. Initialize Deterministic Environment, Memory, Appraisal, and TMS
    game_master = AccountingAndLawEngine(db_path=args.db_path, config=config)
    market_engine = BilateralMarketEngine(config=config, game_master=game_master)
    memory_engine = MultiTierScopedMemory(db_path=args.db_path, config=config)
    appraisal_engine = CognitiveAppraisalAndToM(
        config=config, game_master=game_master, memory_engine=memory_engine
    )
    tms_router = EntropyTMSRouter(db_path=args.db_path, config=config)

    # 2. Initialize White-Box Steered Brain (or FastDeterministicBrain if --mock-brain)
    if args.mock_brain:
        brain: Any = FastDeterministicBrain(config=config)
    else:
        brain = _bootstrap_whitebox_brain(config=config, tms_router=tms_router)

    orchestrator = TurnOrchestrator(
        config=config,
        game_master=game_master,
        market_engine=market_engine,
        memory_engine=memory_engine,
        appraisal_engine=appraisal_engine,
        tms_router=tms_router,
        brain=brain,
    )
    metrics_engine = FourDimensionalMetricsEngine(
        db_path=args.db_path,
        config=config,
        game_master=game_master,
        memory_engine=memory_engine,
    )
    monitor = RealTimeLiveMonitor(config=config)

    history: List[FourDimTickTelemetry] = []
    agent_ids = [a.agent_id for a in config.agents]

    # 3. Main Tick Loop
    for tick in range(1, config.total_ticks + 1):
        macro_state = game_master.step_crisis_regime(tick=tick)
        monitor.log_tick_header(tick=tick, macro=macro_state)

        eff_trust_matrix = memory_engine.get_effective_trust_matrix()
        pairs = market_engine.match_conflict_free_pairs(
            tick=tick,
            agent_ids=agent_ids,
            effective_trust_matrix=eff_trust_matrix,
        )

        for p_idx, (init_id, resp_id) in enumerate(pairs, start=1):
            outcome = orchestrator.execute_dyadic_pair_interaction(
                tick=tick,
                initiator_id=init_id,
                responder_id=resp_id,
            )
            monitor.log_dyadic_interaction(pair_idx=p_idx, outcome=outcome)

        upkeep_report = game_master.execute_end_of_tick_upkeep_and_fiscal_state(
            tick=tick
        )
        monitor.log_upkeep_and_sanctions(tick=tick, upkeep_report=upkeep_report)

        tick_telemetry = metrics_engine.compute_and_persist_tick_telemetry(
            tick=tick,
            upkeep_report=upkeep_report,
            cumulative_tms_queries=orchestrator.cumulative_tms_queries,
            successful_tms_routes=orchestrator.successful_tms_routes,
        )
        monitor.log_four_dim_tick_dashboard(tick_telemetry)
        history.append(tick_telemetry)

    scorecard = metrics_engine.compute_empirical_validation_scorecard(history)
    monitor.print_final_empirical_scorecard(history=history, scorecard=scorecard)
    return history


if __name__ == "__main__":
    run_simulation()