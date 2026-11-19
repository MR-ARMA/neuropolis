from __future__ import annotations

import argparse
import os
from typing import List
import yaml

from src.brain.inference_brain import DualBackendInferenceBrain
from src.brain.model_vault import ModelVaultManager
from src.core.contracts import (
    FourDimTickTelemetry,
    SimulationConfig,
)
from src.core.seed_bundle import SeedBundle
from src.environment.bilateral_market import BilateralMarketEngine
from src.environment.game_master import AccountingAndLawEngine
from src.environment.populations import synthesize_parametric_population
from src.evaluation.live_monitor import RealTimeLiveMonitor
from src.evaluation.metrics import FourDimensionalMetricsEngine
from src.memory.scoped_memory import MultiTierScopedMemory
from src.orchestration.cognitive_appraisal import CognitiveAppraisalAndToM
from src.orchestration.turn_orchestrator import TurnOrchestrator
from src.steering.attention_lab import AttentionSteeringLab
from src.steering.entropy_tms import EntropyTMSRouter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Neuropolis Thesis-Grade Benchmark & Ablation Runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mvp_config.yaml",
        help="Path to YAML experiment configuration",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="laptop",
        choices=["laptop", "kaggle"],
        help="Hardware preset: laptop (Qwen2.5-3B NF4) or kaggle (Qwen2.5-7B NF4 on T4)",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default=None,
        choices=["mvp_6", "dev", "main", "scale"],
        help="Population scale: mvp_6 (N=6), dev (N=12), main (N=24), scale (N=48)",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=None,
        help="Override total simulation ticks",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override master random seed",
    )
    parser.add_argument(
        "--decision-mode",
        type=str,
        default=None,
        choices=["LLM_AUTONOMOUS", "RULE_ONLY", "ZERO_INTELLIGENCE"],
        help="Policy mode: LLM_AUTONOMOUS, RULE_ONLY, or ZERO_INTELLIGENCE",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        choices=["NONE", "TEXT_ONLY", "STEER_ONLY", "TEXT_AND_STEER", "CUE_CONFLICT"],
        help="State delivery channel (C0..C4)",
    )
    parser.add_argument(
        "--steering-method",
        type=str,
        default=None,
        choices=[
            "NONE",
            "PLACEBO_RANDOM",
            "SHUFFLED_STATE",
            "WRONG_SPAN",
            "REVERSED_SIGN",
            "PASTA",
            "ACTADD_CAA",
            "OUR_COUPLED_STEER",
        ],
        help="Active steering intervention or control",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default=None,
        choices=["prefill_only", "prefill_and_decode"],
        help="Steering hook scope (D4 factor)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override additive pre-softmax logit bias dose alpha",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override decoding temperature (0.7 distributional, 0.0 deterministic)",
    )
    parser.add_argument(
        "--tms-mode",
        type=str,
        default=None,
        choices=[
            "OFF",
            "ENTROPY_TRIGGER",
            "RANDOM_TRIGGER",
            "ALWAYS_ASK",
            "ORACLE_TRIGGER",
        ],
        help="Epistemic routing condition",
    )
    parser.add_argument(
        "--omega",
        type=str,
        default=None,
        help="Comma-separated Orthogonal Clamping Vector Omega in {0,1}^4",
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
        default="results/world_state.sqlite",
        help="Path to SQLite world state database (untracked in results/)",
    )
    parser.add_argument(
        "--mock-brain",
        action="store_true",
        help="Run in RULE_ONLY baseline mode without loading LLM weights",
    )
    return parser.parse_args()


def run_simulation() -> List[FourDimTickTelemetry]:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    config = SimulationConfig(**raw_cfg)

    if args.profile == "kaggle":
        os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
        config.model_spec.model_id = "Qwen/Qwen2.5-7B-Instruct"
        config.model_spec.local_dir = "/tmp/models/Qwen--Qwen2.5-7B-Instruct"
        config.model_spec.precision = "float16"
        config.model_spec.load_in_4bit = True

    if args.scale is not None:
        config.population_scale = args.scale  # type: ignore[assignment]
    if args.ticks is not None:
        config.total_ticks = int(args.ticks)
    if args.seed is not None:
        config.random_seed = int(args.seed)
    if args.mock_brain:
        config.experimental_design.decision_mode = "RULE_ONLY"
    if args.decision_mode is not None:
        config.experimental_design.decision_mode = args.decision_mode  # type: ignore[assignment]
    if args.channel is not None:
        config.experimental_design.state_channel = args.channel  # type: ignore[assignment]
    if args.steering_method is not None:
        config.experimental_design.steering_method = args.steering_method  # type: ignore[assignment]
        config.steering_spec.mode = args.steering_method
    if args.scope is not None:
        config.steering_spec.scope = args.scope  # type: ignore[assignment]
    if args.alpha is not None:
        config.steering_spec.alpha_dose = float(args.alpha)
    if args.temperature is not None:
        config.model_spec.temperature = float(args.temperature)
    if args.tms_mode is not None:
        config.experimental_design.tms_mode = args.tms_mode  # type: ignore[assignment]
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

    seed_bundle = SeedBundle(config.random_seed)
    config.agents = synthesize_parametric_population(config, seed_bundle)

    os.makedirs(os.path.dirname(args.db_path) or ".", exist_ok=True)
    if os.path.exists(args.db_path):
        os.remove(args.db_path)

    game_master = AccountingAndLawEngine(db_path=args.db_path, config=config)
    market_engine = BilateralMarketEngine(config=config, game_master=game_master)
    memory_engine = MultiTierScopedMemory(db_path=args.db_path, config=config)
    appraisal_engine = CognitiveAppraisalAndToM(
        config=config, game_master=game_master, memory_engine=memory_engine
    )
    tms_router = EntropyTMSRouter(config=config, db_path=args.db_path)

    if config.experimental_design.decision_mode in ("RULE_ONLY", "ZERO_INTELLIGENCE"):
        steering_lab = AttentionSteeringLab(config=config.steering_spec)
        brain = DualBackendInferenceBrain(
            config=config,
            model=None,
            tokenizer=None,
            steering_lab=steering_lab,
            tms_router=tms_router,
            seed_bundle=seed_bundle,
        )
    else:
        vault = ModelVaultManager(config=config)
        model, tokenizer, calibration_bundle = vault.mount_model_and_calibrate()
        steering_lab = AttentionSteeringLab(
            config=config.steering_spec, calibration=calibration_bundle
        )
        brain = DualBackendInferenceBrain(
            config=config,
            model=model,
            tokenizer=tokenizer,
            steering_lab=steering_lab,
            tms_router=tms_router,
            seed_bundle=seed_bundle,
        )

    run_id = (
        f"run_s{config.random_seed}_{config.experimental_design.decision_mode}_"
        f"{config.experimental_design.state_channel}_{config.experimental_design.steering_method}"
    )
    orchestrator = TurnOrchestrator(
        config=config,
        game_master=game_master,
        market_engine=market_engine,
        memory_engine=memory_engine,
        appraisal_engine=appraisal_engine,
        tms_router=tms_router,
        brain=brain,
        seed_bundle=seed_bundle,
        run_id=run_id,
    )
    metrics_engine = FourDimensionalMetricsEngine(
        db_path=args.db_path,
        config=config,
        game_master=game_master,
        memory_engine=memory_engine,
    )
    monitor = RealTimeLiveMonitor(config=config, seed_bundle=seed_bundle)

    history: List[FourDimTickTelemetry] = []
    agent_ids = [a.agent_id for a in config.agents]

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
            tick=tick,
            pg_contributions=orchestrator.tick_pg_contributions,
            tax_evasion_intents=orchestrator.tick_tax_evasion_intents,
        )
        orchestrator.tick_pg_contributions.clear()
        orchestrator.tick_tax_evasion_intents.clear()

        monitor.log_upkeep_and_sanctions(tick=tick, upkeep_report=upkeep_report)

        tick_telemetry = metrics_engine.compute_and_persist_tick_telemetry(
            tick=tick,
            upkeep_report=upkeep_report,
            cumulative_tms_queries=orchestrator.cumulative_tms_queries,
            successful_tms_routes=orchestrator.successful_tms_routes,
            llm_valid_rate=orchestrator.get_llm_valid_rate(),
            abstain_rate=orchestrator.get_abstain_rate(),
            rule_fallback_rate=orchestrator.get_rule_fallback_rate(),
            tms_entropy_auroc=tms_router.compute_entropy_ignorance_auroc(),
        )
        monitor.log_four_dim_tick_dashboard(tick_telemetry)
        history.append(tick_telemetry)

    monitor.write_run_manifest_and_turn_logs(
        run_id=run_id,
        turn_logs=orchestrator.turn_log_rows,
        llm_valid_rate=orchestrator.get_llm_valid_rate(),
        abstain_rate=orchestrator.get_abstain_rate(),
        rule_fallback_rate=orchestrator.get_rule_fallback_rate(),
        manifest_path=f"results/manifest_{run_id}.json",
        turn_log_path=f"results/turn_logs_{run_id}.jsonl",
    )
    scorecard = metrics_engine.compute_empirical_validation_scorecard(history)
    monitor.print_final_empirical_scorecard(history=history, scorecard=scorecard)
    return history


if __name__ == "__main__":
    run_simulation()