#!/usr/bin/env python3
"""
run_mvp.py
End-to-End Executable Entrypoint for the Modular Mechanistic Social Simulation MVP.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import List

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.brain.inference_brain import (
    BlackBoxAPIBrain,
    PromptSpanCompiler,
    WhiteBoxPyTorchBrain,
)
from src.brain.model_vault import ModelVaultManager
from src.core.contracts import (
    BackendType,
    MacroEconomyState,
    SimulationSpec,
    VerdictStatus,
)
from src.environment.bilateral_market import BilateralTrustMarket
from src.environment.game_master import AccountingAndLawEngine
from src.evaluation.metrics import BenchmarkEvaluator
from src.memory.scoped_memory import ScopedMemoryEngine
from src.orchestration.cognitive_appraisal import CognitiveAppraisalAndToMEngine
from src.orchestration.scheduler import AIMDConcurrencyController, ConflictFreeBatcher
from src.orchestration.turn_orchestrator import TurnExecutionRecord, TurnOrchestrator
from src.run_helpers import synthesize_mvp_population
from src.steering.attention_lab import AttentionSteeringLab, MechanisticCalibrator
from src.steering.entropy_tms import EntropyTMSRouter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mechanistic Social Simulation MVP")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mvp_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=None,
        help="Optional override for number of simulation ticks",
    )
    args = parser.parse_args()

    console = Console()
    with open(args.config, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    spec = SimulationSpec.model_validate(raw_cfg)
    if args.ticks is not None:
        spec.simulation_meta.num_ticks = args.ticks

    seed = spec.simulation_meta.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    console.print(
        Panel.fit(
            f"[bold cyan]MODULAR MECHANISTIC SOCIAL SIMULATION & BEHAVIORAL ECONOMICS ENGINE (MVP)[/bold cyan]\n"
            f"Experiment: [yellow]{spec.simulation_meta.experiment_id}[/yellow] | "
            f"Model: [green]{spec.hardware_and_backend.model_id}[/green] | "
            f"Steering: [magenta]{spec.mechanistic_steering.mode.value}[/magenta] | "
            f"Agents: [bold]{spec.simulation_meta.num_agents}[/bold] | "
            f"Ticks: [bold]{spec.simulation_meta.num_ticks}[/bold]",
            border_style="cyan",
        )
    )

    if os.path.exists(spec.simulation_meta.db_path):
        os.remove(spec.simulation_meta.db_path)
    memory_engine = ScopedMemoryEngine(
        db_path=spec.simulation_meta.db_path,
        spec=spec.memory_and_cognition,
    )
    internal_states, external_states, macro_init = synthesize_mvp_population(spec)
    memory_engine.register_population(internal_states, external_states, macro_init)
    agent_ids = [a.agent_id for a in internal_states]

    tms_router = EntropyTMSRouter(spec.entropy_tms_router)
    vault = ModelVaultManager(spec.hardware_and_backend)

    if spec.hardware_and_backend.backend_type == BackendType.WHITE_BOX_PYTORCH:
        console.print("[cyan]STAGE 1/2:[/cyan] Loading White-Box Model & Running Contrastive SVD Calibration...")
        t0 = time.time()
        model, tokenizer, gate = vault.load_whitebox_model_and_tokenizer()
        calibrator = MechanisticCalibrator(spec.mechanistic_steering)
        calibration_bundle = calibrator.calibrate_or_load(
            model=model,
            tokenizer=tokenizer,
            cache_path=vault.get_calibration_cache_path(),
            device=gate.device,
        )
        steering_lab = AttentionSteeringLab(
            spec=spec.mechanistic_steering,
            calibration=calibration_bundle,
        )
        prompt_compiler = PromptSpanCompiler(tokenizer=tokenizer)
        brain = WhiteBoxPyTorchBrain(
            model=model,
            tokenizer=tokenizer,
            hw_spec=spec.hardware_and_backend,
            steering_lab=steering_lab,
            tms_router=tms_router,
            device=gate.device,
        )
        console.print(
            f"[green]✓ White-Box Brain Ready[/green] on [bold]{gate.device}[/bold] "
            f"in {time.time() - t0:.2f}s (Calibrated Layers: {calibration_bundle.target_layer_indices})"
        )
    else:
        prompt_compiler = PromptSpanCompiler(tokenizer=None)
        brain = BlackBoxAPIBrain(hw_spec=spec.hardware_and_backend, tms_router=tms_router)
        console.print("[yellow]Running in Black-Box API Mode (NoOpBypassSteerer)[/yellow]")

    game_master = AccountingAndLawEngine(
        memory_engine=memory_engine,
        spec=spec.economy_and_institutions,
    )
    market = BilateralTrustMarket(
        memory_engine=memory_engine,
        econ_spec=spec.economy_and_institutions,
        cog_spec=spec.memory_and_cognition,
        rng=rng,
    )
    appraisal_engine = CognitiveAppraisalAndToMEngine(
        cog_spec=spec.memory_and_cognition,
        econ_spec=spec.economy_and_institutions,
    )
    orchestrator = TurnOrchestrator(
        spec=spec,
        memory_engine=memory_engine,
        brain=brain,
        prompt_compiler=prompt_compiler,
        appraisal_engine=appraisal_engine,
        game_master=game_master,
        market=market,
    )
    aimd_controller = AIMDConcurrencyController(spec.hardware_and_backend)
    evaluator = BenchmarkEvaluator(
        memory_engine=memory_engine,
        jsonl_path=spec.simulation_meta.telemetry_jsonl_path,
    )

    tick_history: List[MacroEconomyState] = []
    total_approved_trades = 0

    for tick in range(1, spec.simulation_meta.num_ticks + 1):
        tick_start = time.time()
        hw_stats = aimd_controller.probe_hardware_and_adjust()

        if tick == 4:
            broadcast = (
                "URGENT MARKET SHOCK: Rumors suggest Agent_2 and Agent_4 are hoarding TECH "
                "and exploiting scarcity; food supply chains face temporary price pressure!"
            )
        else:
            broadcast = (
                f"Official Tick {tick} Bulletin: Reference FOOD=${macro_init.market_clearing_price_food:.2f}, "
                f"TECH=${macro_init.market_clearing_price_tech:.2f}. Mention Agent_3 for macro advice."
            )

        directed_requests = market.select_interaction_pairs_mnl(agent_ids)
        matchings = ConflictFreeBatcher.compute_disjoint_matchings(directed_requests)

        tick_records: List[TurnExecutionRecord] = []
        for matching in matchings:
            for initiator_id, responder_id in matching:
                rec = orchestrator.execute_two_wave_dyadic_interaction(
                    tick=tick,
                    initiator_id=initiator_id,
                    responder_id=responder_id,
                    market_broadcast_msg=broadcast,
                )
                tick_records.append(rec)
                if rec.final_verdict.status == VerdictStatus.APPROVED and rec.final_verdict.quantity_transferred > 0:
                    total_approved_trades += 1

        game_master.execute_end_of_tick_upkeep(tick=tick)
        conservation_ok, conservation_err = game_master.verify_global_money_conservation()

        macro_snap = evaluator.compute_and_log_tick_metrics(
            tick=tick,
            agent_ids=agent_ids,
            conservation_ok=conservation_ok,
            conservation_error=conservation_err,
            hw_stats=hw_stats,
        )
        tick_history.append(macro_snap)

        console.print(
            f"[bold cyan]Tick {tick:02d}/{spec.simulation_meta.num_ticks:02d}[/bold cyan] "
            f"({time.time() - tick_start:.1f}s) | "
            f"Pairs: {len(tick_records)} | "
            f"Trust Index: [bold]{macro_snap.societal_trust_index:+.3f}[/bold] | "
            f"Gini: [yellow]{macro_snap.wealth_gini:.3f}[/yellow] | "
            f"Trade Vol: [green]${macro_snap.cumulative_trade_volume:.2f}[/green] | "
            f"TMS Queries: [magenta]{macro_snap.cumulative_tms_queries}[/magenta] | "
            f"VRAM: {hw_stats['vram_allocated_gb']:.2f}GB | "
            f"ACID Drift: {conservation_err:.2e}"
        )

    summary_table = Table(title="MVP Simulation Trajectory & Conservation Audit")
    summary_table.add_column("Tick", justify="right", style="cyan")
    summary_table.add_column("Trust Index", justify="right", style="green")
    summary_table.add_column("Wealth Gini", justify="right", style="yellow")
    summary_table.add_column("Food Price", justify="right")
    summary_table.add_column("Cumul. Volume", justify="right", style="bold green")
    summary_table.add_column("TMS Queries", justify="right", style="magenta")
    summary_table.add_column("Treasury Cash", justify="right")

    for snap in tick_history:
        summary_table.add_row(
            str(snap.tick),
            f"{snap.societal_trust_index:+.4f}",
            f"{snap.wealth_gini:.4f}",
            f"${snap.market_clearing_price_food:.2f}",
            f"${snap.cumulative_trade_volume:.2f}",
            str(snap.cumulative_tms_queries),
            f"${snap.public_treasury_cash:.2f}",
        )
    console.print("\n", summary_table)

    final_ok, final_err = game_master.verify_global_money_conservation()
    status_str = "[bold green]PASS (Exact Conservation)[/bold green]" if final_ok else "[bold red]FAIL[/bold red]"
    console.print(
        Panel.fit(
            f"Global Money Supply Conservation Check: {status_str} (Drift Error = {final_err:.3e})\n"
            f"Total Approved Bilateral Trades: [bold]{total_approved_trades}[/bold] | "
            f"Telemetry Saved: [underline]{spec.simulation_meta.telemetry_jsonl_path}[/underline]",
            border_style="green" if final_ok else "red",
        )
    )


if __name__ == "__main__":
    main()
