from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import dataclasses
import hashlib
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
import yaml

from src.brain.memory_tms import EpisodicMemoryTMS
from src.core.contracts import (
    FactorialCellSpec,
    PolicyMode,
    RepEMode,
    SimulationConfig,
)
from src.environment.game_master import GameMaster, TickMacroSummary
from src.evaluation.metrics import (
    ActiveWindfallShadowState,
    EconomyMetricsCollector,
    EconomySeedClusterMetrics,
    ExogenousWindfallMPCProbe,
)
from src.evaluation.validity_and_stats import (
    clustered_paired_contrast,
    compute_2x2x2_factorial_effects_and_shapley,
)
from src.orchestration.turn_orchestrator import TurnOrchestrator


# ============================================================================
# 1. COMPLETE 2^3 FACTORIAL ABLATION MATRIX + MATCHED SCIENTIFIC CONTROLS
# ============================================================================

def build_full_factorial_and_control_specs() -> Dict[str, FactorialCellSpec]:
    specs: Dict[str, FactorialCellSpec] = {}
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                cid = f"C{a}{b}{c}"
                specs[cid] = FactorialCellSpec(
                    cell_id=cid,
                    deliberation_bottleneck=bool(a),
                    memory_friction=bool(b),
                    endogenous_repe=bool(c),
                    repe_mode=(
                        RepEMode.ENDOGENOUS_PROBE_GATED if c == 1 else RepEMode.OFF
                    ),
                    policy_mode=PolicyMode.LLM_AGENT,
                )

    specs["CTRL_PLACEBO_VEC"] = FactorialCellSpec(
        cell_id="CTRL_PLACEBO_VEC",
        deliberation_bottleneck=True,
        memory_friction=True,
        endogenous_repe=True,
        repe_mode=RepEMode.PLACEBO_RANDOM_VECTOR,
        policy_mode=PolicyMode.LLM_AGENT,
    )
    specs["CTRL_SHUFFLED_LAMBDA"] = FactorialCellSpec(
        cell_id="CTRL_SHUFFLED_LAMBDA",
        deliberation_bottleneck=True,
        memory_friction=True,
        endogenous_repe=True,
        repe_mode=RepEMode.SHUFFLED_LAMBDA,
        policy_mode=PolicyMode.LLM_AGENT,
    )
    specs["BASE_ZERO_INTELLIGENCE"] = FactorialCellSpec(
        cell_id="BASE_ZERO_INTELLIGENCE",
        deliberation_bottleneck=False,
        memory_friction=False,
        endogenous_repe=False,
        repe_mode=RepEMode.OFF,
        policy_mode=PolicyMode.ZERO_INTELLIGENCE,
    )
    specs["BASE_CLOSED_FORM"] = FactorialCellSpec(
        cell_id="BASE_CLOSED_FORM",
        deliberation_bottleneck=False,
        memory_friction=False,
        endogenous_repe=False,
        repe_mode=RepEMode.OFF,
        policy_mode=PolicyMode.CLOSED_FORM_BENCHMARK,
    )
    return specs


# ============================================================================
# 2. SINGLE ECONOMY SEED EXECUTOR (ZERO-REDUNDANCY WINDFALL SHADOW PROBE)
# ============================================================================

def execute_single_economy_seed(
    cell_cfg: SimulationConfig,
    economy_seed: int,
    llm_controller: Any = None,
    eval_windfall_mpc: bool = True,
    windfall_horizon: int = 2,
    output_dir: Optional[str] = None,
) -> EconomySeedClusterMetrics:
    cell_id = cell_cfg.factorial_cell.cell_id
    run_id = f"{cell_cfg.experiment_id}__{cell_id}__seed{economy_seed:03d}"

    gm = GameMaster(config=cell_cfg, economy_seed=economy_seed)
    mem_tms = EpisodicMemoryTMS(
        config=cell_cfg.memory_friction,
        db_path=":memory:",
        use_neural_embeddings=False,
    )
    orch = TurnOrchestrator(
        config=cell_cfg,
        game_master=gm,
        memory_tms=mem_tms,
        llm_controller=llm_controller,
        run_id=run_id,
        economy_seed=economy_seed,
    )

    total_ticks = cell_cfg.burn_in_ticks + cell_cfg.active_ticks
    post_burn_in_summaries: List[TickMacroSummary] = []

    windfall_mpcs: Optional[List[float]] = None
    windfall_counts: Optional[List[int]] = None
    active_shadow: Optional[ActiveWindfallShadowState] = None

    windfall_trigger_tick = cell_cfg.burn_in_ticks + max(
        1, min(4, max(1, cell_cfg.active_ticks - windfall_horizon))
    )

    for tick in range(1, total_ticks + 1):
        if eval_windfall_mpc and tick == windfall_trigger_tick:
            active_shadow = ExogenousWindfallMPCProbe.fork_treated_windfall_branch(
                orchestrator=orch,
                start_tick=tick,
                windfall_amount=20.0,
                horizon_ticks=windfall_horizon,
            )

        summary = orch.run_tick(tick)
        if tick > cell_cfg.burn_in_ticks:
            post_burn_in_summaries.append(summary)

        if active_shadow is not None:
            ExogenousWindfallMPCProbe.step_treated_and_record(
                shadow=active_shadow,
                tick=tick,
                main_control_summary=summary,
            )
            if active_shadow.steps_completed >= active_shadow.horizon_ticks:
                windfall_mpcs, windfall_counts = (
                    ExogenousWindfallMPCProbe.finalize_tercile_mpc(active_shadow)
                )
                active_shadow = None

    if active_shadow is not None:
        windfall_mpcs, windfall_counts = (
            ExogenousWindfallMPCProbe.finalize_tercile_mpc(active_shadow)
        )
        active_shadow = None

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        turn_log_path = os.path.join(
            output_dir, f"turns_{cell_id}_s{economy_seed:03d}.jsonl"
        )
        with open(turn_log_path, "w", encoding="utf-8") as f:
            for rec in orch.turn_records:
                f.write(rec.model_dump_json() + "\n")

    return EconomyMetricsCollector.summarize_single_economy_seed(
        config=cell_cfg,
        economy_seed=economy_seed,
        orchestrator=orch,
        post_burn_in_summaries=post_burn_in_summaries,
        windfall_mpc_terciles=windfall_mpcs,
        windfall_mpc_counts=windfall_counts,
    )


def _parallel_non_llm_worker(
    args_tuple: Tuple[Dict[str, Any], int, bool, int, Optional[str]]
) -> EconomySeedClusterMetrics:
    raw_cfg, seed, eval_mpc, wf_h, out_dir = args_tuple
    cfg = SimulationConfig(**raw_cfg)
    return execute_single_economy_seed(
        cell_cfg=cfg,
        economy_seed=seed,
        llm_controller=None,
        eval_windfall_mpc=eval_mpc,
        windfall_horizon=wf_h,
        output_dir=out_dir,
    )


def _load_checkpoint_safely(ckpt_path: str) -> Dict[int, EconomySeedClusterMetrics]:
    cached: Dict[int, EconomySeedClusterMetrics] = {}
    if not os.path.exists(ckpt_path):
        return cached
    valid_fields = {f.name for f in dataclasses.fields(EconomySeedClusterMetrics)}
    with open(ckpt_path, "r", encoding="utf-8") as cf:
        for row in json.load(cf):
            filtered = {k: v for k, v in row.items() if k in valid_fields}
            sc_obj = EconomySeedClusterMetrics(**filtered)
            cached[sc_obj.economy_seed] = sc_obj
    return cached


def _export_benchmark_csv(
    cell_reports: Dict[str, Dict[str, Any]], output_dir: str
) -> str:
    csv_path = os.path.join(output_dir, "benchmark_summary_table.csv")
    fieldnames = [
        "cell_id",
        "n_economy_seeds_m",
        "json_valid_rate",
        "survival_rate",
        "market_freeze_rate",
        "wealth_gini",
        "mean_bid_ask_spread",
        "mean_clearing_price",
        "mean_capital_liquidation_volume",
        "poor_htm_share",
        "wealthy_htm_share",
        "non_htm_share",
        "htm_debiased_jsd",
        "bargaining_debiased_w1",
        "bargaining_w1_ci95_lo",
        "bargaining_w1_ci95_hi",
        "bargaining_tost_equiv_bins",
        "htm_tost_equiv_bins",
        "windfall_mpc_slope",
        "mpc_tost_equiv_bins",
        "mean_endogenous_lambda",
        "max_sfc_conservation_error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cid, rep in cell_reports.items():
            htm_map = {
                b["bin_label"]: b["sim_cluster_mean"]
                for b in rep.get("htm_per_bin_tost", [])
            }
            w1 = rep["bargaining_w1"]
            writer.writerow(
                {
                    "cell_id": cid,
                    "n_economy_seeds_m": rep["n_economy_seeds_m"],
                    "json_valid_rate": rep["json_valid_rate"],
                    "survival_rate": rep["survival_rate"],
                    "market_freeze_rate": rep["market_freeze_rate"],
                    "wealth_gini": rep.get("wealth_gini", 0.0),
                    "mean_bid_ask_spread": rep.get("mean_bid_ask_spread", 0.0),
                    "mean_clearing_price": rep.get("mean_clearing_price", 12.0),
                    "mean_capital_liquidation_volume": rep.get(
                        "mean_capital_liquidation_volume", 0.0
                    ),
                    "poor_htm_share": htm_map.get("Poor_HtM", 0.0),
                    "wealthy_htm_share": htm_map.get("Wealthy_HtM", 0.0),
                    "non_htm_share": htm_map.get("Non_HtM", 0.0),
                    "htm_debiased_jsd": rep["htm_miller_madow_jsd"]["debiased_jsd"],
                    "bargaining_debiased_w1": w1["debiased_w1"],
                    "bargaining_w1_ci95_lo": w1["ci_95"][0],
                    "bargaining_w1_ci95_hi": w1["ci_95"][1],
                    "bargaining_tost_equiv_bins": sum(
                        1
                        for b in rep["bargaining_per_bin_tost"]
                        if b["statistically_equivalent"]
                    ),
                    "htm_tost_equiv_bins": sum(
                        1
                        for b in rep["htm_per_bin_tost"]
                        if b["statistically_equivalent"]
                    ),
                    "windfall_mpc_slope": rep.get("windfall_mpc_slope", ""),
                    "mpc_tost_equiv_bins": sum(
                        1
                        for b in rep.get("windfall_mpc_per_bin_tost", [])
                        if b["statistically_equivalent"]
                    ),
                    "mean_endogenous_lambda": rep["mean_endogenous_lambda"],
                    "max_sfc_conservation_error": rep["max_sfc_conservation_error"],
                }
            )
    return csv_path


# ============================================================================
# 3. CLI ENTRYPOINT & MASTER ABLATION ORCHESTRATOR
# ============================================================================

def _resolve_git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Neuropolis v2: 2^3 Factorial Ablation & Multi-Seed Clustered Evaluation Runner"
    )
    parser.add_argument("--config", type=str, default="configs/mvp_config.yaml")
    parser.add_argument(
        "--cells",
        type=str,
        default="BASE_ZERO_INTELLIGENCE,BASE_CLOSED_FORM",
    )
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--allow-dev-seeds", action="store_true")
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--active-ticks", type=int, default=None)
    parser.add_argument("--burn-in-ticks", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-windfall-mpc", action="store_true")
    parser.add_argument("--windfall-horizon", type=int, default=2)
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=420.0,
        help="Graceful Kaggle session wall-clock cap (default 420m = 7h); flushes clean reports before stopping.",
    )
    parser.add_argument("--save-turn-logs", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/neuropolis_v2")
    return parser.parse_args()


def _print_report_summary(
    cell_reports: Dict[str, Dict[str, Any]],
    factorial_analysis: Optional[Dict[str, Any]],
    paired_contrasts: Dict[str, Any],
    m_seeds: int,
) -> None:
    power_banner = (
        f"CONFIRMATORY POWER ADEQUATE (M={m_seeds} >= 30)"
        if m_seeds >= 30
        else f"EXPLORATORY / PARTIAL RUN (M={m_seeds} < 30)"
    )
    print("\n" + "=" * 126)
    print(f" NEUROPOLIS v2 — BEHAVIORAL ECONOMICS & CLUSTERED EVALUATION SUMMARY [{power_banner}]")
    print("=" * 126)
    print(
        f"{'Cell ID':<24} | {'M':>3} | {'Valid%':>6} | {'Surv%':>6} | {'Frz%':>5} | "
        f"{'Gini':>5} | {'Sprd':>5} | {'W-HtM':>5} | {'Barg W1 (95% CI)':>21} | {'MPC_Slp':>7} | {'SFC Err':>8}"
    )
    print("-" * 126)
    for cid, rep in cell_reports.items():
        w1_info = rep["bargaining_w1"]
        w1_str = f"{w1_info['debiased_w1']:.3f} [{w1_info['ci_95'][0]:.2f},{w1_info['ci_95'][1]:.2f}]"
        whtm_bin = next(
            (b for b in rep["htm_per_bin_tost"] if b["bin_label"] == "Wealthy_HtM"),
            None,
        )
        whtm_mean = whtm_bin["sim_cluster_mean"] * 100.0 if whtm_bin else 0.0
        mpc_slp = rep.get("windfall_mpc_slope")
        mpc_str = f"{mpc_slp:+.3f}" if mpc_slp is not None else "  N/A  "
        print(
            f"{cid:<24} | {rep['n_economy_seeds_m']:>3d} | "
            f"{rep['json_valid_rate']*100:>5.1f}% | {rep['survival_rate']*100:>5.1f}% | "
            f"{rep['market_freeze_rate']*100:>4.1f}% | {rep.get('wealth_gini', 0.0):>5.3f} | "
            f"{rep.get('mean_bid_ask_spread', 0.0):>5.2f} | {whtm_mean:>4.1f}% | "
            f"{w1_str:>21} | {mpc_str:>7} | {rep['max_sfc_conservation_error']:>8.1e}"
        )

    if paired_contrasts:
        print("\n" + "-" * 126)
        print(" COMMON RANDOM NUMBERS (CRN) PAIRED CLUSTER CONTRASTS (Bargaining W1 Diff; Negative = Improvement)")
        print("-" * 126)
        for label, c_res in paired_contrasts.items():
            print(
                f"  {label:<44}: diff={c_res['paired_mean_diff']:+.4f} "
                f"(SE={c_res['paired_clustered_se']:.4f}, 95% CI={c_res['ci_95']}, "
                f"p={c_res['p_value_two_sided']:.4f})"
            )

    if factorial_analysis and factorial_analysis.get("complete_2x2x2_lattice"):
        print("\n" + "-" * 126)
        print(" 2^3 FACTORIAL MAIN EFFECTS, INTERACTIONS & EXACT SHAPLEY CREDITS (on -Bargaining_W1)")
        print("-" * 126)
        print(f"  Main Effects:   {factorial_analysis['main_effects']}")
        print(f"  Interactions:   {factorial_analysis['interactions']}")
        print(f"  Exact Shapley:  {factorial_analysis['exact_shapley_values']}")
    print("=" * 126 + "\n")


def main() -> None:
    t_run_start = time.time()
    args = parse_args()

    if args.seeds < 30 and not args.allow_dev_seeds:
        raise ValueError(
            f"Confirmatory clustered inference requires --seeds >= 30 (got {args.seeds}). "
            f"Pass --allow-dev-seeds explicitly for smaller seed budgets."
        )

    with open(args.config, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    base_cfg = SimulationConfig(**raw_cfg)

    if args.num_agents is not None:
        base_cfg.num_agents = int(args.num_agents)
    if args.active_ticks is not None:
        base_cfg.active_ticks = int(args.active_ticks)
    if args.burn_in_ticks is not None:
        base_cfg.burn_in_ticks = int(args.burn_in_ticks)

    all_specs = build_full_factorial_and_control_specs()
    cells_arg = args.cells.strip().upper()
    if cells_arg == "ALL":
        selected_cell_ids = [
            "BASE_ZERO_INTELLIGENCE",
            "BASE_CLOSED_FORM",
            "C000", "C100", "C010", "C001", "C110", "C101", "C011", "C111",
            "CTRL_PLACEBO_VEC",
            "CTRL_SHUFFLED_LAMBDA",
        ]
    elif cells_arg == "FACTORIAL_8":
        selected_cell_ids = [
            "C000", "C100", "C010", "C001", "C110", "C101", "C011", "C111",
        ]
    elif cells_arg == "BASELINES":
        selected_cell_ids = ["BASE_ZERO_INTELLIGENCE", "BASE_CLOSED_FORM"]
    else:
        selected_cell_ids = [c.strip() for c in args.cells.split(",") if c.strip()]
        for cid in selected_cell_ids:
            if cid not in all_specs:
                raise ValueError(
                    f"Unknown cell_id '{cid}'. Valid options: {sorted(all_specs.keys())}"
                )

    seed_list = [base_cfg.master_seed + m for m in range(args.seeds)]
    os.makedirs(args.output_dir, exist_ok=True)
    turn_log_dir = (
        os.path.join(args.output_dir, "turn_logs") if args.save_turn_logs else None
    )

    needs_llm = any(
        all_specs[cid].policy_mode == PolicyMode.LLM_AGENT
        for cid in selected_cell_ids
    )
    shared_model = None
    shared_tokenizer = None
    shared_repe_artifacts = None

    if needs_llm:
        from src.brain.inference_brain import (
            TwoStageRepEController,
            WhiteBoxModelLoader,
        )
        from src.brain.repe_manager import RepEManager

        print(
            f"[BOOT] Mounting White-Box model '{base_cfg.model.model_id}' "
            f"(4bit={base_cfg.model.load_in_4bit}, attn={base_cfg.model.attn_implementation})...",
            flush=True,
        )
        shared_model, shared_tokenizer = WhiteBoxModelLoader.load_with_fallback(base_cfg)
        boot_repe = RepEManager(config=base_cfg.repe, seed=base_cfg.master_seed)
        shared_repe_artifacts = boot_repe.load_or_extract_artifacts(
            model=shared_model,
            tokenizer=shared_tokenizer,
            model_id=base_cfg.model.model_id,
        )
        print(
            f"[BOOT] Disjoint RepE artifacts ready (Layer={shared_repe_artifacts.target_layer}, "
            f"Test AUROC={shared_repe_artifacts.probe_test_auroc:.4f}, "
            f"Cos(w, v)={shared_repe_artifacts.probe_steering_cosine:.4f}).",
            flush=True,
        )

    max_runtime_sec = max(60.0, float(args.max_runtime_minutes) * 60.0)
    time_cap_reached = False
    clusters_by_cell: Dict[str, List[EconomySeedClusterMetrics]] = {}

    for cid in selected_cell_ids:
        cell_spec = all_specs[cid]
        cell_cfg = base_cfg.model_copy(
            deep=True, update={"factorial_cell": cell_spec}
        )
        cell_cfg = SimulationConfig(**cell_cfg.model_dump())

        print(
            f"\n[EXEC] Cell {cid:<24} | Policy={cell_cfg.policy_mode.value} | "
            f"A(Delib)={cell_cfg.deliberation.enabled} B(MemFric)={cell_cfg.memory_friction.enabled} "
            f"C(RepE)={cell_cfg.repe.mode.value} | Target Seeds M={len(seed_list)}",
            flush=True,
        )

        if cell_cfg.policy_mode != PolicyMode.LLM_AGENT and args.workers > 1:
            tasks = [
                (
                    cell_cfg.model_dump(),
                    s_val,
                    not args.skip_windfall_mpc,
                    args.windfall_horizon,
                    turn_log_dir,
                )
                for s_val in seed_list
            ]
            seed_results: List[EconomySeedClusterMetrics] = []
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [
                    pool.submit(_parallel_non_llm_worker, t_arg) for t_arg in tasks
                ]
                for fut in as_completed(futures):
                    seed_results.append(fut.result())
            seed_results.sort(key=lambda r: r.economy_seed)
            clusters_by_cell[cid] = seed_results
        else:
            from src.brain.inference_brain import TwoStageRepEController
            from src.brain.repe_manager import RepEManager

            ckpt_path = os.path.join(args.output_dir, f"checkpoint_{cid}.json")
            cached_by_seed = _load_checkpoint_safely(ckpt_path)
            if cached_by_seed:
                print(
                    f"       [RESUME] Loaded {len(cached_by_seed)} cached seeds from {ckpt_path}",
                    flush=True,
                )

            seed_results = []
            seed_durations: List[float] = []

            for s_idx, s_val in enumerate(seed_list, start=1):
                if s_val in cached_by_seed:
                    seed_results.append(cached_by_seed[s_val])
                    continue

                elapsed_total = time.time() - t_run_start
                if elapsed_total >= max_runtime_sec:
                    print(
                        f"       [TIME GUARD] Reached --max-runtime-minutes={args.max_runtime_minutes:.0f}m "
                        f"after {len(seed_results)} seeds of {cid}. Flushing clean report now!",
                        flush=True,
                    )
                    time_cap_reached = True
                    break

                t_seed = time.time()
                if cell_cfg.policy_mode == PolicyMode.LLM_AGENT:
                    repe_mgr = RepEManager(
                        config=cell_cfg.repe,
                        artifacts=shared_repe_artifacts,
                        seed=s_val,
                    )
                    controller = TwoStageRepEController(
                        config=cell_cfg,
                        model=shared_model,
                        tokenizer=shared_tokenizer,
                        repe_manager=repe_mgr,
                        economy_seed=s_val,
                    )
                else:
                    controller = None

                res = execute_single_economy_seed(
                    cell_cfg=cell_cfg,
                    economy_seed=s_val,
                    llm_controller=controller,
                    eval_windfall_mpc=not args.skip_windfall_mpc,
                    windfall_horizon=args.windfall_horizon,
                    output_dir=turn_log_dir,
                )
                dt_seed = time.time() - t_seed
                seed_durations.append(dt_seed)
                seed_results.append(res)
                cached_by_seed[s_val] = res

                with open(ckpt_path, "w", encoding="utf-8") as cf:
                    json.dump(
                        [
                            dataclasses.asdict(cached_by_seed[k])
                            for k in sorted(cached_by_seed.keys())
                        ],
                        cf,
                        indent=2,
                    )

                rem_seeds_cell = len(seed_list) - s_idx
                avg_dt = sum(seed_durations) / len(seed_durations)
                eta_min = (rem_seeds_cell * avg_dt) / 60.0
                print(
                    f"       Seed {s_idx:02d}/{len(seed_list)} (s={s_val}) done in {dt_seed:5.1f}s "
                    f"| Cell ETA={eta_min:5.1f}m | Valid={res.json_valid_rate:.0%} "
                    f"| Gini={res.wealth_gini:.3f} | Spread=${res.mean_bid_ask_spread:.2f} "
                    f"| Freeze={res.consumable_market_freeze_rate:.0%} | SFC={res.max_sfc_conservation_error:.1e}",
                    flush=True,
                )

            if seed_results:
                clusters_by_cell[cid] = seed_results
            if time_cap_reached:
                break

    if not clusters_by_cell:
        print("[WARN] No seeds completed before exit.")
        return

    zi_barg_floor: Optional[float] = None
    zi_mpc_floor: Optional[float] = None
    if "BASE_ZERO_INTELLIGENCE" in clusters_by_cell:
        zi_rep = EconomyMetricsCollector.aggregate_cell_clusters(
            cell_id="BASE_ZERO_INTELLIGENCE",
            seed_clusters=clusters_by_cell["BASE_ZERO_INTELLIGENCE"],
            config=base_cfg,
        )
        zi_barg_floor = float(zi_rep["bargaining_w1"]["debiased_w1"])
        if zi_rep["windfall_mpc_w1"] is not None:
            zi_mpc_floor = float(zi_rep["windfall_mpc_w1"]["debiased_w1"])

    cell_reports: Dict[str, Dict[str, Any]] = {}
    for cid, sc_list in clusters_by_cell.items():
        cell_reports[cid] = EconomyMetricsCollector.aggregate_cell_clusters(
            cell_id=cid,
            seed_clusters=sc_list,
            config=base_cfg,
            zi_bargaining_floor_w1=zi_barg_floor,
            zi_mpc_floor_w1=zi_mpc_floor,
        )

    paired_contrasts: Dict[str, Any] = {}
    contrast_pairs = [
        ("C111_vs_C000_VanillaLLM", "C111", "C000"),
        ("C111_vs_CTRL_PLACEBO_VEC", "C111", "CTRL_PLACEBO_VEC"),
        ("C111_vs_CTRL_SHUFFLED_LAMBDA", "C111", "CTRL_SHUFFLED_LAMBDA"),
        ("C001_RepEOnly_vs_C000_Vanilla", "C001", "C000"),
        ("C111_vs_BASE_ZERO_INTELLIGENCE", "C111", "BASE_ZERO_INTELLIGENCE"),
        (
            "BASE_CLOSED_FORM_vs_BASE_ZERO_INTELLIGENCE",
            "BASE_CLOSED_FORM",
            "BASE_ZERO_INTELLIGENCE",
        ),
    ]
    for label, cell_a, cell_b in contrast_pairs:
        if cell_a in cell_reports and cell_b in cell_reports:
            paired_contrasts[label] = clustered_paired_contrast(
                cell_a_by_seed=cell_reports[cell_a]["seed_level_bargaining_w1"],
                cell_b_by_seed=cell_reports[cell_b]["seed_level_bargaining_w1"],
                n_resamples_b=base_cfg.stats.bootstrap_resamples_b,
                seed=base_cfg.master_seed,
            )

    bits_map: Dict[Tuple[int, int, int], float] = {}
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                cid = f"C{a}{b}{c}"
                if cid in cell_reports:
                    bits_map[(a, b, c)] = -float(
                        cell_reports[cid]["bargaining_w1"]["debiased_w1"]
                    )
    factorial_analysis = (
        compute_2x2x2_factorial_effects_and_shapley(bits_map)
        if len(bits_map) == 8
        else None
    )

    min_completed_m = min(
        len(sc_list) for sc_list in clusters_by_cell.values()
    )
    cfg_sha = hashlib.sha256(base_cfg.model_dump_json().encode("utf-8")).hexdigest()[:16]
    manifest = {
        "experiment_id": base_cfg.experiment_id,
        "git_commit_sha": _resolve_git_sha(),
        "config_sha256": cfg_sha,
        "selected_cells": list(clusters_by_cell.keys()),
        "n_economy_seeds_target": args.seeds,
        "n_economy_seeds_completed_min": min_completed_m,
        "time_cap_triggered": time_cap_reached,
        "power_status": (
            "CONFIRMATORY_ADEQUATE" if min_completed_m >= 30 else "EXPLORATORY_OR_PARTIAL"
        ),
        "num_agents_n": base_cfg.num_agents,
        "burn_in_ticks": base_cfg.burn_in_ticks,
        "active_ticks": base_cfg.active_ticks,
        "model_id": base_cfg.model.model_id,
        "wall_clock_seconds": round(time.time() - t_run_start, 2),
        "probe_artifacts_metadata": (
            {
                "target_layer": shared_repe_artifacts.target_layer,
                "probe_test_auroc": shared_repe_artifacts.probe_test_auroc,
                "probe_permutation_p": shared_repe_artifacts.probe_permutation_p,
                "probe_steering_cosine": shared_repe_artifacts.probe_steering_cosine,
            }
            if shared_repe_artifacts is not None
            else None
        ),
    }

    with open(
        os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2)

    with open(
        os.path.join(args.output_dir, "cluster_metrics.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                cid: [dataclasses.asdict(sc) for sc in sc_list]
                for cid, sc_list in clusters_by_cell.items()
            },
            f,
            indent=2,
        )

    full_report = {
        "manifest": manifest,
        "cell_reports": cell_reports,
        "paired_contrasts": paired_contrasts,
        "factorial_2x2x2_analysis": factorial_analysis,
    }
    report_path = os.path.join(args.output_dir, "factorial_evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    csv_path = _export_benchmark_csv(cell_reports, args.output_dir)

    _print_report_summary(
        cell_reports=cell_reports,
        factorial_analysis=factorial_analysis,
        paired_contrasts=paired_contrasts,
        m_seeds=min_completed_m,
    )
    print(f"[SAVED] JSON Report: {report_path}")
    print(f"[SAVED] CSV Table:   {csv_path}")


if __name__ == "__main__":
    main()