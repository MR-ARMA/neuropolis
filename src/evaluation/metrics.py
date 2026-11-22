from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from src.core.contracts import (
    AgentActionType,
    AgentTurnRecord,
    AssetType,
    OrderSide,
    PerBinTOSTResult,
    PolicyMode,
    SimulationConfig,
)
from src.environment.game_master import GameMaster, TickMacroSummary
from src.environment.needs_engine import NeedsEngine, TwoAssetAccounting
from src.evaluation.validity_and_stats import (
    clustered_bootstrap_mean,
    compute_debiased_wasserstein_curve,
    compute_miller_madow_jsd,
    run_per_bin_tost_family,
)
from src.orchestration.turn_orchestrator import TurnOrchestrator


# ============================================================================
# PUBLISHED HUMAN EMPIRICAL REFERENCE CURVES (PREREGISTERED BENCHMARKS)
# ============================================================================

# 1. Oosterbeek, Sloof, & van de Kuilen (2004, Experimental Economics):
#    Acceptance / order-clearing gradient across 4 relative markup / offer-share bins:
#      Bin 0: Generous / Fair (markup <= 0.05) -> 0.94
#      Bin 1: Moderate markup (0.05 < m <= 0.15) -> 0.76
#      Bin 2: Unfair markup (0.15 < m <= 0.30) -> 0.42
#      Bin 3: Extreme markup (m > 0.30) -> 0.14
HUMAN_BARGAINING_BIN_LABELS = [
    "Bin0_Fair_le_5pct",
    "Bin1_Mod_5_to_15pct",
    "Bin2_Unfair_15_to_30pct",
    "Bin3_Extreme_gt_30pct",
]
HUMAN_BARGAINING_ACCEPT_CURVE = [0.94, 0.76, 0.42, 0.14]
HUMAN_BARGAINING_SE = [0.03, 0.04, 0.05, 0.04]

# 2. Kaplan, Violante, & Weidner (2014, BPEA):
#    US SCF Two-Asset Hand-to-Mouth household distribution:
#      [Poor_HtM, Wealthy_HtM, Non_HtM] = [0.12, 0.20, 0.68]
HUMAN_HTM_LABELS = ["Poor_HtM", "Wealthy_HtM", "Non_HtM"]
HUMAN_HTM_DISTRIBUTION = [0.12, 0.20, 0.68]

# 3. Jappelli & Pistaferri (2014, AEJ: Macroeconomics):
#    Exogenous-windfall MPC across 3 liquid-wealth terciles:
#      [Low_Liquidity_Tercile, Mid_Liquidity_Tercile, High_Liquidity_Tercile]
HUMAN_MPC_TERCILE_LABELS = ["Low_Liq_Tercile", "Mid_Liq_Tercile", "High_Liq_Tercile"]
HUMAN_MPC_CURVE = [0.63, 0.46, 0.31]
HUMAN_MPC_SE = [0.04, 0.03, 0.03]


@dataclass
class EconomySeedClusterMetrics:
    """
    Cluster-level summary for a SINGLE independent economy seed m.
    Every downstream statistical test operates strictly on lists of
    `EconomySeedClusterMetrics` across M >= 30 independent seeds.
    """
    cell_id: str
    economy_seed: int
    policy_mode: str
    # Integrity & Cognitive Control Diagnostics
    json_valid_rate: float
    abstain_rate: float
    cot_truncation_rate: float
    mean_endogenous_lambda: float
    std_endogenous_lambda: float
    mean_relative_steering_norm: float
    survival_rate: float
    consumable_market_freeze_rate: float
    max_sfc_conservation_error: float
    # Two-Asset HtM Shares (Kaplan, Violante, & Weidner 2014)
    poor_htm_share: float
    wealthy_htm_share: float
    non_htm_share: float
    htm_category_counts: List[float]
    # CDA Bargaining / Offer Acceptance Gradient across 4 Markup Bins
    bargaining_bin_accept_rates: List[float]
    bargaining_bin_obs_counts: List[int]
    # Exogenous-Windfall Causal MPC across 3 Liquid-Wealth Terciles (Zero Circularity)
    windfall_mpc_terciles: List[float]
    windfall_mpc_obs_counts: List[int]
    windfall_mpc_low_minus_high_slope: float
    windfall_probe_executed: bool


class ExogenousWindfallMPCProbe:
    """
    Measures strictly non-circular Marginal Propensity to Consume (MPC) via a
    paired exogenous cash windfall experiment (Parker et al. 2013; Jappelli &
    Pistaferri 2014):
      - Clones the post-burn-in economy into two identical shadow branches
        (Control vs. Windfall) with matched RNG streams.
      - In the Windfall branch, transfers an exogenous lump-sum Delta_Y from
        `PublicTreasury` to living households (preserving exact SFC_Error == 0.0).
      - Runs h = 3 ticks on both branches and computes realized peer-to-peer CDA
        consumable spend difference:
            MPC_i = (Spend_i^{Windfall} - Spend_i^{Control}) / Delta_Y
      - Aggregates causal MPC by pre-windfall liquid-wealth tercile.
    """

    @staticmethod
    def run_paired_windfall_probe(
        orchestrator: TurnOrchestrator,
        start_tick: int,
        windfall_amount: float = 20.0,
        horizon_ticks: int = 3,
    ) -> Tuple[List[float], List[int]]:
        gm_base = orchestrator.game_master
        living_agents = [
            (aid, sheet.liquid_cash)
            for aid, sheet in gm_base.households.items()
            if NeedsEngine.is_alive(sheet)
        ]
        if len(living_agents) < 3:
            return ([0.0, 0.0, 0.0], [0, 0, 0])

        # Clone GameMaster & Memory for paired Control vs. Treated Windfall worlds
        # (Use lightweight policy clone so LLM weights are never deep-copied!)
        saved_llm = orchestrator.llm_controller
        orchestrator.llm_controller = None
        try:
            ctrl_orch = copy.deepcopy(orchestrator)
            treat_orch = copy.deepcopy(orchestrator)
        finally:
            orchestrator.llm_controller = saved_llm

        ctrl_orch.llm_controller = saved_llm
        treat_orch.llm_controller = saved_llm

        # Inject exogenous lump-sum windfall from PublicTreasury in Treated world (strict SFC)
        n_living = len(living_agents)
        total_payout = round(windfall_amount * n_living, 8)
        if treat_orch.game_master.treasury_cash < total_payout:
            # Ensure PublicTreasury holds sufficient balance via Firm -> Treasury transfer in BOTH branches
            # or scale windfall to available treasury liquidity so SFC remains strictly 0.0
            eff_windfall = round(
                max(2.0, min(windfall_amount, treat_orch.game_master.treasury_cash / (n_living + 1))),
                2,
            )
        else:
            eff_windfall = round(float(windfall_amount), 2)

        for aid, _ in living_agents:
            treat_orch.game_master.treasury_cash = round(
                treat_orch.game_master.treasury_cash - eff_windfall, 8
            )
            treat_orch.game_master.households[aid].liquid_cash = round(
                treat_orch.game_master.households[aid].liquid_cash + eff_windfall, 8
            )
        treat_orch.game_master.audit_and_assert_sfc(start_tick)

        ctrl_spend: Dict[str, float] = {aid: 0.0 for aid, _ in living_agents}
        treat_spend: Dict[str, float] = {aid: 0.0 for aid, _ in living_agents}

        for step_offset in range(horizon_ticks):
            t_cur = start_tick + step_offset
            s_ctrl = ctrl_orch.run_tick(t_cur)
            s_treat = treat_orch.run_tick(t_cur)
            for aid, _ in living_agents:
                ctrl_spend[aid] += float(
                    s_ctrl.realized_consumable_spend_by_agent.get(aid, 0.0)
                )
                treat_spend[aid] += float(
                    s_treat.realized_consumable_spend_by_agent.get(aid, 0.0)
                )

        # Sort agents by pre-windfall liquid cash into 3 terciles
        living_agents.sort(key=lambda pair: pair[1])
        tercile_splits = np.array_split(np.arange(len(living_agents)), 3)

        tercile_mpcs: List[float] = []
        tercile_counts: List[int] = []
        for idx_arr in tercile_splits:
            vals: List[float] = []
            for pos in idx_arr.tolist():
                aid = living_agents[pos][0]
                delta_c = treat_spend[aid] - ctrl_spend[aid]
                mpc_i = max(-0.5, min(1.5, delta_c / max(eff_windfall, 1e-6)))
                vals.append(mpc_i)
            tercile_mpcs.append(round(float(np.mean(vals)), 4) if vals else 0.0)
            tercile_counts.append(len(vals))

        return (tercile_mpcs, tercile_counts)


class EconomyMetricsCollector:
    """
    Collects strictly objective per-economy-seed metrics from CDA order fills,
    Two-Asset balance sheets, and Stage 1/2 turn audit records.
    """

    @staticmethod
    def summarize_single_economy_seed(
        config: SimulationConfig,
        economy_seed: int,
        orchestrator: TurnOrchestrator,
        post_burn_in_summaries: List[TickMacroSummary],
        windfall_mpc_terciles: Optional[List[float]] = None,
        windfall_mpc_counts: Optional[List[int]] = None,
    ) -> EconomySeedClusterMetrics:
        gm = orchestrator.game_master
        burn_in = config.burn_in_ticks
        active_records = [
            r for r in orchestrator.turn_records if r.tick > burn_in
        ]

        # 1. Cognitive & Endogenous RepE Diagnostics
        n_rec = len(active_records)
        json_valid_rate = (
            round(sum(1 for r in active_records if r.json_valid) / n_rec, 4)
            if n_rec > 0
            else 1.0
        )
        abstain_rate = (
            round(sum(1 for r in active_records if r.abstain) / n_rec, 4)
            if n_rec > 0
            else 0.0
        )
        cot_trunc_rate = (
            round(
                sum(1 for r in active_records if r.cot_truncated_by_bottleneck) / n_rec,
                4,
            )
            if n_rec > 0
            else 0.0
        )

        lambdas = [
            r.probe_measurement.endogenous_lambda
            for r in active_records
            if r.probe_measurement is not None
        ]
        rel_norms = [
            r.probe_measurement.relative_residual_norm_ratio
            for r in active_records
            if r.probe_measurement is not None
        ]
        mean_lam = round(float(np.mean(lambdas)), 4) if lambdas else 0.0
        std_lam = round(float(np.std(lambdas)), 4) if lambdas else 0.0
        mean_rel_norm = round(float(np.mean(rel_norms)), 4) if rel_norms else 0.0

        # 2. Macro Survival, Market Freeze & SFC Conservation Error
        n_agents = max(1, len(gm.households))
        living_now = sum(
            1 for h in gm.households.values() if NeedsEngine.is_alive(h)
        )
        survival_rate = round(living_now / n_agents, 4)
        freeze_rate = (
            round(
                sum(1 for s in post_burn_in_summaries if s.consumable_market_frozen)
                / len(post_burn_in_summaries),
                4,
            )
            if post_burn_in_summaries
            else 0.0
        )
        max_sfc_err = (
            max(s.sfc_audit.conservation_error_abs for s in post_burn_in_summaries)
            if post_burn_in_summaries
            else 0.0
        )

        # 3. Wealthy / Poor HtM Shares (Post-Burn-In Average)
        if post_burn_in_summaries:
            p_htm = round(
                float(np.mean([s.poor_htm_share for s in post_burn_in_summaries])), 4
            )
            w_htm = round(
                float(np.mean([s.wealthy_htm_share for s in post_burn_in_summaries])),
                4,
            )
            n_htm = round(
                float(np.mean([s.non_htm_share for s in post_burn_in_summaries])), 4
            )
        else:
            p_htm, w_htm, n_htm = 0.0, 0.0, 1.0
        htm_counts = [
            round(p_htm * n_agents, 2),
            round(w_htm * n_agents, 2),
            round(n_htm * n_agents, 2),
        ]

        # 4. CDA Bargaining / Order-Acceptance Gradient across 4 Relative Markup Bins
        # Evaluate every consumable ASK limit order submitted post-burn-in relative to
        # the prevailing tick mid/last price: did counterparties accept (fill) or reject (expire)?
        tick_ref_price: Dict[int, float] = {}
        for s in post_burn_in_summaries:
            ref_p = (
                s.consumable_book.mid_price
                or s.consumable_book.last_trade_price
                or 12.0
            )
            tick_ref_price[s.tick] = ref_p

        c_book = gm.exchange.get_book(AssetType.CONSUMABLE)
        bin_filled_flags: List[List[float]] = [[], [], [], []]
        for oid, ord_obj in c_book.orders_by_id.items():
            if ord_obj.tick_submitted <= burn_in:
                continue
            if ord_obj.side != OrderSide.ASK or ord_obj.limit_price is None:
                continue
            ref_p = tick_ref_price.get(ord_obj.tick_submitted, 12.0)
            rel_markup = (ord_obj.limit_price - ref_p) / max(ref_p, 1e-6)
            filled_frac = min(
                1.0, ord_obj.filled_quantity / max(ord_obj.quantity, 1e-6)
            )
            if rel_markup <= 0.05:
                bin_filled_flags[0].append(filled_frac)
            elif rel_markup <= 0.15:
                bin_filled_flags[1].append(filled_frac)
            elif rel_markup <= 0.30:
                bin_filled_flags[2].append(filled_frac)
            else:
                bin_filled_flags[3].append(filled_frac)

        barg_rates = [
            round(float(np.mean(b)), 4) if b else float("nan")
            for b in bin_filled_flags
        ]
        barg_counts = [len(b) for b in bin_filled_flags]

        # 5. Exogenous-Windfall Causal MPC
        if windfall_mpc_terciles is not None and windfall_mpc_counts is not None:
            w_mpcs = list(windfall_mpc_terciles)
            w_cnts = list(windfall_mpc_counts)
            w_ran = True
        else:
            w_mpcs = [float("nan"), float("nan"), float("nan")]
            w_cnts = [0, 0, 0]
            w_ran = False

        mpc_slope = (
            round(w_mpcs[0] - w_mpcs[2], 4)
            if (w_ran and math.isfinite(w_mpcs[0]) and math.isfinite(w_mpcs[2]))
            else float("nan")
        )

        return EconomySeedClusterMetrics(
            cell_id=config.factorial_cell.cell_id,
            economy_seed=economy_seed,
            policy_mode=config.policy_mode.value,
            json_valid_rate=json_valid_rate,
            abstain_rate=abstain_rate,
            cot_truncation_rate=cot_trunc_rate,
            mean_endogenous_lambda=mean_lam,
            std_endogenous_lambda=std_lam,
            mean_relative_steering_norm=mean_rel_norm,
            survival_rate=survival_rate,
            consumable_market_freeze_rate=freeze_rate,
            max_sfc_conservation_error=max_sfc_err,
            poor_htm_share=p_htm,
            wealthy_htm_share=w_htm,
            non_htm_share=n_htm,
            htm_category_counts=htm_counts,
            bargaining_bin_accept_rates=barg_rates,
            bargaining_bin_obs_counts=barg_counts,
            windfall_mpc_terciles=w_mpcs,
            windfall_mpc_obs_counts=w_cnts,
            windfall_mpc_low_minus_high_slope=mpc_slope,
            windfall_probe_executed=w_ran,
        )

    @staticmethod
    def aggregate_cell_clusters(
        cell_id: str,
        seed_clusters: Sequence[EconomySeedClusterMetrics],
        config: SimulationConfig,
        zi_bargaining_floor_w1: Optional[float] = None,
        zi_mpc_floor_w1: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Aggregates M independent economy seeds into clustered bootstrap summaries,
        Per-Bin TOST (Holm-Bonferroni adjusted), Debiased Wasserstein-1, and
        Miller-Madow Jensen-Shannon Divergence.
        """
        stats_cfg = config.stats
        m = len(seed_clusters)

        # 1. Per-Bin TOST for Bargaining Acceptance Gradient (4 Bins)
        barg_cluster_bins: List[List[float]] = [[] for _ in range(4)]
        barg_total_counts = [0, 0, 0, 0]
        complete_barg_curves: List[List[float]] = []

        for sc in seed_clusters:
            row_clean = []
            for b_i in range(4):
                barg_total_counts[b_i] += sc.bargaining_bin_obs_counts[b_i]
                v = sc.bargaining_bin_accept_rates[b_i]
                if math.isfinite(v):
                    barg_cluster_bins[b_i].append(v)
                    row_clean.append(v)
                else:
                    row_clean.append(0.50)
            complete_barg_curves.append(row_clean)

        barg_tost: List[PerBinTOSTResult] = run_per_bin_tost_family(
            benchmark_name="Oosterbeek_2004_CDA_Acceptance_Gradient",
            bin_labels=HUMAN_BARGAINING_BIN_LABELS,
            cluster_bin_values=barg_cluster_bins,
            bin_observation_counts=barg_total_counts,
            human_reference_means=HUMAN_BARGAINING_ACCEPT_CURVE,
            sesoi_margin_delta=stats_cfg.sesoi_ultimatum_bin_diff,
            stats_cfg=stats_cfg,
            seed=config.master_seed,
        )
        human_barg_ceiling_w1 = round(float(np.mean(HUMAN_BARGAINING_SE)), 6)
        barg_w1 = compute_debiased_wasserstein_curve(
            cluster_curves=complete_barg_curves,
            human_curve=HUMAN_BARGAINING_ACCEPT_CURVE,
            zi_null_floor_w1=zi_bargaining_floor_w1,
            human_se_ceiling_w1=human_barg_ceiling_w1,
            seed=config.master_seed,
        )

        # 2. Per-Bin TOST & Miller-Madow JSD for Two-Asset HtM Distribution
        htm_cluster_bins = [
            [sc.poor_htm_share for sc in seed_clusters],
            [sc.wealthy_htm_share for sc in seed_clusters],
            [sc.non_htm_share for sc in seed_clusters],
        ]
        total_htm_counts = [
            float(sum(sc.htm_category_counts[k] for sc in seed_clusters))
            for k in range(3)
        ]
        htm_obs_counts = [max(1, int(round(c))) for c in total_htm_counts]
        htm_tost = run_per_bin_tost_family(
            benchmark_name="KVW_2014_TwoAsset_HtM_Distribution",
            bin_labels=HUMAN_HTM_LABELS,
            cluster_bin_values=htm_cluster_bins,
            bin_observation_counts=htm_obs_counts,
            human_reference_means=HUMAN_HTM_DISTRIBUTION,
            sesoi_margin_delta=stats_cfg.sesoi_whtm_share,
            stats_cfg=stats_cfg,
            seed=config.master_seed + 100,
        )
        htm_jsd = compute_miller_madow_jsd(
            sim_category_counts=total_htm_counts,
            human_category_probs=HUMAN_HTM_DISTRIBUTION,
        )

        # 3. Per-Bin TOST & Debiased W_1 for Exogenous-Windfall Causal MPC
        mpc_cluster_bins: List[List[float]] = [[] for _ in range(3)]
        mpc_total_counts = [0, 0, 0]
        valid_mpc_curves: List[List[float]] = []

        for sc in seed_clusters:
            if not sc.windfall_probe_executed:
                continue
            valid_mpc_curves.append(sc.windfall_mpc_terciles)
            for t_i in range(3):
                mpc_total_counts[t_i] += sc.windfall_mpc_obs_counts[t_i]
                if math.isfinite(sc.windfall_mpc_terciles[t_i]):
                    mpc_cluster_bins[t_i].append(sc.windfall_mpc_terciles[t_i])

        mpc_tost = (
            run_per_bin_tost_family(
                benchmark_name="Jappelli_Pistaferri_2014_Windfall_MPC",
                bin_labels=HUMAN_MPC_TERCILE_LABELS,
                cluster_bin_values=mpc_cluster_bins,
                bin_observation_counts=mpc_total_counts,
                human_reference_means=HUMAN_MPC_CURVE,
                sesoi_margin_delta=stats_cfg.sesoi_mpc_bin_diff,
                stats_cfg=stats_cfg,
                seed=config.master_seed + 200,
            )
            if valid_mpc_curves
            else []
        )
        human_mpc_ceiling_w1 = round(float(np.mean(HUMAN_MPC_SE)), 6)
        mpc_w1 = (
            compute_debiased_wasserstein_curve(
                cluster_curves=valid_mpc_curves,
                human_curve=HUMAN_MPC_CURVE,
                zi_null_floor_w1=zi_mpc_floor_w1,
                human_se_ceiling_w1=human_mpc_ceiling_w1,
                seed=config.master_seed + 200,
            )
            if valid_mpc_curves
            else None
        )

        # 4. Clustered Bootstrap Diagnostics
        valid_boot = clustered_bootstrap_mean(
            [sc.json_valid_rate for sc in seed_clusters],
            min_required_m=stats_cfg.num_economy_seeds_m,
            seed=config.master_seed + 300,
        )
        freeze_boot = clustered_bootstrap_mean(
            [sc.consumable_market_freeze_rate for sc in seed_clusters],
            min_required_m=stats_cfg.num_economy_seeds_m,
            seed=config.master_seed + 301,
        )
        surv_boot = clustered_bootstrap_mean(
            [sc.survival_rate for sc in seed_clusters],
            min_required_m=stats_cfg.num_economy_seeds_m,
            seed=config.master_seed + 302,
        )
        lam_boot = clustered_bootstrap_mean(
            [sc.mean_endogenous_lambda for sc in seed_clusters],
            min_required_m=stats_cfg.num_economy_seeds_m,
            seed=config.master_seed + 303,
        )

        return {
            "cell_id": cell_id,
            "n_economy_seeds_m": m,
            "power_adequate_m_ge_30": m >= stats_cfg.num_economy_seeds_m,
            "max_sfc_conservation_error": max(
                (sc.max_sfc_conservation_error for sc in seed_clusters), default=0.0
            ),
            "json_valid_rate": valid_boot.cluster_mean,
            "json_valid_ci_95": valid_boot.ci_95,
            "market_freeze_rate": freeze_boot.cluster_mean,
            "market_freeze_ci_95": freeze_boot.ci_95,
            "survival_rate": surv_boot.cluster_mean,
            "survival_ci_95": surv_boot.ci_95,
            "mean_endogenous_lambda": lam_boot.cluster_mean,
            "bargaining_w1": barg_w1,
            "bargaining_per_bin_tost": [r.model_dump() for r in barg_tost],
            "htm_miller_madow_jsd": htm_jsd,
            "htm_per_bin_tost": [r.model_dump() for r in htm_tost],
            "windfall_mpc_w1": mpc_w1,
            "windfall_mpc_per_bin_tost": [r.model_dump() for r in mpc_tost],
            "seed_level_bargaining_w1": {
                sc.economy_seed: round(
                    float(
                        np.mean(
                            np.abs(
                                np.cumsum(
                                    np.asarray(
                                        [
                                            v if math.isfinite(v) else 0.50
                                            for v in sc.bargaining_bin_accept_rates
                                        ]
                                    )
                                    - np.asarray(HUMAN_BARGAINING_ACCEPT_CURVE)
                                )
                                / 4.0
                            )
                        )
                    ),
                    6,
                )
                for sc in seed_clusters
            },
        }