from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from src.core.contracts import (
    FourDimTickTelemetry,
    MacroRegimeState,
    SimulationConfig,
)


class RealTimeLiveMonitor:
    """
    Provides real-time ANSI color-coded intra-tick streaming, per-tick 4-Dimensional
    Telemetry Dashboards, non-blocking JSONL event streaming to `data/live_stream.jsonl`,
    and the End-of-Run Empirical Benchmark Scorecard.
    """

    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"

    def __init__(
        self,
        config: SimulationConfig,
        stream_path: str = "data/live_stream.jsonl",
        quiet: bool = False,
    ) -> None:
        self.config = config
        self.stream_path = stream_path
        self.quiet = quiet
        os.makedirs(os.path.dirname(self.stream_path) or ".", exist_ok=True)
        # Truncate stream file at start of run
        with open(self.stream_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "event_type": "RUN_START",
                        "timestamp": time.time(),
                        "experiment_id": config.experiment_id,
                        "omega_vector": config.orthogonal_clamping.as_tuple(),
                        "active_crisis": config.hank_and_crisis_spec.active_crisis,
                    }
                )
                + "\n"
            )

    def _emit_jsonl(self, payload: Dict[str, Any]) -> None:
        payload["timestamp"] = round(time.time(), 4)
        with open(self.stream_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def log_tick_header(self, tick: int, macro: MacroRegimeState) -> None:
        self._emit_jsonl(
            {
                "event_type": "TICK_START",
                "tick": tick,
                "active_crisis": macro.active_crisis,
                "crisis_intensity_s": macro.crisis_intensity_s,
                "tfp_ac": macro.consumable_tfp_ac,
                "ref_price_pc": macro.reference_consumable_price_pc,
                "ref_price_qk": macro.reference_capital_price_qk,
                "capital_yield_rk": macro.capital_yield_rk,
            }
        )
        if self.quiet:
            return
        crisis_badge = (
            f"{self.RED}[CRISIS: {macro.active_crisis} | s_t={macro.crisis_intensity_s:.2f}]{self.RESET}"
            if macro.crisis_intensity_s > 0.0
            else f"{self.GREEN}[REGIME: NORMAL EQUILIBRIUM]{self.RESET}"
        )
        print(
            f"\n{self.BOLD}{self.CYAN}========================================================================================================{self.RESET}"
        )
        print(
            f"{self.BOLD} TICK {tick:02d}{self.RESET} {crisis_badge} | "
            f"TFP A_c={macro.consumable_tfp_ac:.2f} | "
            f"Ref Consumable p_c=${macro.reference_consumable_price_pc:.2f} | "
            f"Ref Capital q_K=${macro.reference_capital_price_qk:.2f} | "
            f"Yield r_K={macro.capital_yield_rk:.3f}"
        )
        print(
            f"{self.CYAN}--------------------------------------------------------------------------------------------------------{self.RESET}"
        )

    def log_dyadic_interaction(self, pair_idx: int, outcome: Dict[str, Any]) -> None:
        intent = outcome["intent"]
        verdict = outcome["verdict"]
        tms_ev = outcome.get("tms_event")

        self._emit_jsonl(
            {
                "event_type": "DYADIC_INTERACTION",
                "tick": outcome["tick"],
                "pair_index": pair_idx,
                "initiator_id": outcome["initiator_id"],
                "responder_id": outcome["responder_id"],
                "initiator_scarcity_f": outcome["initiator_scarcity_f"],
                "effective_trust_ab": outcome["effective_trust_ab"],
                "action_type": intent.action_type,
                "item_id": intent.item_id,
                "quantity": intent.quantity,
                "unit_price": intent.unit_price,
                "is_fire_sale": intent.is_distress_fire_sale,
                "kkt_gouging_index": outcome["kkt_gouging_index"],
                "responder_consent": outcome["responder_consent"],
                "gm_status": verdict.status,
                "gm_error_code": verdict.error_code,
                "tms_event": tms_ev,
            }
        )
        if self.quiet:
            return

        if tms_ev:
            print(
                f"  {self.MAGENTA}[TMS ASK_EXPERT]{self.RESET} {tms_ev['requester_id']} "
                f"(H_bar={tms_ev['entropy']:.2f} > tau) routed {tms_ev['domain_id']} query "
                f"-> {tms_ev['expert_id']} (Competence={tms_ev['updated_competence']:.2f})"
            )

        fire_tag = (
            f" {self.RED}[DISTRESS FIRE-SALE!]{self.RESET}"
            if intent.is_distress_fire_sale
            else ""
        )
        kkt_tag = (
            f" {self.YELLOW}[KKT_GOUGE={outcome['kkt_gouging_index']:.2f}]{self.RESET}"
            if outcome["kkt_gouging_index"] > 0.12
            else ""
        )
        status_color = self.GREEN if verdict.status == "APPROVED" else self.RED
        print(
            f"  {self.BLUE}[PAIR {pair_idx}]{self.RESET} "
            f"{outcome['initiator_id']} (F={outcome['initiator_scarcity_f']:.2f}, T_eff={outcome['effective_trust_ab']:.2f}) "
            f"-> {intent.action_type} {intent.quantity:.0f} {intent.item_id} @ ${intent.unit_price:.2f}"
            f"{fire_tag}{kkt_tag} | "
            f"Resp {outcome['responder_id']}: {outcome['responder_consent']} -> "
            f"{status_color}{verdict.status}{self.RESET}"
        )

    def log_upkeep_and_sanctions(self, tick: int, upkeep_report: Dict[str, Any]) -> None:
        self._emit_jsonl(
            {
                "event_type": "UPKEEP_AND_FISCAL",
                "tick": tick,
                **upkeep_report,
            }
        )
        if self.quiet:
            return
        sanctions = upkeep_report.get("sanctioned_agents", {})
        if sanctions:
            for aid, sinfo in sanctions.items():
                print(
                    f"  {self.YELLOW}[OSTROM GRADUATED SANCTION]{self.RESET} "
                    f"{aid} fined ${sinfo['penalty']:.2f} ({sinfo['points']} pts) -> Credited to PublicTreasury"
                )

    def log_four_dim_tick_dashboard(self, t: FourDimTickTelemetry) -> None:
        self._emit_jsonl(
            {
                "event_type": "FOUR_DIM_TELEMETRY",
                **t.model_dump(),
            }
        )
        if self.quiet:
            return
        sfc_str = (
            f"{self.GREEN}{t.godley_lavoie_error:.2e}{self.RESET}"
            if t.godley_lavoie_error < 1e-6
            else f"{self.RED}{t.godley_lavoie_error:.2e}{self.RESET}"
        )
        print(
            f"{self.CYAN}--------------------------------------------------------------------------------------------------------{self.RESET}"
        )
        print(
            f"  {self.BOLD}[DIM 1: SCARCITY & HANK]{self.RESET}     "
            f"Wealth_Gini={t.wealth_gini:.3f} | Income_Gini={t.disposable_income_gini:.3f} | "
            f"HtM={t.htm_total_share:.1%} (Poor={t.poor_htm_share:.1%}, W-HtM={t.wealthy_htm_share:.1%}) | "
            f"MPC={t.realized_mpc:.2f} | FireSale_DSR={t.distress_sale_ratio_dsr:.1%} | "
            f"Bandwidth_d={t.scarcity_bandwidth_cohen_d:.2f}"
        )
        print(
            f"  {self.BOLD}[DIM 2: 3-TIER TRUST & MKT]{self.RESET}  "
            f"T_soc={t.generalized_trust_tsoc:.3f} | Rel_tau={t.mean_relational_trust_tau:.3f} | "
            f"T_inst={t.institutional_trust_tinst:.3f} | Modularity_Q={t.trade_network_modularity_q:+.3f} | "
            f"PriceDisp_R={t.price_dispersion_ratio_r:.2f} | RejRate={t.unfairness_rejection_rate:.1%} | "
            f"Vol={t.cumulative_trade_volume:.1f}"
        )
        print(
            f"  {self.BOLD}[DIM 3: CONTAGION & TMS]{self.RESET}     "
            f"AffectSync_ASI={t.affective_sync_index_asi:.3f} | Opinion_BC={t.opinion_bimodality_bc:.3f} | "
            f"Cascade={t.complex_contagion_cascade_share:.1%} | "
            f"TMS_Queries={t.cumulative_tms_queries} (Acc={t.tms_routing_accuracy:.0%})"
        )
        print(
            f"  {self.BOLD}[DIM 4: FISCAL & OSTROM]{self.RESET}     "
            f"PG_Provision={t.public_goods_provision_rate_pgpr:.1%} | FreeRider={t.free_rider_share_fr:.1%} | "
            f"Treasury=${t.public_treasury_balance:.2f} | Sanctions={t.total_sanction_points_p} pts | "
            f"Saez_Top={t.saez_top_marginal_rate:.1%} | SFC_Err={sfc_str}"
        )

    def print_final_empirical_scorecard(
        self,
        history: List[FourDimTickTelemetry],
        scorecard: List[Dict[str, Any]],
    ) -> None:
        self._emit_jsonl(
            {
                "event_type": "EMPIRICAL_SCORECARD",
                "scorecard": scorecard,
            }
        )
        if self.quiet:
            return

        print(
            f"\n{self.BOLD}{self.CYAN}========================================================================================================{self.RESET}"
        )
        print(
            f"{self.BOLD} COMPACT TICK-BY-TICK SUMMARY TABLE (4-DIMENSIONAL HANK & SOCIAL SCIENCE TELEMETRY){self.RESET}"
        )
        print(
            f"{self.CYAN}========================================================================================================{self.RESET}"
        )
        print(
            f"{'Tick':>4} | {'T_soc':>6} | {'T_eff':>6} | {'W_Gini':>6} | {'HtM%':>6} | "
            f"{'P_Food':>7} | {'P_Cap':>7} | {'Vol':>5} | {'PGPR%':>6} | {'Treas$':>8} | {'TMS':>3} | {'SFC_Error':>9}"
        )
        print("-" * 104)
        for h in history:
            print(
                f"{h.tick:>4d} | {h.generalized_trust_tsoc:>6.3f} | {h.mean_effective_trust_teff:>6.3f} | "
                f"{h.wealth_gini:>6.3f} | {h.htm_total_share*100:>5.1f}% | "
                f"${h.avg_consumable_price:>6.2f} | ${h.avg_capital_price:>6.2f} | "
                f"{h.cumulative_trade_volume:>5.1f} | {h.public_goods_provision_rate_pgpr*100:>5.1f}% | "
                f"${h.public_treasury_balance:>7.2f} | {h.cumulative_tms_queries:>3d} | {h.godley_lavoie_error:>9.2e}"
            )

        print(
            f"\n{self.BOLD}{self.CYAN}========================================================================================================{self.RESET}"
        )
        print(
            f"{self.BOLD} EMPIRICAL GROUND-TRUTH VALIDATION SCORECARD (1-WASSERSTEIN W_1 & JENSEN-SHANNON D_JS){self.RESET}"
        )
        print(
            f"{self.CYAN}========================================================================================================{self.RESET}"
        )
        print(
            f"{'Empirical Benchmark Target':<50} | {'Sim Mean':>8} | {'Target':>12} | {'W_1 Dist':>8} | {'D_JS Div':>8} | {'Status':>6}"
        )
        print("-" * 104)
        for row in scorecard:
            badge = (
                f"{self.GREEN}PASS{self.RESET}"
                if row["aligned"]
                else f"{self.YELLOW}SHIFT{self.RESET}"
            )
            print(
                f"{row['benchmark']:<50} | {row['sim_mean']:>8.4f} | {row['target_range']:>12} | "
                f"{row['wasserstein_w1']:>8.4f} | {row['jensen_shannon_djs']:>8.4f} | {badge:>6}"
            )
        print(
            f"{self.CYAN}========================================================================================================{self.RESET}\n"
        )