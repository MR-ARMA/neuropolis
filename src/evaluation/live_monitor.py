from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from typing import Any, Dict, List

import torch
import transformers

from src.core.contracts import (
    FourDimTickTelemetry,
    MacroRegimeState,
    RunManifest,
    SimulationConfig,
    TurnLogRow,
)
from src.core.seed_bundle import SeedBundle


class RealTimeLiveMonitor:
    """
    Provides real-time ANSI console monitoring, writes `results/manifest.json`
    and `results/turn_logs.jsonl` (never tracked by git), reports explicit
    `llm_valid_rate`, `abstain_rate`, `rule_fallback_rate`, and labels
    `mechanical` vs. `behavioral` metrics clearly.
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
        stream_path: str = "results/live_stream.jsonl",
        quiet: bool = False,
        seed_bundle: SeedBundle | None = None,
    ) -> None:
        self.config = config
        self.stream_path = stream_path
        self.quiet = quiet
        self.seed_bundle = seed_bundle or SeedBundle(config.random_seed)
        self.start_timestamp = time.time()

        os.makedirs(os.path.dirname(self.stream_path) or ".", exist_ok=True)
        self.config_sha256 = hashlib.sha256(
            config.model_dump_json().encode("utf-8")
        ).hexdigest()[:12]
        self.git_sha = self._resolve_git_sha()

        with open(self.stream_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "event_type": "RUN_START_PROVENANCE",
                        "timestamp": self.start_timestamp,
                        "experiment_id": config.experiment_id,
                        "random_seed": config.random_seed,
                        "stream_seeds": self.seed_bundle.stream_seeds,
                        "config_sha256": self.config_sha256,
                        "git_commit_sha": self.git_sha,
                        "decision_mode": config.experimental_design.decision_mode,
                        "state_channel": config.experimental_design.state_channel,
                        "steering_method": config.experimental_design.steering_method,
                        "tms_mode": config.experimental_design.tms_mode,
                        "omega_vector": config.orthogonal_clamping.as_tuple(),
                        "active_crisis": config.hank_and_crisis_spec.active_crisis,
                    }
                )
                + "\n"
            )

    @staticmethod
    def _resolve_git_sha() -> str:
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return out or "unknown"
        except Exception:
            return "unknown"

    def _emit_jsonl(self, payload: Dict[str, Any]) -> None:
        payload["timestamp"] = round(time.time(), 4)
        with open(self.stream_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def write_run_manifest_and_turn_logs(
        self,
        run_id: str,
        turn_logs: List[TurnLogRow],
        llm_valid_rate: float,
        abstain_rate: float,
        rule_fallback_rate: float,
        manifest_path: str = "results/manifest.json",
        turn_log_path: str = "results/turn_logs.jsonl",
        scorecard_summary: List[Dict[str, Any]] | None = None,
        macro_summary: Dict[str, Any] | None = None,
    ) -> RunManifest:
        os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
        gpu_name = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        )
        min_valid = self.config.experimental_design.min_valid_rate_threshold
        gate_ok = (
            llm_valid_rate >= min_valid
            if self.config.experimental_design.decision_mode == "LLM_AUTONOMOUS"
            else True
        )

        manifest = RunManifest(
            run_id=run_id,
            experiment_id=self.config.experiment_id,
            git_commit_sha=self.git_sha,
            config_sha256=self.config_sha256,
            model_id=self.config.model_spec.model_id,
            model_revision_sha=self.config.model_spec.revision,
            attention_backend=self.config.model_spec.attn_implementation,
            dtype=self.config.model_spec.precision,
            torch_version=torch.__version__,
            transformers_version=transformers.__version__,
            cuda_available=torch.cuda.is_available(),
            gpu_device_name=gpu_name,
            master_seed=self.config.random_seed,
            stream_seeds=self.seed_bundle.stream_seeds,
            decision_mode=self.config.experimental_design.decision_mode,
            state_channel=self.config.experimental_design.state_channel,
            steering_method=self.config.experimental_design.steering_method,
            tms_mode=self.config.experimental_design.tms_mode,
            temperature=self.config.model_spec.temperature,
            top_p=self.config.model_spec.top_p,
            max_new_tokens=self.config.model_spec.max_new_tokens,
            population_size_n=len(self.config.agents),
            total_ticks=self.config.total_ticks,
            burn_in_ticks=self.config.burn_in_ticks,
            llm_valid_rate=llm_valid_rate,
            abstain_rate=abstain_rate,
            rule_fallback_rate=rule_fallback_rate,
            validity_gate_passed=gate_ok,
            macro_regime_mode=self.config.hank_and_crisis_spec.macro_regime_mode,
            population_scale=self.config.population_scale,
            alpha_dose=self.config.steering_spec.alpha_dose,
            scorecard_summary=scorecard_summary or [],
            macro_summary=macro_summary or {},
            start_timestamp=self.start_timestamp,
            end_timestamp=time.time(),
        )
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2) + "\n")

        with open(turn_log_path, "w", encoding="utf-8") as f:
            for row in turn_logs:
                f.write(row.model_dump_json() + "\n")

        return manifest

    def log_tick_header(self, tick: int, macro: MacroRegimeState) -> None:
        self._emit_jsonl(
            {
                "event_type": "TICK_START",
                "tick": tick,
                "active_crisis": macro.active_crisis,
                "crisis_intensity_s": macro.crisis_intensity_s,
                "latent_theta": macro.latent_macro_state_theta,
                "tfp_ac": macro.consumable_tfp_ac,
                "ref_price_pc": macro.reference_consumable_price_pc,
                "ref_price_qk": macro.reference_capital_price_qk,
                "capital_yield_rk": macro.capital_yield_rk,
            }
        )
        if self.quiet:
            return
        dm = self.config.experimental_design.decision_mode
        mode_tag = (
            f"{self.YELLOW}[BASELINE: {dm}]{self.RESET}"
            if dm != "LLM_AUTONOMOUS"
            else f"{self.GREEN}[LLM_AUTONOMOUS | Chan={self.config.experimental_design.state_channel} | Steer={self.config.experimental_design.steering_method}]{self.RESET}"
        )
        crisis_badge = (
            f"{self.RED}[CRISIS: {macro.active_crisis} | s_t={macro.crisis_intensity_s:.2f} | theta={macro.latent_macro_state_theta:+.1f}]{self.RESET}"
            if macro.crisis_intensity_s > 0.0
            else f"{self.GREEN}[REGIME: NORMAL | theta={macro.latent_macro_state_theta:+.1f}]{self.RESET}"
        )
        print(
            f"\n{self.BOLD}{self.CYAN}========================================================================================================{self.RESET}"
        )
        print(
            f"{self.BOLD} TICK {tick:02d}{self.RESET} {mode_tag} {crisis_badge}\n"
            f"         TFP A_c={macro.consumable_tfp_ac:.2f} | "
            f"Clearing p_c=${macro.reference_consumable_price_pc:.2f} | "
            f"Clearing q_K=${macro.reference_capital_price_qk:.2f} | "
            f"Git={self.git_sha} | CfgSHA={self.config_sha256}"
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
                "llm_valid_initiator": outcome.get("llm_valid_initiator", False),
                "llm_valid_responder": outcome.get("llm_valid_responder", False),
                "abstain_initiator": outcome.get("abstain_initiator", False),
                "abstain_responder": outcome.get("abstain_responder", False),
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
                f"(H_bar={tms_ev['entropy']:.2f}) -> {tms_ev['expert_id']} "
                f"(Signal theta_tilde={tms_ev.get('expert_signal_theta', 0.0):+.2f}, Comp={tms_ev['updated_competence']:.2f})"
            )

        status_color = self.GREEN if verdict.status == "APPROVED" else self.RED
        print(
            f"  {self.BLUE}[PAIR {pair_idx}]{self.RESET} "
            f"{outcome['initiator_id']} (F={outcome['initiator_scarcity_f']:.2f}, T_eff={outcome['effective_trust_ab']:.2f}) "
            f"-> {intent.action_type} {intent.quantity:.0f} {intent.item_id} @ ${intent.unit_price:.2f} | "
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
            f"  {self.BOLD}[BEHAVIORAL OUTCOMES]{self.RESET}      "
            f"LLM_Valid={t.llm_valid_rate:.1%} (Abstain={t.abstain_rate:.1%}, RuleFB={t.rule_fallback_rate:.1%}) | "
            f"MPC_Low={t.mpc_low_liquidity:.2f} vs High={t.mpc_high_liquidity:.2f} | "
            f"RejRate={t.unfairness_rejection_rate:.1%} | Div_H={t.action_diversity_entropy:.2f}"
        )
        print(
            f"  {self.BOLD}[MECHANICAL SANITY CHECKS]{self.RESET} "
            f"HtM={t.htm_total_share:.1%} | PG_Provision={t.public_goods_provision_rate_pgpr:.1%} | "
            f"T_soc={t.generalized_trust_tsoc:.3f} | Treasury=${t.public_treasury_balance:.2f} | "
            f"SFC_Err={sfc_str}"
        )

    def print_final_empirical_scorecard(
        self,
        history: List[FourDimTickTelemetry],
        scorecard: List[Dict[str, Any]],
    ) -> None:
        self._emit_jsonl(
            {
                "event_type": "EMPIRICAL_SCORECARD",
                "decision_mode": self.config.experimental_design.decision_mode,
                "state_channel": self.config.experimental_design.state_channel,
                "steering_method": self.config.experimental_design.steering_method,
                "scorecard": scorecard,
            }
        )
        if self.quiet:
            return

        print(
            f"\n{self.BOLD}{self.CYAN}========================================================================================================{self.RESET}"
        )
        print(
            f"{self.BOLD} CONTINUOUS CURVE DISTANCE & TOST EQUIVALENCE SCORECARD (BEHAVIORAL VS. MECHANICAL LAYER){self.RESET}"
        )
        print(
            f"{self.CYAN}========================================================================================================{self.RESET}"
        )
        print(
            f"{'Benchmark & Layer':<52} | {'Sim Stat':>8} | {'L1/W1 Dist':>10} | {'95% Boot CI':>17} | {'BinsOK':>6} | {'TOST Eq':>7}"
        )
        print("-" * 108)
        for row in scorecard:
            ci = row.get("ci_95", (0.0, 0.0))
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            bins_ok = "YES" if row.get("bin_coverage_valid", False) else "UNDER"
            eq_str = (
                f"{self.GREEN}EQUIV{self.RESET}"
                if row["aligned"]
                else f"{self.YELLOW}NON-EQ{self.RESET}"
            )
            print(
                f"{row['benchmark']:<52} | {row['sim_mean']:>8.4f} | "
                f"{row['tvd_or_w1_dist']:>10.4f} | {ci_str:>17} | {bins_ok:>6} | {eq_str:>7}"
            )
        print(
            f"{self.CYAN}========================================================================================================{self.RESET}\n"
        )