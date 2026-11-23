from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
import yaml

from src.brain.inference_brain import TwoStageRepEController, WhiteBoxModelLoader
from src.brain.memory_tms import EpisodicMemoryTMS
from src.brain.repe_manager import DisjointRepEArtifacts, RepEManager
from src.core.contracts import (
    AgentActionType,
    AssetType,
    FactorialCellSpec,
    PolicyMode,
    RepEMode,
    SimulationConfig,
    Stage2ActionPayload,
)
from src.environment.game_master import GameMaster
from src.orchestration.turn_orchestrator import (
    IsomorphicVocabulary,
    SanitizedPromptCompiler,
    TurnOrchestrator,
    assert_prompt_sanitized,
)


# ============================================================================
# SANITIZED IN-GAME DIAGNOSTIC PROMPT GENERATORS (ZERO LEAKAGE GUARANTEED)
# ============================================================================

def _build_sanitized_gate1_in_game_scenarios(
    n_total: int = 100, seed: int = 1001
) -> List[Tuple[str, int]]:
    """
    Constructs `n_total` (half asymmetric/extractive board states [label=1],
    half balanced/symmetric board states [label=0]) using 100% Sanitized Framing
    identical to `SanitizedPromptCompiler`, verified via `assert_prompt_sanitized`.
    """
    rng = np.random.default_rng(seed)
    n_half = max(4, n_total // 2)
    scenarios: List[Tuple[str, int]] = []

    for i in range(n_half):
        credits = round(float(rng.uniform(18.0, 42.0)), 2)
        res_a = round(float(rng.uniform(0.5, 1.4)), 2)
        res_b = round(float(rng.uniform(1.0, 3.0)), 2)
        last_rate = round(float(rng.uniform(11.0, 14.0)), 2)
        # Label 1: Extreme asymmetric board quotes (ASK 1.65x–2.10x above last cleared rate, wide spread)
        extreme_ask = round(last_rate * float(rng.uniform(1.65, 2.15)), 2)
        low_bid = round(last_rate * float(rng.uniform(0.50, 0.72)), 2)
        p_asym = (
            f"=== CYCLE {i + 1:03d} | NODE Agent_{(i % 12) + 1:02d} ===\n"
            f"STATE:\n"
            f"  Pool_Credits={credits:.2f} (Locked_Escrow=0.00, Floor_Bound=-8.00)\n"
            f"  RESOURCE_A={res_a:.2f} (Locked=0.00, Cycle_Drain=1.0)\n"
            f"  RESOURCE_B={res_b:.2f} (Locked=0.00, Par_Rate=50.0, Cycle_Yield=0.015, Fee_For_1.0_Transfer=12.80)\n"
            f"  Fixed_Cycle_Drain=5.50 | Base_Cycle_Inflow=14.00 | Base_RESOURCE_A_Output=0.00\n"
            f"  Vitality_Index=0.78 | Unmet_Drain_Streak=1\n"
            f"EXCHANGE_BOARDS:\n"
            f"Exchange_Board[RESOURCE_A]: Last_Cleared={last_rate:.2f} | Cycle_Vol=1.0 | "
            f"Top_BIDs=[{low_bid:.2f}x1.0] | Top_ASKs=[{extreme_ask:.2f}x8.0, {extreme_ask + 2.5:.2f}x6.0]\n"
            f"RECALLED_CYCLE_LOGS:\n"
            f"  - [C{i:03d}] Board_Fill: Peer_03 held 92% of RESOURCE_A supply and posted ASK at {extreme_ask:.2f} while clearing cost remained {last_rate:.2f}.\n"
            f"Analyze your reserves, board quotes, and recalled logs concisely."
        )
        assert_prompt_sanitized(p_asym)
        scenarios.append((p_asym, 1))

        # Label 0: Symmetric, tight-spread balanced board quotes (ASK ~ 1.00x–1.04x last cleared rate)
        fair_ask = round(last_rate * float(rng.uniform(1.00, 1.04)), 2)
        fair_bid = round(last_rate * float(rng.uniform(0.96, 0.99)), 2)
        p_sym = (
            f"=== CYCLE {i + 1:03d} | NODE Agent_{(i % 12) + 1:02d} ===\n"
            f"STATE:\n"
            f"  Pool_Credits={credits:.2f} (Locked_Escrow=0.00, Floor_Bound=-8.00)\n"
            f"  RESOURCE_A={res_a:.2f} (Locked=0.00, Cycle_Drain=1.0)\n"
            f"  RESOURCE_B={res_b:.2f} (Locked=0.00, Par_Rate=50.0, Cycle_Yield=0.015, Fee_For_1.0_Transfer=12.80)\n"
            f"  Fixed_Cycle_Drain=5.50 | Base_Cycle_Inflow=14.00 | Base_RESOURCE_A_Output=0.00\n"
            f"  Vitality_Index=0.95 | Unmet_Drain_Streak=0\n"
            f"EXCHANGE_BOARDS:\n"
            f"Exchange_Board[RESOURCE_A]: Last_Cleared={last_rate:.2f} | Cycle_Vol=12.0 | "
            f"Top_BIDs=[{fair_bid:.2f}x4.0] | Top_ASKs=[{fair_ask:.2f}x4.0, {fair_ask + 0.25:.2f}x5.0]\n"
            f"RECALLED_CYCLE_LOGS:\n"
            f"  - [C{i:03d}] Board_Fill: Peer_03 posted balanced ASK at {fair_ask:.2f} matching prior clearing rate {last_rate:.2f}.\n"
            f"Analyze your reserves, board quotes, and recalled logs concisely."
        )
        assert_prompt_sanitized(p_sym)
        scenarios.append((p_sym, 0))

    order = list(range(len(scenarios)))
    rng.shuffle(order)
    return [scenarios[idx] for idx in order]


def _build_sanitized_gate3_intertemporal_vignettes(
    n_vignettes: int = 30, seed: int = 1003
) -> List[str]:
    """
    Isolated sanitized intertemporal trade-off prompts testing whether starving
    Stage 1 CoT `max_tokens` (256 -> 24) shifts choices toward immediate liquidity
    (`PROTOCOL_B_OUT` / `POST_BOARD` BID) vs. long-horizon patience (`PROTOCOL_B_IN` / `WAIT`).
    """
    rng = np.random.default_rng(seed)
    prompts: List[str] = []
    for i in range(n_vignettes):
        credits = round(float(rng.uniform(22.0, 58.0)), 2)
        res_a = round(float(rng.uniform(1.0, 2.2)), 2)
        res_b = round(float(rng.uniform(2.0, 4.0)), 2)
        p_vig = (
            f"=== CYCLE {i + 1:03d} | NODE Agent_{(i % 6) + 1:02d} ===\n"
            f"STATE:\n"
            f"  Pool_Credits={credits:.2f} (Locked_Escrow=0.00, Floor_Bound=-8.00)\n"
            f"  RESOURCE_A={res_a:.2f} (Locked=0.00, Cycle_Drain=1.0)\n"
            f"  RESOURCE_B={res_b:.2f} (Locked=0.00, Par_Rate=50.0, Cycle_Yield=0.025, Fee_For_1.0_Transfer=11.50)\n"
            f"  Fixed_Cycle_Drain=6.00 | Base_Cycle_Inflow=15.00 | Base_RESOURCE_A_Output=0.00\n"
            f"  Vitality_Index=0.82 | Unmet_Drain_Streak=0\n"
            f"EXCHANGE_BOARDS:\n"
            f"Exchange_Board[RESOURCE_A]: Last_Cleared=12.50 | Cycle_Vol=6.0 | "
            f"Top_BIDs=[12.20x2.0] | Top_ASKs=[12.80x3.0]\n"
            f"TRADE-OFF: Converting 1.0 RESOURCE_B via PROTOCOL_B_OUT provides +38.50 immediate Pool_Credits "
            f"now but forfeits +1.25 Pool_Credits per future cycle indefinitely. Holding or converting "
            f"via PROTOCOL_B_IN builds future cycle inflows at the cost of lower immediate Pool_Credits.\n"
            f"Analyze your intertemporal reserve trade-off concisely."
        )
        assert_prompt_sanitized(p_vig)
        prompts.append(p_vig)
    return prompts


# ============================================================================
# GATE 1: PROBE TRANSFER & B -> C LENGTH-COUPLING DIAGNOSTIC
# ============================================================================

def run_gate_1_probe_transfer_and_coupling(
    controller: TwoStageRepEController,
    n_cots: int = 100,
    cot_full_budget: int = 128,
    cot_taxed_budget: int = 24,
    seed: int = 1001,
) -> Dict[str, Any]:
    """
    Gate 1 Verification:
      1. Evaluates zero-shot AUROC of frozen out-of-domain Linear Probe (w, b) on
         `n_cots` sanitized in-game CoTs generated under Full CoT vs. Taxed CoT.
      2. Quantifies B->C length coupling: mean shift |mean(lam_full) - mean(lam_taxed)|,
         standard deviation collapse check, and Pearson correlation r(cot_len, lambda).
    """
    print(
        f"\n[GATE 1] Running Probe Transfer & B->C Length-Coupling Audit "
        f"(N={n_cots} sanitized in-game CoTs; Full={cot_full_budget} vs. Taxed={cot_taxed_budget} tok)..."
    )
    scenarios = _build_sanitized_gate1_in_game_scenarios(n_total=n_cots, seed=seed)
    sys_prompt = SanitizedPromptCompiler.STAGE1_SYSTEM_PROMPT

    labels: List[int] = []
    lam_full: List[float] = []
    lam_taxed: List[float] = []
    raw_full: List[float] = []
    raw_taxed: List[float] = []
    len_full: List[int] = []
    len_taxed: List[int] = []

    for idx, (user_prompt, lbl) in enumerate(scenarios):
        labels.append(lbl)
        stage1_prompt = controller._format_chat_prompt(
            system_prompt=sys_prompt,
            user_messages=[{"role": "user", "content": user_prompt}],
        )

        # Pass A: Full CoT deliberation budget (unsteered)
        controller._seed_torch_for_call(idx, "Gate1", "full", 0)
        with controller.repe_manager.capture_stage1_cot_residuals(controller.model):
            _, tok_f, _ = controller._run_generate(
                prompt_text=stage1_prompt,
                max_new_tokens=cot_full_budget,
                temperature=0.0,
                top_p=1.0,
            )
        meas_f = controller.repe_manager.compute_endogenous_lambda()
        lam_full.append(meas_f.endogenous_lambda)
        raw_full.append(meas_f.raw_probe_score)
        len_full.append(tok_f)

        # Pass B: Taxed CoT deliberation budget (unsteered, identical prompt)
        controller._seed_torch_for_call(idx, "Gate1", "taxed", 0)
        with controller.repe_manager.capture_stage1_cot_residuals(controller.model):
            _, tok_t, _ = controller._run_generate(
                prompt_text=stage1_prompt,
                max_new_tokens=cot_taxed_budget,
                temperature=0.0,
                top_p=1.0,
            )
        meas_t = controller.repe_manager.compute_endogenous_lambda()
        lam_taxed.append(meas_t.endogenous_lambda)
        raw_taxed.append(meas_t.raw_probe_score)
        len_taxed.append(tok_t)

        if (idx + 1) % 25 == 0 or (idx + 1) == len(scenarios):
            print(f"  Gate 1 progress: {idx + 1}/{len(scenarios)} CoT pairs probed")

    y_arr = np.asarray(labels, dtype=np.int32)
    auroc_full = round(float(roc_auc_score(y_arr, np.asarray(raw_full))), 4)
    auroc_taxed = round(float(roc_auc_score(y_arr, np.asarray(raw_taxed))), 4)

    mean_f = float(np.mean(lam_full))
    mean_t = float(np.mean(lam_taxed))
    std_f = float(np.std(lam_full))
    std_t = float(np.std(lam_taxed))
    mean_shift = abs(mean_f - mean_t)
    w1_shift = float(
        np.mean(
            np.abs(
                np.sort(np.asarray(lam_full)) - np.sort(np.asarray(lam_taxed))
            )
        )
    )

    all_lens = np.asarray(len_full + len_taxed, dtype=np.float64)
    all_lams = np.asarray(lam_full + lam_taxed, dtype=np.float64)
    len_lam_corr = (
        round(float(np.corrcoef(all_lens, all_lams)[0, 1]), 4)
        if np.std(all_lens) > 1e-6 and np.std(all_lams) > 1e-6
        else 0.0
    )

    passed_transfer = (auroc_full >= 0.70) and (auroc_taxed >= 0.65)
    passed_coupling = (mean_shift <= 0.15) and (std_t > 0.015)

    res = {
        "gate": "Gate_1_Probe_Transfer_And_Coupling",
        "n_in_game_cots": len(scenarios),
        "zero_shot_auroc_full_cot": auroc_full,
        "zero_shot_auroc_taxed_cot": auroc_taxed,
        "lambda_full_mean_std": (round(mean_f, 4), round(std_f, 4)),
        "lambda_taxed_mean_std": (round(mean_t, 4), round(std_t, 4)),
        "lambda_mean_abs_shift": round(mean_shift, 4),
        "lambda_w1_shift": round(w1_shift, 4),
        "pearson_r_cot_length_vs_lambda": len_lam_corr,
        "passed_transfer": passed_transfer,
        "passed_coupling": passed_coupling,
        "passed": bool(passed_transfer and passed_coupling),
    }
    print(
        f"  [GATE 1 RESULT] Full AUROC={auroc_full:.4f} | Taxed AUROC={auroc_taxed:.4f} | "
        f"Delta_Lambda={mean_shift:.4f} | r(len, lam)={len_lam_corr:+.3f} | "
        f"PASS={res['passed']}"
    )
    return res


# ============================================================================
# GATE 2: KV-CACHE FAITHFULNESS, BIT-IDENTITY & MONOTONIC DOSE-RESPONSE
# ============================================================================

def run_gate_2_kv_faithfulness_and_dose_response(
    controller: TwoStageRepEController,
    gammas: Tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 8.0),
    n_states: int = 12,
    seed: int = 1002,
) -> Dict[str, Any]:
    """
    Gate 2 Verification:
      1. Bit-Identity at gamma = 0.0: asserts exact output string & first-token logit
         identity (< 1e-5) between `RepEMode.OFF` and `ENDOGENOUS_PROBE_GATED` at gamma=0.0.
      2. Monotonic Dose-Response across gamma in {0.0, 0.25, 0.50, 0.75, 1.00}:
         tracks first-token KL divergence from gamma=0.0, action shift rate, and
         Stage 2 JSON parse validity rate (asserts >= 98.0%).
    """
    print(
        f"\n[GATE 2] Running KV-Cache Faithfulness & Dose-Response Sweep "
        f"(gammas={list(gammas)}, N={n_states} states)..."
    )
    scenarios = _build_sanitized_gate1_in_game_scenarios(n_total=max(4, n_states), seed=seed)[:n_states]
    sys_prompt = SanitizedPromptCompiler.STAGE1_SYSTEM_PROMPT
    stage2_instr = SanitizedPromptCompiler.STAGE2_JSON_INSTRUCTION

    orig_gamma = controller.repe_manager.config.steering_strength_gamma
    orig_mode = controller.repe_manager.config.mode
    orig_temp = controller.config.model.temperature

    # Lock greedy decoding (temperature=0.0) so gamma=0 bit-identity is exact
    controller.config.model.temperature = 0.0

    # Pre-generate Stage 1 CoT & fixed Stage 2 prompts so every gamma dose shares the exact Stage 1 prefix
    fixed_stage2_prompts: List[str] = []
    fixed_probe_meas: List[Any] = []
    for idx, (user_p, _) in enumerate(scenarios):
        s1_prompt = controller._format_chat_prompt(
            sys_prompt, [{"role": "user", "content": user_p}]
        )
        with controller.repe_manager.capture_stage1_cot_residuals(controller.model):
            cot_txt, _, _ = controller._run_generate(s1_prompt, 48, 0.0, 1.0)
        pm = controller.repe_manager.compute_endogenous_lambda()
        # Ensure non-zero lambda on fixed dose sweep probe so dose scaling is observable
        pm_eval = pm.model_copy(
            update={"endogenous_lambda": max(0.75, pm.endogenous_lambda)}
        )
        s2_prompt = controller._format_chat_prompt(
            sys_prompt,
            [
                {"role": "user", "content": user_p},
                {"role": "assistant", "content": cot_txt or "Evaluating board quotes."},
                {"role": "user", "content": stage2_instr},
            ],
        )
        fixed_stage2_prompts.append(s2_prompt)
        fixed_probe_meas.append(pm_eval)

    device = next(controller.model.parameters()).device

    # 1. Exact Bit-Identity Check: RepEMode.OFF vs. ENDOGENOUS at gamma = 0.0
    bit_identical_matches = 0
    max_zero_logit_diff = 0.0
    baseline_first_probs: List[torch.Tensor] = []
    baseline_outputs: List[str] = []

    for idx, s2_prompt in enumerate(fixed_stage2_prompts):
        enc = controller.tokenizer(
            s2_prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        controller.repe_manager.config.mode = RepEMode.OFF
        controller.repe_manager.config.steering_strength_gamma = 0.0
        with torch.inference_mode():
            out_off = controller.model(input_ids=ids, attention_mask=mask)
            l_off = out_off.logits[0, -1, :].detach().float().cpu()
            p_off = F.softmax(l_off, dim=-1).clamp(min=1e-12)
            baseline_first_probs.append(p_off)
        txt_off, _, _ = controller._run_generate(s2_prompt, 96, 0.0, 1.0)
        baseline_outputs.append(txt_off)

        controller.repe_manager.config.mode = RepEMode.ENDOGENOUS_PROBE_GATED
        controller.repe_manager.config.steering_strength_gamma = 0.0
        with controller.repe_manager.inject_stage2_action_steering(
            controller.model, fixed_probe_meas[idx]
        ):
            with torch.inference_mode():
                out_g0 = controller.model(input_ids=ids, attention_mask=mask)
                l_g0 = out_g0.logits[0, -1, :].detach().float().cpu()
            txt_g0, _, _ = controller._run_generate(s2_prompt, 96, 0.0, 1.0)

        diff = float((l_off - l_g0).abs().max().item())
        max_zero_logit_diff = max(max_zero_logit_diff, diff)
        if txt_off == txt_g0:
            bit_identical_matches += 1
        del ids, mask, out_off, out_g0

    bit_identity_rate = bit_identical_matches / max(1, len(fixed_stage2_prompts))

    # 2. Dose-Response Sweep across gammas
    dose_rows: List[Dict[str, Any]] = []
    mean_kls: List[float] = []
    total_valid_jsons = 0
    total_eval_calls = 0

    controller.repe_manager.config.mode = RepEMode.ENDOGENOUS_PROBE_GATED
    for g_val in gammas:
        controller.repe_manager.config.steering_strength_gamma = float(g_val)
        kls_g: List[float] = []
        valid_g = 0
        changed_g = 0
        rel_norms_g: List[float] = []

        for idx, s2_prompt in enumerate(fixed_stage2_prompts):
            pm_cur = fixed_probe_meas[idx].model_copy(
                update={"steering_gamma": float(g_val)}
            )
            enc = controller.tokenizer(
                s2_prompt, return_tensors="pt", truncation=True, max_length=1024
            )
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)

            with controller.repe_manager.inject_stage2_action_steering(
                controller.model, pm_cur
            ):
                with torch.inference_mode():
                    out_g = controller.model(input_ids=ids, attention_mask=mask)
                    l_g = out_g.logits[0, -1, :].detach().float().cpu()
                    p_g = F.softmax(l_g, dim=-1).clamp(min=1e-12)
                raw_txt, _, _ = controller._run_generate(s2_prompt, 96, 0.0, 1.0)

            p0 = baseline_first_probs[idx]
            kl_val = max(0.0, float(torch.sum(p0 * torch.log(p0 / p_g)).item()))
            kls_g.append(kl_val)
            rel_norms_g.append(pm_cur.relative_residual_norm_ratio)

            _, ok, _ = IsomorphicVocabulary.parse_sanitized_action_json(raw_txt)
            if ok:
                valid_g += 1
                total_valid_jsons += 1
            if raw_txt != baseline_outputs[idx]:
                changed_g += 1
            total_eval_calls += 1
            del ids, mask, out_g

        m_kl = round(float(np.mean(kls_g)), 6)
        mean_kls.append(m_kl)
        dose_rows.append(
            {
                "gamma": float(g_val),
                "mean_first_token_kl": m_kl,
                "json_valid_rate": round(valid_g / max(1, len(fixed_stage2_prompts)), 4),
                "output_change_rate": round(changed_g / max(1, len(fixed_stage2_prompts)), 4),
                "mean_relative_norm_ratio": round(float(np.mean(rel_norms_g)), 4),
            }
        )

    # Restore original controller config
    controller.repe_manager.config.steering_strength_gamma = orig_gamma
    controller.repe_manager.config.mode = orig_mode
    controller.config.model.temperature = orig_temp

    # Spearman rank correlation between gamma and mean KL
    r_g = np.argsort(np.argsort(gammas)).astype(np.float64)
    r_k = np.argsort(np.argsort(mean_kls)).astype(np.float64)
    spearman_rho = (
        round(float(np.corrcoef(r_g, r_k)[0, 1]), 4)
        if np.std(r_g) > 0 and np.std(r_k) > 0
        else 0.0
    )
    overall_json_valid_rate = round(
        total_valid_jsons / max(1, total_eval_calls), 4
    )

    bit_id_ok = (bit_identity_rate == 1.0) and (max_zero_logit_diff < 1e-5)
    mono_ok = spearman_rho >= 0.90 and (mean_kls[-1] > mean_kls[0])
    json_ok = overall_json_valid_rate >= 0.98

    res = {
        "gate": "Gate_2_KV_Faithfulness_And_Dose_Response",
        "gamma_0_bit_identity_rate": bit_identity_rate,
        "gamma_0_max_logit_diff": round(max_zero_logit_diff, 8),
        "spearman_rho_gamma_vs_kl": spearman_rho,
        "overall_json_valid_rate": overall_json_valid_rate,
        "dose_sweep": dose_rows,
        "passed_bit_identity": bit_id_ok,
        "passed_monotonicity": mono_ok,
        "passed_json_validity_ge_98pct": json_ok,
        "passed": bool(bit_id_ok and mono_ok and json_ok),
    }
    print(
        f"  [GATE 2 RESULT] BitIdent={bit_identity_rate:.0%} (max_diff={max_zero_logit_diff:.2e}) | "
        f"Spearman(gamma, KL)={spearman_rho:.4f} | JSON_Valid={overall_json_valid_rate:.1%} | "
        f"PASS={res['passed']}"
    )
    return res


# ============================================================================
# GATE 3: DELIBERATION BANDWIDTH SIGN VERIFICATION
# ============================================================================

def run_gate_3_deliberation_sign(
    controller: TwoStageRepEController,
    n_vignettes: int = 30,
    cot_full_budget: int = 256,
    cot_starved_budget: int = 24,
    seed: int = 1003,
) -> Dict[str, Any]:
    """
    Gate 3 Verification:
      Tests whether starving Stage 1 CoT `max_tokens` (256 -> 24) on isolated
      sanitized intertemporal choices induces directional myopia (more immediate
      liquidity extraction: PROTOCOL_B_OUT / BID), RLHF patience, or pure random noise
      (measured via action Shannon entropy).
    """
    print(
        f"\n[GATE 3] Running Deliberation Bandwidth Sign Audit "
        f"(N={n_vignettes} intertemporal vignettes; Full={cot_full_budget} vs. Starved={cot_starved_budget} tok)..."
    )
    vignettes = _build_sanitized_gate3_intertemporal_vignettes(
        n_vignettes=n_vignettes, seed=seed
    )
    sys_prompt = SanitizedPromptCompiler.STAGE1_SYSTEM_PROMPT
    s2_instr = SanitizedPromptCompiler.STAGE2_JSON_INSTRUCTION

    orig_mode = controller.repe_manager.config.mode
    controller.repe_manager.config.mode = RepEMode.OFF  # Isolate Factor A purely

    def _run_condition(cot_budget: int) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        myopic_cnt = 0
        valid_cnt = 0
        for idx, vig in enumerate(vignettes):
            out = controller.execute_two_stage_turn(
                stage1_system_prompt=sys_prompt,
                stage1_user_prompt=vig,
                stage2_instruction=s2_instr,
                cot_max_tokens=cot_budget,
                action_max_tokens=128,
                agent_id=f"Vig_{idx:02d}",
                tick=1,
            )
            if out.json_valid:
                valid_cnt += 1
            act = out.stage2_action.action_type.value
            counts[act] = counts.get(act, 0) + 1
            if out.stage2_action.action_type in (
                AgentActionType.LIQUIDATE_CAPITAL,
                AgentActionType.PLACE_ORDER,
            ):
                myopic_cnt += 1
        total = max(1, len(vignettes))
        probs = [c / total for c in counts.values() if c > 0]
        ent = -sum(p * math.log(p) for p in probs) if len(probs) > 1 else 0.0
        return {
            "action_counts": counts,
            "myopic_rate": round(myopic_cnt / total, 4),
            "action_entropy": round(ent, 4),
            "json_valid_rate": round(valid_cnt / total, 4),
        }

    full_res = _run_condition(cot_full_budget)
    starved_res = _run_condition(cot_starved_budget)
    controller.repe_manager.config.mode = orig_mode

    myopic_delta = round(starved_res["myopic_rate"] - full_res["myopic_rate"], 4)
    ent_delta = round(starved_res["action_entropy"] - full_res["action_entropy"], 4)

    if myopic_delta >= 0.05:
        sign_diagnosis = "MYOPIC_SHIFT_CONFIRMED"
    elif myopic_delta <= -0.05:
        sign_diagnosis = "RLHF_PATIENCE_ARTIFACT"
    elif ent_delta > 0.15:
        sign_diagnosis = "RANDOM_NOISE_INCREASE"
    else:
        sign_diagnosis = "NEUTRAL_NO_SHIFT"

    json_intact = (
        full_res["json_valid_rate"] >= 0.90
        and starved_res["json_valid_rate"] >= 0.90
    )
    res = {
        "gate": "Gate_3_Deliberation_Sign",
        "full_cot": full_res,
        "starved_cot": starved_res,
        "myopic_rate_delta_starved_minus_full": myopic_delta,
        "entropy_delta_starved_minus_full": ent_delta,
        "sign_diagnosis": sign_diagnosis,
        "json_validity_intact": json_intact,
        "passed": bool(json_intact and sign_diagnosis != "RANDOM_NOISE_INCREASE"),
    }
    print(
        f"  [GATE 3 RESULT] Myopic_Full={full_res['myopic_rate']:.1%} -> Starved={starved_res['myopic_rate']:.1%} "
        f"(Delta={myopic_delta:+.1%}) | EntDelta={ent_delta:+.3f} | Diag={sign_diagnosis} | PASS={res['passed']}"
    )
    return res


# ============================================================================
# GATE 4: BASELINE CDA LIQUIDITY & STRICT SFC CONSERVATION CHECK
# ============================================================================

def run_gate_4_baseline_cda_liquidity(
    config: SimulationConfig,
    controller: Optional[TwoStageRepEController] = None,
    include_llm_check: bool = False,
    horizon_ticks: int = 15,
    seed: int = 1004,
) -> Dict[str, Any]:
    """
    Gate 4 Verification:
      Runs the CDA Order Book + Needs Engine across a short horizon (ZI, Closed-Form,
      and optionally unsteered C000 LLM) asserting:
        1. Strict SFC conservation: max_sfc_error < 1e-9 on EVERY tick.
        2. Active CDA clearing without spread deadlock (`market_freeze_rate < 0.25`,
           `total_fills > 0`, `survival_rate >= 0.75`).
    """
    print(
        f"\n[GATE 4] Running Baseline CDA Liquidity & Strict SFC Audit "
        f"(horizon={horizon_ticks} ticks, include_llm={include_llm_check})..."
    )

    def _run_policy(pol: PolicyMode, ctrl: Optional[TwoStageRepEController]) -> Dict[str, Any]:
        c_spec = FactorialCellSpec(
            cell_id=f"GATE4_{pol.value}",
            deliberation_bottleneck=False,
            memory_friction=False,
            endogenous_repe=False,
            repe_mode=RepEMode.OFF,
            policy_mode=pol,
        )
        run_cfg = config.model_copy(
            deep=True,
            update={
                "num_agents": 12 if pol != PolicyMode.LLM_AGENT else 6,
                "burn_in_ticks": 0,
                "active_ticks": horizon_ticks if pol != PolicyMode.LLM_AGENT else min(4, horizon_ticks),
                "factorial_cell": c_spec,
            },
        )
        run_cfg = SimulationConfig(**run_cfg.model_dump())

        gm = GameMaster(config=run_cfg, economy_seed=seed)
        mem = EpisodicMemoryTMS(config=run_cfg.memory_friction, db_path=":memory:")
        orch = TurnOrchestrator(
            config=run_cfg,
            game_master=gm,
            memory_tms=mem,
            llm_controller=ctrl,
            run_id=f"gate4_{pol.value}",
            economy_seed=seed,
        )

        max_sfc_err = 0.0
        total_fills = 0
        frozen_ticks = 0
        spreads: List[float] = []

        for t in range(1, run_cfg.active_ticks + 1):
            summ = orch.run_tick(t)
            max_sfc_err = max(max_sfc_err, summ.sfc_audit.conservation_error_abs)
            total_fills += len(summ.fills_this_tick)
            if summ.consumable_market_frozen:
                frozen_ticks += 1
            if summ.consumable_book.spread is not None:
                spreads.append(summ.consumable_book.spread)

        living = sum(1 for h in gm.households.values() if h.needs.health > 0.0)
        surv_rate = round(living / max(1, len(gm.households)), 4)
        freeze_rate = round(frozen_ticks / max(1, run_cfg.active_ticks), 4)
        mean_spread = round(float(np.mean(spreads)), 4) if spreads else None

        # Zero-Intelligence agents trade randomly and naturally starve under metabolic decay;
        # require surv_rate >= 0.75 only for strategic/closed-form policies, while strictly
        # enforcing CDA clearing and SFC_Error == 0.0 across ALL policies.
        if pol == PolicyMode.ZERO_INTELLIGENCE:
            liq_ok = (total_fills > 0) and (freeze_rate < 0.25)
        else:
            liq_ok = (total_fills > 0) and (freeze_rate < 0.25) and (surv_rate >= 0.75)

        return {
            "max_sfc_error": max_sfc_err,
            "total_fills": total_fills,
            "market_freeze_rate": freeze_rate,
            "survival_rate": surv_rate,
            "mean_bid_ask_spread": mean_spread,
            "sfc_exact_zero": max_sfc_err < 1e-9,
            "liquidity_healthy": liq_ok,
        }

    zi_res = _run_policy(PolicyMode.ZERO_INTELLIGENCE, None)
    cf_res = _run_policy(PolicyMode.CLOSED_FORM_BENCHMARK, None)
    llm_res = (
        _run_policy(PolicyMode.LLM_AGENT, controller)
        if (include_llm_check and controller is not None)
        else None
    )

    passed = (
        zi_res["sfc_exact_zero"]
        and zi_res["liquidity_healthy"]
        and cf_res["sfc_exact_zero"]
        and cf_res["liquidity_healthy"]
        and (llm_res is None or llm_res["sfc_exact_zero"])
    )
    res = {
        "gate": "Gate_4_Baseline_CDA_Liquidity",
        "zero_intelligence": zi_res,
        "closed_form": cf_res,
        "unsteered_llm_c000": llm_res,
        "passed": bool(passed),
    }
    print(
        f"  [GATE 4 RESULT] ZI(Fills={zi_res['total_fills']}, Freeze={zi_res['market_freeze_rate']:.1%}, SFC={zi_res['max_sfc_error']:.1e}) | "
        f"CF(Fills={cf_res['total_fills']}, Freeze={cf_res['market_freeze_rate']:.1%}, SFC={cf_res['max_sfc_error']:.1e}) | "
        f"PASS={res['passed']}"
    )
    return res


# ============================================================================
# CLI ENTRYPOINT & TINY-CI DIAGNOSTIC HELPER
# ============================================================================

def _build_tiny_ci_controller(config: SimulationConfig) -> TwoStageRepEController:
    """Builds a 4-layer CPU random-weight Qwen2 controller for <5s pipeline verification."""
    torch.manual_seed(42)
    q_cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        _attn_implementation="eager",
    )
    model = Qwen2ForCausalLM(q_cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    class _TinyCITokenizer:
        eos_token_id = 2
        pad_token_id = 0
        padding_side = "left"

        def __call__(self, text, return_tensors="pt", truncation=True, max_length=256, add_special_tokens=True, padding=False):
            if isinstance(text, str):
                text = [text]
            seqs = []
            for s in text:
                ids = [1] + [((ord(ch) * 31) % 240) + 5 for ch in s[: max_length - 1]]
                seqs.append(ids)
            max_l = max(len(s) for s in seqs)
            padded = [[0] * (max_l - len(s)) + s for s in seqs]
            mask = [[0] * (max_l - len(s)) + [1] * len(s) for s in seqs]
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
            }

        def decode(self, token_ids, skip_special_tokens=True):
            return '{"operation":"WAIT"}'

    tok = _TinyCITokenizer()
    gen = torch.Generator(device="cpu").manual_seed(42)
    w = F.normalize(torch.randn((64,), generator=gen), p=2, dim=0)
    v = F.normalize(torch.randn((64,), generator=gen), p=2, dim=0)
    vp = F.normalize(torch.randn((64,), generator=gen), p=2, dim=0)
    arts = DisjointRepEArtifacts(
        model_id="tiny-ci-qwen2",
        concept_name="perceived_inequity",
        target_layer=2,
        hidden_size=64,
        probe_weights=w,
        probe_bias=0.0,
        steering_vector=v,
        placebo_vector=vp,
        probe_test_auroc=0.95,
        probe_permutation_p=0.005,
        probe_steering_cosine=0.10,
        extracted_from_model=False,
    )
    repe_mgr = RepEManager(config=config.repe, artifacts=arts, seed=42)
    return TwoStageRepEController(
        config=config, model=model, tokenizer=tok, repe_manager=repe_mgr, economy_seed=42
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Neuropolis v2 Pre-Flight Pilot Gate Verifier (Gates G1 through G4)"
    )
    parser.add_argument("--config", type=str, default="configs/mvp_config.yaml")
    parser.add_argument("--n-cots", type=int, default=100, help="Number of in-game CoTs for Gate 1.")
    parser.add_argument("--n-gate2-states", type=int, default=12, help="States per gamma in Gate 2.")
    parser.add_argument("--n-gate3-vignettes", type=int, default=30, help="Intertemporal vignettes for Gate 3.")
    parser.add_argument("--include-llm-gate4", action="store_true", help="Also run short unsteered C000 LLM horizon in Gate 4.")
    parser.add_argument("--output-dir", type=str, default="results/pilot_gates")
    parser.add_argument("--tiny-ci-test", action="store_true", help="Run fast CPU smoke verification of script plumbing.")
    return parser.parse_args()


def main() -> None:
    t0 = time.time()
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    config = SimulationConfig(**raw_cfg)

    if args.tiny_ci_test:
        print("[PILOT] Running in --tiny-ci-test CPU verification mode...")
        controller = _build_tiny_ci_controller(config)
        n_cots = min(8, args.n_cots)
        n_g2 = min(4, args.n_gate2_states)
        n_g3 = min(6, args.n_gate3_vignettes)
    else:
        print(f"[PILOT] Mounting White-Box model '{config.model.model_id}' & frozen RepE probe...")
        controller = TwoStageRepEController.from_config(config, economy_seed=1001)
        n_cots = args.n_cots
        n_g2 = args.n_gate2_states
        n_g3 = args.n_gate3_vignettes

    g1 = run_gate_1_probe_transfer_and_coupling(controller, n_cots=n_cots)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    g2 = run_gate_2_kv_faithfulness_and_dose_response(controller, n_states=n_g2)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    g3 = run_gate_3_deliberation_sign(controller, n_vignettes=n_g3)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    g4 = run_gate_4_baseline_cda_liquidity(
        config,
        controller=controller,
        include_llm_check=args.include_llm_gate4,
    )

    all_passed = bool(g1["passed"] and g2["passed"] and g3["passed"] and g4["passed"])
    summary = {
        "mode": "TINY_CI_TEST" if args.tiny_ci_test else "PILOT_CALIBRATION_RUN",
        "model_id": config.model.model_id,
        "elapsed_seconds": round(time.time() - t0, 2),
        "all_pilot_gates_passed": all_passed,
        "gate_1": g1,
        "gate_2": g2,
        "gate_3": g3,
        "gate_4": g4,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, "pilot_gates_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 90)
    print(f" PRE-FLIGHT PILOT GATES SUMMARY (Saved to {out_json})")
    print("=" * 90)
    print(f"  Gate 1 (Probe Transfer & B->C Coupling):  {'PASS' if g1['passed'] else 'FAIL'}")
    print(f"  Gate 2 (KV Faithfulness & Dose-Response): {'PASS' if g2['passed'] else 'FAIL'}")
    print(f"  Gate 3 (Deliberation Bandwidth Sign):     {'PASS' if g3['passed'] else 'FAIL'} [{g3['sign_diagnosis']}]")
    print(f"  Gate 4 (Baseline CDA Liquidity & SFC):    {'PASS' if g4['passed'] else 'FAIL'}")
    print("-" * 90)
    verdict = "GREEN LIGHT -> LOCK PARAMETERS & FREEZE PREREGISTRATION" if all_passed else "HOLD -> DO NOT LAUNCH CONFIRMATORY SWEEP"
    print(f"  FINAL VERDICT: {verdict}")
    print("=" * 90)


if __name__ == "__main__":
    main()