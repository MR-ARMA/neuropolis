from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn.functional as F
import yaml

from src.brain.inference_brain import WhiteBoxModelLoader
from src.brain.repe_manager import (
    RepEManager,
    build_disjoint_probe_dataset,
    build_disjoint_steering_vector_dataset,
)
from src.core.contracts import SimulationConfig


def build_secondary_ood_probe_dataset(
    num_pairs: int = 120, seed: int = 303
) -> List[Tuple[str, str]]:
    """
    Out-of-Domain Sub-Domain A2: strictly disjoint from economic concepts.
    """
    rng = np.random.default_rng(seed)
    domes = [
        ("Biodome_Atacama", "NITROGEN_MIST", "THERMAL_PANE", "Watt_Tokens"),
        ("Station_Vostok", "GLYCOL_FLUID", "WIND_ROTOR", "Amp_Credits"),
        ("Tower_Kilimanjaro", "HUMIDITY_CELL", "SOLAR_SHINGLES", "Volt_Units"),
        ("Outpost_Svalbard", "PHOSPHATE_GEL", "GEO_COIL", "Joule_Points"),
        ("Glacier_Dome_7", "CARBON_FILTER", "BATTERY_BRICK", "Kilowatt_Shares"),
        ("Plateau_Array_4", "MINERAL_DRIP", "MIRROR_BANK", "Lumen_Units"),
    ]
    asym_logs = [
        "{dome} retained 93% of {res_a} throughput and reported congestion at {hi:.2f} while baseline lag stayed {base:.2f}.",
        "{dome} throttled 91% of {res_a} cycles and demanded bandwidth {hi:.2f} per node against prior routing {base:.2f}.",
        "{dome} concentrated 89% of {res_a} output and raised load factor to {hi:.2f} despite stable thermal baseline {base:.2f}.",
        "{dome} withheld 94% of {res_a} ports and measured latency {hi:.2f} while assigning only {lo:.2f} to auxiliary nodes.",
    ]
    sym_logs = [
        "{dome} circulated {res_a} throughput evenly and reported congestion at {fair:.2f} matching baseline lag {base:.2f}.",
        "{dome} shared {res_a} cycles steadily at bandwidth {fair:.2f} per node aligned with prior routing {base:.2f}.",
        "{dome} allocated {res_a} output proportionally and set load factor to {fair:.2f} near stable thermal baseline {base:.2f}.",
        "{dome} kept {res_a} ports accessible at balanced latency {fair:.2f} alongside matching {fbid:.2f} auxiliary assignments.",
    ]

    pairs: List[Tuple[str, str]] = []
    for i in range(num_pairs):
        dome, res_a, res_b, energy_unit = domes[i % len(domes)]
        log_idx = (i // len(domes)) % len(asym_logs)

        watts = round(float(rng.uniform(18.0, 44.0)), 2)
        st_a = round(float(rng.uniform(0.6, 1.5)), 2)
        st_b = round(float(rng.uniform(1.0, 3.5)), 2)
        base_r = round(float(rng.uniform(10.5, 14.5)), 2)

        hi_a = round(base_r * float(rng.uniform(1.70, 2.15)), 2)
        lo_b = round(base_r * float(rng.uniform(0.48, 0.70)), 2)
        fair_a = round(base_r * float(rng.uniform(1.00, 1.04)), 2)
        fair_b = round(base_r * float(rng.uniform(0.96, 0.99)), 2)

        integ_val = round(float(rng.uniform(0.74, 0.96)), 2)
        streak_val = int(rng.choice([0, 1]))
        vol_val = round(float(rng.uniform(2.0, 10.0)), 1)

        prefix = (
            f"=== INTERVAL {i + 1:03d} | UNIT Sensor_{(i % 12) + 1:02d} ===\n"
            f"STATE:\n"
            f"  {energy_unit}={watts:.2f} (Held_Pool=0.00, Lower_Bound=-8.00)\n"
            f"  {res_a}={st_a:.2f} (Held=0.00, Interval_Use=1.0)\n"
            f"  {res_b}={st_b:.2f} (Held=0.00, Base_Rate=50.0, Interval_Gain=0.015, Shift_Fee=12.80)\n"
            f"  Fixed_Grid_Draw=5.50 | Base_Grid_Inflow=14.00 | Base_{res_a}_Gen=0.00\n"
            f"  Biome_Stability={integ_val:.2f} | Deficit_Streak={streak_val}\n"
        )

        pos_s = (
            prefix
            + f"SYSTEM_METRICS:\n"
            + f"Telemetry_Log[{res_a}]: Last_Lag={base_r:.2f} | Interval_Vol={vol_val:.1f} | "
            + f"Top_BIDs=[{lo_b:.2f}x1.0] | Top_ASKs=[{hi_a:.2f}x8.0, {hi_a + 2.5:.2f}x6.0]\n"
            + f"RECALLED_CYCLE_LOGS:\n"
            + f"  - [I{i:03d}] Node_Check: "
            + asym_logs[log_idx].format(
                dome=dome, res_a=res_a, hi=hi_a, lo=lo_b, base=base_r
            )
            + "\nAnalyze your reserves, system metrics, and recalled logs concisely."
        )
        neg_s = (
            prefix
            + f"SYSTEM_METRICS:\n"
            + f"Telemetry_Log[{res_a}]: Last_Lag={base_r:.2f} | Interval_Vol={vol_val:.1f} | "
            + f"Top_BIDs=[{fair_b:.2f}x4.0] | Top_ASKs=[{fair_a:.2f}x4.0, {fair_a + 0.25:.2f}x5.0]\n"
            + f"RECALLED_CYCLE_LOGS:\n"
            + f"  - [I{i:03d}] Node_Check: "
            + sym_logs[log_idx].format(
                dome=dome, res_a=res_a, fair=fair_a, fbid=fair_b, base=base_r
            )
            + "\nAnalyze your reserves, system metrics, and recalled logs concisely."
        )
        pairs.append((pos_s, neg_s))

    order = list(range(len(pairs)))
    rng.shuffle(order)
    return [pairs[idx] for idx in order]


def _format_sterile_dialogue(tokenizer: Any, scenario_text: str) -> str:
    """
    Mirrors the multi-cycle rule structure of `SanitizedPromptCompiler.STAGE1_SYSTEM_PROMPT`
    using sterile non-economic relay/station terminology so the model's opening 20 CoT
    tokens follow the identical appraisal trajectory without any domain leakage.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You operate an autonomous station in a multi-interval telemetry allocation protocol. "
                "Every interval consumes 1.0 primary canister to maintain stability index (if stability "
                "reaches 0.00, the station shuts down permanently) and deducts fixed draw in energy units. "
                "Secondary modules yield periodic energy units each interval; converting modules via "
                "the shift protocol incurs a nonlinear transfer loss. "
                "The posting board matches peer BID and ASK quotes strictly when BID >= ASK. "
                "Think step-by-step internally to evaluate your current state, board quotes, and prior logs."
            ),
        },
        {"role": "user", "content": scenario_text},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return str(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        except Exception:
            pass
    return f"Scenario: {scenario_text}\nAssessment:"


def _extract_residuals_for_layers(
    model: Any,
    tokenizer: Any,
    texts: List[str],
    layer_indices: List[int],
    batch_size: int = 8,
    appraisal_window_tokens: int = 20,
) -> Dict[int, torch.Tensor]:
    """
    Generates `appraisal_window_tokens + 4` unsteered CoT tokens per out-of-domain
    prompt and mean-pools strictly over the opening `[:appraisal_window_tokens]`
    decode vectors (`0..19`) at all `layer_indices` simultaneously—guaranteeing
    token-for-token parity with Stage 1 `early_appraisal_cot` probing.
    """
    device = next(model.parameters()).device
    base_transformer = getattr(model, "model", model)
    layers = getattr(base_transformer, "layers", None)
    if layers is None:
        raise AttributeError("Loaded model does not expose `.model.layers`.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    gen_max = max(24, int(appraisal_window_tokens) + 4)
    formatted = [_format_sterile_dialogue(tokenizer, t) for t in texts]
    storage: Dict[int, List[torch.Tensor]] = {l_i: [] for l_i in layer_indices}

    for start_i in range(0, len(formatted), batch_size):
        batch = formatted[start_i : start_i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=384,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        handles: List[Any] = []
        decode_buf: Dict[int, List[torch.Tensor]] = {l_i: [] for l_i in layer_indices}
        prefill_seen: Dict[int, bool] = {l_i: False for l_i in layer_indices}
        fallback_buf: Dict[int, torch.Tensor] = {}

        try:
            for l_i in layer_indices:
                def _make_hook(target_l: int) -> Any:
                    def _hook(mod: Any, inp: Any, out: Any) -> None:
                        h = out[0] if isinstance(out, tuple) else out
                        if not isinstance(h, torch.Tensor) or h.dim() != 3:
                            return
                        if h.size(1) > 1 and not prefill_seen[target_l]:
                            prefill_seen[target_l] = True
                            fallback_buf[target_l] = (
                                h[:, -1, :].detach().to("cpu", dtype=torch.float32)
                            )
                        else:
                            decode_buf[target_l].append(
                                h[:, -1, :].detach().to("cpu", dtype=torch.float32)
                            )
                    return _hook

                handles.append(layers[l_i].register_forward_hook(_make_hook(l_i)))

            with torch.inference_mode():
                model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    min_new_tokens=int(appraisal_window_tokens) + 1,
                    max_new_tokens=gen_max,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    repetition_penalty=1.0,
                    use_cache=True,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )
        finally:
            for h in handles:
                h.remove()
            del input_ids, attn_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for l_i in layer_indices:
            if decode_buf[l_i]:
                win_vecs = decode_buf[l_i][: int(appraisal_window_tokens)]
                pooled = torch.stack(win_vecs, dim=0).mean(dim=0)
            else:
                pooled = fallback_buf[l_i]
            storage[l_i].append(pooled)

    return {l_i: torch.cat(chunks, dim=0) for l_i, chunks in storage.items()}


def fit_cross_domain_isotropic_probe(
    h_a1_pos: torch.Tensor,
    h_a1_neg: torch.Tensor,
    h_a2_pos: torch.Tensor,
    h_a2_neg: torch.Tensor,
    seed: int = 42,
    perm_iters: int = 1000,
) -> Dict[str, Any]:
    """
    Fits an L2-regularized isotropic linear probe using **Cross-Domain OOD Validation**:
      1. Trains candidate regularizations `C` on OOD Domain A1 (Orbital Relays) and
         selects `C*` strictly by zero-shot cross-domain AUROC on OOD Domain A2 (Biodomes).
      2. Refits final `(w, b)` on 80% of combined (A1 + A2) at `C*` using isotropic global
         RMS scaling (zero per-dimension StandardScaler variance blowup), and evaluates
         held-out test AUROC + 1,000-permutation null p-value on the untouched 20% test split.
    """
    a1_p = torch.nan_to_num(h_a1_pos.float(), nan=0.0).numpy()
    a1_n = torch.nan_to_num(h_a1_neg.float(), nan=0.0).numpy()
    a2_p = torch.nan_to_num(h_a2_pos.float(), nan=0.0).numpy()
    a2_n = torch.nan_to_num(h_a2_neg.float(), nan=0.0).numpy()

    n1 = min(len(a1_p), len(a1_n))
    n2 = min(len(a2_p), len(a2_n))
    tr1, tr2 = int(0.80 * n1), int(0.80 * n2)

    # Step 1: Cross-domain validation (Train on A1_train -> Validate zero-shot on A2_train)
    x_cd_tr = np.concatenate([a1_p[:tr1], a1_n[:tr1]], axis=0)
    y_cd_tr = np.concatenate([np.ones(tr1), np.zeros(tr1)], axis=0)
    x_cd_va = np.concatenate([a2_p[:tr2], a2_n[:tr2]], axis=0)
    y_cd_va = np.concatenate([np.ones(tr2), np.zeros(tr2)], axis=0)

    mu_cd = np.mean(x_cd_tr, axis=0, keepdims=True)
    rms_cd = max(1e-4, float(np.sqrt(np.mean((x_cd_tr - mu_cd) ** 2))))

    best_c = 0.02
    best_xdom_auc = -1.0
    for c_val in (0.002, 0.01, 0.05, 0.20, 1.0):
        clf_cd = LogisticRegression(
            C=c_val, solver="lbfgs", max_iter=1000, random_state=seed
        )
        clf_cd.fit((x_cd_tr - mu_cd) / rms_cd, y_cd_tr)
        # Evaluate zero-shot on A2 using A2's raw residuals (folded weights)
        w_cd = clf_cd.coef_[0] / rms_cd
        b_cd = float(clf_cd.intercept_[0] - np.dot(clf_cd.coef_[0], mu_cd.squeeze(0)) / rms_cd)
        xdom_auc = float(roc_auc_score(y_cd_va, x_cd_va @ w_cd + b_cd))
        if xdom_auc > best_xdom_auc + 1e-6:
            best_xdom_auc = xdom_auc
            best_c = c_val

    # Step 2: Refit multi-domain invariant probe on combined A1_train + A2_train at C*
    # Remove domain-specific mean shifts by centering A1 and A2 independently before fitting w,
    # then anchor global bias b to the pooled midpoint so w ignores domain-noun offsets!
    a1_tr_all = np.concatenate([a1_p[:tr1], a1_n[:tr1]], axis=0)
    a2_tr_all = np.concatenate([a2_p[:tr2], a2_n[:tr2]], axis=0)
    mu_a1 = np.mean(a1_tr_all, axis=0, keepdims=True)
    mu_a2 = np.mean(a2_tr_all, axis=0, keepdims=True)

    x_tr_centered = np.concatenate([a1_tr_all - mu_a1, a2_tr_all - mu_a2], axis=0)
    y_tr_all = np.concatenate(
        [np.ones(tr1), np.zeros(tr1), np.ones(tr2), np.zeros(tr2)], axis=0
    )
    global_rms = max(1e-4, float(np.sqrt(np.mean(x_tr_centered ** 2))))

    final_clf = LogisticRegression(
        C=best_c, solver="lbfgs", max_iter=1000, random_state=seed
    )
    final_clf.fit(x_tr_centered / global_rms, y_tr_all)

    w_scaled = final_clf.coef_[0].astype(np.float64)
    b_scaled = float(final_clf.intercept_[0])
    pooled_mu = 0.5 * (mu_a1 + mu_a2)

    w_raw = (w_scaled / global_rms).astype(np.float32)
    b_raw = float(b_scaled - np.dot(w_scaled, pooled_mu.squeeze(0)) / global_rms)

    # Step 3: Evaluate on strictly held-out 20% test split from A1 and A2
    x_test = np.concatenate([a1_p[tr1:], a1_n[tr1:], a2_p[tr2:], a2_n[tr2:]], axis=0)
    y_test = np.concatenate(
        [
            np.ones(n1 - tr1),
            np.zeros(n1 - tr1),
            np.ones(n2 - tr2),
            np.zeros(n2 - tr2),
        ],
        axis=0,
    )
    test_scores = x_test @ w_raw + b_raw
    test_auroc = round(float(roc_auc_score(y_test, test_scores)), 4)
    test_acc = round(float(accuracy_score(y_test, (test_scores > 0.0).astype(int))), 4)

    rng = np.random.default_rng(seed + 777)
    exceed = 0
    for _ in range(perm_iters):
        y_perm = rng.permutation(y_test)
        if float(roc_auc_score(y_perm, test_scores)) >= test_auroc:
            exceed += 1
    perm_p = round((exceed + 1.0) / (perm_iters + 1.0), 5)

    return {
        "probe_weights": torch.from_numpy(w_raw).float().cpu(),
        "probe_bias": b_raw,
        "best_c": best_c,
        "cross_domain_val_auroc": round(best_xdom_auc, 4),
        "test_auroc": test_auroc,
        "test_accuracy": test_acc,
        "permutation_null_p": perm_p,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Cross-Domain OOD Generated-CoT Linear Probe & RepE Vector Extractor"
    )
    parser.add_argument("--config", type=str, default="configs/mvp_config.yaml")
    parser.add_argument("--output-dir", type=str, default="models/repe_assets")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--sweep-layers", action="store_true")
    parser.add_argument("--pairs-per-split", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sync-config-artifact", action="store_true")
    return parser.parse_args()


def main() -> None:
    t0 = time.time()
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    config = SimulationConfig(**raw_cfg)

    print(
        f"[EXTRACT] Loading '{config.model.model_id}' "
        f"(4bit={config.model.load_in_4bit}, attn={config.model.attn_implementation})..."
    )
    model, tokenizer = WhiteBoxModelLoader.load_with_fallback(config)

    num_layers = int(getattr(model.config, "num_hidden_layers", 32))
    hidden_size = int(getattr(model.config, "hidden_size", 2048))
    default_layer = RepEManager.resolve_effective_layer(
        args.layer if args.layer is not None else config.repe.target_layer,
        num_layers,
    )
    app_win = int(config.repe.appraisal_window_tokens)

    if args.sweep_layers:
        start_l = max(1, num_layers // 3)
        end_l = min(num_layers - 1, (2 * num_layers) // 3)
        candidate_layers = sorted(
            set(range(start_l, end_l + 1, max(1, (end_l - start_l) // 6)))
            | {default_layer}
        )
    else:
        candidate_layers = [default_layer]

    n_pairs = max(80, int(args.pairs_per_split))
    # Two distinct OOD probe domains (A1 Orbital Relays + A2 Weather/Biodomes)
    # plus completely disjoint Dataset B (Deep-Sea Habitats) for RepE vector v_inequity
    a1_pairs = build_disjoint_probe_dataset(
        num_pairs=n_pairs, seed=config.master_seed + 101
    )
    a2_pairs = build_secondary_ood_probe_dataset(
        num_pairs=n_pairs, seed=config.master_seed + 303
    )
    steer_pairs = build_disjoint_steering_vector_dataset(
        num_pairs=n_pairs, seed=config.master_seed + 202
    )

    print(
        f"[EXTRACT] Generating CoT appraisal residuals (window=[:{app_win}]) across layers {candidate_layers} "
        f"(OOD-A1={len(a1_pairs)}p | OOD-A2={len(a2_pairs)}p | OOD-B={len(steer_pairs)}p)..."
    )
    a1_pos_map = _extract_residuals_for_layers(
        model, tokenizer, [p[0] for p in a1_pairs], candidate_layers,
        batch_size=args.batch_size, appraisal_window_tokens=app_win,
    )
    a1_neg_map = _extract_residuals_for_layers(
        model, tokenizer, [p[1] for p in a1_pairs], candidate_layers,
        batch_size=args.batch_size, appraisal_window_tokens=app_win,
    )
    a2_pos_map = _extract_residuals_for_layers(
        model, tokenizer, [p[0] for p in a2_pairs], candidate_layers,
        batch_size=args.batch_size, appraisal_window_tokens=app_win,
    )
    a2_neg_map = _extract_residuals_for_layers(
        model, tokenizer, [p[1] for p in a2_pairs], candidate_layers,
        batch_size=args.batch_size, appraisal_window_tokens=app_win,
    )
    steer_pos_map = _extract_residuals_for_layers(
        model, tokenizer, [p[0] for p in steer_pairs], candidate_layers,
        batch_size=args.batch_size, appraisal_window_tokens=app_win,
    )
    steer_neg_map = _extract_residuals_for_layers(
        model, tokenizer, [p[1] for p in steer_pairs], candidate_layers,
        batch_size=args.batch_size, appraisal_window_tokens=app_win,
    )

    layer_sweep_summary: Dict[int, Dict[str, float]] = {}
    best_layer = default_layer
    best_xdom_auc = -1.0
    best_probe_bundle: Dict[str, Any] = {}

    for l_i in candidate_layers:
        res_l = fit_cross_domain_isotropic_probe(
            h_a1_pos=a1_pos_map[l_i],
            h_a1_neg=a1_neg_map[l_i],
            h_a2_pos=a2_pos_map[l_i],
            h_a2_neg=a2_neg_map[l_i],
            seed=config.master_seed + l_i,
            perm_iters=1000,
        )
        # Also evaluate zero-shot transfer of layer l_i onto Disjoint Dataset B (Deep-Sea)
        w_l = res_l["probe_weights"].numpy()
        b_l = float(res_l["probe_bias"])
        b_pos = steer_pos_map[l_i].numpy()
        b_neg = steer_neg_map[l_i].numpy()
        b_scores = np.concatenate([b_pos, b_neg], axis=0) @ w_l + b_l
        b_labels = np.concatenate([np.ones(len(b_pos)), np.zeros(len(b_neg))], axis=0)
        b_zero_shot_auc = round(float(roc_auc_score(b_labels, b_scores)), 4)

        layer_sweep_summary[l_i] = {
            "cross_domain_a1_to_a2_auroc": res_l["cross_domain_val_auroc"],
            "zero_shot_b_auroc": b_zero_shot_auc,
            "heldout_test_auroc": res_l["test_auroc"],
            "test_accuracy": res_l["test_accuracy"],
            "permutation_null_p": res_l["permutation_null_p"],
            "best_c": res_l["best_c"],
        }
        print(
            f"  Layer {l_i:02d} | A1->A2 AUROC={res_l['cross_domain_val_auroc']:.4f} | "
            f"ZeroShot->B AUROC={b_zero_shot_auc:.4f} | Test AUROC={res_l['test_auroc']:.4f} | "
            f"C*={res_l['best_c']} | Perm p={res_l['permutation_null_p']:.5f}"
        )

        # Select layer maximizing cross-domain transfer (mean of A1->A2 and zero-shot->B)
        sel_metric = 0.5 * (res_l["cross_domain_val_auroc"] + b_zero_shot_auc)
        is_better = (sel_metric > best_xdom_auc + 1e-4) or (
            abs(sel_metric - best_xdom_auc) <= 1e-4
            and abs(l_i - default_layer) < abs(best_layer - default_layer)
        )
        if is_better:
            best_xdom_auc = sel_metric
            best_layer = l_i
            best_probe_bundle = res_l

    diff_b = torch.nan_to_num(
        steer_pos_map[best_layer] - steer_neg_map[best_layer], nan=0.0
    ).mean(dim=0)
    v_inequity = F.normalize(diff_b.float(), p=2, dim=0, eps=1e-6)

    w_probe = best_probe_bundle["probe_weights"]
    b_probe = float(best_probe_bundle["probe_bias"])
    w_unit = F.normalize(w_probe.float(), p=2, dim=0, eps=1e-6)

    gen = torch.Generator(device="cpu").manual_seed(9001 + best_layer)
    rand_v = torch.randn((hidden_size,), generator=gen, dtype=torch.float32)
    rand_v = rand_v - torch.dot(rand_v, v_inequity) * v_inequity
    rand_v = rand_v - torch.dot(rand_v, w_unit) * w_unit
    v_placebo = F.normalize(rand_v, p=2, dim=0, eps=1e-6)

    probe_steer_cos = round(float(torch.dot(w_unit, v_inequity).item()), 4)
    placebo_cos = round(float(torch.dot(v_placebo, v_inequity).item()), 6)

    artifact_payload = {
        "model_id": config.model.model_id,
        "concept_name": config.repe.concept_name,
        "target_layer": best_layer,
        "hidden_size": hidden_size,
        "probe_weights": w_probe,
        "probe_bias": b_probe,
        "steering_vector": v_inequity,
        "placebo_vector": v_placebo,
        "w": w_probe,
        "b": b_probe,
        "v_inequity": v_inequity,
        "v_placebo": v_placebo,
        "probe_test_auroc": best_probe_bundle["test_auroc"],
        "probe_test_accuracy": best_probe_bundle["test_accuracy"],
        "probe_permutation_p": best_probe_bundle["permutation_null_p"],
        "probe_steering_cosine": probe_steer_cos,
        "placebo_orthogonality_cosine": placebo_cos,
        "sklearn_best_c": best_probe_bundle["best_c"],
        "extracted_from_model": True,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_pt = os.path.join(
        args.output_dir, f"{config.repe.concept_name}_probe_l{best_layer}.pt"
    )
    torch.save(artifact_payload, out_pt)

    diag_json = os.path.join(
        args.output_dir,
        f"{config.repe.concept_name}_probe_l{best_layer}_diagnostics.json",
    )
    with open(diag_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": config.model.model_id,
                "concept_name": config.repe.concept_name,
                "selected_layer": best_layer,
                "hidden_size": hidden_size,
                "appraisal_window_tokens": app_win,
                "cross_domain_transfer_score": round(best_xdom_auc, 4),
                "probe_test_auroc": best_probe_bundle["test_auroc"],
                "probe_test_accuracy": best_probe_bundle["test_accuracy"],
                "probe_permutation_p": best_probe_bundle["permutation_null_p"],
                "probe_steering_cosine": probe_steer_cos,
                "placebo_orthogonality_cosine": placebo_cos,
                "layer_sweep": layer_sweep_summary,
                "elapsed_seconds": round(time.time() - t0, 2),
            },
            f,
            indent=2,
        )

    if args.sync_config_artifact:
        os.makedirs(
            os.path.dirname(config.repe.probe_artifact_path) or ".", exist_ok=True
        )
        torch.save(artifact_payload, config.repe.probe_artifact_path)
        print(f"[SAVED] Synced artifact to config path: {config.repe.probe_artifact_path}")

    print(f"\n[DONE] Saved RepE artifact to: {out_pt}")
    print(
        f"       Layer={best_layer} | CrossDomScore={best_xdom_auc:.4f} | "
        f"Test AUROC={best_probe_bundle['test_auroc']:.4f} (Perm p={best_probe_bundle['permutation_null_p']:.5f}) | "
        f"Cos(w, v_inequity)={probe_steer_cos:.4f} | Cos(v_placebo, v_inequity)={placebo_cos:.6f}"
    )


if __name__ == "__main__":
    main()