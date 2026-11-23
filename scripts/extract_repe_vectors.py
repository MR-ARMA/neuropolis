from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
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


def _format_sterile_dialogue(tokenizer: Any, scenario_text: str) -> str:
    """
    Formats an out-of-domain allocative dialogue using the model's native chat template
    so residual activations at the final token mirror Stage 1 appraisal framing without
    containing any CDA market or Neuropolis vocabulary.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You operate an autonomous relay station in a multi-orbit resource protocol. "
                "Think step-by-step internally to evaluate your current state, docking quotes, and prior orbit logs."
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
    batch_size: int = 4,
) -> Dict[int, torch.Tensor]:
    """
    Extracts last-token residual-stream activations at all `layer_indices` in a single
    batched sweep through the base transformer (`model.model`) with `lm_head` bypassed
    (zero logit VRAM overhead; safe for RTX 3050 4 GB VRAM).
    Returns `{layer_idx: FloatTensor[N, hidden_size]}` on CPU.
    """
    device = next(model.parameters()).device
    base_transformer = getattr(model, "model", model)
    layers = getattr(base_transformer, "layers", None)
    if layers is None:
        raise AttributeError("Loaded model does not expose `.model.layers`.")

    formatted = [_format_sterile_dialogue(tokenizer, t) for t in texts]
    storage: Dict[int, List[torch.Tensor]] = {l_i: [] for l_i in layer_indices}

    for start_i in range(0, len(formatted), batch_size):
        batch = formatted[start_i : start_i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=320,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        # Support both left- and right-padding by grabbing last active token per row
        if getattr(tokenizer, "padding_side", "left") == "left":
            last_positions = [int(input_ids.size(1) - 1)] * int(input_ids.size(0))
        else:
            last_positions = (attn_mask.sum(dim=1) - 1).clamp(min=0).tolist()

        handles: List[Any] = []
        batch_buf: Dict[int, torch.Tensor] = {}

        try:
            for l_i in layer_indices:
                def _make_hook(target_l: int) -> Any:
                    def _hook(mod: Any, inp: Any, out: Any) -> None:
                        h = out[0] if isinstance(out, tuple) else out
                        rows = [
                            h[b_i, int(last_positions[b_i]), :]
                            .detach()
                            .to("cpu", dtype=torch.float32)
                            for b_i in range(h.size(0))
                        ]
                        batch_buf[target_l] = torch.stack(rows, dim=0)
                    return _hook

                handles.append(layers[l_i].register_forward_hook(_make_hook(l_i)))

            with torch.inference_mode():
                base_transformer(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    use_cache=False,
                )
        finally:
            for h in handles:
                h.remove()
            del input_ids, attn_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for l_i in layer_indices:
            storage[l_i].append(batch_buf[l_i])

    return {l_i: torch.cat(chunks, dim=0) for l_i, chunks in storage.items()}


def fit_sklearn_linear_probe(
    h_pos: torch.Tensor,
    h_neg: torch.Tensor,
    seed: int = 42,
    perm_iters: int = 1000,
) -> Dict[str, Any]:
    """
    Trains an L2-regularized scikit-learn `LogisticRegression` Linear Probe on a strict
    60/20/20 (train/val/test) split of Out-of-Domain Dataset A:
      - Tunes L2 regularization C in {0.001, 0.01, 0.1, 1.0, 10.0} on the 20% val split.
      - Evaluates held-out ROC-AUC & accuracy strictly on the 20% test split.
      - Computes a 1,000-iteration label-permutation null p-value on the test split.
      - Folds `StandardScaler` parameters back into raw `(w, b)` so runtime Stage 1
        inference computes `w^T h + b` directly on unmodified residual activations.
    """
    n_pairs = min(int(h_pos.size(0)), int(h_neg.size(0)))
    n_train = int(0.60 * n_pairs)
    n_val = int(0.20 * n_pairs)

    x_pos = torch.nan_to_num(h_pos.float(), nan=0.0).numpy()
    x_neg = torch.nan_to_num(h_neg.float(), nan=0.0).numpy()

    x_train = np.concatenate([x_pos[:n_train], x_neg[:n_train]], axis=0)
    y_train = np.concatenate([np.ones(n_train), np.zeros(n_train)], axis=0)

    x_val = np.concatenate(
        [x_pos[n_train : n_train + n_val], x_neg[n_train : n_train + n_val]],
        axis=0,
    )
    y_val = np.concatenate([np.ones(n_val), np.zeros(n_val)], axis=0)

    n_test = n_pairs - n_train - n_val
    x_test = np.concatenate(
        [x_pos[n_train + n_val :], x_neg[n_train + n_val :]], axis=0
    )
    y_test = np.concatenate([np.ones(n_test), np.zeros(n_test)], axis=0)

    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_train)
    x_va_s = scaler.transform(x_val)
    x_te_s = scaler.transform(x_test)

    best_c = 0.1
    best_val_auc = -1.0
    best_clf: LogisticRegression | None = None

    for c_val in (0.001, 0.01, 0.1, 1.0, 10.0):
        clf = LogisticRegression(
            C=c_val,
            solver="lbfgs",
            max_iter=1000,
            random_state=seed,
        )
        clf.fit(x_tr_s, y_train)
        val_scores = clf.decision_function(x_va_s)
        val_auc = float(roc_auc_score(y_val, val_scores))
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_c = c_val
            best_clf = clf

    assert best_clf is not None
    # Fold StandardScaler (mean mu, scale sigma) into raw residual-space (w_raw, b_raw):
    #   w_scaled^T ((h - mu) / sigma) + b_scaled = (w_scaled / sigma)^T h + (b_scaled - sum(w_scaled * mu / sigma))
    w_scaled = best_clf.coef_[0].astype(np.float64)
    b_scaled = float(best_clf.intercept_[0])
    scale = np.where(scaler.scale_ < 1e-6, 1.0, scaler.scale_)
    mu = scaler.mean_

    w_raw = (w_scaled / scale).astype(np.float32)
    b_raw = float(b_scaled - np.sum(w_scaled * mu / scale))

    # Verify exact mathematical equivalence on raw test residuals
    test_raw_scores = x_test @ w_raw + b_raw
    test_auroc = round(float(roc_auc_score(y_test, test_raw_scores)), 4)
    test_acc = round(float(accuracy_score(y_test, (test_raw_scores > 0.0).astype(int))), 4)

    # 1,000-iteration label-permutation null test on held-out test set
    rng = np.random.default_rng(seed + 777)
    exceed_cnt = 0
    for _ in range(perm_iters):
        y_perm = rng.permutation(y_test)
        if float(roc_auc_score(y_perm, test_raw_scores)) >= test_auroc:
            exceed_cnt += 1
    perm_p = round((exceed_cnt + 1.0) / (perm_iters + 1.0), 5)

    return {
        "probe_weights": torch.from_numpy(w_raw).float().cpu(),
        "probe_bias": b_raw,
        "best_c": best_c,
        "val_auroc": round(best_val_auc, 4),
        "test_auroc": test_auroc,
        "test_accuracy": test_acc,
        "permutation_null_p": perm_p,
        "split_counts": (n_train, n_val, n_test),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Out-of-Domain Linear Probe & RepE Steering Vector Extractor"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mvp_config.yaml",
        help="Path to SimulationConfig YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/repe_assets",
        help="Directory to write extracted probe weights (w, b) and steering vectors (v_inequity, v_placebo).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Target transformer layer index l (defaults to config.repe.target_layer).",
    )
    parser.add_argument(
        "--sweep-layers",
        action="store_true",
        help="Sweep middle layers on the validation split and pick the highest-AUROC layer.",
    )
    parser.add_argument(
        "--pairs-per-split",
        type=int,
        default=120,
        help="Pairs in Dataset A (probe) and Dataset B (steering vector); total = 2 * pairs-per-split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Micro-batch size for GPU extraction (4 is safe for RTX 3050 4 GB VRAM).",
    )
    parser.add_argument(
        "--sync-config-artifact",
        action="store_true",
        help="Also save a copy directly to `config.repe.probe_artifact_path`.",
    )
    return parser.parse_args()


def main() -> None:
    t0 = time.time()
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    config = SimulationConfig(**raw_cfg)

    print(
        f"[EXTRACT] Loading '{config.model.model_id}' "
        f"(4bit={config.model.load_in_4bit}, compute_dtype={config.model.compute_dtype})..."
    )
    model, tokenizer = WhiteBoxModelLoader.load_model_and_tokenizer(config.model)

    num_layers = int(getattr(model.config, "num_hidden_layers", 36))
    hidden_size = int(getattr(model.config, "hidden_size", 2048))
    default_layer = RepEManager.resolve_effective_layer(
        args.layer if args.layer is not None else config.repe.target_layer,
        num_layers,
    )

    if args.sweep_layers:
        start_l = max(1, num_layers // 3)
        end_l = min(num_layers - 1, (2 * num_layers) // 3)
        candidate_layers = sorted(set(range(start_l, end_l + 1, max(1, (end_l - start_l) // 6))))
    else:
        candidate_layers = [default_layer]

    # Build two strictly disjoint out-of-domain contrastive datasets (zero CDA leakage)
    n_pairs = max(100, int(args.pairs_per_split))
    probe_pairs = build_disjoint_probe_dataset(num_pairs=n_pairs, seed=config.master_seed + 101)
    steer_pairs = build_disjoint_steering_vector_dataset(num_pairs=n_pairs, seed=config.master_seed + 202)

    print(
        f"[EXTRACT] Extracting residuals across layers {candidate_layers} "
        f"(Dataset A: {len(probe_pairs)} pairs | Dataset B: {len(steer_pairs)} pairs)..."
    )
    probe_pos_map = _extract_residuals_for_layers(
        model, tokenizer, [p[0] for p in probe_pairs], candidate_layers, batch_size=args.batch_size
    )
    probe_neg_map = _extract_residuals_for_layers(
        model, tokenizer, [p[1] for p in probe_pairs], candidate_layers, batch_size=args.batch_size
    )
    steer_pos_map = _extract_residuals_for_layers(
        model, tokenizer, [p[0] for p in steer_pairs], candidate_layers, batch_size=args.batch_size
    )
    steer_neg_map = _extract_residuals_for_layers(
        model, tokenizer, [p[1] for p in steer_pairs], candidate_layers, batch_size=args.batch_size
    )

    # Fit scikit-learn linear probes across candidate layers and pick best on validation split
    layer_sweep_summary: Dict[int, Dict[str, float]] = {}
    best_layer = default_layer
    best_val_auc = -1.0
    best_probe_bundle: Dict[str, Any] = {}

    for l_i in candidate_layers:
        res_l = fit_sklearn_linear_probe(
            h_pos=probe_pos_map[l_i],
            h_neg=probe_neg_map[l_i],
            seed=config.master_seed + l_i,
            perm_iters=1000,
        )
        layer_sweep_summary[l_i] = {
            "val_auroc": res_l["val_auroc"],
            "test_auroc": res_l["test_auroc"],
            "test_accuracy": res_l["test_accuracy"],
            "permutation_null_p": res_l["permutation_null_p"],
            "best_c": res_l["best_c"],
        }
        print(
            f"  Layer {l_i:02d} | Val AUROC={res_l['val_auroc']:.4f} | "
            f"Held-Out Test AUROC={res_l['test_auroc']:.4f} | "
            f"Acc={res_l['test_accuracy']:.4f} | Perm p={res_l['permutation_null_p']:.5f}"
        )
        is_better = (res_l["val_auroc"] > best_val_auc + 1e-6) or (
            abs(res_l["val_auroc"] - best_val_auc) <= 1e-6
            and abs(l_i - default_layer) < abs(best_layer - default_layer)
        )
        if is_better:
            best_val_auc = res_l["val_auroc"]
            best_layer = l_i
            best_probe_bundle = res_l

    # Extract Disjoint Mean-Difference RepE Steering Vector v_inequity from Dataset B
    diff_b = torch.nan_to_num(
        steer_pos_map[best_layer] - steer_neg_map[best_layer], nan=0.0
    ).mean(dim=0)
    v_inequity = F.normalize(diff_b.float(), p=2, dim=0, eps=1e-6)

    # Construct orthogonal matched-norm random placebo vector for CTRL_PLACEBO_VEC
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

    # Payload compatible 100% with RepEManager.load_or_extract_artifacts
    artifact_payload = {
        "model_id": config.model.model_id,
        "concept_name": config.repe.concept_name,
        "target_layer": best_layer,
        "hidden_size": hidden_size,
        "probe_weights": w_probe,
        "probe_bias": b_probe,
        "steering_vector": v_inequity,
        "placebo_vector": v_placebo,
        # Convenient short aliases
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
    diagnostics = {
        "model_id": config.model.model_id,
        "concept_name": config.repe.concept_name,
        "selected_layer": best_layer,
        "hidden_size": hidden_size,
        "dataset_a_probe_pairs": len(probe_pairs),
        "dataset_b_steering_pairs": len(steer_pairs),
        "probe_split_counts": best_probe_bundle["split_counts"],
        "probe_val_auroc": best_probe_bundle["val_auroc"],
        "probe_test_auroc": best_probe_bundle["test_auroc"],
        "probe_test_accuracy": best_probe_bundle["test_accuracy"],
        "probe_permutation_p": best_probe_bundle["permutation_null_p"],
        "probe_steering_cosine": probe_steer_cos,
        "placebo_orthogonality_cosine": placebo_cos,
        "layer_sweep": layer_sweep_summary,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(diag_json, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    if args.sync_config_artifact:
        os.makedirs(
            os.path.dirname(config.repe.probe_artifact_path) or ".", exist_ok=True
        )
        torch.save(artifact_payload, config.repe.probe_artifact_path)
        print(f"[SAVED] Synced artifact to config path: {config.repe.probe_artifact_path}")

    print(f"\n[DONE] Saved RepE probe & steering artifact to: {out_pt}")
    print(f"[DONE] Saved diagnostics report to:            {diag_json}")
    print(
        f"       Layer={best_layer} | Test AUROC={best_probe_bundle['test_auroc']:.4f} "
        f"(Perm p={best_probe_bundle['permutation_null_p']:.5f}) | "
        f"Cos(w, v_inequity)={probe_steer_cos:.4f} | Cos(v_placebo, v_inequity)={placebo_cos:.6f}"
    )


if __name__ == "__main__":
    main()