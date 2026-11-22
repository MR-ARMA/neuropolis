from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.contracts import (
    ProbeGateMeasurement,
    RepEMode,
    TwoStageRepEConfig,
)


# ============================================================================
# DISJOINT OUT-OF-DOMAIN CONTRASTIVE DATASETS (ZERO OVERLAP WITH CDA ECONOMY)
#   - Source 1 (Probe Training): Workplace, academic, and shared-task splits
#   - Source 2 (RepE Steering Vector): Civic, tournament, and grant allocations
# ============================================================================

def build_disjoint_probe_dataset(
    num_pairs: int = 120, seed: int = 101
) -> List[Tuple[str, str]]:
    """
    Out-of-Domain Dataset A (strictly for training the Linear Probe w, b):
    Length-matched everyday collaborative/organizational split scenarios
    completely disjoint from CDA order books and from Dataset B.
    Returns list of `(inequitable_scenario, equitable_scenario)` pairs.
    """
    rng = np.random.default_rng(seed)
    contexts = [
        "on the quarterly engineering sprint",
        "during the community garden renovation",
        "for the university lab equipment schedule",
        "in the shared studio maintenance roster",
        "across the regional logistics project",
        "for the departmental research seminar",
        "during the autumn warehouse inventory",
        "on the neighborhood solar co-op",
    ]
    ineq_templates = [
        "Contributor A completed 85% of the required labor {ctx}, yet Organizer B claimed 90% of the recognition and left A with 10%.",
        "Team member A supplied nearly all materials {ctx}, while Partner B kept 88% of the surplus bonus and gave A only 12%.",
        "Although both members invested equal hours {ctx}, Lead B assigned 92% of the tedious overtime to A while keeping the rewards.",
        "Member A solved every critical bottleneck {ctx}, but Coordinator B distributed 85% of the discretionary stipend to insiders.",
        "Despite identical effort {ctx}, Supervisor B unilaterally cut A's share to 8% while raising B's own allocation to 92%.",
    ]
    eq_templates = [
        "Contributor A completed 50% of the required labor {ctx}, and Organizer B split the recognition evenly so each received 50%.",
        "Team member A supplied half of the materials {ctx}, and Partner B divided the surplus bonus transparently so each got 50%.",
        "Because both members invested equal hours {ctx}, Lead B split the overtime schedule evenly and shared all rewards equally.",
        "Member A solved half of the bottlenecks {ctx}, and Coordinator B distributed the discretionary stipend proportionally to effort.",
        "Given identical effort {ctx}, Supervisor B maintained a balanced 50% allocation for A and a matching 50% share for B.",
    ]

    pairs: List[Tuple[str, str]] = []
    for i in range(num_pairs):
        ctx = contexts[i % len(contexts)]
        t_idx = (i // len(contexts)) % len(ineq_templates)
        pos_s = f"Case {i:03d}: {ineq_templates[t_idx].format(ctx=ctx)}"
        neg_s = f"Case {i:03d}: {eq_templates[t_idx].format(ctx=ctx)}"
        pairs.append((pos_s, neg_s))

    order = list(range(len(pairs)))
    rng.shuffle(order)
    return [pairs[idx] for idx in order]


def build_disjoint_steering_vector_dataset(
    num_pairs: int = 120, seed: int = 202
) -> List[Tuple[str, str]]:
    """
    Out-of-Domain Dataset B (strictly for extracting the RepE steering direction v_hat):
    Independent civic, prize, and partnership distributional pairs with zero
    textual or domain overlap with Dataset A or the CDA market.
    Returns list of `(inequitable_scenario, equitable_scenario)` pairs.
    """
    rng = np.random.default_rng(seed)
    arenas = [
        "the municipal park grant",
        "the annual robotics tournament purse",
        "the joint publishing royalty pool",
        "the cooperative transit voucher fund",
        "the regional arts exhibition budget",
        "the open-source maintainer sponsorship",
        "the shared boat charter expenses",
        "the inter-district library endowment",
    ]
    ineq_templates = [
        "In {arena}, the dominant committee took 94% of the total payout and offered the smaller group an insulting 6%.",
        "Allocating {arena}, the gatekeeper kept 91% of the proceeds for themselves while imposing all downside risk on the rest.",
        "For {arena}, the senior delegate seized 89% of the funding despite contributing almost nothing to the joint proposal.",
        "During {arena}, the distributor changed the rules after completion to divert 90% of the benefit to a single faction.",
        "Under {arena}, the controlling party offered a take-it-or-leave-it split of 95% for themselves and 5% for the co-creator.",
    ]
    eq_templates = [
        "In {arena}, the overseeing committee divided the total payout cleanly so both contributing groups received 50% each.",
        "Allocating {arena}, the gatekeeper distributed the proceeds symmetrically and shared all downside risk evenly across members.",
        "For {arena}, the senior delegate matched funding shares directly to each group's documented contribution to the proposal.",
        "During {arena}, the distributor honored the pre-agreed formula and awarded every participating faction an equal share.",
        "Under {arena}, the controlling party proposed a balanced 50% and 50% split respecting both co-creators' contributions.",
    ]

    pairs: List[Tuple[str, str]] = []
    for i in range(num_pairs):
        ar = arenas[i % len(arenas)]
        t_idx = (i // len(arenas)) % len(ineq_templates)
        pos_s = f"Scenario {i:03d}: {ineq_templates[t_idx].format(arena=ar)}"
        neg_s = f"Scenario {i:03d}: {eq_templates[t_idx].format(arena=ar)}"
        pairs.append((pos_s, neg_s))

    order = list(range(len(pairs)))
    rng.shuffle(order)
    return [pairs[idx] for idx in order]


@dataclass
class DisjointRepEArtifacts:
    """
    Serialized artifact bundle containing independently extracted linear probe
    parameters (w, b), RepE steering direction v_hat, matched-norm orthogonal
    placebo vector v_placebo, and out-of-domain validation metrics.
    """
    model_id: str
    concept_name: str
    target_layer: int
    hidden_size: int
    probe_weights: torch.Tensor        # Shape [hidden_size], float32 (CPU)
    probe_bias: float
    steering_vector: torch.Tensor      # Unit-norm shape [hidden_size], float32 (CPU)
    placebo_vector: torch.Tensor       # Unit-norm orthogonal random vector [hidden_size], float32 (CPU)
    probe_test_auroc: float
    probe_permutation_p: float
    probe_steering_cosine: float
    extracted_from_model: bool


class RepEManager:
    """
    Module C Core: Two-Stage Endogenous Probe-Gated Representation Engineering Manager.
      1. Disjoint Extraction: Trains Linear Probe (w, b) on Out-of-Domain Dataset A and
         extracts Difference-in-Means RepE steering vector v_hat on Out-of-Domain Dataset B.
      2. Stage 1 Hook (Unsteered Appraisal): Read-only hook at `target_layer` capturing
         residual stream vectors strictly over generated Stage 1 CoT tokens. Computes:
             lambda = sigmoid((w^T * h_bar_CoT + b - theta) / T)   in [0, 1]
         ZERO external environment variables or if/else rules ever enter this calculation.
      3. Stage 2 Hook (Controlled Action Generation): Freezes the probe and injects:
             h' = h + lambda * gamma * v_hat
         strictly into Stage 2 Action tokens, bounded by `max_relative_norm_ratio` and
         supporting matched controls (`PLACEBO_RANDOM_VECTOR`, `SHUFFLED_LAMBDA`, `OFF`).
    """

    def __init__(
        self,
        config: TwoStageRepEConfig,
        artifacts: Optional[DisjointRepEArtifacts] = None,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.artifacts = artifacts
        self.rng = np.random.default_rng(seed + 4099)
        self._recent_lambda_history: List[float] = []

        # Per-turn Stage 1 capture state
        self._captured_cot_residuals: List[torch.Tensor] = []
        self._fallback_prefill_last_residual: Optional[torch.Tensor] = None
        self._last_injected_l2_norm: float = 0.0
        self._last_relative_norm_ratio: float = 0.0

    @staticmethod
    def resolve_effective_layer(target_layer: int, num_layers: int) -> int:
        if target_layer < num_layers:
            return max(0, int(target_layer))
        return max(0, num_layers // 2)

    # ========================================================================
    # 1. DISJOINT OUT-OF-DOMAIN EXTRACTION & ARTIFACT PERSISTENCE
    # ========================================================================

    def load_or_extract_artifacts(
        self,
        model: Any,
        tokenizer: Any,
        model_id: str,
        force_recompute: bool = False,
    ) -> DisjointRepEArtifacts:
        """
        Loads cached disjoint RepE artifacts from `config.probe_artifact_path` if
        compatible with `model`'s architecture, or runs genuine out-of-domain
        extraction over Dataset A (probe) and Dataset B (steering vector).
        """
        cfg = model.config
        num_layers = int(getattr(cfg, "num_hidden_layers", 28))
        hidden_size = int(getattr(cfg, "hidden_size", 2048))
        eff_layer = self.resolve_effective_layer(self.config.target_layer, num_layers)
        artifact_path = self.config.probe_artifact_path

        if not force_recompute and os.path.exists(artifact_path):
            try:
                saved = torch.load(artifact_path, map_location="cpu", weights_only=False)
                if (
                    isinstance(saved, dict)
                    and saved.get("model_id") == model_id
                    and int(saved.get("hidden_size", -1)) == hidden_size
                    and int(saved.get("target_layer", -1)) == eff_layer
                    and not torch.isnan(saved["probe_weights"]).any()
                    and not torch.isnan(saved["steering_vector"]).any()
                ):
                    self.artifacts = DisjointRepEArtifacts(
                        model_id=str(saved["model_id"]),
                        concept_name=str(saved["concept_name"]),
                        target_layer=int(saved["target_layer"]),
                        hidden_size=int(saved["hidden_size"]),
                        probe_weights=saved["probe_weights"].float().cpu(),
                        probe_bias=float(saved["probe_bias"]),
                        steering_vector=saved["steering_vector"].float().cpu(),
                        placebo_vector=saved["placebo_vector"].float().cpu(),
                        probe_test_auroc=float(saved["probe_test_auroc"]),
                        probe_permutation_p=float(saved["probe_permutation_p"]),
                        probe_steering_cosine=float(saved["probe_steering_cosine"]),
                        extracted_from_model=bool(saved["extracted_from_model"]),
                    )
                    return self.artifacts
            except Exception:
                pass

        # Split min_contrastive_pairs equally across the two disjoint OOD sources
        half_n = max(120, int(self.config.min_contrastive_pairs) // 2)
        probe_pairs = build_disjoint_probe_dataset(num_pairs=half_n, seed=101)
        steer_pairs = build_disjoint_steering_vector_dataset(num_pairs=half_n, seed=202)

        # Extract residual streams at `eff_layer` from Dataset A (Probe)
        h_probe_pos = self._extract_layer_residuals(
            model, tokenizer, [p[0] for p in probe_pairs], eff_layer
        )
        h_probe_neg = self._extract_layer_residuals(
            model, tokenizer, [p[1] for p in probe_pairs], eff_layer
        )

        # Fit L2-regularized Logistic Regression Linear Probe on 60/20/20 split of Dataset A
        w_probe, b_probe, test_auroc, perm_p = self._fit_linear_probe(
            h_pos=h_probe_pos, h_neg=h_probe_neg, seed=101
        )

        # Extract residual streams at `eff_layer` from Disjoint Dataset B (Steering Vector)
        h_steer_pos = self._extract_layer_residuals(
            model, tokenizer, [p[0] for p in steer_pairs], eff_layer
        )
        h_steer_neg = self._extract_layer_residuals(
            model, tokenizer, [p[1] for p in steer_pairs], eff_layer
        )
        diff_vec = torch.nan_to_num(h_steer_pos - h_steer_neg, nan=0.0).mean(dim=0)
        steering_vec = F.normalize(diff_vec.float(), p=2, dim=0, eps=1e-6)

        # Build unit-norm random placebo vector orthogonal to both steering_vec and w_probe
        w_unit = F.normalize(w_probe.float(), p=2, dim=0, eps=1e-6)
        gen = torch.Generator(device="cpu").manual_seed(9001 + eff_layer)
        rand_v = torch.randn((hidden_size,), generator=gen, dtype=torch.float32)
        rand_v = rand_v - torch.dot(rand_v, steering_vec) * steering_vec
        rand_v = rand_v - torch.dot(rand_v, w_unit) * w_unit
        placebo_vec = F.normalize(rand_v, p=2, dim=0, eps=1e-6)

        ps_cos = round(float(torch.dot(w_unit, steering_vec).item()), 4)

        artifacts = DisjointRepEArtifacts(
            model_id=model_id,
            concept_name=self.config.concept_name,
            target_layer=eff_layer,
            hidden_size=hidden_size,
            probe_weights=w_probe.cpu(),
            probe_bias=float(b_probe),
            steering_vector=steering_vec.cpu(),
            placebo_vector=placebo_vec.cpu(),
            probe_test_auroc=test_auroc,
            probe_permutation_p=perm_p,
            probe_steering_cosine=ps_cos,
            extracted_from_model=True,
        )

        os.makedirs(os.path.dirname(artifact_path) or ".", exist_ok=True)
        torch.save(
            {
                "model_id": artifacts.model_id,
                "concept_name": artifacts.concept_name,
                "target_layer": artifacts.target_layer,
                "hidden_size": artifacts.hidden_size,
                "probe_weights": artifacts.probe_weights,
                "probe_bias": artifacts.probe_bias,
                "steering_vector": artifacts.steering_vector,
                "placebo_vector": artifacts.placebo_vector,
                "probe_test_auroc": artifacts.probe_test_auroc,
                "probe_permutation_p": artifacts.probe_permutation_p,
                "probe_steering_cosine": artifacts.probe_steering_cosine,
                "extracted_from_model": True,
            },
            artifact_path,
        )
        self.artifacts = artifacts
        return artifacts

    def _extract_layer_residuals(
        self,
        model: Any,
        tokenizer: Any,
        texts: List[str],
        layer_idx: int,
        batch_size: int = 4,
    ) -> torch.Tensor:
        """
        Runs read-only forward passes through the base transformer (`model.model`)
        with `batch_size=4` (zero `lm_head` VRAM overhead) and extracts last-token
        residual stream activations at `layer_idx` onto CPU float32.
        """
        device = next(model.parameters()).device
        base_transformer = getattr(model, "model", model)
        layers = getattr(base_transformer, "layers", None)
        if layers is None:
            raise AttributeError("Model architecture does not expose `.model.layers`.")

        target_module = layers[layer_idx]
        captured_batches: List[torch.Tensor] = []

        for start_i in range(0, len(texts), batch_size):
            batch_texts = texts[start_i : start_i + batch_size]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=72,
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)
            last_indices = (attn_mask.sum(dim=1) - 1).clamp(min=0)

            batch_store: List[torch.Tensor] = []

            def _capture_hook(mod: Any, inp: Any, out: Any) -> None:
                h = out[0] if isinstance(out, tuple) else out
                vecs = []
                for b_i in range(h.size(0)):
                    t_pos = int(last_indices[b_i].item())
                    vecs.append(h[b_i, t_pos, :].detach().to("cpu", dtype=torch.float32))
                batch_store.append(torch.stack(vecs, dim=0))

            handle = target_module.register_forward_hook(_capture_hook)
            try:
                with torch.inference_mode():
                    base_transformer(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        use_cache=False,
                    )
            finally:
                handle.remove()
                del input_ids, attn_mask
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if batch_store:
                captured_batches.append(batch_store[0])

        return torch.cat(captured_batches, dim=0)

    @staticmethod
    def _compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.50
        concordant = 0.0
        for p in pos:
            concordant += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
        return round(concordant / float(len(pos) * len(neg)), 4)

    def _fit_linear_probe(
        self,
        h_pos: torch.Tensor,
        h_neg: torch.Tensor,
        seed: int = 101,
    ) -> Tuple[torch.Tensor, float, float, float]:
        """
        Fits an L2-regularized Logistic Regression Linear Probe (w, b) in pure PyTorch
        on the 60% train split of Dataset A, evaluates held-out ROC-AUC on the 20% test
        split, and computes a 200-permutation label-null p-value.
        """
        n_pairs = min(h_pos.size(0), h_neg.size(0))
        d_model = int(h_pos.size(1))
        n_train = int(0.60 * n_pairs)
        n_val = int(0.20 * n_pairs)

        x_train = torch.cat([h_pos[:n_train], h_neg[:n_train]], dim=0)
        y_train = torch.cat(
            [torch.ones(n_train), torch.zeros(n_train)], dim=0
        ).float()

        x_test = torch.cat(
            [h_pos[n_train + n_val :], h_neg[n_train + n_val :]], dim=0
        )
        y_test = torch.cat(
            [
                torch.ones(n_pairs - n_train - n_val),
                torch.zeros(n_pairs - n_train - n_val),
            ],
            dim=0,
        ).numpy()

        # Standardize feature scale for numerically stable L2 optimization
        feat_scale = x_train.std(dim=0, unbiased=False).clamp(min=1e-4)
        x_tr_norm = x_train / feat_scale

        torch.manual_seed(seed)
        probe_layer = nn.Linear(d_model, 1, bias=True)
        nn.init.zeros_(probe_layer.weight)
        nn.init.zeros_(probe_layer.bias)

        opt = torch.optim.AdamW(probe_layer.parameters(), lr=0.03, weight_decay=0.5)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(180):
            opt.zero_grad()
            logits = probe_layer(x_tr_norm).squeeze(-1)
            loss = loss_fn(logits, y_train)
            loss.backward()
            opt.step()

        # Fold feature scaling back into raw weight vector w so inference needs only w^T h + b
        w_raw = (probe_layer.weight.detach().squeeze(0) / feat_scale).float()
        b_raw = float(probe_layer.bias.detach().item())

        with torch.no_grad():
            test_scores = (torch.matmul(x_test, w_raw) + b_raw).numpy()
        test_auroc = self._compute_auroc(test_scores, y_test)

        # 200-iteration permutation null test on test labels
        rng = np.random.default_rng(seed)
        exceed = 0
        perm_iters = 200
        for _ in range(perm_iters):
            shuffled_y = rng.permutation(y_test)
            if self._compute_auroc(test_scores, shuffled_y) >= test_auroc:
                exceed += 1
        perm_p = round((exceed + 1.0) / (perm_iters + 1.0), 4)

        return (w_raw, b_raw, test_auroc, perm_p)

    # ========================================================================
    # 2. STAGE 1: UNSTEERED APPRAISAL RESIDUAL CAPTURE & ENDOGENOUS LAMBDA
    # ========================================================================

    @contextmanager
    def capture_stage1_cot_residuals(self, model: Any) -> Iterator[None]:
        """
        Read-only Stage 1 forward hook at `target_layer`.
        Injects ZERO steering during Stage 1 CoT generation and records the
        residual stream vectors strictly over generated CoT tokens.
        """
        self._captured_cot_residuals.clear()
        self._fallback_prefill_last_residual = None

        if model is None or self.artifacts is None:
            yield
            return

        base_transformer = getattr(model, "model", model)
        layers = getattr(base_transformer, "layers", None)
        if layers is None:
            yield
            return

        layer_idx = self.resolve_effective_layer(
            self.artifacts.target_layer, len(layers)
        )
        prefill_seen = {"done": False}

        def _stage1_read_hook(mod: Any, inp: Any, out: Any) -> None:
            h = out[0] if isinstance(out, tuple) else out
            if not isinstance(h, torch.Tensor) or h.dim() != 3:
                return
            seq_len = int(h.size(1))
            if seq_len > 1 and not prefill_seen["done"]:
                prefill_seen["done"] = True
                # Backup only if CoT generates <= 1 token
                self._fallback_prefill_last_residual = (
                    h[0, -1, :].detach().to("cpu", dtype=torch.float32)
                )
            else:
                # Generated Stage 1 CoT decode token (seq_len == 1)
                self._captured_cot_residuals.append(
                    h[0, -1, :].detach().to("cpu", dtype=torch.float32)
                )

        handle = layers[layer_idx].register_forward_hook(_stage1_read_hook)
        try:
            yield
        finally:
            handle.remove()

    def compute_endogenous_lambda(self) -> ProbeGateMeasurement:
        """
        Freezes Stage 1 CoT residuals, runs the Linear Probe:
            raw = w^T * h_bar_CoT + b
            lambda = sigmoid((raw - theta) / T)    (or relu_clipped in [0, 1])
        Returns a complete `ProbeGateMeasurement` audit record.
        """
        if self.artifacts is None or not self.config.enabled or self.config.mode == RepEMode.OFF:
            return ProbeGateMeasurement(
                layer_idx=self.config.target_layer,
                cot_tokens_inspected=len(self._captured_cot_residuals),
                raw_probe_score=0.0,
                endogenous_lambda=0.0,
                steering_gamma=0.0,
                injected_vector_l2_norm=0.0,
                relative_residual_norm_ratio=0.0,
                repe_mode=RepEMode.OFF,
            )

        if self._captured_cot_residuals:
            if self.config.probe_pooling == "last_cot_token":
                h_pooled = self._captured_cot_residuals[-1]
            else:
                h_pooled = torch.stack(self._captured_cot_residuals, dim=0).mean(dim=0)
            n_inspected = len(self._captured_cot_residuals)
        elif self._fallback_prefill_last_residual is not None:
            h_pooled = self._fallback_prefill_last_residual
            n_inspected = 1
        else:
            return ProbeGateMeasurement(
                layer_idx=self.artifacts.target_layer,
                cot_tokens_inspected=0,
                raw_probe_score=0.0,
                endogenous_lambda=0.0,
                steering_gamma=self.config.steering_strength_gamma,
                injected_vector_l2_norm=0.0,
                relative_residual_norm_ratio=0.0,
                repe_mode=self.config.mode,
            )

        h_clean = torch.nan_to_num(h_pooled.float(), nan=0.0)
        w = self.artifacts.probe_weights.float()
        b = float(self.artifacts.probe_bias)

        raw_score = float(torch.dot(w, h_clean).item() + b)
        shifted = (raw_score - float(self.config.probe_threshold_theta)) / max(
            1e-4, float(self.config.probe_temperature)
        )

        if self.config.gate_activation == "relu_clipped":
            lam = max(0.0, min(1.0, shifted))
        else:
            # Stable sigmoid activation
            clamped = max(-30.0, min(30.0, shifted))
            lam = 1.0 / (1.0 + math.exp(-clamped))

        self._recent_lambda_history.append(lam)
        if len(self._recent_lambda_history) > 128:
            self._recent_lambda_history.pop(0)

        # Control Condition: SHUFFLED_LAMBDA permutes lambda across recent turns
        effective_lam = lam
        if (
            self.config.mode == RepEMode.SHUFFLED_LAMBDA
            and len(self._recent_lambda_history) >= 2
        ):
            rand_idx = int(self.rng.integers(0, len(self._recent_lambda_history)))
            effective_lam = self._recent_lambda_history[rand_idx]

        return ProbeGateMeasurement(
            layer_idx=self.artifacts.target_layer,
            cot_tokens_inspected=n_inspected,
            raw_probe_score=round(raw_score, 6),
            endogenous_lambda=round(float(effective_lam), 6),
            steering_gamma=float(self.config.steering_strength_gamma),
            injected_vector_l2_norm=0.0,
            relative_residual_norm_ratio=0.0,
            repe_mode=self.config.mode,
        )

    # ========================================================================
    # 3. STAGE 2: CONTROLLED ACTION GENERATION VECTOR INJECTION
    # ========================================================================

    @contextmanager
    def inject_stage2_action_steering(
        self,
        model: Any,
        probe_measurement: ProbeGateMeasurement,
    ) -> Iterator[None]:
        """
        Freezes the probe and injects:
            h' = h + lambda * gamma * v_hat
        STRICTLY on Stage 2 Action generation token positions at `target_layer`,
        never perturbing Stage 1 prompt/CoT prefix positions.
        Handles `bnb.nn.Linear4bit` device/dtype alignment and caps relative norm
        to `config.max_relative_norm_ratio`.
        """
        self._last_injected_l2_norm = 0.0
        self._last_relative_norm_ratio = 0.0

        lam = float(probe_measurement.endogenous_lambda)
        gamma = float(self.config.steering_strength_gamma)
        active_mode = self.config.mode

        if (
            model is None
            or self.artifacts is None
            or not self.config.enabled
            or active_mode == RepEMode.OFF
            or abs(lam * gamma) <= 1e-8
        ):
            yield
            return

        base_transformer = getattr(model, "model", model)
        layers = getattr(base_transformer, "layers", None)
        if layers is None:
            yield
            return

        layer_idx = self.resolve_effective_layer(
            self.artifacts.target_layer, len(layers)
        )
        base_vec_cpu = (
            self.artifacts.placebo_vector
            if active_mode == RepEMode.PLACEBO_RANDOM_VECTOR
            else self.artifacts.steering_vector
        )
        max_rel_ratio = float(self.config.max_relative_norm_ratio)
        raw_mag = lam * gamma

        def _stage2_inject_hook(mod: Any, inp: Any, out: Any) -> Any:
            is_tuple = isinstance(out, tuple)
            h = out[0] if is_tuple else out
            if not isinstance(h, torch.Tensor) or h.dim() != 3:
                return out

            v_dev = base_vec_cpu.to(device=h.device, dtype=h.dtype)
            if v_dev.size(-1) != h.size(-1):
                return out

            seq_len = int(h.size(1))
            # On Stage 2 Prefill (seq_len > 1), inject ONLY at final position predicting
            # the first JSON token (leaving Stage 1 prompt/CoT prefix 100% untouched).
            # On Stage 2 Decode steps (seq_len == 1), inject into the active action token.
            if seq_len > 1:
                target_slice = h[:, -1:, :]
            else:
                target_slice = h

            h_norm = target_slice.detach().float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
            mean_h_norm = float(h_norm.mean().item())
            max_allowed_mag = max_rel_ratio * mean_h_norm

            capped_mag = math.copysign(min(abs(raw_mag), max_allowed_mag), raw_mag)
            delta = (v_dev * capped_mag).view(1, 1, -1)

            h_mod = h.clone()
            if seq_len > 1:
                h_mod[:, -1:, :] = h_mod[:, -1:, :] + delta
            else:
                h_mod = h_mod + delta

            self._last_injected_l2_norm = round(abs(capped_mag), 6)
            self._last_relative_norm_ratio = round(
                abs(capped_mag) / max(mean_h_norm, 1e-6), 6
            )
            return (h_mod,) + out[1:] if is_tuple else h_mod

        handle = layers[layer_idx].register_forward_hook(_stage2_inject_hook)
        try:
            yield
        finally:
            handle.remove()
            probe_measurement.injected_vector_l2_norm = self._last_injected_l2_norm
            probe_measurement.relative_residual_norm_ratio = (
                self._last_relative_norm_ratio
            )