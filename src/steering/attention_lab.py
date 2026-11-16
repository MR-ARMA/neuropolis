"""
src/steering/attention_lab.py
MODULE 3 (M3): Mechanistic Attention Steering Lab (`MasterEquationHookEngine`).
Implements TRUE CAUSAL IN-MODEL STEERING:
  1. `MechanisticCalibrator`: Contrastive calibration extracting Gram-Schmidt
     orthonormalized Value-space concept vectors `u_hat_trust` and `u_hat_threat`.
  2. Pre-Softmax Causal Attention Injection (`_make_attn_pre_hook` on `self_attn`):
     Injects the 1D Broadcastable Key-Bias Vector `Delta_B(s)` (combining Term 1 `B_env(E_t)`
     and Term 8 `s_sink(D_t)`) directly into the 4D `attention_mask` entering `Qwen2Attention` /
     `LlamaAttention` BEFORE `torch.softmax`, safely handling both Prefill (`q_len == S`)
     and Autoregressive Decode (`q_len == 1, kv_len == S + step`).
  3. Dual-Space QK-V Value Rotation (`_make_v_hook` on `self_attn.v_proj`):
     Spherical norm-preserving rotation of `V_s` on `SPAN_INCOMING_MSG` tokens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.core.contracts import (
    SPAN_DRIVES_AND_FINANCE,
    SPAN_INCOMING_MSG_PREFIX,
    SPAN_PEER_MEMORY_PREFIX,
    SPAN_SINK,
    EndogenousStatePacket,
    MechanisticSteeringSpec,
    SteeringMode,
    SteeringTelemetryPacket,
    TokenSpanMap,
)


CONTRASTIVE_TRUST_THREAT_PAIRS: List[Tuple[str, str]] = [
    (
        "My partner honored every contract fairly, transparent pricing, mutual cooperation and verified trust.",
        "My partner betrayed the deal, lied about inventory, predatory price gouging and hostile threat.",
    ),
    (
        "Reliable long-term ally sharing honest market supply information and equitable profit split.",
        "Deceptive opportunist attempting to defraud my wallet, exploit scarcity, and ambush my business.",
    ),
    (
        "Full solvency, fair bilateral exchange, warm goodwill, and dependable delivery.",
        "Malicious default, hidden insolvency trap, aggressive extortion, and imminent financial danger.",
    ),
    (
        "Mutually beneficial trade agreement built on integrity, reciprocity, and calm confidence.",
        "Hostile manipulation, severe distrust, broken promise, and alarming existential threat.",
    ),
]


class CalibrationSubspaceBundle(nn.Module):
    """Stores orthonormalized Value-space concept vectors for target layers."""

    def __init__(
        self,
        target_layer_indices: List[int],
        u_trust_by_layer: Dict[int, torch.Tensor],
        u_threat_by_layer: Dict[int, torch.Tensor],
        affect_proj_matrix: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.target_layer_indices = target_layer_indices
        self.u_trust_by_layer = u_trust_by_layer
        self.u_threat_by_layer = u_threat_by_layer
        self.affect_proj_matrix = affect_proj_matrix


class MechanisticCalibrator:
    """
    Extracts and Gram-Schmidt orthonormalizes `u_hat_trust` and `u_hat_threat`
    inside the Value projection (`v_proj`) subspace of the white-box transformer.
    """

    def __init__(self, spec: MechanisticSteeringSpec) -> None:
        self.spec = spec

    @staticmethod
    def get_decoder_layers(model: PreTrainedModel) -> nn.ModuleList:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        raise AttributeError("Unsupported transformer architecture: cannot locate decoder layers.")

    @staticmethod
    def gram_schmidt_orthonormalize(
        v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-7
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Orthonormalizes v1 and v2 in R^{D_v} so that:
          ||u_1||_2 = 1,  ||u_2||_2 = 1,  and  <u_1, u_2> = 0.
        """
        v1_f = v1.float()
        v2_f = v2.float()
        u1 = v1_f / torch.clamp(torch.norm(v1_f, p=2), min=eps)
        v2_orth = v2_f - torch.dot(v2_f, u1) * u1
        v2_orth = v2_orth - torch.dot(v2_orth, u1) * u1
        if torch.norm(v2_orth, p=2) < 1e-5:
            alt = torch.roll(u1, shifts=1, dims=0)
            v2_orth = alt - torch.dot(alt, u1) * u1
            v2_orth = v2_orth - torch.dot(v2_orth, u1) * u1
        u2 = v2_orth / torch.clamp(torch.norm(v2_orth, p=2), min=eps)
        return u1, u2

    def calibrate_or_load(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        cache_path: Path,
        device: torch.device,
    ) -> CalibrationSubspaceBundle:
        layers = self.get_decoder_layers(model)
        num_layers = len(layers)
        center_layer = max(0, min(num_layers - 1, int(num_layers * self.spec.target_layer_fraction)))
        target_indices = sorted(
            list({max(0, center_layer - 1), center_layer, min(num_layers - 1, center_layer + 1)})
        )

        if cache_path.exists():
            try:
                saved = torch.load(cache_path, map_location=device, weights_only=True)
                return CalibrationSubspaceBundle(
                    target_layer_indices=saved["target_layer_indices"],
                    u_trust_by_layer={int(k): v.to(device) for k, v in saved["u_trust"].items()},
                    u_threat_by_layer={int(k): v.to(device) for k, v in saved["u_threat"].items()},
                )
            except Exception:
                pass

        captured_v: Dict[int, List[torch.Tensor]] = {idx: [] for idx in target_indices}
        hooks = []

        def _make_capture_hook(layer_idx: int) -> Callable:
            def _hook(module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
                mean_tok = output.detach().float().mean(dim=1).squeeze(0).cpu()
                captured_v[layer_idx].append(mean_tok)

            return _hook

        for idx in target_indices:
            v_proj = layers[idx].self_attn.v_proj
            hooks.append(v_proj.register_forward_hook(_make_capture_hook(idx)))

        try:
            with torch.no_grad():
                for trust_text, threat_text in CONTRASTIVE_TRUST_THREAT_PAIRS:
                    for text in (trust_text, threat_text):
                        enc = tokenizer(
                            text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=64,
                        ).to(device)
                        model(**enc)
        finally:
            for h in hooks:
                h.remove()

        u_trust_map: Dict[int, torch.Tensor] = {}
        u_threat_map: Dict[int, torch.Tensor] = {}

        for idx in target_indices:
            vecs = captured_v[idx]
            trust_vecs = torch.stack(vecs[0::2], dim=0)
            threat_vecs = torch.stack(vecs[1::2], dim=0)
            raw_trust = trust_vecs.mean(dim=0)
            raw_threat = threat_vecs.mean(dim=0) - trust_vecs.mean(dim=0)
            u_tr, u_th = self.gram_schmidt_orthonormalize(raw_trust, raw_threat)
            u_trust_map[idx] = u_tr.to(device)
            u_threat_map[idx] = u_th.to(device)

        try:
            torch.save(
                {
                    "target_layer_indices": target_indices,
                    "u_trust": u_trust_map,
                    "u_threat": u_threat_map,
                },
                cache_path,
            )
        except Exception:
            pass

        return CalibrationSubspaceBundle(
            target_layer_indices=target_indices,
            u_trust_by_layer=u_trust_map,
            u_threat_by_layer=u_threat_map,
        )


class AttentionSteeringLab:
    """
    Executes the Unified Cognitive Attention Master Equation v2.0 INSIDE the Transformer
    forward pass via:
      1. `self_attn` Pre-Hooks (`_make_attn_pre_hook`): Injects `B_env(E_t)` and `s_sink(D_t)`
         into the 4D additive `attention_mask` prior to `torch.softmax`.
      2. `self_attn.v_proj` Forward Hooks (`_make_v_hook`): Rotates Value vectors on
         `SPAN_INCOMING_MSG` tokens using orthonormal `u_hat_trust` and `u_hat_threat`.
    """

    def __init__(
        self,
        spec: MechanisticSteeringSpec,
        calibration: Optional[CalibrationSubspaceBundle] = None,
    ) -> None:
        self.spec = spec
        self.calibration = calibration
        self._active_hooks: List[Any] = []
        self._last_v_norm_ratios: List[float] = []

    def build_key_bias_vector(
        self,
        span_map: TokenSpanMap,
        state: EndogenousStatePacket,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Builds the 1D Broadcastable Key-Bias Vector `Delta_B(s)` in R^{seq_len}:
          - Term 1a: `+ omega_fin_stress * Stress_fin` on `SPAN_DRIVES_AND_FINANCE`
          - Term 1b: `+ omega_peer_salience * PeerDistrust` on `SPAN_PEER_MEMORY`
          - Bounded tanh ceiling `Z_max * tanh(bias / Z_max)`
          - Term 8: `+ kappa_dissonance_sink * D_t` on `<bos>` (`s = 0`) and attenuation on `SPAN_INCOMING_MSG`.
        """
        bias_1d = torch.zeros(seq_len, device=device, dtype=torch.float32)
        if not self.spec.enabled or self.spec.mode == SteeringMode.NONE:
            return bias_1d.to(dtype)

        fin_indices = [
            i for i in span_map.get_token_indices_for_span(SPAN_DRIVES_AND_FINANCE) if i < seq_len
        ]
        peer_indices = [
            i for i in span_map.get_token_indices_for_span(SPAN_PEER_MEMORY_PREFIX) if i < seq_len
        ]

        if fin_indices and state.financial_stress > 0.0:
            bias_1d[fin_indices] += float(self.spec.omega_fin_stress * state.financial_stress)

        if peer_indices and state.peer_distrust_salience > 0.0:
            bias_1d[peer_indices] += float(
                self.spec.omega_peer_salience * state.peer_distrust_salience
            )

        z_max = max(0.5, float(self.spec.logit_clip_z_max))
        bias_1d = z_max * torch.tanh(bias_1d / z_max)

        if seq_len > 0 and state.dissonance_score_d > 0.0:
            bias_1d[0] += float(self.spec.kappa_dissonance_sink * state.dissonance_score_d)
            msg_indices = [
                i
                for i in span_map.get_token_indices_for_span(SPAN_INCOMING_MSG_PREFIX)
                if 0 < i < seq_len
            ]
            if msg_indices:
                bias_1d[msg_indices] -= float(
                    0.65 * self.spec.kappa_dissonance_sink * state.dissonance_score_d
                )

        return bias_1d.to(dtype)

    @staticmethod
    def apply_variance_preserving_logit_shift(
        raw_qk_logits: torch.Tensor,
        key_bias_1d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pre-Beta Variance-Preserving Logit Renormalization on non-sink tokens (`s >= 1`),
        preventing additive bias terms from artificially inflating Softmax sharpness.
        """
        s_k = raw_qk_logits.shape[-1]
        bias = key_bias_1d[:s_k].to(raw_qk_logits.device, dtype=raw_qk_logits.dtype)

        if s_k <= 2:
            return raw_qk_logits + bias

        z0_non_sink = raw_qk_logits[..., 1:]
        b_non_sink = bias[1:]
        z_shifted = z0_non_sink + b_non_sink

        mean_0 = z0_non_sink.mean(dim=-1, keepdim=True)
        std_0 = torch.clamp(z0_non_sink.std(dim=-1, keepdim=True), min=1e-5)

        mean_raw = z_shifted.mean(dim=-1, keepdim=True)
        std_raw = torch.clamp(z_shifted.std(dim=-1, keepdim=True), min=1e-5)

        scale = std_0 / torch.maximum(std_raw, std_0)
        z_renorm_non_sink = mean_0 + scale * (z_shifted - mean_raw)

        z_sink = raw_qk_logits[..., :1] + bias[:1]
        return torch.cat([z_sink, z_renorm_non_sink], dim=-1)

    @staticmethod
    def apply_dual_space_value_projection(
        v_tensor: torch.Tensor,
        msg_token_indices: List[int],
        u_hat_trust: torch.Tensor,
        u_hat_threat: torch.Tensor,
        cynicism_gamma: float,
        anxiety_eta: float,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, float]:
        """
        Spherical norm-preserving Value-space projection on `SPAN_INCOMING_MSG` tokens:
          V_raw_s = V_s - gamma * <V_s, u_hat_trust> u_hat_trust + eta * ||V_s||_2 u_hat_threat
          V_tilde_s = ||V_s||_2 * (V_raw_s / max(||V_raw_s||_2, eps))
        """
        if not msg_token_indices or (cynicism_gamma <= 0.0 and anxiety_eta <= 0.0):
            return v_tensor, 1.0

        s_len = v_tensor.shape[1]
        valid_idx = [i for i in msg_token_indices if 0 <= i < s_len]
        if not valid_idx:
            return v_tensor, 1.0

        v_out = v_tensor.clone()
        v_slice = v_out[:, valid_idx, :].float()

        u_tr = u_hat_trust.float().to(v_tensor.device)
        u_th = u_hat_threat.float().to(v_tensor.device)

        orig_norm = torch.norm(v_slice, p=2, dim=-1, keepdim=True)

        trust_dot = torch.matmul(v_slice, u_tr).unsqueeze(-1)
        v_mod = v_slice - float(cynicism_gamma) * trust_dot * u_tr.view(1, 1, -1)

        if anxiety_eta > 0.0:
            v_mod = v_mod + float(anxiety_eta) * orig_norm * u_th.view(1, 1, -1)

        mod_norm = torch.clamp(torch.norm(v_mod, p=2, dim=-1, keepdim=True), min=eps)
        v_spherical = orig_norm * (v_mod / mod_norm)

        final_norm = torch.norm(v_spherical, p=2, dim=-1, keepdim=True)
        norm_ratio = float((final_norm / torch.clamp(orig_norm, min=eps)).mean().item())

        v_out[:, valid_idx, :] = v_spherical.to(v_tensor.dtype)
        return v_out, norm_ratio

    def attach_steering_hooks(
        self,
        model: PreTrainedModel,
        span_map: TokenSpanMap,
        state: EndogenousStatePacket,
    ) -> None:
        """
        Attaches BOTH:
          1. Pre-softmax `self_attn` hooks injecting `B_env(E_t)` and `s_sink(D_t)` into the
             4D additive `attention_mask` across the middle-upper semantic layers.
          2. Post-projection `self_attn.v_proj` hooks applying Dual-Space QK-V Value rotation
             across the calibrated layers.
        Handles both Prefill (`q_len == S`) and Autoregressive Decode (`q_len == 1, kv_len == S + step`).
        """
        self.remove_hooks()
        self._last_v_norm_ratios.clear()

        if not self.spec.enabled or self.spec.mode == SteeringMode.NONE:
            return

        layers = MechanisticCalibrator.get_decoder_layers(model)
        num_layers = len(layers)
        center_layer = max(0, min(num_layers - 1, int(num_layers * self.spec.target_layer_fraction)))

        if self.calibration is not None:
            v_target_layers = self.calibration.target_layer_indices
        else:
            v_target_layers = sorted(
                list({max(0, center_layer - 1), center_layer, min(num_layers - 1, center_layer + 1)})
            )

        # Apply pre-softmax attention key-bias across the semantic middle-upper band
        band_half_width = max(2, num_layers // 6)
        attn_target_layers = list(
            range(
                max(0, center_layer - band_half_width),
                min(num_layers, center_layer + band_half_width + 1),
            )
        )
        for idx in v_target_layers:
            if idx not in attn_target_layers:
                attn_target_layers.append(idx)
        attn_target_layers.sort()

        prompt_seq_len = max(1, span_map.total_prompt_tokens)
        key_bias_cpu = self.build_key_bias_vector(
            span_map=span_map,
            state=state,
            seq_len=prompt_seq_len,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        has_nonzero_key_bias = bool(torch.any(torch.abs(key_bias_cpu) > 1e-6).item())

        # 1. Register Pre-Softmax Key-Bias Hook on `layers[idx].self_attn`
        if has_nonzero_key_bias:
            for layer_idx in attn_target_layers:
                attn_module = layers[layer_idx].self_attn

                def _make_attn_pre_hook(bias_vec_cpu: torch.Tensor, p_len: int) -> Callable:
                    def _attn_pre_hook(
                        module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
                    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
                        hidden_states = kwargs.get(
                            "hidden_states", args[0] if len(args) > 0 else None
                        )
                        if hidden_states is None:
                            return args, kwargs

                        bsz, q_len, _ = hidden_states.shape
                        dev = hidden_states.device
                        dt = hidden_states.dtype
                        bias_dev = bias_vec_cpu.to(device=dev, dtype=dt)

                        existing_mask = kwargs.get("attention_mask", None)

                        if existing_mask is not None:
                            # existing_mask shape: [B, 1, q_len, kv_len]
                            kv_len = existing_mask.shape[-1]
                            full_bias = torch.zeros(kv_len, device=dev, dtype=dt)
                            copy_len = min(p_len, kv_len)
                            full_bias[:copy_len] = bias_dev[:copy_len]

                            bias_4d = full_bias.view(1, 1, 1, kv_len)
                            valid_pos = existing_mask > -1e3
                            steered_mask = torch.where(
                                valid_pos, existing_mask + bias_4d, existing_mask
                            )
                            kwargs["attention_mask"] = steered_mask
                        else:
                            if q_len > 1:
                                # Prefill pass without pre-built 4D mask: build causal mask + key bias
                                kv_len = q_len
                                full_bias = torch.zeros(kv_len, device=dev, dtype=dt)
                                copy_len = min(p_len, kv_len)
                                full_bias[:copy_len] = bias_dev[:copy_len]

                                min_val = torch.finfo(dt).min
                                causal_bool = torch.tril(
                                    torch.ones((q_len, kv_len), device=dev, dtype=torch.bool)
                                ).view(1, 1, q_len, kv_len)
                                steered_mask = torch.where(
                                    causal_bool,
                                    full_bias.view(1, 1, 1, kv_len).expand(bsz, 1, q_len, kv_len),
                                    torch.full((bsz, 1, q_len, kv_len), min_val, device=dev, dtype=dt),
                                )
                                kwargs["attention_mask"] = steered_mask
                            else:
                                # Single-token decode (`q_len == 1`): eager_attention_forward slices
                                # `attention_mask[:, :, :, :key_states.shape[-2]]`.
                                max_kv_buffer = p_len + 512
                                full_bias = torch.zeros(max_kv_buffer, device=dev, dtype=dt)
                                full_bias[:p_len] = bias_dev[:p_len]
                                kwargs["attention_mask"] = full_bias.view(
                                    1, 1, 1, max_kv_buffer
                                ).expand(bsz, 1, 1, max_kv_buffer)

                        return args, kwargs

                    return _attn_pre_hook

                handle = attn_module.register_forward_pre_hook(
                    _make_attn_pre_hook(key_bias_cpu, prompt_seq_len),
                    with_kwargs=True,
                )
                self._active_hooks.append(handle)

        # 2. Register Dual-Space QK-V Value Rotation Hook on `layers[idx].self_attn.v_proj`
        msg_indices = span_map.get_token_indices_for_span(SPAN_INCOMING_MSG_PREFIX)
        if self.calibration is not None and msg_indices:
            for layer_idx in v_target_layers:
                u_tr = self.calibration.u_trust_by_layer[layer_idx]
                u_th = self.calibration.u_threat_by_layer[layer_idx]
                v_proj_mod = layers[layer_idx].self_attn.v_proj

                def _make_v_hook(
                    u_trust_vec: torch.Tensor, u_threat_vec: torch.Tensor
                ) -> Callable:
                    def _v_hook(
                        module: nn.Module, inputs: Any, output: torch.Tensor
                    ) -> torch.Tensor:
                        if output.shape[1] <= 1:
                            return output
                        v_steered, ratio = self.apply_dual_space_value_projection(
                            v_tensor=output,
                            msg_token_indices=msg_indices,
                            u_hat_trust=u_trust_vec,
                            u_hat_threat=u_threat_vec,
                            cynicism_gamma=self.spec.gamma_cynicism_scale * state.cynicism_gamma,
                            anxiety_eta=self.spec.eta_anxiety_scale * state.anxiety_eta,
                        )
                        self._last_v_norm_ratios.append(ratio)
                        return v_steered

                    return _v_hook

                handle = v_proj_mod.register_forward_hook(_make_v_hook(u_tr, u_th))
                self._active_hooks.append(handle)

    # Alias preserved for backwards compatibility
    def attach_prefill_value_hooks(
        self,
        model: PreTrainedModel,
        span_map: TokenSpanMap,
        state: EndogenousStatePacket,
    ) -> None:
        self.attach_steering_hooks(model=model, span_map=span_map, state=state)

    def remove_hooks(self) -> None:
        for h in self._active_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._active_hooks.clear()

    def extract_in_model_attention_telemetry(
        self,
        in_model_attn_probs: torch.Tensor,
        span_map: TokenSpanMap,
        state: EndogenousStatePacket,
        tick: int,
    ) -> SteeringTelemetryPacket:
        """
        Extracts exact post-softmax attention mass directly from the real `outputs.attentions`
        tensor emitted inside `Qwen2Attention` during the forward pass.
        """
        s_k = in_model_attn_probs.shape[-1]
        mean_probs = in_model_attn_probs.detach().float().mean(dim=(0, 1, 2))  # [S_k]

        fin_idx = [
            i for i in span_map.get_token_indices_for_span(SPAN_DRIVES_AND_FINANCE) if i < s_k
        ]
        peer_idx = [
            i for i in span_map.get_token_indices_for_span(SPAN_PEER_MEMORY_PREFIX) if i < s_k
        ]
        msg_idx = [
            i for i in span_map.get_token_indices_for_span(SPAN_INCOMING_MSG_PREFIX) if i < s_k
        ]

        mass_fin = float(mean_probs[fin_idx].sum().item()) if fin_idx else 0.0
        mass_peer = float(mean_probs[peer_idx].sum().item()) if peer_idx else 0.0
        mass_msg = float(mean_probs[msg_idx].sum().item()) if msg_idx else 0.0
        mass_sink = float(mean_probs[0].item()) if s_k > 0 else 0.0
        v_ratio = (
            sum(self._last_v_norm_ratios) / len(self._last_v_norm_ratios)
            if self._last_v_norm_ratios
            else 1.0
        )

        return SteeringTelemetryPacket(
            agent_id=state.agent_id,
            tick=tick,
            steering_mode=self.spec.mode if self.spec.enabled else SteeringMode.NONE,
            attn_mass_drives_finance=round(mass_fin, 5),
            attn_mass_peer_memory=round(mass_peer, 5),
            attn_mass_incoming_msg=round(mass_msg, 5),
            attn_mass_sink_bos=round(mass_sink, 5),
            value_projection_norm_ratio=round(v_ratio, 6),
        )

    def compute_steered_attention_telemetry(
        self,
        unsteered_qk_logits: torch.Tensor,
        span_map: TokenSpanMap,
        state: EndogenousStatePacket,
        tick: int,
    ) -> Tuple[torch.Tensor, SteeringTelemetryPacket]:
        """
        Standalone mathematical verification helper used by synthetic tensor unit tests.
        """
        s_k = unsteered_qk_logits.shape[-1]
        key_bias = self.build_key_bias_vector(
            span_map=span_map,
            state=state,
            seq_len=s_k,
            device=unsteered_qk_logits.device,
            dtype=unsteered_qk_logits.dtype,
        )
        steered_logits = self.apply_variance_preserving_logit_shift(
            raw_qk_logits=unsteered_qk_logits,
            key_bias_1d=key_bias,
        )
        attn_probs = torch.softmax(steered_logits, dim=-1)
        telemetry = self.extract_in_model_attention_telemetry(
            in_model_attn_probs=attn_probs,
            span_map=span_map,
            state=state,
            tick=tick,
        )
        return attn_probs, telemetry

    # ==========================================================================
    # POST-MVP MASTER EQUATION STUBS (Preserved with Exact Equations)
    # ==========================================================================
    def apply_construal_rope_warping(
        self,
        key_positions: torch.Tensor,
        anchor_t0: int,
        closeness_map: Dict[str, float],
    ) -> torch.Tensor:
        """
        # TODO(post-mvp): [Master Equation Term 4 - Construal-Level Static Key-Coordinate RoPE Warping]
        Equation: s_tilde(i, s) = t_0 - (t_0 - s) * exp( -lambda_rope * S_closeness(i, s) )
        """
        return key_positions

    def apply_sys1_sys2_competitive_head_gating(
        self,
        head_outputs: torch.Tensor,
        cognitive_load_l: float,
    ) -> torch.Tensor:
        """
        # TODO(post-mvp): [Master Equation Term 5 - System 1 vs. System 2 Competitive Head Gating]
        """
        return head_outputs

    def inject_mirror_neuron_kv_prefix(
        self,
        past_key_values: Any,
        speaker_latent_vec: Optional[torch.Tensor],
    ) -> Any:
        """
        # TODO(post-mvp): [Master Equation Term 6 - Subconscious Emotional Contagion KV-Prefix]
        """
        return past_key_values

    def apply_yerkes_dodson_and_hebbian_trace(
        self,
        logits: torch.Tensor,
        arousal_a: float,
        fatigue_f: float,
        hebbian_trace: Dict[str, float],
    ) -> torch.Tensor:
        """
        # TODO(post-mvp): [Master Equation Terms 7 & 9 - Thermodynamic Softmax & Hebbian Fast-Weights]
        """
        return logits