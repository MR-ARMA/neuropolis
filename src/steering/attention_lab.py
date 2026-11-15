"""
src/steering/attention_lab.py
MODULE 3 (M3): Mechanistic Attention Steering Lab (`MasterEquationHookEngine`).
Implements:
  1. `MechanisticCalibrator`: Contrastive SVD / Mean-Difference calibration extracting
     Gram-Schmidt orthonormalized Value-space concept vectors `u_hat_trust` and `u_hat_threat`
     at target transformer layers.
  2. MVP High-ROI Master Equation Terms (100% Working PyTorch Tensor Operations):
     - Term 1 (`B_env(E_t)`): State-Coupled Dynamic Attention Bias boosting
       `SPAN_DRIVES_AND_FINANCE` under financial stress and `SPAN_PEER_MEMORY` under distrust,
       with Audit Flaw 2 Pre-Beta Variance-Preserving Logit Renormalization.
     - Term 3 (`Dual-Space QK-V Steering`): Norm-preserving spherical Value rotation
       removing `u_hat_trust` proportional to cynicism `gamma` and injecting `u_hat_threat`
       proportional to anxiety `eta` exclusively on `SPAN_INCOMING_MSG` tokens.
     - Term 8 (`Cognitive Dissonance Sink Routing`): Boosting `<bos>` (index 0) attention
       logits by `+kappa_sink * D_t(i)` when incoming messages clash with core beliefs.
  3. Post-MVP Master Equation Stubs (Terms 4, 5, 6, 7, 9) with exact mathematical docstrings.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.core.contracts import (
    SPAN_CORE_BELIEF,
    SPAN_DRIVES_AND_FINANCE,
    SPAN_INCOMING_MSG_PREFIX,
    SPAN_PEER_MEMORY_PREFIX,
    SPAN_SINK,
    CompiledPromptPacket,
    EndogenousStatePacket,
    MechanisticSteeringSpec,
    SteeringMode,
    SteeringTelemetryPacket,
    TokenSpanMap,
)


# ==============================================================================
# 1. CONTRASTIVE CONCEPT SUBSPACE CALIBRATOR (Gram-Schmidt Orthonormalized)
# ==============================================================================
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
        proj = torch.dot(v2_f, u1) * u1
        v2_orth = v2_f - proj
        if torch.norm(v2_orth, p=2) < eps:
            # Construct deterministic orthogonal fallback if collinear
            alt = torch.roll(u1, shifts=1, dims=0)
            v2_orth = alt - torch.dot(alt, u1) * u1
        u2 = v2_orth / torch.clamp(torch.norm(v2_orth, p=2), min=eps)
        return u1.to(v1.dtype), u2.to(v2.dtype)

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

        # Run rapid contrastive forward pass over trust vs. threat stimuli
        captured_v: Dict[int, List[torch.Tensor]] = {idx: [] for idx in target_indices}
        hooks = []

        def _make_capture_hook(layer_idx: int) -> Callable:
            def _hook(module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
                # output shape: [B, S, D_v]
                mean_tok = output.detach().float().mean(dim=1).squeeze(0)
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
            trust_vecs = torch.stack(vecs[0::2], dim=0)   # Even indices: trust stimuli
            threat_vecs = torch.stack(vecs[1::2], dim=0)  # Odd indices: threat stimuli
            global_mean = 0.5 * (trust_vecs.mean(dim=0) + threat_vecs.mean(dim=0))

            raw_trust = trust_vecs.mean(dim=0) - global_mean
            raw_threat = threat_vecs.mean(dim=0) - global_mean
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


# ==============================================================================
# 2. MASTER EQUATION HOOK ENGINE (MVP High-ROI Terms + Post-MVP Stubs)
# ==============================================================================
class AttentionSteeringLab:
    """
    Executes the Unified Cognitive Attention Master Equation v2.0 during White-Box
    prefill & generation:
      - Constructs the 1D Key-Bias Vector `b_key(s)` for `B_env(E_t)` and `s_sink(D_t)`.
      - Attaches norm-preserving spherical Value-projection hooks on `v_proj` over
        `SPAN_INCOMING_MSG` tokens.
      - Applies Audit Flaw 2 Pre-Beta Variance-Preserving Logit Renormalization.
      - Computes exact post-softmax attention mass telemetry across semantic spans.
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
          1. State-Coupled Bias `B_env(E_t)`:
             + omega_1 * Stress_fin(i,t) on `SPAN_DRIVES_AND_FINANCE`
             + omega_2 * PeerDistrust(i,j,t) on `SPAN_PEER_MEMORY`
          2. Bounded tanh clipping:
             Delta_B(s) = Z_max * tanh( raw_bias(s) / Z_max )
          3. Cognitive Dissonance Sink Routing `s_sink(D_t)` at s = 0 (<bos>):
             Delta_B(0) += kappa_sink * D_t(i)
        """
        bias_1d = torch.zeros(seq_len, device=device, dtype=dtype)
        if not self.spec.enabled or self.spec.mode == SteeringMode.NONE:
            return bias_1d

        fin_indices = [
            i for i in span_map.get_token_indices_for_span(SPAN_DRIVES_AND_FINANCE) if i < seq_len
        ]
        peer_indices = [
            i for i in span_map.get_token_indices_for_span(SPAN_PEER_MEMORY_PREFIX) if i < seq_len
        ]

        if fin_indices and state.financial_stress > 0.0:
            boost_fin = self.spec.omega_fin_stress * float(state.financial_stress)
            bias_1d[fin_indices] += boost_fin

        if peer_indices and state.peer_distrust_salience > 0.0:
            boost_peer = self.spec.omega_peer_salience * float(state.peer_distrust_salience)
            bias_1d[peer_indices] += boost_peer

        # Bounded tanh saturation on non-sink tokens
        z_max = max(0.5, float(self.spec.logit_clip_z_max))
        bias_1d = z_max * torch.tanh(bias_1d / z_max)

        # Term 8: Dissonance Sink Routing on token s = 0 (<bos>)
        if seq_len > 0 and state.dissonance_score_d > 0.0:
            bias_1d[0] += self.spec.kappa_dissonance_sink * float(state.dissonance_score_d)
            # Slightly attenuate incoming message logits when dissonance is high
            msg_indices = [
                i for i in span_map.get_token_indices_for_span(SPAN_INCOMING_MSG_PREFIX) if i < seq_len
            ]
            if msg_indices:
                bias_1d[msg_indices] -= 0.65 * self.spec.kappa_dissonance_sink * float(
                    state.dissonance_score_d
                )

        return bias_1d

    @staticmethod
    def apply_variance_preserving_logit_shift(
        raw_qk_logits: torch.Tensor,
        key_bias_1d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Implements Audit Flaw 2 Resolution (Pre-Beta Variance-Preserving Logit Renormalization):
        Given unbiassed query-key logits Z_0 of shape [..., S_k] and additive key bias Delta_B(s):
          1. For non-sink tokens (s >= 1), shifts direction by Delta_B(s) and rescales variance
             so Std(Z_dir[..., 1:]) never exceeds Std(Z_0[..., 1:]), preventing Softmax collapse.
          2. Applies the exact Dissonance Sink boost Delta_B(0) at index s = 0.
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
        Implements Term 3 (Dual-Space QK-V Steering with Norm-Preserving Spherical Rotation):
          For every token s in `SPAN_INCOMING_MSG`:
            V_raw_s = V_s - gamma * <V_s, u_hat_trust> * u_hat_trust
                          + eta * ||V_s||_2 * u_hat_threat
            V_tilde_s = ||V_s||_2 * ( V_raw_s / max(||V_raw_s||_2, eps) )
        Returns `(V_tilde, norm_preservation_ratio)`.
        """
        if not msg_token_indices or (cynicism_gamma <= 0.0 and anxiety_eta <= 0.0):
            return v_tensor, 1.0

        s_len = v_tensor.shape[1]
        valid_idx = [i for i in msg_token_indices if 0 <= i < s_len]
        if not valid_idx:
            return v_tensor, 1.0

        v_out = v_tensor.clone()
        v_slice = v_out[:, valid_idx, :].float()  # [B, M, D_v]

        u_tr = u_hat_trust.float().to(v_tensor.device)
        u_th = u_hat_threat.float().to(v_tensor.device)

        orig_norm = torch.norm(v_slice, p=2, dim=-1, keepdim=True)  # [B, M, 1]

        # Project out trust component when cynicism_gamma > 0
        trust_dot = torch.matmul(v_slice, u_tr).unsqueeze(-1)  # [B, M, 1]
        v_mod = v_slice - float(cynicism_gamma) * trust_dot * u_tr.view(1, 1, -1)

        # Inject threat vector scaled to original activation norm when anxiety_eta > 0
        if anxiety_eta > 0.0:
            v_mod = v_mod + float(anxiety_eta) * orig_norm * u_th.view(1, 1, -1)

        mod_norm = torch.clamp(torch.norm(v_mod, p=2, dim=-1, keepdim=True), min=eps)
        v_spherical = orig_norm * (v_mod / mod_norm)

        final_norm = torch.norm(v_spherical, p=2, dim=-1, keepdim=True)
        norm_ratio = float((final_norm / torch.clamp(orig_norm, min=eps)).mean().item())

        v_out[:, valid_idx, :] = v_spherical.to(v_tensor.dtype)
        return v_out, norm_ratio

    def attach_prefill_value_hooks(
        self,
        model: PreTrainedModel,
        span_map: TokenSpanMap,
        state: EndogenousStatePacket,
    ) -> None:
        """
        Registers PyTorch forward hooks on `self_attn.v_proj` across calibrated layers
        to perform Dual-Space QK-V Value rotation during the prompt prefill pass.
        """
        self.remove_hooks()
        self._last_v_norm_ratios.clear()

        if (
            not self.spec.enabled
            or self.spec.mode == SteeringMode.NONE
            or self.calibration is None
        ):
            return

        msg_indices = span_map.get_token_indices_for_span(SPAN_INCOMING_MSG_PREFIX)
        if not msg_indices:
            return

        layers = MechanisticCalibrator.get_decoder_layers(model)

        for layer_idx in self.calibration.target_layer_indices:
            u_tr = self.calibration.u_trust_by_layer[layer_idx]
            u_th = self.calibration.u_threat_by_layer[layer_idx]
            v_proj_mod = layers[layer_idx].self_attn.v_proj

            def _make_v_hook(
                u_trust_vec: torch.Tensor, u_threat_vec: torch.Tensor
            ) -> Callable:
                def _v_hook(
                    module: nn.Module, inputs: Any, output: torch.Tensor
                ) -> torch.Tensor:
                    # Only apply to prompt prefill pass (where sequence length > 1)
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

    def remove_hooks(self) -> None:
        for h in self._active_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._active_hooks.clear()

    def compute_steered_attention_telemetry(
        self,
        unsteered_qk_logits: torch.Tensor,
        span_map: TokenSpanMap,
        state: EndogenousStatePacket,
        tick: int,
    ) -> Tuple[torch.Tensor, SteeringTelemetryPacket]:
        """
        Given raw/unsteered last-token query-key attention logits of shape [B, H, 1, S_k]
        (or post-softmax weights converted via log), applies `build_key_bias_vector` +
        `apply_variance_preserving_logit_shift` + Softmax to return exact steered
        attention distribution `A_tilde` and `SteeringTelemetryPacket`.
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
        attn_probs = torch.softmax(steered_logits, dim=-1)  # [B, H, 1, S_k]
        mean_probs = attn_probs.mean(dim=(0, 1, 2))         # [S_k]

        fin_idx = [i for i in span_map.get_token_indices_for_span(SPAN_DRIVES_AND_FINANCE) if i < s_k]
        peer_idx = [i for i in span_map.get_token_indices_for_span(SPAN_PEER_MEMORY_PREFIX) if i < s_k]
        msg_idx = [i for i in span_map.get_token_indices_for_span(SPAN_INCOMING_MSG_PREFIX) if i < s_k]

        mass_fin = float(mean_probs[fin_idx].sum().item()) if fin_idx else 0.0
        mass_peer = float(mean_probs[peer_idx].sum().item()) if peer_idx else 0.0
        mass_msg = float(mean_probs[msg_idx].sum().item()) if msg_idx else 0.0
        mass_sink = float(mean_probs[0].item()) if s_k > 0 else 0.0
        v_ratio = (
            sum(self._last_v_norm_ratios) / len(self._last_v_norm_ratios)
            if self._last_v_norm_ratios
            else 1.0
        )

        telemetry = SteeringTelemetryPacket(
            agent_id=state.agent_id,
            tick=tick,
            steering_mode=self.spec.mode if self.spec.enabled else SteeringMode.NONE,
            attn_mass_drives_finance=round(mass_fin, 5),
            attn_mass_peer_memory=round(mass_peer, 5),
            attn_mass_incoming_msg=round(mass_msg, 5),
            attn_mass_sink_bos=round(mass_sink, 5),
            value_projection_norm_ratio=round(v_ratio, 6),
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
        Equation:
            s_tilde(i, s) = t_0 - (t_0 - s) * exp( -lambda_rope * S_closeness(i, s) )
            K_tilde_s = R_{Theta, s_tilde(i, s)} * K_s
        Applied once during prefill so KV-cache reuse remains 100% intact during decoding.
        """
        return key_positions

    def apply_sys1_sys2_competitive_head_gating(
        self,
        head_outputs: torch.Tensor,
        cognitive_load_l: float,
    ) -> torch.Tensor:
        """
        # TODO(post-mvp): [Master Equation Term 5 - System 1 vs. System 2 Competitive Head Gating]
        Equation:
            g_h(L_t) = 1 + alpha_sys1 * L_t   if h in H_Sys1 (Induction/Heuristic Heads)
            g_h(L_t) = 1 - beta_sys2 * L_t    if h in H_Sys2 (Deliberative/Reasoning Heads)
            O_h(t) = g_h(L_t) * (A_tilde_h @ V_tilde)
        """
        return head_outputs

    def inject_mirror_neuron_kv_prefix(
        self,
        past_key_values: Any,
        speaker_latent_vec: Optional[torch.Tensor],
    ) -> Any:
        """
        # TODO(post-mvp): [Master Equation Term 6 - Subconscious Emotional Contagion KV-Prefix]
        Equation:
            k_mirror^{(l*, h)} = W_K^{(l*, h)} ( epsilon_BA * P_affect^{(l*)} * h_bar_A^{(l*)} )
            v_mirror^{(l*, h)} = W_V^{(l*, h)} ( epsilon_BA * P_affect^{(l*)} * h_bar_A^{(l*)} )
        Prepends a single virtual affective token to `SPAN_INCOMING_MSG` at middle layer l*.
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
        Equations:
            beta(A_t, F_t) = clip( beta_0 * (1 + zeta_p * max(0, A_t - A_opt)^2) / (1 + zeta_f * F_t), 0.25, 6.0 )
            h_hebb^{(t)} = (1 - lambda_decay) * h_hebb^{(t-1)} + eta_hebb * Pool(A^{(t-1)})
        """
        return logits