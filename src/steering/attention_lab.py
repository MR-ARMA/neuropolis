from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, List, Tuple
import torch

from src.core.contracts import (
    ALL_SEMANTIC_SPANS,
    EndogenousStatePacket,
    MechanisticCalibrationBundle,
    SPAN_APPRAISAL_AND_TOM,
    SPAN_CORE_BELIEF,
    SPAN_DRIVES_AND_FINANCE,
    SPAN_INCOMING_MSG,
    SPAN_PEER_MEMORY,
    SPAN_SELF_IDENTITY,
    SPAN_SINK,
    SteeredForwardTelemetry,
    SteeringLabConfig,
)


class AttentionSteeringLab:
    """
    M3: White-Box Causal Attention Steering Lab & Unified Master Equation (v2.0).
    Uses zero-allocation in-place tensor operations (`.mul_()`, `.add_()`) inside
    forward hooks so a 3.68 GiB GPU never fragments or runs out of VRAM.
    """

    def __init__(
        self,
        config: SteeringLabConfig,
        calibration: MechanisticCalibrationBundle | None = None,
    ) -> None:
        self.config = config
        self.calibration = calibration
        self.last_telemetry: SteeredForwardTelemetry = SteeredForwardTelemetry()
        self.captured_middle_residual: torch.Tensor | None = None

    def compute_yerkes_dodson_beta(
        self, arousal_a: float, fatigue_f: float
    ) -> float:
        if not self.config.enabled or self.config.mode == "NONE":
            return 1.0
        num = 1.0 + self.config.zeta_panic * (
            max(0.0, arousal_a - self.config.optimal_arousal) ** 2
        )
        den = 1.0 + self.config.zeta_fatigue * fatigue_f
        beta = self.config.beta_base_temp * (num / max(den, 1e-6))
        return round(max(0.25, min(6.0, float(beta))), 4)

    def compute_span_attention_telemetry(
        self, state_pkt: EndogenousStatePacket
    ) -> SteeredForwardTelemetry:
        beta_lc = self.compute_yerkes_dodson_beta(
            state_pkt.arousal_a, state_pkt.fatigue_f
        )
        f_it = state_pkt.financial_stress
        tau_abs = abs(state_pkt.peer_trust_tau)
        d_it = state_pkt.dissonance_d
        hebb = list(state_pkt.hebbian_trace_h)
        while len(hebb) < len(ALL_SEMANTIC_SPANS):
            hebb.append(0.0)

        if not self.config.enabled or self.config.mode == "NONE":
            raw_logits = {sp: 0.0 for sp in ALL_SEMANTIC_SPANS}
        else:
            raw_logits = {
                SPAN_SINK: self.config.kappa_dissonance_sink * d_it,
                SPAN_SELF_IDENTITY: 0.35 + self.config.mu_hebbian * hebb[1],
                SPAN_DRIVES_AND_FINANCE: (
                    self.config.omega_1_drives_finance * f_it
                    + self.config.mu_hebbian * hebb[2]
                ),
                SPAN_CORE_BELIEF: (
                    0.30 + 0.45 * d_it + self.config.mu_hebbian * hebb[3]
                ),
                SPAN_PEER_MEMORY: (
                    self.config.omega_2_peer_memory * tau_abs
                    + self.config.mu_hebbian * hebb[4]
                ),
                SPAN_INCOMING_MSG: (
                    0.65 * (1.0 - 0.45 * d_it)
                    + 0.35 * state_pkt.peer_intimacy_s
                    + self.config.mu_hebbian * hebb[5]
                ),
                SPAN_APPRAISAL_AND_TOM: (
                    0.50 * (1.0 - 0.35 * state_pkt.cognitive_load_l)
                    + self.config.mu_hebbian * hebb[6]
                ),
            }

        vals = torch.tensor(
            [raw_logits[sp] for sp in ALL_SEMANTIC_SPANS], dtype=torch.float32
        )
        clamped = self.config.logit_clamp_max * torch.tanh(
            vals / max(self.config.logit_clamp_max, 1.0)
        )
        probs = torch.softmax(beta_lc * clamped, dim=0).tolist()

        span_dist = {
            sp: round(float(p), 4) for sp, p in zip(ALL_SEMANTIC_SPANS, probs)
        }
        span_dist["yerkes_dodson_beta"] = beta_lc

        lam = self.config.hebbian_decay_lambda
        eta = self.config.hebbian_learning_eta
        updated_hebb = [
            round((1.0 - lam) * float(hebb[i]) + eta * float(probs[i]), 4)
            for i in range(len(ALL_SEMANTIC_SPANS))
        ]

        self.last_telemetry = SteeredForwardTelemetry(
            updated_hebbian_trace=updated_hebb,
            span_attention_distribution=span_dist,
            yerkes_dodson_beta=beta_lc,
        )
        return self.last_telemetry

    @contextmanager
    def attach_hooks(
        self, model: Any, state_pkt: EndogenousStatePacket
    ) -> Iterator[SteeredForwardTelemetry]:
        telemetry = self.compute_span_attention_telemetry(state_pkt)
        self.captured_middle_residual = None

        if (
            model is None
            or not self.config.enabled
            or self.config.mode == "NONE"
        ):
            yield telemetry
            return

        handles: List[Any] = []
        try:
            layers = getattr(getattr(model, "model", model), "layers", None)
            if layers is not None and len(layers) > 0:
                num_layers = len(layers)
                mid_layer = num_layers // 2
                span_map = state_pkt.span_map or {}

                # 1. Middle-Layer Residual Capture & In-Place Mirror Injection (Prefill Only)
                def _mid_layer_hook(module: Any, inputs: Any, output: Any) -> Any:
                    hidden = output[0] if isinstance(output, tuple) else output
                    if (
                        isinstance(hidden, torch.Tensor)
                        and hidden.dim() == 3
                        and hidden.size(1) > 1
                    ):
                        self.captured_middle_residual = (
                            hidden.detach().mean(dim=1).to("cpu", dtype=torch.float32)
                        )
                        if (
                            state_pkt.incoming_mirror_h is not None
                            and state_pkt.empathy_eps > 0.0
                            and SPAN_INCOMING_MSG in span_map
                        ):
                            s_idx, e_idx = span_map[SPAN_INCOMING_MSG]
                            if 0 <= s_idx < e_idx <= hidden.size(1):
                                mir = state_pkt.incoming_mirror_h.to(
                                    device=hidden.device, dtype=hidden.dtype
                                )
                                if mir.dim() == 2 and mir.size(-1) == hidden.size(-1):
                                    hidden[:, s_idx:e_idx, :].add_(
                                        mir.unsqueeze(1),
                                        alpha=float(0.08 * state_pkt.empathy_eps),
                                    )
                    return output

                handles.append(layers[mid_layer].register_forward_hook(_mid_layer_hook))

                # 2. Zero-Allocation In-Place K / V / O Hooks (Prefill Pass Only)
                for l_idx in range(num_layers):
                    attn = getattr(layers[l_idx], "self_attn", None)
                    if attn is None:
                        continue

                    k_proj = getattr(attn, "k_proj", None)
                    if k_proj is not None and (
                        state_pkt.dissonance_d > 0.05
                        or state_pkt.financial_stress > 0.05
                    ):
                        def _k_hook(mod: Any, inp: Any, out: torch.Tensor) -> torch.Tensor:
                            if not isinstance(out, torch.Tensor) or out.size(1) <= 1:
                                return out
                            seq_len = out.size(1)
                            if state_pkt.dissonance_d > 0.05:
                                out[:, 0:1, :].mul_(1.0 + 0.25 * state_pkt.dissonance_d)
                            if (
                                state_pkt.financial_stress > 0.05
                                and SPAN_DRIVES_AND_FINANCE in span_map
                            ):
                                s_f, e_f = span_map[SPAN_DRIVES_AND_FINANCE]
                                if 0 <= s_f < e_f <= seq_len:
                                    out[:, s_f:e_f, :].mul_(
                                        1.0 + 0.22 * state_pkt.financial_stress
                                    )
                            return out

                        handles.append(k_proj.register_forward_hook(_k_hook))

                    v_proj = getattr(attn, "v_proj", None)
                    if (
                        v_proj is not None
                        and num_layers // 3 <= l_idx <= (2 * num_layers) // 3
                    ):
                        def _v_hook(mod: Any, inp: Any, out: torch.Tensor) -> torch.Tensor:
                            if (
                                not isinstance(out, torch.Tensor)
                                or out.size(1) <= 1
                                or SPAN_INCOMING_MSG not in span_map
                            ):
                                return out
                            s_m, e_m = span_map[SPAN_INCOMING_MSG]
                            if 0 <= s_m < e_m <= out.size(1):
                                gamma_cyn = (
                                    max(0.0, -state_pkt.peer_trust_tau)
                                    * self.config.gamma_cynicism_scale
                                )
                                eta_anx = (
                                    state_pkt.financial_stress
                                    * self.config.eta_anxiety_scale
                                )
                                scale = max(
                                    0.75,
                                    min(1.25, 1.0 - 0.15 * gamma_cyn + 0.10 * eta_anx),
                                )
                                out[:, s_m:e_m, :].mul_(scale)
                            return out

                        handles.append(v_proj.register_forward_hook(_v_hook))

                    o_proj = getattr(attn, "o_proj", None)
                    if o_proj is not None and state_pkt.cognitive_load_l > 0.05:
                        def _make_o_hook(layer_i: int) -> Any:
                            def _o_hook(mod: Any, inp: Any, out: torch.Tensor) -> torch.Tensor:
                                if not isinstance(out, torch.Tensor) or out.size(1) <= 1:
                                    return out
                                load_l = state_pkt.cognitive_load_l
                                gain = (
                                    1.0 + 0.10 * self.config.alpha_sys1_gain * load_l
                                    if layer_i < num_layers // 2
                                    else 1.0 - 0.10 * self.config.beta_sys2_atten * load_l
                                )
                                out.mul_(gain)
                                return out

                            return _o_hook

                        handles.append(o_proj.register_forward_hook(_make_o_hook(l_idx)))

            yield telemetry
        finally:
            for h in handles:
                h.remove()


MasterEquationHookEngine = AttentionSteeringLab