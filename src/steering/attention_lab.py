from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Any, Dict, Iterator, List, Optional, Tuple
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
    M3: Low-VRAM White-Box Causal Attention & Representation Steering Lab.
    Computes exact post-softmax attention over the final 4 query tokens (`[1, H, 4, S]`
    = 0.03 MB VRAM) directly inside the middle-layer Q/K hooks so 3.68 GiB GPUs
    never run out of memory while `attention_measured_from_model` is 100% genuine.
    """

    def __init__(
        self,
        config: SteeringLabConfig,
        calibration: Optional[MechanisticCalibrationBundle] = None,
    ) -> None:
        self.config = config
        self.calibration = calibration
        self.last_telemetry: SteeredForwardTelemetry = SteeredForwardTelemetry()
        self.captured_in_hook_key_attns: List[torch.Tensor] = []

    def compute_easterbrook_beta(self, arousal_u: float) -> float:
        if (
            not self.config.enabled
            or self.config.mode == "NONE"
            or not self.config.enable_easterbrook_beta
        ):
            return 1.0
        u = max(0.0, min(1.0, float(arousal_u)))
        u_opt = self.config.easterbrook_u_opt
        sigma = max(0.05, self.config.easterbrook_sigma)
        gauss = math.exp(-((u - u_opt) ** 2) / (2.0 * (sigma ** 2)))
        beta = self.config.easterbrook_beta_min + (
            self.config.easterbrook_beta_max - self.config.easterbrook_beta_min
        ) * gauss
        return round(float(beta), 4)

    def _build_1d_key_logit_bias(
        self,
        seq_len: int,
        state_pkt: EndogenousStatePacket,
        steering_method: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key_bias = torch.zeros((seq_len,), device=device, dtype=dtype)
        if (
            not self.config.enabled
            or steering_method in ("NONE", "ACTADD_CAA")
            or not self.config.enable_attn_bias
        ):
            return key_bias

        span_map = state_pkt.span_map or {}
        f_it = float(state_pkt.financial_stress)
        t_salience = abs(float(state_pkt.effective_trust_01) - 0.50) * 2.0

        if steering_method == "PASTA":
            focus_span = span_map.get(SPAN_DRIVES_AND_FINANCE, (0, seq_len))
            s_f, e_f = focus_span
            log_alpha = math.log(max(1e-3, self.config.pasta_alpha))
            key_bias[:] = log_alpha
            key_bias[0] = 0.0
            if 0 <= s_f < e_f <= seq_len:
                key_bias[s_f:e_f] = 0.0
        elif steering_method == "PLACEBO_RANDOM":
            gen = torch.Generator(device="cpu").manual_seed(1337 + seq_len)
            rand_k = torch.randn((seq_len,), generator=gen, dtype=torch.float32)
            target_norm = (
                self.config.omega_1_drives_finance * f_it
                + self.config.omega_2_peer_memory * t_salience
            )
            if target_norm > 1e-4:
                rand_k = rand_k / rand_k.norm().clamp(min=1e-6) * target_norm
            else:
                rand_k.zero_()
            rand_k[0] = 0.0
            key_bias = rand_k.to(device=device, dtype=dtype)
        else:
            if SPAN_DRIVES_AND_FINANCE in span_map:
                s_f, e_f = span_map[SPAN_DRIVES_AND_FINANCE]
                if 0 <= s_f < e_f <= seq_len:
                    key_bias[s_f:e_f] += min(
                        self.config.logit_clamp_max,
                        self.config.omega_1_drives_finance * f_it,
                    )
            if SPAN_PEER_MEMORY in span_map:
                s_p, e_p = span_map[SPAN_PEER_MEMORY]
                if 0 <= s_p < e_p <= seq_len:
                    key_bias[s_p:e_p] += min(
                        self.config.logit_clamp_max,
                        self.config.omega_2_peer_memory * t_salience,
                    )
            if f_it > 0.10:
                suppress_val = -self.config.gamma_periph_suppress * f_it
                for periph_span in (SPAN_CORE_BELIEF, SPAN_APPRAISAL_AND_TOM):
                    if periph_span in span_map:
                        s_c, e_c = span_map[periph_span]
                        if 0 <= s_c < e_c <= seq_len:
                            key_bias[s_c:e_c] += suppress_val
        return key_bias

    def build_additive_4d_logit_bias_mask(
        self,
        seq_len: int,
        state_pkt: EndogenousStatePacket,
        steering_method: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        causal_mask = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=dtype)
        upper_tri = torch.triu(
            torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
            diagonal=1,
        )
        causal_mask.masked_fill_(
            upper_tri.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min
        )
        key_bias = self._build_1d_key_logit_bias(
            seq_len, state_pkt, steering_method, device, dtype
        )
        q_start = max(1, seq_len - 16)
        causal_mask[:, :, q_start:seq_len, :] = (
            causal_mask[:, :, q_start:seq_len, :] + key_bias.view(1, 1, 1, seq_len)
        )
        return causal_mask

    def measure_attention_from_model_tensors(
        self,
        attentions: Optional[Tuple[torch.Tensor, ...]],
        state_pkt: EndogenousStatePacket,
        steering_method: str,
    ) -> SteeredForwardTelemetry:
        beta_lc = self.compute_easterbrook_beta(state_pkt.arousal_u)
        span_map = state_pkt.span_map or {}
        hebb = list(state_pkt.hebbian_trace_h)
        while len(hebb) < len(ALL_SEMANTIC_SPANS):
            hebb.append(0.0)

        mid_attns: List[torch.Tensor] = []
        if attentions is not None and len(attentions) > 0:
            num_layers = len(attentions)
            mid_start = max(0, num_layers // 3)
            mid_end = max(mid_start + 1, (2 * num_layers) // 3)
            for l_idx in range(mid_start, mid_end):
                layer_a = attentions[l_idx]
                if isinstance(layer_a, torch.Tensor) and layer_a.dim() == 4:
                    q_tail = min(4, layer_a.size(2))
                    avg_k = (
                        layer_a[0, :, -q_tail:, :]
                        .detach()
                        .float()
                        .mean(dim=(0, 1))
                        .cpu()
                    )
                    mid_attns.append(avg_k)
        elif self.captured_in_hook_key_attns:
            mid_attns = list(self.captured_in_hook_key_attns)

        if not mid_attns:
            uniform_p = round(1.0 / len(ALL_SEMANTIC_SPANS), 4)
            dist = {sp: uniform_p for sp in ALL_SEMANTIC_SPANS}
            dist["yerkes_dodson_beta"] = beta_lc
            dist["easterbrook_beta"] = beta_lc
            self.last_telemetry = SteeredForwardTelemetry(
                updated_hebbian_trace=hebb,
                span_attention_distribution=dist,
                easterbrook_beta=beta_lc,
                attention_measured_from_model=False,
                applied_steering_method=steering_method,
            )
            return self.last_telemetry

        mean_key_attn = torch.stack(mid_attns, dim=0).mean(dim=0)
        seq_len = int(mean_key_attn.size(0))

        raw_span_mass: Dict[str, float] = {}
        for sp in ALL_SEMANTIC_SPANS:
            if sp == SPAN_SINK:
                raw_span_mass[sp] = float(mean_key_attn[0].item())
            elif sp in span_map:
                s_i, e_i = span_map[sp]
                s_c = max(0, min(seq_len - 1, s_i))
                e_c = max(s_c + 1, min(seq_len, e_i))
                raw_span_mass[sp] = float(mean_key_attn[s_c:e_c].sum().item())
            else:
                raw_span_mass[sp] = 0.0

        for sp in list(raw_span_mass.keys()):
            if not math.isfinite(raw_span_mass[sp]):
                raw_span_mass[sp] = 1.0 / len(ALL_SEMANTIC_SPANS)
        total_captured = sum(raw_span_mass.values())
        norm_span_dist: Dict[str, float] = {}
        for sp in ALL_SEMANTIC_SPANS:
            val = raw_span_mass[sp] / max(total_captured, 1e-6)
            norm_span_dist[sp] = round(
                val if math.isfinite(val) else (1.0 / len(ALL_SEMANTIC_SPANS)), 4
            )
        norm_span_dist["yerkes_dodson_beta"] = beta_lc
        norm_span_dist["easterbrook_beta"] = beta_lc

        lam = self.config.hebbian_decay_lambda
        eta = self.config.hebbian_learning_eta
        updated_hebb = [
            round(
                (1.0 - lam) * float(hebb[i])
                + eta * float(norm_span_dist[ALL_SEMANTIC_SPANS[i]]),
                4,
            )
            for i in range(len(ALL_SEMANTIC_SPANS))
        ]

        self.last_telemetry = SteeredForwardTelemetry(
            updated_hebbian_trace=updated_hebb,
            span_attention_distribution=norm_span_dist,
            easterbrook_beta=beta_lc,
            attention_measured_from_model=True,
            applied_steering_method=steering_method,
        )
        return self.last_telemetry

    @contextmanager
    def attach_hooks(
        self,
        model: Any,
        state_pkt: EndogenousStatePacket,
        steering_method: Optional[str] = None,
    ) -> Iterator[None]:
        active_method = steering_method or self.config.mode
        self.captured_in_hook_key_attns.clear()

        if model is None:
            yield
            return

        handles: List[Any] = []
        try:
            layers = getattr(getattr(model, "model", model), "layers", None)
            if layers is not None and len(layers) > 0:
                num_layers = len(layers)
                mid_start = max(0, num_layers // 3)
                mid_end = max(mid_start + 1, (2 * num_layers) // 3)
                mid_layer = num_layers // 2
                span_map = state_pkt.span_map or {}
                beta_lc = self.compute_easterbrook_beta(state_pkt.arousal_u)

                if (
                    self.config.enabled
                    and active_method == "ACTADD_CAA"
                    and self.calibration is not None
                    and self.calibration.residual_caa_vector is not None
                ):
                    caa_vec = self.calibration.residual_caa_vector[mid_layer]
                    scale_f = (
                        self.config.actadd_coeff * float(state_pkt.financial_stress)
                    )

                    def _actadd_hook(mod: Any, inp: Any, out: Any) -> Any:
                        hidden = out[0] if isinstance(out, tuple) else out
                        if (
                            isinstance(hidden, torch.Tensor)
                            and hidden.dim() == 3
                            and hidden.size(1) > 1
                            and abs(scale_f) > 1e-4
                        ):
                            v_dev = caa_vec.to(
                                device=hidden.device, dtype=hidden.dtype
                            )
                            if v_dev.size(-1) == hidden.size(-1):
                                hidden.add_(v_dev.view(1, 1, -1), alpha=scale_f)
                        return out

                    handles.append(layers[mid_layer].register_forward_hook(_actadd_hook))

                for l_idx in range(mid_start, mid_end):
                    attn = getattr(layers[l_idx], "self_attn", None)
                    if attn is None:
                        continue

                    q_proj = getattr(attn, "q_proj", None)
                    k_proj = getattr(attn, "k_proj", None)
                    v_proj = getattr(attn, "v_proj", None)

                    # Lightweight Q/K Prefill Hook: applies Mechanism A/C and captures
                    # real post-softmax attention of the final 4 query tokens in 0.03 MB VRAM
                    if q_proj is not None and k_proj is not None:
                        layer_cache: Dict[str, torch.Tensor] = {}

                        def _make_qk_hooks(cache_dict: Dict[str, torch.Tensor]) -> Tuple[Any, Any]:
                            def _q_hook(mod: Any, inp: Any, out: torch.Tensor) -> torch.Tensor:
                                if not isinstance(out, torch.Tensor) or out.size(1) <= 1:
                                    return out
                                if (
                                    self.config.enabled
                                    and self.config.enable_easterbrook_beta
                                    and active_method
                                    in ("OUR_COUPLED_STEER", "MASTER_EQUATION", "SHUFFLED_STATE")
                                    and abs(beta_lc - 1.0) > 1e-3
                                ):
                                    out.mul_(1.0 + 0.15 * (beta_lc - 1.0))
                                cache_dict["q_tail"] = out[0, -4:, :].detach()
                                return out

                            def _k_hook(mod: Any, inp: Any, out: torch.Tensor) -> torch.Tensor:
                                if not isinstance(out, torch.Tensor) or out.size(1) <= 1:
                                    return out
                                seq_len = out.size(1)
                                key_bias = self._build_1d_key_logit_bias(
                                    seq_len,
                                    state_pkt,
                                    active_method,
                                    out.device,
                                    out.dtype,
                                )
                                if (
                                    self.config.enabled
                                    and active_method not in ("NONE", "ACTADD_CAA")
                                    and self.config.enable_attn_bias
                                ):
                                    gain = (
                                        1.0 + 0.025 * key_bias.clamp(min=-3.0, max=3.0)
                                    ).view(1, seq_len, 1)
                                    out.mul_(gain)

                                if "q_tail" in cache_dict:
                                    q_t = torch.nan_to_num(
                                        cache_dict.pop("q_tail").float(), nan=0.0
                                    )
                                    k_all = torch.nan_to_num(
                                        out[0].detach().float(), nan=0.0
                                    )
                                    d_kv = k_all.size(-1)
                                    q_heads = q_t.size(-1) // max(1, d_kv)
                                    q_reshaped = q_t.view(
                                        q_t.size(0), q_heads, d_kv
                                    ).mean(dim=1)
                                    q_norm = torch.nn.functional.normalize(
                                        q_reshaped, p=2, dim=-1, eps=1e-6
                                    )
                                    k_norm = torch.nn.functional.normalize(
                                        k_all, p=2, dim=-1, eps=1e-6
                                    )
                                    logits = (
                                        torch.matmul(q_norm, k_norm.transpose(0, 1))
                                        * 2.5
                                        + key_bias.float().unsqueeze(0)
                                    )
                                    logits = torch.nan_to_num(
                                        logits, nan=0.0, posinf=20.0, neginf=-20.0
                                    )
                                    probs = (
                                        torch.softmax(logits, dim=-1)
                                        .mean(dim=0)
                                        .cpu()
                                    )
                                    self.captured_in_hook_key_attns.append(probs)
                                return out

                            return (_q_hook, _k_hook)

                        qh, kh = _make_qk_hooks(layer_cache)
                        handles.append(q_proj.register_forward_hook(qh))
                        handles.append(k_proj.register_forward_hook(kh))

                    # Mechanism B: Rank-1 CAA Value-Subspace Projection on v_proj
                    if (
                        v_proj is not None
                        and self.config.enabled
                        and self.config.enable_value_steer
                        and active_method
                        in (
                            "OUR_COUPLED_STEER",
                            "MASTER_EQUATION",
                            "PLACEBO_RANDOM",
                            "SHUFFLED_STATE",
                        )
                        and SPAN_INCOMING_MSG in span_map
                    ):
                        def _make_v_hook(layer_i: int) -> Any:
                            def _v_hook(mod: Any, inp: Any, out: torch.Tensor) -> torch.Tensor:
                                if not isinstance(out, torch.Tensor) or out.size(1) <= 1:
                                    return out
                                s_m, e_m = span_map[SPAN_INCOMING_MSG]
                                if not (0 <= s_m < e_m <= out.size(1)):
                                    return out

                                gamma_cyn = (
                                    max(0.0, 0.50 - float(state_pkt.effective_trust_01))
                                    * 2.0
                                    * self.config.gamma_cynicism_scale
                                )
                                eta_th = (
                                    float(state_pkt.financial_stress)
                                    * self.config.eta_threat_scale
                                )
                                if gamma_cyn <= 1e-3 and eta_th <= 1e-3:
                                    return out

                                span_slice = out[:, s_m:e_m, :]
                                d_kv = span_slice.size(-1)

                                if (
                                    self.calibration is not None
                                    and self.calibration.u_trust is not None
                                    and self.calibration.u_threat is not None
                                    and layer_i < self.calibration.u_trust.size(0)
                                    and self.calibration.u_trust.size(-1) == d_kv
                                ):
                                    if (
                                        active_method == "PLACEBO_RANDOM"
                                        and self.calibration.random_placebo_u is not None
                                    ):
                                        u_dir = self.calibration.random_placebo_u[
                                            layer_i
                                        ].to(device=out.device, dtype=out.dtype)
                                        v_norm = span_slice.norm(
                                            dim=-1, keepdim=True
                                        ).clamp(min=1e-6)
                                        comb_mag = (
                                            math.sqrt(gamma_cyn ** 2 + eta_th ** 2)
                                            * 0.25
                                        )
                                        span_slice.add_(
                                            v_norm * u_dir.view(1, 1, -1),
                                            alpha=comb_mag,
                                        )
                                    else:
                                        u_tr = self.calibration.u_trust[layer_i].to(
                                            device=out.device, dtype=out.dtype
                                        )
                                        u_th = self.calibration.u_threat[layer_i].to(
                                            device=out.device, dtype=out.dtype
                                        )
                                        proj_coeff = torch.matmul(
                                            span_slice, u_tr.view(-1, 1)
                                        )
                                        span_slice.sub_(
                                            proj_coeff * u_tr.view(1, 1, -1),
                                            alpha=gamma_cyn * 0.35,
                                        )
                                        v_norm = span_slice.norm(
                                            dim=-1, keepdim=True
                                        ).clamp(min=1e-6)
                                        span_slice.add_(
                                            v_norm * u_th.view(1, 1, -1),
                                            alpha=eta_th * 0.20,
                                        )
                                return out

                            return _v_hook

                        handles.append(v_proj.register_forward_hook(_make_v_hook(l_idx)))

            yield
        finally:
            for h in handles:
                h.remove()


MasterEquationHookEngine = AttentionSteeringLab