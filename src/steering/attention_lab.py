from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
import torch.nn.functional as F

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
    M3: Thesis-Grade White-Box Attention & Representation Steering Engine.
    - Injects true additive pre-softmax logit bias `b_j` via 4D float `attention_mask`.
    - Captures exact eager post-softmax attention weights from `outputs.attentions`
      (freeing non-target layers on the fly when `vocab_size > 1000` so a 28-layer
      1.5B model never runs out of memory on a 3.68 GiB GPU).
    """

    def __init__(
        self,
        config: SteeringLabConfig,
        calibration: Optional[MechanisticCalibrationBundle] = None,
    ) -> None:
        self.config = config
        self.calibration = calibration
        self.last_telemetry: SteeredForwardTelemetry = SteeredForwardTelemetry()
        self.last_relative_residual_mag: float = 0.0
        self.captured_eager_layer_attns: List[torch.Tensor] = []

    def get_target_layer_indices(self, num_layers: int) -> List[int]:
        scope = self.config.layer_scope
        if scope == "early_third":
            return list(range(0, max(1, num_layers // 3)))
        if scope == "late_third":
            return list(range(max(0, (2 * num_layers) // 3), num_layers))
        if scope == "all_layers":
            return list(range(num_layers))
        start = max(0, num_layers // 3)
        end = max(start + 1, (2 * num_layers) // 3)
        return list(range(start, end))

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

    def build_1d_prompt_key_bias(
        self,
        prompt_seq_len: int,
        state_pkt: EndogenousStatePacket,
        steering_method: str,
        alpha_override: Optional[float] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        key_bias = torch.zeros((prompt_seq_len,), device=device, dtype=dtype)
        alpha = (
            float(alpha_override)
            if alpha_override is not None
            else float(self.config.alpha_dose)
        )
        if (
            not self.config.enabled
            or steering_method in ("NONE", "ACTADD_CAA")
            or not self.config.enable_attn_bias
            or abs(alpha) <= 1e-8
        ):
            return key_bias

        span_map = state_pkt.span_map or {}
        f_it = float(state_pkt.financial_stress)
        t_salience = abs(float(state_pkt.effective_trust_01) - 0.50) * 2.0
        sign_mult = -1.0 if steering_method == "REVERSED_SIGN" else 1.0

        if steering_method == "PASTA":
            focus_span = span_map.get(SPAN_DRIVES_AND_FINANCE, (0, prompt_seq_len))
            s_f, e_f = focus_span
            log_alpha = math.log(max(1e-3, self.config.pasta_alpha))
            key_bias[:] = log_alpha
            key_bias[0] = 0.0
            if 0 <= s_f < e_f <= prompt_seq_len:
                key_bias[s_f:e_f] = 0.0
            return key_bias

        if steering_method == "WRONG_SPAN":
            if SPAN_SELF_IDENTITY in span_map:
                s_w, e_w = span_map[SPAN_SELF_IDENTITY]
                if 0 <= s_w < e_w <= prompt_seq_len:
                    val = min(
                        self.config.logit_clamp_max,
                        alpha * self.config.omega_1_drives_finance * max(0.25, f_it),
                    )
                    key_bias[s_w:e_w] += val
            return key_bias

        if steering_method == "PLACEBO_RANDOM":
            active_ref = self.build_1d_prompt_key_bias(
                prompt_seq_len=prompt_seq_len,
                state_pkt=state_pkt,
                steering_method="OUR_COUPLED_STEER",
                alpha_override=alpha,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            target_l2 = float(active_ref.norm(p=2).item())
            if target_l2 <= 1e-6:
                return key_bias
            gen = torch.Generator(device="cpu").manual_seed(1337 + prompt_seq_len)
            rand_k = torch.randn(
                (prompt_seq_len,), generator=gen, dtype=torch.float32
            )
            rand_k[0] = 0.0
            rand_k = rand_k - rand_k.mean()
            rand_k[0] = 0.0
            kl_scale = (
                self.calibration.kl_matched_placebo_scale
                if self.calibration is not None
                else 1.0
            )
            rand_k = (
                rand_k
                / rand_k.norm(p=2).clamp(min=1e-6)
                * (target_l2 * kl_scale)
            )
            return rand_k.to(device=device, dtype=dtype)

        if SPAN_DRIVES_AND_FINANCE in span_map:
            s_f, e_f = span_map[SPAN_DRIVES_AND_FINANCE]
            if 0 <= s_f < e_f <= prompt_seq_len:
                b_fin = sign_mult * min(
                    self.config.logit_clamp_max,
                    alpha * self.config.omega_1_drives_finance * f_it,
                )
                key_bias[s_f:e_f] += b_fin

        if SPAN_PEER_MEMORY in span_map:
            s_p, e_p = span_map[SPAN_PEER_MEMORY]
            if 0 <= s_p < e_p <= prompt_seq_len:
                b_peer = sign_mult * min(
                    self.config.logit_clamp_max,
                    alpha * self.config.omega_2_peer_memory * t_salience,
                )
                key_bias[s_p:e_p] += b_peer

        if f_it > 0.10 and self.config.gamma_periph_suppress > 0.0:
            suppress_val = (
                -sign_mult
                * alpha
                * self.config.gamma_periph_suppress
                * f_it
                * 0.25
            )
            for periph_span in (SPAN_CORE_BELIEF, SPAN_APPRAISAL_AND_TOM):
                if periph_span in span_map:
                    s_c, e_c = span_map[periph_span]
                    if 0 <= s_c < e_c <= prompt_seq_len:
                        key_bias[s_c:e_c] += suppress_val

        return key_bias

    def build_additive_4d_logit_bias_mask(
        self,
        seq_len: int,
        state_pkt: EndogenousStatePacket,
        steering_method: str,
        device: torch.device,
        dtype: torch.dtype,
        alpha_override: Optional[float] = None,
    ) -> torch.Tensor:
        causal_mask = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=dtype)
        upper_tri = torch.triu(
            torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
            diagonal=1,
        )
        # Use -1e4 in fp16/bf16 and finfo.min in fp32 so adding bias never overflows
        mask_fill_val = (
            torch.finfo(dtype).min if dtype == torch.float32 else -10000.0
        )
        causal_mask.masked_fill_(
            upper_tri.unsqueeze(0).unsqueeze(0), mask_fill_val
        )
        key_bias = self.build_1d_prompt_key_bias(
            prompt_seq_len=seq_len,
            state_pkt=state_pkt,
            steering_method=steering_method,
            alpha_override=alpha_override,
            device=device,
            dtype=dtype,
        )
        if float(key_bias.abs().max().item()) <= 1e-8:
            return causal_mask

        if self.config.query_row_scope == "last_token_only":
            q_start = seq_len - 1
        else:
            q_start = max(1, seq_len - 16)

        causal_mask[:, :, q_start:seq_len, :] = (
            causal_mask[:, :, q_start:seq_len, :] + key_bias.view(1, 1, 1, seq_len)
        )
        # Re-enforce causal upper triangle mask
        causal_mask.masked_fill_(
            upper_tri.unsqueeze(0).unsqueeze(0), mask_fill_val
        )
        return causal_mask

    def build_decode_step_4d_bias_mask(
        self,
        prompt_key_bias: torch.Tensor,
        total_kv_len: int,
        steering_method: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        decode_mask = torch.zeros(
            (1, 1, 1, total_kv_len), device=device, dtype=dtype
        )
        if (
            not self.config.enabled
            or self.config.scope == "prefill_only"
            or steering_method in ("NONE", "ACTADD_CAA")
        ):
            return decode_mask

        prompt_len = int(prompt_key_bias.size(0))
        copy_len = min(prompt_len, total_kv_len)
        decode_mask[0, 0, 0, :copy_len] = prompt_key_bias[:copy_len].to(
            device=device, dtype=dtype
        )
        return decode_mask

    @staticmethod
    def _safe_compute_dtype(model: Any) -> torch.dtype:
        emb = getattr(getattr(model, "model", model), "embed_tokens", None)
        if emb is not None and hasattr(emb, "weight") and emb.weight.is_floating_point():
            return emb.weight.dtype
        for p in model.parameters():
            if p.is_floating_point():
                return p.dtype
        return (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else torch.float16
        )

    def shadow_measure_attention(
        self,
        model: Any,
        input_ids: torch.Tensor,
        state_pkt: EndogenousStatePacket,
        steering_method: str,
        alpha_override: Optional[float] = None,
    ) -> SteeredForwardTelemetry:
        emb = getattr(getattr(model, "model", model), "embed_tokens", None)
        device = (
            emb.weight.device
            if (emb is not None and hasattr(emb, "weight"))
            else next(model.parameters()).device
        )
        dtype = self._safe_compute_dtype(model)
        ids = input_ids.to(device)
        seq_len = int(ids.size(1))
        pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)

        mask_4d = self.build_additive_4d_logit_bias_mask(
            seq_len=seq_len,
            state_pkt=state_pkt,
            steering_method=steering_method,
            device=device,
            dtype=dtype,
            alpha_override=alpha_override,
        )
        key_bias = self.build_1d_prompt_key_bias(
            prompt_seq_len=seq_len,
            state_pkt=state_pkt,
            steering_method=steering_method,
            alpha_override=alpha_override,
            device=device,
            dtype=torch.float32,
        )
        bias_l2 = round(float(key_bias.norm(p=2).item()), 4)
        alpha_val = (
            float(alpha_override)
            if alpha_override is not None
            else float(self.config.alpha_dose)
        )

        with self.attach_hooks(
            model, state_pkt, steering_method=steering_method
        ):
            with torch.inference_mode():
                out = model(
                    input_ids=ids,
                    attention_mask=mask_4d,
                    position_ids=pos_ids,
                    use_cache=False,
                    output_attentions=True,
                    return_dict=True,
                )
                attentions = out.attentions

        telemetry = self.measure_attention_from_model_tensors(
            attentions=attentions,
            state_pkt=state_pkt,
            steering_method=steering_method,
            alpha_dose=alpha_val,
            bias_l2_norm=bias_l2,
        )
        del out, attentions, mask_4d
        return telemetry

    def measure_attention_from_model_tensors(
        self,
        attentions: Optional[Tuple[torch.Tensor, ...]],
        state_pkt: EndogenousStatePacket,
        steering_method: str,
        alpha_dose: Optional[float] = None,
        bias_l2_norm: float = 0.0,
    ) -> SteeredForwardTelemetry:
        beta_lc = self.compute_easterbrook_beta(state_pkt.arousal_u)
        span_map = state_pkt.span_map or {}
        hebb = list(state_pkt.hebbian_trace_h)
        while len(hebb) < len(ALL_SEMANTIC_SPANS):
            hebb.append(0.0)
        alpha_val = (
            float(alpha_dose)
            if alpha_dose is not None
            else float(self.config.alpha_dose)
        )

        mid_attns: List[torch.Tensor] = []
        if (
            attentions is not None
            and len(attentions) > 0
            and any(isinstance(a, torch.Tensor) for a in attentions)
        ):
            num_layers = len(attentions)
            target_layers = self.get_target_layer_indices(num_layers)
            for l_idx in target_layers:
                layer_a = attentions[l_idx]
                if isinstance(layer_a, torch.Tensor) and layer_a.dim() == 4:
                    if self.config.query_row_scope == "last_token_only":
                        q_slice = layer_a[0, :, -1:, :]
                    else:
                        q_tail = min(16, layer_a.size(2))
                        q_slice = layer_a[0, :, -q_tail:, :]
                    avg_k = (
                        torch.nan_to_num(q_slice.detach().float(), nan=0.0)
                        .mean(dim=(0, 1))
                        .cpu()
                    )
                    mid_attns.append(avg_k)
        elif self.captured_eager_layer_attns:
            mid_attns = list(self.captured_eager_layer_attns)

        if not mid_attns:
            dist = {sp: 0.0 for sp in ALL_SEMANTIC_SPANS}
            dist["yerkes_dodson_beta"] = beta_lc
            dist["easterbrook_beta"] = beta_lc
            self.last_telemetry = SteeredForwardTelemetry(
                updated_hebbian_trace=hebb,
                span_attention_distribution=dist,
                easterbrook_beta=beta_lc,
                alpha_dose_applied=alpha_val if steering_method != "NONE" else 0.0,
                bias_l2_norm=bias_l2_norm,
                relative_residual_magnitude=self.last_relative_residual_mag,
                attention_measured_from_model=False,
                shadow_eager_verified=False,
                applied_steering_method=steering_method,
                applied_scope=self.config.scope,
            )
            return self.last_telemetry

        mean_key_attn = torch.stack(mid_attns, dim=0).mean(dim=0)
        seq_len = int(mean_key_attn.size(0))

        span_mass: Dict[str, float] = {}
        for sp in ALL_SEMANTIC_SPANS:
            if sp == SPAN_SINK:
                span_mass[sp] = round(float(mean_key_attn[0].item()), 6)
            elif sp in span_map:
                s_i, e_i = span_map[sp]
                s_c = max(0, min(seq_len - 1, s_i))
                e_c = max(s_c + 1, min(seq_len, e_i))
                span_mass[sp] = round(float(mean_key_attn[s_c:e_c].sum().item()), 6)
            else:
                span_mass[sp] = 0.0

        span_mass["yerkes_dodson_beta"] = beta_lc
        span_mass["easterbrook_beta"] = beta_lc

        lam = self.config.hebbian_decay_lambda
        eta = self.config.hebbian_learning_eta
        updated_hebb = [
            round(
                (1.0 - lam) * float(hebb[i])
                + eta * float(span_mass[ALL_SEMANTIC_SPANS[i]]),
                4,
            )
            for i in range(len(ALL_SEMANTIC_SPANS))
        ]

        self.last_telemetry = SteeredForwardTelemetry(
            updated_hebbian_trace=updated_hebb,
            span_attention_distribution=span_mass,
            easterbrook_beta=beta_lc,
            alpha_dose_applied=alpha_val if steering_method != "NONE" else 0.0,
            bias_l2_norm=bias_l2_norm,
            relative_residual_magnitude=self.last_relative_residual_mag,
            attention_measured_from_model=True,
            shadow_eager_verified=True,
            applied_steering_method=steering_method,
            applied_scope=self.config.scope,
        )
        return self.last_telemetry

    def calibrate_kl_matched_placebo(
        self,
        model: Any,
        input_ids: torch.Tensor,
        state_pkt: EndogenousStatePacket,
        alpha_dose: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        emb = getattr(getattr(model, "model", model), "embed_tokens", None)
        device = (
            emb.weight.device
            if (emb is not None and hasattr(emb, "weight"))
            else next(model.parameters()).device
        )
        dtype = self._safe_compute_dtype(model)
        ids = input_ids.to(device)
        seq_len = int(ids.size(1))

        with torch.inference_mode():
            m_none = self.build_additive_4d_logit_bias_mask(
                seq_len, state_pkt, "NONE", device, dtype, alpha_dose
            )
            logits_0 = (
                model(input_ids=ids, attention_mask=m_none).logits[0, -1, :].float()
            )
            p_0 = F.softmax(logits_0, dim=-1).clamp(min=1e-12)

            m_steer = self.build_additive_4d_logit_bias_mask(
                seq_len, state_pkt, "OUR_COUPLED_STEER", device, dtype, alpha_dose
            )
            with self.attach_hooks(model, state_pkt, "OUR_COUPLED_STEER"):
                logits_s = (
                    model(input_ids=ids, attention_mask=m_steer).logits[0, -1, :].float()
                )
            p_s = F.softmax(logits_s, dim=-1).clamp(min=1e-12)
            kl_target = float(torch.sum(p_0 * torch.log(p_0 / p_s)).item())

            if self.calibration is None:
                return (kl_target, kl_target, 1.0)

            best_scale = 1.0
            best_diff = 1e9
            best_kl = 0.0
            for candidate_scale in (0.25, 0.50, 0.75, 1.0, 1.35, 1.75, 2.25, 3.0):
                self.calibration.kl_matched_placebo_scale = candidate_scale
                m_pl = self.build_additive_4d_logit_bias_mask(
                    seq_len, state_pkt, "PLACEBO_RANDOM", device, dtype, alpha_dose
                )
                with self.attach_hooks(model, state_pkt, "PLACEBO_RANDOM"):
                    logits_p = (
                        model(input_ids=ids, attention_mask=m_pl)
                        .logits[0, -1, :]
                        .float()
                    )
                p_p = F.softmax(logits_p, dim=-1).clamp(min=1e-12)
                kl_p = float(torch.sum(p_0 * torch.log(p_0 / p_p)).item())
                if abs(kl_p - kl_target) < best_diff:
                    best_diff = abs(kl_p - kl_target)
                    best_scale = candidate_scale
                    best_kl = kl_p

            self.calibration.kl_matched_placebo_scale = best_scale
            return (round(kl_target, 6), round(best_kl, 6), round(best_scale, 4))

    @contextmanager
    def attach_hooks(
        self,
        model: Any,
        state_pkt: EndogenousStatePacket,
        steering_method: Optional[str] = None,
    ) -> Iterator[None]:
        active_method = steering_method or self.config.mode
        self.last_relative_residual_mag = 0.0
        self.captured_eager_layer_attns.clear()

        if model is None:
            yield
            return

        handles: List[Any] = []
        try:
            layers = getattr(getattr(model, "model", model), "layers", None)
            if layers is not None and len(layers) > 0:
                num_layers = len(layers)
                target_layers = set(self.get_target_layer_indices(num_layers))
                mid_layer = num_layers // 2
                span_map = state_pkt.span_map or {}
                beta_lc = self.compute_easterbrook_beta(state_pkt.arousal_u)
                allow_decode = self.config.scope == "prefill_and_decode"
                is_large_model = (
                    int(getattr(model.config, "vocab_size", 0)) > 1000
                )

                # For large models (vocab > 1000) on 3.68 GiB GPUs, capture target-layer
                # eager attention weights to CPU and free all 28 layers' 4D GPU tensors
                # immediately so peak VRAM is 1 layer (2 MiB) instead of 28 layers (60 MiB).
                if is_large_model:
                    for all_l_idx in range(num_layers):
                        attn_mod = getattr(layers[all_l_idx], "self_attn", None)
                        if attn_mod is None:
                            continue

                        def _make_eager_saver(is_target: bool) -> Any:
                            def _eager_hook(mod: Any, inp: Any, out: Any) -> Any:
                                if (
                                    isinstance(out, tuple)
                                    and len(out) > 1
                                    and isinstance(out[1], torch.Tensor)
                                ):
                                    aw = out[1]
                                    if is_target and aw.dim() == 4 and aw.size(2) > 1:
                                        q_tail = (
                                            1
                                            if self.config.query_row_scope
                                            == "last_token_only"
                                            else min(16, aw.size(2))
                                        )
                                        avg_k = (
                                            torch.nan_to_num(
                                                aw[0, :, -q_tail:, :]
                                                .detach()
                                                .float(),
                                                nan=0.0,
                                            )
                                            .mean(dim=(0, 1))
                                            .cpu()
                                        )
                                        self.captured_eager_layer_attns.append(avg_k)
                                    return (out[0], None) + out[2:]
                                return out

                            return _eager_hook

                        handles.append(
                            attn_mod.register_forward_hook(
                                _make_eager_saver(all_l_idx in target_layers)
                            )
                        )

                if not self.config.enabled or active_method == "NONE":
                    yield
                    return

                if (
                    active_method == "ACTADD_CAA"
                    and self.calibration is not None
                    and self.calibration.residual_caa_vector is not None
                ):
                    caa_vec = self.calibration.residual_caa_vector[mid_layer]
                    scale_f = (
                        self.config.actadd_coeff * float(state_pkt.financial_stress)
                    )

                    def _actadd_hook(mod: Any, inp: Any, out: Any) -> Any:
                        hidden = out[0] if isinstance(out, tuple) else out
                        if isinstance(hidden, torch.Tensor) and hidden.dim() == 3:
                            if not allow_decode and hidden.size(1) <= 1:
                                return out
                            if abs(scale_f) > 1e-5:
                                v_dev = caa_vec.to(
                                    device=hidden.device, dtype=hidden.dtype
                                )
                                if v_dev.size(-1) == hidden.size(-1):
                                    h_norm = float(
                                        hidden.detach()
                                        .float()
                                        .norm(dim=-1)
                                        .mean()
                                        .item()
                                    )
                                    v_norm = float(
                                        (v_dev.float() * scale_f).norm().item()
                                    )
                                    self.last_relative_residual_mag = round(
                                        v_norm / max(h_norm, 1e-6), 4
                                    )
                                    hidden.add_(v_dev.view(1, 1, -1), alpha=scale_f)
                        return out

                    handles.append(
                        layers[mid_layer].register_forward_hook(_actadd_hook)
                    )

                for l_idx in sorted(target_layers):
                    attn = getattr(layers[l_idx], "self_attn", None)
                    if attn is None:
                        continue

                    q_proj = getattr(attn, "q_proj", None)
                    v_proj = getattr(attn, "v_proj", None)

                    if (
                        q_proj is not None
                        and self.config.enable_easterbrook_beta
                        and active_method
                        in (
                            "OUR_COUPLED_STEER",
                            "MASTER_EQUATION",
                            "SHUFFLED_STATE",
                            "REVERSED_SIGN",
                        )
                        and abs(beta_lc - 1.0) > 1e-3
                    ):
                        def _q_hook(
                            mod: Any, inp: Any, out: torch.Tensor
                        ) -> torch.Tensor:
                            if not isinstance(out, torch.Tensor):
                                return out
                            if not allow_decode and out.size(1) <= 1:
                                return out
                            gain_q = 1.0 + 0.15 * (beta_lc - 1.0)
                            out.mul_(gain_q)
                            return out

                        handles.append(q_proj.register_forward_hook(_q_hook))

                    if (
                        v_proj is not None
                        and self.config.enable_value_steer
                        and active_method
                        in (
                            "OUR_COUPLED_STEER",
                            "MASTER_EQUATION",
                            "PLACEBO_RANDOM",
                            "SHUFFLED_STATE",
                            "REVERSED_SIGN",
                        )
                        and SPAN_INCOMING_MSG in span_map
                    ):
                        def _make_v_hook(layer_i: int) -> Any:
                            def _v_hook(
                                mod: Any, inp: Any, out: torch.Tensor
                            ) -> torch.Tensor:
                                if (
                                    not isinstance(out, torch.Tensor)
                                    or out.size(1) <= 1
                                ):
                                    return out
                                s_m, e_m = span_map[SPAN_INCOMING_MSG]
                                if not (0 <= s_m < e_m <= out.size(1)):
                                    return out

                                sign_m = (
                                    -1.0
                                    if active_method == "REVERSED_SIGN"
                                    else 1.0
                                )
                                gamma_cyn = (
                                    max(
                                        0.0,
                                        0.50 - float(state_pkt.effective_trust_01),
                                    )
                                    * 2.0
                                    * self.config.gamma_cynicism_scale
                                )
                                eta_th = (
                                    float(state_pkt.financial_stress)
                                    * self.config.eta_threat_scale
                                )
                                if gamma_cyn <= 1e-4 and eta_th <= 1e-4:
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
                                        and self.calibration.random_placebo_u
                                        is not None
                                    ):
                                        u_dir = self.calibration.random_placebo_u[
                                            layer_i
                                        ].to(device=out.device, dtype=out.dtype)
                                        v_norm = span_slice.norm(
                                            dim=-1, keepdim=True
                                        ).clamp(min=1e-6)
                                        comb_mag = (
                                            math.sqrt(
                                                gamma_cyn ** 2 + eta_th ** 2
                                            )
                                            * 0.25
                                        )
                                        self.last_relative_residual_mag = round(
                                            comb_mag, 4
                                        )
                                        span_slice.add_(
                                            v_norm * u_dir.view(1, 1, -1),
                                            alpha=comb_mag,
                                        )
                                    else:
                                        u_tr = self.calibration.u_trust[
                                            layer_i
                                        ].to(device=out.device, dtype=out.dtype)
                                        u_th = self.calibration.u_threat[
                                            layer_i
                                        ].to(device=out.device, dtype=out.dtype)
                                        proj_coeff = torch.matmul(
                                            span_slice, u_tr.view(-1, 1)
                                        )
                                        span_slice.sub_(
                                            proj_coeff * u_tr.view(1, 1, -1),
                                            alpha=sign_m * gamma_cyn * 0.35,
                                        )
                                        v_norm = span_slice.norm(
                                            dim=-1, keepdim=True
                                        ).clamp(min=1e-6)
                                        span_slice.add_(
                                            v_norm * u_th.view(1, 1, -1),
                                            alpha=sign_m * eta_th * 0.20,
                                        )
                                        self.last_relative_residual_mag = round(
                                            math.sqrt(
                                                (gamma_cyn * 0.35) ** 2
                                                + (eta_th * 0.20) ** 2
                                            ),
                                            4,
                                        )
                                return out

                            return _v_hook

                        handles.append(
                            v_proj.register_forward_hook(_make_v_hook(l_idx))
                        )

            yield
        finally:
            for h in handles:
                h.remove()


MasterEquationHookEngine = AttentionSteeringLab