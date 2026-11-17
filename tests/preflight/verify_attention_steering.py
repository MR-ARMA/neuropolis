from __future__ import annotations

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.brain.model_vault import ModelVaultManager
from src.core.contracts import (
    EndogenousStatePacket,
    SPAN_APPRAISAL_AND_TOM,
    SPAN_CORE_BELIEF,
    SPAN_DRIVES_AND_FINANCE,
    SPAN_INCOMING_MSG,
    SPAN_PEER_MEMORY,
    SPAN_SELF_IDENTITY,
    SPAN_SINK,
    SteeringLabConfig,
)
from src.orchestration.turn_orchestrator import PromptSpanCompiler
from src.steering.attention_lab import AttentionSteeringLab


def _build_tiny_eager_qwen2() -> Qwen2ForCausalLM:
    torch.manual_seed(42)
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        _attn_implementation="eager",
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def test_1_real_in_model_attention_bias_and_placebo_control() -> None:
    """
    Verifies that Mechanism A (`ATTN_BIAS`) genuinely shifts the model's actual
    post-softmax `outputs.attentions` tensor (`attention_measured_from_model == True`)
    onto `SPAN_DRIVES_AND_FINANCE` and `SPAN_PEER_MEMORY`, beating both `NONE`
    and `PLACEBO_RANDOM` (Control C5).
    """
    model = _build_tiny_eager_qwen2()
    calib = ModelVaultManager.build_synthetic_calibration_for_tests(
        num_layers=4, num_heads=4, num_kv_heads=2, head_dim=16, hidden_size=64
    )
    steer_cfg = SteeringLabConfig(
        enabled=True,
        mode="OUR_COUPLED_STEER",
        omega_1_drives_finance=3.20,
        omega_2_peer_memory=2.20,
    )
    lab = AttentionSteeringLab(config=steer_cfg, calibration=calib)

    seq_len = 48
    input_ids = torch.randint(1, 200, (1, seq_len))
    span_map = {
        SPAN_SINK: (0, 1),
        SPAN_SELF_IDENTITY: (1, 8),
        SPAN_DRIVES_AND_FINANCE: (8, 18),
        SPAN_CORE_BELIEF: (18, 26),
        SPAN_PEER_MEMORY: (26, 34),
        SPAN_APPRAISAL_AND_TOM: (34, 40),
        SPAN_INCOMING_MSG: (40, 48),
    }

    pkt_stressed = EndogenousStatePacket(
        agent_id="Agent_01",
        financial_stress=0.95,
        scarcity_deficit_f=0.95,
        effective_trust_01=0.10,
        arousal_u=0.45,
        span_map=span_map,
    )

    with torch.inference_mode():
        # 1. Unsteered pass (NONE)
        mask_none = lab.build_additive_4d_logit_bias_mask(
            seq_len, pkt_stressed, "NONE", input_ids.device, torch.float32
        )
        out_none = model(
            input_ids=input_ids,
            attention_mask=mask_none,
            output_attentions=True,
            return_dict=True,
        )
        tel_none = lab.measure_attention_from_model_tensors(
            out_none.attentions, pkt_stressed, "NONE"
        )

        # 2. Placebo Random pass (PLACEBO_RANDOM - Control C5)
        mask_placebo = lab.build_additive_4d_logit_bias_mask(
            seq_len, pkt_stressed, "PLACEBO_RANDOM", input_ids.device, torch.float32
        )
        out_placebo = model(
            input_ids=input_ids,
            attention_mask=mask_placebo,
            output_attentions=True,
            return_dict=True,
        )
        tel_placebo = lab.measure_attention_from_model_tensors(
            out_placebo.attentions, pkt_stressed, "PLACEBO_RANDOM"
        )

        # 3. Coupled Steering pass (OUR_COUPLED_STEER)
        mask_steered = lab.build_additive_4d_logit_bias_mask(
            seq_len, pkt_stressed, "OUR_COUPLED_STEER", input_ids.device, torch.float32
        )
        out_steered = model(
            input_ids=input_ids,
            attention_mask=mask_steered,
            output_attentions=True,
            return_dict=True,
        )
        tel_steered = lab.measure_attention_from_model_tensors(
            out_steered.attentions, pkt_stressed, "OUR_COUPLED_STEER"
        )

    assert tel_steered.attention_measured_from_model is True
    fin_none = tel_none.span_attention_distribution[SPAN_DRIVES_AND_FINANCE]
    fin_placebo = tel_placebo.span_attention_distribution[SPAN_DRIVES_AND_FINANCE]
    fin_steered = tel_steered.span_attention_distribution[SPAN_DRIVES_AND_FINANCE]

    assert fin_steered > fin_none + 0.15, (
        f"Real model attention on finance span did not increase sufficiently: {fin_steered} vs {fin_none}"
    )
    assert fin_steered > fin_placebo + 0.10, (
        f"Coupled steering did not beat matched-norm random placebo: {fin_steered} vs {fin_placebo}"
    )


def test_2_value_subspace_caa_projection_span_locality() -> None:
    """
    Verifies that Mechanism B (`VALUE_STEER`) modifies `v_proj` activations
    strictly on `SPAN_INCOMING_MSG` tokens using the calibrated CAA vectors
    while leaving tokens outside `SPAN_INCOMING_MSG` untouched.
    """
    model = _build_tiny_eager_qwen2()
    calib = ModelVaultManager.build_synthetic_calibration_for_tests(
        num_layers=4, num_heads=4, num_kv_heads=2, head_dim=16, hidden_size=64
    )
    steer_cfg = SteeringLabConfig(enabled=True, mode="OUR_COUPLED_STEER")
    lab = AttentionSteeringLab(config=steer_cfg, calibration=calib)

    seq_len = 32
    input_ids = torch.randint(1, 200, (1, seq_len))
    span_map = {
        SPAN_SINK: (0, 1),
        SPAN_DRIVES_AND_FINANCE: (1, 16),
        SPAN_INCOMING_MSG: (20, 30),
    }
    pkt = EndogenousStatePacket(
        agent_id="Agent_01",
        financial_stress=0.85,
        effective_trust_01=0.10,
        span_map=span_map,
    )

    v_unsteered = {}
    v_steered = {}

    def _cap_unsteered(mod, inp, out):
        v_unsteered["v"] = out.detach().clone()

    def _cap_steered(mod, inp, out):
        v_steered["v"] = out.detach().clone()

    target_layer = model.model.layers[1].self_attn.v_proj
    h1 = target_layer.register_forward_hook(_cap_unsteered)
    with torch.inference_mode():
        model(input_ids=input_ids)
    h1.remove()

    with lab.attach_hooks(model, pkt, steering_method="OUR_COUPLED_STEER"):
        h2 = target_layer.register_forward_hook(_cap_steered)
        with torch.inference_mode():
            model(input_ids=input_ids)
        h2.remove()

    diff_outside = (v_steered["v"][:, :20, :] - v_unsteered["v"][:, :20, :]).abs().max().item()
    diff_inside = (v_steered["v"][:, 20:30, :] - v_unsteered["v"][:, 20:30, :]).abs().max().item()

    assert diff_outside < 1e-6, f"Value steering leaked outside SPAN_INCOMING_MSG: {diff_outside}"
    assert diff_inside > 1e-3, f"Value steering failed to modify SPAN_INCOMING_MSG: {diff_inside}"


def test_3_easterbrook_inverted_u_curve() -> None:
    """
    Verifies that Mechanism C (`EASTERBROOK_BETA`) follows a genuine inverted-U
    curve in arousal U_{i,t}: peaking at U_opt = 0.45 and decreasing under both
    low arousal (U = 0.05) and high panic overload (U = 0.98).
    """
    lab = AttentionSteeringLab(config=SteeringLabConfig())
    b_low = lab.compute_easterbrook_beta(0.05)
    b_opt = lab.compute_easterbrook_beta(0.45)
    b_high = lab.compute_easterbrook_beta(0.98)

    assert b_opt > b_low + 0.15
    assert b_opt > b_high + 0.25


if __name__ == "__main__":
    print("Running White-Box Attention & CAA Representation Steering Verification Suite...")
    test_1_real_in_model_attention_bias_and_placebo_control()
    print("  [PASS] Check 1: Genuine In-Model Attention Measurement & Placebo Random Control (C5)")
    test_2_value_subspace_caa_projection_span_locality()
    print("  [PASS] Check 2: Rank-1 CAA Value Projection & Strict Span Locality")
    test_3_easterbrook_inverted_u_curve()
    print("  [PASS] Check 3: True Inverted-U Easterbrook Cue-Utilization Beta Curve")
    print("ALL ATTENTION STEERING CHECKS PASSED.")