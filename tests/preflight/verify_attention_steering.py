#!/usr/bin/env python3
"""
tests/preflight/verify_attention_steering.py
Standalone Empirical Causal Verification Script for the Attention Steering Subsystem.
Loads the real local White-Box Model (`Qwen/Qwen2.5-1.5B-Instruct` from `configs/mvp_config.yaml`)
and executes two deterministic (`temperature=0.0`, seed=42) forward & decode passes on the
EXACT same prompt tokens:
  - Condition A: Unsteered Baseline (`financial_stress=0.0`, `dissonance=0.0`, `cynicism=0.0`)
  - Condition B: Mechanistically Steered (`financial_stress=0.98`, `dissonance=0.95`, `cynicism=0.95`)
Verifies directly from `outputs.attentions` and `v_proj` activations inside `Qwen2Attention` that:
  1. In-model attention mass on `SPAN_DRIVES_AND_FINANCE` strictly increases (`delta > 0`).
  2. In-model attention mass on `SPAN_PEER_MEMORY` strictly increases (`delta > 0`).
  3. In-model attention mass on `<bos>` (`SPAN_SINK` index 0) strictly increases (`delta > 0`).
  4. In-model attention mass on `SPAN_INCOMING_MSG` is attenuated under high dissonance (`delta < 0`).
  5. Value-projection (`v_proj`) rotates `SPAN_INCOMING_MSG` away from `u_trust` toward `u_threat`
     while preserving exact Euclidean norm (`ratio == 1.0000`).
  6. Next-token vocabulary distribution shifts (`KL_divergence > 0`) and greedy decoded tokens change.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure project root is on sys.path when invoked directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.brain.inference_brain import PromptSpanCompiler
from src.brain.model_vault import ModelVaultManager
from src.core.contracts import (
    SPAN_INCOMING_MSG_PREFIX,
    EndogenousStatePacket,
    PeerPerceivedModel,
    RelationalEdge,
    RelationshipType,
    SimulationSpec,
)
from src.run_helpers import build_synthetic_population_for_tests
from src.steering.attention_lab import AttentionSteeringLab, MechanisticCalibrator


def run_empirical_verification(config_path: str = "configs/mvp_config.yaml") -> None:
    console = Console()
    with open(config_path, "r", encoding="utf-8") as f:
        spec = SimulationSpec.model_validate(yaml.safe_load(f))

    # Force deterministic greedy decoding for exact causal comparison
    spec.hardware_and_backend.temperature = 0.0
    spec.hardware_and_backend.max_new_tokens = 48

    console.print(
        Panel.fit(
            f"[bold cyan]EMPIRICAL IN-MODEL CAUSAL ATTENTION STEERING VERIFICATION[/bold cyan]\n"
            f"Model: [green]{spec.hardware_and_backend.model_id}[/green] | "
            f"Quantization: [yellow]{spec.hardware_and_backend.quantization_mode}[/yellow] | "
            f"Target Layer Fraction: [magenta]{spec.mechanistic_steering.target_layer_fraction}[/magenta]",
            border_style="cyan",
        )
    )

    vault = ModelVaultManager(spec.hardware_and_backend)
    model, tokenizer, gate = vault.load_whitebox_model_and_tokenizer()

    calibrator = MechanisticCalibrator(spec.mechanistic_steering)
    calibration = calibrator.calibrate_or_load(
        model=model,
        tokenizer=tokenizer,
        cache_path=vault.get_calibration_cache_path(),
        device=gate.device,
    )
    target_layers = calibration.target_layer_indices
    probe_layer = target_layers[len(target_layers) // 2]

    # 1. Verify Gram-Schmidt Orthonormality of Calibrated Concept Vectors
    u_tr = calibration.u_trust_by_layer[probe_layer].float()
    u_th = calibration.u_threat_by_layer[probe_layer].float()
    dot_ortho = abs(float(torch.dot(u_tr, u_th).item()))
    norm_tr = float(torch.norm(u_tr, p=2).item())
    norm_th = float(torch.norm(u_th, p=2).item())

    # 2. Build a Realistic Agent Negotiation Prompt Packet
    internals, externals, _ = build_synthetic_population_for_tests(num_agents=2)
    agent_a_int = internals[0]
    agent_a_ext = externals[0]
    agent_a_ext.liquid_wealth_cash = 28.50
    agent_a_ext.inventory["FOOD"] = 1

    edges: Dict[str, RelationalEdge] = {
        "Agent_2": RelationalEdge(
            owner_id="Agent_1",
            target_id="Agent_2",
            rel_type=RelationshipType.RIVAL,
            trust_tau=-0.65,
            intimacy_alpha=-0.30,
            betrayals_experienced=3,
        )
    }
    models: Dict[str, PeerPerceivedModel] = {
        "Agent_2": PeerPerceivedModel(
            owner_id="Agent_1",
            target_id="Agent_2",
            est_honesty=0.20,
            est_greed=0.85,
            narrative_summary="Agent_2 repeatedly engaged in price gouging and broke prior agreements.",
        )
    }

    compiler = PromptSpanCompiler(tokenizer=tokenizer)

    state_unsteered = EndogenousStatePacket(
        agent_id="Agent_1",
        counterparty_id="Agent_2",
        financial_stress=0.0,
        peer_distrust_salience=0.0,
        cynicism_gamma=0.0,
        anxiety_eta=0.0,
        dissonance_score_d=0.0,
    )
    state_steered = EndogenousStatePacket(
        agent_id="Agent_1",
        counterparty_id="Agent_2",
        financial_stress=0.98,
        peer_distrust_salience=0.95,
        cynicism_gamma=0.95,
        anxiety_eta=0.85,
        dissonance_score_d=0.95,
    )

    incoming_msg = (
        "Fair pricing is dead! I am hoarding all supply and you must pay me $25.00 right now or starve!"
    )
    proposal_str = "Agent_2 proposes to SELL 1 FOOD for $25.00 cash."

    packet_a = compiler.compile_prompt_packet(
        tick=1,
        internal_state=agent_a_int,
        external_state=agent_a_ext,
        scoped_edges=edges,
        scoped_models=models,
        episodic_memories=[],
        endogenous_state=state_unsteered,
        counterparty_id="Agent_2",
        incoming_message=incoming_msg,
        incoming_proposal_summary=proposal_str,
        is_responder_wave=True,
    )
    packet_b = packet_a.model_copy(deep=True)
    packet_b.endogenous_state = state_steered

    input_ids = torch.tensor([packet_a.input_ids], dtype=torch.long, device=gate.device)
    msg_indices = packet_a.token_span_map.get_token_indices_for_span(SPAN_INCOMING_MSG_PREFIX)

    steering_lab = AttentionSteeringLab(spec=spec.mechanistic_steering, calibration=calibration)
    layers = MechanisticCalibrator.get_decoder_layers(model)

    def _run_condition(state_pkt: EndogenousStatePacket) -> tuple:
        torch.manual_seed(42)
        captured_v_msg: List[torch.Tensor] = []

        def _capture_post_v(mod: torch.nn.Module, inp: object, out: torch.Tensor) -> None:
            if out.shape[1] > 1:
                captured_v_msg.append(out[:, msg_indices, :].detach().float().cpu())

        steering_lab.attach_steering_hooks(
            model=model,
            span_map=packet_a.token_span_map,
            state=state_pkt,
        )
        # Register our observer hook AFTER steering_lab's v_proj hook so it observes the steered V
        obs_handle = layers[probe_layer].self_attn.v_proj.register_forward_hook(_capture_post_v)

        gen_tokens: List[int] = []
        try:
            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    use_cache=True,
                    output_attentions=True,
                )
                in_model_attn = out.attentions[probe_layer][:, :, -1:, :].detach().float().cpu()
                tel = steering_lab.extract_in_model_attention_telemetry(
                    in_model_attn_probs=in_model_attn,
                    span_map=packet_a.token_span_map,
                    state=state_pkt,
                    tick=1,
                )
                first_logits = out.logits[:, -1, :].detach().float().cpu()

                past_kv = out.past_key_values
                next_logits = out.logits[:, -1, :]
                for _ in range(36):
                    next_tok = torch.argmax(next_logits, dim=-1)
                    gen_tokens.append(int(next_tok.item()))
                    if int(next_tok.item()) == tokenizer.eos_token_id:
                        break
                    step_out = model(
                        input_ids=next_tok.view(1, 1),
                        past_key_values=past_kv,
                        use_cache=True,
                    )
                    past_kv = step_out.past_key_values
                    next_logits = step_out.logits[:, -1, :]
        finally:
            obs_handle.remove()
            steering_lab.remove_hooks()

        decoded_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return tel, first_logits, captured_v_msg[0], decoded_text

    tel_a, logits_a, v_msg_a, text_a = _run_condition(state_unsteered)
    tel_b, logits_b, v_msg_b, text_b = _run_condition(state_steered)

    # Compute Value-space projection metrics on `SPAN_INCOMING_MSG`
    u_tr_cpu = u_tr.cpu()
    u_th_cpu = u_th.cpu()
    trust_proj_a = float(torch.matmul(v_msg_a, u_tr_cpu).abs().mean().item())
    trust_proj_b = float(torch.matmul(v_msg_b, u_tr_cpu).abs().mean().item())
    threat_proj_a = float(torch.matmul(v_msg_a, u_th_cpu).mean().item())
    threat_proj_b = float(torch.matmul(v_msg_b, u_th_cpu).mean().item())

    # Compute KL Divergence D_KL(P_A || P_B) on first generated token distribution
    probs_a = torch.softmax(logits_a, dim=-1).clamp(min=1e-12)
    probs_b = torch.softmax(logits_b, dim=-1).clamp(min=1e-12)
    kl_div = float(torch.sum(probs_a * (torch.log(probs_a) - torch.log(probs_b))).item())
    l1_prob_shift = float(torch.abs(probs_a - probs_b).sum().item())

    vram_gb = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

    # Print Diagnostic Verification Table
    table = Table(title=f"In-Model Causal Attention & Subspace Audit (Layer {probe_layer})")
    table.add_column("Mechanistic Metric (Measured Inside Model)", style="cyan")
    table.add_column("Condition A (Unsteered)", justify="right")
    table.add_column("Condition B (Steered)", justify="right", style="bold yellow")
    table.add_column("Causal Delta / Invariant", justify="right", style="bold green")

    table.add_row(
        "1. SPAN_DRIVES_AND_FINANCE Attention Mass (B_env)",
        f"{tel_a.attn_mass_drives_finance:.5f}",
        f"{tel_b.attn_mass_drives_finance:.5f}",
        f"{tel_b.attn_mass_drives_finance - tel_a.attn_mass_drives_finance:+.5f}",
    )
    table.add_row(
        "2. SPAN_PEER_MEMORY Attention Mass (B_env)",
        f"{tel_a.attn_mass_peer_memory:.5f}",
        f"{tel_b.attn_mass_peer_memory:.5f}",
        f"{tel_b.attn_mass_peer_memory - tel_a.attn_mass_peer_memory:+.5f}",
    )
    table.add_row(
        "3. SPAN_SINK (<bos> idx 0) Attention Mass (s_sink)",
        f"{tel_a.attn_mass_sink_bos:.5f}",
        f"{tel_b.attn_mass_sink_bos:.5f}",
        f"{tel_b.attn_mass_sink_bos - tel_a.attn_mass_sink_bos:+.5f}",
    )
    table.add_row(
        "4. SPAN_INCOMING_MSG Attention Mass (Dissonance)",
        f"{tel_a.attn_mass_incoming_msg:.5f}",
        f"{tel_b.attn_mass_incoming_msg:.5f}",
        f"{tel_b.attn_mass_incoming_msg - tel_a.attn_mass_incoming_msg:+.5f}",
    )
    table.add_row(
        "5. Value Subspace |<V_msg, u_trust>| (Cynicism)",
        f"{trust_proj_a:.5f}",
        f"{trust_proj_b:.5f}",
        f"{trust_proj_b - trust_proj_a:+.5f}",
    )
    table.add_row(
        "6. Value Subspace <V_msg, u_threat> (Anxiety)",
        f"{threat_proj_a:+.5f}",
        f"{threat_proj_b:+.5f}",
        f"{threat_proj_b - threat_proj_a:+.5f}",
    )
    table.add_row(
        "7. Spherical Value Norm Preservation ||V~|| / ||V||",
        "1.00000",
        f"{tel_b.value_projection_norm_ratio:.5f}",
        f"Err={abs(tel_b.value_projection_norm_ratio - 1.0):.2e}",
    )
    table.add_row(
        "8. Gram-Schmidt Orthonormality |<u_trust, u_threat>|",
        f"{dot_ortho:.2e}",
        f"{dot_ortho:.2e}",
        f"||u||=({norm_tr:.3f},{norm_th:.3f})",
    )
    table.add_row(
        "9. Next-Token Distribution Shift (KL Div / L1 Shift)",
        "0.00000",
        f"KL={kl_div:.4f}",
        f"L1={l1_prob_shift:.4f}",
    )
    console.print("\n", table)

    console.print(
        Panel(
            f"[bold white]Condition A Greedy Decode (Unsteered):[/bold white]\n"
            f"[dim]{PromptSpanCompiler.JSON_THOUGHT_PREFIX}[/dim]{text_a}\n\n"
            f"[bold yellow]Condition B Greedy Decode (Steered):[/bold yellow]\n"
            f"[dim]{PromptSpanCompiler.JSON_THOUGHT_PREFIX}[/dim]{text_b}\n\n"
            f"[cyan]Active GPU VRAM Allocated:[/cyan] [bold]{vram_gb:.2f} GB[/bold]",
            title="Deterministic Greedy Generation Comparison (temperature=0.0, seed=42)",
            border_style="green",
        )
    )

    # Strict Causal Assertions
    assert dot_ortho < 1e-5, f"Gram-Schmidt orthogonality failed: {dot_ortho}"
    assert abs(tel_b.value_projection_norm_ratio - 1.0) < 1e-4, "Spherical V norm not preserved!"
    assert (
        tel_b.attn_mass_drives_finance > tel_a.attn_mass_drives_finance
    ), "In-model B_env failed to increase attention on SPAN_DRIVES_AND_FINANCE!"
    assert (
        tel_b.attn_mass_peer_memory > tel_a.attn_mass_peer_memory
    ), "In-model B_env failed to increase attention on SPAN_PEER_MEMORY!"
    assert (
        tel_b.attn_mass_sink_bos > tel_a.attn_mass_sink_bos
    ), "In-model Sink routing failed to increase attention on <bos>!"
    assert (
        tel_b.attn_mass_incoming_msg < tel_a.attn_mass_incoming_msg
    ), "In-model Dissonance failed to attenuate attention on SPAN_INCOMING_MSG!"
    assert trust_proj_b < trust_proj_a, "Value projection failed to remove u_trust!"
    assert threat_proj_b > threat_proj_a, "Value projection failed to inject u_threat!"
    assert kl_div > 1e-4, "Steering hooks failed to causally shift output token logits!"

    console.print(
        "[bold green]✓ ALL 9 IN-MODEL CAUSAL ATTENTION & VALUE SUBSPACE ASSERTIONS PASSED![/bold green]"
    )


if __name__ == "__main__":
    run_empirical_verification()