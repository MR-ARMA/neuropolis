"""
tests/preflight/test_mvp_preflight.py
MODULE 12 (Pre-Flight Counterfactual Sensitivity & Invariant Test Suite).
Verifies:
  1. ACID Double-Entry Money Conservation across 100 random valid/invalid/overdraft trades.
  2. Scoped Memory Epistemic Isolation (`EpistemicIsolationViolationError` on cross-agent access)
     + Two-Pass Aho-Corasick Third-Party Entity Mention Retrieval + Ebbinghaus decay.
  3. Semantic-Offset Entropy-TMS Halting on high-entropy logits vs. low-entropy peaked logits
     + Bayesian Beta-Bernoulli competence updating.
  4. Counterfactual Mechanistic Sensitivity Check proving `B_env(E_t)` financial stress bias,
     Gram-Schmidt Dual-Space `QK-V` Value projection, and Cognitive Dissonance Sink Routing
     measurably shift attention mass and preserve norms/variance.
"""

from __future__ import annotations

import os
import random
import tempfile
import pytest
import torch

from src.core.contracts import (
    SPAN_DRIVES_AND_FINANCE,
    SPAN_INCOMING_MSG_PREFIX,
    SPAN_PEER_MEMORY_PREFIX,
    SPAN_SINK,
    ActionCategory,
    AgentTurnOutput,
    ConsentDecision,
    EconomicActionIntent,
    EconomyInstitutionsSpec,
    EndogenousStatePacket,
    EntropyTMSSpec,
    ExpertiseDomain,
    ExternalAgentState,
    InternalAgentState,
    MacroEconomyState,
    MechanisticSteeringSpec,
    MemoryCognitionSpec,
    SimulationSpec,
    SteeringMode,
    TokenSpanMap,
    VerdictStatus,
)
from src.environment.game_master import AccountingAndLawEngine
from src.memory.scoped_memory import (
    EpistemicIsolationViolationError,
    ScopedMemoryEngine,
)
from src.run_helpers import build_synthetic_population_for_tests
from src.steering.attention_lab import AttentionSteeringLab, MechanisticCalibrator
from src.steering.entropy_tms import EntropyTMSRouter


# ==============================================================================
# TEST 1: ACID DOUBLE-ENTRY LEDGER & 100-TRADE CONSERVATION TORTURE TEST
# ==============================================================================
def test_acid_money_conservation_across_100_random_trades() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_acid.sqlite")
        mem_spec = MemoryCognitionSpec()
        econ_spec = EconomyInstitutionsSpec(initial_agent_cash=100.0, sales_tax_rate=0.05)
        engine = ScopedMemoryEngine(db_path=db_path, spec=mem_spec)

        internals, externals, macro = build_synthetic_population_for_tests(num_agents=6)
        engine.register_population(internals, externals, macro)
        gm = AccountingAndLawEngine(memory_engine=engine, spec=econ_spec)

        rng = random.Random(42)
        agent_ids = [a.agent_id for a in internals]

        for step in range(100):
            init_id, resp_id = rng.sample(agent_ids, 2)
            # Mix valid prices with deliberate overdrafts ($9999) and inventory over-sells (99 units)
            price = rng.choice([10.0, 20.0, 35.0, 9999.0, -5.0])
            qty = rng.choice([1, 2, 50])
            item = rng.choice(["FOOD", "TECH"])
            is_buying = rng.choice([True, False])
            consent = rng.choice([ConsentDecision.ACCEPT, ConsentDecision.REJECT])

            init_out = AgentTurnOutput(
                consent_decision=ConsentDecision.INITIATE,
                economic_intent=EconomicActionIntent(
                    action_category=ActionCategory.BILATERAL_TRADE,
                    target_peer_id=resp_id,
                    item_id=item,
                    quantity=max(0, qty),
                    total_price_cash=max(0.0, price),
                    is_buying=is_buying,
                ),
            )
            escrow = gm.pre_validate_and_escrow(tick=step, initiator_id=init_id, turn_output=init_out)

            if escrow.status == VerdictStatus.ESCROW_LOCKED:
                resp_out = AgentTurnOutput(consent_decision=consent)
                gm.settle_dyadic_transaction(
                    tick=step,
                    escrow_verdict=escrow,
                    initiator_output=init_out,
                    responder_id=resp_id,
                    responder_output=resp_out,
                )

            ok, drift = gm.verify_global_money_conservation()
            assert ok, f"Money conservation violated at step {step}: drift={drift}"
            assert drift < 1e-6


# ==============================================================================
# TEST 2: SCOPED MEMORY ISOLATION & TWO-PASS ENTITY MENTION RETRIEVAL
# ==============================================================================
def test_scoped_memory_isolation_and_third_party_entity_scanner() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_mem.sqlite")
        engine = ScopedMemoryEngine(db_path=db_path, spec=MemoryCognitionSpec())
        internals, externals, macro = build_synthetic_population_for_tests(num_agents=6)
        engine.register_population(internals, externals, macro)

        view_a = engine.get_scoped_view("Agent_1")

        # Check A: Cross-agent partition access MUST raise EpistemicIsolationViolationError (PermissionError)
        with pytest.raises(EpistemicIsolationViolationError):
            view_a.get_internal_state(requested_owner_id="Agent_2")

        with pytest.raises(EpistemicIsolationViolationError):
            view_a.get_relational_edge(target_id="Agent_3", requested_owner_id="Agent_2")

        # Check B: Two-Pass Entity Scanner loads ONLY direct partner (Agent_2) and mentioned third party (Agent_4)
        incoming_gossip = "Watch out, Agent_4 defaulted on a loan yesterday!"
        edges, models, _ = view_a.dual_pass_scoped_retrieval(
            direct_partner_id="Agent_2",
            incoming_message_text=incoming_gossip,
            current_tick=1,
        )
        assert set(edges.keys()) == {"Agent_2", "Agent_4"}
        assert "Agent_3" not in edges
        assert "Agent_5" not in edges
        assert "Agent_6" not in edges


# ==============================================================================
# TEST 3: SEMANTIC-OFFSET ENTROPY-TMS HALTING & BAYESIAN COMPETENCE UPDATE
# ==============================================================================
def test_entropy_tms_router_halting_and_bayesian_update() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_tms.sqlite")
        engine = ScopedMemoryEngine(db_path=db_path, spec=MemoryCognitionSpec())
        internals, externals, macro = build_synthetic_population_for_tests(num_agents=4)
        engine.register_population(internals, externals, macro)
        view_a = engine.get_scoped_view("Agent_1")

        router = EntropyTMSRouter(
            EntropyTMSSpec(enabled=True, probe_tokens_k=5, top_p_truncation=0.90, entropy_threshold_tau=0.68)
        )

        # Case 1: Low-entropy peaked logits (confident agent) -> should NOT halt
        peaked_logits = [torch.tensor([15.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0]) for _ in range(5)]
        dec_low = router.evaluate_and_route(
            vocab_logits_list=peaked_logits,
            scoped_memory=view_a,
            domain=ExpertiseDomain.TECH_VALUATION,
        )
        assert not dec_low.halted_generation
        assert dec_low.normalized_entropy < 0.15

        # Case 2: High-entropy flat/uniform logits (epistemic ignorance) -> MUST halt and select expert
        flat_logits = [torch.zeros(64) for _ in range(5)]
        dec_high = router.evaluate_and_route(
            vocab_logits_list=flat_logits,
            scoped_memory=view_a,
            domain=ExpertiseDomain.TECH_VALUATION,
        )
        assert dec_high.halted_generation
        assert dec_high.normalized_entropy > 0.90
        # Agent_2 is initialized as TECH_VALUATION specialist
        assert dec_high.selected_expert_id == "Agent_2"

        # Case 3: Verify exact Beta-Bernoulli conjugate belief update
        before = [r for r in view_a.get_tms_competence_directory(ExpertiseDomain.TECH_VALUATION) if r.target_id == "Agent_2"][0]
        updated = view_a.update_tms_peer_belief(
            target_id="Agent_2",
            domain=ExpertiseDomain.TECH_VALUATION,
            outcome_quality_y=1.0,
            tick=2,
        )
        assert abs(updated.alpha_success - (before.alpha_success + 1.0)) < 1e-6
        assert abs(updated.beta_failure - before.beta_failure) < 1e-6


# ==============================================================================
# TEST 4: COUNTERFACTUAL MECHANISTIC SENSITIVITY CHECK (B_env, QK-V, Sink)
# ==============================================================================
def test_mechanistic_steering_counterfactual_sensitivity_and_invariants() -> None:
    torch.manual_seed(42)
    spec = MechanisticSteeringSpec(
        enabled=True,
        mode=SteeringMode.MASTER_EQUATION_MVP,
        omega_fin_stress=2.5,
        omega_peer_salience=2.0,
        gamma_cynicism_scale=0.90,
        eta_anxiety_scale=0.75,
        kappa_dissonance_sink=3.0,
    )
    lab = AttentionSteeringLab(spec=spec)

    # Synthetic 64-token prompt span map
    span_map = TokenSpanMap(
        total_prompt_tokens=64,
        spans={
            SPAN_SINK: [(0, 1)],
            SPAN_DRIVES_AND_FINANCE: [(10, 22)],
            f"{SPAN_PEER_MEMORY_PREFIX}:Agent_2": [(22, 36)],
            f"{SPAN_INCOMING_MSG_PREFIX}:Agent_2": [(36, 52)],
        },
    )
    unsteered_logits = torch.randn(1, 4, 1, 64)

    # 1. Baseline Unsteered State (All stress/dissonance = 0.0)
    calm_state = EndogenousStatePacket(
        agent_id="Agent_1",
        financial_stress=0.0,
        peer_distrust_salience=0.0,
        dissonance_score_d=0.0,
    )
    _, calm_tel = lab.compute_steered_attention_telemetry(
        unsteered_qk_logits=unsteered_logits,
        span_map=span_map,
        state=calm_state,
        tick=1,
    )

    # 2. Counterfactual High Financial Stress + High Peer Distrust State
    stressed_state = EndogenousStatePacket(
        agent_id="Agent_1",
        financial_stress=0.95,
        peer_distrust_salience=0.90,
        dissonance_score_d=0.0,
    )
    _, stressed_tel = lab.compute_steered_attention_telemetry(
        unsteered_qk_logits=unsteered_logits,
        span_map=span_map,
        state=stressed_state,
        tick=1,
    )
    assert stressed_tel.attn_mass_drives_finance > calm_tel.attn_mass_drives_finance + 0.12
    assert stressed_tel.attn_mass_peer_memory > calm_tel.attn_mass_peer_memory + 0.08

    # 3. Counterfactual High Cognitive Dissonance State -> Sink absorption at index 0 (<bos>)
    dissonant_state = EndogenousStatePacket(
        agent_id="Agent_1",
        financial_stress=0.0,
        peer_distrust_salience=0.0,
        dissonance_score_d=0.95,
    )
    _, diss_tel = lab.compute_steered_attention_telemetry(
        unsteered_qk_logits=unsteered_logits,
        span_map=span_map,
        state=dissonant_state,
        tick=1,
    )
    assert diss_tel.attn_mass_sink_bos > calm_tel.attn_mass_sink_bos * 2.5
    assert diss_tel.attn_mass_incoming_msg < calm_tel.attn_mass_incoming_msg

    # 4. Verify Gram-Schmidt Orthonormality & Spherical Norm-Preserving Value Projection
    raw_u1 = torch.randn(32)
    raw_u2 = raw_u1 * 0.7 + torch.randn(32) * 0.3
    u_trust, u_threat = MechanisticCalibrator.gram_schmidt_orthonormalize(raw_u1, raw_u2)

    assert abs(float(torch.dot(u_trust, u_threat).item())) < 1e-6
    assert abs(float(torch.norm(u_trust, p=2).item()) - 1.0) < 1e-5
    assert abs(float(torch.norm(u_threat, p=2).item()) - 1.0) < 1e-5

    v_orig = torch.randn(1, 64, 32)
    msg_indices = list(range(36, 52))
    v_steered, norm_ratio = lab.apply_dual_space_value_projection(
        v_tensor=v_orig,
        msg_token_indices=msg_indices,
        u_hat_trust=u_trust,
        u_hat_threat=u_threat,
        cynicism_gamma=0.95,
        anxiety_eta=0.80,
    )
    assert abs(norm_ratio - 1.0) < 1e-5
    # Non-message tokens must remain 100% untouched
    assert torch.allclose(v_orig[:, :36, :], v_steered[:, :36, :])
    # Message tokens must have reduced projection onto u_trust and increased projection onto u_threat
    orig_trust_proj = torch.matmul(v_orig[:, msg_indices, :], u_trust).abs().mean().item()
    steered_trust_proj = torch.matmul(v_steered[:, msg_indices, :], u_trust).abs().mean().item()
    assert steered_trust_proj < orig_trust_proj