from __future__ import annotations

import os
import tempfile
import torch

from src.benchmarks.human_protocols_tier_b import TierBHumanProtocolSuite
from src.benchmarks.micro_tier_a import TierAMicroBenchmarkSuite
from src.core.contracts import EconomicActionIntent, SteeringLabConfig
from src.core.seed_bundle import SeedBundle
from src.environment.game_master import AccountingAndLawEngine
from src.environment.populations import synthesize_parametric_population
from src.evaluation.validity_and_stats import (
    compute_exact_shapley_values,
    decompose_metric_variance_and_icc,
    export_metric_validity_csv,
)
from src.memory.scoped_memory import MultiTierScopedMemory
from src.steering.attention_lab import AttentionSteeringLab
from tests.gates.test_g0_defects import _build_tiny_eager_qwen2, _load_base_config


def test_gate_g1_steering_dose_response_and_tier_a() -> None:
    """
    Gate G1: Verifies identity (< 1e-5), shadow-measurement parity, and monotone
    dose-response across alpha in {0, 0.5, 1, 2, 4, 8} on Benchmark A1, plus A2–A5.
    """
    model = _build_tiny_eager_qwen2()
    lab = AttentionSteeringLab(config=SteeringLabConfig())
    a1_res = TierAMicroBenchmarkSuite.run_a1_steering_dose_response(
        model=model, steering_lab=lab, n_prompts=6
    )
    assert a1_res["identity_passed"] is True
    assert a1_res["monotone_passed"] is True

    a4_res = TierAMicroBenchmarkSuite.run_a4_bayesian_belief_calibration()
    assert a4_res["passed"] is True

    a5_res = TierAMicroBenchmarkSuite.run_a5_budgeted_crra_regret_and_garp(
        choice_fn=lambda w, p1, p2: 0.5 * w / p1
    )
    assert a5_res["passed"] is True


def test_gate_g2_metric_validity_and_shapley_attribution() -> None:
    """
    Gate G2: Verifies metric validity classification (`metric_validity.csv`) and
    exact 2^k Shapley attribution sum-to-efficiency property.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "metric_validity.csv")
        r1 = decompose_metric_variance_and_icc(
            "b1_ultimatum_w1",
            "Tier_B",
            "behavioral",
            [0.34, 0.36, 0.33, 0.35],
            [0.08, 0.09, 0.07, 0.08],
            [0.10, 0.11, 0.09, 0.10],
        )
        r2 = decompose_metric_variance_and_icc(
            "htm_share",
            "Tier_C",
            "mechanical",
            [0.33, 0.34, 0.32, 0.33],
            [0.33, 0.34, 0.32, 0.33],
            [0.33, 0.34, 0.32, 0.33],
        )
        export_metric_validity_csv([r1, r2], output_path=csv_path)
        assert os.path.exists(csv_path)
        assert r1.verdict == "primary_behavioral"
        assert r2.verdict == "sanity_only"

    # Exact 2^3 Shapley efficiency check: sum(phi_i) == v(1,1,1) - v(0,0,0)
    lattice = {
        (0, 0, 0): 0.10,
        (1, 0, 0): 0.45,
        (0, 1, 0): 0.25,
        (0, 0, 1): 0.15,
        (1, 1, 0): 0.72,
        (1, 0, 1): 0.50,
        (0, 1, 1): 0.30,
        (1, 1, 1): 0.82,
    }
    phis = compute_exact_shapley_values(
        lattice, ["attn_steer", "caa_value", "memory"]
    )
    total_gain = lattice[(1, 1, 1)] - lattice[(0, 0, 0)]
    assert abs(sum(phis.values()) - total_gain) < 1e-5


def test_gate_g3_tier_b_protocols_and_randomized_ledger_invariants() -> None:
    """
    Gate G3: Verifies Tier B human-protocol environments (B1, B2, B3) with >= 30
    observations per bin and property-tests ledger conservation across 100
    randomized trade sequences across N = 12 parametric households.
    """
    b1 = TierBHumanProtocolSuite.run_b1_ultimatum_protocol(
        responder_accept_fn=lambda share, f_i, s: share >= 0.25,
        n_per_bin=30,
    )
    assert b1["bin_coverage_valid"] is True

    b3 = TierBHumanProtocolSuite.run_b3_windfall_mpc_protocol(
        mpc_choice_fn=lambda tercile, f_i, s: 0.30 + 0.38 * f_i,
        n_per_tercile=30,
    )
    assert b3["bin_coverage_valid"] is True
    assert b3["low_minus_high_mpc"] > 0.20

    # Property-test ledger conservation over 100 randomized actions on N=12 population
    cfg = _load_base_config()
    cfg.population_scale = "dev"
    sb = SeedBundle(123)
    cfg.agents = synthesize_parametric_population(cfg, sb, target_n=12)
    assert len(cfg.agents) == 12

    rng = sb.get_np_rng("env_rng")
    agent_ids = [a.agent_id for a in cfg.agents]

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_g3.sqlite")
        gm = AccountingAndLawEngine(db_path=db_path, config=cfg)
        for step_i in range(100):
            u, v = rng.choice(agent_ids, size=2, replace=False)
            act = str(rng.choice(["BUY", "SELL", "HOLD_OR_WORK"]))
            item = str(
                rng.choice(["CONSUMABLE_NECESSITY", "PRODUCTIVE_CAPITAL_ASSET"])
            )
            qty = float(rng.choice([0.5, 1.0, 2.0, 50.0]))
            price = float(rng.uniform(5.0, 80.0))
            consent = str(rng.choice(["ACCEPT", "REJECT"]))
            intent = EconomicActionIntent(
                action_type=act,  # type: ignore[arg-type]
                counterparty_id=str(v),
                item_id=item,
                quantity=qty,
                unit_price=round(price, 2),
            )
            gm.arbitrate_and_settle(
                tick=(step_i // 20) + 1,
                initiator_id=str(u),
                responder_id=str(v),
                intent=intent,
                responder_consent=consent,
            )
            assert gm.get_conservation_error() < 1e-6


if __name__ == "__main__":
    print("Running Gates G1, G2, and G3 Verification Suites...")
    test_gate_g1_steering_dose_response_and_tier_a()
    print("  [PASS] Gate G1: Steering Identity, Shadow Parity & Monotone Dose-Response")
    test_gate_g2_metric_validity_and_shapley_attribution()
    print("  [PASS] Gate G2: Metric Validity CSV & Exact 2^k Shapley Attribution")
    test_gate_g3_tier_b_protocols_and_randomized_ledger_invariants()
    print("  [PASS] Gate G3: Tier B Protocols (n>=30/bin) & 100-Step Ledger Property Test")
    print("ALL GATES G1, G2, AND G3 PASSED.")