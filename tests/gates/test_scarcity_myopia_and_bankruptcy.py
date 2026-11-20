from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import torch
import yaml

from src.brain.inference_brain import DualBackendInferenceBrain
from src.brain.model_vault import ModelVaultManager
from src.core.contracts import (
    EconomicActionIntent,
    EndogenousStatePacket,
    SimulationConfig,
)
from src.environment.bilateral_market import BilateralMarketEngine
from src.environment.game_master import AccountingAndLawEngine
from src.steering.attention_lab import AttentionSteeringLab
from tests.gates.test_g0_defects import _build_tiny_eager_qwen2


def _load_cfg() -> SimulationConfig:
    with open("configs/mvp_config.yaml", "r", encoding="utf-8") as f:
        cfg = SimulationConfig(**yaml.safe_load(f))
    cfg.experimental_design.randomize_agent_identities = False
    return cfg


def test_1_payday_loan_sfc_and_next_tick_debt_promotion() -> None:
    cfg = _load_cfg()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "payday.sqlite")
        gm = AccountingAndLawEngine(db_path=db_path, config=cfg)

        pre_cash = gm.get_agent_ledger("Agent_06")["liquid_wealth_cash"]
        pre_treas = gm.get_public_treasury_balance()
        base_ob = gm.get_agent_ledger("Agent_06")["periodic_obligation_r"]

        loan_intent = EconomicActionIntent(
            action_type="TAKE_PAYDAY_LOAN",
            counterparty_id="PublicTreasury",
            item_id="CASH",
            quantity=30.0,
            unit_price=-40.0,
        )
        verdict = gm.arbitrate_and_settle(
            tick=1,
            initiator_id="Agent_06",
            responder_id="PublicTreasury",
            intent=loan_intent,
            responder_consent="ACCEPT",
        )
        assert verdict.status == "APPROVED"
        assert abs(gm.get_agent_ledger("Agent_06")["liquid_wealth_cash"] - (pre_cash + 30.0)) < 1e-4
        assert abs(gm.get_public_treasury_balance() - (pre_treas - 30.0)) < 1e-4
        assert gm.get_conservation_error() < 1e-9

        # End of Tick 1 upkeep: pays Tick-1 obligation, promotes $40 debt into Tick-2 periodic_obligation_r
        gm.execute_end_of_tick_upkeep_and_fiscal_state(tick=1)
        assert gm.get_conservation_error() < 1e-9
        tick2_ob = gm.get_agent_ledger("Agent_06")["periodic_obligation_r"]
        assert abs(tick2_ob - (base_ob + 40.0)) < 1e-4

        # End of Tick 2 upkeep: pays base_ob + $40, resets Tick-3 obligation back to base_ob
        gm.execute_end_of_tick_upkeep_and_fiscal_state(tick=2)
        assert gm.get_conservation_error() < 1e-9
        tick3_ob = gm.get_agent_ledger("Agent_06")["periodic_obligation_r"]
        assert abs(tick3_ob - base_ob) < 1e-4


def test_2_hard_subsistence_bankruptcy_seizure_and_exclusion() -> None:
    cfg = _load_cfg()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "bankrupt.sqlite")
        gm = AccountingAndLawEngine(db_path=db_path, config=cfg)
        mkt = BilateralMarketEngine(config=cfg, game_master=gm)

        # Force Agent_06 into acute starvation (<0.5 food) and illiquidity (<0.5 * p_c)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                UPDATE agents_external_ledger
                SET liquid_wealth_cash = 2.0,
                    inventory_json = ?
                WHERE agent_id = 'Agent_06';
                """,
                (json.dumps({"CONSUMABLE_NECESSITY": 0.2, "PRODUCTIVE_CAPITAL_ASSET": 1.0}),),
            )
            conn.commit()
        gm.initial_money_supply_m0 = gm._compute_current_total_money()

        upkeep = gm.execute_end_of_tick_upkeep_and_fiscal_state(tick=1)
        assert "Agent_06" in upkeep["bankrupt_agents"]
        assert gm.get_conservation_error() < 1e-9

        dead_row = gm.get_agent_ledger("Agent_06")
        assert dead_row["legal_status"] == "BANKRUPT"
        assert dead_row["liquid_wealth_cash"] == 0.0
        assert dead_row["inventory"]["PRODUCTIVE_CAPITAL_ASSET"] == 0.0

        # Dead agent must be excluded from partner matching
        pairs = mkt.match_conflict_free_pairs(
            tick=2,
            agent_ids=[a.agent_id for a in cfg.agents],
            effective_trust_matrix={},
        )
        matched_ids = {aid for pair in pairs for aid in pair}
        assert "Agent_06" not in matched_ids


def test_3_option_f_desperation_masking_and_6option_prior_padding() -> None:
    cfg = _load_cfg()
    model = _build_tiny_eager_qwen2()
    calib = ModelVaultManager.build_synthetic_calibration_for_tests()
    calib.option_prior_abcde = [0.1, -0.1, 0.0, 0.0, 0.0]  # 5-element legacy prior

    class _CharTok:
        def __call__(self, text: str, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)}

        def encode(self, s: str, add_special_tokens: bool = False):
            return [ord(s.strip()[0]) % 200]

        def decode(self, ids, skip_special_tokens: bool = True):
            return "F"

    lab = AttentionSteeringLab(config=cfg.steering_spec, calibration=calib)
    brain = DualBackendInferenceBrain(
        config=cfg, model=model, tokenizer=_CharTok(), steering_lab=lab
    )
    logits = torch.zeros(256)
    logits[ord("F") % 200] = 15.0  # Strongly favor Option F
    pkt = EndogenousStatePacket(agent_id="Agent_06", financial_stress=0.95)
    gen = torch.Generator().manual_seed(42)

    # Rich agent (own_cash=$100 > $20): Option F MUST be masked to -inf
    rich_prompt = (
        "Phase=INITIATOR_WAVE_1 | Liquid_Cash=$100.00 | Consumable_Stock=0.5 | "
        "Capital_Asset_Stock=0.0 | Counterparty_Consumable_Stock=0.5 | "
        "Counterparty_Capital_Stock=0.0 | Last_Clearing_Price_Consumable=$12.00"
    )
    raw_rich, _ = brain._compute_logit_scored_decision(
        logits, rich_prompt, "Agent_01", 0.0, gen, pkt, "NONE"
    )
    assert json.loads(raw_rich)["action"] != "TAKE_PAYDAY_LOAN"

    # Desperate agent (own_cash=$10 <= $20): Option F unmasked -> TAKE_PAYDAY_LOAN
    poor_prompt = rich_prompt.replace("Liquid_Cash=$100.00", "Liquid_Cash=$10.00")
    raw_poor, _ = brain._compute_logit_scored_decision(
        logits, poor_prompt, "Agent_01", 0.0, gen, pkt, "NONE"
    )
    parsed_poor = json.loads(raw_poor)
    assert parsed_poor["action"] == "TAKE_PAYDAY_LOAN"
    assert parsed_poor["item"] == "CASH"
    assert parsed_poor["qty"] == 30.0
    assert parsed_poor["price"] == -40.0


if __name__ == "__main__":
    print("Running Scarcity Myopia, Payday Loan & Hard Subsistence Bankruptcy Tests...")
    test_1_payday_loan_sfc_and_next_tick_debt_promotion()
    print("  [PASS] Test 1: Payday Loan (+$30 Cash, +$40 Next-Tick Debt) & SFC_Err < 1e-9")
    test_2_hard_subsistence_bankruptcy_seizure_and_exclusion()
    print("  [PASS] Test 2: Hard Subsistence Death, Treasury Seizure (SFC_Err < 1e-9) & Matching Exclusion")
    test_3_option_f_desperation_masking_and_6option_prior_padding()
    print("  [PASS] Test 3: Option F Desperation Masking (cash <= $20) & 6-Option Prior Padding")
    print("ALL SCARCITY MYOPIA & BANKRUPTCY TESTS PASSED.")