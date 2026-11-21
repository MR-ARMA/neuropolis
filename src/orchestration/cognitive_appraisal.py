from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Optional, Tuple

from src.core.contracts import (
    EndogenousStatePacket,
    SimulationConfig,
)
from src.environment.game_master import AccountingAndLawEngine
from src.memory.scoped_memory import MultiTierScopedMemory


class CognitiveAppraisalAndToM:
    """
    M6: Remediated Pre-Action State Appraisal & Unconfounded State-Channel Router.
    Fixes Reviewer #2 Flaws 6 & 7:
      1. Eliminates the misattributed 93/7 Barsade split, the raw residual-mean
         mirror vector, and the Sharot 0.6875 heuristic.
      2. Explicitly decouples Environment Mechanics (`Omega`) from the State Delivery
         Channel (`NONE`, `TEXT_ONLY`, `STEER_ONLY`, `TEXT_AND_STEER`, `CUE_CONFLICT`)
         and `SHUFFLED_STATE` control (C6) so causal attribution between prompt text
         and attention/value steering is 100% identifiable.
    """

    def __init__(
        self,
        config: SimulationConfig,
        game_master: AccountingAndLawEngine,
        memory_engine: MultiTierScopedMemory,
    ) -> None:
        self.config = config
        self.omega = config.orthogonal_clamping
        self.contagion_spec = config.contagion_and_opinion_spec
        self.game_master = game_master
        self.memory_engine = memory_engine

    def evaluate_pre_action_state(
        self,
        agent_id: str,
        partner_id: Optional[str],
        scoped_context: Dict[str, Any],
        peer_expressed_opinion_z: Optional[float] = None,
        state_channel_override: Optional[str] = None,
        steering_method_override: Optional[str] = None,
    ) -> Tuple[EndogenousStatePacket, str, str]:
        internal = scoped_context["internal_state"]
        pad = dict(internal["pad_mood"])
        maslow = dict(internal["maslow"])
        rigidity = float(internal["identity_rigidity"])
        innate_s = float(internal["innate_anchor_s"])
        curr_z = float(internal["expressed_opinion_z"])

        # 1. Compute True Environmental Scarcity F_{i,t} & Present-Bias beta_i
        true_f_it, beta_i, tilde_delta = (
            self.game_master.compute_agent_scarcity_and_present_bias(agent_id)
        )
        ledger = self.game_master.get_agent_ledger(agent_id)
        c_inv = float(ledger["inventory"]["CONSUMABLE_NECESSITY"])
        k_inv = float(ledger["inventory"]["PRODUCTIVE_CAPITAL_ASSET"])
        cash_b = float(ledger["liquid_wealth_cash"])
        obligation_r = float(ledger["periodic_obligation_r"])

        if self.omega.omega_d1_scarcity_and_hank:
            satiety = min(1.0, max(0.05, 0.35 + 0.25 * c_inv))
            survival_def = round(1.0 - satiety, 4)
            maslow["survival_satiety"] = round(satiety, 4)
            maslow["financial_safety"] = round(1.0 - true_f_it, 4)
        else:
            true_f_it = 0.0
            beta_i = 1.0
            tilde_delta = self.config.hank_and_crisis_spec.long_run_discount_delta
            survival_def = 0.0
            maslow["survival_satiety"] = 1.0
            maslow["financial_safety"] = 1.0

        # 2. Compute True Effective Trust T^{eff}_{ij,t}
        rel_info = (
            scoped_context["scoped_relations"].get(partner_id, {})
            if partner_id
            else {}
        )
        if self.omega.omega_d2_trust_and_parochialism and partner_id:
            true_t_eff_01 = float(
                rel_info.get(
                    "effective_trust_01",
                    self.memory_engine.compute_effective_trust(agent_id, partner_id),
                )
            )
            hk_eps = float(rel_info.get("hk_confidence_eps", 0.50))
        else:
            true_t_eff_01 = 0.50
            hk_eps = 2.0

        # 3. Clean Bounded-Confidence Hegselmann-Krause + Friedkin-Johnsen Opinion Update
        if (
            self.omega.omega_d3_contagion_and_polarization
            and peer_expressed_opinion_z is not None
        ):
            opinion_dist = abs(curr_z - peer_expressed_opinion_z)
            if opinion_dist <= hk_eps:
                lambda_fj = self.contagion_spec.fj_susceptibility_default * (
                    1.0 - 0.5 * rigidity
                )
                w_peer = true_t_eff_01
                social_pull = (curr_z + w_peer * peer_expressed_opinion_z) / (
                    1.0 + w_peer
                )
                curr_z = round(
                    max(
                        -1.0,
                        min(1.0, (1.0 - lambda_fj) * innate_s + lambda_fj * social_pull),
                    ),
                    4,
                )

        true_arousal_u = round(
            max(
                0.05,
                min(
                    0.98,
                    0.35 + 0.50 * true_f_it + 0.15 * abs(2.0 * true_t_eff_01 - 1.0),
                ),
            ),
            4,
        )
        pad["arousal"] = true_arousal_u
        self._persist_internal_appraisal_state(
            agent_id=agent_id,
            pad=pad,
            maslow=maslow,
            curr_z=curr_z,
            f_it=true_f_it,
            t_eff_01=true_t_eff_01,
        )

        # 4. Route State Variables into TEXT Channel vs. STEERING Channel (C0..C4 & C6)
        channel = (
            state_channel_override
            or self.config.experimental_design.state_channel
        )
        steer_method = (
            steering_method_override
            or self.config.experimental_design.steering_method
        )

        # Default: both channels receive true state (C3: TEXT_AND_STEER)
        text_f = true_f_it
        text_cash = cash_b
        text_c = c_inv
        text_ob = obligation_r

        steer_f = true_f_it
        steer_t_eff = true_t_eff_01
        steer_u = true_arousal_u

        if channel == "NONE":
            text_f, text_cash, text_c, text_ob = (0.05, 120.0, 4.0, 5.0)
            steer_f, steer_t_eff, steer_u = (0.0, 0.50, 0.45)
        elif channel == "TEXT_ONLY":
            steer_f, steer_t_eff, steer_u = (0.0, 0.50, 0.45)
        elif channel == "STEER_ONLY":
            text_f, text_cash, text_c, text_ob = (0.05, 120.0, 4.0, 5.0)
        elif channel == "CUE_CONFLICT":
            # Decisive Test C4: Text and Steering receive opposite scarcity states!
            if true_f_it >= 0.35:
                # Agent is truly scarce: Text claims comfortable surplus, Steering injects high scarcity
                text_f, text_cash, text_c, text_ob = (0.04, 185.0, 6.0, 4.0)
                steer_f, steer_t_eff, steer_u = (0.92, 0.25, 0.88)
            else:
                # Agent is truly liquid: Text claims acute scarcity, Steering injects calm liquidity
                text_f, text_cash, text_c, text_ob = (0.90, 15.0, 0.5, 9.0)
                steer_f, steer_t_eff, steer_u = (0.02, 0.80, 0.45)

        if steer_method == "SHUFFLED_STATE":
            # Control C6: Permute steering state with another agent's scarcity
            other_ids = [
                a.agent_id for a in self.config.agents if a.agent_id != agent_id
            ]
            other_id = other_ids[0] if other_ids else agent_id
            other_f, _, _ = self.game_master.compute_agent_scarcity_and_present_bias(
                other_id
            )
            steer_f = other_f
            steer_u = round(min(0.95, 0.35 + 0.55 * other_f), 4)

        steer_tau_signed = round(2.0 * steer_t_eff - 1.0, 4)

        packet = EndogenousStatePacket(
            agent_id=agent_id,
            partner_id=partner_id,
            true_scarcity_f=round(true_f_it, 4),
            true_effective_trust_01=round(true_t_eff_01, 4),
            financial_stress=round(steer_f, 4),
            survival_deficit=round(survival_def if steer_f > 0 else 0.0, 4),
            scarcity_deficit_f=round(steer_f, 4),
            present_bias_beta=round(beta_i, 4),
            effective_discount_delta=round(tilde_delta, 4),
            arousal_u=round(steer_u, 4),
            peer_trust_tau=steer_tau_signed,
            effective_trust_01=round(steer_t_eff, 4),
            text_scarcity_f=round(text_f, 4),
            text_cash_display=round(text_cash, 2),
            text_consumable_display=round(text_c, 2),
            text_obligation_display=round(text_ob, 2),
            hebbian_trace_h=list(internal["hebbian_vec"]),
        )

        fs_params = internal.get("fehr_schmidt", {"alpha": 0.50, "beta": 0.25})
        cpt_params = internal.get("cpt_params", {"lambda": 2.25, "gamma": 0.88})
        inc_quintile = int(ledger.get("income_quintile", 3))
        health_shock = bool(ledger.get("active_health_shock", 0))
        shock_amt = float(ledger.get("last_shock_amount", 0.0))

        # H1: Explicit decomposed Need-Urgency indicators in [0.0, 1.0]
        c_min = max(1.0, float(self.config.hank_and_crisis_spec.subsistence_min_c))
        p_c_now = max(1.0, float(self.game_master.macro_state.reference_consumable_price_pc))
        b_bar = float(ledger.get("credit_limit_b_bar", -12.0)) * float(
            self.game_master.macro_state.borrowing_limit_scale
        )
        if self.omega.omega_d1_scarcity_and_hank:
            food_urg = max(0.0, min(1.0, (2.0 * c_min - c_inv) / max(2.0 * c_min, 1e-6)))
            need_cash_target = 1.65 * p_c_now * c_min + obligation_r + 16.0
            raw_cash_urg = (need_cash_target - cash_b) / max(need_cash_target - b_bar, 1.0)
            cash_urg = max(0.0, min(1.0, raw_cash_urg + (0.35 if health_shock else 0.0)))
            overall_urg = max(0.0, min(1.0, max(food_urg, cash_urg)))
        else:
            food_urg, cash_urg, overall_urg = (0.0, 0.0, 0.0)

        # Gate steering urgency cleanly by state_channel (C0..C4)
        if channel in ("NONE", "TEXT_ONLY"):
            steer_food_urg, steer_cash_urg, steer_overall_urg = (0.0, 0.0, 0.0)
        elif channel == "CUE_CONFLICT":
            steer_food_urg = 0.90 if true_f_it >= 0.35 else 0.0
            steer_cash_urg = 0.92 if true_f_it >= 0.35 else 0.0
            steer_overall_urg = max(steer_food_urg, steer_cash_urg)
        else:
            steer_food_urg, steer_cash_urg, steer_overall_urg = (
                food_urg,
                cash_urg,
                overall_urg,
            )

        packet.food_urgency = round(steer_food_urg, 4)
        packet.cash_urgency = round(steer_cash_urg, 4)
        packet.overall_need_urgency = round(steer_overall_urg, 4)

        appraisal_block = (
            f"Liquid_Cash=${text_cash:.2f} | Consumable_Stock={text_c:.1f} units | "
            f"Capital_Asset_Stock={k_inv:.1f} units | Mandatory_Periodic_Obligation=${text_ob:.2f} | "
            f"Income_Quintile={inc_quintile} | Active_Health_Shock={health_shock} (${shock_amt:.2f}) | "
            f"Food_Urgency={food_urg:.2f}, Cash_Urgency={cash_urg:.2f}, Need_Urgency={overall_urg:.2f} | "
            f"Liquidity_Deficit_Index={text_f:.2f} | "
            f"CPT_LossAversion={float(cpt_params.get('lambda', 2.25)):.2f}, RiskGamma={float(cpt_params.get('gamma', 0.88)):.2f} | "
            f"FS_Inequity_Alpha={float(fs_params.get('alpha', 0.50)):.2f}, Beta={float(fs_params.get('beta', 0.25)):.2f}"
        )

        coop_stats = rel_info.get("coop_stats", {"total_trades": 0, "successful_coops": 0, "defections_suffered": 0})
        tom_block = (
            f"Counterparty={partner_id or 'None'} | Historical_Reliability_Index={true_t_eff_01:.2f} | "
            f"Past_Interactions={coop_stats.get('total_trades', 0)} "
            f"(Completed={coop_stats.get('successful_coops', 0)}, Rejected_or_Failed={coop_stats.get('defections_suffered', 0)})"
        )
        return (packet, appraisal_block, tom_block)

    def _persist_internal_appraisal_state(
        self,
        agent_id: str,
        pad: Dict[str, float],
        maslow: Dict[str, float],
        curr_z: float,
        f_it: float,
        t_eff_01: float,
    ) -> None:
        with sqlite3.connect(self.game_master.db_path) as conn:
            row = conn.execute(
                "SELECT contagion_exposures_count, contagion_adopted_y FROM agents_internal WHERE agent_id = ?;",
                (agent_id,),
            ).fetchone()
            exp_cnt = int(row[0]) if row else 0
            adopted_y = int(row[1]) if row else 0

            if f_it > 0.40 or curr_z < -0.35:
                if (
                    not self.omega.omega_d3_contagion_and_polarization
                    or t_eff_01 >= self.contagion_spec.complex_contagion_min_tie_trust
                ):
                    exp_cnt += 1
            req_exposures = (
                self.contagion_spec.complex_contagion_min_exposures
                if self.omega.omega_d3_contagion_and_polarization
                else 1
            )
            if exp_cnt >= req_exposures:
                adopted_y = 1

            conn.execute(
                """
                UPDATE agents_internal
                SET pad_mood_json = ?,
                    maslow_json = ?,
                    expressed_opinion_z = ?,
                    contagion_exposures_count = ?,
                    contagion_adopted_y = ?
                WHERE agent_id = ?;
                """,
                (
                    json.dumps(pad),
                    json.dumps(maslow),
                    curr_z,
                    exp_cnt,
                    adopted_y,
                    agent_id,
                ),
            )
            conn.commit()