from __future__ import annotations

import json
import math
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import torch

from src.core.contracts import (
    EndogenousStatePacket,
    SimulationConfig,
)
from src.environment.game_master import AccountingAndLawEngine
from src.memory.scoped_memory import MultiTierScopedMemory


class CognitiveAppraisalAndToM:
    """
    M6: Pre-Action Neurocognitive Appraisal, Barsade (2002) 93%/7% Affective Contagion,
    Friedkin-Johnsen + Hegselmann-Krause + Sharot (0.6875) Opinion Dynamics,
    Centola-Macy Complex Contagion, and 2nd-Order Quantal Response Theory-of-Mind.
    """

    def __init__(
        self,
        config: SimulationConfig,
        game_master: AccountingAndLawEngine,
        memory_engine: MultiTierScopedMemory,
    ) -> None:
        self.config = config
        self.omega = config.orthogonal_clamping
        self.steer_spec = config.steering_spec
        self.contagion_spec = config.contagion_and_opinion_spec
        self.game_master = game_master
        self.memory_engine = memory_engine

    def evaluate_pre_action_state(
        self,
        agent_id: str,
        partner_id: Optional[str],
        scoped_context: Dict[str, Any],
        incoming_verbal_msg: str = "",
        incoming_mirror_vector: Optional[torch.Tensor] = None,
        peer_expressed_opinion_z: Optional[float] = None,
        is_desirable_signal: bool = True,
    ) -> Tuple[EndogenousStatePacket, str, str]:
        """
        Computes:
        1. Dimension 1 Scarcity F_{i,t}, Present-Bias beta_i(W_{i,t}), and LC-NE Gain beta_{LC}.
        2. Dimension 2 Effective Trust T^{eff}_{ij,t} and Quantal Response ToM Forecast.
        3. Dimension 3 Barsade (93% Non-Verbal / 7% Verbal) Affective Contagion &
           FJ + HK + Sharot (0.6875) Opinion Update + Complex Contagion Cascade check.
        Returns (EndogenousStatePacket, appraisal_text_block, tom_forecast_text_block).
        """
        internal = scoped_context["internal_state"]
        pad = dict(internal["pad_mood"])
        maslow = dict(internal["maslow"])
        fs = internal["fehr_schmidt"]
        rigidity = float(internal["identity_rigidity"])
        innate_s = float(internal["innate_anchor_s"])
        curr_z = float(internal["expressed_opinion_z"])

        # --------------------------------------------------------------------
        # 1. DIMENSION 1: Scarcity F_{i,t}, Present Bias & Satiety Appraisal
        # --------------------------------------------------------------------
        f_it, beta_i, tilde_delta = (
            self.game_master.compute_agent_scarcity_and_present_bias(agent_id)
        )
        ledger = self.game_master.get_agent_ledger(agent_id)
        c_inv = float(ledger["inventory"]["CONSUMABLE_NECESSITY"])
        k_inv = float(ledger["inventory"]["PRODUCTIVE_CAPITAL_ASSET"])
        cash_b = float(ledger["liquid_wealth_cash"])

        if self.omega.omega_d1_scarcity_and_hank:
            satiety = min(1.0, max(0.05, 0.35 + 0.25 * c_inv))
            survival_def = round(1.0 - satiety, 4)
            maslow["survival_satiety"] = round(satiety, 4)
            maslow["financial_safety"] = round(1.0 - f_it, 4)
        else:
            f_it = 0.0
            beta_i = 1.0
            tilde_delta = self.config.hank_and_crisis_spec.long_run_discount_delta
            survival_def = 0.0
            maslow["survival_satiety"] = 1.0
            maslow["financial_safety"] = 1.0

        # --------------------------------------------------------------------
        # 2. DIMENSION 2: Effective Trust T^{eff}_{ij,t}
        # --------------------------------------------------------------------
        rel_info = (
            scoped_context["scoped_relations"].get(partner_id, {})
            if partner_id
            else {}
        )
        if self.omega.omega_d2_trust_and_parochialism and partner_id:
            t_eff_01 = float(
                rel_info.get(
                    "effective_trust_01",
                    self.memory_engine.compute_effective_trust(agent_id, partner_id),
                )
            )
            tau_signed = round(2.0 * t_eff_01 - 1.0, 4)
            intimacy_s = float(rel_info.get("intimacy_alpha", 0.1))
            hk_eps = float(rel_info.get("hk_confidence_eps", 0.50))
        else:
            t_eff_01 = 0.50
            tau_signed = 0.0
            intimacy_s = 0.0
            hk_eps = 2.0

        # --------------------------------------------------------------------
        # 3. DIMENSION 3: Barsade (93% Non-Verbal / 7% Verbal) Contagion & Opinions
        # --------------------------------------------------------------------
        valence_v = float(pad.get("valence", 0.0))
        arousal_a = float(pad.get("arousal", 0.3))
        fatigue_f = float(internal.get("fatigue_f", 0.1))

        if self.omega.omega_d3_contagion_and_polarization:
            # 7% Verbal Sentiment Signal
            verbal_lower = (incoming_verbal_msg or "").lower()
            neg_words = ("crisis", "reject", "unfair", "gouge", "starve", "panic", "crash")
            pos_words = ("fair", "cooperate", "accept", "agree", "trust", "mutual", "benefit")
            v_verbal = 0.0
            for w in pos_words:
                if w in verbal_lower:
                    v_verbal += 0.25
            for w in neg_words:
                if w in verbal_lower:
                    v_verbal -= 0.35
            v_verbal = max(-1.0, min(1.0, v_verbal))

            # 93% Subconscious Non-Verbal Leakage Signal (from Speaker's residual stream vector)
            if incoming_mirror_vector is not None and incoming_mirror_vector.numel() > 0:
                vec_f = incoming_mirror_vector.detach().float()
                v_nonverbal = math.tanh(float(vec_f.mean().item()) * 2.5)
                a_nonverbal = min(1.0, max(0.1, float(vec_f.std().item()) * 0.8))
            else:
                v_nonverbal = -0.35 * f_it + 0.25 * tau_signed
                a_nonverbal = min(1.0, 0.30 + 0.55 * f_it)

            combined_affect_shift = (
                self.contagion_spec.barsade_nonverbal_weight * v_nonverbal
                + self.contagion_spec.barsade_verbal_weight * v_verbal
            )
            if combined_affect_shift < 0.0:
                combined_affect_shift *= self.contagion_spec.negativity_contagion_lambda

            valence_v = max(-1.0, min(1.0, 0.55 * valence_v + 0.45 * combined_affect_shift))
            arousal_a = max(
                0.10,
                min(1.0, 0.50 * arousal_a + 0.35 * f_it + 0.25 * a_nonverbal),
            )
            empathy_eps = max(0.15, min(0.95, 0.55 * (1.0 + intimacy_s)))

            # Friedkin-Johnsen + Bounded-Confidence HK + Sharot (0.6875) Belief Update
            if peer_expressed_opinion_z is not None:
                opinion_dist = abs(curr_z - peer_expressed_opinion_z)
                if opinion_dist <= hk_eps:
                    # Sharot (2011) asymmetry: 1.0 for confirming/desirable, 0.6875 for disconfirming
                    confirming = (curr_z * peer_expressed_opinion_z) >= 0.0 or is_desirable_signal
                    psi_sharot = (
                        self.contagion_spec.sharot_desirable_psi
                        if confirming
                        else self.contagion_spec.sharot_undesirable_psi
                    )
                    lambda_fj = self.contagion_spec.fj_susceptibility_default * (
                        1.0 - 0.5 * rigidity
                    )
                    w_peer = psi_sharot * t_eff_01
                    social_pull = (curr_z + w_peer * peer_expressed_opinion_z) / (
                        1.0 + w_peer
                    )
                    curr_z = (1.0 - lambda_fj) * innate_s + lambda_fj * social_pull
                    curr_z = round(max(-1.0, min(1.0, curr_z)), 4)
                else:
                    # Repulsive polarization when outside HK confidence bound under low trust
                    if t_eff_01 < 0.40:
                        repulse = 0.08 * (1.0 if curr_z >= peer_expressed_opinion_z else -1.0)
                        curr_z = round(max(-1.0, min(1.0, curr_z + repulse)), 4)

                # Cognitive Dissonance Score D_t(i) when peer belief opposes own worldview
                cos_sim = curr_z * peer_expressed_opinion_z
                dissonance_d = round(
                    max(0.0, min(1.0, rigidity * max(0.0, -cos_sim))), 4
                )
            else:
                dissonance_d = round(
                    max(0.0, min(1.0, rigidity * max(0.0, -tau_signed * 0.6))), 4
                )
        else:
            empathy_eps = 0.0
            dissonance_d = 0.0
            arousal_a = 0.45

        # Update cognitive load L_t(i) for Sys1/Sys2 head gating
        if self.omega.omega_d1_scarcity_and_hank:
            cog_load_l = round(
                max(
                    0.0,
                    min(1.0, 0.55 * f_it + 0.30 * arousal_a + 0.15 * fatigue_f),
                ),
                4,
            )
        else:
            cog_load_l = 0.0

        # Persist updated mood, Maslow, and opinion state
        pad["valence"] = round(valence_v, 4)
        pad["arousal"] = round(arousal_a, 4)
        self._persist_internal_appraisal_state(
            agent_id=agent_id,
            pad=pad,
            maslow=maslow,
            curr_z=curr_z,
            f_it=f_it,
            t_eff_01=t_eff_01,
        )

        # Build EndogenousStatePacket for M3: AttentionSteeringLab
        packet = EndogenousStatePacket(
            agent_id=agent_id,
            partner_id=partner_id,
            financial_stress=f_it,
            survival_deficit=survival_def,
            scarcity_deficit_f=f_it,
            present_bias_beta=round(beta_i, 4),
            effective_discount_delta=round(tilde_delta, 4),
            cognitive_load_l=cog_load_l,
            arousal_a=round(arousal_a, 4),
            fatigue_f=round(fatigue_f, 4),
            valence_v=round(valence_v, 4),
            dissonance_d=dissonance_d,
            peer_trust_tau=tau_signed,
            effective_trust_01=t_eff_01,
            peer_intimacy_s=round(intimacy_s, 4),
            empathy_eps=round(empathy_eps, 4),
            hebbian_trace_h=list(internal["hebbian_vec"]),
            incoming_mirror_h=(
                incoming_mirror_vector
                if self.omega.omega_d3_contagion_and_polarization
                else None
            ),
        )

        appraisal_block = (
            f"Scarcity_Deficit_F={f_it:.2f} | Present_Bias_Beta={beta_i:.2f} | "
            f"Effective_Discount_Delta={tilde_delta:.2f} | Cash=${cash_b:.2f} | "
            f"Consumable_Inv={c_inv:.1f} | Capital_Asset_Inv={k_inv:.1f} | "
            f"Fehr_Schmidt=({fs['tier']}: alpha={fs['alpha']:.2f}, beta={fs['beta']:.2f}) | "
            f"Worldview_z={curr_z:+.2f}"
        )

        tom_block = self._build_tom_forecast_block(
            agent_id=agent_id,
            partner_id=partner_id,
            t_eff_01=t_eff_01,
            scoped_context=scoped_context,
        )
        return (packet, appraisal_block, tom_block)

    def _build_tom_forecast_block(
        self,
        agent_id: str,
        partner_id: Optional[str],
        t_eff_01: float,
        scoped_context: Dict[str, Any],
    ) -> str:
        if not partner_id:
            return "No active counterparty."
        pm = scoped_context["scoped_peer_models"].get(partner_id, {})
        traits = pm.get("perceived_traits", {})
        est_alpha = float(traits.get("estimated_alpha_fs", 0.50))
        # Quantal Response acceptance sensitivity to fair vs. gouging splits
        fair_accept_prob = 1.0 / (
            1.0 + math.exp(-3.2 * (0.35 + 0.50 * (t_eff_01 - 0.50)))
        )
        gouge_accept_prob = 1.0 / (
            1.0 + math.exp(-3.2 * (-0.45 - 0.60 * est_alpha))
        )
        return (
            f"Partner={partner_id} | Effective_Trust={t_eff_01:.2f} | "
            f"Est_Inequity_Aversion_Alpha={est_alpha:.2f} | "
            f"QRE_Accept_Prob(Within_ZOPA)={fair_accept_prob:.0%} | "
            f"QRE_Accept_Prob(Price_Gouging)={gouge_accept_prob:.0%}"
        )

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

            # Centola & Macy (2007) Complex Contagion threshold check
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