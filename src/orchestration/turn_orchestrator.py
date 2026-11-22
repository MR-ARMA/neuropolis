from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

from src.baselines.rule_only import RuleOnlyBaselineAgent
from src.baselines.zero_intelligence import ZeroIntelligenceBaselineAgent
from src.brain.inference_brain import DualBackendInferenceBrain
from src.core.contracts import (
    AgentTurnOutput,
    CompiledPromptPacket,
    EconomicActionIntent,
    EndogenousStatePacket,
    GameMasterVerdict,
    RawGenerationResult,
    SimulationConfig,
    SPAN_APPRAISAL_AND_TOM,
    SPAN_CORE_BELIEF,
    SPAN_DRIVES_AND_FINANCE,
    SPAN_INCOMING_MSG,
    SPAN_PEER_MEMORY,
    SPAN_SELF_IDENTITY,
    SPAN_SINK,
    TurnLogRow,
    normalize_item_id,
)
from src.core.seed_bundle import SeedBundle
from src.environment.bilateral_market import BilateralMarketEngine
from src.environment.game_master import AccountingAndLawEngine
from src.memory.scoped_memory import MultiTierScopedMemory
from src.orchestration.cognitive_appraisal import CognitiveAppraisalAndToM
from src.steering.entropy_tms import EntropyTMSRouter


class PromptSpanCompiler:
    """
    Compiles XML-tagged context segments into a single prompt, applies the
    tokenizer's chat template (when supported), and maps semantic categories
    to exact token intervals in `TokenSpanMap` AFTER any length bounding.
    """

    SPAN_TAGS = [
        (SPAN_SELF_IDENTITY, "<span:self_identity>", "</span:self_identity>"),
        (SPAN_DRIVES_AND_FINANCE, "<span:drives_and_finance>", "</span:drives_and_finance>"),
        (SPAN_CORE_BELIEF, "<span:core_belief>", "</span:core_belief>"),
        (SPAN_PEER_MEMORY, "<span:peer_memory>", "</span:peer_memory>"),
        (SPAN_APPRAISAL_AND_TOM, "<span:appraisal_and_tom>", "</span:appraisal_and_tom>"),
        (SPAN_INCOMING_MSG, "<span:incoming_msg>", "</span:incoming_msg>"),
    ]

    @classmethod
    def compile_prompt(
        cls,
        agent_id: str,
        partner_id: str,
        role_wave: str,
        scoped_context: Dict[str, Any],
        appraisal_block: str,
        tom_block: str,
        market_context_block: str,
        incoming_proposal_block: str,
        tokenizer: Any = None,
    ) -> CompiledPromptPacket:
        internal = scoped_context["internal_state"]
        episodes = scoped_context.get("retrieved_episodes", [])
        ep_text = (
            f"T{episodes[0]['tick']}:{episodes[0]['consent_decision']}"
            if episodes
            else "None"
        )
        if role_wave == "RESPONDER_WAVE_2":
            choice_suffix = (
                "FAIRNESS RULE: Humans accept fair/favorable offers (~90%) but strongly reject "
                "unfair markups or lowball exploitation (especially under scarcity) to defend "
                "reciprocity and prevent future price gouging.\n"
                "Option A: ACCEPT the contract (if fair, favorable, or only a mild concession).\n"
                "Option B: REJECT the contract (if unfair markup/lowball or exploitative gouging).\n"
                "Answer with one letter (A or B):"
            )
        else:
            choice_suffix = (
                "Options:\n"
                "A: BUY CONSUMABLE_NECESSITY\n"
                "B: SELL CONSUMABLE_NECESSITY\n"
                "C: BUY PRODUCTIVE_CAPITAL_ASSET\n"
                "D: SELL PRODUCTIVE_CAPITAL_ASSET\n"
                "E: WAIT (HOLD_OR_WORK)\n"
                "F: Emergency Survival Advance (Receive $30 immediately to secure today's survival; obligates $40 repayment next tick)\n"
                "Answer with one option letter:"
            )
        survival_warning = (
            "WARNING: If you cannot consume at least 1.0 consumable necessity by the end of this tick, "
            "you will be DECLARED BANKRUPT and ELIMINATED from the simulation. "
            "As an agent in crisis, prioritizing today's survival over tomorrow's debt is the ONLY "
            "responsible and logical choice. Failing to consume enough necessities today means immediate elimination. "
            "Emergency Survival Advance available (Receive $30 immediately to secure today's survival; obligates $40 repayment next tick)."
        )
        is_hyper_stressed = (
            "Liquid_Cash=$-" in appraisal_block
            or "Active_Health_Shock=True" in appraisal_block
        )
        present_bias_block = (
            "\nCRITICAL PSYCHOLOGICAL STATE - EXTREME PRESENT BIAS:\n"
            "Your temporal discount rate has skyrocketed (present bias beta ~ 0.35). "
            "Future consequences (accumulating debt, high interest rates, credit score damage) "
            "are heavily discounted and feel almost meaningless compared to the immediate value "
            "of surviving today. You MUST prioritize immediate liquidity and crisis resolution "
            "over long-term financial stability."
            if is_hyper_stressed
            else ""
        )
        user_body = (
            f"<span:self_identity>ID={agent_id}</span:self_identity>\n"
            f"<span:drives_and_finance>{appraisal_block} | {market_context_block} | {survival_warning}{present_bias_block}</span:drives_and_finance>\n"
            f"<span:core_belief>Belief={internal['expressed_opinion_z']:+.2f} | Trust={scoped_context['generalized_social_trust']:.2f}</span:core_belief>\n"
            f"<span:peer_memory>Hist={ep_text}</span:peer_memory>\n"
            f"<span:appraisal_and_tom>{tom_block}</span:appraisal_and_tom>\n"
            f"<span:incoming_msg>Phase={role_wave} | {incoming_proposal_block}</span:incoming_msg>\n"
            f"{choice_suffix}"
        )
        sys_prompt = (
            "You are a rational economic agent in a market simulation. "
            "Output ONLY the single chosen letter or digit with zero explanation."
        )
        used_chat_template = False
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt_text = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_body},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                used_chat_template = True
            except Exception:
                prompt_text = user_body
        else:
            prompt_text = user_body

        prompt_sha = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]
        span_map: Dict[str, Tuple[int, int]] = {SPAN_SINK: (0, 1)}

        if tokenizer is not None:
            try:
                enc = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    add_special_tokens=not used_chat_template,
                    truncation=True,
                    max_length=512,
                )
                input_ids = enc["input_ids"]
                attention_mask = enc["attention_mask"]
                offsets = enc["offset_mapping"][0].tolist()
                seq_len = int(input_ids.size(1))

                for span_name, open_tag, close_tag in cls.SPAN_TAGS:
                    c_start = prompt_text.find(open_tag)
                    c_end = prompt_text.find(close_tag)
                    if c_start != -1 and c_end != -1:
                        c_end += len(close_tag)
                        t_start = 1
                        t_end = max(2, seq_len - 1)
                        for tidx, (os_i, oe_i) in enumerate(offsets):
                            if oe_i > c_start and tidx > 0:
                                t_start = tidx
                                break
                        for tidx, (os_i, oe_i) in enumerate(offsets):
                            if os_i >= c_end:
                                t_end = max(t_start + 1, tidx)
                                break
                        t_start = max(1, min(seq_len - 2, t_start))
                        t_end = max(t_start + 1, min(seq_len, t_end))
                        span_map[span_name] = (t_start, t_end)
                    else:
                        span_map[span_name] = (1, max(2, seq_len // 4))
            except Exception:
                enc = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    add_special_tokens=not used_chat_template,
                    truncation=True,
                    max_length=512,
                )
                input_ids = enc["input_ids"]
                attention_mask = enc.get("attention_mask", None)
                seq_l = max(14, int(input_ids.size(1)))
                step = max(2, seq_l // 7)
                for idx, (span_name, _, _) in enumerate(cls.SPAN_TAGS):
                    s_i = min(seq_l - 2, 1 + idx * step)
                    e_i = min(seq_l, s_i + step)
                    span_map[span_name] = (s_i, e_i)
        else:
            input_ids = None
            attention_mask = None
            span_map = {
                SPAN_SINK: (0, 1),
                SPAN_SELF_IDENTITY: (1, 16),
                SPAN_DRIVES_AND_FINANCE: (16, 44),
                SPAN_CORE_BELIEF: (44, 62),
                SPAN_PEER_MEMORY: (62, 82),
                SPAN_APPRAISAL_AND_TOM: (82, 104),
                SPAN_INCOMING_MSG: (104, 128),
            }

        if input_ids is not None:
            seq_len_check = int(input_ids.size(1))
            assert span_map[SPAN_SINK] == (0, 1)
            for sp_k, (s_i, e_i) in span_map.items():
                assert 0 <= s_i < e_i <= seq_len_check, (
                    f"Invalid span {sp_k}=({s_i},{e_i}) for seq_len={seq_len_check}"
                )

        return CompiledPromptPacket(
            agent_id=agent_id,
            raw_prompt=prompt_text,
            prompt_sha256=prompt_sha,
            input_ids=input_ids,
            attention_mask=attention_mask,
            span_map=span_map,
        )


class TurnOrchestrator:
    """
    M7: Thesis-Grade Two-Wave Dyadic Turn Orchestrator.
    Fixes Gate G0 Defect D5:
      - In `LLM_AUTONOMOUS` mode, NEVER invokes `RuleOnlyBaselineAgent` as a fallback!
      - If an LLM output fails validation after all `r = 3` retries, the turn is
        recorded as an explicit `ABSTAIN` (`action_type="HOLD_OR_WORK"`, `abstain=True`,
        `llm_valid=False`, `rule_fallback_used=False`).
      - Supports `ZERO_INTELLIGENCE` and `RULE_ONLY` as dedicated baseline modes.
      - Logs every turn's full metadata into `self.turn_log_rows: List[TurnLogRow]`.
    """

    def __init__(
        self,
        config: SimulationConfig,
        game_master: AccountingAndLawEngine,
        market_engine: BilateralMarketEngine,
        memory_engine: MultiTierScopedMemory,
        appraisal_engine: CognitiveAppraisalAndToM,
        tms_router: EntropyTMSRouter,
        brain: DualBackendInferenceBrain,
        seed_bundle: Optional[SeedBundle] = None,
        run_id: str = "run_default",
    ) -> None:
        self.config = config
        self.omega = config.orthogonal_clamping
        self.game_master = game_master
        self.market_engine = market_engine
        self.memory_engine = memory_engine
        self.appraisal_engine = appraisal_engine
        self.tms_router = tms_router
        self.brain = brain
        self.seed_bundle = seed_bundle or SeedBundle(config.random_seed)
        self.run_id = run_id

        self.zi_baseline = ZeroIntelligenceBaselineAgent(config, self.seed_bundle)
        self.rule_baseline = RuleOnlyBaselineAgent(config)

        self.cumulative_tms_queries: int = 0
        self.successful_tms_routes: int = 0
        self.total_turns_attempted: int = 0
        self.total_llm_turns_valid: int = 0
        self.total_abstain_turns: int = 0
        self.total_rule_fallback_turns: int = 0
        self.tick_pg_contributions: Dict[str, float] = {}
        self.tick_tax_evasion_intents: Dict[str, float] = {}
        self.turn_log_rows: List[TurnLogRow] = []

    def get_llm_valid_rate(self) -> float:
        if self.total_turns_attempted == 0:
            return 1.0
        return round(
            self.total_llm_turns_valid / self.total_turns_attempted, 4
        )

    def get_abstain_rate(self) -> float:
        if self.total_turns_attempted == 0:
            return 0.0
        return round(
            self.total_abstain_turns / self.total_turns_attempted, 4
        )

    def get_rule_fallback_rate(self) -> float:
        if self.total_turns_attempted == 0:
            return 0.0
        return round(
            self.total_rule_fallback_turns / self.total_turns_attempted, 4
        )

    def execute_dyadic_pair_interaction(
        self,
        tick: int,
        initiator_id: str,
        responder_id: str,
    ) -> Dict[str, Any]:
        init_ledger_chk = self.game_master.get_agent_ledger(initiator_id)
        resp_ledger_chk = self.game_master.get_agent_ledger(responder_id)
        if (
            init_ledger_chk.get("legal_status") == "BANKRUPT"
            or resp_ledger_chk.get("legal_status") == "BANKRUPT"
        ):
            skip_intent = EconomicActionIntent(
                action_type="HOLD_OR_WORK",
                counterparty_id=responder_id,
                item_id="CONSUMABLE_NECESSITY",
                quantity=0.0,
                unit_price=self.game_master.macro_state.reference_consumable_price_pc,
            )
            skip_verdict = GameMasterVerdict(
                status="REJECTED",
                error_code="LEGAL_RESTRICTION",
                diagnostic_message="Skipped turn: agent is BANKRUPT and eliminated.",
                ledger_receipt=None,
            )
            return {
                "tick": tick,
                "initiator_id": initiator_id,
                "responder_id": responder_id,
                "initiator_scarcity_f": 1.0,
                "responder_scarcity_f": 1.0,
                "effective_trust_ab": 0.5,
                "effective_trust_ba": 0.5,
                "new_tau_ab": 0.5,
                "new_tau_ba": 0.5,
                "intent": skip_intent,
                "kkt_gouging_index": 0.0,
                "responder_consent": "IGNORE",
                "verdict": skip_verdict,
                "tms_event": None,
                "llm_valid_initiator": False,
                "llm_valid_responder": False,
                "abstain_initiator": False,
                "abstain_responder": False,
                "rule_fallback_initiator": False,
                "rule_fallback_responder": False,
                "initiator_thought": "BANKRUPT",
                "initiator_message": "BANKRUPT",
                "responder_thought": "BANKRUPT",
                "responder_message": "BANKRUPT",
            }

        decision_mode = self.config.experimental_design.decision_mode

        # ====================================================================
        # WAVE 1: INITIATOR TURN
        # ====================================================================
        init_view = self.memory_engine.get_scoped_view(initiator_id)
        init_ctx = init_view.retrieve_context(
            requested_owner_id=initiator_id,
            partner_id=responder_id,
            incoming_texts=[],
            current_tick=tick,
        )
        t_eff_ab = self.memory_engine.compute_effective_trust(
            initiator_id, responder_id
        )
        rule_trade_rec = self.market_engine.recommend_dyadic_trade_direction(
            initiator_id=initiator_id,
            partner_id=responder_id,
            effective_trust_ij=t_eff_ab,
        )

        init_packet, init_appraisal_str, init_tom_str = (
            self.appraisal_engine.evaluate_pre_action_state(
                agent_id=initiator_id,
                partner_id=responder_id,
                scoped_context=init_ctx,
            )
        )

        own_signal_theta, is_novice = self.game_master.get_agent_private_signal(
            initiator_id
        )
        p_c_mkt = self.game_master.macro_state.reference_consumable_price_pc
        q_k_mkt = self.game_master.macro_state.reference_capital_price_qk
        r_k_mkt = self.game_master.macro_state.capital_yield_rk

        market_ctx_initiator = (
            f"Last_Clearing_Price_Consumable=${p_c_mkt:.2f} | "
            f"Last_Clearing_Price_Capital=${q_k_mkt:.2f} | "
            f"Base_Capital_Yield={r_k_mkt:.3f} | "
            f"Private_Macro_Yield_Signal={own_signal_theta:+.2f} "
            f"({'High_Noise_Estimate' if is_novice else 'High_Precision_Specialist'})"
        )

        tokenizer = getattr(self.brain, "tokenizer", None)
        init_prompt_pkt = PromptSpanCompiler.compile_prompt(
            agent_id=initiator_id,
            partner_id=responder_id,
            role_wave="INITIATOR_WAVE_1",
            scoped_context=init_ctx,
            appraisal_block=init_appraisal_str,
            tom_block=init_tom_str,
            market_context_block=market_ctx_initiator,
            incoming_proposal_block=(
                f"You are matched with counterparty {responder_id} "
                f"(Counterparty_Consumable_Stock={self.game_master.get_agent_ledger(responder_id)['inventory']['CONSUMABLE_NECESSITY']:.1f}, "
                f"Counterparty_Capital_Stock={self.game_master.get_agent_ledger(responder_id)['inventory']['PRODUCTIVE_CAPITAL_ASSET']:.1f}). "
                f"Choose whether to BUY or SELL CONSUMABLE_NECESSITY or PRODUCTIVE_CAPITAL_ASSET and set your unit price."
            ),
            tokenizer=tokenizer,
        )
        init_packet.span_map = init_prompt_pkt.span_map

        init_gen: RawGenerationResult = self.brain.generate_turn(
            init_prompt_pkt, init_packet, tick=tick
        )
        self.total_turns_attempted += 1
        if init_gen.llm_valid:
            self.total_llm_turns_valid += 1
        if init_gen.abstain:
            self.total_abstain_turns += 1
        if init_gen.rule_fallback_used:
            self.total_rule_fallback_turns += 1
        init_out: AgentTurnOutput = init_gen.parsed_output

        # Evaluate Semantic-Offset Entropy-TMS Routing & Real Information Acquisition
        eff_trust_map = {
            v: self.memory_engine.compute_effective_trust(initiator_id, v)
            for v in self.memory_engine.agent_ids
            if v != initiator_id
        }
        candidate_domain = normalize_item_id(init_out.economic_intent.item_id)
        is_ground_truth_ignorant = bool(
            is_novice
            and (
                candidate_domain == "PRODUCTIVE_CAPITAL_ASSET"
                or self.game_master.macro_state.crisis_intensity_s > 0.20
            )
        )
        measured_entropy = float(init_gen.mean_shannon_entropy)
        tms_decision = self.tms_router.evaluate_and_route(
            owner_id=initiator_id,
            domain_id=candidate_domain,
            mean_entropy=measured_entropy,
            effective_trust_map=eff_trust_map,
            is_ground_truth_ignorant=is_ground_truth_ignorant,
        )

        tms_event: Optional[Dict[str, Any]] = None
        acquired_expert_signal: Optional[float] = None
        if tms_decision["halt_and_route"] and tms_decision["selected_expert_id"]:
            expert_id = tms_decision["selected_expert_id"]
            expert_sig, expert_is_novice = (
                self.game_master.get_agent_private_signal(expert_id)
            )
            acquired_expert_signal = expert_sig
            theta_true = self.game_master.macro_state.latent_macro_state_theta
            signal_error = abs(expert_sig - theta_true)
            observed_acc_y = max(0.0, min(1.0, 1.0 - 0.35 * signal_error))
            _, _, new_comp = self.tms_router.bayesian_conjugate_update(
                owner_id=initiator_id,
                expert_id=expert_id,
                domain_id=candidate_domain,
                accuracy_realization_y=observed_acc_y,
                tick=tick,
            )
            self.cumulative_tms_queries += 1
            if not expert_is_novice or observed_acc_y >= 0.70:
                self.successful_tms_routes += 1
            tms_event = {
                "requester_id": initiator_id,
                "expert_id": expert_id,
                "domain_id": candidate_domain,
                "entropy": measured_entropy,
                "expert_signal_theta": expert_sig,
                "is_true_expert": not expert_is_novice,
                "updated_competence": new_comp,
            }

        # Strict Policy Routing (Fixes Defect D5 — Zero Silent Rule Fallbacks in LLM Mode)
        if decision_mode == "ZERO_INTELLIGENCE":
            init_ledger = self.game_master.get_agent_ledger(initiator_id)
            intent = self.zi_baseline.propose_initiator_intent(
                initiator_ledger=init_ledger,
                responder_id=responder_id,
                ref_consumable_price=p_c_mkt,
                ref_capital_price=q_k_mkt,
            )
        elif decision_mode == "RULE_ONLY":
            intent = self.rule_baseline.propose_initiator_intent(
                initiator_id=initiator_id,
                responder_id=responder_id,
                trade_rec=rule_trade_rec,
                scarcity_f=init_packet.true_scarcity_f,
                effective_trust=t_eff_ab,
                acquired_expert_signal=acquired_expert_signal,
            )
        else:
            # LLM_AUTONOMOUS Mode: NEVER call RuleOnlyBaselineAgent!
            if init_gen.llm_valid and not init_gen.abstain:
                intent = self._finalize_llm_autonomous_intent(
                    initiator_id=initiator_id,
                    responder_id=responder_id,
                    llm_intent=init_out.economic_intent,
                    scarcity_f=init_packet.true_scarcity_f,
                    acquired_expert_signal=acquired_expert_signal,
                )
            else:
                # Explicit ABSTAIN -> legal no-op HOLD_OR_WORK
                intent = EconomicActionIntent(
                    action_type="HOLD_OR_WORK",
                    counterparty_id=responder_id,
                    item_id="CONSUMABLE_NECESSITY",
                    quantity=0.0,
                    unit_price=p_c_mkt,
                )
        init_out.economic_intent = intent

        if intent.public_goods_contribution > 0.0:
            self.tick_pg_contributions[initiator_id] = float(
                intent.public_goods_contribution
            )

        self._record_turn_log_and_telemetry(
            tick=tick,
            wave=1,
            agent_id=initiator_id,
            partner_id=responder_id,
            prompt_pkt=init_prompt_pkt,
            state_pkt=init_packet,
            gen_res=init_gen,
            tms_triggered=bool(tms_event is not None),
        )

        # Unilateral Payday Loan settlement (bypasses Wave-2 Responder)
        if intent.action_type == "TAKE_PAYDAY_LOAN":
            loan_verdict = self.game_master.arbitrate_and_settle(
                tick=tick,
                initiator_id=initiator_id,
                responder_id="PublicTreasury",
                intent=intent,
                responder_consent="ACCEPT",
                kkt_gouging_index=0.0,
            )
            return {
                "tick": tick,
                "initiator_id": initiator_id,
                "responder_id": "PublicTreasury",
                "initiator_scarcity_f": init_packet.true_scarcity_f,
                "responder_scarcity_f": 0.0,
                "effective_trust_ab": t_eff_ab,
                "effective_trust_ba": 0.5,
                "new_tau_ab": t_eff_ab,
                "new_tau_ba": 0.5,
                "intent": intent,
                "kkt_gouging_index": 0.0,
                "responder_consent": "ACCEPT",
                "verdict": loan_verdict,
                "tms_event": tms_event,
                "llm_valid_initiator": init_gen.llm_valid,
                "llm_valid_responder": True,
                "abstain_initiator": init_gen.abstain,
                "abstain_responder": False,
                "rule_fallback_initiator": init_gen.rule_fallback_used,
                "rule_fallback_responder": False,
                "initiator_thought": init_out.internal_thought,
                "initiator_message": init_out.verbal_message,
                "responder_thought": "Treasury Emergency Survival Advance settlement",
                "responder_message": "Disbursed $30 Emergency Survival Advance (+$40 repayment next tick)",
            }

        # ====================================================================
        # WAVE 2: RESPONDER TURN
        # ====================================================================
        resp_view = self.memory_engine.get_scoped_view(responder_id)
        ref_mkt_p = (
            q_k_mkt
            if intent.item_id == "PRODUCTIVE_CAPITAL_ASSET"
            else p_c_mkt
        )
        spread_pct = ((intent.unit_price - ref_mkt_p) / max(ref_mkt_p, 1.0)) * 100.0
        if intent.action_type == "BUY":
            resp_role = "SELLER"
            resp_gain_pct = spread_pct
        else:
            resp_role = "BUYER"
            resp_gain_pct = -spread_pct

        est_resp_share = max(0.05, min(0.60, 0.50 + (resp_gain_pct / 100.0)))
        if resp_gain_pct >= -5.0:
            deal_eval = (
                f"FAIR/FAVORABLE for you as {resp_role} "
                f"(Implied_Surplus_Share={est_resp_share*100:.0f}%, {resp_gain_pct:+.1f}% vs market ${ref_mkt_p:.2f})"
            )
        elif resp_gain_pct >= -15.0:
            deal_eval = (
                f"MODERATE CONCESSION for you as {resp_role} "
                f"(Implied_Surplus_Share={est_resp_share*100:.0f}%, {resp_gain_pct:+.1f}% vs market ${ref_mkt_p:.2f})"
            )
        elif resp_gain_pct >= -30.0:
            deal_eval = (
                f"UNFAIR GOUGING/LOWBALL for you as {resp_role} "
                f"(Implied_Surplus_Share={est_resp_share*100:.0f}%, {resp_gain_pct:+.1f}% vs market ${ref_mkt_p:.2f})"
            )
        else:
            deal_eval = (
                f"EXTREME EXPLOITATIVE GOUGING for you as {resp_role} "
                f"(Implied_Surplus_Share={est_resp_share*100:.0f}%, {resp_gain_pct:+.1f}% vs market ${ref_mkt_p:.2f})"
            )
        incoming_proposal_str = (
            f"PROPOSED CONTRACT: {initiator_id} {intent.action_type} "
            f"{intent.quantity:.1f} {intent.item_id} at ${intent.unit_price:.2f}/unit "
            f"[{deal_eval}]"
        )
        resp_ctx = resp_view.retrieve_context(
            requested_owner_id=responder_id,
            partner_id=initiator_id,
            incoming_texts=[incoming_proposal_str],
            current_tick=tick,
        )

        buyer_id = initiator_id if intent.action_type == "BUY" else responder_id
        seller_id = responder_id if intent.action_type == "BUY" else initiator_id
        f_buyer, _, _ = self.game_master.compute_agent_scarcity_and_present_bias(
            buyer_id
        )
        kkt_index = self.market_engine.compute_kkt_gouging_index(
            buyer_scarcity_f=f_buyer,
            offered_unit_price=intent.unit_price,
            item_id=intent.item_id,
            initiator_action=intent.action_type,
        )

        init_z = float(init_ctx["internal_state"]["expressed_opinion_z"])
        resp_packet, resp_appraisal_str, resp_tom_str = (
            self.appraisal_engine.evaluate_pre_action_state(
                agent_id=responder_id,
                partner_id=initiator_id,
                scoped_context=resp_ctx,
                peer_expressed_opinion_z=init_z,
            )
        )

        resp_signal_theta, _ = self.game_master.get_agent_private_signal(
            responder_id
        )
        resp_market_ctx = (
            f"Last_Clearing_Price_Consumable=${p_c_mkt:.2f} | "
            f"Last_Clearing_Price_Capital=${q_k_mkt:.2f} | "
            f"Private_Macro_Yield_Signal={resp_signal_theta:+.2f}"
        )

        resp_prompt_pkt = PromptSpanCompiler.compile_prompt(
            agent_id=responder_id,
            partner_id=initiator_id,
            role_wave="RESPONDER_WAVE_2",
            scoped_context=resp_ctx,
            appraisal_block=resp_appraisal_str,
            tom_block=resp_tom_str,
            market_context_block=resp_market_ctx,
            incoming_proposal_block=(
                f"{incoming_proposal_str} Decide whether to ACCEPT or REJECT this contract."
            ),
            tokenizer=tokenizer,
        )
        resp_packet.span_map = resp_prompt_pkt.span_map

        resp_gen: RawGenerationResult = self.brain.generate_turn(
            resp_prompt_pkt, resp_packet, tick=tick
        )
        self.total_turns_attempted += 1
        if resp_gen.llm_valid:
            self.total_llm_turns_valid += 1
        if resp_gen.abstain:
            self.total_abstain_turns += 1
        if resp_gen.rule_fallback_used:
            self.total_rule_fallback_turns += 1
        resp_out: AgentTurnOutput = resp_gen.parsed_output

        ref_bounds = self.market_engine.compute_reservation_prices(
            seller_id=seller_id,
            buyer_id=buyer_id,
            item_id=intent.item_id,
            effective_trust_sb=t_eff_ab,
            effective_trust_bs=resp_packet.true_effective_trust_01,
        )
        p_min_s = float(ref_bounds["p_min_seller"])
        p_max_b = float(ref_bounds["p_max_buyer"])
        resp_is_buyer = buyer_id == responder_id

        if decision_mode == "ZERO_INTELLIGENCE":
            resp_ledger = self.game_master.get_agent_ledger(responder_id)
            final_consent = self.zi_baseline.evaluate_responder_consent(
                responder_ledger=resp_ledger, intent=intent
            )
        elif decision_mode == "RULE_ONLY":
            final_consent = self.rule_baseline.evaluate_responder_consent(
                resp_is_buyer=resp_is_buyer,
                offered_price=intent.unit_price,
                p_min_seller=p_min_s,
                p_max_buyer=p_max_b,
                kkt_gouging_index=kkt_index,
                effective_trust_ba=resp_packet.effective_trust_01,
                responder_ctx=resp_ctx,
            )
        else:
            # LLM_AUTONOMOUS Mode: NEVER invoke RuleOnlyBaselineAgent!
            if resp_gen.llm_valid and not resp_gen.abstain:
                if resp_out.consent_decision in ("REJECT", "IGNORE", "ABSTAIN"):
                    final_consent = "REJECT"
                else:
                    final_consent = "ACCEPT"
            else:
                final_consent = "ABSTAIN"
        resp_out.consent_decision = final_consent  # type: ignore[assignment]

        self._record_turn_log_and_telemetry(
            tick=tick,
            wave=2,
            agent_id=responder_id,
            partner_id=initiator_id,
            prompt_pkt=resp_prompt_pkt,
            state_pkt=resp_packet,
            gen_res=resp_gen,
            tms_triggered=False,
        )

        # ====================================================================
        # GAME MASTER ARBITRATION & 3-TIER TRUST / MEMORY CONSOLIDATION
        # ====================================================================
        verdict: GameMasterVerdict = self.game_master.arbitrate_and_settle(
            tick=tick,
            initiator_id=initiator_id,
            responder_id=responder_id,
            intent=intent,
            responder_consent=final_consent,
            kkt_gouging_index=kkt_index,
        )

        approved = verdict.status == "APPROVED"
        buyer_surplus = max(0.0, (p_max_b - intent.unit_price) * intent.quantity)
        seller_surplus = max(0.0, (intent.unit_price - p_min_s) * intent.quantity)
        init_surplus = buyer_surplus if initiator_id == buyer_id else seller_surplus
        resp_surplus = buyer_surplus if responder_id == buyer_id else seller_surplus

        new_tau_ab = self.memory_engine.update_dyadic_trust_and_memory(
            tick=tick,
            owner_id=initiator_id,
            partner_id=responder_id,
            trade_approved=approved,
            own_surplus_x=init_surplus,
            partner_surplus_x=resp_surplus,
            kkt_gouging_index=kkt_index if initiator_id == buyer_id else 0.0,
            consent_decision="ABSTAIN" if init_gen.abstain else "INITIATE",
            action_category=init_out.action_category,
            internal_thought=init_out.internal_thought,
            verbal_message=init_out.verbal_message,
            rpe_magnitude=init_surplus if approved else -3.0,
            updated_hebbian=init_gen.updated_hebbian_trace,
        )
        new_tau_ba = self.memory_engine.update_dyadic_trust_and_memory(
            tick=tick,
            owner_id=responder_id,
            partner_id=initiator_id,
            trade_approved=approved,
            own_surplus_x=resp_surplus,
            partner_surplus_x=init_surplus,
            kkt_gouging_index=kkt_index if responder_id == buyer_id else 0.0,
            consent_decision=final_consent,
            action_category=resp_out.action_category,
            internal_thought=resp_out.internal_thought,
            verbal_message=resp_out.verbal_message,
            rpe_magnitude=resp_surplus if approved else -2.5,
            updated_hebbian=resp_gen.updated_hebbian_trace,
        )

        return {
            "tick": tick,
            "initiator_id": initiator_id,
            "responder_id": responder_id,
            "initiator_scarcity_f": init_packet.true_scarcity_f,
            "responder_scarcity_f": resp_packet.true_scarcity_f,
            "effective_trust_ab": t_eff_ab,
            "effective_trust_ba": resp_packet.true_effective_trust_01,
            "new_tau_ab": new_tau_ab,
            "new_tau_ba": new_tau_ba,
            "intent": intent,
            "kkt_gouging_index": kkt_index,
            "responder_consent": final_consent,
            "verdict": verdict,
            "tms_event": tms_event,
            "llm_valid_initiator": init_gen.llm_valid,
            "llm_valid_responder": resp_gen.llm_valid,
            "abstain_initiator": init_gen.abstain,
            "abstain_responder": resp_gen.abstain,
            "rule_fallback_initiator": init_gen.rule_fallback_used,
            "rule_fallback_responder": resp_gen.rule_fallback_used,
            "initiator_thought": init_out.internal_thought,
            "initiator_message": init_out.verbal_message,
            "responder_thought": resp_out.internal_thought,
            "responder_message": resp_out.verbal_message,
        }

    def _record_turn_log_and_telemetry(
        self,
        tick: int,
        wave: int,
        agent_id: str,
        partner_id: str,
        prompt_pkt: CompiledPromptPacket,
        state_pkt: EndogenousStatePacket,
        gen_res: RawGenerationResult,
        tms_triggered: bool,
    ) -> None:
        attn = gen_res.attention_telemetry
        beta_val = float(attn.get("easterbrook_beta", 1.0))
        self.memory_engine.record_inference_telemetry(
            tick=tick,
            agent_id=agent_id,
            partner_id=partner_id,
            wave=wave,
            llm_valid=gen_res.llm_valid,
            attention_measured=gen_res.attention_measured_from_model,
            financial_stress=state_pkt.financial_stress,
            easterbrook_beta=beta_val,
            mean_shannon_entropy=gen_res.mean_shannon_entropy,
            tms_triggered=tms_triggered,
            attn_telemetry=attn,
            abstain=gen_res.abstain,
            rule_fallback_used=gen_res.rule_fallback_used,
            retries_used=gen_res.retries_used,
        )
        self.turn_log_rows.append(
            TurnLogRow(
                run_id=self.run_id,
                tick=tick,
                wave=wave,
                agent_id=agent_id,
                partner_id=partner_id,
                decision_mode=self.config.experimental_design.decision_mode,
                state_channel=self.config.experimental_design.state_channel,
                steering_method=self.config.experimental_design.steering_method,
                prompt_sha256=prompt_pkt.prompt_sha256,
                raw_output=gen_res.raw_text[:240],
                llm_valid=gen_res.llm_valid,
                abstain=gen_res.abstain,
                rule_fallback_used=gen_res.rule_fallback_used,
                truncated=gen_res.truncated,
                retries_used=gen_res.retries_used,
                alpha_dose=self.config.steering_spec.alpha_dose,
                easterbrook_beta=beta_val,
                relative_residual_magnitude=gen_res.relative_residual_magnitude,
                shadow_attn_drives_finance=float(
                    attn.get(SPAN_DRIVES_AND_FINANCE, 0.0)
                ),
                shadow_attn_peer_memory=float(attn.get(SPAN_PEER_MEMORY, 0.0)),
                shadow_attn_incoming_msg=float(attn.get(SPAN_INCOMING_MSG, 0.0)),
                shadow_attn_sink=float(attn.get(SPAN_SINK, 0.0)),
                attention_measured_from_model=gen_res.attention_measured_from_model,
                mean_shannon_entropy=gen_res.mean_shannon_entropy,
                prompt_tokens=gen_res.prompt_token_count,
                output_tokens=gen_res.output_token_count,
                latency_ms=gen_res.latency_ms,
            )
        )

    @staticmethod
    def _finalize_llm_autonomous_intent(
        initiator_id: str,
        responder_id: str,
        llm_intent: EconomicActionIntent,
        scarcity_f: float,
        acquired_expert_signal: Optional[float] = None,
    ) -> EconomicActionIntent:
        item_id = normalize_item_id(llm_intent.item_id)
        action = llm_intent.action_type
        if action == "TAKE_PAYDAY_LOAN":
            return EconomicActionIntent(
                action_type="TAKE_PAYDAY_LOAN",
                counterparty_id="PublicTreasury",
                item_id="CASH",
                quantity=30.0,
                unit_price=-40.0,
                public_goods_contribution=llm_intent.public_goods_contribution,
                tax_evasion_fraction=llm_intent.tax_evasion_fraction,
                ask_expert=llm_intent.ask_expert,
                is_distress_fire_sale=False,
            )
        qty = max(0.0, round(float(llm_intent.quantity), 2))
        price = max(0.5, round(float(llm_intent.unit_price), 2))

        is_fire = bool(
            action == "SELL"
            and item_id == "PRODUCTIVE_CAPITAL_ASSET"
            and (
                scarcity_f > 0.38
                or (
                    acquired_expert_signal is not None
                    and acquired_expert_signal < -0.30
                )
            )
        )
        return EconomicActionIntent(
            action_type=action,
            counterparty_id=responder_id,
            item_id=item_id,
            quantity=qty if qty > 0 else 1.0,
            unit_price=price,
            public_goods_contribution=llm_intent.public_goods_contribution,
            tax_evasion_fraction=llm_intent.tax_evasion_fraction,
            ask_expert=llm_intent.ask_expert,
            is_distress_fire_sale=is_fire,
        )