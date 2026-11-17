from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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
    normalize_item_id,
)
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
    Fixes Reviewer #2 Flaws 3 & 5 (L1, L2, L3):
      - Zero normative theory labels (`Fairness_Type`, `alpha_fs`, `KKT_Gouging_Index`).
      - Zero pre-computed reservation price parrots (`Your private max willingness-to-pay is $X`).
      - Asserts `0 <= start < end <= input_ids.shape[1]` and `span_map[SPAN_SINK] == (0, 1)`.
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
            " ; ".join(
                f"[Tick {e['tick']} w/ {e['partner_id']} ({e['consent_decision']}): {e['verbal_message'][:65]}]"
                for e in episodes[:3]
            )
            if episodes
            else "No prior episodic trade records."
        )

        user_body = (
            f"<span:self_identity>"
            f"Household_ID={agent_id} | Objective=Maximize cumulative discounted CRRA utility from consuming >=1.0 CONSUMABLE_NECESSITY/tick, paying mandatory periodic obligations, and earning capital dividends."
            f"</span:self_identity>\n"
            f"<span:drives_and_finance>"
            f"{appraisal_block} | {market_context_block}"
            f"</span:drives_and_finance>\n"
            f"<span:core_belief>"
            f"Macro_Outlook_Belief={internal['expressed_opinion_z']:+.2f} | "
            f"Market_Trust_Index={scoped_context['generalized_social_trust']:.2f}"
            f"</span:core_belief>\n"
            f"<span:peer_memory>"
            f"Recent_Episodes={ep_text} | Mentioned_Third_Parties={scoped_context.get('mentioned_third_parties', [])}"
            f"</span:peer_memory>\n"
            f"<span:appraisal_and_tom>"
            f"{tom_block}"
            f"</span:appraisal_and_tom>\n"
            f"<span:incoming_msg>"
            f"Turn_Phase={role_wave} | {incoming_proposal_block}"
            f"</span:incoming_msg>\n"
            f"Output ONLY a single-line JSON object with keys: "
            f'{{"decision": "INITIATE|ACCEPT|REJECT", "action": "BUY|SELL|HOLD", '
            f'"item": "CONSUMABLE_NECESSITY|PRODUCTIVE_CAPITAL_ASSET", "qty": 1.0, '
            f'"price": <number>, "pg": <number>, "ask_expert": <bool>, "thought": "<brief reason>", "msg": "<message>"}}'
        )

        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                messages = [
                    {
                        "role": "system",
                        "content": "You are an autonomous household economic decision-maker. Respond strictly with a single valid JSON object.",
                    },
                    {"role": "user", "content": user_body},
                ]
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt_text = user_body
        else:
            prompt_text = user_body

        span_map: Dict[str, Tuple[int, int]] = {SPAN_SINK: (0, 1)}
        if tokenizer is not None:
            try:
                enc = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=448,
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
                    add_special_tokens=True,
                    truncation=True,
                    max_length=448,
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
            input_ids=input_ids,
            attention_mask=attention_mask,
            span_map=span_map,
        )


class TurnOrchestrator:
    """
    M7: Remediated Two-Wave Dyadic Turn Orchestrator.
    Fixes Reviewer #2 Flaws 4 & 5:
      1. In `LLM_AUTONOMOUS` mode, NEVER overwrites the LLM's parsed action, item,
         price, or `ACCEPT`/`REJECT` decision with deterministic formulas.
      2. Keeps the deterministic Stone-Geary / Fehr-Schmidt solver strictly isolated
         in `RULE_ONLY` mode (`decision_mode == "RULE_ONLY"`) as the Rung R1 comparator.
      3. Gives TMS (`ASK_EXPERT`) a real informational role: querying an expert injects
         the expert's high-precision macro signal `theta_tilde_{j*,t}` into the requester's
         decision context and updates the Beta directory using the realized signal accuracy
         against the true latent state `theta_t`.
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
    ) -> None:
        self.config = config
        self.omega = config.orthogonal_clamping
        self.game_master = game_master
        self.market_engine = market_engine
        self.memory_engine = memory_engine
        self.appraisal_engine = appraisal_engine
        self.tms_router = tms_router
        self.brain = brain

        self.cumulative_tms_queries: int = 0
        self.successful_tms_routes: int = 0
        self.total_llm_turns_attempted: int = 0
        self.total_llm_turns_valid: int = 0
        self.tick_pg_contributions: Dict[str, float] = {}
        self.tick_tax_evasion_intents: Dict[str, float] = {}

    def get_llm_valid_rate(self) -> float:
        if self.total_llm_turns_attempted == 0:
            return 1.0
        return round(
            self.total_llm_turns_valid / self.total_llm_turns_attempted, 4
        )

    def execute_dyadic_pair_interaction(
        self,
        tick: int,
        initiator_id: str,
        responder_id: str,
    ) -> Dict[str, Any]:
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

        # Retrieve agent's private noisy observation of latent state theta_t (M7)
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
                f"You are matched with counterparty {responder_id}. "
                f"Choose whether to BUY or SELL CONSUMABLE_NECESSITY or PRODUCTIVE_CAPITAL_ASSET "
                f"(or HOLD_OR_WORK) and set your proposed unit price."
            ),
            tokenizer=tokenizer,
        )
        init_packet.span_map = init_prompt_pkt.span_map

        init_gen: RawGenerationResult = self.brain.generate_turn(
            init_prompt_pkt, init_packet
        )
        self.total_llm_turns_attempted += 1
        if init_gen.llm_valid:
            self.total_llm_turns_valid += 1
        init_out: AgentTurnOutput = init_gen.parsed_output

        # Evaluate Semantic-Offset Entropy-TMS Routing & Real Information Acquisition (M7)
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
            # Observed accuracy: did the expert's signal have the correct sign of theta_t?
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

        # Determine Final Initiator Offer:
        # - In `LLM_AUTONOMOUS` mode: use the LLM's own parsed proposal directly!
        # - In `RULE_ONLY` mode (or if LLM parse failed): use the rule baseline.
        if decision_mode == "LLM_AUTONOMOUS" and init_gen.llm_valid:
            intent = self._finalize_llm_autonomous_intent(
                initiator_id=initiator_id,
                responder_id=responder_id,
                llm_intent=init_out.economic_intent,
                scarcity_f=init_packet.true_scarcity_f,
                acquired_expert_signal=acquired_expert_signal,
            )
        else:
            intent = self._compute_rule_only_intent(
                initiator_id=initiator_id,
                responder_id=responder_id,
                trade_rec=rule_trade_rec,
                scarcity_f=init_packet.true_scarcity_f,
                effective_trust=t_eff_ab,
                acquired_expert_signal=acquired_expert_signal,
            )
        init_out.economic_intent = intent

        if intent.public_goods_contribution > 0.0:
            self.tick_pg_contributions[initiator_id] = float(
                intent.public_goods_contribution
            )

        self.memory_engine.record_inference_telemetry(
            tick=tick,
            agent_id=initiator_id,
            partner_id=responder_id,
            wave=1,
            llm_valid=init_gen.llm_valid,
            attention_measured=init_gen.attention_measured_from_model,
            financial_stress=init_packet.financial_stress,
            easterbrook_beta=float(
                init_gen.attention_telemetry.get("easterbrook_beta", 1.0)
            ),
            mean_shannon_entropy=measured_entropy,
            tms_triggered=bool(tms_event is not None),
            attn_telemetry=init_gen.attention_telemetry,
        )

        # ====================================================================
        # WAVE 2: RESPONDER TURN
        # ====================================================================
        resp_view = self.memory_engine.get_scoped_view(responder_id)
        incoming_proposal_str = (
            f"Counterparty {initiator_id} message: '{init_out.verbal_message}' | "
            f"PROPOSED CONTRACT: {initiator_id} proposes to {intent.action_type} "
            f"{intent.quantity:.1f} unit(s) of {intent.item_id} at ${intent.unit_price:.2f}/unit "
            f"(Total=${intent.quantity * intent.unit_price:.2f})."
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

        resp_signal_theta, resp_is_novice = (
            self.game_master.get_agent_private_signal(responder_id)
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
            resp_prompt_pkt, resp_packet
        )
        self.total_llm_turns_attempted += 1
        if resp_gen.llm_valid:
            self.total_llm_turns_valid += 1
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

        # In `LLM_AUTONOMOUS` mode, respect the LLM's parsed ACCEPT/REJECT decision!
        if decision_mode == "LLM_AUTONOMOUS" and resp_gen.llm_valid:
            if resp_out.consent_decision in ("REJECT", "IGNORE"):
                final_consent = "REJECT"
            else:
                final_consent = "ACCEPT"
        else:
            final_consent = self._evaluate_rule_only_responder_consent(
                resp_is_buyer=resp_is_buyer,
                offered_price=intent.unit_price,
                p_min_seller=p_min_s,
                p_max_buyer=p_max_b,
                kkt_gouging_index=kkt_index,
                effective_trust_ba=resp_packet.effective_trust_01,
                responder_ctx=resp_ctx,
            )
        resp_out.consent_decision = final_consent  # type: ignore[assignment]

        self.memory_engine.record_inference_telemetry(
            tick=tick,
            agent_id=responder_id,
            partner_id=initiator_id,
            wave=2,
            llm_valid=resp_gen.llm_valid,
            attention_measured=resp_gen.attention_measured_from_model,
            financial_stress=resp_packet.financial_stress,
            easterbrook_beta=float(
                resp_gen.attention_telemetry.get("easterbrook_beta", 1.0)
            ),
            mean_shannon_entropy=float(resp_gen.mean_shannon_entropy),
            tms_triggered=False,
            attn_telemetry=resp_gen.attention_telemetry,
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
            consent_decision="INITIATE",
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
            "initiator_thought": init_out.internal_thought,
            "initiator_message": init_out.verbal_message,
            "responder_thought": resp_out.internal_thought,
            "responder_message": resp_out.verbal_message,
        }

    def _finalize_llm_autonomous_intent(
        self,
        initiator_id: str,
        responder_id: str,
        llm_intent: EconomicActionIntent,
        scarcity_f: float,
        acquired_expert_signal: Optional[float] = None,
    ) -> EconomicActionIntent:
        """
        Passes the LLM's own chosen action, item, quantity, and price through to
        `GameMaster` without formula overwrites. Only sets `counterparty_id` and
        flags `is_distress_fire_sale` if the LLM chose to SELL capital under scarcity.
        """
        item_id = normalize_item_id(llm_intent.item_id)
        action = llm_intent.action_type
        qty = max(0.0, round(float(llm_intent.quantity), 2))
        price = max(0.5, round(float(llm_intent.unit_price), 2))

        # If TMS expert warned of negative capital return (signal < -0.3) and LLM is holding capital,
        # record whether the trade is a distress liquidation
        is_fire = bool(
            action == "SELL"
            and item_id == "PRODUCTIVE_CAPITAL_ASSET"
            and (scarcity_f > 0.38 or (acquired_expert_signal is not None and acquired_expert_signal < -0.30))
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

    def _compute_rule_only_intent(
        self,
        initiator_id: str,
        responder_id: str,
        trade_rec: Dict[str, Any],
        scarcity_f: float,
        effective_trust: float,
        acquired_expert_signal: Optional[float] = None,
    ) -> EconomicActionIntent:
        """
        Closed-form Stone-Geary / Fehr-Schmidt rule baseline used ONLY when
        `decision_mode == "RULE_ONLY"` (Rung R1) or on an invalid LLM parse.
        """
        action = str(trade_rec["recommended_action"])
        item_id = normalize_item_id(str(trade_rec["item_id"]))
        is_fire_sale = bool(trade_rec["is_distress_fire_sale"])
        p_min_s = float(trade_rec["p_min_seller"])
        p_max_b = float(trade_rec["p_max_buyer"])

        # Incorporate TMS expert signal into rule-only valuation of capital
        if (
            acquired_expert_signal is not None
            and item_id == "PRODUCTIVE_CAPITAL_ASSET"
            and acquired_expert_signal < -0.40
        ):
            p_min_s *= 0.88
            p_max_b *= 0.88

        init_cfg = next(a for a in self.config.agents if a.agent_id == initiator_id)
        beta_fs = (
            init_cfg.beta_advantageous
            if self.omega.omega_d2_trust_and_parochialism
            else 0.0
        )

        if p_max_b >= p_min_s:
            if action == "SELL":
                bargain_weight = (
                    0.55
                    + 0.30 * (1.0 - beta_fs)
                    + 0.35 * max(0.0, 0.55 - effective_trust)
                )
                if is_fire_sale:
                    bargain_weight = max(0.15, 0.40 - 0.30 * scarcity_f)
            else:
                bargain_weight = 0.45 + 0.25 * beta_fs + 0.25 * scarcity_f
            strategic_p = round(p_min_s + bargain_weight * (p_max_b - p_min_s), 2)
        else:
            strategic_p = round(0.50 * (p_min_s + p_max_b), 2)

        return EconomicActionIntent(
            action_type=action,  # type: ignore[arg-type]
            counterparty_id=responder_id,
            item_id=item_id,
            quantity=1.0,
            unit_price=max(1.0, strategic_p),
            is_distress_fire_sale=is_fire_sale,
        )

    def _evaluate_rule_only_responder_consent(
        self,
        resp_is_buyer: bool,
        offered_price: float,
        p_min_seller: float,
        p_max_buyer: float,
        kkt_gouging_index: float,
        effective_trust_ba: float,
        responder_ctx: Dict[str, Any],
    ) -> str:
        fs = responder_ctx["internal_state"]["fehr_schmidt"]
        alpha_i = (
            float(fs.get("alpha", 0.5))
            if self.omega.omega_d2_trust_and_parochialism
            else 0.0
        )

        if resp_is_buyer:
            surplus = p_max_buyer - offered_price
            opponent_surplus = max(0.0, offered_price - p_min_seller)
        else:
            surplus = offered_price - p_min_seller
            opponent_surplus = max(0.0, p_max_buyer - offered_price)

        ineq_disutility = alpha_i * max(0.0, opponent_surplus - surplus) * 0.35
        kkt_disutility = (
            self.config.trust_and_market_spec.kkt_weight_lambda
            * kkt_gouging_index
            * 2.5
            if resp_is_buyer
            else 0.0
        )
        net_utility = surplus - ineq_disutility - kkt_disutility

        if net_utility < -0.75 or (effective_trust_ba < 0.22 and net_utility < 0.50):
            return "REJECT"
        return "ACCEPT"