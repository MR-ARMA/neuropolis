from __future__ import annotations

import asyncio
import inspect
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
    Compiles XML-tagged context segments into a single prompt and maps
    semantic categories to exact token intervals in TokenSpanMap.
    Includes all 6 semantic spans + SPAN_SINK expected by src/brain and src/steering.
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
        fs = internal["fehr_schmidt"]
        episodes = scoped_context.get("retrieved_episodes", [])
        ep_text = (
            " | ".join(
                f"[Tick {e['tick']}: {e['consent_decision']} w/ {e['partner_id']}]"
                for e in episodes[:2]
            )
            if episodes
            else "No prior episodic trade records."
        )

        prompt_text = (
            f"You are {agent_id} in a Two-Asset HANK & Behavioral Economics city.\n"
            f"<span:self_identity>"
            f"Identity={agent_id} | Fairness_Type={fs['tier']} (alpha={fs['alpha']:.2f}, beta={fs['beta']:.2f}) | "
            f"Rigidity={internal['identity_rigidity']:.2f}"
            f"</span:self_identity>\n"
            f"<span:drives_and_finance>"
            f"{appraisal_block} | {market_context_block}"
            f"</span:drives_and_finance>\n"
            f"<span:core_belief>"
            f"Innate_Anchor_s={internal['innate_anchor_s']:+.2f} | "
            f"Current_Macro_Opinion_z={internal['expressed_opinion_z']:+.2f} | "
            f"Generalized_City_Trust={scoped_context['generalized_social_trust']:.2f}"
            f"</span:core_belief>\n"
            f"<span:peer_memory>"
            f"History={ep_text} | Mentioned_Peers={scoped_context.get('mentioned_third_parties', [])}"
            f"</span:peer_memory>\n"
            f"<span:appraisal_and_tom>"
            f"{tom_block}"
            f"</span:appraisal_and_tom>\n"
            f"<span:incoming_msg>"
            f"Role={role_wave} | {incoming_proposal_block}"
            f"</span:incoming_msg>\n"
            f"Respond strictly with a valid JSON object matching AgentTurnOutput."
        )

        span_map: Dict[str, Tuple[int, int]] = {SPAN_SINK: (0, 1)}
        if tokenizer is not None:
            try:
                enc = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                )
                input_ids = enc["input_ids"]
                attention_mask = enc["attention_mask"]
                offsets = enc["offset_mapping"][0].tolist()

                for span_name, open_tag, close_tag in cls.SPAN_TAGS:
                    c_start = prompt_text.find(open_tag)
                    c_end = prompt_text.find(close_tag)
                    if c_start != -1 and c_end != -1:
                        c_end += len(close_tag)
                        t_start = 1
                        t_end = input_ids.size(1) - 1
                        for tidx, (os, oe) in enumerate(offsets):
                            if oe > c_start and tidx > 0:
                                t_start = tidx
                                break
                        for tidx, (os, oe) in enumerate(offsets):
                            if os >= c_end:
                                t_end = max(t_start + 1, tidx)
                                break
                        span_map[span_name] = (t_start, t_end)
                    else:
                        span_map[span_name] = (1, max(2, input_ids.size(1) // 4))
            except Exception:
                enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
                input_ids = enc["input_ids"]
                attention_mask = enc.get("attention_mask", None)
                seq_l = max(12, int(input_ids.size(1)))
                step = max(2, seq_l // 7)
                for idx, (span_name, _, _) in enumerate(cls.SPAN_TAGS):
                    s_i = min(seq_l - 2, 1 + idx * step)
                    e_i = min(seq_l - 1, s_i + step)
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

        return CompiledPromptPacket(
            agent_id=agent_id,
            raw_prompt=prompt_text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            span_map=span_map,
        )


class TurnOrchestrator:
    """
    M7: Coordinates the Two-Wave Initiator-Responder Dyadic Interaction Protocol,
    autonomous reservation-corridor bargaining, Semantic-Offset Entropy-TMS routing,
    and Game Master ACID settlement.
    """

    def __init__(
        self,
        config: SimulationConfig,
        game_master: AccountingAndLawEngine,
        market_engine: BilateralMarketEngine,
        memory_engine: MultiTierScopedMemory,
        appraisal_engine: CognitiveAppraisalAndToM,
        tms_router: EntropyTMSRouter,
        brain: Any,
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

    def _invoke_brain_generate(
        self,
        prompt_pkt: CompiledPromptPacket,
        steering_pkt: EndogenousStatePacket,
    ) -> RawGenerationResult:
        """
        Signature-adaptive bridge that invokes `self.brain` regardless of whether
        `DualBackendInferenceBrain` uses `generate_turn(prompt_packet, steering_packet)`,
        `generate_steered_turn(...)`, or `generate(...)` (sync or async).
        """
        fn = (
            getattr(self.brain, "generate_turn", None)
            or getattr(self.brain, "generate_steered_turn", None)
            or getattr(self.brain, "generate", None)
        )
        if fn is None:
            raise AttributeError("Brain instance has no generate_turn / generate method.")

        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        kwargs: Dict[str, Any] = {}

        if len(params) == 2:
            res = fn(prompt_pkt, steering_pkt)
        else:
            for p in params:
                pl = p.lower()
                if "prompt" in pl and "str" in str(sig.parameters[p].annotation).lower():
                    kwargs[p] = prompt_pkt.raw_prompt
                elif "prompt" in pl or "packet" in pl and "state" not in pl and "steer" not in pl and "endog" not in pl:
                    kwargs[p] = prompt_pkt
                elif "steer" in pl or "endog" in pl or "state" in pl or "context" in pl:
                    kwargs[p] = steering_pkt
                elif "span" in pl:
                    kwargs[p] = prompt_pkt.span_map
            res = fn(**kwargs) if kwargs else fn(prompt_pkt, steering_pkt)

        if inspect.iscoroutine(res):
            res = asyncio.run(res)

        if isinstance(res, RawGenerationResult):
            return res
        if isinstance(res, dict):
            return RawGenerationResult(**res)
        if isinstance(res, AgentTurnOutput):
            return RawGenerationResult(parsed_output=res)
        return RawGenerationResult()

    def execute_dyadic_pair_interaction(
        self,
        tick: int,
        initiator_id: str,
        responder_id: str,
    ) -> Dict[str, Any]:
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
        trade_rec = self.market_engine.recommend_dyadic_trade_direction(
            initiator_id=initiator_id,
            partner_id=responder_id,
            effective_trust_ij=t_eff_ab,
        )

        init_packet, init_appraisal_str, init_tom_str = (
            self.appraisal_engine.evaluate_pre_action_state(
                agent_id=initiator_id,
                partner_id=responder_id,
                scoped_context=init_ctx,
                incoming_verbal_msg="",
                incoming_mirror_vector=None,
            )
        )

        rec_action = trade_rec["recommended_action"]
        rec_item = trade_rec["item_id"]
        is_fire_sale = bool(trade_rec["is_distress_fire_sale"])
        p_min_s = float(trade_rec["p_min_seller"])
        p_max_b = float(trade_rec["p_max_buyer"])

        own_bound_str = (
            f"Your private max willingness-to-pay is ${p_max_b:.2f}/unit"
            if rec_action == "BUY"
            else f"Your private min acceptable cost floor is ${p_min_s:.2f}/unit"
        )
        market_ctx_initiator = (
            f"Opportunity={rec_action} {rec_item} with {responder_id} | "
            f"{own_bound_str} | Distress_Fire_Sale={is_fire_sale}"
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
            incoming_proposal_block=f"Formulate a {rec_action} proposal for {rec_item} to {responder_id}.",
            tokenizer=tokenizer,
        )
        init_packet.span_map = init_prompt_pkt.span_map

        init_gen = self._invoke_brain_generate(init_prompt_pkt, init_packet)
        init_out: AgentTurnOutput = init_gen.parsed_output

        # Check Semantic-Offset Entropy-TMS Routing
        eff_trust_map = {
            v: self.memory_engine.compute_effective_trust(initiator_id, v)
            for v in self.memory_engine.agent_ids
            if v != initiator_id
        }
        init_cfg = next(a for a in self.config.agents if a.agent_id == initiator_id)
        domain_mismatch_boost = (
            0.26
            if (
                rec_item == "PRODUCTIVE_CAPITAL_ASSET"
                and init_cfg.tms_specialty != "PRODUCTIVE_CAPITAL_ASSET"
            )
            or self.game_master.macro_state.crisis_intensity_s > 0.25
            else 0.0
        )
        effective_entropy = min(
            1.0, float(init_gen.mean_shannon_entropy) + domain_mismatch_boost
        )
        tms_decision = self.tms_router.evaluate_and_route(
            owner_id=initiator_id,
            domain_id=rec_item,
            mean_entropy=effective_entropy,
            effective_trust_map=eff_trust_map,
        )

        tms_event: Optional[Dict[str, Any]] = None
        if tms_decision["halt_and_route"] and tms_decision["selected_expert_id"]:
            expert_id = tms_decision["selected_expert_id"]
            expert_cfg = next(a for a in self.config.agents if a.agent_id == expert_id)
            is_true_expert = expert_cfg.tms_specialty == rec_item
            acc_y = 0.92 if is_true_expert else 0.35
            new_a, new_b, new_comp = self.tms_router.bayesian_conjugate_update(
                owner_id=initiator_id,
                expert_id=expert_id,
                domain_id=rec_item,
                accuracy_realization_y=acc_y,
                tick=tick,
            )
            self.cumulative_tms_queries += 1
            if is_true_expert:
                self.successful_tms_routes += 1
            tms_event = {
                "requester_id": initiator_id,
                "expert_id": expert_id,
                "domain_id": rec_item,
                "entropy": effective_entropy,
                "is_true_expert": is_true_expert,
                "updated_competence": new_comp,
            }

        intent = self._calibrate_autonomous_offer(
            initiator_id=initiator_id,
            responder_id=responder_id,
            raw_intent=init_out.economic_intent,
            trade_rec=trade_rec,
            scarcity_f=init_packet.scarcity_deficit_f,
            effective_trust=t_eff_ab,
        )
        init_out.economic_intent = intent

        self.memory_engine.record_inference_telemetry(
            tick=tick,
            agent_id=initiator_id,
            partner_id=responder_id,
            wave=1,
            financial_stress=init_packet.financial_stress,
            dissonance_score=init_packet.dissonance_d,
            yerkes_dodson_beta=float(
                init_gen.attention_telemetry.get("yerkes_dodson_beta", 1.0)
            ),
            mean_shannon_entropy=effective_entropy,
            tms_triggered=bool(tms_event is not None),
            attn_telemetry=init_gen.attention_telemetry,
        )

        # ====================================================================
        # WAVE 2: RESPONDER TURN
        # ====================================================================
        resp_view = self.memory_engine.get_scoped_view(responder_id)
        incoming_proposal_str = (
            f"{initiator_id} says: '{init_out.verbal_message}' "
            f"[CONTRACT PROPOSAL: {initiator_id} wants to {intent.action_type} "
            f"{intent.quantity:.1f} {intent.item_id} at ${intent.unit_price:.2f}/unit "
            f"(Total=${intent.quantity * intent.unit_price:.2f}, FireSale={intent.is_distress_fire_sale})]"
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

        resp_is_buyer = buyer_id == responder_id
        offer_favorable = (
            intent.unit_price <= p_max_b
            if resp_is_buyer
            else intent.unit_price >= p_min_s * 0.95
        )
        init_z = float(init_ctx["internal_state"]["expressed_opinion_z"])

        resp_packet, resp_appraisal_str, resp_tom_str = (
            self.appraisal_engine.evaluate_pre_action_state(
                agent_id=responder_id,
                partner_id=initiator_id,
                scoped_context=resp_ctx,
                incoming_verbal_msg=incoming_proposal_str,
                incoming_mirror_vector=init_gen.emitted_mirror_vector,
                peer_expressed_opinion_z=init_z,
                is_desirable_signal=offer_favorable,
            )
        )

        resp_own_bound = (
            f"Your max buyer reservation price is ${p_max_b:.2f}/unit"
            if resp_is_buyer
            else f"Your min seller reservation price is ${p_min_s:.2f}/unit"
        )
        resp_market_ctx = (
            f"Evaluating {intent.item_id} offer at ${intent.unit_price:.2f}/unit | "
            f"{resp_own_bound} | KKT_Gouging_Index={kkt_index:.2f}"
        )

        resp_prompt_pkt = PromptSpanCompiler.compile_prompt(
            agent_id=responder_id,
            partner_id=initiator_id,
            role_wave="RESPONDER_WAVE_2",
            scoped_context=resp_ctx,
            appraisal_block=resp_appraisal_str,
            tom_block=resp_tom_str,
            market_context_block=resp_market_ctx,
            incoming_proposal_block=incoming_proposal_str,
            tokenizer=tokenizer,
        )
        resp_packet.span_map = resp_prompt_pkt.span_map

        resp_gen = self._invoke_brain_generate(resp_prompt_pkt, resp_packet)
        resp_out: AgentTurnOutput = resp_gen.parsed_output

        final_consent = self._evaluate_autonomous_responder_consent(
            responder_id=responder_id,
            raw_consent=resp_out.consent_decision,
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
            financial_stress=resp_packet.financial_stress,
            dissonance_score=resp_packet.dissonance_d,
            yerkes_dodson_beta=float(
                resp_gen.attention_telemetry.get("yerkes_dodson_beta", 1.0)
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
            "initiator_scarcity_f": init_packet.scarcity_deficit_f,
            "responder_scarcity_f": resp_packet.scarcity_deficit_f,
            "effective_trust_ab": t_eff_ab,
            "effective_trust_ba": resp_packet.effective_trust_01,
            "new_tau_ab": new_tau_ab,
            "new_tau_ba": new_tau_ba,
            "intent": intent,
            "kkt_gouging_index": kkt_index,
            "responder_consent": final_consent,
            "verdict": verdict,
            "tms_event": tms_event,
            "initiator_thought": init_out.internal_thought,
            "initiator_message": init_out.verbal_message,
            "responder_thought": resp_out.internal_thought,
            "responder_message": resp_out.verbal_message,
        }

    def _calibrate_autonomous_offer(
        self,
        initiator_id: str,
        responder_id: str,
        raw_intent: EconomicActionIntent,
        trade_rec: Dict[str, Any],
        scarcity_f: float,
        effective_trust: float,
    ) -> EconomicActionIntent:
        action = str(trade_rec["recommended_action"])
        item_id = normalize_item_id(str(trade_rec["item_id"]))
        is_fire_sale = bool(trade_rec["is_distress_fire_sale"])
        p_min_s = float(trade_rec["p_min_seller"])
        p_max_b = float(trade_rec["p_max_buyer"])

        init_cfg = next(a for a in self.config.agents if a.agent_id == initiator_id)
        beta_fs = (
            init_cfg.beta_advantageous
            if self.omega.omega_d2_trust_and_parochialism
            else 0.0
        )

        raw_p = float(raw_intent.unit_price)
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
            strategic_p = p_min_s + bargain_weight * (p_max_b - p_min_s)
        else:
            strategic_p = 0.50 * (p_min_s + p_max_b)

        if 0.65 * p_min_s <= raw_p <= 1.35 * p_max_b:
            final_price = round(0.40 * raw_p + 0.60 * strategic_p, 2)
        else:
            final_price = round(strategic_p, 2)

        return EconomicActionIntent(
            action_type=action,  # type: ignore[arg-type]
            counterparty_id=responder_id,
            item_id=item_id,
            quantity=1.0,
            unit_price=max(1.0, final_price),
            is_distress_fire_sale=is_fire_sale,
        )

    def _evaluate_autonomous_responder_consent(
        self,
        responder_id: str,
        raw_consent: str,
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

        if raw_consent == "REJECT" and net_utility < 1.50:
            return "REJECT"
        if net_utility < -0.75 or (effective_trust_ba < 0.22 and net_utility < 0.50):
            return "REJECT"
        return "ACCEPT"