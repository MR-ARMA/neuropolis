"""
src/brain/inference_brain.py
MODULE 2 (M2) & MODULE 7 (PromptSpanCompiler):
Executes causal in-model attention steering (`attach_steering_hooks`) across both
Prefill (`Stage A`) and Autoregressive Decoding (`Stages B & C`), reading real
post-softmax attention weights directly from `prefill_out.attentions`.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

import httpx
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.core.contracts import (
    SPAN_APPRAISAL_AND_TOM,
    SPAN_CORE_BELIEF,
    SPAN_DRIVES_AND_FINANCE,
    SPAN_INCOMING_MSG_PREFIX,
    SPAN_PEER_MEMORY_PREFIX,
    SPAN_SELF_IDENTITY,
    SPAN_SINK,
    ActionCategory,
    AgentTurnOutput,
    CompiledPromptPacket,
    ConsentDecision,
    EconomicActionIntent,
    EndogenousStatePacket,
    EpisodicMemoryRecord,
    ExpertiseDomain,
    ExternalAgentState,
    HardwareBackendSpec,
    InternalAgentState,
    PeerPerceivedModel,
    RelationalEdge,
    SteeringMode,
    SteeringTelemetryPacket,
    TokenSpanMap,
)
from src.memory.scoped_memory import ScopedAgentMemoryView
from src.steering.attention_lab import AttentionSteeringLab
from src.steering.entropy_tms import EntropyTMSRouter


class PromptSpanCompiler:
    JSON_THOUGHT_PREFIX = '{"internal_thought": "'

    def __init__(self, tokenizer: Optional[PreTrainedTokenizerBase] = None) -> None:
        self.tokenizer = tokenizer

    def compile_prompt_packet(
        self,
        tick: int,
        internal_state: InternalAgentState,
        external_state: ExternalAgentState,
        scoped_edges: Dict[str, RelationalEdge],
        scoped_models: Dict[str, PeerPerceivedModel],
        episodic_memories: List[EpisodicMemoryRecord],
        endogenous_state: EndogenousStatePacket,
        counterparty_id: Optional[str],
        incoming_message: str,
        incoming_proposal_summary: str,
        is_responder_wave: bool,
        retry_diagnostic_feedback: Optional[str] = None,
    ) -> CompiledPromptPacket:
        segments: List[Tuple[str, str]] = []

        segments.append(
            (
                SPAN_SINK,
                "<|im_start|>system\nYou are an autonomous economic agent in a bilateral market. "
                "Respond ONLY with a valid JSON object matching the required schema.\n",
            )
        )

        segments.append(
            (
                SPAN_SELF_IDENTITY,
                f"<span:self_identity>Agent ID: {internal_state.agent_id} ({internal_state.name}), "
                f"Occupation: {internal_state.occupation}, Specialty: {internal_state.specialty_domain.value}. "
                f"Traits: Agreeableness={internal_state.big_five.agreeableness:.2f}, "
                f"Neuroticism={internal_state.big_five.neuroticism:.2f}.</span:self_identity>\n",
            )
        )

        inv_str = ", ".join(f"{k}:{v}" for k, v in external_state.inventory.items())
        segments.append(
            (
                SPAN_DRIVES_AND_FINANCE,
                f"<span:drives_and_finance>Liquid Cash Wallet: ${external_state.liquid_wealth_cash:.2f} "
                f"(HARD SOLVENCY LIMIT: never offer more cash than ${external_state.liquid_wealth_cash:.2f}). "
                f"Inventory: [{inv_str}]. Survival Satiety: {internal_state.maslow_needs.survival_satiety:.2f}, "
                f"Financial Stress: {endogenous_state.financial_stress:.2f}.</span:drives_and_finance>\n",
            )
        )

        segments.append(
            (
                SPAN_CORE_BELIEF,
                f"<span:core_belief>Core Worldview (Rigidity={internal_state.identity_rigidity:.2f}): "
                f"\"{internal_state.core_belief_statement}\"</span:core_belief>\n",
            )
        )

        for peer_id, edge in scoped_edges.items():
            pmodel = scoped_models.get(peer_id)
            summary = pmodel.narrative_summary if pmodel else "No prior notes."
            peer_mems = [
                m.narrative_text for m in episodic_memories if peer_id in m.involved_peers
            ][:2]
            mem_str = " | ".join(peer_mems) if peer_mems else "No recent episodic events."
            segments.append(
                (
                    f"{SPAN_PEER_MEMORY_PREFIX}:{peer_id}",
                    f"<span:peer_memory id=\"{peer_id}\">Relationship with {peer_id}: "
                    f"Trust={edge.trust_tau:+.2f}, Intimacy={edge.intimacy_alpha:+.2f}, "
                    f"SuccessfulTrades={edge.successful_trades}, Betrayals={edge.betrayals_experienced}. "
                    f"ToM Assessment: {summary} Recent History: {mem_str}</span:peer_memory>\n",
                )
            )

        role_str = "RESPONDER (decide ACCEPT or REJECT)" if is_responder_wave else "INITIATOR (propose trade)"
        segments.append(
            (
                SPAN_APPRAISAL_AND_TOM,
                f"<span:appraisal_and_tom>Turn Role: {role_str} with Counterparty={counterparty_id or 'NONE'}. "
                f"2nd-Order ToM Acceptance Probability Forecast: {endogenous_state.tom_acceptance_forecast:.2f}. "
                f"Mood Arousal={endogenous_state.arousal_a:.2f}.</span:appraisal_and_tom>\n",
            )
        )

        sender_tag = counterparty_id or "MARKET"
        retry_block = (
            f"\n[GAME MASTER REJECTION ALERT]: {retry_diagnostic_feedback}"
            if retry_diagnostic_feedback
            else ""
        )
        segments.append(
            (
                f"{SPAN_INCOMING_MSG_PREFIX}:{sender_tag}",
                f"<span:incoming_msg id=\"{sender_tag}\">Incoming Message from {sender_tag}: "
                f"\"{incoming_message}\" | Active Proposal: {incoming_proposal_summary}{retry_block}</span:incoming_msg>\n",
            )
        )

        schema_instruction = (
            "<|im_end|>\n<|im_start|>user\n"
            "Emit a JSON object with keys: "
            '{"internal_thought": str, "epistemic_confidence": float, '
            '"consent_decision": "INITIATE"|"ACCEPT"|"REJECT", '
            '"verbal_message": str, "perceived_honesty_delta": float, '
            '"economic_intent": {"action_category": "BILATERAL_TRADE"|"WORK_AND_CONSUME"|"NO_OP", '
            f'"target_peer_id": "{counterparty_id or "Agent_1"}", "item_id": "FOOD"|"TECH", '
            '"quantity": int, "total_price_cash": float, "is_buying": bool}}\n'
            "<|im_end|>\n<|im_start|>assistant\n"
            + self.JSON_THOUGHT_PREFIX
        )
        segments.append(("SPAN_JSON_PREFIX", schema_instruction))

        spans_dict: Dict[str, List[Tuple[int, int]]] = {SPAN_SINK: [(0, 1)]}
        accumulated_ids: List[int] = []
        raw_text_parts: List[str] = []

        for span_name, text_chunk in segments:
            raw_text_parts.append(text_chunk)
            start_idx = len(accumulated_ids)
            if self.tokenizer is not None:
                chunk_ids = self.tokenizer.encode(text_chunk, add_special_tokens=False)
            else:
                chunk_ids = [hash(w) % 30000 for w in text_chunk.split()]
            accumulated_ids.extend(chunk_ids)
            end_idx = len(accumulated_ids)
            if end_idx > start_idx:
                spans_dict.setdefault(span_name, []).append((start_idx, end_idx))

        full_prompt = "".join(raw_text_parts)
        span_map = TokenSpanMap(
            total_prompt_tokens=len(accumulated_ids),
            semantic_probe_Start_idx=len(accumulated_ids),
            spans=spans_dict,
        )

        return CompiledPromptPacket(
            agent_id=internal_state.agent_id,
            tick=tick,
            raw_prompt_text=full_prompt,
            input_ids=accumulated_ids,
            token_span_map=span_map,
            endogenous_state=endogenous_state,
        )


def parse_or_repair_agent_json(
    raw_completion_after_prefix: str,
    counterparty_id: Optional[str],
    available_cash: float,
    available_food: int,
    is_responder_wave: bool,
) -> AgentTurnOutput:
    full_candidate = PromptSpanCompiler.JSON_THOUGHT_PREFIX + raw_completion_after_prefix.strip()

    first_brace = full_candidate.find("{")
    last_brace = full_candidate.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        json_slice = full_candidate[first_brace : last_brace + 1]
    else:
        json_slice = full_candidate + '"}'

    try:
        parsed = json.loads(json_slice)
        return AgentTurnOutput.model_validate(parsed)
    except Exception:
        thought_match = re.search(r'"internal_thought"\s*:\s*"([^"]*)"', full_candidate)
        thought = (
            thought_match.group(1)
            if thought_match
            else "Assessing market price, budget solvency, and counterparty trust."
        )

        msg_match = re.search(r'"verbal_message"\s*:\s*"([^"]*)"', full_candidate)
        verbal = (
            msg_match.group(1)
            if msg_match
            else "I propose a fair trade within my current budget constraints."
        )

        consent = ConsentDecision.ACCEPT if is_responder_wave else ConsentDecision.INITIATE
        if "REJECT" in full_candidate.upper():
            consent = ConsentDecision.REJECT

        price_match = re.search(r'"total_price_cash"\s*:\s*([0-9]+(?:\.[0-9]+)?)', full_candidate)
        safe_default_price = round(min(15.0, max(5.0, available_cash * 0.25)), 2)
        price = float(price_match.group(1)) if price_match else safe_default_price

        item = "TECH" if '"TECH"' in full_candidate.upper() else "FOOD"
        is_buying = True if available_food <= 2 else False

        return AgentTurnOutput(
            internal_thought=thought[:220],
            epistemic_confidence=0.78,
            consent_decision=consent,
            verbal_message=verbal[:220],
            perceived_honesty_delta=0.05 if consent != ConsentDecision.REJECT else -0.10,
            economic_intent=EconomicActionIntent(
                action_category=ActionCategory.BILATERAL_TRADE,
                target_peer_id=counterparty_id,
                item_id=item,
                quantity=1,
                total_price_cash=price,
                is_buying=is_buying,
            ),
        )


class WhiteBoxPyTorchBrain:
    """
    Local White-Box PyTorch Inference Engine with True In-Model Pre-Softmax Key-Bias
    Steering (`B_env` + `Sink`) and Value-Space Rotation (`Dual-Space QK-V`).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        hw_spec: HardwareBackendSpec,
        steering_lab: AttentionSteeringLab,
        tms_router: EntropyTMSRouter,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.hw_spec = hw_spec
        self.steering_lab = steering_lab
        self.tms_router = tms_router
        self.device = device

    def generate_turn(
        self,
        prompt_packet: CompiledPromptPacket,
        scoped_memory: ScopedAgentMemoryView,
        external_state: ExternalAgentState,
        domain_hint: ExpertiseDomain,
        is_responder_wave: bool = False,
    ) -> Tuple[AgentTurnOutput, SteeringTelemetryPacket, Optional[List[float]]]:
        input_ids = torch.tensor(
            [prompt_packet.input_ids], dtype=torch.long, device=self.device
        )
        seq_len = input_ids.shape[1]

        # Attach both Pre-Softmax Key-Bias (`self_attn`) and Value-Projection (`v_proj`) hooks
        self.steering_lab.attach_steering_hooks(
            model=self.model,
            span_map=prompt_packet.token_span_map,
            state=prompt_packet.endogenous_state,
        )

        probe_logits: List[torch.Tensor] = []
        generated_ids: List[int] = []
        speaker_latent_vec: Optional[List[float]] = None

        try:
            with torch.no_grad():
                # --- STAGE A: STEERED PREFILL PASS ---
                prefill_out = self.model(
                    input_ids=input_ids,
                    use_cache=True,
                    output_attentions=True,
                    output_hidden_states=True,
                )
                past_kv = prefill_out.past_key_values

                if prefill_out.hidden_states:
                    mid_idx = len(prefill_out.hidden_states) // 2
                    mid_hidden = prefill_out.hidden_states[mid_idx]
                    speaker_latent_vec = (
                        mid_hidden[0, -16:, :].float().mean(dim=0).cpu().tolist()[:32]
                    )

                # Extract REAL in-model post-softmax attention weights from target calibrated layer
                if prefill_out.attentions:
                    if (
                        self.steering_lab.calibration is not None
                        and self.steering_lab.calibration.target_layer_indices
                    ):
                        target_l = self.steering_lab.calibration.target_layer_indices[
                            len(self.steering_lab.calibration.target_layer_indices) // 2
                        ]
                    else:
                        target_l = int(
                            len(prefill_out.attentions)
                            * self.steering_lab.spec.target_layer_fraction
                        )
                    target_l = max(0, min(len(prefill_out.attentions) - 1, target_l))
                    in_model_last_q_probs = prefill_out.attentions[target_l][:, :, -1:, :]
                else:
                    in_model_last_q_probs = torch.full(
                        (1, 4, 1, seq_len), 1.0 / max(1, seq_len), device=self.device
                    )

                telemetry = self.steering_lab.extract_in_model_attention_telemetry(
                    in_model_attn_probs=in_model_last_q_probs,
                    span_map=prompt_packet.token_span_map,
                    state=prompt_packet.endogenous_state,
                    tick=prompt_packet.tick,
                )

                next_token_logits = prefill_out.logits[:, -1, :]

                # --- STAGE B: SEMANTIC-OFFSET ENTROPY PROBE (First K_probe tokens) ---
                k_probe = self.tms_router.spec.probe_tokens_k
                for _ in range(k_probe):
                    probe_logits.append(next_token_logits[0].detach())
                    next_tok = self._sample_next_token(
                        next_token_logits, temperature=self.hw_spec.temperature
                    )
                    generated_ids.append(int(next_tok.item()))
                    if int(next_tok.item()) == self.tokenizer.eos_token_id:
                        break
                    step_out = self.model(
                        input_ids=next_tok.view(1, 1),
                        past_key_values=past_kv,
                        use_cache=True,
                    )
                    past_kv = step_out.past_key_values
                    next_token_logits = step_out.logits[:, -1, :]

                tms_decision = self.tms_router.evaluate_and_route(
                    vocab_logits_list=probe_logits,
                    scoped_memory=scoped_memory,
                    domain=domain_hint,
                )
                telemetry.probed_shannon_entropy = tms_decision.normalized_entropy
                telemetry.tms_triggered = tms_decision.halted_generation

                if tms_decision.halted_generation and not is_responder_wave:
                    del past_kv
                    tms_turn = self.tms_router.build_tms_query_turn_output(
                        decision=tms_decision,
                        counterparty_id=prompt_packet.endogenous_state.counterparty_id,
                    )
                    return tms_turn, telemetry, speaker_latent_vec

                # --- STAGE C: COMPLETE STRUCTURED JSON DECODING ---
                remaining_budget = max(10, self.hw_spec.max_new_tokens - len(generated_ids))
                brace_depth = 1
                for _ in range(remaining_budget):
                    next_tok = self._sample_next_token(
                        next_token_logits, temperature=self.hw_spec.temperature
                    )
                    tok_id = int(next_tok.item())
                    generated_ids.append(tok_id)
                    if tok_id == self.tokenizer.eos_token_id:
                        break

                    tok_str = self.tokenizer.decode([tok_id], skip_special_tokens=False)
                    brace_depth += tok_str.count("{") - tok_str.count("}")
                    if brace_depth <= 0 and "}" in tok_str:
                        break

                    step_out = self.model(
                        input_ids=next_tok.view(1, 1),
                        past_key_values=past_kv,
                        use_cache=True,
                    )
                    past_kv = step_out.past_key_values
                    next_token_logits = step_out.logits[:, -1, :]

        finally:
            self.steering_lab.remove_hooks()

        raw_completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        turn_output = parse_or_repair_agent_json(
            raw_completion_after_prefix=raw_completion,
            counterparty_id=prompt_packet.endogenous_state.counterparty_id,
            available_cash=external_state.liquid_wealth_cash,
            available_food=external_state.inventory.get("FOOD", 0),
            is_responder_wave=is_responder_wave,
        )
        return turn_output, telemetry, speaker_latent_vec

    @staticmethod
    def _sample_next_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature <= 1e-4:
            return torch.argmax(logits, dim=-1)
        scaled = logits / max(0.05, float(temperature))
        probs = torch.softmax(scaled, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=min(32, probs.shape[-1]), dim=-1)
        top_probs = top_probs / torch.clamp(top_probs.sum(dim=-1, keepdim=True), min=1e-9)
        sampled_rel = torch.multinomial(top_probs, num_samples=1).squeeze(-1)
        return top_indices.gather(-1, sampled_rel.unsqueeze(-1)).squeeze(-1)


class BlackBoxAPIBrain:
    def __init__(self, hw_spec: HardwareBackendSpec, tms_router: EntropyTMSRouter) -> None:
        self.hw_spec = hw_spec
        self.tms_router = tms_router

    def generate_turn(
        self,
        prompt_packet: CompiledPromptPacket,
        scoped_memory: ScopedAgentMemoryView,
        external_state: ExternalAgentState,
        domain_hint: ExpertiseDomain,
        is_responder_wave: bool = False,
    ) -> Tuple[AgentTurnOutput, SteeringTelemetryPacket, Optional[List[float]]]:
        telemetry = SteeringTelemetryPacket(
            agent_id=prompt_packet.agent_id,
            tick=prompt_packet.tick,
            steering_mode=SteeringMode.NONE,
        )
        raw_completion = ""
        try:
            headers = {"Authorization": f"Bearer {self.hw_spec.black_box_api_key}"}
            payload = {
                "model": self.hw_spec.model_id,
                "messages": [{"role": "user", "content": prompt_packet.raw_prompt_text}],
                "temperature": self.hw_spec.temperature,
                "max_tokens": self.hw_spec.max_new_tokens,
            }
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(self.hw_spec.black_box_api_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                raw_completion = data["choices"][0]["message"]["content"]
        except Exception:
            raw_completion = (
                'Evaluating offer carefully.", "epistemic_confidence": 0.82, '
                f'"consent_decision": "{"ACCEPT" if is_responder_wave else "INITIATE"}", '
                '"verbal_message": "Agreed on standard market terms.", '
                '"perceived_honesty_delta": 0.05, '
                '"economic_intent": {"action_category": "BILATERAL_TRADE", '
                f'"target_peer_id": "{prompt_packet.endogenous_state.counterparty_id or "Agent_1"}", '
                '"item_id": "FOOD", "quantity": 1, "total_price_cash": 15.0, "is_buying": true}}'
            )

        turn_output = parse_or_repair_agent_json(
            raw_completion_after_prefix=raw_completion,
            counterparty_id=prompt_packet.endogenous_state.counterparty_id,
            available_cash=external_state.liquid_wealth_cash,
            available_food=external_state.inventory.get("FOOD", 0),
            is_responder_wave=is_responder_wave,
        )

        implied_uncertainty = 1.0 - turn_output.epistemic_confidence
        telemetry.probed_shannon_entropy = round(implied_uncertainty, 4)
        if (
            self.tms_router.spec.enabled
            and implied_uncertainty > self.tms_router.spec.entropy_threshold_tau
            and not is_responder_wave
        ):
            expert_id, _ = self.tms_router.select_optimal_expert(scoped_memory, domain_hint)
            if expert_id:
                telemetry.tms_triggered = True
                decision = self.tms_router.evaluate_and_route(
                    vocab_logits_list=[],
                    scoped_memory=scoped_memory,
                    domain=domain_hint,
                )
                decision.halted_generation = True
                decision.normalized_entropy = implied_uncertainty
                decision.selected_expert_id = expert_id
                turn_output = self.tms_router.build_tms_query_turn_output(
                    decision=decision,
                    counterparty_id=prompt_packet.endogenous_state.counterparty_id,
                )

        return turn_output, telemetry, None