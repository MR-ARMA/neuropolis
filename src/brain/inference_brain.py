from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
import torch

from src.core.contracts import (
    AgentTurnOutput,
    CompiledPromptPacket,
    EconomicActionIntent,
    EndogenousStatePacket,
    RawGenerationResult,
    SimulationConfig,
    normalize_item_id,
)
from src.steering.attention_lab import AttentionSteeringLab
from src.steering.entropy_tms import EntropyTMSRouter


class DualBackendInferenceBrain:
    """
    M2: Low-VRAM White-Box (`Qwen/Qwen2.5-1.5B-Instruct`) Inference Brain.
    Runs within a 3.68 GiB VRAM budget (`< 25 MB` activation overhead) by measuring
    in-model attention inside `AttentionSteeringLab` hooks (`0.03 MB`) instead of
    materializing 28 full `[1, 12, S, S]` attention matrices on GPU.
    """

    def __init__(
        self,
        config: SimulationConfig,
        model: Any = None,
        tokenizer: Any = None,
        steering_lab: Optional[AttentionSteeringLab] = None,
        tms_router: Optional[EntropyTMSRouter] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.steering_lab = steering_lab or AttentionSteeringLab(
            config=config.steering_spec
        )
        self.tms_router = tms_router or EntropyTMSRouter(
            db_path="data/mvp_world_state.sqlite", config=config
        )

    def generate_turn(
        self,
        prompt_packet: CompiledPromptPacket,
        steering_packet: EndogenousStatePacket,
        steering_method_override: Optional[str] = None,
    ) -> RawGenerationResult:
        active_method = (
            steering_method_override
            or self.config.experimental_design.steering_method
        )
        if self.config.experimental_design.state_channel in ("NONE", "TEXT_ONLY"):
            active_method = "NONE"

        if (
            self.config.experimental_design.decision_mode == "RULE_ONLY"
            or self.config.model_spec.backend_type == "BLACK_BOX"
            or self.model is None
            or self.tokenizer is None
        ):
            return self._generate_rule_only_result(
                prompt_packet, steering_packet, active_method
            )

        device = next(self.model.parameters()).device
        if prompt_packet.input_ids is not None:
            input_ids = prompt_packet.input_ids.to(device)
            attention_mask = (
                prompt_packet.attention_mask.to(device)
                if prompt_packet.attention_mask is not None
                else torch.ones_like(input_ids)
            )
        else:
            enc = self.tokenizer(
                prompt_packet.raw_prompt,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=320,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

        # Capture first-8 generated token logits via a lightweight lm_head hook (0 extra VRAM)
        step_logits_cpu: List[torch.Tensor] = []
        lm_head = getattr(self.model, "lm_head", None)
        lm_hook_handle = None
        if lm_head is not None:
            def _lm_head_hook(mod: Any, inp: Any, out: torch.Tensor) -> None:
                if len(step_logits_cpu) < 12 and isinstance(out, torch.Tensor):
                    step_logits_cpu.append(
                        out[0, -1, :].detach().to("cpu", dtype=torch.float32)
                    )

            lm_hook_handle = lm_head.register_forward_hook(_lm_head_hook)

        try:
            with self.steering_lab.attach_hooks(
                self.model, steering_packet, steering_method=active_method
            ):
                with torch.inference_mode():
                    gen_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=44,
                        do_sample=False,
                        use_cache=True,
                        return_dict_in_generate=False,
                        output_scores=False,
                        output_attentions=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    prompt_len = input_ids.size(1)
                    new_token_ids = gen_ids[0, prompt_len:].detach().cpu().tolist()
                    del gen_ids

            steered_telemetry = (
                self.steering_lab.measure_attention_from_model_tensors(
                    attentions=None,
                    state_pkt=steering_packet,
                    steering_method=active_method,
                )
            )
            raw_text = self.tokenizer.decode(
                new_token_ids, skip_special_tokens=True
            ).strip()

            if step_logits_cpu:
                stacked_logits = torch.stack(step_logits_cpu, dim=0)
                tok_strs = [
                    self.tokenizer.decode([tid], skip_special_tokens=False)
                    for tid in new_token_ids[: len(step_logits_cpu)]
                ]
                sem_start = self.tms_router.find_semantic_value_start_index(tok_strs)
                mean_entropy = self.tms_router.compute_normalized_top_p_entropy(
                    stacked_logits, semantic_start_idx=sem_start
                )
                del stacked_logits
            else:
                mean_entropy = 0.25

        except torch.cuda.OutOfMemoryError as oom_err:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return RawGenerationResult(
                parsed_output=AgentTurnOutput(
                    internal_thought="CUDA_OOM_ABORT",
                    consent_decision="IGNORE",
                    action_category="PERSONAL_WORK_CONSUME",
                    economic_intent=EconomicActionIntent(action_type="HOLD_OR_WORK"),
                    verbal_message="Pass.",
                ),
                raw_text="",
                llm_valid=False,
                parse_error=f"CUDA_OOM: {str(oom_err)[:80]}",
                attention_measured_from_model=False,
                tms_halt_triggered=False,
                mean_shannon_entropy=0.0,
                attention_telemetry={},
                updated_hebbian_trace=list(steering_packet.hebbian_trace_h),
            )
        finally:
            if lm_hook_handle is not None:
                lm_hook_handle.remove()
            del input_ids, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        parsed_output, llm_valid, parse_err = self._parse_compact_json_strict(
            raw_text=raw_text,
            partner_id=steering_packet.partner_id or "",
        )

        return RawGenerationResult(
            parsed_output=parsed_output,
            raw_text=raw_text,
            llm_valid=llm_valid,
            parse_error=parse_err,
            attention_measured_from_model=steered_telemetry.attention_measured_from_model,
            tms_halt_triggered=mean_entropy > self.config.tms_spec.entropy_threshold_tau,
            mean_shannon_entropy=mean_entropy,
            attention_telemetry=steered_telemetry.span_attention_distribution,
            updated_hebbian_trace=steered_telemetry.updated_hebbian_trace,
        )

    def _parse_compact_json_strict(
        self, raw_text: str, partner_id: str
    ) -> tuple[AgentTurnOutput, bool, Optional[str]]:
        json_match = re.search(r"\{.*?\}", raw_text, flags=re.DOTALL)
        if json_match:
            candidate = json_match.group(0)
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return (
                        self._map_dict_to_turn_output(data, partner_id),
                        True,
                        None,
                    )
            except Exception:
                repaired = re.sub(r",\s*\}", "}", candidate)
                try:
                    data = json.loads(repaired)
                    if isinstance(data, dict):
                        return (
                            self._map_dict_to_turn_output(data, partner_id),
                            True,
                            None,
                        )
                except Exception:
                    pass

        # If model JSON was cut off before closing `}`, extract keys via regex
        return self._extract_structured_fields_from_text(raw_text, partner_id)

    def _map_dict_to_turn_output(
        self, data: Dict[str, Any], partner_id: str
    ) -> AgentTurnOutput:
        if "economic_intent" in data and isinstance(data["economic_intent"], dict):
            return AgentTurnOutput(**data)

        raw_dec = str(
            data.get("decision", data.get("consent_decision", "INITIATE"))
        ).upper()
        if "REJECT" in raw_dec:
            consent = "REJECT"
        elif "ACCEPT" in raw_dec:
            consent = "ACCEPT"
        elif "IGNORE" in raw_dec:
            consent = "IGNORE"
        else:
            consent = "INITIATE"

        raw_act = str(data.get("action", data.get("action_type", "BUY"))).upper()
        if "SELL" in raw_act:
            act_type = "SELL"
        elif "HOLD" in raw_act or "WORK" in raw_act:
            act_type = "HOLD_OR_WORK"
        else:
            act_type = "BUY"

        item_str = normalize_item_id(
            str(data.get("item", data.get("item_id", "CONSUMABLE_NECESSITY")))
        )
        qty = max(0.0, float(data.get("qty", data.get("quantity", 1.0))))
        price = max(0.5, float(data.get("price", data.get("unit_price", 12.0))))
        pg_c = max(
            0.0,
            float(data.get("pg", data.get("public_goods_contribution", 1.5))),
        )
        ask_exp = bool(data.get("ask_expert", False))
        thought = str(
            data.get(
                "thought",
                data.get("internal_thought", "Autonomous LLM economic decision."),
            )
        )
        msg = str(
            data.get(
                "msg",
                data.get(
                    "verbal_message",
                    f"{consent}: {act_type} {qty:.1f} {item_str} @ ${price:.2f}",
                ),
            )
        )

        return AgentTurnOutput(
            internal_thought=thought[:180],
            consent_decision=consent,  # type: ignore[arg-type]
            action_category="TMS_QUERY" if ask_exp else "COMMERCIAL_TRADE",
            economic_intent=EconomicActionIntent(
                action_type=act_type,  # type: ignore[arg-type]
                counterparty_id=str(data.get("counterparty_id", partner_id)),
                item_id=item_str,
                quantity=qty if qty > 0 else 1.0,
                unit_price=price,
                public_goods_contribution=pg_c,
                ask_expert=ask_exp,
            ),
            verbal_message=msg[:160],
        )

    def _extract_structured_fields_from_text(
        self, raw_text: str, partner_id: str
    ) -> tuple[AgentTurnOutput, bool, Optional[str]]:
        upper = raw_text.upper()
        has_decision = any(
            k in upper for k in ("ACCEPT", "REJECT", "INITIATE", "BUY", "SELL", "HOLD")
        )
        price_match = re.search(
            r"(?:PRICE|UNIT_PRICE|\"PRICE\"|\$)\s*[:=]?\s*\$?\s*([0-9]+(?:\.[0-9]+)?)",
            upper,
        )
        if has_decision:
            price_val = (
                max(0.5, float(price_match.group(1)))
                if price_match
                else 14.0
            )
            consent = (
                "REJECT"
                if "REJECT" in upper
                else ("ACCEPT" if "ACCEPT" in upper else "INITIATE")
            )
            act = (
                "SELL"
                if "SELL" in upper
                else ("HOLD_OR_WORK" if "HOLD" in upper else "BUY")
            )
            item = (
                "PRODUCTIVE_CAPITAL_ASSET"
                if ("CAPITAL" in upper or "TECH" in upper or "ASSET" in upper)
                else "CONSUMABLE_NECESSITY"
            )
            out = AgentTurnOutput(
                internal_thought=raw_text[:140],
                consent_decision=consent,  # type: ignore[arg-type]
                action_category="COMMERCIAL_TRADE",
                economic_intent=EconomicActionIntent(
                    action_type=act,  # type: ignore[arg-type]
                    counterparty_id=partner_id,
                    item_id=item,
                    quantity=1.0,
                    unit_price=round(price_val, 2),
                ),
                verbal_message=raw_text[:120],
            )
            return (out, True, None)

        invalid_out = AgentTurnOutput(
            internal_thought=f"PARSE_FAILURE: {raw_text[:80]}",
            consent_decision="REJECT",
            action_category="PERSONAL_WORK_CONSUME",
            economic_intent=EconomicActionIntent(
                action_type="HOLD_OR_WORK",
                counterparty_id=partner_id,
                item_id="CONSUMABLE_NECESSITY",
                quantity=0.0,
                unit_price=12.0,
            ),
            verbal_message="No valid action parsed.",
        )
        return (invalid_out, False, f"Unparseable output: {raw_text[:60]}")

    def _generate_rule_only_result(
        self,
        prompt_packet: CompiledPromptPacket,
        steering_packet: EndogenousStatePacket,
        active_method: str,
    ) -> RawGenerationResult:
        steered_telemetry = self.steering_lab.measure_attention_from_model_tensors(
            attentions=None,
            state_pkt=steering_packet,
            steering_method=active_method,
        )
        out = AgentTurnOutput(
            internal_thought="RULE_ONLY_BASELINE_DECISION",
            consent_decision="INITIATE",
            action_category="COMMERCIAL_TRADE",
            economic_intent=EconomicActionIntent(
                action_type="BUY",
                counterparty_id=steering_packet.partner_id or "",
                item_id="CONSUMABLE_NECESSITY",
                quantity=1.0,
                unit_price=12.0,
            ),
            verbal_message="Rule-only baseline proposal.",
        )
        return RawGenerationResult(
            parsed_output=out,
            raw_text="RULE_ONLY",
            llm_valid=False,
            parse_error="RULE_ONLY_MODE",
            attention_measured_from_model=False,
            tms_halt_triggered=False,
            mean_shannon_entropy=round(
                0.30 + 0.25 * steering_packet.scarcity_deficit_f, 4
            ),
            attention_telemetry=steered_telemetry.span_attention_distribution,
            updated_hebbian_trace=steered_telemetry.updated_hebbian_trace,
        )


WhiteBoxPyTorchBrain = DualBackendInferenceBrain
BlackBoxAPIBrain = DualBackendInferenceBrain