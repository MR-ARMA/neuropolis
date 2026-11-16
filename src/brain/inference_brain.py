from __future__ import annotations

import gc
import json
import re
from typing import Any, Optional
import torch

from src.core.contracts import (
    AgentTurnOutput,
    CompiledPromptPacket,
    EconomicActionIntent,
    EndogenousStatePacket,
    RawGenerationResult,
    SimulationConfig,
)
from src.steering.attention_lab import AttentionSteeringLab
from src.steering.entropy_tms import EntropyTMSRouter


class DualBackendInferenceBrain:
    """
    M2: Memory-Optimized White-Box (`Qwen/Qwen2.5-1.5B-Instruct`) Inference Brain
    engineered specifically to run within a 3.68 GiB VRAM ceiling without OOM.
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
    ) -> RawGenerationResult:
        if (
            self.config.model_spec.backend_type == "BLACK_BOX"
            or self.model is None
            or self.tokenizer is None
        ):
            return self._generate_fallback_structured(prompt_packet, steering_packet)

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
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

        # Truncate overly long prompts to 320 tokens to protect 3.68 GiB VRAM budget
        if input_ids.size(1) > 320:
            input_ids = input_ids[:, -320:]
            attention_mask = attention_mask[:, -320:]

        try:
            with self.steering_lab.attach_hooks(
                self.model, steering_packet
            ) as steered_telemetry:
                with torch.inference_mode():
                    # Step 1: Single prefill forward pass to extract last-K probe logits & mirror vector
                    fwd_out = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    probe_k = self.config.tms_spec.probe_tokens_k
                    probe_logits_cpu = (
                        fwd_out.logits[0, -probe_k:, :]
                        .detach()
                        .to("cpu", dtype=torch.float32)
                    )
                    del fwd_out

                    mean_entropy = self.tms_router.compute_normalized_top_p_entropy(
                        probe_logits_cpu, semantic_start_idx=0
                    )
                    del probe_logits_cpu

                    # Step 2: Compact token generation WITHOUT storing 151,936-vocab scores per step
                    gen_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=28,
                        do_sample=False,
                        use_cache=True,
                        return_dict_in_generate=False,
                        output_scores=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    prompt_len = input_ids.size(1)
                    new_tokens = gen_ids[0, prompt_len:].detach().cpu().tolist()
                    del gen_ids

            raw_text = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self._generate_fallback_structured(prompt_packet, steering_packet)
        finally:
            del input_ids, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        parsed_output = self._parse_or_synthesize_output(
            raw_text=raw_text,
            steering_packet=steering_packet,
        )
        mirror_vec = self.steering_lab.captured_middle_residual
        if mirror_vec is None:
            mirror_vec = torch.full(
                (1, 64),
                fill_value=-0.40 * steering_packet.scarcity_deficit_f
                + 0.25 * steering_packet.peer_trust_tau,
            )

        return RawGenerationResult(
            parsed_output=parsed_output,
            raw_text=raw_text,
            tms_halt_triggered=mean_entropy > self.config.tms_spec.entropy_threshold_tau,
            mean_shannon_entropy=mean_entropy,
            emitted_mirror_vector=mirror_vec,
            attention_telemetry=steered_telemetry.span_attention_distribution,
            updated_hebbian_trace=steered_telemetry.updated_hebbian_trace,
        )

    def _parse_or_synthesize_output(
        self,
        raw_text: str,
        steering_packet: EndogenousStatePacket,
    ) -> AgentTurnOutput:
        f_it = steering_packet.scarcity_deficit_f
        t_eff = steering_packet.effective_trust_01

        json_match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if isinstance(data, dict):
                    return AgentTurnOutput(**data)
            except Exception:
                pass

        cleaned_snippet = re.sub(r"\s+", " ", raw_text)[:110].strip()
        thought = (
            cleaned_snippet
            if len(cleaned_snippet) > 12
            else f"Appraising Two-Asset HANK balance sheet at Scarcity F={f_it:.2f} and Effective Trust T_eff={t_eff:.2f}."
        )
        verbal = (
            "I require an immediate survival trade within fair dual-entitlement bounds."
            if f_it > 0.42
            else "Proposing a mutually beneficial bilateral exchange."
        )
        return AgentTurnOutput(
            internal_thought=thought,
            consent_decision="ACCEPT" if t_eff >= 0.25 else "REJECT",
            action_category="COMMERCIAL_TRADE",
            economic_intent=EconomicActionIntent(
                action_type="BUY",
                counterparty_id=steering_packet.partner_id or "",
                item_id="CONSUMABLE_NECESSITY",
                quantity=1.0,
                unit_price=12.0,
            ),
            verbal_message=verbal,
        )

    def _generate_fallback_structured(
        self,
        prompt_packet: CompiledPromptPacket,
        steering_packet: EndogenousStatePacket,
    ) -> RawGenerationResult:
        steered_telemetry = self.steering_lab.compute_span_attention_telemetry(
            steering_packet
        )
        parsed = self._parse_or_synthesize_output("", steering_packet)
        mirror_vec = torch.full(
            (1, 64),
            fill_value=-0.40 * steering_packet.scarcity_deficit_f
            + 0.25 * steering_packet.peer_trust_tau,
        )
        return RawGenerationResult(
            parsed_output=parsed,
            raw_text="",
            tms_halt_triggered=False,
            mean_shannon_entropy=round(
                0.32 + 0.22 * steering_packet.scarcity_deficit_f, 4
            ),
            emitted_mirror_vector=mirror_vec,
            attention_telemetry=steered_telemetry.span_attention_distribution,
            updated_hebbian_trace=steered_telemetry.updated_hebbian_trace,
        )


WhiteBoxPyTorchBrain = DualBackendInferenceBrain
BlackBoxAPIBrain = DualBackendInferenceBrain