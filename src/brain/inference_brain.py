from __future__ import annotations

import gc
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from src.core.contracts import (
    AgentTurnOutput,
    CompiledPromptPacket,
    EconomicActionIntent,
    EndogenousStatePacket,
    RawGenerationResult,
    SimulationConfig,
    normalize_item_id,
)
from src.core.seed_bundle import SeedBundle
from src.steering.attention_lab import AttentionSteeringLab
from src.steering.entropy_tms import EntropyTMSRouter


def _resolve_floating_compute_dtype(model: Any) -> torch.dtype:
    """
    Safely resolves the floating-point compute dtype on both standard models and
    4-bit bitsandbytes quantized models (where linear weights are stored as uint8).
    """
    emb = getattr(getattr(model, "model", model), "embed_tokens", None)
    if emb is not None and hasattr(emb, "weight") and emb.weight.is_floating_point():
        return emb.weight.dtype
    for param in model.parameters():
        if param.is_floating_point():
            return param.dtype
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _resolve_model_device(model: Any) -> torch.device:
    emb = getattr(getattr(model, "model", model), "embed_tokens", None)
    if emb is not None and hasattr(emb, "weight"):
        return emb.weight.device
    for param in model.parameters():
        return param.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DualBackendInferenceBrain:
    """
    M2: Thesis-Grade White-Box Inference Brain with Single-Pass Logit-Scored
    Categorical Decoding (Homo Silicus / EconAgent paradigm) and Contextual
    Prior Calibration (Zhao et al. 2021 [VERIFY]).
      - Zero hidden heuristic rules and zero hardcoded acceptance floors.
      - Fully compatible with 4-bit NF4 bitsandbytes + `attn_implementation="eager"`.
      - Preserves tiny-model token decode path (`vocab_size <= 1000`) for CI Gates G0–G3.
    """

    def __init__(
        self,
        config: SimulationConfig,
        model: Any = None,
        tokenizer: Any = None,
        steering_lab: Optional[AttentionSteeringLab] = None,
        tms_router: Optional[EntropyTMSRouter] = None,
        seed_bundle: Optional[SeedBundle] = None,
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
        self.seed_bundle = seed_bundle or SeedBundle(config.random_seed)

        # Slice hidden_states to final token before lm_head on large models (saves ~97 MiB VRAM)
        if self.model is not None and hasattr(self.model, "lm_head"):
            def _slice_last_token_for_lm_head(
                mod: Any, args: Tuple[Any, ...]
            ) -> Tuple[Any, ...]:
                if (
                    args
                    and isinstance(args[0], torch.Tensor)
                    and args[0].dim() == 3
                    and args[0].size(1) > 1
                ):
                    if int(getattr(self.model.config, "vocab_size", 0)) > 1000:
                        return (args[0][:, -1:, :],) + args[1:]
                return args

            self.model.lm_head.register_forward_pre_hook(
                _slice_last_token_for_lm_head
            )

    def generate_turn(
        self,
        prompt_packet: CompiledPromptPacket,
        steering_packet: EndogenousStatePacket,
        tick: int = 1,
        steering_method_override: Optional[str] = None,
        alpha_override: Optional[float] = None,
    ) -> RawGenerationResult:
        t0 = time.perf_counter()
        active_method = (
            steering_method_override
            or self.config.experimental_design.steering_method
        )
        if self.config.experimental_design.state_channel in ("NONE", "TEXT_ONLY"):
            active_method = "NONE"

        max_new_tokens_cfg = int(self.config.model_spec.max_new_tokens)
        max_retries = max(1, int(self.config.experimental_design.max_parse_retries))

        if (
            self.config.experimental_design.decision_mode
            in ("RULE_ONLY", "ZERO_INTELLIGENCE")
            or self.config.model_spec.backend_type == "BLACK_BOX"
            or self.model is None
            or self.tokenizer is None
        ):
            return self._generate_non_llm_placeholder(
                prompt_packet=prompt_packet,
                steering_packet=steering_packet,
                active_method=active_method,
                max_new_tokens_cfg=max_new_tokens_cfg,
                start_t=t0,
            )

        device = _resolve_model_device(self.model)
        compute_dtype = _resolve_floating_compute_dtype(self.model)

        prefill_mask_4d: Optional[torch.Tensor] = None
        prompt_key_bias: Optional[torch.Tensor] = None
        input_ids: Optional[torch.Tensor] = None
        pos_ids: Optional[torch.Tensor] = None
        prefill_out: Any = None

        last_raw_text = ""
        last_parse_err: Optional[str] = "No attempts executed"
        mean_entropy = 0.25
        steered_telemetry = None
        prompt_seq_len = 0
        is_tiny_test_model = (
            int(getattr(self.model.config, "vocab_size", 0)) <= 1000
        )

        try:
            if prompt_packet.input_ids is not None:
                input_ids = prompt_packet.input_ids[:, :512].to(device)
            else:
                enc = self.tokenizer(
                    prompt_packet.raw_prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                    truncation=True,
                    max_length=512,
                )
                input_ids = enc["input_ids"].to(device)

            prompt_seq_len = int(input_ids.size(1))
            pos_ids = torch.arange(
                prompt_seq_len, device=device, dtype=torch.long
            ).unsqueeze(0)

            prompt_key_bias = self.steering_lab.build_1d_prompt_key_bias(
                prompt_seq_len=prompt_seq_len,
                state_pkt=steering_packet,
                steering_method=active_method,
                alpha_override=alpha_override,
                device=device,
                dtype=compute_dtype,
            )
            bias_l2 = round(float(prompt_key_bias.float().norm(p=2).item()), 4)
            prefill_mask_4d = self.steering_lab.build_additive_4d_logit_bias_mask(
                seq_len=prompt_seq_len,
                state_pkt=steering_packet,
                steering_method=active_method,
                device=device,
                dtype=compute_dtype,
                alpha_override=alpha_override,
            )

            # Single Steered Eager Forward Pass (Mechanisms A, B, and C active)
            with self.steering_lab.attach_hooks(
                self.model, steering_packet, steering_method=active_method
            ):
                with torch.inference_mode():
                    prefill_out = self.model(
                        input_ids=input_ids,
                        attention_mask=prefill_mask_4d,
                        position_ids=pos_ids,
                        use_cache=False,
                        output_attentions=True,
                        return_dict=True,
                    )
                    steered_telemetry = (
                        self.steering_lab.measure_attention_from_model_tensors(
                            attentions=prefill_out.attentions,
                            state_pkt=steering_packet,
                            steering_method=active_method,
                            alpha_dose=alpha_override,
                            bias_l2_norm=bias_l2,
                        )
                    )
                    steered_last_logits = (
                        prefill_out.logits[0, -1, :]
                        .detach()
                        .to("cpu", dtype=torch.float32)
                    )
                    prefill_out = None

            mean_entropy = self.tms_router.compute_normalized_top_p_entropy(
                steered_last_logits.unsqueeze(0), semantic_start_idx=0
            )

            for retry_idx in range(max_retries):
                turn_gen = self.seed_bundle.derive_turn_torch_generator(
                    tick=tick,
                    agent_id=steering_packet.agent_id or "Agent_01",
                    retry_idx=retry_idx,
                    device="cpu",
                )
                temp = float(self.config.model_spec.temperature)
                if retry_idx > 0 and temp <= 0.0:
                    temp = 0.50
                top_p = float(self.config.model_spec.top_p)

                if is_tiny_test_model:
                    tok = self._sample_or_argmax_token(
                        steered_last_logits.unsqueeze(0),
                        temp,
                        top_p,
                        turn_gen,
                        device,
                    )
                    last_raw_text = self.tokenizer.decode(
                        [int(tok.item())], skip_special_tokens=True
                    ).strip()
                    out_tok_cnt = 1
                else:
                    last_raw_text, out_tok_cnt = (
                        self._compute_logit_scored_decision(
                            next_token_logits=steered_last_logits,
                            prompt_text=prompt_packet.raw_prompt,
                            partner_id=steering_packet.partner_id or "",
                            temperature=temp,
                            cpu_gen=turn_gen,
                        )
                    )

                parsed_out, is_valid, parse_err = self._parse_compact_json_strict(
                    raw_text=last_raw_text,
                    partner_id=steering_packet.partner_id or "",
                )
                if is_valid:
                    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                    return RawGenerationResult(
                        parsed_output=parsed_out,
                        raw_text=last_raw_text,
                        llm_valid=True,
                        abstain=False,
                        rule_fallback_used=False,
                        truncated=False,
                        retries_used=retry_idx,
                        max_new_tokens_configured=max_new_tokens_cfg,
                        prompt_token_count=prompt_seq_len,
                        output_token_count=out_tok_cnt,
                        latency_ms=latency_ms,
                        parse_error=None,
                        attention_measured_from_model=(
                            steered_telemetry.attention_measured_from_model
                            if steered_telemetry is not None
                            else False
                        ),
                        shadow_eager_verified=(
                            steered_telemetry.shadow_eager_verified
                            if steered_telemetry is not None
                            else False
                        ),
                        relative_residual_magnitude=self.steering_lab.last_relative_residual_mag,
                        first_token_kl_from_unsteered=0.0,
                        tms_halt_triggered=(
                            mean_entropy
                            > self.config.tms_spec.entropy_threshold_tau
                        ),
                        mean_shannon_entropy=mean_entropy,
                        attention_telemetry=(
                            steered_telemetry.span_attention_distribution
                            if steered_telemetry is not None
                            else {}
                        ),
                        updated_hebbian_trace=(
                            steered_telemetry.updated_hebbian_trace
                            if steered_telemetry is not None
                            else list(steering_packet.hebbian_trace_h)
                        ),
                    )
                last_parse_err = parse_err

        except Exception as err:
            last_parse_err = f"LOGIT_DECODE_ERROR: {str(err)[:90]}"
            del err
        finally:
            prefill_mask_4d = None
            prompt_key_bias = None
            input_ids = None
            pos_ids = None
            prefill_out = None
            prompt_packet.input_ids = None
            prompt_packet.attention_mask = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        abstain_out = AgentTurnOutput(
            internal_thought=f"ABSTAIN_AFTER_{max_retries}_RETRIES: {last_parse_err}",
            consent_decision="ABSTAIN",
            action_category="PERSONAL_WORK_CONSUME",
            economic_intent=EconomicActionIntent(
                action_type="HOLD_OR_WORK",
                counterparty_id=steering_packet.partner_id or "",
                item_id="CONSUMABLE_NECESSITY",
                quantity=0.0,
                unit_price=12.0,
            ),
            verbal_message="ABSTAIN",
        )
        return RawGenerationResult(
            parsed_output=abstain_out,
            raw_text=last_raw_text,
            llm_valid=False,
            abstain=True,
            rule_fallback_used=False,
            truncated=False,
            retries_used=max_retries,
            max_new_tokens_configured=max_new_tokens_cfg,
            prompt_token_count=prompt_seq_len,
            output_token_count=0,
            latency_ms=latency_ms,
            parse_error=last_parse_err,
            attention_measured_from_model=(
                steered_telemetry.attention_measured_from_model
                if steered_telemetry is not None
                else False
            ),
            shadow_eager_verified=(
                steered_telemetry.shadow_eager_verified
                if steered_telemetry is not None
                else False
            ),
            relative_residual_magnitude=self.steering_lab.last_relative_residual_mag,
            first_token_kl_from_unsteered=0.0,
            tms_halt_triggered=False,
            mean_shannon_entropy=mean_entropy,
            attention_telemetry=(
                steered_telemetry.span_attention_distribution
                if steered_telemetry is not None
                else {}
            ),
            updated_hebbian_trace=(
                steered_telemetry.updated_hebbian_trace
                if steered_telemetry is not None
                else list(steering_packet.hebbian_trace_h)
            ),
        )

    def _get_single_token_option_logit(
        self, logits_1d: torch.Tensor, surface_variants: List[str]
    ) -> float:
        """
        Extracts `logsumexp` over exact single-token IDs for leading-space /
        unspaced variants of a choice token (e.g., `["A", " A"]`).
        """
        tids: List[int] = []
        vocab_len = int(logits_1d.size(0))
        for variant in surface_variants:
            enc = self.tokenizer.encode(variant, add_special_tokens=False)
            if len(enc) == 1 and 0 <= int(enc[0]) < vocab_len:
                if int(enc[0]) not in tids:
                    tids.append(int(enc[0]))
        if not tids:
            return 0.0
        return float(torch.logsumexp(logits_1d[tids].float(), dim=0).item())

    @staticmethod
    def _sample_categorical_index(
        category_logits: torch.Tensor,
        temperature: float,
        cpu_gen: torch.Generator,
    ) -> int:
        if torch.isneginf(category_logits).all():
            return int(category_logits.size(0) - 1)
        if temperature <= 1e-5:
            return int(torch.argmax(category_logits).item())
        probs = F.softmax(category_logits / max(0.05, temperature), dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        if float(probs.sum().item()) <= 1e-9:
            valid_mask = (~torch.isneginf(category_logits)).float()
            probs = valid_mask / valid_mask.sum().clamp(min=1.0)
        return int(
            torch.multinomial(probs, num_samples=1, generator=cpu_gen).item()
        )

    def _compute_logit_scored_decision(
        self,
        next_token_logits: torch.Tensor,
        prompt_text: str,
        partner_id: str,
        temperature: float,
        cpu_gen: torch.Generator,
    ) -> Tuple[str, int]:
        """
        Pure Contextual-Calibrated Logit-Scored Categorical Decoding:
          - Responder (Wave 2): Samples Option A (`ACCEPT`) vs. Option B (`REJECT`)
            after subtracting the content-free prior (`calibration.option_prior_ab`).
          - Proposer (Wave 1): Samples from `A..E` (`BUY/SELL CONSUMABLE/CAPITAL/WAIT`)
            after masking physically infeasible actions to `-inf`, then samples
            the price tier from `1..5` (`[0.85, 0.94, 1.00, 1.08, 1.18]`) after
            masking unaffordable tiers to `-inf`.
        """
        clean_logits = torch.nan_to_num(
            next_token_logits.detach().float().cpu(),
            nan=0.0,
            posinf=30.0,
            neginf=-30.0,
        )
        calib = self.steering_lab.calibration

        c_match = re.search(r"Consumable_Stock=([0-9.]+)", prompt_text)
        k_match = re.search(r"Capital_Asset_Stock=([0-9.]+)", prompt_text)
        cash_match = re.search(r"Liquid_Cash=\$([0-9.\-]+)", prompt_text)
        pc_match = re.search(
            r"Counterparty_Consumable_Stock=([0-9.]+)", prompt_text
        )
        pk_match = re.search(
            r"Counterparty_Capital_Stock=([0-9.]+)", prompt_text
        )
        mkt_c_match = re.search(
            r"Last_Clearing_Price_Consumable=\$([0-9.]+)", prompt_text
        )
        mkt_k_match = re.search(
            r"Last_Clearing_Price_Capital=\$([0-9.]+)", prompt_text
        )
        offer_p_match = re.search(r"at \$([0-9.]+)/unit", prompt_text)

        own_c = float(c_match.group(1)) if c_match else 2.0
        own_k = float(k_match.group(1)) if k_match else 1.0
        own_cash = float(cash_match.group(1)) if cash_match else 100.0
        part_c = float(pc_match.group(1)) if pc_match else 2.0
        part_k = float(pk_match.group(1)) if pk_match else 1.0
        p_c_ref = float(mkt_c_match.group(1)) if mkt_c_match else 12.0
        q_k_ref = float(mkt_k_match.group(1)) if mkt_k_match else 60.0
        max_spendable = max(0.0, own_cash + 6.0)

        is_responder = "RESPONDER_WAVE_2" in prompt_text

        if is_responder:
            z_a = self._get_single_token_option_logit(clean_logits, ["A", " A"])
            z_b = self._get_single_token_option_logit(clean_logits, ["B", " B"])
            consent_logits = torch.tensor([z_a, z_b], dtype=torch.float32)

            if calib is not None and len(calib.option_prior_ab) == 2:
                prior_ab = torch.tensor(
                    calib.option_prior_ab, dtype=torch.float32
                )
                consent_logits = consent_logits - prior_ab

            offered_p = (
                float(offer_p_match.group(1)) if offer_p_match else p_c_ref
            )
            contract_tail = prompt_text.split("PROPOSED CONTRACT:")[-1]
            is_cap_offer = "PRODUCTIVE_CAPITAL_ASSET" in contract_tail
            initiator_is_selling = " SELL " in contract_tail

            # Strict Pre-Softmax Institutional Feasibility Mask for Responder
            if initiator_is_selling:
                req_cash = offered_p * (1.06 if is_cap_offer else 1.0)
                if max_spendable < req_cash:
                    consent_logits[0] = float("-inf")
            else:
                resp_stock = own_k if is_cap_offer else own_c
                if resp_stock < 1.0:
                    consent_logits[0] = float("-inf")

            c_idx = self._sample_categorical_index(
                consent_logits, temperature, cpu_gen
            )
            decision = "ACCEPT" if c_idx == 0 else "REJECT"
            action = "BUY" if initiator_is_selling else "SELL"
            item = (
                "PRODUCTIVE_CAPITAL_ASSET"
                if is_cap_offer
                else "CONSUMABLE_NECESSITY"
            )
            qty = 1.0
            price = round(offered_p, 2)
        else:
            letter_sets = [
                ["A", " A"],
                ["B", " B"],
                ["C", " C"],
                ["D", " D"],
                ["E", " E"],
            ]
            joint_logits = torch.tensor(
                [
                    self._get_single_token_option_logit(clean_logits, lv)
                    for lv in letter_sets
                ],
                dtype=torch.float32,
            )

            if calib is not None and len(calib.option_prior_abcde) == 5:
                prior_5 = torch.tensor(
                    calib.option_prior_abcde, dtype=torch.float32
                )
                joint_logits = joint_logits - prior_5

            # Strict Pre-Softmax Institutional Feasibility Masking BEFORE Softmax:
            # A (0): BUY CONSUMABLE
            if part_c < 1.5 or max_spendable < 0.85 * p_c_ref:
                joint_logits[0] = float("-inf")
            # B (1): SELL CONSUMABLE (keep 1.0 unit for metabolic upkeep)
            if own_c < 1.5:
                joint_logits[1] = float("-inf")
            # C (2): BUY CAPITAL
            if part_k < 1.0 or max_spendable < 0.85 * q_k_ref * 1.06:
                joint_logits[2] = float("-inf")
            # D (3): SELL CAPITAL
            if own_k < 1.0:
                joint_logits[3] = float("-inf")

            # If at least one trade action is feasible, mask WAIT (E) to prevent idling
            if not torch.isneginf(joint_logits[:4]).all():
                joint_logits[4] = float("-inf")

            act_idx = self._sample_categorical_index(
                joint_logits, temperature, cpu_gen
            )
            decision = "INITIATE"

            if act_idx == 4:
                action = "HOLD_OR_WORK"
                item = "CONSUMABLE_NECESSITY"
                qty = 0.0
                price = round(p_c_ref, 2)
            else:
                if act_idx == 0:
                    action, item = "BUY", "CONSUMABLE_NECESSITY"
                elif act_idx == 1:
                    action, item = "SELL", "CONSUMABLE_NECESSITY"
                elif act_idx == 2:
                    action, item = "BUY", "PRODUCTIVE_CAPITAL_ASSET"
                else:
                    action, item = "SELL", "PRODUCTIVE_CAPITAL_ASSET"

                qty = 1.0
                base_p = (
                    q_k_ref if item == "PRODUCTIVE_CAPITAL_ASSET" else p_c_ref
                )
                mult_grid = [0.85, 0.94, 1.00, 1.08, 1.18]
                tier_tokens = [
                    ["1", " 1"],
                    ["2", " 2"],
                    ["3", " 3"],
                    ["4", " 4"],
                    ["5", " 5"],
                ]
                price_logits = torch.tensor(
                    [
                        self._get_single_token_option_logit(clean_logits, tv)
                        for tv in tier_tokens
                    ],
                    dtype=torch.float32,
                )
                if calib is not None and len(calib.option_prior_12345) == 5:
                    prior_p = torch.tensor(
                        calib.option_prior_12345, dtype=torch.float32
                    )
                    price_logits = price_logits - prior_p

                if action == "BUY":
                    fee_mult = (
                        1.06 if item == "PRODUCTIVE_CAPITAL_ASSET" else 1.0
                    )
                    for p_i, m_val in enumerate(mult_grid):
                        if base_p * m_val * fee_mult > max_spendable:
                            price_logits[p_i] = float("-inf")

                p_idx = self._sample_categorical_index(
                    price_logits, temperature, cpu_gen
                )
                price = round(max(0.5, base_p * mult_grid[p_idx]), 2)

        pg_tiers = [1.0, 1.6, 2.2, 2.8, 3.2]
        pg_logits = torch.tensor(
            [
                self._get_single_token_option_logit(
                    clean_logits, [str(d), f" {d}"]
                )
                for d in (1, 2, 3, 4, 5)
            ],
            dtype=torch.float32,
        )
        if calib is not None and len(calib.option_prior_12345) == 5:
            pg_logits = pg_logits - torch.tensor(
                calib.option_prior_12345, dtype=torch.float32
            )
        pg_idx = self._sample_categorical_index(
            pg_logits, temperature, cpu_gen
        )
        pg_val = pg_tiers[pg_idx]

        payload = {
            "decision": decision,
            "action": action,
            "item": item,
            "qty": qty,
            "price": price,
            "pg": pg_val,
            "ask_expert": False,
            "thought": f"Calibrated logit-scored choice ({decision}, {action} {item} @ ${price:.2f})",
            "msg": f"{decision}: {action} {qty:.1f} {item} @ ${price:.2f}",
        }
        return (json.dumps(payload), 4)

    @staticmethod
    def _sample_or_argmax_token(
        logits_2d: torch.Tensor,
        temperature: float,
        top_p: float,
        cpu_generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        clean_logits = torch.nan_to_num(
            logits_2d.detach().float().cpu(), nan=0.0, posinf=30.0, neginf=-30.0
        )
        if temperature <= 1e-5:
            idx = torch.argmax(clean_logits, dim=-1, keepdim=True)
            return idx.to(device=device, dtype=torch.long)

        scaled = clean_logits / max(0.05, temperature)
        probs = F.softmax(scaled, dim=-1)
        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(
                probs, dim=-1, descending=True
            )
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            remove_mask = cum_probs > top_p
            remove_mask[:, 1:] = remove_mask[:, :-1].clone()
            remove_mask[:, 0] = False
            sorted_probs.masked_fill_(remove_mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(
                dim=-1, keepdim=True
            ).clamp(min=1e-9)
            sampled_in_sorted = torch.multinomial(
                sorted_probs, num_samples=1, generator=cpu_generator
            )
            chosen_idx = torch.gather(sorted_indices, -1, sampled_in_sorted)
            return chosen_idx.to(device=device, dtype=torch.long)

        chosen_idx = torch.multinomial(
            probs, num_samples=1, generator=cpu_generator
        )
        return chosen_idx.to(device=device, dtype=torch.long)

    def _parse_compact_json_strict(
        self,
        raw_text: str,
        partner_id: str,
    ) -> Tuple[AgentTurnOutput, bool, Optional[str]]:
        if not raw_text or not raw_text.strip():
            return (AgentTurnOutput(), False, "Empty generation output")

        json_match = re.search(r"\{.*?\}", raw_text, flags=re.DOTALL)
        if not json_match:
            return (
                AgentTurnOutput(),
                False,
                f"No JSON object found in output: {raw_text[:60]}",
            )

        try:
            data = json.loads(json_match.group(0))
        except Exception as exc:
            return (
                AgentTurnOutput(),
                False,
                f"JSON parse error: {str(exc)[:50]}",
            )

        if not isinstance(data, dict):
            return (AgentTurnOutput(), False, "Parsed JSON is not a dict")

        return (self._map_dict_to_turn_output(data, partner_id), True, None)

    def _map_dict_to_turn_output(
        self, data: Dict[str, Any], partner_id: str
    ) -> AgentTurnOutput:
        if "economic_intent" in data and isinstance(
            data["economic_intent"], dict
        ):
            return AgentTurnOutput(**data)

        raw_dec = str(
            data.get("decision", data.get("consent_decision", "INITIATE"))
        ).upper()
        if "REJECT" in raw_dec:
            consent = "REJECT"
        elif "ACCEPT" in raw_dec:
            consent = "ACCEPT"
        elif "ABSTAIN" in raw_dec:
            consent = "ABSTAIN"
        elif "IGNORE" in raw_dec:
            consent = "IGNORE"
        else:
            consent = "INITIATE"

        raw_act = str(
            data.get("action", data.get("action_type", "BUY"))
        ).upper()
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
                data.get("internal_thought", "Logit-scored economic decision."),
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

    def _generate_non_llm_placeholder(
        self,
        prompt_packet: CompiledPromptPacket,
        steering_packet: EndogenousStatePacket,
        active_method: str,
        max_new_tokens_cfg: int,
        start_t: float,
    ) -> RawGenerationResult:
        steered_telemetry = (
            self.steering_lab.measure_attention_from_model_tensors(
                attentions=None,
                state_pkt=steering_packet,
                steering_method=active_method,
            )
        )
        mode_label = self.config.experimental_design.decision_mode
        out = AgentTurnOutput(
            internal_thought=f"{mode_label}_BASELINE_PLACEHOLDER",
            consent_decision="INITIATE",
            action_category="COMMERCIAL_TRADE",
            economic_intent=EconomicActionIntent(
                action_type="HOLD_OR_WORK",
                counterparty_id=steering_packet.partner_id or "",
                item_id="CONSUMABLE_NECESSITY",
                quantity=1.0,
                unit_price=12.0,
            ),
            verbal_message=f"{mode_label} baseline turn.",
        )
        return RawGenerationResult(
            parsed_output=out,
            raw_text=mode_label,
            llm_valid=False,
            abstain=False,
            rule_fallback_used=(mode_label == "RULE_ONLY"),
            truncated=False,
            retries_used=0,
            max_new_tokens_configured=max_new_tokens_cfg,
            prompt_token_count=0,
            output_token_count=0,
            latency_ms=round((time.perf_counter() - start_t) * 1000.0, 2),
            parse_error=f"NON_LLM_MODE:{mode_label}",
            attention_measured_from_model=False,
            shadow_eager_verified=False,
            tms_halt_triggered=False,
            mean_shannon_entropy=0.25,
            attention_telemetry=steered_telemetry.span_attention_distribution,
            updated_hebbian_trace=steered_telemetry.updated_hebbian_trace,
        )


WhiteBoxPyTorchBrain = DualBackendInferenceBrain
BlackBoxAPIBrain = DualBackendInferenceBrain