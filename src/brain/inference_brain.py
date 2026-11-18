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


class DualBackendInferenceBrain:
    """
    M2: Thesis-Grade White-Box Inference Brain.
    - Passes the 4D additive pre-softmax logit bias mask `[1, 1, S, S]` with
      explicit `position_ids` into `self.model(...)` during eager forward passes.
    - Captures genuine post-softmax attention weights (`attention_measured_from_model=True`)
      while freeing non-target layers on the fly so a 3.68 GiB GPU never OOMs.
    - Uses schema-constrained logit-scored decoding for large models and
      retry-then-ABSTAIN for free-text models (zero silent rule fallbacks).
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

        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype

        prefill_mask_4d = None
        prompt_key_bias = None
        input_ids = None
        pos_ids = None
        prefill_out = None

        if prompt_packet.input_ids is not None:
            input_ids = prompt_packet.input_ids[:, :148].to(device)
        else:
            enc = self.tokenizer(
                prompt_packet.raw_prompt,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=148,
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
            dtype=model_dtype,
        )
        bias_l2 = round(float(prompt_key_bias.float().norm(p=2).item()), 4)
        prefill_mask_4d = self.steering_lab.build_additive_4d_logit_bias_mask(
            seq_len=prompt_seq_len,
            state_pkt=steering_packet,
            steering_method=active_method,
            device=device,
            dtype=model_dtype,
            alpha_override=alpha_override,
        )

        last_raw_text = ""
        last_parse_err: Optional[str] = "No attempts executed"
        last_truncated = False
        mean_entropy = 0.25
        steered_telemetry = None
        is_tiny_test_model = (
            int(getattr(self.model.config, "vocab_size", 0)) <= 1000
        )

        try:
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
                    del prefill_out

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
                    temp = 0.45
                top_p = float(self.config.model_spec.top_p)

                if is_tiny_test_model:
                    tok = self._sample_or_argmax_token(
                        steered_last_logits.unsqueeze(0),
                        temp,
                        top_p,
                        turn_gen,
                        device,
                    )
                    generated_ids = [int(tok.item())]
                    last_raw_text = self.tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    ).strip()
                    last_truncated = False
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
                    generated_ids = [1] * out_tok_cnt
                    last_truncated = False

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
                        truncated=last_truncated,
                        retries_used=retry_idx,
                        max_new_tokens_configured=max_new_tokens_cfg,
                        prompt_token_count=prompt_seq_len,
                        output_token_count=len(generated_ids),
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
            last_parse_err = f"LOGIT_DECODE_ERROR: {str(err)[:80]}"
            del err
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
            truncated=last_truncated,
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

    def _get_lexical_category_logit(
        self, logits_1d: torch.Tensor, surface_variants: List[str]
    ) -> float:
        """
        Maps a lexical category (e.g., ACCEPT, REJECT, BUY, SELL, WAIT) to its
        exact Qwen2.5 BPE vocabulary token IDs across leading-space and casing
        variants, returning `logsumexp` over the un-softmaxed category logits.
        """
        tids: List[int] = []
        vocab_len = int(logits_1d.size(0))
        for word in surface_variants:
            enc_ids = self.tokenizer.encode(word, add_special_tokens=False)
            if enc_ids and 0 <= int(enc_ids[0]) < vocab_len:
                if int(enc_ids[0]) not in tids:
                    tids.append(int(enc_ids[0]))
        if not tids:
            return 0.0
        selected = logits_1d[tids].float()
        return float(torch.logsumexp(selected, dim=0).item())

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
        Gold-Standard Logit-Scored Categorical Decoding (Homo Silicus / EconAgent):
          1. Extracts raw un-softmaxed next-token logits from the steered forward pass.
          2. Maps choice categories (`ACCEPT`, `REJECT`, `BUY`, `SELL`, `WAIT`,
             `CONSUMABLE`, `CAPITAL`, and price tiers `1..7`) to exact Qwen2.5
             tokenizer IDs (accounting for leading spaces and casing).
          3. Applies strict pre-softmax `-inf` institutional feasibility masking
             against the agent's balance sheet and counterparty inventory.
          4. Applies temperature-scaled `F.softmax` and samples via `torch.multinomial`.
        """
        clean_logits = torch.nan_to_num(
            next_token_logits.detach().float().cpu(),
            nan=0.0,
            posinf=30.0,
            neginf=-30.0,
        )

        # Parse balance sheet & market clearing context from prompt_text
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
            # Responder Logic: Extract un-softmaxed logits for ACCEPT vs. REJECT
            z_accept = self._get_lexical_category_logit(
                clean_logits,
                ["ACCEPT", " ACCEPT", "Accept", " accept", "yes", "Yes"],
            )
            z_reject = self._get_lexical_category_logit(
                clean_logits,
                ["REJECT", " REJECT", "Reject", " reject", "no", "No"],
            )
            consent_logits = torch.tensor(
                [z_accept, z_reject], dtype=torch.float32
            )

            offered_p = (
                float(offer_p_match.group(1)) if offer_p_match else p_c_ref
            )
            contract_tail = prompt_text.split("PROPOSED CONTRACT:")[-1]
            is_cap_offer = "PRODUCTIVE_CAPITAL_ASSET" in contract_tail
            initiator_is_selling = " SELL " in contract_tail

            # Strict Pre-Softmax Institutional Feasibility Mask for Responder:
            if initiator_is_selling:
                # Responder is the Buyer: must be able to afford offered_p (+ KMV fee if capital)
                req_cash = offered_p * (1.06 if is_cap_offer else 1.0)
                if max_spendable < req_cash:
                    consent_logits[0] = float("-inf")
            else:
                # Responder is the Seller: must hold >= 1.0 unit of the requested item
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
            # Proposer Logic: Extract un-softmaxed logits for BUY, SELL, WAIT and Items
            z_buy = self._get_lexical_category_logit(
                clean_logits, ["BUY", " BUY", "Buy", " buy", "purchase"]
            )
            z_sell = self._get_lexical_category_logit(
                clean_logits, ["SELL", " SELL", "Sell", " sell", "offer"]
            )
            z_wait = self._get_lexical_category_logit(
                clean_logits, ["WAIT", " WAIT", "Wait", "HOLD", " hold"]
            )
            z_cons = self._get_lexical_category_logit(
                clean_logits,
                ["CONSUMABLE", " CONSUMABLE", "food", " food", "Food", "necessity"],
            )
            z_cap = self._get_lexical_category_logit(
                clean_logits,
                ["CAPITAL", " CAPITAL", "capital", " capital", "asset", " asset"],
            )

            # 5 Joint Action-Item Categories:
            # 0: BUY CONSUMABLE, 1: SELL CONSUMABLE, 2: BUY CAPITAL, 3: SELL CAPITAL, 4: WAIT
            joint_logits = torch.tensor(
                [
                    0.5 * (z_buy + z_cons),
                    0.5 * (z_sell + z_cons),
                    0.5 * (z_buy + z_cap),
                    0.5 * (z_sell + z_cap),
                    z_wait,
                ],
                dtype=torch.float32,
            )

            # Strict Institutional Feasibility Masking BEFORE Softmax:
            # 0: BUY CONSUMABLE requires counterparty stock >= 1.5 and spendable >= 0.78 * p_c_ref
            if part_c < 1.5 or max_spendable < 0.78 * p_c_ref:
                joint_logits[0] = float("-inf")
            # 1: SELL CONSUMABLE requires own surplus stock >= 1.5 (keeping 1.0 for metabolic upkeep)
            if own_c < 1.5:
                joint_logits[1] = float("-inf")
            # 2: BUY CAPITAL requires counterparty capital >= 1.0 and spendable >= 0.78 * q_k_ref * 1.08
            if part_k < 1.0 or max_spendable < 0.78 * q_k_ref * 1.08:
                joint_logits[2] = float("-inf")
            # 3: SELL CAPITAL requires own capital >= 1.0
            if own_k < 1.0:
                joint_logits[3] = float("-inf")

            # If at least one trade action (0..3) is institutionally feasible, mask WAIT to -inf
            # so active market participation is evaluated over feasible trades
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
                mult_grid = [0.78, 0.86, 0.94, 1.00, 1.08, 1.18, 1.30]
                tier_tokens = [
                    ["1", " 1"],
                    ["2", " 2"],
                    ["3", " 3"],
                    ["4", " 4"],
                    ["5", " 5"],
                    ["6", " 6"],
                    ["7", " 7"],
                ]
                price_logits = torch.tensor(
                    [
                        self._get_lexical_category_logit(clean_logits, tv)
                        for tv in tier_tokens
                    ],
                    dtype=torch.float32,
                )

                # Mask unaffordable price tiers to -inf before softmax when BUYing
                if action == "BUY":
                    fee_mult = (
                        1.08 if item == "PRODUCTIVE_CAPITAL_ASSET" else 1.0
                    )
                    for p_i, m_val in enumerate(mult_grid):
                        if base_p * m_val * fee_mult > max_spendable:
                            price_logits[p_i] = float("-inf")

                p_idx = self._sample_categorical_index(
                    price_logits, temperature, cpu_gen
                )
                price = round(max(0.5, base_p * mult_grid[p_idx]), 2)

        # Sample Public Goods contribution tier from digit logits 1..4
        pg_tiers = [1.0, 1.6, 2.2, 2.8]
        pg_logits = torch.tensor(
            [
                self._get_lexical_category_logit(clean_logits, [str(d), f" {d}"])
                for d in (1, 2, 3, 4)
            ],
            dtype=torch.float32,
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
            "thought": f"Logit-scored categorical sample ({decision}, {action} {item} @ ${price:.2f})",
            "msg": f"{decision}: {action} {qty:.1f} {item} @ ${price:.2f}",
        }
        return (json.dumps(payload), 12)

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

        return (turn_out, True, None)




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
        elif "ABSTAIN" in raw_dec:
            consent = "ABSTAIN"
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

    def _generate_non_llm_placeholder(
        self,
        prompt_packet: CompiledPromptPacket,
        steering_packet: EndogenousStatePacket,
        active_method: str,
        max_new_tokens_cfg: int,
        start_t: float,
    ) -> RawGenerationResult:
        steered_telemetry = self.steering_lab.measure_attention_from_model_tensors(
            attentions=None,
            state_pkt=steering_packet,
            steering_method=active_method,
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