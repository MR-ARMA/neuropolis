from __future__ import annotations

import gc
import hashlib
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.brain.repe_manager import DisjointRepEArtifacts, RepEManager
from src.core.contracts import (
    AgentActionType,
    ModelRuntimeConfig,
    ProbeGateMeasurement,
    SimulationConfig,
    Stage2ActionPayload,
)
from src.orchestration.turn_orchestrator import (
    IsomorphicVocabulary,
    TwoStageGenerationOutput,
)


def has_native_hardware_bf16() -> bool:
    """
    Returns True strictly on Ampere+ GPUs (compute capability >= 8.0).
    Tesla T4 (Turing sm_75) lacks native BF16 tensor cores and must use float16.
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8


class WhiteBoxModelLoader:
    """
    Hardware-Optimized White-Box Loader for Llama-3.1 / Qwen-2.5 Instruct models.
      - Uses hardware compute capability (`sm >= 8.0`) so Tesla T4 runs native float16
        while RTX 3050 / A100 / H100 run native bfloat16.
      - Enforces `tokenizer.pad_token = tokenizer.eos_token` and syncs
        `model.config.pad_token_id` (required for batched Llama-3.1 extraction).
      - Provides `load_with_fallback(config)` for automatic Llama-3.1 -> Qwen2.5-3B fallback.
    """

    FALLBACK_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
    FALLBACK_LOCAL_DIR = "./models/Qwen--Qwen2.5-3B-Instruct"
    FALLBACK_TARGET_LAYER = 18

    @staticmethod
    def resolve_local_or_hub_path(model_cfg: ModelRuntimeConfig) -> str:
        # Check if config.json exists directly in local_dir
        if os.path.isdir(model_cfg.local_dir) and os.path.exists(
            os.path.join(model_cfg.local_dir, "config.json")
        ):
            return model_cfg.local_dir
            
        # Check if it exists in a weights subdirectory (for Kaggle compatibility)
        weights_sub = os.path.join(model_cfg.local_dir, "weights")
        if os.path.isdir(weights_sub) and os.path.exists(
            os.path.join(weights_sub, "config.json")
        ):
            return weights_sub
            
        return model_cfg.model_id

    @classmethod
    def load_model_and_tokenizer(
        cls, model_cfg: ModelRuntimeConfig
    ) -> Tuple[Any, Any]:
        source = cls.resolve_local_or_hub_path(model_cfg)
        has_cuda = torch.cuda.is_available()

        tok_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if source == model_cfg.model_id:
            tok_kwargs["revision"] = model_cfg.revision
        tokenizer = AutoTokenizer.from_pretrained(source, **tok_kwargs)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        if has_cuda:
            torch.cuda.empty_cache()
            if model_cfg.compute_dtype == "bfloat16" and has_native_hardware_bf16():
                compute_dtype = torch.bfloat16
            elif model_cfg.compute_dtype == "float32":
                compute_dtype = torch.float32
            else:
                compute_dtype = torch.float16
        else:
            compute_dtype = torch.float32

        load_kwargs: Dict[str, Any] = {
            "attn_implementation": model_cfg.attn_implementation,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if source == model_cfg.model_id:
            load_kwargs["revision"] = model_cfg.revision

        if has_cuda and model_cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=model_cfg.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            load_kwargs["quantization_config"] = bnb_cfg
            load_kwargs["device_map"] = {"": 0}
        elif has_cuda:
            load_kwargs["dtype"] = compute_dtype
            load_kwargs["device_map"] = {"": 0}
        else:
            load_kwargs["dtype"] = torch.float32
            load_kwargs["device_map"] = "cpu"

        model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        actual_impl = getattr(
            model.config, "_attn_implementation", model_cfg.attn_implementation
        )
        assert actual_impl == model_cfg.attn_implementation, (
            f"Attention backend mismatch: requested={model_cfg.attn_implementation}, actual={actual_impl}"
        )
        return (model, tokenizer)

    @classmethod
    def load_with_fallback(
        cls, config: SimulationConfig
    ) -> Tuple[Any, Any]:
        try:
            return cls.load_model_and_tokenizer(config.model)
        except Exception as primary_err:
            if config.model.model_id == cls.FALLBACK_MODEL_ID:
                raise
            print(
                f"[LOADER WARN] Primary model '{config.model.model_id}' failed ({primary_err}). "
                f"Falling back automatically to '{cls.FALLBACK_MODEL_ID}'..."
            )
            os.environ.pop("HF_TOKEN", None)
            config.model.model_id = cls.FALLBACK_MODEL_ID
            config.model.local_dir = cls.FALLBACK_LOCAL_DIR
            config.repe.target_layer = cls.FALLBACK_TARGET_LAYER
            config.repe.probe_artifact_path = (
                f"artifacts/probes/inequity_probe_l{cls.FALLBACK_TARGET_LAYER}.pt"
            )
            return cls.load_model_and_tokenizer(config.model)


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)


class PerSampleBudgetLogitsProcessor(LogitsProcessor):
    """
    Enforces exact per-agent Stage 1 Bounded Deliberation Budgets (Factor A)
    inside a single fused GPU batch: once batch element `i` reaches its personal
    `cot_max_tokens[i]`, emits `eos_token_id` strictly for row `i` without
    perturbing any active token logits or blocking calmer agents in the batch.
    """

    def __init__(
        self, prompt_len: int, budgets: List[int], eos_token_id: int
    ) -> None:
        self.prompt_len = int(prompt_len)
        self.budgets = [max(1, int(b)) for b in budgets]
        self.eos_token_id = int(eos_token_id)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        gen_step = int(input_ids.shape[1]) - self.prompt_len
        for row_idx, max_toks in enumerate(self.budgets):
            if row_idx < scores.shape[0] and gen_step >= max_toks:
                scores[row_idx, :] = float("-inf")
                scores[row_idx, self.eos_token_id] = 0.0
        return scores


class TwoStageRepEController:
    """
    Optimized Two-Stage Probe-Gated RepE Master Controller.
      - Adds `execute_two_stage_batch()` to run all B living agents per tick in
        ONE Stage-1 CoT GPU batch + ONE Stage-2 Action JSON GPU batch (~15x speedup
        on Tesla T4 while preserving per-agent CoT budgets and per-agent RepE lambdas).
      - Eliminates per-turn `gc.collect()` and `torch.cuda.empty_cache()` stalls.
      - Retains `execute_two_stage_turn()` for 100% compatibility with Pilot Gates G1–G4.
    """

    def __init__(
        self,
        config: SimulationConfig,
        model: Any,
        tokenizer: Any,
        repe_manager: RepEManager,
        economy_seed: Optional[int] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.repe_manager = repe_manager
        self.economy_seed = (
            economy_seed if economy_seed is not None else config.master_seed
        )

    @classmethod
    def from_config(
        cls,
        config: SimulationConfig,
        economy_seed: Optional[int] = None,
        force_recompute_probe: bool = False,
    ) -> "TwoStageRepEController":
        model, tokenizer = WhiteBoxModelLoader.load_with_fallback(config)
        repe_mgr = RepEManager(
            config=config.repe,
            seed=economy_seed if economy_seed is not None else config.master_seed,
        )
        repe_mgr.load_or_extract_artifacts(
            model=model,
            tokenizer=tokenizer,
            model_id=config.model.model_id,
            force_recompute=force_recompute_probe,
        )
        return cls(
            config=config,
            model=model,
            tokenizer=tokenizer,
            repe_manager=repe_mgr,
            economy_seed=economy_seed,
        )

    def _seed_torch_for_call(
        self, tick: int, agent_id: str, stage: str, retry_idx: int
    ) -> None:
        key = f"{self.economy_seed}::{tick}::{agent_id}::{stage}::{retry_idx}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        sub_seed = int(digest[:8], 16) & 0x7FFFFFFF
        torch.manual_seed(sub_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sub_seed)

    def _format_chat_prompt(
        self,
        system_prompt: str,
        user_messages: List[Dict[str, str]],
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}] + user_messages
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return str(
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            except Exception:
                pass
        parts = [f"System: {system_prompt}"]
        for m in user_messages:
            parts.append(f"{m['role'].capitalize()}: {m['content']}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _run_generate_batch(
        self,
        prompt_texts: List[str],
        max_new_tokens_per_item: List[int],
        temperature: float,
        top_p: float,
    ) -> List[Tuple[str, int, bool]]:
        """
        Fused GPU batched generation with per-sample `max_new_tokens` enforcement
        and zero per-turn `torch.cuda.empty_cache()` synchronization stalls.
        """
        device = next(self.model.parameters()).device
        self.tokenizer.padding_side = "left"
        eos_id = self.tokenizer.eos_token_id
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else eos_id
        )

        enc = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_len = int(input_ids.size(1))
        global_max_new = max(1, max(max_new_tokens_per_item))

        do_sample = temperature > 1e-5
        gen_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": global_max_new,
            "do_sample": do_sample,
            "use_cache": True,
            "pad_token_id": pad_id if pad_id is not None else eos_id,
            "eos_token_id": eos_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)
        else:
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None
            gen_kwargs["top_k"] = None
            gen_kwargs["repetition_penalty"] = 1.0

        # Attach per-sample CoT budget enforcer only when batch items have differing budgets
        if (
            len(prompt_texts) > 1
            and eos_id is not None
            and len(set(max_new_tokens_per_item)) > 1
        ):
            gen_kwargs["logits_processor"] = LogitsProcessorList(
                [
                    PerSampleBudgetLogitsProcessor(
                        prompt_len=prompt_len,
                        budgets=max_new_tokens_per_item,
                        eos_token_id=int(eos_id),
                    )
                ]
            )

        with torch.inference_mode():
            out_ids = self.model.generate(**gen_kwargs)

        new_tokens_batch = out_ids[:, prompt_len:].detach().cpu().tolist()
        del input_ids, attention_mask, out_ids

        outputs: List[Tuple[str, int, bool]] = []
        stop_ids = {t for t in (eos_id, pad_id) if t is not None}
        for idx, tok_seq in enumerate(new_tokens_batch):
            budget_i = int(max_new_tokens_per_item[idx])
            # Trim tokens after first EOS / PAD
            trimmed: List[int] = []
            hit_eos = False
            for tid in tok_seq[:budget_i]:
                if tid in stop_ids:
                    hit_eos = True
                    break
                trimmed.append(tid)
            num_gen = len(trimmed)
            hit_limit = (num_gen >= budget_i) and (not hit_eos)
            decoded = self.tokenizer.decode(
                trimmed, skip_special_tokens=True
            ).strip()
            outputs.append((decoded, num_gen, hit_limit))
        return outputs

    def _run_generate(
        self,
        prompt_text: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Tuple[str, int, bool]:
        """Single-prompt wrapper (used by `scripts/run_pilot_gates.py`)."""
        return self._run_generate_batch(
            prompt_texts=[prompt_text],
            max_new_tokens_per_item=[max_new_tokens],
            temperature=temperature,
            top_p=top_p,
        )[0]

    def execute_two_stage_batch(
        self,
        stage1_system_prompts: List[str],
        stage1_user_prompts: List[str],
        stage2_instruction: str,
        cot_max_tokens_list: List[int],
        action_max_tokens: int,
        agent_ids: List[str],
        tick: int,
        parse_action_fn: Callable[
            [str], Tuple[Stage2ActionPayload, bool, Optional[str]]
        ] = IsomorphicVocabulary.parse_sanitized_action_json,
    ) -> List[TwoStageGenerationOutput]:
        """
        Executes Stage 1 (Unsteered Appraisal CoT + Vectorized Linear Probe) and
        Stage 2 (Per-Agent Steered JSON Action) across all B agents in two single
        batched GPU forward passes, retrying ONLY invalid JSON indices.
        """
        t0 = time.perf_counter()
        bsz = len(agent_ids)
        base_temp = float(self.config.model.temperature)
        top_p = float(self.config.model.top_p)
        max_retries = max(1, int(self.config.model.max_json_retries))

        # PASS 1: Batched Unsteered Stage 1 CoT + Zero-Sync GPU Residual Capture
        stage1_prompts = [
            self._format_chat_prompt(
                system_prompt=stage1_system_prompts[i],
                user_messages=[{"role": "user", "content": stage1_user_prompts[i]}],
            )
            for i in range(bsz)
        ]
        self._seed_torch_for_call(tick, "_".join(agent_ids[:3]), "stage1_batch", 0)

        with self.repe_manager.capture_stage1_cot_residuals(self.model):
            s1_outputs = self._run_generate_batch(
                prompt_texts=stage1_prompts,
                max_new_tokens_per_item=cot_max_tokens_list,
                temperature=base_temp,
                top_p=top_p,
            )

        probe_measurements = self.repe_manager.compute_endogenous_lambda_batch(bsz)

        # PASS 2: Batched Steered Stage 2 JSON Action (retry ONLY failed indices)
        stage2_prompts = [
            self._format_chat_prompt(
                system_prompt=stage1_system_prompts[i],
                user_messages=[
                    {"role": "user", "content": stage1_user_prompts[i]},
                    {"role": "assistant", "content": s1_outputs[i][0] or "Evaluating reserves."},
                    {"role": "user", "content": stage2_instruction},
                ],
            )
            for i in range(bsz)
        ]

        final_actions: List[Optional[Stage2ActionPayload]] = [None] * bsz
        final_raws: List[str] = [""] * bsz
        final_valids: List[bool] = [False] * bsz
        final_retries: List[int] = [max_retries] * bsz
        final_act_toks: List[int] = [0] * bsz

        pending_indices = list(range(bsz))
        for retry_idx in range(max_retries):
            if not pending_indices:
                break
            self._seed_torch_for_call(
                tick, "_".join(agent_ids[i] for i in pending_indices[:3]), "stage2_batch", retry_idx
            )
            retry_temp = base_temp if retry_idx == 0 else max(0.25, base_temp * 0.75)
            sub_prompts = [stage2_prompts[i] for i in pending_indices]
            sub_pms = [probe_measurements[i] for i in pending_indices]
            sub_budgets = [int(action_max_tokens)] * len(pending_indices)

            with self.repe_manager.inject_stage2_action_steering_batch(
                model=self.model, probe_measurements=sub_pms
            ):
                s2_sub_outs = self._run_generate_batch(
                    prompt_texts=sub_prompts,
                    max_new_tokens_per_item=sub_budgets,
                    temperature=retry_temp,
                    top_p=top_p,
                )

            next_pending: List[int] = []
            for local_i, orig_i in enumerate(pending_indices):
                raw_json, act_cnt, _ = s2_sub_outs[local_i]
                final_raws[orig_i] = raw_json
                final_act_toks[orig_i] = act_cnt
                parsed, ok, _ = parse_action_fn(raw_json)
                if ok:
                    final_actions[orig_i] = parsed
                    final_valids[orig_i] = True
                    final_retries[orig_i] = retry_idx
                else:
                    next_pending.append(orig_i)
            pending_indices = next_pending

        per_agent_ms = round(((time.perf_counter() - t0) * 1000.0) / max(1, bsz), 2)
        results: List[TwoStageGenerationOutput] = []
        for i in range(bsz):
            cot_txt, cot_cnt, cot_trunc = s1_outputs[i]
            is_ok = final_valids[i]
            payload = (
                final_actions[i]
                if (is_ok and final_actions[i] is not None)
                else Stage2ActionPayload(action_type=AgentActionType.ABSTAIN)
            )
            results.append(
                TwoStageGenerationOutput(
                    stage1_cot_text=cot_txt,
                    cot_tokens_generated=cot_cnt,
                    cot_truncated_by_bottleneck=cot_trunc,
                    probe_measurement=probe_measurements[i],
                    stage2_raw_json=final_raws[i],
                    stage2_action=payload,
                    json_valid=is_ok,
                    abstain=not is_ok,
                    retries_used=final_retries[i],
                    action_tokens_generated=final_act_toks[i],
                    latency_ms=per_agent_ms,
                )
            )
        return results

    def execute_two_stage_turn(
        self,
        stage1_system_prompt: str,
        stage1_user_prompt: str,
        stage2_instruction: str,
        cot_max_tokens: int,
        action_max_tokens: int,
        agent_id: str,
        tick: int,
        parse_action_fn: Callable[
            [str], Tuple[Stage2ActionPayload, bool, Optional[str]]
        ] = IsomorphicVocabulary.parse_sanitized_action_json,
    ) -> TwoStageGenerationOutput:
        """Single-agent wrapper delegating cleanly to `execute_two_stage_batch`."""
        return self.execute_two_stage_batch(
            stage1_system_prompts=[stage1_system_prompt],
            stage1_user_prompts=[stage1_user_prompt],
            stage2_instruction=stage2_instruction,
            cot_max_tokens_list=[cot_max_tokens],
            action_max_tokens=action_max_tokens,
            agent_ids=[agent_id],
            tick=tick,
            parse_action_fn=parse_action_fn,
        )[0]