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


class TwoStageRepEController:
    """
    Master LLM Inference Controller implementing `TwoStageLLMControllerProtocol`.
    Executes Pass 1 (Unsteered CoT Appraisal + Frozen Linear Probe -> lambda)
    followed by Pass 2 (Action-Only Vector Injection h' = h + lambda*gamma*v -> JSON).
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

    def _run_generate(
        self,
        prompt_text: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Tuple[str, int, bool]:
        device = next(self.model.parameters()).device
        enc = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=1536,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_len = int(input_ids.size(1))
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        do_sample = temperature > 1e-5
        gen_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max(1, int(max_new_tokens)),
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

        try:
            with torch.inference_mode():
                out_ids = self.model.generate(**gen_kwargs)
            new_tokens = out_ids[0, prompt_len:].detach().cpu().tolist()
            num_gen = len(new_tokens)
            hit_limit = num_gen >= int(max_new_tokens) and (
                eos_id is None or (num_gen > 0 and new_tokens[-1] != eos_id)
            )
            decoded = self.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()
            return (decoded, num_gen, hit_limit)
        finally:
            del input_ids, attention_mask
            if "out_ids" in locals():
                del out_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        t0 = time.perf_counter()
        base_temp = float(self.config.model.temperature)
        top_p = float(self.config.model.top_p)
        max_retries = max(1, int(self.config.model.max_json_retries))

        stage1_prompt = self._format_chat_prompt(
            system_prompt=stage1_system_prompt,
            user_messages=[{"role": "user", "content": stage1_user_prompt}],
        )
        self._seed_torch_for_call(tick, agent_id, "stage1_cot", 0)

        with self.repe_manager.capture_stage1_cot_residuals(self.model):
            cot_text, cot_tok_count, cot_truncated = self._run_generate(
                prompt_text=stage1_prompt,
                max_new_tokens=cot_max_tokens,
                temperature=base_temp,
                top_p=top_p,
            )

        probe_measurement: ProbeGateMeasurement = (
            self.repe_manager.compute_endogenous_lambda()
        )

        stage2_prompt = self._format_chat_prompt(
            system_prompt=stage1_system_prompt,
            user_messages=[
                {"role": "user", "content": stage1_user_prompt},
                {"role": "assistant", "content": cot_text or "Evaluating reserves."},
                {"role": "user", "content": stage2_instruction},
            ],
        )

        last_raw_json = ""
        last_action_tokens = 0

        for retry_idx in range(max_retries):
            self._seed_torch_for_call(tick, agent_id, "stage2_action", retry_idx)
            retry_temp = base_temp if retry_idx == 0 else max(0.25, base_temp * 0.75)

            with self.repe_manager.inject_stage2_action_steering(
                model=self.model,
                probe_measurement=probe_measurement,
            ):
                raw_json, act_tok_cnt, _ = self._run_generate(
                    prompt_text=stage2_prompt,
                    max_new_tokens=action_max_tokens,
                    temperature=retry_temp,
                    top_p=top_p,
                )

            last_raw_json = raw_json
            last_action_tokens = act_tok_cnt

            parsed_payload, is_valid, _ = parse_action_fn(raw_json)
            if is_valid:
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return TwoStageGenerationOutput(
                    stage1_cot_text=cot_text,
                    cot_tokens_generated=cot_tok_count,
                    cot_truncated_by_bottleneck=cot_truncated,
                    probe_measurement=probe_measurement,
                    stage2_raw_json=raw_json,
                    stage2_action=parsed_payload,
                    json_valid=True,
                    abstain=False,
                    retries_used=retry_idx,
                    action_tokens_generated=act_tok_cnt,
                    latency_ms=latency_ms,
                )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return TwoStageGenerationOutput(
            stage1_cot_text=cot_text,
            cot_tokens_generated=cot_tok_count,
            cot_truncated_by_bottleneck=cot_truncated,
            probe_measurement=probe_measurement,
            stage2_raw_json=last_raw_json,
            stage2_action=Stage2ActionPayload(action_type=AgentActionType.ABSTAIN),
            json_valid=False,
            abstain=True,
            retries_used=max_retries,
            action_tokens_generated=last_action_tokens,
            latency_ms=latency_ms,
        )