"""
src/brain/model_vault.py
MODULE 1 (M1 - Part A): ModelVaultManager & BackendCapabilityGate.
Uses native `torch.bfloat16` on Ampere+ CUDA GPUs to prevent float16 activation
overflow on Qwen2.5-1.5B / 7B models while keeping VRAM at ~1.07 GB in 4-bit NF4.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.core.contracts import BackendType, HardwareBackendSpec


class BackendCapabilityGate:
    def __init__(
        self,
        backend_type: BackendType,
        supports_whitebox_steering: bool,
        device: torch.device,
        model_fingerprint: str,
    ) -> None:
        self.backend_type = backend_type
        self.supports_whitebox_steering = supports_whitebox_steering
        self.device = device
        self.model_fingerprint = model_fingerprint


class ModelVaultManager:
    def __init__(self, spec: HardwareBackendSpec) -> None:
        self.spec = spec
        sanitized_name = spec.model_id.replace("/", "--")
        self.model_root = Path(spec.models_dir) / sanitized_name
        self.weights_dir = self.model_root / "weights"
        self.cache_dir = self.model_root / "mechanistic_cache"
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def resolve_device(self) -> torch.device:
        if self.spec.device.startswith("cuda") and torch.cuda.is_available():
            return torch.device(self.spec.device)
        return torch.device("cpu")

    def compute_model_fingerprint(self) -> str:
        payload = f"{self.spec.model_id}::{self.spec.quantization_mode}::bf16"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def get_calibration_cache_path(self) -> Path:
        fp = self.compute_model_fingerprint()
        return self.cache_dir / f"calibration_v2_{fp}.pt"

    def ensure_local_weights(self) -> str:
        has_safetensors = any(self.weights_dir.glob("*.safetensors"))
        if not has_safetensors:
            print(f"[ModelVault] Downloading '{self.spec.model_id}' into {self.weights_dir} ...")
            snapshot_download(
                repo_id=self.spec.model_id,
                local_dir=str(self.weights_dir),
                ignore_patterns=["*.msgpack", "*.h5", "original/*"],
            )
        return str(self.weights_dir)

    def load_whitebox_model_and_tokenizer(
        self,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, BackendCapabilityGate]:
        device = self.resolve_device()
        fingerprint = self.compute_model_fingerprint()
        load_source = self.ensure_local_weights()

        tokenizer = AutoTokenizer.from_pretrained(load_source, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        vram_cap_mib = max(1800, int(self.spec.vram_ceiling_gb * 1024))
        max_memory_map = {0: f"{vram_cap_mib}MiB", "cpu": "12GiB"}

        model_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
            "low_cpu_mem_usage": True,
        }

        if device.type == "cuda":
            compute_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            if self.spec.quantization_mode.lower() == "nf4_4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
                model_kwargs["quantization_config"] = bnb_config
                model_kwargs["device_map"] = "auto"
                model_kwargs["max_memory"] = max_memory_map
                model_kwargs["dtype"] = compute_dtype
            else:
                model_kwargs["dtype"] = compute_dtype
                model_kwargs["device_map"] = "auto"
                model_kwargs["max_memory"] = max_memory_map
        else:
            model_kwargs["dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(load_source, **model_kwargs)
        if device.type == "cuda" and "device_map" not in model_kwargs:
            model = model.to(device)
        model.eval()

        input_device = getattr(model, "device", device)
        gate = BackendCapabilityGate(
            backend_type=BackendType.WHITE_BOX_PYTORCH,
            supports_whitebox_steering=True,
            device=input_device,
            model_fingerprint=fingerprint,
        )
        return model, tokenizer, gate
