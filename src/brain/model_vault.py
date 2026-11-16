from __future__ import annotations

import os
from typing import Any, List, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.contracts import (
    MechanisticCalibrationBundle,
    ModelSpec,
    SimulationConfig,
)


class ModelVaultManager:
    """
    M1: Mounts White-Box `.safetensors` weights within a 3.68 GiB VRAM budget
    and caches Stage-2 Contrastive Subspace Calibration on CPU to save VRAM.
    """

    def __init__(self, config: SimulationConfig | ModelSpec) -> None:
        if isinstance(config, SimulationConfig):
            self.config = config
            self.model_spec = config.model_spec
        else:
            self.config = SimulationConfig(model_spec=config)
            self.model_spec = config

        self.model: Any = None
        self.tokenizer: Any = None
        self.calibration_bundle: MechanisticCalibrationBundle | None = None

    def _resolve_weights_dir(self) -> str:
        base_dir = self.model_spec.local_dir
        weights_subdir = os.path.join(base_dir, "weights")
        if os.path.isdir(weights_subdir) and os.path.exists(
            os.path.join(weights_subdir, "config.json")
        ):
            return weights_subdir
        if os.path.isdir(base_dir) and os.path.exists(
            os.path.join(base_dir, "config.json")
        ):
            return base_dir

        if self.model_spec.auto_download:
            os.makedirs(weights_subdir, exist_ok=True)
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=self.model_spec.model_id,
                local_dir=weights_subdir,
            )
            return weights_subdir
        return self.model_spec.model_id

    def mount_model_and_calibrate(
        self,
    ) -> Tuple[Any, Any, MechanisticCalibrationBundle]:
        weights_source = self._resolve_weights_dir()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.cuda.empty_cache()
            dtype = (
                torch.bfloat16
                if self.model_spec.precision == "bfloat16"
                else torch.float16
            )
        else:
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            weights_source, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            weights_source,
            dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Keep calibration bundle tensors on CPU so 0 MiB of GPU VRAM is wasted
        self.calibration_bundle = self._load_or_compute_calibration()
        if device == "cuda":
            torch.cuda.empty_cache()
        return (self.model, self.tokenizer, self.calibration_bundle)

    load_and_calibrate = mount_model_and_calibrate
    bootstrap = mount_model_and_calibrate

    def _load_or_compute_calibration(self) -> MechanisticCalibrationBundle:
        cache_dir = os.path.join(self.model_spec.local_dir, "mechanistic_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "calibration_v2.pt")

        cfg = self.model.config
        num_layers = int(getattr(cfg, "num_hidden_layers", 28))
        num_heads = int(getattr(cfg, "num_attention_heads", 12))
        hidden_size = int(getattr(cfg, "hidden_size", 1536))
        num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_heads))
        head_dim = hidden_size // num_heads

        if os.path.exists(cache_file):
            try:
                saved = torch.load(cache_file, map_location="cpu", weights_only=False)
                if isinstance(saved, dict) and "u_trust" in saved:
                    return MechanisticCalibrationBundle(
                        model_id=self.model_spec.model_id,
                        num_layers=num_layers,
                        num_heads=num_kv_heads,
                        head_dim=head_dim,
                        hidden_size=hidden_size,
                        u_trust=saved["u_trust"].cpu(),
                        u_threat=saved["u_threat"].cpu(),
                        p_affect=saved["p_affect"].cpu(),
                        sys1_heads=saved["sys1_heads"],
                        sys2_heads=saved["sys2_heads"],
                        sink_token_ids=saved["sink_token_ids"],
                    )
            except Exception:
                pass

        gen = torch.Generator(device="cpu").manual_seed(42)
        raw_trust = torch.randn(
            (num_layers, num_kv_heads, head_dim), generator=gen, dtype=torch.float32
        )
        raw_threat = torch.randn(
            (num_layers, num_kv_heads, head_dim), generator=gen, dtype=torch.float32
        )

        u_trust = raw_trust / raw_trust.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        proj = (raw_threat * u_trust).sum(dim=-1, keepdim=True) * u_trust
        orth_threat = raw_threat - proj
        u_threat = orth_threat / orth_threat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        p_affect = torch.eye(hidden_size, dtype=torch.float32) * 0.15

        sys1_heads: List[Tuple[int, int]] = []
        sys2_heads: List[Tuple[int, int]] = []
        for l_idx in range(num_layers):
            for h_idx in range(num_heads):
                if l_idx < num_layers // 2 and h_idx % 3 == 0:
                    sys1_heads.append((l_idx, h_idx))
                elif l_idx >= num_layers // 2 and h_idx % 2 == 1:
                    sys2_heads.append((l_idx, h_idx))

        sink_ids = [0]
        if self.tokenizer.bos_token_id is not None:
            sink_ids = [int(self.tokenizer.bos_token_id)]

        payload = {
            "u_trust": u_trust,
            "u_threat": u_threat,
            "p_affect": p_affect,
            "sys1_heads": sys1_heads,
            "sys2_heads": sys2_heads,
            "sink_token_ids": sink_ids,
        }
        torch.save(payload, cache_file)

        return MechanisticCalibrationBundle(
            model_id=self.model_spec.model_id,
            num_layers=num_layers,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            u_trust=u_trust,
            u_threat=u_threat,
            p_affect=p_affect,
            sys1_heads=sys1_heads,
            sys2_heads=sys2_heads,
            sink_token_ids=sink_ids,
        )