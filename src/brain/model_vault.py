from __future__ import annotations

import os
from typing import Any, List, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.contracts import (
    MechanisticCalibrationBundle,
    ModelSpec,
    SimulationConfig,
)


# 16 Contrastive Economic Prompts for Genuine CAA Subspace Extraction (Rimsky et al. 2024)
CONTRASTIVE_TRUST_PAIRS: List[Tuple[str, str]] = [
    (
        "Counterparty has honored every trade contract fairly and offers transparent market pricing.",
        "Counterparty repeatedly breached contracts, hid defects, and attempts predatory price gouging.",
    ),
    (
        "Trusted partner proposes a mutually beneficial exchange with verified inventory and fair margin.",
        "Opportunistic rival demands an exploitative markup while concealing severe solvency risk.",
    ),
    (
        "Long-standing cooperative neighbor shares accurate supply information and splits surplus equally.",
        "Deceptive trader fabricates scarcity rumors to extract unfair rent from desperate buyers.",
    ),
    (
        "Reliable merchant offers a fair bilateral contract honoring historical reciprocity.",
        "Hostile speculator exploits urgent household need with an extortionate ultimatum.",
    ),
    (
        "Verified collaborator fulfills delivery obligations on time and maintains high credibility.",
        "Defaulting counterparty reneges on promised delivery and pockets the advance payment.",
    ),
    (
        "Honest producer prices necessities near production cost to sustain community trust.",
        "Predatory monopolist corners the necessity market to gouge liquidity-constrained households.",
    ),
    (
        "Transparent partner discloses exact asset yields and proposes an equitable risk-sharing trade.",
        "Manipulative broker misrepresents failing capital assets to offload losses onto others.",
    ),
    (
        "Reciprocal ally extends fair terms during a market downturn without exploiting distress.",
        "Adversarial trader weaponizes a liquidity crunch to seize collateral at fire-sale prices.",
    ),
]

CONTRASTIVE_THREAT_PAIRS: List[Tuple[str, str]] = [
    (
        "Emergency liquidity crisis: cash reserves are exhausted, subsistence food stock is zero, and debt default is imminent.",
        "Comfortable liquidity surplus: cash reserves are abundant, food stocks are full, and all obligations are paid.",
    ),
    (
        "Severe stagflation shock has wiped out household purchasing power and threatens immediate starvation.",
        "Stable macroeconomic equilibrium with low inflation, steady wages, and secure household consumption.",
    ),
    (
        "Urgent solvency threat: mandatory housing obligation exceeds remaining cash and credit limit is frozen.",
        "High financial safety: liquid savings easily cover all periodic obligations with ample buffer.",
    ),
    (
        "Acute survival deficit: household must liquidate productive capital at a steep loss today just to eat.",
        "Long-horizon prosperity: household accumulates productive capital and earns steady dividend income.",
    ),
    (
        "Systemic credit crunch: borrowing is blocked, wages have been cut, and default penalties are triggered.",
        "Fluid credit access: low borrowing rates, rising real wages, and zero risk of default.",
    ),
    (
        "Household faces immediate consumption collapse unless emergency liquidity is secured this turn.",
        "Household enjoys a multi-period consumption buffer and can bargain patiently.",
    ),
    (
        "High-stress insolvency warning: net worth is eroding rapidly under compounding debt service.",
        "Strong balance sheet growth: positive net savings and appreciating capital holdings.",
    ),
    (
        "Critical subsistence shortfall: zero consumable inventory remaining before end-of-tick upkeep.",
        "Abundant consumable inventory: six ticks of subsistence reserves safely stored.",
    ),
]


class ModelVaultManager:
    """
    M1: Mounts White-Box `.safetensors` weights (`attn_implementation='eager'`)
    and executes genuine Contrastive Activation Addition (CAA) forward-pass
    calibration over paired economic prompts to extract empirical `u_trust`,
    `u_threat`, and `residual_caa_vector` directions, plus leave-one-out probe accuracy.
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
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            weights_source, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use SDPA (which upcasts QK^T softmax to fp32 internally so Qwen2 never
        # overflows fp16) while q_proj/k_proj/v_proj hooks still execute in full!
        self.model = AutoModelForCausalLM.from_pretrained(
            weights_source,
            dtype=dtype,
            device_map=device,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.calibration_bundle = self._load_or_compute_caa_calibration()
        if device == "cuda":
            torch.cuda.empty_cache()
        return (self.model, self.tokenizer, self.calibration_bundle)

    load_and_calibrate = mount_model_and_calibrate
    bootstrap = mount_model_and_calibrate

    def _load_or_compute_caa_calibration(self) -> MechanisticCalibrationBundle:
        cache_dir = os.path.join(self.model_spec.local_dir, "mechanistic_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "caa_calibration_v3.pt")

        cfg = self.model.config
        num_layers = int(getattr(cfg, "num_hidden_layers", 28))
        num_heads = int(getattr(cfg, "num_attention_heads", 12))
        num_kv_heads = int(getattr(cfg, "num_key_value_heads", 2))
        hidden_size = int(getattr(cfg, "hidden_size", 1536))
        head_dim = hidden_size // num_heads

        if os.path.exists(cache_file):
            try:
                saved = torch.load(cache_file, map_location="cpu", weights_only=False)
                if (
                    isinstance(saved, dict)
                    and saved.get("calibrated_from_model") is True
                    and "u_trust" in saved
                ):
                    return MechanisticCalibrationBundle(
                        model_id=self.model_spec.model_id,
                        num_layers=num_layers,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        hidden_size=hidden_size,
                        u_trust=saved["u_trust"].cpu(),
                        u_threat=saved["u_threat"].cpu(),
                        random_placebo_u=saved["random_placebo_u"].cpu(),
                        residual_caa_vector=saved["residual_caa_vector"].cpu(),
                        probe_accuracy_trust=float(saved["probe_accuracy_trust"]),
                        probe_accuracy_threat=float(saved["probe_accuracy_threat"]),
                        calibrated_from_model=True,
                        sink_token_ids=saved["sink_token_ids"],
                    )
            except Exception:
                pass

        # Execute Real Forward Passes over Contrastive Economic Prompts
        u_trust, _, acc_trust = self._extract_contrastive_subspace(
            pairs=CONTRASTIVE_TRUST_PAIRS,
            num_layers=num_layers,
            hidden_size=hidden_size,
        )
        u_threat, res_caa, acc_threat = self._extract_contrastive_subspace(
            pairs=CONTRASTIVE_THREAT_PAIRS,
            num_layers=num_layers,
            hidden_size=hidden_size,
        )

        # Gram-Schmidt orthogonalize u_threat against u_trust per layer
        proj = (u_threat * u_trust).sum(dim=-1, keepdim=True) * u_trust
        u_threat = F.normalize(u_threat - proj, p=2, dim=-1, eps=1e-6)

        # Create matched-norm random placebo direction (Control C5) orthogonal to both
        gen = torch.Generator(device="cpu").manual_seed(self.config.random_seed + 999)
        rand_raw = torch.randn(u_trust.shape, generator=gen, dtype=torch.float32)
        rand_raw = rand_raw - (rand_raw * u_trust).sum(dim=-1, keepdim=True) * u_trust
        rand_raw = rand_raw - (rand_raw * u_threat).sum(dim=-1, keepdim=True) * u_threat
        random_placebo_u = F.normalize(rand_raw, p=2, dim=-1, eps=1e-6)

        sink_ids = [0]
        if self.tokenizer.bos_token_id is not None:
            sink_ids = [int(self.tokenizer.bos_token_id)]

        payload = {
            "u_trust": u_trust,
            "u_threat": u_threat,
            "random_placebo_u": random_placebo_u,
            "residual_caa_vector": res_caa,
            "probe_accuracy_trust": acc_trust,
            "probe_accuracy_threat": acc_threat,
            "calibrated_from_model": True,
            "sink_token_ids": sink_ids,
        }
        torch.save(payload, cache_file)

        return MechanisticCalibrationBundle(
            model_id=self.model_spec.model_id,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            u_trust=u_trust,
            u_threat=u_threat,
            random_placebo_u=random_placebo_u,
            residual_caa_vector=res_caa,
            probe_accuracy_trust=acc_trust,
            probe_accuracy_threat=acc_threat,
            calibrated_from_model=True,
            sink_token_ids=sink_ids,
        )

    def _extract_contrastive_subspace(
        self,
        pairs: List[Tuple[str, str]],
        num_layers: int,
        hidden_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Runs paired positive/negative prompts through `self.model`, captures the
        final-token `v_proj` and residual activations at every layer, computes the
        Difference-in-Means CAA vector, and evaluates Leave-One-Out probe accuracy.
        """
        device = next(self.model.parameters()).device
        layers = self.model.model.layers
        d_kv = int(layers[0].self_attn.v_proj.out_features)

        v_diffs: List[torch.Tensor] = []
        h_diffs: List[torch.Tensor] = []

        for pos_text, neg_text in pairs:
            v_pos, h_pos = self._capture_single_pass_activations(
                pos_text, device, num_layers, d_kv, hidden_size
            )
            v_neg, h_neg = self._capture_single_pass_activations(
                neg_text, device, num_layers, d_kv, hidden_size
            )
            v_diffs.append(v_pos - v_neg)
            h_diffs.append(h_pos - h_neg)

        stacked_v = torch.stack(v_diffs, dim=0)  # [N_pairs, L, d_kv]
        stacked_h = torch.stack(h_diffs, dim=0)  # [N_pairs, L, hidden_size]

        mean_v_dir = F.normalize(stacked_v.mean(dim=0), p=2, dim=-1, eps=1e-6)
        mean_h_dir = F.normalize(stacked_h.mean(dim=0), p=2, dim=-1, eps=1e-6)

        # Evaluate Leave-One-Out (LOO) classification accuracy across middle layers
        mid_start = num_layers // 3
        mid_end = (2 * num_layers) // 3
        n_pairs = len(pairs)
        correct = 0
        total = 0

        for idx in range(n_pairs):
            loo_mask = [i for i in range(n_pairs) if i != idx]
            loo_dir = F.normalize(
                stacked_v[loo_mask].mean(dim=0), p=2, dim=-1, eps=1e-6
            )
            test_vec = stacked_v[idx]
            for l_i in range(mid_start, mid_end):
                dot_val = float(torch.dot(test_vec[l_i], loo_dir[l_i]).item())
                if dot_val > 0.0:
                    correct += 1
                total += 1

        loo_acc = round(correct / max(1, total), 4)
        return (mean_v_dir.cpu(), mean_h_dir.cpu(), loo_acc)

    def _capture_single_pass_activations(
        self,
        text: str,
        device: torch.device,
        num_layers: int,
        d_kv: int,
        hidden_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        captured_v = torch.zeros((num_layers, d_kv), dtype=torch.float32)
        captured_h = torch.zeros((num_layers, hidden_size), dtype=torch.float32)
        handles: List[Any] = []

        layers = self.model.model.layers
        try:
            for l_idx in range(num_layers):
                def _make_v_hook(li: int) -> Any:
                    def _vhook(mod: Any, inp: Any, out: torch.Tensor) -> None:
                        captured_v[li] = (
                            out[0, -1, :].detach().to("cpu", dtype=torch.float32)
                        )
                    return _vhook

                def _make_h_hook(li: int) -> Any:
                    def _hhook(mod: Any, inp: Any, out: Any) -> None:
                        h = out[0] if isinstance(out, tuple) else out
                        captured_h[li] = (
                            h[0, -1, :].detach().to("cpu", dtype=torch.float32)
                        )
                    return _hhook

                handles.append(
                    layers[l_idx].self_attn.v_proj.register_forward_hook(
                        _make_v_hook(l_idx)
                    )
                )
                handles.append(
                    layers[l_idx].register_forward_hook(_make_h_hook(l_idx))
                )

            enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
            input_ids = enc["input_ids"].to(device)
            with torch.inference_mode():
                self.model(input_ids=input_ids, use_cache=False)
            del input_ids
        finally:
            for h in handles:
                h.remove()

        return (captured_v, captured_h)

    @staticmethod
    def build_synthetic_calibration_for_tests(
        num_layers: int = 4,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        head_dim: int = 16,
        hidden_size: int = 64,
    ) -> MechanisticCalibrationBundle:
        """Explicitly labeled synthetic bundle used ONLY in fast unit tests without LLM weights."""
        d_kv = num_kv_heads * head_dim
        gen = torch.Generator(device="cpu").manual_seed(42)
        u_tr = F.normalize(
            torch.randn((num_layers, d_kv), generator=gen), p=2, dim=-1
        )
        u_th = F.normalize(
            torch.randn((num_layers, d_kv), generator=gen), p=2, dim=-1
        )
        u_pl = F.normalize(
            torch.randn((num_layers, d_kv), generator=gen), p=2, dim=-1
        )
        res_caa = F.normalize(
            torch.randn((num_layers, hidden_size), generator=gen), p=2, dim=-1
        )
        return MechanisticCalibrationBundle(
            model_id="synthetic-unit-test-qwen2",
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            u_trust=u_tr,
            u_threat=u_th,
            random_placebo_u=u_pl,
            residual_caa_vector=res_caa,
            probe_accuracy_trust=1.0,
            probe_accuracy_threat=1.0,
            calibrated_from_model=False,
            sink_token_ids=[0],
        )