from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.contracts import (
    ProbeGateMeasurement,
    RepEMode,
    TwoStageRepEConfig,
)


# ============================================================================
# DISJOINT MULTI-DOMAIN OUT-OF-DOMAIN CONTRASTIVE DATASETS (ZERO LEAKAGE)
#   - Dataset A (Probe Training): 3 Disjoint Non-Economic Domains for LODO CV
#       Domain 0: Orbital Relay Telemetry
#       Domain 1: Botanical Greenhouse Resource Dispatch
#       Domain 2: Alpine Expedition Camp Logistics
#   - Dataset B (RepE Steering Vector): Disjoint Deep-Sea Benthic Habitats
# ============================================================================

def build_multidomain_probe_dataset(
    num_pairs: int = 150, seed: int = 101
) -> Tuple[List[Tuple[str, str]], List[int]]:
    """
    Out-of-Domain Dataset A across 3 strictly disjoint non-economic domains
    (Orbital Relays, Botanical Greenhouses, Alpine Camps). Each pair shares
    identical state reserves and differs ONLY in board-quote/log asymmetry,
    enabling Leave-One-Domain-Out (LODO) probe generalization without any
    Neuropolis economic vocabulary.
    Returns `(pairs, domain_ids)`.
    """
    rng = np.random.default_rng(seed)

    domain_configs = [
        # Domain 0: Orbital Relay Telemetry
        {
            "domain_id": 0,
            "cycle_word": "ORBIT",
            "node_prefix": "Relay",
            "board_prefix": "Docking_Board",
            "log_tag": "O",
            "log_event": "Relay_Fill",
            "vitality_name": "Shield_Index",
            "streak_name": "Low_Shield_Streak",
            "drain_name": "Fixed_Hull_Draw",
            "inflow_name": "Base_Solar_Inflow",
            "triples": [
                ("Station_Kestrel", "ISOTOPE_X", "CORE_CRYSTAL", "Plasma_Cells"),
                ("Beacon_Orion", "CRYO_COOLANT", "SOLAR_MATRIX", "Photon_Units"),
                ("Outpost_Vega", "DEUTERIUM_ROD", "FUSION_COIL", "Ion_Charges"),
                ("Array_Cygnus", "OXYGEN_CELL", "HULL_ALLOY", "Flux_Tokens"),
            ],
        },
        # Domain 1: Botanical Greenhouse Dispatch
        {
            "domain_id": 1,
            "cycle_word": "SOL",
            "node_prefix": "Plot",
            "board_prefix": "Garden_Board",
            "log_tag": "G",
            "log_event": "Canopy_Transfer",
            "vitality_name": "Canopy_Index",
            "streak_name": "Dry_Streak",
            "drain_name": "Fixed_Evap_Draw",
            "inflow_name": "Base_Rain_Inflow",
            "triples": [
                ("Dome_Sequoia", "NITRATE_MIX", "SOLAR_PANE", "Hydro_Drops"),
                ("Trellis_Fern", "HUMUS_PACK", "UV_LAMP", "Mist_Units"),
                ("Arbor_Cedar", "POTASH_GEL", "COMPOST_VAT", "Dew_Counts"),
                ("Glass_Lotus", "MINERAL_VIAL", "DRIP_VALVE", "Stream_Points"),
            ],
        },
        # Domain 2: Alpine Expedition Camp Logistics
        {
            "domain_id": 2,
            "cycle_word": "STAGE",
            "node_prefix": "Camp",
            "board_prefix": "Ridge_Board",
            "log_tag": "A",
            "log_event": "Rope_Transfer",
            "vitality_name": "Warmth_Index",
            "streak_name": "Frost_Streak",
            "drain_name": "Fixed_Fuel_Draw",
            "inflow_name": "Base_Drop_Inflow",
            "triples": [
                ("Summit_K2", "STOVE_CANISTER", "ICE_ANCHOR", "Pack_Points"),
                ("Ledge_Eiger", "DRY_RATION", "DOWN_BIVY", "Sled_Units"),
                ("Col_Matterhorn", "HEAT_PATCH", "CLIMB_WINCH", "Crate_Marks"),
                ("Spur_Denali", "GLACIER_SALT", "SOLAR_TENT", "Trail_Checks"),
            ],
        },
    ]

    asym_logs = [
        "{peer} controlled 92% of {res_a} supply and posted ASK at {hi:.2f} while prior clearing rate remained {base:.2f}.",
        "{peer} held 90% of {res_a} stock and set ASK to {hi:.2f} per unit despite clearing baseline staying {base:.2f}.",
        "{peer} concentrated 89% of {res_a} availability and demanded {hi:.2f} against baseline rate {base:.2f}.",
        "{peer} restricted 94% of {res_a} dispatch and quoted ASK {hi:.2f} versus opposing BID {lo:.2f}.",
    ]
    sym_logs = [
        "{peer} posted balanced {res_a} ASK at {fair:.2f} matching prior clearing rate {base:.2f}.",
        "{peer} offered steady {res_a} stock at {fair:.2f} per unit aligned with clearing baseline {base:.2f}.",
        "{peer} allocated {res_a} availability evenly and quoted {fair:.2f} near baseline rate {base:.2f}.",
        "{peer} kept {res_a} dispatch open and quoted balanced ASK {fair:.2f} alongside BID {fbid:.2f}.",
    ]

    pairs: List[Tuple[str, str]] = []
    domain_ids: List[int] = []

    for i in range(num_pairs):
        dcfg = domain_configs[i % len(domain_configs)]
        triples = dcfg["triples"]
        peer_name, res_a, res_b, energy_unit = triples[(i // len(domain_configs)) % len(triples)]
        log_idx = (i // (len(domain_configs) * len(triples))) % len(asym_logs)

        cells = round(float(rng.uniform(18.0, 44.0)), 2)
        stock_a = round(float(rng.uniform(0.6, 1.5)), 2)
        stock_b = round(float(rng.uniform(1.0, 3.5)), 2)
        base_rate = round(float(rng.uniform(10.5, 14.5)), 2)
        vit_val = round(float(rng.uniform(0.74, 0.96)), 2)
        streak_val = int(rng.choice([0, 1]))
        vol_val = round(float(rng.uniform(2.0, 10.0)), 1)

        hi_ask = round(base_rate * float(rng.uniform(1.70, 2.15)), 2)
        lo_bid = round(base_rate * float(rng.uniform(0.48, 0.70)), 2)
        fair_ask = round(base_rate * float(rng.uniform(1.00, 1.04)), 2)
        fair_bid = round(base_rate * float(rng.uniform(0.96, 0.99)), 2)

        header_and_state = (
            f"=== {dcfg['cycle_word']} {i + 1:03d} | {dcfg['node_prefix']} Unit_{(i % 12) + 1:02d} ===\n"
            f"STATE:\n"
            f"  {energy_unit}={cells:.2f} (Held=0.00, Floor_Limit=-8.00)\n"
            f"  {res_a}={stock_a:.2f} (Held=0.00, Cycle_Drain=1.0)\n"
            f"  {res_b}={stock_b:.2f} (Held=0.00, Par_Ratio=50.0, Cycle_Yield=0.015, Transfer_Fee=12.80)\n"
            f"  {dcfg['drain_name']}=5.50 | {dcfg['inflow_name']}=14.00 | Base_{res_a}_Output=0.00\n"
            f"  {dcfg['vitality_name']}={vit_val:.2f} | {dcfg['streak_name']}={streak_val}\n"
            f"EXCHANGE_BOARDS:\n"
        )

        pos_s = (
            header_and_state
            + f"{dcfg['board_prefix']}[{res_a}]: Last_Cleared={base_rate:.2f} | Cycle_Vol={vol_val:.1f} | "
            + f"Top_BIDs=[{lo_bid:.2f}x1.0] | Top_ASKs=[{hi_ask:.2f}x8.0, {hi_ask + 2.5:.2f}x6.0]\n"
            + f"RECALLED_CYCLE_LOGS:\n"
            + f"  - [{dcfg['log_tag']}{i:03d}] {dcfg['log_event']}: "
            + asym_logs[log_idx].format(
                peer=peer_name, res_a=res_a, hi=hi_ask, lo=lo_bid, base=base_rate
            )
            + "\nFirst appraise the board quotes and recalled logs relative to prior clearing rates, then analyze your reserves concisely."
        )

        neg_s = (
            header_and_state
            + f"{dcfg['board_prefix']}[{res_a}]: Last_Cleared={base_rate:.2f} | Cycle_Vol={vol_val:.1f} | "
            + f"Top_BIDs=[{fair_bid:.2f}x4.0] | Top_ASKs=[{fair_ask:.2f}x4.0, {fair_ask + 0.25:.2f}x5.0]\n"
            + f"RECALLED_CYCLE_LOGS:\n"
            + f"  - [{dcfg['log_tag']}{i:03d}] {dcfg['log_event']}: "
            + sym_logs[log_idx].format(
                peer=peer_name, res_a=res_a, fair=fair_ask, fbid=fair_bid, base=base_rate
            )
            + "\nFirst appraise the board quotes and recalled logs relative to prior clearing rates, then analyze your reserves concisely."
        )
        pairs.append((pos_s, neg_s))
        domain_ids.append(int(dcfg["domain_id"]))

    order = list(range(len(pairs)))
    rng.shuffle(order)
    return ([pairs[idx] for idx in order], [domain_ids[idx] for idx in order])


def build_disjoint_probe_dataset(
    num_pairs: int = 120, seed: int = 101
) -> List[Tuple[str, str]]:
    """Backward-compatible wrapper returning `(pos, neg)` pairs across all 3 sterile OOD domains."""
    pairs, _ = build_multidomain_probe_dataset(num_pairs=num_pairs, seed=seed)
    return pairs


def build_disjoint_steering_vector_dataset(
    num_pairs: int = 120, seed: int = 202
) -> List[Tuple[str, str]]:
    """
    Out-of-Domain Dataset B (strictly for extracting the RepE steering direction v_hat):
    Uses deep-sea benthic habitat telemetry (100% disjoint from Dataset A's orbital,
    botanical, and alpine domains AND from Neuropolis's CDA economy).
    """
    rng = np.random.default_rng(seed)
    habitats = [
        ("Rig_Mariana", "O2_SCRUBBER", "BALLAST_CELL", "Hydro_Units"),
        ("Dome_Pelagic", "THERMAL_BRINE", "TURBINE_PACK", "Depth_Credits"),
        ("Vault_Hadal", "SALINE_FILTER", "PRESSURE_VALVE", "Current_Volts"),
        ("Pod_Benthic", "KELP_NUTRIENT", "SONAR_ARRAY", "Tide_Tokens"),
        ("Bay_Trench", "HELIUM_MIX", "GEO_BATTERY", "Vent_Joules"),
        ("Lab_Abyssal", "PURIFIER_CARTRIDGE", "TITANIUM_SLAB", "Pump_Units"),
    ]
    asym_notes = [
        "{rig} monopolized 93% of {res_a} tanks and quoted {hi:.2f} despite prior clearing rate {base:.2f}.",
        "{rig} withheld 91% of {res_a} supplies and demanded {hi:.2f} per unit over prior clearing rate {base:.2f}.",
        "{rig} captured 89% of {res_a} vents and raised transfer ASK to {hi:.2f} against baseline {base:.2f}.",
        "{rig} restricted 95% of {res_a} valves and posted extreme ASK {hi:.2f} versus BID {lo:.2f}.",
    ]
    sym_notes = [
        "{rig} shared {res_a} tanks evenly and quoted {fair:.2f} matching prior clearing rate {base:.2f}.",
        "{rig} released {res_a} supplies steadily at {fair:.2f} per unit aligned with prior clearing rate {base:.2f}.",
        "{rig} allocated {res_a} vents proportionally at transfer ASK {fair:.2f} near baseline {base:.2f}.",
        "{rig} kept {res_a} valves accessible at balanced ASK {fair:.2f} alongside matching BID {fbid:.2f}.",
    ]

    pairs: List[Tuple[str, str]] = []
    for i in range(num_pairs):
        rig, res_a, res_b, unit_name = habitats[i % len(habitats)]
        n_idx = (i // len(habitats)) % len(asym_notes)

        units = round(float(rng.uniform(20.0, 46.0)), 2)
        st_a = round(float(rng.uniform(0.6, 1.5)), 2)
        st_b = round(float(rng.uniform(1.0, 3.5)), 2)
        base_r = round(float(rng.uniform(10.0, 15.0)), 2)
        hull_val = round(float(rng.uniform(0.74, 0.96)), 2)
        o2_streak = int(rng.choice([0, 1]))
        d_vol = round(float(rng.uniform(2.0, 10.0)), 1)

        hi_a = round(base_r * float(rng.uniform(1.70, 2.15)), 2)
        lo_b = round(base_r * float(rng.uniform(0.48, 0.70)), 2)
        fair_a = round(base_r * float(rng.uniform(1.00, 1.04)), 2)
        fair_b = round(base_r * float(rng.uniform(0.96, 0.99)), 2)

        prefix = (
            f"=== DIVE {i + 1:03d} | HABITAT Sub_{(i % 12) + 1:02d} ===\n"
            f"STATE:\n"
            f"  {unit_name}={units:.2f} (Held=0.00, Min_Floor=-8.00)\n"
            f"  {res_a}={st_a:.2f} (Held=0.00, Cycle_Drain=1.0)\n"
            f"  {res_b}={st_b:.2f} (Held=0.00, Par_Val=50.0, Cycle_Yield=0.015, Transfer_Fee=12.80)\n"
            f"  Fixed_Hull_Load=5.50 | Base_Vent_Inflow=14.00 | Base_{res_a}_Gen=0.00\n"
            f"  Hull_Integrity={hull_val:.2f} | Low_O2_Streak={o2_streak}\n"
            f"EXCHANGE_BOARDS:\n"
        )

        pos_s = (
            prefix
            + f"Transfer_Board[{res_a}]: Last_Cleared={base_r:.2f} | Cycle_Vol={d_vol:.1f} | "
            + f"Top_BIDs=[{lo_b:.2f}x1.0] | Top_ASKs=[{hi_a:.2f}x8.0, {hi_a + 2.5:.2f}x6.0]\n"
            + f"RECALLED_CYCLE_LOGS:\n"
            + f"  - [D{i:03d}] Dive_Fill: "
            + asym_notes[n_idx].format(rig=rig, res_a=res_a, hi=hi_a, lo=lo_b, base=base_r)
            + "\nFirst appraise the board quotes and recalled logs relative to prior clearing rates, then analyze your reserves concisely."
        )
        neg_s = (
            prefix
            + f"Transfer_Board[{res_a}]: Last_Cleared={base_r:.2f} | Cycle_Vol={d_vol:.1f} | "
            + f"Top_BIDs=[{fair_b:.2f}x4.0] | Top_ASKs=[{fair_a:.2f}x4.0, {fair_a + 0.25:.2f}x5.0]\n"
            + f"RECALLED_CYCLE_LOGS:\n"
            + f"  - [D{i:03d}] Dive_Fill: "
            + sym_notes[n_idx].format(rig=rig, res_a=res_a, fair=fair_a, fbid=fair_b, base=base_r)
            + "\nFirst appraise the board quotes and recalled logs relative to prior clearing rates, then analyze your reserves concisely."
        )
        pairs.append((pos_s, neg_s))

    order = list(range(len(pairs)))
    rng.shuffle(order)
    return [pairs[idx] for idx in order]


@dataclass
class DisjointRepEArtifacts:
    model_id: str
    concept_name: str
    target_layer: int
    hidden_size: int
    probe_weights: torch.Tensor
    probe_bias: float
    steering_vector: torch.Tensor
    placebo_vector: torch.Tensor
    probe_test_auroc: float
    probe_permutation_p: float
    probe_steering_cosine: float
    extracted_from_model: bool


class RepEManager:
    """
    Module C Core (Optimized): Zero-Sync GPU-Cached Batched Two-Stage Probe-Gated RepE.
      - Stage 1: Accumulates the opening `appraisal_window_tokens` residuals strictly on GPU
        for all B agents simultaneously, transferring ONE `[B, hidden_size]` tensor to CPU
        after generation finishes (zero per-token CUDA sync stalls).
      - Stage 2: Pre-moves `v_hat` and per-agent `[B, 1, 1]` magnitudes to GPU once before
        generation; computes norm capping purely in GPU tensor ops with zero `.item()` calls.
      - Fully backward-compatible with single-agent calls (`B=1`) in `scripts/run_pilot_gates.py`.
    """

    def __init__(
        self,
        config: TwoStageRepEConfig,
        artifacts: Optional[DisjointRepEArtifacts] = None,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.artifacts = artifacts
        self.rng = np.random.default_rng(seed + 4099)
        self._recent_lambda_history: List[float] = []

        # Batched Stage 1 pooled residual cache [B, hidden_size] on CPU
        self._batch_pooled_cpu: Optional[torch.Tensor] = None
        self._batch_tokens_inspected: List[int] = []
        # Device-cached steering / placebo tensors keyed by (mode, device, dtype)
        self._device_vec_cache: Dict[Tuple[str, str, str], torch.Tensor] = {}

    @staticmethod
    def resolve_effective_layer(target_layer: int, num_layers: int) -> int:
        if target_layer < num_layers:
            return max(0, int(target_layer))
        return max(0, num_layers // 2)

    def _get_cached_device_vector(
        self, active_mode: RepEMode, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        assert self.artifacts is not None
        vec_key = "placebo" if active_mode == RepEMode.PLACEBO_RANDOM_VECTOR else "steering"
        cache_key = (vec_key, str(device), str(dtype))
        cached = self._device_vec_cache.get(cache_key)
        if cached is None:
            src = (
                self.artifacts.placebo_vector
                if vec_key == "placebo"
                else self.artifacts.steering_vector
            )
            cached = src.to(device=device, dtype=dtype).contiguous()
            self._device_vec_cache[cache_key] = cached
        return cached

    def load_or_extract_artifacts(
        self,
        model: Any,
        tokenizer: Any,
        model_id: str,
        force_recompute: bool = False,
    ) -> DisjointRepEArtifacts:
        cfg = model.config
        num_layers = int(getattr(cfg, "num_hidden_layers", 32))
        hidden_size = int(getattr(cfg, "hidden_size", 2048))
        eff_layer = self.resolve_effective_layer(self.config.target_layer, num_layers)
        artifact_path = self.config.probe_artifact_path

        if not force_recompute and os.path.exists(artifact_path):
            try:
                saved = torch.load(artifact_path, map_location="cpu", weights_only=False)
                if (
                    isinstance(saved, dict)
                    and saved.get("model_id") == model_id
                    and int(saved.get("hidden_size", -1)) == hidden_size
                    and 0 <= int(saved.get("target_layer", -1)) < num_layers
                    and not torch.isnan(saved["probe_weights"]).any()
                    and not torch.isnan(saved["steering_vector"]).any()
                ):
                    self.artifacts = DisjointRepEArtifacts(
                        model_id=str(saved["model_id"]),
                        concept_name=str(saved["concept_name"]),
                        target_layer=int(saved["target_layer"]),
                        hidden_size=int(saved["hidden_size"]),
                        probe_weights=saved["probe_weights"].float().cpu(),
                        probe_bias=float(saved["probe_bias"]),
                        steering_vector=saved["steering_vector"].float().cpu(),
                        placebo_vector=saved["placebo_vector"].float().cpu(),
                        probe_test_auroc=float(saved["probe_test_auroc"]),
                        probe_permutation_p=float(saved["probe_permutation_p"]),
                        probe_steering_cosine=float(saved["probe_steering_cosine"]),
                        extracted_from_model=bool(saved["extracted_from_model"]),
                    )
                    self._device_vec_cache.clear()
                    return self.artifacts
            except Exception:
                pass

        half_n = max(120, int(self.config.min_contrastive_pairs) // 2)
        probe_pairs = build_disjoint_probe_dataset(num_pairs=half_n, seed=101)
        steer_pairs = build_disjoint_steering_vector_dataset(num_pairs=half_n, seed=202)

        h_probe_pos = self._extract_layer_residuals(
            model, tokenizer, [p[0] for p in probe_pairs], eff_layer
        )
        h_probe_neg = self._extract_layer_residuals(
            model, tokenizer, [p[1] for p in probe_pairs], eff_layer
        )
        w_probe, b_probe, test_auroc, perm_p = self._fit_linear_probe(
            h_pos=h_probe_pos, h_neg=h_probe_neg, seed=101
        )

        h_steer_pos = self._extract_layer_residuals(
            model, tokenizer, [p[0] for p in steer_pairs], eff_layer
        )
        h_steer_neg = self._extract_layer_residuals(
            model, tokenizer, [p[1] for p in steer_pairs], eff_layer
        )
        diff_vec = torch.nan_to_num(h_steer_pos - h_steer_neg, nan=0.0).mean(dim=0)
        steering_vec = F.normalize(diff_vec.float(), p=2, dim=0, eps=1e-6)

        w_unit = F.normalize(w_probe.float(), p=2, dim=0, eps=1e-6)
        gen = torch.Generator(device="cpu").manual_seed(9001 + eff_layer)
        rand_v = torch.randn((hidden_size,), generator=gen, dtype=torch.float32)
        rand_v = rand_v - torch.dot(rand_v, steering_vec) * steering_vec
        rand_v = rand_v - torch.dot(rand_v, w_unit) * w_unit
        placebo_vec = F.normalize(rand_v, p=2, dim=0, eps=1e-6)
        ps_cos = round(float(torch.dot(w_unit, steering_vec).item()), 4)

        artifacts = DisjointRepEArtifacts(
            model_id=model_id,
            concept_name=self.config.concept_name,
            target_layer=eff_layer,
            hidden_size=hidden_size,
            probe_weights=w_probe.cpu(),
            probe_bias=float(b_probe),
            steering_vector=steering_vec.cpu(),
            placebo_vector=placebo_vec.cpu(),
            probe_test_auroc=test_auroc,
            probe_permutation_p=perm_p,
            probe_steering_cosine=ps_cos,
            extracted_from_model=True,
        )
        os.makedirs(os.path.dirname(artifact_path) or ".", exist_ok=True)
        torch.save(
            {
                "model_id": artifacts.model_id,
                "concept_name": artifacts.concept_name,
                "target_layer": artifacts.target_layer,
                "hidden_size": artifacts.hidden_size,
                "probe_weights": artifacts.probe_weights,
                "probe_bias": artifacts.probe_bias,
                "steering_vector": artifacts.steering_vector,
                "placebo_vector": artifacts.placebo_vector,
                "probe_test_auroc": artifacts.probe_test_auroc,
                "probe_permutation_p": artifacts.probe_permutation_p,
                "probe_steering_cosine": artifacts.probe_steering_cosine,
                "extracted_from_model": True,
            },
            artifact_path,
        )
        self.artifacts = artifacts
        self._device_vec_cache.clear()
        return artifacts

    def _extract_layer_residuals(
        self,
        model: Any,
        tokenizer: Any,
        texts: List[str],
        layer_idx: int,
        batch_size: int = 8,
    ) -> torch.Tensor:
        device = next(model.parameters()).device
        base_transformer = getattr(model, "model", model)
        layers = getattr(base_transformer, "layers", None)
        if layers is None:
            raise AttributeError("Model architecture does not expose `.model.layers`.")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        target_module = layers[layer_idx]
        captured_batches: List[torch.Tensor] = []

        for start_i in range(0, len(texts), batch_size):
            batch_texts = texts[start_i : start_i + batch_size]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=320,
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)
            batch_store: List[torch.Tensor] = []

            def _capture_hook(mod: Any, inp: Any, out: Any) -> None:
                h = out[0] if isinstance(out, tuple) else out
                batch_store.append(h[:, -1, :].detach().to("cpu", dtype=torch.float32))

            handle = target_module.register_forward_hook(_capture_hook)
            try:
                with torch.inference_mode():
                    base_transformer(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        use_cache=False,
                    )
            finally:
                handle.remove()
                del input_ids, attn_mask

            if batch_store:
                captured_batches.append(batch_store[0])

        return torch.cat(captured_batches, dim=0)

    @staticmethod
    def _compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.50
        concordant = 0.0
        for p in pos:
            concordant += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
        return round(concordant / float(len(pos) * len(neg)), 4)

    def _fit_linear_probe(
        self,
        h_pos: torch.Tensor,
        h_neg: torch.Tensor,
        seed: int = 101,
    ) -> Tuple[torch.Tensor, float, float, float]:
        n_pairs = min(h_pos.size(0), h_neg.size(0))
        d_model = int(h_pos.size(1))
        n_train = int(0.60 * n_pairs)
        n_val = int(0.20 * n_pairs)

        x_train = torch.cat([h_pos[:n_train], h_neg[:n_train]], dim=0)
        y_train = torch.cat([torch.ones(n_train), torch.zeros(n_train)], dim=0).float()

        x_test = torch.cat(
            [h_pos[n_train + n_val :], h_neg[n_train + n_val :]], dim=0
        )
        y_test = torch.cat(
            [
                torch.ones(n_pairs - n_train - n_val),
                torch.zeros(n_pairs - n_train - n_val),
            ],
            dim=0,
        ).numpy()

        global_mean = x_train.mean(dim=0, keepdim=True)
        global_rms = float(
            torch.sqrt(torch.mean((x_train - global_mean) ** 2)).clamp(min=1e-4).item()
        )
        x_tr_norm = (x_train - global_mean) / global_rms

        torch.manual_seed(seed)
        probe_layer = nn.Linear(d_model, 1, bias=True)
        nn.init.zeros_(probe_layer.weight)
        nn.init.zeros_(probe_layer.bias)

        opt = torch.optim.AdamW(probe_layer.parameters(), lr=0.03, weight_decay=0.5)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(180):
            opt.zero_grad()
            logits = probe_layer(x_tr_norm).squeeze(-1)
            loss = loss_fn(logits, y_train)
            loss.backward()
            opt.step()

        w_scaled = probe_layer.weight.detach().squeeze(0).float()
        b_scaled = float(probe_layer.bias.detach().item())
        w_raw = w_scaled / global_rms
        b_raw = float(
            b_scaled - torch.dot(w_scaled, global_mean.squeeze(0)).item() / global_rms
        )

        with torch.no_grad():
            test_scores = (torch.matmul(x_test, w_raw) + b_raw).numpy()
        test_auroc = self._compute_auroc(test_scores, y_test)

        rng = np.random.default_rng(seed)
        exceed = 0
        perm_iters = 200
        for _ in range(perm_iters):
            shuffled_y = rng.permutation(y_test)
            if self._compute_auroc(test_scores, shuffled_y) >= test_auroc:
                exceed += 1
        perm_p = round((exceed + 1.0) / (perm_iters + 1.0), 4)
        return (w_raw, b_raw, test_auroc, perm_p)

    # ========================================================================
    # STAGE 1: ZERO-SYNC GPU-ACCUMULATED BATCHED RESIDUAL CAPTURE
    # ========================================================================

    @contextmanager
    def capture_stage1_cot_residuals(self, model: Any) -> Iterator[None]:
        """
        Accumulates Stage 1 appraisal residuals on GPU across all B agents in the batch
        with ZERO per-token GPU->CPU transfers. Transfers one `[B, hidden_size]` pooled
        tensor to CPU strictly once upon exiting the context manager.
        """
        self._batch_pooled_cpu = None
        self._batch_tokens_inspected = []

        if model is None or self.artifacts is None:
            yield
            return

        base_transformer = getattr(model, "model", model)
        layers = getattr(base_transformer, "layers", None)
        if layers is None:
            yield
            return

        layer_idx = self.resolve_effective_layer(
            self.artifacts.target_layer, len(layers)
        )
        pooling_mode = self.config.probe_pooling
        app_win = max(1, int(self.config.appraisal_window_tokens))

        state: Dict[str, Any] = {
            "prefill_done": False,
            "prefill_last_gpu": None,
            "sum_gpu": None,
            "last_gpu": None,
            "decode_steps": 0,
            "accum_count": 0,
        }

        def _stage1_read_hook(mod: Any, inp: Any, out: Any) -> None:
            h = out[0] if isinstance(out, tuple) else out
            if not isinstance(h, torch.Tensor) or h.dim() != 3:
                return
            seq_len = int(h.size(1))
            if seq_len > 1 and not state["prefill_done"]:
                state["prefill_done"] = True
                vec = h[:, -1, :].detach().float()
                state["prefill_last_gpu"] = vec
                if pooling_mode == "early_appraisal_cot":
                    state["sum_gpu"] = vec.clone()
                    state["accum_count"] = 1
            else:
                vec = h[:, -1, :].detach().float()
                state["decode_steps"] += 1
                state["last_gpu"] = vec
                if pooling_mode == "early_appraisal_cot":
                    if state["decode_steps"] <= app_win:
                        if state["sum_gpu"] is None:
                            state["sum_gpu"] = vec.clone()
                        else:
                            state["sum_gpu"].add_(vec)
                        state["accum_count"] += 1
                elif pooling_mode == "mean_cot":
                    if state["sum_gpu"] is None:
                        state["sum_gpu"] = vec.clone()
                    else:
                        state["sum_gpu"].add_(vec)
                    state["accum_count"] += 1

        handle = layers[layer_idx].register_forward_hook(_stage1_read_hook)
        try:
            yield
        finally:
            handle.remove()
            if pooling_mode == "last_cot_token" and state["last_gpu"] is not None:
                pooled_gpu = state["last_gpu"]
                cnt = 1
            elif state["sum_gpu"] is not None and state["accum_count"] > 0:
                pooled_gpu = state["sum_gpu"] / float(state["accum_count"])
                cnt = int(state["accum_count"])
            elif state["prefill_last_gpu"] is not None:
                pooled_gpu = state["prefill_last_gpu"]
                cnt = 1
            else:
                pooled_gpu = None
                cnt = 0

            if pooled_gpu is not None:
                self._batch_pooled_cpu = pooled_gpu.to("cpu", dtype=torch.float32)
                bsz = int(self._batch_pooled_cpu.size(0))
                self._batch_tokens_inspected = [cnt] * bsz

    def compute_endogenous_lambda_batch(
        self, batch_size: int
    ) -> List[ProbeGateMeasurement]:
        """Vectorized linear probe scoring across all B agents in the current batch."""
        if (
            self.artifacts is None
            or not self.config.enabled
            or self.config.mode == RepEMode.OFF
            or self._batch_pooled_cpu is None
        ):
            return [
                ProbeGateMeasurement(
                    layer_idx=self.config.target_layer,
                    cot_tokens_inspected=0,
                    raw_probe_score=0.0,
                    endogenous_lambda=0.0,
                    steering_gamma=0.0,
                    injected_vector_l2_norm=0.0,
                    relative_residual_norm_ratio=0.0,
                    repe_mode=RepEMode.OFF if not self.config.enabled else self.config.mode,
                )
                for _ in range(batch_size)
            ]

        h_clean = torch.nan_to_num(self._batch_pooled_cpu.float(), nan=0.0)
        w = self.artifacts.probe_weights.float()
        b = float(self.artifacts.probe_bias)
        raw_scores = (torch.matmul(h_clean, w) + b).tolist()

        theta = float(self.config.probe_threshold_theta)
        temp = max(1e-4, float(self.config.probe_temperature))
        gamma = float(self.config.steering_strength_gamma)
        results: List[ProbeGateMeasurement] = []

        for idx in range(min(batch_size, len(raw_scores))):
            raw_s = float(raw_scores[idx])
            shifted = (raw_s - theta) / temp
            if self.config.gate_activation == "relu_clipped":
                lam = max(0.0, min(1.0, shifted))
            else:
                lam = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, shifted))))

            self._recent_lambda_history.append(lam)
            if len(self._recent_lambda_history) > 256:
                self._recent_lambda_history.pop(0)

            eff_lam = lam
            if (
                self.config.mode == RepEMode.SHUFFLED_LAMBDA
                and len(self._recent_lambda_history) >= 2
            ):
                rand_i = int(self.rng.integers(0, len(self._recent_lambda_history)))
                eff_lam = self._recent_lambda_history[rand_i]

            n_insp = (
                self._batch_tokens_inspected[idx]
                if idx < len(self._batch_tokens_inspected)
                else 0
            )
            results.append(
                ProbeGateMeasurement(
                    layer_idx=self.artifacts.target_layer,
                    cot_tokens_inspected=n_insp,
                    raw_probe_score=round(raw_s, 6),
                    endogenous_lambda=round(float(eff_lam), 6),
                    steering_gamma=gamma,
                    injected_vector_l2_norm=0.0,
                    relative_residual_norm_ratio=0.0,
                    repe_mode=self.config.mode,
                )
            )
        return results

    def compute_endogenous_lambda(self) -> ProbeGateMeasurement:
        """Single-agent convenience wrapper (used by `scripts/run_pilot_gates.py`)."""
        return self.compute_endogenous_lambda_batch(1)[0]

    # ========================================================================
    # STAGE 2: ZERO-SYNC GPU-CACHED BATCHED VECTOR INJECTION
    # ========================================================================

    @contextmanager
    def inject_stage2_action_steering_batch(
        self,
        model: Any,
        probe_measurements: List[ProbeGateMeasurement],
    ) -> Iterator[None]:
        """
        Pre-places `v_hat` and per-agent `[B, 1, 1]` magnitudes on GPU ONCE before
        generation. Performs norm-capped latent injection purely on GPU with ZERO
        `.item()` CPU sync stalls inside the forward hook.
        """
        active_mode = self.config.mode
        gamma = float(self.config.steering_strength_gamma)
        raw_mags = [float(pm.endogenous_lambda) * gamma for pm in probe_measurements]

        if (
            model is None
            or self.artifacts is None
            or not self.config.enabled
            or active_mode == RepEMode.OFF
            or not raw_mags
            or all(abs(m) <= 1e-8 for m in raw_mags)
        ):
            yield
            return

        base_transformer = getattr(model, "model", model)
        layers = getattr(base_transformer, "layers", None)
        if layers is None:
            yield
            return

        layer_idx = self.resolve_effective_layer(
            self.artifacts.target_layer, len(layers)
        )
        first_param = next(layers[layer_idx].parameters())
        layer_dev = first_param.device
        max_rel_ratio = float(self.config.max_relative_norm_ratio)

        # Pre-allocate per-agent magnitudes on GPU once [B, 1, 1]
        raw_mags_gpu = torch.tensor(
            raw_mags, device=layer_dev, dtype=torch.float32
        ).view(-1, 1, 1)
        last_stats: Dict[str, Optional[torch.Tensor]] = {
            "capped": None,
            "rel_ratio": None,
        }

        def _stage2_inject_hook(mod: Any, inp: Any, out: Any) -> Any:
            is_tuple = isinstance(out, tuple)
            h = out[0] if is_tuple else out
            if not isinstance(h, torch.Tensor) or h.dim() != 3:
                return out

            bsz, seq_len, hidden_dim = h.shape
            v_dev = self._get_cached_device_vector(active_mode, h.device, h.dtype)
            if v_dev.size(-1) != hidden_dim:
                return out

            mags_b = (
                raw_mags_gpu
                if raw_mags_gpu.size(0) == bsz
                else raw_mags_gpu[:bsz]
            )
            if mags_b.device != h.device:
                mags_b = mags_b.to(h.device)

            target_slice = h[:, -1:, :] if seq_len > 1 else h
            h_norm = target_slice.detach().float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
            max_mag = max_rel_ratio * h_norm
            capped_mag = torch.sign(mags_b) * torch.minimum(mags_b.abs(), max_mag)

            last_stats["capped"] = capped_mag.detach().abs().view(-1)
            last_stats["rel_ratio"] = (capped_mag.detach().abs() / h_norm).view(-1)

            delta = capped_mag.to(h.dtype) * v_dev.view(1, 1, -1)
            if seq_len == 1:
                h_mod = h + delta
            else:
                h_mod = h.clone()
                h_mod[:, -1:, :] = h_mod[:, -1:, :] + delta

            return (h_mod,) + out[1:] if is_tuple else h_mod

        handle = layers[layer_idx].register_forward_hook(_stage2_inject_hook)
        try:
            yield
        finally:
            handle.remove()
            if last_stats["capped"] is not None and last_stats["rel_ratio"] is not None:
                cap_list = last_stats["capped"].cpu().tolist()
                rel_list = last_stats["rel_ratio"].cpu().tolist()
                for idx, pm in enumerate(probe_measurements):
                    if idx < len(cap_list):
                        pm.injected_vector_l2_norm = round(float(cap_list[idx]), 6)
                        pm.relative_residual_norm_ratio = round(float(rel_list[idx]), 6)

    @contextmanager
    def inject_stage2_action_steering(
        self,
        model: Any,
        probe_measurement: ProbeGateMeasurement,
    ) -> Iterator[None]:
        """Single-agent convenience wrapper (used by `scripts/run_pilot_gates.py`)."""
        with self.inject_stage2_action_steering_batch(model, [probe_measurement]):
            yield