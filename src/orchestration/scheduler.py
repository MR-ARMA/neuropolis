"""
src/orchestration/scheduler.py
MODULE 10 (M10): Graph-Coloring Conflict-Free Batcher & AIMD Hardware Profiler.
Implements:
  1. `ConflictFreeBatcher`: Partitions directed interaction requests into mutually
     disjoint matchings `M_1, ..., M_K` via greedy Vizing edge-coloring so no agent
     ever participates in two simultaneous transactions within the same wave.
  2. `AIMDConcurrencyController`: Probes free GPU VRAM (`torch.cuda`) and system RAM
     (`psutil`) to dynamically adjust batch concurrency `C_current`.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple
import psutil
import torch

from src.core.contracts import HardwareBackendSpec


class ConflictFreeBatcher:
    """
    Partitions a list of directed dyadic interaction requests `(initiator_id, responder_id)`
    into disjoint matchings `M_1, M_2, ..., M_K` where every agent appears at most once per
    matching round.
    """

    @staticmethod
    def compute_disjoint_matchings(
        directed_requests: List[Tuple[str, str]],
    ) -> List[List[Tuple[str, str]]]:
        """
        Produces conflict-free edge colorings. Deduplicates unordered agent pairs per tick
        so a pair `(A, B)` executes one clean Two-Wave negotiation per tick.
        """
        seen_undirected: Set[Tuple[str, str]] = set()
        unique_edges: List[Tuple[str, str]] = []

        for u, v in directed_requests:
            if u == v or not u or not v:
                continue
            canon = (u, v) if u < v else (v, u)
            if canon not in seen_undirected:
                seen_undirected.add(canon)
                unique_edges.append((u, v))

        matchings: List[List[Tuple[str, str]]] = []
        busy_agents_per_color: List[Set[str]] = []

        for u, v in unique_edges:
            placed = False
            for color_idx, matching in enumerate(matchings):
                busy = busy_agents_per_color[color_idx]
                if u not in busy and v not in busy:
                    matching.append((u, v))
                    busy.add(u)
                    busy.add(v)
                    placed = True
                    break
            if not placed:
                matchings.append([(u, v)])
                busy_agents_per_color.append({u, v})

        return matchings


class AIMDConcurrencyController:
    """
    Additive-Increase / Multiplicative-Decrease (AIMD) Hardware Concurrency Gate.
    """

    def __init__(self, spec: HardwareBackendSpec) -> None:
        self.spec = spec
        self.current_concurrency: int = max(1, spec.batch_size)
        self.max_concurrency: int = max(1, spec.batch_size * 4)

    def probe_hardware_and_adjust(self) -> Dict[str, float]:
        ram_info = psutil.virtual_memory()
        ram_used_gb = (ram_info.total - ram_info.available) / (1024**3)
        ram_pct = ram_info.percent

        vram_allocated_gb = 0.0
        if torch.cuda.is_available():
            vram_allocated_gb = torch.cuda.memory_allocated() / (1024**3)

        # Multiplicative-Decrease if approaching 4GB VRAM ceiling
        if vram_allocated_gb > self.spec.vram_ceiling_gb * 0.90 or ram_pct > 90.0:
            self.current_concurrency = max(1, self.current_concurrency // 2)
        elif (
            vram_allocated_gb < self.spec.vram_ceiling_gb * 0.65
            and self.current_concurrency < self.spec.batch_size
        ):
            self.current_concurrency += 1

        return {
            "vram_allocated_gb": round(vram_allocated_gb, 3),
            "system_ram_used_gb": round(ram_used_gb, 2),
            "active_concurrency": float(self.current_concurrency),
        }