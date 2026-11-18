from __future__ import annotations

import hashlib
import random
from typing import Dict
import numpy as np
import torch


class SeedBundle:
    """
    Deterministic Common Random Numbers (CRN) Manager (Resolves Defect D10).
    Splits a single `master_seed` into 6 cryptographically isolated PRNG streams
    so that ablating any component (e.g., LLM sampling or placebo directions)
    never perturbs the environment, population draw, partner matching, or crisis shock path.
    """

    STREAM_NAMES = (
        "env_rng",
        "pop_rng",
        "match_rng",
        "shock_rng",
        "llm_rng",
        "placebo_rng",
    )

    def __init__(self, master_seed: int = 42) -> None:
        self.master_seed = int(master_seed)
        self.stream_seeds: Dict[str, int] = {}
        self.py_rngs: Dict[str, random.Random] = {}
        self.np_rngs: Dict[str, np.random.Generator] = {}

        for s_name in self.STREAM_NAMES:
            derived = self._derive_sub_seed(self.master_seed, s_name)
            self.stream_seeds[s_name] = derived
            self.py_rngs[s_name] = random.Random(derived)
            self.np_rngs[s_name] = np.random.default_rng(derived)

    @staticmethod
    def _derive_sub_seed(master_seed: int, tag: str) -> int:
        payload = f"neuropolis_crn::{master_seed}::{tag}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        return int(digest[:8], 16) & 0x7FFFFFFF

    def get_py_rng(self, stream_name: str) -> random.Random:
        if stream_name not in self.py_rngs:
            raise KeyError(f"Unknown PRNG stream: {stream_name}")
        return self.py_rngs[stream_name]

    def get_np_rng(self, stream_name: str) -> np.random.Generator:
        if stream_name not in self.np_rngs:
            raise KeyError(f"Unknown PRNG stream: {stream_name}")
        return self.np_rngs[stream_name]

    def derive_turn_torch_generator(
        self,
        tick: int,
        agent_id: str,
        retry_idx: int = 0,
        device: str | torch.device = "cpu",
    ) -> torch.Generator:
        r"""
        Derives a reproducible per-call `torch.Generator` for stochastic LLM
        sampling (`temperature > 0`) and retry attempts $r \in \{0, 1, 2\}$.
        """
        base = self.stream_seeds["llm_rng"]
        sub_seed = self._derive_sub_seed(
            base, f"tick={tick}::agent={agent_id}::retry={retry_idx}"
        )
        gen = torch.Generator(device=device)
        gen.manual_seed(sub_seed)
        return gen

    def to_manifest_dict(self) -> Dict[str, int]:
        return {"master_seed": self.master_seed, **self.stream_seeds}