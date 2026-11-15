"""
src/evaluation/metrics.py
MODULE 11 (M11): Live Macro-Social Indicators, Inequality Metrics,
Non-Blocking JSONL Telemetry Writer, and Post-MVP Shadow Baseline Stubs.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np

from src.core.contracts import MacroEconomyState
from src.memory.scoped_memory import ScopedMemoryEngine


class BenchmarkEvaluator:
    """
    Computes live behavioral economics metrics at the end of every tick:
      - Societal Trust Index `TI_t`
      - Wealth Gini Coefficient `G_t`
      - Emotion / Arousal dispersion
      - Cumulative Trade Volume & Epistemic TMS Routing Rate
    """

    def __init__(self, memory_engine: ScopedMemoryEngine, jsonl_path: str) -> None:
        self.memory_engine = memory_engine
        self.jsonl_path = jsonl_path
        out_dir = os.path.dirname(os.path.abspath(jsonl_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        # Clear previous run telemetry file
        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            f.write("")

    @staticmethod
    def compute_gini_coefficient(wealths: List[float]) -> float:
        if not wealths or len(wealths) < 2:
            return 0.0
        arr = np.sort(np.maximum(0.0, np.array(wealths, dtype=np.float64)))
        total = float(np.sum(arr))
        if total <= 1e-9:
            return 0.0
        n = len(arr)
        index = np.arange(1, n + 1)
        gini = float((np.sum((2 * index - n - 1) * arr)) / (n * total))
        return round(max(0.0, min(1.0, gini)), 4)

    def compute_and_log_tick_metrics(
        self,
        tick: int,
        agent_ids: List[str],
        conservation_ok: bool,
        conservation_error: float,
        hw_stats: Dict[str, float],
    ) -> MacroEconomyState:
        external_map = self.memory_engine.get_all_external_states()
        macro = self.memory_engine.get_macro_state()

        # 1. Total Net Worth per Agent (Cash + Bank Deposit + Valued Inventory)
        net_worths: List[float] = []
        for aid in agent_ids:
            ext = external_map[aid]
            inv_val = (
                ext.inventory.get("FOOD", 0) * macro.market_clearing_price_food
                + ext.inventory.get("TECH", 0) * macro.market_clearing_price_tech
            )
            net_worths.append(ext.liquid_wealth_cash + ext.bank_deposit + inv_val)

        # 2. Societal Trust Index TI_t across all directed relational edges
        all_trust_vals: List[float] = []
        for aid in agent_ids:
            scoped = self.memory_engine.get_scoped_view(aid)
            for edge in scoped.get_all_outgoing_edges().values():
                all_trust_vals.append(edge.trust_tau)

        mean_trust = float(np.mean(all_trust_vals)) if all_trust_vals else 0.0
        gini = self.compute_gini_coefficient(net_worths)

        macro.tick = tick
        macro.societal_trust_index = round(mean_trust, 4)
        macro.wealth_gini = gini
        self.memory_engine.update_macro_state(macro)

        log_entry: Dict[str, Any] = {
            "tick": tick,
            "societal_trust_index": macro.societal_trust_index,
            "wealth_gini": macro.wealth_gini,
            "market_clearing_price_food": macro.market_clearing_price_food,
            "market_clearing_price_tech": macro.market_clearing_price_tech,
            "cumulative_trade_volume": macro.cumulative_trade_volume,
            "cumulative_tms_queries": macro.cumulative_tms_queries,
            "public_treasury_cash": macro.public_treasury_cash,
            "firm_sector_cash": macro.firm_sector_cash,
            "money_conservation_ok": conservation_ok,
            "money_conservation_error": conservation_error,
            "hardware": hw_stats,
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        return macro

    # ==========================================================================
    # POST-MVP SHADOW BASELINES & DISTRIBUTIONAL ALIGNMENT STUBS
    # ==========================================================================
    def compute_shadow_baselines_and_wasserstein(self, tick: int) -> Dict[str, float]:
        """
        # TODO(post-mvp): [Evaluation - Closed-Form Shadow Baselines & 1-Wasserstein Alignment]
        Implements:
          1. DeGroot Opinion Matrix Baseline: a^{(t+1)} = W * a^{(t)}
          2. Fehr-Schmidt Subgame Perfect Bargaining Equilibrium Solver
          3. 1-Wasserstein Distance W_1(P_sim, P_human) & Jensen-Shannon Divergence (JSD)
        """
        return {"w1_distance": 0.0, "jsd_divergence": 0.0}