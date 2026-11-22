from __future__ import annotations

from dataclasses import dataclass
import math

from src.core.contracts import DeliberationBottleneckConfig, NeedsState


@dataclass(frozen=True)
class DeliberationAllocation:
    """
    Domain-general cognitive budget allocated to an agent for a single turn.
      - `cot_max_tokens` is dynamically compressed under high physiological/liquidity stress
        when Factor A (`DeliberationBottleneckConfig.enabled`) is ON.
      - `action_json_max_tokens` is strictly invariant to stress so cognitive narrowing
        never degrades JSON formatting validity.
    """
    cot_max_tokens: int
    action_json_max_tokens: int
    stress_input: float
    health_input: float
    effective_load: float
    bottleneck_enabled: bool
    compression_ratio: float


class DeliberationBudgetController:
    """
    Factor A (Domain-General Cognitive Bottleneck): Bounded Deliberation Budget.
    Replaces lexical token-masking hacks (PASTA) with a topic-blind computational
    bandwidth constraint on Stage 1 internal Chain-of-Thought (CoT) length:

        B_CoT(S_eff) = floor( B_min + (B_max - B_min) * exp(-k_delib * S_eff) )

    where S_eff combines objective physiological/liquidity stress S_{i,t} in [0, 1]
    with acute vitality degradation max(0, 0.50 - H_{i,t}).
    """

    def __init__(self, config: DeliberationBottleneckConfig) -> None:
        self.config = config

    def compute_allocation(self, needs: NeedsState) -> DeliberationAllocation:
        """Computes Stage 1 CoT token budget and stress-invariant Stage 2 JSON budget."""
        stress = max(0.0, min(1.0, float(needs.composite_stress)))
        health = max(0.0, min(1.0, float(needs.health)))

        b_max = int(self.config.cot_max_tokens_calm)
        b_min = int(self.config.cot_min_tokens_stressed)
        action_budget = int(self.config.action_json_max_tokens)

        if not self.config.enabled:
            return DeliberationAllocation(
                cot_max_tokens=b_max,
                action_json_max_tokens=action_budget,
                stress_input=stress,
                health_input=health,
                effective_load=0.0,
                bottleneck_enabled=False,
                compression_ratio=1.0,
            )

        # Objective acute vitality penalty amplifies cognitive load when H < 0.50
        health_load_addon = 0.25 * max(0.0, 0.50 - health) / 0.50
        s_eff = max(0.0, min(1.0, stress + health_load_addon))

        decay = math.exp(-float(self.config.stress_decay_steepness) * s_eff)
        raw_budget = b_min + (b_max - b_min) * decay
        cot_tokens = max(b_min, min(b_max, int(math.floor(raw_budget))))
        comp_ratio = round(cot_tokens / max(1.0, float(b_max)), 4)

        return DeliberationAllocation(
            cot_max_tokens=cot_tokens,
            action_json_max_tokens=action_budget,
            stress_input=round(stress, 6),
            health_input=round(health, 6),
            effective_load=round(s_eff, 6),
            bottleneck_enabled=True,
            compression_ratio=comp_ratio,
        )
    