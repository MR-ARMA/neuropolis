from __future__ import annotations

from typing import List
import numpy as np

from src.core.contracts import AgentSeedConfig, BigFiveTraits, SimulationConfig
from src.core.seed_bundle import SeedBundle


SCALE_TO_AGENT_COUNT = {
    "mvp_6": 6,
    "dev": 12,
    "main": 24,
    "scale": 48,
}


def synthesize_parametric_population(
    config: SimulationConfig,
    seed_bundle: SeedBundle,
    target_n: int | None = None,
) -> List[AgentSeedConfig]:
    """
    Generates a stratified, survey-calibrated household population of size
    `N in {6, 12, 24, 48}` using `SeedBundle.pop_rng` (Section 3 & Defect D9):
      - Log-normal liquid cash and periodic wage distributions calibrated to
        SCF liquid-to-income quartiles (Kaplan, Violante, & Weidner 2014 [VERIFY]).
      - Fehr & Schmidt (1999) [VERIFY] Table III inequity-aversion distribution:
        30% Selfish (0, 0), 30% Moderate (0.5, 0.25), 30% Strong (1.0, 0.6),
        10% Hyper-Egalitarian (4.0, 0.6).
      - Seed-permuted neutral agent IDs (`Agent_01`..`Agent_N`) so labels are
        statistically independent of endowments and roles.
    """
    n_agents = target_n or SCALE_TO_AGENT_COUNT.get(config.population_scale, len(config.agents) or 6)
    if n_agents == len(config.agents) and config.population_scale == "mvp_6":
        agents_list = [ag.model_copy(deep=True) for ag in config.agents]
    else:
        rng = seed_bundle.get_np_rng("pop_rng")
        agents_list = []

        fs_tiers = [
            ("SELFISH", 0.0, 0.0),
            ("SELFISH", 0.0, 0.0),
            ("SELFISH", 0.0, 0.0),
            ("MODERATE", 0.50, 0.25),
            ("MODERATE", 0.50, 0.25),
            ("MODERATE", 0.50, 0.25),
            ("STRONG", 1.00, 0.60),
            ("STRONG", 1.00, 0.60),
            ("STRONG", 1.00, 0.60),
            ("HYPER_EGALITARIAN", 4.00, 0.60),
        ]

        for idx in range(n_agents):
            stratum = idx % 6
            if stratum in (0, 1):
                # Non-HtM Consumable Producer (33.3%: liquid cash & food surplus)
                role = "Consumable_Producer"
                w_tier = "MID"
                cash = float(np.clip(rng.normal(112.0, 16.0), 80.0, 160.0))
                c_inv = float(np.clip(rng.normal(7.5, 1.5), 4.5, 11.0))
                k_inv = float(rng.choice([0.0, 1.0]))
                wage = float(np.clip(rng.normal(16.5, 2.0), 12.0, 22.0))
                ob_r = round(0.35 * wage, 2)
                prod_q = round(float(np.clip(rng.normal(3.4, 0.4), 2.5, 4.5)), 2)
                spec = "CONSUMABLE_NECESSITY"
                exp_e = round(float(rng.uniform(0.55, 0.85)), 2)
            elif stratum in (2, 3):
                # Non-HtM Capital Specialist (33.3%: high cash & capital, buys food)
                role = "Capital_Specialist"
                w_tier = "HIGH"
                cash = float(np.clip(rng.normal(195.0, 28.0), 135.0, 280.0))
                c_inv = float(rng.choice([2.0, 3.0]))
                k_inv = float(np.clip(rng.integers(3, 6), 2.0, 6.0))
                wage = float(np.clip(rng.normal(28.0, 3.5), 21.0, 36.0))
                ob_r = round(0.28 * wage, 2)
                prod_q = 0.0
                spec = "PRODUCTIVE_CAPITAL_ASSET"
                exp_e = round(float(rng.uniform(0.20, 0.45)), 2)
            elif stratum == 4:
                # Wealthy Hand-to-Mouth (16.7%: illiquid capital >= 2, low liquid cash)
                role = "Illiquid_Asset_Holder"
                w_tier = "MID"
                cash = float(np.clip(rng.normal(42.0, 6.0), 28.0, 54.0))
                c_inv = 1.0
                k_inv = float(rng.choice([2.0, 3.0]))
                wage = float(np.clip(rng.normal(14.5, 1.8), 11.0, 18.0))
                ob_r = round(0.42 * wage, 2)
                prod_q = 0.0
                spec = "CONSUMABLE_NECESSITY"
                exp_e = round(float(rng.uniform(0.65, 0.90)), 2)
            else:
                # Poor Hand-to-Mouth (16.7%: low liquid cash, 0 capital, high obligation ratio)
                role = "Service_Laborer"
                w_tier = "LOW"
                cash = float(np.clip(rng.normal(34.0, 5.0), 24.0, 44.0))
                c_inv = 1.0
                k_inv = 0.0
                wage = float(np.clip(rng.normal(11.0, 1.2), 9.0, 14.0))
                ob_r = round(0.50 * wage, 2)
                prod_q = 0.0
                spec = "CONSUMABLE_NECESSITY"
                exp_e = round(float(rng.uniform(0.80, 1.00)), 2)

            fs_name, a_fs, b_fs = fs_tiers[idx % len(fs_tiers)]
            lam_cpt = round(float(np.clip(rng.normal(2.25, 0.35), 1.40, 3.20)), 2)
            gam_cpt = round(float(np.clip(rng.normal(0.88, 0.06), 0.70, 0.98)), 2)
            rigidity = round(float(rng.uniform(0.30, 0.80)), 2)
            anchor_s = round(float(rng.uniform(-0.85, 0.85)), 2)

            agents_list.append(
                AgentSeedConfig(
                    agent_id=f"Agent_{idx + 1:02d}",
                    occupation_role=role,
                    wage_tier=w_tier,  # type: ignore[arg-type]
                    liquid_wealth_cash=round(cash, 2),
                    consumable_necessity_inv=round(c_inv, 2),
                    productive_capital_inv=round(k_inv, 2),
                    base_income_wage=round(wage, 2),
                    periodic_obligation_r=ob_r,
                    shock_exposure_e=exp_e,
                    base_consumable_prod_q=prod_q,
                    fehr_schmidt_tier=fs_name,  # type: ignore[arg-type]
                    alpha_disadvantageous=a_fs,
                    beta_advantageous=b_fs,
                    loss_aversion_lambda=lam_cpt,
                    risk_tolerance_gamma=gam_cpt,
                    identity_rigidity=rigidity,
                    innate_opinion_anchor_s=anchor_s,
                    tms_specialty=spec,
                    big_five=BigFiveTraits(
                        openness=round(float(rng.uniform(0.35, 0.85)), 2),
                        conscientiousness=round(float(rng.uniform(0.45, 0.90)), 2),
                        extraversion=round(float(rng.uniform(0.35, 0.85)), 2),
                        agreeableness=round(float(rng.uniform(0.35, 0.85)), 2),
                        neuroticism=round(float(rng.uniform(0.25, 0.85)), 2),
                    ),
                )
            )

    if config.experimental_design.randomize_agent_identities:
        py_rng = seed_bundle.get_py_rng("pop_rng")
        neutral_ids = [f"Agent_{i:02d}" for i in range(1, len(agents_list) + 1)]
        py_rng.shuffle(neutral_ids)
        for ag, new_id in zip(agents_list, neutral_ids):
            ag.agent_id = new_id
        agents_list.sort(key=lambda a: a.agent_id)

    return agents_list