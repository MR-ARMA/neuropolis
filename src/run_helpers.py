"""
src/run_helpers.py
Self-contained population synthesizer for both `run_mvp.py` and `pytest` pre-flight tests.
"""

from __future__ import annotations

from typing import List, Tuple

from src.core.contracts import (
    BigFiveTraits,
    ExpertiseDomain,
    ExternalAgentState,
    FehrSchmidtParams,
    InternalAgentState,
    LegalStatus,
    MacroEconomyState,
    MaslowNeeds,
    MoodPAD,
    ProspectTheoryParams,
    SimulationSpec,
)


def synthesize_mvp_population(
    spec: SimulationSpec,
) -> Tuple[List[InternalAgentState], List[ExternalAgentState], MacroEconomyState]:
    archetypes = [
        (
            "Agent_1",
            "Elena Rostova",
            "Agri-Supply Wholesaler",
            ExpertiseDomain.FOOD_SUPPLY_CHAIN,
            "Fair transparent food pricing sustains community survival and long-term trust.",
            0.75,
            BigFiveTraits(openness=0.65, conscientiousness=0.80, extraversion=0.60, agreeableness=0.78, neuroticism=0.25),
            {"FOOD": 6, "TECH": 1},
        ),
        (
            "Agent_2",
            "Marcus Vance",
            "Hardware Arbitrageur",
            ExpertiseDomain.TECH_VALUATION,
            "Profit maximization and aggressive price bargaining reward the sharpest trader.",
            0.82,
            BigFiveTraits(openness=0.70, conscientiousness=0.75, extraversion=0.72, agreeableness=0.32, neuroticism=0.40),
            {"FOOD": 2, "TECH": 4},
        ),
        (
            "Agent_3",
            "Dr. Amara Okafor",
            "Monetary Economist",
            ExpertiseDomain.MACRO_INFLATION,
            "Monetary stability and strict solvency discipline prevent systemic credit crises.",
            0.68,
            BigFiveTraits(openness=0.85, conscientiousness=0.88, extraversion=0.50, agreeableness=0.65, neuroticism=0.30),
            {"FOOD": 3, "TECH": 2},
        ),
        (
            "Agent_4",
            "Viktor Petrov",
            "Risk & Credit Auditor",
            ExpertiseDomain.CREDIT_SOLVENCY,
            "Never trust unverified promises; demand collateral and punish contract defaults.",
            0.88,
            BigFiveTraits(openness=0.45, conscientiousness=0.90, extraversion=0.35, agreeableness=0.28, neuroticism=0.72),
            {"FOOD": 4, "TECH": 2},
        ),
        (
            "Agent_5",
            "Sofia Lin",
            "Cooperative Artisan",
            ExpertiseDomain.FOOD_SUPPLY_CHAIN,
            "Reciprocity, mutual aid, and equitable surplus sharing build resilient markets.",
            0.60,
            BigFiveTraits(openness=0.78, conscientiousness=0.70, extraversion=0.68, agreeableness=0.88, neuroticism=0.35),
            {"FOOD": 5, "TECH": 1},
        ),
        (
            "Agent_6",
            "Devon Brooks",
            "Speculative Broker",
            ExpertiseDomain.TECH_VALUATION,
            "Scarcity creates leverage; charge a premium whenever counterparties are desperate.",
            0.75,
            BigFiveTraits(openness=0.60, conscientiousness=0.55, extraversion=0.80, agreeableness=0.35, neuroticism=0.65),
            {"FOOD": 1, "TECH": 3},
        ),
    ]

    num_agents = min(spec.simulation_meta.num_agents, len(archetypes))
    internal_list: List[InternalAgentState] = []
    external_list: List[ExternalAgentState] = []

    for i in range(num_agents):
        aid, name, occ, dom, belief, rigid, b5, inv = archetypes[i]
        internal_list.append(
            InternalAgentState(
                agent_id=aid,
                name=name,
                occupation=occ,
                specialty_domain=dom,
                core_belief_statement=belief,
                identity_rigidity=rigid,
                big_five=b5,
                cpt_params=ProspectTheoryParams(
                    loss_aversion_lambda=round(1.8 + 1.2 * b5.neuroticism, 2),
                    risk_curvature_gamma=0.88,
                    discount_factor_delta=0.95,
                ),
                fehr_schmidt=FehrSchmidtParams(
                    disadvantageous_inequity_alpha=round(0.8 + 1.4 * b5.agreeableness, 2),
                    advantageous_inequity_beta=round(0.2 + 0.5 * b5.agreeableness, 2),
                ),
                mood_pad=MoodPAD(valence=0.15, arousal=0.25 + 0.2 * b5.neuroticism, dominance=0.5),
                maslow_needs=MaslowNeeds(
                    survival_satiety=0.55 if inv["FOOD"] <= 2 else 0.85,
                    safety_security=0.75,
                    social_belonging=0.65,
                    esteem_status=0.55,
                ),
                fatigue_level=0.10,
            )
        )
        external_list.append(
            ExternalAgentState(
                agent_id=aid,
                liquid_wealth_cash=spec.economy_and_institutions.initial_agent_cash,
                bank_deposit=0.0,
                outstanding_debt=0.0,
                credit_score=720.0,
                inventory=dict(inv),
                base_wage=spec.economy_and_institutions.base_wage_per_tick,
                social_status=0.5,
                legal_status=LegalStatus.LAW_ABIDING,
            )
        )

    total_agent_cash = sum(e.liquid_wealth_cash for e in external_list)
    econ = spec.economy_and_institutions
    m0_total = (
        total_agent_cash
        + econ.initial_treasury_cash
        + econ.initial_bank_reserves
        + econ.initial_firm_sector_cash
    )

    macro = MacroEconomyState(
        tick=0,
        central_bank_base_money_m0=m0_total,
        commercial_bank_reserves=econ.initial_bank_reserves,
        public_treasury_cash=econ.initial_treasury_cash,
        firm_sector_cash=econ.initial_firm_sector_cash,
        market_clearing_price_food=econ.reference_food_price,
        market_clearing_price_tech=econ.reference_tech_price,
    )
    return internal_list, external_list, macro


def build_synthetic_population_for_tests(
    num_agents: int = 6,
) -> Tuple[List[InternalAgentState], List[ExternalAgentState], MacroEconomyState]:
    spec = SimulationSpec()
    spec.simulation_meta.num_agents = num_agents
    return synthesize_mvp_population(spec)
