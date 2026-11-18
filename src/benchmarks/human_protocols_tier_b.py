from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple
import numpy as np

from src.evaluation.validity_and_stats import (
    UnderfilledBinWarning,
    compute_curve_distances_and_tost,
)


# Published Human Reference Curves & Meta-Analytic Standard Errors [VERIFY]:
# B1: Oosterbeek, Sloof, & van de Kuilen (2004, Experimental Economics) [VERIFY]
#     Ultimatum acceptance probability across 4 offer-fairness / markup bins:
B1_OOSTERBEEK_OFFER_SHARES = [0.45, 0.35, 0.22, 0.08]
B1_OOSTERBEEK_ACCEPT_RATES = [0.94, 0.76, 0.42, 0.14]
B1_OOSTERBEEK_HUMAN_SE = [0.03, 0.04, 0.05, 0.04]

# B2: Fehr & Gächter (2000, AER) [VERIFY] Public Goods contribution rate (% of endowment)
#     across 4 periods WITH punishment vs. WITHOUT punishment:
B2_FEHR_GACHTER_WITH_PUNISHMENT = [0.68, 0.76, 0.83, 0.88]
B2_FEHR_GACHTER_NO_PUNISHMENT = [0.52, 0.41, 0.31, 0.22]
B2_HUMAN_SE = [0.04, 0.04, 0.04, 0.04]

# B3: Jappelli & Pistaferri (2014, AEJ: Macroeconomics) [VERIFY]
#     Marginal Propensity to Consume (MPC) across 3 Cash-on-Hand Terciles (Low, Mid, High):
B3_JAPPELLI_MPC_TERCILES = [0.63, 0.46, 0.31]
B3_HUMAN_SE = [0.04, 0.03, 0.03]

# B4: Berg, Dickhaut, & McCabe (1995, GEB) [VERIFY] Trust Game:
#     Mean fraction returned by Trustee across 3 Trustor investment levels (Low, Mid, High):
B4_BDM_TRUST_RETURN_CURVE = [0.22, 0.44, 0.62]
B4_HUMAN_SE = [0.04, 0.05, 0.05]

# B5: Kaplan, Violante, & Weidner (2014, BPEA) [VERIFY] / Distress Asset Liquidation:
#     Illiquid asset sale propensity across 3 liquidity-stress terciles (Calm, Moderate, Acute):
B5_DISTRESS_SALE_CURVE = [0.06, 0.24, 0.58]
B5_HUMAN_SE = [0.03, 0.04, 0.05]


class TierBHumanProtocolSuite:
    """
    Section 7 Tier B: Standalone Human-Protocol Alignment Environments (B1..B5).
    Compares whole conditional response curves (never single threshold flags)
    using continuous Wasserstein-1 (W_1) and L_1 distances, 10,000-resample
    bootstrap 95% CIs, and TOST equivalence tests against human meta-analytic
    standard errors. Enforces >= 30 observations per bin (Resolves Defect D9).
    """

    @staticmethod
    def run_b1_ultimatum_protocol(
        responder_accept_fn: Callable[[float, float, int], bool],
        n_per_bin: int = 30,
        sesoi_w1: float = 0.05,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Protocol B1: One-Shot Ultimatum Game across the 4 offer-share bins.
        `responder_accept_fn(offer_share, responder_scarcity_f, trial_seed) -> bool`.
        """
        rng = np.random.default_rng(seed)
        sim_rates: List[float] = []
        bin_counts: List[int] = []

        for b_idx, offer_sh in enumerate(B1_OOSTERBEEK_OFFER_SHARES):
            accepts = 0
            for t_i in range(n_per_bin):
                scarcity_f = float(rng.uniform(0.10, 0.65))
                sub_seed = int(seed + b_idx * 1000 + t_i)
                if responder_accept_fn(offer_sh, scarcity_f, sub_seed):
                    accepts += 1
            sim_rates.append(round(accepts / max(1, n_per_bin), 4))
            bin_counts.append(n_per_bin)

        stats_res = compute_curve_distances_and_tost(
            sim_curve=sim_rates,
            human_curve=B1_OOSTERBEEK_ACCEPT_RATES,
            human_se=B1_OOSTERBEEK_HUMAN_SE,
            bin_counts=bin_counts,
            min_bin_n=30,
            sesoi=sesoi_w1,
            seed=seed,
        )
        return {
            "protocol_id": "B1_Ultimatum_Oosterbeek_2004",
            "offer_share_bins": B1_OOSTERBEEK_OFFER_SHARES,
            "human_curve": B1_OOSTERBEEK_ACCEPT_RATES,
            "sim_curve": sim_rates,
            "bin_counts": bin_counts,
            **stats_res,
        }

    @staticmethod
    def run_b2_public_goods_protocol(
        contrib_rate_fn: Callable[[int, bool, int], float],
        n_per_period: int = 30,
        with_punishment: bool = True,
        sesoi_w1: float = 0.05,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Protocol B2: Public Goods Game across 4 periods with or without punishment.
        """
        sim_curve: List[float] = []
        bin_counts: List[int] = []
        target_curve = (
            B2_FEHR_GACHTER_WITH_PUNISHMENT
            if with_punishment
            else B2_FEHR_GACHTER_NO_PUNISHMENT
        )

        for period_idx in range(4):
            rates = [
                max(
                    0.0,
                    min(
                        1.0,
                        contrib_rate_fn(
                            period_idx,
                            with_punishment,
                            seed + period_idx * 500 + i,
                        ),
                    ),
                )
                for i in range(n_per_period)
            ]
            sim_curve.append(round(float(np.mean(rates)), 4))
            bin_counts.append(n_per_period)

        stats_res = compute_curve_distances_and_tost(
            sim_curve=sim_curve,
            human_curve=target_curve,
            human_se=B2_HUMAN_SE,
            bin_counts=bin_counts,
            min_bin_n=30,
            sesoi=sesoi_w1,
            seed=seed,
        )
        return {
            "protocol_id": "B2_PublicGoods_FehrGachter_2000",
            "with_punishment": with_punishment,
            "human_curve": target_curve,
            "sim_curve": sim_curve,
            "bin_counts": bin_counts,
            **stats_res,
        }

    @staticmethod
    def run_b3_windfall_mpc_protocol(
        mpc_choice_fn: Callable[[int, float, int], float],
        n_per_tercile: int = 30,
        sesoi_w1: float = 0.05,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Protocol B3: Windfall Consumption MPC Vignette across 3 Liquidity Terciles
        (Tercile 0 = Low Liquidity, Tercile 1 = Mid Liquidity, Tercile 2 = High Liquidity).
        """
        rng = np.random.default_rng(seed)
        sim_curve: List[float] = []
        bin_counts: List[int] = []
        scarcity_levels = [0.85, 0.42, 0.08]

        all_x: List[float] = []
        all_y: List[float] = []

        for t_idx, f_lvl in enumerate(scarcity_levels):
            mpcs: List[float] = []
            for i in range(n_per_tercile):
                jitter_f = float(np.clip(f_lvl + rng.normal(0.0, 0.04), 0.0, 1.0))
                val = max(
                    0.0,
                    min(1.0, mpc_choice_fn(t_idx, jitter_f, seed + t_idx * 500 + i)),
                )
                mpcs.append(val)
                all_x.append(float(t_idx))
                all_y.append(val)
            sim_curve.append(round(float(np.mean(mpcs)), 4))
            bin_counts.append(n_per_tercile)

        # OLS regression slope across terciles
        x_arr = np.asarray(all_x, dtype=np.float64)
        y_arr = np.asarray(all_y, dtype=np.float64)
        ols_slope = round(
            float(np.cov(x_arr, y_arr)[0, 1] / max(1e-6, np.var(x_arr))), 4
        )

        stats_res = compute_curve_distances_and_tost(
            sim_curve=sim_curve,
            human_curve=B3_JAPPELLI_MPC_TERCILES,
            human_se=B3_HUMAN_SE,
            bin_counts=bin_counts,
            min_bin_n=30,
            sesoi=sesoi_w1,
            seed=seed,
        )
        return {
            "protocol_id": "B3_WindfallMPC_JappelliPistaferri_2014",
            "human_curve": B3_JAPPELLI_MPC_TERCILES,
            "sim_curve": sim_curve,
            "ols_tercile_slope": ols_slope,
            "low_minus_high_mpc": round(sim_curve[0] - sim_curve[2], 4),
            "bin_counts": bin_counts,
            **stats_res,
        }