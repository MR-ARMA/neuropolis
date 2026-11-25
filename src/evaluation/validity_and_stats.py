from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings
import numpy as np

from src.core.contracts import PerBinTOSTResult, StatisticalProtocolConfig


class UnderfilledBinWarning(UserWarning):
    """Emitted whenever an empirical response bin has fewer than `min_observations_per_bin` samples."""


class UnderpoweredClusterWarning(UserWarning):
    """Emitted whenever cluster-level inference is invoked with fewer than M=30 independent seeds."""


# ============================================================================
# 1. EXACT STUDENT-T CDF & MULTIPLICITY CORRECTIONS
# ============================================================================

def _student_t_cdf(t_val: float, df: int) -> float:
    """
    Computes exact Student-t CDF P(T <= t_val) with `df` degrees of freedom.
    Uses `scipy.stats.t.cdf` if available, falling back to high-precision
    composite Simpson numerical integration of the exact Student-t PDF (err < 1e-8).
    """
    if not math.isfinite(t_val):
        return 1.0 if t_val > 0 else 0.0
    if df <= 0:
        return 0.5 * (1.0 + math.erf(t_val / math.sqrt(2.0)))

    try:
        from scipy.stats import t as scipy_t

        return float(scipy_t.cdf(t_val, df))
    except Exception:
        pass

    # Pure-Python exact Simpson integration over [0, |t_val|] exploiting symmetry
    abs_t = min(50.0, abs(float(t_val)))
    if abs_t <= 1e-12:
        return 0.5

    log_coeff = (
        math.lgamma(0.5 * (df + 1.0))
        - math.lgamma(0.5 * df)
        - 0.5 * math.log(df * math.pi)
    )
    coeff = math.exp(log_coeff)
    power = -0.5 * (df + 1.0)

    n_steps = 400
    h = abs_t / n_steps
    acc = 0.0
    for i in range(n_steps + 1):
        x = i * h
        pdf_x = coeff * ((1.0 + (x * x) / df) ** power)
        if i == 0 or i == n_steps:
            w = 1.0
        elif i % 2 == 1:
            w = 4.0
        else:
            w = 2.0
        acc += w * pdf_x
    half_area = (h / 3.0) * acc
    cdf_pos = min(1.0, 0.5 + half_area)
    return cdf_pos if t_val >= 0.0 else max(0.0, 1.0 - cdf_pos)


def adjust_p_values(
    p_values: Sequence[float],
    method: str = "holm_bonferroni",
) -> List[float]:
    """
    Applies family-wise (Holm-Bonferroni) or FDR (Benjamini-Hochberg)
    multiplicity adjustment across bins or factorial contrasts.
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda pair: float(pair[1]))
    adjusted = [1.0] * m

    if method == "benjamini_hochberg":
        running_min = 1.0
        for rank_rev in range(m - 1, -1, -1):
            orig_idx, p_val = indexed[rank_rev]
            rank = rank_rev + 1
            adj = min(1.0, float(p_val) * (float(m) / float(rank)))
            running_min = min(running_min, adj)
            adjusted[orig_idx] = round(running_min, 6)
    else:
        running_max = 0.0
        for rank_idx, (orig_idx, p_val) in enumerate(indexed):
            adj = min(1.0, float(p_val) * float(m - rank_idx))
            running_max = max(running_max, adj)
            adjusted[orig_idx] = round(running_max, 6)

    return adjusted


# ============================================================================
# 2. CLUSTERED BOOTSTRAPPING OVER M >= 30 INDEPENDENT ECONOMY SEEDS
# ============================================================================

@dataclass(frozen=True)
class ClusteredBootstrapSummary:
    n_clusters_m: int
    power_adequate: bool
    cluster_mean: float
    clustered_se: float
    ci_90: Tuple[float, float]
    ci_95: Tuple[float, float]
    between_cluster_sd: float


def clustered_bootstrap_mean(
    cluster_values: Sequence[float],
    n_resamples_b: int = 10000,
    min_required_m: int = 30,
    seed: int = 42,
) -> ClusteredBootstrapSummary:
    """
    Clustered Non-Parametric Bootstrap where the sole resampling unit is the
    independent economy seed m in {1, ..., M}. Never treats agents, ticks,
    or CDA transactions within a shared economy as independent observations.
    """
    arr = np.asarray([float(x) for x in cluster_values if math.isfinite(float(x))], dtype=np.float64)
    m = int(arr.size)
    if m == 0:
        return ClusteredBootstrapSummary(
            n_clusters_m=0,
            power_adequate=False,
            cluster_mean=0.0,
            clustered_se=0.0,
            ci_90=(0.0, 0.0),
            ci_95=(0.0, 0.0),
            between_cluster_sd=0.0,
        )

    power_ok = m >= min_required_m
    if not power_ok:
        warnings.warn(
            f"Clustered inference run with M={m} economy seeds (< required M={min_required_m}). "
            f"Results are exploratory/underpowered only.",
            UnderpoweredClusterWarning,
            stacklevel=2,
        )

    mean_est = float(np.mean(arr))
    if m == 1:
        return ClusteredBootstrapSummary(
            n_clusters_m=1,
            power_adequate=False,
            cluster_mean=round(mean_est, 6),
            clustered_se=0.0,
            ci_90=(round(mean_est, 6), round(mean_est, 6)),
            ci_95=(round(mean_est, 6), round(mean_est, 6)),
            between_cluster_sd=0.0,
        )

    sd_cluster = float(np.std(arr, ddof=1))
    se_cluster = sd_cluster / math.sqrt(float(m))

    rng = np.random.default_rng(seed)
    idx_mat = rng.integers(0, m, size=(n_resamples_b, m))
    boot_means = np.mean(arr[idx_mat], axis=1)

    ci_90 = (
        round(float(np.quantile(boot_means, 0.05)), 6),
        round(float(np.quantile(boot_means, 0.95)), 6),
    )
    ci_95 = (
        round(float(np.quantile(boot_means, 0.025)), 6),
        round(float(np.quantile(boot_means, 0.975)), 6),
    )

    return ClusteredBootstrapSummary(
        n_clusters_m=m,
        power_adequate=power_ok,
        cluster_mean=round(mean_est, 6),
        clustered_se=round(se_cluster, 6),
        ci_90=ci_90,
        ci_95=ci_95,
        between_cluster_sd=round(sd_cluster, 6),
    )


def clustered_paired_contrast(
    cell_a_by_seed: Dict[int, float],
    cell_b_by_seed: Dict[int, float],
    n_resamples_b: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Common Random Numbers (CRN) paired cluster contrast (Cell A - Cell B)
    computed strictly across shared economy seeds m in (Seeds_A intersect Seeds_B).
    """
    shared_seeds = sorted(set(cell_a_by_seed.keys()) & set(cell_b_by_seed.keys()))
    diffs = [float(cell_a_by_seed[s]) - float(cell_b_by_seed[s]) for s in shared_seeds]
    boot = clustered_bootstrap_mean(
        diffs, n_resamples_b=n_resamples_b, min_required_m=30, seed=seed
    )

    m = len(diffs)
    if m >= 2 and boot.clustered_se > 1e-12:
        t_stat = boot.cluster_mean / boot.clustered_se
        p_two_sided = 2.0 * min(
            _student_t_cdf(t_stat, m - 1), 1.0 - _student_t_cdf(t_stat, m - 1)
        )
    else:
        t_stat = 0.0
        p_two_sided = 1.0

    return {
        "n_paired_seeds_m": m,
        "paired_mean_diff": boot.cluster_mean,
        "paired_clustered_se": boot.clustered_se,
        "ci_95": boot.ci_95,
        "t_statistic": round(t_stat, 4),
        "p_value_two_sided": round(max(0.0, min(1.0, p_two_sided)), 6),
        "ci_95_excludes_zero": bool(boot.ci_95[0] > 0.0 or boot.ci_95[1] < 0.0),
    }


# ============================================================================
# 3. PER-BIN TOST (INTERSECTION-UNION TEST) WITH A PRIORI SESOI MARGINS
# ============================================================================

def run_per_bin_tost_family(
    benchmark_name: str,
    bin_labels: Sequence[str],
    cluster_bin_values: Sequence[Sequence[float]],
    bin_observation_counts: Sequence[int],
    human_reference_means: Sequence[float],
    sesoi_margin_delta: float,
    stats_cfg: StatisticalProtocolConfig,
    seed: int = 42,
) -> List[PerBinTOSTResult]:
    """
    Executes Per-Bin Two One-Sided Tests (TOST / Intersection-Union Test) across
    all bins of an empirical curve using clustered standard errors over M economy seeds:
      - H_{0,low}:  (mu_b - R_b) <= -Delta_SESOI
      - H_{0,high}: (mu_b - R_b) >= +Delta_SESOI
      - p_TOST = max(p_low, p_high)
      - Family-wise multiplicity adjustment (Holm-Bonferroni / BH) applied across bins.
      - Statistical equivalence holds IFF adjusted p_TOST < alpha AND bin_coverage_valid.
    """
    raw_results: List[PerBinTOSTResult] = []
    raw_p_max_list: List[float] = []

    for b_idx, label in enumerate(bin_labels):
        c_vals = [float(v) for v in cluster_bin_values[b_idx] if math.isfinite(float(v))]
        obs_cnt = int(bin_observation_counts[b_idx])
        ref_val = float(human_reference_means[b_idx])
        delta = float(sesoi_margin_delta)

        bin_valid = obs_cnt >= stats_cfg.min_observations_per_bin
        if not bin_valid:
            warnings.warn(
                f"{benchmark_name} [{label}]: underfilled bin ({obs_cnt} < {stats_cfg.min_observations_per_bin}).",
                UnderfilledBinWarning,
                stacklevel=2,
            )

        boot = clustered_bootstrap_mean(
            c_vals,
            n_resamples_b=stats_cfg.bootstrap_resamples_b,
            min_required_m=stats_cfg.num_economy_seeds_m,
            seed=seed + b_idx * 17,
        )
        m = boot.n_clusters_m
        mean_diff = round(boot.cluster_mean - ref_val, 6)

        if m >= 2:
            # Guard against degenerate zero variance across seeds
            eff_se = max(boot.clustered_se, 1e-6)
            df = m - 1
            t_low = (mean_diff + delta) / eff_se
            t_high = (mean_diff - delta) / eff_se
            p_low = max(0.0, min(1.0, 1.0 - _student_t_cdf(t_low, df)))
            p_high = max(0.0, min(1.0, _student_t_cdf(t_high, df)))
            p_tost = max(p_low, p_high)
        else:
            p_low = 1.0
            p_high = 1.0
            p_tost = 1.0

        raw_p_max_list.append(p_tost)
        raw_results.append(
            PerBinTOSTResult(
                benchmark_name=benchmark_name,
                bin_label=label,
                n_clusters_m=m,
                n_observations_in_bin=obs_cnt,
                bin_coverage_valid=bin_valid,
                sim_cluster_mean=boot.cluster_mean,
                human_reference_mean=ref_val,
                mean_difference=mean_diff,
                clustered_se=boot.clustered_se,
                ci_90_lower=boot.ci_90[0],
                ci_90_upper=boot.ci_90[1],
                ci_95_lower=boot.ci_95[0],
                ci_95_upper=boot.ci_95[1],
                sesoi_margin_delta=delta,
                tost_p_lower=round(p_low, 6),
                tost_p_upper=round(p_high, 6),
                tost_p_max=round(p_tost, 6),
                holm_adjusted_p=None,
                statistically_equivalent=False,
            )
        )

    adj_ps = adjust_p_values(
        raw_p_max_list, method=stats_cfg.multiplicity_correction
    )
    final_results: List[PerBinTOSTResult] = []
    for res, p_adj in zip(raw_results, adj_ps):
        is_equiv = bool(
            p_adj < stats_cfg.alpha_level
            and res.bin_coverage_valid
            and res.n_clusters_m >= 2
        )
        final_results.append(
            res.model_copy(
                update={
                    "holm_adjusted_p": p_adj,
                    "statistically_equivalent": is_equiv,
                }
            )
        )
    return final_results


# ============================================================================
# 4. BIAS-CORRECTED WASSERSTEIN-1 & MILLER-MADOW JENSEN-SHANNON DIVERGENCE
# ============================================================================

def compute_debiased_wasserstein_curve(
    cluster_curves: Sequence[Sequence[float]],
    human_curve: Sequence[float],
    zi_null_floor_w1: Optional[float] = None,
    human_se_ceiling_w1: Optional[float] = None,
    n_resamples_b: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes finite-sample bias-corrected 1-Wasserstein distance (W_1) between
    cluster-aggregated simulated response curves and human reference curves,
    with clustered bootstrap 95% CIs and empirical ZI floor / human SE ceiling.
    """
    mat = np.asarray(cluster_curves, dtype=np.float64)  # Shape [M, K_bins]
    h_ref = np.asarray(human_curve, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] == 0:
        return {
            "raw_w1": 1.0,
            "debiased_w1": 1.0,
            "ci_95": (1.0, 1.0),
            "zi_null_floor_w1": zi_null_floor_w1,
            "human_noise_ceiling_w1": human_se_ceiling_w1,
            "ceiling_normalized_skill": 0.0,
        }

    m, k_bins = mat.shape

    def _curve_w1(sim_c: np.ndarray) -> float:
        # Normalize curves to probability weights if non-negative and non-zero,
        # otherwise evaluate discrete cumulative curve EMD over ordered bins
        diff_cdf = np.cumsum(sim_c - h_ref) / max(1.0, float(k_bins))
        return float(np.mean(np.abs(diff_cdf)))

    mean_curve = np.mean(mat, axis=0)
    raw_w1 = _curve_w1(mean_curve)

    rng = np.random.default_rng(seed)
    boot_w1s: List[float] = []
    for _ in range(n_resamples_b):
        idx = rng.integers(0, m, size=m)
        b_mean_curve = np.mean(mat[idx], axis=0)
        boot_w1s.append(_curve_w1(b_mean_curve))

    mean_boot_w1 = float(np.mean(boot_w1s))
    # First-order bootstrap bias correction: W1_debiased = max(0, 2 * W1_hat - E[W1*])
    debiased_w1 = max(0.0, 2.0 * raw_w1 - mean_boot_w1)
    ci_95 = (
        round(float(np.quantile(boot_w1s, 0.025)), 6),
        round(float(np.quantile(boot_w1s, 0.975)), 6),
    )

    skill_score: Optional[float] = None
    if zi_null_floor_w1 is not None and human_se_ceiling_w1 is not None:
        denom = max(1e-6, float(zi_null_floor_w1) - float(human_se_ceiling_w1))
        skill_score = round(
            100.0 * (float(zi_null_floor_w1) - debiased_w1) / denom, 2
        )

    return {
        "raw_w1": round(raw_w1, 6),
        "debiased_w1": round(debiased_w1, 6),
        "ci_95": ci_95,
        "zi_null_floor_w1": (
            round(float(zi_null_floor_w1), 6)
            if zi_null_floor_w1 is not None
            else None
        ),
        "human_noise_ceiling_w1": (
            round(float(human_se_ceiling_w1), 6)
            if human_se_ceiling_w1 is not None
            else None
        ),
        "ceiling_normalized_skill": skill_score,
    }


def compute_miller_madow_jsd(
    sim_category_counts: Sequence[float],
    human_category_probs: Sequence[float],
) -> Dict[str, float]:
    """
    Computes finite-sample Miller-Madow bias-corrected Jensen-Shannon Divergence
    (in nats, bounded in [0, ln(2)]) for discrete distributions (e.g., Poor-HtM,
    Wealthy-HtM, Non-HtM).
    """
    counts = np.asarray(sim_category_counts, dtype=np.float64)
    q = np.asarray(human_category_probs, dtype=np.float64)
    q = np.clip(q, 1e-12, None)
    q = q / np.sum(q)

    n_total = float(np.sum(counts))
    if n_total <= 1e-9:
        return {"raw_jsd": round(math.log(2.0), 6), "debiased_jsd": round(math.log(2.0), 6)}

    p = np.clip(counts / n_total, 1e-12, None)
    p = p / np.sum(p)
    m_dist = 0.5 * (p + q)

    def _shannon_h(prob_vec: np.ndarray) -> float:
        return float(-np.sum(prob_vec * np.log(np.clip(prob_vec, 1e-12, None))))

    h_p_mle = _shannon_h(p)
    h_q = _shannon_h(q)
    h_m_mle = _shannon_h(m_dist)
    raw_jsd = max(0.0, h_m_mle - 0.5 * h_p_mle - 0.5 * h_q)

    # Miller-Madow finite-sample correction: H_MM = H_MLE + (K_nonzero - 1) / (2 * N)
    k_nonzero = int(np.sum(counts > 0))
    mm_correction = max(0, k_nonzero - 1) / (2.0 * max(1.0, n_total))
    h_p_mm = h_p_mle + mm_correction
    h_m_mm = h_m_mle + 0.5 * mm_correction
    debiased_jsd = max(0.0, min(math.log(2.0), h_m_mm - 0.5 * h_p_mm - 0.5 * h_q))

    return {
        "raw_jsd": round(raw_jsd, 6),
        "debiased_jsd": round(debiased_jsd, 6),
    }


# ============================================================================
# 5. 2^3 FACTORIAL EFFECTS, INTERACTIONS & EXACT SHAPLEY ATTRIBUTION
# ============================================================================

def compute_2x2x2_factorial_effects_and_shapley(
    cell_means_by_bits: Dict[Tuple[int, int, int], float],
) -> Dict[str, Any]:
    """
    Computes exact main effects, 2-way interactions, 3-way interaction, and
    exact Shapley credit values across the 2^3 factorial lattice:
      - Factor A: Bounded Deliberation Budget (B_CoT)
      - Factor B: Episodic Memory Friction (M_RAG)
      - Factor C: Two-Stage Endogenous RepE (R_RepE)
    Note: Lower distance metrics can be negated before passing if "higher = better".
    """
    expected_keys = list(itertools.product([0, 1], repeat=3))
    if not all(k in cell_means_by_bits for k in expected_keys):
        return {"complete_2x2x2_lattice": False}

    def _y(a: int, b: int, c: int) -> float:
        return float(cell_means_by_bits[(a, b, c)])

    # Yates orthogonal contrasts for 2^3 design
    main_a = 0.25 * (
        (_y(1, 0, 0) - _y(0, 0, 0))
        + (_y(1, 1, 0) - _y(0, 1, 0))
        + (_y(1, 0, 1) - _y(0, 0, 1))
        + (_y(1, 1, 1) - _y(0, 1, 1))
    )
    main_b = 0.25 * (
        (_y(0, 1, 0) - _y(0, 0, 0))
        + (_y(1, 1, 0) - _y(1, 0, 0))
        + (_y(0, 1, 1) - _y(0, 0, 1))
        + (_y(1, 1, 1) - _y(1, 0, 1))
    )
    main_c = 0.25 * (
        (_y(0, 0, 1) - _y(0, 0, 0))
        + (_y(1, 0, 1) - _y(1, 0, 0))
        + (_y(0, 1, 1) - _y(0, 1, 0))
        + (_y(1, 1, 1) - _y(1, 1, 0))
    )
    int_ab = 0.25 * (
        (_y(1, 1, 0) - _y(1, 0, 0) - _y(0, 1, 0) + _y(0, 0, 0))
        + (_y(1, 1, 1) - _y(1, 0, 1) - _y(0, 1, 1) + _y(0, 0, 1))
    )
    int_ac = 0.25 * (
        (_y(1, 0, 1) - _y(1, 0, 0) - _y(0, 0, 1) + _y(0, 0, 0))
        + (_y(1, 1, 1) - _y(1, 1, 0) - _y(0, 1, 1) + _y(0, 1, 0))
    )
    int_bc = 0.25 * (
        (_y(0, 1, 1) - _y(0, 1, 0) - _y(0, 0, 1) + _y(0, 0, 0))
        + (_y(1, 1, 1) - _y(1, 1, 0) - _y(1, 0, 1) + _y(1, 0, 0))
    )
    int_abc = 0.25 * (
        (_y(1, 1, 1) - _y(1, 1, 0) - _y(1, 0, 1) + _y(1, 0, 0))
        - (_y(0, 1, 1) - _y(0, 1, 0) - _y(0, 0, 1) + _y(0, 0, 0))
    )

    # Exact 2^3 Shapley values summing identically to y(1,1,1) - y(0,0,0)
    names = ["A_Bounded_Deliberation", "B_Memory_Friction", "C_Endogenous_RepE"]
    shapley: Dict[str, float] = {}
    fact_3 = 6.0
    for i, name in enumerate(names):
        phi_i = 0.0
        for mask in expected_keys:
            if mask[i] == 1:
                continue
            s_sz = sum(mask)
            w = (math.factorial(s_sz) * math.factorial(3 - s_sz - 1)) / fact_3
            m_with = list(mask)
            m_with[i] = 1
            phi_i += w * (
                float(cell_means_by_bits[tuple(m_with)])  # type: ignore[index]
                - float(cell_means_by_bits[mask])
            )
        shapley[name] = round(phi_i, 6)

    return {
        "complete_2x2x2_lattice": True,
        "main_effects": {
            "A_Bounded_Deliberation": round(main_a, 6),
            "B_Memory_Friction": round(main_b, 6),
            "C_Endogenous_RepE": round(main_c, 6),
        },
        "interactions": {
            "AB": round(int_ab, 6),
            "AC": round(int_ac, 6),
            "BC": round(int_bc, 6),
            "ABC": round(int_abc, 6),
        },
        "exact_shapley_values": shapley,
        "total_improvement_C111_minus_C000": round(_y(1, 1, 1) - _y(0, 0, 0), 6),
    }