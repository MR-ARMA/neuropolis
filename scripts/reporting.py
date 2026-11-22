from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluation.metrics import (
    HUMAN_BARGAINING_ACCEPT_CURVE,
    HUMAN_BARGAINING_BIN_LABELS,
    HUMAN_BARGAINING_SE,
    HUMAN_HTM_DISTRIBUTION,
    HUMAN_HTM_LABELS,
    HUMAN_MPC_CURVE,
    HUMAN_MPC_SE,
    HUMAN_MPC_TERCILE_LABELS,
)
from src.evaluation.validity_and_stats import (
    compute_2x2x2_factorial_effects_and_shapley,
)


def _apply_publication_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "grid.linestyle": "--",
            "grid.alpha": 0.45,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#cccccc",
        }
    )


def _add_watermark_if_synthetic(fig: plt.Figure, is_synthetic: bool) -> None:
    if is_synthetic:
        fig.text(
            0.5,
            0.5,
            "SYNTHETIC - PIPELINE TEST",
            fontsize=34,
            color="red",
            alpha=0.22,
            ha="center",
            va="center",
            rotation=28,
            fontweight="bold",
        )


def plot_fig1_factorial_ablation_and_shapley(
    report: Dict[str, Any],
    out_dir: str,
    is_synthetic: bool = False,
) -> str:
    """
    Figure 1:
      - Left: Debiased 1-Wasserstein Distance (W_1) with 95% Clustered Bootstrap CIs
        across all executed 2^3 factorial cells & matched controls, anchored to the
        Zero-Intelligence Null Floor and Human Meta-Analytic SE Noise Ceiling.
      - Right: Exact 2^3 Factorial Main Effects, Interactions (AB, AC, BC, ABC),
        and Exact Shapley Attribution (phi_A, phi_B, phi_C).
    """
    cell_reports: Dict[str, Dict[str, Any]] = report["cell_reports"]
    factorial: Optional[Dict[str, Any]] = report.get("factorial_2x2x2_analysis")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax_left, ax_right = axes

    cell_ids = list(cell_reports.keys())
    w1_vals = [float(cell_reports[c]["bargaining_w1"]["debiased_w1"]) for c in cell_ids]
    ci_lows = [
        max(0.0, w1_vals[i] - float(cell_reports[c]["bargaining_w1"]["ci_95"][0]))
        for i, c in enumerate(cell_ids)
    ]
    ci_highs = [
        max(0.0, float(cell_reports[c]["bargaining_w1"]["ci_95"][1]) - w1_vals[i])
        for i, c in enumerate(cell_ids)
    ]

    palette = []
    for cid in cell_ids:
        if cid == "C111":
            palette.append("#1b9e77")
        elif cid.startswith("CTRL_"):
            palette.append("#d95f02")
        elif cid.startswith("BASE_"):
            palette.append("#7570b3")
        elif cid == "C000":
            palette.append("#666666")
        else:
            palette.append("#386cb0")

    y_pos = np.arange(len(cell_ids))
    ax_left.barh(
        y_pos,
        w1_vals,
        xerr=[ci_lows, ci_highs],
        color=palette,
        alpha=0.88,
        capsize=3.5,
        edgecolor="#222222",
        linewidth=0.7,
    )
    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels(cell_ids, fontsize=9.5)
    ax_left.invert_yaxis()
    ax_left.set_xlabel("Debiased 1-Wasserstein Distance $W_1$ (Lower = Closer to Human Curve)")
    ax_left.set_title("(A) $2^3$ Factorial Cells & Matched Controls (95% Clustered CI)")

    # Overlay empirical Human SE noise ceiling and ZI null floor if present
    human_ceil = float(np.mean(HUMAN_BARGAINING_SE))
    ax_left.axvline(
        human_ceil, color="#1b9e77", linestyle=":", linewidth=1.6, label=f"Human SE Ceiling ({human_ceil:.3f})"
    )
    if "BASE_ZERO_INTELLIGENCE" in cell_reports:
        zi_w1 = float(cell_reports["BASE_ZERO_INTELLIGENCE"]["bargaining_w1"]["debiased_w1"])
        ax_left.axvline(
            zi_w1, color="#e7298a", linestyle="--", linewidth=1.4, label=f"ZI Null Floor ({zi_w1:.3f})"
        )
    ax_left.legend(loc="lower right", fontsize=8.5)

    # Panel B: 2^3 Main Effects, Interactions & Exact Shapley Attribution
    if factorial and factorial.get("complete_2x2x2_lattice"):
        labels_b = [
            "Shapley: Delib (A)",
            "Shapley: MemFric (B)",
            "Shapley: RepE (C)",
            "Interact: A×B",
            "Interact: A×C",
            "Interact: B×C",
            "Interact: A×B×C",
        ]
        shap = factorial["exact_shapley_values"]
        inter = factorial["interactions"]
        vals_b = [
            float(shap["A_Bounded_Deliberation"]),
            float(shap["B_Memory_Friction"]),
            float(shap["C_Endogenous_RepE"]),
            float(inter["AB"]),
            float(inter["AC"]),
            float(inter["BC"]),
            float(inter["ABC"]),
        ]
        colors_b = ["#1b9e77" if v >= 0 else "#d95f02" for v in vals_b]
        yb = np.arange(len(labels_b))
        ax_right.barh(yb, vals_b, color=colors_b, alpha=0.88, edgecolor="#222222", linewidth=0.7)
        ax_right.axvline(0.0, color="#222222", linewidth=1.0)
        ax_right.set_yticks(yb)
        ax_right.set_yticklabels(labels_b, fontsize=9.5)
        ax_right.invert_yaxis()
        ax_right.set_xlabel("Alignment Gain ($-\\Delta W_1$; Positive = Improvement)")
        ax_right.set_title("(B) Exact $2^3$ Shapley Credit & Interaction Contrasts")
    else:
        ax_right.text(
            0.5,
            0.5,
            "Run --cells FACTORIAL_8 or ALL\nto populate complete 2^3 Shapley & Interactions.",
            ha="center",
            va="center",
            fontsize=10,
            color="#555555",
        )
        ax_right.set_axis_off()

    _add_watermark_if_synthetic(fig, is_synthetic)
    plt.tight_layout()
    out_png = os.path.join(out_dir, "fig1_factorial_2x2x2_ablation.png")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_fig2_conditional_bargaining_and_mpc_curves(
    report: Dict[str, Any],
    out_dir: str,
    is_synthetic: bool = False,
) -> str:
    """
    Figure 2:
      - Left: CDA Bargaining / Ultimatum Acceptance Gradient across 4 Markup Bins
        vs. Oosterbeek et al. (2004) Human Curve and +/- SESOI Equivalence Band.
      - Right: Exogenous-Windfall Causal MPC across 3 Liquid-Wealth Terciles
        vs. Jappelli & Pistaferri (2014) Human Curve and +/- SESOI Band.
    """
    cell_reports: Dict[str, Dict[str, Any]] = report["cell_reports"]
    highlight_cells = [
        c
        for c in ("C111", "C000", "CTRL_PLACEBO_VEC", "BASE_ZERO_INTELLIGENCE", "BASE_CLOSED_FORM")
        if c in cell_reports
    ]
    if not highlight_cells:
        highlight_cells = list(cell_reports.keys())[:4]

    style_map = {
        "C111": ("#1b9e77", "-o", "Neuropolis Full (C111)"),
        "C000": ("#386cb0", "--s", "Vanilla LLM (C000)"),
        "CTRL_PLACEBO_VEC": ("#d95f02", ":^", "Placebo Vector Control"),
        "BASE_ZERO_INTELLIGENCE": ("#7570b3", "-.d", "Zero-Intelligence CDA"),
        "BASE_CLOSED_FORM": ("#666666", "--x", "Closed-Form Benchmark"),
    }

    fig, (ax_barg, ax_mpc) = plt.subplots(1, 2, figsize=(13.2, 5.0))

    # Left Panel: Bargaining Acceptance Curve
    x_b = np.arange(len(HUMAN_BARGAINING_BIN_LABELS))
    h_b = np.asarray(HUMAN_BARGAINING_ACCEPT_CURVE, dtype=np.float64)
    sesoi_b = float(
        cell_reports[highlight_cells[0]]["bargaining_per_bin_tost"][0]["sesoi_margin_delta"]
    )
    ax_barg.fill_between(
        x_b,
        np.clip(h_b - sesoi_b, 0.0, 1.0),
        np.clip(h_b + sesoi_b, 0.0, 1.0),
        color="#222222",
        alpha=0.14,
        label=f"Human SESOI Band ($\\pm${sesoi_b:.2f})",
    )
    ax_barg.plot(
        x_b, h_b, color="#111111", linewidth=2.4, marker="*", markersize=9, label="Human Meta-Analysis (Oosterbeek 2004)"
    )

    for cid in highlight_cells:
        bins = cell_reports[cid]["bargaining_per_bin_tost"]
        means = [float(b["sim_cluster_mean"]) for b in bins]
        err_lo = [max(0.0, means[i] - float(bins[i]["ci_95_lower"])) for i in range(len(bins))]
        err_hi = [max(0.0, float(bins[i]["ci_95_upper"]) - means[i]) for i in range(len(bins))]
        col, fmt, lbl = style_map.get(cid, ("#444444", "-o", cid))
        ax_barg.errorbar(
            x_b, means, yerr=[err_lo, err_hi], fmt=fmt, color=col, linewidth=1.8, capsize=3, label=lbl
        )

    ax_barg.set_xticks(x_b)
    ax_barg.set_xticklabels(["Fair\n(<=5%)", "Moderate\n(5–15%)", "Unfair\n(15–30%)", "Extreme\n(>30%)"])
    ax_barg.set_ylim(-0.02, 1.05)
    ax_barg.set_ylabel("CDA Acceptance / Order Fill Probability")
    ax_barg.set_title("(A) Conditional Bargaining Acceptance Curve")
    ax_barg.legend(loc="lower left", fontsize=8.0)

    # Right Panel: Exogenous-Windfall Causal MPC Curve
    x_m = np.arange(len(HUMAN_MPC_TERCILE_LABELS))
    h_m = np.asarray(HUMAN_MPC_CURVE, dtype=np.float64)
    first_mpc_bins = cell_reports[highlight_cells[0]].get("windfall_mpc_per_bin_tost", [])
    sesoi_m = float(first_mpc_bins[0]["sesoi_margin_delta"]) if first_mpc_bins else 0.06

    ax_mpc.fill_between(
        x_m,
        np.clip(h_m - sesoi_m, 0.0, 1.0),
        np.clip(h_m + sesoi_m, 0.0, 1.0),
        color="#222222",
        alpha=0.14,
        label=f"Human SESOI Band ($\\pm${sesoi_m:.2f})",
    )
    ax_mpc.plot(
        x_m, h_m, color="#111111", linewidth=2.4, marker="*", markersize=9, label="Human Windfall MPC (Jappelli 2014)"
    )

    for cid in highlight_cells:
        bins = cell_reports[cid].get("windfall_mpc_per_bin_tost", [])
        if not bins:
            continue
        means = [float(b["sim_cluster_mean"]) for b in bins]
        err_lo = [max(0.0, means[i] - float(bins[i]["ci_95_lower"])) for i in range(len(bins))]
        err_hi = [max(0.0, float(bins[i]["ci_95_upper"]) - means[i]) for i in range(len(bins))]
        col, fmt, lbl = style_map.get(cid, ("#444444", "-o", cid))
        ax_mpc.errorbar(
            x_m, means, yerr=[err_lo, err_hi], fmt=fmt, color=col, linewidth=1.8, capsize=3, label=lbl
        )

    ax_mpc.set_xticks(x_m)
    ax_mpc.set_xticklabels(["Low Liquidity\n(Tercile 1)", "Mid Liquidity\n(Tercile 2)", "High Liquidity\n(Tercile 3)"])
    ax_mpc.set_ylim(-0.05, 1.00)
    ax_mpc.set_ylabel("Exogenous-Windfall Causal MPC ($\\Delta C / \\Delta Y$)")
    ax_mpc.set_title("(B) Cross-Sectional Windfall MPC Heterogeneity")
    ax_mpc.legend(loc="upper right", fontsize=8.0)

    _add_watermark_if_synthetic(fig, is_synthetic)
    plt.tight_layout()
    out_png = os.path.join(out_dir, "fig2_bargaining_and_mpc_curves.png")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_fig3_per_bin_tost_equivalence_forest(
    report: Dict[str, Any],
    out_dir: str,
    target_cell: str = "C111",
    is_synthetic: bool = False,
) -> str:
    """
    Figure 3: Per-Bin TOST (Intersection-Union Test) Equivalence Forest Plot.
    Plots exact (Sim_Cluster_Mean - Human_Ref) with bold 90% Clustered CIs (TOST alpha=0.05)
    and thin 95% Clustered CIs against the shaded a priori [-SESOI, +SESOI] equivalence zone.
    """
    cell_reports = report["cell_reports"]
    if target_cell not in cell_reports:
        target_cell = list(cell_reports.keys())[-1]
    rep = cell_reports[target_cell]

    all_bins: List[Dict[str, Any]] = []
    all_bins.extend(rep.get("bargaining_per_bin_tost", []))
    all_bins.extend(rep.get("htm_per_bin_tost", []))
    all_bins.extend(rep.get("windfall_mpc_per_bin_tost", []))

    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.52 * len(all_bins) + 1.4)))
    y_pos = np.arange(len(all_bins))

    for idx, b in enumerate(all_bins):
        diff = float(b["mean_difference"])
        ref = float(b["human_reference_mean"])
        delta = float(b["sesoi_margin_delta"])
        ci90_lo = float(b["ci_90_lower"]) - ref
        ci90_hi = float(b["ci_90_upper"]) - ref
        ci95_lo = float(b["ci_95_lower"]) - ref
        ci95_hi = float(b["ci_95_upper"]) - ref
        equiv = bool(b["statistically_equivalent"])
        valid_cov = bool(b["bin_coverage_valid"])
        p_adj = b.get("holm_adjusted_p", b["tost_p_max"])

        # Per-row shaded SESOI equivalence box [-delta, +delta]
        ax.barh(idx, 2.0 * delta, left=-delta, height=0.68, color="#b2df8a", alpha=0.35, edgecolor="none")
        ax.plot([-delta, -delta], [idx - 0.34, idx + 0.34], color="#1b9e77", linestyle=":", linewidth=1.0)
        ax.plot([delta, delta], [idx - 0.34, idx + 0.34], color="#1b9e77", linestyle=":", linewidth=1.0)

        col = "#1b9e77" if equiv else ("#e7298a" if not valid_cov else "#d95f02")
        # Thin 95% CI whisker + thick 90% TOST CI bar
        ax.plot([ci95_lo, ci95_hi], [idx, idx], color=col, linewidth=1.3, alpha=0.85)
        ax.plot([ci90_lo, ci90_hi], [idx, idx], color=col, linewidth=4.0, solid_capstyle="round")
        ax.scatter([diff], [idx], color="#111111", s=26, zorder=5)

        tag = "EQUIV" if equiv else ("UNDERFILLED" if not valid_cov else "NON-EQ")
        ax.text(
            max(ci95_hi, delta) + 0.015,
            idx,
            f"p_Holm={p_adj:.3f} [{tag}] (n={b['n_observations_in_bin']})",
            va="center",
            fontsize=8.2,
            color=col,
            fontweight="bold" if equiv else "normal",
        )

    ax.axvline(0.0, color="#222222", linewidth=1.0, linestyle="--")
    y_labels = [
        f"{b['benchmark_name'].split('_')[0]} :: {b['bin_label']}" for b in all_bins
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=8.8)
    ax.invert_yaxis()
    ax.set_xlabel("Cluster Mean Difference ($\\hat{\\mu}_b - R_{\\mathrm{Human}, b}$) with Bold 90% TOST CI & Thin 95% CI")
    ax.set_title(f"Per-Bin TOST Equivalence Analysis — Condition `{target_cell}` (Shaded = A Priori $\\pm$SESOI Margin)")
    ax.set_xlim(-0.45, 0.65)

    _add_watermark_if_synthetic(fig, is_synthetic)
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"fig3_per_bin_tost_forest_{target_cell}.png")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_fig4_two_asset_htm_distribution(
    report: Dict[str, Any],
    out_dir: str,
    is_synthetic: bool = False,
) -> str:
    """
    Figure 4: Stacked Two-Asset Hand-to-Mouth Distribution (Poor-HtM, Wealthy-HtM, Non-HtM)
    compared directly against Kaplan, Violante, & Weidner (2014) US SCF benchmark.
    """
    cell_reports = report["cell_reports"]
    labels = ["Human SCF\n(KVW 2014)"] + list(cell_reports.keys())
    poor_vals = [HUMAN_HTM_DISTRIBUTION[0]]
    wealthy_vals = [HUMAN_HTM_DISTRIBUTION[1]]
    non_vals = [HUMAN_HTM_DISTRIBUTION[2]]

    for cid in cell_reports:
        bins = {b["bin_label"]: float(b["sim_cluster_mean"]) for b in cell_reports[cid]["htm_per_bin_tost"]}
        poor_vals.append(bins.get("Poor_HtM", 0.0))
        wealthy_vals.append(bins.get("Wealthy_HtM", 0.0))
        non_vals.append(bins.get("Non_HtM", 1.0))

    x = np.arange(len(labels))
    p_arr = np.asarray(poor_vals)
    w_arr = np.asarray(wealthy_vals)
    n_arr = np.asarray(non_vals)

    fig, ax = plt.subplots(figsize=(max(8.5, 1.05 * len(labels)), 4.8))
    ax.bar(x, p_arr, label="Poor Hand-to-Mouth (P-HtM)", color="#e7298a", alpha=0.88, edgecolor="#222222")
    ax.bar(x, w_arr, bottom=p_arr, label="Wealthy Hand-to-Mouth (W-HtM)", color="#d95f02", alpha=0.88, edgecolor="#222222")
    ax.bar(x, n_arr, bottom=p_arr + w_arr, label="Non-HtM (Liquid Savers)", color="#1b9e77", alpha=0.80, edgecolor="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8.8)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Steady-State Household Share")
    ax.set_title("Two-Asset Household Liquidity Composition vs. KVW (2014) US SCF Benchmark")
    ax.legend(loc="upper right", ncol=3, fontsize=8.5)

    _add_watermark_if_synthetic(fig, is_synthetic)
    plt.tight_layout()
    out_png = os.path.join(out_dir, "fig4_two_asset_htm_distribution.png")
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_png


def export_latex_summary_table(report: Dict[str, Any], out_dir: str, is_synthetic: bool) -> str:
    status_note = "SYNTHETIC PIPELINE TEST" if is_synthetic else "EXECUTED FROM CLUSTERED ARTIFACTS"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Neuropolis v2 $2^3$ Factorial Clustered Evaluation ({status_note})}}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Condition & $M$ & Valid \% & Freeze \% & Barg. $W_1$ (95\% CI) & HtM JSD & Max SFC Err \\",
        r"\midrule",
    ]
    for cid, rep in report["cell_reports"].items():
        w1 = rep["bargaining_w1"]
        jsd = rep["htm_miller_madow_jsd"]["debiased_jsd"]
        safe_cid = cid.replace("_", r"\_")
        lines.append(
            f"{safe_cid} & {rep['n_economy_seeds_m']} & {rep['json_valid_rate']*100:.1f}\\% & "
            f"{rep['market_freeze_rate']*100:.1f}\\% & "
            f"{w1['debiased_w1']:.3f} [{w1['ci_95'][0]:.3f}, {w1['ci_95'][1]:.3f}] & "
            f"{jsd:.4f} & {rep['max_sfc_conservation_error']:.1e} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out_tex = os.path.join(out_dir, "table_factorial_summary.tex")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_tex


def _build_synthetic_pipeline_test_report() -> Dict[str, Any]:
    """Strictly for unit-testing the plotting pipeline with visible SYNTHETIC watermark."""
    cells = [
        "BASE_ZERO_INTELLIGENCE",
        "C000", "C100", "C010", "C001", "C110", "C101", "C011", "C111",
        "CTRL_PLACEBO_VEC",
    ]
    w1_demo = {
        "BASE_ZERO_INTELLIGENCE": 0.28,
        "C000": 0.22, "C100": 0.19, "C010": 0.20, "C001": 0.14,
        "C110": 0.17, "C101": 0.11, "C011": 0.12, "C111": 0.08,
        "CTRL_PLACEBO_VEC": 0.18,
    }
    cell_reports: Dict[str, Any] = {}
    for cid in cells:
        w1 = w1_demo[cid]
        shift = 0.02 if cid == "C111" else (0.14 if cid == "BASE_ZERO_INTELLIGENCE" else 0.08)
        barg_bins = []
        for lbl, h_val in zip(HUMAN_BARGAINING_BIN_LABELS, HUMAN_BARGAINING_ACCEPT_CURVE):
            sim_m = round(max(0.02, min(0.98, h_val - shift)), 4)
            diff = round(sim_m - h_val, 4)
            barg_bins.append(
                {
                    "benchmark_name": "Oosterbeek_2004",
                    "bin_label": lbl,
                    "n_clusters_m": 30,
                    "n_observations_in_bin": 120,
                    "bin_coverage_valid": True,
                    "sim_cluster_mean": sim_m,
                    "human_reference_mean": h_val,
                    "mean_difference": diff,
                    "clustered_se": 0.015,
                    "ci_90_lower": round(sim_m - 0.025, 4),
                    "ci_90_upper": round(sim_m + 0.025, 4),
                    "ci_95_lower": round(sim_m - 0.030, 4),
                    "ci_95_upper": round(sim_m + 0.030, 4),
                    "sesoi_margin_delta": 0.08,
                    "tost_p_max": 0.008 if abs(diff) < 0.05 else 0.42,
                    "holm_adjusted_p": 0.016 if abs(diff) < 0.05 else 0.65,
                    "statistically_equivalent": abs(diff) < 0.05,
                }
            )
        htm_bins = [
            {
                "benchmark_name": "KVW_2014_HtM",
                "bin_label": lbl,
                "n_clusters_m": 30,
                "n_observations_in_bin": 90,
                "bin_coverage_valid": True,
                "sim_cluster_mean": h_val,
                "human_reference_mean": h_val,
                "mean_difference": 0.0,
                "clustered_se": 0.01,
                "ci_90_lower": h_val - 0.016,
                "ci_90_upper": h_val + 0.016,
                "ci_95_lower": h_val - 0.020,
                "ci_95_upper": h_val + 0.020,
                "sesoi_margin_delta": 0.05,
                "tost_p_max": 0.002,
                "holm_adjusted_p": 0.006,
                "statistically_equivalent": True,
            }
            for lbl, h_val in zip(HUMAN_HTM_LABELS, HUMAN_HTM_DISTRIBUTION)
        ]
        mpc_bins = [
            {
                "benchmark_name": "Jappelli_2014_MPC",
                "bin_label": lbl,
                "n_clusters_m": 30,
                "n_observations_in_bin": 80,
                "bin_coverage_valid": True,
                "sim_cluster_mean": round(max(0.05, h_val - 0.5 * shift), 4),
                "human_reference_mean": h_val,
                "mean_difference": round(-0.5 * shift, 4),
                "clustered_se": 0.014,
                "ci_90_lower": round(h_val - 0.5 * shift - 0.022, 4),
                "ci_90_upper": round(h_val - 0.5 * shift + 0.022, 4),
                "ci_95_lower": round(h_val - 0.5 * shift - 0.028, 4),
                "ci_95_upper": round(h_val - 0.5 * shift + 0.028, 4),
                "sesoi_margin_delta": 0.06,
                "tost_p_max": 0.012 if abs(0.5 * shift) < 0.03 else 0.50,
                "holm_adjusted_p": 0.024 if abs(0.5 * shift) < 0.03 else 0.70,
                "statistically_equivalent": abs(0.5 * shift) < 0.03,
            }
            for lbl, h_val in zip(HUMAN_MPC_TERCILE_LABELS, HUMAN_MPC_CURVE)
        ]
        cell_reports[cid] = {
            "n_economy_seeds_m": 30,
            "json_valid_rate": 0.98,
            "market_freeze_rate": 0.04,
            "survival_rate": 0.96,
            "max_sfc_conservation_error": 0.0,
            "bargaining_w1": {"debiased_w1": w1, "ci_95": (max(0.0, w1 - 0.02), w1 + 0.02)},
            "bargaining_per_bin_tost": barg_bins,
            "htm_miller_madow_jsd": {"debiased_jsd": round(0.15 * w1, 4)},
            "htm_per_bin_tost": htm_bins,
            "windfall_mpc_per_bin_tost": mpc_bins,
        }

    bits_map = {
        (a, b, c): -w1_demo[f"C{a}{b}{c}"]
        for a in (0, 1)
        for b in (0, 1)
        for c in (0, 1)
    }
    return {
        "cell_reports": cell_reports,
        "factorial_2x2x2_analysis": compute_2x2x2_factorial_effects_and_shapley(bits_map),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neuropolis v2 Thesis Publication Figure & LaTeX Table Generator"
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default="results/neuropolis_v2/factorial_evaluation_report.json",
        help="Path to `factorial_evaluation_report.json` generated by `run_mvp.py`.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/neuropolis_v2/figures",
        help="Directory to write 300-DPI PNG/PDF figures and LaTeX tables.",
    )
    parser.add_argument(
        "--target-tost-cell",
        type=str,
        default="C111",
        help="Cell ID to display in the Figure 3 Per-Bin TOST Forest Plot.",
    )
    parser.add_argument(
        "--synthetic-pipeline-test",
        action="store_true",
        help="Generate watermarked `SYNTHETIC - PIPELINE TEST` figures strictly to verify plotting code.",
    )
    args = parser.parse_args()

    _apply_publication_style()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.synthetic_pipeline_test:
        report = _build_synthetic_pipeline_test_report()
        is_synth = True
    else:
        if not os.path.exists(args.report_json):
            print("RESULTS_STATUS: NOT_EXECUTED")
            print(
                f"No executed report found at '{args.report_json}'. "
                f"Run `run_mvp.py` first, or pass `--synthetic-pipeline-test` to unit-test plotting."
            )
            return
        with open(args.report_json, "r", encoding="utf-8") as f:
            report = json.load(f)
        is_synth = False

    f1 = plot_fig1_factorial_ablation_and_shapley(report, args.output_dir, is_synthetic=is_synth)
    f2 = plot_fig2_conditional_bargaining_and_mpc_curves(report, args.output_dir, is_synthetic=is_synth)
    f3 = plot_fig3_per_bin_tost_equivalence_forest(
        report, args.output_dir, target_cell=args.target_tost_cell, is_synthetic=is_synth
    )
    f4 = plot_fig4_two_asset_htm_distribution(report, args.output_dir, is_synthetic=is_synth)
    tex = export_latex_summary_table(report, args.output_dir, is_synthetic=is_synth)

    print("[REPORTING] Generated publication artifacts:")
    for path in (f1, f2, f3, f4, tex):
        print(f"  - {path}")


if __name__ == "__main__":
    main()