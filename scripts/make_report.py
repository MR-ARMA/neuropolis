from __future__ import annotations

import csv
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple


SCALE_NORM_REF = {
    "oosterbeek": 0.45,
    "mpc": 0.35,
    "htm": 0.35,
    "pgpr": 0.35,
}


def _closeness_pct(l1_dist: float, key: str) -> float:
    norm = SCALE_NORM_REF.get(key, 0.35)
    return round(max(0.0, 100.0 * (1.0 - float(l1_dist) / norm)), 1)


def _extract_row_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    sc = m.get("scorecard_summary", [])
    mac = m.get("macro_summary", {})

    def _find_sc(substr: str) -> Dict[str, Any]:
        for item in sc:
            if substr.lower() in str(item.get("benchmark", "")).lower():
                return item
        return {}

    b1 = _find_sc("Oosterbeek")
    b2 = _find_sc("Jappelli")
    m1 = _find_sc("Kaplan")
    m2 = _find_sc("Fehr-Gachter")

    b1_l1 = float(b1.get("tvd_or_w1_dist", 0.35))
    b2_l1 = float(b2.get("tvd_or_w1_dist", 0.35))
    m1_l1 = float(m1.get("tvd_or_w1_dist", 0.35))
    m2_l1 = float(m2.get("tvd_or_w1_dist", 0.35))

    c_b1 = _closeness_pct(b1_l1, "oosterbeek")
    c_b2 = _closeness_pct(b2_l1, "mpc")
    c_m1 = _closeness_pct(m1_l1, "htm")
    c_m2 = _closeness_pct(m2_l1, "pgpr")
    composite = round(0.25 * (c_b1 + c_b2 + c_m1 + c_m2), 1)

    p0 = float(mac.get("p_c_start", 12.0))
    p1 = float(mac.get("p_c_end", 12.0))
    infl_pct = round(((p1 - p0) / max(1.0, p0)) * 100.0, 1)

    return {
        "run_id": str(m.get("run_id", "run")),
        "model": str(m.get("model_id", "unknown")).split("/")[-1],
        "scale": str(m.get("population_scale", "dev")),
        "regime": str(m.get("macro_regime_mode", "crisis")),
        "seed": int(m.get("master_seed", 42)),
        "mode": str(m.get("decision_mode", "")),
        "channel": str(m.get("state_channel", "")),
        "steering": str(m.get("steering_method", "")),
        "alpha": float(m.get("alpha_dose", 3.0)),
        "valid_pct": round(float(m.get("llm_valid_rate", 0.0)) * 100.0, 1),
        "abstain_pct": round(float(m.get("abstain_rate", 0.0)) * 100.0, 1),
        "b1_sim": float(b1.get("sim_mean", 0.0)),
        "b1_l1": b1_l1,
        "b1_close": c_b1,
        "b1_bins": "YES" if b1.get("bin_coverage_valid", False) else "UNDER",
        "b1_tost": "EQUIV" if b1.get("aligned", False) else "NON-EQ",
        "b2_sim": float(b2.get("sim_mean", 0.0)),
        "b2_l1": b2_l1,
        "b2_close": c_b2,
        "b2_tost": "EQUIV" if b2.get("aligned", False) else "NON-EQ",
        "m1_sim": float(m1.get("sim_mean", mac.get("mean_htm", 0.0))),
        "m1_l1": m1_l1,
        "m1_close": c_m1,
        "m1_tost": "EQUIV" if m1.get("aligned", False) else "NON-EQ",
        "m2_sim": float(m2.get("sim_mean", mac.get("mean_pgpr", 0.0))),
        "m2_l1": m2_l1,
        "m2_close": c_m2,
        "m2_tost": "EQUIV" if m2.get("aligned", False) else "NON-EQ",
        "mean_t_soc": float(mac.get("mean_t_soc", 0.0)),
        "mean_rej": float(mac.get("mean_rej_rate", 0.0)),
        "mean_mpc_low": float(mac.get("mean_mpc_low", 0.0)),
        "infl_pct": infl_pct,
        "sanctions": int(mac.get("total_sanctions", 0)),
        "sfc_err": float(mac.get("max_sfc_err", 0.0)),
        "composite_close": composite,
        "tost_passes": sum(
            int(bool(x.get("aligned", False))) for x in (b1, b2, m1, m2)
        ),
    }


def _load_all_manifests(results_dir: str) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(results_dir, "*manifest*.json")))
    records: List[Dict[str, Any]] = []
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                records.append(_extract_row_metrics(json.load(f)))
        except Exception:
            continue
    return records


def generate_markdown_report(results_dir: str = "results") -> str:
    records = _load_all_manifests(results_dir)
    if not records:
        return (
            "# Neuropolis Master Empirical Benchmark & Ablation Report\n\n"
            "`RESULTS_STATUS: NOT_EXECUTED`\n\n"
            "## Table 1: Master Empirical Alignment\n\n"
            "| Condition | Decision Mode | State Channel | Steering | Valid % | Oosterbeek L1 | MPC Slope | HtM % | PG % | Composite % |\n"
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
            "| All Pre-Registered Conditions | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |\n"
        )

    lines: List[str] = [
        "# Neuropolis Master Empirical Benchmark & Mechanism Attribution Report",
        "",
        "`RESULTS_STATUS: EXECUTED_FROM_ARTIFACTS`",
        "",
        "## Table 1 — Master Empirical Alignment Table (vs. Published Scientific Literature)",
        "",
        "| Run ID | Model | Scale | Regime | Mode / Channel / Steering | α | Valid % | Oosterbeek (2004) Sim (L1, Close%, Bins, TOST) | Jappelli-Pistaferri (2014) MPC (L1, Close%, TOST) | KVW (2014) HtM vs 33.3% (L1, Close%, TOST) | Fehr-Gächter (2000) PG vs 75–80% (L1, Close%, TOST) | T_soc | SFC_Err | Composite % |",
        "| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in records:
        cfg_str = f"`{r['mode']}` / `{r['channel']}` / `{r['steering']}`"
        oost_str = f"{r['b1_sim']:.2f} ({r['b1_l1']:.3f}, {r['b1_close']:.1f}%, {r['b1_bins']}, {r['b1_tost']})"
        mpc_str = f"{r['b2_sim']:+.2f} ({r['b2_l1']:.3f}, {r['b2_close']:.1f}%, {r['b2_tost']})"
        htm_str = f"{r['m1_sim']*100:.1f}% ({r['m1_l1']:.3f}, {r['m1_close']:.1f}%, {r['m1_tost']})"
        pg_str = f"{r['m2_sim']*100:.1f}% ({r['m2_l1']:.3f}, {r['m2_close']:.1f}%, {r['m2_tost']})"
        lines.append(
            f"| `{r['run_id']}` | `{r['model']}` | `{r['scale']}` | `{r['regime']}` | {cfg_str} | "
            f"{r['alpha']:.1f} | {r['valid_pct']:.1f}% | {oost_str} | {mpc_str} | {htm_str} | {pg_str} | "
            f"{r['mean_t_soc']:.3f} | {r['sfc_err']:.1e} | **{r['composite_close']:.1f}%** |"
        )
    lines.append(
        "\n*Footnote: `Closeness % = max(0.0, 100.0 * (1.0 - L1_dist / scale_norm))`, where `scale_norm` is `0.45` (Oosterbeek) and `0.35` (MPC, HtM, PG).*\n"
    )

    # Table 2 — Mechanism & Ablation Marginal Effect Analysis
    lines.extend(
        [
            "## Table 2 — Mechanism & Ablation Marginal Effect Analysis (Paired Contrasts)",
            "",
            "| Contrast (Treatment vs. Reference) | Model / Scale / Regime | ΔAccept | ΔOosterbeek L1 | ΔMPC Slope | ΔHtM | ΔPG % | ΔT_soc | ΔComposite % |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    groups: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((r["model"], r["scale"], r["regime"], r["seed"]), []).append(r)

    pair_specs = [
        ("OUR_COUPLED_STEER vs TEXT_ONLY", ("LLM_AUTONOMOUS", "TEXT_AND_STEER", "OUR_COUPLED_STEER"), ("LLM_AUTONOMOUS", "TEXT_ONLY", "NONE")),
        ("STEER_ONLY vs TEXT_AND_STEER", ("LLM_AUTONOMOUS", "STEER_ONLY", "OUR_COUPLED_STEER"), ("LLM_AUTONOMOUS", "TEXT_AND_STEER", "OUR_COUPLED_STEER")),
        ("CUE_CONFLICT vs TEXT_AND_STEER", ("LLM_AUTONOMOUS", "CUE_CONFLICT", "OUR_COUPLED_STEER"), ("LLM_AUTONOMOUS", "TEXT_AND_STEER", "OUR_COUPLED_STEER")),
        ("PASTA vs OUR_COUPLED_STEER", ("LLM_AUTONOMOUS", "TEXT_AND_STEER", "PASTA"), ("LLM_AUTONOMOUS", "TEXT_AND_STEER", "OUR_COUPLED_STEER")),
        ("ACTADD_CAA vs OUR_COUPLED_STEER", ("LLM_AUTONOMOUS", "TEXT_AND_STEER", "ACTADD_CAA"), ("LLM_AUTONOMOUS", "TEXT_AND_STEER", "OUR_COUPLED_STEER")),
        ("PLACEBO_RANDOM vs TEXT_ONLY", ("LLM_AUTONOMOUS", "TEXT_AND_STEER", "PLACEBO_RANDOM"), ("LLM_AUTONOMOUS", "TEXT_ONLY", "NONE")),
        ("RULE_ONLY vs LLM_AUTONOMOUS", ("RULE_ONLY", None, None), ("LLM_AUTONOMOUS", "TEXT_AND_STEER", "OUR_COUPLED_STEER")),
    ]

    def _match_spec(rlist: List[Dict[str, Any]], spec: Tuple[Optional[str], Optional[str], Optional[str]]) -> Optional[Dict[str, Any]]:
        for r in rlist:
            if spec[0] and r["mode"] != spec[0]:
                continue
            if spec[1] and r["channel"] != spec[1]:
                continue
            if spec[2] and r["steering"] != spec[2]:
                continue
            return r
        return None

    t2_count = 0
    for (mod, sc_lbl, reg, sd), rlist in groups.items():
        for label, tr_spec, ref_spec in pair_specs:
            tr = _match_spec(rlist, tr_spec)
            rf = _match_spec(rlist, ref_spec)
            if tr and rf:
                t2_count += 1
                lines.append(
                    f"| {label} | `{mod}` / `{sc_lbl}` / `{reg}` (s{sd}) | "
                    f"{tr['b1_sim'] - rf['b1_sim']:+.3f} | {tr['b1_l1'] - rf['b1_l1']:+.3f} | "
                    f"{tr['b2_sim'] - rf['b2_sim']:+.3f} | {(tr['m1_sim'] - rf['m1_sim'])*100:+.1f}pp | "
                    f"{(tr['m2_sim'] - rf['m2_sim'])*100:+.1f}pp | {tr['mean_t_soc'] - rf['mean_t_soc']:+.3f} | "
                    f"{tr['composite_close'] - rf['composite_close']:+.1f}pp |"
                )
    if t2_count == 0:
        lines.append("| *No matched treatment-vs-reference pairs executed yet* | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |")

    # Table 3 — Macro Regime Sensitivity (Normal vs Crisis)
    lines.extend(
        [
            "",
            "## Table 3 — Macro Regime Sensitivity (Crisis vs. Normal Baseline)",
            "",
            "| Model / Scale / Condition | ΔRejRate | Δp_c Inflation % | ΔHtM | ΔMPC_Low | ΔPG % | ΔT_soc | ΔSanctions | ΔComposite % |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    reg_groups: Dict[Tuple[str, str, str, str, str, int], Dict[str, Dict[str, Any]]] = {}
    for r in records:
        k = (r["model"], r["scale"], r["mode"], r["channel"], r["steering"], r["seed"])
        reg_groups.setdefault(k, {})[r["regime"]] = r

    t3_count = 0
    for k, pair in reg_groups.items():
        if "crisis" in pair and "normal" in pair:
            c, n = pair["crisis"], pair["normal"]
            t3_count += 1
            lines.append(
                f"| `{k[0]}` / `{k[1]}` / `{k[3]}`+`{k[4]}` | "
                f"{(c['mean_rej'] - n['mean_rej'])*100:+.1f}pp | {c['infl_pct'] - n['infl_pct']:+.1f}pp | "
                f"{(c['m1_sim'] - n['m1_sim'])*100:+.1f}pp | {c['mean_mpc_low'] - n['mean_mpc_low']:+.3f} | "
                f"{(c['m2_sim'] - n['m2_sim'])*100:+.1f}pp | {c['mean_t_soc'] - n['mean_t_soc']:+.3f} | "
                f"{c['sanctions'] - n['sanctions']:+d} | {c['composite_close'] - n['composite_close']:+.1f}pp |"
            )
    if t3_count == 0:
        lines.append("| *No paired normal-vs-crisis runs executed yet* | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |")

    # Table 4 — Cross-Scale & Cross-Model Robustness
    lines.extend(
        [
            "",
            "## Table 4 — Cross-Scale & Cross-Model Robustness",
            "",
            "| Model | Scale | Runs | Mean Composite % | BinsOK Rate | Mean TOST Passes (/4) | Max SFC_Err |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    rob_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in records:
        rob_groups.setdefault((r["model"], r["scale"]), []).append(r)
    for (mod, sc_lbl), rlist in sorted(rob_groups.items()):
        n_r = len(rlist)
        m_comp = sum(x["composite_close"] for x in rlist) / n_r
        b_ok = sum(1 for x in rlist if x["b1_bins"] == "YES") / n_r * 100.0
        m_tost = sum(x["tost_passes"] for x in rlist) / n_r
        max_sfc = max(x["sfc_err"] for x in rlist)
        lines.append(
            f"| `{mod}` | `{sc_lbl}` | {n_r} | **{m_comp:.1f}%** | {b_ok:.0f}% | {m_tost:.2f} / 4 | {max_sfc:.1e} |"
        )

    return "\n".join(lines) + "\n"


def generate_latex_table(results_dir: str = "results") -> str:
    records = _load_all_manifests(results_dir)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Neuropolis Master Empirical Alignment Report (Auto-Generated)}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllrrrrrr}",
        r"\toprule",
        r"Model & Scale & Regime & Condition & Valid\% & Oost. $L_1$ & MPC & HtM\% & PG\% & Comp.\% \\",
        r"\midrule",
    ]
    if not records:
        lines.append(r"\multicolumn{10}{c}{\texttt{RESULTS\_STATUS: NOT\_EXECUTED (TBD)}} \\")
    else:
        for r in records:
            cond = f"{r['channel']}/{r['steering']}".replace("_", r"\_")
            mod = r["model"].replace("_", r"\_")
            sc = r["scale"].replace("_", r"\_")
            lines.append(
                f"{mod} & {sc} & {r['regime']} & {cond} & {r['valid_pct']:.1f}\\% & "
                f"{r['b1_l1']:.3f} & {r['b2_sim']:+.2f} & {r['m1_sim']*100:.1f}\\% & "
                f"{r['m2_sim']*100:.1f}\\% & {r['composite_close']:.1f}\\% \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def export_csv_report(results_dir: str = "results", csv_path: str = "results/MASTER_BENCHMARK_REPORT.csv") -> None:
    records = _load_all_manifests(results_dir)
    if not records:
        return
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    md_out = generate_markdown_report("results")
    tex_out = generate_latex_table("results")
    for fname in ("results/MASTER_BENCHMARK_REPORT.md", "results/REPORT_TABLES.md"):
        with open(fname, "w", encoding="utf-8") as f:
            f.write(md_out)
    for fname in ("results/MASTER_BENCHMARK_REPORT.tex", "results/REPORT_TABLES.tex"):
        with open(fname, "w", encoding="utf-8") as f:
            f.write(tex_out)
    export_csv_report("results", "results/MASTER_BENCHMARK_REPORT.csv")
    print(md_out)