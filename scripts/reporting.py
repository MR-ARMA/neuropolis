from __future__ import annotations

import csv
import glob
import json
import os
from typing import Any, Dict, List


def _find_scorecard_entry(scorecard: List[Dict[str, Any]], keyword: str) -> Dict[str, Any]:
    for entry in scorecard:
        if isinstance(entry, dict) and keyword.lower() in str(entry.get("benchmark", "")).lower():
            return entry
    return {}


def load_run_json_reports(
    search_patterns: tuple[str, ...] = (
        "outputs/**/*.json",
        "results/*manifest*.json",
    )
) -> List[Dict[str, Any]]:
    """
    Zero-regex JSON report loader reading run manifests/summaries safely with `.get()` defaults.
    """
    discovered_files: List[str] = []
    for pattern in search_patterns:
        for path in sorted(glob.glob(pattern, recursive=True)):
            if path not in discovered_files:
                discovered_files.append(path)

    rows: List[Dict[str, Any]] = []
    for fpath in discovered_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
        except Exception:
            continue

        macro = data.get("macro_summary", {})
        if not isinstance(macro, dict):
            macro = {}
        scorecard = data.get("scorecard_summary", [])
        if not isinstance(scorecard, list):
            scorecard = []

        b1 = _find_scorecard_entry(scorecard, "Oosterbeek")
        b2 = _find_scorecard_entry(scorecard, "Jappelli")
        m1 = _find_scorecard_entry(scorecard, "Kaplan")
        m2 = _find_scorecard_entry(scorecard, "Fehr-Gachter")

        p0 = float(macro.get("p_c_start", 12.0))
        p1 = float(macro.get("p_c_end", p0))
        infl_pct = round(((p1 - p0) / max(1.0, p0)) * 100.0, 2)

        rows.append(
            {
                "file": os.path.basename(fpath),
                "run_id": str(data.get("run_id", os.path.splitext(os.path.basename(fpath))[0])),
                "model": str(data.get("model_id", data.get("model", "Unknown"))).split("/")[-1],
                "regime": str(data.get("macro_regime_mode", data.get("regime", "crisis"))),
                "channel": str(data.get("state_channel", data.get("channel", "NONE"))),
                "steering": str(data.get("steering_method", data.get("steering", "NONE"))),
                "llm_valid_pct": round(float(data.get("llm_valid_rate", 0.0)) * 100.0, 1),
                "public_goods_provision": round(
                    float(macro.get("mean_pgpr", data.get("public_goods_provision", 0.0))) * 100.0,
                    2,
                ),
                "t_soc": round(float(macro.get("mean_t_soc", data.get("t_soc", 0.0))), 4),
                "rejection_rate": round(
                    float(macro.get("mean_rej_rate", data.get("rejection_rate", 0.0))) * 100.0,
                    2,
                ),
                "mpc_low": round(float(macro.get("mean_mpc_low", data.get("mpc_low", 0.0))), 4),
                "mpc_slope": round(float(b2.get("sim_mean", 0.0)), 4),
                "mpc_l1": round(float(b2.get("tvd_or_w1_dist", 0.0)), 4),
                "mpc_tost": "EQUIV" if b2.get("aligned", False) else "NON-EQ",
                "oosterbeek_l1": round(float(b1.get("tvd_or_w1_dist", 0.0)), 4),
                "oosterbeek_bins": "YES" if b1.get("bin_coverage_valid", False) else "UNDER",
                "oosterbeek_tost": "EQUIV" if b1.get("aligned", False) else "NON-EQ",
                "htm_pct": round(float(macro.get("mean_htm", m1.get("sim_mean", 0.0))) * 100.0, 2),
                "htm_tost": "EQUIV" if m1.get("aligned", False) else "NON-EQ",
                "pg_tost": "EQUIV" if m2.get("aligned", False) else "NON-EQ",
                "urgent_buys": int(macro.get("urgent_food_buys", data.get("urgent_buys", 0))),
                "payday_loans": int(
                    macro.get("payday_loan_cnt", macro.get("payday_loans_taken", data.get("payday_loans", 0)))
                ),
                "bankrupt_pct": round(float(macro.get("bankrupt_pct", data.get("bankrupt_pct", 0.0))), 2),
                "inflation_pct": infl_pct,
                "sfc_err": float(macro.get("max_sfc_err", data.get("sfc_err", 0.0))),
            }
        )
    return rows


def render_and_export_summary(
    csv_out: str = "outputs/parsed_benchmark_summary.csv",
) -> str:
    rows = load_run_json_reports()
    if not rows:
        return "No JSON report files found in `outputs/**/*.json` or `results/*manifest*.json`."

    os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    header = (
        "| Run ID | Model | Regime | Channel / Steering | PG % (TOST) | HtM % (TOST) | "
        "Oost. L1 (Bins, TOST) | MPC_Low / Slope (L1, TOST) | Rej % | T_soc | "
        "Urgent Buys | Payday Loans | Bankr % | Infl % | SFC_Err |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    )
    body_lines = []
    for r in rows:
        body_lines.append(
            f"| `{r['run_id']}` | `{r['model']}` | `{r['regime']}` | "
            f"`{r['channel']}`/`{r['steering']}` | "
            f"{r['public_goods_provision']:.1f}% ({r['pg_tost']}) | "
            f"{r['htm_pct']:.1f}% ({r['htm_tost']}) | "
            f"{r['oosterbeek_l1']:.3f} ({r['oosterbeek_bins']}, {r['oosterbeek_tost']}) | "
            f"{r['mpc_low']:.2f} / {r['mpc_slope']:+.2f} ({r['mpc_l1']:.3f}, {r['mpc_tost']}) | "
            f"{r['rejection_rate']:.1f}% | {r['t_soc']:.3f} | "
            f"{r['urgent_buys']} | {r['payday_loans']} | "
            f"{r['bankrupt_pct']:.1f}% | {r['inflation_pct']:+.1f}% | {r['sfc_err']:.1e} |"
        )
    return header + "\n".join(body_lines) + "\n"


if __name__ == "__main__":
    print(render_and_export_summary())