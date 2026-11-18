from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List


def generate_markdown_report(results_dir: str = "results") -> str:
    """
    Section 0 & Defect D14 Automated Report Generator:
    Reads stored `manifest.json` and scorecard artifacts from `results_dir`.
    If no executed confirmatory artifacts exist, stamps `RESULTS_STATUS: NOT_EXECUTED`
    and fills cells with `TBD` (never fabricating or hard-coding pilot numbers).
    """
    manifest_files = sorted(glob.glob(os.path.join(results_dir, "*manifest*.json")))
    if not manifest_files:
        return (
            "# Neuropolis Automated Benchmark & Ablation Report\n\n"
            "`RESULTS_STATUS: NOT_EXECUTED`\n\n"
            "## 1. Confirmatory Ablation Summary Table (Pre-Registered Contrasts)\n\n"
            "| Condition / Component | Decision Mode | State Channel | Steering Method | LLM Valid % | B1 Ultimatum W1 (95% CI) | B3 MPC Slope (95% CI) | TOST Verdict |\n"
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
            "| C0: Null State Baseline | `LLM_AUTONOMOUS` | `NONE` | `NONE` | `TBD` | `TBD` | `TBD` | `TBD` |\n"
            "| C1: Prompt Text Only | `LLM_AUTONOMOUS` | `TEXT_ONLY` | `NONE` | `TBD` | `TBD` | `TBD` | `TBD` |\n"
            "| C2: Steering Only | `LLM_AUTONOMOUS` | `STEER_ONLY` | `OUR_COUPLED_STEER` | `TBD` | `TBD` | `TBD` | `TBD` |\n"
            "| C3: Coupled Text + Steer | `LLM_AUTONOMOUS` | `TEXT_AND_STEER` | `OUR_COUPLED_STEER` | `TBD` | `TBD` | `TBD` | `TBD` |\n"
            "| C4: Cue Conflict (2x2) | `LLM_AUTONOMOUS` | `CUE_CONFLICT` | `OUR_COUPLED_STEER` | `TBD` | `TBD` | `TBD` | `TBD` |\n"
            "| C5: KL-Matched Placebo | `LLM_AUTONOMOUS` | `TEXT_AND_STEER` | `PLACEBO_RANDOM` | `TBD` | `TBD` | `TBD` | `TBD` |\n"
            "| C6: Shuffled State | `LLM_AUTONOMOUS` | `TEXT_AND_STEER` | `SHUFFLED_STATE` | `TBD` | `TBD` | `TBD` | `TBD` |\n"
            "| C8a: PASTA Baseline | `LLM_AUTONOMOUS` | `TEXT_AND_STEER` | `PASTA` | `TBD` | `TBD` | `TBD` | `TBD` |\n"
            "| C8b: ActAdd/CAA Baseline | `LLM_AUTONOMOUS` | `TEXT_AND_STEER` | `ACTADD_CAA` | `TBD` | `TBD` | `TBD` | `TBD` |\n"
            "| C10a: Rule-Only Baseline | `RULE_ONLY` | `NONE` | `NONE` | `N/A` | `TBD` | `TBD` | `TBD` |\n"
            "| C10b: Zero-Intelligence | `ZERO_INTELLIGENCE` | `NONE` | `NONE` | `N/A` | `TBD` | `TBD` | `TBD` |\n"
        )

    rows_md: List[str] = []
    for mf in manifest_files:
        with open(mf, "r", encoding="utf-8") as f:
            m: Dict[str, Any] = json.load(f)
        rows_md.append(
            f"| `{m.get('run_id', 'run')}` | `{m.get('decision_mode')}` | "
            f"`{m.get('state_channel')}` | `{m.get('steering_method')}` | "
            f"`{float(m.get('llm_valid_rate', 0.0))*100:.1f}%` | "
            f"`{float(m.get('abstain_rate', 0.0))*100:.1f}%` | "
            f"`{m.get('git_commit_sha')}` |"
        )

    header = (
        "# Neuropolis Automated Benchmark & Ablation Report\n\n"
        "`RESULTS_STATUS: EXECUTED_FROM_ARTIFACTS`\n\n"
        "| Run ID | Decision Mode | State Channel | Steering Method | LLM Valid % | Abstain % | Git SHA |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    )
    return header + "\n".join(rows_md) + "\n"


def generate_latex_table(results_dir: str = "results") -> str:
    manifest_files = sorted(glob.glob(os.path.join(results_dir, "*manifest*.json")))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Neuropolis Confirmatory Ablation Results (Generated Automatically)}",
        r"\begin{tabular}{llllrrr}",
        r"\toprule",
        r"Condition & Mode & Channel & Steering & Valid \% & $W_1$ Dist & TOST $p$ \\",
        r"\midrule",
    ]
    if not manifest_files:
        lines.append(r"All Pre-Registered Contrasts & \multicolumn{6}{c}{\texttt{RESULTS\_STATUS: NOT\_EXECUTED (TBD)}} \\")
    else:
        for mf in manifest_files:
            with open(mf, "r", encoding="utf-8") as f:
                m = json.load(f)
            lines.append(
                f"{m.get('run_id')} & {m.get('decision_mode')} & {m.get('state_channel')} & "
                f"{m.get('steering_method')} & {float(m.get('llm_valid_rate', 0.0))*100:.1f}\\% & TBD & TBD \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    md_out = generate_markdown_report("results")
    os.makedirs("results", exist_ok=True)
    with open("results/REPORT_TABLES.md", "w", encoding="utf-8") as f:
        f.write(md_out)
    with open("results/REPORT_TABLES.tex", "w", encoding="utf-8") as f:
        f.write(generate_latex_table("results"))
    print(md_out)