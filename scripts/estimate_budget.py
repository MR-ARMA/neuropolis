from __future__ import annotations

import argparse
from typing import Dict, List


def estimate_compute_budget(
    tokens_per_sec: float = 420.0,
    mean_prompt_tokens: int = 280,
    mean_output_tokens: int = 56,
) -> List[Dict[str, float | str]]:
    """
    Section 4 Budget Estimator:
    Calculates expected token volume and GPU-hours per experimental phase.
    """
    tokens_per_call = mean_prompt_tokens + mean_output_tokens
    phases = [
        ("P1_Dose_Response_and_CAA", 12, 1, 300),
        ("P2_Tier_A_Micro_Benchmarks", 16, 1, 300),
        ("P3_Tier_B_Human_Protocols", 10, 1, 150),
        ("P4_Pilot_and_Power_Calibration", 15, 24, 30),
        ("P6_Confirmatory_X1_to_X8_T1", 120, 24, 60),
        ("P7_Generalization_T2_and_Scale48", 90, 24, 60),
    ]
    rows: List[Dict[str, float | str]] = []
    for name, runs, agents, ticks in phases:
        total_calls = runs * agents * ticks
        total_tokens = total_calls * tokens_per_call
        gpu_hours = (total_tokens / max(1.0, tokens_per_sec)) / 3600.0
        rows.append(
            {
                "phase": name,
                "runs": runs,
                "total_llm_calls": total_calls,
                "total_tokens_m": round(total_tokens / 1e6, 2),
                "estimated_gpu_hours": round(gpu_hours, 2),
            }
        )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neuropolis Compute Budget Estimator")
    parser.add_argument("--tps", type=float, default=420.0, help="Measured tokens/sec")
    args = parser.parse_args()
    table = estimate_compute_budget(tokens_per_sec=args.tps)
    print("================================================================================")
    print(" NEUROPOLIS PHASED COMPUTE BUDGET ESTIMATE")
    print("================================================================================")
    print(f"{'Phase':<34} | {'Runs':>6} | {'LLM Calls':>10} | {'Tokens (M)':>10} | {'GPU Hours':>9}")
    print("-" * 80)
    total_h = 0.0
    for r in table:
        total_h += float(r["estimated_gpu_hours"])
        print(
            f"{str(r['phase']):<34} | {int(r['runs']):>6d} | {int(r['total_llm_calls']):>10d} | "
            f"{float(r['total_tokens_m']):>10.2f} | {float(r['estimated_gpu_hours']):>9.2f}"
        )
    print("-" * 80)
    print(f"TOTAL ESTIMATED GPU HOURS: {total_h:.2f} hrs")