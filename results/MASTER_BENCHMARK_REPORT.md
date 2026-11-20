# Neuropolis Master Empirical Benchmark & Mechanism Attribution Report

`RESULTS_STATUS: EXECUTED_FROM_ARTIFACTS`

## Table 1 — Master Empirical Alignment Table (vs. Published Scientific Literature)

| Run ID | Model | Scale | Regime | Mode / Channel / Steering | α | Valid % | Oosterbeek (2004) Sim (L1, Close%, Bins, TOST) | Jappelli-Pistaferri (2014) MPC (L1, Close%, TOST) | KVW (2014) HtM vs 33.3% (L1, Close%, TOST) | Fehr-Gächter (2000) PG vs 75–80% (L1, Close%, TOST) | T_soc | SFC_Err | Composite % |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `run_s42_RULE_ONLY_TEXT_AND_STEER_OUR_COUPLED_STEER` | `Qwen2.5-1.5B-Instruct` | `dev` | `crisis` | `RULE_ONLY` / `TEXT_AND_STEER` / `OUR_COUPLED_STEER` | 3.0 | 0.0% | 0.00 (0.350, 22.2%, UNDER, NON-EQ) | +0.00 (0.350, 0.0%, NON-EQ) | 0.0% (0.350, 0.0%, NON-EQ) | 0.0% (0.350, 0.0%, NON-EQ) | 0.000 | 0.0e+00 | **5.5%** |

*Footnote: `Closeness % = max(0.0, 100.0 * (1.0 - L1_dist / scale_norm))`, where `scale_norm` is `0.45` (Oosterbeek) and `0.35` (MPC, HtM, PG).*

## Table 2 — Mechanism & Ablation Marginal Effect Analysis (Paired Contrasts)

| Contrast (Treatment vs. Reference) | Model / Scale / Regime | ΔAccept | ΔOosterbeek L1 | ΔMPC Slope | ΔHtM | ΔPG % | ΔT_soc | ΔComposite % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *No matched treatment-vs-reference pairs executed yet* | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

## Table 3 — Macro Regime Sensitivity (Crisis vs. Normal Baseline)

| Model / Scale / Condition | ΔRejRate | Δp_c Inflation % | ΔHtM | ΔMPC_Low | ΔPG % | ΔT_soc | ΔSanctions | ΔComposite % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *No paired normal-vs-crisis runs executed yet* | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

## Table 4 — Cross-Scale & Cross-Model Robustness

| Model | Scale | Runs | Mean Composite % | BinsOK Rate | Mean TOST Passes (/4) | Max SFC_Err |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Qwen2.5-1.5B-Instruct` | `dev` | 1 | **5.5%** | 0% | 0.00 / 4 | 0.0e+00 |
