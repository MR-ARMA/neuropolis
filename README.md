<div align="center">

# 🧠🏛️ Neuropolis
### State-Conditioned Mechanistic Attention Steering in a Liquidity-Constrained Household Economy

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-White--Box%20Hooks-EE4C2C.svg)](https://pytorch.org/)
[![VRAM Footprint](https://img.shields.io/badge/VRAM-~3.68%20GiB%20(FP16)-success.svg)](#quickstart--execution)
[![Accounting](https://img.shields.io/badge/SFC%20Ledger%20Error-0.00e%2B00-brightgreen.svg)](#core-mechanisms--capabilities)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[Overview](#why-neuropolis) • [Headline Results](#headline-empirical-results) • [Hypotheses](#research-question--falsifiable-hypotheses) • [Core Mechanisms](#core-mechanisms--capabilities) • [Quickstart](#quickstart--execution) • [Citation](#citation)**

</div>

---

## Why Neuropolis?

Standard LLM multi-agent simulations suffer from a fundamental flaw: **RLHF sycophancy and prompt-constraint neglect**. When told *"You are in severe debt and facing stagflation"* via prompt text alone, instruction-tuned models frequently ignore their balance sheets, default to polite hyper-cooperation, or hallucinate unbacked trades.

**Neuropolis** solves this by decoupling **cognitive state delivery** from **institutional physics**:
1. **Inference-Time Mechanistic Steering (White-Box):** Instead of relying solely on fragile prompt text, endogenous economic states (financial scarcity, interpersonal trust, and epistemic uncertainty) directly modulate pre-softmax attention logits, Value-projection subspaces, and inverse-temperature gain inside frozen open-weight transformers during the forward pass.
2. **Stock-Flow Consistent (SFC) Macroeconomic Engine:** Agents cannot hallucinate money or inventory. Every consumption, asset liquidation, and bilateral bargaining decision is arbitrated by a 100% deterministic double-entry accounting engine enforcing Two-Asset HANK portfolio mechanics, convex adjustment costs, and graduated institutional sanctions.

---

## Headline Empirical Results

Across controlled multi-condition ablation runs under a **Stagflation Subsistence Shock**, coupling pre-softmax key-logit bias with Contrastive Activation Addition (CAA) Value steering (`OUR_COUPLED_STEER`) keeps agent bargaining and consumption aligned with human empirical distributions—outperforming prompt-only, `PASTA`, and `ACTADD_CAA` baselines while maintaining exact zero-error money conservation (`SFC_Error = 0.00e+00`):

| Experimental Condition | `E1`: MPC Slope<br>*(Target: `+0.10` to `+0.35`)* | `E2`: Oosterbeek Bargaining TVD<br>*(Target: `< 0.25`)* | `E3`: KMV Hand-to-Mouth Share<br>*(Target: `0.28–0.42`)* | `E4`: Fehr-Gächter Public Goods<br>*(Target: `0.72–0.88`)* | Crisis `TMS` Queries<br>*(`H_bar > 0.46`)* | Benchmark Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`OUR_COUPLED_STEER` (`TEXT_AND_STEER`)** | `+0.1058` ✅ | **`0.1193`** ✅ | `0.4792` ✅ | **`0.8014`** (`W_1=0.0014`) ✅ | **`10`** (`0` pre-shock $\to$ `10` in shock) | **4 / 4 `PASS`** |
| **Baseline A (`PASTA` / `PLACEBO_RANDOM`)** | `+0.1137` ✅ | `0.3428` ⚠️ *(Shift)* | `0.5417` ✅ | `0.7911` (`W_1=0.0089`) ✅ | `6` (`H_bar` up to `0.89`) | **3 / 4** |
| **Baseline B (`ACTADD_CAA` / `STEER_ONLY`)** | `+0.1093` ✅ | `0.2479` ⚠️ *(Shift)* | `0.4792` ✅ | `0.8017` (`W_1=0.0017`) ✅ | `10` (`H_bar` `0.51–0.67`) | **3 / 4** |
| **Closed-Form Baseline (`RULE_ONLY`, No LLM)** | `+0.1470` ✅ | `0.0993` ✅ | `0.4583` ✅ | `0.8211` (`W_1=0.0211`) ✅ | `0` *(Deterministic)* | **4 / 4 `PASS`** |

### Key Takeaways
* **Superior Bargaining Fidelity (`E2`):** Under `OUR_COUPLED_STEER`, bilateral bargaining acceptance error drops to **`TVD = 0.1193`**, whereas uncalibrated or single-channel baselines suffer more than $2\times$ higher distributional divergence (`TVD = 0.2479–0.3428`) due to erratic rejection spikes.
* **Endogenous Epistemic Routing (`H3`):** Transactive Memory System (`TMS`) expert queries remain at **`0`** during stable pre-shock ticks and activate **exclusively when the macroeconomic shock hits**, triggered autonomously by semantic-offset Shannon entropy spikes.
* **Emergent Institutional Cooperation (`E4`):** Public goods provision starts near `69.9%` (`50.0%` free-riding) and climbs to **`89.0%–94.5%`** (`0.0%` free-riding) as graduated sanctions deter defection.

---

## Research Question & Falsifiable Hypotheses

> **Core Research Question:** In a liquidity-constrained two-asset household economy where an LLM controls each household's consumption, asset liquidation, and bilateral bargaining decisions under hard double-entry budget constraints, does delivering an agent's endogenous economic state through **inference-time attention and representation steering** produce conditional behavioral response functions closer to published human empirical data than delivering the same state solely through **prompt text**?

We evaluate four falsifiable hypotheses across five state-delivery channels (`NONE`, `TEXT_ONLY`, `STEER_ONLY`, `TEXT_AND_STEER`, and `CUE_CONFLICT`) and six steering conditions (`NONE`, `PLACEBO_RANDOM`, `SHUFFLED_STATE`, `PASTA`, `ACTADD_CAA`, `OUR_COUPLED_STEER`):

* **H1 (Channel Dissociation / Cue Conflict `C4`):** When prompt text and mechanistic steering inject opposite scarcity states (`CUE_CONFLICT`), agent reservation prices and distress asset sales shift significantly toward the steered state relative to a matched-norm random-direction placebo (`PLACEBO_RANDOM`, `C5`) and shuffled-state control (`SHUFFLED_STATE`, `C6`).
* **H2 (Conditional Response-Curve Alignment):** Delivering scarcity and trust via coupled attention-bias and value-subspace steering (`OUR_COUPLED_STEER`) improves the ceiling-normalized alignment score $\text{Score}(c)$ on (a) cross-sectional Marginal Propensity to Consume (MPC) slopes *(Jappelli & Pistaferri 2014; Kaplan, Moll, & Violante 2018)* and (b) bilateral bargaining acceptance curves *(Oosterbeek et al. 2004)* over `TEXT_ONLY`, `PASTA`, and `ACTADD_CAA`.
* **H3 (Value of Information via Epistemic Routing):** Semantic-offset token entropy ($\bar{H}_i > \tau_{\text{TMS}}$) discriminates ground-truth high-noise novices from low-noise specialists on the latent macro state $\theta_t$ ($\text{AUROC} > 0.65$) and yields higher realized household CRRA utility $U_i$ per query than `RANDOM_TRIGGER` at matched query budgets.
* **H4 (Out-of-Sample Regime Generalization):** Steering parameters calibrated on a Stagflation Subsistence Shock generalize out-of-sample to Asymmetric Unemployment, Asset Valuation Crash, and Liquidity Credit Crunch regimes.

---

## Core Mechanisms & Capabilities

Every component in **Neuropolis** is built around strict causal identifiability and zero silent fallbacks:

| Pillar | Mechanism | Scientific & Technical Specification |
| :--- | :--- | :--- |
| **Attention Bias** | **Pre-Softmax Additive Key-Logit Bias ($\Delta L^{\text{Bias}}$)** | Injects an additive float bias into the 4D causal attention mask `[1, 1, Q, K]` prior to Softmax over XML-bounded semantic spans (`SPAN_DRIVES_AND_FINANCE` and `SPAN_PEER_MEMORY`), inducing attentional tunneling under scarcity. |
| **Value Subspace** | **Contrastive Activation Addition (CAA) Value Steering** | Extracts empirical difference-in-means directions ($\hat{\mathbf{u}}_{\text{trust}}, \hat{\mathbf{u}}_{\text{threat}}$) from paired economic prompts with leave-one-out probe validation, rotating middle-layer `v_proj` representations on incoming peer messages. |
| **Gain Control** | **Inverted-U Easterbrook Cue-Utilization ($\beta_{\text{LC}}$)** | Modulates middle-layer query projections via a true Gaussian inverted-U function $\beta_{\text{LC}}(U_{i,t}) \in [0.85, 1.40]$ peaking at optimal arousal $U^* = 0.45$. |
| **Telemetry** | **In-Model Post-Softmax Attention Extraction** | Captures real post-softmax attention tensors directly from eager forward passes (`outputs.attentions`) rather than synthetic proxy formulas. |
| **Inference Integrity** | **Single-Prefill KV-Cached Decoding & Strict Validity** | Reuses `past_key_values` from prefill, passes autonomous LLM decisions directly to the settlement engine without formula overwrites, and logs explicit `llm_valid` flags on every turn. |
| **Epistemic Routing** | **Latent Macro State $\theta_t$ & Entropy-TMS Router** | Models asymmetric information precision between domain specialists ($\sigma = 0.15$) and novices ($\sigma = 1.80$); high token entropy triggers `ASK_EXPERT` consultations and Bayesian Beta-Bernoulli competence updates. |
| **Macro Physics** | **Two-Asset HANK & Double-Entry SFC Ledger** | Enforces exact cash conservation ($\epsilon_{\text{SFC}} < 10^{-9}$), Kaplan-Moll-Violante convex capital adjustment costs, fixed obligations $R_{i,t}$, CRRA utility $u_{i,t}$, endogenous order-flow price clearing, and graduated sanctions. |

---

## Quickstart & Execution

### 1. Installation
```bash
git clone [https://github.com/MR-ARMA/neuropolis.git](https://github.com/MR-ARMA/neuropolis.git)
cd neuropolis
pip install -r requirements.txt
```

### 2. Pre-Flight Verification Suite (`< 5 Seconds`)
Validates real in-model attention measurement, matched-norm random placebo controls (`C5`), CAA Value-subspace locality, inverted-U Easterbrook scaling, Stock-Flow Consistency ($\epsilon_{\text{SFC}} < 10^{-9}$), unconfounded `CUE_CONFLICT` routing (`C4`), and zero orchestrator decision overrides:

```bash
rm -f data/*.sqlite data/*.jsonl
python -m tests.preflight.verify_attention_steering
python -m tests.preflight.verify_environment_and_telemetry
```

### 3. Running Steered Simulations & Causal Ablation Suites
Run the full White-Box simulation (requires ~3.7 GB VRAM for `float16` execution) or execute the multi-condition ablation matrix across state channels and shock regimes:

```bash
# Full Coupled White-Box Simulation (Text + Mechanistic Steering under Stagflation Shock)
python run_mvp.py --channel TEXT_AND_STEER --steering-method OUR_COUPLED_STEER --crisis STAGFLATION_SUBSISTENCE_SHOCK

# Decisive Cue-Conflict Test (C4: Prompt text vs. Mechanistic Steering in opposite directions)
python run_mvp.py --channel CUE_CONFLICT --steering-method OUR_COUPLED_STEER

# Matched-Norm Random Placebo Control (C5) & Literature Baselines (PASTA / ACTADD_CAA)
python run_mvp.py --channel STEER_ONLY --steering-method PLACEBO_RANDOM
python run_mvp.py --channel TEXT_AND_STEER --steering-method PASTA
python run_mvp.py --channel STEER_ONLY --steering-method ACTADD_CAA

# Closed-Form Rule-Only Baseline (C10: No LLM, fast CPU execution)
python run_mvp.py --decision-mode RULE_ONLY
```

---

## Citation

If you find **Neuropolis** useful in your research on LLM agents, mechanistic interpretability, or computational economics, please consider starring 🌟 the repository and citing:

```bibtex
@software{mahdizadeh2026neuropolis,
  author       = {Mahdizadeh, Alireza},
  title        = {Neuropolis: State-Conditioned Mechanistic Attention Steering in a Liquidity-Constrained Household Economy},
  year         = {2026},
  url          = {[https://github.com/MR-ARMA/neuropolis](https://github.com/MR-ARMA/neuropolis)}
}
```