# Preregistration Protocol: Endogenous Probe-Gated Representation Engineering and Domain-General Cognitive Bottlenecks in a Continuous Double Auction Two-Asset Economy

**Project:** Neuropolis v2  
**Status:** `PREREGISTRATION_DRAFT — PARAMETERS TO BE LOCKED AFTER PILOT GATES G1–G4`  
**Results Status:** `RESULTS_STATUS: NOT_EXECUTED`  
**Target Registry:** Open Science Framework (OSF) / Git Tagged Commit Hash (`TBD_AT_FREEZE`)

---

## 1. Strict Boundary Between Pilot/Calibration Phase and Confirmatory Phase

To prevent Researcher Degrees of Freedom (forking paths), hyperparameter overfitting, and circular tuning on evaluation benchmarks, this study enforces an **airtight, three-way split** across random economy seeds, prompt calibration datasets, and macroeconomic shock regimes.

### 1.1 Disjoint Data & Seed Partitions
| Partition | Seed Range (`economy_seed`) | Macroeconomic Regimes Permitted | Permitted Operations |
| :--- | :--- | :--- | :--- |
| **Phase I: Pilot & Calibration (`P_cal`)** | `S_cal = {1001, ..., 1010}` ($M_{\text{cal}} = 10$) | `NONE`, `TFP_SUBSISTENCE_CONTRACTION` | Execute Pre-Flight Pilot Gates G1–G4; select target layer $l^*$, probe threshold $\tau^*$, and steering gain $\gamma^*$; verify JSON validity $\ge 98\%$ and CDA liquidity. |
| **Phase II: Confirmatory Sweep (`P_conf`)** | `S_conf = {2001, ..., 2030}` ($M_{\text{conf}} = 30$) | `NONE`, `TFP_SUBSISTENCE_CONTRACTION`, `ASYMMETRIC_WAGE_SHOCK` | **Frozen config only.** Run complete $2^3$ factorial matrix + matched negative controls once. Zero parameter tuning permitted after tag freeze. |
| **Phase III: Held-Out Generalization (`P_hold`)** | `S_hold = {3001, ..., 3030}` ($M_{\text{hold}} = 30$) | `CAPITAL_YIELD_COLLAPSE`, `CREDIT_WEDGE_CRUNCH` | Opened strictly once after Phase II analysis to test out-of-sample regime generalization and secondary model-family replication. |

### 1.2 Freeze & Deviation Protocol
1. **Hard Freeze Gate:** Phase II (`P_conf`) may begin **if and only if** all four Pre-Flight Pilot Gates (`scripts/run_pilot_gates.py`) pass on `P_cal`, the locked parameters $(l^*, \tau^*, \gamma^*, B_{\min}, K_{\min}, \sigma_{\max})$ are recorded in Section 7 of this document, and the repository commit hash is tagged (`git tag prereg-v2-freeze`) and archived on OSF.
2. **Protocol Deviations:** Any modification after the freeze tag (e.g., bug fix, exclusion rule update, or exploratory post-hoc probe) must be logged with a timestamp and rationale in `prereg/deviations.md` and labeled **Exploratory** in all tables and figures.
3. **Disjoint Out-of-Domain Probe Extraction:** The Linear Probe $(\mathbf{w}^{(l^*)}, b^{(l^*)})$ is trained on sterile Out-of-Domain Dataset A (collaborative workplace/academic splits, $N = 120$ matched pairs), and the RepE steering vector $\hat{\mathbf{v}}_{\text{inequity}}^{(l^*)}$ is extracted independently from sterile Out-of-Domain Dataset B (civic/grant/tournament allocations, $N = 120$ matched pairs). Neither dataset shares vocabulary or structure with the Neuropolis Continuous Double Auction (CDA) economy.

---

## 2. Research Questions & Falsifiable Primary Hypotheses

**Primary Research Question:** In a strictly Stock-Flow Consistent (SFC) two-asset household economy cleared via a Continuous Double Auction (CDA) under sanitized, jargon-free framing, do **Two-Stage Endogenous Probe-Gated RepE** and **Domain-General Cognitive Bottlenecks** (Bounded Deliberation Budget and Episodic Memory Friction) align LLM agent response curves with published human empirical benchmarks better than unconstrained prompting and matched mechanistic controls?

| ID | Pre-Registered Hypothesis | Primary Estimand | Statistical Test | Explicit Falsification Criterion |
| :---: | :--- | :--- | :--- | :--- |
| **H1** | Endogenous Probe-Gated RepE (`C111`) improves alignment with the Oosterbeek et al. (2004) bargaining acceptance curve over unconstrained Vanilla LLM (`C000`) | Paired cluster difference $\Delta W_1 = W_1(\text{C111}) - W_1(\text{C000})$ across $M = 30$ CRN seeds | Paired Clustered Bootstrap ($B=10,000$) & paired Student-$t$ ($df=29$, Holm-Bonferroni) | 95% clustered CI of $\Delta W_1$ includes $0$ or $\Delta W_1 \ge 0$ |
| **H2** | Endogenous RepE alignment gain is direction- and state-specific (not a generic perturbation artifact) | Paired contrasts $W_1(\text{C111}) - W_1(\text{CTRL\_PLACEBO\_VEC})$ and $W_1(\text{C111}) - W_1(\text{CTRL\_SHUFFLED\_LAMBDA})$ | Paired Clustered Bootstrap & paired Student-$t$ (Holm-Bonferroni) | Either 95% CI includes $0$ or either control matches/beats `C111` |
| **H3** | Sign-reversed steering (`CTRL_SIGN_FLIP`, $-\gamma \hat{\mathbf{v}}$) reverses the directional behavioral shift induced by $+\gamma \hat{\mathbf{v}}$ | Mean order acceptance & reservation markup shift under $+\gamma$ vs. $-\gamma$ | Paired Clustered Bootstrap | Behavioral shift fails to invert sign or overlaps with $\gamma = 0$ |
| **H4** | Full architecture (`C111`) achieves Per-Bin TOST statistical equivalence with published human curves (Bargaining, Two-Asset HtM, Windfall MPC) | Per-bin mean difference $\hat{\mu}_b - R_{\text{Human},b}$ against $\pm \Delta_{\text{SESOI}}$ | Per-Bin Intersection-Union TOST with clustered SE (Holm-Bonferroni) | Adjusted $p_{\text{TOST}} \ge 0.05$ or bin count $< 30$ |
| **H5** | Cognitive control components exhibit non-additive synergy across the $2^3$ factorial lattice | Exact $2^3$ interaction contrasts ($\beta_{AB}, \beta_{AC}, \beta_{BC}, \beta_{ABC}$) and Shapley shares ($\phi_A, \phi_B, \phi_C$) | Nested F-test (additive vs. interaction model across $M=30$ seeds) | Interaction terms do not improve fit ($p \ge 0.05$) |
| **H6** | Parameters locked on `P_cal` generalize out-of-sample to held-out shock regimes (`P_hold`) | Paired $\Delta W_1$ on `CAPITAL_YIELD_COLLAPSE` & `CREDIT_WEDGE_CRUNCH` | Paired Clustered Bootstrap on $S_{\text{hold}}$ | Effect sign reverses or 95% CI includes $0$ |

*Commitment to Null Results:* Any hypothesis that meets its falsification criterion will be reported prominently in the main results table as **Not Supported**, alongside exact point estimates and 95% clustered CIs.

---

## 3. Exact $2^3$ Factorial Arms & Isolated Mechanisms

All LLM arms receive identical **Sanitized Framing** prompts (verified at runtime by `assert_prompt_sanitized`, banning all behavioral economics jargon such as *unfair, ultimatum, loan, debt, gouging, scarcity, loss aversion, human, trust*). Steering hooks are strictly forbidden during Stage 1 CoT and strictly restricted to Stage 2 Action tokens, with **zero post-hoc logit modifications** and **zero silent rule fallbacks** (exhausted JSON retries emit explicit `ABSTAIN`).

### 3.1 Isolated Mechanisms
* **Factor A — Bounded Deliberation Budget ($B_{\text{CoT}}$):** Dynamically compresses `max_tokens` **strictly for Stage 1 hidden CoT** via $B_{\text{CoT}}(S_{\text{eff}}) = \lfloor B_{\min} + (B_{\max} - B_{\min})\exp(-k_{\text{delib}} S_{\text{eff}}) \rfloor$, while leaving Stage 2 JSON Action `max_tokens` (`128`) strictly invariant to stress.
* **Factor B — Episodic Memory Friction ($M_{\text{RAG}}$):** Dynamically contracts episodic RAG retrieval depth $K(S) \in [K_{\min}, K_{\max}]$ and injects Gaussian retrieval noise $\sigma(S) = \sigma_{\max} S$ into query embeddings and cosine similarity rankings, mechanistically inducing salience bias and imperfect recall.
* **Factor C — Two-Stage Endogenous Probe-Gated RepE ($R_{\text{RepE}}$):** Stage 1 generates an unsteered hidden CoT; the frozen linear probe $(\mathbf{w}^{(l^*)}, b^{(l^*)})$ computes $\lambda_{i,t} = \sigma\big((\mathbf{w}^{\top}\bar{\mathbf{h}}_{\text{CoT}}^{(l^*)} + b - \tau^*)/T\big) \in [0, 1]$ purely from the model's CoT residual stream; Stage 2 injects $\mathbf{h}' = \mathbf{h} + \lambda_{i,t} \gamma^* \hat{\mathbf{v}}_{\text{inequity}}^{(l^*)}$ strictly into Action tokens (capped at relative residual norm $\le 0.25$).

### 3.2 Complete $2^3$ Factorial Design Matrix
| Cell ID | Factor A ($B_{\text{CoT}}$) | Factor B ($M_{\text{RAG}}$) | Factor C ($R_{\text{RepE}}$) | Experimental Role |
| :---: | :---: | :---: | :---: | :--- |
| **`C000`** | OFF | OFF | OFF | Unconstrained Sanitized Vanilla LLM Baseline |
| **`C100`** | ON | OFF | OFF | Isolated Main Effect: Bounded Deliberation |
| **`C010`** | OFF | ON | OFF | Isolated Main Effect: Episodic Memory Friction |
| **`C001`** | OFF | OFF | ON | Isolated Main Effect: Two-Stage Endogenous RepE |
| **`C110`** | ON | ON | OFF | Joint Domain-General Cognitive Bottlenecks (No RepE) |
| **`C101`** | ON | OFF | ON | Two-Way Interaction: Deliberation $\times$ RepE |
| **`C011`** | OFF | ON | ON | Two-Way Interaction: Memory Friction $\times$ RepE |
| **`C111`** | ON | ON | ON | Full Neuropolis Cognitive Control Architecture |

---

## 4. Exact Negative & Positive Control Suite (Paired via Common Random Numbers)

Every control condition runs on the exact same `S_conf` economy seeds as `C000`–`C111`:

| Control ID | Exact Specification | Confound Ruled Out |
| :--- | :--- | :--- |
| **`CTRL_PLACEBO_VEC`** | Identical to `C111`, replacing $\hat{\mathbf{v}}_{\text{inequity}}$ with a unit-norm random vector $\hat{\mathbf{v}}_{\text{rand}}$ orthogonal to both $\hat{\mathbf{v}}_{\text{inequity}}$ and $\mathbf{w}_{\text{probe}}$, injected with the exact same L2 norm $\lambda \gamma^*$ at layer $l^*$. | Rules out generic activation-perturbation / noise-injection artifacts. |
| **`CTRL_ADJACENT_CONCEPT`** | Identical to `C111`, replacing $\hat{\mathbf{v}}_{\text{inequity}}$ with an orthogonalized adjacent-concept direction (e.g., general lexical urgency/negative-valence extracted from non-distributional texts) matched on L2 norm. | Rules out broad sentiment/valence drift rather than distributional inequity representation. |
| **`CTRL_SIGN_FLIP`** | Identical to `C111`, negating steering gain ($-\gamma^* \hat{\mathbf{v}}_{\text{inequity}}$). | Verifies causal directional specificity (bidirectional dose response). |
| **`CTRL_SHUFFLED_LAMBDA`** | Identical to `C111`, permuting the endogenous gate scalar $\lambda_{i,t}$ across turns/agents within the same tick. | Verifies that *endogenous, state-contingent* gating (not constant background injection) drives the effect. |
| **`BASE_ZERO_INTELLIGENCE`** | Budget-constrained Zero-Intelligence CDA traders (Gode & Sunder, 1993). | Establishes the empirical null floor ($D_{\text{ZI}}$) and proves market freezes stem from cognition, not CDA mechanics. |
| **`BASE_CLOSED_FORM`** | Non-LLM deterministic Stone-Geary / Two-Asset policy operating on the identical CDA order book. | Establishes the non-LLM behavioral reference comparator. |

---

## 5. *A Priori* SESOI Margins & Per-Bin TOST Protocol

Alignment with published empirical human benchmarks is tested **per bin** via the **Intersection-Union Two One-Sided Tests (TOST)** procedure using clustered standard errors over $M$ independent economy seeds ($\text{df} = M - 1$), with family-wise **Holm-Bonferroni** adjustment across all bins within each benchmark family and a strict minimum coverage floor of $n \ge 30$ observations per bin.

| Empirical Benchmark | Published Human Reference & Bins | *A Priori* SESOI Margin ($\pm \Delta_{\text{SESOI}}$) | Justification of SESOI |
| :--- | :--- | :---: | :--- |
| **1. CDA Bargaining / Offer Acceptance Curve** | Oosterbeek, Sloof, & van de Kuilen (2004, *Exp. Econ.*) meta-analysis across 4 relative markup bins: `[0.94, 0.76, 0.42, 0.14]` (meta-analytic SEs `[0.03, 0.04, 0.05, 0.04]`) | **$\pm 0.08$** ($\pm 8\text{ pp}$) | Twice the average cross-study meta-analytic SE ($\approx 0.04$), representing the smallest practically distinguishable behavioral shift across cultural replications. |
| **2. Two-Asset Hand-to-Mouth (HtM) Composition** | Kaplan, Violante, & Weidner (2014, *BPEA*) US SCF microdata: `[Poor_HtM=0.12, Wealthy_HtM=0.20, Non_HtM=0.68]` | **$\pm 0.05$** ($\pm 5\text{ pp}$) | Matches empirical cross-survey SCF/PSID sampling variation across business-cycle states. |
| **3. Exogenous-Windfall Causal MPC Heterogeneity** | Jappelli & Pistaferri (2014, *AEJ: Macro*) liquidity terciles: `[Low=0.63, Mid=0.46, High=0.31]` (measured strictly via paired Treasury windfall shadow worlds, zero circularity) | **$\pm 0.06$** | Half of the empirical inter-tercile MPC step ($\approx 0.16$) and $\approx 1.7\times$ empirical survey SE ($0.035$). |

**Distributional Summary Metrics:** Alongside Per-Bin TOST, we report (a) **Bootstrap-Debiased 1-Wasserstein Distance ($W_1$)** with 95% Clustered Bootstrap CIs ($B = 10,000$ resamples over economy seeds) anchored between the empirical ZI Null Floor and Human SE Noise Ceiling, and (b) **Miller-Madow Finite-Sample Debiased Jensen-Shannon Divergence ($\text{JSD}$)** on discrete household liquidity strata.

---

## 6. Statistical Power Justification for $M \ge 30$ Independent Economy Seeds

Because agents interact endogenously inside a shared CDA limit order book, all observations within a single simulation run are correlated; therefore, **the independent statistical unit is strictly the economy seed $m \in \{1, \dots, M\}$** (Clustered Bootstrap and Cluster-Level Student-$t$, $\text{df} = M - 1$).

1. **Paired Confirmatory Contrasts (H1, H2, H6):** Under Common Random Numbers (CRN) across paired seeds, let the standardized paired effect size be $d_z = \Delta / \sigma_{\text{diff}}$. For a two-sided test targeting $1 - \beta = 0.80$ power at Holm-Bonferroni adjusted $\alpha_{\text{adj}} = 0.05 / 4 = 0.0125$ (critical $t_{0.99375, 29} \approx 2.66$), the minimum sample size to detect a medium-to-large paired effect ($d_z = 0.65$) is:
   $$M \ge \left(\frac{t_{1-\alpha_{\text{adj}}/2,\, M-1} + t_{1-\beta,\, M-1}}{d_z}\right)^2 \approx \left(\frac{2.66 + 0.85}{0.65}\right)^2 \approx 29.2 \implies \mathbf{M = 30\text{ economy seeds}}.$$
2. **Per-Bin TOST Equivalence (H4):** For one-sided $\alpha = 0.05$ TOST with $\Delta_{\text{SESOI}} = 0.08$ and expected between-economy cluster standard deviation $\sigma_{\text{cluster}} \approx 0.11$ (verified during Phase I Pilot), achieving $80\%$ power when true $|\mu_b - R_b| \le 0.015$ requires $\text{SE}_{\text{cluster}} = \sigma_{\text{cluster}} / \sqrt{M} \le (0.08 - 0.015) / (1.70 + 0.85) \approx 0.0255 \implies M \ge (0.11 / 0.0255)^2 \approx 18.6$, comfortably satisfied by **$M = 30$ independent seeds** (with option to expand to $M = 50$ if Phase I pilot cluster SD exceeds $0.14$).

---

## 7. Pre-Flight Pilot Go/No-Go Gates & Frozen Parameter Ledger

Before opening `S_conf`, `scripts/run_pilot_gates.py` must pass all four diagnostic gates:

| Gate | Requirement & Pass Criterion | Recorded Pilot Value | Status |
| :---: | :--- | :---: | :---: |
| **Gate 1** | **Probe Transfer & B$\to$C Coupling:** Zero-shot OOD probe AUROC on ~100 sanitized in-game CoTs $\ge 0.70$ (full CoT) and $\ge 0.65$ (taxed CoT); $|\bar{\lambda}_{\text{full}} - \bar{\lambda}_{\text{taxed}}| \le 0.15$. | `TBD` | `NOT_EXECUTED` |
| **Gate 2** | **KV-Cache Faithfulness & Dose-Response:** $\gamma = 0.0$ output 100% bit-identical to `OFF`; monotone shift across $\gamma \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$ (Spearman $\rho \ge 0.85$); Stage 2 JSON validity $\ge 98.0\%$ across all doses. | `TBD` | `NOT_EXECUTED` |
| **Gate 3** | **Deliberation Bandwidth Sign:** Isolated intertemporal trade-off test comparing `cot_max_tokens=256` vs. `24` verifying whether starvation induces directional myopia vs. noise/RLHF-patience while preserving JSON validity $\ge 95\%$. | `TBD` | `NOT_EXECUTED` |
| **Gate 4** | **Baseline CDA Liquidity & Strict SFC:** Unsteered Baseline + Needs Engine clears CDA without spread deadlock (`market_freeze_rate < 0.25`) and strictly maintains `max_sfc_conservation_error < 1e-9`. | `TBD` | `NOT_EXECUTED` |

### Frozen Parameter Lock Table (Completed Strictly After Passing Gates G1–G4)
* **Primary Model ID & Pinned Revision:** `TBD` (`Qwen/Qwen2.5-3B-Instruct`, NF4 double-quant / `bfloat16`)
* **Locked Target Layer ($l^*$):** `TBD`
* **Locked Probe Threshold ($\tau^*$) & Temperature ($T_{\text{probe}}^*$):** `TBD`
* **Locked Steering Strength ($\gamma^*$):** `TBD`
* **Locked Deliberation Bounds ($B_{\min}, B_{\max}, k_{\text{delib}}$):** `TBD`
* **Locked Memory Friction Bounds ($K_{\min}, K_{\max}, \sigma_{\max}$):** `TBD`
* **Pre-Registration Freeze Git Commit SHA:** `TBD`