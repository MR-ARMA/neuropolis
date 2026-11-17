# Neuropolis: State-Conditioned Mechanistic Attention Steering in a Liquidity-Constrained Household Economy

## 1. Research Question & Falsifiable Hypotheses

`neuropolis` investigates a focused, falsifiable question at the intersection of mechanistic interpretability and computational behavioral economics:

> **Core Research Question:** In a liquidity-constrained two-asset household economy where an LLM controls each household's consumption, asset liquidation, and bilateral bargaining decisions under hard double-entry budget constraints, does delivering an agent's endogenous economic state through **inference-time attention and representation steering** produce conditional behavioral response functions closer to published human empirical data than delivering the same state solely through **prompt text**?

We formalize four pre-registered, falsifiable hypotheses evaluated across five state-delivery channels (`NONE`, `TEXT_ONLY`, `STEER_ONLY`, `TEXT_AND_STEER`, and `CUE_CONFLICT`) and six steering conditions (`NONE`, `PLACEBO_RANDOM`, `SHUFFLED_STATE`, `PASTA`, `ACTADD_CAA`, `OUR_COUPLED_STEER`):

- **H1 (Channel Dissociation / Cue Conflict C4):** When prompt text and mechanistic steering inject opposite scarcity states (`CUE_CONFLICT`), agent reservation prices and distress asset sales shift significantly toward the steered state relative to a matched-norm random-direction placebo (`PLACEBO_RANDOM`, C5) and shuffled-state control (`SHUFFLED_STATE`, C6).
- **H2 (Conditional Response-Curve Alignment):** Delivering scarcity and trust via coupled attention-bias and value-subspace steering (`OUR_COUPLED_STEER`) improves the ceiling-normalized alignment score $\text{Score}(c)$ on (a) cross-sectional Marginal Propensity to Consume (MPC) slopes (Jappelli & Pistaferri 2014; Kaplan, Moll, & Violante 2018) and (b) bilateral bargaining acceptance curves (Oosterbeek et al. 2004) over `TEXT_ONLY`, `PASTA`, and `ACTADD_CAA`.
- **H3 (Value of Information via Epistemic Routing):** Semantic-offset token entropy ($\bar{H}_i > \tau_{\text{TMS}}$) discriminates ground-truth high-noise novices from low-noise specialists on the latent macro state $\theta_t$ ($\text{AUROC} > 0.65$) and yields higher realized household CRRA utility $U_i$ per query than `RANDOM_TRIGGER` at matched query budgets.
- **H4 (Out-of-Sample Regime Generalization):** Steering parameters calibrated on a Stagflation Subsistence Shock generalize out-of-sample to Asymmetric Unemployment, Asset Valuation Crash, and Liquidity Credit Crunch regimes.

---

## 2. Claims & Implementation Ledger (Codebase Audit Table)

Every mechanism in this repository is explicitly tracked below by implementation status, file path, and class/method reference:

| Mechanism / Component | Status | File & Method Reference | Notes & Scope |
| :--- | :--- | :--- | :--- |
| **Mechanism A: Pre-Softmax Additive Key-Logit Bias ($\Delta L^{\text{Bias}}$)** | **IMPLEMENTED** | `src/steering/attention_lab.py :: build_additive_4d_logit_bias_mask` | Injects additive float bias into the 4D causal `attention_mask` `[1, 1, Q, K]` prior to Softmax on `SPAN_DRIVES_AND_FINANCE` and `SPAN_PEER_MEMORY`. |
| **Mechanism B: Contrastive Activation Addition (CAA) Value Steering** | **IMPLEMENTED** | `src/brain/model_vault.py :: _extract_contrastive_subspace` & `src/steering/attention_lab.py :: attach_hooks` | Extracts empirical difference-in-means `u_trust` and `u_threat` from 16 paired economic prompts with LOO probe accuracy; projects `v_proj` on `SPAN_INCOMING_MSG`. |
| **Mechanism C: Inverted-U Easterbrook Cue-Utilization ($\beta_{\text{LC}}$)** | **IMPLEMENTED** | `src/steering/attention_lab.py :: compute_easterbrook_beta` | True Gaussian inverted-U $\beta_{\text{LC}}(U_{i,t}) \in [0.85, 1.40]$ peaking at $U^* = 0.45$, applied to `q_proj` in middle layers. |
| **In-Model Post-Softmax Attention Telemetry (`outputs.attentions`)** | **IMPLEMENTED** | `src/steering/attention_lab.py :: measure_attention_from_model_tensors` | Extracts actual post-softmax attention tensors from eager forward pass (`attention_measured_from_model=True`). |
| **Single-Prefill KV-Cached Decoding & Strict `llm_valid` Tracking** | **IMPLEMENTED** | `src/brain/inference_brain.py :: generate_turn` | Reuses `past_key_values` from prefill; records `llm_valid: bool` and `parse_error` per turn without silent heuristic substitution. |
| **Autonomous LLM Decisions Without Formula Overwrites** | **IMPLEMENTED** | `src/orchestration/turn_orchestrator.py :: _finalize_llm_autonomous_intent` | In `LLM_AUTONOMOUS` mode, passes the LLM's parsed action, item, quantity, price, and `ACCEPT`/`REJECT` decision directly to `GameMaster`. |
| **Unconfounded State Channels (`C0`–`C4`) & Placebo Controls (`C5`, `C6`)** | **IMPLEMENTED** | `src/orchestration/cognitive_appraisal.py :: evaluate_pre_action_state` | Decouples `NONE`, `TEXT_ONLY`, `STEER_ONLY`, `TEXT_AND_STEER`, and `CUE_CONFLICT` from world physics $\mathbf{\Omega}$. |
| **Two-Asset HANK Portfolio, Obligations $R_{i,t}$, CRRA Utility & Endogenous Prices** | **IMPLEMENTED** | `src/environment/game_master.py :: execute_end_of_tick_upkeep_and_fiscal_state` | Enforces double-entry cash conservation ($\epsilon_{\text{SFC}} < 10^{-9}$), KMV convex adjustment costs, obligations $R_{i,t}$, CRRA utility $u_{i,t}$, and order-flow price clearing. |
| **Latent Macro State $\theta_t$, Noisy Private Signals & TMS AUROC** | **IMPLEMENTED** | `src/environment/game_master.py :: _generate_latent_signals_for_tick` & `src/steering/entropy_tms.py` | Specialists observe $\theta_t$ with $\sigma = 0.15$ vs. novices $\sigma = 1.80$; `ASK_EXPERT` transfers expert signal $\tilde{\theta}_{j^*,t}$. |
| **Ray Cluster Sharding, A* Urban Mobility & Web Studio UI** | **DEFERRED** | Out of scope for Paper 1 | Deferred to future scaling work; current repository is a self-contained single-node experimental laboratory ($N = 6\text{–}200$). |

---

## 3. Execution & Experimental Protocol

### Step 1: Run the Pre-Flight Verification Suite (`< 5 Seconds`)
Verifies real in-model attention measurement on `Qwen2ForCausalLM`, matched-norm random placebo controls (C5), CAA Value-subspace locality, inverted-U Easterbrook scaling, Stock-Flow Consistency ($\epsilon_{\text{SFC}} < 10^{-9}$), unconfounded `CUE_CONFLICT` routing (C4), and zero orchestrator decision overrides:

```bash
rm -f data/*.sqlite data/*.jsonl
python -m tests.preflight.verify_attention_steering
python -m tests.preflight.verify_environment_and_telemetry