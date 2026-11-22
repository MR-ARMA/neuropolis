# Neuropolis: Endogenous Representation Engineering and Domain-General Cognitive Bottlenecks for Behavioral Alignment of LLM Economic Agents

`RESULTS_STATUS: NOT_EXECUTED` *(Architecture initialized; confirmatory multi-seed runs pending Module A–D implementation)*

---

## 1. Abstract & Scientific Objective

Large Language Model (LLM) agents deployed in multi-agent economic environments routinely exhibit systematic behavioral distortions—excessive RLHF politeness, hyper-rational verbal compliance, immunity to resource scarcity, and hallucinated financial accounting—that diverge sharply from empirical human microeconomic behavior. Prior attempts to mitigate these failures rely either on **prompt priming** (leaking behavioral theory labels directly into system prompts, inducing demand effects) or on **externalized steering heuristics** (using environment-level `if/else` triggers or hardcoded logit overrides that bypass the model's own cognition).

**Neuropolis** introduces a modular, causally identifiable AI control architecture designed to answer a single, falsifiable research question:

> **Primary Research Question:** In a strictly Stock-Flow Consistent (SFC) two-asset household economy cleared via a **Continuous Double Auction (CDA)**, do **Two-Stage Probe-Gated Representation Engineering (Endogenous RepE)** and **Domain-General Cognitive Bottlenecks** (Bounded Deliberation and Episodic Memory Friction) align LLM agent behavior with empirical human behavioral economics curves—without prompt leakage, without external steering triggers, and without rule-based action overrides?

---

## 2. System Architecture

Neuropolis strictly decouples **Environmental Physics** from **Agent Cognition** across four architectural pillars:

```text
========================================================================================================
                 NEUROPOLIS v2: STRICT SFC PHYSICS + ENDOGENOUS COGNITIVE CONTROL
========================================================================================================

  [1. STRICT SFC ENVIRONMENT & CDA ORDER BOOK]
   • Two-Asset Household Ledger: Liquid Cash b_{i,t} (r_b, credit limit b_bar, borrowing wedge kappa)
                               + Illiquid Productive Capital a_{i,t} (r_k > r_b, convex cost chi(d,a))
                               + Consumable Necessity c_{i,t} (metabolic requirement c_min)
   • Continuous Double Auction (CDA): Price-Time Priority Limit Order Book (Bids/Asks, Escrow Lock)
   • Needs Engine: Sub-subsistence consumption degrades physiological Health H_{i,t} in [0, 1]
                                          │
                                          │ Objective Balance Sheet, Order Book Snapshot & Needs State
                                          │ (ZERO Behavioral Theory Labels / ZERO Prompt Leakage)
                                          ▼
  [2. DOMAIN-GENERAL COGNITIVE BOTTLENECKS (Driven by Endogenous Stress S_{i,t})]
   ┌──────────────────────────────────────────┐   ┌───────────────────────────────────────────────────┐
   │ BOTTLENECK A: EPISODIC MEMORY FRICTION   │   │ BOTTLENECK B: BOUNDED DELIBERATION BUDGET         │
   │ • Contracts RAG Top-K: K(S) in [K_min,K] │   │ • Truncates `max_tokens` STRICTLY on Stage 1 CoT: │
   │ • Injects retrieval score noise sigma(S) │   │     B_CoT(S_{i,t}) in [B_min, B_max]              │
   │ • Mechanistically models Salience Bias   │   │ • Preserves full budget for Stage 2 Action JSON   │
   └────────────────────┬─────────────────────┘   └─────────────────────────┬─────────────────────────┘
                        └─────────────────────────┬─────────────────────────┘
                                                  ▼
  [3. TWO-STAGE PROBE-GATED RepE (100% ENDOGENOUS STEERING — ZERO EXTERNAL TRIGGERS)]
   ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
   │ STAGE 1: UNSTEERED APPRAISAL PASS (Hidden Chain-of-Thought)                                     │
   │  • Model generates private CoT up to B_CoT(S_{i,t}) tokens with ZERO steering hooks active.     │
   │  • Linear Probe reads pooled CoT residual stream h_bar_CoT^(l) at layer l:                      │
   │        lambda_{i,t} = g( w_probe^(l)^T * h_bar_CoT^(l) + b_probe^(l) )   in [0, 1]              │
   │  • Detects emergent internal representation of perceived inequity purely from model cognition.  │
   └──────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                  │ Continuous Scalar lambda_{i,t}
                                                  ▼
   ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
   │ STAGE 2: STEERED ACTION GENERATION PASS (Structured JSON Action)                                │
   │  • Conditioned on the exact Stage 1 CoT prefix, model generates the JSON order/portfolio action.│
   │  • Residual hook injects:   h_t^(l) <-- h_t^(l) + lambda_{i,t} * gamma * v_hat_inequity^(l)     │
   │    STRICTLY on Stage 2 Action tokens (never on prompt or Stage 1 CoT tokens).                   │
   └──────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                  │ Structured Order / Capital Action (or ABSTAIN)
                                                  ▼
  [4. ORDER BOOK MATCHING ENGINE & DOUBLE-ENTRY SFC ACCOUNTING AUDIT]
   • Locks collateral in Escrow -> Price-Time Priority Matching -> Settles Sectoral Balance Sheets
   • Enforces Global Monetary & Asset Conservation Error < 1e-9 on every tick