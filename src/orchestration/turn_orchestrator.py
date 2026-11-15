"""
src/orchestration/turn_orchestrator.py
MODULE 7 (M7): Two-Wave Dyadic Turn Orchestrator & Self-Correction Retry Circuit.
Implements Audit Flaw 3 Resolution:
  - Wave 1 (Initiator): Appraises state -> Compiles XML Span Prompt -> Executes Steered
    Brain (or TMS Router) -> Validates via `GameMaster.pre_validate_and_escrow` with up
    to `K_retry = 2` self-correction retries if insolvent or out-of-inventory.
  - Wave 2 (Responder): Injects Initiator's verified Escrow proposal, verbal message,
    and latent mirror vector into `SPAN_INCOMING_MSG` -> Executes Responder Steered Brain
    -> Settles atomically via `GameMaster.settle_dyadic_transaction` -> Updates Slovic
    asymmetric trust & Ebbinghaus episodic memory.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

from src.brain.inference_brain import (
    BlackBoxAPIBrain,
    PromptSpanCompiler,
    WhiteBoxPyTorchBrain,
)
from src.core.contracts import (
    ActionCategory,
    AgentTurnOutput,
    ConsentDecision,
    EconomicActionIntent,
    ExpertiseDomain,
    GameMasterVerdict,
    SimulationSpec,
    SteeringTelemetryPacket,
    VerdictStatus,
)
from src.environment.bilateral_market import BilateralTrustMarket
from src.environment.game_master import AccountingAndLawEngine
from src.memory.scoped_memory import ScopedMemoryEngine
from src.orchestration.cognitive_appraisal import CognitiveAppraisalAndToMEngine


class TurnExecutionRecord:
    """Complete telemetry record for a single dyadic negotiation round."""

    def __init__(
        self,
        tick: int,
        initiator_id: str,
        responder_id: str,
        initiator_output: AgentTurnOutput,
        responder_output: Optional[AgentTurnOutput],
        initiator_telemetry: SteeringTelemetryPacket,
        responder_telemetry: Optional[SteeringTelemetryPacket],
        final_verdict: GameMasterVerdict,
        retries_used: int,
    ) -> None:
        self.tick = tick
        self.initiator_id = initiator_id
        self.responder_id = responder_id
        self.initiator_output = initiator_output
        self.responder_output = responder_output
        self.initiator_telemetry = initiator_telemetry
        self.responder_telemetry = responder_telemetry
        self.final_verdict = final_verdict
        self.retries_used = retries_used


class TurnOrchestrator:
    """
    Coordinates the Two-Wave Initiator/Responder protocol across all modules.
    """

    def __init__(
        self,
        spec: SimulationSpec,
        memory_engine: ScopedMemoryEngine,
        brain: Union[WhiteBoxPyTorchBrain, BlackBoxAPIBrain],
        prompt_compiler: PromptSpanCompiler,
        appraisal_engine: CognitiveAppraisalAndToMEngine,
        game_master: AccountingAndLawEngine,
        market: BilateralTrustMarket,
    ) -> None:
        self.spec = spec
        self.memory_engine = memory_engine
        self.brain = brain
        self.prompt_compiler = prompt_compiler
        self.appraisal_engine = appraisal_engine
        self.game_master = game_master
        self.market = market

    def execute_two_wave_dyadic_interaction(
        self,
        tick: int,
        initiator_id: str,
        responder_id: str,
        market_broadcast_msg: str = "",
    ) -> TurnExecutionRecord:
        """
        Executes a complete conflict-free dyadic interaction `(initiator_id -> responder_id)`:
          1. Wave 1: Initiator generates offer (with up to K_retry = 2 Game Master self-corrections).
          2. If Initiator emitted a `TMS_QUERY`, updates Bayesian TMS directory and completes turn.
          3. Wave 2: Responder evaluates Escrowed offer and emits `ACCEPT` or `REJECT`.
          4. Atomic Settlement + Slovic Trust Update + Ebbinghaus Memory Encoding.
        """
        init_scoped = self.memory_engine.get_scoped_view(initiator_id)
        init_internal = init_scoped.get_internal_state()
        init_external = self.memory_engine.get_external_state(initiator_id)

        # Dual-Pass Scoped Retrieval for Initiator
        init_edges, init_models, init_memories = init_scoped.dual_pass_scoped_retrieval(
            direct_partner_id=responder_id,
            incoming_message_text=market_broadcast_msg,
            current_tick=tick,
        )

        # Compute relationship-modulated reservation price hint
        edge_to_resp = init_edges.get(responder_id)
        res_price = (
            self.market.compute_reservation_price("FOOD", edge_to_resp, is_seller=False)
            if edge_to_resp
            else self.spec.economy_and_institutions.reference_food_price
        )

        init_endogenous = self.appraisal_engine.appraise_turn_state(
            internal_state=init_internal,
            external_state=init_external,
            counterparty_id=responder_id,
            scoped_edges=init_edges,
            scoped_models=init_models,
            incoming_message=market_broadcast_msg or f"Negotiating with {responder_id}.",
            proposed_price_hint=res_price,
        )

        # --- WAVE 1: INITIATOR GENERATION & SELF-CORRECTION RETRY CIRCUIT ---
        max_retries = self.spec.economy_and_institutions.max_action_retries
        retry_feedback: Optional[str] = None
        retries_used = 0
        escrow_verdict: Optional[GameMasterVerdict] = None
        init_output: Optional[AgentTurnOutput] = None
        init_telemetry: Optional[SteeringTelemetryPacket] = None
        speaker_latent: Optional[List[float]] = None

        for attempt in range(max_retries + 1):
            prompt_packet = self.prompt_compiler.compile_prompt_packet(
                tick=tick,
                internal_state=init_internal,
                external_state=init_external,
                scoped_edges=init_edges,
                scoped_models=init_models,
                episodic_memories=init_memories,
                endogenous_state=init_endogenous,
                counterparty_id=responder_id,
                incoming_message=market_broadcast_msg or f"Ready to trade with {responder_id}.",
                incoming_proposal_summary=f"Suggested fair reservation price: ${res_price:.2f} for 1 FOOD.",
                is_responder_wave=False,
                retry_diagnostic_feedback=retry_feedback,
            )

            init_output, init_telemetry, speaker_latent = self.brain.generate_turn(
                prompt_packet=prompt_packet,
                scoped_memory=init_scoped,
                external_state=init_external,
                domain_hint=init_internal.specialty_domain,
                is_responder_wave=False,
            )
            init_output.economic_intent.target_peer_id = responder_id

            escrow_verdict = self.game_master.pre_validate_and_escrow(
                tick=tick,
                initiator_id=initiator_id,
                turn_output=init_output,
            )

            if escrow_verdict.status in {VerdictStatus.ESCROW_LOCKED, VerdictStatus.APPROVED}:
                break

            # Rejected by Game Master -> feed exact diagnostic back into next retry
            retries_used += 1
            retry_feedback = escrow_verdict.diagnostic_feedback

        assert init_output is not None and init_telemetry is not None and escrow_verdict is not None

        # If Initiator still failed after K_retry attempts, fall back to safe NO_OP
        if escrow_verdict.status not in {VerdictStatus.ESCROW_LOCKED, VerdictStatus.APPROVED}:
            init_output.economic_intent.action_category = ActionCategory.NO_OP
            return TurnExecutionRecord(
                tick=tick,
                initiator_id=initiator_id,
                responder_id=responder_id,
                initiator_output=init_output,
                responder_output=None,
                initiator_telemetry=init_telemetry,
                responder_telemetry=None,
                final_verdict=escrow_verdict,
                retries_used=retries_used,
            )

        # Handle TMS Expert Query Turn (when EntropyTMSRouter halted generation)
        if init_output.economic_intent.action_category == ActionCategory.TMS_QUERY and init_output.tms_query:
            q = init_output.tms_query
            init_scoped.update_tms_peer_belief(
                target_id=q.queried_expert_id,
                domain=q.domain,
                outcome_quality_y=0.85,
                tick=tick,
            )
            macro = self.memory_engine.get_macro_state()
            macro.cumulative_tms_queries += 1
            self.memory_engine.update_macro_state(macro)
            return TurnExecutionRecord(
                tick=tick,
                initiator_id=initiator_id,
                responder_id=q.queried_expert_id,
                initiator_output=init_output,
                responder_output=None,
                initiator_telemetry=init_telemetry,
                responder_telemetry=None,
                final_verdict=escrow_verdict,
                retries_used=retries_used,
            )

        # If non-bilateral action (e.g., WORK_AND_CONSUME / NO_OP), return immediately
        if escrow_verdict.status != VerdictStatus.ESCROW_LOCKED:
            return TurnExecutionRecord(
                tick=tick,
                initiator_id=initiator_id,
                responder_id=responder_id,
                initiator_output=init_output,
                responder_output=None,
                initiator_telemetry=init_telemetry,
                responder_telemetry=None,
                final_verdict=escrow_verdict,
                retries_used=retries_used,
            )

        # --- WAVE 2: RESPONDER EVALUATION & ATOMIC DOUBLE-ENTRY SETTLEMENT ---
        resp_scoped = self.memory_engine.get_scoped_view(responder_id)
        resp_internal = resp_scoped.get_internal_state()
        resp_external = self.memory_engine.get_external_state(responder_id)

        # Pass 1 + Pass 2 Entity-Mention Scoped Retrieval on Initiator's verbal message!
        resp_edges, resp_models, resp_memories = resp_scoped.dual_pass_scoped_retrieval(
            direct_partner_id=initiator_id,
            incoming_message_text=init_output.verbal_message,
            current_tick=tick,
        )

        intent = init_output.economic_intent
        action_verb = "BUY" if intent.is_buying else "SELL"
        proposal_summary = (
            f"{initiator_id} proposes to {action_verb} {intent.quantity} {intent.item_id} "
            f"for total cash ${intent.total_price_cash:.2f} (Funds/Inventory verified in Escrow)."
        )

        resp_endogenous = self.appraisal_engine.appraise_turn_state(
            internal_state=resp_internal,
            external_state=resp_external,
            counterparty_id=initiator_id,
            scoped_edges=resp_edges,
            scoped_models=resp_models,
            incoming_message=init_output.verbal_message,
            proposed_price_hint=intent.total_price_cash,
        )

        resp_prompt_packet = self.prompt_compiler.compile_prompt_packet(
            tick=tick,
            internal_state=resp_internal,
            external_state=resp_external,
            scoped_edges=resp_edges,
            scoped_models=resp_models,
            episodic_memories=resp_memories,
            endogenous_state=resp_endogenous,
            counterparty_id=initiator_id,
            incoming_message=init_output.verbal_message,
            incoming_proposal_summary=proposal_summary,
            is_responder_wave=True,
        )
        resp_prompt_packet.incoming_speaker_latent_vec = speaker_latent

        resp_output, resp_telemetry, _ = self.brain.generate_turn(
            prompt_packet=resp_prompt_packet,
            scoped_memory=resp_scoped,
            external_state=resp_external,
            domain_hint=resp_internal.specialty_domain,
            is_responder_wave=True,
        )

        # Economic rationality guardrail on Responder consent when offer is mutually profitable & solvent
        fair_val = (
            self.spec.economy_and_institutions.reference_food_price
            if intent.item_id == "FOOD"
            else self.spec.economy_and_institutions.reference_tech_price
        ) * intent.quantity
        edge_resp_to_init = resp_edges.get(initiator_id)
        trust_val = edge_resp_to_init.trust_tau if edge_resp_to_init else 0.0

        if intent.is_buying:
            # Initiator is buying -> Responder is selling
            has_stock = resp_external.inventory.get(intent.item_id, 0) >= intent.quantity
            attractive_price = intent.total_price_cash >= fair_val * (0.82 - 0.10 * trust_val)
            if has_stock and attractive_price and trust_val > -0.5:
                resp_output.consent_decision = ConsentDecision.ACCEPT
        else:
            # Initiator is selling -> Responder is buying
            can_afford = resp_external.liquid_wealth_cash >= intent.total_price_cash
            attractive_price = intent.total_price_cash <= fair_val * (1.18 + 0.10 * trust_val)
            if can_afford and attractive_price and trust_val > -0.5:
                resp_output.consent_decision = ConsentDecision.ACCEPT

        # Settle atomically via Game Master
        final_verdict = self.game_master.settle_dyadic_transaction(
            tick=tick,
            escrow_verdict=escrow_verdict,
            initiator_output=init_output,
            responder_id=responder_id,
            responder_output=resp_output,
        )

        # Slovic Asymmetric Trust & Episodic Memory Update
        self.market.update_dyadic_trust_and_memories(
            tick=tick,
            initiator_id=initiator_id,
            responder_id=responder_id,
            initiator_output=init_output,
            responder_output=resp_output,
            verdict=final_verdict,
        )

        return TurnExecutionRecord(
            tick=tick,
            initiator_id=initiator_id,
            responder_id=responder_id,
            initiator_output=init_output,
            responder_output=resp_output,
            initiator_telemetry=init_telemetry,
            responder_telemetry=resp_telemetry,
            final_verdict=final_verdict,
            retries_used=retries_used,
        )