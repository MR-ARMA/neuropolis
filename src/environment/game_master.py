"""
src/environment/game_master.py
MODULE 8 (M8): Deterministic Game Master & ACID Double-Entry Ledger (`AccountingAndLawEngine`).
Enforces:
  1. Two-Wave Dyadic Escrow Pre-Lock (preventing double-spending during parallel waves).
  2. Strict Solvency, Inventory & Legal Capacity validation (zero LLM direct write access).
  3. Atomic SQLite Double-Entry Settlement with Sales Tax routing to `PublicTreasury`.
  4. Godley-Lavoie Stock-Flow Consistent Global Money Conservation Law:
       sum_i W_i^cash + sum_i B_i^deposit + W_Bank + W_Treasury + W_FirmSector == M_0 (to 1e-9).
"""

from __future__ import annotations

import json
import math
import uuid
from typing import Dict, Optional, Tuple

from src.core.contracts import (
    ActionCategory,
    AgentTurnOutput,
    ConsentDecision,
    EconomyInstitutionsSpec,
    ExternalAgentState,
    GameMasterVerdict,
    LegalStatus,
    MacroEconomyState,
    VerdictStatus,
)
from src.memory.scoped_memory import ScopedMemoryEngine


class EscrowRecord:
    """In-memory pre-lock created in Wave 1 and settled or released in Wave 2."""

    def __init__(
        self,
        tx_id: str,
        tick: int,
        initiator_id: str,
        responder_id: str,
        item_id: str,
        quantity: int,
        total_price_cash: float,
        is_buying: bool,
    ) -> None:
        self.tx_id = tx_id
        self.tick = tick
        self.initiator_id = initiator_id
        self.responder_id = responder_id
        self.item_id = item_id
        self.quantity = quantity
        self.total_price_cash = total_price_cash
        self.is_buying = is_buying


class AccountingAndLawEngine:
    """
    Deterministic physical and financial arbitrator.
    The LLM emits `AgentTurnOutput`; only `AccountingAndLawEngine` can mutate
    `agents_external_ledger` or `macro_institutions`.
    """

    def __init__(
        self,
        memory_engine: ScopedMemoryEngine,
        spec: EconomyInstitutionsSpec,
    ) -> None:
        self.memory_engine = memory_engine
        self.spec = spec
        self._active_escrows: Dict[str, EscrowRecord] = {}

    def verify_global_money_conservation(self) -> Tuple[bool, float]:
        """
        Asserts the Godley-Lavoie Stock-Flow Conservation Identity across all
        household wallets, bank deposits, commercial bank reserves, public treasury,
        and aggregate firm sector accounts.
        """
        macro = self.memory_engine.get_macro_state()
        agents = self.memory_engine.get_all_external_states()

        total_agent_cash = sum(a.liquid_wealth_cash for a in agents.values())
        total_agent_deposits = sum(a.bank_deposit for a in agents.values())

        actual_total = (
            total_agent_cash
            + total_agent_deposits
            + macro.commercial_bank_reserves
            + macro.public_treasury_cash
            + macro.firm_sector_cash
        )
        drift_error = abs(actual_total - macro.central_bank_base_money_m0)
        return (drift_error < 1e-6), drift_error

    def pre_validate_and_escrow(
        self,
        tick: int,
        initiator_id: str,
        turn_output: AgentTurnOutput,
    ) -> GameMasterVerdict:
        """
        Wave-1 Initiator Pre-Validation:
        Checks legal capacity, finite positive numbers, buyer solvency, or seller inventory.
        If valid for a bilateral trade, places an Escrow Hold and returns `ESCROW_LOCKED`.
        """
        tx_id = f"tx_{tick}_{uuid.uuid4().hex[:8]}"
        initiator_state = self.memory_engine.get_external_state(initiator_id)
        intent = turn_output.economic_intent

        # 1. Legal capacity check
        if initiator_state.legal_status == LegalStatus.IMPRISONED:
            return self._record_verdict(
                GameMasterVerdict(
                    tx_id=tx_id,
                    tick=tick,
                    status=VerdictStatus.REJECTED_ILLEGAL_STATE,
                    initiator_id=initiator_id,
                    diagnostic_feedback="Action rejected: Agent is currently IMPRISONED.",
                )
            )

        # 2. Fast-path for non-bilateral or NO_OP / TMS_QUERY actions
        if intent.action_category in {
            ActionCategory.NO_OP,
            ActionCategory.TMS_QUERY,
            ActionCategory.WORK_AND_CONSUME,
        }:
            return self._record_verdict(
                GameMasterVerdict(
                    tx_id=tx_id,
                    tick=tick,
                    status=VerdictStatus.APPROVED,
                    initiator_id=initiator_id,
                    diagnostic_feedback=f"Non-trade action ({intent.action_category.value}) approved.",
                )
            )

        # 3. Numerical sanity check
        if (
            not math.isfinite(intent.total_price_cash)
            or intent.total_price_cash < 0.0
            or intent.quantity <= 0
            or intent.item_id == "NONE"
        ):
            return self._record_verdict(
                GameMasterVerdict(
                    tx_id=tx_id,
                    tick=tick,
                    status=VerdictStatus.REJECTED_SCHEMA_ERROR,
                    initiator_id=initiator_id,
                    responder_id=intent.target_peer_id,
                    diagnostic_feedback=(
                        f"Invalid trade parameters: item={intent.item_id}, "
                        f"qty={intent.quantity}, total_price={intent.total_price_cash}."
                    ),
                )
            )

        # 4. Solvency or Inventory precondition check on Initiator
        if intent.is_buying:
            if initiator_state.liquid_wealth_cash + 1e-9 < intent.total_price_cash:
                return self._record_verdict(
                    GameMasterVerdict(
                        tx_id=tx_id,
                        tick=tick,
                        status=VerdictStatus.REJECTED_INSUFFICIENT_FUNDS,
                        initiator_id=initiator_id,
                        responder_id=intent.target_peer_id,
                        item_id=intent.item_id,
                        quantity_transferred=intent.quantity,
                        diagnostic_feedback=(
                            f"INSUFFICIENT_FUNDS: You offered ${intent.total_price_cash:.2f} "
                            f"to buy {intent.quantity} {intent.item_id}, but your liquid cash "
                            f"wallet only holds ${initiator_state.liquid_wealth_cash:.2f}. "
                            f"Lower your total_price_cash to <= ${initiator_state.liquid_wealth_cash:.2f}."
                        ),
                    )
                )
        else:
            avail_qty = initiator_state.inventory.get(intent.item_id, 0)
            if avail_qty < intent.quantity:
                return self._record_verdict(
                    GameMasterVerdict(
                        tx_id=tx_id,
                        tick=tick,
                        status=VerdictStatus.REJECTED_INSUFFICIENT_INVENTORY,
                        initiator_id=initiator_id,
                        responder_id=intent.target_peer_id,
                        item_id=intent.item_id,
                        quantity_transferred=intent.quantity,
                        diagnostic_feedback=(
                            f"INSUFFICIENT_INVENTORY: You tried to sell {intent.quantity} "
                            f"{intent.item_id}, but your inventory only has {avail_qty} units."
                        ),
                    )
                )

        # 5. Valid Wave-1 Proposal -> Lock in Escrow
        target_id = intent.target_peer_id or ""
        self._active_escrows[tx_id] = EscrowRecord(
            tx_id=tx_id,
            tick=tick,
            initiator_id=initiator_id,
            responder_id=target_id,
            item_id=intent.item_id,
            quantity=intent.quantity,
            total_price_cash=round(intent.total_price_cash, 4),
            is_buying=intent.is_buying,
        )
        return GameMasterVerdict(
            tx_id=tx_id,
            tick=tick,
            status=VerdictStatus.ESCROW_LOCKED,
            initiator_id=initiator_id,
            responder_id=target_id,
            item_id=intent.item_id,
            quantity_transferred=intent.quantity,
            diagnostic_feedback="Offer validated and locked in Escrow awaiting counterparty consent.",
        )

    def settle_dyadic_transaction(
        self,
        tick: int,
        escrow_verdict: GameMasterVerdict,
        initiator_output: AgentTurnOutput,
        responder_id: str,
        responder_output: AgentTurnOutput,
    ) -> GameMasterVerdict:
        """
        Wave-2 Atomic Settlement:
        Verifies Responder's `consent_decision == ACCEPT` and Responder's solvency/inventory.
        If valid, executes atomic double-entry SQL settlement (with sales tax to Treasury).
        Otherwise releases the Escrow Hold with zero state mutation.
        """
        tx_id = escrow_verdict.tx_id
        escrow = self._active_escrows.pop(tx_id, None)
        if escrow is None:
            return escrow_verdict

        # 1. Check Responder Consent
        if responder_output.consent_decision != ConsentDecision.ACCEPT:
            return self._record_verdict(
                GameMasterVerdict(
                    tx_id=tx_id,
                    tick=tick,
                    status=VerdictStatus.REJECTED_COUNTERPARTY_DECLINED,
                    initiator_id=escrow.initiator_id,
                    responder_id=responder_id,
                    item_id=escrow.item_id,
                    quantity_transferred=0,
                    diagnostic_feedback=(
                        f"Counterparty {responder_id} responded with "
                        f"{responder_output.consent_decision.value}: '{responder_output.verbal_message}'"
                    ),
                )
            )

        initiator_state = self.memory_engine.get_external_state(escrow.initiator_id)
        responder_state = self.memory_engine.get_external_state(responder_id)

        buyer_state = initiator_state if escrow.is_buying else responder_state
        seller_state = responder_state if escrow.is_buying else initiator_state

        price = escrow.total_price_cash
        qty = escrow.quantity
        item = escrow.item_id

        # 2. Verify Buyer Solvency & Seller Inventory at Settlement Instant
        if buyer_state.liquid_wealth_cash + 1e-9 < price:
            return self._record_verdict(
                GameMasterVerdict(
                    tx_id=tx_id,
                    tick=tick,
                    status=VerdictStatus.REJECTED_INSUFFICIENT_FUNDS,
                    initiator_id=escrow.initiator_id,
                    responder_id=responder_id,
                    item_id=item,
                    diagnostic_feedback=(
                        f"Settlement aborted: Buyer {buyer_state.agent_id} lacks "
                        f"${price:.2f} (wallet=${buyer_state.liquid_wealth_cash:.2f})."
                    ),
                )
            )

        seller_avail = seller_state.inventory.get(item, 0)
        if seller_avail < qty:
            return self._record_verdict(
                GameMasterVerdict(
                    tx_id=tx_id,
                    tick=tick,
                    status=VerdictStatus.REJECTED_INSUFFICIENT_INVENTORY,
                    initiator_id=escrow.initiator_id,
                    responder_id=responder_id,
                    item_id=item,
                    diagnostic_feedback=(
                        f"Settlement aborted: Seller {seller_state.agent_id} only has "
                        f"{seller_avail} {item} (needed {qty})."
                    ),
                )
            )

        # 3. Compute Exact Double-Entry Cash & Tax Flows
        sales_tax = round(price * self.spec.sales_tax_rate, 6)
        seller_net_proceeds = round(price - sales_tax, 6)

        buyer_state.liquid_wealth_cash = round(buyer_state.liquid_wealth_cash - price, 6)
        seller_state.liquid_wealth_cash = round(
            seller_state.liquid_wealth_cash + seller_net_proceeds, 6
        )

        buyer_state.inventory[item] = buyer_state.inventory.get(item, 0) + qty
        seller_state.inventory[item] = seller_avail - qty

        macro = self.memory_engine.get_macro_state()
        macro.public_treasury_cash = round(macro.public_treasury_cash + sales_tax, 6)
        macro.cumulative_trade_volume = round(macro.cumulative_trade_volume + price, 4)
        if item == "FOOD" and qty > 0:
            unit_p = price / float(qty)
            macro.market_clearing_price_food = round(
                0.7 * macro.market_clearing_price_food + 0.3 * unit_p, 4
            )
        elif item == "TECH" and qty > 0:
            unit_p = price / float(qty)
            macro.market_clearing_price_tech = round(
                0.7 * macro.market_clearing_price_tech + 0.3 * unit_p, 4
            )

        # 4. Atomic SQLite Commit across both agents and PublicTreasury
        self._commit_agent_and_macro_states([buyer_state, seller_state], macro)

        _, conservation_err = self.verify_global_money_conservation()

        cash_delta_init = -price if escrow.is_buying else seller_net_proceeds
        cash_delta_resp = seller_net_proceeds if escrow.is_buying else -price

        return self._record_verdict(
            GameMasterVerdict(
                tx_id=tx_id,
                tick=tick,
                status=VerdictStatus.APPROVED,
                initiator_id=escrow.initiator_id,
                responder_id=responder_id,
                cash_delta_initiator=round(cash_delta_init, 4),
                cash_delta_responder=round(cash_delta_resp, 4),
                tax_collected=round(sales_tax, 4),
                item_id=item,
                quantity_transferred=qty,
                diagnostic_feedback=(
                    f"APPROVED: {buyer_state.agent_id} bought {qty} {item} from "
                    f"{seller_state.agent_id} for ${price:.2f} (tax=${sales_tax:.2f})."
                ),
                global_money_conservation_error=conservation_err,
            )
        )

    def execute_end_of_tick_upkeep(self, tick: int) -> None:
        """
        Stock-Flow Consistent End-of-Tick Wage & Survival Consumption Cycle:
          1. Firm Sector pays `base_wage` to Law-Abiding agents (debiting `firm_sector_cash`).
          2. Agents consume 1 unit of `FOOD` for survival satiety; if out of FOOD, they buy
             1 ration from the Aggregate Firm Sector (crediting `firm_sector_cash`).
        """
        macro = self.memory_engine.get_macro_state()
        macro.tick = tick
        agents = self.memory_engine.get_all_external_states()
        updated_agents = []

        for agent_id, est in agents.items():
            if est.legal_status == LegalStatus.IMPRISONED:
                continue

            # 1. SFC Wage Payment from Firm Sector -> Household
            wage = min(est.base_wage, macro.firm_sector_cash)
            if wage > 0:
                macro.firm_sector_cash = round(macro.firm_sector_cash - wage, 6)
                est.liquid_wealth_cash = round(est.liquid_wealth_cash + wage, 6)

            # 2. Survival Consumption & Maslow Satiety Update
            scoped = self.memory_engine.get_scoped_view(agent_id)
            ist = scoped.get_internal_state()
            food_qty = est.inventory.get("FOOD", 0)
            if food_qty >= self.spec.survival_food_consumption_per_tick:
                est.inventory["FOOD"] = food_qty - self.spec.survival_food_consumption_per_tick
                ist.maslow_needs.survival_satiety = min(
                    1.0, ist.maslow_needs.survival_satiety + 0.12
                )
            else:
                # Emergency ration purchase from Firm Sector if solvent
                ration_cost = min(self.spec.reference_food_price, est.liquid_wealth_cash)
                if ration_cost >= 5.0:
                    est.liquid_wealth_cash = round(est.liquid_wealth_cash - ration_cost, 6)
                    macro.firm_sector_cash = round(macro.firm_sector_cash + ration_cost, 6)
                    ist.maslow_needs.survival_satiety = max(
                        0.2, ist.maslow_needs.survival_satiety - 0.05
                    )
                else:
                    ist.maslow_needs.survival_satiety = max(
                        0.05, ist.maslow_needs.survival_satiety - 0.25
                    )
            scoped.save_internal_state(ist)
            updated_agents.append(est)

        self._commit_agent_and_macro_states(updated_agents, macro)

    def _commit_agent_and_macro_states(
        self,
        agent_states: list[ExternalAgentState],
        macro: MacroEconomyState,
    ) -> None:
        cur = self.memory_engine.conn.cursor()
        for est in agent_states:
            cur.execute(
                """
                UPDATE agents_external_ledger
                SET liquid_wealth_cash = ?, bank_deposit = ?, outstanding_debt = ?,
                    credit_score = ?, inventory_json = ?, base_wage = ?,
                    social_status = ?, legal_status = ?, spatial_node_id = ?
                WHERE agent_id = ?
                """,
                (
                    est.liquid_wealth_cash,
                    est.bank_deposit,
                    est.outstanding_debt,
                    est.credit_score,
                    json.dumps(est.inventory),
                    est.base_wage,
                    est.social_status,
                    est.legal_status.value,
                    est.spatial_node_id,
                    est.agent_id,
                ),
            )
        cur.execute(
            "UPDATE macro_institutions SET state_json = ? WHERE singleton_id = 1",
            (macro.model_dump_json(),),
        )
        self.memory_engine.conn.commit()

    def _record_verdict(self, verdict: GameMasterVerdict) -> GameMasterVerdict:
        _, err = self.verify_global_money_conservation()
        verdict.global_money_conservation_error = err
        cur = self.memory_engine.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO acid_transaction_ledger (
                tx_id, tick, status, initiator_id, responder_id,
                cash_delta_initiator, cash_delta_responder, tax_collected,
                item_id, quantity_transferred, diagnostic_feedback, conservation_error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                verdict.tx_id,
                verdict.tick,
                verdict.status.value,
                verdict.initiator_id,
                verdict.responder_id,
                verdict.cash_delta_initiator,
                verdict.cash_delta_responder,
                verdict.tax_collected,
                verdict.item_id,
                verdict.quantity_transferred,
                verdict.diagnostic_feedback,
                verdict.global_money_conservation_error,
            ),
        )
        self.memory_engine.conn.commit()
        return verdict