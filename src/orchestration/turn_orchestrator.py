from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple
import numpy as np

from src.brain.deliberation_budget import (
    DeliberationAllocation,
    DeliberationBudgetController,
)
from src.brain.memory_tms import EpisodicMemoryTMS, RetrievalResult
from src.core.contracts import (
    AgentActionType,
    AgentTurnRecord,
    AssetType,
    CDAOrder,
    HouseholdBalanceSheet,
    OrderBookSnapshot,
    OrderSide,
    OrderType,
    PolicyMode,
    ProbeGateMeasurement,
    SimulationConfig,
    Stage2ActionPayload,
)
from src.environment.game_master import (
    ActionExecutionOutcome,
    GameMaster,
    TickMacroSummary,
)
from src.environment.needs_engine import NeedsEngine, TwoAssetAccounting


class PromptLeakageError(ValueError):
    """Raised if any banned behavioral economics or priming term leaks into a compiled prompt."""


BANNED_LEAKAGE_PATTERNS: Tuple[str, ...] = (
    r"\bunfair\b",
    r"\bfairness\b",
    r"\bultimatum\b",
    r"\bloan\b",
    r"\bdebt\b",
    r"\bgouging\b",
    r"\bgouge\b",
    r"\binequity\b",
    r"\bfehr\b",
    r"\bschmidt\b",
    r"\bkahneman\b",
    r"\btversky\b",
    r"\bloss\s+aversion\b",
    r"\bprospect\s+theory\b",
    r"\bscarcity\b",
    r"\bpanic\b",
    r"\bhoarding\b",
    r"\bbank\s+run\b",
    r"\brecession\b",
    r"\bstagflation\b",
    r"\baltruism\b",
    r"\bselfish\b",
    r"\bgreedy\b",
    r"\bhand-to-mouth\b",
    r"\btrust\b",
    r"\bbetray\b",
    r"\bpunish\b",
    r"\bhuman\b",
    r"\bhumans\b",
    r"\bpsychology\b",
    r"\bsystem\s*1\b",
    r"\bsystem\s*2\b",
    r"\bhyperbolic\b",
    r"\bimpulsive\b",
)
_COMPILED_LEAKAGE_REGEX = re.compile(
    "|".join(BANNED_LEAKAGE_PATTERNS), flags=re.IGNORECASE
)


def assert_prompt_sanitized(prompt_text: str) -> None:
    """Hard runtime firewall guaranteeing zero behavioral framing or theory leakage."""
    match = _COMPILED_LEAKAGE_REGEX.search(prompt_text)
    if match is not None:
        raise PromptLeakageError(
            f"Sanitization Firewall Triggered: banned priming term '{match.group(0)}' detected in prompt."
        )


@dataclass
class TwoStageGenerationOutput:
    stage1_cot_text: str
    cot_tokens_generated: int
    cot_truncated_by_bottleneck: bool
    probe_measurement: Optional[ProbeGateMeasurement]
    stage2_raw_json: str
    stage2_action: Stage2ActionPayload
    json_valid: bool
    abstain: bool
    retries_used: int
    action_tokens_generated: int
    latency_ms: float


class TwoStageLLMControllerProtocol(Protocol):
    def execute_two_stage_turn(
        self,
        stage1_system_prompt: str,
        stage1_user_prompt: str,
        stage2_instruction: str,
        cot_max_tokens: int,
        action_max_tokens: int,
        agent_id: str,
        tick: int,
        parse_action_fn: Any,
    ) -> TwoStageGenerationOutput:
        ...


class IsomorphicVocabulary:
    ASSET_TO_TOKEN = {
        AssetType.CONSUMABLE: "RESOURCE_A",
        AssetType.CAPITAL: "RESOURCE_B",
    }
    TOKEN_TO_ASSET = {
        "RESOURCE_A": AssetType.CONSUMABLE,
        "TOKEN_A": AssetType.CONSUMABLE,
        "CONSUMABLE": AssetType.CONSUMABLE,
        "RESOURCE_B": AssetType.CAPITAL,
        "TOKEN_B": AssetType.CAPITAL,
        "CAPITAL": AssetType.CAPITAL,
    }

    @classmethod
    def parse_sanitized_action_json(
        cls, raw_text: str
    ) -> Tuple[Stage2ActionPayload, bool, Optional[str]]:
        if not raw_text or not raw_text.strip():
            return (
                Stage2ActionPayload(action_type=AgentActionType.ABSTAIN),
                False,
                "Empty Stage 2 output",
            )

        json_match = re.search(r"\{.*?\}", raw_text, flags=re.DOTALL)
        if not json_match:
            return (
                Stage2ActionPayload(action_type=AgentActionType.ABSTAIN),
                False,
                f"No JSON object found: {raw_text[:60]}",
            )

        candidate = json_match.group(0)
        try:
            data = json.loads(candidate)
        except Exception as exc:
            cleaned = re.sub(r",\s*\}", "}", candidate)
            try:
                data = json.loads(cleaned)
            except Exception:
                return (
                    Stage2ActionPayload(action_type=AgentActionType.ABSTAIN),
                    False,
                    f"JSON decode error: {str(exc)[:50]}",
                )

        if not isinstance(data, dict):
            return (
                Stage2ActionPayload(action_type=AgentActionType.ABSTAIN),
                False,
                "Parsed JSON is not an object",
            )

        op_raw = str(data.get("operation", data.get("action_type", ""))).strip().upper()
        op_map = {
            "POST_BOARD": AgentActionType.PLACE_ORDER,
            "PLACE_ORDER": AgentActionType.PLACE_ORDER,
            "CANCEL_BOARD": AgentActionType.CANCEL_ORDER,
            "CANCEL_ORDER": AgentActionType.CANCEL_ORDER,
            "PROTOCOL_B_OUT": AgentActionType.LIQUIDATE_CAPITAL,
            "LIQUIDATE_CAPITAL": AgentActionType.LIQUIDATE_CAPITAL,
            "PROTOCOL_B_IN": AgentActionType.INVEST_CAPITAL,
            "INVEST_CAPITAL": AgentActionType.INVEST_CAPITAL,
            "WAIT": AgentActionType.CONSUME_AND_HOLD,
            "CONSUME_AND_HOLD": AgentActionType.CONSUME_AND_HOLD,
            "ABSTAIN": AgentActionType.ABSTAIN,
        }
        if op_raw not in op_map:
            return (
                Stage2ActionPayload(action_type=AgentActionType.ABSTAIN),
                False,
                f"Unknown operation '{op_raw}'",
            )
        act_type = op_map[op_raw]

        asset_raw = str(data.get("resource", data.get("asset", "RESOURCE_A"))).strip().upper()
        asset = cls.TOKEN_TO_ASSET.get(asset_raw, AssetType.CONSUMABLE)

        side_raw = data.get("side")
        side: Optional[OrderSide] = None
        if side_raw is not None:
            s_up = str(side_raw).strip().upper()
            if s_up in ("BID", "BUY"):
                side = OrderSide.BID
            elif s_up in ("ASK", "SELL"):
                side = OrderSide.ASK

        ot_raw = str(data.get("mode", data.get("order_type", "LIMIT"))).strip().upper()
        order_type = OrderType.MARKET if ot_raw == "MARKET" else OrderType.LIMIT

        raw_price = data.get("rate", data.get("limit_price"))
        limit_price: Optional[float] = (
            float(raw_price) if raw_price is not None and float(raw_price) > 0 else None
        )
        qty = float(data.get("units", data.get("quantity", 0.0)) or 0.0)
        cancel_id = data.get("order_id", data.get("cancel_order_id"))
        transfer_u = float(
            data.get("transfer_units", data.get("capital_transfer_units", qty)) or 0.0
        )

        try:
            payload = Stage2ActionPayload(
                action_type=act_type,
                asset=asset,
                side=side,
                order_type=order_type,
                limit_price=limit_price,
                quantity=qty if act_type == AgentActionType.PLACE_ORDER else 0.0,
                cancel_order_id=str(cancel_id) if cancel_id else None,
                capital_transfer_units=(
                    transfer_u
                    if act_type
                    in (
                        AgentActionType.LIQUIDATE_CAPITAL,
                        AgentActionType.INVEST_CAPITAL,
                    )
                    else 0.0
                ),
            )
            return (payload, True, None)
        except Exception as exc:
            return (
                Stage2ActionPayload(action_type=AgentActionType.ABSTAIN),
                False,
                f"Schema validation error: {str(exc)[:80]}",
            )


class SanitizedPromptCompiler:
    """
    Compiles sanitized, jargon-free prompts for Stage 1 (Private Appraisal) and Stage 2.
    Instructs the node to appraise board quotes & recalled logs first so the opening
    20-token Primary Appraisal window reflects board conditions rather than copying state headers.
    """

    STAGE1_SYSTEM_PROMPT = (
        "You operate an autonomous node in a multi-cycle resource allocation protocol. "
        "Every cycle drains 1.0 unit of RESOURCE_A to sustain Vitality_Index (if Vitality_Index "
        "reaches 0.00, the node terminates permanently) and deducts Fixed_Cycle_Drain in Pool_Credits. "
        "RESOURCE_B yields periodic Pool_Credits each cycle; converting between RESOURCE_B and "
        "Pool_Credits via Transfer_Protocol_B incurs a nonlinear conversion fee. "
        "Exchange_Board_A matches peer BID and ASK postings strictly when BID >= ASK. "
        "First appraise the board quotes and recalled logs relative to prior clearing rates, then analyze your reserves concisely."
    )

    STAGE2_JSON_INSTRUCTION = (
        "Emit ONLY a single valid JSON object with no extra text. Allowed operations:\n"
        '1. Post on board: {"operation":"POST_BOARD","resource":"RESOURCE_A|RESOURCE_B","side":"BID|ASK","mode":"LIMIT|MARKET","rate":<number>,"units":<number>}\n'
        '2. Cancel open entry: {"operation":"CANCEL_BOARD","order_id":"<id>"}\n'
        '3. Convert RESOURCE_B to Pool_Credits: {"operation":"PROTOCOL_B_OUT","transfer_units":<number>}\n'
        '4. Convert Pool_Credits to RESOURCE_B: {"operation":"PROTOCOL_B_IN","transfer_units":<number>}\n'
        '5. No operation: {"operation":"WAIT"}'
    )

    @classmethod
    def _format_board(cls, snap: OrderBookSnapshot) -> str:
        tok = IsomorphicVocabulary.ASSET_TO_TOKEN[snap.asset]
        bids_str = (
            ", ".join(f"{lvl.price:.2f}x{lvl.total_quantity:.1f}" for lvl in snap.bids_top5)
            or "EMPTY"
        )
        asks_str = (
            ", ".join(f"{lvl.price:.2f}x{lvl.total_quantity:.1f}" for lvl in snap.asks_top5)
            or "EMPTY"
        )
        last_str = f"{snap.last_trade_price:.2f}" if snap.last_trade_price is not None else "NONE"
        return (
            f"Exchange_Board[{tok}]: Last_Cleared={last_str} | Cycle_Vol={snap.tick_volume:.1f} | "
            f"Top_BIDs=[{bids_str}] | Top_ASKs=[{asks_str}]"
        )

    @classmethod
    def compile_stage1_prompt(
        cls,
        tick: int,
        sheet: HouseholdBalanceSheet,
        board_snapshots: Dict[AssetType, OrderBookSnapshot],
        open_orders: List[CDAOrder],
        retrieval: RetrievalResult,
        peer_directory: List[Tuple[str, float, int]],
        cfg: SimulationConfig,
    ) -> Tuple[str, str]:
        kmv_1unit_fee = TwoAssetAccounting.compute_kmv_adjustment_cost_cash(
            1.0, sheet.illiquid_capital, cfg.economy
        )
        open_str = (
            "; ".join(
                f"{o.order_id}({IsomorphicVocabulary.ASSET_TO_TOKEN[o.asset]} {o.side.value} "
                f"{o.remaining_quantity:.1f}@{o.limit_price:.2f})"
                for o in open_orders[:4]
            )
            if open_orders
            else "NONE"
        )
        mem_lines = (
            "\n".join(f"  - {r.summary_text}" for r in retrieval.records)
            if retrieval.records
            else "  - NONE"
        )
        peer_str = (
            ", ".join(f"{pid}(rate={rate:.2f}, n={cnt})" for pid, rate, cnt in peer_directory)
            if peer_directory
            else "NONE"
        )
        boards_text = "\n".join(
            cls._format_board(snap) for snap in board_snapshots.values()
        )

        user_prompt = (
            f"=== CYCLE {tick:03d} | NODE {sheet.agent_id} ===\n"
            f"STATE:\n"
            f"  Pool_Credits={sheet.liquid_cash:.2f} (Locked_Escrow={sheet.escrow_cash:.2f}, Floor_Bound={sheet.borrowing_limit_b_bar:.2f})\n"
            f"  RESOURCE_A={sheet.consumable_inventory:.2f} (Locked={sheet.escrow_consumable:.2f}, Cycle_Drain={cfg.economy.subsistence_min_c:.1f})\n"
            f"  RESOURCE_B={sheet.illiquid_capital:.2f} (Locked={sheet.escrow_capital:.2f}, Par_Rate={cfg.economy.capital_unit_par_value:.1f}, "
            f"Cycle_Yield={cfg.economy.capital_return_rate_rk:.3f}, Fee_For_1.0_Transfer={kmv_1unit_fee:.2f})\n"
            f"  Fixed_Cycle_Drain={sheet.periodic_obligation_r:.2f} | Base_Cycle_Inflow={sheet.base_wage:.2f} | Base_RESOURCE_A_Output={sheet.base_consumable_production_q:.2f}\n"
            f"  Vitality_Index={sheet.needs.health:.2f} | Unmet_Drain_Streak={sheet.needs.subsistence_deficit_streak}\n"
            f"OPEN_BOARD_ENTRIES: {open_str}\n"
            f"EXCHANGE_BOARDS:\n{boards_text}\n"
            f"PEER_DIRECTORY: {peer_str}\n"
            f"RECALLED_CYCLE_LOGS (Top-{retrieval.top_k_used}):\n{mem_lines}\n"
            f"First appraise the board quotes and recalled logs relative to prior clearing rates, then analyze your reserves concisely."
        )

        assert_prompt_sanitized(cls.STAGE1_SYSTEM_PROMPT)
        assert_prompt_sanitized(user_prompt)
        assert_prompt_sanitized(cls.STAGE2_JSON_INSTRUCTION)
        return (cls.STAGE1_SYSTEM_PROMPT, user_prompt)


class ClosedFormBenchmarkPolicy:
    @staticmethod
    def select_action(
        tick: int,
        sheet: HouseholdBalanceSheet,
        gm: GameMaster,
    ) -> Stage2ActionPayload:
        c_book = gm.exchange.get_book(AssetType.CONSUMABLE)
        c_snap = c_book.get_snapshot(tick)
        anchor = c_snap.mid_price or c_snap.last_trade_price or 12.0
        spendable = max(0.0, sheet.liquid_cash - sheet.borrowing_limit_b_bar)

        if (
            sheet.consumable_inventory < 1.5
            and spendable < anchor
            and sheet.illiquid_capital >= 1.0
        ):
            return Stage2ActionPayload(
                action_type=AgentActionType.LIQUIDATE_CAPITAL,
                asset=AssetType.CAPITAL,
                capital_transfer_units=1.0,
            )

        if sheet.consumable_inventory >= 2.5:
            ask_p = c_book.quantize_price(
                (c_snap.best_bid + 0.10) if c_snap.best_bid is not None else anchor * 1.02
            )
            qty = min(2.0, max(1.0, round(sheet.consumable_inventory - 1.5, 2)))
            return Stage2ActionPayload(
                action_type=AgentActionType.PLACE_ORDER,
                asset=AssetType.CONSUMABLE,
                side=OrderSide.ASK,
                order_type=OrderType.LIMIT,
                limit_price=ask_p,
                quantity=qty,
            )

        if sheet.consumable_inventory < 2.5 and spendable >= 5.0:
            target_bid = (
                c_snap.best_ask if c_snap.best_ask is not None else anchor * 1.04
            )
            bid_p = c_book.quantize_price(min(spendable, target_bid))
            if bid_p >= gm.cda_cfg.min_order_price and spendable >= bid_p:
                return Stage2ActionPayload(
                    action_type=AgentActionType.PLACE_ORDER,
                    asset=AssetType.CONSUMABLE,
                    side=OrderSide.BID,
                    order_type=OrderType.LIMIT,
                    limit_price=bid_p,
                    quantity=1.0,
                )

        fee_1 = TwoAssetAccounting.compute_kmv_adjustment_cost_cash(
            1.0, sheet.illiquid_capital, gm.economy_cfg
        )
        invest_cost = gm.economy_cfg.capital_unit_par_value + fee_1
        if (
            sheet.consumable_inventory >= 2.5
            and sheet.liquid_cash > invest_cost + 5.0 * sheet.periodic_obligation_r
        ):
            return Stage2ActionPayload(
                action_type=AgentActionType.INVEST_CAPITAL,
                asset=AssetType.CAPITAL,
                capital_transfer_units=1.0,
            )

        return Stage2ActionPayload(action_type=AgentActionType.CONSUME_AND_HOLD)


class TurnOrchestrator:
    def __init__(
        self,
        config: SimulationConfig,
        game_master: GameMaster,
        memory_tms: EpisodicMemoryTMS,
        llm_controller: Optional[TwoStageLLMControllerProtocol] = None,
        run_id: str = "run_v2",
        economy_seed: Optional[int] = None,
    ) -> None:
        self.config = config
        self.game_master = game_master
        self.memory_tms = memory_tms
        self.llm_controller = llm_controller
        self.run_id = run_id
        self.economy_seed = (
            economy_seed if economy_seed is not None else config.master_seed
        )
        self.deliberation_ctrl = DeliberationBudgetController(config.deliberation)
        self.rng = np.random.default_rng(self.economy_seed + 1009)
        self.turn_records: List[AgentTurnRecord] = []

    def run_tick(self, tick: int) -> TickMacroSummary:
        self.game_master.begin_tick(tick)
        arrival_order = list(self.game_master.households.keys())
        self.rng.shuffle(arrival_order)

        for agent_id in arrival_order:
            sheet = self.game_master.households[agent_id]
            if not NeedsEngine.is_alive(sheet):
                continue
            self._execute_single_agent_turn(tick, agent_id)

        summary = self.game_master.step_end_of_tick_physics(tick)
        self._record_post_tick_sanitized_episodes(tick, summary)
        return summary

    def _execute_single_agent_turn(self, tick: int, agent_id: str) -> AgentTurnRecord:
        t_start = time.perf_counter()
        sheet = self.game_master.households[agent_id]
        pre_cash = sheet.liquid_cash
        pre_cap = sheet.illiquid_capital
        pre_cons = sheet.consumable_inventory
        pre_health = sheet.needs.health
        pre_stress = sheet.needs.composite_stress

        alloc: DeliberationAllocation = self.deliberation_ctrl.compute_allocation(
            sheet.needs
        )
        mem_handle = self.memory_tms.get_scoped_handle(agent_id)
        snapshots = self.game_master.exchange.get_all_snapshots(tick)
        c_snap = snapshots[AssetType.CONSUMABLE]
        query_str = (
            f"Cycle {tick} RESOURCE_A={pre_cons:.1f} Pool_Credits={pre_cash:.1f} "
            f"Vitality={pre_health:.2f} Board_Last={c_snap.last_trade_price}"
        )
        retrieval: RetrievalResult = mem_handle.retrieve(
            agent_id=agent_id,
            query_text=query_str,
            current_tick=tick,
            stress_index=pre_stress,
            rng=self.rng,
        )
        peer_dir = mem_handle.get_top_peers_summary(agent_id=agent_id, max_peers=3)
        open_orders = []
        for book in self.game_master.exchange.books.values():
            open_orders.extend(book.get_open_orders_for_agent(agent_id))

        policy = self.config.policy_mode
        if policy == PolicyMode.ZERO_INTELLIGENCE:
            payload = self.game_master.propose_zero_intelligence_action(
                tick, agent_id, zi_rng=self.rng
            )
            gen_out = TwoStageGenerationOutput(
                stage1_cot_text="ZERO_INTELLIGENCE_POLICY",
                cot_tokens_generated=0,
                cot_truncated_by_bottleneck=False,
                probe_measurement=None,
                stage2_raw_json=payload.model_dump_json(),
                stage2_action=payload,
                json_valid=True,
                abstain=False,
                retries_used=0,
                action_tokens_generated=0,
                latency_ms=round((time.perf_counter() - t_start) * 1000.0, 3),
            )
        elif policy == PolicyMode.CLOSED_FORM_BENCHMARK:
            payload = ClosedFormBenchmarkPolicy.select_action(
                tick, sheet, self.game_master
            )
            gen_out = TwoStageGenerationOutput(
                stage1_cot_text="CLOSED_FORM_BENCHMARK_POLICY",
                cot_tokens_generated=0,
                cot_truncated_by_bottleneck=False,
                probe_measurement=None,
                stage2_raw_json=payload.model_dump_json(),
                stage2_action=payload,
                json_valid=True,
                abstain=False,
                retries_used=0,
                action_tokens_generated=0,
                latency_ms=round((time.perf_counter() - t_start) * 1000.0, 3),
            )
        else:
            if self.llm_controller is None:
                raise RuntimeError(
                    "PolicyMode.LLM_AGENT requires an attached TwoStageLLMControllerProtocol (Module C)."
                )
            sys_prompt, user_prompt = SanitizedPromptCompiler.compile_stage1_prompt(
                tick=tick,
                sheet=sheet,
                board_snapshots=snapshots,
                open_orders=open_orders,
                retrieval=retrieval,
                peer_directory=peer_dir,
                cfg=self.config,
            )
            gen_out = self.llm_controller.execute_two_stage_turn(
                stage1_system_prompt=sys_prompt,
                stage1_user_prompt=user_prompt,
                stage2_instruction=SanitizedPromptCompiler.STAGE2_JSON_INSTRUCTION,
                cot_max_tokens=alloc.cot_max_tokens,
                action_max_tokens=alloc.action_json_max_tokens,
                agent_id=agent_id,
                tick=tick,
                parse_action_fn=IsomorphicVocabulary.parse_sanitized_action_json,
            )

        exec_outcome: ActionExecutionOutcome = self.game_master.execute_agent_action(
            tick=tick,
            agent_id=agent_id,
            payload=gen_out.stage2_action,
        )

        for fill in exec_outcome.fills:
            tok = IsomorphicVocabulary.ASSET_TO_TOKEN[fill.asset]
            buyer_handle = self.memory_tms.get_scoped_handle(fill.buyer_id)
            seller_handle = self.memory_tms.get_scoped_handle(fill.seller_id)

            buyer_handle.write_episode(
                agent_id=fill.buyer_id,
                tick=tick,
                event_type="TRADE_FILL",
                summary_text=(
                    f"[C{tick:03d}] Board_Fill: Side=BID {tok} Units={fill.quantity:.1f} "
                    f"Rate={fill.execution_price:.2f} Peer={fill.seller_id}"
                ),
                salience_score=1.35,
            )
            seller_handle.write_episode(
                agent_id=fill.seller_id,
                tick=tick,
                event_type="TRADE_FILL",
                summary_text=(
                    f"[C{tick:03d}] Board_Fill: Side=ASK {tok} Units={fill.quantity:.1f} "
                    f"Rate={fill.execution_price:.2f} Peer={fill.buyer_id}"
                ),
                salience_score=1.35,
            )
            buyer_handle.update_peer_reliability(fill.buyer_id, fill.seller_id, 1.0, tick)
            seller_handle.update_peer_reliability(fill.seller_id, fill.buyer_id, 1.0, tick)

        if exec_outcome.accepted and exec_outcome.action_type in (
            AgentActionType.LIQUIDATE_CAPITAL,
            AgentActionType.INVEST_CAPITAL,
        ):
            proto_label = (
                "PROTOCOL_B_OUT"
                if exec_outcome.action_type == AgentActionType.LIQUIDATE_CAPITAL
                else "PROTOCOL_B_IN"
            )
            mem_handle.write_episode(
                agent_id=agent_id,
                tick=tick,
                event_type="CAPITAL_TRANSFER",
                summary_text=(
                    f"[C{tick:03d}] {proto_label}: Units={gen_out.stage2_action.capital_transfer_units:.1f} "
                    f"Fee={exec_outcome.kmv_fee_paid:.2f} Credits_Delta={exec_outcome.net_cash_delta:+.2f}"
                ),
                salience_score=1.60,
            )

        record = AgentTurnRecord(
            run_id=self.run_id,
            economy_seed=self.economy_seed,
            tick=tick,
            agent_id=agent_id,
            policy_mode=policy,
            factorial_cell_id=self.config.factorial_cell.cell_id,
            pre_turn_cash=pre_cash,
            pre_turn_capital=pre_cap,
            pre_turn_consumable=pre_cons,
            pre_turn_health=pre_health,
            pre_turn_stress=pre_stress,
            rag_top_k_used=retrieval.top_k_used,
            rag_noise_sigma_used=retrieval.noise_sigma_used,
            cot_token_budget_allocated=alloc.cot_max_tokens,
            cot_tokens_generated=gen_out.cot_tokens_generated,
            cot_truncated_by_bottleneck=gen_out.cot_truncated_by_bottleneck,
            stage1_cot_text=gen_out.stage1_cot_text,
            probe_measurement=gen_out.probe_measurement,
            stage2_raw_json=gen_out.stage2_raw_json,
            stage2_action=gen_out.stage2_action,
            json_valid=gen_out.json_valid,
            abstain=gen_out.abstain,
            retries_used=gen_out.retries_used,
            action_tokens_generated=gen_out.action_tokens_generated,
            latency_ms=gen_out.latency_ms,
            execution_status=exec_outcome.reason,
        )
        self.turn_records.append(record)
        return record

    def _record_post_tick_sanitized_episodes(
        self, tick: int, summary: TickMacroSummary
    ) -> None:
        for aid, sheet in self.game_master.households.items():
            handle = self.memory_tms.get_scoped_handle(aid)
            if sheet.needs.subsistence_deficit_streak > 0:
                handle.write_episode(
                    agent_id=aid,
                    tick=tick,
                    event_type="SUBSISTENCE_DEFICIT",
                    summary_text=(
                        f"[C{tick:03d}] Drain_Shortfall: RESOURCE_A_Used={sheet.needs.last_consumed_units:.2f} "
                        f"Vitality_Index={sheet.needs.health:.2f}"
                    ),
                    salience_score=2.20,
                )
            if aid in summary.obligation_defaults_this_tick:
                handle.write_episode(
                    agent_id=aid,
                    tick=tick,
                    event_type="SUBSISTENCE_DEFICIT",
                    summary_text=(
                        f"[C{tick:03d}] Cycle_Drain_Gap: Pool_Credits={sheet.liquid_cash:.2f} "
                        f"Floor_Bound={sheet.borrowing_limit_b_bar:.2f}"
                    ),
                    salience_score=1.95,
                )