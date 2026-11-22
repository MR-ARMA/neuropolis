from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import uuid

from src.core.contracts import (
    AssetType,
    CDAMarketConfig,
    CDAOrder,
    OrderBookSnapshot,
    OrderSide,
    OrderStatus,
    OrderType,
    PriceLevelDepth,
    TradeFill,
)


@dataclass
class OrderEscrowRelease:
    """Tracks exact collateral to release back to an agent upon cancellation, expiry, or price improvement."""
    order_id: str
    agent_id: str
    asset: AssetType
    side: OrderSide
    cash_to_release: float = 0.0
    asset_units_to_release: float = 0.0


@dataclass
class OrderSubmissionResult:
    """Atomic microstructure outcome returned by the CDA Matching Engine to GameMaster."""
    order: CDAOrder
    fills: List[TradeFill]
    price_improvement_refunds: List[Tuple[str, float]]  # (buyer_id, cash_refund)
    unfilled_ioc_release: Optional[OrderEscrowRelease] = None


class AssetOrderBook:
    """
    Pure Price-Time Priority Continuous Double Auction (CDA) Order Book for a single asset.
      - Bids sorted by (-limit_price, sequence_num): highest price first, earliest arrival first.
      - Asks sorted by (+limit_price, sequence_num): lowest price first, earliest arrival first.
      - Transactions execute ONLY when Bid >= Ask (or MARKET order sweeps resting opposite book).
      - Execution price equals the resting Maker's limit price (standard CDA microstructure).
      - Self-trade prevention skips crossing orders belonging to the same agent_id.
      - Zero AMM pricing, zero synthetic price nudges, zero hidden liquidity providers.
    """

    def __init__(self, asset: AssetType, config: CDAMarketConfig) -> None:
        self.asset = asset
        self.config = config
        self.bids: List[CDAOrder] = []
        self.asks: List[CDAOrder] = []
        self.orders_by_id: Dict[str, CDAOrder] = {}
        # Tracks exact remaining locked escrow per resting order for <1e-9 SFC verification
        self.locked_cash_by_order: Dict[str, float] = {}
        self.locked_asset_by_order: Dict[str, float] = {}

        self.last_trade_price: Optional[float] = None
        self.tick_volume: float = 0.0
        self.current_tick_tracker: int = -1

    def quantize_price(self, raw_price: float) -> float:
        """Snaps price to the discrete tick_size grid within [min_order_price, max_order_price]."""
        clipped = max(
            self.config.min_order_price,
            min(self.config.max_order_price, float(raw_price)),
        )
        steps = round(clipped / self.config.tick_size)
        return round(max(self.config.min_order_price, steps * self.config.tick_size), 4)

    def _sort_books(self) -> None:
        self.bids.sort(key=lambda o: (-(o.limit_price or 0.0), o.sequence_num))
        self.asks.sort(key=lambda o: ((o.limit_price or 1e9), o.sequence_num))

    def best_bid(self) -> Optional[float]:
        self._sort_books()
        return self.bids[0].limit_price if self.bids else None

    def best_ask(self) -> Optional[float]:
        self._sort_books()
        return self.asks[0].limit_price if self.asks else None

    def estimate_market_sweep_cost(self, side: OrderSide, quantity: float, exclude_agent_id: str) -> Tuple[float, float]:
        """
        Estimates the maximum fillable quantity and exact cash cost to sweep resting liquidity
        for an incoming MARKET order, excluding the submitting agent's own resting orders.
        Returns `(fillable_quantity, estimated_gross_cash)`.
        """
        self._sort_books()
        resting_book = self.asks if side == OrderSide.BID else self.bids
        rem = float(quantity)
        fillable = 0.0
        gross_cash = 0.0
        for resting in resting_book:
            if rem <= 1e-9:
                break
            if resting.agent_id == exclude_agent_id:
                continue
            avail = resting.remaining_quantity
            if avail <= 1e-9 or resting.limit_price is None:
                continue
            take = min(rem, avail)
            fillable += take
            gross_cash += take * resting.limit_price
            rem -= take
        return (round(fillable, 6), round(gross_cash, 8))

    def submit_order(
        self,
        order: CDAOrder,
        locked_cash: float,
        locked_asset: float,
        sales_tax_rate: float,
    ) -> OrderSubmissionResult:
        """
        Matches an incoming LIMIT or MARKET order against the opposite resting book.
        Resting limit orders remain queued until filled, cancelled, or TTL-expired.
        MARKET orders are Immediate-Or-Cancel (IOC): any unfilled remainder releases escrow immediately.
        """
        if order.tick_submitted != self.current_tick_tracker:
            self.current_tick_tracker = order.tick_submitted
            self.tick_volume = 0.0

        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            order.limit_price = self.quantize_price(order.limit_price)

        self.orders_by_id[order.order_id] = order
        self.locked_cash_by_order[order.order_id] = round(float(locked_cash), 8)
        self.locked_asset_by_order[order.order_id] = round(float(locked_asset), 8)

        self._sort_books()
        fills: List[TradeFill] = []
        price_improvement_refunds: List[Tuple[str, float]] = []

        if order.side == OrderSide.BID:
            opposite_book = self.asks
            idx = 0
            while order.remaining_quantity > 1e-9 and idx < len(opposite_book):
                resting_ask = opposite_book[idx]
                if resting_ask.agent_id == order.agent_id:
                    idx += 1
                    continue
                ask_price = resting_ask.limit_price
                if ask_price is None:
                    idx += 1
                    continue

                # Strict CDA Crossing Condition: Bid >= Ask (for LIMIT orders)
                if (
                    order.order_type == OrderType.LIMIT
                    and (order.limit_price is None or order.limit_price + 1e-9 < ask_price)
                ):
                    break

                # For MARKET BID, ensure remaining locked cash covers at least partial fill
                avail_bid_cash = self.locked_cash_by_order[order.order_id]
                max_affordable_qty = (
                    avail_bid_cash / ask_price
                    if order.order_type == OrderType.MARKET
                    else order.remaining_quantity
                )
                fill_qty = round(
                    min(
                        order.remaining_quantity,
                        resting_ask.remaining_quantity,
                        max_affordable_qty,
                    ),
                    6,
                )
                if fill_qty <= 1e-9:
                    break

                exec_price = ask_price
                gross_val = round(fill_qty * exec_price, 8)
                tax_paid = round(gross_val * sales_tax_rate, 8)

                # Update fill quantities and statuses
                order.filled_quantity = round(order.filled_quantity + fill_qty, 6)
                resting_ask.filled_quantity = round(
                    resting_ask.filled_quantity + fill_qty, 6
                )
                order.status = (
                    OrderStatus.FILLED
                    if order.remaining_quantity <= 1e-9
                    else OrderStatus.PARTIALLY_FILLED
                )
                resting_ask.status = (
                    OrderStatus.FILLED
                    if resting_ask.remaining_quantity <= 1e-9
                    else OrderStatus.PARTIALLY_FILLED
                )

                # Deduct settled collateral from escrow trackers & compute price improvement refund
                if order.order_type == OrderType.LIMIT and order.limit_price is not None:
                    escrow_consumed = round(fill_qty * order.limit_price, 8)
                    refund = round(max(0.0, escrow_consumed - gross_val), 8)
                    self.locked_cash_by_order[order.order_id] = round(
                        max(0.0, self.locked_cash_by_order[order.order_id] - escrow_consumed),
                        8,
                    )
                    if refund > 1e-9:
                        price_improvement_refunds.append((order.agent_id, refund))
                else:
                    self.locked_cash_by_order[order.order_id] = round(
                        max(0.0, self.locked_cash_by_order[order.order_id] - gross_val),
                        8,
                    )

                self.locked_asset_by_order[resting_ask.order_id] = round(
                    max(0.0, self.locked_asset_by_order[resting_ask.order_id] - fill_qty),
                    8,
                )

                fill = TradeFill(
                    fill_id=f"fill_{order.tick_submitted:04d}_{uuid.uuid4().hex[:8]}",
                    tick=order.tick_submitted,
                    asset=self.asset,
                    buy_order_id=order.order_id,
                    sell_order_id=resting_ask.order_id,
                    buyer_id=order.agent_id,
                    seller_id=resting_ask.agent_id,
                    execution_price=exec_price,
                    quantity=fill_qty,
                    gross_value=gross_val,
                    sales_tax_paid=tax_paid,
                    maker_side=OrderSide.ASK,
                )
                fills.append(fill)
                self.last_trade_price = exec_price
                self.tick_volume = round(self.tick_volume + fill_qty, 6)

                if resting_ask.remaining_quantity <= 1e-9:
                    opposite_book.pop(idx)
                else:
                    idx += 1

        else:
            # Incoming ASK order matching against resting BIDS
            opposite_book = self.bids
            idx = 0
            while order.remaining_quantity > 1e-9 and idx < len(opposite_book):
                resting_bid = opposite_book[idx]
                if resting_bid.agent_id == order.agent_id:
                    idx += 1
                    continue
                bid_price = resting_bid.limit_price
                if bid_price is None:
                    idx += 1
                    continue

                # Strict CDA Crossing Condition: Bid >= Ask (for LIMIT orders)
                if (
                    order.order_type == OrderType.LIMIT
                    and (order.limit_price is None or bid_price + 1e-9 < order.limit_price)
                ):
                    break

                fill_qty = round(
                    min(order.remaining_quantity, resting_bid.remaining_quantity), 6
                )
                if fill_qty <= 1e-9:
                    break

                exec_price = bid_price
                gross_val = round(fill_qty * exec_price, 8)
                tax_paid = round(gross_val * sales_tax_rate, 8)

                order.filled_quantity = round(order.filled_quantity + fill_qty, 6)
                resting_bid.filled_quantity = round(
                    resting_bid.filled_quantity + fill_qty, 6
                )
                order.status = (
                    OrderStatus.FILLED
                    if order.remaining_quantity <= 1e-9
                    else OrderStatus.PARTIALLY_FILLED
                )
                resting_bid.status = (
                    OrderStatus.FILLED
                    if resting_bid.remaining_quantity <= 1e-9
                    else OrderStatus.PARTIALLY_FILLED
                )

                self.locked_cash_by_order[resting_bid.order_id] = round(
                    max(0.0, self.locked_cash_by_order[resting_bid.order_id] - gross_val),
                    8,
                )
                self.locked_asset_by_order[order.order_id] = round(
                    max(0.0, self.locked_asset_by_order[order.order_id] - fill_qty),
                    8,
                )

                fill = TradeFill(
                    fill_id=f"fill_{order.tick_submitted:04d}_{uuid.uuid4().hex[:8]}",
                    tick=order.tick_submitted,
                    asset=self.asset,
                    buy_order_id=resting_bid.order_id,
                    sell_order_id=order.order_id,
                    buyer_id=resting_bid.agent_id,
                    seller_id=order.agent_id,
                    execution_price=exec_price,
                    quantity=fill_qty,
                    gross_value=gross_val,
                    sales_tax_paid=tax_paid,
                    maker_side=OrderSide.BID,
                )
                fills.append(fill)
                self.last_trade_price = exec_price
                self.tick_volume = round(self.tick_volume + fill_qty, 6)

                if resting_bid.remaining_quantity <= 1e-9:
                    opposite_book.pop(idx)
                else:
                    idx += 1

        # Handle resting vs. IOC release for unfilled remainder
        unfilled_release: Optional[OrderEscrowRelease] = None
        if order.remaining_quantity > 1e-9:
            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BID:
                    self.bids.append(order)
                else:
                    self.asks.append(order)
                self._sort_books()
            else:
                # MARKET order is IOC: release remaining locked cash/asset immediately
                rem_cash = self.locked_cash_by_order.pop(order.order_id, 0.0)
                rem_asset = self.locked_asset_by_order.pop(order.order_id, 0.0)
                if order.filled_quantity <= 1e-9:
                    order.status = OrderStatus.CANCELLED
                unfilled_release = OrderEscrowRelease(
                    order_id=order.order_id,
                    agent_id=order.agent_id,
                    asset=self.asset,
                    side=order.side,
                    cash_to_release=rem_cash,
                    asset_units_to_release=rem_asset,
                )
        else:
            # Fully filled: release any tiny rounding dust in locked cash tracker
            rem_cash = self.locked_cash_by_order.pop(order.order_id, 0.0)
            rem_asset = self.locked_asset_by_order.pop(order.order_id, 0.0)
            if rem_cash > 1e-9 or rem_asset > 1e-9:
                unfilled_release = OrderEscrowRelease(
                    order_id=order.order_id,
                    agent_id=order.agent_id,
                    asset=self.asset,
                    side=order.side,
                    cash_to_release=rem_cash,
                    asset_units_to_release=rem_asset,
                )

        return OrderSubmissionResult(
            order=order,
            fills=fills,
            price_improvement_refunds=price_improvement_refunds,
            unfilled_ioc_release=unfilled_release,
        )

    def cancel_order(self, order_id: str, requesting_agent_id: Optional[str] = None) -> Optional[OrderEscrowRelease]:
        """Cancels an open limit order and returns its exact remaining locked collateral."""
        order = self.orders_by_id.get(order_id)
        if order is None:
            return None
        if requesting_agent_id is not None and order.agent_id != requesting_agent_id:
            return None
        if order.status not in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED):
            return None

        target_book = self.bids if order.side == OrderSide.BID else self.asks
        self.bids = [o for o in self.bids if o.order_id != order_id] if order.side == OrderSide.BID else self.bids
        self.asks = [o for o in self.asks if o.order_id != order_id] if order.side == OrderSide.ASK else self.asks

        order.status = OrderStatus.CANCELLED
        rem_cash = self.locked_cash_by_order.pop(order_id, 0.0)
        rem_asset = self.locked_asset_by_order.pop(order_id, 0.0)
        return OrderEscrowRelease(
            order_id=order_id,
            agent_id=order.agent_id,
            asset=self.asset,
            side=order.side,
            cash_to_release=rem_cash,
            asset_units_to_release=rem_asset,
        )

    def cancel_all_agent_orders(self, agent_id: str) -> List[OrderEscrowRelease]:
        """Cancels all open resting orders for a specific agent (e.g., on economic death)."""
        open_ids = [
            o.order_id
            for o in (self.bids + self.asks)
            if o.agent_id == agent_id
        ]
        releases: List[OrderEscrowRelease] = []
        for oid in open_ids:
            rel = self.cancel_order(oid, requesting_agent_id=agent_id)
            if rel is not None:
                releases.append(rel)
        return releases

    def expire_stale_orders(self, current_tick: int) -> List[OrderEscrowRelease]:
        """Cancels resting orders older than `order_ttl_ticks` and returns their escrow."""
        ttl = self.config.order_ttl_ticks
        stale_ids = [
            o.order_id
            for o in (self.bids + self.asks)
            if (current_tick - o.tick_submitted) >= ttl
        ]
        releases: List[OrderEscrowRelease] = []
        for oid in stale_ids:
            rel = self.cancel_order(oid)
            if rel is not None:
                releases.append(rel)
        return releases

    def total_locked_cash(self) -> float:
        return round(sum(self.locked_cash_by_order.values()), 8)

    def total_locked_asset(self) -> float:
        return round(sum(self.locked_asset_by_order.values()), 8)

    def get_open_orders_for_agent(self, agent_id: str) -> List[CDAOrder]:
        return [o for o in (self.bids + self.asks) if o.agent_id == agent_id]

    def get_snapshot(self, tick: int) -> OrderBookSnapshot:
        """Builds an aggregated Level-2 Top-5 depth snapshot visible to agents."""
        self._sort_books()
        bb = self.bids[0].limit_price if self.bids else None
        ba = self.asks[0].limit_price if self.asks else None
        mid = round(0.5 * (bb + ba), 4) if (bb is not None and ba is not None) else None
        spread = round(ba - bb, 4) if (bb is not None and ba is not None) else None

        def _aggregate_top5(orders: List[CDAOrder]) -> List[PriceLevelDepth]:
            levels: Dict[float, Tuple[float, int]] = {}
            for o in orders:
                if o.limit_price is None or o.remaining_quantity <= 1e-9:
                    continue
                qty, cnt = levels.get(o.limit_price, (0.0, 0))
                levels[o.limit_price] = (round(qty + o.remaining_quantity, 6), cnt + 1)
            result = [
                PriceLevelDepth(price=p, total_quantity=q, order_count=c)
                for p, (q, c) in list(levels.items())[:5]
            ]
            return result

        tick_vol = self.tick_volume if tick == self.current_tick_tracker else 0.0
        return OrderBookSnapshot(
            tick=tick,
            asset=self.asset,
            best_bid=bb,
            best_ask=ba,
            mid_price=mid,
            spread=spread,
            last_trade_price=self.last_trade_price,
            tick_volume=tick_vol,
            bids_top5=_aggregate_top5(self.bids),
            asks_top5=_aggregate_top5(self.asks),
        )


class ContinuousDoubleAuctionExchange:
    """
    Multi-asset Continuous Double Auction Exchange hosting independent Limit Order Books
    for `AssetType.CONSUMABLE` and (optionally) `AssetType.CAPITAL`.
    """

    def __init__(self, config: CDAMarketConfig) -> None:
        self.config = config
        self.sequence_counter: int = 0
        self.books: Dict[AssetType, AssetOrderBook] = {
            AssetType.CONSUMABLE: AssetOrderBook(AssetType.CONSUMABLE, config),
        }
        if config.enable_capital_secondary_book:
            self.books[AssetType.CAPITAL] = AssetOrderBook(AssetType.CAPITAL, config)

    def next_sequence_num(self) -> int:
        self.sequence_counter += 1
        return self.sequence_counter

    def get_book(self, asset: AssetType) -> AssetOrderBook:
        if asset not in self.books:
            raise ValueError(f"CDA order book is not enabled for asset {asset.value}.")
        return self.books[asset]

    def expire_all_stale_orders(self, current_tick: int) -> List[OrderEscrowRelease]:
        all_releases: List[OrderEscrowRelease] = []
        for book in self.books.values():
            all_releases.extend(book.expire_stale_orders(current_tick))
        return all_releases

    def cancel_all_agent_orders(self, agent_id: str) -> List[OrderEscrowRelease]:
        all_releases: List[OrderEscrowRelease] = []
        for book in self.books.values():
            all_releases.extend(book.cancel_all_agent_orders(agent_id))
        return all_releases

    def total_exchange_locked_cash(self) -> float:
        return round(sum(b.total_locked_cash() for b in self.books.values()), 8)

    def get_all_snapshots(self, tick: int) -> Dict[AssetType, OrderBookSnapshot]:
        return {asset: book.get_snapshot(tick) for asset, book in self.books.items()}