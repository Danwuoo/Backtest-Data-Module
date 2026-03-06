from __future__ import annotations

from dataclasses import dataclass, field

from okx_trading_platform.domain import (
    OrderBookLevel,
    OrderBookSnapshot,
    TradingProfile,
    utc_now,
)


@dataclass
class OkxOrderBook:
    inst_id: str
    profile: TradingProfile
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    sequence_id: int | None = None
    checksum: int | None = None

    def apply_snapshot(
        self,
        bids: list[list[str | float]],
        asks: list[list[str | float]],
        *,
        sequence_id: int | None = None,
        checksum: int | None = None,
    ) -> None:
        self.bids.clear()
        self.asks.clear()
        self._merge(self.bids, bids)
        self._merge(self.asks, asks)
        self.sequence_id = sequence_id
        self.checksum = checksum

    def apply_update(
        self,
        bids: list[list[str | float]],
        asks: list[list[str | float]],
        *,
        sequence_id: int | None = None,
        checksum: int | None = None,
    ) -> None:
        self._merge(self.bids, bids)
        self._merge(self.asks, asks)
        self.sequence_id = sequence_id
        self.checksum = checksum

    def rebuild(self, snapshot: OrderBookSnapshot) -> None:
        self.apply_snapshot(
            [[level.price, level.size] for level in snapshot.bids],
            [[level.price, level.size] for level in snapshot.asks],
            sequence_id=snapshot.sequence_id,
            checksum=snapshot.checksum,
        )

    def best_bid(self) -> OrderBookLevel | None:
        if not self.bids:
            return None
        price = max(self.bids)
        return OrderBookLevel(price=price, size=self.bids[price])

    def best_ask(self) -> OrderBookLevel | None:
        if not self.asks:
            return None
        price = min(self.asks)
        return OrderBookLevel(price=price, size=self.asks[price])

    def snapshot(self, depth: int = 25) -> OrderBookSnapshot:
        bids = [
            OrderBookLevel(price=price, size=size)
            for price, size in sorted(self.bids.items(), reverse=True)[:depth]
        ]
        asks = [
            OrderBookLevel(price=price, size=size)
            for price, size in sorted(self.asks.items())[:depth]
        ]
        return OrderBookSnapshot(
            inst_id=self.inst_id,
            profile=self.profile,
            bids=bids,
            asks=asks,
            sequence_id=self.sequence_id,
            checksum=self.checksum,
            updated_at=utc_now(),
        )

    @staticmethod
    def _merge(levels: dict[float, float], updates: list[list[str | float]]) -> None:
        for raw_price, raw_size, *_rest in updates:
            price = float(raw_price)
            size = float(raw_size)
            if size == 0:
                levels.pop(price, None)
                continue
            levels[price] = size
