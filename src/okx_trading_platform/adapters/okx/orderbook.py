from __future__ import annotations

from dataclasses import dataclass, field

from okx_trading_platform.domain import (
    OrderBookLevel,
    OrderBookSnapshot,
    utc_now,
)


@dataclass
class OkxOrderBook:
    inst_id: str
    profile_id: str
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    instrument_id: str | None = None
    sequence_id: int | None = None
    prev_sequence_id: int | None = None
    checksum: int | None = None
    gap_detected: bool = False

    def apply_snapshot(
        self,
        bids: list[list[str | float]],
        asks: list[list[str | float]],
        *,
        instrument_id: str | None = None,
        sequence_id: int | None = None,
        checksum: int | None = None,
    ) -> OrderBookSnapshot:
        self.bids.clear()
        self.asks.clear()
        self._merge(self.bids, bids)
        self._merge(self.asks, asks)
        self.instrument_id = instrument_id or self.instrument_id
        self.sequence_id = sequence_id
        self.prev_sequence_id = None
        self.checksum = checksum
        self.gap_detected = False
        return self.snapshot()

    def apply_update(
        self,
        bids: list[list[str | float]],
        asks: list[list[str | float]],
        *,
        instrument_id: str | None = None,
        sequence_id: int | None = None,
        prev_sequence_id: int | None = None,
        checksum: int | None = None,
    ) -> OrderBookSnapshot:
        self.gap_detected = self._detect_gap(
            sequence_id=sequence_id,
            prev_sequence_id=prev_sequence_id,
        )
        self._merge(self.bids, bids)
        self._merge(self.asks, asks)
        self.instrument_id = instrument_id or self.instrument_id
        self.prev_sequence_id = prev_sequence_id
        self.sequence_id = sequence_id
        self.checksum = checksum
        return self.snapshot()

    def rebuild(self, snapshot: OrderBookSnapshot) -> None:
        self.apply_snapshot(
            [[level.price, level.size] for level in snapshot.bids],
            [[level.price, level.size] for level in snapshot.asks],
            instrument_id=snapshot.instrument_id,
            sequence_id=snapshot.sequence_id,
            checksum=snapshot.checksum,
        )
        self.prev_sequence_id = snapshot.prev_sequence_id
        self.gap_detected = snapshot.gap_detected

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
            instrument_id=self.instrument_id,
            inst_id=self.inst_id,
            profile_id=self.profile_id,
            bids=bids,
            asks=asks,
            sequence_id=self.sequence_id,
            prev_sequence_id=self.prev_sequence_id,
            gap_detected=self.gap_detected,
            checksum=self.checksum,
            updated_at=utc_now(),
        )

    def _detect_gap(
        self,
        *,
        sequence_id: int | None,
        prev_sequence_id: int | None,
    ) -> bool:
        if sequence_id is None or prev_sequence_id is None:
            return False
        if self.sequence_id is None:
            return prev_sequence_id != sequence_id - 1
        return prev_sequence_id != self.sequence_id

    @staticmethod
    def _merge(levels: dict[float, float], updates: list[list[str | float]]) -> None:
        for raw_price, raw_size, *_rest in updates:
            price = float(raw_price)
            size = float(raw_size)
            if size == 0:
                levels.pop(price, None)
                continue
            levels[price] = size
