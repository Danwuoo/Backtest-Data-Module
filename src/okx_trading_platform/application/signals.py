from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Iterable

from okx_trading_platform.domain import (
    InstrumentKind,
    OrderIntent,
    OrderSide,
    SignalEnvelope,
    TdMode,
    TradingProfile,
)


class SignalProvider(ABC):
    @abstractmethod
    def next_signals(self, market_state: dict) -> Iterable[SignalEnvelope]:
        raise NotImplementedError


class ManualSignalProvider(SignalProvider):
    def __init__(self) -> None:
        self._queue: deque[SignalEnvelope] = deque()

    def enqueue(self, signal: SignalEnvelope) -> None:
        self._queue.append(signal)

    def next_signals(self, market_state: dict) -> Iterable[SignalEnvelope]:
        del market_state
        while self._queue:
            yield self._queue.popleft()


class ReferenceBreakoutSignalProvider(SignalProvider):
    def __init__(
        self,
        *,
        bot_name: str,
        profile: TradingProfile,
        inst_id: str,
        instrument_kind: InstrumentKind,
        trigger_spread: float = 0.002,
        size: float = 1.0,
    ) -> None:
        self.bot_name = bot_name
        self.profile = profile
        self.inst_id = inst_id
        self.instrument_kind = instrument_kind
        self.trigger_spread = trigger_spread
        self.size = size

    def next_signals(self, market_state: dict) -> Iterable[SignalEnvelope]:
        best_bid = float(market_state["best_bid"])
        best_ask = float(market_state["best_ask"])
        last_price = float(market_state["last_price"])
        mid = (best_bid + best_ask) / 2
        if last_price >= mid * (1 + self.trigger_spread):
            yield self._signal(OrderSide.BUY, last_price, mid)
        elif last_price <= mid * (1 - self.trigger_spread):
            yield self._signal(OrderSide.SELL, last_price, mid)

    def _signal(
        self, side: OrderSide, last_price: float, mid_price: float
    ) -> SignalEnvelope:
        return SignalEnvelope(
            bot_name=self.bot_name,
            inst_id=self.inst_id,
            profile=self.profile,
            instrument_kind=self.instrument_kind,
            side=side,
            size=self.size,
            signal_type="reference_breakout",
            metadata={"last_price": last_price, "mid_price": mid_price},
        )


def signal_to_order_intent(signal: SignalEnvelope) -> OrderIntent:
    return OrderIntent(
        profile=signal.profile,
        instrument_kind=signal.instrument_kind,
        inst_id=signal.inst_id,
        side=signal.side,
        size=signal.size,
        price=signal.price,
        bot_name=signal.bot_name,
        source=signal.signal_type,
        metadata=signal.metadata,
        td_mode=TdMode.CASH
        if signal.instrument_kind == InstrumentKind.SPOT
        else TdMode.ISOLATED,
    )


def signals_to_order_intents(signals: Iterable[SignalEnvelope]) -> list[OrderIntent]:
    return [signal_to_order_intent(signal) for signal in signals]
