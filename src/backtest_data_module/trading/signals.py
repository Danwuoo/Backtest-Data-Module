from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Iterable

from backtest_data_module.trading.domain import (
    InstrumentKind,
    OrderSide,
    SignalEnvelope,
    TradingProfile,
)


class SignalProvider(ABC):
    @abstractmethod
    def next_signals(self, market_state: dict) -> Iterable[SignalEnvelope]:
        raise NotImplementedError


class ManualSignalProvider(SignalProvider):
    """供 CLI 與整合測試直接推送訊號。"""

    def __init__(self) -> None:
        self._queue: deque[SignalEnvelope] = deque()

    def enqueue(self, signal: SignalEnvelope) -> None:
        self._queue.append(signal)

    def next_signals(self, market_state: dict) -> Iterable[SignalEnvelope]:
        while self._queue:
            yield self._queue.popleft()


class ReferenceBreakoutSignalProvider(SignalProvider):
    """用簡單 breakout 當作 v1 參考策略，後續可替換成 AI/量化模型。"""

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
            yield SignalEnvelope(
                bot_name=self.bot_name,
                inst_id=self.inst_id,
                profile=self.profile,
                instrument_kind=self.instrument_kind,
                side=OrderSide.BUY,
                size=self.size,
                signal_type="reference_breakout",
                metadata={"last_price": last_price, "mid_price": mid},
            )
        elif last_price <= mid * (1 - self.trigger_spread):
            yield SignalEnvelope(
                bot_name=self.bot_name,
                inst_id=self.inst_id,
                profile=self.profile,
                instrument_kind=self.instrument_kind,
                side=OrderSide.SELL,
                size=self.size,
                signal_type="reference_breakout",
                metadata={"last_price": last_price, "mid_price": mid},
            )
