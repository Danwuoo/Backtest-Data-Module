from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Iterable

from okx_trading_platform.domain import (
    InstrumentConfig,
    InstrumentKind,
    OrderPlan,
    OrderSide,
    OrderType,
    PositionIntent,
    SignalMode,
    SleeveConfig,
    TargetSignal,
    TdMode,
)


class InferenceProvider(ABC):
    @abstractmethod
    def infer_targets(self, market_state: dict) -> Iterable[TargetSignal]:
        raise NotImplementedError


class ManualInferenceProvider(InferenceProvider):
    def __init__(self) -> None:
        self._queue: deque[TargetSignal] = deque()

    def enqueue(self, signal: TargetSignal) -> None:
        self._queue.append(signal)

    def infer_targets(self, market_state: dict) -> Iterable[TargetSignal]:
        del market_state
        while self._queue:
            yield self._queue.popleft()


class RuleBaselineInferenceProvider(InferenceProvider):
    def __init__(
        self,
        *,
        profile_id: str,
        strategy_id: str,
        model_version_id: str,
        instrument_id: str,
        inst_id: str,
        kind: InstrumentKind,
        threshold_bps: float = 20.0,
        target_size: float = 1.0,
        signal_mode: SignalMode = SignalMode.PRIMARY,
    ) -> None:
        self.profile_id = profile_id
        self.strategy_id = strategy_id
        self.model_version_id = model_version_id
        self.instrument_id = instrument_id
        self.inst_id = inst_id
        self.kind = kind
        self.threshold_bps = threshold_bps
        self.target_size = target_size
        self.signal_mode = signal_mode

    def infer_targets(self, market_state: dict) -> Iterable[TargetSignal]:
        best_bid = float(market_state["best_bid"])
        best_ask = float(market_state["best_ask"])
        last_price = float(market_state["last_price"])
        mid = (best_bid + best_ask) / 2
        drift_bps = ((last_price - mid) / mid) * 10_000
        if drift_bps >= self.threshold_bps:
            yield self._signal(OrderSide.BUY, last_price, mid, drift_bps)
        elif drift_bps <= -self.threshold_bps:
            yield self._signal(OrderSide.SELL, last_price, mid, drift_bps)

    def _signal(
        self,
        side: OrderSide,
        last_price: float,
        mid_price: float,
        drift_bps: float,
    ) -> TargetSignal:
        return TargetSignal(
            profile_id=self.profile_id,
            strategy_id=self.strategy_id,
            model_version_id=self.model_version_id,
            instrument_id=self.instrument_id,
            inst_id=self.inst_id,
            kind=self.kind,
            side=side,
            target_size=self.target_size,
            signal_mode=self.signal_mode,
            features={
                "best_bid": last_price if side == OrderSide.SELL else mid_price,
                "best_ask": last_price if side == OrderSide.BUY else mid_price,
                "last_price": last_price,
                "mid_price": mid_price,
                "drift_bps": drift_bps,
            },
            metadata={"provider": "rule_baseline"},
        )


def target_to_position_intent(
    target: TargetSignal, sleeve: SleeveConfig
) -> PositionIntent:
    return PositionIntent(
        profile_id=target.profile_id,
        strategy_id=target.strategy_id,
        model_version_id=target.model_version_id,
        sleeve_id=sleeve.sleeve_id,
        instrument_id=target.instrument_id,
        inst_id=target.inst_id,
        kind=target.kind,
        side=target.side,
        target_size=target.target_size * sleeve.capital_allocation,
        max_leverage=sleeve.max_leverage,
        risk_budget=sleeve.risk_budget,
        metadata={"target_signal_id": target.target_signal_id, **target.metadata},
    )


def position_intent_to_order_plan(
    intent: PositionIntent,
    instrument: InstrumentConfig,
) -> OrderPlan:
    return OrderPlan(
        profile_id=intent.profile_id,
        strategy_id=intent.strategy_id,
        model_version_id=intent.model_version_id,
        sleeve_id=intent.sleeve_id,
        instrument_id=instrument.instrument_id,
        inst_id=instrument.inst_id,
        kind=instrument.kind,
        side=intent.side,
        size=intent.target_size,
        order_type=OrderType.MARKET,
        td_mode=(
            TdMode.CASH if instrument.kind == InstrumentKind.SPOT else TdMode.ISOLATED
        ),
        min_notional=instrument.min_notional,
        source="rule_baseline",
        metadata={"risk_budget": intent.risk_budget, **intent.metadata},
    )
