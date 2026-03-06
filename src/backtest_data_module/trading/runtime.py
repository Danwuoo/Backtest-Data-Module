from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable
from uuid import uuid4

from backtest_data_module.trading.domain import (
    OrderBookSnapshot,
    OrderIntent,
    ServiceHeartbeat,
    ServiceStatus,
    SignalEnvelope,
    TradingProfile,
    enum_value,
    utc_now,
)
from backtest_data_module.trading.okx import OkxExecutionService
from backtest_data_module.trading.orderbook import OkxOrderBook
from backtest_data_module.trading.risk import RiskService
from backtest_data_module.trading.signals import SignalProvider


@dataclass
class ServiceRuntime:
    service_name: str
    profile: TradingProfile
    instance_id: str = field(
        default_factory=lambda: os.getenv("HOSTNAME", str(uuid4()))
    )
    status: ServiceStatus = ServiceStatus.STARTING
    metadata: dict = field(default_factory=dict)

    def heartbeat(self) -> ServiceHeartbeat:
        return ServiceHeartbeat(
            service_name=self.service_name,
            instance_id=self.instance_id,
            profile=enum_value(self.profile),
            status=enum_value(self.status),
            metadata=self.metadata,
            last_seen_at=utc_now(),
        )

    def set_running(self) -> None:
        self.status = ServiceStatus.RUNNING


@dataclass
class MarketDataRuntime(ServiceRuntime):
    books: dict[str, OkxOrderBook] = field(default_factory=dict)

    def upsert_snapshot(
        self,
        inst_id: str,
        bids: list[list[str | float]],
        asks: list[list[str | float]],
    ) -> OrderBookSnapshot:
        book = self.books.setdefault(inst_id, OkxOrderBook(inst_id, self.profile))
        book.apply_snapshot(bids, asks)
        return book.snapshot()

    def apply_delta(
        self,
        inst_id: str,
        bids: list[list[str | float]],
        asks: list[list[str | float]],
    ) -> OrderBookSnapshot:
        book = self.books.setdefault(inst_id, OkxOrderBook(inst_id, self.profile))
        book.apply_update(bids, asks)
        return book.snapshot()


@dataclass
class ExecutionRuntime(ServiceRuntime):
    execution_service: OkxExecutionService | None = None

    def submit(self, intent: OrderIntent) -> dict:
        if self.execution_service is None:
            raise RuntimeError("execution service not configured")
        state = self.execution_service.submit_order(intent)
        return state.model_dump(mode="json")


@dataclass
class RiskRuntime(ServiceRuntime):
    risk_service: RiskService | None = None

    def evaluate(self, *args, **kwargs):
        if self.risk_service is None:
            raise RuntimeError("risk service not configured")
        return self.risk_service.evaluate_order(*args, **kwargs)


@dataclass
class StrategyRunnerRuntime(ServiceRuntime):
    signal_provider: SignalProvider | None = None

    def generate_signals(self, market_state: dict) -> list[SignalEnvelope]:
        if self.signal_provider is None:
            return []
        return list(self.signal_provider.next_signals(market_state))

    @staticmethod
    def to_order_intents(signals: Iterable[SignalEnvelope]) -> list[OrderIntent]:
        intents: list[OrderIntent] = []
        for signal in signals:
            intents.append(
                OrderIntent(
                    profile=signal.profile,
                    instrument_kind=signal.instrument_kind,
                    inst_id=signal.inst_id,
                    side=signal.side,
                    size=signal.size,
                    price=signal.price,
                    bot_name=signal.bot_name,
                    source=signal.signal_type,
                    metadata=signal.metadata,
                    td_mode="cash" if signal.instrument_kind == "spot" else "isolated",
                )
            )
        return intents
