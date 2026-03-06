from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable
from uuid import uuid4

from okx_trading_platform.adapters.okx import OkxExchangeGateway, OkxOrderBook
from okx_trading_platform.application.signals import SignalProvider
from okx_trading_platform.domain import (
    OrderBookSnapshot,
    OrderIntent,
    ServiceHeartbeat,
    ServiceStatus,
    SignalEnvelope,
    TradingProfile,
    utc_now,
)
from okx_trading_platform.domain.risk import RiskManager


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
            profile=self.profile,
            status=self.status,
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
    gateway: OkxExchangeGateway | None = None

    def submit(self, intent: OrderIntent) -> dict:
        if self.gateway is None:
            raise RuntimeError("execution gateway is not configured")
        return self.gateway.submit_order(intent).model_dump(mode="json")


@dataclass
class RiskRuntime(ServiceRuntime):
    risk_manager: RiskManager | None = None

    def evaluate(self, *args, **kwargs):
        if self.risk_manager is None:
            raise RuntimeError("risk manager is not configured")
        return self.risk_manager.evaluate_order(*args, **kwargs)


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
                    td_mode="cash"
                    if signal.instrument_kind == "spot"
                    else "isolated",
                )
            )
        return intents
