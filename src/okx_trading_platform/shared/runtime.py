from __future__ import annotations

import os
from dataclasses import dataclass, field
from uuid import uuid4

from okx_trading_platform.adapters.okx import OkxExchangeGateway, OkxOrderBook
from okx_trading_platform.application.signals import InferenceProvider
from okx_trading_platform.domain import (
    OrderBookSnapshot,
    OrderPlan,
    PositionIntent,
    ServiceHeartbeat,
    ServiceStatus,
    TargetSignal,
    TradingEnvironment,
    utc_now,
)
from okx_trading_platform.domain.risk import RiskManager
from okx_trading_platform.shared.data_lake import DataLakeRecord, DataLakeWriter


@dataclass
class ServiceRuntime:
    service_name: str
    profile_id: str
    environment: TradingEnvironment
    instance_id: str = field(
        default_factory=lambda: os.getenv("HOSTNAME", str(uuid4()))
    )
    status: ServiceStatus = ServiceStatus.STARTING
    metadata: dict = field(default_factory=dict)

    def heartbeat(self) -> ServiceHeartbeat:
        return ServiceHeartbeat(
            service_name=self.service_name,
            instance_id=self.instance_id,
            profile_id=self.profile_id,
            status=self.status,
            metadata=self.metadata,
            last_seen_at=utc_now(),
        )

    def set_running(self) -> None:
        self.status = ServiceStatus.RUNNING


@dataclass
class MarketDataRuntime(ServiceRuntime):
    books: dict[str, OkxOrderBook] = field(default_factory=dict)
    data_lake: DataLakeWriter | None = None

    def upsert_snapshot(
        self,
        inst_id: str,
        bids: list[list[str | float]],
        asks: list[list[str | float]],
        *,
        instrument_id: str | None = None,
        sequence_id: int | None = None,
    ) -> OrderBookSnapshot:
        book = self.books.setdefault(inst_id, OkxOrderBook(inst_id, self.profile_id))
        snapshot = book.apply_snapshot(
            bids,
            asks,
            instrument_id=instrument_id,
            sequence_id=sequence_id,
        )
        self._persist(
            "bronze", inst_id, "book_snapshot", snapshot.model_dump(mode="json")
        )
        return snapshot

    def apply_delta(
        self,
        inst_id: str,
        bids: list[list[str | float]],
        asks: list[list[str | float]],
        *,
        instrument_id: str | None = None,
        sequence_id: int | None = None,
        prev_sequence_id: int | None = None,
    ) -> OrderBookSnapshot:
        book = self.books.setdefault(inst_id, OkxOrderBook(inst_id, self.profile_id))
        snapshot = book.apply_update(
            bids,
            asks,
            instrument_id=instrument_id,
            sequence_id=sequence_id,
            prev_sequence_id=prev_sequence_id,
        )
        self._persist("silver", inst_id, "book_delta", snapshot.model_dump(mode="json"))
        return snapshot

    def _persist(self, layer: str, inst_id: str, stream: str, payload: dict) -> None:
        if self.data_lake is None:
            return
        self.data_lake.write(
            DataLakeRecord(
                layer=layer,
                dt=utc_now().date().isoformat(),
                profile_id=self.profile_id,
                venue="okx",
                inst_id=inst_id,
                stream=stream,
                payload=payload,
            )
        )


@dataclass
class InferenceRuntime(ServiceRuntime):
    provider: InferenceProvider | None = None

    def infer_targets(self, market_state: dict) -> list[TargetSignal]:
        if self.provider is None:
            return []
        return list(self.provider.infer_targets(market_state))


@dataclass
class PortfolioRuntime(ServiceRuntime):
    def build_position_intents(
        self, intents: list[PositionIntent]
    ) -> list[PositionIntent]:
        return intents


@dataclass
class ExecutionPolicyRuntime(ServiceRuntime):
    def build_order_plans(self, plans: list[OrderPlan]) -> list[OrderPlan]:
        return plans


@dataclass
class ExecutionRuntime(ServiceRuntime):
    gateway: OkxExchangeGateway | None = None

    def submit(self, plan: OrderPlan) -> dict:
        if self.gateway is None:
            raise RuntimeError("execution gateway is not configured")
        return self.gateway.submit_order(plan).model_dump(mode="json")


@dataclass
class RiskRuntime(ServiceRuntime):
    risk_manager: RiskManager | None = None

    def evaluate(self, *args, **kwargs):
        if self.risk_manager is None:
            raise RuntimeError("risk manager is not configured")
        return self.risk_manager.evaluate_order(*args, **kwargs)


@dataclass
class ReplayRuntime(ServiceRuntime):
    data_lake: DataLakeWriter | None = None

    def replay(self, payload: dict) -> dict:
        return {"status": "scheduled", "payload": payload}
