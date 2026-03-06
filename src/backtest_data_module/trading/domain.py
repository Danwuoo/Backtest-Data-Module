from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """統一使用 UTC 時間戳，避免服務間比較時出現時區誤差。"""
    return datetime.now(timezone.utc)


def enum_value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    return value


class TradingProfile(str, Enum):
    DEMO = "demo"
    LIVE = "live"


class InstrumentKind(str, Enum):
    SPOT = "spot"
    SWAP = "swap"


class BotStatus(str, Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class TdMode(str, Enum):
    CASH = "cash"
    ISOLATED = "isolated"


class OrderLifecycleState(str, Enum):
    PENDING_RISK = "pending_risk"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    FAILED = "failed"


class ServiceStatus(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPED = "stopped"


class DeploymentStatus(str, Enum):
    DEPLOYED = "deployed"
    STOPPED = "stopped"
    FAILED = "failed"


class SignalEnvelope(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    signal_id: str = Field(default_factory=lambda: str(uuid4()))
    bot_name: str
    inst_id: str
    profile: TradingProfile
    instrument_kind: InstrumentKind
    side: OrderSide
    size: float
    price: float | None = None
    signal_type: str = "manual"
    emitted_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProfileConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    profile: TradingProfile
    name: str
    rest_base_url: str
    public_ws_url: str
    private_ws_url: str
    is_simulated: bool
    credential_env_prefix: str
    description: str | None = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class InstrumentConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    inst_id: str
    kind: InstrumentKind
    profile: TradingProfile
    allow_trading: bool = True
    tick_size: float | None = None
    lot_size: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BotConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    name: str
    profile: TradingProfile
    signal_provider: str
    instrument_ids: list[str] = Field(default_factory=list)
    status: BotStatus = BotStatus.DISABLED
    config: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class OrderIntent(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    order_id: str = Field(default_factory=lambda: str(uuid4()))
    client_order_id: str = Field(default_factory=lambda: str(uuid4()))
    profile: TradingProfile
    instrument_kind: InstrumentKind
    inst_id: str
    side: OrderSide
    size: float = Field(gt=0)
    order_type: OrderType = OrderType.MARKET
    td_mode: TdMode = TdMode.CASH
    price: float | None = None
    bot_name: str | None = None
    source: str = "manual"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class OrderState(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    order_id: str
    client_order_id: str
    profile: TradingProfile
    inst_id: str
    instrument_kind: InstrumentKind
    side: OrderSide
    size: float
    filled_size: float = 0.0
    avg_price: float | None = None
    price: float | None = None
    order_type: OrderType = OrderType.MARKET
    td_mode: TdMode = TdMode.CASH
    status: OrderLifecycleState = OrderLifecycleState.PENDING_RISK
    exchange_order_id: str | None = None
    bot_name: str | None = None
    rejection_reason: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utc_now)


class FillRecord(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    fill_id: str = Field(default_factory=lambda: str(uuid4()))
    order_id: str
    inst_id: str
    profile: TradingProfile
    fill_price: float
    fill_size: float
    fee: float = 0.0
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class PositionSnapshot(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    inst_id: str
    profile: TradingProfile
    instrument_kind: InstrumentKind
    quantity: float
    avg_price: float | None = None
    unrealized_pnl: float = 0.0
    td_mode: TdMode = TdMode.CASH
    updated_at: datetime = Field(default_factory=utc_now)


class BalanceSnapshot(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    profile: TradingProfile
    currency: str
    available: float
    cash_balance: float
    equity: float
    updated_at: datetime = Field(default_factory=utc_now)


class OrderBookLevel(BaseModel):
    price: float
    size: float


class OrderBookSnapshot(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    inst_id: str
    profile: TradingProfile
    bids: list[OrderBookLevel] = Field(default_factory=list)
    asks: list[OrderBookLevel] = Field(default_factory=list)
    sequence_id: int | None = None
    checksum: int | None = None
    updated_at: datetime = Field(default_factory=utc_now)


class ServiceHeartbeat(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    service_name: str
    instance_id: str
    profile: TradingProfile
    status: ServiceStatus = ServiceStatus.RUNNING
    metadata: dict[str, Any] = Field(default_factory=dict)
    last_seen_at: datetime = Field(default_factory=utc_now)


class DeploymentRecord(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    deployment_id: str = Field(default_factory=lambda: str(uuid4()))
    bot_name: str
    profile: TradingProfile
    status: DeploymentStatus = DeploymentStatus.DEPLOYED
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class KillSwitchState(BaseModel):
    activated: bool = False
    reason: str | None = None
    updated_at: datetime = Field(default_factory=utc_now)


class RiskDecision(BaseModel):
    approved: bool
    reason: str | None = None
    applied_limits: dict[str, Any] = Field(default_factory=dict)
