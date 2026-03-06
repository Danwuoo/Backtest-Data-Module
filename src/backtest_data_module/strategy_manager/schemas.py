import datetime as dt
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from backtest_data_module.trading.domain import (
    BalanceSnapshot,
    BotConfig,
    DeploymentRecord,
    InstrumentConfig,
    KillSwitchState,
    OrderIntent,
    OrderState,
    PositionSnapshot,
    ProfileConfig,
    ServiceHeartbeat,
)


class RunBase(BaseModel):
    strategy_name: str
    strategy_version: str
    hyperparameters: dict[str, Any]
    orchestrator_type: str


class RunCreate(RunBase):
    pass


class RunUpdate(BaseModel):
    status: Optional[str] = None
    metrics_uri: Optional[str] = None
    error_message: Optional[str] = None


class Run(RunBase):
    model_config = ConfigDict(from_attributes=True)

    run_id: UUID
    timestamp: dt.datetime
    status: str
    metrics_uri: Optional[str] = None
    error_message: Optional[str] = None


class ProfileCreate(ProfileConfig):
    pass


class ProfileUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    is_active: bool | None = None


class Profile(ProfileConfig):
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class InstrumentCreate(InstrumentConfig):
    pass


class Instrument(InstrumentConfig):
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

    instrument_id: str | None = None


class BotCreate(BotConfig):
    pass


class BotUpdate(BaseModel):
    status: str | None = None
    instrument_ids: list[str] | None = None
    config: dict[str, Any] | None = None


class Bot(BotConfig):
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

    bot_id: str | None = None


class OrderCreate(OrderIntent):
    pass


class Order(OrderState):
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class DeploymentCreate(BaseModel):
    profile: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Deployment(DeploymentRecord):
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class Balance(BalanceSnapshot):
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class Position(PositionSnapshot):
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class HeartbeatCreate(ServiceHeartbeat):
    pass


class Heartbeat(ServiceHeartbeat):
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class KillSwitchUpdate(BaseModel):
    activated: bool
    reason: str | None = None


class KillSwitch(KillSwitchState):
    model_config = ConfigDict(from_attributes=True)


class CancelOrderRequest(BaseModel):
    profile: str
    inst_id: str
    order_id: str | None = None
    client_order_id: str | None = None
