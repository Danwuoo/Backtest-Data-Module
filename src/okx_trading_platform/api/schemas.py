from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from okx_trading_platform.domain import (
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


class ProfileCreate(ProfileConfig):
    pass


class Profile(ProfileConfig):
    model_config = ConfigDict(use_enum_values=True)


class InstrumentCreate(InstrumentConfig):
    pass


class Instrument(InstrumentConfig):
    model_config = ConfigDict(use_enum_values=True)


class BotCreate(BotConfig):
    pass


class Bot(BotConfig):
    model_config = ConfigDict(use_enum_values=True)


class DeploymentCreate(BaseModel):
    profile: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Deployment(DeploymentRecord):
    model_config = ConfigDict(use_enum_values=True)


class OrderCreate(OrderIntent):
    pass


class Order(OrderState):
    model_config = ConfigDict(use_enum_values=True)


class Balance(BalanceSnapshot):
    model_config = ConfigDict(use_enum_values=True)


class Position(PositionSnapshot):
    model_config = ConfigDict(use_enum_values=True)


class HeartbeatCreate(ServiceHeartbeat):
    pass


class Heartbeat(ServiceHeartbeat):
    model_config = ConfigDict(use_enum_values=True)


class KillSwitchUpdate(BaseModel):
    activated: bool
    reason: str | None = None


class KillSwitch(KillSwitchState):
    model_config = ConfigDict()


class CancelOrderRequest(BaseModel):
    profile: str
    inst_id: str
    order_id: str | None = None
    client_order_id: str | None = None
