from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from okx_trading_platform.domain import (
    AlertPolicy,
    AlertRecord,
    AllocatorConfig,
    BacktestRun,
    BalanceSnapshot,
    DatasetRecord,
    FeatureSet,
    FillRecord,
    IncidentRecord,
    InstrumentConfig,
    KillSwitchState,
    LiveRun,
    ModelVersion,
    OrderPlan,
    OrderState,
    PaperRun,
    PositionSnapshot,
    ProfileConfig,
    RiskPolicyConfig,
    ServiceHeartbeat,
    SleeveConfig,
    StrategyConfig,
)


class ProfileCreate(ProfileConfig):
    pass


class Profile(ProfileConfig):
    model_config = ConfigDict(use_enum_values=True)


class RiskPolicyCreate(RiskPolicyConfig):
    pass


class RiskPolicy(RiskPolicyConfig):
    model_config = ConfigDict(use_enum_values=True)


class AllocatorCreate(AllocatorConfig):
    pass


class Allocator(AllocatorConfig):
    model_config = ConfigDict(use_enum_values=True)


class SleeveCreate(SleeveConfig):
    pass


class Sleeve(SleeveConfig):
    model_config = ConfigDict(use_enum_values=True)


class InstrumentCreate(InstrumentConfig):
    pass


class Instrument(InstrumentConfig):
    model_config = ConfigDict(use_enum_values=True)


class StrategyCreate(StrategyConfig):
    pass


class Strategy(StrategyConfig):
    model_config = ConfigDict(use_enum_values=True)


class ModelVersionCreate(ModelVersion):
    pass


class Model(ModelVersion):
    model_config = ConfigDict(use_enum_values=True)


class DatasetCreate(DatasetRecord):
    pass


class Dataset(DatasetRecord):
    model_config = ConfigDict(use_enum_values=True)


class FeatureCreate(FeatureSet):
    pass


class Feature(FeatureSet):
    model_config = ConfigDict(use_enum_values=True)


class BacktestCreate(BacktestRun):
    pass


class Backtest(BacktestRun):
    model_config = ConfigDict(use_enum_values=True)


class PaperRunCreate(PaperRun):
    pass


class PaperRunSchema(PaperRun):
    model_config = ConfigDict(use_enum_values=True)


class LiveRunCreate(LiveRun):
    pass


class LiveRunSchema(LiveRun):
    model_config = ConfigDict(use_enum_values=True)


class OrderCreate(OrderPlan):
    pass


class Order(OrderState):
    model_config = ConfigDict(use_enum_values=True)


class FillCreate(FillRecord):
    pass


class Fill(FillRecord):
    model_config = ConfigDict(use_enum_values=True)


class Position(PositionSnapshot):
    model_config = ConfigDict(use_enum_values=True)


class Balance(BalanceSnapshot):
    model_config = ConfigDict(use_enum_values=True)


class HeartbeatCreate(ServiceHeartbeat):
    pass


class Heartbeat(ServiceHeartbeat):
    model_config = ConfigDict(use_enum_values=True)


class IncidentCreate(IncidentRecord):
    pass


class Incident(IncidentRecord):
    model_config = ConfigDict(use_enum_values=True)


class AlertPolicyCreate(AlertPolicy):
    pass


class AlertPolicySchema(AlertPolicy):
    model_config = ConfigDict(use_enum_values=True)


class AlertCreate(AlertRecord):
    pass


class Alert(AlertRecord):
    model_config = ConfigDict(use_enum_values=True)


class KillSwitchUpdate(BaseModel):
    activated: bool
    reason: str | None = None


class KillSwitch(KillSwitchState):
    model_config = ConfigDict()


class CancelOrderRequest(BaseModel):
    order_id: str
