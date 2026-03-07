from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def enum_value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    return value


def make_id(prefix: str) -> str:
    return f"{prefix}-{uuid4()}"


class TradingEnvironment(str, Enum):
    DEMO = "demo"
    LIVE = "live"


class AccountScope(str, Enum):
    MAIN = "main"
    SUB_ACCOUNT = "sub_account"


class InstrumentKind(str, Enum):
    SPOT = "spot"
    SWAP = "swap"


class StrategyStatus(str, Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    KILLED = "killed"


class ModelKind(str, Enum):
    RULE_BASELINE = "rule_baseline"
    SHADOW = "shadow"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RunType(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


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
    PENDING_POLICY = "pending_policy"
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


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class SignalMode(str, Enum):
    PRIMARY = "primary"
    SHADOW = "shadow"


class ProfileConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    profile_id: str
    name: str
    environment: TradingEnvironment
    account_scope: AccountScope = AccountScope.MAIN
    account_label: str = "main"
    base_currency: str = "USDT"
    rest_base_url: str
    public_ws_url: str
    private_ws_url: str
    credential_env_prefix: str
    risk_policy_id: str
    allocator_id: str
    default_sleeve_id: str
    description: str | None = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class RiskPolicyConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    risk_policy_id: str
    profile_id: str
    name: str
    config: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class AllocatorConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    allocator_id: str
    profile_id: str
    name: str
    policy_type: str = "single_sleeve"
    config: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SleeveConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    sleeve_id: str
    profile_id: str
    name: str
    sleeve_type: str = "primary"
    capital_allocation: float = 1.0
    risk_budget: float = 1.0
    max_leverage: float = 1.0
    allowed_instrument_ids: list[str] = Field(default_factory=list)
    is_active: bool = True
    is_killed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class InstrumentConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    instrument_id: str
    profile_id: str
    inst_id: str
    inst_id_code: str | None = None
    kind: InstrumentKind
    inst_family: str | None = None
    base_currency: str | None = None
    quote_currency: str | None = None
    settle_currency: str | None = None
    tick_size: float | None = None
    lot_size: float | None = None
    min_size: float | None = None
    min_notional: float | None = None
    allow_trading: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    last_synced_at: datetime = Field(default_factory=utc_now)


class StrategyConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    strategy_id: str
    profile_id: str
    name: str
    baseline_provider: str = "reference_breakout"
    status: StrategyStatus = StrategyStatus.DISABLED
    allowed_instrument_ids: list[str] = Field(default_factory=list)
    primary_model_version_id: str | None = None
    is_killed: bool = False
    config: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ModelVersion(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    model_version_id: str
    strategy_id: str
    profile_id: str
    name: str
    kind: ModelKind = ModelKind.RULE_BASELINE
    artifact_uri: str | None = None
    is_primary: bool = True
    shadow_enabled: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class DatasetRecord(BaseModel):
    dataset_id: str
    profile_id: str
    name: str
    layer: str
    path: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class FeatureSet(BaseModel):
    feature_id: str
    profile_id: str
    name: str
    schema_version: str
    path: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class RunRecord(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    run_id: str
    profile_id: str
    strategy_id: str
    model_version_id: str | None = None
    sleeve_id: str | None = None
    status: RunStatus = RunStatus.PENDING
    run_type: RunType
    artifact_path: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class BacktestRun(RunRecord):
    run_type: RunType = RunType.BACKTEST


class PaperRun(RunRecord):
    run_type: RunType = RunType.PAPER


class LiveRun(RunRecord):
    run_type: RunType = RunType.LIVE


class TargetSignal(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    target_signal_id: str = Field(default_factory=lambda: make_id("target"))
    profile_id: str
    strategy_id: str
    model_version_id: str
    sleeve_id: str | None = None
    instrument_id: str
    inst_id: str
    kind: InstrumentKind
    side: OrderSide
    target_size: float = Field(gt=0)
    confidence: float = 1.0
    signal_mode: SignalMode = SignalMode.PRIMARY
    features: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class PositionIntent(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    intent_id: str = Field(default_factory=lambda: make_id("intent"))
    profile_id: str
    strategy_id: str
    model_version_id: str
    sleeve_id: str
    instrument_id: str
    inst_id: str
    kind: InstrumentKind
    side: OrderSide
    target_size: float = Field(gt=0)
    max_leverage: float = 1.0
    risk_budget: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class OrderPlan(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    order_plan_id: str = Field(default_factory=lambda: make_id("plan"))
    profile_id: str
    strategy_id: str
    model_version_id: str
    sleeve_id: str
    instrument_id: str
    inst_id: str
    kind: InstrumentKind
    side: OrderSide
    size: float = Field(gt=0)
    order_type: OrderType = OrderType.MARKET
    td_mode: TdMode = TdMode.CASH
    price: float | None = None
    min_notional: float | None = None
    notional: float | None = None
    rate_bucket: str | None = None
    source: str = "manual"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class OrderState(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    order_id: str = Field(default_factory=lambda: make_id("order"))
    client_order_id: str = Field(default_factory=lambda: make_id("clord"))
    order_plan_id: str | None = None
    profile_id: str
    strategy_id: str | None = None
    model_version_id: str | None = None
    sleeve_id: str | None = None
    instrument_id: str | None = None
    inst_id: str
    kind: InstrumentKind
    side: OrderSide
    size: float = Field(gt=0)
    filled_size: float = 0.0
    avg_price: float | None = None
    price: float | None = None
    order_type: OrderType = OrderType.MARKET
    td_mode: TdMode = TdMode.CASH
    status: OrderLifecycleState = OrderLifecycleState.PENDING_RISK
    exchange_order_id: str | None = None
    source: str = "manual"
    rejection_reason: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class FillRecord(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    fill_id: str = Field(default_factory=lambda: make_id("fill"))
    order_id: str
    profile_id: str
    instrument_id: str | None = None
    inst_id: str
    fill_price: float
    fill_size: float
    fee: float = 0.0
    funding_cost: float = 0.0
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class LedgerEntry(BaseModel):
    ledger_entry_id: str = Field(default_factory=lambda: make_id("ledger"))
    profile_id: str
    order_id: str | None = None
    fill_id: str | None = None
    currency: str = "USDT"
    amount: float
    entry_type: str
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class FundingEntry(BaseModel):
    funding_entry_id: str = Field(default_factory=lambda: make_id("funding"))
    profile_id: str
    instrument_id: str | None = None
    inst_id: str
    amount: float
    rate: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class PnLSnapshot(BaseModel):
    pnl_snapshot_id: str = Field(default_factory=lambda: make_id("pnl"))
    profile_id: str
    strategy_id: str | None = None
    sleeve_id: str | None = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    net_pnl: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class RiskSnapshot(BaseModel):
    risk_snapshot_id: str = Field(default_factory=lambda: make_id("risk"))
    profile_id: str
    order_id: str | None = None
    strategy_id: str | None = None
    sleeve_id: str | None = None
    stage: str = "pre_trade"
    approved: bool = True
    reason: str | None = None
    applied_limits: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ExecutionSnapshot(BaseModel):
    execution_snapshot_id: str = Field(default_factory=lambda: make_id("exec"))
    profile_id: str
    order_id: str
    status: str
    signal_ts: datetime | None = None
    risk_ts: datetime | None = None
    send_ts: datetime | None = None
    ack_ts: datetime | None = None
    fill_ts: datetime | None = None
    send_latency_ms: float | None = None
    ack_latency_ms: float | None = None
    fill_latency_ms: float | None = None
    slippage_bps: float | None = None
    maker_ratio: float | None = None
    taker_ratio: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class PositionSnapshot(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    position_snapshot_id: str = Field(default_factory=lambda: make_id("position"))
    profile_id: str
    sleeve_id: str | None = None
    instrument_id: str | None = None
    inst_id: str
    kind: InstrumentKind
    quantity: float
    avg_price: float | None = None
    unrealized_pnl: float = 0.0
    td_mode: TdMode = TdMode.CASH
    updated_at: datetime = Field(default_factory=utc_now)


class BalanceSnapshot(BaseModel):
    balance_snapshot_id: str = Field(default_factory=lambda: make_id("balance"))
    profile_id: str
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

    instrument_id: str | None = None
    inst_id: str
    profile_id: str
    bids: list[OrderBookLevel] = Field(default_factory=list)
    asks: list[OrderBookLevel] = Field(default_factory=list)
    sequence_id: int | None = None
    prev_sequence_id: int | None = None
    gap_detected: bool = False
    checksum: int | None = None
    updated_at: datetime = Field(default_factory=utc_now)


class ServiceHeartbeat(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    service_name: str
    instance_id: str
    profile_id: str
    status: ServiceStatus = ServiceStatus.RUNNING
    metadata: dict[str, Any] = Field(default_factory=dict)
    last_seen_at: datetime = Field(default_factory=utc_now)


class KillSwitchState(BaseModel):
    activated: bool = False
    reason: str | None = None
    updated_at: datetime = Field(default_factory=utc_now)


class IncidentRecord(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    incident_id: str = Field(default_factory=lambda: make_id("incident"))
    profile_id: str
    severity: AlertSeverity
    title: str
    message: str
    status: AlertStatus = AlertStatus.OPEN
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class AlertPolicy(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    alert_policy_id: str
    profile_id: str
    name: str
    severity_threshold: AlertSeverity = AlertSeverity.WARNING
    channel: str = "slack"
    is_active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class AlertRecord(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    alert_id: str = Field(default_factory=lambda: make_id("alert"))
    profile_id: str
    incident_id: str | None = None
    severity: AlertSeverity
    title: str
    message: str
    channel: str = "slack"
    status: AlertStatus = AlertStatus.OPEN
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class RiskDecision(BaseModel):
    approved: bool
    stage: str = "pre_trade"
    reason: str | None = None
    applied_limits: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
