from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import Boolean, DateTime, Float, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from okx_trading_platform.shared.db import Base


def generate_id() -> str:
    return str(uuid.uuid4())


class TimestampMixin:
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime,
        default=dt.datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime,
        default=dt.datetime.utcnow,
        onupdate=dt.datetime.utcnow,
        nullable=False,
    )


class PayloadMixin:
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)


class NamedResourceMixin(TimestampMixin, PayloadMixin):
    name: Mapped[str] = mapped_column(String, nullable=False)


# Legacy v1 tables kept for migration/backfill support.
class TradingProfileModel(Base):
    __tablename__ = "trading_profiles"

    profile: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    rest_base_url: Mapped[str] = mapped_column(String, nullable=False)
    public_ws_url: Mapped[str] = mapped_column(String, nullable=False)
    private_ws_url: Mapped[str] = mapped_column(String, nullable=False)
    credential_env_prefix: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    is_simulated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime,
        default=dt.datetime.utcnow,
        onupdate=dt.datetime.utcnow,
        nullable=False,
    )


class InstrumentModel(Base):
    __tablename__ = "trading_instruments"
    __table_args__ = (
        UniqueConstraint("profile", "inst_id", name="uq_profile_inst_id"),
    )

    instrument_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=generate_id
    )
    inst_id: Mapped[str] = mapped_column(String, nullable=False)
    profile: Mapped[str] = mapped_column(String, nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    allow_trading: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    tick_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    lot_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime,
        default=dt.datetime.utcnow,
        onupdate=dt.datetime.utcnow,
        nullable=False,
    )


class BotModel(Base):
    __tablename__ = "trading_bots"

    bot_id: Mapped[str] = mapped_column(String, primary_key=True, default=generate_id)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    profile: Mapped[str] = mapped_column(String, nullable=False)
    signal_provider: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="disabled")
    instrument_ids: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    config_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime,
        default=dt.datetime.utcnow,
        onupdate=dt.datetime.utcnow,
        nullable=False,
    )


class OrderRecordModel(Base):
    __tablename__ = "trading_orders"

    order_id: Mapped[str] = mapped_column(String, primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    profile: Mapped[str] = mapped_column(String, nullable=False)
    inst_id: Mapped[str] = mapped_column(String, nullable=False)
    instrument_kind: Mapped[str] = mapped_column(String, nullable=False)
    side: Mapped[str] = mapped_column(String, nullable=False)
    size: Mapped[float] = mapped_column(Float, nullable=False)
    filled_size: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    order_type: Mapped[str] = mapped_column(String, nullable=False)
    td_mode: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending_risk")
    exchange_order_id: Mapped[str | None] = mapped_column(String, nullable=True)
    bot_name: Mapped[str | None] = mapped_column(String, nullable=True)
    source: Mapped[str] = mapped_column(String, nullable=False, default="manual")
    rejection_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    raw_payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime,
        default=dt.datetime.utcnow,
        onupdate=dt.datetime.utcnow,
        nullable=False,
    )


class FillRecordModel(Base):
    __tablename__ = "trading_fills"

    fill_id: Mapped[str] = mapped_column(String, primary_key=True, default=generate_id)
    order_id: Mapped[str] = mapped_column(String, nullable=False)
    profile: Mapped[str] = mapped_column(String, nullable=False)
    inst_id: Mapped[str] = mapped_column(String, nullable=False)
    fill_price: Mapped[float] = mapped_column(Float, nullable=False)
    fill_size: Mapped[float] = mapped_column(Float, nullable=False)
    fee: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    raw_payload: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )


class PositionSnapshotModel(Base):
    __tablename__ = "trading_positions"
    __table_args__ = (
        UniqueConstraint("profile", "inst_id", name="uq_profile_position"),
    )

    position_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=generate_id
    )
    profile: Mapped[str] = mapped_column(String, nullable=False)
    inst_id: Mapped[str] = mapped_column(String, nullable=False)
    instrument_kind: Mapped[str] = mapped_column(String, nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    avg_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    unrealized_pnl: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    td_mode: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )


class BalanceSnapshotModel(Base):
    __tablename__ = "trading_balances"
    __table_args__ = (
        UniqueConstraint("profile", "currency", name="uq_profile_currency"),
    )

    balance_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=generate_id
    )
    profile: Mapped[str] = mapped_column(String, nullable=False)
    currency: Mapped[str] = mapped_column(String, nullable=False)
    available: Mapped[float] = mapped_column(Float, nullable=False)
    cash_balance: Mapped[float] = mapped_column(Float, nullable=False)
    equity: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )


class ServiceHeartbeatModel(Base):
    __tablename__ = "service_heartbeats"
    __table_args__ = (
        UniqueConstraint("service_name", "instance_id", name="uq_service_instance"),
    )

    heartbeat_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=generate_id
    )
    service_name: Mapped[str] = mapped_column(String, nullable=False)
    instance_id: Mapped[str] = mapped_column(String, nullable=False)
    profile: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="running")
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    last_seen_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )


class DeploymentRecordModel(Base):
    __tablename__ = "deployment_history"

    deployment_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=generate_id
    )
    bot_name: Mapped[str] = mapped_column(String, nullable=False)
    profile: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="deployed")
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )


class KillSwitchModel(Base):
    __tablename__ = "kill_switch"

    kill_switch_id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    activated: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    reason: Mapped[str | None] = mapped_column(String, nullable=True)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )


# V2 platform tables.
class ProfileV2Model(Base, NamedResourceMixin):
    __tablename__ = "platform_profiles"

    profile_id: Mapped[str] = mapped_column(String, primary_key=True)
    environment: Mapped[str] = mapped_column(String, nullable=False)
    account_scope: Mapped[str] = mapped_column(String, nullable=False)


class RiskPolicyV2Model(Base, NamedResourceMixin):
    __tablename__ = "platform_risk_policies"

    risk_policy_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)


class AllocatorV2Model(Base, NamedResourceMixin):
    __tablename__ = "platform_allocators"

    allocator_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    policy_type: Mapped[str] = mapped_column(String, nullable=False)


class SleeveV2Model(Base, NamedResourceMixin):
    __tablename__ = "platform_sleeves"

    sleeve_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    sleeve_type: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class InstrumentV2Model(Base, PayloadMixin):
    __tablename__ = "platform_instruments"
    __table_args__ = (
        UniqueConstraint("profile_id", "inst_id", name="uq_platform_profile_inst"),
    )

    instrument_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    inst_id: Mapped[str] = mapped_column(String, nullable=False)
    inst_id_code: Mapped[str | None] = mapped_column(String, nullable=True)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    allow_trading: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime,
        default=dt.datetime.utcnow,
        onupdate=dt.datetime.utcnow,
        nullable=False,
    )


class StrategyV2Model(Base, NamedResourceMixin):
    __tablename__ = "platform_strategies"

    strategy_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)


class ModelVersionV2Model(Base, NamedResourceMixin):
    __tablename__ = "platform_model_versions"

    model_version_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    strategy_id: Mapped[str] = mapped_column(String, nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class DatasetV2Model(Base, NamedResourceMixin):
    __tablename__ = "platform_datasets"

    dataset_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    layer: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False)


class FeatureV2Model(Base, NamedResourceMixin):
    __tablename__ = "platform_features"

    feature_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    schema_version: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False)


class DatasetVersionV2Model(Base, PayloadMixin):
    __tablename__ = "platform_dataset_versions"
    __table_args__ = (
        UniqueConstraint("dataset_id", "version", name="uq_platform_dataset_version"),
    )

    dataset_version_id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String, nullable=False)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[str] = mapped_column(String, nullable=False)
    layer: Mapped[str] = mapped_column(String, nullable=False)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )


class RunArtifactV2Model(Base, PayloadMixin):
    __tablename__ = "platform_run_artifacts"

    artifact_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, nullable=False)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    run_type: Mapped[str] = mapped_column(String, nullable=False)
    artifact_type: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )


class RunBaseModel(Base, PayloadMixin):
    __abstract__ = True

    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    strategy_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime,
        default=dt.datetime.utcnow,
        onupdate=dt.datetime.utcnow,
        nullable=False,
    )


class BacktestRunV2Model(RunBaseModel):
    __tablename__ = "platform_backtest_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)


class PaperRunV2Model(RunBaseModel):
    __tablename__ = "platform_paper_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)


class LiveRunV2Model(RunBaseModel):
    __tablename__ = "platform_live_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)


class OrderPlanV2Model(Base, PayloadMixin):
    __tablename__ = "platform_order_plans"

    order_plan_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    strategy_id: Mapped[str] = mapped_column(String, nullable=False)
    sleeve_id: Mapped[str] = mapped_column(String, nullable=False)
    inst_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="planned")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime,
        default=dt.datetime.utcnow,
        onupdate=dt.datetime.utcnow,
        nullable=False,
    )


class OrderV2Model(Base, PayloadMixin):
    __tablename__ = "platform_orders"

    order_id: Mapped[str] = mapped_column(String, primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    strategy_id: Mapped[str | None] = mapped_column(String, nullable=True)
    sleeve_id: Mapped[str | None] = mapped_column(String, nullable=True)
    inst_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime,
        default=dt.datetime.utcnow,
        onupdate=dt.datetime.utcnow,
        nullable=False,
    )


class FillV2Model(Base, PayloadMixin):
    __tablename__ = "platform_fills"

    fill_id: Mapped[str] = mapped_column(String, primary_key=True)
    order_id: Mapped[str] = mapped_column(String, nullable=False)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    inst_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )


class LedgerEntryV2Model(Base, PayloadMixin):
    __tablename__ = "platform_ledger_entries"

    ledger_entry_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    order_id: Mapped[str | None] = mapped_column(String, nullable=True)
    entry_type: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )


class FundingEntryV2Model(Base, PayloadMixin):
    __tablename__ = "platform_funding_entries"

    funding_entry_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    inst_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )


class PnLSnapshotV2Model(Base, PayloadMixin):
    __tablename__ = "platform_pnl_snapshots"

    pnl_snapshot_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )


class RiskSnapshotV2Model(Base, PayloadMixin):
    __tablename__ = "platform_risk_snapshots"

    risk_snapshot_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    order_id: Mapped[str | None] = mapped_column(String, nullable=True)
    stage: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )


class ExecutionSnapshotV2Model(Base, PayloadMixin):
    __tablename__ = "platform_execution_snapshots"

    execution_snapshot_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    order_id: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )


class PositionV2Model(Base, PayloadMixin):
    __tablename__ = "platform_positions"
    __table_args__ = (
        UniqueConstraint(
            "profile_id", "inst_id", "sleeve_id", name="uq_platform_position"
        ),
    )

    position_snapshot_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    sleeve_id: Mapped[str | None] = mapped_column(String, nullable=True)
    inst_id: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)


class BalanceV2Model(Base, PayloadMixin):
    __tablename__ = "platform_balances"
    __table_args__ = (
        UniqueConstraint("profile_id", "currency", name="uq_platform_currency"),
    )

    balance_snapshot_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    currency: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)


class IncidentV2Model(Base, PayloadMixin):
    __tablename__ = "platform_incidents"

    incident_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    severity: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime,
        default=dt.datetime.utcnow,
        onupdate=dt.datetime.utcnow,
        nullable=False,
    )


class AlertPolicyV2Model(Base, NamedResourceMixin):
    __tablename__ = "platform_alert_policies"

    alert_policy_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    severity_threshold: Mapped[str] = mapped_column(String, nullable=False)
    channel: Mapped[str] = mapped_column(String, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class AlertV2Model(Base, PayloadMixin):
    __tablename__ = "platform_alerts"

    alert_id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False)
    incident_id: Mapped[str | None] = mapped_column(String, nullable=True)
    severity: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, nullable=False
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime,
        default=dt.datetime.utcnow,
        onupdate=dt.datetime.utcnow,
        nullable=False,
    )
