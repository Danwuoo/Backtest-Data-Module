from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import Boolean, DateTime, Float, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from okx_trading_platform.shared.db import Base


def generate_id() -> str:
    return str(uuid.uuid4())


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
