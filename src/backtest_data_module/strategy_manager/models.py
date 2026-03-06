import datetime as dt
import uuid
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from .database import Base


class Run(Base):
    __tablename__ = "runs"

    run_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=dt.datetime.utcnow)
    strategy_name = Column(String, nullable=False)
    strategy_version = Column(String, nullable=False)
    hyperparameters = Column(JSON, nullable=False)
    orchestrator_type = Column(String, nullable=False)
    metrics_uri = Column(String, nullable=True)
    status = Column(
        Enum("PENDING", "RUNNING", "FAILED", "COMPLETED", name="run_status"),
        nullable=False,
    )
    error_message = Column(String, nullable=True)


def generate_id() -> str:
    return str(uuid.uuid4())


class TradingProfileModel(Base):
    __tablename__ = "trading_profiles"

    profile = Column(String, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    rest_base_url = Column(String, nullable=False)
    public_ws_url = Column(String, nullable=False)
    private_ws_url = Column(String, nullable=False)
    credential_env_prefix = Column(String, nullable=False)
    description = Column(String, nullable=True)
    is_simulated = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(
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

    instrument_id = Column(String, primary_key=True, default=generate_id)
    inst_id = Column(String, nullable=False)
    profile = Column(String, nullable=False)
    kind = Column(String, nullable=False)
    allow_trading = Column(Boolean, default=True, nullable=False)
    tick_size = Column(Float, nullable=True)
    lot_size = Column(Float, nullable=True)
    metadata_json = Column(JSON, default=dict, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=dt.datetime.utcnow,
        onupdate=dt.datetime.utcnow,
        nullable=False,
    )


class BotModel(Base):
    __tablename__ = "trading_bots"

    bot_id = Column(String, primary_key=True, default=generate_id)
    name = Column(String, unique=True, nullable=False)
    profile = Column(String, nullable=False)
    signal_provider = Column(String, nullable=False)
    status = Column(String, nullable=False, default="disabled")
    instrument_ids = Column(JSON, default=list, nullable=False)
    config_json = Column(JSON, default=dict, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=dt.datetime.utcnow,
        onupdate=dt.datetime.utcnow,
        nullable=False,
    )


class OrderRecordModel(Base):
    __tablename__ = "trading_orders"

    order_id = Column(String, primary_key=True)
    client_order_id = Column(String, unique=True, nullable=False)
    profile = Column(String, nullable=False)
    inst_id = Column(String, nullable=False)
    instrument_kind = Column(String, nullable=False)
    side = Column(String, nullable=False)
    size = Column(Float, nullable=False)
    filled_size = Column(Float, nullable=False, default=0.0)
    avg_price = Column(Float, nullable=True)
    price = Column(Float, nullable=True)
    order_type = Column(String, nullable=False)
    td_mode = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending_risk")
    exchange_order_id = Column(String, nullable=True)
    bot_name = Column(String, nullable=True)
    source = Column(String, nullable=False, default="manual")
    rejection_reason = Column(String, nullable=True)
    raw_payload = Column(JSON, default=dict, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=dt.datetime.utcnow,
        onupdate=dt.datetime.utcnow,
        nullable=False,
    )


class FillRecordModel(Base):
    __tablename__ = "trading_fills"

    fill_id = Column(String, primary_key=True, default=generate_id)
    order_id = Column(String, nullable=False)
    profile = Column(String, nullable=False)
    inst_id = Column(String, nullable=False)
    fill_price = Column(Float, nullable=False)
    fill_size = Column(Float, nullable=False)
    fee = Column(Float, nullable=False, default=0.0)
    raw_payload = Column(JSON, default=dict, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)


class PositionSnapshotModel(Base):
    __tablename__ = "trading_positions"
    __table_args__ = (
        UniqueConstraint("profile", "inst_id", name="uq_profile_position"),
    )

    position_id = Column(String, primary_key=True, default=generate_id)
    profile = Column(String, nullable=False)
    inst_id = Column(String, nullable=False)
    instrument_kind = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    avg_price = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, nullable=False, default=0.0)
    td_mode = Column(String, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)


class BalanceSnapshotModel(Base):
    __tablename__ = "trading_balances"
    __table_args__ = (
        UniqueConstraint("profile", "currency", name="uq_profile_currency"),
    )

    balance_id = Column(String, primary_key=True, default=generate_id)
    profile = Column(String, nullable=False)
    currency = Column(String, nullable=False)
    available = Column(Float, nullable=False)
    cash_balance = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)


class ServiceHeartbeatModel(Base):
    __tablename__ = "service_heartbeats"
    __table_args__ = (
        UniqueConstraint("service_name", "instance_id", name="uq_service_instance"),
    )

    heartbeat_id = Column(String, primary_key=True, default=generate_id)
    service_name = Column(String, nullable=False)
    instance_id = Column(String, nullable=False)
    profile = Column(String, nullable=False)
    status = Column(String, nullable=False, default="running")
    metadata_json = Column(JSON, default=dict, nullable=False)
    last_seen_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)


class DeploymentRecordModel(Base):
    __tablename__ = "deployment_history"

    deployment_id = Column(String, primary_key=True, default=generate_id)
    bot_name = Column(String, nullable=False)
    profile = Column(String, nullable=False)
    status = Column(String, nullable=False, default="deployed")
    metadata_json = Column(JSON, default=dict, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)


class KillSwitchModel(Base):
    __tablename__ = "kill_switch"

    kill_switch_id = Column(Integer, primary_key=True, default=1)
    activated = Column(Boolean, default=False, nullable=False)
    reason = Column(String, nullable=True)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
