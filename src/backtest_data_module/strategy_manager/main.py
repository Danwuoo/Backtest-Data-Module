import os
from contextlib import asynccontextmanager
from typing import List
import uuid

from fastapi import Depends, FastAPI, HTTPException, Query, Security
from sqlalchemy.orm import Session

from backtest_data_module.strategy_manager import models, schemas
from backtest_data_module.strategy_manager.auth import get_api_key
from backtest_data_module.strategy_manager.database import engine, get_db
from backtest_data_module.trading.domain import (
    BotStatus,
    DeploymentStatus,
    OrderLifecycleState,
    TradingProfile,
    enum_value,
)
from backtest_data_module.trading.okx import (
    ClientOrderIdCache,
    OkxExecutionService,
    OkxRestClient,
    OkxWebSocketRouter,
)
from backtest_data_module.trading.risk import RiskService
from backtest_data_module.trading.settings import get_okx_profile_settings

models.Base.metadata.create_all(bind=engine)

risk_service = RiskService()
execution_service = OkxExecutionService(
    rest_client=OkxRestClient(),
    router=OkxWebSocketRouter(),
    dedupe_cache=ClientOrderIdCache(),
)


def ensure_default_profiles(db: Session) -> None:
    for profile in TradingProfile:
        profile_value = enum_value(profile)
        existing = db.get(models.TradingProfileModel, profile_value)
        if existing:
            continue
        settings = get_okx_profile_settings(profile)
        db.add(
            models.TradingProfileModel(
                profile=profile_value,
                name=profile_value,
                rest_base_url=settings.rest_base_url,
                public_ws_url=settings.public_ws_url,
                private_ws_url=settings.private_ws_url,
                credential_env_prefix=settings.credential_env_prefix,
                description=f"Default {profile_value} profile",
                is_simulated=settings.simulated_trading,
                is_active=True,
            )
        )
    db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    models.Base.metadata.create_all(bind=engine)
    db = next(get_db())
    try:
        ensure_default_profiles(db)
    finally:
        db.close()
    yield


app = FastAPI(lifespan=lifespan)


def _profile_schema(model: models.TradingProfileModel) -> schemas.Profile:
    return schemas.Profile(
        profile=model.profile,
        name=model.name,
        rest_base_url=model.rest_base_url,
        public_ws_url=model.public_ws_url,
        private_ws_url=model.private_ws_url,
        is_simulated=model.is_simulated,
        credential_env_prefix=model.credential_env_prefix,
        description=model.description,
        is_active=model.is_active,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def _instrument_schema(model: models.InstrumentModel) -> schemas.Instrument:
    return schemas.Instrument(
        instrument_id=model.instrument_id,
        inst_id=model.inst_id,
        kind=model.kind,
        profile=model.profile,
        allow_trading=model.allow_trading,
        tick_size=model.tick_size,
        lot_size=model.lot_size,
        metadata=model.metadata_json,
    )


def _bot_schema(model: models.BotModel) -> schemas.Bot:
    return schemas.Bot(
        bot_id=model.bot_id,
        name=model.name,
        profile=model.profile,
        signal_provider=model.signal_provider,
        instrument_ids=model.instrument_ids,
        status=model.status,
        config=model.config_json,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def _order_schema(model: models.OrderRecordModel) -> schemas.Order:
    return schemas.Order(
        order_id=model.order_id,
        client_order_id=model.client_order_id,
        profile=model.profile,
        inst_id=model.inst_id,
        instrument_kind=model.instrument_kind,
        side=model.side,
        size=model.size,
        filled_size=model.filled_size,
        avg_price=model.avg_price,
        price=model.price,
        order_type=model.order_type,
        td_mode=model.td_mode,
        status=model.status,
        exchange_order_id=model.exchange_order_id,
        bot_name=model.bot_name,
        rejection_reason=model.rejection_reason,
        raw_payload=model.raw_payload,
        updated_at=model.updated_at,
    )


def _deployment_schema(model: models.DeploymentRecordModel) -> schemas.Deployment:
    return schemas.Deployment(
        deployment_id=model.deployment_id,
        bot_name=model.bot_name,
        profile=model.profile,
        status=model.status,
        metadata=model.metadata_json,
        created_at=model.created_at,
    )


def _balance_schema(model: models.BalanceSnapshotModel) -> schemas.Balance:
    return schemas.Balance(
        profile=model.profile,
        currency=model.currency,
        available=model.available,
        cash_balance=model.cash_balance,
        equity=model.equity,
        updated_at=model.updated_at,
    )


def _position_schema(model: models.PositionSnapshotModel) -> schemas.Position:
    return schemas.Position(
        inst_id=model.inst_id,
        profile=model.profile,
        instrument_kind=model.instrument_kind,
        quantity=model.quantity,
        avg_price=model.avg_price,
        unrealized_pnl=model.unrealized_pnl,
        td_mode=model.td_mode,
        updated_at=model.updated_at,
    )


def _heartbeat_schema(model: models.ServiceHeartbeatModel) -> schemas.Heartbeat:
    return schemas.Heartbeat(
        service_name=model.service_name,
        instance_id=model.instance_id,
        profile=model.profile,
        status=model.status,
        metadata=model.metadata_json,
        last_seen_at=model.last_seen_at,
    )


def _kill_switch_schema(model: models.KillSwitchModel) -> schemas.KillSwitch:
    return schemas.KillSwitch(
        activated=model.activated,
        reason=model.reason,
        updated_at=model.updated_at,
    )


def _get_kill_switch(db: Session) -> models.KillSwitchModel:
    kill_switch = db.get(models.KillSwitchModel, 1)
    if kill_switch is None:
        kill_switch = models.KillSwitchModel(kill_switch_id=1, activated=False)
        db.add(kill_switch)
        db.commit()
        db.refresh(kill_switch)
    return kill_switch


def _validate_profile(db: Session, profile: str) -> models.TradingProfileModel:
    ensure_default_profiles(db)
    model = db.get(models.TradingProfileModel, profile)
    if model is None:
        raise HTTPException(status_code=404, detail="Profile not found")
    return model


def _get_balance_for_profile(
    db: Session, profile: str, currency: str = "USDT"
) -> models.BalanceSnapshotModel | None:
    return (
        db.query(models.BalanceSnapshotModel)
        .filter(
            models.BalanceSnapshotModel.profile == profile,
            models.BalanceSnapshotModel.currency == currency,
        )
        .first()
    )


def _get_positions_for_profile(
    db: Session, profile: str
) -> list[models.PositionSnapshotModel]:
    return (
        db.query(models.PositionSnapshotModel)
        .filter(models.PositionSnapshotModel.profile == profile)
        .all()
    )


def _get_open_orders_for_profile(db: Session, profile: str) -> int:
    return (
        db.query(models.OrderRecordModel)
        .filter(
            models.OrderRecordModel.profile == profile,
            models.OrderRecordModel.status.in_(
                [
                    enum_value(OrderLifecycleState.APPROVED),
                    enum_value(OrderLifecycleState.SUBMITTED),
                    enum_value(OrderLifecycleState.PARTIALLY_FILLED),
                ]
            ),
        )
        .count()
    )


def _mark_price_from_order(order: schemas.OrderCreate) -> float | None:
    if order.price is not None:
        return order.price
    mark_price = order.metadata.get("mark_price")
    if mark_price is None:
        return None
    return float(mark_price)


def _upsert_heartbeat(
    db: Session, heartbeat: schemas.HeartbeatCreate
) -> models.ServiceHeartbeatModel:
    model = (
        db.query(models.ServiceHeartbeatModel)
        .filter(
            models.ServiceHeartbeatModel.service_name == heartbeat.service_name,
            models.ServiceHeartbeatModel.instance_id == heartbeat.instance_id,
        )
        .first()
    )
    if model is None:
        model = models.ServiceHeartbeatModel(
            service_name=heartbeat.service_name,
            instance_id=heartbeat.instance_id,
            profile=heartbeat.profile,
        )
        db.add(model)
    model.profile = heartbeat.profile
    model.status = heartbeat.status
    model.metadata_json = heartbeat.metadata
    model.last_seen_at = heartbeat.last_seen_at
    db.commit()
    db.refresh(model)
    return model


def _upsert_balance(
    db: Session, balance: schemas.Balance
) -> models.BalanceSnapshotModel:
    model = (
        db.query(models.BalanceSnapshotModel)
        .filter(
            models.BalanceSnapshotModel.profile == balance.profile,
            models.BalanceSnapshotModel.currency == balance.currency,
        )
        .first()
    )
    if model is None:
        model = models.BalanceSnapshotModel(
            profile=balance.profile,
            currency=balance.currency,
        )
        db.add(model)
    model.available = balance.available
    model.cash_balance = balance.cash_balance
    model.equity = balance.equity
    model.updated_at = balance.updated_at
    db.commit()
    db.refresh(model)
    return model


def _upsert_position(
    db: Session, position: schemas.Position
) -> models.PositionSnapshotModel:
    model = (
        db.query(models.PositionSnapshotModel)
        .filter(
            models.PositionSnapshotModel.profile == position.profile,
            models.PositionSnapshotModel.inst_id == position.inst_id,
        )
        .first()
    )
    if model is None:
        model = models.PositionSnapshotModel(
            profile=position.profile,
            inst_id=position.inst_id,
        )
        db.add(model)
    model.instrument_kind = position.instrument_kind
    model.quantity = position.quantity
    model.avg_price = position.avg_price
    model.unrealized_pnl = position.unrealized_pnl
    model.td_mode = position.td_mode
    model.updated_at = position.updated_at
    db.commit()
    db.refresh(model)
    return model


@app.post("/runs", response_model=schemas.Run)
def create_run(
    run: schemas.RunCreate,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    db_run = models.Run(**run.model_dump(), status="PENDING")
    db.add(db_run)
    db.commit()
    db.refresh(db_run)
    return db_run


@app.get("/runs", response_model=List[schemas.Run])
def read_runs(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    return db.query(models.Run).offset(skip).limit(limit).all()


@app.get("/runs/{run_id}", response_model=schemas.Run)
def read_run(
    run_id: uuid.UUID,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    db_run = db.query(models.Run).filter(models.Run.run_id == run_id).first()
    if db_run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return db_run


@app.put("/runs/{run_id}", response_model=schemas.Run)
def update_run(
    run_id: uuid.UUID,
    run_update: schemas.RunUpdate,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    db_run = db.query(models.Run).filter(models.Run.run_id == run_id).first()
    if db_run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    for key, value in run_update.model_dump(exclude_unset=True).items():
        setattr(db_run, key, value)

    db.commit()
    db.refresh(db_run)
    return db_run


@app.get("/profiles", response_model=list[schemas.Profile])
def list_profiles(
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    ensure_default_profiles(db)
    profiles = db.query(models.TradingProfileModel).all()
    return [_profile_schema(profile) for profile in profiles]


@app.post("/profiles", response_model=schemas.Profile)
def create_profile(
    profile: schemas.ProfileCreate,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    profile_value = enum_value(profile.profile)
    existing = db.get(models.TradingProfileModel, profile_value)
    if existing is not None:
        raise HTTPException(status_code=409, detail="Profile already exists")
    model = models.TradingProfileModel(
        profile=profile_value,
        name=profile.name,
        rest_base_url=profile.rest_base_url,
        public_ws_url=profile.public_ws_url,
        private_ws_url=profile.private_ws_url,
        credential_env_prefix=profile.credential_env_prefix,
        description=profile.description,
        is_simulated=profile.is_simulated,
        is_active=profile.is_active,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return _profile_schema(model)


@app.post("/instruments", response_model=schemas.Instrument)
def create_instrument(
    instrument: schemas.InstrumentCreate,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    profile_value = enum_value(instrument.profile)
    _validate_profile(db, profile_value)
    existing = (
        db.query(models.InstrumentModel)
        .filter(
            models.InstrumentModel.profile == profile_value,
            models.InstrumentModel.inst_id == instrument.inst_id,
        )
        .first()
    )
    if existing is not None:
        raise HTTPException(status_code=409, detail="Instrument already exists")
    model = models.InstrumentModel(
        inst_id=instrument.inst_id,
        profile=profile_value,
        kind=enum_value(instrument.kind),
        allow_trading=instrument.allow_trading,
        tick_size=instrument.tick_size,
        lot_size=instrument.lot_size,
        metadata_json=instrument.metadata,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return _instrument_schema(model)


@app.get("/instruments", response_model=list[schemas.Instrument])
def list_instruments(
    profile: str | None = None,
    kind: str | None = None,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    query = db.query(models.InstrumentModel)
    if profile:
        query = query.filter(models.InstrumentModel.profile == profile)
    if kind:
        query = query.filter(models.InstrumentModel.kind == kind)
    return [_instrument_schema(model) for model in query.all()]


@app.post("/bots", response_model=schemas.Bot)
def create_bot(
    bot: schemas.BotCreate,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    profile_value = enum_value(bot.profile)
    _validate_profile(db, profile_value)
    existing = (
        db.query(models.BotModel)
        .filter(models.BotModel.name == bot.name)
        .first()
    )
    if existing is not None:
        raise HTTPException(status_code=409, detail="Bot already exists")
    model = models.BotModel(
        name=bot.name,
        profile=profile_value,
        signal_provider=bot.signal_provider,
        status=enum_value(bot.status),
        instrument_ids=bot.instrument_ids,
        config_json=bot.config,
        created_at=bot.created_at,
        updated_at=bot.updated_at,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return _bot_schema(model)


@app.get("/bots", response_model=list[schemas.Bot])
def list_bots(
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    return [_bot_schema(model) for model in db.query(models.BotModel).all()]


@app.post("/bots/{bot_name}/enable", response_model=schemas.Bot)
def enable_bot(
    bot_name: str,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    model = db.query(models.BotModel).filter(models.BotModel.name == bot_name).first()
    if model is None:
        raise HTTPException(status_code=404, detail="Bot not found")
    model.status = enum_value(BotStatus.ENABLED)
    db.commit()
    db.refresh(model)
    return _bot_schema(model)


@app.post("/bots/{bot_name}/disable", response_model=schemas.Bot)
def disable_bot(
    bot_name: str,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    model = db.query(models.BotModel).filter(models.BotModel.name == bot_name).first()
    if model is None:
        raise HTTPException(status_code=404, detail="Bot not found")
    model.status = enum_value(BotStatus.DISABLED)
    db.commit()
    db.refresh(model)
    return _bot_schema(model)


@app.post("/bots/{bot_name}/deploy", response_model=schemas.Deployment)
def deploy_bot(
    bot_name: str,
    deployment: schemas.DeploymentCreate,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    bot = db.query(models.BotModel).filter(models.BotModel.name == bot_name).first()
    if bot is None:
        raise HTTPException(status_code=404, detail="Bot not found")
    record = models.DeploymentRecordModel(
        bot_name=bot_name,
        profile=deployment.profile,
        status=enum_value(DeploymentStatus.DEPLOYED),
        metadata_json=deployment.metadata,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return _deployment_schema(record)


@app.post("/orders", response_model=schemas.Order)
def create_order(
    order: schemas.OrderCreate,
    submit: bool = Query(default=False),
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    profile_value = enum_value(order.profile)
    _validate_profile(db, profile_value)
    kill_switch = _get_kill_switch(db)
    if kill_switch.activated:
        raise HTTPException(status_code=423, detail=kill_switch.reason or "kill switch")

    instrument = (
        db.query(models.InstrumentModel)
        .filter(
            models.InstrumentModel.profile == profile_value,
            models.InstrumentModel.inst_id == order.inst_id,
        )
        .first()
    )
    if instrument is None or not instrument.allow_trading:
        raise HTTPException(status_code=400, detail="Instrument not allowlisted")

    balance_model = _get_balance_for_profile(db, profile_value)
    positions = _get_positions_for_profile(db, profile_value)
    decision = risk_service.evaluate_order(
        order,
        balance=_balance_schema(balance_model) if balance_model else None,
        positions=[_position_schema(position) for position in positions],
        open_orders_count=_get_open_orders_for_profile(db, profile_value),
        mark_price=_mark_price_from_order(order),
        daily_realized_pnl=float(order.metadata.get("daily_realized_pnl", 0.0)),
        consecutive_errors=int(order.metadata.get("consecutive_errors", 0)),
        market_data_fresh=bool(order.metadata.get("market_data_fresh", True)),
    )
    status = (
        enum_value(OrderLifecycleState.APPROVED)
        if decision.approved
        else enum_value(OrderLifecycleState.REJECTED)
    )
    rejection_reason = decision.reason
    raw_payload = {"risk": decision.model_dump(mode="json")}

    if submit and decision.approved:
        try:
            state = execution_service.submit_order(order)
            status = enum_value(state.status)
            rejection_reason = state.rejection_reason
            raw_payload = state.raw_payload
        except ValueError as exc:
            status = enum_value(OrderLifecycleState.FAILED)
            rejection_reason = str(exc)
        except Exception as exc:  # pragma: no cover - 保留實盤故障訊息
            status = enum_value(OrderLifecycleState.FAILED)
            rejection_reason = str(exc)

    record = models.OrderRecordModel(
        order_id=order.order_id,
        client_order_id=order.client_order_id,
        profile=profile_value,
        inst_id=order.inst_id,
        instrument_kind=enum_value(order.instrument_kind),
        side=enum_value(order.side),
        size=order.size,
        price=order.price,
        order_type=enum_value(order.order_type),
        td_mode=enum_value(order.td_mode),
        status=status,
        bot_name=order.bot_name,
        source=order.source,
        rejection_reason=rejection_reason,
        raw_payload=raw_payload,
        created_at=order.created_at,
        updated_at=order.created_at,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return _order_schema(record)


@app.get("/orders", response_model=list[schemas.Order])
def list_orders(
    profile: str | None = None,
    status: str | None = None,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    query = db.query(models.OrderRecordModel)
    if profile:
        query = query.filter(models.OrderRecordModel.profile == profile)
    if status:
        query = query.filter(models.OrderRecordModel.status == status)
    query = query.order_by(models.OrderRecordModel.created_at.desc())
    return [_order_schema(model) for model in query.all()]


@app.post("/orders/cancel", response_model=schemas.Order)
def cancel_order(
    request: schemas.CancelOrderRequest,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    query = db.query(models.OrderRecordModel).filter(
        models.OrderRecordModel.profile == request.profile,
        models.OrderRecordModel.inst_id == request.inst_id,
    )
    if request.order_id:
        query = query.filter(models.OrderRecordModel.order_id == request.order_id)
    elif request.client_order_id:
        query = query.filter(
            models.OrderRecordModel.client_order_id == request.client_order_id
        )
    else:
        raise HTTPException(status_code=400, detail="Missing order identifier")

    model = query.first()
    if model is None:
        raise HTTPException(status_code=404, detail="Order not found")

    model.status = enum_value(OrderLifecycleState.CANCELED)
    model.updated_at = models.dt.datetime.utcnow()
    db.commit()
    db.refresh(model)
    return _order_schema(model)


@app.get("/positions", response_model=list[schemas.Position])
def list_positions(
    profile: str | None = None,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    query = db.query(models.PositionSnapshotModel)
    if profile:
        query = query.filter(models.PositionSnapshotModel.profile == profile)
    return [_position_schema(model) for model in query.all()]


@app.post("/positions", response_model=schemas.Position)
def upsert_position(
    position: schemas.Position,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    _validate_profile(db, enum_value(position.profile))
    return _position_schema(_upsert_position(db, position))


@app.get("/balances", response_model=list[schemas.Balance])
def list_balances(
    profile: str | None = None,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    query = db.query(models.BalanceSnapshotModel)
    if profile:
        query = query.filter(models.BalanceSnapshotModel.profile == profile)
    return [_balance_schema(model) for model in query.all()]


@app.post("/balances", response_model=schemas.Balance)
def upsert_balance(
    balance: schemas.Balance,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    _validate_profile(db, enum_value(balance.profile))
    return _balance_schema(_upsert_balance(db, balance))


@app.get("/services", response_model=list[schemas.Heartbeat])
def list_services(
    profile: str | None = None,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    query = db.query(models.ServiceHeartbeatModel)
    if profile:
        query = query.filter(models.ServiceHeartbeatModel.profile == profile)
    return [_heartbeat_schema(model) for model in query.all()]


@app.post("/services", response_model=schemas.Heartbeat)
def post_service_heartbeat(
    heartbeat: schemas.HeartbeatCreate,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    _validate_profile(db, enum_value(heartbeat.profile))
    return _heartbeat_schema(_upsert_heartbeat(db, heartbeat))


@app.get("/kill-switch", response_model=schemas.KillSwitch)
def get_kill_switch(
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    return _kill_switch_schema(_get_kill_switch(db))


@app.put("/kill-switch", response_model=schemas.KillSwitch)
def update_kill_switch(
    update: schemas.KillSwitchUpdate,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key),
):
    del api_key
    model = _get_kill_switch(db)
    model.activated = update.activated
    model.reason = update.reason
    model.updated_at = models.dt.datetime.utcnow()
    if update.activated:
        risk_service.activate_kill_switch(update.reason or "manual stop")
    else:
        risk_service.release_kill_switch()
    db.commit()
    db.refresh(model)
    return _kill_switch_schema(model)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {
        "status": "ok",
        "auto_submit": os.getenv("CONTROL_API_AUTO_SUBMIT", "0"),
    }
