from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Security
from sqlalchemy.orm import Session

from okx_trading_platform.application import (
    CancelOrderCommand,
    ControlPlaneError,
    ControlPlaneService,
    PlatformRepository,
)
from okx_trading_platform.application.persistence import Base
from okx_trading_platform.domain import TradingProfile
from okx_trading_platform.domain.risk import RiskManager
from okx_trading_platform.shared.auth import get_api_key
from okx_trading_platform.shared.db import engine, get_db

from . import schemas


def get_risk_manager() -> RiskManager:
    return RiskManager()


def get_control_plane(db: Session = Depends(get_db)) -> ControlPlaneService:
    return ControlPlaneService(
        PlatformRepository(db),
        risk_manager=get_risk_manager(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    Base.metadata.create_all(bind=engine)
    db = next(get_db())
    try:
        get_control_plane(db).bootstrap_profiles()
    finally:
        db.close()
    yield


app = FastAPI(title="OKX Trading Platform", lifespan=lifespan)


def _handle_error(exc: ControlPlaneError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=exc.detail)


@app.get("/profiles", response_model=list[schemas.Profile])
def list_profiles(
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_profiles()


@app.post("/profiles", response_model=schemas.Profile)
def create_profile(
    profile: schemas.ProfileCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    try:
        return service.create_profile(profile)
    except ControlPlaneError as exc:
        _handle_error(exc)


@app.get("/instruments", response_model=list[schemas.Instrument])
def list_instruments(
    profile: str | None = None,
    kind: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_instruments(profile=profile, kind=kind)


@app.post("/instruments", response_model=schemas.Instrument)
def create_instrument(
    instrument: schemas.InstrumentCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    try:
        return service.create_instrument(instrument)
    except ControlPlaneError as exc:
        _handle_error(exc)


@app.get("/bots", response_model=list[schemas.Bot])
def list_bots(
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_bots()


@app.post("/bots", response_model=schemas.Bot)
def create_bot(
    bot: schemas.BotCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    try:
        return service.create_bot(bot)
    except ControlPlaneError as exc:
        _handle_error(exc)


@app.post("/bots/{bot_name}/enable", response_model=schemas.Bot)
def enable_bot(
    bot_name: str,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    try:
        return service.enable_bot(bot_name)
    except ControlPlaneError as exc:
        _handle_error(exc)


@app.post("/bots/{bot_name}/disable", response_model=schemas.Bot)
def disable_bot(
    bot_name: str,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    try:
        return service.disable_bot(bot_name)
    except ControlPlaneError as exc:
        _handle_error(exc)


@app.post("/bots/{bot_name}/deploy", response_model=schemas.Deployment)
def deploy_bot(
    bot_name: str,
    deployment: schemas.DeploymentCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    try:
        return service.deploy_bot(
            bot_name=bot_name,
            profile=TradingProfile(deployment.profile),
            metadata=deployment.metadata,
        )
    except ControlPlaneError as exc:
        _handle_error(exc)


@app.get("/orders", response_model=list[schemas.Order])
def list_orders(
    profile: str | None = None,
    status: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_orders(profile=profile, status=status)


@app.post("/orders", response_model=schemas.Order)
def create_order(
    order: schemas.OrderCreate,
    submit: bool = Query(default=False),
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    try:
        return service.create_order(order, submit=submit)
    except ControlPlaneError as exc:
        _handle_error(exc)


@app.post("/orders/cancel", response_model=schemas.Order)
def cancel_order(
    request: schemas.CancelOrderRequest,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    try:
        return service.cancel_order(
            CancelOrderCommand(
                profile=request.profile,
                inst_id=request.inst_id,
                order_id=request.order_id,
                client_order_id=request.client_order_id,
            )
        )
    except ControlPlaneError as exc:
        _handle_error(exc)


@app.get("/positions", response_model=list[schemas.Position])
def list_positions(
    profile: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_positions(profile=profile)


@app.post("/positions", response_model=schemas.Position)
def upsert_position(
    position: schemas.Position,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    try:
        return service.upsert_position(position)
    except ControlPlaneError as exc:
        _handle_error(exc)


@app.get("/balances", response_model=list[schemas.Balance])
def list_balances(
    profile: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_balances(profile=profile)


@app.post("/balances", response_model=schemas.Balance)
def upsert_balance(
    balance: schemas.Balance,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    try:
        return service.upsert_balance(balance)
    except ControlPlaneError as exc:
        _handle_error(exc)


@app.get("/services", response_model=list[schemas.Heartbeat])
def list_services(
    profile: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_services(profile=profile)


@app.post("/services", response_model=schemas.Heartbeat)
def upsert_service_heartbeat(
    heartbeat: schemas.HeartbeatCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    try:
        return service.upsert_service(heartbeat)
    except ControlPlaneError as exc:
        _handle_error(exc)


@app.get("/kill-switch", response_model=schemas.KillSwitch)
def get_kill_switch(
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.get_kill_switch()


@app.put("/kill-switch", response_model=schemas.KillSwitch)
def update_kill_switch(
    update: schemas.KillSwitchUpdate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.update_kill_switch(
        activated=update.activated,
        reason=update.reason,
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "control-api",
        "auto_submit": os.getenv("CONTROL_API_AUTO_SUBMIT", "0"),
    }
