from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Security
from sqlalchemy.orm import Session

from okx_trading_platform.application import (
    AuditEventPipeline,
    CancelOrderCommand,
    ControlPlaneError,
    ControlPlaneService,
    PlatformRepository,
    ReadModelProjector,
)
from okx_trading_platform.application.persistence import Base
from okx_trading_platform.domain.risk import RiskManager
from okx_trading_platform.shared.auth import get_api_key
from okx_trading_platform.shared.data_lake import DataLakeWriter
from okx_trading_platform.shared.db import engine, get_db
from okx_trading_platform.shared.settings import get_platform_settings

from . import schemas

settings = get_platform_settings()
lake_writer = DataLakeWriter(
    settings.platform_data_root,
    settings.duckdb_path,
    hot_cache_root=settings.hot_cache_root,
    object_store_backend=settings.object_store_backend,
    object_store_bucket=settings.object_store_bucket,
    object_store_endpoint=settings.object_store_endpoint,
    object_store_region=settings.object_store_region,
    object_store_access_key_id=settings.object_store_access_key_id,
    object_store_secret_access_key=settings.object_store_secret_access_key,
    object_store_prefix=settings.object_store_prefix,
    checkpoint_root=settings.checkpoint_root,
)


def get_risk_manager() -> RiskManager:
    return RiskManager()


def get_control_plane(db: Session = Depends(get_db)) -> ControlPlaneService:
    repository = PlatformRepository(db)
    return ControlPlaneService(
        repository,
        risk_manager=get_risk_manager(),
        audit_pipeline=AuditEventPipeline(
            lake=lake_writer,
            projector=ReadModelProjector(repository),
            stream=f"{settings.redpanda_topic_prefix}.control-plane",
            source_service="control-api",
        ),
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


app = FastAPI(title="OKX Trading Platform V2", lifespan=lifespan)


def _handle_error(exc: ControlPlaneError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=exc.detail)


def _call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except ControlPlaneError as exc:
        _handle_error(exc)


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
    return _call(service.create_profile, profile)


@app.get("/risk-policies", response_model=list[schemas.RiskPolicy])
def list_risk_policies(
    profile_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_risk_policies(profile_id=profile_id)


@app.post("/risk-policies", response_model=schemas.RiskPolicy)
def create_risk_policy(
    policy: schemas.RiskPolicyCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return _call(service.create_risk_policy, policy)


@app.get("/allocators", response_model=list[schemas.Allocator])
def list_allocators(
    profile_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_allocators(profile_id=profile_id)


@app.post("/allocators", response_model=schemas.Allocator)
def create_allocator(
    allocator: schemas.AllocatorCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return _call(service.create_allocator, allocator)


@app.get("/sleeves", response_model=list[schemas.Sleeve])
def list_sleeves(
    profile_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_sleeves(profile_id=profile_id)


@app.post("/sleeves", response_model=schemas.Sleeve)
def create_sleeve(
    sleeve: schemas.SleeveCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return _call(service.create_sleeve, sleeve)


@app.get("/instruments", response_model=list[schemas.Instrument])
def list_instruments(
    profile_id: str | None = None,
    kind: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_instruments(profile_id=profile_id, kind=kind)


@app.post("/instruments", response_model=schemas.Instrument)
def create_instrument(
    instrument: schemas.InstrumentCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return _call(service.create_instrument, instrument)


@app.get("/strategies", response_model=list[schemas.Strategy])
def list_strategies(
    profile_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_strategies(profile_id=profile_id)


@app.post("/strategies", response_model=schemas.Strategy)
def create_strategy(
    strategy: schemas.StrategyCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return _call(service.create_strategy, strategy)


@app.get("/models", response_model=list[schemas.Model])
def list_models(
    profile_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_model_versions(profile_id=profile_id)


@app.post("/models", response_model=schemas.Model)
def create_model(
    version: schemas.ModelVersionCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return _call(service.create_model_version, version)


@app.get("/datasets", response_model=list[schemas.Dataset])
def list_datasets(
    profile_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_datasets(profile_id=profile_id)


@app.post("/datasets", response_model=schemas.Dataset)
def create_dataset(
    dataset: schemas.DatasetCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return _call(service.create_dataset, dataset)


@app.get("/features", response_model=list[schemas.Feature])
def list_features(
    profile_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_features(profile_id=profile_id)


@app.post("/features", response_model=schemas.Feature)
def create_feature(
    feature: schemas.FeatureCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return _call(service.create_feature, feature)


@app.get("/dataset-versions", response_model=list[schemas.DatasetVersionSchema])
def list_dataset_versions(
    profile_id: str | None = None,
    dataset_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_dataset_versions(profile_id=profile_id, dataset_id=dataset_id)


@app.post("/dataset-versions", response_model=schemas.DatasetVersionSchema)
def create_dataset_version(
    version: schemas.DatasetVersionCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return _call(service.create_dataset_version, version)


@app.get("/run-artifacts", response_model=list[schemas.RunArtifactSchema])
def list_run_artifacts(
    profile_id: str | None = None,
    run_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_run_artifacts(profile_id=profile_id, run_id=run_id)


@app.post("/run-artifacts", response_model=schemas.RunArtifactSchema)
def create_run_artifact(
    artifact: schemas.RunArtifactCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return _call(service.create_run_artifact, artifact)


@app.get("/backtests", response_model=list[schemas.Backtest])
def list_backtests(
    profile_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_backtests(profile_id=profile_id)


@app.post("/backtests", response_model=schemas.Backtest)
def create_backtest(
    run: schemas.BacktestCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return _call(service.create_backtest, run)


@app.get("/paper-runs", response_model=list[schemas.PaperRunSchema])
def list_paper_runs(
    profile_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_paper_runs(profile_id=profile_id)


@app.post("/paper-runs", response_model=schemas.PaperRunSchema)
def create_paper_run(
    run: schemas.PaperRunCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return _call(service.create_paper_run, run)


@app.get("/live-runs", response_model=list[schemas.LiveRunSchema])
def list_live_runs(
    profile_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_live_runs(profile_id=profile_id)


@app.post("/live-runs", response_model=schemas.LiveRunSchema)
def create_live_run(
    run: schemas.LiveRunCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return _call(service.create_live_run, run)


@app.get("/orders", response_model=list[schemas.Order])
def list_orders(
    profile_id: str | None = None,
    status: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_orders(profile_id=profile_id, status=status)


@app.post("/orders", response_model=schemas.Order)
def create_order(
    order: schemas.OrderCreate,
    submit: bool = Query(default=False),
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return _call(service.create_order, order, submit=submit)


@app.post("/orders/cancel", response_model=schemas.Order)
def cancel_order(
    request: schemas.CancelOrderRequest,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return _call(service.cancel_order, CancelOrderCommand(order_id=request.order_id))


@app.get("/fills", response_model=list[schemas.Fill])
def list_fills(
    profile_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_fills(profile_id=profile_id)


@app.post("/fills", response_model=schemas.Fill)
def create_fill(
    fill: schemas.FillCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return _call(service.create_fill, fill)


@app.get("/ledger")
def list_ledger(
    profile_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_ledger(profile_id=profile_id)


@app.get("/funding")
def list_funding(
    profile_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_funding(profile_id=profile_id)


@app.get("/pnl")
def list_pnl(
    profile_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_pnl(profile_id=profile_id)


@app.get("/risk-snapshots")
def list_risk_snapshots(
    profile_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_risk_snapshots(profile_id=profile_id)


@app.get("/execution-snapshots")
def list_execution_snapshots(
    profile_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_execution_snapshots(profile_id=profile_id)


@app.get("/positions", response_model=list[schemas.Position])
def list_positions(
    profile_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_positions(profile_id=profile_id)


@app.post("/positions", response_model=schemas.Position)
def upsert_position(
    position: schemas.Position,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return _call(service.upsert_position, position)


@app.get("/balances", response_model=list[schemas.Balance])
def list_balances(
    profile_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_balances(profile_id=profile_id)


@app.post("/balances", response_model=schemas.Balance)
def upsert_balance(
    balance: schemas.Balance,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return _call(service.upsert_balance, balance)


@app.get("/services", response_model=list[schemas.Heartbeat])
def list_services(
    profile_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_services(profile_id=profile_id)


@app.post("/services", response_model=schemas.Heartbeat)
def upsert_service_heartbeat(
    heartbeat: schemas.HeartbeatCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return _call(service.upsert_service, heartbeat)


@app.get("/incidents", response_model=list[schemas.Incident])
def list_incidents(
    profile_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_incidents(profile_id=profile_id)


@app.post("/incidents", response_model=schemas.Incident)
def create_incident(
    incident: schemas.IncidentCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return _call(service.create_incident, incident)


@app.get("/alert-policies", response_model=list[schemas.AlertPolicySchema])
def list_alert_policies(
    profile_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_alert_policies(profile_id=profile_id)


@app.post("/alert-policies", response_model=schemas.AlertPolicySchema)
def create_alert_policy(
    policy: schemas.AlertPolicyCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return _call(service.create_alert_policy, policy)


@app.get("/alerts", response_model=list[schemas.Alert])
def list_alerts(
    profile_id: str | None = None,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return service.list_alerts(profile_id=profile_id)


@app.post("/alerts", response_model=schemas.Alert)
def create_alert(
    alert: schemas.AlertCreate,
    service: ControlPlaneService = Depends(get_control_plane),
    api_key: str | None = Security(get_api_key),
):
    del api_key
    return _call(service.create_alert, alert)


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
