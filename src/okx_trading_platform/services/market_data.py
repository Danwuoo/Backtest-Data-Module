from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI

from okx_trading_platform.adapters.okx import build_market_data_ingestion_plan
from okx_trading_platform.shared.data_lake import DataLakeWriter
from okx_trading_platform.shared.runtime import MarketDataRuntime
from okx_trading_platform.shared.settings import get_platform_settings

settings = get_platform_settings()
ingestion_plan = build_market_data_ingestion_plan(settings)
runtime = MarketDataRuntime(
    service_name="market-data-service",
    profile_id=settings.baseline_profile_id,
    environment=settings.trading_environment,
    data_lake=DataLakeWriter(settings.platform_data_root, settings.duckdb_path),
)
runtime.set_running()
app = FastAPI(title="OKX Market Data Service")


@app.get("/healthz")
def healthz() -> dict:
    return {
        "service": runtime.service_name,
        "status": runtime.status,
        "profile_id": runtime.profile_id,
        "environment": runtime.environment,
        "order_books": len(runtime.books),
        "whitelist_source": ingestion_plan.whitelist_source,
        "whitelist_symbols": len(ingestion_plan.whitelist),
    }


@app.get("/heartbeat")
def heartbeat() -> dict:
    return runtime.heartbeat().model_dump(mode="json")


@app.get("/ingestion-plan")
def get_ingestion_plan() -> dict:
    return asdict(ingestion_plan)
