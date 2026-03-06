from __future__ import annotations

from fastapi import FastAPI

from okx_trading_platform.shared.runtime import MarketDataRuntime
from okx_trading_platform.shared.settings import get_platform_settings

settings = get_platform_settings()
runtime = MarketDataRuntime(
    service_name="market-data-service",
    profile=settings.trading_profile,
)
runtime.set_running()
app = FastAPI(title="OKX Market Data Service")


@app.get("/healthz")
def healthz() -> dict:
    return {
        "service": runtime.service_name,
        "status": runtime.status,
        "profile": runtime.profile,
        "order_books": len(runtime.books),
    }


@app.get("/heartbeat")
def heartbeat() -> dict:
    return runtime.heartbeat().model_dump(mode="json")
