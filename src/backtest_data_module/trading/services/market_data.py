from __future__ import annotations

import os

from fastapi import FastAPI

from backtest_data_module.trading.domain import TradingProfile
from backtest_data_module.trading.runtime import MarketDataRuntime

runtime = MarketDataRuntime(
    service_name="market-data-service",
    profile=os.getenv("TRADING_PROFILE", TradingProfile.DEMO),
)
runtime.set_running()
app = FastAPI()


@app.get("/healthz")
def healthz() -> dict:
    return {
        "service": runtime.service_name,
        "status": runtime.status,
        "profile": runtime.profile,
        "books": len(runtime.books),
    }


@app.get("/heartbeat")
def heartbeat() -> dict:
    return runtime.heartbeat().model_dump(mode="json")
