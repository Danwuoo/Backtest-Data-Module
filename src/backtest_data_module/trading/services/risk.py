from __future__ import annotations

import os

from fastapi import FastAPI

from backtest_data_module.trading.domain import TradingProfile
from backtest_data_module.trading.risk import RiskService
from backtest_data_module.trading.runtime import RiskRuntime

runtime = RiskRuntime(
    service_name="risk-service",
    profile=os.getenv("TRADING_PROFILE", TradingProfile.DEMO),
    risk_service=RiskService(),
)
runtime.set_running()
app = FastAPI()


@app.get("/healthz")
def healthz() -> dict:
    return {
        "service": runtime.service_name,
        "status": runtime.status,
        "profile": runtime.profile,
        "kill_switch": runtime.risk_service.kill_switch.model_dump(mode="json"),
    }


@app.get("/heartbeat")
def heartbeat() -> dict:
    return runtime.heartbeat().model_dump(mode="json")
