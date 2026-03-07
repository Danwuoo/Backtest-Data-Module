from __future__ import annotations

from fastapi import FastAPI

from okx_trading_platform.domain.risk import RiskManager
from okx_trading_platform.shared.runtime import RiskRuntime
from okx_trading_platform.shared.settings import get_platform_settings

settings = get_platform_settings()
risk_manager = RiskManager()
runtime = RiskRuntime(
    service_name="risk-service",
    profile_id=settings.baseline_profile_id,
    environment=settings.trading_environment,
    risk_manager=risk_manager,
)
runtime.set_running()
app = FastAPI(title="OKX Risk Service")


@app.get("/healthz")
def healthz() -> dict:
    return {
        "service": runtime.service_name,
        "status": runtime.status,
        "profile_id": runtime.profile_id,
        "environment": runtime.environment,
        "limits": runtime.risk_manager.limits.__dict__,
    }


@app.get("/heartbeat")
def heartbeat() -> dict:
    return runtime.heartbeat().model_dump(mode="json")
