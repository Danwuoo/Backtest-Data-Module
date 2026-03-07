from __future__ import annotations

from fastapi import FastAPI

from okx_trading_platform.shared.runtime import ExecutionPolicyRuntime
from okx_trading_platform.shared.settings import get_platform_settings

settings = get_platform_settings()
runtime = ExecutionPolicyRuntime(
    service_name="execution-policy-service",
    profile_id=settings.baseline_profile_id,
    environment=settings.trading_environment,
)
runtime.set_running()
app = FastAPI(title="OKX Execution Policy Service")


@app.get("/healthz")
def healthz() -> dict:
    return {
        "service": runtime.service_name,
        "status": runtime.status,
        "profile_id": runtime.profile_id,
        "environment": runtime.environment,
    }


@app.get("/heartbeat")
def heartbeat() -> dict:
    return runtime.heartbeat().model_dump(mode="json")
