from __future__ import annotations

from fastapi import FastAPI

from okx_trading_platform.adapters.okx import (
    ClientOrderIdCache,
    OkxExchangeGateway,
    OkxRestClient,
    OkxWebSocketRouter,
)
from okx_trading_platform.shared.runtime import ExecutionRuntime
from okx_trading_platform.shared.settings import get_platform_settings

settings = get_platform_settings()
gateway = OkxExchangeGateway(
    rest_client=OkxRestClient(),
    router=OkxWebSocketRouter(),
    dedupe_cache=ClientOrderIdCache(),
)
runtime = ExecutionRuntime(
    service_name="execution-service",
    profile_id=settings.baseline_profile_id,
    environment=settings.trading_environment,
    gateway=gateway,
)
runtime.set_running()
app = FastAPI(title="OKX Execution Service")


@app.get("/healthz")
def healthz() -> dict:
    return {
        "service": runtime.service_name,
        "status": runtime.status,
        "profile_id": runtime.profile_id,
        "environment": runtime.environment,
        "ws_public": gateway.rest_client.websocket_url(
            environment=runtime.environment,
            private=False,
        ),
        "ws_private": gateway.rest_client.websocket_url(
            environment=runtime.environment,
            private=True,
        ),
    }


@app.get("/heartbeat")
def heartbeat() -> dict:
    return runtime.heartbeat().model_dump(mode="json")
