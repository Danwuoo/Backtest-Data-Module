from __future__ import annotations

import os

from fastapi import FastAPI

from backtest_data_module.trading.domain import TradingProfile
from backtest_data_module.trading.okx import (
    ClientOrderIdCache,
    OkxExecutionService,
    OkxRestClient,
    OkxWebSocketRouter,
)
from backtest_data_module.trading.runtime import ExecutionRuntime

runtime = ExecutionRuntime(
    service_name="execution-service",
    profile=os.getenv("TRADING_PROFILE", TradingProfile.DEMO),
    execution_service=OkxExecutionService(
        rest_client=OkxRestClient(),
        router=OkxWebSocketRouter(),
        dedupe_cache=ClientOrderIdCache(),
    ),
)
runtime.set_running()
app = FastAPI()


@app.get("/healthz")
def healthz() -> dict:
    return {
        "service": runtime.service_name,
        "status": runtime.status,
        "profile": runtime.profile,
        "ws_public": runtime.execution_service.rest_client.websocket_url(
            profile=runtime.profile,
            private=False,
        ),
        "ws_private": runtime.execution_service.rest_client.websocket_url(
            profile=runtime.profile,
            private=True,
        ),
    }


@app.get("/heartbeat")
def heartbeat() -> dict:
    return runtime.heartbeat().model_dump(mode="json")
