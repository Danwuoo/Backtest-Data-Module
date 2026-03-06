from __future__ import annotations

from fastapi import FastAPI

from okx_trading_platform.application import ReferenceBreakoutSignalProvider
from okx_trading_platform.shared.runtime import StrategyRunnerRuntime
from okx_trading_platform.shared.settings import get_platform_settings

settings = get_platform_settings()
runtime = StrategyRunnerRuntime(
    service_name="strategy-runner",
    profile=settings.trading_profile,
    signal_provider=ReferenceBreakoutSignalProvider(
        bot_name=settings.reference_bot_name,
        profile=settings.trading_profile,
        inst_id=settings.reference_inst_id,
        instrument_kind=settings.reference_instrument_kind,
        trigger_spread=settings.reference_trigger_spread,
        size=settings.reference_order_size,
    ),
)
runtime.set_running()
app = FastAPI(title="OKX Strategy Runner")


@app.get("/healthz")
def healthz() -> dict:
    return {
        "service": runtime.service_name,
        "status": runtime.status,
        "profile": runtime.profile,
        "provider": type(runtime.signal_provider).__name__,
    }


@app.get("/heartbeat")
def heartbeat() -> dict:
    return runtime.heartbeat().model_dump(mode="json")
