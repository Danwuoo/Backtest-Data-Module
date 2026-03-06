from __future__ import annotations

import os

from fastapi import FastAPI

from backtest_data_module.trading.domain import InstrumentKind, TradingProfile
from backtest_data_module.trading.runtime import StrategyRunnerRuntime
from backtest_data_module.trading.signals import ReferenceBreakoutSignalProvider

profile = os.getenv("TRADING_PROFILE", TradingProfile.DEMO)
runtime = StrategyRunnerRuntime(
    service_name="strategy-runner",
    profile=profile,
    signal_provider=ReferenceBreakoutSignalProvider(
        bot_name=os.getenv("REFERENCE_BOT_NAME", "reference-breakout"),
        profile=profile,
        inst_id=os.getenv("REFERENCE_INST_ID", "BTC-USDT-SWAP"),
        instrument_kind=os.getenv("REFERENCE_INSTRUMENT_KIND", InstrumentKind.SWAP),
        trigger_spread=float(os.getenv("REFERENCE_TRIGGER_SPREAD", "0.002")),
        size=float(os.getenv("REFERENCE_ORDER_SIZE", "1")),
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
        "provider": runtime.signal_provider.__class__.__name__,
    }


@app.get("/heartbeat")
def heartbeat() -> dict:
    return runtime.heartbeat().model_dump(mode="json")
