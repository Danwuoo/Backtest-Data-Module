from __future__ import annotations

from fastapi import FastAPI

from okx_trading_platform.application import RuleBaselineInferenceProvider
from okx_trading_platform.shared.runtime import InferenceRuntime
from okx_trading_platform.shared.settings import get_platform_settings

settings = get_platform_settings()
runtime = InferenceRuntime(
    service_name="model-inference-service",
    profile_id=settings.baseline_profile_id,
    environment=settings.trading_environment,
    provider=RuleBaselineInferenceProvider(
        profile_id=settings.baseline_profile_id,
        strategy_id=settings.baseline_strategy_id,
        model_version_id=settings.baseline_model_version_id,
        instrument_id=settings.baseline_instrument_id,
        inst_id=settings.baseline_inst_id,
        kind=settings.baseline_instrument_kind,
        threshold_bps=settings.baseline_threshold_bps,
        target_size=settings.baseline_target_size,
    ),
)
runtime.set_running()
app = FastAPI(title="OKX Model Inference Service")


@app.get("/healthz")
def healthz() -> dict:
    return {
        "service": runtime.service_name,
        "status": runtime.status,
        "profile_id": runtime.profile_id,
        "environment": runtime.environment,
        "provider": type(runtime.provider).__name__,
    }


@app.get("/heartbeat")
def heartbeat() -> dict:
    return runtime.heartbeat().model_dump(mode="json")
