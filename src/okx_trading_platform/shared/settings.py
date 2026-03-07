from __future__ import annotations

import os
from dataclasses import dataclass

from okx_trading_platform.domain import InstrumentKind, TradingEnvironment


def parse_environment(value: str | TradingEnvironment | None) -> TradingEnvironment:
    if isinstance(value, TradingEnvironment):
        return value
    if value is None:
        return TradingEnvironment.DEMO
    return TradingEnvironment(str(value).lower())


def parse_instrument_kind(value: str | InstrumentKind | None) -> InstrumentKind:
    if isinstance(value, InstrumentKind):
        return value
    if value is None:
        return InstrumentKind.SWAP
    return InstrumentKind(str(value).lower())


@dataclass(frozen=True)
class PlatformSettings:
    control_api_url: str
    control_api_auto_submit: bool
    trading_environment: TradingEnvironment
    platform_data_root: str
    duckdb_path: str
    bronze_ttl_days: int
    redis_url: str
    baseline_strategy_id: str
    baseline_model_version_id: str
    baseline_profile_id: str
    baseline_sleeve_id: str
    baseline_instrument_id: str
    baseline_inst_id: str
    baseline_instrument_kind: InstrumentKind
    baseline_threshold_bps: float
    baseline_target_size: float


def get_platform_settings() -> PlatformSettings:
    environment = parse_environment(os.getenv("TRADING_PROFILE"))
    profile_id = os.getenv("BASELINE_PROFILE_ID", f"{environment.value}-main")
    return PlatformSettings(
        control_api_url=os.getenv("CONTROL_API_URL", "http://127.0.0.1:8000"),
        control_api_auto_submit=os.getenv("CONTROL_API_AUTO_SUBMIT", "0") == "1",
        trading_environment=environment,
        platform_data_root=os.getenv("PLATFORM_DATA_ROOT", "./data/lake"),
        duckdb_path=os.getenv("DUCKDB_PATH", "./data/platform.duckdb"),
        bronze_ttl_days=int(os.getenv("OKX_BRONZE_TTL_DAYS", "7")),
        redis_url=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
        baseline_strategy_id=os.getenv("BASELINE_STRATEGY_ID", "reference-breakout"),
        baseline_model_version_id=os.getenv(
            "BASELINE_MODEL_VERSION_ID", "reference-breakout-baseline-v1"
        ),
        baseline_profile_id=profile_id,
        baseline_sleeve_id=os.getenv(
            "BASELINE_SLEEVE_ID", f"{profile_id}-default-sleeve"
        ),
        baseline_instrument_id=os.getenv(
            "BASELINE_INSTRUMENT_ID",
            f"{profile_id}:{os.getenv('BASELINE_INST_ID', 'BTC-USDT-SWAP')}",
        ),
        baseline_inst_id=os.getenv("BASELINE_INST_ID", "BTC-USDT-SWAP"),
        baseline_instrument_kind=parse_instrument_kind(
            os.getenv("BASELINE_INSTRUMENT_KIND", "swap")
        ),
        baseline_threshold_bps=float(os.getenv("BASELINE_THRESHOLD_BPS", "20")),
        baseline_target_size=float(os.getenv("BASELINE_TARGET_SIZE", "1")),
    )
