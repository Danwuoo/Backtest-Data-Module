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


def parse_csv(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    items = [item.strip() for item in value.split(",")]
    return tuple(item for item in items if item)


def parse_instrument_kinds(value: str | None) -> tuple[InstrumentKind, ...]:
    raw_values = parse_csv(value)
    if not raw_values:
        return (InstrumentKind.SPOT, InstrumentKind.SWAP)
    return tuple(parse_instrument_kind(item) for item in raw_values)


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
    okx_universe_source: str
    okx_public_instrument_kinds: tuple[InstrumentKind, ...]
    okx_inst_id_whitelist: tuple[str, ...]
    okx_tier_a_inst_ids: tuple[str, ...]
    okx_tier_b_inst_ids: tuple[str, ...]
    okx_public_ws_batch_size: int
    okx_instrument_refresh_seconds: int
    okx_account_poll_seconds: int
    okx_fill_poll_seconds: int
    okx_fee_poll_seconds: int
    okx_funding_poll_seconds: int
    okx_trade_flush_interval_seconds: int
    okx_book_sampling_interval_ms: int
    okx_rest_public_limit_per_2s: int
    okx_rest_private_limit_per_2s: int
    okx_rest_backfill_limit_per_2s: int
    okx_ws_raw_ttl_days: int
    okx_book_delta_ttl_days: int


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
        okx_universe_source=os.getenv("OKX_UNIVERSE_SOURCE", "control_plane"),
        okx_public_instrument_kinds=parse_instrument_kinds(
            os.getenv("OKX_PUBLIC_INSTRUMENT_KINDS", "spot,swap")
        ),
        okx_inst_id_whitelist=parse_csv(os.getenv("OKX_INST_ID_WHITELIST")),
        okx_tier_a_inst_ids=parse_csv(
            os.getenv(
                "OKX_TIER_A_INST_IDS",
                os.getenv("BASELINE_INST_ID", "BTC-USDT-SWAP"),
            )
        ),
        okx_tier_b_inst_ids=parse_csv(os.getenv("OKX_TIER_B_INST_IDS")),
        okx_public_ws_batch_size=int(os.getenv("OKX_PUBLIC_WS_BATCH_SIZE", "50")),
        okx_instrument_refresh_seconds=int(
            os.getenv("OKX_INSTRUMENT_REFRESH_SECONDS", "21600")
        ),
        okx_account_poll_seconds=int(os.getenv("OKX_ACCOUNT_POLL_SECONDS", "30")),
        okx_fill_poll_seconds=int(os.getenv("OKX_FILL_POLL_SECONDS", "10")),
        okx_fee_poll_seconds=int(os.getenv("OKX_FEE_POLL_SECONDS", "900")),
        okx_funding_poll_seconds=int(os.getenv("OKX_FUNDING_POLL_SECONDS", "600")),
        okx_trade_flush_interval_seconds=int(
            os.getenv("OKX_TRADE_FLUSH_INTERVAL_SECONDS", "1")
        ),
        okx_book_sampling_interval_ms=int(
            os.getenv("OKX_BOOK_SAMPLING_INTERVAL_MS", "1000")
        ),
        okx_rest_public_limit_per_2s=int(
            os.getenv("OKX_REST_PUBLIC_LIMIT_PER_2S", "10")
        ),
        okx_rest_private_limit_per_2s=int(
            os.getenv("OKX_REST_PRIVATE_LIMIT_PER_2S", "5")
        ),
        okx_rest_backfill_limit_per_2s=int(
            os.getenv("OKX_REST_BACKFILL_LIMIT_PER_2S", "2")
        ),
        okx_ws_raw_ttl_days=int(os.getenv("OKX_WS_RAW_TTL_DAYS", "3")),
        okx_book_delta_ttl_days=int(os.getenv("OKX_BOOK_DELTA_TTL_DAYS", "2")),
    )
