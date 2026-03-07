from __future__ import annotations

from dataclasses import dataclass

from okx_trading_platform.shared.settings import PlatformSettings

from .settings import get_okx_profile_settings


@dataclass(frozen=True)
class IngestionRateLimit:
    bucket: str
    limit: int
    interval_seconds: int
    applies_to: str
    notes: str = ""


@dataclass(frozen=True)
class IngestionJob:
    job_id: str
    stream: str
    transport: str
    cadence: str
    storage_layer: str
    retention_class: str
    symbol_scope: str
    rate_limit_bucket: str | None = None
    batch_size: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class MarketDataIngestionPlan:
    venue: str
    environment: str
    whitelist_source: str
    whitelist: list[str]
    tier_a: list[str]
    tier_b: list[str]
    instrument_kinds: list[str]
    public_jobs: list[IngestionJob]
    account_jobs: list[IngestionJob]
    rate_limits: list[IngestionRateLimit]
    credential_env_vars: list[str]


def build_market_data_ingestion_plan(
    settings: PlatformSettings,
) -> MarketDataIngestionPlan:
    profile_settings = get_okx_profile_settings(settings.trading_environment)
    prefix = profile_settings.credential_env_prefix.upper()
    whitelist = list(settings.okx_inst_id_whitelist)
    whitelist_source = (
        "env" if whitelist else settings.okx_universe_source.replace("_", "-")
    )
    symbol_scope = (
        ", ".join(whitelist)
        if whitelist
        else "control-plane allowlist or dynamic discovery"
    )
    tier_a_scope = _scope_text(settings.okx_tier_a_inst_ids, "tier-a allowlist")
    tier_b_scope = _scope_text(settings.okx_tier_b_inst_ids, "tier-b allowlist")
    account_scope = (
        symbol_scope
        if whitelist
        else "account-scoped symbols discovered from balances, positions, and fills"
    )

    public_jobs = [
        IngestionJob(
            job_id="instrument-refresh",
            stream="instrument_meta",
            transport="rest/public",
            cadence=f"every {settings.okx_instrument_refresh_seconds}s",
            storage_layer="silver",
            retention_class="long",
            symbol_scope="spot,swap instrument catalogs",
            rate_limit_bucket="rest_public_reference",
            batch_size=settings.okx_public_ws_batch_size,
            notes=(
                "Refresh trading status, lot size, tick size, "
                "and instrument metadata."
            ),
        ),
        IngestionJob(
            job_id="trade-stream",
            stream="trades",
            transport="ws/public",
            cadence=(
                "continuous, flush every "
                f"{settings.okx_trade_flush_interval_seconds}s"
            ),
            storage_layer="silver",
            retention_class="long",
            symbol_scope=symbol_scope,
            batch_size=settings.okx_public_ws_batch_size,
            notes="Primary source for long-lived trades and derived 1s bars.",
        ),
        IngestionJob(
            job_id="bars-rollup",
            stream="bars_1s",
            transport="derived",
            cadence=f"every {settings.okx_trade_flush_interval_seconds}s",
            storage_layer="silver",
            retention_class="long",
            symbol_scope=symbol_scope,
            notes=(
                "Aggregate 1s bars from normalized trades "
                "instead of refetching candles."
            ),
        ),
        IngestionJob(
            job_id="books5-sampler",
            stream="books5_1s",
            transport="ws/public -> sampler",
            cadence=f"sample every {settings.okx_book_sampling_interval_ms}ms",
            storage_layer="silver",
            retention_class="long",
            symbol_scope=tier_a_scope,
            batch_size=settings.okx_public_ws_batch_size,
            notes=(
                "Only for Tier A symbols that need "
                "microstructure and replay fidelity."
            ),
        ),
        IngestionJob(
            job_id="tob-sampler",
            stream="tob_1s",
            transport="ws/public -> sampler",
            cadence=f"sample every {settings.okx_book_sampling_interval_ms}ms",
            storage_layer="silver",
            retention_class="long",
            symbol_scope=tier_b_scope,
            batch_size=settings.okx_public_ws_batch_size,
            notes="Use for Tier B symbols instead of permanently storing books5.",
        ),
        IngestionJob(
            job_id="raw-ws-ring-buffer",
            stream="ws_raw",
            transport="ws/public",
            cadence="continuous",
            storage_layer="bronze",
            retention_class=f"short ({settings.okx_ws_raw_ttl_days}d TTL)",
            symbol_scope=symbol_scope,
            notes="Short-lived raw frame archive for parser and incident debugging.",
        ),
        IngestionJob(
            job_id="raw-book-delta-ring-buffer",
            stream="book_delta_raw",
            transport="ws/public",
            cadence="continuous",
            storage_layer="bronze",
            retention_class=f"short ({settings.okx_book_delta_ttl_days}d TTL)",
            symbol_scope=tier_a_scope,
            notes=(
                "Short-lived raw delta archive for gap recovery "
                "and replay debugging."
            ),
        ),
        IngestionJob(
            job_id="funding-poller",
            stream="funding",
            transport="rest/public",
            cadence=f"every {settings.okx_funding_poll_seconds}s",
            storage_layer="silver",
            retention_class="long",
            symbol_scope=symbol_scope,
            rate_limit_bucket="rest_public_reference",
            notes=(
                "Capture funding rates and realized funding history "
                "for swap instruments."
            ),
        ),
    ]
    account_jobs = [
        IngestionJob(
            job_id="balance-sync",
            stream="balances",
            transport="rest/private",
            cadence=f"every {settings.okx_account_poll_seconds}s",
            storage_layer="silver",
            retention_class="long",
            symbol_scope=account_scope,
            rate_limit_bucket="rest_private_account",
            notes="Reconcile available balance, cash balance, and equity snapshots.",
        ),
        IngestionJob(
            job_id="position-sync",
            stream="positions",
            transport="rest/private",
            cadence=f"every {settings.okx_account_poll_seconds}s",
            storage_layer="silver",
            retention_class="long",
            symbol_scope=account_scope,
            rate_limit_bucket="rest_private_account",
            notes=(
                "Reconcile open positions for live, paper, "
                "and backtest calibration."
            ),
        ),
        IngestionJob(
            job_id="fill-sync",
            stream="fills",
            transport="rest/private",
            cadence=f"every {settings.okx_fill_poll_seconds}s",
            storage_layer="silver",
            retention_class="long",
            symbol_scope=account_scope,
            rate_limit_bucket="rest_private_account",
            notes="Capture fills for execution attribution and account-state repair.",
        ),
        IngestionJob(
            job_id="fee-sync",
            stream="fee_schedule",
            transport="rest/private",
            cadence=f"every {settings.okx_fee_poll_seconds}s",
            storage_layer="silver",
            retention_class="long",
            symbol_scope="account-level fee tiers",
            rate_limit_bucket="rest_private_account",
            notes="Refresh maker/taker fee tiers used by live and replay cost models.",
        ),
    ]
    rate_limits = [
        IngestionRateLimit(
            bucket="rest_public_reference",
            limit=settings.okx_rest_public_limit_per_2s,
            interval_seconds=2,
            applies_to="instruments, funding, and other public reference pulls",
            notes=(
                "Conservative application-level budget; adjust only "
                "after observing stable headroom."
            ),
        ),
        IngestionRateLimit(
            bucket="rest_private_account",
            limit=settings.okx_rest_private_limit_per_2s,
            interval_seconds=2,
            applies_to="balances, positions, fills, and fee polling",
            notes=(
                "Shared across account reconciliation tasks to "
                "avoid private endpoint bursts."
            ),
        ),
        IngestionRateLimit(
            bucket="rest_backfill",
            limit=settings.okx_rest_backfill_limit_per_2s,
            interval_seconds=2,
            applies_to="historical backfill jobs",
            notes=(
                "Keep backfills isolated from steady-state capture "
                "so live sync keeps priority."
            ),
        ),
    ]
    return MarketDataIngestionPlan(
        venue="okx",
        environment=settings.trading_environment.value,
        whitelist_source=whitelist_source,
        whitelist=whitelist,
        tier_a=list(settings.okx_tier_a_inst_ids),
        tier_b=list(settings.okx_tier_b_inst_ids),
        instrument_kinds=[kind.value for kind in settings.okx_public_instrument_kinds],
        public_jobs=public_jobs,
        account_jobs=account_jobs,
        rate_limits=rate_limits,
        credential_env_vars=[
            f"{prefix}_API_KEY",
            f"{prefix}_SECRET_KEY",
            f"{prefix}_PASSPHRASE",
        ],
    )


def _scope_text(inst_ids: tuple[str, ...], fallback: str) -> str:
    return ", ".join(inst_ids) if inst_ids else fallback
