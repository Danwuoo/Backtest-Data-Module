# Configuration

## Core environment variables

- `CONTROL_API_URL`: control-plane base URL
- `CONTROL_API_DATABASE_URL`: SQLAlchemy database URL
- `CONTROL_API_KEY`: API key for protected routes
- `TRADING_PROFILE`: `demo` or `live`
- `BASELINE_PROFILE_ID`: default profile ID used by service entrypoints

## Data plane

- `PLATFORM_DATA_ROOT`: Parquet lake root
- `DUCKDB_PATH`: DuckDB catalog path
- `OKX_BRONZE_TTL_DAYS`: bronze retention window
- `REDIS_URL`: Redis Streams connection URL

## Market data ingestion

- `OKX_UNIVERSE_SOURCE`: `control_plane` or `env`
- `OKX_PUBLIC_INSTRUMENT_KINDS`: default `spot,swap`
- `OKX_INST_ID_WHITELIST`: comma-separated symbol allowlist override
- `OKX_TIER_A_INST_IDS`: comma-separated symbols that keep long-lived `books5_1s`
- `OKX_TIER_B_INST_IDS`: comma-separated symbols that keep long-lived `tob_1s`
- `OKX_PUBLIC_WS_BATCH_SIZE`: subscription batch size target
- `OKX_INSTRUMENT_REFRESH_SECONDS`: public instrument metadata refresh cadence
- `OKX_ACCOUNT_POLL_SECONDS`: balances and positions reconciliation cadence
- `OKX_FILL_POLL_SECONDS`: fills reconciliation cadence
- `OKX_FEE_POLL_SECONDS`: fee tier refresh cadence
- `OKX_FUNDING_POLL_SECONDS`: funding refresh cadence
- `OKX_TRADE_FLUSH_INTERVAL_SECONDS`: trade-to-`bars_1s` rollup cadence
- `OKX_BOOK_SAMPLING_INTERVAL_MS`: `books5_1s` or `tob_1s` sampling cadence
- `OKX_REST_PUBLIC_LIMIT_PER_2S`: public REST application budget
- `OKX_REST_PRIVATE_LIMIT_PER_2S`: private REST application budget
- `OKX_REST_BACKFILL_LIMIT_PER_2S`: backfill REST application budget
- `OKX_WS_RAW_TTL_DAYS`: raw websocket archive TTL
- `OKX_BOOK_DELTA_TTL_DAYS`: raw book delta archive TTL

## Rule baseline defaults

- `BASELINE_STRATEGY_ID`
- `BASELINE_MODEL_VERSION_ID`
- `BASELINE_INST_ID`
- `BASELINE_INSTRUMENT_ID`
- `BASELINE_INSTRUMENT_KIND`
- `BASELINE_THRESHOLD_BPS`
- `BASELINE_TARGET_SIZE`

## OKX credentials

- `OKX_DEMO_API_KEY`
- `OKX_DEMO_SECRET_KEY`
- `OKX_DEMO_PASSPHRASE`
- `OKX_LIVE_API_KEY`
- `OKX_LIVE_SECRET_KEY`
- `OKX_LIVE_PASSPHRASE`

## Alerting

- `SLACK_WEBHOOK_URL`
