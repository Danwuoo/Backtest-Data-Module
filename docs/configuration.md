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
