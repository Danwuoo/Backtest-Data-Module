# Configuration

## Core

- `CONTROL_API_URL`: control-plane base URL
- `CONTROL_API_DATABASE_URL`: SQLAlchemy database URL
- `CONTROL_API_KEY`: API key for protected routes
- `TRADING_PROFILE`: `demo` or `live`
- `BASELINE_PROFILE_ID`: default profile ID used by service entrypoints

## Lake-First Data Plane

- `PLATFORM_DATA_ROOT`: local lake root used by DuckDB and the filesystem object-store backend
- `DUCKDB_PATH`: DuckDB catalog path
- `HOT_CACHE_ROOT`: local hot-cache root for frequently queried data
- `CHECKPOINT_ROOT`: checkpoint directory for local consumers and maintenance jobs
- `OKX_BRONZE_TTL_DAYS`: bronze retention window
- `COMPACTION_INTERVAL_SECONDS`: compaction cadence for lake maintenance
- `ARTIFACT_RETENTION_DAYS`: default retention horizon for run artifacts

Object store settings:

- `OBJECT_STORE_BACKEND`: `filesystem` or `s3`
- `OBJECT_STORE_BUCKET`: required when `OBJECT_STORE_BACKEND=s3`
- `OBJECT_STORE_ENDPOINT`: MinIO or S3-compatible endpoint
- `OBJECT_STORE_REGION`
- `OBJECT_STORE_ACCESS_KEY_ID`
- `OBJECT_STORE_SECRET_ACCESS_KEY`
- `OBJECT_STORE_PREFIX`

Event bus settings:

- `EVENT_BUS_BACKEND`: currently `redpanda` by default
- `REDPANDA_BROKERS`: comma-separated broker list
- `REDPANDA_TOPIC_PREFIX`: topic namespace prefix, default `okx-platform`
- `REDIS_URL`: still available for cache and non-critical coordination

## Market Data

- `OKX_UNIVERSE_SOURCE`: `control_plane` or `env`
- `OKX_PUBLIC_INSTRUMENT_KINDS`: default `spot,swap`
- `OKX_INST_ID_WHITELIST`: comma-separated symbol allowlist override
- `OKX_TIER_A_INST_IDS`: symbols that keep long-lived `books5_1s`
- `OKX_TIER_B_INST_IDS`: symbols that keep long-lived `tob_1s`
- `OKX_PUBLIC_WS_BATCH_SIZE`: websocket subscription batch size
- `OKX_INSTRUMENT_REFRESH_SECONDS`
- `OKX_ACCOUNT_POLL_SECONDS`
- `OKX_FILL_POLL_SECONDS`
- `OKX_FEE_POLL_SECONDS`
- `OKX_FUNDING_POLL_SECONDS`
- `OKX_TRADE_FLUSH_INTERVAL_SECONDS`
- `OKX_BOOK_SAMPLING_INTERVAL_MS`
- `OKX_REST_PUBLIC_LIMIT_PER_2S`
- `OKX_REST_PRIVATE_LIMIT_PER_2S`
- `OKX_REST_BACKFILL_LIMIT_PER_2S`
- `OKX_WS_RAW_TTL_DAYS`
- `OKX_BOOK_DELTA_TTL_DAYS`
- `ENABLE_MARKET_DATA_WORKER`: `1` enables the background ingest worker

## Replay and Run Materialization

- `ENABLE_REPLAY_WORKER`: `1` enables the background replay worker

Rule baseline defaults:

- `BASELINE_STRATEGY_ID`
- `BASELINE_MODEL_VERSION_ID`
- `BASELINE_INST_ID`
- `BASELINE_INSTRUMENT_ID`
- `BASELINE_INSTRUMENT_KIND`
- `BASELINE_THRESHOLD_BPS`
- `BASELINE_TARGET_SIZE`

## Credentials and Secrets

OKX credentials:

- `OKX_DEMO_API_KEY`
- `OKX_DEMO_SECRET_KEY`
- `OKX_DEMO_PASSPHRASE`
- `OKX_LIVE_API_KEY`
- `OKX_LIVE_SECRET_KEY`
- `OKX_LIVE_PASSPHRASE`

Alerting:

- `SLACK_WEBHOOK_URL`

The application reads environment variables directly. It does not auto-load `.env` files, so use `docker compose --env-file .env.demo ...` or export the variables in the shell before startup.
