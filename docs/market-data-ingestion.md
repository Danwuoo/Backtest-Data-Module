# Market Data Ingestion

## Goal

The market-data service should be able to start from a whitelist universe, ingest long-lived research datasets, and stay inside a controlled application-side rate budget before any production API keys are enabled.

This document defines:

- the capture flow
- the application-side rate limits
- the symbol tiering used for storage control
- where OKX credentials must be injected

## Capture Flow

The capture flow is split into four phases.

### Phase 1: Instrument bootstrap

Refresh the instrument catalog by instrument kind on a timer.

- source: public REST
- outputs: `instrument_meta`
- cadence: every `OKX_INSTRUMENT_REFRESH_SECONDS`
- purpose: trading status, tick size, lot size, instrument family, quote/base/settle metadata

### Phase 2: Public live market capture

Subscribe to public market streams for the active whitelist universe.

- `trades`
  - source of truth for long-lived trade history
  - source of truth for derived `bars_1s`
- `books5_1s`
  - only for `Tier A`
  - sampled from the live public book feed
- `tob_1s`
  - only for `Tier B`
  - sampled from the live public best-bid-offer feed or derived from the maintained book
- `ws_raw`
  - short-lived ring buffer only
- `book_delta_raw`
  - short-lived ring buffer only

The service should not permanently store full-depth raw deltas for the whole universe.

### Phase 3: Derived market datasets

Derived datasets should be written from normalized public streams instead of rescanning raw order book data.

- derive `bars_1s` from normalized `trades`
- derive `tob_1s` or `books5_1s` from live in-memory books at the sampling interval
- write long-lived research tables as flat typed Parquet, not raw nested payload blobs

### Phase 4: Private account reconciliation

Private account data is required for:

- `fills`
- `positions`
- `balances`
- `fee_schedule`

These datasets should be reconciled on a timer with a separate private REST budget so that account sync cannot starve public market capture.

For fee sync requests:

- use `instId` for `SPOT`
- use `instFamily` for `SWAP`, `FUTURES`, and `OPTION`

## Rate-Limit Policy

The values below are conservative application-side budgets. They are intentionally lower than typical exchange-side ceilings so the platform has room for retries, reconnects, and future background jobs.

| Bucket | Default | Window | Used by |
| --- | ---: | ---: | --- |
| `rest_public_reference` | `10` | `2s` | instruments, funding, public metadata |
| `rest_private_account` | `5` | `2s` | balances, positions, fills, fee polling |
| `rest_backfill` | `2` | `2s` | historical backfills |

Additional operational rules:

- keep public live capture on websocket whenever possible
- keep historical backfills in a separate budget from steady-state ingestion
- do not run backfills with the same bucket as live account reconciliation
- keep `fills` polling faster than `balances` and `positions`
- if reconnects spike, pause backfills first

## Default Ingestion Settings

The service now exposes its ingestion plan at `GET /ingestion-plan`.

Default environment variables:

- `OKX_UNIVERSE_SOURCE=control_plane`
- `OKX_PUBLIC_INSTRUMENT_KINDS=spot,swap`
- `OKX_INST_ID_WHITELIST=`
- `OKX_TIER_A_INST_IDS=BTC-USDT-SWAP,ETH-USDT-SWAP`
- `OKX_TIER_B_INST_IDS=`
- `OKX_PUBLIC_WS_BATCH_SIZE=50`
- `OKX_INSTRUMENT_REFRESH_SECONDS=21600`
- `OKX_ACCOUNT_POLL_SECONDS=30`
- `OKX_FILL_POLL_SECONDS=10`
- `OKX_FEE_POLL_SECONDS=900`
- `OKX_FUNDING_POLL_SECONDS=600`
- `OKX_TRADE_FLUSH_INTERVAL_SECONDS=1`
- `OKX_BOOK_SAMPLING_INTERVAL_MS=1000`
- `OKX_REST_PUBLIC_LIMIT_PER_2S=10`
- `OKX_REST_PRIVATE_LIMIT_PER_2S=5`
- `OKX_REST_BACKFILL_LIMIT_PER_2S=2`
- `OKX_WS_RAW_TTL_DAYS=3`
- `OKX_BOOK_DELTA_TTL_DAYS=2`

## Tiering

Use the lake policy together with the capture plan:

- `Tier A`: long-retain `trades + books5_1s + bars_1s`
- `Tier B`: long-retain `trades + tob_1s + bars_1s`
- `Tier C`: long-retain `bars_1s`, with `trades` optional and time-limited

This tiering is how the platform stays under the `300 GB` lake cap while still ingesting the full whitelist universe.

## Where To Put API Credentials

The code reads credentials from environment variables. It does not automatically load `.env` files by itself.

Credential variable names:

- demo: `OKX_DEMO_API_KEY`, `OKX_DEMO_SECRET_KEY`, `OKX_DEMO_PASSPHRASE`
- live: `OKX_LIVE_API_KEY`, `OKX_LIVE_SECRET_KEY`, `OKX_LIVE_PASSPHRASE`

Recommended placement:

- Docker Compose: copy `.env.demo.example` to `.env.demo` or `.env.live.example` to `.env.live`, fill in the credentials there, and start Compose with `--env-file`
- local shell: export the same variables in the shell before starting the services

The market-data service now needs these credentials for private account capture, not only the execution service.

## Startup Checklist

1. Fill demo or live credentials into the matching environment variables.
2. Set the whitelist and tier lists.
3. Start `control-api`, `redis`, and `market-data-service`.
4. Check `GET /healthz`.
5. Check `GET /ingestion-plan`.
6. Verify the credential variable names match the selected profile.
7. Only then enable live private capture and backfills.
