# Data Lake Policy

## Goal

This platform uses a local `DuckDB + Parquet` lake for research, replay, backtests, paper trading, and live-run artifacts.

The storage target is:

- keep the lake under `300 GB`
- keep long-lived data queryable without rescanning raw order book deltas
- support a full-time ingest policy across the platform whitelist universe
- avoid permanent storage of full-depth raw market streams

`All OKX symbols forever` is not compatible with a hard `300 GB` cap if `trades` and `books5` are kept long-term. In practice, "full universe" means the full platform whitelist, with tiered retention by symbol importance.

## Storage Model

The lake is organized into three layers:

- `bronze`: raw websocket messages, raw order book deltas, and debug-grade events
- `silver`: normalized research datasets used by replay, feature generation, and backtests
- `gold`: feature sets, model-ready datasets, and run artifacts

Retention follows these rules:

- long retention: `1s bars`, `trades`, `top-of-book` or `books5` sampling, `funding`, `fee`, `fills`, `positions`, `balances`, feature parquet, and run artifacts
- short retention: raw order book deltas, raw websocket payloads, and debug events
- no retention: permanent full-depth raw streams across all symbols

## Capacity Budget

The lake should not be operated above `260 GB` in steady state. The remaining headroom is reserved for compaction, backfill, schema changes, and run spikes.

Recommended budget:

| Area | Target |
| --- | ---: |
| `silver/bars_1s` | `60 GB` |
| `silver/trades` | `80 GB` |
| `silver/books5_or_tob` | `25 GB` |
| `silver/funding_fee_account` | `10 GB` |
| `gold/features` | `50 GB` |
| `gold/run_artifacts` | `15 GB` |
| `bronze/raw_ring_buffer` | `20 GB` |
| free headroom | `40 GB` |

If the lake exceeds the target, eviction and downsampling must happen before new long-lived datasets are added.

## Universe Tiers

The whitelist universe is split into three tiers.

### Tier A

Use for the highest-liquidity and highest-priority trading symbols.

- target size: `20-30` symbols
- long retention: `bars_1s`, `trades`, `books5_1s`, `funding`, `fee`, account streams
- short retention: raw deltas and raw websocket messages

### Tier B

Use for symbols that remain research-relevant but do not justify permanent `books5`.

- target size: `50-100` symbols
- long retention: `bars_1s`, `trades`, `tob_1s`, `funding`, `fee`
- short retention: raw deltas and raw websocket messages

### Tier C

Use for the rest of the whitelist.

- long retention: `bars_1s`, `funding`, `fee`
- optional retention: `trades` for `30-90 days`
- no permanent order book storage

If permanent `trades` are required for more symbols, the allowed counts in `Tier A` and `Tier B` must be reduced.

## Dataset Catalog

### Long Retention

#### `bars_1s`

The canonical long-lived market dataset. Coarser bars are derived from this table.

Required columns:

- `ts`
- `inst_id`
- `open`
- `high`
- `low`
- `close`
- `vol_base`
- `vol_quote`
- `trade_count`
- `buy_vol`
- `sell_vol`
- `mark_px`
- `index_px`

#### `trades`

Required columns:

- `ts`
- `inst_id`
- `trade_id`
- `px`
- `sz`
- `side`
- `is_taker`
- `sequence_id`

#### `tob_1s`

Use for Tier B or any symbol where `books5` is too expensive.

Required columns:

- `ts`
- `inst_id`
- `bid_px1`
- `bid_sz1`
- `ask_px1`
- `ask_sz1`
- `mid_px`
- `spread_bps`

#### `books5_1s`

Use only for Tier A symbols that need microstructure features or execution replay.

Required columns:

- `ts`
- `inst_id`
- `bid_px1` to `bid_px5`
- `bid_sz1` to `bid_sz5`
- `ask_px1` to `ask_px5`
- `ask_sz1` to `ask_sz5`
- `mid_px`
- `spread_bps`

Only one of `tob_1s` or `books5_1s` should be stored long-term for the same symbol. Do not store both permanently.

#### `funding`

Required columns:

- `funding_ts`
- `inst_id`
- `rate`
- `realized_amount`

#### `fee_schedule`

Required columns:

- `effective_from`
- `inst_id`
- `maker_fee`
- `taker_fee`
- `tier`

#### `fills`

Required columns:

- `ts`
- `profile_id`
- `inst_id`
- `order_id`
- `trade_id`
- `px`
- `sz`
- `fee`
- `liquidity_flag`

#### `positions`

Required columns:

- `ts`
- `profile_id`
- `inst_id`
- `qty`
- `avg_px`
- `upl`
- `leverage`
- `margin_mode`

#### `balances`

Required columns:

- `ts`
- `profile_id`
- `currency`
- `available`
- `cash_balance`
- `equity`

#### `feature parquet`

Features are computed once, versioned, and stored as reusable Parquet datasets.

- keep only feature sets referenced by a model version, training job, or replay run
- do not rescan raw book data for repeated feature generation
- store `schema_version`, `feature_set_id`, and source dataset lineage

#### `run artifacts`

Long-lived artifacts include:

- backtest summaries
- paper-run summaries
- live-run summaries
- metrics tables
- plots and attribution outputs

Do not keep full event-by-event replay traces permanently unless they are explicitly pinned for investigation.

### Short Retention

#### `ws_raw`

- source: raw websocket frames
- TTL: `3-7 days`
- usage: parser debugging, data recovery, exchange incident analysis

#### `book_delta_raw`

- source: raw order book increment stream
- TTL: `1-3 days`
- usage: gap recovery, replay troubleshooting, schema validation

#### `debug_events`

- source: internal service debug payloads
- TTL: `1-3 days`
- usage: short-lived investigations only

### No Retention

The following data must not be stored permanently:

- full-market, all-symbol, full-depth raw book streams
- all raw websocket payloads beyond the short retention window
- debug-level internal events beyond the short retention window

## Partitioning

All datasets use `Parquet + ZSTD`.

Canonical market-data layout:

```text
<layer>/dt=YYYY-MM-DD/venue=okx/inst_id=<inst_id>/stream=<stream_type>/
```

Canonical account-data layout:

```text
<layer>/dt=YYYY-MM-DD/profile_id=<profile_id>/venue=okx/inst_id=<inst_id>/stream=<stream_type>/
```

Guidelines:

- use `UTC` for all timestamps and partition dates
- add `hour=HH` for high-volume streams when daily partitions grow too large
- target `128-256 MB` Parquet files for high-volume streams
- target `32-64 MB` Parquet files for low-volume streams
- compact small files on a schedule

## Compression and Schema Rules

- use `ZSTD` compression for every Parquet dataset
- prefer typed columns over JSON payload blobs
- keep only columns used by research, training, replay, or operations
- avoid nested raw payloads in long-lived silver datasets
- use fixed-scale numeric types where possible instead of unconstrained floating-point blobs

Long-lived research tables must not simply persist `model_dump()` payloads. Raw nested payloads are acceptable in `bronze`, but silver and gold datasets should be flat, typed, and schema-versioned.

## Retention and Eviction Order

When the lake approaches capacity, reduce storage in this order:

1. delete expired `debug_events`
2. delete expired `ws_raw`
3. delete expired `book_delta_raw`
4. remove non-pinned scratch feature sets
5. drop `Tier C` trade history older than policy
6. downgrade `Tier B` from `trades` retention if needed
7. reduce `Tier A` and `Tier B` symbol counts

Do not delete:

- pinned feature sets used by active models
- recent run artifacts required for comparison or audit
- funding, fee, fills, positions, and balances needed for reconciliation

## Operational Controls

Run a daily storage report for:

- bytes by `layer`
- bytes by `stream`
- bytes by `inst_id`
- file count by partition
- row count by dataset
- top growth contributors over the last `1`, `7`, and `30` days

The report should trigger action when:

- total lake size exceeds `260 GB`
- any single dataset exceeds its budget by more than `10%`
- raw bronze retention exceeds its TTL target
- file fragmentation causes partitions to fall below target Parquet size

## Feature and Run Policy

- compute features once and store them in gold datasets
- tie each feature set to a `schema_version`
- tie each feature set to source datasets and time windows
- keep run artifacts small and queryable
- store summary metrics, attribution tables, and compact plots before storing raw traces

Feature generation should be incremental. Rebuilding features from raw order book history should be reserved for explicit backfills or schema upgrades.

## Implementation Notes

This document is the target storage policy for the platform.

The current codebase already writes partitioned Parquet with `ZSTD` and registers datasets in DuckDB, but the long-lived market datasets should evolve toward flat typed schemas rather than generic payload dumps. The platform should also distinguish global market-data partitions from account-scoped partitions when adding new writers and compaction jobs.
