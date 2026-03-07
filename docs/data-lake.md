# Data Lake Policy

## Goal

The platform now targets a lake-first event-sourcing model:

- object store is the durable system of record for immutable audit history
- DuckDB is the local query engine and hot-cache entrypoint for research
- Postgres stores control-plane read models that can be rebuilt from audit
- research, replay, backtests, paper trading, and live-run records share the same lake contract

The old `300 GB` limit is a local hot-cache operating budget, not a global retention ceiling. Long-lived history can live in S3-compatible object storage while local DuckDB keeps only the data needed for current work.

## Layers

The lake is organized into four layers:

- `audit`: append-only control-plane and workflow events with manifests and checkpoints
- `bronze`: short-lived raw websocket and raw delta capture
- `silver`: normalized typed datasets used for research and replay
- `gold`: feature sets, model-ready datasets, run summaries, and reusable artifacts

Recommended long-lived datasets:

- `trades`
- `bars_1s`
- `books5_1s` for Tier A
- `tob_1s` for Tier B
- `funding`
- `fee_schedule`
- `fills`
- `positions`
- `balances`
- feature datasets
- run summaries and artifacts

Short-lived datasets:

- `ws_raw`
- `book_delta_raw`
- `debug_events`

## Event Contract

Every cross-service event written into the audit layer uses `EventEnvelope` with these fields:

- `event_id`
- `event_type`
- `schema_version`
- `occurred_at`
- `ingested_at`
- `profile_id`
- `strategy_id`
- `run_id`
- `inst_id`
- `correlation_id`
- `causation_id`
- `source_service`
- `payload`
- `payload_hash`

The current implementation appends audit events to Parquet segments, writes a manifest per segment, and updates a checkpoint per stream partition. Control-plane writes use this path before projecting the Postgres read model.

## Partition Layout

All Parquet data uses `ZSTD` compression. The partition key is standardized as `profile_id=` everywhere.

Typed datasets:

```text
<layer>/dt=YYYY-MM-DD/profile_id=<profile_id>/venue=<venue>/inst_id=<inst_id>/stream=<stream>/part-*.parquet
```

Optional partitions can add `run_id=` and `hour=` when needed.

Audit segments:

```text
audit/dt=YYYY-MM-DD/stream=<stream>/partition=<partition_id>/segment-<uuid>.parquet
audit/_manifests/stream=<stream>/partition=<partition_id>/<segment_id>.json
audit/_checkpoints/stream=<stream>/partition=<partition_id>.json
```

## Query Path

DuckDB is the read entrypoint for research data:

```sql
select *
from read_parquet('data/lake/silver/dt=*/profile_id=demo-main/venue=okx/inst_id=BTC-USDT-SWAP/stream=trades/*.parquet')
order by ts desc
limit 100;
```

CLI wrappers:

- `okx-platform lake sql --sql "..."`
- `okx-platform lake datasets --include-versions`
- `okx-platform lake artifacts --run-id ...`
- `okx-platform lake doctor`
- `okx-platform lake compact`
- `okx-platform lake rebuild-read-models`

## Retention and Tiering

Tiering remains the control mechanism for market-data cost:

- Tier A: `trades + books5_1s + bars_1s + funding + fee + account datasets`
- Tier B: `trades + tob_1s + bars_1s + funding + fee`
- Tier C: `bars_1s + funding + fee`, with optional time-limited `trades`

The platform should not permanently retain full-depth raw book streams for the whole universe. `book_delta_raw` and `ws_raw` are for bounded recovery and incident analysis windows only.

## Dataset Catalog and Lineage

Datasets, dataset versions, features, runs, and artifacts are registered in the control plane with lake metadata such as:

- logical name
- physical URI
- schema version
- manifest URI
- checksum
- row, file, and byte stats
- watermark range
- retention class
- quality state
- pinned flag
- source dataset or feature lineage
- producing run ID

One logical dataset can have many materialized versions. One run can own many artifacts.

## Current Implementation Notes

This repo includes the first operational slice of the target architecture:

- local filesystem object store by default, with S3-compatible support
- Redpanda client scaffolding
- synchronous audit append plus read-model projection inside the control API
- background market-data and replay workers behind feature flags

It is not yet a fully separate distributed `audit-writer` service plus standalone `read-model-projector` service. The interfaces and storage layout are in place so those pieces can be split later without changing the lake contract.
