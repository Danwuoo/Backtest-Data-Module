# API

## Control-Plane Surface

The control API is the write path for control-plane records and the catalog surface for data lake assets. Create routes append an audit event first and then project the read model in-process.

Core resources:

- `GET/POST /profiles`
- `GET/POST /risk-policies`
- `GET/POST /allocators`
- `GET/POST /sleeves`
- `GET/POST /instruments`
- `GET/POST /strategies`
- `GET/POST /models`
- `GET/POST /datasets`
- `GET/POST /features`
- `GET/POST /dataset-versions`
- `GET/POST /run-artifacts`
- `GET/POST /backtests`
- `GET/POST /paper-runs`
- `GET/POST /live-runs`

Trading and state:

- `GET/POST /orders`
- `POST /orders/cancel`
- `GET/POST /fills`
- `GET /ledger`
- `GET /funding`
- `GET /pnl`
- `GET /risk-snapshots`
- `GET /execution-snapshots`
- `GET/POST /positions`
- `GET/POST /balances`
- `GET/POST /services`
- `GET/PUT /kill-switch`

Incidents and alerting:

- `GET/POST /incidents`
- `GET/POST /alert-policies`
- `GET/POST /alerts`

## Data Lake Metadata

`DatasetRecord`, `FeatureSet`, and `RunRecord` now carry lake-oriented metadata instead of only a path field.

Common fields exposed by dataset and artifact resources include:

- logical name
- physical URI
- schema version
- manifest URI
- checksum
- row, file, and byte statistics
- watermark start and end
- retention class
- quality state
- pinned flag
- source lineage references
- producing run ID

`/dataset-versions` stores materialized versions for one logical dataset. `/run-artifacts` stores one or more artifacts per run and can point back to the dataset version that produced them.

Supported list filters:

- `/dataset-versions`: `profile_id`, `dataset_id`
- `/run-artifacts`: `profile_id`, `run_id`
- `/backtests`, `/paper-runs`, `/live-runs`: `profile_id`

## Health and Auth

Protected endpoints use the `X-API-KEY` header when `CONTROL_API_KEY` is configured.

Operational endpoints exposed by services include:

- `GET /healthz`
- `GET /heartbeat`
- `GET /ingestion-plan` on `market-data-service`
