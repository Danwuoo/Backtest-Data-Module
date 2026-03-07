# CLI

Install the package in editable mode to get the `okx-platform` command.

## Control-Plane Commands

- `okx-platform status`
- `okx-platform profiles`
- `okx-platform strategies --profile-id demo-main`
- `okx-platform models --profile-id demo-main`
- `okx-platform sleeves --profile-id demo-main`
- `okx-platform allocators --profile-id demo-main`
- `okx-platform runs backtests --profile-id demo-main`
- `okx-platform runs paper --profile-id demo-main`
- `okx-platform runs live --profile-id demo-main`
- `okx-platform incidents --profile-id demo-main`
- `okx-platform alerts --profile-id demo-main`
- `okx-platform order --profile-id demo-main --strategy-id ... --model-version-id ... --sleeve-id ... --instrument-id ... --inst-id BTC-USDT-SWAP --instrument-kind swap --side buy --size 1`
- `okx-platform stop-all --reason "manual intervention"`
- `okx-platform migrate`
- `okx-platform cutover`

## Lake Commands

The CLI is the primary user-facing query surface for the local research lake.

- `okx-platform lake datasets --profile-id demo-main`
- `okx-platform lake datasets --profile-id demo-main --include-versions`
- `okx-platform lake features --profile-id demo-main`
- `okx-platform lake runs --profile-id demo-main`
- `okx-platform lake artifacts --run-id bt-1`
- `okx-platform lake sql --sql "select count(*) from read_parquet('data/lake/silver/**/*.parquet')"`
- `okx-platform lake sql --file .\\queries\\recent_trades.sql`
- `okx-platform lake doctor`
- `okx-platform lake compact --glob-pattern "silver\\**\\*.parquet" --min-file-size-mb 8`
- `okx-platform lake rebuild-read-models --stream okx-platform.control-plane`

## Notes

- `lake sql` runs directly against DuckDB.
- `lake doctor` reports local layer sizes plus audit manifest and checkpoint counts.
- `lake rebuild-read-models` replays audit events from the immutable audit layer into Postgres read models.
