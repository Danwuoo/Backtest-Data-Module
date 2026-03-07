# Operations

## Standard Workflow

1. Run `alembic upgrade head`.
2. Start infrastructure dependencies: `postgres`, `minio`, `redpanda`, and `redis`.
3. Start `control-api` and verify `/profiles`, `/kill-switch`, and `/healthz`.
4. Start `market-data-service` and `replay-service`.
5. Leave `ENABLE_MARKET_DATA_WORKER=0` and `ENABLE_REPLAY_WORKER=0` until credentials, whitelist, and lake paths are verified.
6. Start inference, portfolio, execution policy, risk, and execution services.
7. Verify `/services` heartbeats and service-specific `/healthz` payloads.
8. Seed strategies, model versions, instruments, balances, and any required datasets or features.
9. Run manual demo orders before enabling automated flow.

## Lake Operations

Useful maintenance commands:

- `okx-platform lake doctor`
- `okx-platform lake compact`
- `okx-platform lake rebuild-read-models --stream okx-platform.control-plane`

Use `lake rebuild-read-models` when read models need to be reconstructed from audit history after an interruption, migration, or catalog corruption event.

## Kill Switch

The kill switch blocks new orders platform-wide.

- Read the current state with `GET /kill-switch`
- Activate it with `PUT /kill-switch`
- Use `okx-platform stop-all --reason ...` for the fastest CLI path

## Migration and Cutover

- Use Alembic for schema creation before application startup.
- V1 `trading_*` tables are read during bootstrap and mapped into V2 `platform_*` tables.
- `service_heartbeats` remain operational state and are not backfilled.
- Audit data in the lake is the durable recovery path for new control-plane write models.
