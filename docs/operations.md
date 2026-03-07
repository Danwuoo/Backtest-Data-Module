# Operations

## Standard workflow

1. Run `alembic upgrade head`.
2. Start `control-api`.
3. Verify `/profiles`, `/kill-switch`, and `/healthz`.
4. Start market data, inference, portfolio, execution policy, risk, execution, and replay services.
5. Verify `/services` heartbeats.
6. Seed strategies, model versions, instruments, and balances.
7. Run manual demo orders before enabling automated flow.

## Kill switch

The kill switch blocks new orders platform-wide.

- Read the current state with `GET /kill-switch`
- Activate it with `PUT /kill-switch`
- Use `okx-platform stop-all --reason ...` for the fastest CLI path

## Migration and cutover

- Use Alembic for schema creation before application startup.
- V1 `trading_*` tables are read during bootstrap and mapped into V2 `platform_*` tables.
- `service_heartbeats` remain operational state and are not backfilled.
