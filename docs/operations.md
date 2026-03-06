# Operations

## Standard workflow

1. Start `control-api`.
2. Validate `/profiles` and `/healthz`.
3. Allowlist instruments.
4. Register or enable bots.
5. Start service processes.
6. Verify `/services` heartbeats.
7. Place manual demo orders before enabling automated order flow.

## Kill switch

The kill switch is a platform-wide block on new orders.

- Read the current state with `GET /kill-switch`
- Activate it with `PUT /kill-switch`
- Use `okx-platform stop-all --reason ...` for the fastest CLI path

## Recovery

After service restart, re-check:

- balances
- positions
- open orders
- service heartbeats
- demo/live profile selection
