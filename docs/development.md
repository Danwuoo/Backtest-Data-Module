# Development

## Repository rules

- keep platform logic in `okx_trading_platform`
- keep FastAPI handlers thin and push orchestration into `ControlPlaneService`
- keep OKX normalization, sequence handling, and payload translation inside adapter modules
- keep PostgreSQL access inside repository classes
- prefer payload-backed V2 tables for API resources and keep legacy V1 tables only for migration/backfill

## Local validation

```bash
flake8 .
pytest -q
```

## Testing focus

- legacy-to-v2 migration
- order book sequence validation and gap detection
- rule-baseline target generation and order-plan conversion
- risk decisions and snapshot persistence
- fill to ledger/PnL/execution snapshot flow
- CLI command delegation
- service health contracts and Compose topology
