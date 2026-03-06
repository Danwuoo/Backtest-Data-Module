# Architecture

The repository is split into a small number of explicit layers:

- `domain`: trading models, enums, and risk rules
- `adapters/okx`: OKX-specific signing, routing, and order book helpers
- `application`: orchestration, repositories, and signal providers
- `api`: the FastAPI control plane
- `services`: deployable service entrypoints
- `shared`: settings, auth, database wiring, logging, and service runtimes

## Services

- `control-api`: CRUD and orchestration for profiles, instruments, bots, orders, balances, positions, service heartbeats, and the kill switch
- `market-data-service`: market data runtime and order book holder
- `execution-service`: OKX execution gateway wrapper
- `risk-service`: deterministic risk runtime
- `strategy-runner`: reference strategy provider and signal-to-order conversion

## Storage

- PostgreSQL stores control-plane state and trading records
- Redis is reserved for pub/sub, short-lived cache, and distributed coordination
- High-frequency market data is not persisted in PostgreSQL
