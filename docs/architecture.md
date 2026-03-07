# Architecture

The repository is split into explicit V2 layers:

- `domain`: profiles, sleeves, strategies, model versions, runs, orders, ledgers, incidents, and risk models
- `adapters/okx`: OKX signing, REST routing, instrument normalization, rate limiting, and order book sequence helpers
- `application`: control-plane orchestration, repositories, rule-baseline inference, and legacy-to-v2 migration
- `api`: the FastAPI control plane
- `services`: deployable runtime entrypoints
- `shared`: settings, DB wiring, lake writer, runtime helpers, notifier, and Redis stream abstractions

## Services

- `control-api`: CRUD and orchestration for V2 resources
- `market-data-service`: snapshot + delta order book handling, sequence validation, and lake writes
- `model-inference-service`: registry-pinned rule baseline inference
- `portfolio-service`: target signal to position intent conversion
- `execution-policy-service`: position intent to order plan conversion
- `risk-service`: pre-trade, post-trade, and portfolio risk runtime
- `execution-service`: OKX submission, reconciliation, and execution analytics
- `replay-service`: replay, backtest, paper-run, and walk-forward scaffolding

## Storage

- PostgreSQL stores authoritative control-plane and trading state
- Redis Streams carries runtime events between services
- DuckDB + Parquet stores bronze/silver/gold artifacts under `PLATFORM_DATA_ROOT`
- Raw bronze market data is short-lived; normalized silver and feature/run gold artifacts are long-lived
