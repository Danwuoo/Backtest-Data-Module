# OKX Trading Platform V2

OKX Trading Platform V2 is a control plane and service runtime for OKX trading that now includes replay, research artifacts, richer risk controls, execution analytics, sleeves, allocators, and alerting. The repository still targets a single-node deployment, but the runtime surface is no longer limited to a thin order gateway.

## Scope

- `control-api` manages profiles, sleeves, allocators, risk policies, strategies, model versions, datasets, features, runs, orders, fills, ledgers, PnL, incidents, alerts, balances, positions, and service heartbeats.
- `market-data-service`, `model-inference-service`, `portfolio-service`, `execution-policy-service`, `risk-service`, `execution-service`, and `replay-service` provide deployable entrypoints.
- PostgreSQL stores authoritative control-plane and trading state.
- Redis Streams is the intended event bus between runtime services.
- DuckDB + Parquet under `PLATFORM_DATA_ROOT` stores bronze/silver/gold research artifacts.

## Runtime Topology

- `market-data-service`: order book sequence handling, instrument metadata sync, and lake writes.
- `model-inference-service`: rule-baseline inference with registry-pinned model versions.
- `portfolio-service`: sleeve-aware target to position intent conversion.
- `execution-policy-service`: position intent to exchange-ready order plan conversion.
- `risk-service`: pre-trade, post-trade, and portfolio risk decisions.
- `execution-service`: OKX-aware submission, reconciliation, and execution analytics.
- `replay-service`: replay, backtest, paper-run, and walk-forward scaffolding.

## Quick Start

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
uvicorn okx_trading_platform.services.control_api:app --reload
```

In another shell:

```bash
okx-platform status
okx-platform profiles
okx-platform strategies --profile-id demo-main
```

## Validation

Any source change should pass:

```bash
flake8 .
pytest -q
```
