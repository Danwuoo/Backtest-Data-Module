# OKX Trading Platform

OKX Trading Platform is a focused control plane and service runtime for automated trading on OKX. The repository supports `demo` and `live` profiles, `spot` and `USDT-settled swap` instruments, and a small set of services that can run on a single VPS with Docker Compose.

## Scope

- `control-api` manages profiles, allowlisted instruments, bots, orders, balances, positions, service heartbeats, and the kill switch.
- `market-data-service`, `execution-service`, `risk-service`, and `strategy-runner` provide deployable service entrypoints.
- `okx-platform` is the CLI for control plane operations.
- AI models, research notebooks, backtesting, and generic data platform modules are intentionally out of scope for this repository.

## Project Layout

- `src/okx_trading_platform/api`: FastAPI control plane and request schemas
- `src/okx_trading_platform/application`: orchestration and repository layer
- `src/okx_trading_platform/domain`: typed models and risk logic
- `src/okx_trading_platform/adapters/okx`: OKX REST/WS routing, signing, and order book helpers
- `src/okx_trading_platform/services`: deployable service entrypoints
- `src/okx_trading_platform/shared`: shared auth, database, settings, logging, and runtime helpers
- `docs/`: product, deployment, and operations documentation

## Quick Start

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
uvicorn okx_trading_platform.services.control_api:app --reload
```

In another shell:

```bash
okx-platform profiles
okx-platform status
```

## Validation

Any source change must pass:

```bash
flake8 .
pytest -q
```
