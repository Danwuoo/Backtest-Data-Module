# API

## Core resources

- `GET/POST /profiles`
- `GET/POST /risk-policies`
- `GET/POST /allocators`
- `GET/POST /sleeves`
- `GET/POST /instruments`
- `GET/POST /strategies`
- `GET/POST /models`
- `GET/POST /datasets`
- `GET/POST /features`
- `GET/POST /backtests`
- `GET/POST /paper-runs`
- `GET/POST /live-runs`

## Trading and state

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

## Incidents and alerting

- `GET/POST /incidents`
- `GET/POST /alert-policies`
- `GET/POST /alerts`

## Authentication

Protected endpoints use the `X-API-KEY` header when `CONTROL_API_KEY` is configured.
