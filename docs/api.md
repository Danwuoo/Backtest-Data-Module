# API

## Endpoints

- `GET /profiles`
- `POST /profiles`
- `GET /instruments`
- `POST /instruments`
- `GET /bots`
- `POST /bots`
- `POST /bots/{bot_name}/enable`
- `POST /bots/{bot_name}/disable`
- `POST /bots/{bot_name}/deploy`
- `GET /orders`
- `POST /orders`
- `POST /orders/cancel`
- `GET /positions`
- `POST /positions`
- `GET /balances`
- `POST /balances`
- `GET /services`
- `POST /services`
- `GET /kill-switch`
- `PUT /kill-switch`
- `GET /healthz`

## Authentication

Protected endpoints use the `X-API-KEY` header when `CONTROL_API_KEY` is configured.
