# Deployment

## Local

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
alembic upgrade head
uvicorn okx_trading_platform.services.control_api:app --reload
```

## Docker Compose

The repository ships with a single-node V2 topology:

- `postgres`
- `redis`
- `control-api`
- `market-data-service`
- `model-inference-service`
- `portfolio-service`
- `execution-policy-service`
- `execution-service`
- `replay-service`
- `risk-service`

Start demo mode:

```bash
cp .env.demo.example .env.demo
docker compose --profile demo --env-file .env.demo up --build
```

Start live mode:

```bash
cp .env.live.example .env.live
docker compose --profile live --env-file .env.live up --build
```

Keep demo and live credentials isolated and point both modes at their own profile IDs.

OKX credentials are read from environment variables and should be placed in the matching env file:

- demo: `OKX_DEMO_API_KEY`, `OKX_DEMO_SECRET_KEY`, `OKX_DEMO_PASSPHRASE`
- live: `OKX_LIVE_API_KEY`, `OKX_LIVE_SECRET_KEY`, `OKX_LIVE_PASSPHRASE`

The market-data service and execution service both need these variables when private capture is enabled.
