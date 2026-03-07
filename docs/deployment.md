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
docker compose --profile demo --env-file .env.demo.example up --build
```

Start live mode:

```bash
docker compose --profile live --env-file .env.live.example up --build
```

Keep demo and live credentials isolated and point both modes at their own profile IDs.
