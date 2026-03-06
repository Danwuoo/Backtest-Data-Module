# Deployment

## Local

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
uvicorn okx_trading_platform.services.control_api:app --reload
```

## Docker Compose

The repository ships with a single-node Compose topology:

- `postgres`
- `redis`
- `control-api`
- `market-data-service`
- `execution-service`
- `risk-service`
- `strategy-runner`

Start demo mode:

```bash
docker compose --profile demo --env-file .env.demo.example up --build
```

Start live mode:

```bash
docker compose --profile live --env-file .env.live.example up --build
```

Keep demo and live credentials isolated. Do not reuse API keys across profiles.
