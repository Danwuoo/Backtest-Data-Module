FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src

CMD [
  "uvicorn",
  "backtest_data_module.trading.services.control_api:app",
  "--host",
  "0.0.0.0",
  "--port",
  "8000"
]
