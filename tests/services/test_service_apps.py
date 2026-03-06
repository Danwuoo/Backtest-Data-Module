from fastapi.testclient import TestClient

from okx_trading_platform.services.execution import app as execution_app
from okx_trading_platform.services.market_data import app as market_data_app
from okx_trading_platform.services.risk import app as risk_app
from okx_trading_platform.services.strategy_runner import app as strategy_runner_app


def test_market_data_service_health():
    client = TestClient(market_data_app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["service"] == "market-data-service"


def test_execution_service_health():
    client = TestClient(execution_app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert "ws_public" in response.json()


def test_risk_service_health():
    client = TestClient(risk_app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["service"] == "risk-service"


def test_strategy_runner_health():
    client = TestClient(strategy_runner_app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["provider"] == "ReferenceBreakoutSignalProvider"
