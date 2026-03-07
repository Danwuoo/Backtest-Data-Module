from fastapi.testclient import TestClient

from okx_trading_platform.services.execution import app as execution_app
from okx_trading_platform.services.execution_policy import app as execution_policy_app
from okx_trading_platform.services.market_data import app as market_data_app
from okx_trading_platform.services.model_inference import app as model_inference_app
from okx_trading_platform.services.portfolio import app as portfolio_app
from okx_trading_platform.services.replay import app as replay_app
from okx_trading_platform.services.risk import app as risk_app


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


def test_model_inference_service_health():
    client = TestClient(model_inference_app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["provider"] == "RuleBaselineInferenceProvider"


def test_portfolio_service_health():
    client = TestClient(portfolio_app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["service"] == "portfolio-service"


def test_execution_policy_service_health():
    client = TestClient(execution_policy_app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["service"] == "execution-policy-service"


def test_replay_service_health():
    client = TestClient(replay_app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["service"] == "replay-service"
