from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backtest_data_module.strategy_manager.auth import get_api_key as real_get_api_key
from backtest_data_module.strategy_manager.database import Base, aget_db
from backtest_data_module.strategy_manager.main import app, get_db

SQLALCHEMY_DATABASE_URL = "sqlite:///./platform_test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


async def override_get_api_key():
    return None


app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[aget_db] = override_get_db
app.dependency_overrides[real_get_api_key] = override_get_api_key

client = TestClient(app)


@pytest.fixture(scope="function", autouse=True)
def setup_teardown():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_profiles_endpoint_bootstraps_demo_and_live():
    response = client.get("/profiles")
    assert response.status_code == 200
    profiles = {item["profile"] for item in response.json()}
    assert profiles == {"demo", "live"}


def test_control_api_order_flow_and_kill_switch():
    balance = client.post(
        "/balances",
        json={
            "profile": "demo",
            "currency": "USDT",
            "available": 100,
            "cash_balance": 100,
            "equity": 100,
        },
    )
    assert balance.status_code == 200

    instrument = client.post(
        "/instruments",
        json={
            "inst_id": "BTC-USDT-SWAP",
            "kind": "swap",
            "profile": "demo",
            "allow_trading": True,
            "metadata": {},
        },
    )
    assert instrument.status_code == 200

    order = client.post(
        "/orders",
        json={
            "profile": "demo",
            "instrument_kind": "swap",
            "inst_id": "BTC-USDT-SWAP",
            "side": "buy",
            "size": 1,
            "td_mode": "isolated",
            "metadata": {"mark_price": 10},
        },
    )
    assert order.status_code == 200
    assert order.json()["status"] == "approved"

    stop = client.put(
        "/kill-switch",
        json={"activated": True, "reason": "manual stop"},
    )
    assert stop.status_code == 200
    assert stop.json()["activated"] is True

    blocked = client.post(
        "/orders",
        json={
            "profile": "demo",
            "instrument_kind": "swap",
            "inst_id": "BTC-USDT-SWAP",
            "side": "buy",
            "size": 1,
            "td_mode": "isolated",
            "metadata": {"mark_price": 10},
        },
    )
    assert blocked.status_code == 423


def test_bot_deploy_and_service_heartbeat():
    bot = client.post(
        "/bots",
        json={
            "name": "reference-breakout",
            "profile": "demo",
            "signal_provider": "reference_breakout",
            "instrument_ids": ["BTC-USDT-SWAP"],
            "status": "disabled",
            "config": {},
        },
    )
    assert bot.status_code == 200

    deploy = client.post(
        "/bots/reference-breakout/deploy",
        json={"profile": "demo", "metadata": {"commit": "abc123"}},
    )
    assert deploy.status_code == 200
    assert deploy.json()["status"] == "deployed"

    heartbeat = client.post(
        "/services",
        json={
            "service_name": "risk-service",
            "instance_id": "instance-1",
            "profile": "demo",
            "status": "running",
            "metadata": {"version": "v1"},
        },
    )
    assert heartbeat.status_code == 200

    services = client.get("/services")
    assert services.status_code == 200
    assert services.json()[0]["service_name"] == "risk-service"


def test_auth_failure_can_still_be_overridden():
    def mock_get_api_key_unauthorized():
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

    app.dependency_overrides[real_get_api_key] = mock_get_api_key_unauthorized
    response = client.get("/profiles")
    assert response.status_code == 401
    app.dependency_overrides[real_get_api_key] = override_get_api_key
