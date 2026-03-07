from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from okx_trading_platform.api.app import app, get_api_key, get_db
from okx_trading_platform.application.persistence import (
    Base,
    BotModel,
    OrderRecordModel,
    TradingProfileModel,
)
from okx_trading_platform.shared.db import aget_db

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
app.dependency_overrides[get_api_key] = override_get_api_key

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
    profile_ids = {item["profile_id"] for item in response.json()}
    assert profile_ids == {"demo-main", "live-main"}


def test_v2_order_flow_creates_fill_ledger_pnl_and_alerts():
    profiles = client.get("/profiles")
    assert profiles.status_code == 200

    strategy = client.post(
        "/strategies",
        json={
            "strategy_id": "reference-breakout",
            "profile_id": "demo-main",
            "name": "reference-breakout",
            "baseline_provider": "reference_breakout",
            "status": "enabled",
            "allowed_instrument_ids": ["demo-main:BTC-USDT-SWAP"],
            "primary_model_version_id": "reference-breakout-baseline-v1",
            "config": {},
        },
    )
    assert strategy.status_code == 200

    model = client.post(
        "/models",
        json={
            "model_version_id": "reference-breakout-baseline-v1",
            "strategy_id": "reference-breakout",
            "profile_id": "demo-main",
            "name": "baseline-v1",
            "kind": "rule_baseline",
            "metadata": {},
        },
    )
    assert model.status_code == 200

    instrument = client.post(
        "/instruments",
        json={
            "instrument_id": "demo-main:BTC-USDT-SWAP",
            "profile_id": "demo-main",
            "inst_id": "BTC-USDT-SWAP",
            "kind": "swap",
            "min_notional": 5,
            "allow_trading": True,
            "metadata": {},
        },
    )
    assert instrument.status_code == 200

    balance = client.post(
        "/balances",
        json={
            "balance_snapshot_id": "demo-main:USDT",
            "profile_id": "demo-main",
            "currency": "USDT",
            "available": 100,
            "cash_balance": 100,
            "equity": 100,
        },
    )
    assert balance.status_code == 200

    order = client.post(
        "/orders",
        json={
            "profile_id": "demo-main",
            "strategy_id": "reference-breakout",
            "model_version_id": "reference-breakout-baseline-v1",
            "sleeve_id": "demo-main-default-sleeve",
            "instrument_id": "demo-main:BTC-USDT-SWAP",
            "inst_id": "BTC-USDT-SWAP",
            "kind": "swap",
            "side": "buy",
            "size": 1,
            "td_mode": "isolated",
            "metadata": {"mark_price": 10},
        },
    )
    assert order.status_code == 200
    order_payload = order.json()
    assert order_payload["status"] == "approved"

    fill = client.post(
        "/fills",
        json={
            "order_id": order_payload["order_id"],
            "profile_id": "demo-main",
            "instrument_id": "demo-main:BTC-USDT-SWAP",
            "inst_id": "BTC-USDT-SWAP",
            "fill_price": 10,
            "fill_size": 1,
            "fee": 0.1,
            "raw_payload": {"latency_ms": 50},
        },
    )
    assert fill.status_code == 200

    ledger = client.get("/ledger", params={"profile_id": "demo-main"})
    pnl = client.get("/pnl", params={"profile_id": "demo-main"})
    risk = client.get("/risk-snapshots", params={"profile_id": "demo-main"})
    execution = client.get("/execution-snapshots", params={"profile_id": "demo-main"})
    assert len(ledger.json()) == 2
    assert pnl.json()[0]["net_pnl"] == -0.1
    assert len(risk.json()) >= 2
    assert len(execution.json()) >= 2

    incident = client.post(
        "/incidents",
        json={
            "profile_id": "demo-main",
            "severity": "critical",
            "title": "latency incident",
            "message": "latency breach detected",
            "metadata": {},
        },
    )
    assert incident.status_code == 200
    alerts = client.get("/alerts", params={"profile_id": "demo-main"})
    assert alerts.status_code == 200
    assert alerts.json()[0]["title"] == "latency incident"


def test_legacy_bot_and_orders_migrate_into_v2():
    db = TestingSessionLocal()
    try:
        db.add(
            TradingProfileModel(
                profile="demo",
                name="demo",
                rest_base_url="https://www.okx.com",
                public_ws_url="wss://wspap.okx.com:8443/ws/v5/public",
                private_ws_url="wss://wspap.okx.com:8443/ws/v5/private",
                credential_env_prefix="OKX_DEMO",
                is_simulated=True,
                is_active=True,
            )
        )
        db.add(
            BotModel(
                name="legacy-bot",
                profile="demo",
                signal_provider="reference_breakout",
                status="enabled",
                instrument_ids=["BTC-USDT-SWAP"],
                config_json={},
            )
        )
        db.add(
            OrderRecordModel(
                order_id="legacy-order",
                client_order_id="legacy-client-order",
                profile="demo",
                inst_id="BTC-USDT-SWAP",
                instrument_kind="swap",
                side="buy",
                size=1,
                order_type="market",
                td_mode="isolated",
                status="approved",
                source="manual",
                raw_payload={},
            )
        )
        db.commit()
    finally:
        db.close()

    profiles = client.get("/profiles")
    assert profiles.status_code == 200

    strategies = client.get("/strategies", params={"profile_id": "demo-main"})
    orders = client.get("/orders", params={"profile_id": "demo-main"})
    assert strategies.status_code == 200
    assert any(item["strategy_id"] == "legacy-bot" for item in strategies.json())
    assert any(item["order_id"] == "legacy-order" for item in orders.json())


def test_auth_failure_can_be_overridden():
    def mock_get_api_key_unauthorized():
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    app.dependency_overrides[get_api_key] = mock_get_api_key_unauthorized
    response = client.get("/profiles")
    assert response.status_code == 401
    app.dependency_overrides[get_api_key] = override_get_api_key
