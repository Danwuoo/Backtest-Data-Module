import base64
import hashlib
import hmac

import httpx
import pytest

from okx_trading_platform.adapters.okx import (
    ClientOrderIdCache,
    OkxExchangeGateway,
    OkxRequestSigner,
    OkxRestClient,
    OkxWebSocketRouter,
    RateLimitGovernor,
)
from okx_trading_platform.domain import (
    InstrumentKind,
    OrderPlan,
    OrderSide,
    OrderType,
    TdMode,
    TradingEnvironment,
)


def test_okx_request_signer_matches_hmac_sha256():
    timestamp = "2024-01-01T00:00:00.000Z"
    path = "/api/v5/trade/order"
    body = '{"instId":"BTC-USDT-SWAP"}'
    expected = base64.b64encode(
        hmac.new(
            b"secret",
            f"{timestamp}POST{path}{body}".encode("utf-8"),
            hashlib.sha256,
        ).digest()
    ).decode("utf-8")
    assert (
        OkxRequestSigner.sign(
            secret_key="secret",
            timestamp=timestamp,
            method="POST",
            request_path=path,
            body=body,
        )
        == expected
    )


def test_okx_rest_client_adds_demo_header():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    client = OkxRestClient(http_client=httpx.Client(transport=transport))
    response = client.request(
        environment=TradingEnvironment.DEMO,
        method="GET",
        path="/api/v5/public/time",
    )
    assert response.json() == {"ok": True}
    assert captured["headers"]["x-simulated-trading"] == "1"


def test_okx_websocket_router_prefers_ws_for_market_orders():
    router = OkxWebSocketRouter(websocket_available=True)
    plan = OrderPlan(
        profile_id="demo-main",
        strategy_id="strategy-a",
        model_version_id="model-a",
        sleeve_id="demo-main-default-sleeve",
        instrument_id="demo-main:BTC-USDT-SWAP",
        inst_id="BTC-USDT-SWAP",
        kind=InstrumentKind.SWAP,
        side=OrderSide.BUY,
        size=1,
        order_type=OrderType.MARKET,
        td_mode=TdMode.ISOLATED,
    )
    assert router.choose_submit_route(plan) == "ws"


def test_rate_limit_governor_blocks_bucket_when_limit_hit():
    governor = RateLimitGovernor()
    assert governor.allow("demo:BTC-USDT-SWAP", limit=1, interval_seconds=60) is True
    assert governor.allow("demo:BTC-USDT-SWAP", limit=1, interval_seconds=60) is False


def test_execution_gateway_rejects_duplicate_client_order_id():
    class FakeResponse:
        def json(self):
            return {"data": [{"sCode": "0", "ordId": "123"}]}

    class FakeRestClient:
        def request(self, **kwargs):
            return FakeResponse()

    service = OkxExchangeGateway(
        rest_client=FakeRestClient(),
        router=OkxWebSocketRouter(websocket_available=False),
        dedupe_cache=ClientOrderIdCache(),
    )
    plan = OrderPlan(
        order_plan_id="plan-1",
        profile_id="demo-main",
        strategy_id="strategy-a",
        model_version_id="model-a",
        sleeve_id="demo-main-default-sleeve",
        instrument_id="demo-main:BTC-USDT-SWAP",
        inst_id="BTC-USDT-SWAP",
        kind=InstrumentKind.SWAP,
        side=OrderSide.BUY,
        size=1,
        order_type=OrderType.LIMIT,
        td_mode=TdMode.ISOLATED,
        price=30000,
        metadata={"client_order_id": "cl-1"},
    )
    state = service.submit_order(plan)
    assert state.exchange_order_id == "123"
    with pytest.raises(ValueError, match="Duplicate client_order_id"):
        service.submit_order(plan)
