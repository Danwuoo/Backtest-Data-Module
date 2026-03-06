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
)
from okx_trading_platform.domain import (
    InstrumentKind,
    OrderIntent,
    OrderSide,
    OrderType,
    TdMode,
    TradingProfile,
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
        profile=TradingProfile.DEMO,
        method="GET",
        path="/api/v5/public/time",
    )
    assert response.json() == {"ok": True}
    assert captured["headers"]["x-simulated-trading"] == "1"


def test_okx_websocket_router_prefers_ws_for_market_orders():
    router = OkxWebSocketRouter(websocket_available=True)
    intent = OrderIntent(
        profile=TradingProfile.DEMO,
        instrument_kind=InstrumentKind.SWAP,
        inst_id="BTC-USDT-SWAP",
        side=OrderSide.BUY,
        size=1,
        order_type=OrderType.MARKET,
        td_mode=TdMode.ISOLATED,
    )
    assert router.choose_submit_route(intent) == "ws"


def test_okx_websocket_router_falls_back_to_rest_for_limit_orders():
    router = OkxWebSocketRouter(websocket_available=True)
    intent = OrderIntent(
        profile=TradingProfile.DEMO,
        instrument_kind=InstrumentKind.SPOT,
        inst_id="BTC-USDT",
        side=OrderSide.BUY,
        size=1,
        order_type=OrderType.LIMIT,
        td_mode=TdMode.CASH,
        price=30000,
    )
    assert router.choose_submit_route(intent) == "rest"


def test_execution_service_rejects_duplicate_client_order_id():
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
    intent = OrderIntent(
        profile=TradingProfile.DEMO,
        instrument_kind=InstrumentKind.SWAP,
        inst_id="BTC-USDT-SWAP",
        side=OrderSide.BUY,
        size=1,
        order_type=OrderType.LIMIT,
        td_mode=TdMode.ISOLATED,
        price=30000,
        client_order_id="cl-1",
    )
    state = service.submit_order(intent)
    assert state.exchange_order_id == "123"
    with pytest.raises(ValueError, match="Duplicate client_order_id"):
        service.submit_order(intent)
