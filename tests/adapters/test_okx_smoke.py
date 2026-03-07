from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import websockets

from okx_trading_platform.adapters.okx import (
    OkxRestClient,
    build_okx_trade_fee_params,
)
from okx_trading_platform.domain import TradingEnvironment


def _load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def _inst_type(inst_id: str, fallback: str) -> str:
    if fallback:
        return fallback.upper()
    if inst_id.endswith("-SWAP"):
        return "SWAP"
    if inst_id.count("-") == 1:
        return "SPOT"
    return "SWAP"


def _smoke_env_cases() -> list[tuple[Path, str]]:
    return [
        (Path(".env.demo.example"), "OKX_DEMO"),
        (Path(".env.live.example"), "OKX_LIVE"),
    ]


@pytest.fixture(params=_smoke_env_cases(), ids=lambda case: case[0].name)
def okx_env(request, monkeypatch):
    if os.getenv("RUN_OKX_SMOKE") != "1":
        pytest.skip("set RUN_OKX_SMOKE=1 to run external OKX smoke tests")
    env_path, prefix = request.param
    values = _load_env(env_path)
    if not all(values.get(name) for name in [
        f"{prefix}_API_KEY",
        f"{prefix}_SECRET_KEY",
        f"{prefix}_PASSPHRASE",
    ]):
        pytest.skip(f"missing credentials in {env_path.name}")
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    environment = TradingEnvironment(values["TRADING_PROFILE"].lower())
    inst_id = values.get("BASELINE_INST_ID", "BTC-USDT-SWAP")
    inst_type = _inst_type(inst_id, values.get("BASELINE_INSTRUMENT_KIND", ""))
    return {
        "environment": environment,
        "inst_id": inst_id,
        "inst_type": inst_type,
    }


def test_okx_rest_read_only_smoke(okx_env):
    client = OkxRestClient()
    checks = [
        (
            "public_time",
            {"method": "GET", "path": "/api/v5/public/time", "auth": False},
        ),
        (
            "public_instruments",
            {
                "method": "GET",
                "path": "/api/v5/public/instruments",
                "params": {"instType": okx_env["inst_type"]},
                "auth": False,
            },
        ),
        (
            "funding_history",
            {
                "method": "GET",
                "path": "/api/v5/public/funding-rate-history",
                "params": {"instId": okx_env["inst_id"], "limit": "1"},
                "auth": False,
            },
        ),
        ("balance", {"method": "GET", "path": "/api/v5/account/balance", "auth": True}),
        (
            "positions",
            {"method": "GET", "path": "/api/v5/account/positions", "auth": True},
        ),
        (
            "trade_fee",
            {
                "method": "GET",
                "path": "/api/v5/account/trade-fee",
                "params": build_okx_trade_fee_params(
                    inst_type=okx_env["inst_type"],
                    inst_id=okx_env["inst_id"],
                ),
                "auth": True,
            },
        ),
        (
            "fills_history",
            {
                "method": "GET",
                "path": "/api/v5/trade/fills-history",
                "params": {"instType": okx_env["inst_type"], "limit": "1"},
                "auth": True,
            },
        ),
    ]

    for name, request_args in checks:
        response = client.request(environment=okx_env["environment"], **request_args)
        assert response.status_code == 200, name
        payload = response.json()
        assert payload.get("code") == "0", (name, payload.get("msg"))


@pytest.mark.asyncio
async def test_okx_public_ws_trades_smoke(okx_env):
    client = OkxRestClient()
    url = client.websocket_url(environment=okx_env["environment"], private=False)
    async with websockets.connect(url, ping_interval=None, close_timeout=2) as ws:
        await ws.send(
            json.dumps(
                {
                    "op": "subscribe",
                    "args": [
                        {"channel": "trades", "instId": okx_env["inst_id"]},
                    ],
                }
            )
        )
        subscribed = False
        for _ in range(6):
            raw = await ws.recv()
            message = json.loads(raw)
            if message.get("event") == "subscribe":
                subscribed = True
                continue
            if message.get("arg", {}).get("channel") == "trades":
                assert subscribed
                return
            if message.get("event") == "error":
                pytest.fail(message.get("msg", "websocket subscribe failed"))
        pytest.fail("did not receive a trades payload after subscribing")
