from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any

import httpx

from okx_trading_platform.domain import (
    BalanceSnapshot,
    FillRecord,
    InstrumentKind,
    OrderIntent,
    OrderLifecycleState,
    OrderState,
    OrderType,
    PositionSnapshot,
    TdMode,
    TradingProfile,
    enum_value,
    utc_now,
)

from .settings import get_okx_profile_settings


class ClientOrderIdCache:
    """Track submitted client order IDs to avoid duplicate sends."""

    def __init__(self) -> None:
        self._submitted: set[str] = set()

    def seen(self, client_order_id: str) -> bool:
        return client_order_id in self._submitted

    def add(self, client_order_id: str) -> None:
        self._submitted.add(client_order_id)


class OkxRequestSigner:
    @staticmethod
    def sign(
        *,
        secret_key: str,
        timestamp: str,
        method: str,
        request_path: str,
        body: str = "",
    ) -> str:
        payload = f"{timestamp}{method.upper()}{request_path}{body}"
        digest = hmac.new(
            secret_key.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(digest).decode("utf-8")


class OkxWebSocketRouter:
    def __init__(self, *, websocket_available: bool = True) -> None:
        self.websocket_available = websocket_available

    def choose_submit_route(self, intent: OrderIntent) -> str:
        if self.websocket_available and intent.order_type == OrderType.MARKET:
            return "ws"
        return "rest"

    def choose_cancel_route(self, *, websocket_available: bool | None = None) -> str:
        if websocket_available is None:
            websocket_available = self.websocket_available
        return "ws" if websocket_available else "rest"


class OkxRestClient:
    def __init__(self, http_client: httpx.Client | None = None) -> None:
        self._client = http_client or httpx.Client(timeout=10.0)

    def request(
        self,
        *,
        profile: TradingProfile,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        auth: bool = False,
    ) -> httpx.Response:
        settings = get_okx_profile_settings(profile)
        headers: dict[str, str] = {}
        content = json.dumps(body) if body else ""
        if auth:
            credentials = settings.credentials
            if not all(
                [credentials.api_key, credentials.secret_key, credentials.passphrase]
            ):
                raise ValueError(
                    f"Missing credentials for profile '{enum_value(profile)}'"
                )
            timestamp = utc_now().isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            )
            headers.update(
                {
                    "OK-ACCESS-KEY": credentials.api_key or "",
                    "OK-ACCESS-PASSPHRASE": credentials.passphrase or "",
                    "OK-ACCESS-TIMESTAMP": timestamp,
                    "OK-ACCESS-SIGN": OkxRequestSigner.sign(
                        secret_key=credentials.secret_key or "",
                        timestamp=timestamp,
                        method=method,
                        request_path=path,
                        body=content,
                    ),
                }
            )
        if settings.simulated_trading:
            headers["x-simulated-trading"] = "1"
        return self._client.request(
            method=method.upper(),
            url=f"{settings.rest_base_url}{path}",
            params=params,
            content=content or None,
            headers=headers,
        )

    def websocket_url(self, *, profile: TradingProfile, private: bool) -> str:
        settings = get_okx_profile_settings(profile)
        return settings.private_ws_url if private else settings.public_ws_url


@dataclass
class OkxExchangeGateway:
    rest_client: OkxRestClient
    router: OkxWebSocketRouter
    dedupe_cache: ClientOrderIdCache

    def submit_order(
        self,
        intent: OrderIntent,
        *,
        websocket_submitter: Any | None = None,
    ) -> OrderState:
        if self.dedupe_cache.seen(intent.client_order_id):
            raise ValueError("Duplicate client_order_id")
        route = self.router.choose_submit_route(intent)
        payload = build_okx_order_payload(intent)
        self.dedupe_cache.add(intent.client_order_id)
        if route == "ws" and websocket_submitter is not None:
            response = websocket_submitter.submit(payload)
        else:
            response = self.rest_client.request(
                profile=intent.profile,
                method="POST",
                path="/api/v5/trade/order",
                body=payload,
                auth=True,
            ).json()
        return normalize_order_state(intent, response)

    def cancel_order(
        self,
        *,
        profile: TradingProfile,
        inst_id: str,
        order_id: str | None = None,
        client_order_id: str | None = None,
        websocket_submitter: Any | None = None,
    ) -> dict[str, Any]:
        route = self.router.choose_cancel_route()
        payload = {"instId": inst_id}
        if order_id:
            payload["ordId"] = order_id
        if client_order_id:
            payload["clOrdId"] = client_order_id
        if route == "ws" and websocket_submitter is not None:
            return websocket_submitter.cancel(payload)
        return self.rest_client.request(
            profile=profile,
            method="POST",
            path="/api/v5/trade/cancel-order",
            body=payload,
            auth=True,
        ).json()

    def reconcile_positions(
        self, *, profile: TradingProfile, raw_positions: list[dict[str, Any]]
    ) -> list[PositionSnapshot]:
        snapshots: list[PositionSnapshot] = []
        for raw_position in raw_positions:
            inst_type = raw_position.get("instType", "").lower()
            kind = (
                InstrumentKind.SWAP if inst_type == "swap" else InstrumentKind.SPOT
            )
            td_mode = (
                TdMode.ISOLATED
                if raw_position.get("mgnMode") == "isolated"
                else TdMode.CASH
            )
            snapshots.append(
                PositionSnapshot(
                    inst_id=raw_position["instId"],
                    profile=profile,
                    instrument_kind=kind,
                    quantity=float(raw_position.get("pos", 0.0)),
                    avg_price=_maybe_float(raw_position.get("avgPx")),
                    unrealized_pnl=float(raw_position.get("upl", 0.0) or 0.0),
                    td_mode=td_mode,
                )
            )
        return snapshots

    def reconcile_balances(
        self, *, profile: TradingProfile, raw_balances: list[dict[str, Any]]
    ) -> list[BalanceSnapshot]:
        snapshots: list[BalanceSnapshot] = []
        for raw_balance in raw_balances:
            snapshots.append(
                BalanceSnapshot(
                    profile=profile,
                    currency=raw_balance["ccy"],
                    available=float(raw_balance.get("availBal", 0.0) or 0.0),
                    cash_balance=float(raw_balance.get("cashBal", 0.0) or 0.0),
                    equity=float(raw_balance.get("eq", 0.0) or 0.0),
                )
            )
        return snapshots

    def normalize_fills(
        self,
        *,
        profile: TradingProfile,
        order_id: str,
        inst_id: str,
        raw_fills: list[dict[str, Any]],
    ) -> list[FillRecord]:
        fills: list[FillRecord] = []
        for raw_fill in raw_fills:
            fills.append(
                FillRecord(
                    order_id=order_id,
                    inst_id=inst_id,
                    profile=profile,
                    fill_price=float(raw_fill.get("fillPx", 0.0) or 0.0),
                    fill_size=float(raw_fill.get("fillSz", 0.0) or 0.0),
                    fee=float(raw_fill.get("fee", 0.0) or 0.0),
                    raw_payload=raw_fill,
                )
            )
        return fills


def build_okx_order_payload(intent: OrderIntent) -> dict[str, Any]:
    payload = {
        "instId": intent.inst_id,
        "side": enum_value(intent.side),
        "ordType": enum_value(intent.order_type),
        "sz": str(intent.size),
        "clOrdId": intent.client_order_id,
        "tdMode": enum_value(intent.td_mode),
    }
    if intent.order_type == OrderType.LIMIT and intent.price is not None:
        payload["px"] = str(intent.price)
    return payload


def normalize_order_state(
    intent: OrderIntent, raw_response: dict[str, Any]
) -> OrderState:
    data = raw_response.get("data", [])
    first = data[0] if data else {}
    state = first.get("sCode")
    status = OrderLifecycleState.SUBMITTED
    rejection_reason = None
    exchange_order_id = first.get("ordId")
    if state and state != "0":
        status = OrderLifecycleState.FAILED
        rejection_reason = first.get("sMsg") or raw_response.get("msg")
    return OrderState(
        order_id=intent.order_id,
        client_order_id=intent.client_order_id,
        profile=intent.profile,
        inst_id=intent.inst_id,
        instrument_kind=intent.instrument_kind,
        side=intent.side,
        size=intent.size,
        price=intent.price,
        order_type=intent.order_type,
        td_mode=intent.td_mode,
        status=status,
        exchange_order_id=exchange_order_id,
        bot_name=intent.bot_name,
        rejection_reason=rejection_reason,
        raw_payload=raw_response,
    )


def _maybe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
