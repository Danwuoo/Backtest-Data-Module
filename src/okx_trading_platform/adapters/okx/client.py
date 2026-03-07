from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import httpx

from okx_trading_platform.domain import (
    BalanceSnapshot,
    FillRecord,
    FundingEntry,
    InstrumentConfig,
    InstrumentKind,
    OrderLifecycleState,
    OrderPlan,
    OrderState,
    PositionSnapshot,
    TdMode,
    TradingEnvironment,
    enum_value,
    utc_now,
)

from .settings import get_okx_profile_settings


class ClientOrderIdCache:
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

    def choose_submit_route(self, plan: OrderPlan) -> str:
        if self.websocket_available and enum_value(plan.order_type) == "market":
            return "ws"
        return "rest"

    def choose_cancel_route(self, *, websocket_available: bool | None = None) -> str:
        if websocket_available is None:
            websocket_available = self.websocket_available
        return "ws" if websocket_available else "rest"


class RateLimitGovernor:
    def __init__(self) -> None:
        self._windows: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, bucket: str, *, limit: int, interval_seconds: int = 2) -> bool:
        now = time.monotonic()
        queue = self._windows[bucket]
        while queue and now - queue[0] >= interval_seconds:
            queue.popleft()
        if len(queue) >= limit:
            return False
        queue.append(now)
        return True

    def snapshot(self, bucket: str) -> dict[str, int]:
        return {"bucket": bucket, "in_flight": len(self._windows[bucket])}


class OkxRestClient:
    def __init__(self, http_client: httpx.Client | None = None) -> None:
        self._client = http_client or httpx.Client(timeout=10.0)

    def request(
        self,
        *,
        environment: TradingEnvironment,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        auth: bool = False,
    ) -> httpx.Response:
        settings = get_okx_profile_settings(environment)
        headers: dict[str, str] = {}
        content = json.dumps(body) if body else ""
        if auth:
            credentials = settings.credentials
            if not all(
                [credentials.api_key, credentials.secret_key, credentials.passphrase]
            ):
                raise ValueError(
                    f"Missing credentials for environment '{enum_value(environment)}'"
                )
            timestamp = (
                utc_now().isoformat(timespec="milliseconds").replace("+00:00", "Z")
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

    def websocket_url(self, *, environment: TradingEnvironment, private: bool) -> str:
        settings = get_okx_profile_settings(environment)
        return settings.private_ws_url if private else settings.public_ws_url


@dataclass
class OkxExchangeGateway:
    rest_client: OkxRestClient
    router: OkxWebSocketRouter
    dedupe_cache: ClientOrderIdCache
    governor: RateLimitGovernor = field(default_factory=RateLimitGovernor)

    def submit_order(
        self,
        plan: OrderPlan,
        *,
        environment: TradingEnvironment = TradingEnvironment.DEMO,
        websocket_submitter: Any | None = None,
        rate_limit: int = 60,
    ) -> OrderState:
        client_order_id = plan.metadata.get("client_order_id", plan.order_plan_id)
        if self.dedupe_cache.seen(client_order_id):
            raise ValueError("Duplicate client_order_id")
        bucket = plan.rate_bucket or f"{environment.value}:{plan.inst_id}"
        if not self.governor.allow(bucket, limit=rate_limit):
            raise ValueError("Rate limit exceeded for bucket")

        route = self.router.choose_submit_route(plan)
        payload = build_okx_order_payload(plan, client_order_id=client_order_id)
        self.dedupe_cache.add(client_order_id)
        if route == "ws" and websocket_submitter is not None:
            response = websocket_submitter.submit(payload)
        else:
            response = self.rest_client.request(
                environment=environment,
                method="POST",
                path="/api/v5/trade/order",
                body=payload,
                auth=True,
            ).json()
        return normalize_order_state(plan, client_order_id, response)

    def cancel_order(
        self,
        *,
        environment: TradingEnvironment,
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
            environment=environment,
            method="POST",
            path="/api/v5/trade/cancel-order",
            body=payload,
            auth=True,
        ).json()

    def normalize_instrument(
        self, *, profile_id: str, raw: dict[str, Any]
    ) -> InstrumentConfig:
        kind = (
            InstrumentKind.SWAP
            if raw.get("instType", "").lower() == "swap"
            else InstrumentKind.SPOT
        )
        return InstrumentConfig(
            instrument_id=f"{profile_id}:{raw['instId']}",
            profile_id=profile_id,
            inst_id=raw["instId"],
            inst_id_code=raw.get("instIdCode"),
            kind=kind,
            inst_family=raw.get("instFamily"),
            base_currency=raw.get("baseCcy"),
            quote_currency=raw.get("quoteCcy"),
            settle_currency=raw.get("settleCcy"),
            tick_size=_maybe_float(raw.get("tickSz")),
            lot_size=_maybe_float(raw.get("lotSz")),
            min_size=_maybe_float(raw.get("minSz")),
            min_notional=(
                _maybe_float(raw.get("minSz")) * _maybe_float(raw.get("tickSz"))
                if raw.get("minSz") and raw.get("tickSz")
                else None
            ),
            allow_trading=raw.get("state", "live") == "live",
            metadata=raw,
        )

    def reconcile_positions(
        self, *, profile_id: str, raw_positions: list[dict[str, Any]]
    ) -> list[PositionSnapshot]:
        snapshots: list[PositionSnapshot] = []
        for raw_position in raw_positions:
            inst_type = raw_position.get("instType", "").lower()
            kind = InstrumentKind.SWAP if inst_type == "swap" else InstrumentKind.SPOT
            td_mode = (
                TdMode.ISOLATED
                if raw_position.get("mgnMode") == "isolated"
                else TdMode.CASH
            )
            snapshots.append(
                PositionSnapshot(
                    profile_id=profile_id,
                    position_snapshot_id=f"{profile_id}:{raw_position['instId']}",
                    sleeve_id=f"{profile_id}-default-sleeve",
                    instrument_id=f"{profile_id}:{raw_position['instId']}",
                    inst_id=raw_position["instId"],
                    kind=kind,
                    quantity=float(raw_position.get("pos", 0.0) or 0.0),
                    avg_price=_maybe_float(raw_position.get("avgPx")),
                    unrealized_pnl=float(raw_position.get("upl", 0.0) or 0.0),
                    td_mode=td_mode,
                )
            )
        return snapshots

    def reconcile_balances(
        self, *, profile_id: str, raw_balances: list[dict[str, Any]]
    ) -> list[BalanceSnapshot]:
        snapshots: list[BalanceSnapshot] = []
        for raw_balance in raw_balances:
            snapshots.append(
                BalanceSnapshot(
                    balance_snapshot_id=f"{profile_id}:{raw_balance['ccy']}",
                    profile_id=profile_id,
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
        profile_id: str,
        order_id: str,
        inst_id: str,
        raw_fills: list[dict[str, Any]],
    ) -> list[FillRecord]:
        fills: list[FillRecord] = []
        for raw_fill in raw_fills:
            fills.append(
                FillRecord(
                    order_id=order_id,
                    profile_id=profile_id,
                    instrument_id=f"{profile_id}:{inst_id}",
                    inst_id=inst_id,
                    fill_price=float(raw_fill.get("fillPx", 0.0) or 0.0),
                    fill_size=float(raw_fill.get("fillSz", 0.0) or 0.0),
                    fee=float(raw_fill.get("fee", 0.0) or 0.0),
                    raw_payload=raw_fill,
                )
            )
        return fills

    def normalize_funding(
        self,
        *,
        profile_id: str,
        inst_id: str,
        raw_entry: dict[str, Any],
    ) -> FundingEntry:
        return FundingEntry(
            profile_id=profile_id,
            instrument_id=f"{profile_id}:{inst_id}",
            inst_id=inst_id,
            amount=float(raw_entry.get("fundingFee", 0.0) or 0.0),
            rate=_maybe_float(raw_entry.get("fundingRate")),
            metadata=raw_entry,
        )


def build_okx_order_payload(plan: OrderPlan, *, client_order_id: str) -> dict[str, Any]:
    payload = {
        "instId": plan.inst_id,
        "side": enum_value(plan.side),
        "ordType": enum_value(plan.order_type),
        "sz": str(plan.size),
        "clOrdId": client_order_id,
        "tdMode": enum_value(plan.td_mode),
    }
    if enum_value(plan.order_type) == "limit" and plan.price is not None:
        payload["px"] = str(plan.price)
    if plan.metadata.get("exp_time"):
        payload["expTime"] = str(plan.metadata["exp_time"])
    return payload


def normalize_order_state(
    plan: OrderPlan,
    client_order_id: str,
    raw_response: dict[str, Any],
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
        client_order_id=client_order_id,
        order_plan_id=plan.order_plan_id,
        profile_id=plan.profile_id,
        strategy_id=plan.strategy_id,
        model_version_id=plan.model_version_id,
        sleeve_id=plan.sleeve_id,
        instrument_id=plan.instrument_id,
        inst_id=plan.inst_id,
        kind=plan.kind,
        side=plan.side,
        size=plan.size,
        price=plan.price,
        order_type=plan.order_type,
        td_mode=plan.td_mode,
        status=status,
        exchange_order_id=exchange_order_id,
        source=plan.source,
        rejection_reason=rejection_reason,
        raw_payload=raw_response,
    )


def _maybe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
