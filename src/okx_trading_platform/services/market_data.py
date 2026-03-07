from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import httpx
import websockets
from fastapi import FastAPI

from okx_trading_platform.adapters.okx import (
    ClientOrderIdCache,
    OkxExchangeGateway,
    OkxRestClient,
    OkxWebSocketRouter,
    build_market_data_ingestion_plan,
    build_okx_trade_fee_params,
)
from okx_trading_platform.domain import InstrumentKind
from okx_trading_platform.shared.data_lake import DataLakeWriter
from okx_trading_platform.shared.runtime import MarketDataRuntime
from okx_trading_platform.shared.settings import PlatformSettings, get_platform_settings

settings = get_platform_settings()
ingestion_plan = build_market_data_ingestion_plan(settings)
rest_client = OkxRestClient()
gateway = OkxExchangeGateway(
    rest_client=rest_client,
    router=OkxWebSocketRouter(),
    dedupe_cache=ClientOrderIdCache(),
)
runtime = MarketDataRuntime(
    service_name="market-data-service",
    profile_id=settings.baseline_profile_id,
    environment=settings.trading_environment,
    data_lake=DataLakeWriter(
        settings.platform_data_root,
        settings.duckdb_path,
        hot_cache_root=settings.hot_cache_root,
        object_store_backend=settings.object_store_backend,
        object_store_bucket=settings.object_store_bucket,
        object_store_endpoint=settings.object_store_endpoint,
        object_store_region=settings.object_store_region,
        object_store_access_key_id=settings.object_store_access_key_id,
        object_store_secret_access_key=settings.object_store_secret_access_key,
        object_store_prefix=settings.object_store_prefix,
        checkpoint_root=settings.checkpoint_root,
    ),
)
runtime.set_running()


class MarketDataWorker:
    def __init__(
        self,
        *,
        settings: PlatformSettings,
        runtime: MarketDataRuntime,
        gateway: OkxExchangeGateway,
    ) -> None:
        self.settings = settings
        self.runtime = runtime
        self.gateway = gateway
        self.tasks: list[asyncio.Task] = []
        self.stats: dict[str, Any] = {
            "enabled": settings.enable_market_data_worker,
            "public_batches": 0,
            "rest_cycles": 0,
            "rows_written": 0,
            "last_error": None,
            "last_error_at": None,
            "last_success_at": None,
        }

    def _active_whitelist(self) -> tuple[str, ...]:
        return self.settings.okx_inst_id_whitelist or (self.settings.baseline_inst_id,)

    def _has_private_credentials(self) -> bool:
        from okx_trading_platform.adapters.okx import get_okx_profile_settings

        credentials = get_okx_profile_settings(self.settings.trading_environment).credentials
        return bool(
            credentials.api_key and credentials.secret_key and credentials.passphrase
        )

    async def start(self) -> None:
        if not self.settings.enable_market_data_worker:
            return
        self.tasks = [
            asyncio.create_task(
                self._run_loop(
                    "instrument-refresh",
                    self._refresh_instruments,
                    self.settings.okx_instrument_refresh_seconds,
                )
            ),
            asyncio.create_task(
                self._run_loop(
                    "funding-refresh",
                    self._refresh_funding,
                    self.settings.okx_funding_poll_seconds,
                )
            ),
            asyncio.create_task(
                self._run_loop(
                    "public-capture",
                    self._capture_public_batches,
                    max(5, self.settings.okx_trade_flush_interval_seconds),
                )
            ),
            asyncio.create_task(
                self._run_loop(
                    "account-sync",
                    self._sync_private_account,
                    self.settings.okx_account_poll_seconds,
                )
            ),
            asyncio.create_task(
                self._run_loop(
                    "compaction",
                    self._compact_lake,
                    self.settings.compaction_interval_seconds,
                )
            ),
        ]

    async def stop(self) -> None:
        for task in self.tasks:
            task.cancel()
        for task in self.tasks:
            with suppress(asyncio.CancelledError):
                await task
        self.tasks.clear()

    async def _run_loop(self, job_name: str, func, interval_seconds: int) -> None:
        while True:
            try:
                await func()
                self.stats["rest_cycles"] += 1
                self.stats["last_success_at"] = datetime.now(timezone.utc).isoformat()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - exercised via service integration
                await self._report_failure(job_name, exc)
            await asyncio.sleep(max(1, interval_seconds))

    async def _refresh_instruments(self) -> None:
        rows: list[dict[str, Any]] = []
        whitelist = set(self._active_whitelist())
        for kind in self.settings.okx_public_instrument_kinds:
            response = await asyncio.to_thread(
                self.gateway.rest_client.request,
                environment=self.settings.trading_environment,
                method="GET",
                path="/api/v5/public/instruments",
                params={"instType": kind.value.upper()},
                auth=False,
            )
            payload = response.json().get("data", [])
            for raw in payload:
                if whitelist and raw.get("instId") not in whitelist:
                    continue
                instrument = self.gateway.normalize_instrument(
                    profile_id=self.runtime.profile_id,
                    raw=raw,
                )
                rows.append(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "inst_id": instrument.inst_id,
                        "instrument_id": instrument.instrument_id,
                        "kind": instrument.kind,
                        "inst_family": instrument.inst_family,
                        "tick_size": instrument.tick_size,
                        "lot_size": instrument.lot_size,
                        "min_size": instrument.min_size,
                        "allow_trading": instrument.allow_trading,
                    }
                )
        if rows:
            self.runtime.persist_rows(
                layer="silver",
                stream="instrument_meta",
                rows=rows,
                logical_name="instrument_meta",
            )
            self.stats["rows_written"] += len(rows)

    async def _refresh_funding(self) -> None:
        rows: list[dict[str, Any]] = []
        for inst_id in self._active_whitelist():
            if not inst_id.endswith("-SWAP"):
                continue
            response = await asyncio.to_thread(
                self.gateway.rest_client.request,
                environment=self.settings.trading_environment,
                method="GET",
                path="/api/v5/public/funding-rate-history",
                params={"instId": inst_id, "limit": "1"},
                auth=False,
            )
            for raw in response.json().get("data", []):
                rows.append(
                    {
                        "funding_ts": raw.get("fundingTime"),
                        "inst_id": inst_id,
                        "rate": float(raw.get("fundingRate", 0.0) or 0.0),
                        "realized_amount": float(raw.get("realizedRate", 0.0) or 0.0),
                    }
                )
        if rows:
            self.runtime.persist_rows(
                layer="silver",
                stream="funding",
                rows=rows,
                logical_name="funding",
            )
            self.stats["rows_written"] += len(rows)

    async def _capture_public_batches(self) -> None:
        whitelist = self._active_whitelist()
        if not whitelist:
            return
        trade_rows = await self._capture_public_ws_once(channel="trades", inst_ids=whitelist)
        if trade_rows:
            self.runtime.persist_rows(
                layer="silver",
                stream="trades",
                rows=trade_rows,
                logical_name="trades",
            )
            bars = self._rollup_bars(trade_rows)
            if bars:
                self.runtime.persist_rows(
                    layer="silver",
                    stream="bars_1s",
                    rows=bars,
                    logical_name="bars_1s",
                )
            self.stats["rows_written"] += len(trade_rows) + len(bars)
        tier_a_rows = await self._capture_public_ws_once(
            channel="books5",
            inst_ids=self.settings.okx_tier_a_inst_ids or whitelist,
        )
        if tier_a_rows:
            self.runtime.persist_rows(
                layer="silver",
                stream="books5_1s",
                rows=tier_a_rows,
                logical_name="books5_1s",
            )
            self.stats["rows_written"] += len(tier_a_rows)
        tier_b_rows = [self._books5_to_tob(row) for row in tier_a_rows if row["inst_id"] in set(self.settings.okx_tier_b_inst_ids)]
        if tier_b_rows:
            self.runtime.persist_rows(
                layer="silver",
                stream="tob_1s",
                rows=tier_b_rows,
                logical_name="tob_1s",
            )
            self.stats["rows_written"] += len(tier_b_rows)
        self.stats["public_batches"] += 1

    async def _sync_private_account(self) -> None:
        if not self._has_private_credentials():
            return
        balances_response = await asyncio.to_thread(
            self.gateway.rest_client.request,
            environment=self.settings.trading_environment,
            method="GET",
            path="/api/v5/account/balance",
            auth=True,
        )
        balance_rows: list[dict[str, Any]] = []
        for entry in balances_response.json().get("data", []):
            for detail in entry.get("details", []):
                balance_rows.append(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "profile_id": self.runtime.profile_id,
                        "currency": detail.get("ccy"),
                        "available": float(detail.get("availBal", 0.0) or 0.0),
                        "cash_balance": float(detail.get("cashBal", 0.0) or 0.0),
                        "equity": float(detail.get("eq", 0.0) or 0.0),
                    }
                )
        if balance_rows:
            self.runtime.persist_rows(
                layer="silver",
                stream="balances",
                rows=balance_rows,
                logical_name="balances",
            )
        positions_response = await asyncio.to_thread(
            self.gateway.rest_client.request,
            environment=self.settings.trading_environment,
            method="GET",
            path="/api/v5/account/positions",
            auth=True,
        )
        position_rows = [
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "profile_id": self.runtime.profile_id,
                "inst_id": item.get("instId"),
                "qty": float(item.get("pos", 0.0) or 0.0),
                "avg_px": float(item.get("avgPx", 0.0) or 0.0),
                "upl": float(item.get("upl", 0.0) or 0.0),
                "leverage": float(item.get("lever", 0.0) or 0.0),
                "margin_mode": item.get("mgnMode"),
            }
            for item in positions_response.json().get("data", [])
        ]
        if position_rows:
            self.runtime.persist_rows(
                layer="silver",
                stream="positions",
                rows=position_rows,
                logical_name="positions",
            )
        fills_response = await asyncio.to_thread(
            self.gateway.rest_client.request,
            environment=self.settings.trading_environment,
            method="GET",
            path="/api/v5/trade/fills-history",
            params={"instType": self._default_inst_type(), "limit": "100"},
            auth=True,
        )
        fill_rows = [
            {
                "ts": item.get("ts"),
                "profile_id": self.runtime.profile_id,
                "inst_id": item.get("instId"),
                "order_id": item.get("ordId"),
                "trade_id": item.get("tradeId"),
                "px": float(item.get("fillPx", 0.0) or 0.0),
                "sz": float(item.get("fillSz", 0.0) or 0.0),
                "fee": float(item.get("fee", 0.0) or 0.0),
                "liquidity_flag": item.get("execType"),
            }
            for item in fills_response.json().get("data", [])
        ]
        if fill_rows:
            self.runtime.persist_rows(
                layer="silver",
                stream="fills",
                rows=fill_rows,
                logical_name="fills",
            )
        first_inst_id = self._active_whitelist()[0]
        fee_response = await asyncio.to_thread(
            self.gateway.rest_client.request,
            environment=self.settings.trading_environment,
            method="GET",
            path="/api/v5/account/trade-fee",
            params=build_okx_trade_fee_params(
                inst_type=self._default_inst_type(),
                inst_id=first_inst_id,
            ),
            auth=True,
        )
        fee_rows = [
            {
                "effective_from": datetime.now(timezone.utc).isoformat(),
                "inst_id": first_inst_id,
                "maker_fee": float(item.get("maker", 0.0) or 0.0),
                "taker_fee": float(item.get("taker", 0.0) or 0.0),
                "tier": item.get("level"),
            }
            for item in fee_response.json().get("data", [])
        ]
        if fee_rows:
            self.runtime.persist_rows(
                layer="silver",
                stream="fee_schedule",
                rows=fee_rows,
                logical_name="fee_schedule",
            )
        self.stats["rows_written"] += (
            len(balance_rows) + len(position_rows) + len(fill_rows) + len(fee_rows)
        )

    async def _compact_lake(self) -> None:
        self.runtime.data_lake.compact()

    async def _capture_public_ws_once(
        self,
        *,
        channel: str,
        inst_ids: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        if not inst_ids:
            return []
        url = self.gateway.rest_client.websocket_url(
            environment=self.settings.trading_environment,
            private=False,
        )
        rows: list[dict[str, Any]] = []
        async with websockets.connect(url, ping_interval=None, close_timeout=2) as ws:
            args = [{"channel": channel, "instId": inst_id} for inst_id in inst_ids]
            await ws.send(json.dumps({"op": "subscribe", "args": args}))
            deadline = asyncio.get_running_loop().time() + 4
            while asyncio.get_running_loop().time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1)
                except asyncio.TimeoutError:
                    break
                message = json.loads(raw)
                self.runtime.persist_rows(
                    layer="bronze",
                    stream="ws_raw",
                    rows=[
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "channel": channel,
                            "message": message,
                        }
                    ],
                    logical_name="ws_raw",
                )
                if message.get("event") == "error":
                    raise RuntimeError(message.get("msg", f"Failed to subscribe to {channel}"))
                if message.get("event") == "subscribe":
                    continue
                arg = message.get("arg", {})
                if arg.get("channel") != channel:
                    continue
                if channel == "trades":
                    rows.extend(self._trade_rows(arg.get("instId"), message.get("data", [])))
                elif channel == "books5":
                    rows.extend(self._books5_rows(arg.get("instId"), message))
                if rows:
                    break
        return rows

    def _trade_rows(self, inst_id: str | None, payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in payload:
            rows.append(
                {
                    "ts": self._parse_ts(item.get("ts")),
                    "inst_id": inst_id,
                    "trade_id": item.get("tradeId"),
                    "px": float(item.get("px", 0.0) or 0.0),
                    "sz": float(item.get("sz", 0.0) or 0.0),
                    "side": item.get("side"),
                    "is_taker": item.get("execType", "").lower() in {"t", "maker"},
                    "sequence_id": int(item.get("seqId", 0) or 0),
                }
            )
        return rows

    def _books5_rows(self, inst_id: str | None, message: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        action = message.get("action", "snapshot")
        for item in message.get("data", []):
            bids = item.get("bids", [])
            asks = item.get("asks", [])
            if action == "snapshot":
                snapshot = self.runtime.upsert_snapshot(
                    inst_id or "",
                    bids,
                    asks,
                    sequence_id=int(item.get("seqId", 0) or 0),
                )
            else:
                snapshot = self.runtime.apply_delta(
                    inst_id or "",
                    bids,
                    asks,
                    sequence_id=int(item.get("seqId", 0) or 0),
                    prev_sequence_id=int(item.get("prevSeqId", 0) or 0),
                )
                self.runtime.persist_rows(
                    layer="bronze",
                    stream="book_delta_raw",
                    rows=[
                        {
                            "ts": self._parse_ts(item.get("ts")),
                            "inst_id": inst_id,
                            "message": item,
                        }
                    ],
                    inst_id=inst_id,
                    logical_name="book_delta_raw",
                )
            row = {
                "ts": self._parse_ts(item.get("ts")),
                "inst_id": snapshot.inst_id,
                "mid_px": self._mid_price(snapshot),
                "spread_bps": self._spread_bps(snapshot),
            }
            for level in range(5):
                bid = snapshot.bids[level] if len(snapshot.bids) > level else None
                ask = snapshot.asks[level] if len(snapshot.asks) > level else None
                row[f"bid_px{level + 1}"] = None if bid is None else bid.price
                row[f"bid_sz{level + 1}"] = None if bid is None else bid.size
                row[f"ask_px{level + 1}"] = None if ask is None else ask.price
                row[f"ask_sz{level + 1}"] = None if ask is None else ask.size
            rows.append(row)
        return rows

    @staticmethod
    def _books5_to_tob(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "ts": row["ts"],
            "inst_id": row["inst_id"],
            "bid_px1": row.get("bid_px1"),
            "bid_sz1": row.get("bid_sz1"),
            "ask_px1": row.get("ask_px1"),
            "ask_sz1": row.get("ask_sz1"),
            "mid_px": row.get("mid_px"),
            "spread_bps": row.get("spread_bps"),
        }

    @staticmethod
    def _rollup_bars(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            ts = str(row["ts"])[:19]
            buckets[(row["inst_id"], ts)].append(row)
        bars: list[dict[str, Any]] = []
        for (inst_id, ts), items in sorted(buckets.items()):
            prices = [float(item["px"]) for item in items]
            sizes = [float(item["sz"]) for item in items]
            buy_vol = sum(size for item, size in zip(items, sizes) if item["side"] == "buy")
            sell_vol = sum(
                size for item, size in zip(items, sizes) if item["side"] == "sell"
            )
            bars.append(
                {
                    "ts": ts,
                    "inst_id": inst_id,
                    "open": prices[0],
                    "high": max(prices),
                    "low": min(prices),
                    "close": prices[-1],
                    "vol_base": sum(sizes),
                    "vol_quote": sum(price * size for price, size in zip(prices, sizes)),
                    "trade_count": len(items),
                    "buy_vol": buy_vol,
                    "sell_vol": sell_vol,
                    "mark_px": prices[-1],
                    "index_px": prices[-1],
                }
            )
        return bars

    @staticmethod
    def _mid_price(snapshot) -> float | None:
        if not snapshot.bids or not snapshot.asks:
            return None
        return (snapshot.bids[0].price + snapshot.asks[0].price) / 2

    @staticmethod
    def _spread_bps(snapshot) -> float | None:
        mid = MarketDataWorker._mid_price(snapshot)
        if mid in (None, 0) or not snapshot.bids or not snapshot.asks:
            return None
        spread = snapshot.asks[0].price - snapshot.bids[0].price
        return (spread / mid) * 10_000

    @staticmethod
    def _parse_ts(value: Any) -> str | None:
        if value in (None, ""):
            return None
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).isoformat()

    def _default_inst_type(self) -> str:
        baseline_kind = self.settings.baseline_instrument_kind
        if isinstance(baseline_kind, InstrumentKind):
            return baseline_kind.value.upper()
        return str(baseline_kind).upper()

    async def _report_failure(self, job_name: str, exc: Exception) -> None:
        self.stats["last_error"] = f"{job_name}: {exc}"
        self.stats["last_error_at"] = datetime.now(timezone.utc).isoformat()
        try:
            async with httpx.AsyncClient(
                base_url=self.settings.control_api_url,
                timeout=10.0,
            ) as client:
                await client.post(
                    "/incidents",
                    json={
                        "profile_id": self.runtime.profile_id,
                        "severity": "warning",
                        "title": f"market-data {job_name} failure",
                        "message": str(exc),
                        "metadata": {"job": job_name},
                    },
                )
        except Exception:
            pass


worker = MarketDataWorker(settings=settings, runtime=runtime, gateway=gateway)


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    await worker.start()
    try:
        yield
    finally:
        await worker.stop()


app = FastAPI(title="OKX Market Data Service", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict:
    return {
        "service": runtime.service_name,
        "status": runtime.status,
        "profile_id": runtime.profile_id,
        "environment": runtime.environment,
        "order_books": len(runtime.books),
        "whitelist_source": ingestion_plan.whitelist_source,
        "whitelist_symbols": len(ingestion_plan.whitelist),
        "worker": worker.stats,
    }


@app.get("/heartbeat")
def heartbeat() -> dict:
    return runtime.heartbeat().model_dump(mode="json")


@app.get("/ingestion-plan")
def get_ingestion_plan() -> dict:
    return asdict(ingestion_plan)
