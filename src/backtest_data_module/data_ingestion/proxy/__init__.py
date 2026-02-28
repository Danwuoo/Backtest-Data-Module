from __future__ import annotations

import logging
import json
import time
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.responses import Response
import httpx
from tenacity import AsyncRetrying, wait_random_exponential, stop_after_attempt

from backtest_data_module.data_ingestion.py.rate_limiter import RateLimiter
from backtest_data_module.data_ingestion.py.redis_rate_limiter import (
    RedisRateLimiter,
)
from backtest_data_module.data_ingestion.py.caching import ICache, LRUCache


logger = logging.getLogger(__name__)


def create_proxy_app(
    target_base_url: str,
    *,
    rate_limiter: RateLimiter | RedisRateLimiter | None = None,
    cache: ICache | None = None,
) -> FastAPI:
    """建立反向 proxy FastAPI 應用程式。"""

    limiter = rate_limiter or RateLimiter(calls=5, period=1)
    proxy_cache = cache or LRUCache(capacity=128, ttl=60)
    app = FastAPI()
    client = httpx.AsyncClient(base_url=target_base_url)

    async def _forward(
        method: str, url: str, params: dict
    ) -> tuple[httpx.Response, int]:
        retryer = AsyncRetrying(
            wait=wait_random_exponential(min=1, max=4),
            stop=stop_after_attempt(3),
            reraise=True,
        )

        async def _do_request() -> httpx.Response:
            resp = await client.request(method, url, params=params)
            resp.raise_for_status()
            return resp

        resp = await retryer(_do_request)
        attempts = int(retryer.statistics.get("attempt_number", 1))
        return resp, attempts

    @app.api_route("/{path:path}", methods=["GET"])  # 簡化僅支援 GET
    async def proxy(path: str, request: Request) -> Response:
        start = time.monotonic()
        await limiter.acquire()
        cache_key = f"{path}?{request.query_params}"
        cached = await proxy_cache.get(cache_key)
        if cached is not None:
            latency = time.monotonic() - start
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "path": request.url.path,
                "status": cached["status"],
                "latency": latency,
                "retries": 0,
                "remaining_quota": limiter.remaining_tokens,
            }
            logger.info(json.dumps(log_data, ensure_ascii=False))
            return Response(
                content=cached["content"],
                status_code=cached["status"],
                media_type=cached["media_type"],
            )
        try:
            resp, attempts = await _forward(
                request.method,
                f"/{path}",
                dict(request.query_params),
            )
        except httpx.HTTPError as exc:  # pragma: no cover - 轉換成 502
            latency = time.monotonic() - start
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "path": request.url.path,
                "status": 502,
                "latency": latency,
                "retries": attempts if 'attempts' in locals() else 1,
                "remaining_quota": limiter.remaining_tokens,
            }
            logger.error(json.dumps(log_data, ensure_ascii=False))
            logger.error("proxy error: %s", exc)
            return Response(content=str(exc), status_code=502)
        await proxy_cache.set(
            cache_key,
            {
                "content": resp.content,
                "status": resp.status_code,
                "media_type": resp.headers.get("content-type"),
            },
        )
        latency = time.monotonic() - start
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path,
            "status": resp.status_code,
            "latency": latency,
            "retries": attempts,
            "remaining_quota": limiter.remaining_tokens,
        }
        logger.info(json.dumps(log_data, ensure_ascii=False))
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await client.aclose()

    return app
