from __future__ import annotations

import asyncio
import time
from redis.asyncio import Redis

from backtest_data_module.data_ingestion.metrics import (
    RATE_LIMIT_429_COUNTER,
    REMAINING_GAUGE,
)

_LUA_SCRIPT = """
local calls = tonumber(ARGV[1])
local period = tonumber(ARGV[2])
local burst = tonumber(ARGV[3])
local now = tonumber(ARGV[4])
local ttl = tonumber(ARGV[5])
local rate = calls / period
local data = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
if not tokens then
  tokens = burst
  ts = now
end
if ts == nil then ts = now end
local elapsed = now - ts
if elapsed < 0 then elapsed = 0 end
tokens = math.min(burst, tokens + elapsed * rate)
local allowed = 0
local wait = 0
if tokens >= 1 then
  allowed = 1
  tokens = tokens - 1
else
  wait = (1 - tokens) / rate
end
redis.call('HMSET', KEYS[1], 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', KEYS[1], ttl)
return {allowed, tokens, wait}
"""


class RedisRateLimiter:
    """利用 Redis 網路共用 token bucket 紀錄。"""

    def __init__(
        self,
        *,
        redis: Redis,
        key: str,
        calls: int,
        period: float,
        burst: int | None = None,
        fail_threshold: int = 3,
    ) -> None:
        self.redis = redis
        self.key = key
        self.calls = calls
        self.period = period
        self.burst = burst or calls
        self.fail_threshold = fail_threshold
        self.fail_count = 0
        self._tokens = float(self.burst)

    async def acquire(self) -> None:
        while True:
            allowed, tokens, wait = await self.redis.eval(
                _LUA_SCRIPT,
                1,
                self.key,
                self.calls,
                self.period,
                self.burst,
                time.time(),
                int(self.period * 2),
            )
            self._tokens = float(tokens)
            if int(allowed) == 1:
                REMAINING_GAUGE.labels(endpoint=self.key).set(self._tokens)
                return
            await asyncio.sleep(float(wait))

    def record_failure(
        self, *, status_code: int | None = None, timeout: bool = False
    ) -> None:
        if status_code == 429 or timeout:
            self.fail_count += 1
            if status_code == 429:
                RATE_LIMIT_429_COUNTER.labels(endpoint=self.key).inc()
        else:
            self.fail_count = 0

        if self.fail_count >= self.fail_threshold:
            self.calls = max(1, int(self.calls * 0.8))
            self.burst = max(1, int(self.burst * 0.8))
            self.fail_count = 0

    @property
    def remaining_tokens(self) -> float:
        return self._tokens


__all__ = ["RedisRateLimiter"]
