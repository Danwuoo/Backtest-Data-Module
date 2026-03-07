from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import redis


@dataclass
class InMemoryStreamBus:
    messages: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def publish(self, stream: str, payload: dict[str, Any]) -> None:
        self.messages.setdefault(stream, []).append(payload)

    def read(self, stream: str) -> list[dict[str, Any]]:
        return list(self.messages.get(stream, []))


class RedisStreamBus:
    def __init__(self, redis_url: str) -> None:
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)

    def publish(self, stream: str, payload: dict[str, Any]) -> None:
        self.client.xadd(stream, {"payload": json.dumps(payload, default=str)})

    def read(self, stream: str, count: int = 10) -> list[dict[str, Any]]:
        entries = self.client.xrevrange(stream, count=count)
        return [json.loads(item["payload"]) for _, item in entries]
