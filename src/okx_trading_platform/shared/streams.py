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


class RedpandaStreamBus:
    def __init__(self, *, brokers: tuple[str, ...], topic_prefix: str = "") -> None:
        self.brokers = ",".join(brokers)
        self.topic_prefix = topic_prefix.strip(".")
        self._producer = None
        self._consumer = None

    def _topic(self, stream: str) -> str:
        if not self.topic_prefix:
            return stream
        return f"{self.topic_prefix}.{stream}"

    def _ensure_producer(self):
        if self._producer is None:
            from kafka import KafkaProducer

            self._producer = KafkaProducer(
                bootstrap_servers=self.brokers,
                value_serializer=lambda value: json.dumps(value, default=str).encode(
                    "utf-8"
                ),
            )
        return self._producer

    def _ensure_consumer(self):
        if self._consumer is None:
            from kafka import KafkaConsumer

            self._consumer = KafkaConsumer(
                bootstrap_servers=self.brokers,
                auto_offset_reset="earliest",
                enable_auto_commit=False,
                consumer_timeout_ms=1000,
                value_deserializer=lambda value: json.loads(value.decode("utf-8")),
            )
        return self._consumer

    def publish(self, stream: str, payload: dict[str, Any]) -> None:
        producer = self._ensure_producer()
        future = producer.send(self._topic(stream), payload)
        future.get(timeout=10)

    def read(self, stream: str, count: int = 10) -> list[dict[str, Any]]:
        consumer = self._ensure_consumer()
        consumer.subscribe([self._topic(stream)])
        results: list[dict[str, Any]] = []
        for message in consumer:
            results.append(message.value)
            if len(results) >= count:
                break
        return results


def build_stream_bus(
    *,
    backend: str,
    redis_url: str,
    redpanda_brokers: tuple[str, ...],
    redpanda_topic_prefix: str,
):
    normalized = backend.lower()
    if normalized == "inmemory":
        return InMemoryStreamBus()
    if normalized == "redis":
        return RedisStreamBus(redis_url)
    if normalized in {"redpanda", "kafka"}:
        return RedpandaStreamBus(
            brokers=redpanda_brokers,
            topic_prefix=redpanda_topic_prefix,
        )
    raise ValueError(f"Unsupported event bus backend: {backend}")
