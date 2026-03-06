from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


class Notifier:
    def notify(self, *, title: str, message: str) -> None:
        raise NotImplementedError


class NullNotifier(Notifier):
    def notify(self, *, title: str, message: str) -> None:
        del title, message


@dataclass
class SlackWebhookNotifier(Notifier):
    webhook_url: str
    http_client: httpx.Client | None = None

    def notify(self, *, title: str, message: str) -> None:
        client = self.http_client or httpx.Client(timeout=10.0)
        client.post(self.webhook_url, json={"text": f"{title}\n{message}"})


def build_default_notifier() -> Notifier:
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return NullNotifier()
    return SlackWebhookNotifier(webhook_url=webhook_url)
