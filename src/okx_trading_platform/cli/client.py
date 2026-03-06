from __future__ import annotations

import os
from typing import Any

import httpx


class ControlApiClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        resolved_base_url = base_url or os.getenv(
            "CONTROL_API_URL", "http://127.0.0.1:8000"
        )
        resolved_api_key = api_key or os.getenv("CONTROL_API_KEY") or os.getenv(
            "STRATEGY_MANAGER_API_KEY"
        )
        headers = {"X-API-KEY": resolved_api_key} if resolved_api_key else {}
        self._client = http_client or httpx.Client(
            base_url=resolved_base_url,
            headers=headers,
            timeout=10.0,
        )

    def status(self) -> dict[str, Any]:
        return {
            "kill_switch": self._get("/kill-switch").json(),
            "services": self._get("/services").json(),
            "bots": self._get("/bots").json(),
            "profiles": self._get("/profiles").json(),
        }

    def list_services(self, *, profile: str | None = None) -> list[dict[str, Any]]:
        response = self._client.get(
            "/services",
            params={"profile": profile} if profile else None,
        )
        response.raise_for_status()
        return response.json()

    def list_profiles(self) -> list[dict[str, Any]]:
        return self._get("/profiles").json()

    def list_instruments(
        self, *, profile: str | None = None, kind: str | None = None
    ) -> list[dict[str, Any]]:
        params = {}
        if profile:
            params["profile"] = profile
        if kind:
            params["kind"] = kind
        response = self._client.get("/instruments", params=params or None)
        response.raise_for_status()
        return response.json()

    def deploy(self, bot_name: str, profile: str) -> dict[str, Any]:
        response = self._client.post(
            f"/bots/{bot_name}/deploy",
            json={"profile": profile, "metadata": {}},
        )
        response.raise_for_status()
        return response.json()

    def enable_bot(self, bot_name: str) -> dict[str, Any]:
        response = self._client.post(f"/bots/{bot_name}/enable")
        response.raise_for_status()
        return response.json()

    def disable_bot(self, bot_name: str) -> dict[str, Any]:
        response = self._client.post(f"/bots/{bot_name}/disable")
        response.raise_for_status()
        return response.json()

    def create_order(
        self, payload: dict[str, Any], *, submit: bool = False
    ) -> dict[str, Any]:
        response = self._client.post(
            f"/orders?submit={'true' if submit else 'false'}",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def cancel_order(
        self,
        *,
        profile: str,
        inst_id: str,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        response = self._client.post(
            "/orders/cancel",
            json={
                "profile": profile,
                "inst_id": inst_id,
                "order_id": order_id,
                "client_order_id": client_order_id,
            },
        )
        response.raise_for_status()
        return response.json()

    def stop_all(self, reason: str) -> dict[str, Any]:
        response = self._client.put(
            "/kill-switch",
            json={"activated": True, "reason": reason},
        )
        response.raise_for_status()
        return response.json()

    def _get(self, path: str) -> httpx.Response:
        response = self._client.get(path)
        response.raise_for_status()
        return response
