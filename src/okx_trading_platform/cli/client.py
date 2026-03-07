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
        resolved_api_key = api_key or os.getenv("CONTROL_API_KEY")
        headers = {"X-API-KEY": resolved_api_key} if resolved_api_key else {}
        self._client = http_client or httpx.Client(
            base_url=resolved_base_url,
            headers=headers,
            timeout=10.0,
        )

    def status(self) -> dict[str, Any]:
        return {
            "kill_switch": self._get("/kill-switch").json(),
            "profiles": self._get("/profiles").json(),
            "strategies": self._get("/strategies").json(),
            "services": self._get("/services").json(),
            "alerts": self._get("/alerts").json(),
        }

    def list_profiles(self) -> list[dict[str, Any]]:
        return self._get("/profiles").json()

    def list_strategies(self, *, profile_id: str | None = None) -> list[dict[str, Any]]:
        params = {"profile_id": profile_id} if profile_id else None
        return self._get("/strategies", params=params).json()

    def list_models(self, *, profile_id: str | None = None) -> list[dict[str, Any]]:
        params = {"profile_id": profile_id} if profile_id else None
        return self._get("/models", params=params).json()

    def list_datasets(self, *, profile_id: str | None = None) -> list[dict[str, Any]]:
        params = {"profile_id": profile_id} if profile_id else None
        return self._get("/datasets", params=params).json()

    def list_features(self, *, profile_id: str | None = None) -> list[dict[str, Any]]:
        params = {"profile_id": profile_id} if profile_id else None
        return self._get("/features", params=params).json()

    def list_dataset_versions(
        self,
        *,
        profile_id: str | None = None,
        dataset_id: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if profile_id:
            params["profile_id"] = profile_id
        if dataset_id:
            params["dataset_id"] = dataset_id
        return self._get("/dataset-versions", params=params or None).json()

    def list_run_artifacts(
        self,
        *,
        profile_id: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if profile_id:
            params["profile_id"] = profile_id
        if run_id:
            params["run_id"] = run_id
        return self._get("/run-artifacts", params=params or None).json()

    def list_sleeves(self, *, profile_id: str | None = None) -> list[dict[str, Any]]:
        params = {"profile_id": profile_id} if profile_id else None
        return self._get("/sleeves", params=params).json()

    def list_allocators(self, *, profile_id: str | None = None) -> list[dict[str, Any]]:
        params = {"profile_id": profile_id} if profile_id else None
        return self._get("/allocators", params=params).json()

    def list_runs(
        self, *, path: str, profile_id: str | None = None
    ) -> list[dict[str, Any]]:
        params = {"profile_id": profile_id} if profile_id else None
        return self._get(path, params=params).json()

    def list_incidents(self, *, profile_id: str | None = None) -> list[dict[str, Any]]:
        params = {"profile_id": profile_id} if profile_id else None
        return self._get("/incidents", params=params).json()

    def list_alerts(self, *, profile_id: str | None = None) -> list[dict[str, Any]]:
        params = {"profile_id": profile_id} if profile_id else None
        return self._get("/alerts", params=params).json()

    def create_order(
        self, payload: dict[str, Any], *, submit: bool = False
    ) -> dict[str, Any]:
        response = self._client.post(
            f"/orders?submit={'true' if submit else 'false'}",
            json=payload,
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

    def migrate(self) -> list[dict[str, Any]]:
        return self.list_profiles()

    def cutover(self) -> dict[str, Any]:
        return self.status()

    def _get(
        self, path: str, *, params: dict[str, Any] | None = None
    ) -> httpx.Response:
        response = self._client.get(path, params=params)
        response.raise_for_status()
        return response
