from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI

from okx_trading_platform.domain import RunType
from okx_trading_platform.shared.data_lake import DataLakeWriter
from okx_trading_platform.shared.runtime import ReplayRuntime
from okx_trading_platform.shared.settings import PlatformSettings, get_platform_settings

settings = get_platform_settings()
runtime = ReplayRuntime(
    service_name="replay-service",
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


class ReplayWorker:
    def __init__(self, *, settings: PlatformSettings, runtime: ReplayRuntime) -> None:
        self.settings = settings
        self.runtime = runtime
        self.task: asyncio.Task | None = None
        self.stats: dict[str, Any] = {
            "enabled": settings.enable_replay_worker,
            "executed_runs": 0,
            "last_error": None,
            "last_error_at": None,
            "last_success_at": None,
        }

    async def start(self) -> None:
        if not self.settings.enable_replay_worker:
            return
        self.task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        if self.task is None:
            return
        self.task.cancel()
        with suppress(asyncio.CancelledError):
            await self.task
        self.task = None

    async def _run_loop(self) -> None:
        while True:
            try:
                await self._execute_pending_runs()
                self.stats["last_success_at"] = datetime.now(timezone.utc).isoformat()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - exercised in service integration
                self.stats["last_error"] = str(exc)
                self.stats["last_error_at"] = datetime.now(timezone.utc).isoformat()
            await asyncio.sleep(5)

    async def _execute_pending_runs(self) -> None:
        async with httpx.AsyncClient(
            base_url=self.settings.control_api_url,
            timeout=20.0,
        ) as client:
            for run_type, path in (
                (RunType.BACKTEST, "/backtests"),
                (RunType.PAPER, "/paper-runs"),
                (RunType.LIVE, "/live-runs"),
            ):
                response = await client.get(path, params={"profile_id": self.runtime.profile_id})
                response.raise_for_status()
                for run in response.json():
                    if run.get("status") != "pending":
                        continue
                    await self._execute_run(client, run_type, path, run)

    async def _execute_run(
        self,
        client: httpx.AsyncClient,
        run_type: RunType,
        path: str,
        run: dict[str, Any],
    ) -> None:
        running_payload = dict(run)
        running_payload["status"] = "running"
        running_payload["started_at"] = datetime.now(timezone.utc).isoformat()
        await client.post(path, json=running_payload)

        summary_rows = [
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "run_id": run["run_id"],
                "run_type": run_type.value,
                "strategy_id": run["strategy_id"],
                "model_version_id": run.get("model_version_id"),
                "input_dataset_version_ids": run.get("input_dataset_version_ids", []),
                "input_feature_ids": run.get("input_feature_ids", []),
                "status": "completed",
            }
        ]
        artifact_type = f"{run_type.value}_summary"
        artifact_path = self.runtime.materialize_artifact(
            run_id=run["run_id"],
            artifact_type=artifact_type,
            rows=summary_rows,
            inst_id=run.get("config", {}).get("inst_id"),
        )
        if artifact_path is None:
            return

        logical_dataset_id = f"{run_type.value}-summary"
        dataset_payload = {
            "dataset_id": logical_dataset_id,
            "profile_id": run["profile_id"],
            "name": f"{run_type.value} summary",
            "logical_name": logical_dataset_id,
            "layer": "gold",
            "path": artifact_path,
            "physical_uri": artifact_path,
            "manifest_uri": None,
            "schema_version": "v1",
            "row_count": len(summary_rows),
            "file_count": 1,
            "byte_count": None,
            "watermark_start": summary_rows[0]["ts"],
            "watermark_end": summary_rows[0]["ts"],
            "retention_class": "long",
            "quality_state": "ready",
            "pinned": True,
            "source_dataset_ids": [],
            "source_feature_ids": run.get("input_feature_ids", []),
            "producing_run_id": run["run_id"],
            "metadata": {"run_type": run_type.value},
        }
        await client.post("/datasets", json=dataset_payload)
        version_payload = runtime.data_lake.build_dataset_version_payload(
            dataset_id=logical_dataset_id,
            profile_id=run["profile_id"],
            logical_name=logical_dataset_id,
            layer="gold",
            path=artifact_path,
            row_count=len(summary_rows),
            file_count=1,
            watermark_start=datetime.now(timezone.utc),
            watermark_end=datetime.now(timezone.utc),
            producing_run_id=run["run_id"],
            metadata={"run_type": run_type.value},
        )
        await client.post("/dataset-versions", json=version_payload)

        artifact_payload = {
            "artifact_id": f"{run['run_id']}:{artifact_type}",
            "run_id": run["run_id"],
            "profile_id": run["profile_id"],
            "run_type": run_type.value,
            "artifact_type": artifact_type,
            "name": artifact_type,
            "logical_name": artifact_type,
            "path": artifact_path,
            "physical_uri": artifact_path,
            "manifest_uri": None,
            "checksum": None,
            "size_bytes": None,
            "dataset_version_id": version_payload["dataset_version_id"],
            "feature_id": None,
            "metadata": {"run_type": run_type.value},
        }
        await client.post("/run-artifacts", json=artifact_payload)

        completed_payload = dict(running_payload)
        completed_payload["status"] = "completed"
        completed_payload["artifact_path"] = artifact_path
        completed_payload["artifact_ids"] = [artifact_payload["artifact_id"]]
        completed_payload["completed_at"] = datetime.now(timezone.utc).isoformat()
        completed_payload["materialization_id"] = version_payload["dataset_version_id"]
        completed_payload["row_count"] = len(summary_rows)
        completed_payload["file_count"] = 1
        completed_payload["quality_state"] = "ready"
        completed_payload["pinned"] = True
        await client.post(path, json=completed_payload)
        self.stats["executed_runs"] += 1


worker = ReplayWorker(settings=settings, runtime=runtime)


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    await worker.start()
    try:
        yield
    finally:
        await worker.stop()


app = FastAPI(title="OKX Replay Service", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict:
    return {
        "service": runtime.service_name,
        "status": runtime.status,
        "profile_id": runtime.profile_id,
        "environment": runtime.environment,
        "worker": worker.stats,
    }


@app.get("/heartbeat")
def heartbeat() -> dict:
    return runtime.heartbeat().model_dump(mode="json")
