from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
import hashlib

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


UNKNOWN_PARTITION_VALUE = "_global"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _ensure_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_ensure_jsonable(item) for item in value]
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(_ensure_jsonable(value), sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_json_dumps(value).encode("utf-8"))


def _normalize_partition(value: str | None) -> str:
    if value is None:
        return UNKNOWN_PARTITION_VALUE
    normalized = str(value).strip()
    return normalized or UNKNOWN_PARTITION_VALUE


@dataclass(frozen=True)
class DataLakeRecord:
    layer: str
    dt: str
    profile_id: str
    venue: str
    inst_id: str
    stream: str
    payload: dict[str, Any]
    idempotency_key: str | None = None


@dataclass(frozen=True)
class EventEnvelope:
    event_type: str
    source_service: str
    payload: dict[str, Any]
    schema_version: str = "v1"
    event_id: str = field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = field(default_factory=utc_now)
    ingested_at: datetime = field(default_factory=utc_now)
    profile_id: str | None = None
    strategy_id: str | None = None
    run_id: str | None = None
    inst_id: str | None = None
    correlation_id: str | None = None
    causation_id: str | None = None
    payload_hash: str = ""

    def __post_init__(self) -> None:
        if not self.payload_hash:
            object.__setattr__(self, "payload_hash", _sha256_json(self.payload))

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "schema_version": self.schema_version,
            "occurred_at": self.occurred_at,
            "ingested_at": self.ingested_at,
            "profile_id": self.profile_id,
            "strategy_id": self.strategy_id,
            "run_id": self.run_id,
            "inst_id": self.inst_id,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "source_service": self.source_service,
            "payload": _json_dumps(self.payload),
            "payload_hash": self.payload_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EventEnvelope:
        copy = dict(payload)
        for key in ("occurred_at", "ingested_at"):
            value = copy.get(key)
            if isinstance(value, str):
                copy[key] = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if isinstance(copy.get("payload"), str):
            copy["payload"] = json.loads(copy["payload"])
        return cls(**copy)


@dataclass(frozen=True)
class DatasetBatch:
    layer: str
    dt: str
    profile_id: str
    venue: str
    stream: str
    rows: list[dict[str, Any]]
    inst_id: str | None = None
    run_id: str | None = None
    hour: str | None = None
    logical_name: str | None = None
    schema_version: str = "v1"
    retention_class: str = "long"
    quality_state: str = "ready"
    idempotency_key: str | None = None
    partition_overrides: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditSegmentManifest:
    segment_id: str
    stream: str
    partition_id: str
    offset_start: int
    offset_end: int
    row_count: int
    schema_version: str
    object_key: str
    object_uri: str
    manifest_key: str
    manifest_uri: str
    checksum: str
    previous_hash: str | None
    manifest_hash: str
    checkpoint_key: str
    created_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "stream": self.stream,
            "partition_id": self.partition_id,
            "offset_start": self.offset_start,
            "offset_end": self.offset_end,
            "row_count": self.row_count,
            "schema_version": self.schema_version,
            "object_key": self.object_key,
            "object_uri": self.object_uri,
            "manifest_key": self.manifest_key,
            "manifest_uri": self.manifest_uri,
            "checksum": self.checksum,
            "previous_hash": self.previous_hash,
            "manifest_hash": self.manifest_hash,
            "checkpoint_key": self.checkpoint_key,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AuditSegmentManifest:
        copy = dict(payload)
        created_at = copy.get("created_at")
        if isinstance(created_at, str):
            copy["created_at"] = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        return cls(**copy)


class ObjectStore:
    def put_bytes(self, key: str, payload: bytes) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def read_bytes(self, key: str) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError

    def exists(self, key: str) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def list_keys(self, prefix: str) -> list[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def resolve_uri(self, key: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def write_json(self, key: str, payload: dict[str, Any]) -> str:
        return self.put_bytes(key, _json_dumps(payload).encode("utf-8"))

    def read_json(self, key: str) -> dict[str, Any]:
        return json.loads(self.read_bytes(key).decode("utf-8"))


class LocalObjectStore(ObjectStore):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, key: str, payload: bytes) -> str:
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return self.resolve_uri(key)

    def read_bytes(self, key: str) -> bytes:
        return (self.root / key).read_bytes()

    def exists(self, key: str) -> bool:
        return (self.root / key).exists()

    def list_keys(self, prefix: str) -> list[str]:
        target = self.root / prefix
        if target.is_file():
            return [prefix.replace("\\", "/")]
        if not target.exists():
            return []
        return sorted(
            str(path.relative_to(self.root)).replace("\\", "/")
            for path in target.rglob("*")
            if path.is_file()
        )

    def resolve_uri(self, key: str) -> str:
        return str((self.root / key).resolve())


class S3ObjectStore(ObjectStore):
    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        prefix: str = "",
    ) -> None:
        import boto3

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )

    def _qualified_key(self, key: str) -> str:
        relative = key.strip("/")
        if not self.prefix:
            return relative
        return f"{self.prefix}/{relative}"

    def put_bytes(self, key: str, payload: bytes) -> str:
        qualified = self._qualified_key(key)
        self.client.put_object(Bucket=self.bucket, Key=qualified, Body=payload)
        return self.resolve_uri(key)

    def read_bytes(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=self._qualified_key(key))
        return response["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._qualified_key(key))
        except self.client.exceptions.ClientError:
            return False
        return True

    def list_keys(self, prefix: str) -> list[str]:
        qualified_prefix = self._qualified_key(prefix)
        paginator = self.client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=qualified_prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if self.prefix:
                    key = key[len(self.prefix) + 1 :]
                keys.append(key)
        return sorted(keys)

    def resolve_uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{self._qualified_key(key)}"


class SchemaRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def validate(self, dataset_id: str, schema_version: str, schema: pa.Schema) -> None:
        payload = self._schema_payload(dataset_id, schema_version, schema)
        registry_path = self.root / f"{dataset_id}.json"
        if not registry_path.exists():
            registry_path.write_text(_json_dumps(payload), encoding="utf-8")
            return
        existing = json.loads(registry_path.read_text(encoding="utf-8"))
        existing_fields = {
            field["name"]: field["type"]
            for field in existing.get("fields", [])
            if field.get("schema_version") == schema_version
        }
        for field in payload["fields"]:
            current_type = existing_fields.get(field["name"])
            if current_type is not None and current_type != field["type"]:
                raise ValueError(
                    f"Schema drift detected for {dataset_id}:{field['name']} "
                    f"({current_type} -> {field['type']})"
                )
        missing = sorted(set(existing_fields) - {field["name"] for field in payload["fields"]})
        if missing:
            raise ValueError(
                f"Non-additive schema change detected for {dataset_id}: missing {missing}"
            )
        registry_path.write_text(_json_dumps(payload), encoding="utf-8")

    @staticmethod
    def _schema_payload(
        dataset_id: str, schema_version: str, schema: pa.Schema
    ) -> dict[str, Any]:
        return {
            "dataset_id": dataset_id,
            "schema_version": schema_version,
            "updated_at": utc_now(),
            "fields": [
                {
                    "name": field.name,
                    "type": str(field.type),
                    "schema_version": schema_version,
                }
                for field in schema
            ],
        }


class DataLakeWriter:
    def __init__(
        self,
        root: str,
        duckdb_path: str,
        *,
        hot_cache_root: str | None = None,
        object_store_backend: str = "filesystem",
        object_store_bucket: str | None = None,
        object_store_endpoint: str | None = None,
        object_store_region: str | None = None,
        object_store_access_key_id: str | None = None,
        object_store_secret_access_key: str | None = None,
        object_store_prefix: str = "",
        checkpoint_root: str | None = None,
        object_store: ObjectStore | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.duckdb_path = duckdb_path
        self.hot_cache_root = Path(hot_cache_root or self.root / "_hot_cache")
        self.hot_cache_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_root = Path(checkpoint_root or self.root / "_checkpoints")
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.idempotency_root = self.root / "_idempotency"
        self.idempotency_root.mkdir(parents=True, exist_ok=True)
        self.schema_registry = SchemaRegistry(self.root / "_schema_registry")
        self.object_store = object_store or self._build_object_store(
            backend=object_store_backend,
            bucket=object_store_bucket,
            endpoint=object_store_endpoint,
            region=object_store_region,
            access_key_id=object_store_access_key_id,
            secret_access_key=object_store_secret_access_key,
            prefix=object_store_prefix,
        )

    def _build_object_store(
        self,
        *,
        backend: str,
        bucket: str | None,
        endpoint: str | None,
        region: str | None,
        access_key_id: str | None,
        secret_access_key: str | None,
        prefix: str,
    ) -> ObjectStore:
        normalized = backend.lower()
        if normalized in {"filesystem", "local"}:
            return LocalObjectStore(self.root)
        if normalized == "s3":
            if not bucket:
                raise ValueError("object_store_bucket is required for s3 backend")
            return S3ObjectStore(
                bucket=bucket,
                endpoint_url=endpoint,
                region_name=region,
                access_key_id=access_key_id,
                secret_access_key=secret_access_key,
                prefix=prefix,
            )
        raise ValueError(f"Unsupported object store backend: {backend}")

    def write(self, record: DataLakeRecord) -> Path:
        return self.write_batch(
            DatasetBatch(
                layer=record.layer,
                dt=record.dt,
                profile_id=record.profile_id,
                venue=record.venue,
                inst_id=record.inst_id,
                stream=record.stream,
                rows=[record.payload],
                idempotency_key=record.idempotency_key,
            )
        )

    def write_batch(self, batch: DatasetBatch) -> Path:
        if not batch.rows:
            raise ValueError("batch rows must not be empty")
        if batch.idempotency_key:
            existing = self._lookup_idempotency(batch.idempotency_key)
            if existing is not None:
                return Path(existing)
        table = pa.Table.from_pylist([_ensure_jsonable(row) for row in batch.rows])
        dataset_id = self._dataset_id(batch)
        self.schema_registry.validate(dataset_id, batch.schema_version, table.schema)
        relative_dir = self._partition_dir(
            layer=batch.layer,
            dt=batch.dt,
            profile_id=batch.profile_id,
            venue=batch.venue,
            inst_id=batch.inst_id,
            stream=batch.stream,
            run_id=batch.run_id,
            hour=batch.hour,
            extra=batch.partition_overrides,
        )
        target_dir = self.root / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"part-{uuid4()}.parquet"
        pq.write_table(table, file_path, compression="zstd")
        if batch.idempotency_key:
            self._store_idempotency(batch.idempotency_key, file_path)
        return file_path

    def append_events(
        self,
        stream: str,
        events: list[EventEnvelope],
        *,
        partition_id: str = "0",
        offset_start: int | None = None,
    ) -> AuditSegmentManifest | None:
        if not events:
            return None
        partition_id = _normalize_partition(partition_id)
        checkpoint_key = f"audit/_checkpoints/stream={stream}/partition={partition_id}.json"
        checkpoint = (
            self.object_store.read_json(checkpoint_key)
            if self.object_store.exists(checkpoint_key)
            else {}
        )
        segment_id = str(uuid4())
        row_dicts = [event.as_dict() for event in events]
        dt_value = events[0].occurred_at.date().isoformat()
        if offset_start is None:
            offset_start = int(checkpoint.get("next_offset", 0))
        offset_end = offset_start + len(events) - 1
        object_key = (
            f"audit/dt={dt_value}/stream={stream}/partition={partition_id}/"
            f"segment-{segment_id}.parquet"
        )
        table = pa.Table.from_pylist([_ensure_jsonable(row) for row in row_dicts])
        buffer = io.BytesIO()
        pq.write_table(table, buffer, compression="zstd")
        payload = buffer.getvalue()
        checksum = _sha256_bytes(payload)
        object_uri = self.object_store.put_bytes(object_key, payload)
        manifest_key = (
            f"audit/_manifests/stream={stream}/partition={partition_id}/{segment_id}.json"
        )
        previous_hash = checkpoint.get("latest_manifest_hash")
        manifest_base = {
            "segment_id": segment_id,
            "stream": stream,
            "partition_id": partition_id,
            "offset_start": offset_start,
            "offset_end": offset_end,
            "row_count": len(events),
            "schema_version": events[0].schema_version,
            "object_key": object_key,
            "object_uri": object_uri,
            "manifest_key": manifest_key,
            "manifest_uri": self.object_store.resolve_uri(manifest_key),
            "checksum": checksum,
            "previous_hash": previous_hash,
            "checkpoint_key": checkpoint_key,
            "created_at": utc_now(),
        }
        manifest_hash = _sha256_json(manifest_base)
        manifest = AuditSegmentManifest(manifest_hash=manifest_hash, **manifest_base)
        self.object_store.write_json(manifest_key, manifest.as_dict())
        self.object_store.write_json(
            checkpoint_key,
            {
                "stream": stream,
                "partition_id": partition_id,
                "latest_manifest_hash": manifest_hash,
                "next_offset": offset_end + 1,
                "latest_segment_id": segment_id,
                "updated_at": utc_now(),
            },
        )
        return manifest

    def read_audit_events(self, *, stream: str | None = None) -> list[EventEnvelope]:
        prefix = "audit/_manifests"
        if stream:
            prefix = f"{prefix}/stream={stream}"
        manifests = [
            AuditSegmentManifest.from_dict(self.object_store.read_json(key))
            for key in self.object_store.list_keys(prefix)
            if key.endswith(".json")
        ]
        manifests.sort(key=lambda item: (item.stream, item.partition_id, item.offset_start))
        events: list[EventEnvelope] = []
        for manifest in manifests:
            raw = self.object_store.read_bytes(manifest.object_key)
            table = pq.read_table(pa.BufferReader(raw))
            for row in table.to_pylist():
                events.append(EventEnvelope.from_dict(row))
        return events

    def build_dataset_version_payload(
        self,
        *,
        dataset_id: str,
        profile_id: str,
        logical_name: str,
        layer: str,
        path: str,
        schema_version: str = "v1",
        checksum: str | None = None,
        row_count: int | None = None,
        file_count: int | None = None,
        byte_count: int | None = None,
        watermark_start: datetime | None = None,
        watermark_end: datetime | None = None,
        producing_run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return {
            "dataset_version_id": f"{dataset_id}:{version}",
            "dataset_id": dataset_id,
            "profile_id": profile_id,
            "version": version,
            "logical_name": logical_name,
            "layer": layer,
            "path": path,
            "physical_uri": path,
            "schema_version": schema_version,
            "checksum": checksum,
            "row_count": row_count,
            "file_count": file_count,
            "byte_count": byte_count,
            "watermark_start": watermark_start,
            "watermark_end": watermark_end,
            "producing_run_id": producing_run_id,
            "metadata": metadata or {},
        }

    def register_dataset(self, name: str, glob_pattern: str) -> None:
        with duckdb.connect(self.duckdb_path) as con:
            con.execute(
                "CREATE OR REPLACE VIEW "
                f"{name} AS SELECT * FROM read_parquet('{glob_pattern}')"
            )

    def query(self, sql: str) -> list[dict[str, Any]]:
        with duckdb.connect(self.duckdb_path) as con:
            result = con.execute(sql)
            columns = [item[0] for item in result.description]
            return [dict(zip(columns, row)) for row in result.fetchall()]

    def compact(
        self,
        *,
        glob_pattern: str | None = None,
        min_file_size_bytes: int = 8 * 1024 * 1024,
    ) -> dict[str, Any]:
        if glob_pattern:
            candidate_files = [Path(path) for path in self.root.glob(glob_pattern)]
        else:
            candidate_files = list(self.root.rglob("*.parquet"))
        grouped: dict[Path, list[Path]] = {}
        for path in candidate_files:
            if "_manifests" in path.parts or "_checkpoints" in path.parts or "audit" in path.parts:
                continue
            grouped.setdefault(path.parent, []).append(path)
        compacted_groups = 0
        compacted_files = 0
        bytes_rewritten = 0
        for parent, files in grouped.items():
            eligible = [path for path in files if path.stat().st_size < min_file_size_bytes]
            if len(eligible) < 2:
                continue
            tables = [pq.read_table(path) for path in eligible]
            compacted = pa.concat_tables(tables)
            target = parent / f"compacted-{uuid4()}.parquet"
            pq.write_table(compacted, target, compression="zstd")
            for path in eligible:
                bytes_rewritten += path.stat().st_size
                path.unlink(missing_ok=True)
                compacted_files += 1
            compacted_groups += 1
        return {
            "compacted_groups": compacted_groups,
            "compacted_files": compacted_files,
            "bytes_rewritten": bytes_rewritten,
        }

    def cleanup_layer(self, *, layer: str, ttl_days: int) -> list[Path]:
        cutoff = utc_now() - timedelta(days=ttl_days)
        removed: list[Path] = []
        layer_root = self.root / layer
        if not layer_root.exists():
            return removed
        for path in layer_root.rglob("*.parquet"):
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if modified < cutoff:
                path.unlink(missing_ok=True)
                removed.append(path)
        return removed

    def cleanup_bronze(self, *, ttl_days: int) -> list[Path]:
        return self.cleanup_layer(layer="bronze", ttl_days=ttl_days)

    def doctor(self) -> dict[str, Any]:
        layer_stats: dict[str, dict[str, int]] = {}
        for path in self.root.rglob("*"):
            if not path.is_file() or path.name.endswith(".json"):
                continue
            try:
                layer = path.relative_to(self.root).parts[0]
            except ValueError:
                continue
            stats = layer_stats.setdefault(layer, {"bytes": 0, "files": 0})
            stats["bytes"] += path.stat().st_size
            stats["files"] += 1
        return {
            "root": str(self.root.resolve()),
            "duckdb_path": str(Path(self.duckdb_path).resolve()),
            "layer_stats": layer_stats,
            "audit_manifest_count": len(
                [
                    key
                    for key in self.object_store.list_keys("audit/_manifests")
                    if key.endswith(".json")
                ]
            ),
            "audit_checkpoint_count": len(
                [
                    key
                    for key in self.object_store.list_keys("audit/_checkpoints")
                    if key.endswith(".json")
                ]
            ),
        }

    def _dataset_id(self, batch: DatasetBatch) -> str:
        logical_name = batch.logical_name or batch.stream
        return f"{batch.layer}.{logical_name}".replace("/", "_")

    def _partition_dir(
        self,
        *,
        layer: str,
        dt: str,
        profile_id: str | None,
        venue: str | None,
        inst_id: str | None,
        stream: str,
        run_id: str | None,
        hour: str | None,
        extra: dict[str, str] | None = None,
    ) -> Path:
        parts = [layer, f"dt={dt}", f"profile_id={_normalize_partition(profile_id)}"]
        if venue:
            parts.append(f"venue={_normalize_partition(venue)}")
        if inst_id:
            parts.append(f"inst_id={_normalize_partition(inst_id)}")
        parts.append(f"stream={_normalize_partition(stream)}")
        if run_id:
            parts.append(f"run_id={_normalize_partition(run_id)}")
        if hour:
            parts.append(f"hour={_normalize_partition(hour)}")
        for key, value in sorted((extra or {}).items()):
            parts.append(f"{key}={_normalize_partition(value)}")
        return Path(*parts)

    def _lookup_idempotency(self, idempotency_key: str) -> str | None:
        marker = self.idempotency_root / f"{_sha256_json(idempotency_key)}.json"
        if not marker.exists():
            return None
        payload = json.loads(marker.read_text(encoding="utf-8"))
        return payload["path"]

    def _store_idempotency(self, idempotency_key: str, path: Path) -> None:
        marker = self.idempotency_root / f"{_sha256_json(idempotency_key)}.json"
        marker.write_text(
            _json_dumps({"idempotency_key": idempotency_key, "path": str(path.resolve())}),
            encoding="utf-8",
        )
