from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class DataLakeRecord:
    layer: str
    dt: str
    profile_id: str
    venue: str
    inst_id: str
    stream: str
    payload: dict


class DataLakeWriter:
    def __init__(self, root: str, duckdb_path: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.duckdb_path = duckdb_path

    def write(self, record: DataLakeRecord) -> Path:
        target_dir = (
            self.root
            / record.layer
            / f"dt={record.dt}"
            / f"profile={record.profile_id}"
            / f"venue={record.venue}"
            / f"inst_id={record.inst_id}"
            / f"stream={record.stream}"
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"part-{uuid4()}.parquet"
        table = pa.Table.from_pylist([record.payload])
        pq.write_table(table, file_path, compression="zstd")
        return file_path

    def register_dataset(self, name: str, glob_pattern: str) -> None:
        with duckdb.connect(self.duckdb_path) as con:
            con.execute(
                "CREATE OR REPLACE VIEW "
                f"{name} AS SELECT * FROM read_parquet('{glob_pattern}')"
            )

    def cleanup_bronze(self, *, ttl_days: int) -> list[Path]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
        removed: list[Path] = []
        bronze_root = self.root / "bronze"
        if not bronze_root.exists():
            return removed
        for path in bronze_root.rglob("*.parquet"):
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if modified < cutoff:
                path.unlink(missing_ok=True)
                removed.append(path)
        return removed
