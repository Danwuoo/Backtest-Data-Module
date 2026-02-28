from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl

from backtest_data_module.utils.notify import Notifier, SlackNotifier

if TYPE_CHECKING:  # pragma: no cover - type checking imports
    from backtest_data_module.data_storage.storage_backend import HybridStorageManager


@dataclass
class CatalogEntry:
    """Catalog 資料結構，包含版本與分割區資訊。"""

    table_name: str
    version: int
    tier: str
    location: str
    schema_hash: str
    row_count: int = 0
    partition_keys: str = ""
    lineage: str = ""
    created_at: str | None = None


class Catalog:
    """使用 SQLite 紀錄資料表所在層級與 schema。"""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS catalog (
                table_name TEXT,
                version INTEGER,
                tier TEXT,
                location TEXT,
                schema_hash TEXT,
                row_count INTEGER DEFAULT 0,
                partition_keys TEXT DEFAULT "",
                lineage TEXT DEFAULT "",
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (table_name, version)
            )
            """
        )
        self.conn.commit()

    def upsert(self, entry: CatalogEntry) -> None:
        """新增一筆表格版本紀錄。"""
        with self.conn:
            cur = self.conn.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM catalog WHERE table_name=?",
                (entry.table_name,),
            )
            version = cur.fetchone()[0]
            entry.version = version
            entry.created_at = entry.created_at or datetime.utcnow().isoformat()
            self.conn.execute(
                """
                INSERT INTO catalog (
                    table_name, version, tier, location, schema_hash,
                    row_count, partition_keys, lineage, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.table_name,
                    entry.version,
                    entry.tier,
                    entry.location,
                    entry.schema_hash,
                    entry.row_count,
                    entry.partition_keys,
                    entry.lineage,
                    entry.created_at,
                ),
            )

    def update_tier(self, table_name: str, tier: str, location: str) -> None:
        """更新表格所在層級。"""
        with self.conn:
            self.conn.execute(
                """
                UPDATE catalog
                SET tier=?, location=?
                WHERE table_name=?
                  AND version=(SELECT MAX(version) FROM catalog WHERE table_name=?)
                """,
                (tier, location, table_name, table_name),
            )

    def get(self, table_name: str) -> CatalogEntry | None:
        cur = self.conn.execute(
            (
                "SELECT table_name, version, tier, location, schema_hash, row_count,"
                " partition_keys, lineage, created_at FROM catalog"
                " WHERE table_name=? ORDER BY version DESC LIMIT 1"
            ),
            (table_name,),
        )
        row = cur.fetchone()
        if row:
            return CatalogEntry(*row)
        return None


def send_slack_alert(message: str, webhook_url: str | None = None) -> None:
    """已廢棄：改用 Notifier 介面。"""
    SlackNotifier(webhook_url).send(message)


def check_drift(
    manager: "HybridStorageManager", notifier: Notifier | None = None
) -> list[str]:
    """重新計算 schema hash，若不一致則警告並更新紀錄。"""
    from .storage_backend import HybridStorageManager

    if not isinstance(manager, HybridStorageManager):
        raise TypeError("manager must be HybridStorageManager")

    catalog = manager.catalog
    cur = catalog.conn.execute(
        """
        SELECT c.table_name, c.tier, c.schema_hash
        FROM catalog c
        JOIN (
            SELECT table_name, MAX(version) AS ver FROM catalog GROUP BY table_name
        ) m ON c.table_name = m.table_name AND c.version = m.ver
        """
    )
    mismatches = []
    for table, tier, stored_hash in cur.fetchall():
        backend = manager._backend_for(tier)
        try:
            df = backend.read(table)
        except KeyError:
            continue
        if isinstance(df, pl.DataFrame):
            schema_repr = str(df.schema)
        else:
            schema_repr = str({col: str(dtype) for col, dtype in df.dtypes.items()})
        new_hash = hashlib.sha256(schema_repr.encode()).hexdigest()
        catalog.conn.execute(
            """
            UPDATE catalog
            SET schema_hash=?, row_count=?
            WHERE table_name=?
              AND version=(SELECT MAX(version) FROM catalog WHERE table_name=?)
            """,
            (new_hash, len(df), table, table),
        )
        if new_hash != stored_hash:
            mismatches.append(table)
            if notifier:
                notifier.send(f"Schema drift detected for {table}")
    catalog.conn.commit()
    return mismatches
