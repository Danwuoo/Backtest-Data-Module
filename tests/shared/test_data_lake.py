from pathlib import Path

from okx_trading_platform.shared.data_lake import (
    DataLakeRecord,
    DataLakeWriter,
    EventEnvelope,
)


def test_data_lake_writes_partitioned_parquet(tmp_path):
    writer = DataLakeWriter(str(tmp_path / "lake"), str(tmp_path / "test.duckdb"))
    path = writer.write(
        DataLakeRecord(
            layer="silver",
            dt="2026-03-07",
            profile_id="demo-main",
            venue="okx",
            inst_id="BTC-USDT-SWAP",
            stream="trades",
            payload={"price": 100.0, "size": 1.0},
        )
    )

    assert path.suffix == ".parquet"
    assert "dt=2026-03-07" in str(path)
    assert Path(path).exists()
    assert "profile_id=demo-main" in str(path)


def test_data_lake_appends_audit_events_with_manifest_and_checkpoint(tmp_path):
    writer = DataLakeWriter(str(tmp_path / "lake"), str(tmp_path / "test.duckdb"))
    manifest = writer.append_events(
        "okx-platform.control-plane",
        [
            EventEnvelope(
                event_type="dataset.upserted",
                source_service="control-api",
                profile_id="demo-main",
                payload={"dataset_id": "trades", "profile_id": "demo-main"},
            )
        ],
    )

    assert manifest is not None
    assert Path(manifest.object_uri).exists()
    assert Path(manifest.manifest_uri).exists()
    assert writer.read_audit_events(stream="okx-platform.control-plane")[0].event_type == "dataset.upserted"


def test_data_lake_idempotency_returns_existing_path(tmp_path):
    writer = DataLakeWriter(str(tmp_path / "lake"), str(tmp_path / "test.duckdb"))
    first = writer.write(
        DataLakeRecord(
            layer="silver",
            dt="2026-03-07",
            profile_id="demo-main",
            venue="okx",
            inst_id="BTC-USDT-SWAP",
            stream="trades",
            payload={"price": 100.0, "size": 1.0},
            idempotency_key="demo-main:trades:1",
        )
    )
    second = writer.write(
        DataLakeRecord(
            layer="silver",
            dt="2026-03-07",
            profile_id="demo-main",
            venue="okx",
            inst_id="BTC-USDT-SWAP",
            stream="trades",
            payload={"price": 100.0, "size": 1.0},
            idempotency_key="demo-main:trades:1",
        )
    )

    assert first == second
