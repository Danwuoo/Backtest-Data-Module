from pathlib import Path

from okx_trading_platform.shared.data_lake import DataLakeRecord, DataLakeWriter


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
