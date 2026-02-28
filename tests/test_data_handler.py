import pytest
import polars as pl
from polars.testing import assert_frame_equal
import pyarrow as pa
from datetime import datetime

from backtest_data_module.data_handler import DataHandler
from backtest_data_module.data_storage.storage_backend import (
    DuckHot,
    TimescaleWarm,
    S3Cold,
    HybridStorageManager,
)


@pytest.fixture
def hot_store():
    return DuckHot()


class MockTimescaleWarm(TimescaleWarm):
    def __init__(self):
        super().__init__(dsn=None)


@pytest.fixture
def warm_store():
    return MockTimescaleWarm()


@pytest.fixture
def cold_store():
    return S3Cold()


@pytest.fixture
def storage_manager(hot_store, warm_store, cold_store):
    return HybridStorageManager(
        hot_store=hot_store, warm_store=warm_store, cold_store=cold_store
    )


@pytest.fixture
def data_handler(storage_manager):
    return DataHandler(storage_manager)


@pytest.fixture
def sample_df():
    return pl.DataFrame(
        {
            "symbol": ["AAPL", "GOOG"],
            "price": [150.0, 2800.0],
            "timestamp": [
                datetime(2021, 6, 1, 12, 0, 0),
                datetime(2021, 6, 1, 12, 0, 0),
            ],
        }
    )


def test_read_write(data_handler, sample_df):
    print("Writing to hot store...")
    data_handler.storage_manager.write(sample_df, "test_table", tier="hot")
    print("Reading from data handler...")
    read_df = data_handler.read("test_table")
    print("Asserting dataframes are equal...")
    assert_frame_equal(sample_df, read_df)
    print("Test finished.")


def test_migrate(data_handler, sample_df):
    data_handler.storage_manager.write(sample_df, "test_table", tier="hot")
    data_handler.migrate("test_table", "hot", "warm")
    with pytest.raises(KeyError):
        data_handler.storage_manager.read("test_table", tiers=["hot"])
    read_df = data_handler.storage_manager.read("test_table", tiers=["warm"])
    assert_frame_equal(sample_df, read_df)


def test_cold_to_warm_promotion(data_handler, sample_df):
    data_handler.storage_manager.write(sample_df, "test_table", tier="cold")
    read_df = data_handler.read("test_table")
    assert_frame_equal(sample_df, read_df)
    # Check if the table was promoted to the warm tier
    promoted_df = data_handler.storage_manager.read("test_table", tiers=["warm"])
    assert_frame_equal(sample_df, promoted_df)


@pytest.mark.asyncio
async def test_stream(data_handler):
    symbols = ["AAPL", "GOOG"]
    freq = "1s"
    batches = [batch async for batch in data_handler.stream(symbols, freq)]
    assert len(batches) == 20
    for batch in batches:
        assert isinstance(batch, pa.RecordBatch)
        assert "symbol" in batch.schema.names
        assert "price" in batch.schema.names
        assert "timestamp" in batch.schema.names


def test_register_arrow(data_handler, sample_df):
    table_name = "arrow_table"
    arrow_table = sample_df.to_arrow()
    data_handler.register_arrow(table_name, arrow_table)
    read_df = data_handler.storage_manager.hot_store.read(table_name)
    assert_frame_equal(sample_df, read_df)


def test_validate_schema(data_handler, sample_df):
    assert data_handler.validate_schema(sample_df, "tick_schema")
    assert not data_handler.validate_schema(sample_df, "bar_schema")
