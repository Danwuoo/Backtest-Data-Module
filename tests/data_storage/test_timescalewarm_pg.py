import io

import duckdb
import polars as pl
from polars.testing import assert_frame_equal

from backtest_data_module.data_storage.storage_backend import TimescaleWarm


class DummyCopy:
    def __init__(self, con, stmt):
        self.con = con
        self.stmt = stmt
        self.buf = io.StringIO()

    def __enter__(self):
        return self

    def write(self, data: str) -> None:
        self.buf.write(data)

    def __exit__(self, exc_type, exc, tb):
        self.buf.seek(0)
        df = pl.read_csv(self.buf)
        table = self.stmt.split()[1].strip('"')
        self.con.register("tmp", df.to_arrow())
        self.con.execute(f'INSERT INTO "{table}" SELECT * FROM tmp')
        self.con.unregister("tmp")


class DummyCursor:
    def __init__(self, con):
        self.con = con

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def execute(self, q: str) -> None:
        self.con.execute(q)

    def copy(self, stmt: str):
        return DummyCopy(self.con, stmt)


class DummyConn:
    def __init__(self):
        self.db = duckdb.connect()

    def cursor(self):
        return DummyCursor(self.db)

    def commit(self):
        return None


def test_timescalewarm_pg_write_read():
    class MockWarm(TimescaleWarm):
        def __init__(self):
            self.conn = DummyConn()
            self.use_pg = True
            self._tables = set()
            self._table_types = {}

        def read(self, table: str) -> pl.DataFrame:
            try:
                return self.conn.db.execute(f'SELECT * FROM "{table}"').pl()
            except duckdb.CatalogException as e:
                raise KeyError(table) from e

    warm = MockWarm()
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    warm.write(df, "tbl")
    out = warm.read("tbl")
    assert_frame_equal(out, df)
