from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pandas as pd
from airflow.decorators import dag, task

from backtest_data_module.data_processing.pipeline import Pipeline
from backtest_data_module.data_storage.storage_backend import HybridStorageManager
from backtest_data_module.zxq.pipeline.loader import load_steps_from_yaml


def save_raw_to_parquet(symbol: str, date: str, df: pd.DataFrame) -> Path:
    """將原始資料存為 Parquet。"""
    dest = Path(f"data/staging/{symbol}")
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{date}.parquet"
    df.to_parquet(path)
    return path


def load_raw_from_parquet(symbol: str, date: str) -> pd.DataFrame:
    """讀取暫存的 Parquet 檔。"""
    path = Path(f"data/staging/{symbol}/{date}.parquet")
    return pd.read_parquet(path)


@dag(schedule_interval="0 */6 * * *", start_date=datetime(2024, 1, 1), catchup=False)
def ingest_process_dag():
    """示範的 Airflow DAG，包含四個基本任務"""

    @task
    def fetch_raw(symbol: str, date: str) -> pd.DataFrame:
        """抓取原始資料"""
        data = {"date": [date], "asset": [symbol], "value": [1.0]}
        return pd.DataFrame(data)

    @task
    def convert_parquet(df: pd.DataFrame, symbol: str, date: str) -> str:
        """轉換為 Parquet 格式"""
        path = save_raw_to_parquet(symbol, date, df)
        return str(path)

    @task(task_id="transform_pipeline")
    def run_pipeline(path: str, symbol: str, date: str) -> None:
        """執行資料處理流程"""
        df = load_raw_from_parquet(symbol, date)
        steps = [cls() for cls in load_steps_from_yaml("steps.yaml")]
        pipeline = Pipeline(steps)
        manager = HybridStorageManager()
        pipeline.run(df, manager, input_tables=[path], output_table=f"{symbol}_{date}")

    @task
    def load_warm():
        """佔位任務，可擴充其他動作"""
        return "done"

    f = fetch_raw("AAPL", "2024-01-01")
    p = convert_parquet(f, "AAPL", "2024-01-01")
    run_pipeline(p, "AAPL", "2024-01-01") >> load_warm()


dag = ingest_process_dag()
