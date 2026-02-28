from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd
from prefect import flow, task, get_run_logger

from backtest_data_module.data_processing.pipeline import Pipeline
from backtest_data_module.zxq.pipeline.loader import load_steps_from_yaml


@task
def run_pipeline(df: pd.DataFrame, steps: Iterable) -> pd.DataFrame:
    """執行 Pipeline 並記錄時間與資料筆數"""
    logger = get_run_logger()
    pipeline = Pipeline(list(steps))
    start = datetime.now()
    result = pipeline.process(df)
    duration = (datetime.now() - start).total_seconds()
    logger.info(f"處理完成，共 {len(result)} rows，耗時 {duration:.2f}s")
    return result


@flow
def data_pipeline_flow(df: pd.DataFrame) -> pd.DataFrame:
    """範例 Prefect Flow"""
    classes = load_steps_from_yaml("steps.yaml")
    steps = [cls() for cls in classes]
    return run_pipeline(df, steps)
