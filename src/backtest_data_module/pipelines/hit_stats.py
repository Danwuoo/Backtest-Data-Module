from __future__ import annotations

import yaml
from prefect import flow, get_run_logger
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule

from backtest_data_module.data_storage.storage_backend import HybridStorageManager

# 從配置檔讀取排程
try:
    with open("storage.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
except FileNotFoundError:
    cfg = {}
SCHEDULE = cfg.get("hit_stats_schedule", "0 1 * * *")


@flow
def hit_stats_flow() -> None:
    """計算 7 日命中率並自動遷移冷門表格。"""
    logger = get_run_logger()
    manager = HybridStorageManager()
    stats = manager.compute_7day_hits()
    logger.info(f"7日命中統計: {stats}")
    manager.migrate_low_hit_tables()


DeploymentSpec(
    flow=hit_stats_flow,
    name="hit-stats-maintenance",
    schedule=CronSchedule(cron=SCHEDULE, timezone="UTC"),
    tags=["maintenance"],
)
