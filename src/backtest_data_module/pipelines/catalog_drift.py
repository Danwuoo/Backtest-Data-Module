from prefect import flow, get_run_logger
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule

import os

from backtest_data_module.data_storage.storage_backend import HybridStorageManager
from backtest_data_module.data_storage.catalog import check_drift
from backtest_data_module.utils.notify import SlackNotifier


@flow
def catalog_drift_flow() -> None:
    """每日檢查 Catalog schema 是否漂移。"""
    logger = get_run_logger()
    manager = HybridStorageManager()
    notifier = SlackNotifier(os.getenv("SLACK_WEBHOOK"))
    mismatches = check_drift(manager, notifier=notifier)
    if mismatches:
        logger.warning(f"Schema drift detected: {', '.join(mismatches)}")
    else:
        logger.info("No schema drift detected")


DeploymentSpec(
    flow=catalog_drift_flow,
    name="catalog-drift-check",
    schedule=CronSchedule(cron="0 0 * * *", timezone="UTC"),
    tags=["maintenance"],
)
