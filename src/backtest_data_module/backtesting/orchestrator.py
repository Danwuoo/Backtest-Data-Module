from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable, List, Type

import httpx
import pandas as pd
import polars as pl
import ray

from backtest_data_module.backtesting.engine import Backtest
from backtest_data_module.backtesting.execution import Execution
from backtest_data_module.backtesting.performance import Performance
from backtest_data_module.backtesting.portfolio import Portfolio
from backtest_data_module.backtesting.strategy import StrategyBase
from backtest_data_module.data_handler import DataHandler
from backtest_data_module.data_processing.cross_validation import (
    combinatorial_purged_cv,
    walk_forward_split,
)
from backtest_data_module.reporting.report import ReportGen
from backtest_data_module.strategy_manager.registry import strategy_registry
from backtest_data_module.utils.json_encoder import CustomJSONEncoder


def _load_strategy_cls(strategy_module: str, strategy_class: str) -> Type[StrategyBase]:
    module = importlib.import_module(strategy_module)
    strategy_cls = getattr(module, strategy_class)
    if not isinstance(strategy_cls, type) or not issubclass(strategy_cls, StrategyBase):
        raise TypeError(f"{strategy_class} is not a StrategyBase subclass")
    return strategy_cls


@ray.remote
def run_backtest_slice(
    strategy_module: str,
    strategy_class: str,
    portfolio_cls: Type[Portfolio],
    execution_cls: Type[Execution],
    performance_cls: Type[Performance],
    strategy_params: dict,
    portfolio_params: dict,
    execution_params: dict,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    slice_id: int,
) -> dict:
    """
    在獨立的 Ray 程序中執行單一回測切片。
    """
    strategy_cls = _load_strategy_cls(strategy_module, strategy_class)
    strategy = strategy_cls(**strategy_params)
    portfolio = portfolio_cls(**portfolio_params)
    execution = execution_cls(**execution_params)
    performance = performance_cls()

    backtest = Backtest(
        strategy,
        portfolio,
        execution,
        performance,
        pl.from_pandas(test_data.reset_index()),
    )
    backtest.run()

    return {
        "slice_id": slice_id,
        "train_start": train_data.index.min(),
        "train_end": train_data.index.max(),
        "test_start": test_data.index.min(),
        "test_end": test_data.index.max(),
        "metrics": backtest.results["performance"].metrics,
    }


class Orchestrator:
    def __init__(
        self,
        data_handler: DataHandler,
        portfolio_cls: Type[Portfolio],
        execution_cls: Type[Execution],
        performance_cls: Type[Performance],
        strategy_name: str | None = None,
        strategy_cls: Type[StrategyBase] | None = None,
        register_api: str | None = None,
    ):
        if strategy_cls is None and strategy_name is None:
            raise ValueError("Either strategy_name or strategy_cls must be provided")

        self.data_handler = data_handler
        self.strategy_cls = strategy_cls or self._resolve_strategy_cls(strategy_name)
        self.strategy_name = strategy_name or self.strategy_cls.__name__

        # 允許透過名稱查詢策略，同時避免重複註冊報錯
        try:
            strategy_registry.register(self.strategy_name, self.strategy_cls)
        except ValueError:
            pass

        self.portfolio_cls = portfolio_cls
        self.execution_cls = execution_cls
        self.performance_cls = performance_cls
        self.results: dict[str, Any] = {}
        self.run_id: str | None = None
        self.hyperparams: dict[str, Any] | None = None
        self.config: dict[str, Any] = {}
        self.register_api = register_api
        if self.register_api:
            self.api_client = httpx.Client(
                base_url=self.register_api,
                headers={"X-API-KEY": os.environ.get("STRATEGY_MANAGER_API_KEY")},
            )
        else:
            self.api_client = None

    def _resolve_strategy_cls(self, strategy_name: str | None) -> Type[StrategyBase]:
        if strategy_name is None:
            raise ValueError("strategy_name is required")

        try:
            return strategy_registry.get_strategy(strategy_name)
        except ValueError:
            builtin_modules = {
                "SmaCrossover": (
                    "backtest_data_module.backtesting.strategies.sma_crossover",
                    "SmaCrossover",
                ),
                "MeanReversion": (
                    "backtest_data_module.backtesting.strategies.mean_reversion",
                    "MeanReversion",
                ),
            }
            if strategy_name not in builtin_modules:
                raise

            module_name, class_name = builtin_modules[strategy_name]
            return _load_strategy_cls(module_name, class_name)

    @staticmethod
    def _build_component(factory: Callable[..., Any], params: dict[str, Any]) -> Any:
        try:
            return factory(**params)
        except TypeError:
            if params:
                raise
            return factory()

    def _create_run(self, orchestrator_type: str, configured_run_id: str | None = None):
        if not self.register_api:
            self.run_id = configured_run_id or f"local-run-{os.urandom(4).hex()}"
            return
        response = self.api_client.post(
            "/runs",
            json={
                "strategy_name": self.strategy_name,
                "strategy_version": "0.1.0",  # 實際應填入版本資訊
                "hyperparameters": self.hyperparams,
                "orchestrator_type": orchestrator_type,
            },
        )
        response.raise_for_status()
        self.run_id = response.json()["run_id"]

    def _update_run_status(
        self,
        status: str,
        metrics_uri: str | None = None,
        error_message: str | None = None,
    ) -> None:
        if not self.register_api or not self.run_id:
            return

        json_payload = {"status": status}
        if metrics_uri:
            json_payload["metrics_uri"] = metrics_uri
        if error_message:
            json_payload["error_message"] = error_message

        response = self.api_client.put(
            f"/runs/{self.run_id}",
            json=json_payload,
        )
        response.raise_for_status()

    def _get_slices(self, config: dict, data: pd.DataFrame):
        if "walk_forward" in config:
            wf_config = config["walk_forward"]
            return walk_forward_split(
                n_samples=len(data),
                train_size=wf_config["train_period"],
                test_size=wf_config["test_period"],
                step_size=wf_config["step_size"],
            )
        if "cpcv" in config:
            cpcv_config = config["cpcv"]
            n_samples = len(data)
            embargo_pct = cpcv_config.get("embargo_pct", 0.0)
            embargo = int(n_samples * embargo_pct)
            return combinatorial_purged_cv(
                n_samples=n_samples,
                n_splits=cpcv_config["N"],
                n_test_splits=cpcv_config["k"],
                embargo=embargo,
            )
        raise ValueError("No valid configuration found for walk_forward or cpcv")

    def run(self, config: dict, data: pd.DataFrame) -> List[dict]:
        self.hyperparams = config.get("strategy_params", {})
        self.config = config
        self._create_run(
            "walk_forward" if "walk_forward" in config else "cpcv",
            configured_run_id=config.get("run_id"),
        )
        self._update_run_status("RUNNING")

        try:
            results = []
            slices = self._get_slices(config, data)

            for i, (train_indices, test_indices) in enumerate(slices):
                train_data = data.iloc[train_indices]
                test_data = data.iloc[test_indices]

                strategy = self.strategy_cls(**self.hyperparams)
                portfolio = self._build_component(
                    self.portfolio_cls, config.get("portfolio_params", {})
                )
                execution = self._build_component(
                    self.execution_cls, config.get("execution_params", {})
                )
                performance = self.performance_cls()

                backtest = Backtest(
                    strategy,
                    portfolio,
                    execution,
                    performance,
                    pl.from_pandas(test_data.reset_index()),
                )
                backtest.run()

                slice_results = {
                    "slice_id": i,
                    "train_start": train_data.index.min(),
                    "train_end": train_data.index.max(),
                    "test_start": test_data.index.min(),
                    "test_end": test_data.index.max(),
                    "metrics": backtest.results["performance"].metrics,
                }
                results.append(slice_results)

            self.results = {"run_id": self.run_id, "slices": results}
            metrics_uri = f"metrics/{self.run_id}_summary.json"
            self.to_json(metrics_uri)
            self._update_run_status("COMPLETED", metrics_uri=metrics_uri)
            return results
        except Exception as e:
            self._update_run_status("FAILED", error_message=str(e))
            raise

    def run_ray(self, config: dict, data: pd.DataFrame) -> List[dict]:
        self.hyperparams = config.get("strategy_params", {})
        self.config = config
        self._create_run(
            "walk_forward" if "walk_forward" in config else "cpcv",
            configured_run_id=config.get("run_id"),
        )
        self._update_run_status("RUNNING")

        try:
            pythonpath_entries = []
            existing_pythonpath = os.environ.get("PYTHONPATH")
            if existing_pythonpath:
                pythonpath_entries.append(existing_pythonpath)
            pythonpath_entries.append(str(Path.cwd() / "src"))
            ray.init(
                ignore_reinit_error=True,
                runtime_env={
                    "env_vars": {
                        "PYTHONPATH": os.pathsep.join(pythonpath_entries)
                    }
                },
            )

            results_refs = []
            slices = self._get_slices(config, data)

            for i, (train_indices, test_indices) in enumerate(slices):
                train_data = data.iloc[train_indices]
                test_data = data.iloc[test_indices]

                results_refs.append(
                    run_backtest_slice.remote(
                        self.strategy_cls.__module__,
                        self.strategy_cls.__name__,
                        self.portfolio_cls,
                        self.execution_cls,
                        self.performance_cls,
                        self.hyperparams,
                        config.get("portfolio_params", {}),
                        config.get("execution_params", {}),
                        train_data,
                        test_data,
                        i,
                    )
                )

            results = ray.get(results_refs)
            self.results = {"run_id": self.run_id, "slices": results}
            metrics_uri = f"metrics/{self.run_id}_summary.json"
            self.to_json(metrics_uri)
            self._update_run_status("COMPLETED", metrics_uri=metrics_uri)
            ray.shutdown()
            return results
        except Exception as e:
            self._update_run_status("FAILED", error_message=str(e))
            ray.shutdown()
            raise

    def to_json(self, filepath: str):
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4, cls=CustomJSONEncoder)

    def generate_reports(self, output_dir: str = "."):
        if not self.run_id:
            raise ValueError("Run ID not set. Please run a backtest first.")

        report_gen = ReportGen(
            self.run_id,
            self.results,
            self.strategy_cls.__name__,
            self.config.get("strategy_params", {}),
        )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 產生 JSON 報告
        json_report = report_gen.generate_json()
        json_filepath = output_path / f"{self.run_id}_report.json"
        with json_filepath.open("w", encoding="utf-8") as f:
            f.write(json_report)

        # 產生 PDF 報告
        pdf_filepath = output_path / f"{self.run_id}_report.pdf"
        report_gen.generate_pdf(pdf_filepath)
