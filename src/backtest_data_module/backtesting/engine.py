from __future__ import annotations

import json
import warnings
from collections import deque

import numpy as np
import polars as pl

try:
    import cupy as cp
    from cupy import cuda

    CUPY_AVAILABLE = True
except Exception:  # noqa: BLE001
    cp = None
    cuda = None
    CUPY_AVAILABLE = False

from backtest_data_module.backtesting.events import FillEvent, OrderEvent, SignalEvent
from backtest_data_module.backtesting.execution import Execution
from backtest_data_module.backtesting.performance import Performance
from backtest_data_module.backtesting.portfolio import Portfolio
from backtest_data_module.backtesting.strategy import StrategyBase
from backtest_data_module.data_handler import DataHandler
from backtest_data_module.utils.profiler import Profiler


class Backtest:
    def __init__(
        self,
        strategy: StrategyBase,
        portfolio: Portfolio,
        execution: Execution,
        performance: Performance,
        data: pl.DataFrame,
    ):
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution = execution
        self.performance = performance
        self.data = data
        self.events = deque()
        self.results = {}
        self.device = self.strategy.device
        self.precision = self.strategy.precision
        self.quantization_bits = self.strategy.quantization_bits
        self.time_col = (
            "date"
            if "date" in self.data.columns
            else "timestamp" if "timestamp" in self.data.columns else None
        )
        if self.time_col is None:
            raise ValueError("Backtest data must contain 'date' or 'timestamp' column")

        if self.device == "cuda" and not CUPY_AVAILABLE:
            warnings.warn("cupy 未安裝，將回退至 CPU 執行")
            self.device = "cpu"
        self.profiler = Profiler() if self.device == "cuda" and cp else None
        self.data_handler = DataHandler(None)

    def run(self):
        if self.profiler:
            self.profiler.start()

        if self.device == "cuda" and cp:
            data = cp.asarray(self.data.to_numpy())
            if self.quantization_bits:
                data = self.data_handler.quantize(data, self.quantization_bits)
        else:
            data = self.data

        signals = self.strategy.on_data(data)
        for signal in signals:
            self.events.append(signal)

        if self.device == "cuda" and cp:
            graph = cuda.Graph()
            with graph.capture():
                while self.events:
                    event = self.events.popleft()

                    if isinstance(event, SignalEvent):
                        order = OrderEvent(asset=event.asset, quantity=event.quantity)
                        self.events.append(order)
                    elif isinstance(event, OrderEvent):
                        timestamp = (
                            self.data.filter(pl.col("asset") == event.asset)
                            .select(self.time_col)
                            .row(0)[0]
                        )
                        self.execution.place_order(event, timestamp)
                    elif isinstance(event, FillEvent):
                        self.portfolio.update([event.__dict__])
            self.graph_exec = graph.instantiate()
            self.graph_exec.launch()
        else:
            while self.events:
                event = self.events.popleft()

                if isinstance(event, SignalEvent):
                    order = OrderEvent(asset=event.asset, quantity=event.quantity)
                    self.events.append(order)
                elif isinstance(event, OrderEvent):
                    timestamp = (
                        self.data.filter(pl.col("asset") == event.asset)
                        .select(self.time_col)
                        .row(0)[0]
                    )
                    self.execution.place_order(event, timestamp)
                elif isinstance(event, FillEvent):
                    self.portfolio.update([event.__dict__])

        # 在回測結束時處理所有訂單
        for timestamp in self.data[self.time_col].unique():
            market_data_at_timestamp = self.data.filter(
                pl.col(self.time_col) == timestamp
            ).to_dict(as_series=False)
            market_data_dict = {}
            for i in range(len(market_data_at_timestamp["asset"])):
                asset = market_data_at_timestamp["asset"][i]
                market_data_dict[asset] = {
                    "close": market_data_at_timestamp["close"][i]
                }

            fills = self.execution.process_orders(timestamp, market_data_dict)
            for fill in fills:
                self.portfolio.update([fill])

        # 更新投資組合績效
        if self.device == "cuda" and cp:
            xp = cp
            if self.precision == "amp":
                cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        else:
            xp = np

        for timestamp in self.data[self.time_col].unique():
            last_prices = (
                self.data.filter(pl.col(self.time_col) <= timestamp)
                .group_by("asset")
                .last()
                .select(["asset", "close"])
                .to_dict()
            )
            last_prices = {
                a: p
                for a, p in zip(last_prices["asset"], last_prices["close"])
            }
            self.performance.nav_series.append(
                self.portfolio.cash
                + sum(
                    p.market_value(last_prices.get(asset, 0))
                    for asset, p in self.portfolio.positions.items()
                )
            )

        self.performance.returns = (
            xp.diff(xp.asarray(self.performance.nav_series))
            / xp.asarray(self.performance.nav_series[:-1])
            if self.performance.nav_series and len(self.performance.nav_series) > 1
            else xp.array([])
        )

        last_prices = (
            self.data.group_by("asset")
            .last()
            .select(["asset", "close"])
            .to_dict()
        )
        last_prices = {
            a: p
            for a, p in zip(last_prices["asset"], last_prices["close"])
        }
        self.results = {
            "pnl": self.portfolio.get_pnl(last_prices),
            "fills": self.portfolio.fills,
            "performance": self.performance.compute_metrics(),
        }

        if self.profiler:
            self.profiler.stop()
            self.profiler.print_report()

    def to_json(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)
