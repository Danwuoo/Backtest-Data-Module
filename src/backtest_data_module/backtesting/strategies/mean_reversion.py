from __future__ import annotations

from typing import List

import polars as pl

from backtest_data_module.backtesting.events import SignalEvent
from backtest_data_module.backtesting.strategy import StrategyBase


class MeanReversion(StrategyBase):
    def __init__(
        self,
        window: int = 20,
        threshold: float = 1.5,
        params: dict | None = None,
    ):
        merged = dict(params or {})
        window = int(merged.get("window", window))
        threshold = float(merged.get("threshold", threshold))
        if window <= 0:
            raise ValueError("window must be positive")

        super().__init__({"window": window, "threshold": threshold})
        self.window = window
        self.threshold = threshold

    def on_data(self, data: pl.DataFrame) -> List[SignalEvent]:
        signals = []
        for asset in data["asset"].unique().to_list():
            asset_data = data.filter(pl.col("asset") == asset)

            # Calculate rolling mean and std
            asset_data = asset_data.with_columns(
                pl.col("close").rolling_mean(window_size=self.window).alias("mean"),
                pl.col("close").rolling_std(window_size=self.window).alias("std"),
            )

            # Generate signals
            asset_data = asset_data.with_columns(
                pl.when(
                    asset_data["close"]
                    > asset_data["mean"] + self.threshold * asset_data["std"]
                )
                .then(-1)
                .when(
                    asset_data["close"]
                    < asset_data["mean"] - self.threshold * asset_data["std"]
                )
                .then(1)
                .otherwise(0)
                .alias("signal")
            )

            # Create SignalEvents
            for row in asset_data.iter_rows(named=True):
                if row["signal"] == 1:
                    signals.append(
                        SignalEvent(asset=asset, quantity=100, direction="long")
                    )
                elif row["signal"] == -1:
                    signals.append(
                        SignalEvent(asset=asset, quantity=-100, direction="short")
                    )
        return signals
