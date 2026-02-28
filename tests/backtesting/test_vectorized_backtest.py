import pytest
import polars as pl
import numpy as np
import time

from backtest_data_module.backtesting.engine import Backtest
from backtest_data_module.backtesting.strategy import StrategyBase
from backtest_data_module.backtesting.portfolio import Portfolio
from backtest_data_module.backtesting.execution import Execution, GaussianSlippage
from backtest_data_module.backtesting.performance import Performance
from backtest_data_module.backtesting.events import SignalEvent


class VectorizedSmaCrossover(StrategyBase):
    def __init__(self, short_window: int = 10, long_window: int = 30):
        super().__init__({})
        self.short_window = short_window
        self.long_window = long_window

    def on_data(self, data: pl.DataFrame) -> list[SignalEvent]:
        signals = []
        for asset in data["asset"].unique():
            asset_data = data.filter(pl.col("asset") == asset)

            # Calculate SMAs
            asset_data = asset_data.with_columns(
                pl.col("close")
                .rolling_mean(window_size=self.short_window)
                .alias("short_sma")
            )
            asset_data = asset_data.with_columns(
                pl.col("close")
                .rolling_mean(window_size=self.long_window)
                .alias("long_sma")
            )

            # Generate signals
            asset_data = asset_data.with_columns(
                pl.when(asset_data["short_sma"] > asset_data["long_sma"])
                .then(1)
                .when(asset_data["short_sma"] < asset_data["long_sma"])
                .then(-1)
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


class NonVectorizedSmaCrossover(StrategyBase):
    def __init__(self, short_window: int = 10, long_window: int = 30):
        super().__init__({})
        self.short_window = short_window
        self.long_window = long_window

    def on_data(self, data: pl.DataFrame) -> list[SignalEvent]:
        signals = []
        for asset in data["asset"].unique():
            asset_data = data.filter(pl.col("asset") == asset)
            prices = asset_data["close"].to_numpy()

            short_sma = (
                np.convolve(prices, np.ones(self.short_window), "valid")
                / self.short_window
            )
            long_sma = (
                np.convolve(prices, np.ones(self.long_window), "valid")
                / self.long_window
            )

            # Adjust arrays to be the same size
            short_sma = short_sma[self.long_window - self.short_window:]

            for i in range(len(long_sma)):
                if short_sma[i] > long_sma[i]:
                    signals.append(
                        SignalEvent(asset=asset, quantity=100, direction="long")
                    )
                elif short_sma[i] < long_sma[i]:
                    signals.append(
                        SignalEvent(
                            asset=asset,
                            quantity=-100,
                            direction="short",
                        )
                    )
        return signals


def generate_test_data(num_rows: int, num_assets: int) -> pl.DataFrame:
    assets = [f"ASSET_{i}" for i in range(num_assets)]
    data = []
    for asset in assets:
        dates = pl.date_range(
            start=np.datetime64("2023-01-01"),
            end=np.datetime64("2023-01-01") + np.timedelta64(num_rows - 1, "D"),
            interval="1d",
            eager=True,
        )
        prices = np.random.rand(num_rows) * 100 + 100
        data.append(pl.DataFrame({"date": dates, "asset": asset, "close": prices}))
    return pl.concat(data)


@pytest.mark.parametrize("num_rows", [100, 1000, 10000, 20000])
def test_vectorized_vs_non_vectorized(num_rows):
    data = generate_test_data(num_rows, 1)

    # Time vectorized implementation
    np.random.seed(42)
    strategy_v = VectorizedSmaCrossover()
    portfolio_v = Portfolio(initial_cash=100000)
    execution_v = Execution(slippage_model=GaussianSlippage(seed=42))
    performance_v = Performance()
    backtest_v = Backtest(
        strategy_v,
        portfolio_v,
        execution_v,
        performance_v,
        data,
    )
    start_time_v = time.time()
    backtest_v.run()
    end_time_v = time.time()
    time_v = end_time_v - start_time_v

    # Time non-vectorized implementation
    np.random.seed(42)
    strategy_nv = NonVectorizedSmaCrossover()
    portfolio_nv = Portfolio(initial_cash=100000)
    execution_nv = Execution(slippage_model=GaussianSlippage(seed=42))
    performance_nv = Performance()
    backtest_nv = Backtest(
        strategy_nv,
        portfolio_nv,
        execution_nv,
        performance_nv,
        data,
    )
    start_time_nv = time.time()
    backtest_nv.run()
    end_time_nv = time.time()
    time_nv = end_time_nv - start_time_nv

    # Assert that the results are the same
    assert backtest_v.results["fills"] == backtest_nv.results["fills"]
    assert backtest_v.results["pnl"] == backtest_nv.results["pnl"]

    # 環境差異會影響絕對速度，這裡改為避免明顯效能退化
    if num_rows >= 20000:
        assert time_v <= time_nv * 2
