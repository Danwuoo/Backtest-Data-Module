from __future__ import annotations

from collections import defaultdict, deque
from typing import List

import polars as pl

from backtest_data_module.backtesting.events import MarketEvent, SignalEvent
from backtest_data_module.backtesting.strategy import StrategyBase


class SmaCrossover(StrategyBase):
    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 30,
        params: dict | None = None,
    ):
        merged = dict(params or {})
        short_window = int(merged.get("short_window", short_window))
        long_window = int(merged.get("long_window", long_window))
        if short_window <= 0 or long_window <= 0:
            raise ValueError("window size must be positive")
        if short_window > long_window:
            raise ValueError("short_window must be <= long_window")

        super().__init__(
            {
                "short_window": short_window,
                "long_window": long_window,
            }
        )
        self.short_window = short_window
        self.long_window = long_window
        self._history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.long_window + 1)
        )

    def _signal_from_prices(self, prices: list[float], asset: str) -> list[SignalEvent]:
        if len(prices) <= self.long_window:
            return []

        short_sma = sum(prices[-self.short_window :]) / self.short_window
        long_sma = sum(prices[-self.long_window :]) / self.long_window
        if short_sma > long_sma:
            return [SignalEvent(asset=asset, quantity=100, direction="long")]
        if short_sma < long_sma:
            return [SignalEvent(asset=asset, quantity=-100, direction="short")]
        return []

    def _on_market_event(self, event: MarketEvent) -> List[SignalEvent]:
        if not event.data:
            return []

        signals: list[SignalEvent] = []
        for asset, row in event.data.items():
            if "close" not in row:
                continue
            self._history[asset].append(float(row["close"]))
            signals.extend(self._signal_from_prices(list(self._history[asset]), asset))
        return signals

    def _on_dataframe(self, data: pl.DataFrame) -> List[SignalEvent]:
        signals: list[SignalEvent] = []
        for asset in data["asset"].unique().to_list():
            asset_prices = (
                data.filter(pl.col("asset") == asset)
                .select("close")
                .to_series()
                .to_list()
            )
            for idx in range(self.long_window - 1, len(asset_prices)):
                signals.extend(
                    self._signal_from_prices(asset_prices[: idx + 1], asset)
                )
        return signals

    def on_data(self, data: pl.DataFrame | MarketEvent) -> List[SignalEvent]:
        if isinstance(data, MarketEvent):
            return self._on_market_event(data)
        return self._on_dataframe(data)
