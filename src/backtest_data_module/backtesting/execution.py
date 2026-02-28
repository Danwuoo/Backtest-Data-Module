from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from datetime import date, datetime, timedelta
from numbers import Real
from typing import Any, Dict, List

import numpy as np

from backtest_data_module.backtesting.events import FillEvent, OrderEvent


class CommissionModel(ABC):
    @abstractmethod
    def calculate(self, quantity: float, price: float) -> float:
        pass


class FlatCommission(CommissionModel):
    def __init__(self, fee: float = 0.0005):
        self.fee = fee

    def calculate(self, quantity: float, price: float) -> float:
        return self.fee * abs(quantity) * price


class SlippageModel(ABC):
    @abstractmethod
    def apply(self, price: Any, quantity: float) -> float:
        pass


class GaussianSlippage(SlippageModel):
    def __init__(self, mu: float = 0, sigma: float = 0.001, seed: int = None):
        self.mu = mu
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def apply(self, price: Any, quantity: float) -> float:
        # 支援直接價格值與 {"close": price} 兩種輸入格式
        base_price = price.get("close") if isinstance(price, dict) else price
        return float(base_price) * (1 + self.rng.normal(self.mu, self.sigma))


class LatencyModel(ABC):
    @abstractmethod
    def get_delay(self) -> float:
        pass


class PoissonLatency(LatencyModel):
    def __init__(self, lam: float = 0.01):  # Average delay of 10ms
        self.lam = lam

    def get_delay(self) -> float:
        return np.random.poisson(self.lam)


class Execution:
    def __init__(
        self,
        commission_model: CommissionModel = FlatCommission(),
        slippage_model: SlippageModel = GaussianSlippage(seed=42),
        latency_model: LatencyModel = PoissonLatency(),
    ):
        self.commission_model = commission_model
        self.slippage_model = slippage_model
        self.latency_model = latency_model
        self.order_queue = deque()

    def _normalize_time(self, value: Real | date | datetime) -> float | datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        return float(value)

    def _execution_time(self, timestamp: Real | date | datetime, delay: float):
        normalized = self._normalize_time(timestamp)
        if isinstance(normalized, datetime):
            return normalized + timedelta(seconds=delay)
        return normalized + float(delay)

    def place_order(self, order: OrderEvent, timestamp: Real | date | datetime):
        delay = self.latency_model.get_delay()
        execution_time = self._execution_time(timestamp, delay)
        self.order_queue.append((order, execution_time))

    def process_orders(
        self,
        current_time: Real | date | datetime,
        price_data: Dict[str, Any],
    ) -> List[FillEvent]:
        fills = []
        current_cmp = self._normalize_time(current_time)

        # 依執行時間排序訂單以維持 FIFO
        self.order_queue = deque(sorted(self.order_queue, key=lambda x: x[1]))

        while (
            self.order_queue
            and self._normalize_time(self.order_queue[0][1]) <= current_cmp
        ):
            order, _execution_time = self.order_queue.popleft()

            if order.asset not in price_data:
                # 若該資產無價格資料則略過
                continue

            price = price_data[order.asset]
            slipped_price = self.slippage_model.apply(price, order.quantity)
            commission = self.commission_model.calculate(order.quantity, slipped_price)

            fills.append(
                {
                    "asset": order.asset,
                    "quantity": order.quantity,
                    "price": slipped_price,
                    "commission": commission,
                }
            )
        return fills
