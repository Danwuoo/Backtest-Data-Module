from __future__ import annotations

from dataclasses import dataclass

from backtest_data_module.trading.domain import (
    BalanceSnapshot,
    KillSwitchState,
    OrderIntent,
    PositionSnapshot,
    RiskDecision,
    enum_value,
)


@dataclass(frozen=True)
class RiskLimits:
    min_notional: float = 5.0
    isolated_margin_buffer_ratio: float = 0.15
    max_position_notional: float = 25.0
    max_open_orders: int = 4
    max_daily_loss: float = 5.0
    max_consecutive_errors: int = 3


class RiskService:
    """所有訂單意圖先經過這一層，避免策略直接打到交易所。"""

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()
        self.kill_switch = KillSwitchState()

    def activate_kill_switch(self, reason: str) -> KillSwitchState:
        self.kill_switch = KillSwitchState(activated=True, reason=reason)
        return self.kill_switch

    def release_kill_switch(self) -> KillSwitchState:
        self.kill_switch = KillSwitchState(activated=False, reason=None)
        return self.kill_switch

    def evaluate_order(
        self,
        intent: OrderIntent,
        *,
        balance: BalanceSnapshot | None,
        positions: list[PositionSnapshot],
        open_orders_count: int,
        mark_price: float | None,
        daily_realized_pnl: float = 0.0,
        consecutive_errors: int = 0,
        market_data_fresh: bool = True,
    ) -> RiskDecision:
        applied_limits = {
            "min_notional": self.limits.min_notional,
            "max_position_notional": self.limits.max_position_notional,
            "max_open_orders": self.limits.max_open_orders,
        }

        if self.kill_switch.activated:
            return RiskDecision(
                approved=False,
                reason=self.kill_switch.reason or "kill switch activated",
                applied_limits=applied_limits,
            )
        if not market_data_fresh:
            return RiskDecision(
                approved=False,
                reason="market data stale",
                applied_limits=applied_limits,
            )
        if consecutive_errors >= self.limits.max_consecutive_errors:
            return RiskDecision(
                approved=False,
                reason="execution circuit breaker active",
                applied_limits=applied_limits,
            )
        if open_orders_count >= self.limits.max_open_orders:
            return RiskDecision(
                approved=False,
                reason="too many open orders",
                applied_limits=applied_limits,
            )
        if daily_realized_pnl <= -abs(self.limits.max_daily_loss):
            return RiskDecision(
                approved=False,
                reason="daily loss limit reached",
                applied_limits=applied_limits,
            )
        if mark_price is None:
            return RiskDecision(
                approved=False,
                reason="missing mark price",
                applied_limits=applied_limits,
            )

        notional = intent.size * mark_price
        if notional < self.limits.min_notional:
            return RiskDecision(
                approved=False,
                reason="order notional below minimum",
                applied_limits=applied_limits,
            )

        if balance is None:
            return RiskDecision(
                approved=False,
                reason="missing balance snapshot",
                applied_limits=applied_limits,
            )

        required_equity = notional
        if enum_value(intent.td_mode) == "isolated":
            required_equity *= 1 + self.limits.isolated_margin_buffer_ratio

        if balance.available < required_equity:
            return RiskDecision(
                approved=False,
                reason="insufficient available balance",
                applied_limits=applied_limits,
            )

        current_notional = 0.0
        for position in positions:
            if position.inst_id == intent.inst_id:
                current_notional = abs(position.quantity) * mark_price
                break

        if current_notional + notional > self.limits.max_position_notional:
            return RiskDecision(
                approved=False,
                reason="position exposure limit exceeded",
                applied_limits=applied_limits,
            )

        return RiskDecision(approved=True, applied_limits=applied_limits)
