from __future__ import annotations

from dataclasses import dataclass

from .models import (
    BalanceSnapshot,
    InstrumentConfig,
    KillSwitchState,
    OrderPlan,
    PositionSnapshot,
    RiskDecision,
    RiskPolicyConfig,
    RiskSnapshot,
    SleeveConfig,
    StrategyConfig,
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
    max_leverage: float = 3.0
    max_loss_per_trade: float = 3.0
    max_loss_per_hour: float = 8.0
    max_loss_per_day: float = 15.0
    max_funding_abs: float = 1.0
    min_liquidation_distance_ratio: float = 0.05
    max_latency_ms: float = 1500.0


class RiskManager:
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
        plan: OrderPlan,
        *,
        instrument: InstrumentConfig | None,
        sleeve: SleeveConfig | None,
        strategy: StrategyConfig | None,
        policy: RiskPolicyConfig | None,
        balance: BalanceSnapshot | None,
        positions: list[PositionSnapshot],
        open_orders_count: int,
        mark_price: float | None,
        daily_realized_pnl: float = 0.0,
        hourly_realized_pnl: float = 0.0,
        funding_cost: float = 0.0,
        consecutive_errors: int = 0,
        consecutive_losses: int = 0,
        liquidation_distance_ratio: float | None = None,
        volatility_regime: str | None = None,
        spread_bps: float | None = None,
        impact_bps: float | None = None,
        observed_latency_ms: float | None = None,
        market_data_fresh: bool = True,
    ) -> RiskDecision:
        applied_limits = self._policy_limits(policy)
        min_notional = plan.min_notional or applied_limits["min_notional"]
        max_position_notional = applied_limits["max_position_notional"]
        max_leverage = applied_limits["max_leverage"]

        if self.kill_switch.activated:
            return self._rejected(applied_limits, "platform kill switch activated")
        if sleeve and sleeve.is_killed:
            return self._rejected(applied_limits, "sleeve kill switch activated")
        if strategy and strategy.is_killed:
            return self._rejected(applied_limits, "strategy kill switch activated")
        if not market_data_fresh:
            return self._rejected(applied_limits, "market data is stale")
        if consecutive_errors >= applied_limits["max_consecutive_errors"]:
            return self._rejected(applied_limits, "execution circuit breaker is active")
        if consecutive_losses >= 3:
            return self._rejected(applied_limits, "consecutive loss cooldown is active")
        if open_orders_count >= applied_limits["max_open_orders"]:
            return self._rejected(applied_limits, "too many open orders")
        if daily_realized_pnl <= -abs(applied_limits["max_loss_per_day"]):
            return self._rejected(applied_limits, "daily loss limit reached")
        if hourly_realized_pnl <= -abs(applied_limits["max_loss_per_hour"]):
            return self._rejected(applied_limits, "hourly loss limit reached")
        if abs(funding_cost) > applied_limits["max_funding_abs"]:
            return self._rejected(applied_limits, "funding exposure limit reached")
        if (
            observed_latency_ms
            and observed_latency_ms > applied_limits["max_latency_ms"]
        ):
            return self._rejected(applied_limits, "latency degradation guard triggered")
        if spread_bps and spread_bps > 35:
            return self._rejected(applied_limits, "spread guard triggered")
        if impact_bps and impact_bps > 50:
            return self._rejected(applied_limits, "impact guard triggered")
        if volatility_regime == "halt":
            return self._rejected(applied_limits, "volatility regime guard triggered")
        if mark_price is None:
            return self._rejected(applied_limits, "missing mark price")

        notional = (plan.notional or 0.0) or plan.size * mark_price
        if notional < min_notional:
            return self._rejected(applied_limits, "order notional is below the minimum")
        if (
            instrument
            and instrument.min_notional
            and notional < instrument.min_notional
        ):
            return self._rejected(applied_limits, "instrument min notional not met")
        if instrument and instrument.min_size and plan.size < instrument.min_size:
            return self._rejected(applied_limits, "instrument min size not met")
        if sleeve and sleeve.max_leverage > max_leverage:
            return self._rejected(applied_limits, "sleeve leverage exceeds risk policy")
        if (
            liquidation_distance_ratio is not None
            and liquidation_distance_ratio
            < applied_limits["min_liquidation_distance_ratio"]
        ):
            return self._rejected(
                applied_limits, "liquidation distance guard triggered"
            )
        if balance is None:
            return self._rejected(applied_limits, "missing balance snapshot")

        required_equity = notional
        if enum_value(plan.td_mode) == "isolated":
            required_equity *= 1 + applied_limits["isolated_margin_buffer_ratio"]
        if balance.available < required_equity:
            return self._rejected(applied_limits, "insufficient available balance")

        current_notional = 0.0
        for position in positions:
            if position.inst_id == plan.inst_id:
                current_notional = abs(position.quantity) * mark_price
                break
        if current_notional + notional > max_position_notional:
            return self._rejected(applied_limits, "position exposure limit exceeded")

        metrics = {
            "notional": notional,
            "required_equity": required_equity,
            "open_orders_count": open_orders_count,
            "daily_realized_pnl": daily_realized_pnl,
        }
        return RiskDecision(
            approved=True,
            stage="pre_trade",
            applied_limits=applied_limits,
            metrics=metrics,
        )

    def build_snapshot(
        self,
        *,
        profile_id: str,
        order_id: str | None,
        strategy_id: str | None,
        sleeve_id: str | None,
        decision: RiskDecision,
    ) -> RiskSnapshot:
        return RiskSnapshot(
            profile_id=profile_id,
            order_id=order_id,
            strategy_id=strategy_id,
            sleeve_id=sleeve_id,
            stage=decision.stage,
            approved=decision.approved,
            reason=decision.reason,
            applied_limits=decision.applied_limits,
            metrics=decision.metrics,
        )

    def portfolio_snapshot(
        self,
        *,
        profile_id: str,
        strategy_id: str | None,
        sleeve_id: str | None,
        realized_pnl: float,
        unrealized_pnl: float,
    ) -> RiskSnapshot:
        approved = realized_pnl > -abs(self.limits.max_loss_per_day)
        reason = None if approved else "portfolio drawdown limit reached"
        return RiskSnapshot(
            profile_id=profile_id,
            strategy_id=strategy_id,
            sleeve_id=sleeve_id,
            stage="portfolio",
            approved=approved,
            reason=reason,
            applied_limits={"max_loss_per_day": self.limits.max_loss_per_day},
            metrics={
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
            },
        )

    def execution_snapshot(
        self,
        *,
        profile_id: str,
        order_id: str,
        latency_ms: float | None,
    ) -> RiskSnapshot:
        approved = latency_ms is None or latency_ms <= self.limits.max_latency_ms
        reason = None if approved else "latency degradation guard triggered"
        return RiskSnapshot(
            profile_id=profile_id,
            order_id=order_id,
            stage="post_trade",
            approved=approved,
            reason=reason,
            applied_limits={"max_latency_ms": self.limits.max_latency_ms},
            metrics={"latency_ms": latency_ms},
        )

    def _policy_limits(self, policy: RiskPolicyConfig | None) -> dict[str, float]:
        raw = policy.config if policy else {}
        return {
            "min_notional": float(raw.get("min_notional", self.limits.min_notional)),
            "isolated_margin_buffer_ratio": float(
                raw.get(
                    "isolated_margin_buffer_ratio",
                    self.limits.isolated_margin_buffer_ratio,
                )
            ),
            "max_position_notional": float(
                raw.get(
                    "max_position_notional",
                    self.limits.max_position_notional,
                )
            ),
            "max_open_orders": int(
                raw.get("max_open_orders", self.limits.max_open_orders)
            ),
            "max_consecutive_errors": int(
                raw.get(
                    "max_consecutive_errors",
                    self.limits.max_consecutive_errors,
                )
            ),
            "max_leverage": float(raw.get("max_leverage", self.limits.max_leverage)),
            "max_loss_per_hour": float(
                raw.get("max_loss_per_hour", self.limits.max_loss_per_hour)
            ),
            "max_loss_per_day": float(
                raw.get("max_loss_per_day", self.limits.max_loss_per_day)
            ),
            "max_funding_abs": float(
                raw.get("max_funding_abs", self.limits.max_funding_abs)
            ),
            "min_liquidation_distance_ratio": float(
                raw.get(
                    "min_liquidation_distance_ratio",
                    self.limits.min_liquidation_distance_ratio,
                )
            ),
            "max_latency_ms": float(
                raw.get("max_latency_ms", self.limits.max_latency_ms)
            ),
        }

    @staticmethod
    def _rejected(applied_limits: dict[str, float], reason: str) -> RiskDecision:
        return RiskDecision(
            approved=False,
            stage="pre_trade",
            reason=reason,
            applied_limits=applied_limits,
        )
