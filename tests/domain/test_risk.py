from okx_trading_platform.domain import (
    BalanceSnapshot,
    InstrumentConfig,
    InstrumentKind,
    OrderPlan,
    OrderSide,
    RiskPolicyConfig,
    SleeveConfig,
    StrategyConfig,
    StrategyStatus,
    TdMode,
)
from okx_trading_platform.domain.risk import RiskLimits, RiskManager


def test_risk_service_blocks_when_platform_kill_switch_active():
    manager = RiskManager()
    manager.activate_kill_switch("manual stop")
    decision = manager.evaluate_order(
        OrderPlan(
            profile_id="demo-main",
            strategy_id="strategy-a",
            model_version_id="model-a",
            sleeve_id="demo-main-default-sleeve",
            instrument_id="demo-main:BTC-USDT-SWAP",
            inst_id="BTC-USDT-SWAP",
            kind=InstrumentKind.SWAP,
            side=OrderSide.BUY,
            size=1,
            td_mode=TdMode.ISOLATED,
        ),
        instrument=InstrumentConfig(
            instrument_id="demo-main:BTC-USDT-SWAP",
            profile_id="demo-main",
            inst_id="BTC-USDT-SWAP",
            kind=InstrumentKind.SWAP,
            min_notional=5,
        ),
        sleeve=SleeveConfig(
            sleeve_id="demo-main-default-sleeve",
            profile_id="demo-main",
            name="primary",
        ),
        strategy=StrategyConfig(
            strategy_id="strategy-a",
            profile_id="demo-main",
            name="strategy-a",
            status=StrategyStatus.ENABLED,
        ),
        policy=RiskPolicyConfig(
            risk_policy_id="policy-a",
            profile_id="demo-main",
            name="default",
        ),
        balance=BalanceSnapshot(
            profile_id="demo-main",
            balance_snapshot_id="demo-main:USDT",
            currency="USDT",
            available=100,
            cash_balance=100,
            equity=100,
        ),
        positions=[],
        open_orders_count=0,
        mark_price=10,
    )
    assert decision.approved is False
    assert decision.reason == "platform kill switch activated"


def test_risk_service_blocks_on_insufficient_balance():
    manager = RiskManager(limits=RiskLimits(min_notional=5, max_position_notional=100))
    decision = manager.evaluate_order(
        OrderPlan(
            profile_id="demo-main",
            strategy_id="strategy-a",
            model_version_id="model-a",
            sleeve_id="demo-main-default-sleeve",
            instrument_id="demo-main:BTC-USDT-SWAP",
            inst_id="BTC-USDT-SWAP",
            kind=InstrumentKind.SWAP,
            side=OrderSide.BUY,
            size=2,
            td_mode=TdMode.ISOLATED,
        ),
        instrument=InstrumentConfig(
            instrument_id="demo-main:BTC-USDT-SWAP",
            profile_id="demo-main",
            inst_id="BTC-USDT-SWAP",
            kind=InstrumentKind.SWAP,
            min_notional=5,
        ),
        sleeve=SleeveConfig(
            sleeve_id="demo-main-default-sleeve",
            profile_id="demo-main",
            name="primary",
        ),
        strategy=StrategyConfig(
            strategy_id="strategy-a",
            profile_id="demo-main",
            name="strategy-a",
            status=StrategyStatus.ENABLED,
        ),
        policy=RiskPolicyConfig(
            risk_policy_id="policy-a",
            profile_id="demo-main",
            name="default",
        ),
        balance=BalanceSnapshot(
            profile_id="demo-main",
            balance_snapshot_id="demo-main:USDT",
            currency="USDT",
            available=10,
            cash_balance=10,
            equity=10,
        ),
        positions=[],
        open_orders_count=0,
        mark_price=10,
    )
    assert decision.approved is False
    assert decision.reason == "insufficient available balance"
