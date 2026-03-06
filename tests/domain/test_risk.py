from okx_trading_platform.domain import (
    BalanceSnapshot,
    InstrumentKind,
    OrderIntent,
    OrderSide,
    TdMode,
    TradingProfile,
)
from okx_trading_platform.domain.risk import RiskLimits, RiskManager


def test_risk_service_blocks_when_kill_switch_active():
    service = RiskManager()
    service.activate_kill_switch("manual stop")
    decision = service.evaluate_order(
        OrderIntent(
            profile=TradingProfile.DEMO,
            instrument_kind=InstrumentKind.SWAP,
            inst_id="BTC-USDT-SWAP",
            side=OrderSide.BUY,
            size=1,
            td_mode=TdMode.ISOLATED,
            metadata={"mark_price": 10},
        ),
        balance=BalanceSnapshot(
            profile=TradingProfile.DEMO,
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
    assert decision.reason == "manual stop"


def test_risk_service_blocks_on_insufficient_balance():
    service = RiskManager(limits=RiskLimits(min_notional=5, max_position_notional=100))
    decision = service.evaluate_order(
        OrderIntent(
            profile=TradingProfile.DEMO,
            instrument_kind=InstrumentKind.SWAP,
            inst_id="BTC-USDT-SWAP",
            side=OrderSide.BUY,
            size=2,
            td_mode=TdMode.ISOLATED,
        ),
        balance=BalanceSnapshot(
            profile=TradingProfile.DEMO,
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
