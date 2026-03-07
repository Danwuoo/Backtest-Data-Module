from okx_trading_platform.application.signals import (
    RuleBaselineInferenceProvider,
    position_intent_to_order_plan,
    target_to_position_intent,
)
from okx_trading_platform.domain import (
    InstrumentConfig,
    InstrumentKind,
    SleeveConfig,
)


def test_rule_baseline_provider_emits_target_signal_on_breakout():
    provider = RuleBaselineInferenceProvider(
        profile_id="demo-main",
        strategy_id="reference-breakout",
        model_version_id="reference-breakout-baseline-v1",
        instrument_id="demo-main:BTC-USDT-SWAP",
        inst_id="BTC-USDT-SWAP",
        kind=InstrumentKind.SWAP,
        threshold_bps=10,
        target_size=2,
    )
    signals = list(
        provider.infer_targets(
            {
                "best_bid": 100,
                "best_ask": 101,
                "last_price": 102,
            }
        )
    )

    assert len(signals) == 1
    assert signals[0].side == "buy"
    assert signals[0].target_size == 2


def test_target_signal_converts_to_order_plan():
    provider = RuleBaselineInferenceProvider(
        profile_id="demo-main",
        strategy_id="reference-breakout",
        model_version_id="reference-breakout-baseline-v1",
        instrument_id="demo-main:BTC-USDT-SWAP",
        inst_id="BTC-USDT-SWAP",
        kind=InstrumentKind.SWAP,
        threshold_bps=10,
        target_size=1,
    )
    signal = list(
        provider.infer_targets(
            {
                "best_bid": 100,
                "best_ask": 101,
                "last_price": 99,
            }
        )
    )[0]
    sleeve = SleeveConfig(
        sleeve_id="demo-main-default-sleeve",
        profile_id="demo-main",
        name="primary",
        capital_allocation=0.5,
        risk_budget=0.25,
    )
    instrument = InstrumentConfig(
        instrument_id="demo-main:BTC-USDT-SWAP",
        profile_id="demo-main",
        inst_id="BTC-USDT-SWAP",
        kind=InstrumentKind.SWAP,
        min_notional=5,
    )

    intent = target_to_position_intent(signal, sleeve)
    plan = position_intent_to_order_plan(intent, instrument)

    assert intent.target_size == 0.5
    assert plan.sleeve_id == sleeve.sleeve_id
    assert plan.instrument_id == instrument.instrument_id
