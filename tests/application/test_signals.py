from okx_trading_platform.application.signals import (
    ManualSignalProvider,
    ReferenceBreakoutSignalProvider,
    signals_to_order_intents,
)
from okx_trading_platform.domain import InstrumentKind, SignalEnvelope, TradingProfile


def test_manual_signal_provider_yields_enqueued_signal():
    provider = ManualSignalProvider()
    signal = SignalEnvelope(
        bot_name="manual",
        inst_id="BTC-USDT-SWAP",
        profile=TradingProfile.DEMO,
        instrument_kind=InstrumentKind.SWAP,
        side="buy",
        size=1,
    )
    provider.enqueue(signal)
    signals = list(provider.next_signals({}))
    assert len(signals) == 1
    assert signals[0].bot_name == "manual"


def test_reference_breakout_provider_emits_signal():
    provider = ReferenceBreakoutSignalProvider(
        bot_name="reference",
        profile=TradingProfile.DEMO,
        inst_id="BTC-USDT-SWAP",
        instrument_kind=InstrumentKind.SWAP,
        trigger_spread=0.001,
        size=1,
    )
    signals = list(
        provider.next_signals(
            {"best_bid": 100, "best_ask": 101, "last_price": 102}
        )
    )
    assert signals
    intents = signals_to_order_intents(signals)
    assert intents[0].td_mode == "isolated"
