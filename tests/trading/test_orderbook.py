from backtest_data_module.trading.domain import TradingProfile
from backtest_data_module.trading.orderbook import OkxOrderBook


def test_orderbook_apply_update_removes_zero_size_levels():
    book = OkxOrderBook("BTC-USDT-SWAP", TradingProfile.DEMO)
    book.apply_snapshot([["100", "2"], ["99", "1"]], [["101", "3"]])
    book.apply_update([["100", "0"], ["98", "4"]], [["101", "2"]])
    snapshot = book.snapshot()

    assert [level.price for level in snapshot.bids] == [99.0, 98.0]
    assert snapshot.asks[0].size == 2.0


def test_orderbook_rebuild_uses_snapshot_state():
    book = OkxOrderBook("BTC-USDT-SWAP", TradingProfile.DEMO)
    book.apply_snapshot([["100", "2"]], [["101", "3"]], sequence_id=1)
    snapshot = book.snapshot()

    rebuilt = OkxOrderBook("BTC-USDT-SWAP", TradingProfile.DEMO)
    rebuilt.rebuild(snapshot)

    assert rebuilt.best_bid().price == 100.0
    assert rebuilt.best_ask().price == 101.0
    assert rebuilt.sequence_id == 1
