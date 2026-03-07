from okx_trading_platform.adapters.okx import OkxOrderBook


def test_orderbook_detects_gap_when_prev_seq_does_not_match():
    book = OkxOrderBook("BTC-USDT-SWAP", "demo-main")
    book.apply_snapshot([["100", "2"]], [["101", "3"]], sequence_id=10)
    snapshot = book.apply_update(
        [["100", "1"]],
        [["101", "2"]],
        sequence_id=12,
        prev_sequence_id=8,
    )

    assert snapshot.gap_detected is True
    assert snapshot.prev_sequence_id == 8


def test_orderbook_rebuild_uses_snapshot_state():
    book = OkxOrderBook("BTC-USDT-SWAP", "demo-main")
    snapshot = book.apply_snapshot(
        [["100", "2"]],
        [["101", "3"]],
        instrument_id="demo-main:BTC-USDT-SWAP",
        sequence_id=1,
    )

    rebuilt = OkxOrderBook("BTC-USDT-SWAP", "demo-main")
    rebuilt.rebuild(snapshot)

    assert rebuilt.best_bid().price == 100.0
    assert rebuilt.best_ask().price == 101.0
    assert rebuilt.instrument_id == "demo-main:BTC-USDT-SWAP"
