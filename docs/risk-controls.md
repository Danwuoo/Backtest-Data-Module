# Risk Controls

V2 risk control is no longer limited to deterministic pre-trade checks.

## Pre-trade

- platform kill switch
- per-sleeve and per-strategy kill switch
- stale market data block
- max open orders
- min notional and instrument min size
- available balance validation
- isolated margin buffer
- position exposure cap
- spread, impact, volatility, and latency guards

## Post-trade and portfolio

- execution latency degradation guard
- funding exposure cap
- liquidation distance guard
- hourly and daily loss limits
- consecutive-loss cooldown
- persisted `risk_snapshots` for accepted and rejected flow

Risk evaluation is implemented in `okx_trading_platform.domain.risk.RiskManager`. All live order flow should pass through the control plane or an equivalent risk gate before reaching the execution gateway.
