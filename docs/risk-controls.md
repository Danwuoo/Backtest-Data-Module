# Risk Controls

The v1 platform includes deterministic pre-trade checks:

- kill switch
- stale market data block
- consecutive error circuit breaker
- maximum open orders
- daily loss guard
- minimum notional
- isolated margin buffer
- available balance validation
- maximum position exposure

Risk evaluation is implemented in `okx_trading_platform.domain.risk.RiskManager`. Trading logic must go through the control plane or an equivalent risk gate before reaching the execution gateway.
