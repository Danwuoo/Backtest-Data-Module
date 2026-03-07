# OKX Trading Platform V2

This documentation covers the V2 repository shape after the move from a narrow control plane to a researchable, replayable, sleeve-ready trading platform.

Use this project when you need:

- a control plane for `demo` and `live` OKX profiles
- a split runtime for market data, inference, portfolio, execution policy, risk, execution, and replay
- deterministic + portfolio-aware risk controls
- Parquet/DuckDB artifacts for replay, backtests, paper runs, and feature storage
- alerting, incidents, and execution attribution without a separate dashboard
