# OKX Trading Platform

This documentation covers the production-facing shape of the repository after the migration away from research and backtesting modules.

Use this project when you need:

- a control plane for OKX `demo` and `live` profiles
- a thin service topology that fits a single VPS
- deterministic order validation and kill-switch controls
- a CLI for operations without a separate web dashboard

The system is designed to be extended with additional signal providers later, but the current repository is intentionally optimized for platform reliability first.
