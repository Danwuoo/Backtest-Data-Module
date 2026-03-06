# Development

## Repository rules

- keep platform logic in `okx_trading_platform`
- keep route handlers thin
- keep exchange payload normalization inside adapter modules
- keep database access inside repository classes

## Local validation

```bash
flake8 .
pytest -q
```

## Testing focus

- API behavior and kill-switch flow
- OKX signing and request routing
- order book merge and rebuild behavior
- risk limits
- CLI command delegation
- service health contracts
