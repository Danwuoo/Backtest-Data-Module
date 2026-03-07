# CLI

Install the package in editable mode to get the `okx-platform` command.

## Commands

- `okx-platform status`
- `okx-platform profiles`
- `okx-platform strategies --profile-id demo-main`
- `okx-platform models --profile-id demo-main`
- `okx-platform sleeves --profile-id demo-main`
- `okx-platform allocators --profile-id demo-main`
- `okx-platform runs backtests --profile-id demo-main`
- `okx-platform runs paper --profile-id demo-main`
- `okx-platform runs live --profile-id demo-main`
- `okx-platform incidents --profile-id demo-main`
- `okx-platform alerts --profile-id demo-main`
- `okx-platform order --profile-id demo-main --strategy-id ... --model-version-id ... --sleeve-id ... --instrument-id ... --inst-id BTC-USDT-SWAP --instrument-kind swap --side buy --size 1`
- `okx-platform stop-all --reason "manual intervention"`
- `okx-platform migrate`
- `okx-platform cutover`
