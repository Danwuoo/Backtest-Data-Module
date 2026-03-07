from __future__ import annotations

import json

import typer

from .client import ControlApiClient

app = typer.Typer(help="Operate the OKX Trading Platform control plane.")
runs_app = typer.Typer(help="Inspect backtest, paper, and live runs.")
app.add_typer(runs_app, name="runs")


def _platform_client(base_url: str | None = None) -> ControlApiClient:
    return ControlApiClient(base_url=base_url)


def _emit(payload) -> None:
    typer.echo(json.dumps(payload, indent=2, default=str))


@app.command("status")
def status(
    base_url: str | None = typer.Option(None, help="Control API base URL.")
) -> None:
    _emit(_platform_client(base_url).status())


@app.command("profiles")
def profiles(
    base_url: str | None = typer.Option(None, help="Control API base URL.")
) -> None:
    _emit(_platform_client(base_url).list_profiles())


@app.command("strategies")
def strategies(
    profile_id: str | None = typer.Option(None, "--profile-id", help="Profile ID."),
    base_url: str | None = typer.Option(None, help="Control API base URL."),
) -> None:
    _emit(_platform_client(base_url).list_strategies(profile_id=profile_id))


@app.command("models")
def models(
    profile_id: str | None = typer.Option(None, "--profile-id", help="Profile ID."),
    base_url: str | None = typer.Option(None, help="Control API base URL."),
) -> None:
    _emit(_platform_client(base_url).list_models(profile_id=profile_id))


@app.command("sleeves")
def sleeves(
    profile_id: str | None = typer.Option(None, "--profile-id", help="Profile ID."),
    base_url: str | None = typer.Option(None, help="Control API base URL."),
) -> None:
    _emit(_platform_client(base_url).list_sleeves(profile_id=profile_id))


@app.command("allocators")
def allocators(
    profile_id: str | None = typer.Option(None, "--profile-id", help="Profile ID."),
    base_url: str | None = typer.Option(None, help="Control API base URL."),
) -> None:
    _emit(_platform_client(base_url).list_allocators(profile_id=profile_id))


@runs_app.command("backtests")
def backtests(
    profile_id: str | None = typer.Option(None, "--profile-id", help="Profile ID."),
    base_url: str | None = typer.Option(None, help="Control API base URL."),
) -> None:
    _emit(
        _platform_client(base_url).list_runs(path="/backtests", profile_id=profile_id)
    )


@runs_app.command("paper")
def paper_runs(
    profile_id: str | None = typer.Option(None, "--profile-id", help="Profile ID."),
    base_url: str | None = typer.Option(None, help="Control API base URL."),
) -> None:
    _emit(
        _platform_client(base_url).list_runs(path="/paper-runs", profile_id=profile_id)
    )


@runs_app.command("live")
def live_runs(
    profile_id: str | None = typer.Option(None, "--profile-id", help="Profile ID."),
    base_url: str | None = typer.Option(None, help="Control API base URL."),
) -> None:
    _emit(
        _platform_client(base_url).list_runs(path="/live-runs", profile_id=profile_id)
    )


@app.command("incidents")
def incidents(
    profile_id: str | None = typer.Option(None, "--profile-id", help="Profile ID."),
    base_url: str | None = typer.Option(None, help="Control API base URL."),
) -> None:
    _emit(_platform_client(base_url).list_incidents(profile_id=profile_id))


@app.command("alerts")
def alerts(
    profile_id: str | None = typer.Option(None, "--profile-id", help="Profile ID."),
    base_url: str | None = typer.Option(None, help="Control API base URL."),
) -> None:
    _emit(_platform_client(base_url).list_alerts(profile_id=profile_id))


@app.command("order")
def order(
    profile_id: str = typer.Option(..., "--profile-id", help="Profile ID."),
    strategy_id: str = typer.Option(..., "--strategy-id", help="Strategy ID."),
    model_version_id: str = typer.Option(
        ..., "--model-version-id", help="Model version ID."
    ),
    sleeve_id: str = typer.Option(..., "--sleeve-id", help="Sleeve ID."),
    instrument_id: str = typer.Option(..., "--instrument-id", help="Instrument ID."),
    inst_id: str = typer.Option(
        ..., "--inst-id", help="Exchange instrument identifier."
    ),
    instrument_kind: str = typer.Option(..., "--instrument-kind", help="spot or swap."),
    side: str = typer.Option(..., "--side", help="buy or sell."),
    size: float = typer.Option(..., "--size", help="Order size."),
    td_mode: str = typer.Option("cash", "--td-mode", help="cash or isolated."),
    order_type: str = typer.Option("market", "--order-type", help="market or limit."),
    price: float | None = typer.Option(None, "--price", help="Limit price."),
    submit: bool = typer.Option(False, "--submit", help="Forward to OKX."),
    base_url: str | None = typer.Option(None, help="Control API base URL."),
) -> None:
    payload = {
        "profile_id": profile_id,
        "strategy_id": strategy_id,
        "model_version_id": model_version_id,
        "sleeve_id": sleeve_id,
        "instrument_id": instrument_id,
        "inst_id": inst_id,
        "kind": instrument_kind,
        "side": side,
        "size": size,
        "td_mode": td_mode,
        "order_type": order_type,
    }
    if price is not None:
        payload["price"] = price
    _emit(_platform_client(base_url).create_order(payload, submit=submit))


@app.command("stop-all")
def stop_all(
    reason: str = typer.Option(..., "--reason", help="Kill switch reason."),
    base_url: str | None = typer.Option(None, help="Control API base URL."),
) -> None:
    _emit(_platform_client(base_url).stop_all(reason))


@app.command("migrate")
def migrate(
    base_url: str | None = typer.Option(None, help="Control API base URL.")
) -> None:
    _emit(_platform_client(base_url).migrate())


@app.command("cutover")
def cutover(
    base_url: str | None = typer.Option(None, help="Control API base URL.")
) -> None:
    _emit(_platform_client(base_url).cutover())
