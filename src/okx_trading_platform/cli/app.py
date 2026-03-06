from __future__ import annotations

import json

import typer

from .client import ControlApiClient

app = typer.Typer(help="Operate the OKX Trading Platform control plane.")


def _platform_client(base_url: str | None = None) -> ControlApiClient:
    return ControlApiClient(base_url=base_url)


def _emit(payload) -> None:
    typer.echo(json.dumps(payload, indent=2, default=str))


@app.command("status")
def status(
    base_url: str | None = typer.Option(None, help="Control API base URL.")
) -> None:
    _emit(_platform_client(base_url).status())


@app.command("deploy")
def deploy(
    bot_name: str = typer.Argument(..., help="Bot name."),
    profile: str = typer.Option(..., "--profile", help="Target trading profile."),
    base_url: str | None = typer.Option(None, help="Control API base URL."),
) -> None:
    _emit(_platform_client(base_url).deploy(bot_name, profile))


@app.command("enable")
def enable(
    bot_name: str = typer.Argument(..., help="Bot name."),
    base_url: str | None = typer.Option(None, help="Control API base URL."),
) -> None:
    _emit(_platform_client(base_url).enable_bot(bot_name))


@app.command("disable")
def disable(
    bot_name: str = typer.Argument(..., help="Bot name."),
    base_url: str | None = typer.Option(None, help="Control API base URL."),
) -> None:
    _emit(_platform_client(base_url).disable_bot(bot_name))


@app.command("order")
def order(
    profile: str = typer.Option(..., "--profile", help="Trading profile."),
    inst_id: str = typer.Option(..., "--inst-id", help="Instrument identifier."),
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
        "profile": profile,
        "inst_id": inst_id,
        "instrument_kind": instrument_kind,
        "side": side,
        "size": size,
        "td_mode": td_mode,
        "order_type": order_type,
    }
    if price is not None:
        payload["price"] = price
    _emit(_platform_client(base_url).create_order(payload, submit=submit))


@app.command("cancel")
def cancel(
    profile: str = typer.Option(..., "--profile", help="Trading profile."),
    inst_id: str = typer.Option(..., "--inst-id", help="Instrument identifier."),
    order_id: str | None = typer.Option(None, "--order-id", help="Platform order ID."),
    client_order_id: str | None = typer.Option(
        None, "--client-order-id", help="Client order ID."
    ),
    base_url: str | None = typer.Option(None, help="Control API base URL."),
) -> None:
    _emit(
        _platform_client(base_url).cancel_order(
            profile=profile,
            inst_id=inst_id,
            order_id=order_id,
            client_order_id=client_order_id,
        )
    )


@app.command("stop-all")
def stop_all(
    reason: str = typer.Option(..., "--reason", help="Kill switch reason."),
    base_url: str | None = typer.Option(None, help="Control API base URL."),
) -> None:
    _emit(_platform_client(base_url).stop_all(reason))


@app.command("services")
def services(
    profile: str | None = typer.Option(None, "--profile", help="Trading profile."),
    base_url: str | None = typer.Option(None, help="Control API base URL."),
) -> None:
    _emit(_platform_client(base_url).list_services(profile=profile))


@app.command("profiles")
def profiles(
    base_url: str | None = typer.Option(None, help="Control API base URL.")
) -> None:
    _emit(_platform_client(base_url).list_profiles())


@app.command("instruments")
def instruments(
    profile: str | None = typer.Option(None, "--profile", help="Trading profile."),
    kind: str | None = typer.Option(None, "--kind", help="spot or swap."),
    base_url: str | None = typer.Option(None, help="Control API base URL."),
) -> None:
    _emit(_platform_client(base_url).list_instruments(profile=profile, kind=kind))
