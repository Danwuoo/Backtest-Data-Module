from __future__ import annotations

import json
from pathlib import Path

import typer

from okx_trading_platform.application import ReadModelProjector
from okx_trading_platform.application.repositories import PlatformRepository
from okx_trading_platform.shared import DataLakeWriter, SessionLocal, get_platform_settings

from .client import ControlApiClient

app = typer.Typer(help="Operate the OKX Trading Platform control plane.")
runs_app = typer.Typer(help="Inspect backtest, paper, and live runs.")
lake_app = typer.Typer(help="Query and maintain the local data lake.")
app.add_typer(runs_app, name="runs")
app.add_typer(lake_app, name="lake")


def _platform_client(base_url: str | None = None) -> ControlApiClient:
    return ControlApiClient(base_url=base_url)


def _emit(payload) -> None:
    typer.echo(json.dumps(payload, indent=2, default=str))


def _lake_writer() -> DataLakeWriter:
    settings = get_platform_settings()
    return DataLakeWriter(
        settings.platform_data_root,
        settings.duckdb_path,
        hot_cache_root=settings.hot_cache_root,
        object_store_backend=settings.object_store_backend,
        object_store_bucket=settings.object_store_bucket,
        object_store_endpoint=settings.object_store_endpoint,
        object_store_region=settings.object_store_region,
        object_store_access_key_id=settings.object_store_access_key_id,
        object_store_secret_access_key=settings.object_store_secret_access_key,
        object_store_prefix=settings.object_store_prefix,
        checkpoint_root=settings.checkpoint_root,
    )


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


@lake_app.command("datasets")
def lake_datasets(
    profile_id: str | None = typer.Option(None, "--profile-id", help="Profile ID."),
    include_versions: bool = typer.Option(
        False, "--include-versions", help="Include dataset versions."
    ),
    base_url: str | None = typer.Option(None, help="Control API base URL."),
) -> None:
    client = _platform_client(base_url)
    payload = {"datasets": client.list_datasets(profile_id=profile_id)}
    if include_versions:
        payload["dataset_versions"] = client.list_dataset_versions(profile_id=profile_id)
    _emit(payload)


@lake_app.command("features")
def lake_features(
    profile_id: str | None = typer.Option(None, "--profile-id", help="Profile ID."),
    base_url: str | None = typer.Option(None, help="Control API base URL."),
) -> None:
    _emit(_platform_client(base_url).list_features(profile_id=profile_id))


@lake_app.command("runs")
def lake_runs(
    profile_id: str | None = typer.Option(None, "--profile-id", help="Profile ID."),
    base_url: str | None = typer.Option(None, help="Control API base URL."),
) -> None:
    client = _platform_client(base_url)
    runs = (
        client.list_runs(path="/backtests", profile_id=profile_id)
        + client.list_runs(path="/paper-runs", profile_id=profile_id)
        + client.list_runs(path="/live-runs", profile_id=profile_id)
    )
    runs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    _emit(runs)


@lake_app.command("artifacts")
def lake_artifacts(
    run_id: str = typer.Option(..., "--run-id", help="Run ID."),
    profile_id: str | None = typer.Option(None, "--profile-id", help="Profile ID."),
    base_url: str | None = typer.Option(None, help="Control API base URL."),
) -> None:
    _emit(_platform_client(base_url).list_run_artifacts(profile_id=profile_id, run_id=run_id))


@lake_app.command("sql")
def lake_sql(
    sql: str | None = typer.Option(None, "--sql", help="SQL text to execute."),
    file: Path | None = typer.Option(None, "--file", help="Path to a SQL file."),
) -> None:
    if bool(sql) == bool(file):
        raise typer.BadParameter("Specify exactly one of --sql or --file.")
    query = sql or file.read_text(encoding="utf-8")
    _emit(_lake_writer().query(query))


@lake_app.command("doctor")
def lake_doctor() -> None:
    _emit(_lake_writer().doctor())


@lake_app.command("compact")
def lake_compact(
    glob_pattern: str | None = typer.Option(
        None, "--glob-pattern", help="Optional parquet glob relative to lake root."
    ),
    min_file_size_mb: int = typer.Option(
        8, "--min-file-size-mb", help="Compact files smaller than this size."
    ),
) -> None:
    _emit(
        _lake_writer().compact(
            glob_pattern=glob_pattern,
            min_file_size_bytes=min_file_size_mb * 1024 * 1024,
        )
    )


@lake_app.command("rebuild-read-models")
def lake_rebuild_read_models(
    stream: str | None = typer.Option(None, "--stream", help="Audit stream filter."),
) -> None:
    writer = _lake_writer()
    events = writer.read_audit_events(stream=stream)
    db = SessionLocal()
    try:
        projector = ReadModelProjector(PlatformRepository(db))
        for event in events:
            projector.project_event(event)
    finally:
        db.close()
    _emit({"stream": stream, "projected_events": len(events)})


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
