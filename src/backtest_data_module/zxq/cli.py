import shutil
from pathlib import Path
from typing import Type

import json
import pandas as pd
import typer
import yaml


from backtest_data_module.backtesting.execution import Execution
from backtest_data_module.backtesting.orchestrator import Orchestrator
from backtest_data_module.backtesting.performance import Performance
from backtest_data_module.backtesting.portfolio import Portfolio
from backtest_data_module.backtesting.strategy import StrategyBase
from backtest_data_module.data_handler import DataHandler
from backtest_data_module.data_processing.cross_validation import walk_forward_split
from backtest_data_module.data_storage import (
    Catalog,
    CatalogEntry,
    HybridStorageManager,
)
from backtest_data_module.reporting.report import ReportGen

SNAPSHOT_DIR = Path("snapshots")
RESTORE_DIR = Path("restored")


def restore_snapshot(snapshot: Path) -> None:
    """é‚„åŽŸæŒ‡å®šçš„ snapshot æª”æ¡ˆã€‚"""
    if not snapshot.exists():
        raise FileNotFoundError(snapshot)
    RESTORE_DIR.mkdir(exist_ok=True)
    shutil.unpack_archive(str(snapshot), str(RESTORE_DIR))


def verify_latest() -> None:
    """å°‹æ‰¾ä¸¦é‚„åŽŸæœ€æ–°çš„ snapshotã€‚"""
    snapshots = sorted(
        SNAPSHOT_DIR.glob("*.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not snapshots:
        typer.echo("æ‰¾ä¸åˆ°ä»»ä½• snapshot")
        return
    latest = snapshots[0]
    typer.echo(f"æ­£åœ¨é‚„åŽŸ {latest.name}...")
    restore_snapshot(latest)
    typer.echo("é‚„åŽŸå®Œæˆ")


app = typer.Typer(help="ZXQuant CLI å·¥å…·")

audit_app = typer.Typer(help="ç¨½æ ¸ç›¸é—œæŒ‡ä»¤")
app.add_typer(audit_app, name="audit")

storage_app = typer.Typer(help="å„²å­˜ç›¸é—œæŒ‡ä»¤")
app.add_typer(storage_app, name="storage")

backup_app = typer.Typer(help="å‚™ä»½èˆ‡é‚„åŽŸæŒ‡ä»¤")
app.add_typer(backup_app, name="backup")

orchestrator_app = typer.Typer(help="Orchestrator æŒ‡ä»¤")
app.add_typer(orchestrator_app, name="orchestrator")


@audit_app.command()
def trace(table: str, db: str = ":memory:") -> None:
    """å¾ž Catalog è®€å–è³‡æ–™è¡¨æ‰€åœ¨å±¤ç´šèˆ‡ä½ç½®ã€‚"""
    catalog = Catalog(db_path=db)
    entry: CatalogEntry | None = catalog.get(table)
    if not entry:
        typer.echo(f"æ‰¾ä¸åˆ°è¡¨æ ¼ {table}")
        raise typer.Exit(code=1)

    msg = (
        f"Table {entry.table_name} is at {entry.tier} (location: {entry.location})\n"
        f"Schema: {entry.schema_hash}"
    )
    typer.echo(msg)


@app.command()
def walk_forward(
    samples: int,
    train_size: int,
    test_size: int,
    step_size: int,
) -> None:
    """Walk-Forward è³‡æ–™åˆ‡åˆ†ã€‚"""
    for train_idx, test_idx in walk_forward_split(
        samples, train_size, test_size, step_size
    ):
        typer.echo(f"{train_idx} {test_idx}")


@storage_app.command()
def migrate(
    table: str = typer.Option(..., "--table", help="è¦ç§»å‹•çš„è¡¨æ ¼åç¨±"),
    to: str = typer.Option(..., "--to", help="ç›®æ¨™å±¤ç´š"),
    db: str = typer.Option(":memory:", "--db", help="Catalog ä½ç½®"),
    dry_run: bool = typer.Option(False, "--dry-run", help="åƒ…é¡¯ç¤ºé æœŸå‹•ä½œ"),
) -> None:
    """å°‡è¡¨æ ¼æ¬ç§»è‡³æŒ‡å®šå±¤ç´šã€‚"""
    manager = HybridStorageManager(catalog=Catalog(db_path=db))
    entry = manager.catalog.get(table)
    if entry is None:
        typer.echo(f"æ‰¾ä¸åˆ°è¡¨æ ¼ {table}")
        raise typer.Exit(code=1)
    if dry_run:
        typer.echo(f"預計將 {table} 從 {entry.tier} 移至 {to}")
    else:
        manager.migrate(table, entry.tier, to)
        typer.echo(f"å·²å°‡ {table} å¾ž {entry.tier} ç§»è‡³ {to}")


@backup_app.command()
def verify(latest: bool = False) -> None:
    """é©—è­‰æˆ–é‚„åŽŸå‚™ä»½ã€‚"""
    if latest:
        verify_latest()
    else:
        typer.echo("è«‹ä½¿ç”¨ --latest åƒæ•¸")


def _get_strategy_cls(strategy_name: str) -> Type[StrategyBase]:
    # This is a simple way to get the strategy class.
    # In a real application, you would have a more robust mechanism
    # for discovering and loading strategies.
    if strategy_name == "SmaCrossover":
        from backtest_data_module.backtesting.strategies.sma_crossover import (
            SmaCrossover,
        )

        return SmaCrossover
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def _run_orchestrator(config_path: Path, use_ray: bool):
    with open(config_path, encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)

    # This is a sample dataframe. In a real application, you would load
    # data from the DataHandler.
    data = pd.DataFrame(
        {
            "asset": ["AAPL"] * 1000,
            "close": [100 + i + (i % 5) * 5 for i in range(1000)],
        }
    )
    data["date"] = pd.to_datetime(pd.date_range(start="2020-01-01", periods=1000))
    data = data.set_index("date")

    storage_manager = HybridStorageManager({})
    data_handler = DataHandler(storage_manager)
    strategy_cls = _get_strategy_cls(config["strategy_cls"])

    orchestrator = Orchestrator(
        data_handler=data_handler,
        strategy_cls=strategy_cls,
        portfolio_cls=Portfolio,
        execution_cls=Execution,
        performance_cls=Performance,
    )

    if use_ray:
        orchestrator.run_ray(config, data)
    else:
        orchestrator.run(config, data)

    output_file = f"{config['run_id']}_results.json"
    orchestrator.to_json(output_file)
    orchestrator.generate_reports()
    typer.echo(f"å›žæ¸¬å®Œæˆï¼Œçµæžœå·²å¯«å…¥ {output_file}")


@orchestrator_app.command("run-wfa")
def run_wfa(
    config_path: Path = typer.Option(
        ..., "--config", help="Walk-Forward è¨­å®šæª”è·¯å¾‘"
    ),
    use_ray: bool = typer.Option(
        True, "--ray/--no-ray", help="æ˜¯å¦ä½¿ç”¨ Ray å¹³è¡ŒåŸ·è¡Œ"
    ),
):
    """åŸ·è¡Œ Walk-Forward åˆ†æžã€‚"""
    _run_orchestrator(config_path, use_ray)


@orchestrator_app.command("run-cpcv")
def run_cpcv(
    config_path: Path = typer.Option(
        ..., "--config", help="CPCV è¨­å®šæª”è·¯å¾‘"
    ),
    use_ray: bool = typer.Option(
        True, "--ray/--no-ray", help="æ˜¯å¦ä½¿ç”¨ Ray å¹³è¡ŒåŸ·è¡Œ"
    ),
):
    """åŸ·è¡Œ Combinatorial Purged Cross-Validationã€‚"""
    _run_orchestrator(config_path, use_ray)


report_app = typer.Typer(help="Report generation commands")
app.add_typer(report_app, name="report")


@report_app.command("generate")
def generate_report(
    run_id: str = typer.Option(..., "--run-id", help="å›žæ¸¬åŸ·è¡Œçš„ Run ID"),
    fmt: str = typer.Option("pdf", "--fmt", help="è¼¸å‡ºæ ¼å¼ (pdf, json)"),
):
    """ç‚ºæŒ‡å®š run ID ç”¢ç”Ÿå ±å‘Šã€‚"""
    results_path = Path(f"{run_id}_results.json")
    if not results_path.exists():
        typer.echo(f"æ‰¾ä¸åˆ° Run ID '{run_id}' çš„çµæžœæª”æ¡ˆ")
        raise typer.Exit(code=1)

    with open(results_path) as f:
        results = json.load(f)

    report_gen = ReportGen(run_id, results)

    if fmt == "pdf":
        output_file = Path(f"report_{run_id}.pdf")
        report_gen.generate_pdf(output_file)
        typer.echo(f"å·²ç”¢ç”Ÿ PDF å ±å‘Šæ–¼ {output_file}")
    elif fmt == "json":
        output_file = Path(f"report_{run_id}.json")
        json_output = report_gen.generate_json()
        with open(output_file, "w") as f:
            f.write(json_output)
        typer.echo(f"å·²ç”¢ç”Ÿ JSON å ±å‘Šæ–¼ {output_file}")
    else:
        typer.echo(f"æœªçŸ¥çš„æ ¼å¼ï¼š{fmt}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
