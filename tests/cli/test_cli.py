import importlib
import json

from typer.testing import CliRunner

from okx_trading_platform.cli import app

cli_module = importlib.import_module("okx_trading_platform.cli.app")


class FakeControlApiClient:
    def status(self):
        return {"kill_switch": {"activated": False}, "alerts": []}

    def stop_all(self, reason: str):
        return {"activated": True, "reason": reason}

    def list_profiles(self):
        return [{"profile_id": "demo-main"}]

    def list_strategies(self, profile_id=None):
        return [{"strategy_id": "reference-breakout", "profile_id": profile_id}]

    def list_alerts(self, profile_id=None):
        return [{"title": "incident", "profile_id": profile_id}]

    def list_datasets(self, profile_id=None):
        return [{"dataset_id": "silver.trades", "profile_id": profile_id}]

    def list_features(self, profile_id=None):
        return [{"feature_id": "feat-1", "profile_id": profile_id}]

    def list_dataset_versions(self, profile_id=None, dataset_id=None):
        return [{"dataset_version_id": "silver.trades:v1", "dataset_id": dataset_id}]

    def list_run_artifacts(self, profile_id=None, run_id=None):
        return [{"artifact_id": f"{run_id}:summary", "profile_id": profile_id}]


class FakeLakeWriter:
    def query(self, sql):
        return [{"sql": sql}]

    def doctor(self):
        return {"layer_stats": {"silver": {"files": 1, "bytes": 10}}}

    def compact(self, glob_pattern=None, min_file_size_bytes=0):
        return {
            "glob_pattern": glob_pattern,
            "min_file_size_bytes": min_file_size_bytes,
            "compacted_groups": 1,
        }

    def read_audit_events(self, stream=None):
        return ["event-1", "event-2"]


class FakeSession:
    def close(self):
        return None


class FakeProjector:
    projected = 0

    def __init__(self, repository):
        self.repository = repository

    def project_event(self, event):
        del event
        type(self).projected += 1


class FakeRepository:
    def __init__(self, db):
        self.db = db


def test_platform_status_cli(monkeypatch):
    monkeypatch.setattr(
        cli_module,
        "_platform_client",
        lambda base_url=None: FakeControlApiClient(),
    )
    runner = CliRunner()
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["kill_switch"]["activated"] is False


def test_platform_stop_all_cli(monkeypatch):
    monkeypatch.setattr(
        cli_module,
        "_platform_client",
        lambda base_url=None: FakeControlApiClient(),
    )
    runner = CliRunner()
    result = runner.invoke(app, ["stop-all", "--reason", "panic"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["reason"] == "panic"


def test_platform_profiles_cli(monkeypatch):
    monkeypatch.setattr(
        cli_module,
        "_platform_client",
        lambda base_url=None: FakeControlApiClient(),
    )
    runner = CliRunner()
    result = runner.invoke(app, ["profiles"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload[0]["profile_id"] == "demo-main"


def test_platform_strategies_cli(monkeypatch):
    monkeypatch.setattr(
        cli_module,
        "_platform_client",
        lambda base_url=None: FakeControlApiClient(),
    )
    runner = CliRunner()
    result = runner.invoke(app, ["strategies", "--profile-id", "demo-main"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload[0]["strategy_id"] == "reference-breakout"


def test_platform_lake_datasets_cli(monkeypatch):
    monkeypatch.setattr(
        cli_module,
        "_platform_client",
        lambda base_url=None: FakeControlApiClient(),
    )
    runner = CliRunner()
    result = runner.invoke(app, ["lake", "datasets", "--profile-id", "demo-main", "--include-versions"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["datasets"][0]["dataset_id"] == "silver.trades"
    assert payload["dataset_versions"][0]["dataset_version_id"] == "silver.trades:v1"


def test_platform_lake_sql_cli(monkeypatch):
    monkeypatch.setattr(cli_module, "_lake_writer", lambda: FakeLakeWriter())
    runner = CliRunner()
    result = runner.invoke(app, ["lake", "sql", "--sql", "select 1"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload[0]["sql"] == "select 1"


def test_platform_lake_doctor_cli(monkeypatch):
    monkeypatch.setattr(cli_module, "_lake_writer", lambda: FakeLakeWriter())
    runner = CliRunner()
    result = runner.invoke(app, ["lake", "doctor"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["layer_stats"]["silver"]["files"] == 1


def test_platform_lake_rebuild_read_models_cli(monkeypatch):
    FakeProjector.projected = 0
    monkeypatch.setattr(cli_module, "_lake_writer", lambda: FakeLakeWriter())
    monkeypatch.setattr(cli_module, "SessionLocal", lambda: FakeSession())
    monkeypatch.setattr(cli_module, "ReadModelProjector", FakeProjector)
    monkeypatch.setattr(cli_module, "PlatformRepository", FakeRepository)
    runner = CliRunner()
    result = runner.invoke(app, ["lake", "rebuild-read-models", "--stream", "control"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["stream"] == "control"
    assert payload["projected_events"] == 2
    assert FakeProjector.projected == 2
