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
