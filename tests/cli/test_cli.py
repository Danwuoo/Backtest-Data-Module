import importlib
import json

from typer.testing import CliRunner

from okx_trading_platform.cli import app

cli_module = importlib.import_module("okx_trading_platform.cli.app")


class FakeControlApiClient:
    def status(self):
        return {"kill_switch": {"activated": False}}

    def stop_all(self, reason: str):
        return {"activated": True, "reason": reason}

    def list_profiles(self):
        return [{"profile": "demo"}]


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
    result = runner.invoke(
        app,
        ["stop-all", "--reason", "panic"],
    )
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
    assert payload[0]["profile"] == "demo"
