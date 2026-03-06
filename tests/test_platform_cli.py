import json

from typer.testing import CliRunner

from backtest_data_module.zxq import app


class FakeControlApiClient:
    def status(self):
        return {"kill_switch": {"activated": False}}

    def stop_all(self, reason: str):
        return {"activated": True, "reason": reason}


def test_platform_status_cli(monkeypatch):
    monkeypatch.setattr(
        "backtest_data_module.zxq.cli._platform_client",
        lambda base_url=None: FakeControlApiClient(),
    )
    runner = CliRunner()
    result = runner.invoke(app, ["platform", "status"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["kill_switch"]["activated"] is False


def test_platform_stop_all_cli(monkeypatch):
    monkeypatch.setattr(
        "backtest_data_module.zxq.cli._platform_client",
        lambda base_url=None: FakeControlApiClient(),
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["platform", "stop-all", "--reason", "panic"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["reason"] == "panic"
