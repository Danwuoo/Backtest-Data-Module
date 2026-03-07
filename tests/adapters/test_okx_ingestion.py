from okx_trading_platform.adapters.okx import build_market_data_ingestion_plan
from okx_trading_platform.shared.settings import get_platform_settings


def test_build_market_data_ingestion_plan_uses_env_scope(monkeypatch):
    monkeypatch.setenv("TRADING_PROFILE", "demo")
    monkeypatch.setenv("OKX_UNIVERSE_SOURCE", "env")
    monkeypatch.setenv("OKX_INST_ID_WHITELIST", "BTC-USDT-SWAP,ETH-USDT-SWAP")
    monkeypatch.setenv("OKX_TIER_A_INST_IDS", "BTC-USDT-SWAP")
    monkeypatch.setenv("OKX_TIER_B_INST_IDS", "ETH-USDT-SWAP")

    settings = get_platform_settings()
    plan = build_market_data_ingestion_plan(settings)

    assert plan.whitelist_source == "env"
    assert plan.whitelist == ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]
    assert plan.tier_a == ["BTC-USDT-SWAP"]
    assert plan.tier_b == ["ETH-USDT-SWAP"]
    assert plan.credential_env_vars == [
        "OKX_DEMO_API_KEY",
        "OKX_DEMO_SECRET_KEY",
        "OKX_DEMO_PASSPHRASE",
    ]
    assert any(job.stream == "bars_1s" for job in plan.public_jobs)
    assert any(job.stream == "balances" for job in plan.account_jobs)
