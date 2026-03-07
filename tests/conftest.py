import os
import sys

os.environ.setdefault(
    "CONTROL_API_DATABASE_URL", "sqlite:///./test_okx_trading_platform.db"
)
os.environ.setdefault("TRADING_PROFILE", "demo")
os.environ.setdefault("BASELINE_PROFILE_ID", "demo-main")
os.environ.setdefault("PLATFORM_DATA_ROOT", "./test-data/lake")
os.environ.setdefault("DUCKDB_PATH", "./test-data/platform.duckdb")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
