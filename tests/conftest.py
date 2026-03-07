import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

TEST_ROOT = Path(tempfile.mkdtemp(prefix="okx-platform-tests-"))

os.environ.setdefault(
    "CONTROL_API_DATABASE_URL", "sqlite:///./test_okx_trading_platform.db"
)
os.environ.setdefault("TRADING_PROFILE", "demo")
os.environ.setdefault("BASELINE_PROFILE_ID", "demo-main")
os.environ.setdefault("PLATFORM_DATA_ROOT", str(TEST_ROOT / "lake"))
os.environ.setdefault("DUCKDB_PATH", str(TEST_ROOT / "platform.duckdb"))
os.environ.setdefault("HOT_CACHE_ROOT", str(TEST_ROOT / "hot-cache"))
os.environ.setdefault("CHECKPOINT_ROOT", str(TEST_ROOT / "checkpoints"))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_artifacts():
    yield
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
    shutil.rmtree("test-data", ignore_errors=True)
    for db_path in ("platform_test.db", "test_okx_trading_platform.db"):
        try:
            Path(db_path).unlink(missing_ok=True)
        except PermissionError:
            pass
