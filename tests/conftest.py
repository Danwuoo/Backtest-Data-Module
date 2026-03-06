import os
import sys

os.environ.setdefault(
    "CONTROL_API_DATABASE_URL", "sqlite:///./test_okx_trading_platform.db"
)
os.environ.setdefault("TRADING_PROFILE", "demo")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
