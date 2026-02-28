from __future__ import annotations

import importlib
import sys

_BASE = "backtest_data_module.data_storage"

# 先註冊子模組，避免舊路徑匯入失敗
for _submodule in ("storage_backend", "catalog", "migrations"):
    sys.modules[__name__ + f".{_submodule}"] = importlib.import_module(
        f"{_BASE}.{_submodule}"
    )

from backtest_data_module.data_storage import *  # noqa: F401,F403,E402
