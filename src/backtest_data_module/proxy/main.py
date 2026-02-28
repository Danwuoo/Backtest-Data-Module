import os

from backtest_data_module.data_ingestion.proxy import create_proxy_app

# 讀取目標 API URL，預設可修改為實際位址
TARGET = os.environ.get("PROXY_TARGET", "https://api.example.com")
app = create_proxy_app(TARGET)
