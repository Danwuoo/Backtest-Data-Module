import asyncio

from backtest_data_module.data_ingestion.py.api_client import ApiClient
from backtest_data_module.data_ingestion.py.caching import LRUCache
from backtest_data_module.data_ingestion.py.data_source import APIDataSource
from backtest_data_module.data_ingestion.py.rate_limiter import RateLimiter


async def main():
    # 設定速率限制器：每秒 5 次呼叫
    rate_limiter = RateLimiter(calls=5, period=1)

    # 設定快取：容量 100 筆
    cache = LRUCache(capacity=100)

    # 使用 context manager 建立 API client，透過 proxy 轉發
    async with ApiClient(
        base_url="https://jsonplaceholder.typicode.com",
        proxy_base_url="http://localhost:8000",
    ) as api_client:
        api_source = APIDataSource(
            api_client=api_client,
            rate_limiter=rate_limiter,
            cache=cache,
            endpoint="todos"
        )

        tasks = [api_source.read(params={"id": i}) for i in range(1, 11)]
        results = await asyncio.gather(*tasks)

        for result in results:
            print(result)

if __name__ == "__main__":
    asyncio.run(main())
