from __future__ import annotations

import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


async def get_api_key(api_key: str | None = Security(api_key_header)) -> str | None:
    expected = os.getenv("CONTROL_API_KEY") or os.getenv("STRATEGY_MANAGER_API_KEY")
    if not expected:
        return api_key
    if api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key
