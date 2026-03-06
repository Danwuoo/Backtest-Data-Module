from __future__ import annotations

import os
from dataclasses import dataclass

from okx_trading_platform.domain import TradingProfile

OKX_LIVE_REST_URL = "https://www.okx.com"
OKX_DEMO_REST_URL = "https://www.okx.com"
OKX_LIVE_PUBLIC_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
OKX_LIVE_PRIVATE_WS_URL = "wss://ws.okx.com:8443/ws/v5/private"
OKX_DEMO_PUBLIC_WS_URL = "wss://wspap.okx.com:8443/ws/v5/public"
OKX_DEMO_PRIVATE_WS_URL = "wss://wspap.okx.com:8443/ws/v5/private"


@dataclass(frozen=True)
class OkxCredentials:
    api_key: str | None
    secret_key: str | None
    passphrase: str | None


@dataclass(frozen=True)
class OkxProfileSettings:
    profile: TradingProfile
    rest_base_url: str
    public_ws_url: str
    private_ws_url: str
    simulated_trading: bool
    credential_env_prefix: str

    @property
    def credentials(self) -> OkxCredentials:
        prefix = self.credential_env_prefix.upper()
        return OkxCredentials(
            api_key=os.getenv(f"{prefix}_API_KEY"),
            secret_key=os.getenv(f"{prefix}_SECRET_KEY"),
            passphrase=os.getenv(f"{prefix}_PASSPHRASE"),
        )


def get_okx_profile_settings(profile: TradingProfile) -> OkxProfileSettings:
    if profile == TradingProfile.DEMO:
        return OkxProfileSettings(
            profile=profile,
            rest_base_url=os.getenv("OKX_DEMO_REST_URL", OKX_DEMO_REST_URL),
            public_ws_url=os.getenv("OKX_DEMO_PUBLIC_WS_URL", OKX_DEMO_PUBLIC_WS_URL),
            private_ws_url=os.getenv(
                "OKX_DEMO_PRIVATE_WS_URL", OKX_DEMO_PRIVATE_WS_URL
            ),
            simulated_trading=True,
            credential_env_prefix=os.getenv(
                "OKX_DEMO_CREDENTIAL_ENV_PREFIX", "OKX_DEMO"
            ),
        )
    return OkxProfileSettings(
        profile=profile,
        rest_base_url=os.getenv("OKX_LIVE_REST_URL", OKX_LIVE_REST_URL),
        public_ws_url=os.getenv("OKX_LIVE_PUBLIC_WS_URL", OKX_LIVE_PUBLIC_WS_URL),
        private_ws_url=os.getenv("OKX_LIVE_PRIVATE_WS_URL", OKX_LIVE_PRIVATE_WS_URL),
        simulated_trading=False,
        credential_env_prefix=os.getenv("OKX_LIVE_CREDENTIAL_ENV_PREFIX", "OKX_LIVE"),
    )
