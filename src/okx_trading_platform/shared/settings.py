from __future__ import annotations

import os
from dataclasses import dataclass

from okx_trading_platform.domain import InstrumentKind, TradingProfile


def parse_profile(value: str | TradingProfile | None) -> TradingProfile:
    if isinstance(value, TradingProfile):
        return value
    if value is None:
        return TradingProfile.DEMO
    return TradingProfile(str(value).lower())


def parse_instrument_kind(value: str | InstrumentKind | None) -> InstrumentKind:
    if isinstance(value, InstrumentKind):
        return value
    if value is None:
        return InstrumentKind.SWAP
    return InstrumentKind(str(value).lower())


@dataclass(frozen=True)
class PlatformSettings:
    control_api_url: str
    control_api_auto_submit: bool
    trading_profile: TradingProfile
    reference_bot_name: str
    reference_inst_id: str
    reference_instrument_kind: InstrumentKind
    reference_trigger_spread: float
    reference_order_size: float


def get_platform_settings() -> PlatformSettings:
    return PlatformSettings(
        control_api_url=os.getenv("CONTROL_API_URL", "http://127.0.0.1:8000"),
        control_api_auto_submit=os.getenv("CONTROL_API_AUTO_SUBMIT", "0") == "1",
        trading_profile=parse_profile(os.getenv("TRADING_PROFILE")),
        reference_bot_name=os.getenv("REFERENCE_BOT_NAME", "reference-breakout"),
        reference_inst_id=os.getenv("REFERENCE_INST_ID", "BTC-USDT-SWAP"),
        reference_instrument_kind=parse_instrument_kind(
            os.getenv("REFERENCE_INSTRUMENT_KIND", "swap")
        ),
        reference_trigger_spread=float(
            os.getenv("REFERENCE_TRIGGER_SPREAD", "0.002")
        ),
        reference_order_size=float(os.getenv("REFERENCE_ORDER_SIZE", "1")),
    )
