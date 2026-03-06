from __future__ import annotations

from dataclasses import dataclass

from okx_trading_platform.adapters.okx import (
    ClientOrderIdCache,
    OkxExchangeGateway,
    OkxRestClient,
    OkxWebSocketRouter,
    get_okx_profile_settings,
)
from okx_trading_platform.domain import (
    BalanceSnapshot,
    BotConfig,
    BotStatus,
    DeploymentRecord,
    InstrumentConfig,
    KillSwitchState,
    OrderIntent,
    OrderLifecycleState,
    OrderState,
    PositionSnapshot,
    ProfileConfig,
    ServiceHeartbeat,
    TradingProfile,
)
from okx_trading_platform.domain.risk import RiskManager

from .repositories import PlatformRepository


class ControlPlaneError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class CancelOrderCommand:
    profile: str
    inst_id: str
    order_id: str | None = None
    client_order_id: str | None = None


class ControlPlaneService:
    def __init__(
        self,
        repository: PlatformRepository,
        *,
        risk_manager: RiskManager,
        execution_gateway: OkxExchangeGateway | None = None,
    ) -> None:
        self.repository = repository
        self.risk_manager = risk_manager
        self.execution_gateway = execution_gateway or OkxExchangeGateway(
            rest_client=OkxRestClient(),
            router=OkxWebSocketRouter(),
            dedupe_cache=ClientOrderIdCache(),
        )

    def bootstrap_profiles(self) -> list[ProfileConfig]:
        created_profiles: list[ProfileConfig] = []
        for profile in TradingProfile:
            if self.repository.get_profile(profile) is not None:
                continue
            settings = get_okx_profile_settings(profile)
            created_profiles.append(
                self.repository.create_profile(
                    ProfileConfig(
                        profile=profile,
                        name=profile.value,
                        rest_base_url=settings.rest_base_url,
                        public_ws_url=settings.public_ws_url,
                        private_ws_url=settings.private_ws_url,
                        is_simulated=settings.simulated_trading,
                        credential_env_prefix=settings.credential_env_prefix,
                        description=f"Default {profile.value} trading profile",
                        is_active=True,
                    )
                )
            )
        return created_profiles

    def list_profiles(self) -> list[ProfileConfig]:
        self.bootstrap_profiles()
        return self.repository.list_profiles()

    def create_profile(self, profile: ProfileConfig) -> ProfileConfig:
        if self.repository.get_profile(profile.profile) is not None:
            raise ControlPlaneError(409, "Profile already exists")
        return self.repository.create_profile(profile)

    def list_instruments(
        self, *, profile: str | None = None, kind: str | None = None
    ) -> list[InstrumentConfig]:
        return self.repository.list_instruments(profile=profile, kind=kind)

    def create_instrument(self, instrument: InstrumentConfig) -> InstrumentConfig:
        self._require_profile(instrument.profile)
        if self.repository.get_instrument(
            profile=instrument.profile, inst_id=instrument.inst_id
        ):
            raise ControlPlaneError(409, "Instrument already exists")
        return self.repository.create_instrument(instrument)

    def list_bots(self) -> list[BotConfig]:
        return self.repository.list_bots()

    def create_bot(self, bot: BotConfig) -> BotConfig:
        self._require_profile(bot.profile)
        if self.repository.get_bot(bot.name):
            raise ControlPlaneError(409, "Bot already exists")
        return self.repository.create_bot(bot)

    def enable_bot(self, bot_name: str) -> BotConfig:
        return self._set_bot_status(bot_name, BotStatus.ENABLED)

    def disable_bot(self, bot_name: str) -> BotConfig:
        return self._set_bot_status(bot_name, BotStatus.DISABLED)

    def deploy_bot(
        self, *, bot_name: str, profile: TradingProfile, metadata: dict
    ) -> DeploymentRecord:
        self._require_profile(profile)
        if self.repository.get_bot(bot_name) is None:
            raise ControlPlaneError(404, "Bot not found")
        return self.repository.create_deployment(
            bot_name=bot_name,
            profile=profile,
            metadata=metadata,
        )

    def create_order(self, intent: OrderIntent, *, submit: bool = False) -> OrderState:
        profile = self._require_profile(intent.profile)
        kill_switch = self.repository.get_kill_switch()
        if kill_switch.activated:
            raise ControlPlaneError(423, kill_switch.reason or "Kill switch active")

        instrument = self.repository.get_instrument(
            profile=profile.profile, inst_id=intent.inst_id
        )
        if instrument is None or not instrument.allow_trading:
            raise ControlPlaneError(400, "Instrument is not allowlisted")

        decision = self.risk_manager.evaluate_order(
            intent,
            balance=self.repository.get_balance(profile=profile.profile),
            positions=self.repository.list_positions(profile=profile.profile),
            open_orders_count=self.repository.count_open_orders(profile.profile),
            mark_price=self._mark_price(intent),
            daily_realized_pnl=float(intent.metadata.get("daily_realized_pnl", 0.0)),
            consecutive_errors=int(intent.metadata.get("consecutive_errors", 0)),
            market_data_fresh=bool(intent.metadata.get("market_data_fresh", True)),
        )
        state = OrderState(
            order_id=intent.order_id,
            client_order_id=intent.client_order_id,
            profile=intent.profile,
            inst_id=intent.inst_id,
            instrument_kind=intent.instrument_kind,
            side=intent.side,
            size=intent.size,
            price=intent.price,
            order_type=intent.order_type,
            td_mode=intent.td_mode,
            status=OrderLifecycleState.APPROVED
            if decision.approved
            else OrderLifecycleState.REJECTED,
            bot_name=intent.bot_name,
            rejection_reason=decision.reason,
            raw_payload={"risk": decision.model_dump(mode="json")},
        )
        if submit and decision.approved:
            try:
                state = self.execution_gateway.submit_order(intent)
            except ValueError as exc:
                state.status = OrderLifecycleState.FAILED
                state.rejection_reason = str(exc)
            except Exception as exc:  # pragma: no cover - external transport failure
                state.status = OrderLifecycleState.FAILED
                state.rejection_reason = str(exc)
        return self.repository.create_order(state, source=intent.source)

    def list_orders(
        self, *, profile: str | None = None, status: str | None = None
    ) -> list[OrderState]:
        return self.repository.list_orders(profile=profile, status=status)

    def cancel_order(self, command: CancelOrderCommand) -> OrderState:
        if not command.order_id and not command.client_order_id:
            raise ControlPlaneError(400, "Missing order identifier")
        model = self.repository.get_order_for_cancel(
            profile=command.profile,
            inst_id=command.inst_id,
            order_id=command.order_id,
            client_order_id=command.client_order_id,
        )
        if model is None:
            raise ControlPlaneError(404, "Order not found")
        return self.repository.cancel_order(model)

    def list_positions(self, *, profile: str | None = None) -> list[PositionSnapshot]:
        return self.repository.list_positions(profile=profile)

    def upsert_position(self, position: PositionSnapshot) -> PositionSnapshot:
        self._require_profile(position.profile)
        return self.repository.upsert_position(position)

    def list_balances(self, *, profile: str | None = None) -> list[BalanceSnapshot]:
        return self.repository.list_balances(profile=profile)

    def upsert_balance(self, balance: BalanceSnapshot) -> BalanceSnapshot:
        self._require_profile(balance.profile)
        return self.repository.upsert_balance(balance)

    def list_services(self, *, profile: str | None = None) -> list[ServiceHeartbeat]:
        return self.repository.list_services(profile=profile)

    def upsert_service(self, heartbeat: ServiceHeartbeat) -> ServiceHeartbeat:
        self._require_profile(heartbeat.profile)
        return self.repository.upsert_heartbeat(heartbeat)

    def get_kill_switch(self) -> KillSwitchState:
        return self.repository.get_kill_switch()

    def update_kill_switch(
        self, *, activated: bool, reason: str | None
    ) -> KillSwitchState:
        if activated:
            self.risk_manager.activate_kill_switch(reason or "manual stop")
        else:
            self.risk_manager.release_kill_switch()
        return self.repository.update_kill_switch(activated=activated, reason=reason)

    def _require_profile(self, profile: str | TradingProfile) -> ProfileConfig:
        self.bootstrap_profiles()
        model = self.repository.get_profile(profile)
        if model is None:
            raise ControlPlaneError(404, "Profile not found")
        return model

    def _set_bot_status(self, bot_name: str, status: BotStatus) -> BotConfig:
        bot = self.repository.update_bot_status(bot_name, status)
        if bot is None:
            raise ControlPlaneError(404, "Bot not found")
        return bot

    @staticmethod
    def _mark_price(intent: OrderIntent) -> float | None:
        if intent.price is not None:
            return intent.price
        mark_price = intent.metadata.get("mark_price")
        if mark_price is None:
            return None
        return float(mark_price)
