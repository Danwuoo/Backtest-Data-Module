from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import Session

from okx_trading_platform.domain import (
    BalanceSnapshot,
    BotConfig,
    BotStatus,
    DeploymentRecord,
    DeploymentStatus,
    InstrumentConfig,
    InstrumentKind,
    KillSwitchState,
    OrderLifecycleState,
    OrderSide,
    OrderState,
    OrderType,
    PositionSnapshot,
    ProfileConfig,
    ServiceHeartbeat,
    ServiceStatus,
    TdMode,
    TradingProfile,
)

from .persistence import (
    BalanceSnapshotModel,
    BotModel,
    DeploymentRecordModel,
    InstrumentModel,
    KillSwitchModel,
    OrderRecordModel,
    PositionSnapshotModel,
    ServiceHeartbeatModel,
    TradingProfileModel,
)


class PlatformRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_profile(self, profile: str | TradingProfile) -> ProfileConfig | None:
        model = self.db.get(TradingProfileModel, self._enum(profile))
        if model is None:
            return None
        return self._profile_to_domain(model)

    def list_profiles(self) -> list[ProfileConfig]:
        return [
            self._profile_to_domain(model)
            for model in self.db.query(TradingProfileModel).order_by(
                TradingProfileModel.profile.asc()
            )
        ]

    def create_profile(self, profile: ProfileConfig) -> ProfileConfig:
        model = TradingProfileModel(
            profile=self._enum(profile.profile),
            name=profile.name,
            rest_base_url=profile.rest_base_url,
            public_ws_url=profile.public_ws_url,
            private_ws_url=profile.private_ws_url,
            credential_env_prefix=profile.credential_env_prefix,
            description=profile.description,
            is_simulated=profile.is_simulated,
            is_active=profile.is_active,
            created_at=profile.created_at,
            updated_at=profile.updated_at,
        )
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return self._profile_to_domain(model)

    def get_instrument(
        self, *, profile: str | TradingProfile, inst_id: str
    ) -> InstrumentConfig | None:
        model = (
            self.db.query(InstrumentModel)
            .filter(
                InstrumentModel.profile == self._enum(profile),
                InstrumentModel.inst_id == inst_id,
            )
            .first()
        )
        if model is None:
            return None
        return self._instrument_to_domain(model)

    def list_instruments(
        self, *, profile: str | None = None, kind: str | None = None
    ) -> list[InstrumentConfig]:
        query = self.db.query(InstrumentModel)
        if profile:
            query = query.filter(InstrumentModel.profile == profile)
        if kind:
            query = query.filter(InstrumentModel.kind == kind)
        return [self._instrument_to_domain(model) for model in query.all()]

    def create_instrument(self, instrument: InstrumentConfig) -> InstrumentConfig:
        model = InstrumentModel(
            inst_id=instrument.inst_id,
            kind=self._enum(instrument.kind),
            profile=self._enum(instrument.profile),
            allow_trading=instrument.allow_trading,
            tick_size=instrument.tick_size,
            lot_size=instrument.lot_size,
            metadata_json=instrument.metadata,
        )
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return self._instrument_to_domain(model)

    def get_bot(self, bot_name: str) -> BotConfig | None:
        model = self.db.query(BotModel).filter(BotModel.name == bot_name).first()
        if model is None:
            return None
        return self._bot_to_domain(model)

    def list_bots(self) -> list[BotConfig]:
        return [self._bot_to_domain(model) for model in self.db.query(BotModel).all()]

    def create_bot(self, bot: BotConfig) -> BotConfig:
        model = BotModel(
            name=bot.name,
            profile=self._enum(bot.profile),
            signal_provider=bot.signal_provider,
            status=self._enum(bot.status),
            instrument_ids=bot.instrument_ids,
            config_json=bot.config,
            created_at=bot.created_at,
            updated_at=bot.updated_at,
        )
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return self._bot_to_domain(model)

    def update_bot_status(self, bot_name: str, status: BotStatus) -> BotConfig | None:
        model = self.db.query(BotModel).filter(BotModel.name == bot_name).first()
        if model is None:
            return None
        model.status = self._enum(status)
        model.updated_at = dt.datetime.utcnow()
        self.db.commit()
        self.db.refresh(model)
        return self._bot_to_domain(model)

    def create_deployment(
        self,
        *,
        bot_name: str,
        profile: str | TradingProfile,
        metadata: dict,
    ) -> DeploymentRecord:
        model = DeploymentRecordModel(
            bot_name=bot_name,
            profile=self._enum(profile),
            status=DeploymentStatus.DEPLOYED.value,
            metadata_json=metadata,
        )
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return self._deployment_to_domain(model)

    def list_orders(
        self, *, profile: str | None = None, status: str | None = None
    ) -> list[OrderState]:
        query = self.db.query(OrderRecordModel)
        if profile:
            query = query.filter(OrderRecordModel.profile == profile)
        if status:
            query = query.filter(OrderRecordModel.status == status)
        query = query.order_by(OrderRecordModel.created_at.desc())
        return [self._order_to_domain(model) for model in query.all()]

    def get_order_for_cancel(
        self,
        *,
        profile: str | TradingProfile,
        inst_id: str,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> OrderRecordModel | None:
        query = self.db.query(OrderRecordModel).filter(
            OrderRecordModel.profile == self._enum(profile),
            OrderRecordModel.inst_id == inst_id,
        )
        if order_id:
            query = query.filter(OrderRecordModel.order_id == order_id)
        elif client_order_id:
            query = query.filter(OrderRecordModel.client_order_id == client_order_id)
        return query.first()

    def create_order(self, state: OrderState, *, source: str = "manual") -> OrderState:
        model = OrderRecordModel(
            order_id=state.order_id,
            client_order_id=state.client_order_id,
            profile=self._enum(state.profile),
            inst_id=state.inst_id,
            instrument_kind=self._enum(state.instrument_kind),
            side=self._enum(state.side),
            size=state.size,
            filled_size=state.filled_size,
            avg_price=state.avg_price,
            price=state.price,
            order_type=self._enum(state.order_type),
            td_mode=self._enum(state.td_mode),
            status=self._enum(state.status),
            exchange_order_id=state.exchange_order_id,
            bot_name=state.bot_name,
            source=source,
            rejection_reason=state.rejection_reason,
            raw_payload=state.raw_payload,
            created_at=state.updated_at,
            updated_at=state.updated_at,
        )
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return self._order_to_domain(model)

    def cancel_order(self, model: OrderRecordModel) -> OrderState:
        model.status = OrderLifecycleState.CANCELED.value
        model.updated_at = dt.datetime.utcnow()
        self.db.commit()
        self.db.refresh(model)
        return self._order_to_domain(model)

    def count_open_orders(self, profile: str | TradingProfile) -> int:
        return (
            self.db.query(OrderRecordModel)
            .filter(
                OrderRecordModel.profile == self._enum(profile),
                OrderRecordModel.status.in_(
                    [
                        OrderLifecycleState.APPROVED.value,
                        OrderLifecycleState.SUBMITTED.value,
                        OrderLifecycleState.PARTIALLY_FILLED.value,
                    ]
                ),
            )
            .count()
        )

    def list_positions(self, *, profile: str | None = None) -> list[PositionSnapshot]:
        query = self.db.query(PositionSnapshotModel)
        if profile:
            query = query.filter(PositionSnapshotModel.profile == profile)
        return [self._position_to_domain(model) for model in query.all()]

    def upsert_position(self, position: PositionSnapshot) -> PositionSnapshot:
        model = (
            self.db.query(PositionSnapshotModel)
            .filter(
                PositionSnapshotModel.profile == position.profile,
                PositionSnapshotModel.inst_id == position.inst_id,
            )
            .first()
        )
        if model is None:
            model = PositionSnapshotModel(
                profile=self._enum(position.profile),
                inst_id=position.inst_id,
            )
            self.db.add(model)
        model.instrument_kind = self._enum(position.instrument_kind)
        model.quantity = position.quantity
        model.avg_price = position.avg_price
        model.unrealized_pnl = position.unrealized_pnl
        model.td_mode = self._enum(position.td_mode)
        model.updated_at = position.updated_at
        self.db.commit()
        self.db.refresh(model)
        return self._position_to_domain(model)

    def get_balance(
        self, *, profile: str | TradingProfile, currency: str = "USDT"
    ) -> BalanceSnapshot | None:
        model = (
            self.db.query(BalanceSnapshotModel)
            .filter(
                BalanceSnapshotModel.profile == self._enum(profile),
                BalanceSnapshotModel.currency == currency,
            )
            .first()
        )
        if model is None:
            return None
        return self._balance_to_domain(model)

    def list_balances(self, *, profile: str | None = None) -> list[BalanceSnapshot]:
        query = self.db.query(BalanceSnapshotModel)
        if profile:
            query = query.filter(BalanceSnapshotModel.profile == profile)
        return [self._balance_to_domain(model) for model in query.all()]

    def upsert_balance(self, balance: BalanceSnapshot) -> BalanceSnapshot:
        model = (
            self.db.query(BalanceSnapshotModel)
            .filter(
                BalanceSnapshotModel.profile == balance.profile,
                BalanceSnapshotModel.currency == balance.currency,
            )
            .first()
        )
        if model is None:
            model = BalanceSnapshotModel(
                profile=self._enum(balance.profile),
                currency=balance.currency,
            )
            self.db.add(model)
        model.available = balance.available
        model.cash_balance = balance.cash_balance
        model.equity = balance.equity
        model.updated_at = balance.updated_at
        self.db.commit()
        self.db.refresh(model)
        return self._balance_to_domain(model)

    def list_services(self, *, profile: str | None = None) -> list[ServiceHeartbeat]:
        query = self.db.query(ServiceHeartbeatModel)
        if profile:
            query = query.filter(ServiceHeartbeatModel.profile == profile)
        return [self._heartbeat_to_domain(model) for model in query.all()]

    def upsert_heartbeat(self, heartbeat: ServiceHeartbeat) -> ServiceHeartbeat:
        model = (
            self.db.query(ServiceHeartbeatModel)
            .filter(
                ServiceHeartbeatModel.service_name == heartbeat.service_name,
                ServiceHeartbeatModel.instance_id == heartbeat.instance_id,
            )
            .first()
        )
        if model is None:
            model = ServiceHeartbeatModel(
                service_name=heartbeat.service_name,
                instance_id=heartbeat.instance_id,
                profile=self._enum(heartbeat.profile),
            )
            self.db.add(model)
        model.profile = self._enum(heartbeat.profile)
        model.status = self._enum(heartbeat.status)
        model.metadata_json = heartbeat.metadata
        model.last_seen_at = heartbeat.last_seen_at
        self.db.commit()
        self.db.refresh(model)
        return self._heartbeat_to_domain(model)

    def get_kill_switch(self) -> KillSwitchState:
        model = self.db.get(KillSwitchModel, 1)
        if model is None:
            model = KillSwitchModel(kill_switch_id=1, activated=False)
            self.db.add(model)
            self.db.commit()
            self.db.refresh(model)
        return self._kill_switch_to_domain(model)

    def update_kill_switch(
        self, activated: bool, reason: str | None
    ) -> KillSwitchState:
        model = self.db.get(KillSwitchModel, 1)
        if model is None:
            model = KillSwitchModel(kill_switch_id=1)
            self.db.add(model)
        model.activated = activated
        model.reason = reason
        model.updated_at = dt.datetime.utcnow()
        self.db.commit()
        self.db.refresh(model)
        return self._kill_switch_to_domain(model)

    @staticmethod
    def _enum(value) -> str:
        return value.value if hasattr(value, "value") else value

    def _profile_to_domain(self, model: TradingProfileModel) -> ProfileConfig:
        return ProfileConfig(
            profile=TradingProfile(model.profile),
            name=model.name,
            rest_base_url=model.rest_base_url,
            public_ws_url=model.public_ws_url,
            private_ws_url=model.private_ws_url,
            is_simulated=model.is_simulated,
            credential_env_prefix=model.credential_env_prefix,
            description=model.description,
            is_active=model.is_active,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    def _instrument_to_domain(self, model: InstrumentModel) -> InstrumentConfig:
        return InstrumentConfig(
            instrument_id=model.instrument_id,
            inst_id=model.inst_id,
            kind=InstrumentKind(model.kind),
            profile=TradingProfile(model.profile),
            allow_trading=model.allow_trading,
            tick_size=model.tick_size,
            lot_size=model.lot_size,
            metadata=model.metadata_json,
        )

    def _bot_to_domain(self, model: BotModel) -> BotConfig:
        return BotConfig(
            bot_id=model.bot_id,
            name=model.name,
            profile=TradingProfile(model.profile),
            signal_provider=model.signal_provider,
            instrument_ids=model.instrument_ids,
            status=BotStatus(model.status),
            config=model.config_json,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    def _deployment_to_domain(self, model: DeploymentRecordModel) -> DeploymentRecord:
        return DeploymentRecord(
            deployment_id=model.deployment_id,
            bot_name=model.bot_name,
            profile=TradingProfile(model.profile),
            status=DeploymentStatus(model.status),
            metadata=model.metadata_json,
            created_at=model.created_at,
        )

    def _order_to_domain(self, model: OrderRecordModel) -> OrderState:
        return OrderState(
            order_id=model.order_id,
            client_order_id=model.client_order_id,
            profile=TradingProfile(model.profile),
            inst_id=model.inst_id,
            instrument_kind=InstrumentKind(model.instrument_kind),
            side=OrderSide(model.side),
            size=model.size,
            filled_size=model.filled_size,
            avg_price=model.avg_price,
            price=model.price,
            order_type=OrderType(model.order_type),
            td_mode=TdMode(model.td_mode),
            status=OrderLifecycleState(model.status),
            exchange_order_id=model.exchange_order_id,
            bot_name=model.bot_name,
            rejection_reason=model.rejection_reason,
            raw_payload=model.raw_payload,
            updated_at=model.updated_at,
        )

    def _position_to_domain(self, model: PositionSnapshotModel) -> PositionSnapshot:
        return PositionSnapshot(
            inst_id=model.inst_id,
            profile=TradingProfile(model.profile),
            instrument_kind=InstrumentKind(model.instrument_kind),
            quantity=model.quantity,
            avg_price=model.avg_price,
            unrealized_pnl=model.unrealized_pnl,
            td_mode=TdMode(model.td_mode),
            updated_at=model.updated_at,
        )

    def _balance_to_domain(self, model: BalanceSnapshotModel) -> BalanceSnapshot:
        return BalanceSnapshot(
            profile=TradingProfile(model.profile),
            currency=model.currency,
            available=model.available,
            cash_balance=model.cash_balance,
            equity=model.equity,
            updated_at=model.updated_at,
        )

    def _heartbeat_to_domain(self, model: ServiceHeartbeatModel) -> ServiceHeartbeat:
        return ServiceHeartbeat(
            service_name=model.service_name,
            instance_id=model.instance_id,
            profile=TradingProfile(model.profile),
            status=ServiceStatus(model.status),
            metadata=model.metadata_json,
            last_seen_at=model.last_seen_at,
        )

    def _kill_switch_to_domain(self, model: KillSwitchModel) -> KillSwitchState:
        return KillSwitchState(
            activated=model.activated,
            reason=model.reason,
            updated_at=model.updated_at,
        )
