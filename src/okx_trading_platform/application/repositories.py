from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import Session

from okx_trading_platform.adapters.okx import get_okx_profile_settings
from okx_trading_platform.domain import (
    AccountScope,
    AlertPolicy,
    AlertRecord,
    AlertSeverity,
    AllocatorConfig,
    BacktestRun,
    BalanceSnapshot,
    DatasetRecord,
    ExecutionSnapshot,
    FeatureSet,
    FillRecord,
    FundingEntry,
    IncidentRecord,
    InstrumentConfig,
    InstrumentKind,
    KillSwitchState,
    LedgerEntry,
    LiveRun,
    ModelKind,
    ModelVersion,
    OrderLifecycleState,
    OrderPlan,
    OrderState,
    PaperRun,
    PnLSnapshot,
    PositionSnapshot,
    ProfileConfig,
    RiskPolicyConfig,
    RiskSnapshot,
    ServiceHeartbeat,
    SleeveConfig,
    StrategyConfig,
    StrategyStatus,
    TradingEnvironment,
)

from .persistence import (
    AlertPolicyV2Model,
    AlertV2Model,
    AllocatorV2Model,
    BacktestRunV2Model,
    BalanceSnapshotModel,
    BalanceV2Model,
    BotModel,
    DatasetV2Model,
    DeploymentRecordModel,
    ExecutionSnapshotV2Model,
    FeatureV2Model,
    FillRecordModel,
    FillV2Model,
    FundingEntryV2Model,
    IncidentV2Model,
    InstrumentModel,
    InstrumentV2Model,
    KillSwitchModel,
    LedgerEntryV2Model,
    LiveRunV2Model,
    ModelVersionV2Model,
    OrderPlanV2Model,
    OrderRecordModel,
    OrderV2Model,
    PaperRunV2Model,
    PnLSnapshotV2Model,
    PositionSnapshotModel,
    PositionV2Model,
    ProfileV2Model,
    RiskPolicyV2Model,
    RiskSnapshotV2Model,
    ServiceHeartbeatModel,
    SleeveV2Model,
    StrategyV2Model,
    TradingProfileModel,
)


class PlatformRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def ensure_default_profile_stack(
        self, environment: TradingEnvironment
    ) -> ProfileConfig:
        profile_id = f"{environment.value}-main"
        risk_policy_id = f"{profile_id}-default-risk"
        allocator_id = f"{profile_id}-default-allocator"
        sleeve_id = f"{profile_id}-default-sleeve"
        alert_policy_id = f"{profile_id}-default-alert-policy"
        settings = get_okx_profile_settings(environment)

        if self.get_risk_policy(risk_policy_id) is None:
            self.create_risk_policy(
                RiskPolicyConfig(
                    risk_policy_id=risk_policy_id,
                    profile_id=profile_id,
                    name=f"{environment.value} default risk",
                    config={
                        "min_notional": 5.0,
                        "max_position_notional": 25.0,
                        "max_open_orders": 4,
                        "max_leverage": 3.0,
                    },
                )
            )
        if self.get_allocator(allocator_id) is None:
            self.create_allocator(
                AllocatorConfig(
                    allocator_id=allocator_id,
                    profile_id=profile_id,
                    name=f"{environment.value} default allocator",
                    config={"mode": "single_sleeve"},
                )
            )
        if self.get_sleeve(sleeve_id) is None:
            self.create_sleeve(
                SleeveConfig(
                    sleeve_id=sleeve_id,
                    profile_id=profile_id,
                    name=f"{environment.value} primary sleeve",
                    capital_allocation=1.0,
                    risk_budget=1.0,
                    max_leverage=1.0,
                )
            )
        if self.get_alert_policy(alert_policy_id) is None:
            self.create_alert_policy(
                AlertPolicy(
                    alert_policy_id=alert_policy_id,
                    profile_id=profile_id,
                    name=f"{environment.value} default alerts",
                    severity_threshold=AlertSeverity.WARNING,
                )
            )
        profile = self.get_profile(profile_id)
        if profile is None:
            profile = self.create_profile(
                ProfileConfig(
                    profile_id=profile_id,
                    name=profile_id,
                    environment=environment,
                    account_scope=AccountScope.MAIN,
                    account_label="main",
                    rest_base_url=settings.rest_base_url,
                    public_ws_url=settings.public_ws_url,
                    private_ws_url=settings.private_ws_url,
                    credential_env_prefix=settings.credential_env_prefix,
                    risk_policy_id=risk_policy_id,
                    allocator_id=allocator_id,
                    default_sleeve_id=sleeve_id,
                    description=f"Default {environment.value} profile",
                )
            )
        return profile

    def migrate_legacy_data(self) -> None:
        if self.db.query(ProfileV2Model).count():
            return

        for env in TradingEnvironment:
            self.ensure_default_profile_stack(env)

        for legacy in self.db.query(TradingProfileModel).all():
            profile_id = f"{legacy.profile}-main"
            profile = self.get_profile(profile_id)
            if profile is None:
                continue
            profile.description = legacy.description or profile.description
            profile.updated_at = legacy.updated_at
            self._upsert_profile(profile)

        for legacy in self.db.query(InstrumentModel).all():
            profile_id = f"{legacy.profile}-main"
            self.create_instrument(
                InstrumentConfig(
                    instrument_id=legacy.instrument_id
                    or f"{profile_id}:{legacy.inst_id}",
                    profile_id=profile_id,
                    inst_id=legacy.inst_id,
                    kind=InstrumentKind(legacy.kind),
                    tick_size=legacy.tick_size,
                    lot_size=legacy.lot_size,
                    allow_trading=legacy.allow_trading,
                    metadata=legacy.metadata_json,
                )
            )

        for legacy in self.db.query(BotModel).all():
            profile_id = f"{legacy.profile}-main"
            strategy_id = legacy.name
            model_version_id = f"{strategy_id}-baseline-v1"
            if self.get_strategy(strategy_id) is None:
                self.create_strategy(
                    StrategyConfig(
                        strategy_id=strategy_id,
                        profile_id=profile_id,
                        name=legacy.name,
                        baseline_provider=legacy.signal_provider,
                        status=StrategyStatus(legacy.status),
                        allowed_instrument_ids=list(legacy.instrument_ids),
                        primary_model_version_id=model_version_id,
                        config=legacy.config_json,
                    )
                )
            if self.get_model_version(model_version_id) is None:
                self.create_model_version(
                    ModelVersion(
                        model_version_id=model_version_id,
                        strategy_id=strategy_id,
                        profile_id=profile_id,
                        name="baseline-v1",
                        kind=ModelKind.RULE_BASELINE,
                        metadata={"migrated_from_bot": legacy.name},
                    )
                )

        for legacy in self.db.query(DeploymentRecordModel).all():
            profile_id = f"{legacy.profile}-main"
            strategy_id = legacy.bot_name
            self.create_live_run(
                LiveRun(
                    run_id=legacy.deployment_id,
                    profile_id=profile_id,
                    strategy_id=strategy_id,
                    model_version_id=f"{strategy_id}-baseline-v1",
                    sleeve_id=f"{profile_id}-default-sleeve",
                    status=legacy.status,
                    metrics={},
                    config=legacy.metadata_json,
                )
            )

        for legacy in self.db.query(OrderRecordModel).all():
            profile_id = f"{legacy.profile}-main"
            self.create_order(
                OrderState(
                    order_id=legacy.order_id,
                    client_order_id=legacy.client_order_id,
                    profile_id=profile_id,
                    strategy_id=legacy.bot_name,
                    model_version_id=(
                        f"{legacy.bot_name}-baseline-v1" if legacy.bot_name else None
                    ),
                    sleeve_id=f"{profile_id}-default-sleeve",
                    instrument_id=f"{profile_id}:{legacy.inst_id}",
                    inst_id=legacy.inst_id,
                    kind=InstrumentKind(legacy.instrument_kind),
                    side=legacy.side,
                    size=legacy.size,
                    filled_size=legacy.filled_size,
                    avg_price=legacy.avg_price,
                    price=legacy.price,
                    order_type=legacy.order_type,
                    td_mode=legacy.td_mode,
                    status=legacy.status,
                    exchange_order_id=legacy.exchange_order_id,
                    source=legacy.source,
                    rejection_reason=legacy.rejection_reason,
                    raw_payload=legacy.raw_payload,
                    created_at=legacy.created_at,
                    updated_at=legacy.updated_at,
                )
            )

        for legacy in self.db.query(FillRecordModel).all():
            profile_id = f"{legacy.profile}-main"
            self.create_fill(
                FillRecord(
                    fill_id=legacy.fill_id,
                    order_id=legacy.order_id,
                    profile_id=profile_id,
                    instrument_id=f"{profile_id}:{legacy.inst_id}",
                    inst_id=legacy.inst_id,
                    fill_price=legacy.fill_price,
                    fill_size=legacy.fill_size,
                    fee=legacy.fee,
                    raw_payload=legacy.raw_payload,
                    created_at=legacy.created_at,
                )
            )

        for legacy in self.db.query(PositionSnapshotModel).all():
            profile_id = f"{legacy.profile}-main"
            self.upsert_position(
                PositionSnapshot(
                    position_snapshot_id=f"{profile_id}:{legacy.inst_id}",
                    profile_id=profile_id,
                    sleeve_id=f"{profile_id}-default-sleeve",
                    instrument_id=f"{profile_id}:{legacy.inst_id}",
                    inst_id=legacy.inst_id,
                    kind=InstrumentKind(legacy.instrument_kind),
                    quantity=legacy.quantity,
                    avg_price=legacy.avg_price,
                    unrealized_pnl=legacy.unrealized_pnl,
                    td_mode=legacy.td_mode,
                    updated_at=legacy.updated_at,
                )
            )

        for legacy in self.db.query(BalanceSnapshotModel).all():
            profile_id = f"{legacy.profile}-main"
            self.upsert_balance(
                BalanceSnapshot(
                    balance_snapshot_id=f"{profile_id}:{legacy.currency}",
                    profile_id=profile_id,
                    currency=legacy.currency,
                    available=legacy.available,
                    cash_balance=legacy.cash_balance,
                    equity=legacy.equity,
                    updated_at=legacy.updated_at,
                )
            )

        kill = self.db.get(KillSwitchModel, 1)
        if kill is not None:
            self.update_kill_switch(kill.activated, kill.reason)

    def _save(self, model) -> None:
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)

    # Profiles and config resources
    def list_profiles(self) -> list[ProfileConfig]:
        query = self.db.query(ProfileV2Model).order_by(ProfileV2Model.created_at.asc())
        return [
            ProfileConfig.model_validate(model.payload_json) for model in query.all()
        ]

    def get_profile(self, profile_id: str) -> ProfileConfig | None:
        model = self.db.get(ProfileV2Model, profile_id)
        return (
            None if model is None else ProfileConfig.model_validate(model.payload_json)
        )

    def create_profile(self, profile: ProfileConfig) -> ProfileConfig:
        return self._upsert_profile(profile)

    def _upsert_profile(self, profile: ProfileConfig) -> ProfileConfig:
        model = self.db.get(ProfileV2Model, profile.profile_id) or ProfileV2Model(
            profile_id=profile.profile_id,
            name=profile.name,
            environment=profile.environment,
            account_scope=profile.account_scope,
        )
        model.name = profile.name
        model.environment = profile.environment
        model.account_scope = profile.account_scope
        model.payload_json = profile.model_dump(mode="json")
        self._save(model)
        return ProfileConfig.model_validate(model.payload_json)

    def list_risk_policies(
        self, profile_id: str | None = None
    ) -> list[RiskPolicyConfig]:
        query = self.db.query(RiskPolicyV2Model)
        if profile_id:
            query = query.filter(RiskPolicyV2Model.profile_id == profile_id)
        query = query.order_by(RiskPolicyV2Model.created_at.asc())
        return [
            RiskPolicyConfig.model_validate(model.payload_json) for model in query.all()
        ]

    def get_risk_policy(self, risk_policy_id: str) -> RiskPolicyConfig | None:
        model = self.db.get(RiskPolicyV2Model, risk_policy_id)
        return (
            None
            if model is None
            else RiskPolicyConfig.model_validate(model.payload_json)
        )

    def create_risk_policy(self, policy: RiskPolicyConfig) -> RiskPolicyConfig:
        model = self.db.get(
            RiskPolicyV2Model, policy.risk_policy_id
        ) or RiskPolicyV2Model(
            risk_policy_id=policy.risk_policy_id,
            profile_id=policy.profile_id,
            name=policy.name,
        )
        model.profile_id = policy.profile_id
        model.name = policy.name
        model.payload_json = policy.model_dump(mode="json")
        self._save(model)
        return RiskPolicyConfig.model_validate(model.payload_json)

    def list_allocators(self, profile_id: str | None = None) -> list[AllocatorConfig]:
        query = self.db.query(AllocatorV2Model)
        if profile_id:
            query = query.filter(AllocatorV2Model.profile_id == profile_id)
        query = query.order_by(AllocatorV2Model.created_at.asc())
        return [
            AllocatorConfig.model_validate(model.payload_json) for model in query.all()
        ]

    def get_allocator(self, allocator_id: str) -> AllocatorConfig | None:
        model = self.db.get(AllocatorV2Model, allocator_id)
        return (
            None
            if model is None
            else AllocatorConfig.model_validate(model.payload_json)
        )

    def create_allocator(self, allocator: AllocatorConfig) -> AllocatorConfig:
        model = self.db.get(
            AllocatorV2Model, allocator.allocator_id
        ) or AllocatorV2Model(
            allocator_id=allocator.allocator_id,
            profile_id=allocator.profile_id,
            name=allocator.name,
            policy_type=allocator.policy_type,
        )
        model.profile_id = allocator.profile_id
        model.name = allocator.name
        model.policy_type = allocator.policy_type
        model.payload_json = allocator.model_dump(mode="json")
        self._save(model)
        return AllocatorConfig.model_validate(model.payload_json)

    def list_sleeves(self, profile_id: str | None = None) -> list[SleeveConfig]:
        query = self.db.query(SleeveV2Model)
        if profile_id:
            query = query.filter(SleeveV2Model.profile_id == profile_id)
        query = query.order_by(SleeveV2Model.created_at.asc())
        return [
            SleeveConfig.model_validate(model.payload_json) for model in query.all()
        ]

    def get_sleeve(self, sleeve_id: str) -> SleeveConfig | None:
        model = self.db.get(SleeveV2Model, sleeve_id)
        return (
            None if model is None else SleeveConfig.model_validate(model.payload_json)
        )

    def create_sleeve(self, sleeve: SleeveConfig) -> SleeveConfig:
        model = self.db.get(SleeveV2Model, sleeve.sleeve_id) or SleeveV2Model(
            sleeve_id=sleeve.sleeve_id,
            profile_id=sleeve.profile_id,
            name=sleeve.name,
            sleeve_type=sleeve.sleeve_type,
            is_active=sleeve.is_active,
        )
        model.profile_id = sleeve.profile_id
        model.name = sleeve.name
        model.sleeve_type = sleeve.sleeve_type
        model.is_active = sleeve.is_active
        model.payload_json = sleeve.model_dump(mode="json")
        self._save(model)
        return SleeveConfig.model_validate(model.payload_json)

    def list_instruments(
        self, profile_id: str | None = None, kind: str | None = None
    ) -> list[InstrumentConfig]:
        query = self.db.query(InstrumentV2Model)
        if profile_id:
            query = query.filter(InstrumentV2Model.profile_id == profile_id)
        if kind:
            query = query.filter(InstrumentV2Model.kind == kind)
        query = query.order_by(InstrumentV2Model.created_at.asc())
        return [
            InstrumentConfig.model_validate(model.payload_json) for model in query.all()
        ]

    def get_instrument(self, profile_id: str, inst_id: str) -> InstrumentConfig | None:
        model = (
            self.db.query(InstrumentV2Model)
            .filter(
                InstrumentV2Model.profile_id == profile_id,
                InstrumentV2Model.inst_id == inst_id,
            )
            .first()
        )
        return (
            None
            if model is None
            else InstrumentConfig.model_validate(model.payload_json)
        )

    def create_instrument(self, instrument: InstrumentConfig) -> InstrumentConfig:
        existing = self.get_instrument(instrument.profile_id, instrument.inst_id)
        if existing is not None:
            return existing
        model = InstrumentV2Model(
            instrument_id=instrument.instrument_id,
            profile_id=instrument.profile_id,
            inst_id=instrument.inst_id,
            inst_id_code=instrument.inst_id_code,
            kind=instrument.kind,
            allow_trading=instrument.allow_trading,
            payload_json=instrument.model_dump(mode="json"),
        )
        self._save(model)
        return InstrumentConfig.model_validate(model.payload_json)

    def list_strategies(self, profile_id: str | None = None) -> list[StrategyConfig]:
        query = self.db.query(StrategyV2Model)
        if profile_id:
            query = query.filter(StrategyV2Model.profile_id == profile_id)
        query = query.order_by(StrategyV2Model.created_at.asc())
        return [
            StrategyConfig.model_validate(model.payload_json) for model in query.all()
        ]

    def get_strategy(self, strategy_id: str) -> StrategyConfig | None:
        model = self.db.get(StrategyV2Model, strategy_id)
        return (
            None if model is None else StrategyConfig.model_validate(model.payload_json)
        )

    def create_strategy(self, strategy: StrategyConfig) -> StrategyConfig:
        model = self.db.get(StrategyV2Model, strategy.strategy_id) or StrategyV2Model(
            strategy_id=strategy.strategy_id,
            profile_id=strategy.profile_id,
            name=strategy.name,
            status=strategy.status,
        )
        model.profile_id = strategy.profile_id
        model.name = strategy.name
        model.status = strategy.status
        model.payload_json = strategy.model_dump(mode="json")
        self._save(model)
        return StrategyConfig.model_validate(model.payload_json)

    def list_model_versions(self, profile_id: str | None = None) -> list[ModelVersion]:
        query = self.db.query(ModelVersionV2Model)
        if profile_id:
            query = query.filter(ModelVersionV2Model.profile_id == profile_id)
        query = query.order_by(ModelVersionV2Model.created_at.asc())
        return [
            ModelVersion.model_validate(model.payload_json) for model in query.all()
        ]

    def get_model_version(self, model_version_id: str) -> ModelVersion | None:
        model = self.db.get(ModelVersionV2Model, model_version_id)
        return (
            None if model is None else ModelVersion.model_validate(model.payload_json)
        )

    def create_model_version(self, version: ModelVersion) -> ModelVersion:
        model = self.db.get(
            ModelVersionV2Model, version.model_version_id
        ) or ModelVersionV2Model(
            model_version_id=version.model_version_id,
            profile_id=version.profile_id,
            strategy_id=version.strategy_id,
            name=version.name,
            kind=version.kind,
            is_primary=version.is_primary,
        )
        model.profile_id = version.profile_id
        model.strategy_id = version.strategy_id
        model.name = version.name
        model.kind = version.kind
        model.is_primary = version.is_primary
        model.payload_json = version.model_dump(mode="json")
        self._save(model)
        return ModelVersion.model_validate(model.payload_json)

    def list_datasets(self, profile_id: str | None = None) -> list[DatasetRecord]:
        query = self.db.query(DatasetV2Model)
        if profile_id:
            query = query.filter(DatasetV2Model.profile_id == profile_id)
        query = query.order_by(DatasetV2Model.created_at.asc())
        return [
            DatasetRecord.model_validate(model.payload_json) for model in query.all()
        ]

    def create_dataset(self, dataset: DatasetRecord) -> DatasetRecord:
        model = self.db.get(DatasetV2Model, dataset.dataset_id) or DatasetV2Model(
            dataset_id=dataset.dataset_id,
            profile_id=dataset.profile_id,
            name=dataset.name,
            layer=dataset.layer,
            path=dataset.path,
        )
        model.profile_id = dataset.profile_id
        model.name = dataset.name
        model.layer = dataset.layer
        model.path = dataset.path
        model.payload_json = dataset.model_dump(mode="json")
        self._save(model)
        return DatasetRecord.model_validate(model.payload_json)

    def list_features(self, profile_id: str | None = None) -> list[FeatureSet]:
        query = self.db.query(FeatureV2Model)
        if profile_id:
            query = query.filter(FeatureV2Model.profile_id == profile_id)
        query = query.order_by(FeatureV2Model.created_at.asc())
        return [FeatureSet.model_validate(model.payload_json) for model in query.all()]

    def create_feature(self, feature: FeatureSet) -> FeatureSet:
        model = self.db.get(FeatureV2Model, feature.feature_id) or FeatureV2Model(
            feature_id=feature.feature_id,
            profile_id=feature.profile_id,
            name=feature.name,
            schema_version=feature.schema_version,
            path=feature.path,
        )
        model.profile_id = feature.profile_id
        model.name = feature.name
        model.schema_version = feature.schema_version
        model.path = feature.path
        model.payload_json = feature.model_dump(mode="json")
        self._save(model)
        return FeatureSet.model_validate(model.payload_json)

    def list_backtests(self, profile_id: str | None = None) -> list[BacktestRun]:
        query = self.db.query(BacktestRunV2Model)
        if profile_id:
            query = query.filter(BacktestRunV2Model.profile_id == profile_id)
        query = query.order_by(BacktestRunV2Model.created_at.desc())
        return [BacktestRun.model_validate(model.payload_json) for model in query.all()]

    def create_backtest(self, run: BacktestRun) -> BacktestRun:
        model = self.db.get(BacktestRunV2Model, run.run_id) or BacktestRunV2Model(
            run_id=run.run_id,
            profile_id=run.profile_id,
            strategy_id=run.strategy_id,
            status=run.status,
        )
        model.profile_id = run.profile_id
        model.strategy_id = run.strategy_id
        model.status = run.status
        model.payload_json = run.model_dump(mode="json")
        self._save(model)
        return BacktestRun.model_validate(model.payload_json)

    def list_paper_runs(self, profile_id: str | None = None) -> list[PaperRun]:
        query = self.db.query(PaperRunV2Model)
        if profile_id:
            query = query.filter(PaperRunV2Model.profile_id == profile_id)
        query = query.order_by(PaperRunV2Model.created_at.desc())
        return [PaperRun.model_validate(model.payload_json) for model in query.all()]

    def create_paper_run(self, run: PaperRun) -> PaperRun:
        model = self.db.get(PaperRunV2Model, run.run_id) or PaperRunV2Model(
            run_id=run.run_id,
            profile_id=run.profile_id,
            strategy_id=run.strategy_id,
            status=run.status,
        )
        model.profile_id = run.profile_id
        model.strategy_id = run.strategy_id
        model.status = run.status
        model.payload_json = run.model_dump(mode="json")
        self._save(model)
        return PaperRun.model_validate(model.payload_json)

    def list_live_runs(self, profile_id: str | None = None) -> list[LiveRun]:
        query = self.db.query(LiveRunV2Model)
        if profile_id:
            query = query.filter(LiveRunV2Model.profile_id == profile_id)
        query = query.order_by(LiveRunV2Model.created_at.desc())
        return [LiveRun.model_validate(model.payload_json) for model in query.all()]

    def create_live_run(self, run: LiveRun) -> LiveRun:
        model = self.db.get(LiveRunV2Model, run.run_id) or LiveRunV2Model(
            run_id=run.run_id,
            profile_id=run.profile_id,
            strategy_id=run.strategy_id,
            status=run.status,
        )
        model.profile_id = run.profile_id
        model.strategy_id = run.strategy_id
        model.status = run.status
        model.payload_json = run.model_dump(mode="json")
        self._save(model)
        return LiveRun.model_validate(model.payload_json)

    def create_order_plan(self, plan: OrderPlan) -> OrderPlan:
        model = self.db.get(OrderPlanV2Model, plan.order_plan_id) or OrderPlanV2Model(
            order_plan_id=plan.order_plan_id,
            profile_id=plan.profile_id,
            strategy_id=plan.strategy_id,
            sleeve_id=plan.sleeve_id,
            inst_id=plan.inst_id,
            status="planned",
        )
        model.payload_json = plan.model_dump(mode="json")
        self._save(model)
        return OrderPlan.model_validate(model.payload_json)

    def list_orders(
        self, profile_id: str | None = None, status: str | None = None
    ) -> list[OrderState]:
        query = self.db.query(OrderV2Model)
        if profile_id:
            query = query.filter(OrderV2Model.profile_id == profile_id)
        if status:
            query = query.filter(OrderV2Model.status == status)
        query = query.order_by(OrderV2Model.created_at.desc())
        return [OrderState.model_validate(model.payload_json) for model in query.all()]

    def get_order(self, order_id: str) -> OrderState | None:
        model = self.db.get(OrderV2Model, order_id)
        return None if model is None else OrderState.model_validate(model.payload_json)

    def create_order(self, state: OrderState) -> OrderState:
        model = self.db.get(OrderV2Model, state.order_id) or OrderV2Model(
            order_id=state.order_id,
            client_order_id=state.client_order_id,
            profile_id=state.profile_id,
            strategy_id=state.strategy_id,
            sleeve_id=state.sleeve_id,
            inst_id=state.inst_id,
            status=state.status,
        )
        model.client_order_id = state.client_order_id
        model.profile_id = state.profile_id
        model.strategy_id = state.strategy_id
        model.sleeve_id = state.sleeve_id
        model.inst_id = state.inst_id
        model.status = state.status
        model.payload_json = state.model_dump(mode="json")
        self._save(model)
        return OrderState.model_validate(model.payload_json)

    def cancel_order(self, order_id: str) -> OrderState | None:
        state = self.get_order(order_id)
        if state is None:
            return None
        state.status = OrderLifecycleState.CANCELED
        state.updated_at = dt.datetime.utcnow()
        return self.create_order(state)

    def count_open_orders(self, profile_id: str) -> int:
        return (
            self.db.query(OrderV2Model)
            .filter(
                OrderV2Model.profile_id == profile_id,
                OrderV2Model.status.in_(
                    [
                        OrderLifecycleState.APPROVED.value,
                        OrderLifecycleState.SUBMITTED.value,
                        OrderLifecycleState.PARTIALLY_FILLED.value,
                    ]
                ),
            )
            .count()
        )

    def list_fills(self, profile_id: str | None = None) -> list[FillRecord]:
        query = self.db.query(FillV2Model)
        if profile_id:
            query = query.filter(FillV2Model.profile_id == profile_id)
        query = query.order_by(FillV2Model.created_at.desc())
        return [FillRecord.model_validate(model.payload_json) for model in query.all()]

    def create_fill(self, fill: FillRecord) -> FillRecord:
        model = self.db.get(FillV2Model, fill.fill_id) or FillV2Model(
            fill_id=fill.fill_id,
            order_id=fill.order_id,
            profile_id=fill.profile_id,
            inst_id=fill.inst_id,
        )
        model.order_id = fill.order_id
        model.profile_id = fill.profile_id
        model.inst_id = fill.inst_id
        model.payload_json = fill.model_dump(mode="json")
        self._save(model)
        return FillRecord.model_validate(model.payload_json)

    def list_ledger(self, profile_id: str | None = None) -> list[LedgerEntry]:
        query = self.db.query(LedgerEntryV2Model)
        if profile_id:
            query = query.filter(LedgerEntryV2Model.profile_id == profile_id)
        query = query.order_by(LedgerEntryV2Model.created_at.desc())
        return [LedgerEntry.model_validate(model.payload_json) for model in query.all()]

    def create_ledger_entry(self, entry: LedgerEntry) -> LedgerEntry:
        model = self.db.get(
            LedgerEntryV2Model, entry.ledger_entry_id
        ) or LedgerEntryV2Model(
            ledger_entry_id=entry.ledger_entry_id,
            profile_id=entry.profile_id,
            order_id=entry.order_id,
            entry_type=entry.entry_type,
        )
        model.profile_id = entry.profile_id
        model.order_id = entry.order_id
        model.entry_type = entry.entry_type
        model.payload_json = entry.model_dump(mode="json")
        self._save(model)
        return LedgerEntry.model_validate(model.payload_json)

    def list_funding(self, profile_id: str | None = None) -> list[FundingEntry]:
        query = self.db.query(FundingEntryV2Model)
        if profile_id:
            query = query.filter(FundingEntryV2Model.profile_id == profile_id)
        query = query.order_by(FundingEntryV2Model.created_at.desc())
        return [
            FundingEntry.model_validate(model.payload_json) for model in query.all()
        ]

    def create_funding_entry(self, entry: FundingEntry) -> FundingEntry:
        model = self.db.get(
            FundingEntryV2Model, entry.funding_entry_id
        ) or FundingEntryV2Model(
            funding_entry_id=entry.funding_entry_id,
            profile_id=entry.profile_id,
            inst_id=entry.inst_id,
        )
        model.profile_id = entry.profile_id
        model.inst_id = entry.inst_id
        model.payload_json = entry.model_dump(mode="json")
        self._save(model)
        return FundingEntry.model_validate(model.payload_json)

    def list_pnl(self, profile_id: str | None = None) -> list[PnLSnapshot]:
        query = self.db.query(PnLSnapshotV2Model)
        if profile_id:
            query = query.filter(PnLSnapshotV2Model.profile_id == profile_id)
        query = query.order_by(PnLSnapshotV2Model.created_at.desc())
        return [PnLSnapshot.model_validate(model.payload_json) for model in query.all()]

    def create_pnl_snapshot(self, snapshot: PnLSnapshot) -> PnLSnapshot:
        model = self.db.get(
            PnLSnapshotV2Model, snapshot.pnl_snapshot_id
        ) or PnLSnapshotV2Model(
            pnl_snapshot_id=snapshot.pnl_snapshot_id,
            profile_id=snapshot.profile_id,
        )
        model.profile_id = snapshot.profile_id
        model.payload_json = snapshot.model_dump(mode="json")
        self._save(model)
        return PnLSnapshot.model_validate(model.payload_json)

    def list_risk_snapshots(self, profile_id: str | None = None) -> list[RiskSnapshot]:
        query = self.db.query(RiskSnapshotV2Model)
        if profile_id:
            query = query.filter(RiskSnapshotV2Model.profile_id == profile_id)
        query = query.order_by(RiskSnapshotV2Model.created_at.desc())
        return [
            RiskSnapshot.model_validate(model.payload_json) for model in query.all()
        ]

    def create_risk_snapshot(self, snapshot: RiskSnapshot) -> RiskSnapshot:
        model = self.db.get(
            RiskSnapshotV2Model, snapshot.risk_snapshot_id
        ) or RiskSnapshotV2Model(
            risk_snapshot_id=snapshot.risk_snapshot_id,
            profile_id=snapshot.profile_id,
            order_id=snapshot.order_id,
            stage=snapshot.stage,
        )
        model.profile_id = snapshot.profile_id
        model.order_id = snapshot.order_id
        model.stage = snapshot.stage
        model.payload_json = snapshot.model_dump(mode="json")
        self._save(model)
        return RiskSnapshot.model_validate(model.payload_json)

    def list_execution_snapshots(
        self, profile_id: str | None = None
    ) -> list[ExecutionSnapshot]:
        query = self.db.query(ExecutionSnapshotV2Model)
        if profile_id:
            query = query.filter(ExecutionSnapshotV2Model.profile_id == profile_id)
        query = query.order_by(ExecutionSnapshotV2Model.created_at.desc())
        return [
            ExecutionSnapshot.model_validate(model.payload_json)
            for model in query.all()
        ]

    def create_execution_snapshot(
        self, snapshot: ExecutionSnapshot
    ) -> ExecutionSnapshot:
        model = self.db.get(
            ExecutionSnapshotV2Model, snapshot.execution_snapshot_id
        ) or ExecutionSnapshotV2Model(
            execution_snapshot_id=snapshot.execution_snapshot_id,
            profile_id=snapshot.profile_id,
            order_id=snapshot.order_id,
            status=snapshot.status,
        )
        model.profile_id = snapshot.profile_id
        model.order_id = snapshot.order_id
        model.status = snapshot.status
        model.payload_json = snapshot.model_dump(mode="json")
        self._save(model)
        return ExecutionSnapshot.model_validate(model.payload_json)

    def list_positions(self, profile_id: str | None = None) -> list[PositionSnapshot]:
        query = self.db.query(PositionV2Model)
        if profile_id:
            query = query.filter(PositionV2Model.profile_id == profile_id)
        query = query.order_by(PositionV2Model.updated_at.desc())
        return [
            PositionSnapshot.model_validate(model.payload_json) for model in query.all()
        ]

    def upsert_position(self, position: PositionSnapshot) -> PositionSnapshot:
        model = self.db.get(
            PositionV2Model, position.position_snapshot_id
        ) or PositionV2Model(
            position_snapshot_id=position.position_snapshot_id,
            profile_id=position.profile_id,
            sleeve_id=position.sleeve_id,
            inst_id=position.inst_id,
            updated_at=position.updated_at,
        )
        model.profile_id = position.profile_id
        model.sleeve_id = position.sleeve_id
        model.inst_id = position.inst_id
        model.updated_at = position.updated_at
        model.payload_json = position.model_dump(mode="json")
        self._save(model)
        return PositionSnapshot.model_validate(model.payload_json)

    def get_balance(
        self, profile_id: str, currency: str = "USDT"
    ) -> BalanceSnapshot | None:
        model = (
            self.db.query(BalanceV2Model)
            .filter(
                BalanceV2Model.profile_id == profile_id,
                BalanceV2Model.currency == currency,
            )
            .first()
        )
        return (
            None
            if model is None
            else BalanceSnapshot.model_validate(model.payload_json)
        )

    def list_balances(self, profile_id: str | None = None) -> list[BalanceSnapshot]:
        query = self.db.query(BalanceV2Model)
        if profile_id:
            query = query.filter(BalanceV2Model.profile_id == profile_id)
        query = query.order_by(BalanceV2Model.updated_at.desc())
        return [
            BalanceSnapshot.model_validate(model.payload_json) for model in query.all()
        ]

    def upsert_balance(self, balance: BalanceSnapshot) -> BalanceSnapshot:
        model = self.db.get(
            BalanceV2Model, balance.balance_snapshot_id
        ) or BalanceV2Model(
            balance_snapshot_id=balance.balance_snapshot_id,
            profile_id=balance.profile_id,
            currency=balance.currency,
            updated_at=balance.updated_at,
        )
        model.profile_id = balance.profile_id
        model.currency = balance.currency
        model.updated_at = balance.updated_at
        model.payload_json = balance.model_dump(mode="json")
        self._save(model)
        return BalanceSnapshot.model_validate(model.payload_json)

    def list_services(self, profile_id: str | None = None) -> list[ServiceHeartbeat]:
        query = self.db.query(ServiceHeartbeatModel)
        if profile_id:
            query = query.filter(ServiceHeartbeatModel.profile == profile_id)
        query = query.order_by(ServiceHeartbeatModel.last_seen_at.desc())
        return [
            ServiceHeartbeat(
                service_name=model.service_name,
                instance_id=model.instance_id,
                profile_id=model.profile,
                status=model.status,
                metadata=model.metadata_json,
                last_seen_at=model.last_seen_at,
            )
            for model in query.all()
        ]

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
                profile=heartbeat.profile_id,
            )
        model.profile = heartbeat.profile_id
        model.status = heartbeat.status
        model.metadata_json = heartbeat.metadata
        model.last_seen_at = heartbeat.last_seen_at
        self._save(model)
        return ServiceHeartbeat(
            service_name=model.service_name,
            instance_id=model.instance_id,
            profile_id=model.profile,
            status=model.status,
            metadata=model.metadata_json,
            last_seen_at=model.last_seen_at,
        )

    def get_kill_switch(self) -> KillSwitchState:
        model = self.db.get(KillSwitchModel, 1)
        if model is None:
            model = KillSwitchModel(kill_switch_id=1, activated=False)
            self._save(model)
        return KillSwitchState(
            activated=model.activated,
            reason=model.reason,
            updated_at=model.updated_at,
        )

    def update_kill_switch(
        self, activated: bool, reason: str | None
    ) -> KillSwitchState:
        model = self.db.get(KillSwitchModel, 1)
        if model is None:
            model = KillSwitchModel(kill_switch_id=1)
        model.activated = activated
        model.reason = reason
        model.updated_at = dt.datetime.utcnow()
        self._save(model)
        return KillSwitchState(
            activated=model.activated,
            reason=model.reason,
            updated_at=model.updated_at,
        )

    def list_incidents(self, profile_id: str | None = None) -> list[IncidentRecord]:
        query = self.db.query(IncidentV2Model)
        if profile_id:
            query = query.filter(IncidentV2Model.profile_id == profile_id)
        query = query.order_by(IncidentV2Model.created_at.desc())
        return [
            IncidentRecord.model_validate(model.payload_json) for model in query.all()
        ]

    def create_incident(self, incident: IncidentRecord) -> IncidentRecord:
        model = self.db.get(IncidentV2Model, incident.incident_id) or IncidentV2Model(
            incident_id=incident.incident_id,
            profile_id=incident.profile_id,
            severity=incident.severity,
            status=incident.status,
            title=incident.title,
        )
        model.profile_id = incident.profile_id
        model.severity = incident.severity
        model.status = incident.status
        model.title = incident.title
        model.payload_json = incident.model_dump(mode="json")
        self._save(model)
        return IncidentRecord.model_validate(model.payload_json)

    def list_alert_policies(self, profile_id: str | None = None) -> list[AlertPolicy]:
        query = self.db.query(AlertPolicyV2Model)
        if profile_id:
            query = query.filter(AlertPolicyV2Model.profile_id == profile_id)
        query = query.order_by(AlertPolicyV2Model.created_at.asc())
        return [AlertPolicy.model_validate(model.payload_json) for model in query.all()]

    def get_alert_policy(self, alert_policy_id: str) -> AlertPolicy | None:
        model = self.db.get(AlertPolicyV2Model, alert_policy_id)
        return None if model is None else AlertPolicy.model_validate(model.payload_json)

    def create_alert_policy(self, policy: AlertPolicy) -> AlertPolicy:
        model = self.db.get(
            AlertPolicyV2Model, policy.alert_policy_id
        ) or AlertPolicyV2Model(
            alert_policy_id=policy.alert_policy_id,
            profile_id=policy.profile_id,
            name=policy.name,
            severity_threshold=policy.severity_threshold,
            channel=policy.channel,
            is_active=policy.is_active,
        )
        model.profile_id = policy.profile_id
        model.name = policy.name
        model.severity_threshold = policy.severity_threshold
        model.channel = policy.channel
        model.is_active = policy.is_active
        model.payload_json = policy.model_dump(mode="json")
        self._save(model)
        return AlertPolicy.model_validate(model.payload_json)

    def list_alerts(self, profile_id: str | None = None) -> list[AlertRecord]:
        query = self.db.query(AlertV2Model)
        if profile_id:
            query = query.filter(AlertV2Model.profile_id == profile_id)
        query = query.order_by(AlertV2Model.created_at.desc())
        return [AlertRecord.model_validate(model.payload_json) for model in query.all()]

    def create_alert(self, alert: AlertRecord) -> AlertRecord:
        model = self.db.get(AlertV2Model, alert.alert_id) or AlertV2Model(
            alert_id=alert.alert_id,
            profile_id=alert.profile_id,
            incident_id=alert.incident_id,
            severity=alert.severity,
            status=alert.status,
            title=alert.title,
        )
        model.profile_id = alert.profile_id
        model.incident_id = alert.incident_id
        model.severity = alert.severity
        model.status = alert.status
        model.title = alert.title
        model.payload_json = alert.model_dump(mode="json")
        self._save(model)
        return AlertRecord.model_validate(model.payload_json)
