from __future__ import annotations

from dataclasses import dataclass

from okx_trading_platform.adapters.okx import (
    ClientOrderIdCache,
    OkxExchangeGateway,
    OkxRestClient,
    OkxWebSocketRouter,
)
from okx_trading_platform.domain import (
    AlertPolicy,
    AlertRecord,
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
    KillSwitchState,
    LedgerEntry,
    LiveRun,
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
    TradingEnvironment,
)
from okx_trading_platform.domain.risk import RiskManager
from okx_trading_platform.shared.notify import build_default_notifier

from .repositories import PlatformRepository


class ControlPlaneError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass
class CancelOrderCommand:
    order_id: str


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
        self.notifier = build_default_notifier()

    def bootstrap_profiles(self) -> list[ProfileConfig]:
        self.repository.migrate_legacy_data()
        return [
            self.repository.ensure_default_profile_stack(TradingEnvironment.DEMO),
            self.repository.ensure_default_profile_stack(TradingEnvironment.LIVE),
        ]

    def list_profiles(self) -> list[ProfileConfig]:
        self.bootstrap_profiles()
        return self.repository.list_profiles()

    def create_profile(self, profile: ProfileConfig) -> ProfileConfig:
        if self.repository.get_profile(profile.profile_id) is not None:
            raise ControlPlaneError(409, "Profile already exists")
        return self.repository.create_profile(profile)

    def list_risk_policies(
        self, *, profile_id: str | None = None
    ) -> list[RiskPolicyConfig]:
        return self.repository.list_risk_policies(profile_id=profile_id)

    def create_risk_policy(self, policy: RiskPolicyConfig) -> RiskPolicyConfig:
        self._require_profile(policy.profile_id)
        return self.repository.create_risk_policy(policy)

    def list_allocators(
        self, *, profile_id: str | None = None
    ) -> list[AllocatorConfig]:
        return self.repository.list_allocators(profile_id=profile_id)

    def create_allocator(self, allocator: AllocatorConfig) -> AllocatorConfig:
        self._require_profile(allocator.profile_id)
        return self.repository.create_allocator(allocator)

    def list_sleeves(self, *, profile_id: str | None = None) -> list[SleeveConfig]:
        return self.repository.list_sleeves(profile_id=profile_id)

    def create_sleeve(self, sleeve: SleeveConfig) -> SleeveConfig:
        self._require_profile(sleeve.profile_id)
        return self.repository.create_sleeve(sleeve)

    def list_instruments(
        self,
        *,
        profile_id: str | None = None,
        kind: str | None = None,
    ) -> list[InstrumentConfig]:
        return self.repository.list_instruments(profile_id=profile_id, kind=kind)

    def create_instrument(self, instrument: InstrumentConfig) -> InstrumentConfig:
        self._require_profile(instrument.profile_id)
        return self.repository.create_instrument(instrument)

    def list_strategies(self, *, profile_id: str | None = None) -> list[StrategyConfig]:
        return self.repository.list_strategies(profile_id=profile_id)

    def create_strategy(self, strategy: StrategyConfig) -> StrategyConfig:
        self._require_profile(strategy.profile_id)
        return self.repository.create_strategy(strategy)

    def list_model_versions(
        self, *, profile_id: str | None = None
    ) -> list[ModelVersion]:
        return self.repository.list_model_versions(profile_id=profile_id)

    def create_model_version(self, version: ModelVersion) -> ModelVersion:
        self._require_profile(version.profile_id)
        return self.repository.create_model_version(version)

    def list_datasets(self, *, profile_id: str | None = None) -> list[DatasetRecord]:
        return self.repository.list_datasets(profile_id=profile_id)

    def create_dataset(self, dataset: DatasetRecord) -> DatasetRecord:
        self._require_profile(dataset.profile_id)
        return self.repository.create_dataset(dataset)

    def list_features(self, *, profile_id: str | None = None) -> list[FeatureSet]:
        return self.repository.list_features(profile_id=profile_id)

    def create_feature(self, feature: FeatureSet) -> FeatureSet:
        self._require_profile(feature.profile_id)
        return self.repository.create_feature(feature)

    def list_backtests(self, *, profile_id: str | None = None) -> list[BacktestRun]:
        return self.repository.list_backtests(profile_id=profile_id)

    def create_backtest(self, run: BacktestRun) -> BacktestRun:
        self._require_profile(run.profile_id)
        return self.repository.create_backtest(run)

    def list_paper_runs(self, *, profile_id: str | None = None) -> list[PaperRun]:
        return self.repository.list_paper_runs(profile_id=profile_id)

    def create_paper_run(self, run: PaperRun) -> PaperRun:
        self._require_profile(run.profile_id)
        return self.repository.create_paper_run(run)

    def list_live_runs(self, *, profile_id: str | None = None) -> list[LiveRun]:
        return self.repository.list_live_runs(profile_id=profile_id)

    def create_live_run(self, run: LiveRun) -> LiveRun:
        self._require_profile(run.profile_id)
        return self.repository.create_live_run(run)

    def list_orders(
        self, *, profile_id: str | None = None, status: str | None = None
    ) -> list[OrderState]:
        return self.repository.list_orders(profile_id=profile_id, status=status)

    def create_order(self, plan: OrderPlan, *, submit: bool = False) -> OrderState:
        profile = self._require_profile(plan.profile_id)
        instrument = self.repository.get_instrument(plan.profile_id, plan.inst_id)
        if instrument is None or not instrument.allow_trading:
            raise ControlPlaneError(400, "Instrument is not allowlisted")

        strategy = self.repository.get_strategy(plan.strategy_id)
        sleeve = self.repository.get_sleeve(plan.sleeve_id)
        policy = self.repository.get_risk_policy(profile.risk_policy_id)
        if sleeve is None:
            raise ControlPlaneError(400, "Sleeve not found")
        mark_price = plan.price or float(plan.metadata.get("mark_price", 0.0) or 0.0)
        mark_price = mark_price or None
        decision = self.risk_manager.evaluate_order(
            plan,
            instrument=instrument,
            sleeve=sleeve,
            strategy=strategy,
            policy=policy,
            balance=self.repository.get_balance(profile.profile_id),
            positions=self.repository.list_positions(profile_id=profile.profile_id),
            open_orders_count=self.repository.count_open_orders(profile.profile_id),
            mark_price=mark_price,
            daily_realized_pnl=float(plan.metadata.get("daily_realized_pnl", 0.0)),
            hourly_realized_pnl=float(plan.metadata.get("hourly_realized_pnl", 0.0)),
            funding_cost=float(plan.metadata.get("funding_cost", 0.0)),
            consecutive_errors=int(plan.metadata.get("consecutive_errors", 0)),
            consecutive_losses=int(plan.metadata.get("consecutive_losses", 0)),
            liquidation_distance_ratio=plan.metadata.get("liquidation_distance_ratio"),
            volatility_regime=plan.metadata.get("volatility_regime"),
            spread_bps=plan.metadata.get("spread_bps"),
            impact_bps=plan.metadata.get("impact_bps"),
            observed_latency_ms=plan.metadata.get("latency_ms"),
            market_data_fresh=bool(plan.metadata.get("market_data_fresh", True)),
        )
        self.repository.create_order_plan(plan)
        state = OrderState(
            order_plan_id=plan.order_plan_id,
            client_order_id=plan.metadata.get("client_order_id", plan.order_plan_id),
            profile_id=plan.profile_id,
            strategy_id=plan.strategy_id,
            model_version_id=plan.model_version_id,
            sleeve_id=plan.sleeve_id,
            instrument_id=plan.instrument_id,
            inst_id=plan.inst_id,
            kind=plan.kind,
            side=plan.side,
            size=plan.size,
            price=plan.price,
            order_type=plan.order_type,
            td_mode=plan.td_mode,
            status=(
                OrderLifecycleState.APPROVED
                if decision.approved
                else OrderLifecycleState.REJECTED
            ),
            source=plan.source,
            rejection_reason=decision.reason,
            raw_payload={"risk": decision.model_dump(mode="json")},
        )
        self.repository.create_risk_snapshot(
            self.risk_manager.build_snapshot(
                profile_id=plan.profile_id,
                order_id=state.order_id,
                strategy_id=plan.strategy_id,
                sleeve_id=plan.sleeve_id,
                decision=decision,
            )
        )
        if submit and decision.approved:
            try:
                state = self.execution_gateway.submit_order(
                    plan,
                    environment=profile.environment,
                )
            except ValueError as exc:
                state.status = OrderLifecycleState.FAILED
                state.rejection_reason = str(exc)
        self.repository.create_order(state)
        self.repository.create_execution_snapshot(
            ExecutionSnapshot(
                profile_id=plan.profile_id,
                order_id=state.order_id,
                status=state.status,
                signal_ts=plan.created_at,
                risk_ts=self.repository.list_risk_snapshots(profile_id=plan.profile_id)[
                    0
                ].created_at,
                send_ts=state.updated_at if submit and decision.approved else None,
                metadata={"source": plan.source},
            )
        )
        return state

    def cancel_order(self, command: CancelOrderCommand) -> OrderState:
        model = self.repository.cancel_order(command.order_id)
        if model is None:
            raise ControlPlaneError(404, "Order not found")
        return model

    def list_fills(self, *, profile_id: str | None = None) -> list[FillRecord]:
        return self.repository.list_fills(profile_id=profile_id)

    def create_fill(self, fill: FillRecord) -> FillRecord:
        order = self.repository.get_order(fill.order_id)
        if order is None:
            raise ControlPlaneError(404, "Order not found")
        created = self.repository.create_fill(fill)
        direction = -1 if order.side == "buy" else 1
        notional = fill.fill_price * fill.fill_size * direction
        self.repository.create_ledger_entry(
            LedgerEntry(
                profile_id=fill.profile_id,
                order_id=fill.order_id,
                fill_id=fill.fill_id,
                amount=notional,
                entry_type="trade_cashflow",
                description="Trade cashflow",
            )
        )
        self.repository.create_ledger_entry(
            LedgerEntry(
                profile_id=fill.profile_id,
                order_id=fill.order_id,
                fill_id=fill.fill_id,
                amount=-abs(fill.fee),
                entry_type="fee",
                description="Exchange fee",
            )
        )
        if fill.funding_cost:
            self.repository.create_funding_entry(
                FundingEntry(
                    profile_id=fill.profile_id,
                    instrument_id=fill.instrument_id,
                    inst_id=fill.inst_id,
                    amount=fill.funding_cost,
                    metadata={"fill_id": fill.fill_id},
                )
            )
        self.repository.create_pnl_snapshot(
            PnLSnapshot(
                profile_id=fill.profile_id,
                strategy_id=order.strategy_id,
                sleeve_id=order.sleeve_id,
                realized_pnl=-abs(fill.fee),
                net_pnl=-abs(fill.fee),
                metadata={"fill_id": fill.fill_id},
            )
        )
        self.repository.create_risk_snapshot(
            self.risk_manager.execution_snapshot(
                profile_id=fill.profile_id,
                order_id=fill.order_id,
                latency_ms=fill.raw_payload.get("latency_ms"),
            )
        )
        self.repository.create_execution_snapshot(
            ExecutionSnapshot(
                profile_id=fill.profile_id,
                order_id=fill.order_id,
                status="filled",
                fill_ts=fill.created_at,
                slippage_bps=self._slippage_bps(order.price, fill.fill_price),
                metadata={"fill_id": fill.fill_id},
            )
        )
        return created

    def list_ledger(self, *, profile_id: str | None = None) -> list[LedgerEntry]:
        return self.repository.list_ledger(profile_id=profile_id)

    def list_funding(self, *, profile_id: str | None = None) -> list[FundingEntry]:
        return self.repository.list_funding(profile_id=profile_id)

    def list_pnl(self, *, profile_id: str | None = None) -> list[PnLSnapshot]:
        return self.repository.list_pnl(profile_id=profile_id)

    def list_risk_snapshots(
        self, *, profile_id: str | None = None
    ) -> list[RiskSnapshot]:
        return self.repository.list_risk_snapshots(profile_id=profile_id)

    def list_execution_snapshots(
        self, *, profile_id: str | None = None
    ) -> list[ExecutionSnapshot]:
        return self.repository.list_execution_snapshots(profile_id=profile_id)

    def list_positions(
        self, *, profile_id: str | None = None
    ) -> list[PositionSnapshot]:
        return self.repository.list_positions(profile_id=profile_id)

    def upsert_position(self, position: PositionSnapshot) -> PositionSnapshot:
        self._require_profile(position.profile_id)
        return self.repository.upsert_position(position)

    def list_balances(self, *, profile_id: str | None = None) -> list[BalanceSnapshot]:
        return self.repository.list_balances(profile_id=profile_id)

    def upsert_balance(self, balance: BalanceSnapshot) -> BalanceSnapshot:
        self._require_profile(balance.profile_id)
        return self.repository.upsert_balance(balance)

    def list_services(self, *, profile_id: str | None = None) -> list[ServiceHeartbeat]:
        return self.repository.list_services(profile_id=profile_id)

    def upsert_service(self, heartbeat: ServiceHeartbeat) -> ServiceHeartbeat:
        self._require_profile(heartbeat.profile_id)
        return self.repository.upsert_heartbeat(heartbeat)

    def list_incidents(self, *, profile_id: str | None = None) -> list[IncidentRecord]:
        return self.repository.list_incidents(profile_id=profile_id)

    def create_incident(self, incident: IncidentRecord) -> IncidentRecord:
        self._require_profile(incident.profile_id)
        created = self.repository.create_incident(incident)
        for policy in self.repository.list_alert_policies(
            profile_id=incident.profile_id
        ):
            if self._severity_rank(incident.severity) < self._severity_rank(
                policy.severity_threshold
            ):
                continue
            self.create_alert(
                AlertRecord(
                    profile_id=incident.profile_id,
                    incident_id=incident.incident_id,
                    severity=incident.severity,
                    title=incident.title,
                    message=incident.message,
                    channel=policy.channel,
                )
            )
        return created

    def list_alert_policies(
        self, *, profile_id: str | None = None
    ) -> list[AlertPolicy]:
        return self.repository.list_alert_policies(profile_id=profile_id)

    def create_alert_policy(self, policy: AlertPolicy) -> AlertPolicy:
        self._require_profile(policy.profile_id)
        return self.repository.create_alert_policy(policy)

    def list_alerts(self, *, profile_id: str | None = None) -> list[AlertRecord]:
        return self.repository.list_alerts(profile_id=profile_id)

    def create_alert(self, alert: AlertRecord) -> AlertRecord:
        created = self.repository.create_alert(alert)
        self.notifier.notify(title=created.title, message=created.message)
        return created

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

    def _require_profile(self, profile_id: str) -> ProfileConfig:
        self.bootstrap_profiles()
        model = self.repository.get_profile(profile_id)
        if model is None:
            raise ControlPlaneError(404, "Profile not found")
        return model

    @staticmethod
    def _severity_rank(severity) -> int:
        order = {"info": 1, "warning": 2, "critical": 3}
        return order[str(severity)]

    @staticmethod
    def _slippage_bps(order_price: float | None, fill_price: float) -> float | None:
        if order_price in (None, 0):
            return None
        return ((fill_price - order_price) / order_price) * 10_000
