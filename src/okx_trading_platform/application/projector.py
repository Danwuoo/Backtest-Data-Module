from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from okx_trading_platform.domain import (
    AlertPolicy,
    AlertRecord,
    BacktestRun,
    BalanceSnapshot,
    DatasetRecord,
    DatasetVersion,
    ExecutionSnapshot,
    FeatureSet,
    FillRecord,
    FundingEntry,
    IncidentRecord,
    InstrumentConfig,
    LiveRun,
    ModelVersion,
    OrderPlan,
    OrderState,
    PaperRun,
    PnLSnapshot,
    PositionSnapshot,
    ProfileConfig,
    RiskPolicyConfig,
    RiskSnapshot,
    RunArtifact,
    ServiceHeartbeat,
    SleeveConfig,
    StrategyConfig,
)
from okx_trading_platform.shared.data_lake import DataLakeWriter, EventEnvelope

from .repositories import PlatformRepository


class ReadModelProjector:
    def __init__(self, repository: PlatformRepository) -> None:
        self.repository = repository

    def project_event(self, event: EventEnvelope) -> Any:
        payload = event.payload
        event_type = event.event_type
        if event_type == "profile.upserted":
            return self.repository.create_profile(ProfileConfig.model_validate(payload))
        if event_type == "risk_policy.upserted":
            return self.repository.create_risk_policy(
                RiskPolicyConfig.model_validate(payload)
            )
        if event_type == "allocator.upserted":
            from okx_trading_platform.domain import AllocatorConfig

            return self.repository.create_allocator(AllocatorConfig.model_validate(payload))
        if event_type == "sleeve.upserted":
            return self.repository.create_sleeve(SleeveConfig.model_validate(payload))
        if event_type == "instrument.upserted":
            return self.repository.create_instrument(
                InstrumentConfig.model_validate(payload)
            )
        if event_type == "strategy.upserted":
            return self.repository.create_strategy(StrategyConfig.model_validate(payload))
        if event_type == "model_version.upserted":
            return self.repository.create_model_version(ModelVersion.model_validate(payload))
        if event_type == "dataset.upserted":
            return self.repository.create_dataset(DatasetRecord.model_validate(payload))
        if event_type == "feature.upserted":
            return self.repository.create_feature(FeatureSet.model_validate(payload))
        if event_type == "dataset_version.upserted":
            return self.repository.create_dataset_version(
                DatasetVersion.model_validate(payload)
            )
        if event_type == "run_artifact.upserted":
            return self.repository.create_run_artifact(
                RunArtifact.model_validate(payload)
            )
        if event_type == "backtest_run.upserted":
            return self.repository.create_backtest(BacktestRun.model_validate(payload))
        if event_type == "paper_run.upserted":
            return self.repository.create_paper_run(PaperRun.model_validate(payload))
        if event_type == "live_run.upserted":
            return self.repository.create_live_run(LiveRun.model_validate(payload))
        if event_type == "order_plan.upserted":
            return self.repository.create_order_plan(OrderPlan.model_validate(payload))
        if event_type == "order_state.upserted":
            return self.repository.create_order(OrderState.model_validate(payload))
        if event_type == "fill.upserted":
            return self.repository.create_fill(FillRecord.model_validate(payload))
        if event_type == "ledger_entry.upserted":
            from okx_trading_platform.domain import LedgerEntry

            return self.repository.create_ledger_entry(
                LedgerEntry.model_validate(payload)
            )
        if event_type == "funding_entry.upserted":
            return self.repository.create_funding_entry(
                FundingEntry.model_validate(payload)
            )
        if event_type == "pnl_snapshot.upserted":
            return self.repository.create_pnl_snapshot(PnLSnapshot.model_validate(payload))
        if event_type == "risk_snapshot.upserted":
            return self.repository.create_risk_snapshot(
                RiskSnapshot.model_validate(payload)
            )
        if event_type == "execution_snapshot.upserted":
            return self.repository.create_execution_snapshot(
                ExecutionSnapshot.model_validate(payload)
            )
        if event_type == "position.upserted":
            return self.repository.upsert_position(PositionSnapshot.model_validate(payload))
        if event_type == "balance.upserted":
            return self.repository.upsert_balance(BalanceSnapshot.model_validate(payload))
        if event_type == "service_heartbeat.upserted":
            return self.repository.upsert_heartbeat(
                ServiceHeartbeat.model_validate(payload)
            )
        if event_type == "incident.upserted":
            return self.repository.create_incident(IncidentRecord.model_validate(payload))
        if event_type == "alert_policy.upserted":
            return self.repository.create_alert_policy(AlertPolicy.model_validate(payload))
        if event_type == "alert.upserted":
            return self.repository.create_alert(AlertRecord.model_validate(payload))
        raise ValueError(f"Unsupported event type: {event_type}")


@dataclass
class AuditEventPipeline:
    lake: DataLakeWriter
    projector: ReadModelProjector
    stream: str
    source_service: str

    def append_and_project(
        self,
        *,
        event_type: str,
        payload: dict[str, Any],
        profile_id: str | None = None,
        strategy_id: str | None = None,
        run_id: str | None = None,
        inst_id: str | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
        schema_version: str = "v1",
    ) -> Any:
        envelope = EventEnvelope(
            event_type=event_type,
            source_service=self.source_service,
            payload=payload,
            profile_id=profile_id,
            strategy_id=strategy_id,
            run_id=run_id,
            inst_id=inst_id,
            correlation_id=correlation_id,
            causation_id=causation_id,
            schema_version=schema_version,
        )
        self.lake.append_events(self.stream, [envelope])
        return self.projector.project_event(envelope)
