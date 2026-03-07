"""platform v2 baseline

Revision ID: 0001_platform_v2_baseline
Revises:
Create Date: 2026-03-07 00:00:00.000000
"""

from __future__ import annotations

from alembic import op

from okx_trading_platform.application.persistence import (
    AlertPolicyV2Model,
    AlertV2Model,
    AllocatorV2Model,
    BacktestRunV2Model,
    BalanceV2Model,
    DatasetV2Model,
    DatasetVersionV2Model,
    ExecutionSnapshotV2Model,
    FeatureV2Model,
    FillV2Model,
    FundingEntryV2Model,
    IncidentV2Model,
    InstrumentV2Model,
    LedgerEntryV2Model,
    LiveRunV2Model,
    ModelVersionV2Model,
    OrderPlanV2Model,
    OrderV2Model,
    PaperRunV2Model,
    PnLSnapshotV2Model,
    PositionV2Model,
    ProfileV2Model,
    RiskPolicyV2Model,
    RiskSnapshotV2Model,
    RunArtifactV2Model,
    SleeveV2Model,
    StrategyV2Model,
)

revision = "0001_platform_v2_baseline"
down_revision = None
branch_labels = None
depends_on = None


TABLES = [
    ProfileV2Model.__table__,
    RiskPolicyV2Model.__table__,
    AllocatorV2Model.__table__,
    SleeveV2Model.__table__,
    InstrumentV2Model.__table__,
    StrategyV2Model.__table__,
    ModelVersionV2Model.__table__,
    DatasetV2Model.__table__,
    DatasetVersionV2Model.__table__,
    FeatureV2Model.__table__,
    RunArtifactV2Model.__table__,
    BacktestRunV2Model.__table__,
    PaperRunV2Model.__table__,
    LiveRunV2Model.__table__,
    OrderPlanV2Model.__table__,
    OrderV2Model.__table__,
    FillV2Model.__table__,
    LedgerEntryV2Model.__table__,
    FundingEntryV2Model.__table__,
    PnLSnapshotV2Model.__table__,
    RiskSnapshotV2Model.__table__,
    ExecutionSnapshotV2Model.__table__,
    PositionV2Model.__table__,
    BalanceV2Model.__table__,
    IncidentV2Model.__table__,
    AlertPolicyV2Model.__table__,
    AlertV2Model.__table__,
]


def upgrade() -> None:
    bind = op.get_bind()
    for table in TABLES:
        table.create(bind, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    for table in reversed(TABLES):
        table.drop(bind, checkfirst=True)
