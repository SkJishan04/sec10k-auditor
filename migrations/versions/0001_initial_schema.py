"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-01-15 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "filings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("company_name", sa.String(length=255), nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("cik", sa.String(length=10), nullable=False),
        sa.Column("fiscal_year", sa.Integer(), nullable=False),
        sa.Column("filing_type", sa.String(length=10), nullable=False, server_default="10-K"),
        sa.Column("source_url", sa.String(length=1024), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("page_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_filings_ticker", "filings", ["ticker"])
    op.create_index("ix_filings_cik", "filings", ["cik"])

    op.create_table(
        "filing_chunks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "filing_id",
            sa.Integer(),
            sa.ForeignKey("filings.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
    )
    op.create_index("ix_filing_chunks_filing_id", "filing_chunks", ["filing_id"])

    op.create_table(
        "analysis_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "filing_id",
            sa.Integer(),
            sa.ForeignKey("filings.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("report_json", sa.JSON(), nullable=True),
        sa.Column("retrieval_latency_ms", sa.Float(), nullable=True),
        sa.Column("generation_latency_ms", sa.Float(), nullable=True),
        sa.Column("total_cost_usd", sa.Float(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_analysis_runs_filing_id", "analysis_runs", ["filing_id"])


def downgrade() -> None:
    op.drop_index("ix_analysis_runs_filing_id", table_name="analysis_runs")
    op.drop_table("analysis_runs")
    op.drop_index("ix_filing_chunks_filing_id", table_name="filing_chunks")
    op.drop_table("filing_chunks")
    op.drop_index("ix_filings_cik", table_name="filings")
    op.drop_index("ix_filings_ticker", table_name="filings")
    op.drop_table("filings")