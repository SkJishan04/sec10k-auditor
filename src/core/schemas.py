"""
Pydantic schemas shared across layers (API request/response bodies, LLM
structured-output targets, and internal service contracts).

Keeping these in one module means the API layer, the LLM structured-output
layer, and the persistence layer all agree on the exact same shape for a
financial metric or risk finding -- there is only one definition to keep
in sync.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class FilingType(str, Enum):
    FORM_10K = "10-K"
    FORM_10Q = "10-Q"


class FilingStatus(str, Enum):
    PENDING = "pending"
    INGESTING = "ingesting"
    READY = "ready"
    FAILED = "failed"


class RiskSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(str, Enum):
    OFF_BALANCE_SHEET = "off_balance_sheet_liability"
    REVENUE_RECOGNITION = "aggressive_revenue_recognition"
    RELATED_PARTY = "related_party_transaction"
    GOING_CONCERN = "going_concern"
    CONTINGENT_LIABILITY = "contingent_liability"
    OTHER = "other"


class FilingCreate(BaseModel):
    """Payload to register a new filing for ingestion."""

    company_name: str = Field(..., min_length=1, max_length=255)
    ticker: str = Field(..., min_length=1, max_length=10)
    cik: str = Field(..., description="SEC Central Index Key, zero-padded to 10 digits")
    fiscal_year: int = Field(..., ge=1994, le=2100)
    filing_type: FilingType = FilingType.FORM_10K
    source_url: str | None = Field(
        default=None, description="Direct URL to the filing document on SEC EDGAR"
    )

    @field_validator("cik")
    @classmethod
    def pad_cik(cls, v: str) -> str:
        digits = v.strip().lstrip("0") or "0"
        if not digits.isdigit():
            raise ValueError("CIK must be numeric")
        return digits.zfill(10)


class FilingRead(BaseModel):
    id: int
    company_name: str
    ticker: str
    cik: str
    fiscal_year: int
    filing_type: FilingType
    status: FilingStatus
    chunk_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class SourceSpan(BaseModel):
    """Precise pointer back to the retrieved text a claim was derived from,
    used both for UI citations and for the hallucination guard's verification."""

    chunk_id: int
    page_number: int | None = None
    excerpt: str = Field(..., max_length=600)


class ExtractedMetric(BaseModel):
    """A single numeric financial fact extracted by the LLM, always tied to
    a verifiable source span. This is the atomic unit the hallucination
    guard checks against retrieved text before it is ever persisted."""

    metric_name: str = Field(..., examples=["Total Long-Term Debt"])
    value: float
    unit: str = Field(..., examples=["USD_thousands", "USD_millions", "percent", "ratio"])
    period: str = Field(..., examples=["FY2023", "Q4 2023"])
    source: SourceSpan
    confidence: float = Field(..., ge=0.0, le=1.0)


class RiskFinding(BaseModel):
    """A single flagged risk derived from one or more extracted metrics and
    supporting narrative disclosure text."""

    category: RiskCategory
    severity: RiskSeverity
    title: str = Field(..., max_length=200)
    explanation: str = Field(..., max_length=2000)
    supporting_metrics: list[ExtractedMetric] = Field(default_factory=list)
    sources: list[SourceSpan] = Field(default_factory=list)


class FinancialRiskReport(BaseModel):
    """The full structured output of an analysis run -- this is what the LLM
    is prompted to produce, and what is validated end-to-end before storage."""

    filing_id: int
    summary: str = Field(..., max_length=3000)
    extracted_metrics: list[ExtractedMetric]
    risk_findings: list[RiskFinding]
    overall_risk_severity: RiskSeverity
    numeric_hallucination_flags: list[str] = Field(
        default_factory=list,
        description="Any extracted values that failed source verification "
        "and were excluded from the report.",
    )


class AnalysisRunRead(BaseModel):
    id: int
    filing_id: int
    status: str
    report: FinancialRiskReport | None
    retrieval_latency_ms: float | None
    generation_latency_ms: float | None
    total_cost_usd: float | None
    created_at: datetime

    model_config = {"from_attributes": True}


class RetrievedChunk(BaseModel):
    """Internal contract returned by the hybrid retriever."""

    chunk_id: int
    text: str
    page_number: int | None
    dense_score: float
    sparse_score: float
    fused_score: float