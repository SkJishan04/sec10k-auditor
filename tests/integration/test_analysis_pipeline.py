import pytest

from src.agents.orchestrator import OrchestratorResult
from src.core.exceptions import AuditorError
from src.core.schemas import FilingType, FinancialRiskReport, RiskSeverity
from src.db.models import Filing, FilingChunk
from src.db.repository import AnalysisRepository, FilingRepository
from src.services.analysis_service import AnalysisService


class _StubOrchestrator:
    def analyze(self, filing_id, chunk_records):
        report = FinancialRiskReport(
            filing_id=filing_id,
            summary="Stubbed summary",
            extracted_metrics=[],
            risk_findings=[],
            overall_risk_severity=RiskSeverity.LOW,
        )
        return OrchestratorResult(
            report=report, retrieval_latency_ms=12.0, generation_latency_ms=34.0, total_cost_usd=0.01
        )


def test_run_analysis_persists_report(db_session):
    filing_repo = FilingRepository(db_session)
    analysis_repo = AnalysisRepository(db_session)

    filing = filing_repo.create(
        Filing(
            company_name="Example Corp",
            ticker="EXMP",
            cik="0000320193",
            fiscal_year=2023,
            filing_type=FilingType.FORM_10K.value,
            status="ready",
        )
    )
    filing_repo.add_chunks(filing.id, [FilingChunk(chunk_index=0, page_number=1, text="Some filing text.")])

    service = AnalysisService(analysis_repo, filing_repo, _StubOrchestrator())
    run = service.run_analysis(filing.id)

    assert run.status == "completed"
    assert run.report_json["summary"] == "Stubbed summary"
    assert run.retrieval_latency_ms == 12.0


def test_run_analysis_rejects_non_ready_filing(db_session):
    filing_repo = FilingRepository(db_session)
    analysis_repo = AnalysisRepository(db_session)

    filing = filing_repo.create(
        Filing(
            company_name="Pending Corp",
            ticker="PEND",
            cik="0000000001",
            fiscal_year=2023,
            filing_type=FilingType.FORM_10K.value,
            status="pending",
        )
    )

    service = AnalysisService(analysis_repo, filing_repo, _StubOrchestrator())

    with pytest.raises(AuditorError):
        service.run_analysis(filing.id)