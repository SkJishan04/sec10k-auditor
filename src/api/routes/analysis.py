"""Analysis run trigger and retrieval endpoints."""

from fastapi import APIRouter, Depends, status

from src.api.dependencies import get_analysis_service
from src.core.schemas import AnalysisRunRead, FinancialRiskReport
from src.db.models import AnalysisRun
from src.services.analysis_service import AnalysisService

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


def _to_read(run: AnalysisRun) -> AnalysisRunRead:
    report = FinancialRiskReport.model_validate(run.report_json) if run.report_json else None
    return AnalysisRunRead(
        id=run.id,
        filing_id=run.filing_id,
        status=run.status,
        report=report,
        retrieval_latency_ms=run.retrieval_latency_ms,
        generation_latency_ms=run.generation_latency_ms,
        total_cost_usd=run.total_cost_usd,
        created_at=run.created_at,
    )


@router.post(
    "/filings/{filing_id}/run", response_model=AnalysisRunRead, status_code=status.HTTP_201_CREATED
)
def run_analysis(
    filing_id: int, service: AnalysisService = Depends(get_analysis_service)
) -> AnalysisRunRead:
    return _to_read(service.run_analysis(filing_id))


@router.get("/filings/{filing_id}", response_model=list[AnalysisRunRead])
def list_runs(
    filing_id: int, service: AnalysisService = Depends(get_analysis_service)
) -> list[AnalysisRunRead]:
    return [_to_read(r) for r in service.list_runs(filing_id)]


@router.get("/{run_id}", response_model=AnalysisRunRead)
def get_run(run_id: int, service: AnalysisService = Depends(get_analysis_service)) -> AnalysisRunRead:
    return _to_read(service.get_run(run_id))