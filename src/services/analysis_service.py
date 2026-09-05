"""Analysis run business logic: validates filing readiness, invokes the
orchestrator, and persists the resulting report and run metrics."""

from src.agents.orchestrator import AnalysisOrchestrator
from src.config.logging_config import get_logger
from src.core.exceptions import AuditorError
from src.core.schemas import FilingStatus
from src.db.models import AnalysisRun
from src.db.repository import AnalysisRepository, FilingRepository
from src.retrieval.hybrid_retriever import ChunkRecord

logger = get_logger(__name__)


class AnalysisService:
    def __init__(
        self,
        analysis_repo: AnalysisRepository,
        filing_repo: FilingRepository,
        orchestrator: AnalysisOrchestrator,
    ) -> None:
        self._analysis_repo = analysis_repo
        self._filing_repo = filing_repo
        self._orchestrator = orchestrator

    def run_analysis(self, filing_id: int) -> AnalysisRun:
        filing = self._filing_repo.get(filing_id)
        if filing.status != FilingStatus.READY.value:
            raise AuditorError(
                f"Filing {filing_id} is not ready for analysis (status={filing.status})"
            )

        run = self._analysis_repo.create(AnalysisRun(filing_id=filing_id, status="running"))

        persisted_chunks = self._filing_repo.get_chunks(filing_id)
        chunk_records = [
            ChunkRecord(chunk_id=c.id, text=c.text, page_number=c.page_number, filing_id=filing_id)
            for c in persisted_chunks
        ]

        try:
            result = self._orchestrator.analyze(filing_id=filing_id, chunk_records=chunk_records)
            run.status = "completed"
            run.report_json = result.report.model_dump(mode="json")
            run.retrieval_latency_ms = result.retrieval_latency_ms
            run.generation_latency_ms = result.generation_latency_ms
            run.total_cost_usd = result.total_cost_usd
        except Exception as exc:
            run.status = "failed"
            run.error_message = str(exc)
            logger.error("analysis_service.run_failed", filing_id=filing_id, error=str(exc))

        return self._analysis_repo.update(run)

    def get_run(self, run_id: int) -> AnalysisRun:
        return self._analysis_repo.get(run_id)

    def list_runs(self, filing_id: int) -> list[AnalysisRun]:
        return self._analysis_repo.list_for_filing(filing_id)