"""Filing registration and ingestion endpoints."""

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile, status

from src.api.dependencies import get_filing_service, get_hybrid_retriever
from src.config.logging_config import get_logger
from src.core.schemas import FilingCreate, FilingRead, FilingStatus, FilingType
from src.db.models import Filing
from src.db.repository import FilingRepository
from src.services.filing_service import FilingService

logger = get_logger(__name__)
router = APIRouter(prefix="/api/filings", tags=["filings"])


def _to_read(filing: Filing) -> FilingRead:
    return FilingRead(
        id=filing.id,
        company_name=filing.company_name,
        ticker=filing.ticker,
        cik=filing.cik,
        fiscal_year=filing.fiscal_year,
        filing_type=FilingType(filing.filing_type),
        status=FilingStatus(filing.status),
        chunk_count=len(filing.chunks),
        created_at=filing.created_at,
    )


@router.post("", response_model=FilingRead, status_code=status.HTTP_201_CREATED)
def create_filing(
    payload: FilingCreate, service: FilingService = Depends(get_filing_service)
) -> FilingRead:
    return _to_read(service.create_filing(payload))


@router.get("", response_model=list[FilingRead])
def list_filings(
    limit: int = 50, offset: int = 0, service: FilingService = Depends(get_filing_service)
) -> list[FilingRead]:
    return [_to_read(f) for f in service.list_filings(limit=limit, offset=offset)]


@router.get("/{filing_id}", response_model=FilingRead)
def get_filing(filing_id: int, service: FilingService = Depends(get_filing_service)) -> FilingRead:
    return _to_read(service.get_filing(filing_id))


@router.post("/{filing_id}/ingest", response_model=FilingRead, status_code=status.HTTP_202_ACCEPTED)
async def ingest_filing(
    filing_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    service: FilingService = Depends(get_filing_service),
) -> FilingRead:
    """Accepts a directly uploaded 10-K PDF and ingests it in the background."""
    pdf_bytes = await file.read()
    background_tasks.add_task(_run_bytes_ingestion, filing_id, pdf_bytes)
    return _to_read(service.get_filing(filing_id))


@router.post(
    "/{filing_id}/ingest-from-url", response_model=FilingRead, status_code=status.HTTP_202_ACCEPTED
)
async def ingest_filing_from_url(
    filing_id: int,
    background_tasks: BackgroundTasks,
    service: FilingService = Depends(get_filing_service),
) -> FilingRead:
    """Downloads the filing's registered source_url from SEC EDGAR and ingests it."""
    filing = service.get_filing(filing_id)
    background_tasks.add_task(_run_url_ingestion, filing_id)
    return _to_read(filing)


def _run_bytes_ingestion(filing_id: int, pdf_bytes: bytes) -> None:
    from src.db.session import session_scope

    with session_scope() as db:
        service = FilingService(FilingRepository(db), get_hybrid_retriever())
        try:
            service.ingest_from_bytes(filing_id, pdf_bytes)
        except Exception as exc:
            logger.error("filing.ingest_task_failed", filing_id=filing_id, error=str(exc))


async def _run_url_ingestion(filing_id: int) -> None:
    from src.db.session import session_scope

    with session_scope() as db:
        service = FilingService(FilingRepository(db), get_hybrid_retriever())
        try:
            await service.ingest_from_url(filing_id)
        except Exception as exc:
            logger.error("filing.ingest_url_task_failed", filing_id=filing_id, error=str(exc))