"""
Filing lifecycle business logic: registration, PDF ingestion (from bytes
or a source URL), chunking, and indexing into the hybrid retriever.
Depends only on the repository abstraction and the retriever, so it can be
unit-tested against an in-memory SQLite session with no FastAPI or network
dependency in the loop.
"""

from src.config.logging_config import get_logger
from src.core.schemas import FilingCreate, FilingStatus
from src.data.chunker import FilingChunker
from src.data.edgar_client import EdgarClient
from src.data.pdf_parser import PdfParser
from src.db.models import Filing, FilingChunk
from src.db.repository import FilingRepository
from src.retrieval.hybrid_retriever import ChunkRecord, HybridRetriever

logger = get_logger(__name__)


class FilingService:
    def __init__(self, filing_repo: FilingRepository, retriever: HybridRetriever) -> None:
        self._repo = filing_repo
        self._retriever = retriever
        self._parser = PdfParser()
        self._chunker = FilingChunker()
        self._edgar = EdgarClient()

    def create_filing(self, payload: FilingCreate) -> Filing:
        filing = Filing(
            company_name=payload.company_name,
            ticker=payload.ticker,
            cik=payload.cik,
            fiscal_year=payload.fiscal_year,
            filing_type=payload.filing_type.value,
            source_url=payload.source_url,
            status=FilingStatus.PENDING.value,
        )
        return self._repo.create(filing)

    def get_filing(self, filing_id: int) -> Filing:
        return self._repo.get(filing_id)

    def list_filings(self, limit: int = 50, offset: int = 0) -> list[Filing]:
        return self._repo.list_all(limit=limit, offset=offset)

    async def ingest_from_url(self, filing_id: int) -> None:
        filing = self._repo.get(filing_id)
        if not filing.source_url:
            raise ValueError(f"Filing {filing_id} has no source_url set")

        self._repo.update_status(filing_id, FilingStatus.INGESTING.value)
        try:
            pdf_bytes = await self._edgar.download_document(filing.source_url)
            self._ingest_bytes(filing_id, pdf_bytes)
        except Exception:
            self._repo.update_status(filing_id, FilingStatus.FAILED.value)
            raise

    def ingest_from_bytes(self, filing_id: int, pdf_bytes: bytes) -> None:
        self._repo.update_status(filing_id, FilingStatus.INGESTING.value)
        try:
            self._ingest_bytes(filing_id, pdf_bytes)
        except Exception:
            self._repo.update_status(filing_id, FilingStatus.FAILED.value)
            raise

    def _ingest_bytes(self, filing_id: int, pdf_bytes: bytes) -> None:
        parsed = self._parser.parse(pdf_bytes)
        chunks = self._chunker.chunk(parsed)

        chunk_models = [
            FilingChunk(chunk_index=c.chunk_index, page_number=c.page_number, text=c.text)
            for c in chunks
        ]
        self._repo.add_chunks(filing_id, chunk_models)

        persisted_chunks = self._repo.get_chunks(filing_id)
        records = [
            ChunkRecord(chunk_id=c.id, text=c.text, page_number=c.page_number, filing_id=filing_id)
            for c in persisted_chunks
        ]
        self._retriever.index_chunks(records)

        filing = self._repo.get(filing_id)
        filing.page_count = parsed.page_count
        self._repo.update_status(filing_id, FilingStatus.READY.value)

        logger.info(
            "filing_service.ingest_complete",
            filing_id=filing_id,
            pages=parsed.page_count,
            chunks=len(chunk_models),
        )