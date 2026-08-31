"""
Repository layer: isolates all raw SQLAlchemy queries behind a
domain-oriented interface. Services depend on `FilingRepository` /
`AnalysisRepository`, never on `Session` directly -- this is what lets the
service layer be unit-tested against an in-memory SQLite session without
any mocking of query internals.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.exceptions import AnalysisNotFoundError, FilingNotFoundError
from src.db.models import AnalysisRun, Filing, FilingChunk


class FilingRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create(self, filing: Filing) -> Filing:
        self._db.add(filing)
        self._db.commit()
        self._db.refresh(filing)
        return filing

    def get(self, filing_id: int) -> Filing:
        filing = self._db.get(Filing, filing_id)
        if filing is None:
            raise FilingNotFoundError(f"Filing {filing_id} not found")
        return filing

    def list_all(self, limit: int = 50, offset: int = 0) -> list[Filing]:
        stmt = select(Filing).order_by(Filing.created_at.desc()).limit(limit).offset(offset)
        return list(self._db.execute(stmt).scalars().all())

    def update_status(self, filing_id: int, status: str) -> Filing:
        filing = self.get(filing_id)
        filing.status = status
        self._db.commit()
        self._db.refresh(filing)
        return filing

    def add_chunks(self, filing_id: int, chunks: list[FilingChunk]) -> None:
        for chunk in chunks:
            chunk.filing_id = filing_id
            self._db.add(chunk)
        self._db.commit()

    def get_chunks(self, filing_id: int) -> list[FilingChunk]:
        stmt = select(FilingChunk).where(FilingChunk.filing_id == filing_id).order_by(
            FilingChunk.chunk_index
        )
        return list(self._db.execute(stmt).scalars().all())


class AnalysisRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create(self, run: AnalysisRun) -> AnalysisRun:
        self._db.add(run)
        self._db.commit()
        self._db.refresh(run)
        return run

    def get(self, run_id: int) -> AnalysisRun:
        run = self._db.get(AnalysisRun, run_id)
        if run is None:
            raise AnalysisNotFoundError(f"Analysis run {run_id} not found")
        return run

    def update(self, run: AnalysisRun) -> AnalysisRun:
        self._db.add(run)
        self._db.commit()
        self._db.refresh(run)
        return run

    def list_for_filing(self, filing_id: int) -> list[AnalysisRun]:
        stmt = (
            select(AnalysisRun)
            .where(AnalysisRun.filing_id == filing_id)
            .order_by(AnalysisRun.created_at.desc())
        )
        return list(self._db.execute(stmt).scalars().all())