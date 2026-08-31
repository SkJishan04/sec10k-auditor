"""
Thin client around the public SEC EDGAR full-text search and filing-index
APIs. SEC EDGAR requires a descriptive User-Agent header identifying the
requester; using a browser-like or empty User-Agent gets you rate-limited
or blocked, so this is read from configuration rather than hardcoded.
"""

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.logging_config import get_logger
from src.config.settings import get_settings

logger = get_logger(__name__)

EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


class EdgarClient:
    """Fetches filing metadata and documents from SEC EDGAR."""

    def __init__(self, timeout_seconds: float = 15.0) -> None:
        settings = get_settings()
        self._headers = {"User-Agent": settings.edgar_user_agent}
        self._timeout = timeout_seconds

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def get_company_submissions(self, cik: str) -> dict:
        """Return the raw JSON submissions index for a company (zero-padded CIK)."""
        url = EDGAR_SUBMISSIONS_URL.format(cik=cik)
        async with httpx.AsyncClient(headers=self._headers, timeout=self._timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def download_document(self, url: str) -> bytes:
        """Download a filing document (PDF or HTML) given its direct EDGAR URL."""
        logger.info("edgar.download_document.start", url=url)
        async with httpx.AsyncClient(headers=self._headers, timeout=self._timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            logger.info("edgar.download_document.complete", url=url, bytes=len(response.content))
            return response.content

    def find_latest_10k_document_url(self, submissions: dict) -> str | None:
        """Locate the most recent 10-K primary document URL from a submissions payload."""
        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accession_numbers = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        cik = str(int(submissions.get("cik", "0")))

        for form, accession, primary_doc in zip(forms, accession_numbers, primary_docs):
            if form == "10-K":
                accession_no_dashes = accession.replace("-", "")
                return (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik}/{accession_no_dashes}/{primary_doc}"
                )
        return None