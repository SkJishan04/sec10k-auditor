"""
Parses SEC 10-K PDF filings into page-level text with page number tracking.

Page numbers are preserved end-to-end (parser -> chunker -> vector store ->
retrieved chunk -> LLM citation) so every extracted metric can be traced
back to a specific page, which is what makes the hallucination guard and
the UI citations meaningful rather than decorative.
"""

import io
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.core.exceptions import FilingParsingError


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class ParsedFiling:
    pages: list[ParsedPage]

    @property
    def full_text(self) -> str:
        return "\n".join(page.text for page in self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)


class PdfParser:
    """Extracts normalized text from a 10-K PDF, one page at a time."""

    def parse(self, pdf_bytes: bytes) -> ParsedFiling:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
        except PdfReadError as exc:
            raise FilingParsingError(f"Could not open PDF: {exc}") from exc

        if len(reader.pages) == 0:
            raise FilingParsingError("PDF contains no pages")

        pages: list[ParsedPage] = []
        for index, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception as exc:  # pypdf can raise various parser-internal errors
                raise FilingParsingError(f"Failed to extract text from page {index}: {exc}") from exc
            pages.append(ParsedPage(page_number=index, text=self._normalize(raw_text)))

        if all(not p.text.strip() for p in pages):
            raise FilingParsingError(
                "No extractable text found in any page (document may be a scanned image "
                "without an OCR text layer)"
            )

        return ParsedFiling(pages=pages)

    @staticmethod
    def _normalize(text: str) -> str:
        # Collapse the hard line-wraps and repeated whitespace that PDF text
        # extraction typically introduces, without destroying paragraph breaks.
        lines = [line.strip() for line in text.splitlines()]
        collapsed: list[str] = []
        for line in lines:
            if line:
                collapsed.append(line)
            elif collapsed and collapsed[-1] != "":
                collapsed.append("")
        return " ".join(collapsed).replace("  ", " ").strip()