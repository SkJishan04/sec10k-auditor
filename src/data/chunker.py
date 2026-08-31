"""
Splits parsed filing text into overlapping, page-anchored chunks suitable
for embedding.

The overlap exists so a financial figure and the sentence that gives it
context aren't split across a chunk boundary with no overlap to recover
from. Page numbers are carried per-chunk (a chunk may span two pages; we
attribute it to the page where it starts) to preserve citation accuracy.
"""

from dataclasses import dataclass

from src.data.pdf_parser import ParsedFiling


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    text: str
    page_number: int | None


class FilingChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, parsed_filing: ParsedFiling) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        chunk_index = 0

        for page in parsed_filing.pages:
            words = page.text.split(" ")
            if not words or words == [""]:
                continue

            start = 0
            while start < len(words):
                end = min(start + self._words_per_chunk(), len(words))
                chunk_text = " ".join(words[start:end]).strip()
                if chunk_text:
                    chunks.append(
                        TextChunk(
                            chunk_index=chunk_index,
                            text=chunk_text,
                            page_number=page.page_number,
                        )
                    )
                    chunk_index += 1
                if end == len(words):
                    break
                start = end - self._words_per_overlap()

        return chunks

    def _words_per_chunk(self) -> int:
        # Approximate ~5.5 characters per word (English financial prose average).
        return max(1, self.chunk_size // 6)

    def _words_per_overlap(self) -> int:
        return max(0, self.chunk_overlap // 6)