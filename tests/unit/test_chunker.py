import pytest

from src.data.chunker import FilingChunker
from src.data.pdf_parser import ParsedFiling, ParsedPage


def test_chunker_respects_overlap_and_page_boundaries():
    page = ParsedPage(page_number=1, text=" ".join(f"word{i}" for i in range(300)))
    parsed = ParsedFiling(pages=[page])
    chunker = FilingChunker(chunk_size=300, chunk_overlap=60)

    chunks = chunker.chunk(parsed)

    assert len(chunks) > 1
    assert all(c.page_number == 1 for c in chunks)
    first_words = set(chunks[0].text.split(" "))
    second_words = set(chunks[1].text.split(" "))
    assert first_words & second_words


def test_chunker_skips_empty_pages():
    pages = [ParsedPage(page_number=1, text=""), ParsedPage(page_number=2, text="real content here")]
    parsed = ParsedFiling(pages=pages)
    chunker = FilingChunker()

    chunks = chunker.chunk(parsed)

    assert len(chunks) == 1
    assert chunks[0].page_number == 2


def test_chunker_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        FilingChunker(chunk_size=100, chunk_overlap=100)