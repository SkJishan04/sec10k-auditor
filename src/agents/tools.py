"""
Tool abstractions used by the orchestrator.

Rather than a dynamic ReAct-style loop where the LLM decides which tool to
call next, this orchestrator uses a deterministic plan (retrieve, then
extract) built from explicit tool objects. In a numeric-sensitive
financial-auditing context, predictable execution order and cost matters
more than the flexibility a free-form tool-calling loop would add — this
still demonstrates the tool-abstraction pattern while keeping behavior
auditable and reproducible.
"""

from dataclasses import dataclass

from src.core.schemas import FinancialRiskReport, RetrievedChunk
from src.llm.base_provider import BaseLLMProvider
from src.llm.prompts import build_extraction_prompt
from src.retrieval.hybrid_retriever import ChunkRecord, HybridRetriever


@dataclass
class RetrievalTool:
    """Retrieves the most relevant filing chunks for a query via hybrid search."""

    retriever: HybridRetriever
    name: str = "retrieve_filing_chunks"
    description: str = (
        "Retrieve the most relevant filing text chunks for a natural-language "
        "query using hybrid dense + BM25 search."
    )

    def run(
        self, query: str, filing_id: int, all_chunks: list[ChunkRecord], top_k: int = 4
    ) -> list[RetrievedChunk]:
        return self.retriever.retrieve(
            query=query, filing_id=filing_id, all_chunks=all_chunks, top_k=top_k
        )


@dataclass
class ExtractionTool:
    """Generates a structured financial risk report from retrieved chunks."""

    llm_provider: BaseLLMProvider
    name: str = "extract_financial_report"
    description: str = (
        "Generate a structured financial risk report from retrieved filing "
        "chunks using the configured LLM provider."
    )

    def run(self, filing_id: int, chunks: list[RetrievedChunk]) -> tuple[FinancialRiskReport, dict]:
        prompt = build_extraction_prompt(filing_id=filing_id, chunks=chunks)
        return self.llm_provider.generate_structured(prompt=prompt, schema=FinancialRiskReport)