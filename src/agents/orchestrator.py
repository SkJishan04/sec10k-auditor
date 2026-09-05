"""
Orchestrates the end-to-end analysis pipeline: retrieve chunks for a fixed
set of canonical risk queries, deduplicate and rank them, run structured
extraction, then pass the result through the hallucination guard.
"""

import time
from dataclasses import dataclass

from src.config.logging_config import get_logger
from src.core.schemas import FinancialRiskReport
from src.llm.base_provider import BaseLLMProvider
from src.llm.hallucination_guard import HallucinationGuard
from src.retrieval.hybrid_retriever import ChunkRecord, HybridRetriever
from src.agents.tools import ExtractionTool, RetrievalTool

logger = get_logger(__name__)

CANONICAL_RISK_QUERIES = [
    "off-balance sheet arrangements and liabilities",
    "revenue recognition policy and timing",
    "related party transactions",
    "contingent liabilities and legal proceedings",
    "going concern and liquidity risk",
    "total long-term debt and lease obligations",
]


@dataclass
class OrchestratorResult:
    report: FinancialRiskReport
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_cost_usd: float


class AnalysisOrchestrator:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm_provider: BaseLLMProvider,
        guard: HallucinationGuard | None = None,
        queries: list[str] | None = None,
    ) -> None:
        self._retrieval_tool = RetrievalTool(retriever=retriever)
        self._extraction_tool = ExtractionTool(llm_provider=llm_provider)
        self._guard = guard or HallucinationGuard()
        self._queries = queries or CANONICAL_RISK_QUERIES

    def analyze(self, filing_id: int, chunk_records: list[ChunkRecord]) -> OrchestratorResult:
        t0 = time.perf_counter()
        retrieved: dict[int, object] = {}
        for query in self._queries:
            hits = self._retrieval_tool.run(
                query=query, filing_id=filing_id, all_chunks=chunk_records, top_k=4
            )
            for hit in hits:
                existing = retrieved.get(hit.chunk_id)
                if existing is None or hit.fused_score > existing.fused_score:
                    retrieved[hit.chunk_id] = hit
        retrieval_latency_ms = (time.perf_counter() - t0) * 1000

        ranked_chunks = sorted(retrieved.values(), key=lambda c: c.fused_score, reverse=True)[:20]
        logger.info(
            "orchestrator.retrieval_complete",
            filing_id=filing_id,
            unique_chunks=len(retrieved),
            used_chunks=len(ranked_chunks),
        )

        t1 = time.perf_counter()
        raw_report, usage = self._extraction_tool.run(filing_id=filing_id, chunks=ranked_chunks)
        generation_latency_ms = (time.perf_counter() - t1) * 1000

        verified_report = self._guard.verify(raw_report, ranked_chunks)
        logger.info(
            "orchestrator.analysis_complete",
            filing_id=filing_id,
            metrics_kept=len(verified_report.extracted_metrics),
            metrics_flagged=len(verified_report.numeric_hallucination_flags),
        )

        return OrchestratorResult(
            report=verified_report,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_cost_usd=usage.get("cost_usd", 0.0),
        )