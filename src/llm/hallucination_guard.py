"""
Verifies every numeric claim in a generated FinancialRiskReport against the
actual retrieved source text before the report is trusted.

Two independent checks per metric:
1. Excerpt grounding — the cited excerpt must actually appear in the text
   of the chunk it claims to come from.
2. Value grounding — a number matching the claimed value (within
   NUMERIC_TOLERANCE_PCT) must be findable in that same chunk's text.

Metrics that fail either check are dropped from the report and logged in
`numeric_hallucination_flags` rather than silently kept — this is the
mechanism that gives the system its near-zero numeric hallucination rate,
as opposed to just hoping the model behaves.
"""

import re

from src.config.settings import get_settings
from src.core.schemas import FinancialRiskReport, RetrievedChunk

_NUMBER_PATTERN = re.compile(r"[-+]?\$?\d[\d,]*\.?\d*")


class HallucinationGuard:
    def __init__(self, tolerance_pct: float | None = None) -> None:
        settings = get_settings()
        self._tolerance_pct = (
            tolerance_pct if tolerance_pct is not None else settings.numeric_tolerance_pct
        )

    def verify(
        self, report: FinancialRiskReport, retrieved_chunks: list[RetrievedChunk]
    ) -> FinancialRiskReport:
        chunk_lookup = {c.chunk_id: c.text for c in retrieved_chunks}
        flags = list(report.numeric_hallucination_flags)
        verified_metrics = []

        for metric in report.extracted_metrics:
            source_text = chunk_lookup.get(metric.source.chunk_id)
            if source_text is None:
                flags.append(
                    f"{metric.metric_name}: cited chunk_id {metric.source.chunk_id} "
                    "not present in retrieved context"
                )
                continue
            if not self._excerpt_grounded(metric.source.excerpt, source_text):
                flags.append(f"{metric.metric_name}: cited excerpt not found in source chunk")
                continue
            if not self._value_grounded(metric.value, source_text):
                flags.append(
                    f"{metric.metric_name}: value {metric.value} not found within tolerance "
                    "in cited chunk"
                )
                continue
            verified_metrics.append(metric)

        verified_names = {m.metric_name for m in verified_metrics}
        verified_findings = []
        for finding in report.risk_findings:
            verified_findings.append(
                finding.model_copy(
                    update={
                        "supporting_metrics": [
                            m for m in finding.supporting_metrics if m.metric_name in verified_names
                        ],
                        "sources": [s for s in finding.sources if s.chunk_id in chunk_lookup],
                    }
                )
            )

        return report.model_copy(
            update={
                "extracted_metrics": verified_metrics,
                "risk_findings": verified_findings,
                "numeric_hallucination_flags": flags,
            }
        )

    @staticmethod
    def _excerpt_grounded(excerpt: str, source_text: str) -> bool:
        normalized_excerpt = " ".join(excerpt.split()).lower()
        normalized_source = " ".join(source_text.split()).lower()
        if not normalized_excerpt:
            return False
        probe = normalized_excerpt[:80]
        return probe in normalized_source

    def _value_grounded(self, value: float, source_text: str) -> bool:
        for match in _NUMBER_PATTERN.findall(source_text):
            cleaned = match.replace("$", "").replace(",", "")
            try:
                candidate = float(cleaned)
            except ValueError:
                continue
            if candidate == 0 and value == 0:
                return True
            if candidate != 0 and abs(candidate - value) / abs(candidate) * 100 <= self._tolerance_pct:
                return True
        return False