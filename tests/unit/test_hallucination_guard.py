from src.core.schemas import ExtractedMetric, FinancialRiskReport, RetrievedChunk, RiskSeverity, SourceSpan
from src.llm.hallucination_guard import HallucinationGuard


def _chunk(chunk_id: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, text=text, page_number=1, dense_score=0.9, sparse_score=0.8, fused_score=0.85
    )


def _report_with_metric(metric: ExtractedMetric) -> FinancialRiskReport:
    return FinancialRiskReport(
        filing_id=1,
        summary="test",
        extracted_metrics=[metric],
        risk_findings=[],
        overall_risk_severity=RiskSeverity.LOW,
    )


def test_guard_keeps_grounded_metric():
    text = "Total long-term debt was $340 million as of December 31, 2023."
    chunks = [_chunk(1, text)]
    metric = ExtractedMetric(
        metric_name="Total Long-Term Debt",
        value=340.0,
        unit="USD_millions",
        period="FY2023",
        confidence=0.9,
        source=SourceSpan(chunk_id=1, page_number=1, excerpt=text),
    )

    verified = HallucinationGuard(tolerance_pct=0.5).verify(_report_with_metric(metric), chunks)

    assert len(verified.extracted_metrics) == 1
    assert verified.numeric_hallucination_flags == []


def test_guard_flags_unsupported_value():
    text = "Total long-term debt was $340 million as of December 31, 2023."
    chunks = [_chunk(1, text)]
    metric = ExtractedMetric(
        metric_name="Total Long-Term Debt",
        value=9999.0,
        unit="USD_millions",
        period="FY2023",
        confidence=0.9,
        source=SourceSpan(chunk_id=1, page_number=1, excerpt=text),
    )

    verified = HallucinationGuard(tolerance_pct=0.5).verify(_report_with_metric(metric), chunks)

    assert len(verified.extracted_metrics) == 0
    assert len(verified.numeric_hallucination_flags) == 1


def test_guard_flags_metric_citing_unknown_chunk():
    chunks = [_chunk(1, "Total long-term debt was $340 million.")]
    metric = ExtractedMetric(
        metric_name="Total Long-Term Debt",
        value=340.0,
        unit="USD_millions",
        period="FY2023",
        confidence=0.9,
        source=SourceSpan(chunk_id=99, page_number=1, excerpt="irrelevant"),
    )

    verified = HallucinationGuard().verify(_report_with_metric(metric), chunks)

    assert verified.extracted_metrics == []
    assert "chunk_id 99" in verified.numeric_hallucination_flags[0]