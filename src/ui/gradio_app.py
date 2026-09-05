"""
Gradio UI for the SEC 10-K Financial Risk Auditor.

This replaces the static HTML/JS frontend. It is mounted directly into the
FastAPI app (see src/api/main.py) via gradio.mount_gradio_app, so it runs
in the same process and calls the service layer in-process rather than
looping back through HTTP -- there is no redundant network hop and no
separate CORS concern, while the underlying business logic is exactly the
same FilingService / AnalysisService used by the REST routes.
"""

import pandas as pd
import gradio as gr

from src.api.dependencies import get_hybrid_retriever, get_orchestrator
from src.core.schemas import FilingCreate
from src.db.repository import AnalysisRepository, FilingRepository
from src.db.session import session_scope
from src.services.analysis_service import AnalysisService
from src.services.filing_service import FilingService

_FILINGS_COLUMNS = ["ID", "Company", "Ticker", "FY", "Status", "Chunks"]
_METRICS_COLUMNS = ["Metric", "Value", "Unit", "Period", "Page"]
_FINDINGS_COLUMNS = ["Severity", "Category", "Title", "Explanation"]


def _list_filings_df() -> pd.DataFrame:
    with session_scope() as db:
        filings = FilingRepository(db).list_all(limit=100)
        rows = [
            {
                "ID": f.id,
                "Company": f.company_name,
                "Ticker": f.ticker,
                "FY": f.fiscal_year,
                "Status": f.status,
                "Chunks": len(f.chunks),
            }
            for f in filings
        ]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=_FILINGS_COLUMNS)


def create_filing(company_name: str, ticker: str, cik: str, fiscal_year: float):
    if not company_name or not ticker or not cik or not fiscal_year:
        return "⚠️ All fields are required.", _list_filings_df()

    try:
        payload = FilingCreate(
            company_name=company_name, ticker=ticker, cik=cik, fiscal_year=int(fiscal_year)
        )
    except Exception as exc:
        return f"⚠️ Invalid input: {exc}", _list_filings_df()

    with session_scope() as db:
        service = FilingService(FilingRepository(db), get_hybrid_retriever())
        filing = service.create_filing(payload)
        message = f"✅ Created filing **#{filing.id}** for {filing.company_name} ({filing.status})"

    return message, _list_filings_df()


def ingest_pdf(filing_id: float, pdf_file):
    if pdf_file is None:
        return "⚠️ Upload a PDF first.", _list_filings_df()
    if filing_id is None:
        return "⚠️ Enter a filing ID.", _list_filings_df()

    with open(pdf_file.name, "rb") as f:
        pdf_bytes = f.read()

    with session_scope() as db:
        service = FilingService(FilingRepository(db), get_hybrid_retriever())
        try:
            service.ingest_from_bytes(int(filing_id), pdf_bytes)
            message = f"✅ Ingested PDF into filing **#{int(filing_id)}**."
        except Exception as exc:
            message = f"❌ Ingestion failed: {exc}"

    return message, _list_filings_df()


def run_analysis(filing_id: float):
    empty_metrics = pd.DataFrame(columns=_METRICS_COLUMNS)
    empty_findings = pd.DataFrame(columns=_FINDINGS_COLUMNS)

    if filing_id is None:
        return "⚠️ Enter a filing ID.", "", empty_metrics, empty_findings

    with session_scope() as db:
        analysis_service = AnalysisService(
            AnalysisRepository(db), FilingRepository(db), get_orchestrator()
        )
        try:
            run = analysis_service.run_analysis(int(filing_id))
        except Exception as exc:
            return f"❌ {exc}", "", empty_metrics, empty_findings

        if run.status != "completed" or not run.report_json:
            status = f"⚠️ Run status: {run.status}. {run.error_message or ''}"
            return status, "", empty_metrics, empty_findings

        report = run.report_json
        status_msg = (
            f"✅ Completed — retrieval {run.retrieval_latency_ms:.0f}ms, "
            f"generation {run.generation_latency_ms:.0f}ms, cost ${run.total_cost_usd:.4f}"
        )

        summary_md = (
            f"### Overall Risk: **{report['overall_risk_severity'].upper()}**\n\n{report['summary']}"
        )
        if report.get("numeric_hallucination_flags"):
            flags = "\n".join(f"- {flag}" for flag in report["numeric_hallucination_flags"])
            summary_md += f"\n\n**⚠️ Flagged (excluded) claims:**\n{flags}"

        metrics_df = (
            pd.DataFrame(
                [
                    {
                        "Metric": m["metric_name"],
                        "Value": m["value"],
                        "Unit": m["unit"],
                        "Period": m["period"],
                        "Page": m["source"]["page_number"],
                    }
                    for m in report["extracted_metrics"]
                ]
            )
            if report["extracted_metrics"]
            else empty_metrics
        )

        findings_df = (
            pd.DataFrame(
                [
                    {
                        "Severity": f["severity"],
                        "Category": f["category"],
                        "Title": f["title"],
                        "Explanation": f["explanation"],
                    }
                    for f in report["risk_findings"]
                ]
            )
            if report["risk_findings"]
            else empty_findings
        )

    return status_msg, summary_md, metrics_df, findings_df


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="SEC 10-K Financial Risk Auditor", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 📊 SEC 10-K RAG & Financial Statement Auditor\n"
            "Agentic hybrid-RAG risk analysis for SEC 10-K filings. Every extracted number "
            "is verified against its source text by a hallucination guard before being shown."
        )

        with gr.Tab("1. Register Filing"):
            with gr.Row():
                company_input = gr.Textbox(label="Company Name", placeholder="Apple Inc.")
                ticker_input = gr.Textbox(label="Ticker", placeholder="AAPL")
            with gr.Row():
                cik_input = gr.Textbox(label="CIK", placeholder="0000320193")
                year_input = gr.Number(label="Fiscal Year", value=2023, precision=0)
            create_btn = gr.Button("Create Filing", variant="primary")
            create_status = gr.Markdown()

        with gr.Tab("2. Ingest PDF"):
            filing_id_ingest = gr.Number(label="Filing ID", precision=0)
            pdf_upload = gr.File(label="10-K PDF", file_types=[".pdf"])
            ingest_btn = gr.Button("Ingest PDF", variant="primary")
            ingest_status = gr.Markdown()

        with gr.Tab("3. Filings"):
            refresh_btn = gr.Button("Refresh")
            filings_table = gr.Dataframe(
                headers=_FILINGS_COLUMNS, value=_list_filings_df(), interactive=False
            )

        with gr.Tab("4. Run Analysis"):
            filing_id_analyze = gr.Number(label="Filing ID", precision=0)
            run_btn = gr.Button("Run Analysis", variant="primary")
            run_status = gr.Markdown()
            summary_output = gr.Markdown()
            metrics_output = gr.Dataframe(label="Extracted Metrics", interactive=False)
            findings_output = gr.Dataframe(label="Risk Findings", interactive=False)

        create_btn.click(
            create_filing,
            inputs=[company_input, ticker_input, cik_input, year_input],
            outputs=[create_status, filings_table],
        )
        ingest_btn.click(
            ingest_pdf, inputs=[filing_id_ingest, pdf_upload], outputs=[ingest_status, filings_table]
        )
        refresh_btn.click(_list_filings_df, outputs=filings_table)
        run_btn.click(
            run_analysis,
            inputs=filing_id_analyze,
            outputs=[run_status, summary_output, metrics_output, findings_output],
        )

    return demo


demo = build_demo()