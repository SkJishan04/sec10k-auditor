# 📊 SEC 10-K RAG & Financial Statement Auditor

**Agentic Hybrid-RAG system for automated financial risk detection in SEC 10-K filings**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-blue.svg)](#testing)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](#docker)

<!-- 📸 IMAGE PLACEHOLDER 1: Hero banner / architecture illustration -->
<!-- ![Project Banner](docs/images/banner.png) -->

---

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Motivation](#motivation)
4. [Key Features](#key-features)
5. [System Workflow](#system-workflow)
6. [Architecture](#architecture)
7. [Tech Stack](#tech-stack)
8. [Project Structure](#project-structure)
9. [Usage Examples](#usage-examples)
10. [Results](#results)
11. [Evaluation Methodology](#evaluation-methodology)
12. [Setup & Installation](#setup--installation)
13. [Environment Variables](#environment-variables)
14. [Testing](#testing)
15. [Docker](#docker)
16. [CI/CD](#cicd)
17. [Limitations](#limitations)
18. [Future Improvements](#future-improvements)
19. [License](#license)

---

## Overview

This project is a **production-oriented Retrieval-Augmented Generation (RAG) system** that automates one of the most time-consuming tasks in financial due diligence: extracting and cross-checking numerical disclosures buried inside SEC 10-K filings.

It combines **hybrid retrieval** (dense semantic search + sparse BM25 keyword search), a **DPO-fine-tuned Llama-3 8B model** specialized for numeric extraction, and a custom **hallucination guard** that verifies every extracted figure against its retrieved source text before it is ever surfaced to a user.

The result is a system that produces a structured, source-cited **Financial Risk Report** for a given filing — flagging things like off-balance-sheet liabilities, aggressive revenue recognition, and going-concern language — with every numeric claim traceable back to a specific page.

## Problem Statement

Financial analysts, auditors, and M&A due-diligence teams routinely spend **hundreds of hours per deal** manually reading 10-K filings that can run 150–300+ pages. Key risks are often buried in footnotes:

- Off-balance-sheet financing arrangements
- Revenue recognition policies that front-load earnings
- Related-party transactions
- Contingent liabilities and litigation reserves
- Going-concern qualifications

Manual review is slow, inconsistent across analysts, and error-prone at scale — and naive LLM summarization introduces a worse failure mode: **numeric hallucination**, where a model confidently reports a dollar figure that doesn't actually appear in the source document.

## Motivation

> *"Financial due diligence in M&A takes weeks due to manual parsing of SEC 10-K footnoted risks. This system was built to cut financial-audit extraction time by automating retrieval and extraction — while treating numeric hallucination as a hard failure mode to engineer against, not an acceptable tradeoff."*

This project exists to demonstrate that LLM systems can be built **responsibly** in high-stakes, numeric-sensitive domains — by pairing generation with retrieval-grounded verification rather than trusting model output at face value.

## Key Features

- 🔍 **Hybrid Retrieval** — Dense vector search (sentence-transformers) fused with sparse BM25 keyword search, so exact figures and rare terms are recovered alongside semantic matches.
- 🧠 **Fine-Tuned Numeric Extraction** — Llama-3 8B specialized via QLoRA + DPO specifically to reduce numeric hallucination during financial figure extraction.
- ✅ **Hallucination Guard** — Every extracted metric is checked against its cited source span before being included in a report; unverifiable claims are flagged and excluded, not silently kept.
- 🤖 **Agentic Orchestration** — Tool-calling orchestrator plans retrieval and extraction steps rather than using a single fixed prompt.
- 📑 **Structured, Cited Output** — Reports are validated Pydantic objects with page-level source citations for every claim.
- ⚙️ **Config-Driven Pipeline** — Retrieval weights, model provider, and thresholds are all environment-driven, not hardcoded.
- 📈 **Observability** — Per-run latency and cost tracking, structured JSON logging.
- 🧪 **Evaluation Harness** — Retrieval quality and numeric-accuracy evaluation against a golden dataset.

## System Workflow

<!-- 📸 IMAGE PLACEHOLDER 2: Optional custom workflow illustration -->
<!-- ![Workflow Illustration](docs/images/workflow.png) -->

```mermaid
flowchart LR
    A[📄 SEC 10-K PDF] --> B[PDF Parser<br/>page-level extraction]
    B --> C[Chunker<br/>page-anchored, overlapping]
    C --> D[(Embedding Model)]
    C --> E[(BM25 Index)]
    D --> F[(Chroma Vector Store)]
    F --> G{Hybrid Retriever<br/>weighted fusion}
    E --> G
    G --> H[Agentic Orchestrator]
    H --> I[DPO-Tuned LLM<br/>numeric extraction]
    I --> J[Hallucination Guard<br/>source verification]
    J -->|verified| K[✅ Financial Risk Report]
    J -->|unverified| L[🚩 Flagged & Excluded]
```

**Flow summary:**
1. A 10-K PDF is parsed page-by-page and split into overlapping, page-anchored chunks.
2. Chunks are embedded and indexed in both a dense vector store and an in-memory BM25 index.
3. A user query (or automated risk-scan trigger) is routed through the hybrid retriever, which fuses dense + sparse scores.
4. The agentic orchestrator plans which extraction/analysis tools to call and in what order.
5. The fine-tuned LLM extracts structured numeric metrics and risk narratives from retrieved chunks.
6. The hallucination guard cross-checks every number against its cited source text before it's persisted or shown.

## Architecture

```mermaid
flowchart TB
    subgraph Client
        UI[Frontend<br/>HTML/CSS/JS]
    end

    subgraph API["FastAPI Application"]
        Routes[Routes Layer<br/>filings.py / analysis.py]
        Services[Service Layer<br/>filing_service / analysis_service]
        Repo[Repository Layer<br/>data access abstraction]
    end

    subgraph AI["AI/ML Pipeline"]
        Retriever[Hybrid Retriever]
        Orchestrator[Agentic Orchestrator]
        LLM[LLM Provider<br/>Anthropic / Local QLoRA]
        Guard[Hallucination Guard]
    end

    subgraph Data["Data Layer"]
        PG[(PostgreSQL<br/>filings, chunks, runs)]
        Chroma[(ChromaDB<br/>dense vectors)]
    end

    subgraph Ingestion
        EDGAR[EDGAR Client]
        Parser[PDF Parser]
        Chunker[Chunker]
    end

    UI --> Routes
    Routes --> Services
    Services --> Repo
    Repo --> PG
    Services --> Retriever
    Retriever --> Chroma
    Retriever --> Orchestrator
    Orchestrator --> LLM
    LLM --> Guard
    Guard --> Services
    EDGAR --> Parser --> Chunker --> Retriever
```

**Design principles:**
- **Layered separation** — routes never touch the database directly; services never import FastAPI.
- **Repository pattern** — all SQL lives behind `FilingRepository` / `AnalysisRepository`, enabling unit tests against in-memory SQLite.
- **Provider abstraction** — `base_provider.py` defines the LLM interface; Anthropic and local QLoRA implementations are interchangeable via a config flag.

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM / GenAI** | Anthropic Claude, Llama-3 8B (QLoRA + DPO) | Report generation & specialized numeric extraction |
| **Fine-tuning** | `transformers`, `peft`, `trl`, `bitsandbytes` | Parameter-efficient fine-tuning with preference optimization |
| **Retrieval** | `sentence-transformers`, `rank-bm25`, ChromaDB | Hybrid dense + sparse semantic search |
| **Backend** | FastAPI, Pydantic v2 | Async REST API with validated I/O |
| **Database** | PostgreSQL, SQLAlchemy 2.0, Alembic | Persistence & schema migrations |
| **Ingestion** | `httpx`, `tenacity`, `pypdf` | EDGAR fetching, retries, PDF parsing |
| **Observability** | `structlog` | Structured JSON logging, cost/latency tracking |
| **Frontend** | HTML / CSS / vanilla JS | Lightweight report viewer |
| **Testing** | `pytest`, `pytest-asyncio`, `pytest-cov` | Unit, integration, and eval tests |
| **Tooling** | `ruff`, `mypy`, Docker, Docker Compose | Linting, typing, containerization |

## Project Structure

```text
sec10k-auditor/
├── src/
│   ├── config/          # Settings & logging configuration
│   ├── core/            # Shared schemas & exceptions (no external deps)
│   ├── data/            # EDGAR client, PDF parser, chunker
│   ├── retrieval/        # Embeddings, vector store, BM25, hybrid fusion
│   ├── llm/              # Provider abstraction, prompts, hallucination guard
│   ├── agents/           # Tool-calling orchestrator
│   ├── db/               # SQLAlchemy models, session, repositories
│   ├── services/          # Business logic (filing & analysis services)
│   └── api/               # FastAPI app, routes, dependencies
├── migrations/            # Alembic schema migrations
├── training/               # QLoRA/DPO dataset prep & training scripts
├── evaluation/              # Golden dataset + retrieval/numeric eval
├── frontend/                # Static report viewer UI
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/                  # One-off utility scripts
├── docker-compose.yml
├── Dockerfile
├── alembic.ini
├── pyproject.toml
└── .env.example
```

> Each layer has a single responsibility: `api/` never contains business logic, `services/` never imports FastAPI, and `db/` is the only place raw SQL/ORM queries are written.

## Usage Examples

### 1. Register a filing for ingestion

```bash
curl -X POST http://localhost:8000/api/filings \
  -H "Content-Type: application/json" \
  -d '{
        "company_name": "Example Corp",
        "ticker": "EXMP",
        "cik": "0000320193",
        "fiscal_year": 2023,
        "filing_type": "10-K"
      }'
```

**Response:**
```json
{
  "id": 1,
  "company_name": "Example Corp",
  "ticker": "EXMP",
  "status": "pending",
  "chunk_count": 0,
  "created_at": "2026-01-15T00:00:00Z"
}
```

### 2. Trigger a risk analysis run

```bash
curl -X POST http://localhost:8000/api/analysis/filings/1/run
```

### 3. Sample structured output

```json
{
  "summary": "Example Corp shows moderate risk concentrated in off-balance-sheet lease commitments...",
  "overall_risk_severity": "medium",
  "risk_findings": [
    {
      "category": "off_balance_sheet_liability",
      "severity": "medium",
      "title": "Operating lease commitments not reflected on balance sheet",
      "explanation": "Footnote 12 discloses $340M in future lease obligations...",
      "sources": [{ "chunk_id": 214, "page_number": 87 }]
    }
  ],
  "numeric_hallucination_flags": []
}
```

<!-- 📸 IMAGE PLACEHOLDER 3: Screenshot of frontend report viewer -->
<!-- ![Report Viewer Screenshot](docs/images/report-viewer.png) -->

## Results

> Replace the placeholders below with your actual measured numbers once you run `evaluation/retrieval_eval.py` and `evaluation/numeric_accuracy_eval.py` against your golden dataset.

| Metric | Baseline (Dense-only) | Hybrid RAG (This System) |
|---|---|---|
| Retrieval Recall@8 | — | — |
| Numeric Extraction Accuracy | — | — |
| Numeric Hallucination Rate | — | — |
| Avg. Analysis Latency | — | — |
| Avg. Cost per Filing | — | — |

## Evaluation Methodology

Two complementary evaluation tracks:

**1. Retrieval Evaluation** (`evaluation/retrieval_eval.py`)
Measures whether the hybrid retriever surfaces the correct source chunk for a curated set of known question–answer pairs drawn from real filings, reported as Recall@K and Mean Reciprocal Rank.

**2. Numeric Accuracy Evaluation** (`evaluation/numeric_accuracy_eval.py`)
For each metric in the golden dataset, compares the LLM-extracted value against the ground-truth value within a configurable tolerance (`NUMERIC_TOLERANCE_PCT`), and separately tracks the **hallucination rate** — extracted values with no supporting source span at all.

```mermaid
flowchart LR
    G[Golden Dataset<br/>known Q/A + ground-truth metrics] --> R[Retrieval Eval<br/>Recall@K, MRR]
    G --> N[Numeric Accuracy Eval<br/>tolerance-based comparison]
    N --> H[Hallucination Rate Tracking]
```

## Setup & Installation

### Prerequisites
- Python 3.11+
- PostgreSQL 14+
- Docker & Docker Compose (recommended)

### Local Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/<your-username>/sec10k-auditor.git
cd sec10k-auditor

# 2. Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Configure environment
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY, DATABASE_URL, etc.

# 5. Run database migrations
alembic upgrade head

# 6. Start the API
uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+psycopg2://auditor:auditor@localhost:5432/sec10k_auditor` |
| `CHROMA_PERSIST_DIR` | Local path for vector store persistence | `./data/chroma` |
| `EMBEDDING_MODEL_NAME` | Sentence-transformers model | `all-mpnet-base-v2` |
| `LLM_PROVIDER` | `anthropic` or `local_qlora` | `anthropic` |
| `ANTHROPIC_API_KEY` | API key for Claude | *(required if using Anthropic)* |
| `EDGAR_USER_AGENT` | Required identifying header for SEC EDGAR | *(required)* |
| `HYBRID_DENSE_WEIGHT` / `HYBRID_SPARSE_WEIGHT` | Fusion weights for hybrid retrieval | `0.6` / `0.4` |
| `NUMERIC_TOLERANCE_PCT` | Tolerance for hallucination guard verification | `0.5` |

> Full list in [`.env.example`](.env.example). Never commit a real `.env` file — it's excluded via `.gitignore`.

## Testing

```bash
# Run the full test suite with coverage
pytest --cov=src --cov-report=term-missing

# Run only unit tests
pytest tests/unit

# Run only integration tests
pytest tests/integration
```

Test coverage includes:
- **Unit tests** — chunker boundary behavior, BM25 scoring, hallucination guard logic
- **Integration tests** — full filing ingestion → analysis API flow
- **Evaluation tests** — retrieval recall and numeric extraction accuracy against the golden dataset

## Docker

```bash
# Build and run the full stack (API + PostgreSQL)
docker compose up --build
```

This starts:
- `api` — the FastAPI application
- `db` — PostgreSQL with a persisted volume

Migrations run automatically on container startup.

## CI/CD

> Suggested GitHub Actions pipeline — add `.github/workflows/ci.yml`:

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_USER: auditor
          POSTGRES_PASSWORD: auditor
          POSTGRES_DB: sec10k_auditor
        ports: ["5432:5432"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: mypy src
      - run: alembic upgrade head
      - run: pytest --cov=src
```

Every push runs linting (`ruff`), type checking (`mypy`), migrations, and the full test suite.

## Limitations

- **Scanned/image-only PDFs** are not supported without an OCR layer (no text extraction fallback yet).
- **Single-filing scope** — cross-filing / cross-year trend analysis is not yet implemented.
- **BM25 index is rebuilt per-query** rather than persisted, which is efficient at single-filing scale but would need optimization for very large corpora.
- **Local QLoRA provider requires GPU** — CPU-only environments must use the Anthropic provider.
- **Evaluation golden dataset is manually curated** and currently limited in size; broader coverage would strengthen result confidence.

## Future Improvements

- [ ] OCR fallback for scanned filings (e.g. via `pytesseract`)
- [ ] Cross-filing trend analysis (multi-year risk trajectory)
- [ ] Persisted BM25 index for large-scale corpora
- [ ] Caching layer for repeated retrieval queries
- [ ] Expanded golden evaluation dataset with adversarial numeric cases
- [ ] Support for 10-Q and 8-K filings in addition to 10-K
- [ ] Async background job queue for long-running ingestion/analysis

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  Built as a demonstration of production-oriented AI/ML/GenAI engineering practices.
</p>

