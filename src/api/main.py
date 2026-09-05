"""FastAPI application entrypoint: middleware, exception handling, routing,
and the Gradio UI mounted at the root path."""

from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import analysis, filings
from src.config.logging_config import configure_logging, get_logger
from src.core.exceptions import (
    AnalysisNotFoundError,
    AuditorError,
    FilingNotFoundError,
    FilingParsingError,
    HallucinationDetectedError,
    LLMProviderError,
    RetrievalError,
)
from src.ui.gradio_app import demo as gradio_demo

configure_logging()
logger = get_logger(__name__)

_ERROR_STATUS_MAP: dict[type[AuditorError], int] = {
    FilingNotFoundError: 404,
    AnalysisNotFoundError: 404,
    FilingParsingError: 422,
    RetrievalError: 502,
    LLMProviderError: 502,
    HallucinationDetectedError: 502,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("app.startup")
    yield
    logger.info("app.shutdown")


app = FastAPI(
    title="SEC 10-K RAG & Financial Statement Auditor",
    description="Agentic hybrid-RAG system for automated financial risk detection in SEC 10-K filings.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AuditorError)
async def auditor_error_handler(request: Request, exc: AuditorError) -> JSONResponse:
    status_code = _ERROR_STATUS_MAP.get(type(exc), 400)
    logger.warning("request.error", path=str(request.url), error=str(exc), status_code=status_code)
    return JSONResponse(status_code=status_code, content={"detail": str(exc)})


# REST API stays under /api/* — this is the "real" backend and what the
# test suite exercises directly.
app.include_router(filings.router)
app.include_router(analysis.router)

# Gradio UI mounted at the root path, in the same process, so there is a
# single service to run and no CORS hop between UI and API.
app = gr.mount_gradio_app(app, gradio_demo, path="/")