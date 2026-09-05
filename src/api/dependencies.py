"""
FastAPI dependency wiring. Expensive components (embedding model, vector
store connection, LLM provider) are constructed once via `lru_cache` and
reused across requests; request-scoped components (DB session, services)
are constructed per-request via `Depends`.
"""

from functools import lru_cache

from fastapi import Depends
from sqlalchemy.orm import Session

from src.agents.orchestrator import AnalysisOrchestrator
from src.config.settings import get_settings
from src.db.repository import AnalysisRepository, FilingRepository
from src.db.session import get_db
from src.llm.anthropic_provider import AnthropicProvider
from src.llm.base_provider import BaseLLMProvider
from src.llm.hallucination_guard import HallucinationGuard
from src.llm.local_qlora_provider import LocalQLoRAProvider
from src.retrieval.embedding_service import EmbeddingService
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import ChromaVectorStore
from src.services.analysis_service import AnalysisService
from src.services.filing_service import FilingService


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@lru_cache
def get_vector_store() -> ChromaVectorStore:
    return ChromaVectorStore()


@lru_cache
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(get_embedding_service(), get_vector_store())


@lru_cache
def get_llm_provider() -> BaseLLMProvider:
    settings = get_settings()
    if settings.llm_provider == "anthropic":
        return AnthropicProvider()
    return LocalQLoRAProvider()


@lru_cache
def get_hallucination_guard() -> HallucinationGuard:
    return HallucinationGuard()


@lru_cache
def get_orchestrator() -> AnalysisOrchestrator:
    return AnalysisOrchestrator(
        retriever=get_hybrid_retriever(),
        llm_provider=get_llm_provider(),
        guard=get_hallucination_guard(),
    )


def get_filing_service(db: Session = Depends(get_db)) -> FilingService:
    return FilingService(FilingRepository(db), get_hybrid_retriever())


def get_analysis_service(db: Session = Depends(get_db)) -> AnalysisService:
    return AnalysisService(AnalysisRepository(db), FilingRepository(db), get_orchestrator())