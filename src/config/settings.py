"""
Centralized application configuration.

All runtime configuration is sourced from environment variables (via a .env
file in local development, or real environment variables in deployment).
Nothing here hardcodes secrets or environment-specific values, which keeps
the same codebase portable across dev/staging/prod.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Application ---
    app_env: Literal["development", "staging", "production", "test"] = "development"
    log_level: str = "INFO"

    # --- Database ---
    database_url: str = "postgresql+psycopg2://auditor:auditor@localhost:5432/sec10k_auditor"

    # --- Vector store ---
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "sec10k_filings"

    # --- Embeddings ---
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"

    # --- LLM providers ---
    llm_provider: Literal["anthropic", "local_qlora"] = "anthropic"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    local_base_model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    local_adapter_path: str = "./training/output/dpo_adapter"

    # --- SEC EDGAR ---
    edgar_user_agent: str = "Anonymous anonymous@example.com"

    # --- Retrieval ---
    hybrid_dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    hybrid_sparse_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    retrieval_top_k: int = Field(default=8, ge=1, le=50)

    # --- Hallucination guard ---
    numeric_tolerance_pct: float = Field(default=0.5, ge=0.0)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance so env parsing happens once."""
    return Settings()