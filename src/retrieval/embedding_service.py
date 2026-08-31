"""
Wraps a sentence-transformers model behind a small interface so the rest of
the system depends on an abstraction, not a specific embedding library --
swapping to an API-based embedding model later only touches this file.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config.settings import get_settings


class EmbeddingService:
    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.embedding_model_name
        self._model = self._load_model(self._model_name)

    @staticmethod
    @lru_cache(maxsize=2)
    def _load_model(model_name: str) -> SentenceTransformer:
        return SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._model.get_sentence_embedding_dimension()))
        return self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def embed_query(self, query: str) -> list[float]:
        embedding = self._model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        return embedding[0].tolist()

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()