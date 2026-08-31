"""
Dense vector store abstraction backed by Chroma (persistent, local, and
zero-infra to run -- an appropriate choice for a portfolio-scale project;
swapping to pgvector/Pinecone in production only requires a new
implementation of this same interface).
"""

from typing import Protocol

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config.settings import get_settings


class VectorStore(Protocol):
    def upsert(
        self, ids: list[str], embeddings: list[list[float]], metadatas: list[dict], documents: list[str]
    ) -> None: ...

    def query(self, query_embedding: list[float], top_k: int, where: dict | None = None) -> list[dict]: ...


class ChromaVectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        documents: list[str],
    ) -> None:
        if not ids:
            return
        self._collection.upsert(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents
        )

    def query(self, query_embedding: list[float], top_k: int, where: dict | None = None) -> list[dict]:
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            # Chroma returns cosine *distance*; convert to a similarity score in [0, 1].
            similarity = max(0.0, 1.0 - distance / 2.0)
            hits.append(
                {"id": chunk_id, "document": document, "metadata": metadata, "score": similarity}
            )
        return hits

    def delete_by_filing(self, filing_id: int) -> None:
        self._collection.delete(where={"filing_id": filing_id})