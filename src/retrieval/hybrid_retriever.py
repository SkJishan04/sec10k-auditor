"""
Fuses dense (semantic) and sparse (BM25 keyword) retrieval into a single
ranked list of chunks.

Fusion strategy: normalized weighted-sum of the two scores rather than
reciprocal-rank fusion. Weighted-sum was chosen because both underlying
scores are already normalized to [0, 1] (cosine similarity and
max-normalized BM25), making the fusion weights directly interpretable as
"how much do we trust semantic vs. exact-match signal for this domain" --
which for dense financial disclosures with exact figures, we weight
slightly toward dense (default 0.6/0.4) but keep both signals present.
"""

from dataclasses import dataclass

from src.config.logging_config import get_logger
from src.config.settings import get_settings
from src.core.exceptions import RetrievalError
from src.core.schemas import RetrievedChunk
from src.retrieval.bm25_index import Bm25Index, BM25Document
from src.retrieval.embedding_service import EmbeddingService
from src.retrieval.vector_store import ChromaVectorStore

logger = get_logger(__name__)


@dataclass(frozen=True)
class ChunkRecord:
    """Minimal representation of a persisted chunk needed to build indices."""

    chunk_id: int
    text: str
    page_number: int | None
    filing_id: int


class HybridRetriever:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: ChromaVectorStore,
    ) -> None:
        settings = get_settings()
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._dense_weight = settings.hybrid_dense_weight
        self._sparse_weight = settings.hybrid_sparse_weight
        self._top_k = settings.retrieval_top_k

    def index_chunks(self, chunks: list[ChunkRecord]) -> None:
        """Embed and upsert chunks into the dense vector store. BM25 is built
        lazily per-query from chunk records passed to `retrieve`, since it is
        cheap to rebuild at this scale and avoids a second persistence path."""
        if not chunks:
            return
        embeddings = self._embedding_service.embed_texts([c.text for c in chunks])
        self._vector_store.upsert(
            ids=[str(c.chunk_id) for c in chunks],
            embeddings=embeddings.tolist(),
            metadatas=[
                {"filing_id": c.filing_id, "page_number": c.page_number or -1} for c in chunks
            ],
            documents=[c.text for c in chunks],
        )

    def retrieve(
        self, query: str, filing_id: int, all_chunks: list[ChunkRecord], top_k: int | None = None
    ) -> list[RetrievedChunk]:
        top_k = top_k or self._top_k
        try:
            query_embedding = self._embedding_service.embed_query(query)
            dense_hits = self._vector_store.query(
                query_embedding=query_embedding,
                top_k=top_k * 2,
                where={"filing_id": filing_id},
            )

            bm25_docs = [
                BM25Document(
                    chunk_id=str(c.chunk_id),
                    text=c.text,
                    metadata={"filing_id": c.filing_id, "page_number": c.page_number},
                )
                for c in all_chunks
            ]
            sparse_hits = Bm25Index(bm25_docs).query(query, top_k=top_k * 2)
        except Exception as exc:
            raise RetrievalError(f"Hybrid retrieval failed for query={query!r}: {exc}") from exc

        return self._fuse(dense_hits, sparse_hits, all_chunks, top_k)

    def _fuse(
        self,
        dense_hits: list[dict],
        sparse_hits: list[dict],
        all_chunks: list[ChunkRecord],
        top_k: int,
    ) -> list[RetrievedChunk]:
        chunk_lookup = {str(c.chunk_id): c for c in all_chunks}
        dense_scores = {hit["id"]: hit["score"] for hit in dense_hits}
        sparse_scores = {hit["id"]: hit["score"] for hit in sparse_hits}

        all_ids = set(dense_scores) | set(sparse_scores)
        fused: list[RetrievedChunk] = []
        for chunk_id in all_ids:
            record = chunk_lookup.get(chunk_id)
            if record is None:
                continue
            dense_score = dense_scores.get(chunk_id, 0.0)
            sparse_score = sparse_scores.get(chunk_id, 0.0)
            fused_score = self._dense_weight * dense_score + self._sparse_weight * sparse_score
            fused.append(
                RetrievedChunk(
                    chunk_id=record.chunk_id,
                    text=record.text,
                    page_number=record.page_number,
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                    fused_score=fused_score,
                )
            )

        fused.sort(key=lambda c: c.fused_score, reverse=True)
        logger.info("hybrid_retriever.retrieve", candidates=len(fused), returned=min(top_k, len(fused)))
        return fused[:top_k]