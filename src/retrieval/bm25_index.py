"""
Sparse keyword retrieval using BM25 (Okapi BM25 via rank-bm25).

Dense retrieval alone systematically misses exact numeric tokens and rare
proper nouns (e.g. "$47.2 million", "Topic 606", a specific subsidiary
name) because embeddings compress them toward semantically similar but
lexically different neighbors. BM25 is included specifically to recover
those exact-match cases, and its score is fused with the dense score in
HybridRetriever.
"""

import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9$%.]+")


def _tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_PATTERN.findall(text)]


@dataclass
class BM25Document:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


class Bm25Index:
    """In-memory BM25 index scoped to a single filing's chunks.

    Rebuilt per-filing at query time from the DB rather than persisted
    separately, which keeps it trivially consistent with the source of
    truth (Postgres) at the cost of index build time -- an acceptable
    tradeoff at the scale of a single 10-K (a few hundred chunks).
    """

    def __init__(self, documents: list[BM25Document]) -> None:
        self._documents = documents
        self._corpus_tokens = [_tokenize(doc.text) for doc in documents]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def query(self, query_text: str, top_k: int) -> list[dict]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query_text))
        max_score = max(scores) if len(scores) else 0.0
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        hits: list[dict] = []
        for idx in ranked_indices:
            normalized_score = float(scores[idx] / max_score) if max_score > 0 else 0.0
            doc = self._documents[idx]
            hits.append(
                {
                    "id": doc.chunk_id,
                    "document": doc.text,
                    "metadata": doc.metadata,
                    "score": normalized_score,
                }
            )
        return hits