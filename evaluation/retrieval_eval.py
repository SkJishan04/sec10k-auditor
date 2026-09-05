"""
Retrieval evaluation harness: measures Recall@K and Mean Reciprocal Rank of
the hybrid retriever against curated (query, relevant chunk_ids) pairs.

Usage:
    python evaluation/retrieval_eval.py --dataset evaluation/golden_dataset.json
"""

import argparse
import json

from src.db.repository import FilingRepository
from src.db.session import session_scope
from src.retrieval.embedding_service import EmbeddingService
from src.retrieval.hybrid_retriever import ChunkRecord, HybridRetriever
from src.retrieval.vector_store import ChromaVectorStore


def evaluate(dataset_path: str, top_k: int = 8) -> dict:
    with open(dataset_path) as f:
        dataset = json.load(f)

    retriever = HybridRetriever(EmbeddingService(), ChromaVectorStore())
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []

    with session_scope() as db:
        repo = FilingRepository(db)

        for case in dataset["retrieval_cases"]:
            relevant_ids = set(case["relevant_chunk_ids"])
            if not relevant_ids:
                continue

            persisted_chunks = repo.get_chunks(case["filing_id"])
            all_chunks = [
                ChunkRecord(chunk_id=c.id, text=c.text, page_number=c.page_number, filing_id=case["filing_id"])
                for c in persisted_chunks
            ]

            hits = retriever.retrieve(
                query=case["query"], filing_id=case["filing_id"], all_chunks=all_chunks, top_k=top_k
            )
            retrieved_ids = [h.chunk_id for h in hits]

            hit_count = len(relevant_ids & set(retrieved_ids))
            recalls.append(hit_count / len(relevant_ids))

            rank = next((i + 1 for i, cid in enumerate(retrieved_ids) if cid in relevant_ids), None)
            reciprocal_ranks.append(1 / rank if rank else 0.0)

    return {
        "recall_at_k": sum(recalls) / len(recalls) if recalls else None,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None,
        "cases_evaluated": len(recalls),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation/golden_dataset.json")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset, top_k=args.top_k), indent=2))