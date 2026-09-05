from src.retrieval.bm25_index import BM25Document, Bm25Index


def test_bm25_ranks_exact_keyword_match_highest():
    documents = [
        BM25Document(chunk_id="1", text="The company reported total revenue of $47.2 million in fiscal 2023."),
        BM25Document(chunk_id="2", text="Management discusses general market conditions and competition."),
        BM25Document(chunk_id="3", text="Employee headcount increased by twelve percent year over year."),
    ]
    index = Bm25Index(documents)

    results = index.query("total revenue $47.2 million", top_k=2)

    assert results[0]["id"] == "1"
    assert results[0]["score"] == 1.0


def test_bm25_empty_index_returns_no_hits():
    index = Bm25Index([])
    assert index.query("anything", top_k=5) == []