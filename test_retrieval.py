import numpy as np

from core.vector_store import VectorStore


def make_store(tmp_path):
    store = VectorStore(tmp_path, dim=4)
    vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype="float32")
    chunks = [
        {"chunk_id": "a|p1|c0", "doc_name": "a.pdf", "page": 1, "text": "alpha"},
        {"chunk_id": "b|p2|c0", "doc_name": "b.pdf", "page": 2, "text": "beta"},
        {"chunk_id": "a|p3|c0", "doc_name": "a.pdf", "page": 3, "text": "gamma"},
    ]
    store.add(vecs, chunks)
    return store


def test_search_returns_most_similar_first(tmp_path):
    store = make_store(tmp_path)
    hits = store.search(np.array([0.9, 0.1, 0, 0]), top_k=2)
    assert hits[0][0]["text"] == "alpha"
    assert len(hits) == 2


def test_document_filter_excludes_other_documents(tmp_path):
    store = make_store(tmp_path)
    hits = store.search(np.array([0, 1, 0, 0]), top_k=3, doc_names={"a.pdf"})
    assert hits and all(c["doc_name"] == "a.pdf" for c, _ in hits)


def test_remove_document_rebuilds_index(tmp_path):
    store = make_store(tmp_path)
    assert store.remove_document("a.pdf") == 2
    assert store.index.ntotal == 1
    hits = store.search(np.array([1, 0, 0, 0]), top_k=1)
    assert hits[0][0]["doc_name"] == "b.pdf"


def test_index_persists_to_disk(tmp_path):
    make_store(tmp_path)
    reloaded = VectorStore(tmp_path, dim=4)
    assert reloaded.index.ntotal == 3
    assert [c["text"] for c in reloaded.chunks] == ["alpha", "beta", "gamma"]


def test_list_documents_summarises_counts(tmp_path):
    store = make_store(tmp_path)
    docs = {d["name"]: d for d in store.list_documents()}
    assert docs["a.pdf"]["chunks"] == 2 and docs["a.pdf"]["pages"] == 2
    assert docs["b.pdf"]["chunks"] == 1
