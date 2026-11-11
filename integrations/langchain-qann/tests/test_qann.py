import zlib

import numpy as np
import pytest
import qann
from langchain_core.embeddings import Embeddings

from langchain_qann import QANNVectorStore

DIM = 16


class HashEmbeddings(Embeddings):
    """Deterministic random unit vectors per text."""

    def _one(self, text: str) -> list[float]:
        v = np.random.default_rng(zlib.crc32(text.encode())).standard_normal(DIM)
        return (v / np.linalg.norm(v)).tolist()

    def embed_documents(self, texts):
        return [self._one(t) for t in texts]

    def embed_query(self, text):
        return self._one(text)


@pytest.fixture
def emb():
    return HashEmbeddings()


def corpus(n):
    texts = [f"doc {i}" for i in range(n)]
    metas = [{"year": 2000 + i % 25, "source": ["10-K", "10-Q", "8-K"][i % 3], "tags": ["a", i]} for i in range(n)]
    return texts, metas


@pytest.mark.parametrize("preset", ["hnsw", "flat", "ivf-pq"])
def test_presets_find_their_own_text(emb, preset):
    texts, metas = corpus(1500)
    store = QANNVectorStore.from_texts(texts, emb, metas, index=preset)
    hits = sum(store.similarity_search(t, k=1)[0].page_content == t for t in texts[:100])
    assert hits >= 95


def test_ivf_preset_needs_a_corpus(emb):
    with pytest.raises(ValueError, match="at least 1000"):
        QANNVectorStore.from_texts(["a", "b"], emb, index="ivf-pq")


def test_filters(emb):
    texts, metas = corpus(300)
    store = QANNVectorStore.from_texts(texts, emb, metas, ids=[str(i) for i in range(300)])
    q = "doc 7"

    def check(filter, pred):
        docs = store.similarity_search(q, k=300, filter=filter)
        assert docs and all(pred(d.metadata) for d in docs)
        assert len(docs) == sum(pred(m) for m in metas)

    check({"source": "10-K"}, lambda m: m["source"] == "10-K")
    check({"source": "10-Q", "year": [2001, 2004]}, lambda m: m["source"] == "10-Q" and m["year"] in (2001, 2004))
    check("year >= 2020 AND source = '8-K'", lambda m: m["year"] >= 2020 and m["source"] == "8-K")
    check(("year < ?", [2003]), lambda m: m["year"] < 2003)
    check(qann.field("year").between(2010, 2011), lambda m: 2010 <= m["year"] <= 2011)
    with pytest.raises(ValueError, match="unknown metadata field"):
        store.similarity_search(q, filter={"nope": 1})


def test_metadata_round_trips_and_type_conflicts(emb):
    store = QANNVectorStore(emb)
    store.add_texts(["a", "b", "c"], [{"n": 1, "nested": {"x": [1, 2]}}, {"n": "one"}, {"n": 2.5, "flag": True}], ids=["a", "b", "c"])
    docs = {d.id: d for d in store.get_by_ids(["a", "b", "c"])}
    assert docs["a"].metadata == {"n": 1, "nested": {"x": [1, 2]}}
    assert docs["b"].metadata == {"n": "one"}
    # "n" became a number column; "one" is kept on the document but isn't filterable
    assert {d.id for d in store.similarity_search("a", k=3, filter="n >= 1")} == {"a", "c"}
    assert [d.id for d in store.similarity_search("a", k=3, filter={"flag": True})] == ["c"]


def test_search_settings(emb):
    texts, metas = corpus(1500)
    store = QANNVectorStore.from_texts(texts, emb, metas, index="ivf-pq", search_params={"nprobe": 4})
    base = store.index.base
    assert base.nprobe != 4   # applied per query, not left set
    store.similarity_search("doc 1", k=5, nprobe=64, k_factor=8)
    assert base.nprobe != 64 and store.index.k_factor == 4
    with pytest.raises(ValueError, match="unknown search setting"):
        store.similarity_search("doc 1", k=5, nprob=8)
    with pytest.raises(ValueError, match="no search setting"):
        QANNVectorStore.from_texts(texts[:10], emb).similarity_search("doc 1", nprobe=8)


@pytest.mark.parametrize("mmap", [False, True])
def test_save_and_load(emb, tmp_path, mmap):
    texts, metas = corpus(1200)
    store = QANNVectorStore.from_texts(texts, emb, metas, ids=[f"id{i}" for i in range(1200)], index="ivf-pq")
    store.delete(["id3"])
    path = tmp_path / "store.qann"
    store.save(path)
    loaded = QANNVectorStore.load(path, emb, mmap=mmap)
    assert loaded.index.mapped == mmap
    assert loaded.get_by_ids(["id3"]) == []
    assert loaded.get_by_ids(["id5"]) == store.get_by_ids(["id5"])
    q = "doc 42"
    assert [d.id for d in loaded.similarity_search(q, k=5)] == [d.id for d in store.similarity_search(q, k=5)]
    loaded.add_texts(["new"], [{"year": 1999}], ids=["new"])
    assert loaded.similarity_search("new", k=1, filter={"year": 1999})[0].id == "new"


def test_own_index_and_scores(emb):
    store = QANNVectorStore(emb, index=qann.HNSWIndex(DIM, M=8, metric=qann.Metric.L2))
    store.add_texts(["x", "y"], ids=["x", "y"])
    (doc, dist), = store.similarity_search_with_score("x", k=1)
    assert doc.id == "x" and dist == pytest.approx(0, abs=1e-5)
    (doc, rel), = store.similarity_search_with_relevance_scores("x", k=1)
    assert rel == pytest.approx(1, abs=1e-3)
    foreign = qann.FlatIndex(DIM)
    foreign.add(np.ones((1, DIM), dtype=np.float32))
    with pytest.raises(ValueError, match="outside a QANNVectorStore"):
        QANNVectorStore(emb, index=foreign)


def test_mmr(emb):
    store = QANNVectorStore.from_texts(["a", "a", "b"], emb, ids=["1", "2", "3"])
    assert {d.id for d in store.max_marginal_relevance_search("a", k=2, fetch_k=3, lambda_mult=0.1)} >= {"3"}
