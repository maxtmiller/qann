import collections
import zlib

import numpy as np
import pytest
import qann
import scipy.sparse as sp
from langchain_core.embeddings import Embeddings

from langchain_qann import BM25, QANNVectorStore

DIM = 16
WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()


class HashEmbeddings(Embeddings):
    def _one(self, text):
        v = np.random.default_rng(zlib.crc32(text.encode())).standard_normal(DIM)
        return (v / np.linalg.norm(v)).tolist()

    def embed_documents(self, texts):
        return [self._one(t) for t in texts]

    def embed_query(self, text):
        return self._one(text)


@pytest.fixture
def emb():
    return HashEmbeddings()


def corpus(n, seed=0):
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(WORDS, rng.integers(3, 12))) + f" unique{i}" for i in range(n)]


def stats(bm25):
    return bm25.num_documents, bm25.num_terms, bm25.total_length


def test_bm25_statistics_follow_adds_and_deletes(emb):
    texts = corpus(200)
    store = QANNVectorStore.from_texts(texts, emb, ids=[str(i) for i in range(200)], sparse="bm25")
    store.add_texts(corpus(50, seed=1), ids=[f"b{i}" for i in range(50)])
    store.delete([str(i) for i in range(0, 200, 3)])
    store.add_texts(["replaced text alpha"], ids=["1"])
    remaining = [d.page_content for d in store.get_by_ids(list(store._rows))]
    fresh = BM25()
    fresh.rebuild(remaining)
    assert stats(store._sparse) == stats(fresh)


def test_bm25_matches_batch_scoring():
    texts = corpus(300)
    bm = BM25()
    D = bm.encode_documents(texts)
    q = bm.encode_query("alpha kilo papa")
    qw = dict(zip(q.indices, q.data))
    scores = [sum(v * qw.get(c, 0.0) for c, v in zip(D.indices[D.indptr[i]:D.indptr[i + 1]],
                                                    D.data[D.indptr[i]:D.indptr[i + 1]])) for i in range(D.shape[0])]
    # reference: Lucene BM25 with the IDF on the document side
    toks = [collections.Counter(BM25.terms(t)) for t in texts]
    avgdl = np.mean([sum(c.values()) for c in toks])
    df = collections.Counter(t for c in toks for t in c)
    ref = []
    for c in toks:
        dl = sum(c.values())
        s = 0.0
        for term in set(BM25.terms("alpha kilo papa")):
            tf = c.get(term, 0)
            idf = np.log(1 + (len(texts) - df[term] + 0.5) / (df[term] + 0.5))
            s += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * dl / avgdl))
        ref.append(s)
    assert np.allclose(scores, ref, rtol=1e-5)


def test_keyword_match_needs_the_sparse_side(emb):
    texts = corpus(300)
    store = QANNVectorStore.from_texts(texts, emb, ids=[str(i) for i in range(300)], sparse="bm25")
    # random embeddings can't find "unique123"; BM25 can
    assert store.similarity_search("unique123", k=1, weights=(0, 1))[0].id == "123"
    assert store.similarity_search("unique123", k=1)[0].id == "123"
    assert store.index.weights == (1.0, 1.0)   # per-query weights are restored


def test_hybrid_filters_scores_and_vector_search(emb):
    texts = corpus(300)
    metas = [{"year": 2000 + i % 10} for i in range(300)]
    store = QANNVectorStore.from_texts(texts, emb, metas, sparse="bm25", index="flat")
    docs = store.similarity_search("alpha bravo", k=20, filter="year >= 2008")
    assert len(docs) == 20 and all(d.metadata["year"] >= 2008 for d in docs)
    scored = store.similarity_search_with_relevance_scores("alpha bravo", k=5)
    assert all(0 <= s <= 1 + 1e-6 for _, s in scored)
    # by vector there is no text, so the dense side alone ranks; exact search finds the text itself
    assert store.similarity_search_by_vector(emb.embed_query(texts[7]), k=1)[0].page_content == texts[7]
    assert len(store.max_marginal_relevance_search("alpha", k=3, fetch_k=10)) == 3


def test_tune_hybrid_weights(emb):
    texts = corpus(300)
    store = QANNVectorStore.from_texts(texts, emb, ids=[str(i) for i in range(300)], sparse="bm25")
    queries = [f"unique{i}" for i in range(0, 300, 10)]
    relevant = [{str(i)} for i in range(0, 300, 10)]
    table = store.tune_hybrid_weights(queries, relevant, k=10)
    assert len(table) > 1 and store.index.weights[1] > 0
    assert max(table.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("mmap", [False, True])
def test_hybrid_save_and_load(emb, tmp_path, mmap):
    texts = corpus(1200)
    store = QANNVectorStore.from_texts(texts, emb, ids=[str(i) for i in range(1200)], index="ivf-pq", sparse="bm25")
    store.delete(["5"])
    store.index.weights = (1.0, 0.5)
    path = tmp_path / "hybrid.qann"
    store.save(path)
    loaded = QANNVectorStore.load(path, emb, mmap=mmap)
    assert isinstance(loaded.index, qann.HybridIndex) and loaded.index.weights == (1.0, 0.5)
    assert stats(loaded._sparse) == stats(store._sparse)
    for q in ["alpha kilo", "unique77", texts[3]]:
        assert [d.id for d in loaded.similarity_search(q, k=5)] == [d.id for d in store.similarity_search(q, k=5)]
    loaded.add_texts(["zulu newword"], ids=["new"])
    assert loaded.similarity_search("newword", k=1, weights=(0, 1))[0].id == "new"


class Keywords:
    """A toy custom encoder: one term per distinct word, weight 1."""

    def _row(self, text):
        terms = sorted({zlib.crc32(w.encode()) & (2**31 - 1) for w in text.split()})
        return sp.csr_matrix((np.ones(len(terms), np.float32), ([0] * len(terms), terms)), shape=(1, 2**31))

    def encode_documents(self, texts):
        return sp.vstack([self._row(t) for t in texts]).tocsr()

    def encode_query(self, text):
        return self._row(text)


def test_custom_sparse_encoder(emb, tmp_path):
    texts = corpus(200)
    store = QANNVectorStore.from_texts(texts, emb, ids=[str(i) for i in range(200)], sparse=Keywords())
    assert store.similarity_search("unique42", k=1, weights=(0, 1))[0].id == "42"
    store.save(tmp_path / "k.qann")
    loaded = QANNVectorStore.load(tmp_path / "k.qann", emb, sparse=Keywords())
    assert loaded.similarity_search("unique42", k=1, weights=(0, 1))[0].id == "42"


def test_wrapping_and_errors(emb):
    store = QANNVectorStore(emb, index=qann.HNSWIndex(DIM, metric=qann.Metric.Cosine), sparse="bm25")
    store.add_texts(["alpha", "bravo"], ids=["a", "b"])
    assert isinstance(store.index, qann.HybridIndex)
    with pytest.raises(ValueError, match="sparse must be"):
        QANNVectorStore(emb, sparse="splade")
    with pytest.raises(ValueError, match="needs hybrid search"):
        QANNVectorStore.from_texts(["x"], emb).tune_hybrid_weights(["x"], [{"x"}])
    with pytest.raises(ValueError, match="no search setting"):
        QANNVectorStore.from_texts(["x"], emb).similarity_search("x", weights=(1, 0))
