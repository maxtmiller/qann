import gc
import pathlib
import threading

import numpy as np
import pytest

import qann

sp = pytest.importorskip("scipy.sparse")

N, DIM, VOCAB, NQ = 800, 16, 400, 30


@pytest.fixture(scope="module")
def docs():
    rng = np.random.default_rng(11)
    return {
        "dense": rng.random((N, DIM), dtype=np.float32),
        "sparse": sp.random(N, VOCAB, density=0.03, format="csr", dtype=np.float32, random_state=12),
        "q_dense": rng.random((NQ, DIM), dtype=np.float32),
        "q_sparse": sp.random(NQ, VOCAB, density=0.03, format="csr", dtype=np.float32, random_state=13),
        "ids": np.arange(N) * 5 + 100,
        "group": np.arange(N) % 4,
    }


def make(docs, dense=None, sparse=None):
    h = qann.HybridIndex(dense or qann.FlatIndex(DIM), sparse or qann.InvertedIndex(), custom_ids=True)
    h.add(docs["dense"], docs["sparse"], ids=docs["ids"], metadata={"group": docs["group"]})
    return h


def reference(docs, h, k, allowed=None):
    """Standalone indexes over the same documents, queried and fused by hand."""
    dense = qann.FlatIndex(DIM, custom_ids=True)
    sparse = qann.InvertedIndex(custom_ids=True)
    dense.add(docs["dense"], ids=docs["ids"])
    sparse.add(docs["sparse"], ids=docs["ids"])
    depth = min(k * h.depth_factor, N if allowed is None else len(allowed))
    d = dense.batch_query(docs["q_dense"], depth, allowed=allowed)
    s = sparse.batch_query(docs["q_sparse"], depth, allowed=allowed)
    return qann.fuse([d, s], method=h.method, k=h.rrf_k, weights=list(h.weights), limit=k)


@pytest.mark.parametrize("method", ["rrf", "linear"])
def test_matches_fusing_standalone_indexes(docs, method):
    h = make(docs)
    h.method = method
    h.weights = (0.6, 1.3)
    h.depth_factor = 3
    ids, scores = h.batch_query(docs["q_dense"], docs["q_sparse"], 10)
    want_ids, want_scores = reference(docs, h, 10)
    np.testing.assert_array_equal(ids, want_ids)
    np.testing.assert_array_equal(scores, want_scores)
    assert ids.dtype == np.int64 and scores.dtype == np.float32

    # query on one row matches that row of batch_query, without the padding
    one_ids, one_scores = h.query(docs["q_dense"][3], docs["q_sparse"][[3]], 10)
    keep = ids[3] >= 0
    np.testing.assert_array_equal(one_ids, ids[3][keep])
    np.testing.assert_array_equal(one_scores, scores[3][keep])


def test_filters_apply_to_both_sides(docs):
    h = make(docs)
    allowed = docs["ids"][docs["group"] == 2]
    want = reference(docs, h, 10, allowed=allowed)
    for got in (h.batch_query(docs["q_dense"], docs["q_sparse"], 10, where="group = 2"),
                h.batch_query(docs["q_dense"], docs["q_sparse"], 10, allowed=allowed),
                h.batch_query(docs["q_dense"], docs["q_sparse"], 10, where=qann.field("group") == 2, allowed=docs["ids"])):
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])
    ids, scores = h.batch_query(docs["q_dense"], docs["q_sparse"], 10, where="group = 9")
    assert (ids == -1).all() and (scores == 0).all()


def test_remove_compact_save_load(tmp_path, docs):
    base = qann.IVFIndex(DIM, nlist=100, nprobe=100)
    base.enable_pq(8)
    base.train(docs["dense"], seed=1)
    h = make(docs, dense=qann.RefineIndex(base, k_factor=8), sparse=qann.SparseApproxIndex())
    h.method, h.weights, h.rrf_k, h.depth_factor = "linear", (0.7, 0.3), 30.0, 4

    gone = docs["ids"][::5]
    assert h.remove(gone) == len(gone)
    assert h.size() == N - len(gone)
    before = h.batch_query(docs["q_dense"], docs["q_sparse"], 10)
    assert not np.isin(before[0], gone).any()

    assert h.compact() == len(gone)
    after = h.batch_query(docs["q_dense"], docs["q_sparse"], 10)
    np.testing.assert_array_equal(after[0], before[0])
    np.testing.assert_array_equal(after[1], before[1])
    np.testing.assert_array_equal(h.ids(), np.setdiff1d(docs["ids"], gone))

    h.save(tmp_path / "h.qann")
    loaded = qann.load(tmp_path / "h.qann")
    assert type(loaded) is qann.HybridIndex
    assert (loaded.method, loaded.rrf_k, loaded.depth_factor) == ("linear", 30.0, 4)
    assert loaded.weights == pytest.approx((0.7, 0.3))   # stored as float32
    assert type(loaded.dense) is qann.RefineIndex and type(loaded.sparse) is qann.SparseApproxIndex
    got = loaded.batch_query(docs["q_dense"], docs["q_sparse"], 10)
    np.testing.assert_array_equal(got[0], after[0])
    np.testing.assert_array_equal(got[1], after[1])
    np.testing.assert_array_equal(loaded.get_metadata(docs["ids"][1:3])["group"], [1, 2])
    loaded.dense.base.nprobe = 10        # the parts of a loaded index are still tunable
    loaded.sparse.list_depth = 100
    loaded.batch_query(docs["q_dense"], docs["q_sparse"], 10)


def test_parts_are_protected_while_wrapped(docs):
    dense, sparse = qann.FlatIndex(DIM), qann.InvertedIndex()
    h = qann.HybridIndex(dense, sparse)
    h.add(docs["dense"][:5], docs["sparse"][:5])
    with pytest.raises(ValueError, match="through the wrapper"):
        dense.add(docs["dense"][:1])
    with pytest.raises(ValueError, match="through the wrapper"):
        sparse.add(docs["sparse"][:1])
    with pytest.raises(ValueError, match="through the wrapper"):
        sparse.remove([0])
    with pytest.raises(ValueError):
        qann.HybridIndex(dense, qann.InvertedIndex())   # already wrapped
    with pytest.raises(ValueError):
        qann.RefineIndex(dense)

    # once the wrapper is gone, its (still empty) parts can be used on their own
    alone_dense, alone_sparse = qann.FlatIndex(DIM), qann.InvertedIndex()
    h2 = qann.HybridIndex(alone_dense, alone_sparse)
    del h2
    gc.collect()
    alone_dense.add(docs["dense"][:1])
    alone_sparse.add(docs["sparse"][:1])


def test_bad_input_adds_nothing(docs):
    h = qann.HybridIndex(qann.FlatIndex(DIM, metric=qann.Metric.Cosine), qann.InvertedIndex(), custom_ids=True)
    h.add(docs["dense"][:10], docs["sparse"][:10], ids=docs["ids"][:10])
    rows = docs["dense"][10:12].copy()
    zero = rows.copy()
    zero[1] = 0
    cases = [
        lambda: h.add(zero, docs["sparse"][10:12], ids=[1, 2]),                          # zero row under Cosine
        lambda: h.add(rows, docs["sparse"][10:13], ids=[1, 2]),                          # row counts differ
        lambda: h.add(rows, docs["sparse"][10:12], ids=[1, int(docs["ids"][0])]),       # id taken
        lambda: h.add(rows, docs["sparse"][10:12], ids=[1, 2], metadata={"g": [1]}),    # metadata length
        lambda: h.add(rows[:, :8].copy(), docs["sparse"][10:12], ids=[1, 2]),          # dimension
    ]
    for bad in cases:
        with pytest.raises(ValueError):
            bad()
        assert h.size() == 10 and h.dense.size() == 10 and h.sparse.size() == 10
    with pytest.raises(TypeError):
        h.add(rows, rows, ids=[1, 2])                                                   # sparse must be scipy.sparse


def test_settings_and_query_errors(docs):
    h = make(docs)
    assert (h.method, h.weights, h.rrf_k, h.depth_factor) == ("rrf", (1.0, 1.0), 60.0, 5)
    for setting, value in (("method", "borda"), ("weights", (0, 0)), ("weights", (-1, 1)), ("rrf_k", 0), ("depth_factor", 0)):
        with pytest.raises(ValueError):
            setattr(h, setting, value)
    with pytest.raises(ValueError):
        h.batch_query(docs["q_dense"], docs["q_sparse"][:5], 10)        # row counts differ
    with pytest.raises(ValueError):
        h.query(docs["q_dense"][0], docs["q_sparse"][:2], 10)           # two sparse rows
    with pytest.raises(ValueError):
        h.query(docs["q_dense"][0][:8].copy(), docs["q_sparse"][[0]], 10)
    assert isinstance(h, qann.IndexBase)
    assert not isinstance(h, (qann.DenseIndex, qann.SparseIndex))
    assert h.dim() == DIM


def test_concurrent_adds_queries_and_tuning(docs):
    dense = qann.HNSWIndex(DIM, M=8, ef_construction=40)
    h = qann.HybridIndex(dense, qann.InvertedIndex())
    h.add(docs["dense"][:200], docs["sparse"][:200])
    errors = []

    def run(fn):
        try:
            fn()
        except Exception as e:  # surfaced below; a thread's exception is otherwise lost
            errors.append(e)

    def adder():
        for start in range(200, N, 50):
            h.add(docs["dense"][start:start + 50], docs["sparse"][start:start + 50])

    def querier():
        for _ in range(15):
            ids, _ = h.batch_query(docs["q_dense"], docs["q_sparse"], 10)
            assert ids.max() < N

    def tuner():
        for i in range(100):
            h.weights = (1.0, 0.5 + i % 3)
            h.method = "rrf" if i % 2 else "linear"
            h.dense.ef_search = 16 + i % 32

    threads = [threading.Thread(target=run, args=(f,)) for f in (adder, querier, querier, tuner)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert h.size() == N and h.dense.size() == N and h.sparse.size() == N


def test_loads_file_saved_on_macos():
    # Saved on macOS (format version 9): HNSW (cosine, M 8) + InvertedIndex, 400
    # documents with custom ids 50, 57, ..., every tenth deleted (not compacted),
    # a str column from the first add and an int column from the second (nulls
    # before it), and fusion settings linear, (0.75, 0.5), rrf_k 40, depth 3.
    data = pathlib.Path(__file__).parent / "data"
    h = qann.load(data / "v9_hybrid.qann")
    assert type(h) is qann.HybridIndex
    assert type(h.dense) is qann.HNSWIndex and type(h.sparse) is qann.InvertedIndex
    assert h.dense.metric == qann.Metric.Cosine
    assert (h.method, h.rrf_k, h.depth_factor) == ("linear", 40.0, 3)
    assert h.weights == pytest.approx((0.75, 0.5))
    assert h.size() == 360 and h.metadata_schema == {"lang": "str", "year": "int"}
    md = h.get_metadata([57, 50 + 7 * 351])
    assert md["lang"] == ["en", "en"] and md["year"].mask.tolist() == [True, False]

    qd = np.load(data / "v9_hybrid_q_dense.npy")
    qs = sp.load_npz(data / "v9_hybrid_q_sparse.npz")
    ids, scores = h.batch_query(qd, qs, 10, where="lang = 'en'")
    # SIMD kernels round differently across CPUs, which can swap near-ties
    assert np.mean(ids == np.load(data / "v9_hybrid_ids.npy")) >= 0.98
    np.testing.assert_allclose(scores, np.load(data / "v9_hybrid_scores.npy"), rtol=1e-4, atol=1e-5)

    h.compact()
    assert h.size() == 360 and h.dense.size() == 360 and h.sparse.size() == 360
