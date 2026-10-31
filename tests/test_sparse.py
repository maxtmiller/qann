import pathlib
import threading

import numpy as np
import pytest

import qann

sp = pytest.importorskip("scipy.sparse")

DATA = pathlib.Path(__file__).parent / "data"


def _random_csr(rng, n, vocab, nnz):
    # low term ids are common, like stopwords; integer weights make every sum exact
    rows = np.repeat(np.arange(n), nnz)
    cols = (rng.random(n * nnz) ** 3 * vocab).astype(np.int64)
    vals = rng.integers(1, 6, n * nnz).astype(np.float32)
    m = sp.coo_array((vals, (rows, cols)), shape=(n, vocab)).tocsr()
    m.sum_duplicates()
    return m


def _check_top_k(ids, dists, scores, shares, k):
    # dists are the k best -scores among docs sharing a term; each id truly has its score
    candidates = np.flatnonzero(shares)
    best = np.sort(-scores[candidates])[:k]
    got = ids[ids >= 0]
    assert len(got) == len(best)
    assert np.array_equal(dists[: len(got)], best.astype(np.float32))
    assert np.array_equal(-scores[got], dists[: len(got)])
    assert len(set(got.tolist())) == len(got)


def test_sparse_matches_brute_force():
    rng = np.random.default_rng(3)
    docs = _random_csr(rng, 3000, 400, 12)
    queries = _random_csr(rng, 50, 400, 5)
    index = qann.SparseIndex()
    assert index.add(docs).tolist() == list(range(3000))
    assert index.size() == 3000 and index.postings == docs.nnz

    scores = (queries @ docs.T).toarray()
    shares = ((queries != 0).astype(np.int64) @ (docs != 0).astype(np.int64).T).toarray() > 0
    for k in (1, 10, 4000):
        ids, dists = index.batch_query(queries, k)
        assert ids.shape == (50, k)
        for i in range(50):
            _check_top_k(ids[i], dists[i], scores[i], shares[i], k)
    one_ids, one_dists = index.query(queries[[7]], 10)
    _check_top_k(one_ids, one_dists, scores[7], shares[7], 10)


def test_sparse_input_formats():
    index = qann.SparseIndex()
    dense = np.array([[1, 0, 2], [0, 3, 0]], dtype=np.float64)
    index.add(sp.csr_matrix(dense))
    index.add(sp.csc_array(dense))
    # repeated entries in a row are summed, explicit zeros dropped
    dup = sp.coo_array((np.array([1.0, 1.0, 0.0]), (np.array([0, 0, 0]), np.array([2, 2, 1]))), shape=(1, 3))
    assert index.add(dup).tolist() == [4]
    assert index.terms == 3 and index.postings == 7

    ids, dists = index.query(sp.csr_array(np.array([[0, 0, 1]])), 10)
    assert ids.tolist() == [0, 2, 4] and dists.tolist() == [-2, -2, -2]
    assert index.query(sp.csr_array((1, 3)), 10)[0].size == 0    # an empty row matches nothing
    ids, dists = index.batch_query(sp.csr_array(np.array([[0, 1, 0], [0, 0, 0]])), 3)
    assert ids.tolist() == [[1, 3, -1], [-1, -1, -1]]
    assert dists[0].tolist() == [-3, -3, np.inf]

    with pytest.raises(TypeError, match="scipy.sparse"):
        index.add(dense)
    with pytest.raises(TypeError, match="scipy.sparse"):
        index.query([[1, 0, 0]], 1)
    with pytest.raises(ValueError, match="one row"):
        index.query(sp.csr_array(dense), 1)
    huge = sp.csr_array((np.array([1.0]), np.array([2**32], dtype=np.int64), np.array([0, 1])), shape=(1, 2**33))
    with pytest.raises(ValueError, match="outside"):
        index.add(huge)
    assert index.size() == 5


def test_sparse_ids_filters_metadata_and_save(tmp_path):
    rng = np.random.default_rng(5)
    docs = _random_csr(rng, 200, 60, 8)
    queries = _random_csr(rng, 10, 60, 4)
    ids = np.arange(200) * 3 + 1000
    index = qann.SparseIndex(custom_ids=True)
    index.add(docs, ids, metadata={"odd": np.arange(200) % 2, "tag": [f"t{i % 5}" for i in range(200)]})
    with pytest.raises(ValueError):
        index.add(docs[[0]])    # needs ids
    assert isinstance(index, qann.IndexBase) and not isinstance(index, qann.Index)

    index.remove(ids[::7])
    scores = (queries @ docs.T).toarray()
    shares = ((queries != 0).astype(np.int64) @ (docs != 0).astype(np.int64).T).toarray() > 0
    live = np.ones(200, dtype=bool)
    live[::7] = False

    def check(idx):
        cases = [({}, live),
                 ({"allowed": ids[::3]}, live & (np.arange(200) % 3 == 0)),
                 ({"where": "odd = 1"}, live & (np.arange(200) % 2 == 1)),
                 ({"where": qann.field("tag") == "t2", "allowed": ids[:100]}, live & (np.arange(200) % 5 == 2) & (np.arange(200) < 100))]
        for kwargs, keep in cases:
            got, dists = idx.batch_query(queries, 10, **kwargs)
            for i in range(10):
                slots = np.where(got[i] >= 0, (got[i] - 1000) // 3, -1)
                _check_top_k(slots, dists[i], scores[i], shares[i] & keep, 10)

    check(index)
    assert index.batch_query(queries, 5, allowed=[])[0].max() == -1
    with pytest.raises(ValueError, match="unknown metadata field"):
        index.query(queries[[0]], 5, where="missing = 1")

    before = index.postings
    assert index.compact() == len(ids[::7])
    assert index.postings < before
    check(index)
    assert index.get_metadata([ids[1]])["tag"] == ["t1"]

    path = tmp_path / "sparse.qann"
    index.save(path)
    loaded = qann.load(path)
    assert type(loaded) is qann.SparseIndex
    assert loaded.size() == index.size() and loaded.postings == index.postings
    check(loaded)


def test_sparse_metadata_update_add_and_drop():
    index = qann.SparseIndex(custom_ids=True)
    docs = sp.csr_array(np.array([[1, 0, 2], [0, 3, 1], [2, 1, 0], [1, 1, 1]], dtype=np.float32))
    index.add(docs, [10, 20, 30, 40], metadata={"price": [5, 15, None, 25]})
    q = sp.csr_array(np.array([[1, 1, 1]], dtype=np.float32))

    def match(where):
        return sorted(index.query(q, 10, where=where)[0].tolist())

    index.update_metadata([30, 10], {"price": [50, None]})
    assert match("price > 20") == [30, 40]
    assert match(qann.field("price").is_null()) == [10]
    index.add_metadata_column("tag", "str", ids=[20, 40], values=["x", None])
    assert index.metadata_schema == {"price": "int", "tag": "str"}
    assert match("tag = 'x'") == [20]
    assert index.get_metadata([40, 20], columns=["tag"]) == {"tag": [None, "x"]}
    index.add_metadata_column("rank", "float", default=0.5)
    assert match("rank = 0.5") == [10, 20, 30, 40]
    index.drop_metadata_column("rank")
    with pytest.raises(ValueError, match="unknown metadata field"):
        match("rank = 0.5")
    with pytest.raises(ValueError, match="id 99 is not in the index"):
        index.update_metadata([99], {"price": [1]})
    index.remove([20])
    index.compact()
    assert match("tag IS NULL") == [10, 30, 40]


def test_sparse_concurrent_adds_removes_and_filtered_queries():
    # adds and removes (exclusive lock) interleaved with filtered queries
    # (shared): every result must match the filter and be live at some point
    rng = np.random.default_rng(8)
    index = qann.SparseIndex()
    first = _random_csr(rng, 500, 100, 8)
    index.add(first, metadata={"group": np.arange(500) % 5})
    batches = [_random_csr(rng, 100, 100, 8) for _ in range(15)]
    queries = _random_csr(rng, 50, 100, 4)
    errors = []

    def run(fn):
        try:
            fn()
        except Exception as e:
            errors.append(e)

    def writer():
        for b, batch in enumerate(batches):
            start = 500 + 100 * b
            index.add(batch, metadata={"group": np.arange(start, start + 100) % 5})
            index.remove(np.arange(start - 100, start, 10))
            if b % 5 == 4:
                index.compact()

    def querier():
        for _ in range(30):
            got, _ = index.batch_query(queries, 10, where="group = 3")
            got = got[got >= 0]
            assert (got % 5 == 3).all()

    threads = [threading.Thread(target=run, args=(fn,)) for fn in (writer, querier, querier)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert index.size() == 500 + 100 * len(batches) - 10 * len(batches)
    got, _ = index.batch_query(queries, 10, where="group = 3")
    got = got[got >= 0]
    assert (got % 5 == 3).all() and not np.isin(got, np.arange(400, 1900, 10)).any()


def test_sparse_loads_file_saved_on_macos():
    # Saved on macOS (format version 9): 1000 docs over 300 term ids spread up
    # to 100,000, custom ids 100, 103, ..., every ninth deleted (not
    # compacted), int and string columns with nulls and a float column added
    # by the second of two adds, a forward index and approximate settings.
    index = qann.load(DATA / "v9_sparse.qann")
    assert type(index) is qann.SparseIndex
    assert index.size() == 888 and index.terms == 297 and index.postings == 11200
    assert index.forward_index and (index.query_terms, index.list_depth, index.k_factor) == (3, 40, 4)
    assert index.metadata_schema == {"price": "int", "tag": "str", "score": "float"}
    md = index.get_metadata([100 + 3 * 143, 100 + 3 * 601])    # slot 143: price and tag null
    assert md["price"].mask.tolist() == [True, False] and md["tag"] == [None, "t1"]
    assert md["score"].mask.tolist() == [True, False]

    queries = sp.load_npz(DATA / "v9_sparse_queries.npz")
    where = "price < 30 AND tag != 't0'"
    for name in ("v9_sparse_approx", "v9_sparse"):
        ids, dists = index.batch_query(queries, 10, where=where)
        np.testing.assert_array_equal(ids, np.load(DATA / f"{name}_ids.npy"))
        np.testing.assert_allclose(dists, np.load(DATA / f"{name}_dists.npy"), rtol=1e-5, atol=1e-5)
        index.query_terms, index.list_depth = 0, 0    # then exact

    # compacting keeps the exact results
    index.compact()
    assert index.size() == 888
    again, _ = index.batch_query(queries, 10, where=where)
    np.testing.assert_array_equal(again, ids)


def test_sparse_pruning_switch(tmp_path):
    rng = np.random.default_rng(12)
    docs = _random_csr(rng, 2000, 300, 20)
    queries = _random_csr(rng, 30, 300, 6)
    index = qann.SparseIndex()
    index.add(docs)
    assert index.pruning
    pruned = index.batch_query(queries, 10)
    index.pruning = False
    full = index.batch_query(queries, 10)
    np.testing.assert_array_equal(pruned[1], full[1])
    scores = (queries @ docs.T).toarray()
    for i in range(30):
        assert np.array_equal(-scores[i][pruned[0][i]], pruned[1][i])

    index.save(tmp_path / "p.qann")
    assert qann.load(tmp_path / "p.qann").pruning    # not saved: loads with the default


def test_sparse_approximate_search(tmp_path):
    rng = np.random.default_rng(21)
    docs = _random_csr(rng, 3000, 400, 20).astype(np.float32)
    docs.data *= rng.random(docs.nnz).astype(np.float32)    # spread weights, as learned sparse vectors have
    queries = _random_csr(rng, 50, 400, 12).astype(np.float32)
    index = qann.SparseIndex(forward_index=True)
    index.add(docs)
    assert index.forward_index and (index.query_terms, index.list_depth, index.k_factor) == (0, 0, 10)
    exact_ids, exact_d = index.batch_query(queries, 10)

    index.query_terms, index.list_depth, index.k_factor = 4, 200, 5
    ids, dists = index.batch_query(queries, 10)
    scores = (queries @ docs.T).toarray()
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, exact_ids)])
    assert recall > 0.5
    for i in range(50):    # every result's distance is its exact one
        got = ids[i][ids[i] >= 0]
        np.testing.assert_allclose(dists[i][: len(got)], -scores[i][got], rtol=1e-5, atol=1e-5)
    one_ids, _ = index.query(queries[[3]], 10)
    np.testing.assert_array_equal(one_ids, ids[3][ids[3] >= 0])

    index.save(tmp_path / "approx.qann")
    loaded = qann.load(tmp_path / "approx.qann")
    assert loaded.forward_index and (loaded.query_terms, loaded.list_depth, loaded.k_factor) == (4, 200, 5)
    np.testing.assert_array_equal(loaded.batch_query(queries, 10)[0], ids)

    plain = qann.SparseIndex()
    with pytest.raises(ValueError, match="forward index"):
        plain.list_depth = 100
    with pytest.raises(ValueError, match="forward index"):
        plain.query_terms = 5
    with pytest.raises(ValueError, match="at most 16384"):
        index.list_depth = 16385
    with pytest.raises(ValueError, match="at least 1"):
        index.k_factor = 0
