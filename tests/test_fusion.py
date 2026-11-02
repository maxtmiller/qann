import numpy as np
import pytest

import qann


def test_fuse_one_query():
    ids, scores = qann.fuse([np.array([3, 7, 9]), np.array([7, 4, 9])], limit=5)
    np.testing.assert_array_equal(ids, [7, 9, 3, 4, -1])
    np.testing.assert_allclose(scores, [1 / 62 + 1 / 61, 2 / 63, 1 / 61, 1 / 62, 0], rtol=1e-6)
    assert ids.dtype == np.int64 and scores.dtype == np.float32


def test_fuse_accepts_query_results_and_plain_ids():
    a, b = np.array([3, 7, 9]), np.array([7, 4, 9])
    expected = qann.fuse([a, b])
    dists = np.zeros(3, dtype=np.float32)
    for results in ([(a, dists), (b, dists)], [a.tolist(), (b, dists)], [tuple(a), list(b)],
                    [a.astype(np.int32), b.astype(np.uint16)]):
        got = qann.fuse(results)
        np.testing.assert_array_equal(got[0], expected[0])
        np.testing.assert_array_equal(got[1], expected[1])


def test_fuse_batch_rows_match_single_queries():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 40, size=(50, 20))
    b = rng.integers(0, 40, size=(50, 8))
    b[:, -2:] = -1   # padding, as batch_query returns it
    ids, scores = qann.fuse([(a, np.zeros(a.shape, np.float32)), b], weights=[1.0, 2.0], limit=7)
    assert ids.shape == scores.shape == (50, 7)
    for q in range(50):
        one_ids, one_scores = qann.fuse([a[q], b[q]], weights=[1.0, 2.0], limit=7)
        np.testing.assert_array_equal(ids[q], one_ids)
        np.testing.assert_array_equal(scores[q], one_scores)


def test_fuse_k_weights_and_ties():
    ids, scores = qann.fuse([[1, 2]], k=1.0)
    np.testing.assert_allclose(scores[:2], [1 / 2, 1 / 3], rtol=1e-6)
    np.testing.assert_array_equal(qann.fuse([[1], [2]], weights=[1, 3])[0][:2], [2, 1])
    np.testing.assert_array_equal(qann.fuse([[1, 2], [3, 4]], weights=[1, 0])[0][:3], [1, 2, -1])
    np.testing.assert_array_equal(qann.fuse([[9, 2], [2, 9]])[0][:2], [2, 9])   # tie: smaller id first


def test_fuse_errors():
    with pytest.raises(ValueError):
        qann.fuse([])
    with pytest.raises(ValueError):
        qann.fuse([[1]], limit=0)
    with pytest.raises(ValueError):
        qann.fuse([[1], [2]], weights=[1.0])
    with pytest.raises(ValueError):
        qann.fuse([[1]], weights=[-1.0])
    with pytest.raises(ValueError):
        qann.fuse([[1]], k=0)
    with pytest.raises(ValueError):
        qann.fuse([np.zeros((2, 3), np.int64), np.zeros((3, 3), np.int64)])   # different row counts
    with pytest.raises(ValueError):
        qann.fuse([np.zeros(3, np.int64), np.zeros((1, 3), np.int64)])        # 1D and 2D
    with pytest.raises(TypeError):
        qann.fuse([np.array([1.5, 2.0])])                                     # ids must be integers


def test_hybrid_dense_and_sparse_with_shared_ids():
    sp = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(2)
    n, doc_ids = 500, np.arange(500) * 7 + 100
    dense = qann.FlatIndex(16, custom_ids=True)
    dense.add(rng.random((n, 16), dtype=np.float32), ids=doc_ids)
    sparse = qann.InvertedIndex(custom_ids=True)
    sparse.add(sp.random(n, 300, density=0.05, format="csr", dtype=np.float32, random_state=3), ids=doc_ids)
    np.testing.assert_array_equal(dense.ids(), sparse.ids())

    q_dense = rng.random((20, 16), dtype=np.float32)
    q_sparse = sp.random(20, 300, density=0.05, format="csr", dtype=np.float32, random_state=4)
    d = dense.batch_query(q_dense, 50)
    s = sparse.batch_query(q_sparse, 50)
    ids, scores = qann.fuse([d, s], limit=10)
    assert ids.shape == (20, 10)
    assert np.isin(ids[ids >= 0], doc_ids).all()
    assert (np.diff(scores, axis=1) <= 0).all()   # best first

    # the same as a plain-Python RRF over each row
    for q in range(20):
        ref = {}
        for row in (d[0][q], s[0][q]):
            for rank, i in enumerate(row[row >= 0], start=1):
                ref[int(i)] = ref.get(int(i), 0.0) + 1 / (60 + rank)
        best = sorted(ref.items(), key=lambda t: (-t[1], t[0]))[:10]
        np.testing.assert_array_equal(ids[q][:len(best)], [i for i, _ in best])
        np.testing.assert_allclose(scores[q][:len(best)], [v for _, v in best], rtol=1e-6)
