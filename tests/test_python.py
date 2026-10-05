import numpy as np
import pytest

import qann


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    return rng.random((5_000, 32), dtype=np.float32)


def test_version():
    assert isinstance(qann.__version__, str)


def test_flat_finds_self(data):
    index = qann.FlatIndex(32)
    index.add(data)
    assert index.size() == len(data)
    ids, dists = index.batch_query(data[:10], 5)
    assert ids.shape == (10, 5)
    np.testing.assert_array_equal(ids[:, 0], np.arange(10))
    np.testing.assert_allclose(dists[:, 0], 0, atol=1e-4)


def test_ivf_pq_refine_finds_self(data):
    ivf = qann.IVFIndex(32, nlist=100, nprobe=20)
    ivf.enable_pq(8)
    ivf.train(data, seed=1)
    refine = qann.RefineIndex(ivf, k_factor=10)
    refine.add(data)
    ids, _ = refine.batch_query(data[:50], 10)
    assert ids.shape == (50, 10)
    assert np.mean(ids[:, 0] == np.arange(50)) >= 0.9
