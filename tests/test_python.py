import threading

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


def test_query_takes_1d_vector(data):
    index = qann.FlatIndex(32)
    index.add(data)
    ids, dists = index.query(data[3], 5)
    assert ids.shape == (5,) and dists.shape == (5,)
    assert ids[0] == 3
    with pytest.raises(TypeError):
        index.query(data[3:4], 5)


def test_ivf_pq_refine_finds_self(data):
    ivf = qann.IVFIndex(32, nlist=100, nprobe=20)
    ivf.enable_pq(8)
    ivf.train(data, seed=1)
    refine = qann.RefineIndex(ivf, k_factor=10)
    refine.add(data)
    ids, _ = refine.batch_query(data[:50], 10)
    assert ids.shape == (50, 10)
    assert np.mean(ids[:, 0] == np.arange(50)) >= 0.9


@pytest.fixture(scope="module")
def refine_index(data):
    ivf = qann.IVFIndex(32, nlist=100, nprobe=8)
    ivf.enable_pq(8)
    ivf.train(data, seed=1)
    index = qann.RefineIndex(ivf, k_factor=5)
    index.add(data)
    return index


def assert_same_results(a, b, queries, k=10):
    ids_a, dists_a = a.batch_query(queries, k)
    ids_b, dists_b = b.batch_query(queries, k)
    np.testing.assert_array_equal(ids_a, ids_b)
    np.testing.assert_array_equal(dists_a, dists_b)


@pytest.mark.parametrize("kind", ["flat", "ivf", "refine"])
def test_save_load_round_trip(tmp_path, data, refine_index, kind):
    if kind == "flat":
        index = qann.FlatIndex(32)
        index.add(data)
        expected_type = qann.FlatIndex
    elif kind == "ivf":
        index = qann.IVFIndex(32, nlist=100, nprobe=8)
        index.train(data, seed=1)
        index.add(data)
        expected_type = qann.IVFIndex
    else:
        index = refine_index
        expected_type = qann.RefineIndex

    path = tmp_path / "index.qann"
    index.save(path)
    loaded = qann.load(path)

    assert type(loaded) is expected_type
    assert loaded.size() == index.size()
    assert loaded.dim() == index.dim()
    assert_same_results(index, loaded, data[:50])


def test_save_load_accepts_str_paths(tmp_path, data, refine_index):
    path = str(tmp_path / "index.qann")
    refine_index.save(path)
    assert_same_results(refine_index, qann.load(path), data[:50])


def test_loaded_refine_base_is_tunable(tmp_path, data, refine_index):
    path = tmp_path / "index.qann"
    refine_index.save(path)
    loaded = qann.load(path)

    base = loaded.base
    assert type(base) is qann.IVFIndex
    assert base.nprobe == 8
    base.nprobe = 40
    assert loaded.base.nprobe == 40

    del loaded  # the base must stay usable while Python still holds it
    assert base.size() == len(data)


def test_load_errors(tmp_path, refine_index):
    with pytest.raises(RuntimeError):
        qann.load(tmp_path / "missing.qann")

    path = tmp_path / "index.qann"
    refine_index.save(path)
    good = path.read_bytes()

    path.write_bytes(good[: len(good) // 2])
    with pytest.raises(RuntimeError):
        qann.load(path)

    path.write_bytes(good + b"junk")
    with pytest.raises(RuntimeError):
        qann.load(path)

    path.write_bytes(b"not an index file")
    with pytest.raises(RuntimeError):
        qann.load(path)


def test_batch_query_releases_gil():
    rng = np.random.default_rng(1)
    index = qann.FlatIndex(32)
    index.add(rng.random((20_000, 32), dtype=np.float32))
    queries = rng.random((1_000, 32), dtype=np.float32)

    count = 0
    stop = threading.Event()

    def spin():
        nonlocal count
        while not stop.is_set():
            count += 1

    thread = threading.Thread(target=spin)
    thread.start()
    try:
        before = count
        index.batch_query(queries, 10)
        after = count
    finally:
        stop.set()
        thread.join()

    # With the GIL held for the whole call, the spinning thread could not run at all.
    assert after > before


def test_concurrent_adds_queries_and_tuning(data):
    rng = np.random.default_rng(2)
    ivf = qann.IVFIndex(32, nlist=100, nprobe=8)
    ivf.enable_pq(8)
    ivf.train(data, seed=1)
    index = qann.RefineIndex(ivf, k_factor=5)
    index.add(data[:1_000])

    batches = [rng.random((200, 32), dtype=np.float32) for _ in range(20)]
    queries = rng.random((100, 32), dtype=np.float32)
    errors = []

    def run(fn):
        try:
            fn()
        except Exception as e:  # surfaced below; a thread's exception is otherwise lost
            errors.append(e)

    def adder():
        for batch in batches:
            index.add(batch)

    def querier():
        for _ in range(20):
            ids, _ = index.batch_query(queries, 10)
            assert ids.max() < index.size()
            index.query(queries[0], 5)

    def tuner():
        for i in range(200):
            ivf.nprobe = 4 + i % 16
            index.k_factor = 2 + i % 4

    threads = [threading.Thread(target=run, args=(f,)) for f in (adder, querier, querier, tuner)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert index.size() == 1_000 + 200 * len(batches)
    ids, _ = index.batch_query(batches[-1][:10], 1)
    np.testing.assert_array_equal(ids[:, 0], np.arange(index.size() - 200, index.size() - 190))
