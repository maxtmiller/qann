import pathlib
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


DATA = pathlib.Path(__file__).parent / "data"


@pytest.mark.parametrize("name", ["flat", "ivf_pq", "refine"])
def test_loads_files_from_0_1_0a4(name):
    # Saved by qann 0.1.0a4 (format version 1) with the results it returned.
    index = qann.load(DATA / f"v1_{name}.qann")
    queries = np.load(DATA / "v1_queries.npy")
    expected_ids = np.load(DATA / f"v1_{name}_ids.npy")
    expected_dists = np.load(DATA / f"v1_{name}_dists.npy")

    assert index.size() == 2000
    ids, dists = index.batch_query(queries, 10)
    # SIMD kernels round differently across CPUs, which can swap near-ties.
    assert np.mean(ids == expected_ids) >= 0.99
    np.testing.assert_allclose(dists, expected_dists, rtol=1e-4, atol=1e-5)


def make_custom_index(kind, data, ids):
    if kind == "flat":
        index = qann.FlatIndex(32, custom_ids=True)
    elif kind == "ivf":
        index = qann.IVFIndex(32, nlist=100, nprobe=100, custom_ids=True)
        index.enable_pq(8)
        index.train(data, seed=1)
    else:
        base = qann.IVFIndex(32, nlist=100, nprobe=100)
        base.enable_pq(8)
        base.train(data, seed=1)
        index = qann.RefineIndex(base, k_factor=5, custom_ids=True)
    index.add(data, ids=ids)
    return index


@pytest.mark.parametrize("kind", ["flat", "ivf", "refine"])
def test_custom_ids_and_remove(data, kind):
    ids = np.arange(len(data)) * 10 + 7
    index = make_custom_index(kind, data, ids)

    found, _ = index.batch_query(data[:20], 1)
    if kind != "ivf":  # exact distances: each vector finds itself
        np.testing.assert_array_equal(found[:, 0], ids[:20])
    assert np.isin(found, ids).all()

    deleted = ids[::3]
    assert index.remove(deleted) == len(deleted)
    assert index.remove(deleted) == 0
    assert index.size() == len(data) - len(deleted)
    found, _ = index.batch_query(data[::3][:50], 10)
    assert not np.isin(found, deleted).any()

    # A deleted id can be added again, with a new vector.
    new_vec = np.full((1, 32), 5.0, dtype=np.float32)
    index.add(new_vec, ids=[int(deleted[0])])
    assert index.query(new_vec[0], 1)[0][0] == deleted[0]


def test_ids_accept_any_integer_sequence(data):
    index = qann.FlatIndex(32, custom_ids=True)
    index.add(data[:4], ids=[10, 11, 12, 13])
    index.add(data[4:6], ids=np.array([14, 15], dtype=np.int32))
    index.add(data[6:8], ids=range(16, 18))
    assert index.remove([10]) == 1
    assert index.remove(np.array([11], dtype=np.uint8)) == 1
    assert index.remove(range(12, 14)) == 2
    assert index.remove([]) == 0
    assert index.size() == 4


def test_bad_ids_are_rejected_without_adding(data):
    index = qann.FlatIndex(32, custom_ids=True)
    index.add(data[:3], ids=[1, 2, 3])

    for ids, error in [
        ([4], ValueError),            # wrong length
        ([4, -1], ValueError),        # negative
        ([4, 4], ValueError),         # duplicate in the batch
        ([4, 1], ValueError),         # already in the index
        ([4.0, 5.0], TypeError),      # not integers
        ([[4, 5]], TypeError),        # not one-dimensional
    ]:
        with pytest.raises(error):
            index.add(data[:2], ids=ids)
    with pytest.raises(ValueError):
        index.add(data[:2])           # custom-id index needs ids
    with pytest.raises(ValueError):
        qann.FlatIndex(32).add(data[:2], ids=[1, 2])  # default index refuses them
    assert index.size() == 3


def test_custom_ids_via_make_index(data):
    opts = qann.IndexOptions()
    opts.custom_ids = True
    index = qann.make_index(qann.IndexType.Flat, 32, opts)
    index.add(data[:5], ids=[50, 51, 52, 53, 54])
    assert index.query(data[2], 1)[0][0] == 52


def test_save_load_keeps_ids_and_deletions(tmp_path, data):
    ids = np.arange(len(data)) + 1_000_000
    index = make_custom_index("refine", data, ids)
    index.remove(ids[:100])

    path = tmp_path / "ids.qann"
    index.save(path)
    loaded = qann.load(path)

    assert loaded.size() == index.size()
    assert_same_results(index, loaded, data[:50])
    assert loaded.remove(ids[:100]) == 0     # still deleted
    assert loaded.remove(ids[100:110]) == 10  # still live, under the same ids


def test_concurrent_remove_and_query(data):
    ids = np.arange(len(data)) + 100
    index = make_custom_index("flat", data, ids)
    to_delete = ids[::2]
    errors = []

    def run(fn):
        try:
            fn()
        except Exception as e:
            errors.append(e)

    def remover():
        for chunk in np.array_split(to_delete, 50):
            index.remove(chunk)

    def querier():
        for _ in range(30):
            found, _ = index.batch_query(data[:50], 5)
            assert np.isin(found[found >= 0], ids).all()

    threads = [threading.Thread(target=run, args=(f,)) for f in (remover, querier, querier)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert index.size() == len(data) - len(to_delete)
    found, _ = index.batch_query(data[:50], 10)
    assert not np.isin(found, to_delete).any()


def test_base_of_refine_index_is_protected(data):
    import gc

    base = qann.IVFIndex(32, nlist=100, nprobe=8)
    base.train(data, seed=1)
    refine = qann.RefineIndex(base, k_factor=4)
    refine.add(data[:100])

    with pytest.raises(ValueError, match="through the RefineIndex"):
        base.add(data[:5])
    with pytest.raises(ValueError, match="through the RefineIndex"):
        base.remove([0])
    with pytest.raises(ValueError):
        qann.RefineIndex(base)
    assert base.size() == refine.size() == 100

    # Tuning the base is still allowed, and changes through the RefineIndex reach it.
    base.nprobe = 16
    assert refine.remove([0, 1]) == 2
    assert base.size() == 98

    # Once the RefineIndex is gone, the base can be used on its own.
    del refine
    gc.collect()
    base.add(data[100:105])
    assert base.size() == 103


def test_base_of_loaded_refine_index_is_protected(tmp_path, refine_index, data):
    path = tmp_path / "refine.qann"
    refine_index.save(path)
    loaded = qann.load(path)
    with pytest.raises(ValueError, match="through the RefineIndex"):
        loaded.base.add(data[:2])


def test_precomputed_tables_toggle(data):
    ivf = qann.IVFIndex(32, 100, 20)
    ivf.enable_pq(8)
    assert not ivf.precomputed_tables
    ivf.train(data, seed=1)
    assert ivf.precomputed_tables
    ivf.add(data)

    ids_fast, dists_fast = ivf.batch_query(data[:50], 10)
    ivf.precomputed_tables = False
    assert not ivf.precomputed_tables
    ids_slow, dists_slow = ivf.batch_query(data[:50], 10)

    np.testing.assert_allclose(dists_fast, dists_slow, rtol=1e-3, atol=1e-3)
    assert np.mean(ids_fast == ids_slow) >= 0.98
