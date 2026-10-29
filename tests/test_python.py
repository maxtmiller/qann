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
    elif kind == "hnsw":
        index = qann.HNSWIndex(32, custom_ids=True, seed=1)
    else:
        base = qann.IVFIndex(32, nlist=100, nprobe=100)
        base.enable_pq(8)
        base.train(data, seed=1)
        index = qann.RefineIndex(base, k_factor=5, custom_ids=True)
    index.add(data, ids=ids)
    return index


@pytest.mark.parametrize("kind", ["flat", "ivf", "refine", "hnsw"])
def test_custom_ids_and_remove(data, kind):
    ids = np.arange(len(data)) * 10 + 7
    index = make_custom_index(kind, data, ids)

    found, _ = index.batch_query(data[:20], 1)
    if kind in ("flat", "refine"):  # exact search over exact distances: each vector finds itself
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


@pytest.mark.parametrize("kind", ["flat", "ivf", "refine", "hnsw"])
def test_filtered_query(data, kind):
    ids = np.arange(len(data)) * 10 + 7
    index = make_custom_index(kind, data, ids)
    allowed = ids[::4]
    index.remove(allowed[:10])

    found, _ = index.batch_query(data[:50], 10, allowed=allowed)
    assert np.isin(found, allowed[10:]).all()
    if kind in ("flat", "refine"):  # exact distances: allowed vectors find themselves
        np.testing.assert_array_equal(found[40:50:4, 0], ids[40:50:4])

    one, _ = index.query(data[0], 10, allowed=allowed)
    np.testing.assert_array_equal(one, index.batch_query(data[:1], 10, allowed=allowed)[0][0])

    # A list works too; unknown ids are ignored; fewer allowed than k pads the rows.
    found, dists = index.batch_query(data[:3], 5, allowed=[int(ids[1]), int(ids[2]), 123])
    assert set(found[:, :2].ravel()) == {ids[1], ids[2]}
    assert (found[:, 2:] == -1).all() and np.isinf(dists[:, 2:]).all()

    found, _ = index.query(data[0], 5, allowed=[])
    assert len(found) == 0
    with pytest.raises(TypeError):
        index.query(data[0], 5, allowed=[1.5])


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


@pytest.mark.parametrize("kind", ["flat", "ivf", "refine"])
def test_compact_keeps_ids_and_results(tmp_path, data, kind):
    ids = np.arange(len(data)) * 10 + 7
    index = make_custom_index(kind, data, ids)
    assert index.remove(ids[::4]) == len(ids[::4])
    before_ids, before_dists = index.batch_query(data[:50], 10)

    assert index.compact() == len(ids[::4])
    assert index.compact() == 0
    assert index.size() == len(data) - len(ids[::4])
    after_ids, after_dists = index.batch_query(data[:50], 10)
    np.testing.assert_array_equal(after_ids, before_ids)
    np.testing.assert_array_equal(after_dists, before_dists)

    index.save(tmp_path / "compacted.qann")
    loaded = qann.load(tmp_path / "compacted.qann")
    loaded_ids, _ = loaded.batch_query(data[:50], 10)
    np.testing.assert_array_equal(loaded_ids, before_ids)
    assert loaded.remove([ids[1]]) == 1


def test_compact_auto_ids_continue(data):
    index = qann.FlatIndex(32)
    index.add(data[:10])
    index.remove(range(10))
    assert index.compact() == 10
    index.add(data[:2])
    ids, _ = index.batch_query(data[:2], 1)
    np.testing.assert_array_equal(ids[:, 0], [10, 11])


def test_compact_refused_on_wrapped_base(data):
    base = qann.FlatIndex(32)
    refine = qann.RefineIndex(base, k_factor=2)
    refine.add(data[:100])
    refine.remove([0, 1])
    with pytest.raises(ValueError, match="compact"):
        base.compact()
    assert refine.compact() == 2


@pytest.mark.parametrize("name", ["flat", "ivf_pq", "refine"])
def test_loads_files_from_0_1_0a6(name):
    # Saved by qann 0.1.0a6 (format version 2), with deletions, and the results it returned.
    index = qann.load(DATA / f"v2_{name}.qann")
    queries = np.load(DATA / "v2_queries.npy")
    expected_ids = np.load(DATA / f"v2_{name}_ids.npy")
    expected_dists = np.load(DATA / f"v2_{name}_dists.npy")

    assert index.size() == 1600
    ids, dists = index.batch_query(queries, 10)
    assert np.mean(ids == expected_ids) >= 0.99
    np.testing.assert_allclose(dists, expected_dists, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", ["flat", "ivf_pq", "refine"])
def test_loads_files_from_0_1_0a7(name):
    # Saved by qann 0.1.0a7 (format version 3) after compaction, with the results it returned.
    index = qann.load(DATA / f"v3_{name}.qann")
    queries = np.load(DATA / "v3_queries.npy")
    expected_ids = np.load(DATA / f"v3_{name}_ids.npy")
    expected_dists = np.load(DATA / f"v3_{name}_dists.npy")

    assert index.size() == (1300 if name == "flat" else 1500)
    ids, dists = index.batch_query(queries, 10)
    assert np.mean(ids == expected_ids) >= 0.99
    np.testing.assert_allclose(dists, expected_dists, rtol=1e-4, atol=1e-5)


def test_fast_scan(tmp_path, data):
    def build(fast):
        ivf = qann.IVFIndex(32, nlist=100, nprobe=100)
        ivf.enable_pq(16, fast_scan=fast) if fast else ivf.enable_pq(16, 16)
        ivf.train(data, seed=1)
        ivf.add(data)
        return ivf

    fast, ref = build(True), build(False)
    assert fast.fast_scan and not ref.fast_scan
    ids_f, dists_f = fast.batch_query(data[:100], 10)
    ids_r, _ = ref.batch_query(data[:100], 10)
    assert np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids_f, ids_r)]) >= 0.95
    assert np.mean(ids_f[:, 0] == np.arange(100)) >= 0.9

    fast.save(tmp_path / "fs.qann")
    loaded = qann.load(tmp_path / "fs.qann")
    assert loaded.fast_scan
    np.testing.assert_array_equal(loaded.batch_query(data[:100], 10)[0], ids_f)


def test_fast_scan_rejects_bad_settings():
    ivf = qann.IVFIndex(32, nlist=100)
    with pytest.raises(ValueError):
        ivf.enable_pq(16, 256, fast_scan=True)
    ivf.enable_pq(16, fast_scan=True)
    with pytest.raises(ValueError):
        ivf.pq_distance = qann.PQDistance.SDC


def test_hnsw_finds_neighbors(data):
    index = qann.HNSWIndex(32, M=16, ef_construction=100, seed=1)
    index.add(data)
    assert index.size() == len(data)
    assert (index.M, index.ef_construction, index.ef_search) == (16, 100, 64)

    flat = qann.FlatIndex(32)
    flat.add(data)
    queries = data[:200] + 0.01
    truth, _ = flat.batch_query(queries, 10)
    for ef in (16, 128):
        index.ef_search = ef
        ids, dists = index.batch_query(queries, 10)
        assert ids.shape == (200, 10)
        assert np.all(np.diff(dists, axis=1) >= 0)
        recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, truth)])
        assert recall >= (0.9 if ef == 16 else 0.98)


def test_hnsw_save_load_compact(tmp_path, data):
    index = qann.HNSWIndex(32, M=12, ef_construction=80, seed=1)
    index.add(data)
    index.ef_search = 40
    index.remove(range(0, len(data), 4))
    index.save(tmp_path / "hnsw.qann")

    loaded = qann.load(tmp_path / "hnsw.qann")
    assert isinstance(loaded, qann.HNSWIndex)
    assert (loaded.M, loaded.ef_construction, loaded.ef_search) == (12, 80, 40)
    assert_same_results(index, loaded, data[:50])

    assert loaded.compact() == len(range(0, len(data), 4))
    assert loaded.size() == len(data) - len(range(0, len(data), 4))
    ids, _ = loaded.batch_query(data[1:200:4], 1)   # live vectors find themselves, by their old ids
    assert np.mean(ids[:, 0] == np.arange(1, 200, 4)) >= 0.98
    assert not np.isin(loaded.batch_query(data[:50], 10)[0], np.arange(0, len(data), 4)).any()


def test_hnsw_via_make_index_and_bad_settings():
    opts = qann.IndexOptions()
    opts.hnsw_m = 8
    opts.ef_construction = 40
    opts.ef_search = 20
    index = qann.make_index(qann.IndexType.HNSW, 16, opts)
    assert isinstance(index, qann.HNSWIndex)
    assert (index.M, index.ef_construction, index.ef_search) == (8, 40, 20)

    with pytest.raises(ValueError):
        qann.HNSWIndex(16, M=1)
    with pytest.raises(ValueError):
        index.ef_search = 0


@pytest.mark.parametrize("bits", [8, 4])
def test_hnsw_sq_with_refine(tmp_path, data, bits):
    flat = qann.FlatIndex(32)
    flat.add(data)
    queries = data[:200] + 0.01
    truth, _ = flat.batch_query(queries, 10)

    base = qann.HNSWIndex(32, M=16, ef_construction=100, seed=1)
    assert base.sq_bits == 0
    base.enable_sq(bits)
    assert base.sq_bits == bits
    with pytest.raises(RuntimeError):   # not trained yet
        base.add(data)
    with pytest.raises(ValueError):
        base.train(data[:, :16])
    base.train(data)
    refine = qann.RefineIndex(base, k_factor=4 if bits == 8 else 8)
    refine.add(data)
    ids, dists = refine.batch_query(queries, 10)
    assert np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, truth)]) >= 0.97
    np.testing.assert_allclose(dists[:, 0], ((queries - data[ids[:, 0]]) ** 2).sum(axis=1), rtol=1e-4)

    refine.save(tmp_path / "sq.qann")
    loaded = qann.load(tmp_path / "sq.qann")
    assert loaded.base.sq_bits == bits
    assert_same_results(refine, loaded, queries[:50])


def test_hnsw_sq_options_and_bad_bits(data):
    opts = qann.IndexOptions()
    opts.hnsw_sq_bits = 4
    assert qann.make_index(qann.IndexType.HNSW, 16, opts).sq_bits == 4
    assert qann.HNSWIndex(16).enable_sq() is None   # 8 by default
    for bad in (0, 2, 16):
        with pytest.raises(ValueError):
            qann.HNSWIndex(16).enable_sq(bad)
    with pytest.raises(RuntimeError):   # train() is only for SQ
        qann.HNSWIndex(32).train(data)


def test_cosine_flat_matches_cosine_distance(data):
    index = qann.FlatIndex(32, metric=qann.Metric.Cosine)
    assert index.metric == qann.Metric.Cosine
    index.add(data * np.linspace(0.1, 10, len(data), dtype=np.float32)[:, None])   # lengths don't matter
    queries = data[:20] + 0.05
    ids, dists = index.batch_query(queries, 5)

    for q, row_ids, row_dists in zip(queries, ids, dists):
        expected = qann.cosine_distance(np.repeat(q[None, :], len(data), axis=0), data)
        top = np.argsort(expected, kind="stable")[:5]
        np.testing.assert_array_equal(row_ids, top)
        np.testing.assert_allclose(row_dists, expected[top], atol=1e-5)


def test_inner_product_flat_ranks_by_dot_product(data):
    index = qann.FlatIndex(32, metric=qann.Metric.InnerProduct)
    assert index.metric == qann.Metric.InnerProduct
    scaled = data * np.linspace(0.1, 10, len(data), dtype=np.float32)[:, None]   # lengths matter here
    index.add(scaled)
    queries = data[:20] - 0.5
    ids, dists = index.batch_query(queries, 5)

    dots = queries @ scaled.T
    top = np.argsort(-dots, axis=1, kind="stable")[:, :5]
    np.testing.assert_array_equal(ids, top)
    np.testing.assert_allclose(dists, -np.take_along_axis(dots, top, axis=1), rtol=1e-4)


@pytest.mark.parametrize("fast_scan", [False, True])
def test_inner_product_ivf_pq_with_refine(fast_scan):
    scaled, queries, truth = _ip_clustered()
    ivf = qann.IVFIndex(32, nlist=100, nprobe=20, metric=qann.Metric.InnerProduct)
    ivf.enable_pq(16 if fast_scan else 8, fast_scan=fast_scan)
    with pytest.raises(ValueError):
        ivf.pq_distance = qann.PQDistance.SDC
    ivf.train(scaled, seed=1)
    assert not ivf.precomputed_tables
    index = qann.RefineIndex(ivf, k_factor=10)
    index.add(scaled)
    ids, _ = index.batch_query(queries, 10)
    assert np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, truth)]) >= 0.95


def _ip_clustered():
    # directions follow 20 cluster centers, lengths spread over 0.1x to 10x
    rng = np.random.default_rng(3)
    centers = rng.standard_normal((20, 32))
    scaled = ((centers[rng.integers(0, 20, 5000)] + 0.5 * rng.standard_normal((5000, 32)))
              * 10 ** rng.uniform(-1, 1, (5000, 1))).astype(np.float32)
    queries = (centers[rng.integers(0, 20, 50)] + 0.5 * rng.standard_normal((50, 32))).astype(np.float32)
    flat = qann.FlatIndex(32, metric=qann.Metric.InnerProduct)
    flat.add(scaled)
    truth, _ = flat.batch_query(queries, 10)
    return scaled, queries, truth


@pytest.mark.parametrize("sq", [0, 8])
def test_inner_product_ivf_recall(sq):
    scaled, queries, truth = _ip_clustered()
    ivf = qann.IVFIndex(32, nlist=100, nprobe=20, metric=qann.Metric.InnerProduct)
    if sq:
        ivf.enable_sq(sq)
    ivf.train(scaled, seed=1)
    ivf.add(scaled)
    ids, _ = ivf.batch_query(queries, 10)
    assert np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, truth)]) >= 0.95


@pytest.mark.parametrize("sq", [0, 8])
def test_inner_product_hnsw_recall(sq):
    scaled, queries, truth = _ip_clustered()

    hnsw = qann.HNSWIndex(32, metric=qann.Metric.InnerProduct, seed=1)
    index = hnsw
    if sq:
        hnsw.enable_sq(sq)
        hnsw.train(scaled)
        index = qann.RefineIndex(hnsw, k_factor=2)
    index.add(scaled)
    hnsw.ef_search = 200
    ids, _ = index.batch_query(queries, 10)
    assert np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, truth)]) >= 0.95


def test_cosine_rejects_zero_vectors(data):
    index = qann.FlatIndex(32, metric=qann.Metric.Cosine)
    bad = data[:3].copy()
    bad[1] = 0
    with pytest.raises(ValueError, match="non-zero"):
        index.add(bad)
    assert index.size() == 0
    index.add(data[:3])
    with pytest.raises(ValueError):
        index.query(np.zeros(32, dtype=np.float32), 1)


@pytest.mark.parametrize("kind", ["flat", "ivf", "hnsw"])
def test_cosine_metric_survives_save_load(tmp_path, data, kind):
    if kind == "flat":
        index = qann.FlatIndex(32, metric=qann.Metric.Cosine)
        index.add(data)
    elif kind == "ivf":
        ivf = qann.IVFIndex(32, nlist=100, nprobe=20, metric=qann.Metric.Cosine)
        ivf.enable_pq(16, fast_scan=True)
        ivf.train(data, seed=1)
        index = qann.RefineIndex(ivf, k_factor=10)
        assert index.metric == qann.Metric.Cosine   # taken from the base
        index.add(data)
    else:
        index = qann.HNSWIndex(32, metric=qann.Metric.Cosine)
        index.add(data)

    index.save(tmp_path / "cos.qann")
    loaded = qann.load(tmp_path / "cos.qann")
    assert loaded.metric == qann.Metric.Cosine
    assert_same_results(index, loaded, data[:50])


def test_metric_via_make_index():
    opts = qann.IndexOptions()
    opts.metric = qann.Metric.Cosine
    for t in (qann.IndexType.Flat, qann.IndexType.IVF, qann.IndexType.HNSW):
        assert qann.make_index(t, 16, opts).metric == qann.Metric.Cosine
    assert qann.FlatIndex(16).metric == qann.Metric.L2


@pytest.mark.parametrize("name", ["flat", "ivf_fs", "hnsw"])
def test_loads_files_from_0_1_0a10(name):
    # Saved by qann 0.1.0a10 (format version 4, before the metric), with the results it returned.
    index = qann.load(DATA / f"v4_{name}.qann")
    queries = np.load(DATA / "v4_queries.npy")
    expected_ids = np.load(DATA / f"v4_{name}_ids.npy")
    expected_dists = np.load(DATA / f"v4_{name}_dists.npy")

    assert index.metric == qann.Metric.L2
    ids, dists = index.batch_query(queries, 10)
    np.testing.assert_array_equal(ids, expected_ids)
    np.testing.assert_allclose(dists, expected_dists, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", ["flat", "hnsw"])
def test_loads_files_from_0_1_0a11(name):
    # Saved by qann 0.1.0a11 (format version 5): cosine flat and L2 HNSW, both with deletions.
    index = qann.load(DATA / f"v5_{name}.qann")
    queries = np.load(DATA / "v5_queries.npy")
    expected_ids = np.load(DATA / f"v5_{name}_ids.npy")
    expected_dists = np.load(DATA / f"v5_{name}_dists.npy")

    assert index.metric == (qann.Metric.Cosine if name == "flat" else qann.Metric.L2)
    ids, dists = index.batch_query(queries, 10)
    np.testing.assert_array_equal(ids, expected_ids)
    np.testing.assert_allclose(dists, expected_dists, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("bits", [8, 4])
def test_loads_sq_hnsw_saved_on_macos(bits):
    # Saved on macOS (format version 6): a RefineIndex over an SQ HNSWIndex,
    # with deletions. Linux CI loading it checks the file is portable.
    index = qann.load(DATA / f"v6_hnsw_sq{bits}.qann")
    assert index.base.sq_bits == bits
    ids, dists = index.batch_query(np.load(DATA / "v6_queries.npy"), 10)
    np.testing.assert_array_equal(ids, np.load(DATA / f"v6_hnsw_sq{bits}_ids.npy"))
    np.testing.assert_allclose(dists, np.load(DATA / f"v6_hnsw_sq{bits}_dists.npy"), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("bits", [8, 4])
def test_ivf_sq(tmp_path, data, bits):
    flat = qann.FlatIndex(32)
    flat.add(data)
    queries = data[:200] + 0.01
    truth, _ = flat.batch_query(queries, 10)

    floats = qann.IVFIndex(32, nlist=100, nprobe=20)
    floats.train(data, seed=1)
    floats.add(data)
    float_ids, _ = floats.batch_query(queries, 10)
    float_recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(float_ids, truth)])

    ivf = qann.IVFIndex(32, nlist=100, nprobe=20)
    assert ivf.sq_bits == 0
    ivf.enable_sq(bits)
    assert ivf.sq_bits == bits
    with pytest.raises(RuntimeError):   # not with PQ
        ivf.enable_pq(8)
    with pytest.raises(RuntimeError):   # not trained yet
        ivf.add(data)
    ivf.train(data, seed=1)
    index = qann.RefineIndex(ivf, k_factor=8) if bits == 4 else ivf
    index.add(data)
    ids, _ = index.batch_query(queries, 10)
    assert np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, truth)]) >= float_recall - 0.02

    index.save(tmp_path / "ivf_sq.qann")
    loaded = qann.load(tmp_path / "ivf_sq.qann")
    assert (loaded.base if bits == 4 else loaded).sq_bits == bits
    assert_same_results(index, loaded, queries[:50])


def test_ivf_sq_options_and_bad_bits():
    opts = qann.IndexOptions()
    opts.ivf_sq_bits = 8
    assert qann.make_index(qann.IndexType.IVF, 16, opts).sq_bits == 8
    for bad in (0, 2, 16):
        with pytest.raises(ValueError):
            qann.IVFIndex(16, 100).enable_sq(bad)


@pytest.mark.parametrize("bits", [8, 4])
def test_loads_sq_ivf_saved_on_macos(bits):
    # Saved on macOS (format version 7): SQ IVF with deletions (4 bits under a
    # RefineIndex). Linux CI loading it checks the file is portable.
    index = qann.load(DATA / f"v7_ivf_sq{bits}.qann")
    assert (index.base if bits == 4 else index).sq_bits == bits
    ids, dists = index.batch_query(np.load(DATA / "v7_queries.npy"), 10)
    np.testing.assert_array_equal(ids, np.load(DATA / f"v7_ivf_sq{bits}_ids.npy"))
    np.testing.assert_allclose(dists, np.load(DATA / f"v7_ivf_sq{bits}_dists.npy"), rtol=1e-5, atol=1e-5)


def test_ivf_pq_rotate(tmp_path):
    # Dimension j + 16 is a near-copy of dimension j, so plain PQ quantizes every
    # value twice in different subspaces; the rotation folds each pair together.
    rng = np.random.default_rng(3)
    z = rng.standard_normal((6100, 16)).astype(np.float32)
    x = np.hstack([z, z + 0.05 * rng.standard_normal(z.shape).astype(np.float32)])
    data, queries = x[:6000], x[6000:]
    flat = qann.FlatIndex(32)
    flat.add(data)
    truth, _ = flat.batch_query(queries, 10)

    recall = {}
    for rotate in (False, True):
        ivf = qann.IVFIndex(32, nlist=100, nprobe=100)
        ivf.enable_pq(8, rotate=rotate)
        assert ivf.pq_rotate == rotate
        ivf.train(data, seed=1)
        ivf.add(data)
        ids, _ = ivf.batch_query(queries, 10)
        recall[rotate] = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, truth)])
    assert recall[True] >= recall[False] + 0.1

    ivf.save(tmp_path / "rot.qann")
    loaded = qann.load(tmp_path / "rot.qann")
    assert loaded.pq_rotate
    assert_same_results(ivf, loaded, queries[:50])

    opts = qann.IndexOptions()
    opts.pq_subspaces = 8
    opts.pq_rotate = True
    assert qann.make_index(qann.IndexType.IVF, 32, opts).pq_rotate


def test_loads_rotated_ivf_saved_on_macos():
    # Saved on macOS (format version 8): rotated IVF-PQ under a RefineIndex, with
    # deletions. Linux CI loading it checks the rotation is portable.
    index = qann.load(DATA / "v8_ivf_rot.qann")
    assert index.base.pq_rotate
    ids, dists = index.batch_query(np.load(DATA / "v8_queries.npy"), 10)
    np.testing.assert_array_equal(ids, np.load(DATA / "v8_ivf_rot_ids.npy"))
    np.testing.assert_allclose(dists, np.load(DATA / "v8_ivf_rot_dists.npy"), rtol=1e-5, atol=1e-5)


COLORS = np.array(["red", "green", "blue", "black"])


@pytest.mark.parametrize("kind", ["flat", "ivf", "refine", "hnsw"])
def test_metadata_filters(tmp_path, data, kind):
    ids = np.arange(len(data)) * 10 + 7
    if kind == "flat":
        index = qann.FlatIndex(32, custom_ids=True)
    elif kind == "hnsw":
        index = qann.HNSWIndex(32, custom_ids=True, seed=1)
    else:
        ivf = qann.IVFIndex(32, nlist=100, nprobe=100, custom_ids=(kind == "ivf"))
        ivf.enable_pq(8)
        ivf.train(data, seed=1)
        index = ivf if kind == "ivf" else qann.RefineIndex(ivf, k_factor=5, custom_ids=True)

    i = np.arange(len(data))
    half = len(data) // 2
    index.add(data[:half], ids=ids[:half],
              metadata={"price": i[:half] % 100, "weight": i[:half] * 0.5, "color": COLORS[i[:half] % 4]})
    index.add(data[half:], ids=ids[half:], metadata={"price": i[half:] % 100})   # weight, color null
    assert index.metadata_schema == {"price": "int", "weight": "float", "color": "str"}

    f = qann.field
    where = ((f("price") < 30) & ~(f("color") == "red")) | f("weight").isin([1.0, 2.5])
    match = ((i % 100 < 30) & (i < half) & (i % 4 != 0)) | ((i % 100 < 30) & (i >= half)) | np.isin(i * 0.5, [1.0, 2.5])
    queries = data[:40] + 0.01
    got, dists = index.batch_query(queries, 10, where=where)
    expected, expected_dists = index.batch_query(queries, 10, allowed=ids[match])
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(dists, expected_dists)
    one, _ = index.query(queries[0], 10, where=where)
    np.testing.assert_array_equal(one, got[0])

    # where and allowed together: only vectors passing both
    evens = ids[::2]
    got, _ = index.batch_query(queries, 10, where=where, allowed=evens)
    np.testing.assert_array_equal(got, index.batch_query(queries, 10, allowed=ids[match & (i % 2 == 0)])[0])

    # nulls: rows added without color
    got, _ = index.batch_query(queries, 5, where=f("color").is_null())
    assert np.isin(got[got >= 0], ids[half:]).all()

    index.remove(ids[:10])
    index.save(tmp_path / "md.qann")
    loaded = qann.load(tmp_path / "md.qann")
    assert loaded.metadata_schema == index.metadata_schema
    np.testing.assert_array_equal(loaded.batch_query(queries, 10, where=where)[0],
                                  index.batch_query(queries, 10, where=where)[0])


def test_metadata_input_and_filter_errors(data):
    index = qann.FlatIndex(32)
    index.add(data[:3], metadata={"n": [1, 2, 3], "x": np.array([0.5, 1.5, 2.5], dtype=np.float32),
                                  "s": np.array(["a", "b", "c"], dtype=object), "flag": np.array([True, False, True])})
    assert index.metadata_schema == {"n": "int", "x": "float", "s": "str", "flag": "int"}
    f = qann.field
    assert index.query(data[0], 3, where=f("flag") == 1)[0].tolist() == [0, 2]
    assert index.query(data[0], 3, where=f("x") > np.float32(1.0))[0].size == 2
    assert repr((f("n") >= 2) & ~f("s").isin(["a"])) == "qann.Expr(((n >= 2) & ~s in ['a']))"

    with pytest.raises(ValueError):    # one value per row
        index.add(data[3:5], metadata={"n": [1]})
    with pytest.raises(ValueError):    # a column keeps its type
        index.add(data[3:4], metadata={"n": [1.5]})
    with pytest.raises(TypeError):     # mixed kinds
        index.add(data[3:5], metadata={"s": ["a", 1]})
    with pytest.raises(TypeError):     # not ints, floats or strings
        index.add(data[3:4], metadata={"d": np.array(["2026-01-01"], dtype="datetime64[D]")})
    assert index.size() == 3

    with pytest.raises(ValueError):    # unknown column
        index.query(data[0], 3, where=f("missing") == 1)
    with pytest.raises(ValueError):    # wrong kind of value
        index.query(data[0], 3, where=f("s") == 1)
    for misuse in (lambda: bool(f("n") < 2), lambda: 0 < f("n") < 3, lambda: (f("n") < 2) and (f("n") > 0)):
        with pytest.raises(TypeError):
            misuse()



def test_metadata_sql_where(data):
    index = qann.FlatIndex(32)
    i = np.arange(200)
    index.add(data[:200], metadata={"price": i % 100, "color": COLORS[i % 4]})
    index.add(data[200:210])    # no metadata: price and color null
    f = qann.field
    queries = data[:20] + 0.01

    def same(where, expr):
        np.testing.assert_array_equal(index.batch_query(queries, 10, where=where)[0],
                                      index.batch_query(queries, 10, where=expr)[0])

    same("price < 30 AND color != 'red'", (f("price") < 30) & (f("color") != "red"))
    same("color IN ('red', 'blue') OR price BETWEEN 10 AND 20",
         f("color").isin(["red", "blue"]) | ((f("price") >= 10) & (f("price") <= 20)))
    same(("price < ? AND color = ?", [30, "red"]), (f("price") < 30) & (f("color") == "red"))
    same(("price >= :lo AND price < :hi", {"lo": 5, "hi": 9.5}), (f("price") >= 5) & (f("price") < 9.5))
    same(qann.where("price < ?", [30]) & (f("color") == "red"), (f("price") < 30) & (f("color") == "red"))
    same(f("price").between(10, 20), "price BETWEEN 10 AND 20")
    same(f("price").between(10, 20), (f("price") >= 10) & (f("price") <= 20))
    assert repr(f("price").between(1, 2.5)) == "qann.Expr(((price >= 1) & (price <= 2.5)))"
    ends = index.query(data[0], 210, where=f("price").between(10, 10))[0]
    assert sorted(ends.tolist()) == [10, 110]    # both ends included; nulls never match
    assert index.query(data[0], 10, where=f("price").between(20, 10))[0].size == 0
    assert index.query(queries[0], 10, where="price IS NULL")[0].tolist() == \
        index.query(queries[0], 10, allowed=np.arange(200, 210))[0].tolist()

    # SQL's NOT excludes nulls; ~ includes them
    nulls = np.arange(200, 210)
    assert not np.isin(nulls, index.query(data[200], 210, where="NOT price < 50")[0]).any()
    assert np.isin(nulls, index.query(data[200], 210, where=~(f("price") < 50))[0]).all()
    assert repr(qann.where("NOT (price < 5 OR color IS NULL)")) == "qann.Expr(((price >= 5) & ~is_null(color)))"

    # a parameter is one value, whatever it contains
    assert index.query(queries[0], 10, where=("color = ?", ["red' OR 'a' = 'a"]))[0].size == 0

    with pytest.raises(ValueError, match="unexpected .color. at column 11"):
        index.query(queries[0], 10, where="price < 5 color = 'red'")
    with pytest.raises(ValueError, match="no parameter named 'hi'"):
        qann.where("price < :hi", {"lo": 1})
    with pytest.raises(ValueError, match="placeholders"):
        index.query(queries[0], 10, where=("price < ?", [1, 2]))
    with pytest.raises(ValueError):    # unknown column, found when evaluated
        index.query(queries[0], 10, where="size = 1")

def test_metadata_unsigned_overflow_and_empty_adds(data):
    index = qann.FlatIndex(32)
    with pytest.raises(ValueError):    # would wrap to a negative int64
        index.add(data[:2], metadata={"u": np.array([1, 2**63], dtype=np.uint64)})
    index.add(data[:2], metadata={"u": np.array([1, 2**63 - 1], dtype=np.uint64)})
    assert index.query(data[0], 2, where=qann.field("u") > 0)[0].size == 2

    # an add of no vectors says nothing about a column's type, so it creates none
    index.add(data[:0], metadata={"tag": []})
    assert "tag" not in index.metadata_schema
    index.add(data[2:3], metadata={"tag": ["x"]})
    assert index.metadata_schema["tag"] == "str"


def test_concurrent_metadata_adds_and_filtered_queries(data):
    # adds with metadata (exclusive lock) interleaved with where= queries (shared):
    # every result must satisfy the filter as of some consistent state
    index = qann.HNSWIndex(32, seed=1)
    index.add(data[:500], metadata={"group": np.arange(500) % 5, "name": np.array(["a", "b"])[np.arange(500) % 2]})
    rng = np.random.default_rng(4)
    batches = [rng.random((100, 32), dtype=np.float32) for _ in range(15)]
    queries = rng.random((50, 32), dtype=np.float32)
    f = qann.field
    where = (f("group") == 3) & (f("name") == "b")
    errors = []

    def run(fn):
        try:
            fn()
        except Exception as e:
            errors.append(e)

    def adder():
        for b, batch in enumerate(batches):
            start = 500 + 100 * b
            ids = np.arange(start, start + 100)
            index.add(batch, metadata={"group": ids % 5, "name": np.array(["a", "b"])[ids % 2]})

    def querier():
        for _ in range(30):
            got, _ = index.batch_query(queries, 10, where=where)
            got = got[got >= 0]
            assert ((got % 5 == 3) & (got % 2 == 1)).all()

    threads = [threading.Thread(target=run, args=(fn,)) for fn in (adder, querier, querier)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert index.size() == 500 + 100 * len(batches)
    got, _ = index.batch_query(queries, 10, where=where)
    assert ((got % 5 == 3) & (got % 2 == 1)).all()


def test_loads_metadata_saved_on_macos():
    # Saved on macOS (format version 9): a RefineIndex over IVF-PQ with int,
    # float and string columns, nulls from a second add, and deletions.
    index = qann.load(DATA / "v9_refine_metadata.qann")
    assert index.metadata_schema == {"price": "int", "weight": "float", "color": "str", "tag": "str"}
    f = qann.field
    where = ((f("price") < 40) & (f("color") != "red")) | (f("tag") == "new")
    ids, dists = index.batch_query(np.load(DATA / "v9_queries.npy"), 10, where=where)
    np.testing.assert_array_equal(ids, np.load(DATA / "v9_refine_metadata_ids.npy"))
    np.testing.assert_allclose(dists, np.load(DATA / "v9_refine_metadata_dists.npy"), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", ["flat", "hnsw_sq8", "refine_ivf_fs"])
def test_loads_inner_product_saved_on_macos(name):
    # Saved on macOS (format version 9, Metric.InnerProduct): 2000 clustered
    # 16-dim vectors with lengths over 0.1x to 10x; a Flat with deletions, an
    # HNSW with SQ8, and a RefineIndex over rotated fast-scan IVF-PQ.
    index = qann.load(DATA / f"v9_ip_{name}.qann")
    assert index.metric == qann.Metric.InnerProduct
    ids, dists = index.batch_query(np.load(DATA / "v9_ip_queries.npy"), 10)
    np.testing.assert_array_equal(ids, np.load(DATA / f"v9_ip_{name}_ids.npy"))
    np.testing.assert_allclose(dists, np.load(DATA / f"v9_ip_{name}_dists.npy"), rtol=1e-5, atol=1e-5)


def test_inner_product_hnsw_queries_while_adding():
    scaled, queries, truth = _ip_clustered()
    index = qann.HNSWIndex(32, metric=qann.Metric.InnerProduct, seed=1)
    index.add(scaled[:1000])
    errors = []

    def run(fn):
        try:
            fn()
        except Exception as e:
            errors.append(e)

    def adder():
        for start in range(1000, 5000, 500):
            index.add(scaled[start:start + 500])

    def querier():
        for _ in range(20):
            ids, dists = index.batch_query(queries, 10)
            assert (np.diff(dists, axis=1) >= 0).all()
            assert ((ids >= 0) & (ids < 5000)).all()

    threads = [threading.Thread(target=run, args=(fn,)) for fn in (adder, querier, querier)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    index.ef_search = 200
    ids, _ = index.batch_query(queries, 10)
    assert np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, truth)]) >= 0.95
