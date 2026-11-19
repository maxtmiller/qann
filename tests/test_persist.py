import os
import signal
import subprocess
import sys
import textwrap
import threading

import numpy as np
import pytest

import qann

DIM = 16


@pytest.fixture(scope="module")
def data():
    return np.random.default_rng(3).random((3_000, DIM), dtype=np.float32)


def results(index, queries, **kw):
    ids, dists = index.batch_query(queries, 10, **kw)
    return ids, dists


def test_writes_survive_reopen(tmp_path, data):
    ivf = qann.IVFIndex(DIM, 100)
    ivf.train(data)
    index = qann.RefineIndex(ivf, k_factor=4)
    index.persist(tmp_path / "store")
    assert index.persist_path == tmp_path / "store"

    index.add(data[:2_000], metadata={"g": np.arange(2_000) % 5, "s": ["a"] * 2_000})
    index.add(data[2_000:])
    index.remove(np.arange(0, 2_000, 7))
    index.update_metadata([1, 2], {"g": [50, None]})
    index.add_metadata_column("f", "float", default=0.5)
    index.add_metadata_column("t", "str", ids=[3, 4], values=["x", None])
    index.drop_metadata_column("s")
    expected = results(index, data[:20], where="g < 4")
    meta = index.get_metadata([1, 2, 3, 4])
    del index, ivf

    reopened = qann.open(tmp_path / "store")
    assert isinstance(reopened, qann.RefineIndex)
    assert reopened.size() == 3_000 - len(range(0, 2_000, 7))
    assert reopened.metadata_schema == {"g": "int", "f": "float", "t": "str"}
    got = reopened.get_metadata([1, 2, 3, 4])
    assert got["t"] == meta["t"] == [None, None, "x", None]
    assert np.ma.allequal(got["g"], meta["g"]) and got["g"].mask.tolist() == [False, True, False, False]
    for a, b in zip(results(reopened, data[:20], where="g < 4"), expected):
        np.testing.assert_array_equal(a, b)


def test_custom_ids_and_auto_ids_replay(tmp_path, data):
    flat = qann.FlatIndex(DIM, custom_ids=True)
    flat.persist(tmp_path / "custom")
    flat.add(data[:100], ids=np.arange(100) * 3 + 7)
    flat.remove([7, 10])
    flat.add(data[100:110], ids=[7] + list(range(1_000, 1_009)))  # re-adds a deleted id
    del flat
    flat = qann.open(tmp_path / "custom")
    assert flat.ids().tolist() == sorted([7] + [i * 3 + 7 for i in range(2, 100)] + list(range(1_000, 1_009)))

    auto = qann.FlatIndex(DIM)
    auto.persist(tmp_path / "auto")
    assert auto.add(data[:10]).tolist() == list(range(10))
    auto.remove([3])
    auto.compact()  # checkpoints: compaction isn't logged
    assert auto.add(data[10:12]).tolist() == [10, 11]
    del auto
    auto = qann.open(tmp_path / "auto")
    assert auto.ids().tolist() == [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]
    assert auto.add(data[12:13]).tolist() == [12]


def test_checkpoint_starts_a_new_generation(tmp_path, data):
    index = qann.FlatIndex(DIM)
    index.persist(tmp_path / "s")
    index.add(data[:500])
    assert sorted(os.listdir(tmp_path / "s")) == ["MANIFEST", "index-1.qann", "wal-1.log"]
    index.checkpoint()
    assert sorted(os.listdir(tmp_path / "s")) == ["MANIFEST", "index-2.qann", "wal-2.log"]
    assert os.path.getsize(tmp_path / "s" / "wal-2.log") == 16  # header only
    index.add(data[500:600])
    del index
    assert qann.open(tmp_path / "s").size() == 600


def test_training_checkpoints(tmp_path, data):
    ivf = qann.IVFIndex(DIM, 100)
    ivf.persist(tmp_path / "s")
    ivf.enable_pq(4)
    ivf.train(data)
    ivf.add(data)
    expected = results(ivf, data[:10])
    del ivf
    ivf = qann.open(tmp_path / "s")
    assert ivf.size() == len(data)
    for a, b in zip(results(ivf, data[:10]), expected):
        np.testing.assert_array_equal(a, b)


def test_mmap_open_and_checkpoint(tmp_path, data):
    ivf = qann.IVFIndex(DIM, 100)
    ivf.train(data)
    ivf.persist(tmp_path / "s")
    ivf.add(data[:2_000])
    ivf.checkpoint()
    del ivf
    ivf = qann.open(tmp_path / "s", mmap=True)
    assert ivf.mapped
    ivf.add(data[2_000:])
    ivf.checkpoint()  # saves from the mapped file it then deletes
    expected = results(ivf, data[:10])
    del ivf
    ivf = qann.open(tmp_path / "s", mmap=True)
    assert ivf.size() == len(data)
    for a, b in zip(results(ivf, data[:10]), expected):
        np.testing.assert_array_equal(a, b)


def test_sparse_and_hybrid_replay(tmp_path, data):
    sp = pytest.importorskip("scipy.sparse")
    rows = sp.random(200, 300, density=0.05, format="csr", dtype=np.float32, random_state=4)
    q = sp.random(5, 300, density=0.05, format="csr", dtype=np.float32, random_state=5)

    inv = qann.InvertedIndex()
    inv.persist(tmp_path / "inv")
    inv.add(rows, metadata={"g": np.arange(200) % 3})
    inv.remove([0, 1])
    expected = inv.batch_query(q, 5)
    del inv
    inv = qann.open(tmp_path / "inv")
    for a, b in zip(inv.batch_query(q, 5), expected):
        np.testing.assert_array_equal(a, b)

    h = qann.HybridIndex(qann.FlatIndex(DIM), qann.InvertedIndex(), custom_ids=True)
    h.persist(tmp_path / "hyb")
    h.add(data[:200], rows, ids=np.arange(200) + 1_000)
    h.remove([1_000])
    expected = h.batch_query(data[:5], q, 5)
    del h
    h = qann.open(tmp_path / "hyb")
    assert isinstance(h, qann.HybridIndex) and h.size() == 199
    for a, b in zip(h.batch_query(data[:5], q, 5), expected):
        np.testing.assert_array_equal(a, b)


def crc32c(data):
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x82F63B78 if crc & 1 else crc >> 1
    return crc ^ 0xFFFFFFFF


def test_log_records_use_standard_crc32c(tmp_path, data):
    # the hardware (SSE4.2 / ARM) and table versions must write the same CRCs, so logs move between machines
    assert crc32c(b"123456789") == 0xE3069283
    index = qann.FlatIndex(DIM)
    index.persist(tmp_path / "s")
    index.add(data[:3])
    del index
    log = (tmp_path / "s" / "wal-1.log").read_bytes()
    length = int.from_bytes(log[16:20], "little")
    assert int.from_bytes(log[20:24], "little") == crc32c(log[24:24 + length])
    assert len(log) == 24 + length


def test_torn_and_corrupt_tail_is_dropped(tmp_path, data):
    index = qann.FlatIndex(DIM)
    index.persist(tmp_path / "s")
    index.add(data[:10])
    index.add(data[10:20])
    del index
    log = tmp_path / "s" / "wal-1.log"
    full = log.read_bytes()

    log.write_bytes(full[:-5])  # cut inside the second record
    assert qann.open(tmp_path / "s").size() == 10
    assert len(log.read_bytes()) < len(full) - 5  # truncated to the last whole record

    log.write_bytes(full[:-1] + bytes([full[-1] ^ 0xFF]))  # flipped bit: CRC fails
    index = qann.open(tmp_path / "s")
    assert index.size() == 10
    index.add(data[20:25])  # appends after the dropped record
    del index
    assert qann.open(tmp_path / "s").size() == 15


def test_crash_keeps_every_acknowledged_write(tmp_path):
    # a child process persists writes and reports each acknowledged batch; it is killed mid-stream
    store = tmp_path / "s"
    script = textwrap.dedent(f"""
        import sys, numpy as np, qann
        index = qann.FlatIndex({DIM}, custom_ids=True)
        index.persist({str(store)!r})
        rng = np.random.default_rng(0)
        for i in range(100_000):
            index.add(rng.random((50, {DIM}), dtype=np.float32), ids=np.arange(50) + 50 * i)
            print(i, flush=True)
    """)
    env = dict(os.environ)
    child = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True, env=env)
    acked = -1
    for line in child.stdout:
        acked = int(line)
        if acked >= 40:
            break
    child.send_signal(signal.SIGKILL)
    child.wait()
    index = qann.open(store)
    assert index.size() >= 50 * (acked + 1)
    assert index.size() % 50 == 0  # whole batches only
    assert index.ids().tolist() == list(range(index.size()))


def test_concurrent_writers_share_syncs(tmp_path, data):
    index = qann.FlatIndex(DIM, custom_ids=True)
    index.persist(tmp_path / "s")

    def writer(t):
        for i in range(20):
            start = (t * 20 + i) * 5
            index.add(data[start:start + 5], ids=np.arange(start, start + 5))

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert index.size() == 400
    del index
    assert qann.open(tmp_path / "s").ids().tolist() == list(range(400))


def test_refusals(tmp_path, data):
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "x").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        qann.FlatIndex(DIM).persist(tmp_path / "full")

    index = qann.FlatIndex(DIM)
    index.persist(tmp_path / "a")
    with pytest.raises(ValueError, match="already persisted"):
        index.persist(tmp_path / "b")
    with pytest.raises(ValueError, match="persisted"):
        qann.RefineIndex(index, k_factor=2)

    base = qann.FlatIndex(DIM)
    refine = qann.RefineIndex(base, k_factor=2)
    with pytest.raises(ValueError, match="wrapped"):
        base.persist(tmp_path / "c")
    del refine

    with pytest.raises(ValueError, match="not persisted"):
        qann.FlatIndex(DIM).checkpoint()
    assert qann.FlatIndex(DIM).persist_path is None
    with pytest.raises(RuntimeError, match="MANIFEST"):
        qann.open(tmp_path / "missing")

    # a failed write logs nothing
    index.add(data[:5])
    with pytest.raises(ValueError):
        index.update_metadata([99], {"g": [1]})
    del index
    assert qann.open(tmp_path / "a").size() == 5
