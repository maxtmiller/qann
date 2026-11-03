"""Indexes loaded with qann.load(path, mmap=True) (issue #58): cold and warm page cache,
against the index read into RAM. --kind refine maps a RefineIndex's raw vectors over a
fast-scan IVF base (swept: k_factor); ivf and ivf-sq8 map an IVFIndex's float or SQ8 lists
(swept: nprobe); inverted and sparse-approx map the posting lists of an InvertedIndex
(swept: pruning 1 / 0) or a SparseApproxIndex and its forward index (swept: list_depth),
built from the big-ann sparse track's SPLADE vectors (--size, files as for sparse_bigann.py).

Linux only: the cold runs evict the file from the page cache with posix_fadvise(DONTNEED),
which needs no root. Mapped pages can't be evicted, so every cold run starts from a fresh load.

    python benchmarks/refine_mmap.py --file /path/on/the/disk/to/test.qann
    python benchmarks/refine_mmap.py --kind ivf --file /path/on/the/disk/to/ivf.qann
    python benchmarks/refine_mmap.py --kind inverted --file /path/on/the/disk/to/sparse.qann
"""

import argparse
import os
import pathlib
import sys
import time

import numpy as np
import scipy.sparse as sp

import qann

SPARSE = ("inverted", "sparse-approx")
SWEEP = {"refine": ("kf", [10, 50]), "ivf": ("np", [16, 64]), "ivf-sq8": ("np", [16, 64]),
         "inverted": ("prune", [1, 0]), "sparse-approx": ("depth", [2000, 10000])}


def make_data(n, dim, nq, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((1000, dim), dtype=np.float32) * 3
    def draw(m):
        return centers[rng.integers(0, len(centers), m)] + rng.standard_normal((m, dim), dtype=np.float32)
    return draw(n), draw(nq)


def queries_path(args):
    return args.file.with_suffix(".queries.npz" if args.kind in SPARSE else ".queries.npy")


def build_sparse(args):
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import sparse_bigann as bigann
    base = bigann.read_csr(bigann.fetch(bigann.BASE[args.size] + ".csr", False))
    q = bigann.read_csr(bigann.fetch("queries.dev.csr", False))[:args.queries]
    index = qann.InvertedIndex() if args.kind == "inverted" else qann.SparseApproxIndex()
    t = time.perf_counter()
    for first in range(0, base.shape[0], 500_000):
        index.add(base[first:first + 500_000])
    index.save(args.file)
    sp.save_npz(queries_path(args), sp.csr_matrix(q))
    print(f"built {index.size():,} passages, {index.postings:,} postings in {time.perf_counter() - t:.0f} s, "
          f"file {args.file.stat().st_size / 1e9:.2f} GB", flush=True)


def build(args):
    if args.kind in SPARSE:
        return build_sparse(args)
    x, q = make_data(args.n, args.dim, args.queries)
    ivf = qann.IVFIndex(args.dim, nlist=args.nlist, nprobe=args.nprobe)
    if args.kind == "refine":
        ivf.enable_pq(args.dim // 4, fast_scan=True)
    elif args.kind == "ivf-sq8":
        ivf.enable_sq(8)
    t = time.perf_counter()
    ivf.train(x[:100_000], seed=1)
    index = qann.RefineIndex(ivf, k_factor=10) if args.kind == "refine" else ivf
    for i in range(0, len(x), 100_000):
        index.add(x[i:i + 100_000])
    index.save(args.file)
    np.save(queries_path(args), q)
    print(f"built {args.n} x {args.dim} in {time.perf_counter() - t:.0f} s, file {args.file.stat().st_size / 1e9:.2f} GB",
          flush=True)


def evict(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def load_queries(args):
    """The queries, and each as the one-query input query() takes (a vector, or a 1-row matrix)."""
    if args.kind in SPARSE:
        q = sp.load_npz(queries_path(args)).tocsr()
        return q, [q[i:i + 1] for i in range(q.shape[0])]
    q = np.load(queries_path(args))
    return q, list(q)


def latencies(index, rows, k):
    out = np.empty(len(rows))
    for i, row in enumerate(rows):
        t = time.perf_counter()
        index.query(row, k)
        out[i] = time.perf_counter() - t
    return out * 1e6


def fmt(us):
    return f"mean {us.mean():8.0f}  p50 {np.percentile(us, 50):8.0f}  p99 {np.percentile(us, 99):8.0f} us"


def setting(args, index, v):
    """Sets the swept value (see SWEEP)."""
    if args.kind == "refine":
        index.k_factor = v
    elif args.kind == "inverted":
        index.pruning = bool(v)
    elif args.kind == "sparse-approx":
        index.list_depth = v
    else:
        index.nprobe = v


def measure(args):
    q, rows = load_queries(args)
    cold_q = rows[:args.cold_queries]
    for kf in args.sweep:
        for trial in range(args.trials):
            evict(args.file)
            t = time.perf_counter()
            index = qann.load(args.file, mmap=True)
            load_s = time.perf_counter() - t
            rss = rss_mb()
            setting(args, index, kf)
            qann.set_num_threads(1)
            cold = latencies(index, cold_q, args.k)
            print(f"  mapped  {args.label} {kf:4}  cold #{trial}  {fmt(cold)}   load {load_s:5.2f} s, RSS {rss:6.0f} MB", flush=True)
            del index

        evict(args.file)
        index = qann.load(args.file, mmap=True)
        setting(args, index, kf)
        qann.set_num_threads(0)
        t = time.perf_counter()
        index.batch_query(q, args.k)
        print(f"  mapped  {args.label} {kf:4}  cold batch, all threads: {len(rows) / (time.perf_counter() - t):8.0f} QPS", flush=True)
        qann.set_num_threads(1)
        latencies(index, rows, args.k)
        warm = latencies(index, rows, args.k)
        print(f"  mapped  {args.label} {kf:4}  warm     {fmt(warm)}   ({1e6 / warm.mean():.0f} QPS), RSS {rss_mb():6.0f} MB",
              flush=True)


def reference(args):
    q, rows = load_queries(args)
    t = time.perf_counter()
    index = qann.load(args.file)
    print(f"     ram  load {time.perf_counter() - t:5.2f} s, RSS {rss_mb():6.0f} MB", flush=True)
    mapped = qann.load(args.file, mmap=True)
    for kf in args.sweep:
        setting(args, index, kf)
        setting(args, mapped, kf)
        a, b = index.batch_query(q[:200], args.k), mapped.batch_query(q[:200], args.k)
        assert (a[0] == b[0]).all() and (a[1] == b[1]).all(), "mapped results differ from RAM"
        qann.set_num_threads(1)
        latencies(index, rows, args.k)
        warm = latencies(index, rows, args.k)
        print(f"     ram  {args.label} {kf:4}  warm     {fmt(warm)}   ({1e6 / warm.mean():.0f} QPS)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=pathlib.Path, required=True, help="index file, built if missing")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--cold-queries", type=int, default=300, help="queries timed right after each fresh load")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--kind", choices=list(SWEEP), default="refine")
    ap.add_argument("--size", choices=["small", "1M", "full"], default="1M", help="SPLADE set for the sparse kinds")
    ap.add_argument("--sweep", type=int, nargs="+", help="the swept setting's values (defaults in SWEEP)")
    args = ap.parse_args()
    args.label, default_sweep = SWEEP[args.kind]
    args.sweep = args.sweep or default_sweep

    if sys.platform != "linux":
        sys.exit("cold runs need Linux (posix_fadvise DONTNEED)")
    if not args.file.exists():
        build(args)
    reference(args)
    measure(args)


if __name__ == "__main__":
    main()
