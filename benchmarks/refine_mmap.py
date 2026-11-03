"""RefineIndex with mapped raw vectors (qann.load(path, mmap=True), issue #58): cold and warm
page cache, with and without the MADV_WILLNEED hint, against the index read into RAM.

Linux only: the cold runs evict the file from the page cache with posix_fadvise(DONTNEED),
which needs no root. Mapped pages can't be evicted, so every cold run starts from a fresh load.

    python benchmarks/refine_mmap.py --file /path/on/the/disk/to/test.qann
"""

import argparse
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

import qann


def make_data(n, dim, nq, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((1000, dim), dtype=np.float32) * 3
    def draw(m):
        return centers[rng.integers(0, len(centers), m)] + rng.standard_normal((m, dim), dtype=np.float32)
    return draw(n), draw(nq)


def build(args):
    x, q = make_data(args.n, args.dim, args.queries)
    ivf = qann.IVFIndex(args.dim, nlist=args.nlist, nprobe=args.nprobe)
    ivf.enable_pq(args.dim // 4, fast_scan=True)
    t = time.perf_counter()
    ivf.train(x[:100_000], seed=1)
    index = qann.RefineIndex(ivf, k_factor=10)
    for i in range(0, len(x), 100_000):
        index.add(x[i:i + 100_000])
    index.save(args.file)
    np.save(args.file.with_suffix(".queries.npy"), q)
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


def latencies(index, q, k):
    out = np.empty(len(q))
    for i, row in enumerate(q):
        t = time.perf_counter()
        index.query(row, k)
        out[i] = time.perf_counter() - t
    return out * 1e6


def fmt(us):
    return f"mean {us.mean():8.0f}  p50 {np.percentile(us, 50):8.0f}  p99 {np.percentile(us, 99):8.0f} us"


def measure(args):
    """One mode (the WILLNEED env var is read once per process, so each mode is its own process)."""
    q = np.load(args.file.with_suffix(".queries.npy"))
    cold_q = q[:args.cold_queries]
    for kf in args.k_factors:
        for trial in range(args.trials):
            evict(args.file)
            t = time.perf_counter()
            index = qann.load(args.file, mmap=True)
            load_s = time.perf_counter() - t
            rss = rss_mb()
            index.k_factor = kf
            qann.set_num_threads(1)
            cold = latencies(index, cold_q, args.k)
            print(f"{args.mode:>8}  kf {kf:3}  cold #{trial}  {fmt(cold)}   load {load_s:5.2f} s, RSS {rss:6.0f} MB", flush=True)
            del index

        evict(args.file)
        index = qann.load(args.file, mmap=True)
        index.k_factor = kf
        qann.set_num_threads(0)
        t = time.perf_counter()
        index.batch_query(q, args.k)
        print(f"{args.mode:>8}  kf {kf:3}  cold batch, all threads: {len(q) / (time.perf_counter() - t):8.0f} QPS", flush=True)
        qann.set_num_threads(1)
        latencies(index, q, args.k)
        warm = latencies(index, q, args.k)
        print(f"{args.mode:>8}  kf {kf:3}  warm     {fmt(warm)}   ({1e6 / warm.mean():.0f} QPS), RSS {rss_mb():6.0f} MB",
              flush=True)


def reference(args):
    q = np.load(args.file.with_suffix(".queries.npy"))
    t = time.perf_counter()
    index = qann.load(args.file)
    print(f"     ram  load {time.perf_counter() - t:5.2f} s, RSS {rss_mb():6.0f} MB", flush=True)
    mapped = qann.load(args.file, mmap=True)
    for kf in args.k_factors:
        index.k_factor = mapped.k_factor = kf
        a, b = index.batch_query(q[:200], args.k), mapped.batch_query(q[:200], args.k)
        assert (a[0] == b[0]).all() and (a[1] == b[1]).all(), "mapped results differ from RAM"
        qann.set_num_threads(1)
        latencies(index, q, args.k)
        warm = latencies(index, q, args.k)
        print(f"     ram  kf {kf:3}  warm     {fmt(warm)}   ({1e6 / warm.mean():.0f} QPS)", flush=True)


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
    ap.add_argument("--k-factors", type=int, nargs="+", default=[10, 50])
    ap.add_argument("--mode", choices=["hint", "no-hint"], help=argparse.SUPPRESS)
    args = ap.parse_args()

    if sys.platform != "linux":
        sys.exit("cold runs need Linux (posix_fadvise DONTNEED)")
    if args.mode:
        measure(args)
        return
    if not args.file.exists():
        build(args)
    reference(args)
    for mode in ("hint", "no-hint"):
        env = dict(os.environ)
        env.pop("QANN_NO_WILLNEED", None)
        if mode == "no-hint":
            env["QANN_NO_WILLNEED"] = "1"
        subprocess.run([sys.executable, *sys.argv, "--mode", mode], env=env, check=True)


if __name__ == "__main__":
    main()
