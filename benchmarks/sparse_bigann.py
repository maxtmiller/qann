"""SparseIndex on the big-ann-benchmarks (NeurIPS'23) sparse track (issue #59).

SPLADE vectors of MS MARCO passages (about 127 nonzeros per passage over
~30k terms) and 6,980 dev queries (about 49 nonzeros), with exact top-10
ground truth by dot product. Reports recall@10 and QPS with MaxScore pruning
off and on, on one thread and on all of them, and checks that both modes give
the same distances. With --approx, the index keeps a forward index (twice the
memory) and each query_terms:list_depth:k_factor setting is run as well.

Sizes: small (100k passages, 67 MB download), 1M (667 MB), full (8.8M,
5.9 GB; needs about 20 GB of RAM). Files go in benchmarks/data/sparse/.

    python benchmarks/sparse_bigann.py --size 1M --download
    python benchmarks/sparse_bigann.py --size 1M --approx 10:2000:50 10:10000:10 10:10000:50
"""
import argparse
import gzip
import os
import pathlib
import shutil
import time
import urllib.request

import numpy as np
import scipy.sparse as sp

import qann

URL = "https://storage.googleapis.com/ann-challenge-sparse-vectors/csr/"
DATA = pathlib.Path(__file__).parent / "data" / "sparse"
BASE = {"small": "base_small", "1M": "base_1M", "full": "base_full"}


def fetch(name, download):
    path = DATA / name
    if path.exists():
        return path
    if not download:
        raise SystemExit(f"{path} is missing; pass --download to fetch it")
    DATA.mkdir(parents=True, exist_ok=True)
    gz = name.endswith(".csr")
    print(f"downloading {name}{'.gz' if gz else ''} ...", flush=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(URL + name + (".gz" if gz else "")) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(gzip.GzipFile(fileobj=response) if gz else response, out, 1 << 24)
    os.replace(tmp, path)
    return path


def read_csr(path, rows=None):
    # int64 nrow, ncol, nnz; int64 indptr[nrow + 1]; int32 indices[nnz]; float32 data[nnz]
    with open(path, "rb") as f:
        nrow, ncol, nnz = np.fromfile(f, dtype=np.int64, count=3)
        indptr = np.fromfile(f, dtype=np.int64, count=nrow + 1)
        indices = np.fromfile(f, dtype=np.int32, count=nnz)
        data = np.fromfile(f, dtype=np.float32, count=nnz)
    return sp.csr_array((data, indices, indptr), shape=(nrow, ncol))


def read_gt(path):
    with open(path, "rb") as f:
        n, k = np.fromfile(f, dtype=np.uint32, count=2)
        return np.fromfile(f, dtype=np.int32, count=n * k).reshape(n, k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=BASE, default="small")
    parser.add_argument("--download", action="store_true", help="fetch missing files")
    parser.add_argument("--rounds", type=int, default=3, help="timed runs per setting; the best counts")
    parser.add_argument("--approx", nargs="*", default=[], metavar="TERMS:DEPTH:KFACTOR",
                        help="approximate settings to run too, e.g. 10:10000:50")
    args = parser.parse_args()
    approx = [tuple(int(x) for x in a.split(":")) for a in args.approx]

    base = read_csr(fetch(BASE[args.size] + ".csr", args.download))
    queries = read_csr(fetch("queries.dev.csr", args.download))
    truth = read_gt(fetch(BASE[args.size] + ".dev.gt", args.download))

    index = qann.SparseIndex(forward_index=bool(approx))
    start = time.perf_counter()
    for first in range(0, base.shape[0], 500_000):  # in parts, so the copy qann makes stays small
        index.add(base[first:first + 500_000])
    built = time.perf_counter() - start
    del base
    print(f"{args.size}: {index.size():,} passages, {index.terms:,} terms, {index.postings:,} postings, "
          f"added in {built:.1f}s; {queries.shape[0]:,} queries, k=10; {os.cpu_count()} CPUs")

    def run(label):
        index.batch_query(queries[:200], 10)
        best = float("inf")
        for _ in range(args.rounds):
            start = time.perf_counter()
            ids, dists = index.batch_query(queries, 10)
            best = min(best, time.perf_counter() - start)
        recall = np.mean([len(set(row) & set(gt)) / 10 for row, gt in zip(ids, truth)])
        print(f"  {label:<56} {queries.shape[0] / best:9,.0f} QPS  ({best / queries.shape[0] * 1e3:.2f} ms/query)  "
              f"recall@10 {recall:.4f}", flush=True)
        return dists

    same = True
    for threads in (1, 0):
        qann.set_num_threads(threads)
        name = "1 thread" if threads else "all threads"
        index.query_terms, index.list_depth = 0, 0
        exact = {}
        for pruning in (False, True):
            index.pruning = pruning
            exact[pruning] = run(f"{name}, exact, pruning {'on' if pruning else 'off'}")
        same &= np.array_equal(exact[True], exact[False])
        for terms, depth, factor in approx:
            index.query_terms, index.list_depth, index.k_factor = terms, depth, factor
            run(f"{name}, query_terms {terms} list_depth {depth} k_factor {factor}")
    print(f"  same distances with pruning on and off: {same}")


if __name__ == "__main__":
    main()
