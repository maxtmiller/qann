"""Filtered HNSW recall and latency with a different filter on every query (issue #153).

Each query filters its own random window of a 1,000-value int column, so the
allowed vectors aren't already in cache from the previous query, as with
per-user or per-document filters. Reports recall@10 against exact filtered
search and µs per query on one thread, at ef_search 32 and 64, from 0.2% to
20% allowed. The index (M 16, ef_construction 200) is built once and cached
next to the dataset.

    python benchmarks/hnsw_filter_cold.py sift                                   # SIFT1M
    python benchmarks/hnsw_filter_cold.py data/arxiv-nomic-768-normalized.hdf5   # VIBE / ann-benchmarks layout
    python benchmarks/hnsw_filter_cold.py FILE --queries 300
"""
import argparse
import pathlib
import time

import numpy as np

import qann

K = 10
SELECTIVITIES = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
CATS = 1000


def load(name):
    if name == "sift":
        from bench_indexes import load_dataset
        base, _, queries, _ = load_dataset("sift")
        return base, queries, qann.Metric.L2, pathlib.Path(__file__).parent / "data" / "sift" / "sift.hnsw.qann"
    import h5py
    path = pathlib.Path(name)
    with h5py.File(path) as f:
        base, queries = np.asarray(f["train"], dtype=np.float32), np.asarray(f["test"], dtype=np.float32)
    metric = qann.Metric.L2 if "euclidean" in path.name or "l2" in path.name else qann.Metric.Cosine
    return base, queries, metric, path.with_suffix(".hnsw.qann")


def window(sel, offset):
    w = round(sel * CATS)
    lo = int(offset) % (CATS - w + 1)
    return lo, lo + w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--queries", type=int, default=1000)
    args = parser.parse_args()

    base, queries, metric, cache = load(args.dataset)
    queries = queries[: args.queries]
    cats = np.random.default_rng(1).integers(0, CATS, len(base))
    offsets = np.random.default_rng(7).integers(0, CATS, len(queries))

    if cache.exists():
        index = qann.load(cache)
    else:
        index = qann.HNSWIndex(base.shape[1], M=16, ef_construction=200, seed=1, metric=metric)
        start = time.perf_counter()
        index.add(base, metadata={"cat": cats})
        print(f"built {len(base):,} in {time.perf_counter() - start:.0f} s", flush=True)
        index.save(cache)

    # exact filtered top 10, by brute force over each query's allowed rows
    unit = metric == qann.Metric.Cosine
    if unit:
        base /= np.linalg.norm(base, axis=1, keepdims=True)
    truth = {}
    for sel in SELECTIVITIES:
        rows = []
        for q, o in zip(queries, offsets):
            lo, hi = window(sel, o)
            ok = np.flatnonzero((cats >= lo) & (cats < hi))
            d = -(base[ok] @ (q / np.linalg.norm(q))) if unit else ((base[ok] - q) ** 2).sum(1)
            rows.append(set(ok[np.argsort(d, kind="stable")[:K]].tolist()))
        truth[sel] = rows
    del base

    qann.set_num_threads(1)
    print(f"qann {qann.__version__}, {index.size():,} x {index.dim()}, {len(queries)} queries, one thread")
    for ef in (32, 64):
        index.ef_search = ef
        cells = []
        for sel in SELECTIVITIES:
            hits, start = 0, time.perf_counter()
            for q, o, want in zip(queries, offsets, truth[sel]):
                ids, _ = index.query(q, K, where=("cat >= ? AND cat < ?", list(window(sel, o))))
                hits += len(want & set(ids.tolist()))
            us = (time.perf_counter() - start) / len(queries) * 1e6
            cells.append(f"{sel:.1%}: {hits / (K * len(queries)):.3f} / {us:,.0f} µs")
        print(f"ef_search {ef}:  " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    main()
