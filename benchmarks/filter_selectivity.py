"""Filtered search (issue #37): recall and QPS as the allowed set shrinks.

Run from the repo root after `pip install -e .`:
    python3 benchmarks/filter_selectivity.py                      # SIFT1M, 1000 queries
    python3 benchmarks/filter_selectivity.py --dataset siftsmall

For each selectivity (the fraction of vectors allowed) it draws a random
allowed set, takes the exact filtered top 10 from FlatIndex as ground truth,
and times every index on batch_query(..., allowed=...). Timings alternate
between the indexes in rounds and keep each one's best, so load swings on a
busy machine hit every index alike. Writes
benchmarks/results/filter_selectivity_<dataset>.csv.
"""

import argparse
import csv
import math
import time

import numpy as np

import qann

from bench_indexes import HNSW_EF_CONSTRUCTION, HNSW_M, K, RESULTS_DIR, load_dataset

SELECTIVITIES = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
NPROBE = 16
EF_SEARCH = 64
ROUNDS = 3


def recall(ids, gt):
    # gt rows are padded with -1 when fewer than K vectors are allowed
    hits = total = 0
    for got, truth in zip(ids, gt):
        truth = set(truth[truth >= 0].tolist())
        hits += len(truth & set(got.tolist()))
        total += len(truth)
    return hits / total


def build(base, learn, seed):
    n, dim = base.shape
    nlist = round(math.sqrt(n))
    indexes = {}

    flat = qann.FlatIndex(dim)
    flat.add(base)
    indexes["Flat"] = flat

    ivf = qann.IVFIndex(dim, nlist, NPROBE)
    ivf.train(learn, seed=seed)
    ivf.add(base)
    indexes[f"IVF nprobe={NPROBE}"] = ivf

    fs = qann.IVFIndex(dim, nlist, NPROBE)
    fs.enable_pq(dim // 2, fast_scan=True)
    fs.train(learn, seed=seed)
    fs_refine = qann.RefineIndex(fs, k_factor=4)
    fs_refine.add(base)
    indexes[f"IVF-PQ{dim // 2}x4fs+R4 nprobe={NPROBE}"] = fs_refine

    hnsw = qann.HNSWIndex(dim, M=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION, seed=seed)
    hnsw.add(base)
    hnsw.ef_search = EF_SEARCH
    indexes[f"HNSW ef={EF_SEARCH}"] = hnsw

    sq = qann.HNSWIndex(dim, M=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION, seed=seed)
    sq.enable_sq(8)
    sq.train(learn)
    sq.ef_search = EF_SEARCH
    sq_refine = qann.RefineIndex(sq, k_factor=2)
    sq_refine.add(base)
    indexes[f"HNSW-SQ8+R2 ef={EF_SEARCH}"] = sq_refine

    # the bases must outlive their RefineIndex wrappers
    return indexes, (fs, sq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["siftsmall", "sift"], default="sift")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    base, learn, queries, _ = load_dataset(args.dataset)
    queries = queries[: args.queries]
    n = len(base)
    start = time.perf_counter()
    indexes, _bases = build(base, learn, args.seed)
    print(f"{args.dataset}: built {len(indexes)} indexes in {time.perf_counter() - start:.0f}s, "
          f"{len(queries)} queries\n", flush=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    for sel in SELECTIVITIES:
        allowed = np.sort(rng.choice(n, max(1, round(n * sel)), replace=False)).astype(np.int64)
        gt, _ = indexes["Flat"].batch_query(queries, K, allowed=allowed)

        best = {name: math.inf for name in indexes}
        found = {}
        for _ in range(ROUNDS):
            for name, index in indexes.items():
                t = time.perf_counter()
                found[name], _ = index.batch_query(queries, K, allowed=allowed)
                best[name] = min(best[name], time.perf_counter() - t)

        print(f"{sel:.1%} allowed ({len(allowed)} vectors)")
        for name in indexes:
            r, qps = recall(found[name], gt), len(queries) / best[name]
            rows.append(dict(dataset=args.dataset, selectivity=sel, allowed=len(allowed), index=name,
                             recall=round(r, 4), qps=round(qps)))
            print(f"  {name:<32} recall@{K} {r:.3f}   {qps:>9.0f} qps")
        print(flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"filter_selectivity_{args.dataset}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
