"""Single-query where= cost (issue #72 step 3): filter evaluation vs search.

Run from the repo root after `pip install -e .`:
    python3 benchmarks/where_cost.py                      # SIFT1M
    python3 benchmarks/where_cost.py --dataset siftsmall

Gives every vector synthetic metadata (category: 10 strings, brand: 1000
strings, price: float in [0, 100), year: int 2000-2024), then for each filter
times, per single query on one thread:
  parse   qann.where(sql)
  eval    the filter's evaluation over every slot: query(where=f AND price < 0)
          minus query(where=price < 0). Nothing matches either, so the search
          is skipped and only evaluation is left
  search  query(where=f) per index, with the filter already parsed
and prints eval as a share of each index's filtered query. Configs alternate in
rounds and keep each one's best, so machine load hits all of them alike.
Writes benchmarks/results/where_cost_<dataset>.csv.
"""

import argparse
import csv
import math
import time

import numpy as np

import qann

from bench_indexes import HNSW_EF_CONSTRUCTION, HNSW_M, K, RESULTS_DIR, load_dataset

NPROBE = 16
EF_SEARCH = 64
ROUNDS = 5

BRANDS = ", ".join(f"'b{i}'" for i in range(0, 1000, 50))
FILTERS = [
    "category = 'c3'",
    "brand = 'b17'",
    "price < 10",
    "year BETWEEN 2010 AND 2012",
    "category IN ('c1', 'c2', 'c3')",
    f"brand IN ({BRANDS})",
    "category = 'c3' AND price < 50",
    "category = 'c3' AND price < 50 AND year >= 2015",
    "(category = 'c3' OR brand = 'b17') AND NOT (price >= 90)",
]
NOTHING = "price < 0"


def metadata(n, rng):
    return {
        "category": [f"c{i}" for i in rng.integers(0, 10, n)],
        "brand": [f"b{i}" for i in rng.integers(0, 1000, n)],
        "price": rng.uniform(0, 100, n),
        "year": rng.integers(2000, 2025, n),
    }


def build(base, learn, meta, seed):
    n, dim = base.shape
    indexes = {}

    hnsw = qann.HNSWIndex(dim, M=HNSW_M, ef_construction=HNSW_EF_CONSTRUCTION, seed=seed)
    hnsw.add(base, metadata=meta)
    hnsw.ef_search = EF_SEARCH
    indexes[f"HNSW ef={EF_SEARCH}"] = hnsw

    ivf = qann.IVFIndex(dim, round(math.sqrt(n)), NPROBE)
    ivf.train(learn, seed=seed)
    ivf.add(base, metadata=meta)
    indexes[f"IVF nprobe={NPROBE}"] = ivf

    fs = qann.IVFIndex(dim, round(math.sqrt(n)), NPROBE)
    fs.enable_pq(dim // 2, fast_scan=True)
    fs.train(learn, seed=seed)
    fs_refine = qann.RefineIndex(fs, k_factor=4)
    fs_refine.add(base, metadata=meta)
    indexes[f"IVF-PQ{dim // 2}x4fs+R4 nprobe={NPROBE}"] = fs_refine

    # the base must outlive its RefineIndex wrapper
    return indexes, fs


def per_query(fn, queries):
    t = time.perf_counter()
    for q in queries:
        fn(q)
    return (time.perf_counter() - t) / len(queries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["siftsmall", "sift"], default="sift")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    base, learn, queries, _ = load_dataset(args.dataset)
    queries = queries[: args.queries]
    n = len(base)
    meta = metadata(n, np.random.default_rng(args.seed))
    start = time.perf_counter()
    indexes, _base = build(base, learn, meta, args.seed)
    print(f"{args.dataset}: built {len(indexes)} indexes in {time.perf_counter() - start:.0f}s, "
          f"{len(queries)} single queries, 1 thread\n", flush=True)
    qann.set_num_threads(1)

    probe = next(iter(indexes.values()))
    nothing = qann.where(NOTHING)
    configs = {"unfiltered": {name: (lambda q, i=index: i.query(q, K)) for name, index in indexes.items()}}
    configs["nothing"] = {"eval": lambda q: probe.query(q, K, where=nothing)}
    for sql in FILTERS:
        expr, empty = qann.where(sql), qann.where(f"({sql}) AND {NOTHING}")
        configs[sql] = {"parse": lambda q, s=sql: qann.where(s),
                        "eval": lambda q, e=empty: probe.query(q, K, where=e)}
        for name, index in indexes.items():
            configs[sql][name] = lambda q, i=index, e=expr: i.query(q, K, where=e)

    best = {(c, part): math.inf for c, parts in configs.items() for part in parts}
    for _ in range(ROUNDS):
        for c, parts in configs.items():
            for part, fn in parts.items():
                best[c, part] = min(best[c, part], per_query(fn, queries))

    us = {key: t * 1e6 for key, t in best.items()}
    base_eval = us["nothing", "eval"]
    print(f"empty filter (one leaf + live mask + call overhead): {base_eval:.0f} us")
    print("unfiltered: " + ", ".join(f"{name} {us['unfiltered', name]:.0f} us" for name in indexes) + "\n")

    rows = []
    for sql in FILTERS:
        evaluation = max(0.0, us[sql, "eval"] - base_eval)
        print(f"{sql}")
        print(f"  parse {us[sql, 'parse']:.1f} us, eval {evaluation:.0f} us")
        for name in indexes:
            total = us[sql, name]
            print(f"  {name:<32} {total:>7.0f} us/query   eval share {evaluation / total:>5.1%}")
            rows.append(dict(dataset=args.dataset, filter=sql, index=name, parse_us=round(us[sql, "parse"], 1),
                             eval_us=round(evaluation, 1), query_us=round(total, 1),
                             unfiltered_us=round(us["unfiltered", name], 1)))
        print(flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"where_cost_{args.dataset}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
