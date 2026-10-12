"""HNSW M sweep (issue #33): which M reaches a recall target fastest, per dataset.

Run from the repo root after `pip install -e .`:
    python3 benchmarks/hnsw_m_sweep.py                 # glove-100 and sift
    python3 benchmarks/hnsw_m_sweep.py --dataset sift

For each dataset it builds one float HNSWIndex per (M, ef_construction),
sweeps ef_search for recall, then times the smallest ef that reaches each
recall target. The timings alternate between the graphs in rounds and keep
each one's best, so load swings on a busy machine hit every M alike. Queries
and builds run on all cores. Writes benchmarks/results/hnsw_m_sweep_<dataset>.csv.
"""

import argparse
import csv
import time

import qann

from bench_indexes import DATASETS, K, RESULTS_DIR, load_dataset, recall_at_k, time_queries

CONFIGS = [(16, 200), (32, 300), (48, 400)]  # (M, ef_construction)
EF_SEARCHES = [16, 32, 64, 128, 256, 512, 1024]
TARGETS = [0.90, 0.95, 0.99]
ROUNDS = 3


def sweep(dataset: str, seed: int):
    base, _, queries, gt = load_dataset(dataset)
    dim = base.shape[1]
    metric = getattr(qann.Metric, DATASETS[dataset]["metric"])

    graphs = {}
    for m, efc in CONFIGS:
        hnsw = qann.HNSWIndex(dim, M=m, ef_construction=efc, seed=seed, metric=metric)
        start = time.perf_counter()
        hnsw.add(base)
        build_s = time.perf_counter() - start
        recall = {}
        for ef in EF_SEARCHES:
            hnsw.ef_search = ef
            ids, _ = hnsw.batch_query(queries, K)
            recall[ef] = recall_at_k(ids, gt, K)
        graphs[m] = (hnsw, efc, build_s, recall)
        print(f"{dataset} M{m} (ef_construction {efc}): built in {build_s:.0f}s, recall by ef: "
              + ", ".join(f"{ef}: {r:.3f}" for ef, r in recall.items()), flush=True)

    # (M, target) -> smallest ef reaching it; then time those, alternating between graphs
    plan = {(m, t): next((ef for ef in EF_SEARCHES if g[3][ef] >= t), None) for m, g in graphs.items() for t in TARGETS}
    best = {key: 0.0 for key, ef in plan.items() if ef is not None}
    for _ in range(ROUNDS):
        for m, t in best:
            hnsw = graphs[m][0]
            hnsw.ef_search = plan[(m, t)]
            _, qps = time_queries(hnsw, queries, K, repeats=1)
            best[(m, t)] = max(best[(m, t)], qps)

    rows = []
    for m, (_, efc, build_s, recall) in graphs.items():
        for t in TARGETS:
            ef = plan[(m, t)]
            rows.append(dict(dataset=dataset, M=m, ef_construction=efc, build_s=round(build_s, 1), target=t, ef=ef,
                             recall=None if ef is None else round(recall[ef], 4),
                             qps=None if ef is None else round(best[(m, t)])))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["glove-100", "sift"], action="append")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    for dataset in args.dataset or ["glove-100", "sift"]:
        rows = sweep(dataset, args.seed)   # its graphs are freed on return, before the next dataset builds
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / f"hnsw_m_sweep_{dataset}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n{dataset}: QPS (best of {ROUNDS} alternating rounds) at the smallest ef reaching each recall")
        for r in rows:
            cell = "not reached" if r["ef"] is None else f"ef {r['ef']:<5} recall {r['recall']:.3f}  {r['qps']:>7} qps"
            print(f"  M{r['M']:<3} build {r['build_s']:>5.0f}s  target {r['target']}: {cell}")
        print(flush=True)


if __name__ == "__main__":
    main()
