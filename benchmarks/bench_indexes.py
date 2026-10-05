"""Recall vs QPS benchmark for FlatIndex, IVFIndex and IVFIndex+PQ on SIFT.

Run from the repo root after a Release build:
    PYTHONPATH=build python3 benchmarks/bench_indexes.py                  # siftsmall
    PYTHONPATH=build python3 benchmarks/bench_indexes.py --dataset sift   # SIFT1M

Recall regression gate (used by CI): --write-baseline saves recall for every
row with a fixed seed; --check fails if any row's recall drops below it.
    PYTHONPATH=build python3 benchmarks/bench_indexes.py --seed 1 --write-baseline benchmarks/baseline_siftsmall.json
    PYTHONPATH=build python3 benchmarks/bench_indexes.py --check benchmarks/baseline_siftsmall.json

Datasets go in benchmarks/data/<name>/ (see ftp://ftp.irisa.fr/local/texmex/corpus/).
batch_query and PQ training run on all cores by default (detail::parallel_for);
--threads N limits that (--threads 1 is fully serial). add uses add_batch.
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import vecengine as ve

BENCH_DIR = Path(__file__).parent
DATA_DIR = BENCH_DIR / "data"
RESULTS_DIR = BENCH_DIR / "results"
K = 10
NPROBES = [1, 2, 4, 8, 16, 32, 64, 128]
PQ_SUBSPACES = [8, 16, 32]
REFINE_K_FACTORS = [4, 16]  # re-rank k * k_factor ADC candidates with exact distances


def read_vecs(path: Path, dtype) -> np.ndarray:
    # .fvecs/.ivecs: each row is an int32 dim followed by dim 4-byte values.
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    return np.ascontiguousarray(raw.reshape(-1, dim + 1)[:, 1:].view(dtype))


def load_dataset(name: str):
    d = DATA_DIR / name
    base = read_vecs(d / f"{name}_base.fvecs", np.float32)
    learn = read_vecs(d / f"{name}_learn.fvecs", np.float32)
    queries = read_vecs(d / f"{name}_query.fvecs", np.float32)
    gt = read_vecs(d / f"{name}_groundtruth.ivecs", np.int32)
    return base, learn, queries, gt


def recall_at_k(ids_per_query, gt: np.ndarray, k: int) -> float:
    hits = sum(len(set(ids[:k].tolist()) & set(gt[i, :k].tolist())) for i, ids in enumerate(ids_per_query))
    return hits / (len(ids_per_query) * k)


def time_queries(index, queries: np.ndarray, k: int, repeats: int = 3):
    # Best of `repeats` full passes: the first pass after a build or setting
    # change pays cold caches, so a short warm-up alone undercounts QPS.
    best = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        ids, _ = index.batch_query(queries, k)
        best = min(best, time.perf_counter() - start)
    return ids, len(queries) / best


def build_ivf(dim, nlist, base, learn, pq_m=None, seed=None):
    # With PQ, vectors are added through a RefineIndex so one build serves both
    # the plain PQ rows (query idx) and the re-ranked rows (query refine).
    idx = ve.IVFIndex(dim, nlist)
    if pq_m is not None:
        idx.enable_pq(pq_m)
    start = time.perf_counter()
    idx.train(learn, seed=seed)
    train_s = time.perf_counter() - start
    refine = ve.RefineIndex(idx) if pq_m is not None else None
    start = time.perf_counter()
    (refine or idx).add(base)
    add_s = time.perf_counter() - start
    return idx, refine, train_s, add_s


def run(dataset: str, max_queries: int | None, seed: int | None = None):
    base, learn, queries, gt = load_dataset(dataset)
    if max_queries:
        queries, gt = queries[:max_queries], gt[:max_queries]
    n, dim = base.shape
    nlist = min(65535, max(100, round(math.sqrt(n))))
    print(f"{dataset}: base {base.shape}, learn {learn.shape}, queries {queries.shape}, nlist {nlist}\n")

    rows = []

    def record(name, params, ids, qps, bytes_per_vec, train_s=0.0, add_s=0.0):
        r = recall_at_k(ids, gt, K)
        rows.append(dict(index=name, params=params, recall=r, qps=qps,
                         bytes_per_vec=bytes_per_vec, train_s=train_s, add_s=add_s))
        print(f"  {name:<18} {params:<14} recall@{K} {r:.3f}   {qps:>9.0f} qps   {bytes_per_vec:>4} B/vec")

    print("Flat")
    flat = ve.FlatIndex(dim)
    start = time.perf_counter()
    flat.add(base)
    add_s = time.perf_counter() - start
    ids, qps = time_queries(flat, queries, K)
    record("Flat", "-", ids, qps, dim * 4, add_s=add_s)

    configs = [("IVF", None)] + [(f"IVF-PQ{m}", m) for m in PQ_SUBSPACES]
    for name, m in configs:
        idx, refine, train_s, add_s = build_ivf(dim, nlist, base, learn, m, seed)
        print(f"{name}  (train {train_s:.1f}s, add {add_s:.1f}s)")
        bytes_per_vec = (dim * 4 if m is None else m) + 4  # + uint32 id
        modes = [("", None)] if m is None else [("-ADC", ve.PQDistance.ADC), ("-SDC", ve.PQDistance.SDC)]
        for suffix, mode in modes:
            label = name + suffix
            if mode is not None:
                idx.pq_distance = mode
            for nprobe in (p for p in NPROBES if p <= nlist):
                idx.nprobe = nprobe
                ids, qps = time_queries(idx, queries, K)
                record(label, f"nprobe={nprobe}", ids, qps, bytes_per_vec, train_s, add_s)
        if refine is not None:
            idx.pq_distance = ve.PQDistance.ADC
            for kf in REFINE_K_FACTORS:
                refine.k_factor = kf
                for nprobe in (p for p in NPROBES if p <= nlist):
                    idx.nprobe = nprobe
                    ids, qps = time_queries(refine, queries, K)
                    record(f"{name}-ADC+R{kf}", f"nprobe={nprobe}", ids, qps,
                           bytes_per_vec + dim * 4, train_s, add_s)
        print()

    return rows


def write_csv(rows, path: Path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot(rows, path: Path, title: str, threads: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name in dict.fromkeys(r["index"] for r in rows):
        pts = [(r["recall"], r["qps"]) for r in rows if r["index"] == name]  # nprobe order
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", markersize=3 if len(pts) > 1 else 6, label=name)
    ax.set_yscale("log")
    ax.set_xlabel(f"recall@{K}")
    ax.set_ylabel(f"queries / second ({threads} threads, log)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def row_key(r) -> str:
    return f"{r['index']} {r['params']}"


def write_baseline(rows, path: Path, dataset: str, seed: int):
    recall = {row_key(r): round(r["recall"], 4) for r in rows}
    path.write_text(json.dumps({"dataset": dataset, "seed": seed, "recall": recall}, indent=2) + "\n")
    print(f"wrote baseline {path}")


def check_baseline(rows, baseline: dict, tolerance: float) -> bool:
    current = {row_key(r): r["recall"] for r in rows}
    failures = []
    for key, expected in baseline["recall"].items():
        got = current.get(key)
        if got is None:
            failures.append(f"  {key}: missing from this run")
        elif got < expected - tolerance:
            failures.append(f"  {key}: recall {got:.4f} < baseline {expected:.4f} - {tolerance}")
    if failures:
        print(f"recall regression ({len(failures)} rows):")
        print("\n".join(failures))
        return False
    print(f"recall check passed: {len(baseline['recall'])} rows within {tolerance} of baseline")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="siftsmall", help="siftsmall or sift")
    parser.add_argument("--max-queries", type=int, default=None, help="use only the first N queries")
    parser.add_argument("--seed", type=int, default=None, help="seed k-means for reproducible recall")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="where to write the CSV and plot")
    parser.add_argument("--write-baseline", type=Path, help="save per-row recall to this JSON (requires --seed)")
    parser.add_argument("--check", type=Path, help="fail if recall drops below this baseline JSON (uses its dataset and seed)")
    parser.add_argument("--tolerance", type=float, default=0.02, help="allowed recall drop for --check")
    parser.add_argument("--threads", type=int, default=0, help="threads for vecengine's parallel work; 0 = one per core")
    args = parser.parse_args()

    baseline = None
    if args.check:
        baseline = json.loads(args.check.read_text())
        args.dataset, args.seed = baseline["dataset"], baseline["seed"]
    if args.write_baseline and args.seed is None:
        parser.error("--write-baseline requires --seed")

    ve.set_num_threads(args.threads)
    print(f"threads: {ve.num_threads()}")
    rows = run(args.dataset, args.max_queries, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.dataset}.csv"
    write_csv(rows, csv_path)
    print(f"wrote {csv_path}")
    if args.write_baseline:
        write_baseline(rows, args.write_baseline, args.dataset, args.seed)
    if baseline is not None:
        sys.exit(0 if check_baseline(rows, baseline, args.tolerance) else 1)

    png_path = args.out_dir / f"{args.dataset}.png"
    plot(rows, png_path, f"{args.dataset}: recall vs QPS", ve.num_threads())
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
