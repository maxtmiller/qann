"""RAG retrieval matrix on BEIR (issue #143): quality, latency, memory and build time together.

For each dense index configuration, sweeps its search setting (nprobe or
ef_search) for top-10 queries, what a RAG prompt usually takes, and measures:
  - nDCG@10 against the human relevance labels, and recall@10 against exact search
  - latency per query on one thread, and QPS batched on all cores
  - index size (the saved file) and the part that stays in RAM when loaded
    with mmap=True (a RefineIndex's raw vectors stay on disk)
  - build time (train + add, all cores)
Each configuration's operating point is its cheapest setting whose nDCG@10 is
within 0.005 of exact search. At that point it also measures recall@100
against the labels (top 100, as a re-ranker would take) and runs:
  - hybrid search (HybridIndex with InvertedIndex BM25): linear (1, 1), and
    weights tuned with tune_weights on the even queries, scored on the odd ones
  - a metadata filter (where= on a random int column) keeping 1%, 10% and 50%
    of the corpus: recall@10 against exact filtered search, and latency

Embeddings as for hybrid_beir.py (cached in benchmarks/data/beir/<name>/).

    python benchmarks/rag_matrix.py fiqa scifact
    python benchmarks/rag_matrix.py fiqa --model sentence-transformers/all-MiniLM-L6-v2

A path ending in .hdf5 (ann-benchmarks / VIBE layout: train, test, neighbors)
runs the same matrix at scale without labels: quality is recall@10 against the
file's exact neighbors, the operating point the first setting reaching
RECALL_TARGET, and BM25 / hybrid search are skipped.

    python benchmarks/rag_matrix.py data/arxiv-nomic-768-normalized.hdf5
"""
import argparse
import csv
import math
import pathlib
import tempfile
import time

import numpy as np

import qann
from hybrid_beir import embeddings, labels
from sparse_bm25 import bm25, load

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
K = 10
NDCG_SLACK = 0.005
RECALL_TARGET = 0.95
ROUNDS = 3
SELECTIVITIES = [0.01, 0.1, 0.5]
NPROBES = [1, 2, 4, 8, 16, 32, 64, 128]
EFS = [10, 16, 24, 32, 48, 64, 96, 128, 256, 512]
COSINE = qann.Metric.Cosine


def nlist(n):
    return max(100, round(math.sqrt(n)))


def ivf(sq=None, pq=False, refine=None):
    def build(vecs):
        n, dim = vecs.shape
        base = qann.IVFIndex(dim, nlist(n), metric=COSINE)
        if sq:
            base.enable_sq(sq)
        if pq:
            base.enable_pq(dim // 2, fast_scan=True)
        base.train(vecs, seed=1)
        return base, (qann.RefineIndex(base, refine) if refine else base)
    return build, "nprobe", NPROBES


def hnsw(sq=None, refine=None):
    def build(vecs):
        base = qann.HNSWIndex(vecs.shape[1], M=16, ef_construction=200, seed=1, metric=COSINE)
        if sq:
            base.enable_sq(sq)
            base.train(vecs)
        return base, (qann.RefineIndex(base, refine) if refine else base)
    return build, "ef_search", EFS


def flat():
    def build(vecs):
        index = qann.FlatIndex(vecs.shape[1], metric=COSINE)
        return index, index
    return build, None, [None]


CONFIGS = {
    "Flat": flat(),
    "HNSW": hnsw(),
    "HNSW-SQ8": hnsw(sq=8),
    "HNSW-SQ8+R2": hnsw(sq=8, refine=2),
    "IVF": ivf(),
    "IVF-SQ8": ivf(sq=8),
    "IVF-SQ4+R4": ivf(sq=4, refine=4),
    "IVF-PQfs": ivf(pq=True),
    "IVF-PQfs+R4": ivf(pq=True, refine=4),
}


def timed(fn):
    """(result of the last call, best wall time of ROUNDS calls)."""
    best = math.inf
    for _ in range(ROUNDS):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return out, best


def latency(fn, nq):
    """(result, µs per query on one thread, QPS batched on all cores)."""
    qann.set_num_threads(1)
    out, one = timed(fn)
    qann.set_num_threads(0)
    _, many = timed(fn)
    return out, one / nq * 1e6, nq / many


def sizes(index, n, dim):
    """(bytes per vector in the saved file, in RAM when loaded with mmap=True)."""
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "index.qann"
        index.save(path)
        size = path.stat().st_size
    mapped = n * dim * 4 if isinstance(index, qann.RefineIndex) else 0
    return size / n, (size - mapped) / n


def overlap(found, truth, k):
    return np.mean([len(set(f[:k]) & set(t[:k])) / k for f, t in zip(found, truth)])


def build(config, vecs, D=None):
    """(base, index to query, build seconds); with D, a HybridIndex over it and BM25."""
    make = CONFIGS[config][0]
    t = time.perf_counter()
    base, index = make(vecs)
    if D is None:
        index.add(vecs, metadata={"cat": CATS})
    else:
        index = qann.HybridIndex(index, qann.InvertedIndex())
        index.add(vecs, D, metadata={"cat": CATS})
    return base, index, time.perf_counter() - t


def run(name, model, rows):
    global CATS
    if name.endswith(".hdf5"):
        # ann-benchmarks / VIBE file: no text or labels, so quality is recall@10
        # against its exact neighbors and there is no BM25 or hybrid search
        import h5py
        with h5py.File(name) as f:
            doc_vecs, query_vecs, truth = f["train"][:], f["test"][:], f["neighbors"][:, :10]
        model, D = "-", None
        name = pathlib.Path(name).stem
    else:
        docs, queries, qrels = load(name)
        D, Q = bm25(docs, queries)
        doc_vecs, query_vecs = embeddings(name, model, docs, queries)
        relevant = labels(docs, queries, qrels)
    n, dim = doc_vecs.shape
    nq = len(query_vecs)
    CATS = np.random.default_rng(1).integers(0, 100, n)
    print(f"\n{name}, {model}: {n:,} docs, dim {dim}, {nq:,} test queries", flush=True)
    common = dict(dataset=name, model=model.split("/")[-1], n=n, dim=dim)
    exact = qann.FlatIndex(dim, metric=COSINE)
    exact.add(doc_vecs, metadata={"cat": CATS})
    if D is None:
        exact_ids, exact_ndcg = truth, None
    else:
        exact_ids = exact.batch_query(query_vecs, K)[0]
        exact_ndcg = qann.ndcg(exact_ids, relevant, k=10)
        bm25_row(D, Q, relevant, nq, common, rows)
    sweep_configs(doc_vecs, query_vecs, exact, exact_ids, exact_ndcg, D, Q if D is not None else None,
                  relevant if D is not None else None, common, rows)


def bm25_row(D, Q, relevant, nq, common, rows):
    sparse = qann.InvertedIndex()
    sparse.add(D)
    (ids, _), us, qps = latency(lambda: sparse.batch_query(Q, K), nq)
    rec = qann.recall(sparse.batch_query(Q, 100)[0], relevant, k=100)
    print(f"  BM25 (InvertedIndex)          nDCG@10 {qann.ndcg(ids, relevant, k=10):.3f}  "
          f"recall@100 {rec:.3f}  {us:7.0f} µs  {qps:8.0f} qps")
    rows.append(dict(common, config="BM25", ndcg10=qann.ndcg(ids, relevant, k=10),
                     recall100=rec, us_1t=us, qps_batched=qps))


def sweep_configs(doc_vecs, query_vecs, exact, exact_ids, exact_ndcg, D, Q, relevant, common, rows):
    """Each configuration's sweep, then filters and hybrid search at its operating point:
    nDCG@10 within NDCG_SLACK of exact with labels, recall@10 RECALL_TARGET without."""
    n, dim = doc_vecs.shape
    nq = len(query_vecs)
    print(f"  {'config':<14}{'setting':>14} {'nDCG@10':>8} {'R@10 vs exact':>14} "
          f"{'µs 1t':>7} {'qps batch':>10} {'B/vec':>7} {'RAM B/vec':>10} {'build s':>8}")
    for config, (_, param, sweep) in CONFIGS.items():
        base, index, build_s = build(config, doc_vecs)
        file_b, ram_b = sizes(index, n, dim)
        chosen = None
        if param == "nprobe":
            sweep = [v for v in sweep if v < nlist(n)] + [nlist(n)]
        for value in sweep:
            if param:
                setattr(base, param, value)
            (ids, _), us, qps = latency(lambda: index.batch_query(query_vecs, K), nq)
            ndcg = math.nan if relevant is None else qann.ndcg(ids, relevant, k=10)
            vs_exact = overlap(ids, exact_ids, K)
            setting = f"{param}={value}" if param else "exact"
            print(f"  {config:<14}{setting:>14} {ndcg:8.3f} {vs_exact:14.3f} {us:7.0f} {qps:10.0f} "
                  f"{file_b:7.0f} {ram_b:10.0f} {build_s:8.1f}", flush=True)
            row = dict(common, config=config, param=param, value=value, ndcg10=ndcg,
                       recall10_vs_exact=vs_exact, us_1t=us, qps_batched=qps, bytes_per_vec=file_b,
                       ram_bytes_per_vec_mmap=ram_b, build_s=build_s, operating=False)
            rows.append(row)
            if (vs_exact >= RECALL_TARGET) if relevant is None else (ndcg >= exact_ndcg - NDCG_SLACK):
                chosen = (value, row)
                break
        if chosen is None:
            value, row = sweep[-1], rows[-1]
            print(f"  {config}: never reached the target, using {param}={value}")
        else:
            value, row = chosen
        row["operating"] = True
        if relevant is not None:
            row["recall100"] = qann.recall(index.batch_query(query_vecs, 100)[0], relevant, k=100)

        for sel in SELECTIVITIES:
            where = f"cat < {round(sel * 100)}"
            truth = exact.batch_query(query_vecs, 10, where=where)[0]
            (ids, _), us, _ = latency(lambda: index.batch_query(query_vecs, 10, where=where), nq)
            row[f"filter{sel:g}_recall10"] = overlap(ids, truth, 10)
            row[f"filter{sel:g}_us_1t"] = us

        if D is None:
            print(f"    -> {config} at {f'{param}={value}' if param else 'exact'}: filters R@10 "
                  + ", ".join(f"{s:.0%} {row[f'filter{s:g}_recall10']:.3f} ({row[f'filter{s:g}_us_1t']:.0f} µs)"
                              for s in SELECTIVITIES), flush=True)
            continue
        hbase, hybrid, _ = build(config, doc_vecs, D)
        if param:
            setattr(hbase, param, value)
        (ids, _), us, qps = latency(lambda: hybrid.batch_query(query_vecs, Q, K), nq)
        row["hybrid_ndcg10"] = qann.ndcg(ids, relevant, k=10)
        row["hybrid_recall100"] = qann.recall(hybrid.batch_query(query_vecs, Q, 100)[0], relevant, k=100)
        row["hybrid_us_1t"], row["hybrid_qps_batched"] = us, qps
        odd = slice(1, None, 2)
        row["hybrid_odd_ndcg10_default"] = qann.ndcg(hybrid.batch_query(query_vecs[odd], Q[odd], 10)[0], relevant[odd])
        hybrid.tune_weights(query_vecs[::2], Q[::2], relevant[::2], 10)
        row["hybrid_tuned_weights"] = hybrid.weights
        row["hybrid_odd_ndcg10_tuned"] = qann.ndcg(hybrid.batch_query(query_vecs[odd], Q[odd], 10)[0], relevant[odd])
        print(f"    -> {config} at {f'{param}={value}' if param else 'exact'}: recall@100 {row['recall100']:.3f}, hybrid nDCG@10 {row['hybrid_ndcg10']:.3f} "
              f"({row['hybrid_us_1t']:.0f} µs), held-out tuned {row['hybrid_tuned_weights']} "
              f"{row['hybrid_odd_ndcg10_tuned']:.3f} vs {row['hybrid_odd_ndcg10_default']:.3f}; filters R@10 "
              + ", ".join(f"{s:.0%} {row[f'filter{s:g}_recall10']:.3f} ({row[f'filter{s:g}_us_1t']:.0f} µs)"
                          for s in SELECTIVITIES), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    args = parser.parse_args()
    rows = []
    for name in args.datasets:
        run(name, args.model, rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"rag_matrix_{'_'.join(pathlib.Path(d).stem for d in args.datasets)}.csv"
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
