"""OPQ study (issue #41): does a learned rotation before PQ help qann's IVF-PQ?

Run from the repo root after `pip install -e .`:
    python3 benchmarks/opq_study.py                      # sift and glove-100
    python3 benchmarks/opq_study.py --dataset sift

A rotation R preserves L2 and cosine distances, so the study rotates learn,
base and queries in NumPy and feeds them to the unchanged qann indexes: any
recall difference is the rotation's effect on PQ alone. For each PQ size it
compares four rotations:
  identity   plain PQ
  pca        parametric OPQ: PCA directions assigned to subspaces so each gets
             a similar product of eigenvalues (Ge et al., eigenvalue allocation)
  opq        non-parametric OPQ trained on the vectors, starting from identity
  opq-res    the same trained on IVF residuals (x - nearest coarse centroid),
             which is what IVF-PQ quantizes
Non-parametric OPQ starts from identity rather than from pca (as Ge et al.
suggest): on SIFT1M the pca start is a worse local minimum (NumPy PQ16 error
14.0k from pca vs 9.9k from identity, 10.3k unrotated), because SIFT's natural
dimension order already groups correlated dimensions.
and reports covariance structure, PQ reconstruction error and recall@10 at
fixed nprobe, with and without re-ranking.

qann exposes no PQ decode, so rotations are trained with a small NumPy PQ;
every number reported is then measured with qann's own PQCodebook / IVFIndex.
Writes benchmarks/results/opq_study_<dataset>.csv.
"""

import argparse
import csv
import math
import time

import numpy as np

import qann

from bench_indexes import DATASETS, K, RESULTS_DIR, load_dataset, recall_at_k

OPQ_ITERS = 12        # outer alternations (PQ train / Procrustes)
OPQ_KMEANS_ITERS = 4  # Lloyd steps per subspace per outer iteration, warm-started
OPQ_SAMPLE = 50_000   # vectors used to train rotations
COARSE_SAMPLE = 50_000
MSE_SAMPLE = 5_000
NPROBES = [16, 64]
K_FACTORS = [4, 16]
# dataset -> [(name, num_subspaces, centroids_per_subspace, fast_scan)]
CONFIGS = {
    "sift": [("PQ16", 16, 256, False), ("PQ32", 32, 256, False), ("PQ64x4fs", 64, 16, True)],
    "glove-100": [("PQ10", 10, 256, False), ("PQ20", 20, 256, False), ("PQ50x4fs", 50, 16, True)],
}


def normalize(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------- k-means / PQ

def assign(x, c):
    # argmin_j ||x - c_j||^2 = argmin_j ||c_j||^2 - 2 x.c_j, in blocks to bound memory
    out = np.empty(len(x), dtype=np.int64)
    cn = (c * c).sum(1)
    for s in range(0, len(x), 65536):
        out[s:s + 65536] = np.argmin(cn - 2.0 * x[s:s + 65536] @ c.T, axis=1)
    return out


def kmeans(x, k, iters, rng, init=None):
    c = x[rng.choice(len(x), k, replace=False)].copy() if init is None else init.copy()
    for _ in range(iters):
        a = assign(x, c)
        counts = np.bincount(a, minlength=k)
        sums = np.zeros_like(c)
        np.add.at(sums, a, x)
        empty = counts == 0
        c[~empty] = sums[~empty] / counts[~empty, None]
        c[empty] = x[rng.choice(len(x), empty.sum(), replace=False)]
    return c


def pq_reconstruct(x, m, ks, iters, rng, books=None):
    # trains (or refines) per-subspace codebooks on x; returns (reconstruction, books)
    d = x.shape[1]
    w = d // m
    y = np.empty_like(x)
    new_books = []
    for s in range(m):
        sub = np.ascontiguousarray(x[:, s * w:(s + 1) * w])
        c = kmeans(sub, ks, iters, rng, None if books is None else books[s])
        y[:, s * w:(s + 1) * w] = c[assign(sub, c)]
        new_books.append(c)
    return y, new_books


# ---------------------------------------------------------------- rotations

def pca_rotation(x, m):
    # eigenvalue allocation: largest eigenvalue first, each into the subspace with
    # room whose log-eigenvalue sum is smallest, balancing variance products
    d = x.shape[1]
    w = d // m
    cov = np.cov(x, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    buckets = [[] for _ in range(m)]
    logsum = np.zeros(m)
    for i in order:
        free = [b for b in range(m) if len(buckets[b]) < w]
        b = min(free, key=lambda j: logsum[j])
        buckets[b].append(i)
        logsum[b] += math.log(max(vals[i], 1e-12))
    cols = [i for b in buckets for i in b]
    return vecs[:, cols].astype(np.float32)  # x @ R puts subspace b's directions together


def opq_rotation(x, m, ks, rng, init):
    r = init.copy()
    books = None
    for it in range(OPQ_ITERS):
        xr = x @ r
        y, books = pq_reconstruct(xr, m, ks, OPQ_KMEANS_ITERS if books else 10, rng, books)
        # orthogonal Procrustes: min ||x R - y|| => R = U V^T for x^T y = U S V^T
        u, _, vt = np.linalg.svd(x.T.astype(np.float64) @ y.astype(np.float64))
        r = (u @ vt).astype(np.float32)
    return r


# ---------------------------------------------------------------- measurements

def structure(x, m):
    cov = np.cov(x, rowvar=False)
    d = x.shape[1]
    w = d // m
    mask = np.zeros_like(cov, dtype=bool)
    for s in range(m):
        mask[s * w:(s + 1) * w, s * w:(s + 1) * w] = True
    sq = cov ** 2
    off = sq[~mask].sum() / sq.sum()
    var = np.array([np.trace(cov[s * w:(s + 1) * w, s * w:(s + 1) * w]) for s in range(m)])
    return off, var.max() / max(var.min(), 1e-12)


def pq_mse(train, x, m, ks, seed):
    # qann's PQ: ||x - decode(encode(x))||^2 == distance_adc(table(x), encode(x))
    pq = qann.PQCodebook(x.shape[1], m, ks)
    pq.train(train, seed=seed)
    return float(np.mean([pq.distance_adc(pq.compute_adc_table(v), pq.encode(v)) for v in x]))


def ivf_recall(learn, base, queries, gt, nlist, m, ks, fast, metric, seed):
    ivf = qann.IVFIndex(learn.shape[1], nlist, metric=metric)
    ivf.enable_pq(m, ks, fast_scan=fast)
    ivf.train(learn, seed=seed)
    refine = qann.RefineIndex(ivf)
    refine.add(base)
    out = {}
    for nprobe in NPROBES:
        ivf.nprobe = nprobe
        ids, _ = ivf.batch_query(queries, K)
        out[f"np{nprobe}"] = recall_at_k(ids, gt, K)
        for kf in K_FACTORS:
            refine.k_factor = kf
            ids, _ = refine.batch_query(queries, K)
            out[f"np{nprobe}+R{kf}"] = recall_at_k(ids, gt, K)
    return out


def run(dataset, seed):
    base, learn, queries, gt = load_dataset(dataset)
    metric = getattr(qann.Metric, DATASETS[dataset]["metric"])
    if metric == qann.Metric.Cosine:  # the index normalizes too; rotations are learned on unit vectors
        base, learn, queries = normalize(base), normalize(learn), normalize(queries)
    n, d = base.shape
    nlist = round(math.sqrt(n))
    rng = np.random.default_rng(seed)
    train = learn[rng.choice(len(learn), min(OPQ_SAMPLE, len(learn)), replace=False)]
    mse_x = base[rng.choice(n, MSE_SAMPLE, replace=False)]

    # coarse centroids for residual statistics and opq-res training (NumPy k-means;
    # close to, not identical with, the IVFIndex's own)
    t = time.perf_counter()
    coarse = kmeans(learn[rng.choice(len(learn), min(COARSE_SAMPLE, len(learn)), replace=False)], nlist, 15, rng)
    res_train = train - coarse[assign(train, coarse)]
    res_mse = mse_x - coarse[assign(mse_x, coarse)]
    print(f"{dataset}: n {n}, dim {d}, nlist {nlist}, coarse k-means {time.perf_counter() - t:.0f}s", flush=True)

    rows = []
    for name, m, ks, fast in CONFIGS[dataset]:
        t = time.perf_counter()
        rots = {"identity": np.eye(d, dtype=np.float32), "pca": pca_rotation(train, m)}
        rots["opq"] = opq_rotation(train, m, ks, rng, rots["identity"])
        rots["opq-res"] = opq_rotation(res_train, m, ks, rng, rots["identity"])
        print(f"\n{name}: rotations trained in {time.perf_counter() - t:.0f}s", flush=True)

        for rot, r in rots.items():
            off_v, imb_v = structure(train @ r, m)
            off_r, imb_r = structure(res_train @ r, m)
            mse_v = pq_mse(train @ r, mse_x @ r, m, ks, seed)
            mse_r = pq_mse(res_train @ r, res_mse @ r, m, ks, seed)
            rec = ivf_recall(learn @ r, base @ r, queries @ r, gt, nlist, m, ks, fast, metric, seed)
            row = dict(dataset=dataset, config=name, rotation=rot,
                       offblock_vec=round(off_v, 4), var_ratio_vec=round(imb_v, 2),
                       offblock_res=round(off_r, 4), var_ratio_res=round(imb_r, 2),
                       mse_vec=mse_v, mse_res=mse_r, **{k: round(v, 4) for k, v in rec.items()})
            rows.append(row)
            print(f"  {rot:<9} off-block {off_v:.3f}/{off_r:.3f}  var max/min {imb_v:6.1f}/{imb_r:6.1f}  "
                  f"mse {mse_v:.4g}/{mse_r:.4g}  " + "  ".join(f"{k} {v:.3f}" for k, v in rec.items()), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sift", "glove-100"], action="append")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    for dataset in args.dataset or ["sift", "glove-100"]:
        rows = run(dataset, args.seed)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / f"opq_study_{dataset}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
