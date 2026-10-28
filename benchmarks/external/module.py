# Shared by ann-benchmarks and VIBE: both define BaseANN at algorithms/base/module.py.
import numpy as np
import qann

from ..base.module import BaseANN

METRICS = {
    "euclidean": qann.Metric.L2,
    # Angular and cosine data is normalized here and searched by L2, which ranks unit vectors
    # like cosine. qann.Metric.Cosine does the same internally but rejects zero vectors, and
    # nytimes-256-angular has 239; here they stay zero, as in the hnswlib and faiss adapters
    # (see _metric_for for how they are scored).
    "angular": qann.Metric.L2,
    "cosine": qann.Metric.L2,
    # VIBE's "normalized" datasets are unit length, so L2 ranks like cosine without rescaling.
    "normalized": qann.Metric.L2,
    "ip": qann.Metric.InnerProduct,
}


class _Qann(BaseANN):
    def __init__(self, metric):
        if metric not in METRICS:
            raise NotImplementedError(f"qann does not support metric {metric!r}")
        self.metric = METRICS[metric]
        self.normalize = metric in ("angular", "cosine")
        self.pad = 0

    def _metric_for(self, X):
        # After normalizing, a zero vector is at L2 distance 1 from every query, as if its
        # cosine were 0.5, so it would outrank true neighbors below that. Inner product gives
        # it 0, cosine's value, and ranks unit vectors by cosine too; only used when needed so
        # other datasets keep the L2 path.
        if self.normalize and not np.all(np.any(X != 0, axis=1)):
            return qann.Metric.InnerProduct
        return self.metric

    def _prep(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if self.normalize:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = np.divide(X, norms, out=np.zeros_like(X), where=norms > 0)
        if self.pad:
            X = np.pad(X, ((0, 0), (0, self.pad)))
        return X

    def query(self, v, n):
        ids, _ = self.index.query(self._prep(np.asarray(v).reshape(1, -1))[0], n)
        return ids[ids >= 0]

    def batch_query(self, X, n):
        qann.set_num_threads(0)
        ids, _ = self.index.batch_query(self._prep(X), n)
        self.res = [row[row >= 0] for row in ids]


class QannHNSW(_Qann):
    def __init__(self, metric, M, ef_construction, sq_bits=0):
        super().__init__(metric)
        self.M, self.ef_construction, self.sq_bits = M, ef_construction, sq_bits
        self.ef_search, self.k_factor = None, None

    def fit(self, X):
        qann.set_num_threads(1)
        X = self._prep(X)
        self.metric = self._metric_for(X)
        hnsw = qann.HNSWIndex(X.shape[1], M=self.M, ef_construction=self.ef_construction, metric=self.metric, seed=1)
        if self.sq_bits:
            hnsw.enable_sq(self.sq_bits)
            hnsw.train(X)
            self.index = qann.RefineIndex(hnsw, k_factor=2)
        else:
            self.index = hnsw
        self.hnsw = hnsw
        self.index.add(X)

    def set_query_arguments(self, ef_search, k_factor=None):
        self.ef_search, self.k_factor = ef_search, k_factor
        self.hnsw.ef_search = ef_search
        if k_factor is not None:
            self.index.k_factor = k_factor

    def __str__(self):
        s = f"qann.HNSW(M={self.M}, efc={self.ef_construction}"
        if self.sq_bits:
            s += f", sq={self.sq_bits}, k_factor={self.k_factor}"
        return s + f", ef={self.ef_search})"


def _pq_subspaces(dim, dims_per_subspace):
    # Fast-scan needs an even subspace count, at most 1024, dividing dim.
    target = min(1024, dim // dims_per_subspace)
    return max(m for m in range(2, target + 1, 2) if dim % m == 0)


class QannIVFFastScan(_Qann):
    def __init__(self, metric, nlist, dims_per_subspace, rotate=True):
        super().__init__(metric)
        self.nlist, self.dims_per_subspace, self.rotate = nlist, dims_per_subspace, rotate
        self.nprobe, self.k_factor = None, None

    def fit(self, X):
        qann.set_num_threads(1)
        # Zero padding to a multiple of 8 keeps every distance unchanged and lets 2 or 4 dims per
        # subspace divide evenly (glove-25 padded to 28 could only split into 4 or 14 subspaces).
        self.pad = -X.shape[1] % 8
        X = self._prep(X)
        self.metric = self._metric_for(X)
        dim = X.shape[1]
        self.m = _pq_subspaces(dim, self.dims_per_subspace)
        # qann needs 100 <= nlist <= training vectors; small datasets get fewer clusters.
        self.nlist_used = min(self.nlist, max(100, len(X) // 39))
        self.ivf = qann.IVFIndex(dim, self.nlist_used, metric=self.metric)
        self.ivf.enable_pq(self.m, fast_scan=True, rotate=self.rotate)
        rng = np.random.default_rng(1)
        sample = X[rng.choice(len(X), min(len(X), 256 * self.nlist_used), replace=False)]
        self.ivf.train(sample, seed=1)
        self.index = qann.RefineIndex(self.ivf, k_factor=8)
        self.index.add(X)

    def set_query_arguments(self, nprobe, k_factor):
        self.nprobe, self.k_factor = min(nprobe, self.nlist_used), k_factor
        self.ivf.nprobe = self.nprobe
        self.index.k_factor = k_factor

    def __str__(self):
        return (
            f"qann.IVFFastScan(nlist={self.nlist_used}, m={self.m}, rotate={self.rotate}, "
            f"nprobe={self.nprobe}, k_factor={self.k_factor})"
        )
