"""qann's HNSWIndex against hnswlib on one ann-benchmarks dataset (issue #128), outside Docker so
both can be profiled. Data is prepared the way the two ann-benchmarks adapters prepare it: qann
gets angular data normalized in Python (inner product when the data has zero vectors, else L2),
hnswlib gets it raw in its "cosine" space. Queries go one at a time on one thread, like
ann-benchmarks; qann's time is also shown without the Python normalization.

Built graphs are cached next to the dataset, so a profile doesn't rebuild them:

    python benchmarks/hnsw_vs_hnswlib.py --data ~/qann-bench/ann-benchmarks/data/sift-128-euclidean.hdf5
    perf record -o qann.perf python benchmarks/hnsw_vs_hnswlib.py --data ... --lib qann --profile-ef 40
"""

import argparse
import pathlib
import time

import h5py
import numpy as np

import qann


def load(path):
    with h5py.File(path, "r") as f:
        return f.attrs["distance"], f["train"][:], f["test"][:], f["neighbors"][:, :10]


def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


class Qann:
    name = "qann"

    def __init__(self, args, distance, train):
        self.angular = distance == "angular"
        x = normalize(train) if self.angular else np.ascontiguousarray(train)
        metric = qann.Metric.InnerProduct if self.angular and not np.all(np.any(x != 0, axis=1)) else qann.Metric.L2
        path = args.data.with_suffix(f".qann-M{args.M}-efc{args.efc}.qann")
        if path.exists():
            self.index = qann.load(path)
        else:
            qann.set_num_threads(0)
            self.index = qann.HNSWIndex(x.shape[1], M=args.M, ef_construction=args.efc, metric=metric, seed=1)
            self.index.add(x)
            self.index.save(path)
        qann.set_num_threads(1)
        print(f"qann metric {metric}", flush=True)

    def set_ef(self, ef):
        self.index.ef_search = ef

    def prep(self, q):
        return normalize(q.reshape(1, -1))[0] if self.angular else q

    def query(self, q):
        return self.index.query(self.prep(q), 10)[0]

    def query_prepped(self, q):
        return self.index.query(q, 10)[0]


class Hnswlib:
    name = "hnswlib"

    def __init__(self, args, distance, train):
        import hnswlib
        space = "cosine" if distance == "angular" else "l2"
        self.index = hnswlib.Index(space=space, dim=train.shape[1])
        path = args.data.with_suffix(f".hnswlib-M{args.M}-efc{args.efc}.bin")
        if path.exists():
            self.index.load_index(str(path), max_elements=len(train))
        else:
            self.index.init_index(max_elements=len(train), ef_construction=args.efc, M=args.M)
            self.index.add_items(train, np.arange(len(train)))
            self.index.save_index(str(path))
        self.index.set_num_threads(1)

    def set_ef(self, ef):
        self.index.set_ef(ef)

    def query(self, q):
        return self.index.knn_query(np.expand_dims(q, axis=0), k=10)[0][0]


def run(lib, queries, gt, ef, prepped=False):
    lib.set_ef(ef)
    rows = [lib.prep(q) for q in queries] if prepped else queries
    call = lib.query_prepped if prepped else lib.query
    for q in rows[:200]:
        call(q)
    found = []
    t = time.perf_counter()
    for q in rows:
        found.append(call(q))
    us = (time.perf_counter() - t) / len(rows) * 1e6
    recall = np.mean([len(set(map(int, f)) & set(map(int, g))) / 10 for f, g in zip(found, gt)])
    return recall, us


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=pathlib.Path, required=True, help="an ann-benchmarks .hdf5 file")
    ap.add_argument("--lib", choices=["qann", "hnswlib", "both"], default="both")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--efc", type=int, default=500)
    ap.add_argument("--ef", type=int, nargs="+", default=[10, 20, 40, 80, 120, 200])
    ap.add_argument("--profile-ef", type=int, help="only run every query at this ef, --repeat times, for perf")
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()
    args.data = args.data.expanduser()

    distance, train, test, gt = load(args.data)
    libs = [cls(args, distance, train) for cls in (Qann, Hnswlib) if args.lib in (cls.name, "both")]
    if args.profile_ef:
        for lib in libs:
            for _ in range(args.repeat):
                recall, us = run(lib, test, gt, args.profile_ef)
            print(f"{lib.name:8} ef {args.profile_ef:4}  recall {recall:.4f}  {us:8.1f} us/query", flush=True)
        return
    for ef in args.ef:
        for lib in libs:
            recall, us = run(lib, test, gt, ef)
            extra = ""
            if isinstance(lib, Qann) and lib.angular:
                extra = f"  ({run(lib, test, gt, ef, prepped=True)[1]:.1f} us without the Python normalization)"
            print(f"{lib.name:8} ef {ef:4}  recall {recall:.4f}  {us:8.1f} us/query  {1e6 / us:8.0f} QPS{extra}", flush=True)


if __name__ == "__main__":
    main()
