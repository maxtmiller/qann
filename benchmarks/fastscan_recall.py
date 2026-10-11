# Would PQ fast-scan (4-bit codes + 8-bit lookup tables) cost too much recall?
# Part 1: real IVF-PQ on SIFT1M with 16 vs 256 centroids per subspace, with and
# without re-ranking. Part 2: numpy simulation of rounding the ADC table to
# uint8, on identical 4-bit codes.
#
#   python3 benchmarks/fastscan_recall.py        (needs benchmarks/data/sift)
import pathlib, time
import numpy as np
import qann

D = str(pathlib.Path(__file__).parent / "data" / "sift") + "/"

def fvecs(p):
    a = np.fromfile(p, dtype=np.int32)
    return a.reshape(-1, a[0] + 1)[:, 1:].view(np.float32).copy()

def ivecs(p):
    a = np.fromfile(p, dtype=np.int32)
    return a.reshape(-1, a[0] + 1)[:, 1:].copy()

base, learn = fvecs(D + "sift_base.fvecs"), fvecs(D + "sift_learn.fvecs")
queries, gt = fvecs(D + "sift_query.fvecs"), ivecs(D + "sift_groundtruth.ivecs")[:, :10]

def recall(ids):
    return np.mean([len(set(r) & set(g)) / 10 for r, g in zip(ids, gt)])

print("part 1: IVF-PQ on SIFT1M, nlist 1000, recall@10 over 10k queries", flush=True)
print(f"{'config':10} {'B/vec':>5} {'nprobe':>6} {'PQ only':>8} {'+R4':>6} {'+R16':>6}", flush=True)
for m, K in [(16, 256), (32, 256), (32, 16), (64, 16)]:
    ivf = qann.IVFIndex(128, 1000, 16)
    ivf.enable_pq(m, K)
    t = time.time()
    ivf.train(learn, seed=1)
    refine = qann.RefineIndex(ivf, k_factor=4)
    refine.add(base)
    build = time.time() - t
    bits = 8 if K == 256 else 4
    for nprobe in (16, 64):
        ivf.nprobe = nprobe
        pq_only = recall(ivf.batch_query(queries, 10)[0])
        refine.k_factor = 4
        r4 = recall(refine.batch_query(queries, 10)[0])
        refine.k_factor = 16
        r16 = recall(refine.batch_query(queries, 10)[0])
        print(f"PQ{m}x{bits:<5} {m * bits // 8:>5} {nprobe:>6} {pq_only:>8.3f} {r4:>6.3f} {r16:>6.3f}   (build {build:.0f}s)", flush=True)

# Part 2: same 4-bit codes, float table vs uint8 table (FAISS-style: per-subspace
# bias, one shared scale so every entry fits 0..255; sums fit uint16 for m <= 257).
print("\npart 2: uint8 lookup tables vs float, flat PQ over 100k base vectors, 500 queries", flush=True)
sub = base[:100_000]
qs = queries[:500]
exact = ((qs ** 2).sum(1)[:, None] - 2 * qs @ sub.T + (sub ** 2).sum(1)[None, :])
gt_sub = np.argsort(exact, axis=1)[:, :100]

def rec(dists, kk, truth_k):
    top = np.argpartition(dists, kk, axis=1)[:, :kk]
    return np.mean([len(set(t) & set(g[:truth_k])) / truth_k for t, g in zip(top, gt_sub)])

for m in (32, 64):
    pq = qann.PQCodebook(128, m, 16)
    pq.train(learn, seed=1)
    codes = np.stack([pq.encode(v) for v in sub]).astype(np.int64)     # (n, m)
    flat_idx = codes + (np.arange(m) * 16)[None, :]
    f_d = np.empty((len(qs), len(sub)), np.float32)
    q_d = np.empty((len(qs), len(sub)), np.float32)
    for i, q in enumerate(qs):
        T = pq.compute_adc_table(q).reshape(m, 16)
        f_d[i] = T.reshape(-1)[flat_idx].sum(1)
        bias = T.min(1, keepdims=True)
        scale = 255.0 / (T - bias).max()
        Tq = np.round((T - bias) * scale).astype(np.uint16)              # 0..255
        q_d[i] = Tq.reshape(-1)[flat_idx].sum(1).astype(np.float32)      # uint16 sums
    print(f"PQ{m}x4  recall@10: float {rec(f_d, 10, 10):.3f}  uint8 {rec(q_d, 10, 10):.3f}   "
          f"candidates@100 (what re-ranking sees): float {rec(f_d, 100, 10):.3f}  uint8 {rec(q_d, 100, 10):.3f}", flush=True)
