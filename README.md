# QaNN

QaNN (**Q**uantized **N**earest **N**eighbors) is a Python library for fast nearest-neighbor search over float32 vectors. It offers exact search, inverted-file (IVF) search, product quantization (PQ) compression and exact re-ranking, all implemented in multithreaded, SIMD-accelerated C++ and driven from NumPy.

> **Alpha:** under active development. The API may change between releases.

## Install

```bash
pip install qann
```

Wheels are published for Linux (x86_64, aarch64) and macOS (Apple Silicon, Intel) on Python 3.10 to 3.14. x86_64 builds require AVX2 and FMA. To build from source instead, run `pip install .` in a checkout; this needs CMake 3.20+, a C++20 compiler and, on Linux, OpenBLAS (optional but much faster training).

## Quick start

```python
import numpy as np, qann

x = np.random.rand(100_000, 128).astype(np.float32)

ivf = qann.IVFIndex(128, nlist=1024, nprobe=16)
ivf.enable_pq(16)                           # store each vector in 16 bytes
ivf.train(x, seed=1)
index = qann.RefineIndex(ivf, k_factor=10)  # re-rank PQ candidates exactly
index.add(x)

ids, dists = index.batch_query(x[:5], 10)   # both shaped (5, 10)
```

## Conventions

- **Vectors** are passed as 2D NumPy arrays of shape `(n, dim)`, except single vectors (`query`, `PQCodebook` methods), which are `(dim,)`. float32 C-contiguous arrays are used without copying; anything else (float64, Fortran order, strided slices) is converted first.
- **IDs** are assigned in insertion order by default: the first vector added gets id 0, the next id 1, and so on across all `add` calls. Create an index with `custom_ids=True` to use your own ids instead; see [Custom ids and deleting](#custom-ids-and-deleting).
- **Distances** are squared Euclidean (L2) distances by default, or cosine distances with `metric=qann.Metric.Cosine` (see [Metrics](#metrics)); smallest first either way.
- **Results** from `batch_query` are an `int64` id array and a `float32` distance array, both shaped `(num_queries, k)`. Rows with fewer than `k` hits are padded with id `-1` and distance `inf`.

## Indexes

Every index has the same core methods:

| Method | Description |
| --- | --- |
| `add(data, ids=None)` | Add an `(n, dim)` array of vectors, with `n` ids if the index uses custom ids. |
| `batch_query(queries, k)` | k nearest neighbors for each row of an `(nq, dim)` array, run in parallel. Returns `(ids, dists)`. |
| `query(query, k)` | k nearest neighbors for a single `(dim,)` vector such as `x[i]`. Returns 1D `(ids, dists)`. |
| `remove(ids)` | Delete vectors by id; returns how many were deleted. |
| `compact()` | Free the memory of deleted vectors; returns how many were freed. Ids are unchanged, and so are results, except that `HNSWIndex` rebuilds its graph. |
| `size()` | Number of vectors in the index (deleted ones excluded). |
| `dim()` | Vector dimension. |
| `save(path)` | Save the index to a file; see [Saving and loading](#saving-and-loading). |

### `FlatIndex(dim, *, custom_ids=False)`

Exact brute-force search: no training, 100% recall, and query cost that grows linearly with the number of vectors. Good for up to a few hundred thousand vectors, or as ground truth when measuring recall.

```python
index = qann.FlatIndex(128)
index.add(x)
ids, dists = index.batch_query(x[:10], 10)
```

### `IVFIndex(dim, nlist, nprobe=10, *, custom_ids=False)`

Partitions vectors into `nlist` clusters with k-means and, at query time, scans only the `nprobe` clusters closest to the query. Much faster than flat search at a small cost in recall.

| Member | Description |
| --- | --- |
| `train(data, max_iters=25, seed=None)` | Learn the cluster centroids from a sample of at least `nlist` vectors. Call once, before `add`. Pass `seed` for reproducible results. |
| `enable_pq(num_subspaces, centroids_per_subspace=None, *, fast_scan=False)` | Compress stored vectors with product quantization. Call before `train`. `centroids_per_subspace` defaults to 256 (one byte per slice), or 16 with `fast_scan`; see [Fast-scan](#fast-scan). |
| `fast_scan` | Read-only property: whether `enable_pq(..., fast_scan=True)` was used. |
| `nprobe` | Property: clusters scanned per query. Can be changed at any time; higher means better recall and slower queries. |
| `pq_distance` | Property: how PQ codes are scored, `qann.PQDistance.ADC` (default, more accurate) or `qann.PQDistance.SDC`. |
| `precomputed_tables` | Property: whether ADC queries use lookup tables precomputed at train time, which makes PQ queries faster. On by default; reads `True` once the tables exist (PQ enabled, trained, and `nlist * num_subspaces * centroids_per_subspace * 4` bytes within 256 MB). Set to `False` to free them. |

`nlist` must be between 100 and 65535, and `nprobe` between 1 and `nlist`. A common starting point is `nlist` around `sqrt(n)` to `4 * sqrt(n)` and `nprobe` at 1% to 5% of `nlist`.

**With PQ** (`enable_pq`), each vector is split into `num_subspaces` slices and each slice is stored as one byte, so a 128-dim float32 vector (512 bytes) with 16 subspaces takes 16 bytes. `dim` must be divisible by `num_subspaces`. Distances become approximate, which lowers recall; wrap the index in a `RefineIndex` to recover it.

```python
ivf = qann.IVFIndex(128, nlist=1024, nprobe=16)
ivf.enable_pq(16)
ivf.train(x, seed=1)
ivf.add(x)
ivf.nprobe = 32   # trade speed for recall
```

#### Fast-scan

`enable_pq(num_subspaces, fast_scan=True)` stores each slice in 4 bits (16 centroids per slice) and scores 32 vectors at a time with a single SIMD table lookup per slice (NEON on ARM, SSSE3 on x86). Scanning the codes is about 6x faster than regular PQ at the same bytes per vector. Use twice the `num_subspaces` you would with regular PQ to keep the memory and most of the accuracy, and pair it with a `RefineIndex`:

```python
ivf = qann.IVFIndex(128, nlist=1024, nprobe=16)
ivf.enable_pq(64, fast_scan=True)          # 64 x 4 bits = 32 bytes per vector
ivf.train(x, seed=1)
refine = qann.RefineIndex(ivf, k_factor=16)
refine.add(x)
```

- On SIFT1M with re-ranking (`k_factor=16`), 64 subspaces with fast-scan match 32 regular subspaces (recall@10 0.931 at `nprobe=16`, 0.995 at `nprobe=64`) and beat 16 regular ones, while scanning several times faster.
- Without a `RefineIndex`, recall is lower than regular PQ at the same memory, and the returned distances are approximate: the lookup tables are rounded to 6 bits.
- `num_subspaces` must be even and at most 256, `centroids_per_subspace` must be 16 (the default with `fast_scan`), and `pq_distance` stays `ADC`: setting `SDC` raises `ValueError`.

### `HNSWIndex(dim, M=16, ef_construction=200, *, custom_ids=False, seed=None)`

A graph index (Hierarchical Navigable Small World): every vector is linked to `M` near neighbors (`2 * M` on the bottom level), and a query walks the graph toward the query vector. No training; `add` builds the graph on all cores. It stores vectors in full, so distances are exact, and it needs no representative training sample, at the cost of more memory per vector (the vector plus its links) and slower adds. With `enable_sq()` it stores 8- or 4-bit codes per dimension instead (see below).

| Member | Description |
| --- | --- |
| `ef_search` | Property: how many candidates a query keeps while searching (default 64; queries use at least `k`). Can be changed at any time; higher means better recall and slower queries. |
| `M`, `ef_construction` | Read-only properties: the settings the graph was built with. Higher `M` or `ef_construction` builds a better graph, more slowly. |
| `enable_sq(bits=8)` | Store vectors as scalar codes of `bits` per dimension: 8 or 4 (anything else raises `ValueError`). Call before `train()` and `add()`. |
| `train(x)` | With SQ: learn each dimension's range from sample vectors. Required before `add()`; raises `RuntimeError` without SQ. |
| `sq_bits` | Read-only property: 8 or 4 after `enable_sq()`, else 0. |

```python
index = qann.HNSWIndex(128, M=16, ef_construction=200)
index.add(x)
index.ef_search = 128   # trade speed for recall
ids, dists = index.batch_query(q, 10)
```

On SIFT1M (1M vectors), `M=16, ef_construction=200` reaches recall@10 0.960 at `ef_search=64` (about 80 µs per query on one thread) and 0.995 at 256 (about 340 µs). On all cores, with the machine busy, the float graph built in 75 s and reached 0.996 at 11.4k queries/s; with 8-bit SQ and re-ranking (below) it reached 0.996 at 15.8k/s, close to IVF + fast-scan + `RefineIndex` at 0.995 and 16.9k/s, which built in 5 s. Below 0.96 recall HNSW is the faster of the two on SIFT1M. On GloVe-100 word embeddings (cosine) the order flips: IVF + fast-scan + `RefineIndex` was about twice as fast up to 0.94 recall, and HNSW with `M=16` needed `ef_search=1024` to reach 0.95. Benchmark on your own data before choosing. Choose IVF + fast-scan when you can train on representative data and build time matters; choose HNSW when you can't train up front, or when vectors keep arriving that the IVF clusters were not trained on.

- **Deleting** marks vectors deleted; queries still route through them, so the graph stays connected, but never return them. `compact()` rebuilds the graph from the remaining vectors, which takes about as long as building it again, so call it after many deletes rather than after each one.
- **Builds are parallel,** so the graph differs slightly from run to run. With `seed` and `qann.set_num_threads(1)`, the build is reproducible.

**Scalar quantization (compressed HNSW).** `enable_sq(8)` maps each dimension's trained range onto 256 evenly spaced values, so a vector takes `dim` bytes instead of `4 * dim` (on SIFT, 128 instead of 512); `enable_sq(4)` uses 16 values packed two per byte (64 bytes on SIFT). Distances become approximate, so wrap the index in a `RefineIndex` to re-rank with exact ones: `k_factor=2` is enough for 8 bits; 4 bits needs 4 on SIFT and 8 on GloVe word embeddings. Values outside the training sample's range are clamped, so train on a representative sample (or on the vectors themselves).

```python
index = qann.HNSWIndex(128)
index.enable_sq(8)
index.train(x)
refine = qann.RefineIndex(index, k_factor=2)
refine.add(x)
```

On SIFT1M with the machine busy, at `ef_search=256`: 8 bits + `RefineIndex(k_factor=2)` matched the float graph's recall (0.996) at 1.4x its queries per second on all cores (15.8k vs 11.4k) and 1.7x on one thread, and 4 bits + `k_factor=4` did nearly the same (0.995, 15.6k). The `RefineIndex` keeps full vectors too, so the memory saving needs the plain SQ index: 8 bits alone reaches 0.986, 4 bits alone only 0.86, so 4 bits really needs re-ranking. 4 bits is no faster than 8, since searches wait on memory latency rather than bytes moved; choose it to halve code memory. On GloVe-100 (cosine), 8 bits + `k_factor=2` again matched the floats exactly, while 4 bits alone reached only 0.70 and needed `k_factor=8`.

### `RefineIndex(base, k_factor=10, *, custom_ids=False)`

Wraps a trained, empty approximate index created without custom ids (and never compacted) (give the `RefineIndex` its own `custom_ids` instead). Each query fetches `k * k_factor` candidates from `base`, then re-ranks them with exact distances against full-precision copies of the vectors. Add, remove and compact through the `RefineIndex`: while it wraps a base, calling `add`, `remove` or `compact` on the base itself raises `ValueError` (it would put the two out of step), and a base can only be wrapped by one `RefineIndex` at a time. Tuning the base (`nprobe`, `pq_distance`) is fine. `k_factor` is a property and can be changed at any time, and the read-only `base` property returns the wrapped index, e.g. to change `nprobe`.

This is the usual way to combine PQ's speed with near-exact ranking. It keeps the original vectors in memory alongside the PQ codes.

### `make_index(type, dim, opts=IndexOptions())`

Builds an index from a type and an options object, returning a `FlatIndex`, `IVFIndex` or `HNSWIndex`:

```python
opts = qann.IndexOptions()
opts.nlist, opts.nprobe, opts.pq_subspaces = 1024, 16, 16
ivf = qann.make_index(qann.IndexType.IVF, 128, opts)
ivf.train(x)
```

`IndexType` is `Flat`, `IVF` or `HNSW`. `IndexOptions` fields and defaults: `metric` (`Metric.L2`), plus `capacity` (Flat initial reserve, 1024), `nlist` (100), `nprobe` (10), `pq_subspaces` (0, meaning no PQ) and `pq_centroids` (256) for IVF, and `hnsw_m` (16), `ef_construction` (200), `ef_search` (64) and `hnsw_sq_bits` (0 for floats; 8 or 4 calls `enable_sq()`) for HNSW.

## Metrics

Every index takes `metric=qann.Metric.L2` (the default) or `metric=qann.Metric.Cosine`, fixed at construction and saved with the index. Most text and image embedding models are meant to be compared by cosine similarity, so use `Cosine` for them:

```python
index = qann.HNSWIndex(384, metric=qann.Metric.Cosine)
index.add(embeddings)
ids, dists = index.batch_query(query_embeddings, 10)   # dists = 1 - cosine similarity
```

- **Distances** are `1 - cosine similarity`, from 0 (same direction) to 2 (opposite), matching `qann.cosine_distance`. Similarity is `1 - dist`.
- **Vector length doesn't matter:** each vector is scaled to length 1 when it's added and each query when it's run, so you don't need to normalize anything yourself. A zero vector has no direction, so adding or querying one raises `ValueError`.
- **It works with every index,** including PQ, fast-scan and HNSW. IVF learns its clusters on the scaled vectors, and a `RefineIndex` uses its base's metric. `index.metric` reports it.
- Inner product (for vectors whose length carries meaning) isn't supported yet.

## Custom ids and deleting

By default an index numbers vectors 0, 1, 2, ... in the order they're added. To use your own ids (database keys, document ids, ...), create the index with `custom_ids=True` and pass one id per vector:

```python
index = qann.FlatIndex(128, custom_ids=True)
index.add(vectors, ids=doc_ids)     # doc_ids: one int per row
ids, dists = index.batch_query(queries, 10)   # ids are your doc_ids

index.remove([doc_ids[0], doc_ids[1]])        # returns 2
```

- **Ids** can be a NumPy array of any integer type, a list or a `range`. They must be `>= 0` (`-1` marks missing results) and unique within the index; a bad batch raises an error and adds nothing.
- **Every add must pass ids** on a `custom_ids=True` index, and an index created without it refuses them.
- **`remove(ids)`** works with or without custom ids. Unknown and already deleted ids are skipped, so deleting twice is harmless. A deleted id can be added again, with a new vector.
- **Deleted vectors keep their memory until `compact()`**: queries skip them, but they still take space and scan time. `compact()` frees them in one pass, keeping every id and every query result the same, so call it after a large batch of deletes. It doesn't retrain anything. (`HNSWIndex` is the exception: it rebuilds its graph, so ids stay the same but results can shift slightly.)
- **Default ids never repeat.** New vectors continue after the highest id ever assigned, even after deletes and `compact()` (delete ids 0 to 9, compact, add one more, and it gets id 10).
- For `make_index`, set `IndexOptions.custom_ids`.

## Saving and loading

Any index can be saved to a file and loaded back, including its training, so it doesn't have to be retrained or re-added each run:

```python
index.save("vectors.qann")          # str or pathlib.Path

index = qann.load("vectors.qann")   # returns a FlatIndex, IVFIndex, HNSWIndex or RefineIndex
ids, dists = index.batch_query(queries, 10)
```

- **Everything is restored:** vectors, ids (including custom ids), deletions, IVF centroids, PQ codebooks, HNSW graphs and SQ ranges, the metric and settings such as `nprobe`, `pq_distance`, `precomputed_tables`, `ef_search` and `k_factor`. A loaded index gives identical results and keeps accepting `add` and `remove`.
- **A `RefineIndex` file contains its base index**, so loading one restores both. Tune the base through `loaded.base`, e.g. `loaded.base.nprobe = 32`.
- **Saving is safe to interrupt:** the file is written next to the target as `<path>.tmp` and renamed into place when complete, so a failed save never leaves a partial file or destroys an existing one.
- **Bad files raise `RuntimeError`:** a missing, truncated or corrupted file, or one that isn't a QaNN index, is rejected instead of loading garbage.
- **Files are versioned.** Newer releases keep loading files saved by older ones (files from 0.1.0a4, 0.1.0a6, 0.1.0a7, 0.1.0a10 and 0.1.0a11 still load). A file saved by a newer release than the one loading it raises `RuntimeError` rather than being misread.

`PQCodebook` on its own can't be saved yet; save the `IVFIndex` that uses it instead.

## Product quantization

`PQCodebook(dim, num_subspaces, centroids_per_subspace=256)` exposes the quantizer used inside `IVFIndex` for direct use.

| Method | Description |
| --- | --- |
| `train(data, max_iters=25, seed=None)` | Run k-means in each subspace over an `(n, dim)` array. |
| `encode(vec)` | Encode a `(dim,)` vector into a `uint8` array of `num_subspaces` codes. |
| `compute_adc_table(query)` | Precompute distances from a `(dim,)` query to every centroid; reuse it across many codes. |
| `distance_adc(table, code)` | Approximate squared L2 between the table's query and a code (asymmetric). |
| `distance_sdc(query_code, code)` | Approximate squared L2 between two codes (symmetric). |
| `dim()`, `num_subspaces()`, `centroids_per_subspace()` | Shape of the codebook. |

```python
pq = qann.PQCodebook(128, 16)
pq.train(x, seed=1)
codes = [pq.encode(v) for v in x[:1000]]
table = pq.compute_adc_table(x[0])
dists = [pq.distance_adc(table, c) for c in codes]
```

## Utilities

| Function | Description |
| --- | --- |
| `l2_distance(a, b)` | Row-wise squared L2 distance between two `(m, dim)` arrays; returns shape `(m,)`. |
| `cosine_distance(a, b)` | Row-wise cosine distance (1 minus cosine similarity) between two `(m, dim)` arrays. |
| `set_num_threads(n)` | Threads used for batch queries, adding with PQ and PQ training. `0` (default) uses every core; `1` runs serially. |
| `num_threads()` | Thread count currently in effect. |
| `__version__` | Installed version. |

## Threads and type hints

- **Thread-safe.** Queries, adds and training release Python's GIL while they run, so other Python threads keep running, and several threads can query the same index at once. An `add`, `train` or setting change waits for running queries on that index to finish (and blocks new ones until it's done), so concurrent use never sees a half-updated index. This also covers the base of a `RefineIndex`, e.g. `ivf.nprobe = 32` while queries run through the `RefineIndex` wrapping `ivf`.
- **Typed.** Wheels include type stubs, so editors autocomplete the API and type checkers such as mypy and Pylance/pyright check calls to it.

## More

See [ARCHITECTURE.md](https://github.com/maxtmiller/qann/blob/main/ARCHITECTURE.md) for the design, a file-by-file reference and SIFT1M benchmarks.
