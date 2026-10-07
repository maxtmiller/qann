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
- **IDs** are assigned in insertion order: the first vector added gets id 0, the next id 1, and so on across all `add` calls.
- **Distances** are squared Euclidean (L2) distances, smallest first. For cosine similarity, L2-normalize vectors before adding and querying.
- **Results** from `batch_query` are an `int64` id array and a `float32` distance array, both shaped `(num_queries, k)`. Rows with fewer than `k` hits are padded with id `-1` and distance `inf`.

## Indexes

Every index has the same core methods:

| Method | Description |
| --- | --- |
| `add(data)` | Add an `(n, dim)` array of vectors. |
| `batch_query(queries, k)` | k nearest neighbors for each row of an `(nq, dim)` array, run in parallel. Returns `(ids, dists)`. |
| `query(query, k)` | k nearest neighbors for a single `(dim,)` vector such as `x[i]`. Returns 1D `(ids, dists)`. |
| `size()` | Number of vectors added. |
| `dim()` | Vector dimension. |
| `save(path)` | Save the index to a file; see [Saving and loading](#saving-and-loading). |

### `FlatIndex(dim)`

Exact brute-force search: no training, 100% recall, and query cost that grows linearly with the number of vectors. Good for up to a few hundred thousand vectors, or as ground truth when measuring recall.

```python
index = qann.FlatIndex(128)
index.add(x)
ids, dists = index.batch_query(x[:10], 10)
```

### `IVFIndex(dim, nlist, nprobe=10)`

Partitions vectors into `nlist` clusters with k-means and, at query time, scans only the `nprobe` clusters closest to the query. Much faster than flat search at a small cost in recall.

| Member | Description |
| --- | --- |
| `train(data, max_iters=25, seed=None)` | Learn the cluster centroids from a sample of at least `nlist` vectors. Call once, before `add`. Pass `seed` for reproducible results. |
| `enable_pq(num_subspaces, centroids_per_subspace=256)` | Compress stored vectors with product quantization. Call before `train`. |
| `nprobe` | Property: clusters scanned per query. Can be changed at any time; higher means better recall and slower queries. |
| `pq_distance` | Property: how PQ codes are scored, `qann.PQDistance.ADC` (default, more accurate) or `qann.PQDistance.SDC`. |

`nlist` must be between 100 and 65535, and `nprobe` between 1 and `nlist`. A common starting point is `nlist` around `sqrt(n)` to `4 * sqrt(n)` and `nprobe` at 1% to 5% of `nlist`.

**With PQ** (`enable_pq`), each vector is split into `num_subspaces` slices and each slice is stored as one byte, so a 128-dim float32 vector (512 bytes) with 16 subspaces takes 16 bytes. `dim` must be divisible by `num_subspaces`. Distances become approximate, which lowers recall; wrap the index in a `RefineIndex` to recover it.

```python
ivf = qann.IVFIndex(128, nlist=1024, nprobe=16)
ivf.enable_pq(16)
ivf.train(x, seed=1)
ivf.add(x)
ivf.nprobe = 32   # trade speed for recall
```

### `RefineIndex(base, k_factor=10)`

Wraps a trained, empty approximate index. Each query fetches `k * k_factor` candidates from `base`, then re-ranks them with exact distances against full-precision copies of the vectors. Add vectors through the `RefineIndex`, not the base. `k_factor` is a property and can be changed at any time, and the read-only `base` property returns the wrapped index, e.g. to change `nprobe`.

This is the usual way to combine PQ's speed with near-exact ranking. It keeps the original vectors in memory alongside the PQ codes.

### `make_index(type, dim, opts=IndexOptions())`

Builds an index from a type and an options object, returning a `FlatIndex` or `IVFIndex`:

```python
opts = qann.IndexOptions()
opts.nlist, opts.nprobe, opts.pq_subspaces = 1024, 16, 16
ivf = qann.make_index(qann.IndexType.IVF, 128, opts)
ivf.train(x)
```

`IndexType` is `Flat` or `IVF`. `IndexOptions` fields and defaults: `capacity` (Flat initial reserve, 1024), `nlist` (100), `nprobe` (10), `pq_subspaces` (0, meaning no PQ) and `pq_centroids` (256).

## Saving and loading

Any index can be saved to a file and loaded back, including its training, so it doesn't have to be retrained or re-added each run:

```python
index.save("vectors.qann")          # str or pathlib.Path

index = qann.load("vectors.qann")   # returns a FlatIndex, IVFIndex or RefineIndex
ids, dists = index.batch_query(queries, 10)
```

- **Everything is restored:** vectors, ids, IVF centroids, PQ codebooks and settings such as `nprobe`, `pq_distance` and `k_factor`. A loaded index gives identical results and keeps accepting `add`.
- **A `RefineIndex` file contains its base index**, so loading one restores both. Tune the base through `loaded.base`, e.g. `loaded.base.nprobe = 32`.
- **Saving is safe to interrupt:** the file is written next to the target as `<path>.tmp` and renamed into place when complete, so a failed save never leaves a partial file or destroys an existing one.
- **Bad files raise `RuntimeError`:** a missing, truncated or corrupted file, or one that isn't a QaNN index, is rejected instead of loading garbage.
- **Files are versioned.** A file can be loaded by any QaNN release that supports its format version; if a future release changes the format, loading an older file raises `RuntimeError` rather than misreading it.

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
