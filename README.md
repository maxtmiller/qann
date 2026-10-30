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
- **Distances** are squared Euclidean (L2) distances by default, cosine distances with `metric=qann.Metric.Cosine`, or negative inner products with `metric=qann.Metric.InnerProduct` (see [Metrics](#metrics)); smallest first in every case.
- **Results** from `batch_query` are an `int64` id array and a `float32` distance array, both shaped `(num_queries, k)`. Rows with fewer than `k` hits are padded with id `-1` and distance `inf`.

## Indexes

Every index has the same core methods:

| Method | Description |
| --- | --- |
| `add(data, ids=None, *, metadata=None)` | Add an `(n, dim)` array of vectors, with `n` ids if the index uses custom ids, and optional metadata columns; see [Metadata](#metadata). Returns the vectors' ids. |
| `batch_query(queries, k, *, allowed=None, where=None)` | k nearest neighbors for each row of an `(nq, dim)` array, run in parallel. Returns `(ids, dists)`. `allowed` restricts results to those ids and `where` to vectors whose metadata matches; see [Filtered search](#filtered-search). |
| `query(query, k, *, allowed=None, where=None)` | k nearest neighbors for a single `(dim,)` vector such as `x[i]`. Returns 1D `(ids, dists)`. |
| `metadata_schema` | Each metadata column's name and type (`'int'`, `'float'` or `'str'`). |
| `update_metadata(ids, metadata)` | Overwrite metadata columns of existing vectors in place; see [Metadata](#metadata). |
| `add_metadata_column(name, type, *, default=None, ids=None, values=None)` | Add a metadata column (`'int'`, `'float'` or `'str'`) to the vectors already in the index; see [Metadata](#metadata). |
| `drop_metadata_column(name)` | Remove a metadata column and its values. |
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
| `enable_sq(bits=8)` | Store each vector as scalar codes of 8 or 4 bits per dimension instead of floats (see [SQ](#sq)). Call before `train`; not together with `enable_pq`. |
| `sq_bits` | Read-only property: 8 or 4 after `enable_sq()`, else 0. |
| `nprobe` | Property: clusters scanned per query. Can be changed at any time; higher means better recall and slower queries. |
| `pq_distance` | Property: how PQ codes are scored, `qann.PQDistance.ADC` (default, more accurate) or `qann.PQDistance.SDC` (not with `Metric.InnerProduct`). |
| `precomputed_tables` | Property: whether ADC queries use lookup tables precomputed at train time, which makes PQ queries faster. On by default; reads `True` once the tables exist (PQ enabled, trained, and `nlist * num_subspaces * centroids_per_subspace * 4` bytes within 256 MB). Set to `False` to free them. Always `False` with `Metric.InnerProduct`, which doesn't need them. |

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
- `num_subspaces` must be even and at most 1024, `centroids_per_subspace` must be 16 (the default with `fast_scan`), and `pq_distance` stays `ADC`: setting `SDC` raises `ValueError`.

#### Rotation

`enable_pq(..., rotate=True)` makes `train()` also learn a rotation of the vectors (PCA, with the variance spread evenly over the subspaces), and every vector is rotated before IVF and PQ see it. Rotations don't change distances, so results stay valid; only PQ's error changes. It helps when dimensions are correlated across subspaces, so plain PQ spends bits encoding the same information twice:

```python
ivf.enable_pq(64, fast_scan=True, rotate=True)
```

- On SIFT1M with fast-scan, recall@10 at `nprobe=16` goes from 0.635 to 0.718 without re-ranking, and `k_factor=4` reaches what `k_factor=8` did (0.988 vs 0.956 at `nprobe=64`). With re-ranking that is about 5% more queries per second at the same recall, after the cost of rotating each query.
- It does nothing for embeddings whose dimensions are already uncorrelated (GloVe: within 0.004 either way), and with heavy re-ranking (`k_factor=16`) there is little left to gain. Measure on your data.
- Training takes a little longer (0.4 s more on SIFT1M), and the index stores one `dim x dim` matrix (64 KB at 128 dimensions).


#### SQ

`enable_sq(8)` stores each vector as one byte per dimension (on SIFT, 128 bytes instead of 512) and `enable_sq(4)` as half a byte; `train` also learns each dimension's range. Unlike PQ there are no codebooks, and 8 bits loses little recall, so it needs no `RefineIndex`: it is the IVF option for high recall without keeping the raw vectors in memory. 4 bits loses more and should be wrapped in a `RefineIndex(k_factor=8)`.

```python
ivf = qann.IVFIndex(128, nlist=1024, nprobe=16)
ivf.enable_sq(8)
ivf.train(x, seed=1)
ivf.add(x)
```

On SIFT1M (all cores, busy machine), 8 bits reached recall@10 0.925 at `nprobe=16` and 0.985 at 64 (floats: 0.932 and 0.995) at 1.3 to 1.4x the queries per second of float lists, in a quarter of the memory; 4 bits + `RefineIndex(k_factor=8)` matched the floats' recall exactly at the same speed-up. On one thread the gain mostly disappears: SQ helps because the float scan is limited by memory bandwidth. For the fastest queries when raw vectors fit in memory, fast-scan PQ + `RefineIndex` is still quicker, and HNSW-SQ8 is faster at the same recall for about twice the memory.

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
- **Choosing `M`:** the default 16 suits data like SIFT. For high-dimensional embeddings (word or sentence vectors), use `M=32` to `48` with `ef_construction=300` to `400`: on GloVe-100, `M=48` answered 1.5x more queries per second than `M=16` at 0.90 and 0.95 recall, and only `M=48` reached 0.99. It builds about 5x slower and adds 256 bytes of links per vector.

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

`IndexType` is `Flat`, `IVF` or `HNSW`. `IndexOptions` fields and defaults: `metric` (`Metric.L2`), plus `capacity` (Flat initial reserve, 1024), `nlist` (100), `nprobe` (10), `pq_subspaces` (0, meaning no PQ), `pq_centroids` (256) and `ivf_sq_bits` (0; 8 or 4 calls `enable_sq()`) for IVF, and `hnsw_m` (16), `ef_construction` (200), `ef_search` (64) and `hnsw_sq_bits` (0 for floats; 8 or 4 calls `enable_sq()`) for HNSW.

## Metrics

Every index takes `metric=qann.Metric.L2` (the default), `metric=qann.Metric.Cosine` or `metric=qann.Metric.InnerProduct`, fixed at construction and saved with the index. Most text and image embedding models are meant to be compared by cosine similarity, so use `Cosine` for them:

```python
index = qann.HNSWIndex(384, metric=qann.Metric.Cosine)
index.add(embeddings)
ids, dists = index.batch_query(query_embeddings, 10)   # dists = 1 - cosine similarity
```

- **Distances** are `1 - cosine similarity`, from 0 (same direction) to 2 (opposite), matching `qann.cosine_distance`. Similarity is `1 - dist`.
- **Vector length doesn't matter:** each vector is scaled to length 1 when it's added and each query when it's run, so you don't need to normalize anything yourself. A zero vector has no direction, so adding or querying one raises `ValueError`.
- **It works with every index,** including PQ, fast-scan and HNSW. IVF learns its clusters on the scaled vectors, and a `RefineIndex` uses its base's metric. `index.metric` reports it.

### Inner product

Use `InnerProduct` when a vector's length carries meaning, so the largest dot product `q . x` should come first rather than the closest direction: recommendation (user and item factors, where an item's length reflects its popularity), multi-vector retrieval such as ColBERT or MUVERA, attention keys, and embedding models trained with a dot-product loss.

```python
index = qann.HNSWIndex(64, metric=qann.Metric.InnerProduct)
index.add(item_factors)
ids, dists = index.batch_query(user_factors, 10)   # dists = -(q . x)
```

- **Distances** are `-(q . x)`, so results stay smallest first; the inner product is `-dist`. Vectors are used as given: nothing is normalized, and zero vectors are allowed.
- **It works with every index:** Flat, IVF (floats, SQ, PQ, fast-scan, rotation), HNSW (floats and SQ) and `RefineIndex`. PQ can only use `ADC`: setting `pq_distance` to `SDC` raises `ValueError`, and `precomputed_tables` reads `False`, since one lookup table per query already serves every list.
- **HNSW** answers queries by `-(q . x)` but links its graph as if every stored vector had one more coordinate, `sqrt(M^2 - |x|^2)` with `M` the largest stored length. That turns inner product search into L2 search for the links, which matters a lot: at `ef_search=64`, recall@10 was 0.96 instead of 0.05 on ann-benchmarks' Last.fm factors and 0.50 instead of 0.05 on VIBE's llama-128-ip, against linking by raw inner product as hnswlib does. Only the length of each vector is stored for it, and `M` simply grows when longer vectors are added later.
- **IVF** clusters by L2 as usual and probes the lists whose centroids have the largest `q . c`. With exact lists and `nprobe=64`, recall@10 was 0.98 on Last.fm, 0.88 on VIBE's yi-128-ip and 0.84 on llama-128-ip; with fast-scan PQ (rotated, `dim / 2` subspaces) and `RefineIndex(k_factor=8)`, 0.91, 0.81 and 0.73.

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
- **Default ids never repeat.** New vectors continue after the highest id ever assigned, even after deletes and `compact()` (delete ids 0 to 9, compact, add one more, and it gets id 10). `add()` returns the ids it assigned, so keep those rather than counting. For data that changes (updates, deletes, re-adds), prefer `custom_ids=True` with ids from your own data.
- For `make_index`, set `IndexOptions.custom_ids`.

## Filtered search

To search only some of the vectors ("nearest products in this category", "documents this user may see"), pass the ids that may be returned as `allowed`:

```python
in_stock = product_ids[stock > 0]   # computed from your own metadata
ids, dists = index.batch_query(queries, 10, allowed=in_stock)
```

- **Ids from your own data:** keep metadata wherever you already do and turn the condition into a list of ids, or store it in the index and filter with `where` (see [Metadata](#metadata)). `allowed` takes the same kinds of ids as `remove` (any integer array, list or `range`), with or without custom ids, and one set applies to every row of a `batch_query`.
- **Unknown and deleted ids in `allowed` are ignored.** An empty set returns no results; fewer than `k` allowed vectors gives fewer results (rows padded with `-1` / `inf`).
- **Filtering happens during the search,** not after it, so a selective filter still returns `k` results instead of the few that happen to be in an unfiltered top `k`.
- **Works on every index.** `FlatIndex` scores only the allowed vectors. `IVFIndex` skips the others while scanning, and if its `nprobe` lists hold fewer than `k` allowed vectors it keeps probing the next nearest lists until it has `k`. `HNSWIndex` walks its graph through every node but returns only allowed ones; when few are allowed (below about `sqrt(ef_search * M * size)`, ~3% of SIFT1M), walking the graph would cost more than scoring the allowed vectors directly, so it does that instead, which is also exact. A `RefineIndex` passes the filter to its base and re-ranks as usual.
- **Recall by index:** `FlatIndex` is exact, and `HNSWIndex` holds or gains recall as the filter narrows. `IVFIndex` loses recall as it narrows (on SIFT1M at `nprobe=16`: 0.93 unfiltered, 0.85 with 5% allowed, 0.52 with 0.1%), because the nearest allowed vectors spread over more lists than `nprobe`; raise `nprobe`, or for very selective filters prefer `HNSWIndex` or `FlatIndex`. A `RefineIndex` over IVF is exact when at most `2 * nlist` vectors are allowed (about 0.2% on SIFT1M): it then scores the allowed vectors directly from its raw copies, which is also faster there. Full numbers are in [ARCHITECTURE.md](https://github.com/maxtmiller/qann/blob/main/ARCHITECTURE.md#benchmarks-benchmarksbench_indexespy).
- **Cost** is a pass over `allowed` per call (to look up each id), plus one bit per vector in the index.

## Metadata

Store attributes with each vector and filter on them directly:

```python
index.add(vectors, metadata={"category": categories, "price": prices, "year": years})

f = qann.field
ids, dists = index.batch_query(queries, 10, where=(f("category") == "shoes") & (f("price") < 50))
ids, dists = index.batch_query(queries, 10, where=f("year").isin([2024, 2025]) | f("price").is_null())

# or as a SQL WHERE clause, with values passed separately
ids, dists = index.batch_query(queries, 10, where="category = 'shoes' AND price < 50")
ids, dists = index.batch_query(queries, 10, where=("user_id = ? AND year IN (?, ?)", [uid, 2024, 2025]))
```

- **Columns** are named and hold ints, floats or strings, one value per vector, as NumPy arrays or lists (bool arrays become ints). A column's type is fixed by the first add that has it, or by `add_metadata_column`. Ints are accepted into a float column (so whole-number prices work; ints beyond 2^53 are rounded to the nearest float), but floats into an int column raise `ValueError`.
- **Missing values:** `None` in a column's values is a null. A column added later is null for the vectors already in the index, and an add that leaves a column out stores null for its vectors. Nulls fail every comparison and `isin`, so `~(f("price") < 50)` includes vectors with no price; use `f("price").is_null()` to select or exclude them.
- **Filters:** `==`, `!=`, `<`, `<=`, `>`, `>=`, `.isin([...])`, `.between(lo, hi)` (both ends included) and `.is_null()`, combined with `&` (and), `|` (or) and `~` (not). Parenthesize comparisons, since `&` binds tighter than `<`: `(f("a") < 1) & (f("b") == 2)`. Python's `and`, `or`, `not` and chained comparisons such as `0 < f("a") < 5` raise `TypeError` instead of silently dropping part of the filter. For a long chain built in code, use `qann.any_of(filters)` / `qann.all_of(filters)`: adding terms one `|` at a time copies the chain each step, so it gets slow past a few thousand terms. For one column against many values, `f("price").isin(values)` is better still: it's one condition, so it's scanned once per query instead of once per value.
- **SQL strings:** `where` also takes a SQL `WHERE` clause: `=`, `!=` / `<>`, `<`, `<=`, `>`, `>=`, `[NOT] IN (...)`, `IS [NOT] NULL`, `[NOT] BETWEEN x AND y`, `AND`, `OR`, `NOT` and parentheses, with SQL's precedence and case-insensitive keywords. Strings are `'...'` (`''` for a quote); quote column names that are keywords or contain non-ASCII characters as `"..."`. Numbers that overflow, such as `1e400`, raise `ValueError`. `LIKE`, functions, arithmetic and comparing two columns aren't supported. A syntax error raises `ValueError` with its column. Filters can nest up to 256 levels of parentheses or `NOT`; long `AND` / `OR` chains don't count toward that. A string may have up to 1024 conditions (comparisons, `IN`, `IS NULL`, `BETWEEN`; an `IN` list counts once, however many values it has), so filter strings from users can't make a query scan its columns thousands of times; `qann.set_max_filter_conditions(n)` changes the limit for the whole process. Filters built with `qann.field` aren't limited.
- **Parameters:** pass values as `(sql, [values])` for `?` or `(sql, {name: value})` for `:name` instead of pasting them into the string. They are bound after parsing, so user input can't change the filter (`"1 OR 1 = 1"` is just a value). `qann.where(sql, params)` returns the filter, to reuse it or combine it with `qann.field` filters through `&`, `|` and `~`.
- **`NOT` vs `~`:** SQL's `NOT` excludes nulls, as in SQL: `NOT price < 50` doesn't match vectors with no price, while `~(f("price") < 50)` does.
- **Types:** ints and floats compare as numbers; strings compare with strings (`<` is alphabetical). Filtering on a column the index doesn't have, or comparing a number column with a string, raises `ValueError`.
- **Same search as `allowed`:** a `where` filter becomes the set of matching vectors and runs the [filtered search](#filtered-search) for every index, with the same recall behavior. With both `where` and `allowed`, only vectors passing both are returned.
- **Cost:** each query call scans the columns its filter names once, shared by every row of a `batch_query`: about 0.4 ms per condition per million vectors. Strings are stored once per distinct value, plus 4 bytes per vector; ints and floats take 8 bytes per vector; each column also takes 1 byte per vector for its null flag.
- **Updating:** `index.update_metadata(ids, {"price": [10, None]})` overwrites the given columns for those vectors in place (no re-insert), with one value per id; other columns keep their values. A column the index doesn't have raises `ValueError`, so a typo can't create one.
- **New columns for existing vectors:** `index.add_metadata_column("in_stock", "int", default=1)` gives every vector that value; `index.add_metadata_column("tag", "str", ids=tagged, values=tags)` sets those vectors (one value per id, `None` for null) and leaves the rest null, in one step; with neither, every vector is null. `default` can't be combined with `ids` / `values` (`TypeError`). Later adds that leave the column out store null. `drop_metadata_column(name)` removes a column, freeing its name; `compact()` never removes columns, even ones left with no values. Unknown, deleted or repeated ids raise `ValueError`, and a failed call changes nothing. Strings no vector uses any more are freed by `compact()`. Re-adding a removed id doesn't bring back its old metadata: pass it again with the `add`.
- **Kept with the index:** `remove` and `compact` keep metadata with its vector, and `save` / `qann.load` store it. A `RefineIndex` holds the metadata itself; add through it as usual.

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
- **Files are versioned.** Newer releases keep loading files saved by older ones (files from 0.1.0a4, 0.1.0a6, 0.1.0a7, 0.1.0a10, 0.1.0a11 and 0.1.0a12 still load). A file saved by a newer release than the one loading it raises `RuntimeError` rather than being misread.

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
