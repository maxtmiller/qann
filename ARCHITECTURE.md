# QANN: Architecture & File Reference

## Overview

QANN (Quantized Approximate Nearest Neighbors) is a C++20 vector search library with a Python extension module, `qann`, built on nanobind. The C++ API lives in namespace `vecengine` under `include/vecengine/`. It has three layers:

1. **Distance kernels**: squared L2, cosine distance and inner product with scalar, AVX2+FMA, AVX-512 and ARM NEON paths, selected at compile time.
2. **Quantization and clustering**: a shared k-means helper (BLAS-accelerated when a CBLAS is available), a Product Quantization (PQ) codebook and an 8- / 4-bit scalar quantizer (SQ).
3. **Indexes**: `FlatIndex` (exact brute force), `IVFIndex` (inverted file, optionally PQ- or SQ-compressed), `HNSWIndex` (graph, optionally SQ-compressed) and `RefineIndex` (exact re-ranking wrapper), behind a common `DenseIndex` interface; `InvertedIndex` (exact sparse search) and `SparseApproxIndex` (approximate) behind `SparseIndex`; both families share `IndexBase`.

All arithmetic lives in C++. Python gets zero-copy access to NumPy arrays via `nb::ndarray`. `benchmarks/bench_indexes.py` measures recall vs QPS on the SIFT datasets.

---

## Directory Structure

```
qann/
├── .github/workflows/ci.yml        # build + test (Ubuntu, macOS), recall gate
├── .github/workflows/wheels.yml    # cibuildwheel wheels, published to PyPI on v* tags
├── CMakeLists.txt                  # build orchestrator
├── build.sh                        # configure + build + test in one command
├── pyproject.toml                  # PEP 517 packaging (scikit-build-core)
├── include/vecengine/              # public API
│   ├── aligned_allocator.hpp       # std::allocator with 32/64-byte alignment
│   ├── distance.hpp                # distance function declarations
│   ├── distance_inl.hpp            # inline compile-time SIMD dispatch
│   ├── index.hpp                   # IndexBase, DenseIndex + Neighbor
│   ├── metadata.hpp                # Metadata columns, Expr filters, detail::MetadataStore
│   ├── flat_index.hpp              # exact brute-force index
│   ├── ivf_index.hpp               # IVF index, optional PQ
│   ├── hnsw_index.hpp              # HNSW graph index
│   ├── pq.hpp                      # PQCodebook + PQDistance
│   ├── sq.hpp                      # SQ: 8- and 4-bit scalar quantizer used by HNSWIndex
│   ├── refine_index.hpp            # exact re-ranking wrapper
│   ├── sparse_index.hpp            # SparseIndex: base of the sparse indexes, SparseRows input
│   ├── inverted_index.hpp          # InvertedIndex: exact search over sparse vectors
│   ├── sparse_approx_index.hpp     # SparseApproxIndex: approximate search over sparse vectors
│   └── index_factory.hpp           # make_index() + IndexOptions, save_index() / load_index()
├── src/
│   ├── distance.cpp                # scalar / AVX2 / NEON kernels
│   ├── kmeans.hpp / kmeans.cpp     # internal k-means helper (not public API)
│   ├── parallel.hpp                # internal parallel_for (not public API)
│   ├── serialize.hpp               # internal binary I/O helpers + file header (not public API)
│   ├── fast_scan.hpp / fast_scan.cpp  # 4-bit PQ block layout, uint8 tables, NEON / AVX2 / SSSE3 / scalar kernels
│   ├── rotation.hpp / rotation.cpp    # PCA-balanced rotation for IVF-PQ: Jacobi eigensolver, eigenvalue allocation
│   ├── id_map.cpp                  # IdMap: slots <-> user ids, deletions
│   ├── metadata.cpp                # Expr filters, MetadataStore columns: evaluate to slot bits, compact, save/load
│   ├── index.cpp                   # IndexBase metadata calls, DenseIndex::add_batch / query_batch
│   ├── pq.cpp
│   ├── sq.cpp                      # SQ train / encode / decode, NEON / AVX2 distance
│   ├── flat_index.cpp
│   ├── ivf_index.cpp
│   ├── hnsw_index.cpp
│   ├── refine_index.cpp
│   ├── sparse_index.cpp            # SparseIndex::add_batch / query_batch, canonical rows
│   ├── inverted_index.cpp
│   ├── sparse_approx_index.cpp
│   └── index_factory.cpp
├── tests/
│   ├── test_distance.cpp           # Catch2 v3 tests for every layer
│   ├── test_serialize.cpp          # save/load round trips and corrupt-file rejection
│   ├── test_id_map.cpp             # IdMap, custom ids, deletion and compaction per index, format v2/v3
│   ├── test_fast_scan.cpp          # fast-scan packing, kernels, table rounding, IVF fast-scan
│   ├── test_hnsw.cpp               # HNSW recall, determinism, deletion, compaction, save/load, parallel build, SQ
│   ├── test_ivf_sq.cpp             # IVF SQ8/SQ4: recall, API checks, save/load, corrupt files, compaction, cosine
│   ├── test_metadata.cpp           # metadata store, every filter operator, nulls, where == allowed on every index, save/load, corrupt files, SQL parser (fuzzed against three-valued logic)
│   ├── test_rotation.cpp           # eigensolver, PCA-balanced rotation, rotated IVF-PQ recall, save/load, corrupt files
│   ├── test_filter.cpp             # filtered search on every index: recall vs filtered brute force, tiny / empty / full sets, ids, deletes
│   ├── test_metric.cpp             # cosine metric: exact ranking, normalize_rows, zero vectors, save/load, format v4 files
│   ├── test_inner_product.cpp      # inner product: kernels, exact ranking on every index type, HNSW vs explicit L2 transform, PQ, save/load
│   ├── test_python.py              # pytest smoke, save/load, ids and threading tests
│   ├── test_sparse.cpp / .py       # InvertedIndex and SparseApproxIndex against brute-force dot products, input forms, ids, filters, compaction, save/load, corrupt files
│   ├── load_dense.hpp              # test helpers: load_index() / make_index() checked to give a DenseIndex
│   └── data/                       # index files saved by 0.1.0a4 (v1), 0.1.0a6 (v2), 0.1.0a7 (v3), 0.1.0a10 (v4), 0.1.0a11 (v5), macOS-made SQ8 / SQ4 files (v6), IVF SQ (v7), rotated IVF (v8), inner product, metadata and sparse files (v9) + their results
├── benchmarks/
│   ├── bench_distance.cpp          # Google Benchmark distance kernels
│   ├── bench_indexes.py            # recall vs QPS on SIFT
│   ├── compare.py                  # ref vs working tree: recall, QPS, build times
│   ├── hnsw_m_sweep.py             # HNSW recall and QPS by M on GloVe-100 and SIFT1M (issue #33)
│   ├── filter_selectivity.py       # filtered search recall and QPS vs selectivity on SIFT1M (issue #37)
│   ├── profile_query.cpp           # per-section query time breakdown on SIFT1M (VECENGINE_PROFILE=ON)
│   ├── profile_fastscan.cpp        # per-section single-query profile of IVF fast-scan under a RefineIndex (VECENGINE_PROFILE=ON)
│   ├── filter_sweep.cpp            # thresholds for choosing how a metadata filter is evaluated (issue #72)
│   ├── where_cost.py               # single-query where= cost: filter evaluation vs search (issue #72)
│   ├── sparse_bm25.py              # InvertedIndex BM25 on BEIR: QPS and nDCG@10, pruning off and on (issue #59)
│   ├── sparse_bigann.py            # InvertedIndex and SparseApproxIndex on the big-ann sparse track (SPLADE, issue #59)
│   ├── fastscan_recall.py          # recall of 4-bit codes and uint8 tables on SIFT1M (fast-scan study)
│   ├── fastscan_bench.cpp          # NEON fast-scan kernel vs the scalar ADC scan (ARM only)
│   ├── baseline_siftsmall.json     # recall baseline for the CI gate
│   ├── data/                       # downloaded datasets (git-ignored)
│   └── results/                    # CSV + PNG output (git-ignored)
└── bindings/python_bindings.cpp    # nanobind module
```

---

## Layer 1: Distance Kernels

### `distance.hpp` / `distance_inl.hpp` / `distance.cpp`

```cpp
float l2_distance(const float* a, const float* b, size_t n);      // squared L2
float cosine_distance(const float* a, const float* b, size_t n);  // 1 - cos(a, b)
```

`l2_distance` returns **squared** L2. It ranks neighbors identically to true L2 and skips the `sqrt`. Every index and PQ distance in this repo is squared L2 internally; `Metric::Cosine` reuses it on unit vectors (see `DenseIndex`). `inner_product` (same three paths) serves `Metric::InnerProduct`.

The unsuffixed functions dispatch at compile time in `distance_inl.hpp`:

| Build | Path |
|---|---|
| x86-64 Release (`-mavx2 -mfma`) | `_avx2`: 8 floats per iteration, `_mm256_fmadd_ps` |
| arm64 (Apple Silicon, Graviton) | `_neon`: 4 floats per iteration |
| anything else | `_scalar` |

The `_scalar`, `_avx2` and `_neon` symbols always exist (non-native paths forward to scalar) so tests and benchmarks can call any of them directly. There is no runtime CPU detection, function pointer or virtual call.

These kernels live in `distance.cpp` and are not inlined into other translation units, so each call has a fixed overhead. That is negligible on full-length vectors (128 floats) but dominates on 4 to 16 float PQ slices, which is why the PQ table builds below avoid calling them per centroid.

Cosine returns `1.0f` for zero-magnitude inputs rather than dividing by zero.

### `aligned_allocator.hpp`

`AlignedAllocator<T, Align>` uses `posix_memalign` (or `_aligned_malloc` on MSVC). Aliases: `AVX2Vector<T>` (32-byte) and `AVX512Vector<T>` (64-byte). Used by the benchmarks; the indexes use plain `std::vector` and unaligned loads.

---

## Layer 2: Clustering and Quantization

### `src/kmeans.hpp` / `kmeans.cpp` (`vecengine::detail`)

Shared by `IVFIndex` and `PQCodebook`, so Lloyd's algorithm exists exactly once.

```cpp
KMeansResult kmeans(const float* data, size_t n, size_t stride, size_t d, size_t k, size_t max_iters);
void assign_nearest(const float* data, size_t n, size_t stride, const float* centroids, size_t k, size_t d, uint32_t* out);
size_t nearest_centroid(const float* point, const float* centroids, size_t k, size_t d);
```

- Point `i` is the `d` floats at `data + i * stride`. `stride` and `d` are separate so PQ can cluster one subspace slice of every row in place: pass `data + s * sub_dim`, `stride = dim`, `d = sub_dim`. IVF passes `stride = d = dim`.
- Returned centroids are packed (`k * d`, centroid `c` at `c * d`), never strided.
- Init picks `k` distinct random points (partial Fisher-Yates). Empty clusters are handled by `reseed_empty_clusters`: one pass keeps the E furthest points (distance to their assigned centroid, skipping points within `1e-6`) in a size-E min-heap, and each of the E empty clusters gets a distinct one. If the heap runs out, the largest cluster is split FAISS-style [4]: its centroid is copied and the two copies are scaled by `1 ± 1/1024`, and its size is halved so the next split picks another cluster.
- A final assignment pass runs after the last update, so `assignments` match the returned `centroids`. IVF relies on this to compute PQ training residuals.

**Assignment is ~99.9% of k-means time**, so it is done in bulk by `assign_nearest`:

- **With BLAS (`VECENGINE_USE_BLAS`)**: `argmin_c ‖x − c‖² = argmin_c (‖c‖² − 2 x·c)`, and the `x·c` terms for a block of 1024 points against all centroids are one `cblas_sgemm` call. On Apple that is Accelerate, which runs on the M-series matrix unit; elsewhere any BLAS that ships `cblas.h` (OpenBLAS, netlib CBLAS). `lda = stride` lets PQ subspaces use strided input without copying. The per-row argmin is branchless (conditional selects).
- **Without BLAS**: falls back to `nearest_centroid` per point.

Measured on SIFT's 100k learn set: IVF assignment (k = 1000) 1.54 s → 0.086 s (18×), PQ subspaces 2.9 to 3.7×, with identical assignments. The `‖c‖² − 2x·c` scores are only used for ranking, never as distances.

`nearest_centroid` is the single-point scan, still used by `IVFIndex::add`. `IVFIndex::add_batch` uses `assign_nearest`.

`kmeans` takes an optional `seed`; without one it seeds from `std::random_device`, so training is non-deterministic run to run. `IVFIndex::train` and `PQCodebook::train` take an optional seed too (Python: `train(x, seed=1)`): coarse k-means uses `seed`, PQ uses `seed + 1` onward, one per subspace. With the same seed and the same BLAS, results are bit-identical, which the recall gate relies on.

### `pq.hpp` / `pq.cpp`: `PQCodebook`

Product quantization [1]: splits a `dim`-vector into `m = num_subspaces` slices of `sub_dim = dim / m` floats and quantizes each slice to one of `K = centroids_per_subspace` (max 256) centroids. A vector becomes `m` bytes: at `dim = 128`, `m = 16` that is 512 bytes down to 16.

| Member | Layout | Used by |
|---|---|---|
| `centroids_` | `m * K * sub_dim`, `[subspace][centroid][dim]` | reading whole centroids |
| `centroids_t_` | `m * sub_dim * K`, `[subspace][dim][centroid]` (transposed copy) | ADC and SDC table builds |
| `sdc_table_` | `m * K * K`: distance between centroids `c1`, `c2` of subspace `s` at `s*K*K + c1*K + c2` | `distance_sdc` |

All three are built in `train()`.

- `train(vectors, n)`: one `kmeans` call per subspace, run in parallel with `parallel_for`, then the transposed copy, then `sdc_table_`.
- `compute_adc_table(query, out)` + `distance_adc(table, code)`: **ADC**. The caller-provided `out` holds `m * K` distances from each query slice to every centroid (`table[s * K + c]`); the distance is `m` lookups. `distance_adc` is defined in the header so it inlines into the IVF scan loop.
- `compute_inner_table(query, out)` and `compute_list_term(c, out)`: the two halves of IVF's precomputed tables (see `IVFIndex`), in the same `[s * K + c]` layout. The first gives `−2⟨q_s, y_sj⟩`, the second `‖y_sj‖² + 2⟨c_s, y_sj⟩` for one coarse centroid `c`. Both use the `centroids_t_` loop shape.
- `encode(vec)`: builds the ADC table for `vec`, then takes the argmin of each subspace's row (a vectorized min pass, then a short search for its position).
- `distance_sdc(query_code, code)`: **SDC**. Both sides are codes; `m` lookups into `sdc_table_`, no float math.

**Why `centroids_t_`:** the table loops compute, for one dimension, `out[c] += (q_d − centroid_c[d])²` over all `K` centroids. In `[dim][centroid]` order those `K` values are contiguous, so the compiler vectorizes across centroids with no function calls or horizontal sums. The previous version made `K * m` separate `l2_distance` calls on 4 to 16 floats each (8,192 calls for m = 32). The change made ADC queries 2.0× (m = 8) to 3.6× (m = 32) faster on siftsmall with identical results.

Both distances work because squared L2 decomposes across disjoint slices: `‖a − b‖² = Σₛ ‖aₛ − bₛ‖²`. ADC keeps the query exact and is more accurate; SDC quantizes both sides.

`PQDistance { ADC, SDC }` lives here and selects the mode in `IVFIndex`.

### `src/fast_scan.hpp` / `fast_scan.cpp` (`vecengine::detail`): PQ fast-scan

With 16 centroids per subspace a code is 4 bits, and a row of the ADC table is 16 entries. Rounded to bytes, a row fits one 128-bit register, so one shuffle instruction (`vqtbl1q_u8` on NEON, `_mm_shuffle_epi8` on SSSE3) does 16 table lookups at once. AVX2's `_mm256_shuffle_epi8` shuffles each 128-bit lane separately, so one instruction does the 16 lookups of two adjacent subspaces (their table rows and codes are adjacent 16-byte runs), which is why fast-scan needs an even `m`. This is the fast-scan technique of André et al. [3]; FAISS implements it as `IndexIVFPQFastScan` [4].

- **Block layout.** `IVFIndex` stores a fast-scan list's codes in blocks of `kFsBlock = 32` vectors, `fs_block_bytes(m) = m * 16` bytes each: per subspace, 16 bytes where byte `j` holds vector `j` in its low nibble and vector `j + 16` in its high nibble. The last block is zero-padded; the list's `ids.size()` says how many entries are real. `fs_set` / `fs_get` write and read one code (used by `add`, `add_batch`, `compact` and tests; never in the scan).
- **`quantize_lut(table, m, lut, scale, offset)`** rounds the `m × 16` float table: each row is shifted by its minimum (rows can be negative with precomputed tables, since `C = −2⟨q, y⟩`), and one scale for all rows makes the widest row span 0..`kLutMax` = 63 (`scale = 1` if every row is flat). A distance is `sum / scale + offset`, where `offset` is the sum of the row minimums, accurate to `m / (2 · scale)`.
- **`fast_scan_block(lut, block, m, out)`** writes the 32 `uint16_t` sums of a block: per subspace, two lookups (low nibbles, high nibbles); the lookups of up to 4 subspaces are added in bytes, then widened into four 8-lane accumulators. `m <= kFsMaxSubspaces` (1024) keeps the sums below 65536 (1024 × 63 = 64,512). The AVX2 version sums 4 subspaces per lane (8 per register) before widening and adds the two lanes at the end. The NEON, AVX2 and SSSE3 versions must equal `fast_scan_block_scalar` exactly (tested, including all-`kLutMax` tables and `m` not a multiple of 8).
- **Why 63, not 255.** The kernel is bound by vector arithmetic, not memory, and with byte-wide entries the four widening adds per subspace were half its work. Capping entries at 63 lets four subspaces share one set of widening adds (4 × 63 = 252 fits a byte), 1.22× faster per block at `m = 64`. Rounding error grows 4× to `m / (2 · scale)` with the smaller scale; on SIFT1M PQ64×4 recall@10 without re-ranking moves 0.638 → 0.635 at `nprobe = 16`, and with `k_factor = 16` it is unchanged (0.931, 0.995 at 64).
- **`fast_scan_block_mask(lut, block, m, limit, out)`** is the same kernel plus the threshold test: it compares the four accumulators with `limit` while they are in registers and returns a 32-bit mask, bit `v` set when `out[v] < limit` (NEON: compare, keep one power-of-two weight per lane, horizontal add; SSSE3: flip the top bit so a signed compare orders unsigned values, pack to bytes, `movemask`). The scan converts the k-th-best distance to `ceil((threshold − offset) · scale)`, drops the padding bits, and visits only set bits with `countr_zero`.
- `quantize_lut` has a NEON version (four floats per instruction for the row min/max and the rounding) that matches `quantize_lut_scalar` byte for byte; it takes ~0.15 µs per list instead of ~0.6 µs at `m = 64`. The SSE and AVX2 versions (4 and 8 floats per instruction) match it the same way. Rounding is half to even (`std::nearbyint`), one instruction everywhere (`vcvtnq_u32_f32`, `_mm256_cvtps_epi32`); half away from zero, as `lround` does, took six on x86. The row minimums go into `offset` through 4 interleaved partial sums rather than one chain of `m` dependent adds, in the same order on every platform.

Measured on SIFT1M (`benchmarks/fastscan_recall.py`, with 8-bit tables): rounding the tables to bytes costs no recall (0.482 vs 0.482 for PQ32×4, 0.672 vs 0.673 for PQ64×4, flat PQ over 100k vectors). The 4-bit codes do cost recall before re-ranking (PQ64×4 0.638 vs PQ32×8 0.706 at the same 32 bytes), but with re-ranking (`k_factor = 16`) PQ64×4 matches PQ32×8: 0.931 vs 0.932 at `nprobe = 16`, 0.995 vs 0.995 at 64.

### `sq.hpp` / `sq.cpp`: `SQ` (`vecengine::detail`)

Scalar quantization with `bits` = 8 or 4, used by `HNSWIndex::enable_sq()`. `train` sets each dimension's range to the min and max of the sample: `vmin[d]` is code 0 and `step[d] = (max - min) / max_code()` the spacing (`max_code()` is 255 or 15; `step` is 0 for a constant dimension, which then always encodes as 0). 8-bit codes take one byte per dimension; 4-bit codes are packed two per byte, dimension `2i` in the low nibble of byte `i` and `2i + 1` in the high nibble, so a vector takes `code_bytes()` = `ceil(dim / 2)` bytes and an odd `dim` leaves the last high nibble 0. `encode` rounds to the nearest level and clamps values outside the range; `decode` returns `vmin + code * step`, so a decoded value is within `step / 2` of the original inside the range. `distance(q, code)` is the squared L2 from a float query to the decoded vector without writing it out. Both widths share one scoring path: NEON widens 16 one-byte codes to four float lanes and computes `q - (vmin + c * step)` with FMAs into two accumulators, and for 4 bits it first splits 16 packed bytes into low and high nibbles and interleaves them (`vzip1q_u8` / `vzip2q_u8`) into 32 one-byte codes in dimension order. AVX2 does the same 8 dimensions at a time (`_mm_unpacklo_epi8` for the nibbles). A scalar loop handles the tail. Unlike PQ there are no codebooks: each dimension is independent, so nothing needs k-means, and SQ8 loses far less than PQ at 4x compression; SQ4's 16 levels are coarse enough that it needs re-ranking.

---

## Layer 3: Indexes

### `index.hpp`: `IndexBase` and `DenseIndex`

Indexes form two families under `IndexBase`, which holds what doesn't depend on the vectors' form: the `IdMap` accessor, `remove`, `compact`, `save`, `size` / `slots`, the metadata store and its methods, `where_filter`, the mutex and the `wrapped()` flag. `DenseIndex` (named `Index` before 0.1.0a24) adds dense vectors: `dim`, `metric`, the float `add` / `query` methods and the `*_impl` hooks, implemented by `FlatIndex`, `IVFIndex`, `HNSWIndex` and `RefineIndex`. `SparseIndex` (`sparse_index.hpp`) is the sparse family, implemented by `InvertedIndex` and `SparseApproxIndex`. Each class is one search algorithm, fixed at construction; search settings are setters. The Python bindings mirror the tree: `qann.IndexBase` carries `save`, `remove`, `compact`, `size` and the metadata methods, `qann.DenseIndex` and `qann.SparseIndex` add `add` / `query` / `batch_query` for their kind of vectors, and `qann.Index` is kept as an alias of `qann.DenseIndex` (`m.attr("Index")`; the stubs show it as a `TypeAlias`).

```cpp
void add(span<const float> vec);                  // non-virtual: prepare for the metric, then *_impl
void add_batch(span<const float> vecs, size_t n, const int64_t* ids = nullptr, const Metadata& metadata = {});
vector<Neighbor> query(span<const float> vec, size_t k) const;
vector<vector<Neighbor>> query_batch(span<const float> queries, size_t num_queries, size_t k) const;
vector<Neighbor> query(span<const float> vec, size_t k, span<const int64_t> allowed) const;     // filtered
vector<vector<Neighbor>> query_batch(span<const float> queries, size_t num_queries, size_t k,
                                     span<const int64_t> allowed) const;
vector<Neighbor> query(span<const float> vec, size_t k, const Expr& where) const;              // metadata filter
// also query(vec, k, where, allowed) and query_batch(..., where) / query_batch(..., where, allowed)
vector<std::pair<std::string, ColumnType>> metadata_schema() const;
Metric metric() const noexcept;
virtual size_t remove(span<const int64_t> ids) = 0;
virtual size_t compact() = 0;                   // frees deleted slots, ids unchanged
virtual void save(std::ostream& out) const = 0;
virtual size_t slots() const noexcept = 0;      // current slots, including deleted
virtual bool custom_ids() const noexcept = 0;
virtual bool ids_are_slots() const noexcept = 0;
virtual size_t size() const noexcept = 0;       // live vectors
virtual size_t dim() const noexcept = 0;
```

`Neighbor { size_t index; float distance; }`. Results are sorted nearest first.

**Metrics.** `Metric { L2, Cosine, InnerProduct }` is fixed at construction (`metric_`, set by each constructor, a `RefineIndex` copying its base's). For unit vectors `‖a − b‖² = 2 (1 − cos(a, b))`, so cosine needs no new search code: the public `add`, `add_batch`, `query` and `query_batch` are non-virtual and, for `Cosine`, copy the input through `normalize_rows` (each row scaled to unit length; a zero row throws `invalid_argument`), call the index's private virtual `add_batch_impl` / `query_impl` / `query_batch_impl`, and halve the returned squared L2 into `1 − cos`. `IVFIndex::train` normalizes its training set the same way, so clusters are learned on unit vectors. Normalizing happens exactly once per vector: a `RefineIndex` (a friend of `DenseIndex`) passes its already prepared vectors to the base's `*_impl` methods rather than its public ones, and the default `query_batch_impl` calls `query_impl` per row. (Normalizing a unit vector again can change its last bits, which would make `query` and `query_batch` disagree.)

`Metric::InnerProduct` can't be reduced to L2 at the edge, so the `*_impl` methods score by `-(a . b)` themselves, smallest first like the other metrics: `DenseIndex::exact_score` (squared L2, or `-inner_product`) for Flat, RefineIndex's re-ranking and IVF's float lists and coarse probing; `SQ::inner_product` (the same decode loop as `SQ::distance`) for SQ codes. HNSW searches by `-(q . x)` but links stored vectors as if each had an extra coordinate `sqrt(M^2 - |x|^2)` (Bachrach et al. [7]) (`HNSWIndex::Probe`; only `|x|^2` per slot is kept and `M` is recomputed on add, compact and load). IVF keeps L2 k-means and assignment and probes by `-(q . c)`; for PQ the table of `-(q_s . y_sj)` doesn't depend on the list, so `scan` builds (and for fast-scan quantizes) it once per query and adds each list's `-(q . c)`, with no precomputed tables and no SDC. The measurements behind these choices are in the comments on `HNSWIndex::Probe` and `IVFIndex::Probe`, and on #27.

**Slots and ids.** Every added vector gets a *slot*: its position in storage (0, 1, 2, ... in insertion order). Indexes store and scan slots internally (IVF lists hold slots; `RefineIndex` finds its raw vector by slot). Slots only change in `compact()`, which renumbers them in order. The id users see is separate: with `custom_ids` it is the id passed to `add_batch`; otherwise it is the slot until the first compaction, and the original slot number after it. `Neighbor::index` always reports the id. `add_batch` must keep slots in input order: row `i` of a batch gets the next slot.

**Filtered search.** The `allowed` overloads turn the ids into a `detail::SlotFilter` once per call (`id_map().filter(allowed)`: one bit per slot plus the count of set bits; unknown, deleted and repeated ids are dropped), and every query of a batch shares it. A count of 0 returns empty rows without calling the index. Otherwise the filter goes to `query_impl` / `query_batch_impl` as a pointer, `nullptr` meaning unfiltered, and every index tests `filter ? filter->test(slot) : !is_deleted(slot)` where it used to test only `is_deleted`: deleted slots are never in the filter, so one test covers both. Filtering happens during the search, like deletion, so a selective filter still fills the top k. Per index: `FlatIndex` walks the set bits (`for_each`) instead of every slot; `IVFIndex` keeps probing past `nprobe` (see below); `HNSWIndex` treats disallowed nodes like deleted ones, or scores the allowed slots directly when there are few; `RefineIndex` builds the filter from its own id map and passes it to the base unchanged (its slots are the base's), asking for `min(k * k_factor, count)` candidates. Each index exposes its `IdMap` through a private `id_map()` override.

**Metadata** (`metadata.hpp`, `src/metadata.cpp`, issue #38). `IndexBase` owns a `detail::MetadataStore metadata_`: named columns aligned to slots (row `s` belongs to slot `s`), each `Int64`, `Float64` or `String` (dictionary-encoded: `dict` holds each distinct string once, rows hold `uint32` codes) with a null flag per row. The non-virtual `add_batch` runs `metadata_.check()` (lengths, names, a column keeps its type), then `add_batch_impl`, then `metadata_.append()`, so a throw from either side leaves both unchanged. A column first seen in an add is backfilled with nulls for the earlier rows, and a column an add leaves out gets nulls. A `RefineIndex` calls its base's `add_batch_impl` directly, so the base's store stays empty and the metadata lives in the `RefineIndex`, next to its user ids. `remove` needs nothing (filters drop deleted slots); every `compact_impl` and `RefineIndex::compact` apply the same remap to the columns. A `MetadataColumn` may carry `valid` flags (empty means no nulls; Python's `None` sets them). Int values fit a `Float64` column (converted in `write_row`); the reverse throws. `IndexBase::update_metadata` (issue #73) resolves ids with `IdMap::slots_of` (throws on unknown, deleted or repeated ids), runs `check_update` (a column the store lacks is an error; `add_metadata_column` creates a column of a given type, all null, all one default, or filled for given ids through the same `slots_of` + `update` path, validated by `check_new` before the column exists; `drop_metadata_column` erases one), then `MetadataStore::update` overwrites those rows in place, so it is O(rows changed) with no file-format change. `IndexBase::get_metadata` (issue #77) resolves ids the same way and `MetadataStore::get` copies those rows out (strings looked up in `dict`, nulls as 0 / empty with `valid` cleared); the bindings turn number columns into `numpy.ma` masked arrays and string columns into lists with `None`. Replaced strings stay in the dictionary until compaction: `compact` rebuilds each dictionary from the remaining rows, and a compaction with nothing deleted still calls `drop_unused_strings`.

A filter is an immutable `Expr` tree (`compare`, `isin`, `is_null`, `&`, `|`, `~`; children are shared). `&` and `|` go through `Expr::join`, which merges children of the same op, so `a | b | c | ...` is one `Or` node with n children however it was built. Each `|` copies the left chain's children, so a chain built one term at a time is O(n²); `qann.any_of` / `qann.all_of` call `join` once over the whole list. Making `e = e | t` itself O(1) was considered and left out (#74): `|` would have to extend the left node in place, which is only safe when nothing else holds it, and from Python that means a reference-count check (NumPy's trick) tuned to nanobind's internals; if it misjudged, a filter held in another variable would change silently. Measured on an M3 (2026-10-29), the loop takes 2 ms at 1,000 terms and 1 s at 20,000, against 0.4 ms for `any_of`; long chains are rare, usually one column against many values (`isin`, one scan instead of n), and their per-query evaluation costs far more than building them. A C++-only rvalue overload of `|` / `&` would be safe if the C++ API ever needs it. Each node records its depth, and building one deeper than `Expr::kMaxDepth` (256) throws: `evaluate`, `to_string` and the destructor recurse per level, so this keeps them off the stack limit. The `where` overloads of `query` / `query_batch` call `where_filter()`: `MetadataStore::evaluate` turns the tree into one bit per slot (leaves scan their column into 64-bit words built in a register, with the operator chosen outside the loop and no branches per row; string leaves decide each dictionary entry once and look rows up by code; `&` / `|` / `~` combine words), ANDs in the `allowed` ids when given, and `IdMap::live()` clears deleted slots and counts. From there it is the `SlotFilter` of filtered search, so every index's filtered path, including HNSW's switch to exact scoring and IVF's extra probing, applies unchanged. Nulls fail every comparison and `In`; `Not` is a plain complement, so it includes them. Ints compare exactly against ints and as doubles against floats; a string column takes only strings. Measured on 1M rows (one query, `allowed` of one id so the search is free): 0.4 ms for an int, float or string comparison, 1.2 ms for three combined.

**String rows per code** (#72 step 3). A pass over 1M rows costs ~0.4 ms however few rows match, which was 50-90% of a single filtered query on SIFT1M (`benchmarks/where_cost.py`). So each string column also keeps `rows_of`: for every dictionary code, its non-null rows, as an unordered `uint32` list while the code is rare and as a bitmap once it holds at least 1/32 of the rows and 64 rows (back to a list below 1/64; the 64-row floor stops a tiny first add from leaving bitmaps behind). Lists total at most 4 bytes a row; each bitmap is one bit per row. A string leaf adds up the matching codes' cost, `kScatterCost` (4) scanned rows per listed row and one per bitmap word, and uses them when that is less than the slot count, else falls back to the pass, so `!=` against one of many rare values still scans. `append` indexes its new rows, `update` removes the overwritten rows under their old codes before writing (batched per code, one pass over each list) and adds them under the new ones, and `drop_unused_strings` (so also `compact`), `add_column` and `load` rebuild it. `rebalance` then switches only the codes those rows touched, plus every code once the column has grown by a quarter since the last full pass, so a one-row add into a column of a million distinct strings doesn't walk the dictionary. `remove` needs nothing: deleted slots stay listed until compaction, and `IdMap::live()` clears them from the result as before. It is never saved, so the file format is unchanged.

The thresholds come from `benchmarks/filter_sweep.cpp` (M3, 1M rows, one string at a fraction p of random rows, 2026-10-30): the pass costs ~410 us at any p, a bitmap ~4 us at any p (mostly zeroing the result), and a list 1 us at 0.01%, 7 us at 1%, 27 us at 3% and 159 us at 10%. Per listed row that is 1.7-4.5x a scanned row (rising with p: rows added in order set bits in the same word back to back, a read-modify-write chain), hence `kScatterCost = 4`. A bitmap is faster than a list from about 0.5%, but costs one bit per row against 4 bytes per listed row, which break even at 1/32; switching at 1/32 instead of 0.5% gives up at most ~23 us per filter (a list between 0.5% and 3%) so that a code only takes a bitmap when it is no bigger than the list it replaces (at 0.5%, a bitmap is 6x its list).

**Sorted numeric rows** (#72 step 3). Each `Int64` / `Float64` column keeps `sorted`: its non-null rows (NaN left out) ordered by value, with the values they had then (`sorted_ints` / `sorted_floats`, 12 bytes a row in all). A comparison or `In` leaf binary-searches those values for each bound (`<`, `<=`, `>`, `>=` take a prefix or suffix, `=` and each `In` value a run) and sets the run's rows, when that is cheaper than the pass (`kSortedScatterCost` = 2 per row, against `kScatterCost` = 4 for string lists, since these rows come in random order and don't chain read-modify-writes on one word); `!=` always scans. Inside an `And`, `evaluate` hands all comparisons on one numeric column to `sorted_leaves` together and intersects their runs, so `year BETWEEN 2010 AND 2012` costs its matching rows even though `year >= 2010` alone matches most of the column; if the overlap is still too wide, each comparison is evaluated alone as before. The search uses the saved values, not the column, because an overwritten row may now hold anything: the binary search must see a sorted array. Rows added since the sort (`[sorted_rows, rows())`) and rows overwritten since (`stale`) are tested one by one: stale rows are first cleared from the result, then tested on their current value, so a row that moved into or out of range is right either way. `append` and `update` only extend that tail and `stale`, and `maybe_sort` re-sorts once they reach 1/8 of the rows, so a small add costs its own rows and re-sorting is amortized (measured 80-100 ms for 1M random floats on an M3, at most once per 12.5% growth). `compact`, `add_column` and `load` sort from scratch. The comparisons are the pass's own (`compare`, exact for an int column against an int, in double otherwise), and NaN never matches, so both paths agree; the numeric fuzz test checks that with ties, NaN, -0.0, ints past 2^53, the tail and stale rows. Measured with `benchmarks/filter_sweep.cpp` (M3, 1M floats, 2026-10-30): `price < t` costs 1.3 us at 0.01% of rows, 7 us at 1%, 59 us at 10%, 174 us at 30%, 292 us at 50%, and the pass 383 us above that.

**And narrowing** (#72 step 3). An `And` evaluates its children in order of how few rows they are estimated to match, and once few rows are left it tests only those. `evaluate` splits the children into parts: each child, except that the comparisons on one numeric column form one part (the BETWEEN case above). `estimate()` gives each leaf part its matches and cost without evaluating it: for a string leaf the matching codes' row counts (exact) and list or bitmap cost; for a numeric part the binary-searched runs plus the unsorted tail and stale rows (an upper bound); for `IsNull` and `!=`, the whole column. `estimate_tree` extends this to nested parts: an `And`'s fewest matches, an `Or`'s summed matches (capped at the column), a `Not`'s whole column, and summed costs. Parts run in ascending order of matches (a stable sort, so ties keep the filter's order). After the first, each leaf part either runs in full and is ANDed in as before, or, when the rows still set times `kTestCost` (8) is less than its estimated cost, `narrow()` tests only those rows (iterating set bits, comparisons chosen outside the loop) and clears the ones that fail. A nested part gets the rows still set passed down (`evaluate_tree`'s `within`): an `Or` runs each branch only on the rows no earlier branch matched, a `Not` complements within them, and each leaf makes the same narrow-or-evaluate choice. Results don't depend on the order: `And` is commutative, every leaf still validates its column and values (an unknown column or a wrong-type value throws even when no rows are left), and `narrow` uses the pass's comparisons. A tested row costs ~4-6 scanned rows, measured with `filter_sweep.cpp` on a loaded machine (its values are read out of order), so `kTestCost` = 8 narrows only when it clearly wins. `string_match` also looks `=` and `IN` values up in `code_of` instead of comparing every dictionary entry.

**Factoring.** Before evaluating, `evaluate` calls `factor()`, which pulls a condition shared by two or more `Or` branches out in front: `(a & x) | (a & y) | z` becomes `(a & (x | y)) | z`, so `a` is evaluated once and `x | y` only on its rows. It repeats with the condition found in the most branches (ties broken by `to_string`, so the result doesn't depend on hash order) until no two branches share one, and recurses into the pulled-out `Or`. Conditions are matched with `same()`, node by node, with `to_string` only as the hash key, so two conditions that print alike but differ can't be merged. A branch that is only the shared condition absorbs its group (`a | (a & x)` is `a`). The rows matched never change: the filter logic is two-valued (nulls fail every comparison, `~` is a plain complement), where `&` and `|` distribute exactly. Factoring can nest one level deeper per group, so if the result would pass `Expr::kMaxDepth` the filter is evaluated as given. Measured with `MetadataStore::evaluate` alone on 1M rows of `where_cost.py`'s metadata, this branch against main alternating three rounds on a loaded M3 (2026-10-30): `(category = 'c3' OR brand = 'b17') AND NOT (price >= 90)` 970 -> 245 us (the `Or` is now estimated at ~101k rows and runs first, then `price` is tested on those), `(category = 'c3' AND price < 50) OR (category = 'c3' AND year >= 2015)` 735 -> 425 us. A filter whose shared condition already leaves ~1k rows costs ~10 us more, the price of building the factored tree.

**Row lists for few rows.** Every step above works on a bitmap of all slots, so it costs a pass over `slots / 64` words however few rows are left: counting them, copying the mask for an `Or` branch, `narrow` scanning every word for its set bits. Once a filter's candidates drop below 1/32 of the slots (`few_rows`: a `uint32` row list is then smaller than the bitmap), `evaluate_tree` switches to `evaluate_rows`, which works on a sorted row list: an `And` filters it child by child (fewest estimated matches first), an `Or` gives each branch only the rows no earlier one matched and merges what they keep, a `Not` subtracts, and a leaf tests each row (`narrow` on a list) or, when its own evaluation is cheaper than testing the rows (`brand = 'b17'` through its 1k-row list), evaluates over the column and keeps the listed rows whose bit is set. The switch happens at the top of a nested part handed few rows and inside an `And` once its rows left get few; the bitmap is rebuilt once at the end. `evaluate_tree` also takes the count of `within`'s rows from its caller (an `And` counts as it ANDs, an `Or` as it clears matched rows) instead of counting it again. Measured with `MetadataStore::evaluate` on 1M rows, branch against main alternating three rounds on a quiet M3 (2026-10-30): `(brand = 'b17' AND price < 50) OR (brand = 'b17' AND year >= 2015) OR (brand = 'b17' AND category = 'c1')` 52.5 -> 26.7 us (before factoring it was ~43 us); filters whose candidates stay above 1/32 of the rows are unchanged (one ran ~10% slower after other filters in the same process, but the same alone over 300 rounds: 164 us both).

A full normal form (CNF, an `And` of `Or`s) was left out: it can grow the filter exponentially (`(a & b) | (c & d) | ...` doubles per term), and evaluation already costs about the rows each part touches, so only the targeted rewrite above pays. Measured on SIFT1M (2026-10-30, 1 thread, loaded machine): `category = 'c3'` (10%) 419 -> 12 us, `brand = 'b17'` (0.1%) 352 -> under 1 us, `category IN` 3 values 418 -> 22 us, `brand IN` 20 values 425 -> 13 us; int and float leaves are unchanged.

`Expr::parse` (issue #46) builds the same tree from a SQL `WHERE` clause, so evaluation and search are unchanged. It is a recursive-descent parser (`SqlParser` in `src/metadata.cpp`) over the string itself, with no separate token list: `or_expr` > `and_expr` > `not_expr` > `predicate`, so precedence is SQL's. SQL's `NOT` is three-valued (`NOT (price < 50)` is unknown, so excluded, where price is null), unlike `~`, so the parser never emits `Not` for it: it carries a `neg` flag down the recursion instead. A negated comparison flips its operator (`<` to `>=`), `AND` and `OR` swap (De Morgan), negated `IN` becomes `~In & ~IsNull`, and negated `BETWEEN` becomes `< lo | > hi`. `IS [NOT] NULL` is two-valued, so it may use `~`. Placeholders (`?` from a list, `:name` from a map) are replaced by values as they are read, so a value can't change the tree. Errors throw `std::invalid_argument` with the 1-based column. `AND` / `OR` chains are collected in a loop and joined once; only `NOT` and parentheses recurse, and the parser counts them against the same 256-level cap, so deeply nested input fails with a column instead of overflowing the stack. It also counts conditions (each `predicate`, so an `IN` list or a negated `BETWEEN` is one) against `Expr::max_conditions()` (default 1024, a process-wide atomic read once per parse), because each condition is one scan of its column per query; filters built through `compare` / `join` aren't limited, since they come from code rather than user strings. The Python binding takes an `Expr`, a string, or `(string, list | dict)` for `where=` and parses before taking the index lock.

### `id_map.hpp` / `id_map.cpp`: `IdMap` (`vecengine::detail`)

Each index owns one. It holds `labels_` (slot → id; empty while ids equal slots), `slot_of_` (live custom id → slot, for `remove` and duplicate checks), `next_id_` (the next auto id) and a deleted flag per slot.

- `check(n, ids)` validates an add without changing anything: ids present iff custom ids are on, each `>= 0` (`-1` is the padding id in batch results), no duplicates in the batch or among live ids, and at most `2^32 - 1` slots. `add` runs `check` and then commits.
- `remove(ids, removed_slots)` marks slots deleted and erases their ids from `slot_of_`, so a deleted id can be added again (it gets a new slot). Unknown and already deleted ids are skipped.
- `filter(ids)` builds the `SlotFilter` of a filtered query: the slots of the live ids among `ids`, found the same way `remove` finds them.
- `label(slot)` and `is_deleted(slot)` are inline: queries call them per candidate. `label` returns `labels_[slot]`, or the slot itself while `labels_` is empty.
- Auto ids: while `labels_` is empty and `next_id_ == slots_`, ids are slots and `add` is a counter bump. After a compaction `labels_` holds each slot's id, and `add` appends `next_id_++`. Auto ids are assigned in increasing order and compaction keeps order, so `labels_` stays sorted and `remove` finds a slot with `lower_bound` instead of a hash map (8 bytes per vector, only once compacted).
- `compact(keep_ids)` returns `remap` (old slot → new slot, or `kRemoved`), or `{}` when nothing is deleted. It drops deleted slots, moves each live label forward (`remap[old] <= old`, so in place), rewrites `slot_of_` and clears the deleted flags. `keep_ids = false` is for a `RefineIndex` base: `labels_` stays empty and `next_id_` is reset to the new slot count, so its ids keep equalling its slots.
- `ids_are_slots()`: auto ids, `labels_` empty and `next_id_ == slots_`. `RefineIndex` requires it of its base; it rejects custom ids and also a base that was compacted (or emptied by compaction) before being wrapped.

**Deletion is tombstoning; `compact()` reclaims.** `remove` moves nothing. Queries skip deleted slots *during* the scan (filtering a finished top-k would return fewer than k), so deleted vectors still cost a distance computation and keep their memory until `compact()`.

**Compaction.** `compact()` runs `id_map_.compact(true)` and then moves storage with the remap, keeping live entries in order, in place with a write pointer (no second copy; the remap is 4 bytes per old slot), then `shrink_to_fit`. Flat moves row `old` to row `remap[old]`. IVF filters each list's `ids` with its `codes` or `vecs`, rewrites the ids to their new slots and erases lists left empty; centroids, the PQ codebook and the precomputed tables don't depend on slots, so nothing is retrained. `RefineIndex::compact` compacts its id map, calls the base's private `compact_as_base()` (the same storage compaction with `keep_ids = false`), then moves its own raw rows. Both id maps have the same deleted set (`RefineIndex::remove` deletes the same slots in the base), so both compute the same remap. Results are bit-identical before and after, and a second call returns 0.

Defaults in `src/index.cpp`: `add_batch` calls `add` per row; `query_batch` calls `query` for each row in parallel via `parallel_for`. This is safe because `query` is `const` and touches no shared mutable state, and each result goes to its own slot. Subclasses override `add_batch` only when a whole batch allows a faster path (`IVFIndex`, `RefineIndex`).

### `src/parallel.hpp`: `parallel_for` (`vecengine::detail`)

`parallel_for(n, fn)` calls `fn(i)` once for each `i` in `[0, n)` on up to `num_threads()` `std::thread`s, the caller being one of them. With one thread (or `n <= 1`) it is a plain loop on the caller: no atomics, no threads, exceptions propagate directly. Workers take the next index from a shared atomic counter, so uneven per-item cost (IVF list sizes, M-series performance vs efficiency cores) doesn't leave threads idle. The first exception thrown by `fn` stops the other workers and is rethrown on the caller after all join. Threads are created per call; at batch and training granularity that cost is negligible, so there is no pool. Don't nest it: `fn` must not call `parallel_for` itself.

**Thread count** (`include/vecengine/threads.hpp`, `src/threads.cpp`): `set_num_threads(n)` sets one process-wide count in a relaxed `std::atomic`, safe to change while other threads query. `0` (the default) means one thread per core; `num_threads()` resolves `0` to `hardware_concurrency()` (at least 1) on every read without writing it back. It does not cap the BLAS library's own threads inside `sgemm`; use `VECLIB_MAXIMUM_THREADS` / `OPENBLAS_NUM_THREADS` for those.

### `flat_index.hpp`: `FlatIndex`

Exact search. Vectors stored row-major in one `vector<float>`, one row per slot. `query` scans every live slot (or, filtered, only the allowed ones) with a size-`k` max-heap, drains it back to front for ascending order and reports each slot's id. This is the ground truth for recall tests.

### `ivf_index.hpp`: `IVFIndex`

`IVFIndex(dim, nlist, nprobe = 10, custom_ids = false)`, with `nlist` in `[100, 65535]` (cluster IDs are `uint16_t`) and `nprobe` in `[1, nlist]`.

**Lifecycle:** `enable_pq(...)` or `enable_sq(bits)` (optional, not both) → `train(sample)` → `add(v)` for each vector → `query`. `train` learns the model only; it stores no vectors. Training vectors are not searchable unless they are also added.

- **`train`**: one `kmeans` call for the `nlist` coarse centroids. With PQ on, computes residuals `v − centroid(assignment)` from `kmeans`'s assignments and trains the codebook on them. With SQ on, `detail::SQ::train` learns each dimension's range from the sample itself (unit vectors for Cosine).
- **SQ storage** (`enable_sq(bits)`, 8 or 4): each vector's SQ codes go in its list's `codes` (`sq_code_bytes()` per vector) instead of `vecs`, and `scan` scores them with `SQ::distance`, the kernel HNSW uses. The codes encode the vector, not its residual: a study (`benchmarks/ivf_sq_bench.cpp` on branch `ivf-sq-study`, issue #25) found residual codes no more accurate on SIFT1M (0.924 vs 0.925 recall at `nprobe = 16`), and direct codes need no per-list query offset. The float scan is bandwidth-bound, so SQ8's 4x smaller lists pay off on all cores (1.35-1.7× in that study) more than on one thread (1.0-1.1×, where decoding costs as much as the bytes it saves).
- **`add`**: `nearest_centroid` picks the cluster; the vector's slot is appended to that cluster's `InvertedList`, plus either the raw floats (`vecs`) or the PQ code of its residual (`codes`).
- **`add_batch`**: three phases. (1) One `assign_nearest` call assigns every row. (2) With PQ on, `parallel_for` encodes each residual into its own slot of an `n * m` buffer. (3) A serial loop appends slots and data to the lists in input order, because `unordered_map` and list growth are not thread-safe and slots must follow input order. The ids are checked by `IdMap` before any of this, so a bad batch changes nothing. `assign_nearest` ranks by `‖c‖² − 2x·c`, so a near-tie can pick a different list than `add` would.
- **`query`**:
  1. Coarse: find the `nprobe` nearest coarse centroids.
  2. Fine: scan only those lists with a size-`k` max-heap. One ADC table buffer is allocated per query and reused for every list. Per list, all query-and-cluster-dependent work happens once, before the per-vector loop:
     - PQ off: `l2_distance` per stored vector.
     - ADC with precomputed tables (the default): table = `B[list] + C`, plus `A` on subspace 0's row → `distances_adc` over the list's codes. See below.
     - ADC without them: residual `q − centroid(c)` → `compute_adc_table` → `distances_adc`.
     - Fast-scan (`enable_pq(m, 16, true)`): either float table above → `quantize_lut` → `fast_scan_block` per 32-vector block. The top-k threshold is converted to the table's integer scale once per block (`(threshold − offset) · scale`), so most vectors are rejected by one integer compare; survivors, minus padding, are pushed with `sum / scale + offset`. Distances are approximate; a `RefineIndex` makes them exact. SDC is refused (`set_pq_distance` and `enable_pq` throw `invalid_argument`).
     - SDC: residual → `encode` → `distance_sdc` per code.

The residual depends on the cluster's centroid, so without precomputed tables it is rebuilt per probed list. Stored codes encode `x − c`, the query becomes `q − c`, and `(q − c) − (x − c) = q − x`, so residual distances equal true distances. SDC still rebuilds and encodes the residual per list, so it is slower than precomputed ADC as well as less accurate. ADC is the default.

**Precomputed tables** (FAISS-style [4]). With `y` the PQ reconstruction of a stored residual, the ADC distance expands per subspace as `‖q − c − y‖² = ‖q − c‖² + (‖y‖² + 2⟨c, y⟩) − 2⟨q, y⟩ = A + B + C`:

| Term | Depends on | Built | Stored |
|---|---|---|---|
| `A = ‖q − c‖²` | query, list | coarse search already computes it | the coarse heap (read before `pop`) |
| `B[list][s][j] = ‖y_sj‖² + 2⟨c_s, y_sj⟩` | list | `build_precomputed()`, once per list, in parallel | `precomputed_`, `nlist * m * K` floats, list-major |
| `C[s][j] = −2⟨q_s, y_sj⟩` | query | `compute_inner_table`, once per query | a local `m * K` vector |

Per probed list the table is then `m * K` additions instead of a `sub_dim`-long loop per entry, and `A` is added to the `K` entries of subspace 0 only: each code picks exactly one entry per row, so every distance gets `A` once and `distances_adc` stays unchanged. Results match the per-list path up to float rounding (the tests allow `1e-3`).

`build_precomputed()` runs at the end of `train()`, after `load_body()` (the tables are derived, never saved), and from `set_precomputed_tables(true)`. It leaves `precomputed_` empty, and queries fall back to the per-list build, when PQ is off, the index is untrained, the setting is off, or the tables would exceed `kPrecomputedMaxBytes` (256 MB). SIFT1M with `nlist = 1000` and PQ16 needs 16 MB. `precomputed_tables()` reports whether the tables exist; `set_precomputed_tables(false)` frees them.

`query()` is `coarse()` (a size-`nprobe` heap over exact `l2_distance`s) followed by `scan()` (the fine step above). Each `Probe` carries its list id and its exact coarse distance, which is term `A` of the precomputed tables.

**`query_batch`** (with BLAS; without it, the `DenseIndex` default) replaces the per-query coarse loop. Queries go in blocks of up to 1024, capped so the score buffer (`block * nlist` floats) stays within 32 MB. Per block, `coarse_batch()`:
1. computes `‖c‖²` for every centroid,
2. makes one `cblas_sgemm` call for all `q·c` dot products in the block, on the calling thread so Accelerate/OpenBLAS can use their own threads without oversubscribing `parallel_for`,
3. per query, in `parallel_for`, ranks the lists by `‖c‖² − 2q·c` (`‖q‖²` is the same for every list) and keeps the `nprobe` lowest with `nth_element`,
4. recomputes the exact `‖q − c‖²` for those `nprobe` lists only. The sgemm score suffers cancellation, and `A` feeds straight into the returned distances, so it must match `query()`.

Then `scan()` runs per query in `parallel_for`. Like `assign_nearest`, a near-tie between two centroids' scores can pick a different list than `query()` would, so batch and single results can differ in rare ids; the tests allow 1%.

Every candidate goes through one `push` lambda that checks the `TopK` threshold first and then `is_deleted` (or the filter), and pushes the slot's id.

**Filtered queries** may find fewer than `k` allowed vectors in the `nprobe` nearest lists (at 0.1% of SIFT1M, a list of ~1000 vectors holds one on average). So `query_impl` takes `coarse_all()`: every list, with only the `nprobe` nearest sorted (`partial_sort`), and `scan()` continues past the first `nprobe` while `TopK` holds fewer than `min(k, count)` results, sorting the remaining lists the first time it needs them. A filter that the probed lists satisfy costs only the partial sort over `nlist` distances. A filtered `query_batch` takes the `DenseIndex` default (per-query `query_impl` in parallel) rather than `coarse_batch`, which ranks only `nprobe` lists per query. `nprobe` and `pq_distance` both have setters and can be changed at any time after training; they only affect queries.

**Rotation** (`enable_pq(..., rotate = true)`, `src/rotation.hpp`, issue #41). PQ quantizes each subspace independently, so it wastes bits when dimensions are correlated across subspaces and when some subspaces hold far more variance than others. `train()` learns an orthogonal `R` (`dim × dim`, row-major; a row vector rotates as `x · R`) and then trains the coarse centroids and the codebook on the rotated sample; `add_batch_impl`, `query_impl` and the BLAS `query_batch_impl` rotate their input first (the batch path only after its early returns, which go through `query_impl`, so nothing is rotated twice). Distances are unchanged by a rotation, so everything downstream (coarse search, precomputed tables, fast-scan, results) works in rotated space as is, and a `RefineIndex` keeps re-ranking against its own unrotated vectors.
- `balanced_pca_rotation`: the covariance of the sample (double, `cblas_dsyrk` with BLAS), its eigenvectors from `symmetric_eigen` (cyclic Jacobi, so no LAPACK; O(d³) per sweep, milliseconds at 128 dimensions), then eigenvalue allocation (Ge et al. [5]): largest eigenvalue first, each into the subspace with room whose product of eigenvalues is smallest, so subspaces hold equally hard mixes of high- and low-variance directions.
- `rotate` uses plain loops (auto-vectorized, parallel in blocks of 64 rows) rather than BLAS, so a vector rotates to the same floats on every platform and saved codes stay consistent with queries.
- The study (branch `opq-study`, results on #41) compared this with non-parametric OPQ (Procrustes alternation), which gained nothing: from identity it stays near identity, and from the PCA start it is worse on SIFT, whose natural order already groups correlated dimensions. Measured with the implementation on SIFT1M, PQ64×4 fast-scan, alternating rounds: recall@10 at `nprobe = 16` 0.635 → 0.718 unranked; with re-ranking, `k_factor = 4` matches what 8 did and 8 what 16 did, about 1.05× QPS at equal recall after the per-query rotation (~5% at identical settings). GloVe-100 gains nothing (its dimensions are nearly uncorrelated already). On synthetic data, duplicated dimensions in different subspaces gain 0.17-0.2 recall, while a random rotation of a steep Gaussian spectrum loses 0.01-0.05: it is opt-in for that reason.

### `hnsw_index.hpp`: `HNSWIndex`

`HNSWIndex(dim, M = 16, ef_construction = 200, custom_ids = false, seed)` is a Hierarchical Navigable Small World graph (Malkov & Yashunin [6]). Every vector is a node on level 0; a node's top level is drawn as `floor(-ln(U) / ln(M))`, so each level up holds about 1/M of the nodes. Vectors are stored in full, in slot order (`data_`), or as SQ codes (`codes_`, see below), with ids from an `IdMap` like `FlatIndex`.

**Storage.** Level 0 is one flat `vector<uint32_t>` with `2M + 1` entries per slot (a count, then up to `2M` neighbor slots), so the hot level is contiguous. Upper levels, which hold few nodes, are `upper_[slot][level - 1]`, a list per level. `levels_` holds each slot's top level; `entry_` is a node on the top level `max_level_` (`-1` while empty).

**Search** (`search_layer(q, entry_points, ef, level)`): a beam search with a min-heap of nodes to expand and a max-heap of the best `ef` so far; it stops when the nearest unexpanded node is farther than the worst result and the results are full. Visited nodes are tracked with a per-thread array of generation tags, so nothing is cleared between searches. `query()` descends greedily through the upper levels (`ef = 1`), then searches level 0 with `max(ef_search, k)`. With `skip_deleted`, deleted nodes are still expanded (they keep the graph connected) but never enter the results heap, so a query's beam fills with live nodes even when most nodes are deleted. A filtered query passes its filter, and nodes outside it are handled the same way.

**Filtered queries.** With a fraction `s` allowed, the beam stops only once it holds `ef` allowed nodes, so it expands about `1/s` times as many nodes as an unfiltered search: about `D / s` distances, where `D` (~1,200 on SIFT1M at `ef_search = 64`) is the unfiltered cost. Scoring the allowed slots directly costs `s * slots` and is exact. The two cross at `count² = D * slots`, so `query_impl` scores directly when `count² <= ef * M * slots` (`ef * M` stands in for `D`: 1,024 at the defaults, i.e. below 32k allowed vectors of 1M). Measured on SIFT1M with 1,000 queries, the crossover between graph and direct scoring was between 3% and 5% allowed, for both the float and the SQ8 graph. The graph search keeps 0.99+ recall all the way down, but at 0.1% it ran at ~100 QPS against ~120k for direct scoring. Results by selectivity are under Benchmarks.

**Beam scaling for filters** (#89, `filter_ef_scaling`, on by default, not saved). Because the beam only stops once it holds `ef` allowed nodes, a filtered search ends up nearer the true neighbors than an unfiltered one at the same `ef`, and pays for it: on SIFT1M at `ef_search = 64`, recall 0.998 with 10% allowed against 0.953 unfiltered, at 405 us against 61 us per query. `query_impl` therefore shrinks the beam to `max(k, ef * sqrt(s))`, `s = count / slots`. Measured on SIFT1M (500 queries, 1 thread, on and off alternating, 2026-10-30), filtered recall stays at or above unfiltered at every `ef` tested (ef 64: 0.960 / 0.972 / 0.971 / 0.968 at 50% / 20% / 10% / 5%) and time drops 1.3-3.7x (103 -> 77, 217 -> 109, 405 -> 143, 730 -> 191 us); on GloVe-100 recall stays within 0.01 of unfiltered (ef 64: 0.762-0.776 against 0.764) at 1.3-4.4x less time. `s^0.7` was faster but fell below unfiltered recall (0.935 at 10%). The direct-scoring cutoff uses the scaled `ef`, but a scaled search costs about twice the `D / s` model at small `s` (the beam can't shrink below `k`), so with scaling it doubles `D`: `count² <= 2 * ef * M * slots`. That puts the crossover at 1.8% (ef 32 and 64) and 2.5% (ef 128), against measured ~1.8%, ~2% and ~2.5%. With scaling off, both the beam and the cutoff are as before.

Two alternatives were measured and dropped. Raising the direct-scoring cutoff loses: direct scoring costs ~20 us per 1,000 allowed vectors, so 10% would cost ~2 ms. ACORN-style two-hop expansion, where a rejected neighbor isn't scored but its own allowed neighbors are, does score only allowed nodes, but at equal recall it was slower than the plain search with a smaller `ef` (10% allowed: 200-228 us at recall 0.96-0.97, against 122 us at 0.957 for the plain search at ef 16): the rejected nodes' distances weren't the cost, the oversized beam was.

**Insert.** Descend to the new node's level, then on each level from there down: search with `ef_construction`, pick up to `M` neighbors with the paper's heuristic (`select_neighbors`: sorted nearest first, a candidate is kept only if it is closer to the new node than to every neighbor already kept, which favors neighbors in different directions), and add the reverse links (`add_link`), re-selecting a neighbor's list with the same heuristic when it is full. Candidates sort as `(distance, slot)` pairs, so ties break the same way everywhere.

**Parallel build** (`build(first, n)`, used by `add_batch` and `compact`): levels are drawn serially in slot order first, so a seed fixes them, and storage for the batch is sized before any thread starts, so nothing reallocates. The first `kSerialBuild` (1024) slots of a graph are inserted serially, because vectors inserted at the same time cannot see each other and that costs recall while the graph is small; the rest are inserted with `parallel_for`. Neighbor lists are guarded by a pool of `kLockStripes` (1024) mutexes indexed by `slot % kLockStripes`. A thread holds at most one of them at a time, so threads cannot deadlock waiting on each other: `add_link` holds the neighbor's lock throughout, and a search during an insert copies each list under its lock and computes distances unlocked. Queries never run during an add (the Python bindings lock the index), so they skip the locks. `top_lock_` guards `entry_` and `max_level_`; an insert that raises the top level holds it for its whole insert. A node can be reached through an upper level before its insert reaches the lower ones, so other threads may already have linked into a lower level; `set_own_neighbors` merges those links instead of overwriting them. The test suite was run under ThreadSanitizer clean. Parallel builds lose little quality (on 20k random vectors, 0.21% of vectors fail to find themselves vs 0.17% serially), but they are not reproducible run to run; with `set_num_threads(1)` and a seed they are.

**Deletion and compaction.** `remove()` only marks slots deleted. `compact()` drops deleted vectors and rebuilds the graph from the live ones (`build(0, slots)`): repairing a graph in place is complex and degrades it, and compaction is meant for occasional cleanup. Ids are kept; results can shift slightly because the graph is new.

**Memory access.** A search is limited by memory, not arithmetic: on SIFT1M a query at `ef_search = 64` computes about 1,210 distances and expands 76 nodes, and each neighbor's vector is a random 512-byte read from a 512 MB array (~125 ns per distance before prefetching, against ~17 ns for the arithmetic on data in cache). So `search_layer` expands a node in two passes: the first marks its unvisited neighbors and calls `prefetch_vector` on each, so their reads overlap; the second computes the distances. On level 0 it also prefetches the neighbor list of the node it will most likely expand next (the new top of the candidate heap). The upper levels only need the single closest node, so `greedy_closest` walks to whichever neighbor is closer until none is, with the same prefetching and no heaps. Together these took a single-threaded query from 152–157 to 94–95 µs at `ef_search = 64` (recall 0.958) and from 513–596 to 263–287 µs at 256 (0.994), with identical results; builds, which are mostly searches, got faster too.

**Measured** (SIFT1M, `M = 16`, `ef_construction = 200`, all 8 cores, machine busy, IVF in the same run; full tables under Benchmarks): the float graph builds in 75 s (IVF + PQ64×4 fast-scan + R16: 5.3 s). At 0.995 recall it reaches 11.4k QPS against 16.9k for IVF + fast-scan + R16, 0.67× (it was 0.5× in 0.1.0a10, and 0.33× before the prefetching). Most of the gap is the size of the vectors it reads, which SQ storage (below) cuts from 512 to 128 bytes (8 bits) or 64 (4 bits).

**SQ storage** (`enable_sq(bits)` with `bits` 8 or 4, before `train()` and any add, like `IVFIndex::enable_pq`). `train()` learns a `detail::SQ` from a sample (unit vectors for Cosine, like IVF's training), and adds encode into `codes_` (`code_bytes_` per slot; `data_` stays empty). Everything goes through two helpers: `distance(q, slot)` uses `SQ::distance` on the slot's code, and `vector_of(slot, buf)` returns a pointer into `data_`, or decodes the code into the caller's `buf`. Inserts and neighbor selection use `vector_of` for the node being linked, so the graph is built and searched entirely with SQ distances, and `compact()` rebuilds from the codes. `prefetch_vector` fetches `stored_bytes()` (on SIFT 2 cache lines for SQ8, 1 for SQ4, instead of 8). Query distances are approximate; a `RefineIndex` over the SQ graph restores exact ranking: `k_factor = 2` is enough for SQ8 on SIFT1M because it ranks nearly like the floats, and SQ4 needs 4. Measured on SIFT1M (all cores, machine busy; tables under Benchmarks), at 0.995 recall: SQ8 + R2 15.8k QPS and SQ4 + R4 15.6k, against 11.4k for the float graph and 16.9k for IVF + fast-scan + R16. Single-threaded, SQ8 + R2 is 1.6 to 1.7× the float graph at the same recall. Plain SQ8 uses about 40% of the float graph's memory (128 + 132 bytes of level-0 links per vector vs 512 + 132) and loses a little recall (0.978 vs 0.987 at `ef_search = 128`); SQ4 alone levels off near 0.86, so it needs re-ranking. SQ4 is not faster than SQ8 because a search waits on the latency of each random read (the vector and its 132-byte link list), not on the bytes moved.

### `refine_index.hpp`: `RefineIndex`

Re-ranks an approximate index's shortlist with exact distances, as in Jégou et al. [2]. `RefineIndex(base, k_factor = 10, custom_ids = false)` wraps an approximate index with **no slots and ids equal to its slots** (`ids_are_slots()`: no custom ids, never compacted) by reference and keeps its own row-major copy of the raw vectors. User ids live in the `RefineIndex`'s `IdMap`; the base gets plain slots, so base slot `i` and refine slot `i` are always the same vector. (Checking `base.slots()` rather than `size()` matters: a base whose vectors were all deleted has size 0 but its slots are taken.)

A RefineIndex built this way holds `base` by reference (`Index& base_`); one created by `load_body` owns its base through `owned_base_`, declared before `base_` so it is initialized first, and binds `base_` to it via a private constructor. `base()` exposes the wrapped index either way, so a loaded index can still be tuned.

- `add` / `add_batch`: `id_map_.check(...)` first, then `base.add_batch` (no ids), then `id_map_.add(...)` and the raw floats, so slot `i` is at `data_[i * dim]`. Checking before touching the base keeps the two in step when either side rejects the add (bad ids, untrained base).
- `remove(ids)`: removes the ids from its own map, then removes the same slots from the base (whose ids are its slots), so the base stops returning them.
- `query(q, k)`: asks `base` for `min(k * k_factor, size())` candidates, replaces each approximate distance with the exact `l2_distance` against its raw vector, and returns the exact top `k` with ids from its own map. Returns nothing when every vector is deleted.

Changing the base directly would break that alignment (an add gives the base a vector the `RefineIndex` has no raw copy of; a remove hides a vector the `RefineIndex` still counts), so the constructor sets the base's `wrapped()` flag (an `std::atomic<bool>` on `IndexBase`, cleared by `~RefineIndex`) and rejects a base that is already wrapped. The Python `add`/`remove` bindings check the flag under the index's write lock and raise `ValueError`. The C++ API doesn't enforce it, since `RefineIndex` itself calls `add_batch`/`remove` on its base.

`query_batch` makes one `base.query_batch` call for every query's candidates (so IVF's batched coarse search applies) and then re-ranks each query in `parallel_for`. Without it, the `DenseIndex` default would run `query()` per row and call `base.query()` one query at a time.

It only helps when `base` ranks with approximate distances (IVF + PQ). It restores recall but stores the raw vectors in RAM, so IVF + PQ + re-ranking uses more memory than plain IVF; in production the raw vectors would live on disk and only the candidates would be read. `k_factor` is a query-time setting; `k_factor = 1` returns the base's IDs with exact distances.

### `sparse_index.hpp`: `SparseIndex`, and `inverted_index.hpp`: `InvertedIndex`

The sparse family (issue #59), derived from `IndexBase` rather than `DenseIndex`: no `dim`, and `SparseRows` input (a CSR view: `indptr` as u64, `indices` as u32 term ids, `values` as floats) instead of float rows. `SparseIndex`'s non-virtual `add_batch` / `query_batch` check the rows, check and append metadata, build `allowed` / `where` filters, make each query canonical and run queries in `parallel_for`, calling the private virtual `add_batch_impl` and `query_impl` (as `DenseIndex` does). Ids, deletion, metadata, `where=` and `allowed` come from `IndexBase` unchanged. `InvertedIndex` is the exact algorithm:

- **Storage:** `lists_` maps each term to its `List`: the posting list, `(slot, weight)` pairs in increasing slot order, and the largest and smallest weight in it (kept on add, recomputed on compaction and load, not saved). Adds append, since new slots are the largest. Each row is made canonical first (`canonical_row`: sorted by term, repeats summed in their original order, zeros dropped), so a list never holds a slot twice. A row with no nonzero values still gets a slot.
- **Query (term at a time, `search_one`):** each query term's list gets bounds on what it can add to a score: `hi = max(0, q·max, q·min)` and `lo = min(0, q·max, q·min)` (0 for a vector not in the list, so negative weights are allowed). Lists are taken from the largest `hi` down, shorter first on ties, and each adds `q * w` with `std::fma` into a per-thread dense `scores` array, recording each slot on first touch. Both modes add the same lists in the same order, so pruning returns bit-identical distances. The scratch is zeroed as it is read back, so it costs O(slots) memory per thread once. Distances are `-(q . d)`, like `Metric::InnerProduct`; only vectors sharing a nonzero term with the query are candidates. `query_batch` runs queries in `parallel_for`.
- **Pruning (on by default, `pruning`; MaxScore's term-at-a-time form, Turtle and Flood 1995):** `kth` is a lower bound on the k-th best final score among allowed vectors: a min-heap of k distinct vectors, each entered with its partial score plus the remaining lists' `lo` (entries aren't raised as scores grow, so it lags but stays valid), updated only when a score beats its minimum. Once the remaining lists' summed `hi` can't beat `kth`, an unseen vector can't make the top k, so the rest of the lists only add to seen vectors whose partial score plus the remaining `hi` still beats `kth`; one that falls out is never added to again and the final `TopK` drops it. Those lists are read testing a flag per posting, or, when the vectors seen are under 1/16 of a list, the allowed candidates are collected, sorted and looked up with an exponential search (`seek`). Every bound comparison is widened by a margin for float rounding (`2 · Σ|bounds| · (lists + 2) · 1.2e-7`), so pruning never drops a vector full scoring would keep. Pruning only runs when `k · 128 <= ` the postings in the query's lists, and stops once fewer postings are left than vectors seen: past those points keeping `kth` costs more than it saves (measured on BM25 over BEIR SciFact and FiQA, and on a synthetic corpus with no IDF, its worst case). Measured with `benchmarks/sparse_bm25.py`: FiQA 1.58x at k = 10, 1.08x at k = 100, about 1.0x at k = 1000 and on SciFact; 0.8 to 0.9x on the no-IDF corpus. A document-at-a-time MaxScore was tried first and was 2 to 3x slower than plain term-at-a-time scoring here, since that costs about one add per posting.
- **Compaction:** the `IdMap` remap keeps slot order, so each list is rewritten in place without re-sorting; lists left empty are erased.
- **Not yet:** block-level bounds (Block-Max WAND) to skip within lists.

### `sparse_approx_index.hpp`: `SparseApproxIndex`

Approximate search for learned sparse vectors (issue #59), whose ~49-term queries make exact search read about 2.6M postings each at 1M SPLADE passages.

- **Storage:** like `InvertedIndex`, a `List` per term with the extremes of its weights, but in weight order: `postings[0, sorted)` heaviest first (ties by slot), then the entries added since in slot order. `settle()` runs at the end of an add for each list it touched: once more than 64 entries, and more than an eighth of the sorted ones (at most 4,096), are new, they are sorted and `inplace_merge`d in, O(list) per merge. Plus a forward index (each slot's terms, increasing, and weights, CSR by slot), appended on add and rebuilt from the lists on compaction and load, not saved: about twice `InvertedIndex`'s memory.
- **Query (`query_impl`):** takes the `query_terms` largest values (`nth_element`), reads `postings[0, min(list_depth, sorted))` and `postings[sorted, end)` of each of their lists into a per-thread dense scratch, keeps the `k * k_factor` best allowed partial scores, and re-scores each by merging its forward row with the query's terms in term order, then adding the per-list weights with `std::fma` in the order `InvertedIndex::query_impl` adds its lists (largest bound first, ties to the shorter list). So each result's distance is bit-identical to `InvertedIndex`'s, which the tests check.
- **Measured** (big-ann sparse track 1M, M3, one thread): `10:2000:50` 2,477 QPS at recall@10 0.907, `10:10000:10` (the defaults) 1,621 at 0.962, `10:10000:50` 1,183 at 0.989, against `InvertedIndex`'s 118. Re-scoring by merge was about 1.5x faster than a binary search per query term. The approach was first built into one sparse class, with lists split into a weight-ordered head of 16,384 entries and a slot-ordered rest so exact search kept its sequential access (fully weight-ordered lists made exact search 20-25% slower at 1M); as its own class, the lists can be fully weight-ordered with no cap.

### `index_factory.hpp`: `make_index`

`make_index(IndexType, dim, IndexOptions)` returns `unique_ptr<IndexBase>`: a dense index of `dim` floats, an `InvertedIndex` for `IndexType::Inverted` (`pruning`) or a `SparseApproxIndex` for `IndexType::SparseApprox` (`query_terms`, `list_depth`, `k_factor`); `dim` is unused for the sparse types. `IndexOptions` holds `capacity` (Flat), `nlist`, `nprobe`, `pq_subspaces` (0 = PQ off) and `pq_centroids` (IVF), `hnsw_m`, `ef_construction`, `ef_search` and `ivf_sq_bits` (IVF) and `hnsw_sq_bits` (HNSW) (0 for floats, 8 or 4 calls `enable_sq()`), and `custom_ids` (all). Type-specific calls like `train()` require the concrete `IVFIndex` or `HNSWIndex`.

`load_index(std::istream&)` reads a file header and dispatches to `FlatIndex::load_body`, `IVFIndex::load_body`, `HNSWIndex::load_body`, `RefineIndex::load_body`, `InvertedIndex::load_body` or `SparseApproxIndex::load_body`. It returns `unique_ptr<IndexBase>`; C++ callers `dynamic_cast` to `DenseIndex` for a dense index (the tests use `load_dense` and `make_dense` in `tests/load_dense.hpp`). The path overloads wrap it for files:

- `save_index(index, path)` writes to `path + ".tmp"`, checks the stream after `close()` (write errors don't throw on their own), then renames into place; on any failure it removes the `.tmp` file and rethrows.
- `load_index(path)` opens in binary mode, rethrows `invalid_argument` from constructors and setters (bad `nlist`, `nprobe`, ...) as `runtime_error("corrupt file: ...")` so every bad file reports the same way, and rejects trailing bytes after the index.

---

## File Format (`src/serialize.hpp`)

Binary, little-endian (a `static_assert` requires a little-endian host), every size and count stored as `uint64`. Vectors are stored as a `uint64` element count followed by the raw elements (`write_vec` / `read_vec`).

Every index starts with a 9-byte header: magic `QANN` (`uint32`), format version (`uint32`, currently 9) and an `IndexKind` tag (`uint8`). `DenseIndex::save` writes header + body; `load_index` reads the header (`read_header` returns `{kind, version}`) and passes the version to the type's static `load_body`, which reads only the body. Versions `kMinFormatVersion` (1) through `kFormatVersion` (9) are readable. Since version 5, Flat, IVF and HNSW bodies end with the metric (u8; `read_metric` rejects unknown values); older files load as L2. A Refine stores no metric of its own: it takes its base's. Since version 9 every body ends with its metadata (`MetadataStore::save`): the column count (u64), then per column its name (u64 length + bytes), type (u8), null flags (u8 per slot) and values (`count` i64 or f64; for strings `count` u32 codes, then the dictionary size and each entry as length + bytes). A Refine's base writes an empty section (count 0).

| Kind | Body |
| --- | --- |
| `Flat` (1) | `dim`, `count` (slots), `data` (`count * dim` floats), id map |
| `IVF` (2) | `dim`, `nlist`, `nprobe`, `n_total` (slots), `trained` (u8), `pq_distance` (u8), coarse centroids (`nlist * dim`, or empty if untrained), PQ flag (u8) + `PQCodebook`, `fast_scan` (u8, version 4+) and the rotate flag (u8) + rotation (`dim * dim` floats, or empty while untrained; version 8+) if set, SQ bits (u8, version 7+: 0, 4 or 8) + `vmin` and `step` (`dim` floats each, or empty while untrained) if nonzero, then `nlist` lists in cluster order `0..nlist-1`, each slot `ids` + `codes` (PQ: `n * m` bytes, or `ceil(n / 32) * m * 16` in fast-scan blocks; SQ: `n * dim` bytes for 8 bits, `n * ceil(dim / 2)` for 4) or `vecs` (neither), empty lists written as empty vectors; id map; `use_precomputed` (u8, version 3+) |
| `Refine` (3) | `k_factor`, `count` (slots), the base index (full header + body, same version), `data` (`count * dim` floats), id map |
| `Inverted` (5, version 9+; named `Sparse` before it was released) | `count` (slots), the number of posting lists, then per list in increasing term order the term (u32) and its postings (`write_vec` of `(slot u32, weight f32)` pairs), id map. Loading requires non-empty lists with slots strictly increasing and below `count`, and no term twice |
| `SparseApprox` (6, version 9+) | `count` (slots), `query_terms`, `list_depth`, `k_factor`, the number of posting lists, then per list in increasing term order the term (u32), `sorted` (u64) and its postings (`write_vec` of `(slot u32, weight f32)` pairs), id map. Loading requires `k_factor >= 1`, non-empty lists with `sorted <= length`, `[0, sorted)` heaviest first (ties by slot) and the rest in increasing slot order, slots below `count` and not repeated within a list, and no term twice; the forward index is rebuilt |
| `HNSW` (4) | `dim`, `M`, `ef_construction`, `ef_search`, `count` (slots), `max_level` (i32), `entry` (u32), the level generator's state (version 5: one u64, the splitmix64 state, so adds after a load draw the same levels; version 4 stored `std::mt19937`'s text form, which libc++ and libstdc++ write differently, so a version-4 file continues from an FNV-1a hash of that text), the SQ bits per dimension (u8, version 6+: 0 for floats, 8 or 4) and, if nonzero, `vmin` and `step` (`dim` floats each, or empty while untrained) and the codes (`count * dim` bytes for 8 bits, `count * ceil(dim / 2)` for 4), `data` (`count * dim` floats, empty with SQ), `levels` (u8 per slot), level 0 (`count * (2M + 1)` u32), then each slot's upper-level lists in slot and level order, id map |

The id map (`IdMap::save`, version 2+) is `custom_ids` (u8), the slot count, `next_id` (i64, version 3+), `labels` (empty while ids equal slots) and one deleted flag (u8) per slot. It ends each body (in IVF, only the `use_precomputed` flag follows it). For version 1, `load_body` skips it and starts from a map with every slot live and no custom ids; for version 2, `next_id` is the slot count and auto ids carry no labels. Version 3 added `next_id` and auto-id labels for compaction, and IVF's `use_precomputed` flag (version 2 files load with the tables on).

`PQCodebook` (inside IVF, no header): `dim`, `num_subspaces`, `centroids_per_subspace`, `centroids_`. `centroids_t_` and `sdc_table_` are derived and rebuilt on load by `build_tables()`, the same code `train()` runs, so they come out bit-identical.

**Validation on load.** Files are untrusted input, so every loader checks before allocating or indexing: `read_vec` takes a `max_len` and callers then require the exact expected length; sizes are checked against overflow (`count > UINT64_MAX / dim`); IVF ids must be `< n_total` and list sizes must sum to `n_total`; the PQ codebook's `dim` must match the index, and a fast-scan flag needs 16 centroids, an even `m <= 1024` and ADC; a rotation must be present exactly when its flag is set and the index is trained, have `dim * dim` entries and be orthogonal (`RᵀR = I` within 1e-3); a Refine's `count` must equal its base's `slots()` (queries index the raw vectors by base slot), its base's ids must equal its slots and it must have the same format version. A Refine base must be Flat, IVF or HNSW, which also stops a crafted file from nesting Refines until the stack overflows. An HNSW file's neighbor lists must be within capacity and point at real slots, and its entry point must sit on the top level; the SQ bits must be 0, 4 or 8, and with SQ the ranges must have `dim` entries (or none, with no slots) and the codes exactly `count` rows of the width's code bytes. The id map must match its index's slot count, with no duplicate live ids; auto-id labels must be strictly increasing, `>= 0` and below `next_id`, and without labels `next_id` must equal the slot count (or the index must be empty). Metadata columns must have unique non-empty names, a known type, 0/1 null flags and exactly `count` rows, and string codes of non-null rows must point into a dictionary without duplicates. Any failure throws `runtime_error`.

**Changing the format.** `IndexKind` values are append-only: never renumber or reuse one. A new kind needs no version bump: older readers reject it as an unknown kind (`Inverted` was added within version 9). Bump `kFormatVersion` whenever an existing type's layout changes, and read older versions explicitly in each `load_body` (as version 9 does for versions 1 to 8). `tests/data/` holds files saved by 0.1.0a4 (version 1), 0.1.0a6 (version 2, with deletions and custom ids) 0.1.0a7 (version 3, compacted, with precomputed tables off on the IVF file) 0.1.0a10 (version 4: Flat with deletions, IVF fast-scan, and HNSW with deletions) and 0.1.0a11 (version 5: cosine Flat and HNSW, with deletions) with the results they returned, a version-9 SparseApproxIndex (custom ids, two adds, uncompacted deletions, a metadata column, saved settings; queries in `v9_sparse_approx_queries.npz`), plus version-6 RefineIndexes over SQ8 and SQ4 HNSW graphs (odd `dim`, so 4-bit rows end in half a byte) saved on macOS, version-7 SQ8 IVF and SQ4 IVF under a RefineIndex (odd `dim`), a version-8 rotated IVF-PQ under a RefineIndex, a version-9 RefineIndex with int, float and string metadata (nulls from a second add, deletions), and a version-9 SparseIndex (custom ids, uncompacted deletions, term ids up to 100,000, metadata with nulls; queries in `v9_sparse_queries.npz`), also saved on macOS, so Linux CI checks they are portable; the C++ and Python tests load them to catch compatibility breaks, and re-save them in the current version.

---

## Usage Details

The README keeps to what each feature does and how to call it. This section holds the measurements behind its advice and the finer behavior, in the README's order.

### Choosing an index

- **IVF + fast-scan + `RefineIndex` vs HNSW.** On SIFT1M, `HNSWIndex(M=16, ef_construction=200)` reaches recall@10 0.960 at `ef_search=64` (about 80 µs per query on one thread) and 0.995 at 256 (about 340 µs).
- **All cores, SIFT1M, busy machine:**
  - the float graph built in 75 s and reached 0.996 at 11.4k queries/s;
  - HNSW-SQ8 + `RefineIndex(k_factor=2)` reached 0.996 at 15.8k/s;
  - IVF + fast-scan + `RefineIndex` reached 0.995 at 16.9k/s and built in 5 s.
- **Below 0.96 recall HNSW is faster on SIFT1M.** On GloVe-100 (cosine) the order flips: IVF + fast-scan + `RefineIndex` was about twice as fast up to 0.94 recall, and HNSW with `M=16` needed `ef_search=1024` to reach 0.95.
- **Rule of thumb:** IVF + fast-scan when you can train on representative data and build time matters; HNSW when you can't train up front, or when new vectors keep arriving that IVF's clusters weren't trained on.

### IVF: fast-scan, rotation and SQ

- **Fast-scan** is about 6x faster than regular PQ at scanning codes of the same size: 4-bit codes, scored 32 vectors at a time by one table lookup per subspace (NEON on ARM, AVX2 on x86).
  - On SIFT1M with `RefineIndex(k_factor=16)`, 64 fast-scan subspaces match 32 regular ones (recall@10 0.931 at `nprobe=16`, 0.995 at 64) and beat 16 regular ones, while scanning several times faster.
  - Without a `RefineIndex`, recall is lower than regular PQ at the same memory, and distances are approximate: the lookup tables are rounded to 6 bits.
- **Rotation** (`enable_pq(..., rotate=True)`) learns a PCA rotation that spreads variance evenly over the subspaces.
  - On SIFT1M with fast-scan, recall@10 at `nprobe=16` goes from 0.635 to 0.718 without re-ranking, and `k_factor=4` reaches what `k_factor=8` did (0.988 vs 0.956 at `nprobe=64`). With re-ranking that is about 5% more queries per second at the same recall, after the cost of rotating each query.
  - On GloVe (dimensions already uncorrelated) it's within 0.004 either way, and with `k_factor=16` there is little left to gain.
  - Training takes about 0.4 s longer on SIFT1M, and the index stores one `dim x dim` matrix (64 KB at 128 dimensions).
- **IVF SQ** on SIFT1M (all cores, busy machine):
  - 8 bits reached recall@10 0.925 at `nprobe=16` and 0.985 at 64 (floats: 0.932 and 0.995), at 1.3 to 1.4x the queries per second of float lists, in a quarter of the memory;
  - 4 bits + `RefineIndex(k_factor=8)` matched the floats' recall at the same speed-up;
  - on one thread the gain mostly disappears: SQ helps because the float scan is limited by memory bandwidth.
  - When raw vectors fit in memory, fast-scan PQ + `RefineIndex` is still faster, and HNSW-SQ8 is faster at the same recall for about twice the memory.
- **`precomputed_tables`** exist once PQ is enabled and trained, if `nlist * num_subspaces * centroids_per_subspace * 4` bytes fit in 256 MB. They're always off with `Metric.InnerProduct`, which needs only one table per query.

### HNSW: `M`, deletes and SQ

- **Choosing `M`:** on GloVe-100, `M=48` answered 1.5x more queries per second than `M=16` at 0.90 and 0.95 recall, and only `M=48` reached 0.99. It builds about 5x slower and adds 256 bytes of links per vector.
- **Deletes** only mark vectors: queries still route through them so the graph stays connected, but never return them. `compact()` rebuilds the graph, which takes about as long as building it again.
- **SQ** maps each dimension's trained range onto 256 (8-bit) or 16 (4-bit) evenly spaced values. Values outside the training sample's range are clamped, so train on a representative sample.
- **SQ on SIFT1M,** busy machine, `ef_search=256`:
  - 8 bits + `RefineIndex(k_factor=2)` matched the float graph's recall (0.996) at 1.4x its queries per second on all cores (15.8k vs 11.4k) and 1.7x on one thread;
  - 4 bits + `k_factor=4` did nearly the same (0.995, 15.6k);
  - without re-ranking, 8 bits reaches 0.986 and 4 bits only 0.86.
  - 4 bits is no faster than 8, because searches wait on memory latency rather than bytes moved; choose it to halve code memory.
- **SQ on GloVe-100 (cosine):** 8 bits + `k_factor=2` matched the floats exactly; 4 bits alone reached only 0.70 and needed `k_factor=8`.

### Sparse indexes

- **`InvertedIndex` pruning** gives 1.6x more queries per second at `k = 10` on BEIR FiQA (58k documents) with BM25, and about the same at `k = 1000`. Results are identical with it on or off. `benchmarks/sparse_bm25.py` measures it.
- **`SparseApproxIndex`** on SPLADE vectors of 1M MS MARCO passages (big-ann-benchmarks sparse track), one thread on an M3, against exact search at 118 queries per second:

| `query_terms` / `list_depth` / `k_factor` | recall@10 | queries per second |
| --- | --- | --- |
| 10 / 2,000 / 50 | 0.907 | 2,477 (21x) |
| 10 / 10,000 / 10 (the defaults) | 0.962 | 1,621 (14x) |
| 10 / 10,000 / 50 | 0.989 | 1,183 (10x) |
| 20 / 10,000 / 50 | 0.994 | 653 (5.5x) |

  On a 4-CPU x86 VM the defaults gave 0.955 recall at 10.6x exact search on one thread (614 vs 58 queries per second) and 8.0x on all threads (#59). `benchmarks/sparse_bigann.py --approx` measures it.

### Inner product

- **HNSW** links its graph as if every stored vector had one more coordinate, `sqrt(M^2 - |x|^2)` with `M` the largest stored length, which turns inner product search into L2 search for the links. At `ef_search=64`, recall@10 was 0.96 instead of 0.05 on ann-benchmarks' Last.fm factors, and 0.50 instead of 0.05 on VIBE's llama-128-ip, against linking by raw inner product as hnswlib does.
- **IVF** clusters by L2 and probes the lists whose centroids have the largest `q . c`. With exact lists and `nprobe=64`, recall@10 was 0.98 on Last.fm, 0.88 on VIBE's yi-128-ip and 0.84 on llama-128-ip; with fast-scan PQ (rotated, `dim / 2` subspaces) and `RefineIndex(k_factor=8)`, 0.91, 0.81 and 0.73.

### Filtered search

- **How each index filters:**
  - `FlatIndex` scores only the allowed vectors.
  - `IVFIndex` skips the others while scanning, and keeps probing the next nearest lists until it has `k` allowed vectors.
  - `HNSWIndex` walks its graph through every node but returns only allowed ones.
  - A `RefineIndex` passes the filter to its base and re-ranks as usual.
- **Narrow filters are scored directly, which is exact:**
  - HNSW below about `sqrt(ef_search * M * size)` allowed vectors, about 2% of SIFT1M at `ef_search=64` and 2.5% at 128;
  - a plain or SQ `IVFIndex` at up to `3 * nlist` allowed vectors;
  - a `RefineIndex` over IVF at up to `6 * nlist`, from its raw copies.
  - A PQ `IVFIndex` doesn't do this, so wrap it in a `RefineIndex` for narrow filters.
- **HNSW `filter_ef_scaling`** narrows the search width to `max(k, ef_search * sqrt(s))` for a fraction `s` allowed. A filtered beam finds more of its true neighbors than an unfiltered one, so at the full `ef_search` it overshoots unfiltered recall at several times the cost.
  - With the scaling, filtered recall stays near the unfiltered level: SIFT1M at `ef_search=64` gives 0.96 to 0.975 from 2% to 50% allowed (0.953 unfiltered), and GloVe-100 stays within 0.01 of unfiltered.
  - Queries are 1.3 to 3.7 times faster (10% allowed: 405 to 143 µs).
- **IVF `filter_probe_scaling`** probes about `0.75 * nprobe * s^(-1/3)` lists for a fraction `s` allowed.
  - On SIFT1M at `nprobe=16` that keeps recall at 0.92 to 0.96 from 0.2% to 20% allowed (0.93 unfiltered), where stopping at the first 10 allowed vectors gave 0.60 to 0.88. It costs 1.3 to 5.7 times fewer queries per second.
  - It helps most on hard data (GloVe-100: filtered recall 0.50 to 0.75 off, 0.72 to 0.78 on, 0.81 unfiltered). Where filtered recall is already high (fashion-mnist: 0.98 or more either way), turning it off is safe and up to 1.9 times faster.
- **Cost of `allowed`:** a pass over the ids per call, plus one bit per vector in the index. The per-selectivity numbers are under [Benchmarks](#benchmarks-benchmarksbench_indexespy).

### Metadata

- **Types:** a column's type is fixed by the first add that has it, or by `add_metadata_column`. Bool arrays become ints. Ints are accepted into a float column (ints beyond 2^53 are rounded to the nearest float); floats into an int column raise `ValueError`.
- **Long filter chains built in code:** adding terms one `|` at a time copies the chain each step, so it gets slow past a few thousand terms; use `qann.any_of` / `qann.all_of`. For one column against many values, `isin` is one condition, scanned once per query instead of once per value.
- **SQL strings in detail:**
  - Keywords are case-insensitive and follow SQL's precedence. Strings are `'...'` (`''` for a quote); quote column names that are keywords or contain non-ASCII characters as `"..."`.
  - Numbers that overflow, such as `1e400`, raise `ValueError`, and a syntax error raises `ValueError` with its column.
  - Filters nest up to 256 levels of parentheses or `NOT`; long `AND` / `OR` chains don't count toward that.
  - A string may have up to 1024 conditions (an `IN` list counts once), so filter strings from users can't make a query scan its columns thousands of times. `qann.set_max_filter_conditions(n)` changes the limit for the process; filters built with `qann.field` aren't limited.
- **Cost:** each query call scans the columns its filter names once, shared by every row of a `batch_query`: about 0.4 ms per condition per million vectors. Storage:
  - strings: each distinct value once, plus 4 bytes per vector;
  - ints and floats: 8 bytes per vector;
  - every column: 1 byte per vector for its null flag.
- **`add_metadata_column` details:**
  - `default` can't be combined with `ids` / `values` (`TypeError`).
  - Unknown, deleted or repeated ids raise `ValueError`, and a failed call changes nothing.
  - `compact()` never removes columns, even ones left with no values; it does free strings no vector uses any more.
  - Re-adding a removed id doesn't bring back its old metadata: pass it again with the `add`.

### Saving and loading

- **Interrupted saves:** the file is written next to the target as `<path>.tmp` and renamed into place when complete, so a failed save never leaves a partial file or destroys an existing one.
- **Compatibility:** files saved by 0.1.0a4, 0.1.0a6, 0.1.0a7, 0.1.0a10, 0.1.0a11 and 0.1.0a12 onward all still load (see [File Format](#file-format-srcserializehpp)). A file from a newer release than the one loading it raises `RuntimeError` rather than being misread.

---

## Benchmarks (`benchmarks/bench_indexes.py`)

Recall@10 (overlap of the returned top 10 with the true top 10), QPS on all cores (`query_batch`) and bytes per vector for Flat, IVF, IVF + PQ (ADC and SDC, m ∈ {8, 16, 32}), IVF + fast-scan PQ (m ∈ {32, 64}, 4-bit), IVF + PQ + re-ranking (`k_factor` ∈ {4, 16}), IVF + SQ8, IVF + SQ4 alone and re-ranked (`k_factor = 8`), sweeping `nprobe` from 1 to 128, and HNSW (`M = 16`, `ef_construction = 200`; float, SQ8, SQ4, and each SQ width re-ranked: SQ8 with `k_factor = 2`, SQ4 with 4), sweeping `ef_search` from 10 to 256. Training uses each dataset's `learn` set; `nlist ≈ √N`. Each configuration's QPS is the best of 3 full passes, so cold caches after a build don't distort the first measurement. Writes `benchmarks/results/<dataset>.csv` and a recall-vs-QPS plot.

```bash
PYTHONPATH=build python3 benchmarks/bench_indexes.py                    # siftsmall (10k)
PYTHONPATH=build python3 benchmarks/bench_indexes.py --check benchmarks/baseline_siftsmall.json
PYTHONPATH=build python3 benchmarks/bench_indexes.py --threads 1       # single-threaded (default 0 = all cores)
python3 benchmarks/compare.py [ref] [--dataset sift] [--rounds N]       # ref (default HEAD) vs working tree
PYTHONPATH=build python3 benchmarks/bench_indexes.py --dataset sift     # SIFT1M
PYTHONPATH=build python3 benchmarks/bench_indexes.py --dataset glove-100  # GloVe-100, cosine (pip install h5py)
PYTHONPATH=build python3 benchmarks/bench_indexes.py --max-queries 1000
```

- **Recall gate**: `--seed N --write-baseline FILE` records recall for every row; `--check FILE` reruns with the baseline's dataset and seed and exits 1 if any row drops more than `--tolerance` (default 0.02). `benchmarks/baseline_siftsmall.json` (seed 1) is the committed baseline. Regenerate it when a change is *meant* to move recall.
- **`compare.py`**: builds `ref` in a temporary copy (reusing `build/_deps` sources), runs each side's own `bench_indexes.py` alternately for `--rounds`, keeps the best QPS and fastest build per row, and prints recall, QPS ratios and build times. Speed is compared locally rather than in CI because shared CI runners vary 10 to 20% run to run.

Datasets come from `ftp://ftp.irisa.fr/local/texmex/corpus/` (`siftsmall.tar.gz`, `sift.tar.gz`), extracted into `benchmarks/data/`, and from ann-benchmarks (`http://ann-benchmarks.com/glove-100-angular.hdf5`, saved as `benchmarks/data/glove-100/glove-100-angular.hdf5`). Each dataset sets its metric and PQ sizes in `DATASETS`: PQ subspace counts must divide `dim`, so GloVe (dim 100) uses PQ10 / 20 / 25 and fast-scan 20 / 50 (10 and 25 bytes). GloVe has no learn set, so training uses a fixed random sample of 100k base vectors.

**SIFT1M on all cores, after 0.1.0a11** (`main` at `ecea2bb`, with SQ HNSW; Apple M3, 4 performance + 4 efficiency cores, `nlist = 1000`, 10k queries, seed 1). The machine was busy during this run (load average 13 to 138), so absolute QPS is low (Flat 106 vs 196 on an idle machine, plain IVF 6.9k at `nprobe = 16` vs 10.2k); compare rows within the table.

| Index | B/vec | `nprobe = 16` recall@10 | QPS | `nprobe = 64` recall@10 | QPS |
|---|---|---|---|---|---|
| Flat | 512 | 0.999 (exact scan) | 106 | | |
| Plain IVF | 516 | 0.932 | 6.9k | 0.995 | 1.7k |
| IVF + PQ16 | 20 | 0.555 | 31.5k | 0.565 | 8.9k |
| IVF + PQ64×4 fast-scan | 36 | 0.635 | 53.9k | 0.651 | 14.1k |
| IVF + PQ16 + R16 | 532 | 0.929 | 24.5k | 0.988 | 7.7k |
| IVF + PQ32 + R16 | 548 | 0.932 | 15.0k | 0.995 | 4.5k |
| IVF + PQ32×4 fast-scan + R16 | 532 | 0.912 | 60.1k | 0.962 | 30.2k |
| IVF + PQ64×4 fast-scan + R16 | 548 | 0.931 | 39.4k | 0.995 | 16.9k |

HNSW in the same run (`M = 16`, `ef_construction = 200`; B/vec counts the vector or its codes plus level-0 links, and re-ranked rows add the raw floats):

| Index | B/vec | `ef = 32` recall@10 | QPS | `ef = 64` | QPS | `ef = 128` | QPS | `ef = 256` | QPS | build |
|---|---|---|---|---|---|---|---|---|---|---|
| HNSW (floats) | 644 | 0.897 | 70.4k | 0.960 | 40.8k | 0.987 | 22.4k | 0.996 | 11.4k | 75 s |
| HNSW-SQ8 | 260 | 0.892 | 92.2k | 0.953 | 51.5k | 0.978 | 27.6k | 0.986 | 14.7k | 50 s |
| HNSW-SQ8 + R2 | 772 | 0.896 | 83.7k | 0.960 | 50.7k | 0.987 | 28.6k | 0.996 | 15.8k | |
| HNSW-SQ4 | 196 | 0.805 | 87.0k | 0.840 | 52.1k | 0.853 | 28.9k | 0.857 | 15.3k | 64 s |
| HNSW-SQ4 + R4 | 708 | 0.920 | 70.6k | 0.957 | 47.7k | 0.986 | 28.3k | 0.995 | 15.6k | |

- At 0.93 recall, PQ64×4 fast-scan + R16 is 5.7× plain IVF and 1.6× PQ16 + R16. At 0.995 it is 10× plain IVF and 3.8× PQ32 + R16, the regular-PQ configuration with the same recall.
- PQ32×4 fast-scan is the fastest IVF configuration but tops out at 0.96 recall with R16; use PQ64×4 when recall matters.
- At 0.995, compressed HNSW nearly closes the gap to IVF: SQ8 + R2 at 15.8k and SQ4 + R4 at 15.6k against fast-scan + R16 at 16.9k (0.93×), where the float graph is at 11.4k (0.67×). Around 0.96 recall HNSW is ahead: SQ8 + R2 reaches 0.960 at 50.7k, while fast-scan + R4 reaches 0.956 at `nprobe = 64` with 17.2k.
- SQ8 alone loses little recall (0.978 vs 0.987 at `ef = 128`) at 40% of the float graph's memory; SQ4 alone levels off near 0.86, so it needs re-ranking. SQ4 is no faster than SQ8: a search waits on the latency of each random read, not on the bytes moved.
- PQ alone plateaus at 0.38 / 0.56 / 0.73 recall for m = 8 / 16 / 32; re-ranking lifts m = 16 to 0.99 and m = 32 to 0.995.
- SDC recall *falls* as `nprobe` grows, because more candidates expose its query-side quantization error.

Thread scaling (`set_num_threads`, same commit, every index built once on all cores; QPS by thread count). The machine was busy here too (load average 22 to 90), so the steps are noisy; the HNSW-SQ8 `ef = 256` drop at 8 threads is noise, not a real slowdown.

| Index | recall@10 | 1 thread | 2 | 4 | 8 |
|---|---|---|---|---|---|
| Plain IVF, `nprobe = 16` | 0.932 | 2.3k | 3.7k (1.6×) | 4.2k (1.8×) | 4.7k (2.0×) |
| IVF + PQ16, `nprobe = 16` | 0.555 | 10.3k | 19.1k (1.9×) | 24.9k (2.4×) | 29.0k (2.8×) |
| IVF + PQ16 + R16, `nprobe = 16` | 0.929 | 8.6k | 9.8k (1.1×) | 18.3k (2.1×) | 26.3k (3.1×) |
| IVF + PQ64×4 fast-scan + R16, `nprobe = 16` | 0.931 | 11.1k | 22.8k (2.1×) | 30.5k (2.7×) | 33.2k (3.0×) |
| IVF + PQ64×4 fast-scan + R16, `nprobe = 64` | 0.995 | 5.3k | 9.5k (1.8×) | 12.4k (2.4×) | 14.7k (2.8×) |
| HNSW, `ef = 64` | 0.960 | 12.7k | 19.4k (1.5×) | 26.1k (2.1×) | 32.9k (2.6×) |
| HNSW, `ef = 256` | 0.995 | 3.0k | 6.0k (2.0×) | 8.6k (2.9×) | 9.9k (3.3×) |
| HNSW-SQ8 + R2, `ef = 64` | 0.960 | 20.3k | 31.1k (1.5×) | 34.7k (1.7×) | 42.8k (2.1×) |
| HNSW-SQ8 + R2, `ef = 256` | 0.996 | 5.2k | 8.5k (1.6×) | 11.7k (2.3×) | 9.8k |

- Single-threaded, SQ8 is where compression pays most: HNSW-SQ8 + R2 is 1.6× the float graph at `ef = 64` and 1.7× at 256, with the same recall. On all cores the gain shrinks (about 1.3×), as threads already overlap the memory waits that SQ8 shortens.
- Plain IVF barely scales: it streams 512-byte raw vectors and is limited by memory bandwidth, while PQ and fast-scan scan small codes and keep scaling. The efficiency cores add less than the performance cores.
- On one thread, fast-scan + R16 at 0.995 (5.3k) beats the float HNSW at the same recall (3.0k) and matches SQ8 + R2 (5.2k).

**GloVe-100 on all cores, cosine** (1,183,514 word vectors, dim 100, 10k queries, ground truth from ann-benchmarks' angular neighbors, `nlist = 1088`, seed 1, same commit as the SIFT1M run plus the GloVe loader). The machine was even busier (load average 80 to 135), so QPS is rough; compare within the table. Flat returns recall 1.000, which checks the cosine metric against the dataset's own ground truth.

| Index | B/vec | recall@10 / QPS at `nprobe` 16 | 64 | 128 |
|---|---|---|---|---|
| Plain IVF | 404 | 0.784 / 5.9k | 0.903 / 1.4k | 0.947 / 0.8k |
| IVF + PQ25 + R16 | 429 | 0.783 / 16.9k | 0.900 / 5.6k | 0.942 / 3.0k |
| IVF + PQ50×4 fast-scan + R16 | 429 | 0.782 / 47.6k | 0.897 / 18.6k | 0.938 / 9.5k |

| Index | B/vec | recall@10 / QPS at `ef` 64 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| HNSW (floats) | 532 | 0.761 / 28.2k | 0.884 / 8.7k | 0.926 / 4.4k | 0.956 / 2.3k |
| HNSW-SQ8 | 232 | 0.756 / 34.5k | 0.875 / 10.4k | | |
| HNSW-SQ8 + R2 | 632 | 0.761 / 27.0k | 0.885 / 9.6k | 0.926 / 4.1k | 0.956 / 2.7k |
| HNSW-SQ4 | 182 | 0.624 / 35.3k | 0.697 / 11.1k | | |
| HNSW-SQ4 + R4 | 582 | 0.747 / 34.4k | 0.871 / 10.7k | | |
| HNSW-SQ4 + R8 | 582 | 0.774 / 28.7k | 0.875 / 10.5k | 0.917 / 5.9k | 0.950 / 3.1k |

(`ef` 512 and 1024 come from a second run on the same busy machine; with `k_factor = 8` the base searches with `ef = max(ef_search, 80)`, which is why SQ4 + R8 beats the floats at 64.)

- GloVe is much harder than SIFT: at the same settings every index loses 0.05 to 0.1 recall, and reaching 0.95 takes `nprobe = 128` or `ef_search = 1024` rather than 64 and 256.
- Here IVF + fast-scan + re-ranking is the fastest path up to about 0.94 recall (0.897 at 18.6k vs HNSW's 0.884 at 8.7k), the opposite of SIFT1M. HNSW with `M = 16` reaches 0.95 only at `ef = 1024` (2.3k); a larger `M` is the usual fix for GloVe and was not swept.
- SQ8 holds up on real embeddings: alone it loses 0.01 against the floats, and with `k_factor = 2` it matches them exactly at every `ef`.
- SQ4 does not: 16 levels per dimension are coarse for GloVe's continuous values, so alone it levels off near 0.70 (0.86 on SIFT), and `k_factor = 4` stays 0.01 short. `k_factor = 8` matches the floats within 0.01, so use 8 for SQ4 on embeddings like these.

**HNSW `M` sweep** (`benchmarks/hnsw_m_sweep.py`, issue #33): one float graph per (`M`, `ef_construction`) of (16, 200), (32, 300) and (48, 400), `ef_search` swept over 16 to 1024 (doubling), then the smallest `ef` reaching each recall target timed on all cores. The timings alternate between the three graphs in 3 rounds and keep each one's best, because load swung between about 40 and 390 during the run; a first run timed the configurations one after another and its QPS contradicted itself, so only its recall (identical) is used. Each `ef` step doubles, so a row can overshoot its target, which understates the gaps. SQ8 + R2 matched the float recall at every `M` and `ef` on both datasets (first run).

| GloVe-100 | build | link bytes / vec | QPS at 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|
| `M = 16` | 121 s | 132 | 4.4k (`ef` 512) | 2.4k (`ef` 1024) | not reached |
| `M = 32` | 384 s | 260 | 5.1k (`ef` 256) | 2.7k (`ef` 512) | not reached |
| `M = 48` | 625 s | 388 | 6.7k (`ef` 128) | 3.7k (`ef` 256) | 1.0k (`ef` 1024) |

| SIFT1M | build | link bytes / vec | QPS at 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|
| `M = 16` | 78 s | 132 | 29.6k (`ef` 64) | 30.1k (`ef` 64) | 9.6k (`ef` 256) |
| `M = 32` | 169 s | 260 | 39.9k (`ef` 32) | 23.8k (`ef` 64) | 12.5k (`ef` 128) |
| `M = 48` | 246 s | 388 | 34.2k (`ef` 32) | 36.4k (`ef` 32) | 11.2k (`ef` 128) |

- On GloVe a larger `M` is clearly worth it: `M = 48` is about 1.5× `M = 16` at 0.90 and 0.95, and the only one to reach 0.99 within `ef = 1024`. Recall at `ef = 256` goes 0.884 / 0.941 / 0.962. The cost is a 5× longer build and 256 more bytes of links per vector (more than half the 400-byte vector).
- On SIFT `M = 16` is enough up to 0.95 (the differences there are within one `ef` step); at 0.99 `M = 32` is 1.3× faster and `M = 48` adds nothing.
- So the default stays `M = 16` (fast builds, right for SIFT-like data), and the README advises `M = 32` to 48 for high-dimensional embeddings like GloVe. Even at `M = 48`, IVF + PQ50×4 fast-scan + R16 stays ahead on GloVe up to 0.94 (18.6k at 0.897 in the earlier run, against 6.7k at 0.925 here, though from runs at different load).

**IVF + SQ** (`benchmarks/bench_indexes.py` rows `IVF-SQ8`, `IVF-SQ4`, `IVF-SQ4+R8`; numbers from a separate script that built float, SQ8 and SQ4 + R8 IVF indexes from the same training and timed them in alternating rounds, best of 3, all cores, load 7 to 126):

| SIFT1M | B/vec | `nprobe = 16` recall / QPS | `nprobe = 64` |
|---|---|---|---|
| IVF (floats) | 516 | 0.932 / 9.2k | 0.995 / 2.5k |
| IVF-SQ8 | 132 | 0.925 / 13.0k (1.41×) | 0.985 / 3.3k (1.34×) |
| IVF-SQ4 + R8 | 580 | 0.932 / 11.9k (1.30×) | 0.995 / 3.2k (1.32×) |

| GloVe-100 (second run) | `nprobe = 8` | 16 | 32 | 128 |
|---|---|---|---|---|
| IVF (floats) | 0.707 / 19.4k | 0.784 / 11.3k | 0.849 / 3.1k | 0.947 / 0.8k |
| IVF-SQ8 | 0.702 / 1.53× | 0.778 / 0.89× | 0.840 / 1.34× | 0.934 / 1.56× |
| IVF-SQ4 + R8 | 0.707 / 1.25× | 0.784 / 0.65× | 0.848 / 1.36× | 0.947 / 1.56× |

- SQ8 costs about 0.01 recall (0.013 on GloVe at 128) for 4× smaller lists and 1.3-1.6× QPS; SQ4 + R8 recovers the floats' recall exactly at a similar speed-up, but keeps the raw vectors.
- GloVe at `nprobe = 16` was slower with SQ in both runs (0.68× / 0.85× in the first). It is not explained yet: the float rows' own QPS falls 3.6× from 16 to 32 lists for 2× the work, which suggests the float timing at 16 is the outlier, but the machine was too loaded (load ~100) to settle it.
- Positioning: IVF-SQ8 is the IVF option for high recall without raw vectors in RAM. HNSW-SQ8 reaches the same recall faster for about twice the memory (260 vs 132 B/vec on SIFT), and fast-scan + R16 is faster when the raw vectors fit.

**Filtered search** (`benchmarks/filter_selectivity.py`, issue #37): a random allowed set per selectivity, ground truth from a filtered `FlatIndex` query, 1,000 queries on all cores, timings in alternating rounds (best of 3). Load was 75 to 210 during the run, so the QPS are rough; recall is exact. Recall / QPS:

| SIFT1M allowed | Flat | IVF `nprobe = 16` | IVF-PQ64×4fs + R4 | HNSW `ef = 64` | HNSW-SQ8 + R2 |
|---|---|---|---|---|---|
| 100% | 1.000 / 170 | 0.929 / 5.0k | 0.900 / 8.9k | 0.954 / 5.8k | 0.957 / 6.7k |
| 20% | 1.000 / 280 | 0.891 / 6.0k | 0.880 / 9.9k | 0.992 / 3.4k | 0.993 / 5.3k |
| 5% | 1.000 / 740 | 0.845 / 5.6k | 0.843 / 10.7k | 0.998 / 2.0k | 0.999 / 2.9k |
| 2% | 1.000 / 4.9k | 0.795 / 8.7k | 0.795 / 46.4k | 1.000 / 4.8k | 1.000 / 8.5k |
| 0.5% | 1.000 / 20.6k | 0.704 / 8.2k | 0.704 / 44.0k | 1.000 / 24.2k | 1.000 / 23.1k |
| 0.1% | 1.000 / 135k | 0.519 / 9.4k | 0.750 / 19.4k | 1.000 / 127k | 1.000 / 122k |

- **HNSW** gains recall as the filter narrows (the beam searches longer for `ef` allowed nodes), and slows down until 3.2%, where it switches to scoring the allowed vectors directly: exact, and as fast as `FlatIndex` over the same set. (These rows predate `filter_ef_scaling` (#89), which now narrows the beam so filtered recall stays near unfiltered at 1.3-3.7x less time, with the switch at about 2%; see the HNSW section.) A separate run with the switch forced either way put the crossover between 3% and 5% for both the float and the SQ8 graph; graph search at 0.1% ran at ~100 QPS against ~120k for direct scoring.
- **IVF** lost recall as the filter narrowed (the table above, from before #72): the filtered top 10 at selectivity `s` lie about as deep as the unfiltered top `10 / s`, and those are spread over more lists than `nprobe`. Probing past `nprobe` only guaranteed `k` results, not the right ones. Since #72 a filtered query probes at least `0.75 * nprobe * s^(-1/3)` lists (`filter_probe_scaling`, on by default, not saved), measured on SIFT1M (`nlist` 1000, `nprobe` 16) to keep recall@10 at 0.92 to 0.96 from 0.2% to 20% allowed (unfiltered 0.93), at 1.3 to 5.7 times lower QPS than stopping at the first `k` allowed (plain IVF 0.5%: 170k QPS / 0.71 to 30k / 0.96; fast-scan + re-rank 1%: 104k / 0.75 to 31k / 0.95). `nprobe / s` was tried first and overshot (recall 1.0 at many times the cost). Raising `nprobe` restores recall at the usual cost; below ~1%, `FlatIndex` or `HNSWIndex` over the same vectors are both exact and faster. A `RefineIndex` over an IVF base skips the base when the filter allows at most `6 * nlist` vectors and scores them from its raw copies (`RefineIndex::use_direct` / `direct_scan`, #72). Against the scaled probing above on SIFT1M fast-scan + re-rank 4 (`nlist` 1000, `nprobe` 16), the exact direct scan ran at 110k QPS against 36k at 2,000 allowed, 49k against 37k at 5,000 and 23k against 36k at 10,000, a crossover near 6,400. A bare plain or SQ `IVFIndex` makes the same switch at `3 * nlist` allowed vectors (`IVFIndex::use_direct` / `direct_scan`), finding each through `slot_list_` / `slot_pos_` (6 bytes per vector, kept on add and rebuilt on compact and load, not saved). Against the scaled probing on SIFT1M, plain IVF's direct scan ran at 98k QPS against 44k at 2,000 allowed and 33k against 49k at 5,000, a crossover near 4,000; SQ's direct scoring is somewhat slower, hence 3. PQ has no lookup and doesn't switch: scoring scattered allowed vectors needs one ADC table per list, which measured 2-50x slower.

**Build times** on all cores (this run, busy machine): IVF trains in 2.3 s and adds 1M vectors in 1.1 s (`add_batch`); IVF + PQ16 trains in 5.2 s and adds in 4.0 s; PQ64×4 fast-scan, the fastest PQ build, 3.4 s and 1.9 s; HNSW builds in 50 to 75 s. On an idle machine earlier, IVF + PQ16 trained in 3.4 s (8.6 s on one thread) and added in 3.0 s. Before `assign_nearest`, IVF training took 37 s and IVF + PQ16 55 s.

The sections below are from earlier releases.

**Precomputed tables** (`profile_query`, `nprobe = 16`, 1 thread, both paths in one run): the ADC table build drops from 37.8 to 9.0 µs per query, taking IVF + PQ16 from 131 to 108 µs (+21% QPS). The code scan (~70 µs) is now most of a PQ query.

**Batched coarse search** (`profile_query`, which uses `query_batch`, 1 thread, `VECLIB_MAXIMUM_THREADS=1`): coarse search drops from about 19 µs to 7.5 µs per query. Together with precomputed tables, IVF + PQ16 goes from 131 to 103 µs per query and IVF + PQ16 + R16 from 188 to 149 µs. The machine was loaded (load average 6), so treat the totals as approximate; the coarse section is the reliable comparison. The thread-scaling table above predates both changes.

**Fast-scan** (`profile_query`, 1 thread, `VECLIB_MAXIMUM_THREADS=1`, `nprobe = 16`, IVF-PQ64×4fs vs IVF-PQ16 in the same run, load average 3 to 6): 49 vs 91 µs per query (1.9×), with the code scan at 33.5 vs 66 µs (2.0×); with re-ranking 102 vs 159 µs. In an earlier, more loaded run, `nprobe = 64` gave 184 vs 351 µs (1.9×). PQ64×4fs is also the more accurate of the two after re-ranking (see the fast-scan section). The scan is about half of PQ16's rather than the kernel's 6×: the kernel is ~19 µs per query, rounding the 16 lists' tables ~3 µs (NEON `quantize_lut`; ~10 µs scalar), and the rest is building the tables and pushing survivors. `fast_scan_block_mask` compares each block's sums with the threshold in registers, so blocks with no survivor (most of them once the top k fills) cost nothing beyond the kernel. Coarse search, re-ranking and result handling don't change, which bounds the end-to-end gain (~3.6× at most without re-ranking, ~2.8× with it, even if the scan took no time).

---

## Python Bindings (`bindings/python_bindings.cpp`)

Inputs are float32 C-contiguous NumPy arrays, passed zero-copy as `nb::ndarray` (`FloatMatrix` 2D, `FloatVector` 1D, `ByteVector` uint8 1D). Wrappers validate shapes and raise `ValueError` so mismatched input never reaches C++ in Release builds, where asserts are off.

```python
import numpy as np, qann

x = np.random.rand(10_000, 128).astype(np.float32)

ivf = qann.IVFIndex(128, nlist=100, nprobe=10)
ivf.enable_pq(16)                      # optional; before train/add
ivf.train(x)
refine = qann.RefineIndex(ivf, k_factor=10)   # optional; wrap before adding
refine.add(x)                          # adds to ivf too
ivf.nprobe = 32                        # tune after training
ivf.pq_distance = qann.PQDistance.ADC    # default ADC
ivf.precomputed_tables = False         # default True; frees the precomputed ADC tables
ids, dists = refine.query(x[0], 10)      # (10,) arrays
ids, dists = refine.batch_query(x[:5], 10)  # (5, 10) arrays

pq = qann.PQCodebook(128, 16)
pq.train(x)
code = pq.encode(x[0])                 # (16,) uint8
table = pq.compute_adc_table(x[1])     # (16 * 256,) float32
pq.distance_adc(table, code)
```

`RefineIndex` holds its base by reference; `keep_alive` keeps the Python base object alive as long as the wrapper exists. A `RefineIndex` returned by `load` owns its base instead (`owned_base_`).

Also exposed: `FlatIndex`, `IndexType`, `IndexOptions`, `make_index`, `set_num_threads(n)` / `num_threads()`, and row-wise `l2_distance(a, b)` / `cosine_distance(a, b)`.

Ids: `add(data, ids=None)` is bound once on `DenseIndex`, `remove(ids)` and `compact()` once on `IndexBase`; all three take the write lock and are refused (`ValueError`) on an index a `RefineIndex` wraps. `ids` accepts any sequence of integers: `as_ids` runs `numpy.asarray`, rejects non-integer and non-1D input with `TypeError`, then converts to contiguous `int64`; the stubs show it as `Iterable[int]` (`nb::typed<nb::iterable, int>`). Constructors take `custom_ids` as a keyword-only argument.

Save/load: `save(path)` is bound once on `IndexBase` and inherited by every index class; `qann.load(path)` binds the path overload of `load_index` and returns the concrete type. Paths go through nanobind's `std::filesystem::path` caster, so `str` and `pathlib.Path` both work. `RefineIndex.base` is a read-only property returned with `rv_policy::reference_internal`, which keeps the RefineIndex alive while Python holds its base (needed for a loaded RefineIndex, which owns its base).

**GIL and locking.** Every `DenseIndex` carries a `std::shared_mutex` (`DenseIndex::mutex()`), unused by the C++ library itself. The bindings run index work through `read_locked` (shared: `query`, `batch_query`, `size`, `save`, getters) or `write_locked` (exclusive: `add`, `remove`, `compact`, `train`, `enable_pq`, setters, and `RefineIndex.__init__`, which checks its base is empty). Both release the GIL *before* taking the lock, so a thread waiting for a lock never holds the GIL the lock holder needs to finish; the work inside must not touch Python objects, so NumPy results are built after the lock and GIL are released and reacquired. `lock_chain` locks a `RefineIndex` and then each base under it, always outermost first, because a base is also reachable from Python (e.g. `ivf.add()` while queries run through a `RefineIndex` over `ivf`). `load` only releases the GIL (`call_guard`), since the index it creates isn't shared yet. `PQCodebook` methods keep the GIL.

**Type stubs.** `nanobind_add_stub` (CMake target `qann_stub`) imports the built module and writes `build/qann.pyi`, so stubs always match the bindings. Wheels install it as `qann-stubs/__init__.pyi`: type checkers read installed stubs only from packages (PEP 561), and `qann` itself is a single extension module. The shared `add`/`query`/`batch_query`/`size`/`dim` methods are bound once on `DenseIndex`, so the `DenseIndex` returned by `load` and `make_index` type-checks with them.

Results are NumPy arrays that take ownership of the C++ buffer (no copy): IDs are `int64`, distances `float32`, codes `uint8`. `query` returns 1D arrays with one entry per result. `batch_query` returns `(num_queries, k)` arrays; a row with fewer than `k` results is padded with ID `-1` and distance `inf`.

---

## Build, Test, Benchmark

```bash
pip install .               # builds the Python module into a wheel (no C++ tests)
./build.sh                  # Release build in ./build/, then ctest
./build.sh build Debug      # Debug build (asserts on)
./build/test_vecengine "[pq]"   # run one tag: [l2] [cosine] [index] [ivf] [pq] [adc] [sdc] [refine] [serialize] [file] [precomputed] ...
./build/bench_vecengine
```

- Dependencies come from FetchContent: Catch2 v3.5.3, Google Benchmark v1.8.3, nanobind v3.1.0.
- BLAS for k-means (`VECENGINE_USE_BLAS`): on Apple, `vecengine_core` links `-framework Accelerate` (PUBLIC, so tests and the Python module link it too). Elsewhere `find_package(BLAS)` plus a `cblas.h` search (also under `include/openblas`); on Debian/Ubuntu `apt install libopenblas-dev`. Configure prints which one it used, or that it fell back to the per-point scan. OpenBLAS runs its own threads inside `sgemm`; if PQ training (which calls it from every `parallel_for` worker) oversubscribes, set `OPENBLAS_NUM_THREADS=1`.
- Packaging: `pyproject.toml` uses scikit-build-core, builds only `vecengine_py` and `qann_stub` with `VECENGINE_BUILD_TESTS=OFF` (skips fetching Catch2 and Google Benchmark), and installs the module at the wheel root and the stubs as `qann-stubs/`. Runtime dependency: NumPy.
- `-mavx2 -mfma` are only added on x86 so arm64 builds never see unsupported flags.
- Google Benchmark's own tests are disabled via cache variables to avoid its stale GoogleTest download.
- Link-time optimization was tried and left off: no query speedup, and k-means training got ~35% slower.
- `CMAKE_EXPORT_COMPILE_COMMANDS=ON` writes `build/compile_commands.json` for clangd.
- Targets: `vecengine_core` (static lib), `test_vecengine`, `bench_vecengine`, `vecengine_py` (Python module), `qann_stub` (`qann.pyi`).
- CI (`.github/workflows/ci.yml`, on pushes to `main` and on PRs): builds and runs `ctest` on `ubuntu-24.04` (with OpenBLAS) and `macos-15`, then runs the siftsmall recall gate on macOS only, since the baseline was recorded with Accelerate. siftsmall is cached between runs.

---

## Not Yet Implemented

Future work and ideas are tracked as GitHub issues, not here: see the [open `idea` issues](https://github.com/maxtmiller/qann/issues?q=is%3Aissue+is%3Aopen+label%3Aidea). Each one records what was measured or decided so far.

## References

Methods qann implements, and the benchmarks it is measured with. Numbers match the citations above.

1. H. Jégou, M. Douze, C. Schmid. **Product Quantization for Nearest Neighbor Search.** IEEE TPAMI 33(1):117-128, 2011. [doi:10.1109/TPAMI.2010.57](https://doi.org/10.1109/TPAMI.2010.57). Product quantization with ADC and SDC (`PQCodebook`) and the inverted file over residuals (`IVFIndex` with PQ).
2. H. Jégou, R. Tavenard, M. Douze, L. Amsaleg. **Searching in one billion vectors: re-rank with source coding.** ICASSP 2011, 861-864. [doi:10.1109/ICASSP.2011.5946540](https://doi.org/10.1109/ICASSP.2011.5946540). Re-ranking a shortlist with exact distances (`RefineIndex`).
3. F. André, A.-M. Kermarrec, N. Le Scouarnec. **Cache locality is not enough: High-Performance Nearest Neighbor Search with Product Quantization Fast Scan.** PVLDB 9(4):288-299, 2015. [doi:10.14778/2856318.2856324](https://doi.org/10.14778/2856318.2856324). 4-bit PQ codes scored with SIMD shuffle lookups (`fast_scan`).
4. M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazaré, M. Lomeli, L. Hosseini, H. Jégou. **The Faiss library.** arXiv:2401.08281, 2024. [arxiv.org/abs/2401.08281](https://arxiv.org/abs/2401.08281). Implementation details qann follows: precomputed IVF-PQ tables, the fast-scan block layout, empty-cluster splitting in k-means.
5. T. Ge, K. He, Q. Ke, J. Sun. **Optimized Product Quantization.** IEEE TPAMI 36(4):744-755, 2014. [doi:10.1109/TPAMI.2013.240](https://doi.org/10.1109/TPAMI.2013.240). Eigenvalue allocation for the PCA-balanced rotation (`rotation.cpp`).
6. Y. A. Malkov, D. A. Yashunin. **Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.** IEEE TPAMI 42(4):824-836, 2020. [doi:10.1109/TPAMI.2018.2889473](https://doi.org/10.1109/TPAMI.2018.2889473). The HNSW graph and its neighbor-selection heuristic (`HNSWIndex`).
7. Y. Bachrach, Y. Finkelstein, R. Gilad-Bachrach, L. Katzir, N. Koenigstein, N. Nice, U. Paquet. **Speeding up the Xbox recommender system using a euclidean transformation for inner-product spaces.** RecSys 2014, 257-264. [doi:10.1145/2645710.2645741](https://doi.org/10.1145/2645710.2645741). The extra coordinate that lets HNSW link by L2 for `Metric::InnerProduct`.
8. M. Aumüller, E. Bernhardsson, A. Faithfull. **ANN-Benchmarks: A benchmarking tool for approximate nearest neighbor algorithms.** Information Systems 87:101374, 2020. [doi:10.1016/j.is.2019.02.006](https://doi.org/10.1016/j.is.2019.02.006). External benchmark suite (`benchmarks/external`, branch `external-benchmarks`).
9. E. Jääsaari, V. Hyvönen, M. Ceccarello, T. Roos, M. Aumüller. **VIBE: Vector Index Benchmark for Embeddings.** arXiv:2505.17810, 2025. [arxiv.org/abs/2505.17810](https://arxiv.org/abs/2505.17810). External benchmark suite with embedding and inner product datasets (`benchmarks/external`).
