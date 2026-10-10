# QANN: Architecture & File Reference

## Overview

QANN (Quantized Approximate Nearest Neighbors) is a C++20 vector search library with a Python extension module, `qann`, built on nanobind. The C++ API lives in namespace `vecengine` under `include/vecengine/`. It has three layers:

1. **Distance kernels**: squared L2 and cosine distance with scalar, AVX2+FMA and ARM NEON paths, selected at compile time.
2. **Quantization and clustering**: a shared k-means helper (BLAS-accelerated when a CBLAS is available) and a Product Quantization (PQ) codebook.
3. **Indexes**: `FlatIndex` (exact brute force), `IVFIndex` (inverted file, optionally PQ-compressed) and `RefineIndex` (exact re-ranking wrapper), behind a common `Index` interface.

All arithmetic lives in C++. Python gets zero-copy access to NumPy arrays via `nb::ndarray`. `benchmarks/bench_indexes.py` measures recall vs QPS on the SIFT datasets.

---

## Directory Structure

```
vecengine/
├── .github/workflows/ci.yml        # build + test (Ubuntu, macOS), recall gate
├── CMakeLists.txt                  # build orchestrator
├── build.sh                        # configure + build + test in one command
├── pyproject.toml                  # PEP 517 packaging (scikit-build-core)
├── include/vecengine/              # public API
│   ├── aligned_allocator.hpp       # std::allocator with 32/64-byte alignment
│   ├── distance.hpp                # distance function declarations
│   ├── distance_inl.hpp            # inline compile-time SIMD dispatch
│   ├── index.hpp                   # Index interface + Neighbor
│   ├── flat_index.hpp              # exact brute-force index
│   ├── ivf_index.hpp               # IVF index, optional PQ
│   ├── pq.hpp                      # PQCodebook + PQDistance
│   ├── refine_index.hpp            # exact re-ranking wrapper
│   └── index_factory.hpp           # make_index() + IndexOptions, save_index() / load_index()
├── src/
│   ├── distance.cpp                # scalar / AVX2 / NEON kernels
│   ├── kmeans.hpp / kmeans.cpp     # internal k-means helper (not public API)
│   ├── parallel.hpp                # internal parallel_for (not public API)
│   ├── serialize.hpp               # internal binary I/O helpers + file header (not public API)
│   ├── id_map.cpp                  # IdMap: slots <-> user ids, deletions
│   ├── index.cpp                   # default Index::add_batch / query_batch
│   ├── pq.cpp
│   ├── flat_index.cpp
│   ├── ivf_index.cpp
│   ├── refine_index.cpp
│   └── index_factory.cpp
├── tests/
│   ├── test_distance.cpp           # Catch2 v3 tests for every layer
│   ├── test_serialize.cpp          # save/load round trips and corrupt-file rejection
│   ├── test_id_map.cpp             # IdMap, custom ids and deletion per index, format v2
│   ├── test_python.py              # pytest smoke, save/load, ids and threading tests
│   └── data/                       # index files saved by 0.1.0a4 (format v1) + its results
├── benchmarks/
│   ├── bench_distance.cpp          # Google Benchmark distance kernels
│   ├── bench_indexes.py            # recall vs QPS on SIFT
│   ├── compare.py                  # ref vs working tree: recall, QPS, build times
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

`l2_distance` returns **squared** L2. It ranks neighbors identically to true L2 and skips the `sqrt`. Every index and PQ distance in this repo is squared L2.

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
- Init picks `k` distinct random points (partial Fisher-Yates). Empty clusters are handled by `reseed_empty_clusters`: one pass keeps the E furthest points (distance to their assigned centroid, skipping points within `1e-6`) in a size-E min-heap, and each of the E empty clusters gets a distinct one. If the heap runs out, the largest cluster is split FAISS-style: its centroid is copied and the two copies are scaled by `1 ± 1/1024`, and its size is halved so the next split picks another cluster.
- A final assignment pass runs after the last update, so `assignments` match the returned `centroids`. IVF relies on this to compute PQ training residuals.

**Assignment is ~99.9% of k-means time**, so it is done in bulk by `assign_nearest`:

- **With BLAS (`VECENGINE_USE_BLAS`)**: `argmin_c ‖x − c‖² = argmin_c (‖c‖² − 2 x·c)`, and the `x·c` terms for a block of 1024 points against all centroids are one `cblas_sgemm` call. On Apple that is Accelerate, which runs on the M-series matrix unit; elsewhere any BLAS that ships `cblas.h` (OpenBLAS, netlib CBLAS). `lda = stride` lets PQ subspaces use strided input without copying. The per-row argmin is branchless (conditional selects).
- **Without BLAS**: falls back to `nearest_centroid` per point.

Measured on SIFT's 100k learn set: IVF assignment (k = 1000) 1.54 s → 0.086 s (18×), PQ subspaces 2.9 to 3.7×, with identical assignments. The `‖c‖² − 2x·c` scores are only used for ranking, never as distances.

`nearest_centroid` is the single-point scan, still used by `IVFIndex::add`. `IVFIndex::add_batch` uses `assign_nearest`.

`kmeans` takes an optional `seed`; without one it seeds from `std::random_device`, so training is non-deterministic run to run. `IVFIndex::train` and `PQCodebook::train` take an optional seed too (Python: `train(x, seed=1)`): coarse k-means uses `seed`, PQ uses `seed + 1` onward, one per subspace. With the same seed and the same BLAS, results are bit-identical, which the recall gate relies on.

### `pq.hpp` / `pq.cpp`: `PQCodebook`

Splits a `dim`-vector into `m = num_subspaces` slices of `sub_dim = dim / m` floats and quantizes each slice to one of `K = centroids_per_subspace` (max 256) centroids. A vector becomes `m` bytes: at `dim = 128`, `m = 16` that is 512 bytes down to 16.

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

---

## Layer 3: Indexes

### `index.hpp`: `Index`

```cpp
virtual void add(span<const float> vec) = 0;
virtual void add_batch(span<const float> vecs, size_t n, const int64_t* ids = nullptr);
virtual vector<Neighbor> query(span<const float> vec, size_t k) const = 0;
virtual vector<vector<Neighbor>> query_batch(span<const float> queries, size_t num_queries, size_t k) const;
virtual size_t remove(span<const int64_t> ids) = 0;
virtual void save(std::ostream& out) const = 0;
virtual size_t slots() const noexcept = 0;      // ever added, including deleted
virtual bool custom_ids() const noexcept = 0;
virtual size_t size() const noexcept = 0;       // live vectors
virtual size_t dim() const noexcept = 0;
```

`Neighbor { size_t index; float distance; }`. Results are sorted nearest first.

**Slots and ids.** Every added vector gets a *slot*: its position in insertion order (0, 1, 2, ...), which never changes. Indexes store and scan slots internally (IVF lists hold slots; `RefineIndex` finds its raw vector by slot). The id users see is the slot itself, or with `custom_ids` the id passed to `add_batch`; `Neighbor::index` always reports that id. `add_batch` must keep slots in input order: row `i` of a batch gets the next slot.

### `id_map.hpp` / `id_map.cpp`: `IdMap` (`vecengine::detail`)

Each index owns one. It holds `labels_` (slot → id, empty without custom ids), `slot_of_` (live id → slot, for `remove` and duplicate checks) and a deleted flag per slot.

- `check(n, ids)` validates an add without changing anything: ids present iff custom ids are on, each `>= 0` (`-1` is the padding id in batch results), no duplicates in the batch or among live ids, and at most `2^32 - 1` slots. `add` runs `check` and then commits.
- `remove(ids, removed_slots)` marks slots deleted and erases their ids from `slot_of_`, so a deleted id can be added again (it gets a new slot). Unknown and already deleted ids are skipped.
- `label(slot)` and `is_deleted(slot)` are inline: queries call them per candidate.

**Deletion is tombstoning.** Nothing is moved: moving vectors would change slots, which IVF lists and `RefineIndex` depend on. Queries skip deleted slots *during* the scan (filtering a finished top-k would return fewer than k), so deleted vectors still cost a distance computation and keep their memory until a future compaction.

Defaults in `src/index.cpp`: `add_batch` calls `add` per row; `query_batch` calls `query` for each row in parallel via `parallel_for`. This is safe because `query` is `const` and touches no shared mutable state, and each result goes to its own slot. Subclasses override `add_batch` only when a whole batch allows a faster path (`IVFIndex`, `RefineIndex`).

### `src/parallel.hpp`: `parallel_for` (`vecengine::detail`)

`parallel_for(n, fn)` calls `fn(i)` once for each `i` in `[0, n)` on up to `num_threads()` `std::thread`s, the caller being one of them. With one thread (or `n <= 1`) it is a plain loop on the caller: no atomics, no threads, exceptions propagate directly. Workers take the next index from a shared atomic counter, so uneven per-item cost (IVF list sizes, M-series performance vs efficiency cores) doesn't leave threads idle. The first exception thrown by `fn` stops the other workers and is rethrown on the caller after all join. Threads are created per call; at batch and training granularity that cost is negligible, so there is no pool. Don't nest it: `fn` must not call `parallel_for` itself.

**Thread count** (`include/vecengine/threads.hpp`, `src/threads.cpp`): `set_num_threads(n)` sets one process-wide count in a relaxed `std::atomic`, safe to change while other threads query. `0` (the default) means one thread per core; `num_threads()` resolves `0` to `hardware_concurrency()` (at least 1) on every read without writing it back. It does not cap the BLAS library's own threads inside `sgemm`; use `VECLIB_MAXIMUM_THREADS` / `OPENBLAS_NUM_THREADS` for those.

### `flat_index.hpp`: `FlatIndex`

Exact search. Vectors stored row-major in one `vector<float>`, one row per slot. `query` scans every live slot with a size-`k` max-heap, drains it back to front for ascending order and reports each slot's id. This is the ground truth for recall tests.

### `ivf_index.hpp`: `IVFIndex`

`IVFIndex(dim, nlist, nprobe = 10, custom_ids = false)`, with `nlist` in `[100, 65535]` (cluster IDs are `uint16_t`) and `nprobe` in `[1, nlist]`.

**Lifecycle:** `enable_pq(...)` (optional) → `train(sample)` → `add(v)` for each vector → `query`. `train` learns the model only; it stores no vectors. Training vectors are not searchable unless they are also added.

- **`train`**: one `kmeans` call for the `nlist` coarse centroids. With PQ on, computes residuals `v − centroid(assignment)` from `kmeans`'s assignments and trains the codebook on them.
- **`add`**: `nearest_centroid` picks the cluster; the vector's slot is appended to that cluster's `InvertedList`, plus either the raw floats (`vecs`) or the PQ code of its residual (`codes`).
- **`add_batch`**: three phases. (1) One `assign_nearest` call assigns every row. (2) With PQ on, `parallel_for` encodes each residual into its own slot of an `n * m` buffer. (3) A serial loop appends slots and data to the lists in input order, because `unordered_map` and list growth are not thread-safe and slots must follow input order. The ids are checked by `IdMap` before any of this, so a bad batch changes nothing. `assign_nearest` ranks by `‖c‖² − 2x·c`, so a near-tie can pick a different list than `add` would.
- **`query`**:
  1. Coarse: find the `nprobe` nearest coarse centroids.
  2. Fine: scan only those lists with a size-`k` max-heap. One ADC table buffer is allocated per query and reused for every list. Per list, all query-and-cluster-dependent work happens once, before the per-vector loop:
     - PQ off: `l2_distance` per stored vector.
     - ADC with precomputed tables (the default): table = `B[list] + C`, plus `A` on subspace 0's row → `distances_adc` over the list's codes. See below.
     - ADC without them: residual `q − centroid(c)` → `compute_adc_table` → `distances_adc`.
     - SDC: residual → `encode` → `distance_sdc` per code.

The residual depends on the cluster's centroid, so without precomputed tables it is rebuilt per probed list. Stored codes encode `x − c`, the query becomes `q − c`, and `(q − c) − (x − c) = q − x`, so residual distances equal true distances. SDC still rebuilds and encodes the residual per list, so it is slower than precomputed ADC as well as less accurate. ADC is the default.

**Precomputed tables** (FAISS-style). With `y` the PQ reconstruction of a stored residual, the ADC distance expands per subspace as `‖q − c − y‖² = ‖q − c‖² + (‖y‖² + 2⟨c, y⟩) − 2⟨q, y⟩ = A + B + C`:

| Term | Depends on | Built | Stored |
|---|---|---|---|
| `A = ‖q − c‖²` | query, list | coarse search already computes it | the coarse heap (read before `pop`) |
| `B[list][s][j] = ‖y_sj‖² + 2⟨c_s, y_sj⟩` | list | `build_precomputed()`, once per list, in parallel | `precomputed_`, `nlist * m * K` floats, list-major |
| `C[s][j] = −2⟨q_s, y_sj⟩` | query | `compute_inner_table`, once per query | a local `m * K` vector |

Per probed list the table is then `m * K` additions instead of a `sub_dim`-long loop per entry, and `A` is added to the `K` entries of subspace 0 only: each code picks exactly one entry per row, so every distance gets `A` once and `distances_adc` stays unchanged. Results match the per-list path up to float rounding (the tests allow `1e-3`).

`build_precomputed()` runs at the end of `train()`, after `load_body()` (the tables are derived, never saved), and from `set_precomputed_tables(true)`. It leaves `precomputed_` empty, and queries fall back to the per-list build, when PQ is off, the index is untrained, the setting is off, or the tables would exceed `kPrecomputedMaxBytes` (256 MB). SIFT1M with `nlist = 1000` and PQ16 needs 16 MB. `precomputed_tables()` reports whether the tables exist; `set_precomputed_tables(false)` frees them.

`query()` is `coarse()` (a size-`nprobe` heap over exact `l2_distance`s) followed by `scan()` (the fine step above). Each `Probe` carries its list id and its exact coarse distance, which is term `A` of the precomputed tables.

**`query_batch`** (with BLAS; without it, the `Index` default) replaces the per-query coarse loop. Queries go in blocks of up to 1024, capped so the score buffer (`block * nlist` floats) stays within 32 MB. Per block, `coarse_batch()`:
1. computes `‖c‖²` for every centroid,
2. makes one `cblas_sgemm` call for all `q·c` dot products in the block, on the calling thread so Accelerate/OpenBLAS can use their own threads without oversubscribing `parallel_for`,
3. per query, in `parallel_for`, ranks the lists by `‖c‖² − 2q·c` (`‖q‖²` is the same for every list) and keeps the `nprobe` lowest with `nth_element`,
4. recomputes the exact `‖q − c‖²` for those `nprobe` lists only. The sgemm score suffers cancellation, and `A` feeds straight into the returned distances, so it must match `query()`.

Then `scan()` runs per query in `parallel_for`. Like `assign_nearest`, a near-tie between two centroids' scores can pick a different list than `query()` would, so batch and single results can differ in rare ids; the tests allow 1%.

Every candidate goes through one `push` lambda that checks the `TopK` threshold first and then `is_deleted`, and pushes the slot's id. `nprobe` and `pq_distance` both have setters and can be changed at any time after training; they only affect queries.

### `refine_index.hpp`: `RefineIndex`

`RefineIndex(base, k_factor = 10, custom_ids = false)` wraps an approximate index with **no slots and no custom ids** by reference and keeps its own row-major copy of the raw vectors. User ids live in the `RefineIndex`'s `IdMap`; the base gets plain slots, so base slot `i` and refine slot `i` are always the same vector. (Checking `base.slots()` rather than `size()` matters: a base whose vectors were all deleted has size 0 but its slots are taken.)

A RefineIndex built this way holds `base` by reference (`Index& base_`); one created by `load_body` owns its base through `owned_base_`, declared before `base_` so it is initialized first, and binds `base_` to it via a private constructor. `base()` exposes the wrapped index either way, so a loaded index can still be tuned.

- `add` / `add_batch`: `id_map_.check(...)` first, then `base.add_batch` (no ids), then `id_map_.add(...)` and the raw floats, so slot `i` is at `data_[i * dim]`. Checking before touching the base keeps the two in step when either side rejects the add (bad ids, untrained base).
- `remove(ids)`: removes the ids from its own map, then removes the same slots from the base (whose ids are its slots), so the base stops returning them.
- `query(q, k)`: asks `base` for `min(k * k_factor, size())` candidates, replaces each approximate distance with the exact `l2_distance` against its raw vector, and returns the exact top `k` with ids from its own map. Returns nothing when every vector is deleted.

Changing the base directly would break that alignment (an add gives the base a vector the `RefineIndex` has no raw copy of; a remove hides a vector the `RefineIndex` still counts), so the constructor sets the base's `wrapped()` flag (an `std::atomic<bool>` on `Index`, cleared by `~RefineIndex`) and rejects a base that is already wrapped. The Python `add`/`remove` bindings check the flag under the index's write lock and raise `ValueError`. The C++ API doesn't enforce it, since `RefineIndex` itself calls `add_batch`/`remove` on its base.

`query_batch` makes one `base.query_batch` call for every query's candidates (so IVF's batched coarse search applies) and then re-ranks each query in `parallel_for`. Without it, the `Index` default would run `query()` per row and call `base.query()` one query at a time.

It only helps when `base` ranks with approximate distances (IVF + PQ). It restores recall but stores the raw vectors in RAM, so IVF + PQ + re-ranking uses more memory than plain IVF; in production the raw vectors would live on disk and only the candidates would be read. `k_factor` is a query-time setting; `k_factor = 1` returns the base's IDs with exact distances.

### `index_factory.hpp`: `make_index`

`make_index(IndexType, dim, IndexOptions)` returns `unique_ptr<Index>`. `IndexOptions` holds `capacity` (Flat), `nlist`, `nprobe`, `pq_subspaces` (0 = PQ off) and `pq_centroids` (IVF), and `custom_ids` (both). IVF-specific calls like `train()` require the concrete `IVFIndex`.

`load_index(std::istream&)` reads a file header and dispatches to `FlatIndex::load_body`, `IVFIndex::load_body` or `RefineIndex::load_body`. The path overloads wrap it for files:

- `save_index(index, path)` writes to `path + ".tmp"`, checks the stream after `close()` (write errors don't throw on their own), then renames into place; on any failure it removes the `.tmp` file and rethrows.
- `load_index(path)` opens in binary mode, rethrows `invalid_argument` from constructors and setters (bad `nlist`, `nprobe`, ...) as `runtime_error("corrupt file: ...")` so every bad file reports the same way, and rejects trailing bytes after the index.

---

## File Format (`src/serialize.hpp`)

Binary, little-endian (a `static_assert` requires a little-endian host), every size and count stored as `uint64`. Vectors are stored as a `uint64` element count followed by the raw elements (`write_vec` / `read_vec`).

Every index starts with a 9-byte header: magic `QANN` (`uint32`), format version (`uint32`, currently 2) and an `IndexKind` tag (`uint8`). `Index::save` writes header + body; `load_index` reads the header (`read_header` returns `{kind, version}`) and passes the version to the type's static `load_body`, which reads only the body. Versions `kMinFormatVersion` (1) through `kFormatVersion` (2) are readable.

| Kind | Body |
| --- | --- |
| `Flat` (1) | `dim`, `count` (slots), `data` (`count * dim` floats), id map |
| `IVF` (2) | `dim`, `nlist`, `nprobe`, `n_total` (slots), `trained` (u8), `pq_distance` (u8), coarse centroids (`nlist * dim`, or empty if untrained), PQ flag (u8) + `PQCodebook` if set, then `nlist` lists in cluster order `0..nlist-1`, each slot `ids` + `codes` (PQ) or `vecs` (no PQ), empty lists written as empty vectors; id map |
| `Refine` (3) | `k_factor`, `count` (slots), the base index (full header + body, same version), `data` (`count * dim` floats), id map |

The id map (`IdMap::save`, version 2 only) is `custom_ids` (u8), the slot count, `labels` (empty without custom ids) and one deleted flag (u8) per slot. It is the last section of each body, so a version-2 body is a version-1 body plus one section: for version 1, `load_body` skips it and starts from a map with every slot live and no custom ids.

`PQCodebook` (inside IVF, no header): `dim`, `num_subspaces`, `centroids_per_subspace`, `centroids_`. `centroids_t_` and `sdc_table_` are derived and rebuilt on load by `build_tables()`, the same code `train()` runs, so they come out bit-identical.

**Validation on load.** Files are untrusted input, so every loader checks before allocating or indexing: `read_vec` takes a `max_len` and callers then require the exact expected length; sizes are checked against overflow (`count > UINT64_MAX / dim`); IVF ids must be `< n_total` and list sizes must sum to `n_total`; the PQ codebook's `dim` must match the index; a Refine's `count` must equal its base's `slots()` (queries index the raw vectors by base slot), its base must have no custom ids and the same format version. A Refine base must be Flat or IVF, which also stops a crafted file from nesting Refines until the stack overflows. The id map must match its index's slot count, with no duplicate live ids. Any failure throws `runtime_error`.

**Changing the format.** `IndexKind` values are append-only: never renumber or reuse one. Bump `kFormatVersion` whenever an existing type's layout changes, and read older versions explicitly in each `load_body` (as version 2 does for version 1). `tests/data/` holds files saved by 0.1.0a4 with the results it returned; the C++ and Python tests load them to catch compatibility breaks.

---

## Benchmarks (`benchmarks/bench_indexes.py`)

Recall@10 (overlap of the returned top 10 with the true top 10), QPS on all cores (`query_batch`) and bytes per vector for Flat, IVF, IVF + PQ (ADC and SDC, m ∈ {8, 16, 32}) and IVF + PQ + re-ranking (`k_factor` ∈ {4, 16}), sweeping `nprobe` from 1 to 128. Training uses each dataset's `learn` set; `nlist ≈ √N`. Each configuration's QPS is the best of 3 full passes, so cold caches after a build don't distort the first measurement. Writes `benchmarks/results/<dataset>.csv` and a recall-vs-QPS plot.

```bash
PYTHONPATH=build python3 benchmarks/bench_indexes.py                    # siftsmall (10k)
PYTHONPATH=build python3 benchmarks/bench_indexes.py --check benchmarks/baseline_siftsmall.json
PYTHONPATH=build python3 benchmarks/bench_indexes.py --threads 1       # single-threaded (default 0 = all cores)
python3 benchmarks/compare.py [ref] [--dataset sift] [--rounds N]       # ref (default HEAD) vs working tree
PYTHONPATH=build python3 benchmarks/bench_indexes.py --dataset sift     # SIFT1M
PYTHONPATH=build python3 benchmarks/bench_indexes.py --max-queries 1000
```

- **Recall gate**: `--seed N --write-baseline FILE` records recall for every row; `--check FILE` reruns with the baseline's dataset and seed and exits 1 if any row drops more than `--tolerance` (default 0.02). `benchmarks/baseline_siftsmall.json` (seed 1) is the committed baseline. Regenerate it when a change is *meant* to move recall.
- **`compare.py`**: builds `ref` in a temporary copy (reusing `build/_deps` sources), runs each side's own `bench_indexes.py` alternately for `--rounds`, keeps the best QPS and fastest build per row, and prints recall, QPS ratios and build times. Speed is compared locally rather than in CI because shared CI runners vary 10 to 20% run to run.

Datasets come from `ftp://ftp.irisa.fr/local/texmex/corpus/` (`siftsmall.tar.gz`, `sift.tar.gz`), extracted into `benchmarks/data/`.

**SIFT1M findings** (Apple M3, 4 performance + 4 efficiency cores, `nlist = 1000`, 10k queries, seed 1, idle machine):

Thread scaling at `nprobe = 16` (`set_num_threads`):

| threads | Plain IVF (recall 0.93) | IVF + PQ16 (0.56) | IVF + PQ16 + R16 (0.93) |
|---|---|---|---|
| 1 | 3.3k QPS | 5.7k | 3.6k |
| 2 | 6.4k (1.9×) | 11.0k (1.9×) | 7.4k (2.0×) |
| 4 | 7.6k (2.3×) | 19.2k (3.4×) | 13.3k (3.7×) |
| 8 | 10.6k (3.2×) | 24.5k (4.3×) | 18.0k (4.9×) |

- Plain IVF stalls between 2 and 4 threads while PQ keeps scaling: IVF streams 512-byte raw vectors and is limited by memory bandwidth, PQ scans 16-byte codes. The efficiency cores add 25 to 40% on top of 4 threads.
- On all cores, IVF + PQ16 + re-ranking is 1.7× faster than plain IVF at the same 0.93 recall. Single-threaded, plain IVF was faster; PQ's smaller scan wins once bandwidth is the bottleneck.
- Plain IVF on all cores: 10.2k QPS at 0.93 recall (`nprobe = 16`), 5.3k at 0.98 (`nprobe = 32`), 2.6k at 0.995 (`nprobe = 64`); Flat is 196 QPS.
- PQ alone plateaus at 0.38 / 0.57 / 0.73 recall for m = 8 / 16 / 32; re-ranking lifts m = 16 to 0.99 and m = 32 to 0.999.
- SDC recall *falls* as `nprobe` grows, because more candidates expose its query-side quantization error.

**Build times** on all cores: IVF trains in 2.5 s and adds 1M vectors in 1.3 s (`add_batch`); IVF + PQ16 trains in 3.4 s (8.6 s on one thread) and adds in 3.0 s. Before `assign_nearest`, IVF training took 37 s and IVF + PQ16 55 s.

**Precomputed tables** (`profile_query`, `nprobe = 16`, 1 thread, both paths in one run): the ADC table build drops from 37.8 to 9.0 µs per query, taking IVF + PQ16 from 131 to 108 µs (+21% QPS). The code scan (~70 µs) is now most of a PQ query.

**Batched coarse search** (`profile_query`, which uses `query_batch`, 1 thread, `VECLIB_MAXIMUM_THREADS=1`): coarse search drops from about 19 µs to 7.5 µs per query. Together with precomputed tables, IVF + PQ16 goes from 131 to 103 µs per query and IVF + PQ16 + R16 from 188 to 149 µs. The machine was loaded (load average 6), so treat the totals as approximate; the coarse section is the reliable comparison. The thread-scaling table above predates both changes.

The PQ rows of `benchmarks/results/sift.csv` from this run are ~20% low: other jobs loaded the machine during its second half. The thread-scaling numbers above come from a separate clean run.

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

Ids: `add(data, ids=None)` and `remove(ids)` are bound once on `Index`. `ids` accepts any sequence of integers: `as_ids` runs `numpy.asarray`, rejects non-integer and non-1D input with `TypeError`, then converts to contiguous `int64`; the stubs show it as `Iterable[int]` (`nb::typed<nb::iterable, int>`). Constructors take `custom_ids` as a keyword-only argument.

Save/load: `save(path)` is bound once on `Index` and inherited by every index class; `qann.load(path)` binds the path overload of `load_index` and returns the concrete type. Paths go through nanobind's `std::filesystem::path` caster, so `str` and `pathlib.Path` both work. `RefineIndex.base` is a read-only property returned with `rv_policy::reference_internal`, which keeps the RefineIndex alive while Python holds its base (needed for a loaded RefineIndex, which owns its base).

**GIL and locking.** Every `Index` carries a `std::shared_mutex` (`Index::mutex()`), unused by the C++ library itself. The bindings run index work through `read_locked` (shared: `query`, `batch_query`, `size`, `save`, getters) or `write_locked` (exclusive: `add`, `train`, `enable_pq`, setters, and `RefineIndex.__init__`, which checks its base is empty). Both release the GIL *before* taking the lock, so a thread waiting for a lock never holds the GIL the lock holder needs to finish; the work inside must not touch Python objects, so NumPy results are built after the lock and GIL are released and reacquired. `lock_chain` locks a `RefineIndex` and then each base under it, always outermost first, because a base is also reachable from Python (e.g. `ivf.add()` while queries run through a `RefineIndex` over `ivf`). `load` only releases the GIL (`call_guard`), since the index it creates isn't shared yet. `PQCodebook` methods keep the GIL.

**Type stubs.** `nanobind_add_stub` (CMake target `qann_stub`) imports the built module and writes `build/qann.pyi`, so stubs always match the bindings. Wheels install it as `qann-stubs/__init__.pyi`: type checkers read installed stubs only from packages (PEP 561), and `qann` itself is a single extension module. The shared `add`/`query`/`batch_query`/`size`/`dim` methods are bound once on `Index`, so the `Index` returned by `load` and `make_index` type-checks with them.

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
- CI (`.github/workflows/ci.yml`, on pushes to `main` and on PRs): builds and runs `ctest` on `ubuntu-latest` (with OpenBLAS) and `macos-14`, then runs the siftsmall recall gate on macOS only, since the baseline was recorded with Accelerate. siftsmall is cached between runs.

---

## Not Yet Implemented

- **HNSW**: graph index; slot reserved in `IndexType`.
