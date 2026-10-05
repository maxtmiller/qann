# vecengine: Architecture & File Reference

## Overview

`vecengine` is a C++20 vector search library with a Python extension module built on nanobind. It has three layers:

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
│   └── index_factory.hpp           # make_index() + IndexOptions
├── src/
│   ├── distance.cpp                # scalar / AVX2 / NEON kernels
│   ├── kmeans.hpp / kmeans.cpp     # internal k-means helper (not public API)
│   ├── parallel.hpp                # internal parallel_for (not public API)
│   ├── index.cpp                   # default Index::add_batch / query_batch
│   ├── pq.cpp
│   ├── flat_index.cpp
│   ├── ivf_index.cpp
│   ├── refine_index.cpp
│   └── index_factory.cpp
├── tests/test_distance.cpp         # Catch2 v3 tests for every layer
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
virtual void add_batch(span<const float> vecs, size_t n);
virtual vector<Neighbor> query(span<const float> vec, size_t k) const = 0;
virtual vector<vector<Neighbor>> query_batch(span<const float> queries, size_t num_queries, size_t k) const;
virtual size_t size() const noexcept = 0;
virtual size_t dim() const noexcept = 0;
```

`Neighbor { size_t index; float distance; }`. Results are sorted nearest first.

**Invariant:** `add()` assigns IDs in insertion order starting at 0, and `Neighbor::index` is that ID. `RefineIndex` depends on this. `add_batch` must keep it: row `i` of a batch gets the next ID in order.

Defaults in `src/index.cpp`: `add_batch` calls `add` per row; `query_batch` calls `query` for each row in parallel via `parallel_for`. This is safe because `query` is `const` and touches no shared mutable state, and each result goes to its own slot. Subclasses override `add_batch` only when a whole batch allows a faster path (`IVFIndex`, `RefineIndex`).

### `src/parallel.hpp`: `parallel_for` (`vecengine::detail`)

`parallel_for(n, fn)` calls `fn(i)` once for each `i` in `[0, n)` on up to `hardware_concurrency()` `std::thread`s, the caller being one of them. Workers take the next index from a shared atomic counter, so uneven per-item cost (IVF list sizes, M-series performance vs efficiency cores) doesn't leave threads idle. The first exception thrown by `fn` stops the other workers and is rethrown on the caller after all join. Threads are created per call; at batch and training granularity that cost is negligible, so there is no pool. Don't nest it: `fn` must not call `parallel_for` itself.

### `flat_index.hpp`: `FlatIndex`

Exact search. Vectors stored row-major in one `vector<float>`. `query` scans all of them with a size-`k` max-heap and drains it back to front for ascending order. This is the ground truth for recall tests.

### `ivf_index.hpp`: `IVFIndex`

`IVFIndex(dim, nlist, nprobe = 10)`, with `nlist` in `[100, 65535]` (cluster IDs are `uint16_t`).

**Lifecycle:** `enable_pq(...)` (optional) → `train(sample)` → `add(v)` for each vector → `query`. `train` learns the model only; it stores no vectors. Training vectors are not searchable unless they are also added.

- **`train`**: one `kmeans` call for the `nlist` coarse centroids. With PQ on, computes residuals `v − centroid(assignment)` from `kmeans`'s assignments and trains the codebook on them.
- **`add`**: `nearest_centroid` picks the cluster; the vector's ID is appended to that cluster's `InvertedList`, plus either the raw floats (`vecs`) or the PQ code of its residual (`codes`).
- **`add_batch`**: three phases. (1) One `assign_nearest` call assigns every row. (2) With PQ on, `parallel_for` encodes each residual into its own slot of an `n * m` buffer. (3) A serial loop appends IDs and data to the lists in input order, because `unordered_map` and list growth are not thread-safe and the ID invariant needs input order. `assign_nearest` ranks by `‖c‖² − 2x·c`, so a near-tie can pick a different list than `add` would.
- **`query`**:
  1. Coarse: find the `nprobe` nearest coarse centroids.
  2. Fine: scan only those lists with a size-`k` max-heap. One ADC table buffer is allocated per query and reused for every list. Per list, all query-and-cluster-dependent work happens once, before the per-vector loop:
     - PQ off: `l2_distance` per stored vector.
     - ADC: residual `q − centroid(c)` → `compute_adc_table` → `distance_adc` per code.
     - SDC: residual → `encode` → `distance_sdc` per code.

The residual is rebuilt per probed list because it depends on the cluster's centroid. Stored codes encode `x − c`, the query becomes `q − c`, and `(q − c) − (x − c) = q − x`, so residual distances equal true distances. Inside IVF, SDC costs about the same as ADC (both need a full pass over the centroids per list) and is less accurate. ADC is the default.

`nprobe` and `pq_distance` both have setters and can be changed at any time after training; they only affect queries.

### `refine_index.hpp`: `RefineIndex`

`RefineIndex(base, k_factor = 10)` wraps an **empty, trained** approximate index by reference and keeps its own row-major copy of the raw vectors.

- `add(v)`: forwards to `base`, then appends the raw floats, so ID `i` is at `data_[i * dim]`.
- `add_batch(vecs, n)`: forwards to `base.add_batch` and appends all rows at once.
- `query(q, k)`: asks `base` for `min(k * k_factor, size())` candidates, replaces each approximate distance with the exact `l2_distance` against its raw vector, and returns the exact top `k`.

It only helps when `base` ranks with approximate distances (IVF + PQ). It restores recall but stores the raw vectors in RAM, so IVF + PQ + re-ranking uses more memory than plain IVF; in production the raw vectors would live on disk and only the candidates would be read. `k_factor` is a query-time setting; `k_factor = 1` returns the base's IDs with exact distances.

### `index_factory.hpp`: `make_index`

`make_index(IndexType, dim, IndexOptions)` returns `unique_ptr<Index>`. `IndexOptions` holds `capacity` (Flat), `nlist`, `nprobe`, `pq_subspaces` (0 = PQ off) and `pq_centroids` (IVF). IVF-specific calls like `train()` require the concrete `IVFIndex`.

---

## Benchmarks (`benchmarks/bench_indexes.py`)

Recall@10 (overlap of the returned top 10 with the true top 10), QPS on all cores (`query_batch`) and bytes per vector for Flat, IVF, IVF + PQ (ADC and SDC, m ∈ {8, 16, 32}) and IVF + PQ + re-ranking (`k_factor` ∈ {4, 16}), sweeping `nprobe` from 1 to 128. Training uses each dataset's `learn` set; `nlist ≈ √N`. Each configuration's QPS is the best of 3 full passes, so cold caches after a build don't distort the first measurement. Writes `benchmarks/results/<dataset>.csv` and a recall-vs-QPS plot.

```bash
PYTHONPATH=build python3 benchmarks/bench_indexes.py                    # siftsmall (10k)
PYTHONPATH=build python3 benchmarks/bench_indexes.py --check benchmarks/baseline_siftsmall.json
python3 benchmarks/compare.py [ref] [--dataset sift] [--rounds N]       # ref (default HEAD) vs working tree
PYTHONPATH=build python3 benchmarks/bench_indexes.py --dataset sift     # SIFT1M
PYTHONPATH=build python3 benchmarks/bench_indexes.py --max-queries 1000
```

- **Recall gate**: `--seed N --write-baseline FILE` records recall for every row; `--check FILE` reruns with the baseline's dataset and seed and exits 1 if any row drops more than `--tolerance` (default 0.02). `benchmarks/baseline_siftsmall.json` (seed 1) is the committed baseline. Regenerate it when a change is *meant* to move recall.
- **`compare.py`**: builds `ref` in a temporary copy (reusing `build/_deps` sources), runs each side's own `bench_indexes.py` alternately for `--rounds`, keeps the best QPS and fastest build per row, and prints recall, QPS ratios and build times. Speed is compared locally rather than in CI because shared CI runners vary 10 to 20% run to run.

Datasets come from `ftp://ftp.irisa.fr/local/texmex/corpus/` (`siftsmall.tar.gz`, `sift.tar.gz`), extracted into `benchmarks/data/`.

**SIFT1M findings** (Apple M3, single thread, before the ADC table and k-means speedups):

| recall@10 | Plain IVF | Best IVF + PQ + re-ranking |
|---|---|---|
| ~0.92 | 3.3k QPS (`nprobe = 16`) | 3.1k (m = 16, R16) |
| ~0.97 | 1.76k (`nprobe = 32`) | 1.79k (m = 16, R16) |

- Plain IVF reaches 0.98 recall at 25× Flat's QPS.
- PQ alone plateaus at 0.38 / 0.56 / 0.72 recall for m = 8 / 16 / 32; re-ranking lifts m = 16 to 0.99 and m = 32 to 0.999.
- SDC recall *falls* as `nprobe` grows, because more candidates expose its query-side quantization error.
- Without re-ranking, PQ's advantage is memory (12 to 36 bytes per vector vs 516), not speed.

**Training times** on SIFT1M after `assign_nearest`: IVF 37 s → 3.3 s, IVF + PQ16 55 s → 14.7 s.

---

## Python Bindings (`bindings/python_bindings.cpp`)

Inputs are float32 C-contiguous NumPy arrays, passed zero-copy as `nb::ndarray` (`FloatMatrix` 2D, `FloatVector` 1D, `ByteVector` uint8 1D). Wrappers validate shapes and raise `ValueError` so mismatched input never reaches C++ in Release builds, where asserts are off.

```python
import numpy as np, vecengine as ve

x = np.random.rand(10_000, 128).astype(np.float32)

ivf = ve.IVFIndex(128, nlist=100, nprobe=10)
ivf.enable_pq(16)                      # optional; before train/add
ivf.train(x)
refine = ve.RefineIndex(ivf, k_factor=10)   # optional; wrap before adding
refine.add(x)                          # adds to ivf too
ivf.nprobe = 32                        # tune after training
ivf.pq_distance = ve.PQDistance.ADC    # default ADC
ids, dists = refine.query(x[:1], 10)
ids, dists = refine.batch_query(x[:5], 10)  # (5, 10) arrays

pq = ve.PQCodebook(128, 16)
pq.train(x)
code = pq.encode(x[0])                 # (16,) uint8
table = pq.compute_adc_table(x[1])     # (16 * 256,) float32
pq.distance_adc(table, code)
```

`RefineIndex` holds its base by reference; `keep_alive` keeps the Python base object alive as long as the wrapper exists.

Also exposed: `FlatIndex`, `IndexType`, `IndexOptions`, `make_index`, and row-wise `l2_distance(a, b)` / `cosine_distance(a, b)`.

Results are NumPy arrays that take ownership of the C++ buffer (no copy): IDs are `int64`, distances `float32`, codes `uint8`. `query` returns 1D arrays with one entry per result. `batch_query` returns `(num_queries, k)` arrays; a row with fewer than `k` results is padded with ID `-1` and distance `inf`.

---

## Build, Test, Benchmark

```bash
pip install .               # builds the Python module into a wheel (no C++ tests)
./build.sh                  # Release build in ./build/, then ctest
./build.sh build Debug      # Debug build (asserts on)
./build/test_vecengine "[pq]"   # run one tag: [l2] [cosine] [index] [ivf] [pq] [adc] [sdc] [refine] ...
./build/bench_vecengine
```

- Dependencies come from FetchContent: Catch2 v3.5.3, Google Benchmark v1.8.3, nanobind v1.9.2.
- BLAS for k-means (`VECENGINE_USE_BLAS`): on Apple, `vecengine_core` links `-framework Accelerate` (PUBLIC, so tests and the Python module link it too). Elsewhere `find_package(BLAS)` plus a `cblas.h` search (also under `include/openblas`); on Debian/Ubuntu `apt install libopenblas-dev`. Configure prints which one it used, or that it fell back to the per-point scan. OpenBLAS runs its own threads inside `sgemm`; if PQ training (which calls it from every `parallel_for` worker) oversubscribes, set `OPENBLAS_NUM_THREADS=1`.
- Packaging: `pyproject.toml` uses scikit-build-core, builds only `vecengine_py` with `VECENGINE_BUILD_TESTS=OFF` (skips fetching Catch2 and Google Benchmark), and installs the module at the wheel root. Runtime dependency: NumPy.
- `-mavx2 -mfma` are only added on x86 so arm64 builds never see unsupported flags.
- Google Benchmark's own tests are disabled via cache variables to avoid its stale GoogleTest download.
- Link-time optimization was tried and left off: no query speedup, and k-means training got ~35% slower.
- `CMAKE_EXPORT_COMPILE_COMMANDS=ON` writes `build/compile_commands.json` for clangd.
- Targets: `vecengine_core` (static lib), `test_vecengine`, `bench_vecengine`, `vecengine_py` (Python module).
- CI (`.github/workflows/ci.yml`, on pushes to `main` and on PRs): builds and runs `ctest` on `ubuntu-latest` (with OpenBLAS) and `macos-14`, then runs the siftsmall recall gate on macOS only, since the baseline was recorded with Accelerate. siftsmall is cached between runs.

---

## Not Yet Implemented

- **Thread count setting**: `parallel_for` always uses every core; there is no knob for single-thread measurements or limiting CPU use.
- **IVF-PQ precomputed tables**: FAISS-style decomposition to avoid building an ADC table per probed list.
- **HNSW**: graph index; slot reserved in `IndexType`.
- **Persistence**: no save/load of trained indexes.
