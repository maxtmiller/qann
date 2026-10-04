# vecengine: Architecture & File Reference

## Overview

`vecengine` is a C++20 vector search library with a Python extension module built on nanobind. It has three layers:

1. **Distance kernels**: squared L2 and cosine distance with scalar, AVX2+FMA and ARM NEON paths, selected at compile time.
2. **Quantization and clustering**: a shared k-means helper and a Product Quantization (PQ) codebook.
3. **Indexes**: `FlatIndex` (exact brute force) and `IVFIndex` (inverted file, optionally PQ-compressed), behind a common `Index` interface.

All arithmetic lives in C++. Python gets zero-copy access to NumPy arrays via `nb::ndarray`.

---

## Directory Structure

```
vecengine/
├── CMakeLists.txt                  # build orchestrator
├── build.sh                        # configure + build + test in one command
├── setup.py                        # legacy pip/scikit-build-core shim
├── include/vecengine/              # public API
│   ├── aligned_allocator.hpp       # std::allocator with 32/64-byte alignment
│   ├── distance.hpp                # distance function declarations
│   ├── distance_inl.hpp            # inline compile-time SIMD dispatch
│   ├── index.hpp                   # Index interface + Neighbor
│   ├── flat_index.hpp              # exact brute-force index
│   ├── ivf_index.hpp               # IVF index, optional PQ
│   ├── pq.hpp                      # PQCodebook + PQDistance
│   └── index_factory.hpp           # make_index() + IndexOptions
├── src/
│   ├── distance.cpp                # scalar / AVX2 / NEON kernels
│   ├── kmeans.hpp / kmeans.cpp     # internal k-means helper (not public API)
│   ├── pq.cpp
│   ├── flat_index.cpp
│   ├── ivf_index.cpp
│   └── index_factory.cpp
├── tests/test_distance.cpp         # Catch2 v3 tests for every layer
├── benchmarks/bench_distance.cpp   # Google Benchmark distance kernels
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

Cosine returns `1.0f` for zero-magnitude inputs rather than dividing by zero.

### `aligned_allocator.hpp`

`AlignedAllocator<T, Align>` uses `posix_memalign` (or `_aligned_malloc` on MSVC). Aliases: `AVX2Vector<T>` (32-byte) and `AVX512Vector<T>` (64-byte). Used by the benchmarks; the indexes use plain `std::vector` and unaligned loads.

---

## Layer 2: Clustering and Quantization

### `src/kmeans.hpp` / `kmeans.cpp` (`vecengine::detail`)

Shared by `IVFIndex` and `PQCodebook`, so Lloyd's algorithm exists exactly once.

```cpp
KMeansResult kmeans(const float* data, size_t n, size_t stride, size_t d, size_t k, size_t max_iters);
size_t nearest_centroid(const float* point, const float* centroids, size_t k, size_t d);
```

- Point `i` is the `d` floats at `data + i * stride`. `stride` and `d` are separate so PQ can cluster one subspace slice of every row in place: pass `data + s * sub_dim`, `stride = dim`, `d = sub_dim`. IVF passes `stride = d = dim`.
- Returned centroids are packed (`k * d`, centroid `c` at `c * d`), never strided.
- Init picks `k` distinct random points (partial Fisher-Yates). An empty cluster is reseeded with the point furthest from its assigned centroid.
- A final assignment pass runs after the last update, so `assignments` match the returned `centroids`. IVF relies on this to compute PQ training residuals.
- `nearest_centroid` is the brute-force scan used by the k-means assignment step, `IVFIndex::add` and `PQCodebook::encode`.

Seeded from `std::random_device`, so training is non-deterministic run to run.

### `pq.hpp` / `pq.cpp`: `PQCodebook`

Splits a `dim`-vector into `m = num_subspaces` slices of `sub_dim = dim / m` floats and quantizes each slice to one of `K = centroids_per_subspace` (max 256) centroids. A vector becomes `m` bytes: at `dim = 128`, `m = 16` that is 512 bytes down to 16.

| Member | Layout | Built |
|---|---|---|
| `centroids_` | `m * K * sub_dim`, grouped by subspace: centroid `c` of subspace `s` at `(s * K + c) * sub_dim` | `train()` |
| `sdc_table_` | `m * K * K`: distance between centroids `c1`, `c2` of subspace `s` at `s*K*K + c1*K + c2` | `train()` |

- `train(vectors, n)`: one `kmeans` call per subspace, then fills `sdc_table_` (upper triangle computed, mirrored; diagonal 0).
- `encode(vec)`: per subspace, `nearest_centroid` of the slice. Returns `m` codes.
- `compute_adc_table(query)` + `distance_adc(table, code)`: **ADC**. The table holds `m * K` distances from each query slice to every centroid (`table[s * K + c]`); the distance is `m` lookups. Build the table once and reuse it across many codes.
- `distance_sdc(query_code, code)`: **SDC**. Both sides are codes; `m` lookups into `sdc_table_`, no float math.

Both work because squared L2 decomposes across disjoint slices: `‖a − b‖² = Σₛ ‖aₛ − bₛ‖²`. ADC keeps the query exact and is more accurate; SDC quantizes both sides.

`PQDistance { ADC, SDC }` lives here and selects the mode in `IVFIndex`.

---

## Layer 3: Indexes

### `index.hpp`: `Index`

```cpp
virtual void add(span<const float> vec) = 0;
virtual vector<Neighbor> query(span<const float> vec, size_t k) const = 0;
virtual vector<vector<Neighbor>> query_batch(span<const float> queries, size_t num_queries, size_t k) const = 0;
virtual size_t size() const noexcept = 0;
virtual size_t dim() const noexcept = 0;
```

`Neighbor { size_t index; float distance; }`. IDs are assigned in insertion order starting at 0. Results are sorted nearest first.

### `flat_index.hpp`: `FlatIndex`

Exact search. Vectors stored row-major in one `vector<float>`. `query` scans all of them with a size-`k` max-heap and drains it back to front for ascending order. This is the ground truth for recall tests.

### `ivf_index.hpp`: `IVFIndex`

`IVFIndex(dim, nlist, nprobe = 10)`, with `nlist` in `[100, 65535]` (cluster IDs are `uint16_t`).

**Lifecycle:** `enable_pq(...)` (optional) → `train(sample)` → `add(v)` for each vector → `query`. `train` learns the model only; it stores no vectors. Training vectors are not searchable unless they are also added.

- **`train`**: one `kmeans` call for the `nlist` coarse centroids. With PQ on, computes residuals `v − centroid(assignment)` from `kmeans`'s assignments and trains the codebook on them.
- **`add`**: `nearest_centroid` picks the cluster; the vector's ID is appended to that cluster's `InvertedList`, plus either the raw floats (`vecs`) or the PQ code of its residual (`codes`).
- **`query`**:
  1. Coarse: find the `nprobe` nearest coarse centroids.
  2. Fine: scan only those lists with a size-`k` max-heap. Per list, all query-and-cluster-dependent work happens once, before the per-vector loop:
     - PQ off: `l2_distance` per stored vector.
     - ADC: residual `q − centroid(c)` → `compute_adc_table` → `distance_adc` per code.
     - SDC: residual → `encode` → `distance_sdc` per code.

The residual is rebuilt per probed list because it depends on the cluster's centroid. Stored codes encode `x − c`, the query becomes `q − c`, and `(q − c) − (x − c) = q − x`, so residual distances equal true distances. A consequence: inside IVF, SDC costs about the same as ADC (both need `m * K` distance computations per list) and is less accurate. ADC is the default.

`set_pq_distance(PQDistance)` switches modes at any time; it only affects queries.

**Measured recall@10 vs `FlatIndex`** (5000 uniform random 32-d vectors, `nlist = 100`, `nprobe = 20`, `m = 8`, `K = 256`): plain IVF ~0.80, ADC ~0.63, SDC ~0.48. Uniform random data is the worst case for PQ; real embeddings compress better.

### `index_factory.hpp`: `make_index`

`make_index(IndexType, dim, IndexOptions)` returns `unique_ptr<Index>`. `IndexOptions` holds `capacity` (Flat), `nlist`, `nprobe`, `pq_subspaces` (0 = PQ off) and `pq_centroids` (IVF). IVF-specific calls like `train()` require the concrete `IVFIndex`.

---

## Python Bindings (`bindings/python_bindings.cpp`)

Inputs are float32 C-contiguous NumPy arrays, passed zero-copy as `nb::ndarray` (`FloatMatrix` 2D, `FloatVector` 1D, `ByteVector` uint8 1D). Wrappers validate shapes and raise `ValueError` so mismatched input never reaches C++ in Release builds, where asserts are off.

```python
import numpy as np, vecengine as ve

x = np.random.rand(10_000, 128).astype(np.float32)

ivf = ve.IVFIndex(128, nlist=100, nprobe=10)
ivf.enable_pq(16)                      # optional; before train/add
ivf.train(x)
ivf.add(x)
ivf.pq_distance = ve.PQDistance.SDC    # default ADC
ids, dists = ivf.query(x[:1], 10)
batch = ivf.batch_query(x[:5], 10)     # list of (ids, dists)

pq = ve.PQCodebook(128, 16)
pq.train(x)
code = np.asarray(pq.encode(x[0]), dtype=np.uint8)
table = np.asarray(pq.compute_adc_table(x[1]), dtype=np.float32)
pq.distance_adc(table, code)
```

Also exposed: `FlatIndex`, `IndexType`, `IndexOptions`, `make_index`, and row-wise `l2_distance(a, b)` / `cosine_distance(a, b)`.

`encode` and `compute_adc_table` return Python lists; wrap them with `np.asarray` before passing them back.

---

## Build, Test, Benchmark

```bash
./build.sh                  # Release build in ./build/, then ctest
./build.sh build Debug      # Debug build (asserts on)
./build/test_vecengine "[pq]"   # run one tag: [l2] [cosine] [index] [ivf] [pq] [adc] [sdc] ...
./build/bench_vecengine
```

- Dependencies come from FetchContent: Catch2 v3.5.3, Google Benchmark v1.8.3, nanobind v1.9.2.
- `-mavx2 -mfma` are only added on x86 so arm64 builds never see unsupported flags.
- Google Benchmark's own tests are disabled via cache variables to avoid its stale GoogleTest download.
- `CMAKE_EXPORT_COMPILE_COMMANDS=ON` writes `build/compile_commands.json` for clangd.
- Targets: `vecengine_core` (static lib), `test_vecengine`, `bench_vecengine`, `vecengine_py` (Python module).

---

## Not Yet Implemented

- **HNSW**: graph index; slot reserved in `IndexType`.
- **Index benchmarks**: recall vs QPS for Flat / IVF / IVF+PQ (benchmarks cover distance kernels only).
- **Batched query parallelism**: `query_batch` runs queries sequentially.
- **NumPy returns**: `encode`, `compute_adc_table` and query results come back as Python lists.
- **Persistence**: no save/load of trained indexes.
- **`pyproject.toml`**: PEP 517 packaging; `setup.py` is a legacy shim.
