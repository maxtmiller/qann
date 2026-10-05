#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>

#include <cstdint>
#include <limits>
#include <span>

#include "vecengine/distance.hpp"
#include "vecengine/flat_index.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/pq.hpp"
#include "vecengine/refine_index.hpp"
#include "vecengine/index_factory.hpp"
#include "vecengine/threads.hpp"

namespace nb = nanobind;

// 2D row-major float32 NumPy array, zero-copy via nanobind::ndarray.
using FloatMatrix = nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using FloatVector = nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using ByteVector = nb::ndarray<uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

static std::span<const float> as_span(FloatVector v) { return {v.data(), v.shape(0)}; }
static std::span<const uint8_t> as_span(ByteVector v) { return {v.data(), v.shape(0)}; }

template <class T>
using NumpyArray = nb::ndarray<nb::numpy, T>;

// Moves v into a NumPy array of the given shape without copying; the
// capsule frees the vector when Python drops the array.
template <class T>
static NumpyArray<T> to_numpy(std::vector<T> v, std::initializer_list<std::size_t> shape) {
    auto* owned = new std::vector<T>(std::move(v));
    nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return NumpyArray<T>(owned->data(), shape, owner);
}

// Compute pairwise L2 distances between rows of two matrices (M x D each).
// Returns a length-M array of squared L2 distances.
static NumpyArray<float> batch_l2(FloatMatrix a, FloatMatrix b) {
    if (a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1))
        throw std::invalid_argument("a and b must have the same shape");

    const std::size_t M = a.shape(0);
    const std::size_t D = a.shape(1);
    std::vector<float> out(M);

    for (std::size_t i = 0; i < M; ++i)
        out[i] = vecengine::l2_distance(a.data() + i * D, b.data() + i * D, D);

    return to_numpy(std::move(out), {M});
}

// Compute pairwise cosine distances between rows of two matrices (M x D each).
static NumpyArray<float> batch_cosine(FloatMatrix a, FloatMatrix b) {
    if (a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1))
        throw std::invalid_argument("a and b must have the same shape");

    const std::size_t M = a.shape(0);
    const std::size_t D = a.shape(1);
    std::vector<float> out(M);

    for (std::size_t i = 0; i < M; ++i)
        out[i] = vecengine::cosine_distance(a.data() + i * D, b.data() + i * D, D);

    return to_numpy(std::move(out), {M});
}

// Add all rows of a (N x dim) matrix to the index.
static void index_add(vecengine::Index &self, FloatMatrix data) {
    if (data.shape(1) != self.dim())
        throw std::invalid_argument("vector dimension does not match index dimension");

    self.add_batch(std::span<const float>(data.data(), data.size()), data.shape(0));
}

// Query k-NN for a single vector; returns (indices, distances) as 1D arrays
// with one entry per result (int64, float32).
static std::pair<NumpyArray<int64_t>, NumpyArray<float>>
index_query(const vecengine::Index &self, FloatVector query, std::size_t k) {
    if (query.shape(0) != self.dim())
        throw std::invalid_argument("query dimension does not match index dimension");

    auto results = self.query(as_span(query), k);

    const std::size_t n = results.size();
    std::vector<int64_t> indices(n);
    std::vector<float> distances(n);
    for (std::size_t i = 0; i < n; ++i) {
        indices[i] = static_cast<int64_t>(results[i].index);
        distances[i] = results[i].distance;
    }
    return {to_numpy(std::move(indices), {n}), to_numpy(std::move(distances), {n})};
}

// Query k-NN for each row; returns (indices, distances) as (num_queries, k)
// arrays. Rows with fewer than k results are padded with -1 / inf.
static std::pair<NumpyArray<int64_t>, NumpyArray<float>>
index_batch_query(const vecengine::Index &self, FloatMatrix query, std::size_t k) {
    if (query.shape(1) != self.dim())
        throw std::invalid_argument("query dimension does not match index dimension");

    const std::size_t num_queries = query.shape(0);
    auto results = self.query_batch(std::span<const float>(query.data(), query.size()), num_queries, k);

    std::vector<int64_t> indices(num_queries * k, -1);
    std::vector<float> distances(num_queries * k, std::numeric_limits<float>::infinity());
    for (std::size_t i = 0; i < num_queries; ++i) {
        for (std::size_t j = 0; j < results[i].size(); ++j) {
            indices[i * k + j] = static_cast<int64_t>(results[i][j].index);
            distances[i * k + j] = results[i][j].distance;
        }
    }
    return {to_numpy(std::move(indices), {num_queries, k}), to_numpy(std::move(distances), {num_queries, k})};
}

// Train an IVFIndex's coarse centroids from a (N x dim) matrix of sample vectors.
static void ivf_train(vecengine::IVFIndex &self, FloatMatrix data, std::size_t max_iters, std::optional<uint32_t> seed) {
    if (data.shape(1) != self.dim())
        throw std::invalid_argument("training vector dimension does not match index dimension");
    self.train(std::span<const float>(data.data(), data.size()), data.shape(0), max_iters, seed);
}

static void pq_train(vecengine::PQCodebook &self, FloatMatrix data, std::size_t max_iters, std::optional<uint32_t> seed) {
    if (data.shape(1) != self.dim())
        throw std::invalid_argument("training vector dimension does not match codebook dimension");
    self.train(std::span<const float>(data.data(), data.size()), data.shape(0), max_iters, seed);
}

static NumpyArray<uint8_t> pq_encode(const vecengine::PQCodebook &self, FloatVector vec) {
    if (vec.shape(0) != self.dim())
        throw std::invalid_argument("vector dimension does not match codebook dimension");
    return to_numpy(self.encode(as_span(vec)), {self.num_subspaces()});
}

static NumpyArray<float> pq_compute_adc_table(const vecengine::PQCodebook &self, FloatVector query) {
    if (query.shape(0) != self.dim())
        throw std::invalid_argument("query dimension does not match codebook dimension");
    const std::size_t size = self.num_subspaces() * self.centroids_per_subspace();
    std::vector<float> table(size);
    self.compute_adc_table(as_span(query), table);
    return to_numpy(std::move(table), {size});
}

static float pq_distance_adc(const vecengine::PQCodebook &self, FloatVector table, ByteVector code) {
    if (table.shape(0) != self.num_subspaces() * self.centroids_per_subspace())
        throw std::invalid_argument("table must come from compute_adc_table");
    if (code.shape(0) != self.num_subspaces())
        throw std::invalid_argument("code length must equal num_subspaces");
    return self.distance_adc(as_span(table), as_span(code));
}

static float pq_distance_sdc(const vecengine::PQCodebook &self, ByteVector query_code, ByteVector code) {
    if (query_code.shape(0) != self.num_subspaces() || code.shape(0) != self.num_subspaces())
        throw std::invalid_argument("code length must equal num_subspaces");
    return self.distance_sdc(as_span(query_code), as_span(code));
}

// nanobind's unique_ptr<Base> caster does not reliably downcast to the
// concrete Python type in this nanobind version; release to a raw pointer
// and hand ownership to Python explicitly via take_ownership instead.
static vecengine::Index* make_index_py(vecengine::IndexType type, std::size_t dim, const vecengine::IndexOptions& opts) {
    return vecengine::make_index(type, dim, opts).release();
}

NB_MODULE(qann, m) {
    m.doc() = "qann: Quantized Approximate Nearest Neighbors (Flat, IVF, PQ, re-ranking)";
    m.attr("__version__") = QANN_VERSION;

    m.def("l2_distance", &batch_l2, nb::arg("a"), nb::arg("b"),
          "Squared L2 distance between rows of two (M, D) float32 arrays.");

    m.def("cosine_distance", &batch_cosine, nb::arg("a"), nb::arg("b"),
          "Cosine distance between rows of two (M, D) float32 arrays.");

    m.def("set_num_threads", &vecengine::set_num_threads, nb::arg("n"),
          "Threads for batch queries, batch adds and PQ training; 0 = one per core (default). "
          "Does not limit BLAS's own threads (VECLIB_MAXIMUM_THREADS / OPENBLAS_NUM_THREADS).");
    m.def("num_threads", &vecengine::num_threads,
          "Thread count parallel work will use (resolves the 0 default to the core count).");

    nb::class_<vecengine::Index>(m, "Index");

    nb::class_<vecengine::FlatIndex, vecengine::Index>(m, "FlatIndex")
        .def(nb::init<std::size_t>(), nb::arg("dim"), "Construct FlatIndex with vector dimension")
        .def("add", &index_add, nb::arg("data"), "Add matrix of vectors to index")
        .def("query", &index_query, nb::arg("query"), nb::arg("k"), "Query k-NN for a (dim,) vector; returns 1D (indices, distances)")
        .def("batch_query", &index_batch_query, nb::arg("query"), nb::arg("k"), "Query k-NN for each row; returns (indices, distances) arrays of shape (num_queries, k), padded with -1 / inf")
        .def("size", &vecengine::Index::size, "Get number of indexed vectors")
        .def("dim", &vecengine::Index::dim, "Get vector dimension");

    nb::enum_<vecengine::PQDistance>(m, "PQDistance")
        .value("ADC", vecengine::PQDistance::ADC)
        .value("SDC", vecengine::PQDistance::SDC);

    nb::class_<vecengine::IVFIndex, vecengine::Index>(m, "IVFIndex")
        .def(nb::init<std::size_t, std::size_t, std::size_t>(), nb::arg("dim"), nb::arg("nlist"), nb::arg("nprobe") = 10,
             "Construct IVFIndex with vector dimension, number of coarse clusters, and search breadth")
        .def("train", &ivf_train, nb::arg("data"), nb::arg("max_iters") = 25, nb::arg("seed") = nb::none(),
             "Run k-means over sample vectors to compute coarse centroids")
        .def("enable_pq", &vecengine::IVFIndex::enable_pq,
             nb::arg("num_subspaces"), nb::arg("centroids_per_subspace") = 256,
             "Enable PQ compression of residuals. Call before train()/add().")
        .def_prop_rw("nprobe", &vecengine::IVFIndex::nprobe, &vecengine::IVFIndex::set_nprobe,
                     "Number of clusters scanned per query; can be changed after training")
        .def_prop_rw("pq_distance", &vecengine::IVFIndex::pq_distance, &vecengine::IVFIndex::set_pq_distance,
                     "How queries score PQ codes: PQDistance.ADC (default, more accurate) or PQDistance.SDC")
        .def("add", &index_add, nb::arg("data"), "Add matrix of vectors to index")
        .def("query", &index_query, nb::arg("query"), nb::arg("k"), "Query k-NN for a (dim,) vector; returns 1D (indices, distances)")
        .def("batch_query", &index_batch_query, nb::arg("query"), nb::arg("k"), "Query k-NN for each row; returns (indices, distances) arrays of shape (num_queries, k), padded with -1 / inf")
        .def("size", &vecengine::Index::size, "Get number of indexed vectors")
        .def("dim", &vecengine::Index::dim, "Get vector dimension");

    // Holds `base` by reference; keep_alive ties base's lifetime to the
    // RefineIndex so Python can't free it first.
    nb::class_<vecengine::RefineIndex, vecengine::Index>(m, "RefineIndex")
        .def(nb::init<vecengine::Index&, std::size_t>(), nb::arg("base"), nb::arg("k_factor") = 10,
             nb::keep_alive<1, 2>(),
             "Re-rank an empty, trained approximate index with exact distances. Add vectors through this wrapper.")
        .def_prop_rw("k_factor", &vecengine::RefineIndex::k_factor, &vecengine::RefineIndex::set_k_factor,
                     "Candidates fetched from base per result (k * k_factor)")
        .def("add", &index_add, nb::arg("data"), "Add matrix of vectors to base and raw storage")
        .def("query", &index_query, nb::arg("query"), nb::arg("k"), "Query k-NN for a (dim,) vector; returns 1D (indices, distances)")
        .def("batch_query", &index_batch_query, nb::arg("query"), nb::arg("k"), "Query k-NN for each row; returns (indices, distances) arrays of shape (num_queries, k), padded with -1 / inf")
        .def("size", &vecengine::Index::size, "Get number of indexed vectors")
        .def("dim", &vecengine::Index::dim, "Get vector dimension");

    nb::enum_<vecengine::IndexType>(m, "IndexType")
        .value("Flat", vecengine::IndexType::Flat)
        .value("IVF", vecengine::IndexType::IVF);

    nb::class_<vecengine::IndexOptions>(m, "IndexOptions")
        .def(nb::init<>())
        .def_rw("capacity", &vecengine::IndexOptions::capacity)
        .def_rw("nlist", &vecengine::IndexOptions::nlist)
        .def_rw("nprobe", &vecengine::IndexOptions::nprobe)
        .def_rw("pq_subspaces", &vecengine::IndexOptions::pq_subspaces)
        .def_rw("pq_centroids", &vecengine::IndexOptions::pq_centroids);

    nb::class_<vecengine::PQCodebook>(m, "PQCodebook")
        .def(nb::init<std::size_t, std::size_t, std::size_t>(),
             nb::arg("dim"), nb::arg("num_subspaces"), nb::arg("centroids_per_subspace") = 256,
             "Construct a PQ codebook; dim must be divisible by num_subspaces")
        .def("train", &pq_train, nb::arg("data"), nb::arg("max_iters") = 25, nb::arg("seed") = nb::none(),
             "Run per-subspace k-means over (N, dim) float32 vectors and build the SDC table")
        .def("encode", &pq_encode, nb::arg("vec"), "Encode a (dim,) vector into num_subspaces uint8 codes")
        .def("compute_adc_table", &pq_compute_adc_table, nb::arg("query"),
             "Distances from each query slice to every centroid; reuse across many codes")
        .def("distance_adc", &pq_distance_adc, nb::arg("table"), nb::arg("code"),
             "Approximate squared L2 between the table's query and a code")
        .def("distance_sdc", &pq_distance_sdc, nb::arg("query_code"), nb::arg("code"),
             "Approximate squared L2 between two codes via the precomputed SDC table")
        .def("dim", &vecengine::PQCodebook::dim)
        .def("num_subspaces", &vecengine::PQCodebook::num_subspaces)
        .def("centroids_per_subspace", &vecengine::PQCodebook::centroids_per_subspace);

    m.def("make_index", &make_index_py, nb::arg("type"), nb::arg("dim"), nb::arg("opts") = vecengine::IndexOptions{},
          nb::rv_policy::take_ownership,
          "Construct an index of the given type. Returns a base Index handle; "
          "for IVF-specific methods like train(), construct IVFIndex directly instead.");
}
