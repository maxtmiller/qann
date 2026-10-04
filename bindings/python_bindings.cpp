#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

#include <span>

#include "vecengine/distance.hpp"
#include "vecengine/flat_index.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/pq.hpp"
#include "vecengine/index_factory.hpp"

namespace nb = nanobind;

// 2D row-major float32 NumPy array, zero-copy via nanobind::ndarray.
using FloatMatrix = nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using FloatVector = nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using ByteVector = nb::ndarray<uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

static std::span<const float> as_span(FloatVector v) { return {v.data(), v.shape(0)}; }
static std::span<const uint8_t> as_span(ByteVector v) { return {v.data(), v.shape(0)}; }

// Compute pairwise L2 distances between rows of two matrices (M x D each).
// Returns a length-M vector of squared L2 distances.
static std::vector<float> batch_l2(FloatMatrix a, FloatMatrix b) {
    if (a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1))
        throw std::invalid_argument("a and b must have the same shape");

    const std::size_t M = a.shape(0);
    const std::size_t D = a.shape(1);
    std::vector<float> out(M);

    for (std::size_t i = 0; i < M; ++i)
        out[i] = vecengine::l2_distance(a.data() + i * D, b.data() + i * D, D);

    return out;
}

// Compute pairwise cosine distances between rows of two matrices (M x D each).
static std::vector<float> batch_cosine(FloatMatrix a, FloatMatrix b) {
    if (a.shape(0) != b.shape(0) || a.shape(1) != b.shape(1))
        throw std::invalid_argument("a and b must have the same shape");

    const std::size_t M = a.shape(0);
    const std::size_t D = a.shape(1);
    std::vector<float> out(M);

    for (std::size_t i = 0; i < M; ++i)
        out[i] = vecengine::cosine_distance(a.data() + i * D, b.data() + i * D, D);

    return out;
}

// Add all rows of a (N x dim) matrix to the index.
static void index_add(vecengine::Index &self, FloatMatrix data) {
    if (data.shape(1) != self.dim())
        throw std::invalid_argument("vector dimension does not match index dimension");

    const std::size_t N = data.shape(0);
    const std::size_t D = data.shape(1);
    for (std::size_t i = 0; i < N; ++i)
        self.add(std::span<const float>(data.data() + i * D, D));
}

// Query k-NN for a single vector; returns (indices, distances).
static std::pair<std::vector<std::size_t>, std::vector<float>>
index_query(const vecengine::Index &self, FloatMatrix query, std::size_t k) {
    if (query.shape(1) != self.dim())
        throw std::invalid_argument("query dimension does not match index dimension");

    auto results = self.query(std::span<const float>(query.data(), query.shape(1)), k);

    std::vector<std::size_t> indices(results.size());
    std::vector<float> distances(results.size());
    for (std::size_t i = 0; i < results.size(); ++i) {
        indices[i] = results[i].index;
        distances[i] = results[i].distance;
    }
    return {indices, distances};
}

// Query k-NN for a single vector; returns (indices, distances).
static std::vector<std::pair<std::vector<std::size_t>, std::vector<float>>>
index_batch_query(const vecengine::Index &self, FloatMatrix query, std::size_t k) {
    if (query.shape(1) != self.dim())
        throw std::invalid_argument("query dimension does not match index dimension");

    const std::size_t num_queries = query.shape(0);
    auto results = self.query_batch(std::span<const float>(query.data(), query.size()), num_queries, k);
    
    std::vector<std::pair<std::vector<std::size_t>, std::vector<float>>> batch;
    batch.reserve(num_queries);
    for (std::size_t i = 0; i < results.size(); ++i) {
        std::vector<std::size_t> indices(results[i].size());
        std::vector<float> distances(results[i].size());
        for (std::size_t j = 0; j < results[i].size(); ++j) {
            indices[j] = results[i][j].index;
            distances[j] = results[i][j].distance;
        }
        batch.emplace_back(indices, distances);
    }
    return batch;
}

// Train an IVFIndex's coarse centroids from a (N x dim) matrix of sample vectors.
static void ivf_train(vecengine::IVFIndex &self, FloatMatrix data, std::size_t max_iters) {
    if (data.shape(1) != self.dim())
        throw std::invalid_argument("training vector dimension does not match index dimension");
    self.train(std::span<const float>(data.data(), data.size()), data.shape(0), max_iters);
}

static void pq_train(vecengine::PQCodebook &self, FloatMatrix data, std::size_t max_iters) {
    if (data.shape(1) != self.dim())
        throw std::invalid_argument("training vector dimension does not match codebook dimension");
    self.train(std::span<const float>(data.data(), data.size()), data.shape(0), max_iters);
}

static std::vector<uint8_t> pq_encode(const vecengine::PQCodebook &self, FloatVector vec) {
    if (vec.shape(0) != self.dim())
        throw std::invalid_argument("vector dimension does not match codebook dimension");
    return self.encode(as_span(vec));
}

static std::vector<float> pq_compute_adc_table(const vecengine::PQCodebook &self, FloatVector query) {
    if (query.shape(0) != self.dim())
        throw std::invalid_argument("query dimension does not match codebook dimension");
    return self.compute_adc_table(as_span(query));
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

NB_MODULE(vecengine, m) {
    m.doc() = "vecengine: high-performance vector distance routines";

    m.def("l2_distance", &batch_l2, nb::arg("a"), nb::arg("b"),
          "Squared L2 distance between rows of two (M, D) float32 arrays.");

    m.def("cosine_distance", &batch_cosine, nb::arg("a"), nb::arg("b"),
          "Cosine distance between rows of two (M, D) float32 arrays.");

    nb::class_<vecengine::Index>(m, "Index");

    nb::class_<vecengine::FlatIndex, vecengine::Index>(m, "FlatIndex")
        .def(nb::init<std::size_t>(), nb::arg("dim"), "Construct FlatIndex with vector dimension")
        .def("add", &index_add, nb::arg("data"), "Add matrix of vectors to index")
        .def("query", &index_query, nb::arg("query"), nb::arg("k"), "Query k-NN for a query vector; returns (indices, distances)")
        .def("batch_query", &index_batch_query, nb::arg("query"), nb::arg("k"), "Query k-NN for multiple query vectors; returns list of (indices, distances)")
        .def("size", &vecengine::Index::size, "Get number of indexed vectors")
        .def("dim", &vecengine::Index::dim, "Get vector dimension");

    nb::enum_<vecengine::PQDistance>(m, "PQDistance")
        .value("ADC", vecengine::PQDistance::ADC)
        .value("SDC", vecengine::PQDistance::SDC);

    nb::class_<vecengine::IVFIndex, vecengine::Index>(m, "IVFIndex")
        .def(nb::init<std::size_t, std::size_t, std::size_t>(), nb::arg("dim"), nb::arg("nlist"), nb::arg("nprobe") = 10,
             "Construct IVFIndex with vector dimension, number of coarse clusters, and search breadth")
        .def("train", &ivf_train, nb::arg("data"), nb::arg("max_iters") = 25,
             "Run k-means over sample vectors to compute coarse centroids")
        .def("enable_pq", &vecengine::IVFIndex::enable_pq,
             nb::arg("num_subspaces"), nb::arg("centroids_per_subspace") = 256,
             "Enable PQ compression of residuals. Call before train()/add().")
        .def_prop_rw("pq_distance", &vecengine::IVFIndex::pq_distance, &vecengine::IVFIndex::set_pq_distance,
                     "How queries score PQ codes: PQDistance.ADC (default, more accurate) or PQDistance.SDC")
        .def("add", &index_add, nb::arg("data"), "Add matrix of vectors to index")
        .def("query", &index_query, nb::arg("query"), nb::arg("k"), "Query k-NN for a query vector; returns (indices, distances)")
        .def("batch_query", &index_batch_query, nb::arg("query"), nb::arg("k"), "Query k-NN for multiple query vectors; returns list of (indices, distances)")
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
        .def("train", &pq_train, nb::arg("data"), nb::arg("max_iters") = 25,
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
