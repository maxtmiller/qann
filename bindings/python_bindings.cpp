#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/map.h>

#include <cstdint>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <span>

#include "vecengine/distance.hpp"
#include "vecengine/flat_index.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/hnsw_index.hpp"
#include "vecengine/metadata.hpp"
#include "vecengine/pq.hpp"
#include "vecengine/refine_index.hpp"
#include "vecengine/index_factory.hpp"
#include "vecengine/threads.hpp"

namespace nb = nanobind;

// 2D row-major float32 NumPy array, zero-copy via nanobind::ndarray.
using FloatMatrix = nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using FloatVector = nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using ByteVector = nb::ndarray<uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using IdVector = nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
// What add()/remove() accept as ids; the stubs show it as Iterable[int].
using IdsArg = nb::typed<nb::iterable, int>;

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

// Locks an index and, for a RefineIndex, the chain of bases under it (the
// base is also reachable from Python, e.g. ivf.add() on a wrapped IVFIndex).
// Always taken outermost index first, so lock order is consistent.
template <class Lock>
static std::vector<Lock> lock_chain(const vecengine::Index& index) {
    std::vector<Lock> locks;
    for (const vecengine::Index* p = &index; p != nullptr;) {
        locks.emplace_back(p->mutex());
        auto* refine = dynamic_cast<const vecengine::RefineIndex*>(p);
        p = refine ? &refine->base() : nullptr;
    }
    return locks;
}

// Run f with the GIL released and the index locked: shared for reads,
// exclusive for writes. The GIL is released before locking so a thread
// waiting for the lock never holds the GIL that the lock holder needs back.
// f must not touch Python objects; build NumPy results after it returns.
template <class F>
static auto read_locked(const vecengine::Index& index, F&& f) {
    nb::gil_scoped_release release;
    auto locks = lock_chain<std::shared_lock<std::shared_mutex>>(index);
    return f();
}

template <class F>
static auto write_locked(const vecengine::Index& index, F&& f) {
    nb::gil_scoped_release release;
    auto locks = lock_chain<std::unique_lock<std::shared_mutex>>(index);
    return f();
}

// Converts any sequence of integers (NumPy array of any int dtype, list,
// range, ...) to a 1D int64 array. Floats are rejected rather than truncated.
static IdVector as_ids(nb::handle obj) {
    nb::module_ np = nb::module_::import_("numpy");
    nb::object arr = np.attr("asarray")(obj);
    if (nb::cast<std::size_t>(arr.attr("size")) > 0) {
        const auto kind = nb::cast<std::string>(arr.attr("dtype").attr("kind"));
        if (kind != "i" && kind != "u") throw nb::type_error("ids must be integers");
    }
    if (nb::cast<std::size_t>(arr.attr("ndim")) != 1) throw nb::type_error("ids must be one-dimensional");
    return nb::cast<IdVector>(np.attr("ascontiguousarray")(arr, nb::arg("dtype") = "int64"));
}

// Changing a RefineIndex's base directly would misalign the two (an add would
// give the base a vector the RefineIndex has no copy of), so it is refused.
// Called with the index locked, since a RefineIndex sets the flag under it.
static void refuse_if_wrapped(const vecengine::Index& self, const char* what) {
    if (self.wrapped())
        throw std::invalid_argument(std::string("cannot ") + what + " on the base of a RefineIndex; " + what +
                                    " through the RefineIndex instead");
}

// qann.field(name): a column to compare; its operators build Expr filters.
struct PyField {
    std::string name;
};

static vecengine::Expr field_compare(const PyField& f, vecengine::Expr::Op op, vecengine::Expr::Value value) {
    return vecengine::Expr::compare(f.name, op, std::move(value));
}

// Converts add()'s metadata argument, {name: values}, to columns of n values.
// Values go through numpy.asarray: integer and bool arrays become Int64
// columns, float arrays Float64, and str arrays (or object arrays of str)
// String columns. Must run with the GIL held.
static vecengine::Metadata as_metadata(nb::dict metadata, std::size_t n) {
    nb::module_ np = nb::module_::import_("numpy");
    vecengine::Metadata out;
    for (auto [key, value] : metadata) {
        if (!nb::isinstance<nb::str>(key)) throw nb::type_error("metadata column names must be strings");
        const std::string name = nb::cast<std::string>(key);
        nb::object arr = np.attr("asarray")(value);
        if (nb::cast<std::size_t>(arr.attr("ndim")) != 1)
            throw nb::type_error(("metadata column '" + name + "' must be one-dimensional").c_str());
        if (nb::cast<std::size_t>(arr.attr("shape")[nb::int_(0)]) != n)
            throw std::invalid_argument("metadata column '" + name + "' must have one value per row of data");

        const auto kind = nb::cast<std::string>(arr.attr("dtype").attr("kind"));
        if (kind == "u" && nb::cast<std::size_t>(arr.attr("size")) > 0 &&
            nb::cast<uint64_t>(arr.attr("max")()) > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            throw std::invalid_argument("metadata column '" + name + "' has values above the int64 maximum");
        if (kind == "i" || kind == "u" || kind == "b") {
            auto ints = nb::cast<IdVector>(np.attr("ascontiguousarray")(arr, nb::arg("dtype") = "int64"));
            out.push_back({name, std::vector<int64_t>(ints.data(), ints.data() + ints.shape(0))});
        } else if (kind == "f") {
            auto floats = nb::cast<nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>>(
                np.attr("ascontiguousarray")(arr, nb::arg("dtype") = "float64"));
            out.push_back({name, std::vector<double>(floats.data(), floats.data() + floats.shape(0))});
        } else if (kind == "U" || kind == "O") {
            std::vector<std::string> strings;
            strings.reserve(n);
            // numpy.asarray(["a", 1]) silently makes "1" a string, so check a list's own items
            nb::object items = nb::hasattr(value, "dtype") ? arr.attr("tolist")() : nb::borrow(value);
            for (nb::handle item : items) {
                if (!nb::isinstance<nb::str>(item))
                    throw nb::type_error(("metadata column '" + name + "' mixes strings with other values").c_str());
                strings.push_back(nb::cast<std::string>(item));
            }
            out.push_back({name, std::move(strings)});
        } else {
            throw nb::type_error(("metadata column '" + name + "' must hold ints, floats or strings").c_str());
        }
    }
    return out;
}

// Add all rows of a (N x dim) matrix to the index, with one id per row when
// the index uses custom ids, and optional metadata columns.
static void index_add(vecengine::Index &self, FloatMatrix data, std::optional<IdsArg> ids,
                      std::optional<nb::dict> metadata) {
    if (data.shape(1) != self.dim())
        throw std::invalid_argument("vector dimension does not match index dimension");

    std::optional<IdVector> id_array;
    if (ids) {
        id_array = as_ids(*ids);
        if (id_array->shape(0) != data.shape(0))
            throw std::invalid_argument("ids must have one entry per row of data");
    }

    const vecengine::Metadata md = metadata ? as_metadata(*metadata, data.shape(0)) : vecengine::Metadata{};
    const int64_t* id_ptr = id_array ? id_array->data() : nullptr;
    write_locked(self, [&] {
        refuse_if_wrapped(self, "add");
        self.add_batch(std::span<const float>(data.data(), data.size()), data.shape(0), id_ptr, md);
    });
}

// Delete the vectors with these ids; returns how many were deleted.
static std::size_t index_remove(vecengine::Index& self, IdsArg ids) {
    IdVector id_array = as_ids(ids);
    return write_locked(self, [&] {
        refuse_if_wrapped(self, "remove");
        return self.remove(std::span<const int64_t>(id_array.data(), id_array.shape(0)));
    });
}

// Free the storage of deleted vectors; returns how many slots were reclaimed.
static std::size_t index_compact(vecengine::Index& self) {
    return write_locked(self, [&] {
        refuse_if_wrapped(self, "compact");
        return self.compact();
    });
}

static std::size_t index_size(const vecengine::Index& self) {
    return read_locked(self, [&] { return self.size(); });
}

// Query k-NN for a single vector, among the ids in allowed when given;
// returns (indices, distances) as 1D arrays with one entry per result (int64,
// float32).
static std::pair<NumpyArray<int64_t>, NumpyArray<float>>
index_query(const vecengine::Index &self, FloatVector query, std::size_t k, std::optional<IdsArg> allowed,
            std::optional<vecengine::Expr> where) {
    if (query.shape(0) != self.dim())
        throw std::invalid_argument("query dimension does not match index dimension");

    std::optional<IdVector> allowed_ids;
    if (allowed) allowed_ids = as_ids(*allowed);
    auto results = read_locked(self, [&] {
        if (!allowed_ids && !where) return self.query(as_span(query), k);
        if (!where) return self.query(as_span(query), k, std::span<const int64_t>(allowed_ids->data(), allowed_ids->shape(0)));
        if (!allowed_ids) return self.query(as_span(query), k, *where);
        return self.query(as_span(query), k, *where,
                          std::span<const int64_t>(allowed_ids->data(), allowed_ids->shape(0)));
    });

    const std::size_t n = results.size();
    std::vector<int64_t> indices(n);
    std::vector<float> distances(n);
    for (std::size_t i = 0; i < n; ++i) {
        indices[i] = static_cast<int64_t>(results[i].index);
        distances[i] = results[i].distance;
    }
    return {to_numpy(std::move(indices), {n}), to_numpy(std::move(distances), {n})};
}

// Query k-NN for each row, among the ids in allowed when given; returns
// (indices, distances) as (num_queries, k) arrays. Rows with fewer than k
// results are padded with -1 / inf.
static std::pair<NumpyArray<int64_t>, NumpyArray<float>>
index_batch_query(const vecengine::Index &self, FloatMatrix query, std::size_t k, std::optional<IdsArg> allowed,
                  std::optional<vecengine::Expr> where) {
    if (query.shape(1) != self.dim())
        throw std::invalid_argument("query dimension does not match index dimension");

    const std::size_t num_queries = query.shape(0);
    std::optional<IdVector> allowed_ids;
    if (allowed) allowed_ids = as_ids(*allowed);
    auto results = read_locked(self, [&] {
        const std::span<const float> queries(query.data(), query.size());
        if (!allowed_ids && !where) return self.query_batch(queries, num_queries, k);
        if (!where)
            return self.query_batch(queries, num_queries, k,
                                    std::span<const int64_t>(allowed_ids->data(), allowed_ids->shape(0)));
        if (!allowed_ids) return self.query_batch(queries, num_queries, k, *where);
        return self.query_batch(queries, num_queries, k, *where,
                                std::span<const int64_t>(allowed_ids->data(), allowed_ids->shape(0)));
    });

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
static void hnsw_train(vecengine::HNSWIndex &self, FloatMatrix data) {
    if (data.shape(1) != self.dim())
        throw std::invalid_argument("training vector dimension does not match index dimension");
    write_locked(self, [&] { self.train(std::span<const float>(data.data(), data.size()), data.shape(0)); });
}

static void ivf_train(vecengine::IVFIndex &self, FloatMatrix data, std::size_t max_iters, std::optional<uint32_t> seed) {
    if (data.shape(1) != self.dim())
        throw std::invalid_argument("training vector dimension does not match index dimension");
    write_locked(self, [&] { self.train(std::span<const float>(data.data(), data.size()), data.shape(0), max_iters, seed); });
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

    // Methods bound on Index are inherited by FlatIndex, IVFIndex and RefineIndex,
    // and give type checkers the right methods for what load() and make_index() return.
    // Registered before the index classes, which use it as a default argument.
    nb::enum_<vecengine::Metric>(m, "Metric")
        .value("L2", vecengine::Metric::L2, "Squared Euclidean distance")
        .value("Cosine", vecengine::Metric::Cosine,
               "1 - cosine similarity (0 to 2); vectors are scaled to unit length on add and query")
        .value("InnerProduct", vecengine::Metric::InnerProduct,
               "Negative inner product -(a . b), so the largest inner product comes first");

    // Filters for where=: qann.field("price") < 50, combined with & | ~.
    nb::class_<vecengine::Expr>(m, "Expr", "A metadata filter for where=; build one with qann.field()")
        .def("__and__", [](const vecengine::Expr& a, const vecengine::Expr& b) { return a & b; }, nb::arg("other"))
        .def("__or__", [](const vecengine::Expr& a, const vecengine::Expr& b) { return a | b; }, nb::arg("other"))
        .def("__invert__", [](const vecengine::Expr& a) { return ~a; })
        // `and`, `or`, `not`, `if` and chained comparisons (0 < f < 5) would call this and
        // silently drop part of the filter, so refuse instead
        .def("__bool__", [](const vecengine::Expr&) -> bool {
            throw nb::type_error("a filter has no truth value: combine filters with & | ~ (not and / or / not), "
                                 "and write ranges as (f > 0) & (f < 5)");
        })
        .def("__repr__", [](const vecengine::Expr& e) { return "qann.Expr(" + e.to_string() + ")"; });

    nb::class_<PyField>(m, "Field", "A metadata column to compare in a where= filter; see qann.field()")
        .def_ro("name", &PyField::name)
        .def("__eq__", [](const PyField& f, vecengine::Expr::Value v) { return field_compare(f, vecengine::Expr::Op::Eq, std::move(v)); }, nb::arg("value"))
        .def("__ne__", [](const PyField& f, vecengine::Expr::Value v) { return field_compare(f, vecengine::Expr::Op::Ne, std::move(v)); }, nb::arg("value"))
        .def("__lt__", [](const PyField& f, vecengine::Expr::Value v) { return field_compare(f, vecengine::Expr::Op::Lt, std::move(v)); }, nb::arg("value"))
        .def("__le__", [](const PyField& f, vecengine::Expr::Value v) { return field_compare(f, vecengine::Expr::Op::Le, std::move(v)); }, nb::arg("value"))
        .def("__gt__", [](const PyField& f, vecengine::Expr::Value v) { return field_compare(f, vecengine::Expr::Op::Gt, std::move(v)); }, nb::arg("value"))
        .def("__ge__", [](const PyField& f, vecengine::Expr::Value v) { return field_compare(f, vecengine::Expr::Op::Ge, std::move(v)); }, nb::arg("value"))
        .def("isin", [](const PyField& f, std::vector<vecengine::Expr::Value> values) {
                 return vecengine::Expr::isin(f.name, std::move(values));
             },
             nb::arg("values"), "Matches rows whose value is one of values")
        .def("is_null", [](const PyField& f) { return vecengine::Expr::is_null(f.name); },
             "Matches rows with no value in this column")
        .def("__repr__", [](const PyField& f) { return "qann.field('" + f.name + "')"; });

    m.def("field", [](std::string name) { return PyField{std::move(name)}; }, nb::arg("name"),
          "A metadata column, for filters such as (qann.field('price') < 50) & (qann.field('color') == 'red')");

    nb::class_<vecengine::Index>(m, "Index")
        .def("save", [](const vecengine::Index& self, const std::filesystem::path& path) {
                 read_locked(self, [&] { vecengine::save_index(self, path); });
             },
             nb::arg("path"), "Save the index to a file; load it back with qann.load(path)")
        .def("add", &index_add, nb::arg("data"), nb::arg("ids").none() = nb::none(), nb::kw_only(),
             nb::arg("metadata").none() = nb::none(),
             "Add an (n, dim) array of vectors. Pass ids (n int64 values >= 0) if the index was "
             "created with custom_ids=True; otherwise ids continue after the highest id ever assigned. "
             "metadata is {name: n values} (ints, floats or strings) to filter queries on with where=; a new "
             "column is null for earlier vectors, and a column left out of an add is null for its vectors")
        .def("remove", &index_remove, nb::arg("ids"),
             "Delete the vectors with these ids; unknown ids are skipped. Returns how many were deleted")
        .def("compact", &index_compact,
             "Free the memory of deleted vectors; ids and results are unchanged. Returns how many were freed")
        .def("query", &index_query, nb::arg("query"), nb::arg("k"), nb::kw_only(), nb::arg("allowed").none() = nb::none(),
             nb::arg("where").none() = nb::none(),
             "Query k-NN for a (dim,) vector; returns 1D (indices, distances). With allowed (ids), only those "
             "vectors can be returned; unknown and deleted ids in it are ignored. With where (a filter built from "
             "qann.field), only vectors whose metadata matches; with both, only those that pass both")
        .def("batch_query", &index_batch_query, nb::arg("query"), nb::arg("k"), nb::kw_only(),
             nb::arg("allowed").none() = nb::none(), nb::arg("where").none() = nb::none(),
             "Query k-NN for each row; returns (indices, distances) arrays of shape (num_queries, k), padded with "
             "-1 / inf. allowed (ids) and where (a qann.field filter) restrict the results as in query(), for every row")
        .def_prop_ro("metadata_schema",
                     [](const vecengine::Index& self) {
                         auto schema = read_locked(self, [&] { return self.metadata_schema(); });
                         nb::dict out;
                         for (const auto& [name, type] : schema)
                             out[nb::str(name.c_str())] = type == vecengine::ColumnType::Int64     ? "int"
                                                          : type == vecengine::ColumnType::Float64 ? "float"
                                                                                                   : "str";
                         return out;
                     },
                     "Each metadata column's name and type ('int', 'float' or 'str'), in the order they were added")
        .def_prop_ro("metric", &vecengine::Index::metric, "How vectors are compared (fixed at construction)")
        .def("size", &index_size, "Get number of indexed vectors")
        .def("dim", &vecengine::Index::dim, "Get vector dimension");

    nb::class_<vecengine::FlatIndex, vecengine::Index>(m, "FlatIndex")
        .def("__init__", [](vecengine::FlatIndex* self, std::size_t dim, bool custom_ids, vecengine::Metric metric) {
                 new (self) vecengine::FlatIndex(dim, 1024, custom_ids, metric);
             },
             nb::arg("dim"), nb::kw_only(), nb::arg("custom_ids") = false, nb::arg("metric") = vecengine::Metric::L2,
             "Construct FlatIndex with vector dimension; custom_ids=True makes add() take your own ids");

    nb::enum_<vecengine::PQDistance>(m, "PQDistance")
        .value("ADC", vecengine::PQDistance::ADC)
        .value("SDC", vecengine::PQDistance::SDC);

    nb::class_<vecengine::IVFIndex, vecengine::Index>(m, "IVFIndex")
        .def(nb::init<std::size_t, std::size_t, std::size_t, bool, vecengine::Metric>(), nb::arg("dim"), nb::arg("nlist"),
             nb::arg("nprobe") = 10, nb::kw_only(), nb::arg("custom_ids") = false, nb::arg("metric") = vecengine::Metric::L2,
             "Construct IVFIndex with vector dimension, number of coarse clusters, and search breadth")
        .def("train", &ivf_train, nb::arg("data"), nb::arg("max_iters") = 25, nb::arg("seed") = nb::none(),
             "Run k-means over sample vectors to compute coarse centroids")
        .def("enable_pq",
             [](vecengine::IVFIndex& self, std::size_t num_subspaces, std::optional<std::size_t> centroids_per_subspace,
                bool fast_scan, bool rotate) {
                 const std::size_t k = centroids_per_subspace.value_or(fast_scan ? 16 : 256);
                 write_locked(self, [&] { self.enable_pq(num_subspaces, k, fast_scan, rotate); });
             },
             nb::arg("num_subspaces"), nb::arg("centroids_per_subspace").none() = nb::none(), nb::kw_only(),
             nb::arg("fast_scan") = false, nb::arg("rotate") = false,
             "Enable PQ compression of residuals. Call before train()/add(). centroids_per_subspace defaults to "
             "256, or 16 with fast_scan=True, which stores 4-bit codes and scores 32 vectors per SIMD instruction "
             "(much faster; distances are approximate, so pair it with a RefineIndex). fast_scan needs an even "
             "num_subspaces <= 1024 and ADC. rotate=True makes train() also learn a rotation (PCA, with the "
             "variance spread evenly over the subspaces) applied to every vector first: lower PQ error when "
             "dimensions are correlated (e.g. SIFT), no effect on near-isotropic embeddings")
        .def_prop_ro("pq_rotate",
                     [](const vecengine::IVFIndex& self) { return read_locked(self, [&] { return self.pq_rotate(); }); },
                     "Whether enable_pq(..., rotate=True) was used")
        .def("enable_sq", [](vecengine::IVFIndex& self, int bits) { write_locked(self, [&] { self.enable_sq(bits); }); },
             nb::arg("bits") = 8,
             "Store vectors as scalar codes of 8 bits (4x smaller than floats) or 4 bits (8x smaller) per "
             "dimension instead of raw floats. Call before train()/add(); not with enable_pq(). Distances become "
             "approximate; 8 bits loses little recall, 4 bits should be wrapped in a RefineIndex")
        .def_prop_ro("sq_bits", [](const vecengine::IVFIndex& self) { return read_locked(self, [&] { return self.sq_bits(); }); },
                     "Bits per dimension set by enable_sq() (4 or 8), or 0 without SQ")
        .def_prop_ro("fast_scan",
                     [](const vecengine::IVFIndex& self) { return read_locked(self, [&] { return self.fast_scan(); }); },
                     "Whether enable_pq(..., fast_scan=True) was used")
        .def_prop_rw("nprobe",
                     [](const vecengine::IVFIndex& self) { return read_locked(self, [&] { return self.nprobe(); }); },
                     [](vecengine::IVFIndex& self, std::size_t v) { write_locked(self, [&] { self.set_nprobe(v); }); },
                     "Number of clusters scanned per query; can be changed after training")
        .def_prop_rw("pq_distance",
                     [](const vecengine::IVFIndex& self) { return read_locked(self, [&] { return self.pq_distance(); }); },
                     [](vecengine::IVFIndex& self, vecengine::PQDistance v) { write_locked(self, [&] { self.set_pq_distance(v); }); },
                     "How queries score PQ codes: PQDistance.ADC (default, more accurate) or PQDistance.SDC")
        .def_prop_rw("precomputed_tables",
                     [](const vecengine::IVFIndex& self) { return read_locked(self, [&] { return self.precomputed_tables(); }); },
                     [](vecengine::IVFIndex& self, bool v) { write_locked(self, [&] { self.set_precomputed_tables(v); }); },
                     "Whether ADC queries use per-list tables precomputed at train time (on by default). "
                     "True only once they exist: PQ enabled, trained, and within the 256 MB cap");

    nb::class_<vecengine::HNSWIndex, vecengine::Index>(m, "HNSWIndex")
        .def(nb::init<std::size_t, std::size_t, std::size_t, bool, std::optional<uint32_t>, vecengine::Metric>(),
             nb::arg("dim"), nb::arg("M") = 16, nb::arg("ef_construction") = 200, nb::kw_only(),
             nb::arg("custom_ids") = false, nb::arg("seed") = nb::none(), nb::arg("metric") = vecengine::Metric::L2,
             "Construct an HNSW graph index. add() builds the graph on all cores; no training unless enable_sq() is "
             "used. M is the neighbors per node (2 * M on the bottom level); ef_construction the search width while "
             "inserting. A seed makes the build reproducible when qann.set_num_threads(1) is set")
        .def("enable_sq", [](vecengine::HNSWIndex& self, int bits) { write_locked(self, [&] { self.enable_sq(bits); }); },
             nb::arg("bits") = 8,
             "Store vectors as scalar codes of 8 bits (4x smaller than floats) or 4 bits (8x smaller) per "
             "dimension. Call before train()/add(). Distances become approximate; wrap the index in a "
             "RefineIndex to re-rank with exact ones")
        .def("train", &hnsw_train, nb::arg("data"),
             "Learn each dimension's SQ range from sample vectors; required after enable_sq(), before add(). "
             "Values outside the sample's range are clamped, so use a representative sample")
        .def_prop_ro("sq_bits", [](const vecengine::HNSWIndex& self) { return read_locked(self, [&] { return self.sq_bits(); }); },
                     "Bits per dimension set by enable_sq() (4 or 8), or 0 when vectors are stored as floats")
        .def_prop_rw("ef_search",
                     [](const vecengine::HNSWIndex& self) { return read_locked(self, [&] { return self.ef_search(); }); },
                     [](vecengine::HNSWIndex& self, std::size_t v) { write_locked(self, [&] { self.set_ef_search(v); }); },
                     "Search width per query (queries use max(ef_search, k)); higher is more accurate and slower")
        .def_prop_ro("M", [](const vecengine::HNSWIndex& self) { return self.M(); }, "Neighbors per node (fixed at construction)")
        .def_prop_ro("ef_construction", [](const vecengine::HNSWIndex& self) { return self.ef_construction(); },
                     "Search width used while inserting (fixed at construction)");

    // Holds `base` by reference; keep_alive ties base's lifetime to the
    // RefineIndex so Python can't free it first.
    nb::class_<vecengine::RefineIndex, vecengine::Index>(m, "RefineIndex")
        // The constructor checks base is empty; lock base so a concurrent
        // base.add() can't race that check.
        .def("__init__", [](vecengine::RefineIndex* self, vecengine::Index& base, std::size_t k_factor, bool custom_ids) {
                 write_locked(base, [&] { new (self) vecengine::RefineIndex(base, k_factor, custom_ids); });
             },
             nb::arg("base"), nb::arg("k_factor") = 10, nb::kw_only(), nb::arg("custom_ids") = false,
             nb::keep_alive<1, 2>(),
             "Re-rank an empty, trained approximate index with exact distances. Add vectors through this wrapper.")
        .def_prop_rw("k_factor",
                     [](const vecengine::RefineIndex& self) { return read_locked(self, [&] { return self.k_factor(); }); },
                     [](vecengine::RefineIndex& self, std::size_t v) { write_locked(self, [&] { self.set_k_factor(v); }); },
                     "Candidates fetched from base per result (k * k_factor)")
        .def_prop_ro("base", [](vecengine::RefineIndex& self) -> vecengine::Index& { return self.base(); },
                     nb::rv_policy::reference_internal,
                     "The wrapped approximate index, e.g. to change nprobe on a loaded RefineIndex");

    nb::enum_<vecengine::IndexType>(m, "IndexType")
        .value("Flat", vecengine::IndexType::Flat)
        .value("IVF", vecengine::IndexType::IVF)
        .value("HNSW", vecengine::IndexType::HNSW);

    nb::class_<vecengine::IndexOptions>(m, "IndexOptions")
        .def(nb::init<>())
        .def_rw("capacity", &vecengine::IndexOptions::capacity)
        .def_rw("nlist", &vecengine::IndexOptions::nlist)
        .def_rw("nprobe", &vecengine::IndexOptions::nprobe)
        .def_rw("pq_subspaces", &vecengine::IndexOptions::pq_subspaces)
        .def_rw("pq_centroids", &vecengine::IndexOptions::pq_centroids)
        .def_rw("ivf_sq_bits", &vecengine::IndexOptions::ivf_sq_bits)
        .def_rw("pq_rotate", &vecengine::IndexOptions::pq_rotate)
        .def_rw("hnsw_m", &vecengine::IndexOptions::hnsw_m)
        .def_rw("ef_construction", &vecengine::IndexOptions::ef_construction)
        .def_rw("ef_search", &vecengine::IndexOptions::ef_search)
        .def_rw("hnsw_sq_bits", &vecengine::IndexOptions::hnsw_sq_bits)
        .def_rw("custom_ids", &vecengine::IndexOptions::custom_ids)
        .def_rw("metric", &vecengine::IndexOptions::metric);

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

    m.def("make_index", &vecengine::make_index, nb::arg("type"), nb::arg("dim"), nb::arg("opts") = vecengine::IndexOptions{},
          "Construct a FlatIndex, IVFIndex or HNSWIndex from an IndexType and IndexOptions");

    m.def("load", nb::overload_cast<const std::filesystem::path&>(&vecengine::load_index), nb::arg("path"),
          nb::call_guard<nb::gil_scoped_release>(),
          "Load an index saved with Index.save(path); returns a FlatIndex, IVFIndex, HNSWIndex or RefineIndex");
}
