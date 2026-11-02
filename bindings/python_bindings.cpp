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
#include <nanobind/stl/unordered_map.h>

#include <cstdint>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <span>

#include "vecengine/distance.hpp"
#include "vecengine/flat_index.hpp"
#include "vecengine/fusion.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/hnsw_index.hpp"
#include "vecengine/metadata.hpp"
#include "vecengine/pq.hpp"
#include "vecengine/refine_index.hpp"
#include "vecengine/inverted_index.hpp"
#include "vecengine/sparse_approx_index.hpp"
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
static std::vector<Lock> lock_chain(const vecengine::IndexBase& index) {
    std::vector<Lock> locks;
    for (const vecengine::IndexBase* p = &index; p != nullptr;) {
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
static auto read_locked(const vecengine::IndexBase& index, F&& f) {
    nb::gil_scoped_release release;
    auto locks = lock_chain<std::shared_lock<std::shared_mutex>>(index);
    return f();
}

template <class F>
static auto write_locked(const vecengine::IndexBase& index, F&& f) {
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

using IdMatrix = nb::ndarray<int64_t, nb::c_contig, nb::device::cpu>;
using DistMatrix = nb::ndarray<float, nb::c_contig, nb::device::cpu>;

// qann.fuse: each result is (ids, dists) as query / batch_query return it, or
// ids alone (RRF only). All must be 1D (one query) or all 2D with the same row
// count. Returns (ids, scores) shaped like one query's or batch_query's
// results, with up to limit hits per row, padded with -1 / 0.
static nb::tuple fuse(nb::sequence results, const std::string& method_name, float k,
                      std::optional<std::vector<float>> weights, std::size_t limit) {
    vecengine::FusionMethod method;
    if (method_name == "rrf") method = vecengine::FusionMethod::RRF;
    else if (method_name == "linear") method = vecengine::FusionMethod::Linear;
    else throw std::invalid_argument("method must be 'rrf' or 'linear'; got '" + method_name + "'");
    const bool linear = method == vecengine::FusionMethod::Linear;
    if (limit == 0) throw std::invalid_argument("limit must be at least 1");

    nb::module_ np = nb::module_::import_("numpy");
    std::vector<IdMatrix> id_arrays;
    std::vector<DistMatrix> dist_arrays;
    for (nb::handle item : results) {
        nb::object ids = nb::borrow(item), dists;
        // (ids, dists): a pair whose first element is itself a sequence or array
        if (nb::isinstance<nb::tuple>(item) && nb::len(item) == 2 &&
            nb::cast<std::size_t>(np.attr("ndim")(item[0])) > 0) {
            ids = item[0];
            dists = item[1];
        }
        nb::object arr = np.attr("asarray")(ids);
        if (nb::cast<std::size_t>(arr.attr("size")) > 0) {
            const auto kind = nb::cast<std::string>(arr.attr("dtype").attr("kind"));
            if (kind != "i" && kind != "u") throw nb::type_error("result ids must be integers");
        }
        id_arrays.push_back(nb::cast<IdMatrix>(np.attr("ascontiguousarray")(arr, nb::arg("dtype") = "int64")));
        if (linear) {
            if (!dists.is_valid())
                throw std::invalid_argument("method='linear' needs (ids, dists) results, as query / batch_query return them");
            dist_arrays.push_back(nb::cast<DistMatrix>(np.attr("ascontiguousarray")(dists, nb::arg("dtype") = "float32")));
            const DistMatrix& d = dist_arrays.back();
            const IdMatrix& a = id_arrays.back();
            bool same = d.ndim() == a.ndim();
            for (std::size_t i = 0; same && i < a.ndim(); ++i) same = d.shape(i) == a.shape(i);
            if (!same) throw std::invalid_argument("each result's dists must have the shape of its ids");
        }
    }
    if (id_arrays.empty()) throw std::invalid_argument("fuse needs at least one result list");
    const std::size_t ndim = id_arrays[0].ndim();
    if (ndim != 1 && ndim != 2) throw std::invalid_argument("results must be 1D (one query) or 2D (one row per query)");
    const std::size_t rows = ndim == 1 ? 1 : id_arrays[0].shape(0);
    std::vector<std::span<const int64_t>> id_lists;
    std::vector<std::span<const float>> dist_lists;
    std::vector<std::size_t> widths;
    for (std::size_t i = 0; i < id_arrays.size(); ++i) {
        const IdMatrix& a = id_arrays[i];
        if (a.ndim() != ndim) throw std::invalid_argument("results must all be 1D or all be 2D");
        if (ndim == 2 && a.shape(0) != rows)
            throw std::invalid_argument("2D results must all have the same number of rows (queries)");
        widths.push_back(ndim == 1 ? a.shape(0) : a.shape(1));
        id_lists.emplace_back(a.data(), rows * widths.back());
        if (linear) dist_lists.emplace_back(dist_arrays[i].data(), rows * widths.back());
    }
    const std::vector<float> w = weights.value_or(std::vector<float>{});

    std::vector<std::vector<vecengine::FusedHit>> fused;
    {
        nb::gil_scoped_release release;
        fused = vecengine::fuse_batch(method, id_lists, dist_lists, widths, rows, w, k, limit);
    }
    std::vector<int64_t> ids(rows * limit, -1);
    std::vector<float> scores(rows * limit, 0.0f);
    for (std::size_t q = 0; q < rows; ++q)
        for (std::size_t j = 0; j < fused[q].size(); ++j) {
            ids[q * limit + j] = fused[q][j].id;
            scores[q * limit + j] = fused[q][j].score;
        }
    if (ndim == 1) return nb::make_tuple(to_numpy(std::move(ids), {limit}), to_numpy(std::move(scores), {limit}));
    return nb::make_tuple(to_numpy(std::move(ids), {rows, limit}), to_numpy(std::move(scores), {rows, limit}));
}

// Changing a RefineIndex's base directly would misalign the two (an add would
// give the base a vector the RefineIndex has no copy of), so it is refused.
// Called with the index locked, since a RefineIndex sets the flag under it.
static void refuse_if_wrapped(const vecengine::IndexBase& self, const char* what) {
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

using SqlParams = std::variant<std::vector<vecengine::Expr::Value>, std::unordered_map<std::string, vecengine::Expr::Value>>;

// where=: a qann.field filter, a SQL WHERE string, or (string, params) with
// params a list for ? or a dict for :name.
using WhereArg = std::variant<vecengine::Expr, std::string, std::pair<std::string, SqlParams>>;

static vecengine::Expr parse_sql(const std::string& sql, const std::optional<SqlParams>& params) {
    if (!params) return vecengine::Expr::parse(sql);
    return std::visit([&](const auto& p) { return vecengine::Expr::parse(sql, p); }, *params);
}

static std::optional<vecengine::Expr> to_expr(const std::optional<WhereArg>& where) {
    if (!where) return std::nullopt;
    if (auto* e = std::get_if<vecengine::Expr>(&*where)) return *e;
    if (auto* sql = std::get_if<std::string>(&*where)) return vecengine::Expr::parse(*sql);
    const auto& [sql, params] = std::get<std::pair<std::string, SqlParams>>(*where);
    return parse_sql(sql, params);
}

// Converts one metadata column of n values. Values go through numpy.asarray:
// integer and bool arrays become Int64 columns, float arrays Float64, and str
// arrays (or object arrays of str) String columns. None marks a null, and the
// other values decide the type. `per` says what each value belongs to, for
// errors. Must run with the GIL held.
static vecengine::MetadataColumn as_column(const std::string& name, nb::handle value, std::size_t n, const char* per) {
    nb::module_ np = nb::module_::import_("numpy");
    nb::object arr = np.attr("asarray")(value);
    if (nb::cast<std::size_t>(arr.attr("ndim")) != 1)
        throw nb::type_error(("metadata column '" + name + "' must be one-dimensional").c_str());
    if (nb::cast<std::size_t>(arr.attr("shape")[nb::int_(0)]) != n)
        throw std::invalid_argument("metadata column '" + name + "' must have one value per " + per);

    const auto kind = nb::cast<std::string>(arr.attr("dtype").attr("kind"));
    // numpy.asarray(["a", 1]) silently makes "1" a string, so check a list's own items
    nb::object items = nb::hasattr(value, "dtype") ? arr.attr("tolist")() : nb::borrow(value);
    if (kind == "O") {
        std::vector<uint8_t> valid;
        nb::list present;
        for (nb::handle item : items) {
            valid.push_back(!item.is_none());
            if (!item.is_none()) present.append(item);
        }
        if (std::find(valid.begin(), valid.end(), 0) != valid.end()) {
            const std::size_t m = nb::len(present);
            vecengine::MetadataColumn col = m ? as_column(name, present, m, per)
                                              : vecengine::MetadataColumn(name, std::vector<int64_t>{});
            std::visit([&](auto& v) {  // spread the m values over the n rows; nulls keep a default
                std::decay_t<decltype(v)> full(n);
                for (std::size_t i = 0, j = 0; i < n; ++i)
                    if (valid[i]) full[i] = std::move(v[j++]);
                v = std::move(full);
            }, col.values);
            col.valid = std::move(valid);
            return col;
        }
    }
    if (kind == "u" && nb::cast<std::size_t>(arr.attr("size")) > 0 &&
        nb::cast<uint64_t>(arr.attr("max")()) > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        throw std::invalid_argument("metadata column '" + name + "' has values above the int64 maximum");
    if (kind == "i" || kind == "u" || kind == "b") {
        auto ints = nb::cast<IdVector>(np.attr("ascontiguousarray")(arr, nb::arg("dtype") = "int64"));
        return {name, std::vector<int64_t>(ints.data(), ints.data() + ints.shape(0))};
    }
    if (kind == "f") {
        auto floats = nb::cast<nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>>(
            np.attr("ascontiguousarray")(arr, nb::arg("dtype") = "float64"));
        return {name, std::vector<double>(floats.data(), floats.data() + floats.shape(0))};
    }
    if (kind == "U" || kind == "O") {
        std::vector<std::string> strings;
        strings.reserve(n);
        for (nb::handle item : items) {
            if (!nb::isinstance<nb::str>(item))
                throw nb::type_error(("metadata column '" + name + "' mixes strings with other values").c_str());
            strings.push_back(nb::cast<std::string>(item));
        }
        return {name, std::move(strings)};
    }
    throw nb::type_error(("metadata column '" + name + "' must hold ints, floats or strings").c_str());
}

// Converts a metadata argument, {name: values}, to columns of n values each.
static vecengine::Metadata as_metadata(nb::dict metadata, std::size_t n, const char* per) {
    vecengine::Metadata out;
    for (auto [key, value] : metadata) {
        if (!nb::isinstance<nb::str>(key)) throw nb::type_error("metadata column names must be strings");
        out.push_back(as_column(nb::cast<std::string>(key), value, n, per));
    }
    return out;
}

// Add all rows of a (N x dim) matrix to the index, with one id per row when
// the index uses custom ids, and optional metadata columns. Returns the ids.
static NumpyArray<int64_t> index_add(vecengine::DenseIndex &self, FloatMatrix data, std::optional<IdsArg> ids,
                                     std::optional<nb::dict> metadata) {
    if (data.shape(1) != self.dim())
        throw std::invalid_argument("vector dimension does not match index dimension");

    std::optional<IdVector> id_array;
    if (ids) {
        id_array = as_ids(*ids);
        if (id_array->shape(0) != data.shape(0))
            throw std::invalid_argument("ids must have one entry per row of data");
    }

    const vecengine::Metadata md = metadata ? as_metadata(*metadata, data.shape(0), "row of data") : vecengine::Metadata{};
    const int64_t* id_ptr = id_array ? id_array->data() : nullptr;
    std::vector<int64_t> added = write_locked(self, [&] {
        refuse_if_wrapped(self, "add");
        return self.add_batch(std::span<const float>(data.data(), data.size()), data.shape(0), id_ptr, md);
    });
    const std::size_t n = added.size();
    return to_numpy(std::move(added), {n});
}

// Overwrite metadata columns of the vectors with these ids.
static void index_update_metadata(vecengine::IndexBase& self, IdsArg ids, nb::dict metadata) {
    IdVector id_array = as_ids(ids);
    const vecengine::Metadata md = as_metadata(metadata, id_array.shape(0), "id");
    write_locked(self, [&] {
        refuse_if_wrapped(self, "update metadata");
        self.update_metadata(std::span<const int64_t>(id_array.data(), id_array.shape(0)), md);
    });
}

// Read metadata back as {name: values}: int and float columns as NumPy masked
// arrays (int64 / float64, masked where null), str columns as lists with None
// for null. columns picks and orders the columns; None means all of them.
static nb::dict index_get_metadata(const vecengine::IndexBase& self, IdsArg ids,
                                   std::optional<std::vector<std::string>> columns) {
    IdVector id_array = as_ids(ids);
    vecengine::Metadata md = read_locked(self, [&] {
        return self.get_metadata(std::span<const int64_t>(id_array.data(), id_array.shape(0)),
                                 columns ? &*columns : nullptr);
    });
    nb::module_ ma = nb::module_::import_("numpy.ma");
    nb::dict out;
    for (vecengine::MetadataColumn& col : md) {
        const std::size_t n = col.valid.size();
        nb::object value;
        if (auto* strings = std::get_if<std::vector<std::string>>(&col.values)) {
            nb::list list;
            for (std::size_t i = 0; i < n; ++i)
                list.append(col.valid[i] ? nb::object(nb::str((*strings)[i].c_str(), (*strings)[i].size())) : nb::none());
            value = list;
        } else {
            nb::object data = std::holds_alternative<std::vector<int64_t>>(col.values)
                                  ? nb::cast(to_numpy(std::get<std::vector<int64_t>>(std::move(col.values)), {n}))
                                  : nb::cast(to_numpy(std::get<std::vector<double>>(std::move(col.values)), {n}));
            std::vector<uint8_t> mask(n);
            for (std::size_t i = 0; i < n; ++i) mask[i] = !col.valid[i];
            value = ma.attr("MaskedArray")(data, nb::arg("mask") = to_numpy(std::move(mask), {n}));
        }
        out[nb::str(col.name.c_str(), col.name.size())] = value;
    }
    return out;
}

// Add a metadata column; type is 'int', 'float' or 'str', as metadata_schema
// reports them. Either fill (one value for every vector) or ids and values.
static void index_add_metadata_column(vecengine::IndexBase& self, std::string name, const std::string& type,
                                      std::optional<vecengine::Expr::Value> fill, std::optional<IdsArg> ids,
                                      nb::object values) {
    vecengine::ColumnType t;
    if (type == "int") t = vecengine::ColumnType::Int64;
    else if (type == "float") t = vecengine::ColumnType::Float64;
    else if (type == "str") t = vecengine::ColumnType::String;
    else throw std::invalid_argument("metadata column type must be 'int', 'float' or 'str'; got '" + type + "'");
    if (ids.has_value() != !values.is_none()) throw nb::type_error("ids and values must be passed together");
    if (fill && ids) throw nb::type_error("pass either default or ids and values, not both");
    if (!ids) {
        write_locked(self, [&] {
            refuse_if_wrapped(self, "add a metadata column");
            self.add_metadata_column(std::move(name), t, std::move(fill));
        });
        return;
    }
    IdVector id_array = as_ids(*ids);
    vecengine::MetadataColumn col = as_column(name, values, id_array.shape(0), "id");
    write_locked(self, [&] {
        refuse_if_wrapped(self, "add a metadata column");
        self.add_metadata_column(std::move(name), t, std::span<const int64_t>(id_array.data(), id_array.shape(0)),
                                 std::move(col.values), std::move(col.valid));
    });
}

// Delete the vectors with these ids; returns how many were deleted.
static std::size_t index_remove(vecengine::IndexBase& self, IdsArg ids) {
    IdVector id_array = as_ids(ids);
    return write_locked(self, [&] {
        refuse_if_wrapped(self, "remove");
        return self.remove(std::span<const int64_t>(id_array.data(), id_array.shape(0)));
    });
}

// Free the storage of deleted vectors; returns how many slots were reclaimed.
static std::size_t index_compact(vecengine::IndexBase& self) {
    return write_locked(self, [&] {
        refuse_if_wrapped(self, "compact");
        return self.compact();
    });
}

static std::size_t index_size(const vecengine::IndexBase& self) {
    return read_locked(self, [&] { return self.size(); });
}

// One query's results as 1D (ids, distances) arrays (int64, float32).
static std::pair<NumpyArray<int64_t>, NumpyArray<float>> results_1d(const std::vector<vecengine::Neighbor>& results) {
    const std::size_t n = results.size();
    std::vector<int64_t> indices(n);
    std::vector<float> distances(n);
    for (std::size_t i = 0; i < n; ++i) {
        indices[i] = static_cast<int64_t>(results[i].index);
        distances[i] = results[i].distance;
    }
    return {to_numpy(std::move(indices), {n}), to_numpy(std::move(distances), {n})};
}

// Each query's results as (num_queries, k) arrays, rows padded with -1 / inf.
static std::pair<NumpyArray<int64_t>, NumpyArray<float>>
results_padded(const std::vector<std::vector<vecengine::Neighbor>>& results, std::size_t k) {
    const std::size_t num_queries = results.size();
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

// Query k-NN for a single vector, among the ids in allowed when given;
// returns (indices, distances) as 1D arrays with one entry per result (int64,
// float32).
static std::pair<NumpyArray<int64_t>, NumpyArray<float>>
index_query(const vecengine::DenseIndex &self, FloatVector query, std::size_t k, std::optional<IdsArg> allowed,
            std::optional<WhereArg> where_arg) {
    if (query.shape(0) != self.dim())
        throw std::invalid_argument("query dimension does not match index dimension");

    const std::optional<vecengine::Expr> where = to_expr(where_arg);
    std::optional<IdVector> allowed_ids;
    if (allowed) allowed_ids = as_ids(*allowed);
    auto results = read_locked(self, [&] {
        if (!allowed_ids && !where) return self.query(as_span(query), k);
        if (!where) return self.query(as_span(query), k, std::span<const int64_t>(allowed_ids->data(), allowed_ids->shape(0)));
        if (!allowed_ids) return self.query(as_span(query), k, *where);
        return self.query(as_span(query), k, *where,
                          std::span<const int64_t>(allowed_ids->data(), allowed_ids->shape(0)));
    });
    return results_1d(results);
}

// Query k-NN for each row, among the ids in allowed when given; returns
// (indices, distances) as (num_queries, k) arrays. Rows with fewer than k
// results are padded with -1 / inf.
static std::pair<NumpyArray<int64_t>, NumpyArray<float>>
index_batch_query(const vecengine::DenseIndex &self, FloatMatrix query, std::size_t k, std::optional<IdsArg> allowed,
                  std::optional<WhereArg> where_arg) {
    if (query.shape(1) != self.dim())
        throw std::invalid_argument("query dimension does not match index dimension");

    const std::size_t num_queries = query.shape(0);
    const std::optional<vecengine::Expr> where = to_expr(where_arg);
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
    return results_padded(results, k);
}

// A batch of sparse rows copied from Python, and the SparseRows view of it.
struct OwnedRows {
    std::vector<uint64_t> indptr;
    std::vector<uint32_t> indices;
    std::vector<float> values;

    vecengine::SparseRows view() const { return {indptr, indices, values}; }
};

// Converts a scipy.sparse matrix (or anything with format, indptr, indices
// and data; other formats go through tocsr()) to CSR rows: term ids must be
// in [0, 2^32), values become float32. Must run with the GIL held.
static OwnedRows as_sparse_rows(nb::handle obj, const char* what) {
    nb::object x = nb::borrow(obj);
    if (!nb::hasattr(x, "format") || !nb::hasattr(x, "tocsr"))
        throw nb::type_error((std::string(what) + " must be a scipy.sparse matrix, e.g. scipy.sparse.csr_array").c_str());
    if (nb::cast<std::string>(x.attr("format")) != "csr") x = x.attr("tocsr")();

    nb::module_ np = nb::module_::import_("numpy");
    auto as = [&](const char* name, const char* dtype) { return np.attr("ascontiguousarray")(x.attr(name), nb::arg("dtype") = dtype); };
    OwnedRows rows;
    auto indptr = nb::cast<IdVector>(as("indptr", "int64"));
    rows.indptr.assign(indptr.data(), indptr.data() + indptr.shape(0));
    auto indices = nb::cast<IdVector>(as("indices", "int64"));
    for (std::size_t i = 0; i < indices.shape(0); ++i) {
        if (indices.data()[i] < 0 || indices.data()[i] > static_cast<int64_t>(UINT32_MAX))
            throw std::invalid_argument(std::string(what) + " has a column index outside [0, 2^32)");
        rows.indices.push_back(static_cast<uint32_t>(indices.data()[i]));
    }
    auto values = nb::cast<FloatVector>(as("data", "float32"));
    rows.values.assign(values.data(), values.data() + values.shape(0));
    return rows;
}

// Add the rows of a sparse matrix, with ids and metadata as in DenseIndex.add.
static NumpyArray<int64_t> sparse_add(vecengine::SparseIndex& self, nb::handle data, std::optional<IdsArg> ids,
                                      std::optional<nb::dict> metadata) {
    const OwnedRows rows = as_sparse_rows(data, "data");
    std::optional<IdVector> id_array;
    if (ids) {
        id_array = as_ids(*ids);
        if (id_array->shape(0) != rows.view().rows()) throw std::invalid_argument("ids must have one entry per row of data");
    }
    const vecengine::Metadata md = metadata ? as_metadata(*metadata, rows.view().rows(), "row of data") : vecengine::Metadata{};
    std::vector<int64_t> added = write_locked(self, [&] {
        return self.add_batch(rows.view(), id_array ? id_array->data() : nullptr, md);
    });
    const std::size_t n = added.size();
    return to_numpy(std::move(added), {n});
}

// SparseIndex.query_batch with whichever of allowed and where are given.
static std::vector<std::vector<vecengine::Neighbor>>
sparse_search(const vecengine::SparseIndex& self, const OwnedRows& rows, std::size_t k, std::optional<IdsArg> allowed,
              std::optional<WhereArg> where_arg) {
    const std::optional<vecengine::Expr> where = to_expr(where_arg);
    std::optional<IdVector> allowed_ids;
    if (allowed) allowed_ids = as_ids(*allowed);
    return read_locked(self, [&] {
        if (!allowed_ids && !where) return self.query_batch(rows.view(), k);
        const std::span<const int64_t> ids = allowed_ids ? std::span<const int64_t>(allowed_ids->data(), allowed_ids->shape(0))
                                                         : std::span<const int64_t>();
        if (!where) return self.query_batch(rows.view(), k, ids);
        if (!allowed_ids) return self.query_batch(rows.view(), k, *where);
        return self.query_batch(rows.view(), k, *where, ids);
    });
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
    m.def("fuse", &fuse, nb::arg("results"), nb::kw_only(), nb::arg("method") = "rrf", nb::arg("k") = 60.0f,
          nb::arg("weights").none() = nb::none(), nb::arg("limit") = 10,
          "Merge ranked results from several retrievers into one ranking. results is a list of (ids, dists) "
          "as query / batch_query return them (or, for 'rrf', of id arrays), all 1D (one query) or all 2D with "
          "one row per query; row q of each is fused into row q. method 'rrf' (reciprocal rank fusion, the "
          "default) scores each id weight / (k + rank) per list holding it, rank counting from 1: positions "
          "only, no tuning needed. method 'linear' scores weight * (max - d) / (max - min), each list's "
          "distances min-max normalized per query over its returned hits: usually better than 'rrf' when "
          "weights are tuned on labeled queries. weights: one per list (default all 1); 0 ignores a list. "
          "-1 padding is skipped. Returns (ids, scores), highest score first (ties: smaller id first), up "
          "to limit per row, padded with -1 / 0. Fetch several times limit from each retriever, or good "
          "results never reach the fusion");

    // Methods bound on DenseIndex are inherited by FlatIndex, IVFIndex and RefineIndex,
    // and give type checkers the right methods for what load() and make_index() return.
    // Registered before the index classes, which use it as a default argument.
    nb::enum_<vecengine::Metric>(m, "Metric")
        .value("L2", vecengine::Metric::L2, "Squared Euclidean distance")
        .value("Cosine", vecengine::Metric::Cosine,
               "1 - cosine similarity (0 to 2); vectors are scaled to unit length on add and query")
        .value("InnerProduct", vecengine::Metric::InnerProduct,
               "Negative inner product -(a . b), so the largest inner product comes first");

    // Filters for where=: qann.field("price") < 50, combined with & | ~.
    nb::class_<vecengine::Expr>(m, "Expr", "A metadata filter for where=; build one with qann.field() or qann.where()")
        .def("__and__", [](const vecengine::Expr& a, const vecengine::Expr& b) { return a & b; }, nb::arg("other"))
        .def("__or__", [](const vecengine::Expr& a, const vecengine::Expr& b) { return a | b; }, nb::arg("other"))
        .def("__invert__", [](const vecengine::Expr& a) { return ~a; })
        // `and`, `or`, `not`, `if` and chained comparisons (0 < f < 5) would call this and
        // silently drop part of the filter, so refuse instead
        .def("__bool__", [](const vecengine::Expr&) -> bool {
            throw nb::type_error("a filter has no truth value: combine filters with & | ~ (not and / or / not), "
                                 "and write ranges as f.between(0, 5) or (f > 0) & (f < 5)");
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
        .def("between", [](const PyField& f, vecengine::Expr::Value lo, vecengine::Expr::Value hi) {
                 return vecengine::Expr::between(f.name, std::move(lo), std::move(hi));
             },
             nb::arg("lo"), nb::arg("hi"),
             "Matches rows with lo <= value <= hi, like SQL's BETWEEN: (field >= lo) & (field <= hi)")
        .def("__repr__", [](const PyField& f) { return "qann.field('" + f.name + "')"; });

    m.def("field", [](std::string name) { return PyField{std::move(name)}; }, nb::arg("name"),
          "A metadata column, for filters such as (qann.field('price') < 50) & (qann.field('color') == 'red')");

    m.def("where", &parse_sql, nb::arg("sql"), nb::arg("params").none() = nb::none(),
          "A filter from a SQL WHERE clause, e.g. \"category = 'shoes' AND price < ?\" with params [50]: "
          "comparisons (= != <> < <= > >=), [NOT] IN (...), IS [NOT] NULL, [NOT] BETWEEN x AND y, AND, OR, "
          "NOT and parentheses. Placeholders take values from params (a list for ?, a dict for :name), so "
          "user input never changes the filter's structure. NOT follows SQL: NOT (price < 50) excludes rows "
          "with no price, unlike ~. Combines with qann.field filters through & | ~; raises ValueError "
          "naming the column of a syntax error");

    m.def("any_of", [](std::vector<vecengine::Expr> filters) {
              if (filters.empty()) throw std::invalid_argument("any_of needs at least one filter");
              return vecengine::Expr::join(vecengine::Expr::Op::Or, std::move(filters));
          },
          nb::arg("filters"),
          "Matches rows matching any of filters: f1 | f2 | ..., built in one step, so a long chain costs "
          "O(n) where adding terms one | at a time costs O(n^2)");
    m.def("all_of", [](std::vector<vecengine::Expr> filters) {
              if (filters.empty()) throw std::invalid_argument("all_of needs at least one filter");
              return vecengine::Expr::join(vecengine::Expr::Op::And, std::move(filters));
          },
          nb::arg("filters"), "Matches rows matching all of filters: f1 & f2 & ..., built in one step, like any_of");

    m.def("get_max_filter_conditions", &vecengine::Expr::max_conditions,
          "The most conditions a SQL where= string may have (default 1024); see set_max_filter_conditions");
    m.def("set_max_filter_conditions", &vecengine::Expr::set_max_conditions, nb::arg("n"),
          "Set the most conditions (comparisons, IN, IS NULL, BETWEEN; an IN list counts once) a SQL where= "
          "string may have, for the whole process. Longer strings raise ValueError, so user input can't make "
          "a query scan its columns thousands of times. Filters built with qann.field aren't limited");

    nb::class_<vecengine::IndexBase>(m, "IndexBase", "What every index has: ids, deletion, metadata and saving")
        .def("save", [](const vecengine::IndexBase& self, const std::filesystem::path& path) {
                 read_locked(self, [&] { vecengine::save_index(self, path); });
             },
             nb::arg("path"), "Save the index to a file; load it back with qann.load(path)")
        .def("update_metadata", &index_update_metadata, nb::arg("ids"), nb::arg("metadata"),
             "Overwrite metadata in place: metadata is {name: one value per id}, None sets a value to null, and "
             "columns left out keep their values. Unknown, deleted or repeated ids, and columns the index "
             "doesn't have (see add_metadata_column), raise ValueError; a failed call changes nothing")
        .def("get_metadata", &index_get_metadata, nb::arg("ids"), nb::arg("columns").none() = nb::none(),
             "Read metadata back: {name: one value per id} for the columns named (all of them, in the order "
             "they were added, when columns is None). int and float columns are numpy.ma masked arrays (int64 / "
             "float64) masked where a value is null; str columns are lists with None for null. Unknown, deleted "
             "or repeated ids, and columns the index doesn't have or named twice, raise ValueError")
        .def("add_metadata_column", &index_add_metadata_column, nb::arg("name"), nb::arg("type"), nb::kw_only(),
             nb::arg("default").none() = nb::none(), nb::arg("ids").none() = nb::none(),
             nb::arg("values").none() = nb::none(),
             "Add a metadata column of type 'int', 'float' or 'str'. With default, every vector gets that value; "
             "with ids and values (one per id, None for null), those vectors get them and the rest are null; "
             "with neither, every vector is null. default can't be combined with ids and values (TypeError). "
             "Later adds that leave the column out store null. Raises ValueError, changing nothing, if the "
             "name is taken, a value doesn't fit the type, or an id is unknown, deleted or repeated")
        .def("drop_metadata_column", [](vecengine::IndexBase& self, const std::string& name) {
                 write_locked(self, [&] {
                     refuse_if_wrapped(self, "drop a metadata column");
                     self.drop_metadata_column(name);
                 });
             },
             nb::arg("name"), "Remove a metadata column and its values from every vector; raises ValueError if there is none")
        .def("remove", &index_remove, nb::arg("ids"),
             "Delete the vectors with these ids; unknown ids are skipped. Returns how many were deleted")
        .def("compact", &index_compact,
             "Free the memory of deleted vectors; ids and results are unchanged. Returns how many were freed")
        .def_prop_ro("metadata_schema",
                     [](const vecengine::IndexBase& self) {
                         auto schema = read_locked(self, [&] { return self.metadata_schema(); });
                         nb::dict out;
                         for (const auto& [name, type] : schema)
                             out[nb::str(name.c_str())] = type == vecengine::ColumnType::Int64     ? "int"
                                                          : type == vecengine::ColumnType::Float64 ? "float"
                                                                                                   : "str";
                         return out;
                     },
                     "Each metadata column's name and type ('int', 'float' or 'str'), in the order they were added")
        .def("ids",
             [](const vecengine::IndexBase& self) {
                 std::vector<int64_t> ids = read_locked(self, [&] { return self.ids(); });
                 const std::size_t n = ids.size();
                 return to_numpy(std::move(ids), {n});
             },
             "Every live vector's id (int64), in ascending order")
        .def("size", &index_size, "Get number of indexed vectors");

    nb::class_<vecengine::DenseIndex, vecengine::IndexBase>(m, "DenseIndex", "An index of dense float vectors")
        .def("add", &index_add, nb::arg("data"), nb::arg("ids").none() = nb::none(), nb::kw_only(),
             nb::arg("metadata").none() = nb::none(),
             "Add an (n, dim) array of vectors. Pass ids (n int64 values >= 0) if the index was "
             "created with custom_ids=True; otherwise ids continue after the highest id ever assigned. "
             "metadata is {name: n values} (ints, floats or strings) to filter queries on with where=; a new "
             "column is null for earlier vectors, and a column left out of an add is null for its vectors. "
             "None in a column's values is a null. Returns the vectors' ids (int64), the given ones or the "
             "assigned ones")
        .def("query", &index_query, nb::arg("query"), nb::arg("k"), nb::kw_only(), nb::arg("allowed").none() = nb::none(),
             nb::arg("where").none() = nb::none(),
             "Query k-NN for a (dim,) vector; returns 1D (indices, distances). With allowed (ids), only those "
             "vectors can be returned; unknown and deleted ids in it are ignored. With where (a qann.field filter, "
             "a SQL WHERE string as in qann.where, or (string, params)), only vectors whose metadata matches; "
             "with both, only those that pass both")
        .def("batch_query", &index_batch_query, nb::arg("query"), nb::arg("k"), nb::kw_only(),
             nb::arg("allowed").none() = nb::none(), nb::arg("where").none() = nb::none(),
             "Query k-NN for each row; returns (indices, distances) arrays of shape (num_queries, k), padded with "
             "-1 / inf. allowed (ids) and where (a filter, SQL string or (string, params)) restrict the results as "
             "in query(), for every row")
        .def_prop_ro("metric", &vecengine::DenseIndex::metric, "How vectors are compared (fixed at construction)")
        .def_prop_ro("stores_vectors", &vecengine::DenseIndex::stores_vectors,
                     "Whether get_vectors() works: the index keeps its vectors as floats, not only PQ or SQ codes")
        .def("get_vectors",
             [](const vecengine::DenseIndex& self, IdsArg ids) {
                 IdVector id_array = as_ids(ids);
                 std::vector<float> out = read_locked(self, [&] {
                     return self.get_vectors(std::span<const int64_t>(id_array.data(), id_array.shape(0)));
                 });
                 const std::size_t n = id_array.shape(0), dim = self.dim();
                 return to_numpy(std::move(out), {n, dim});
             },
             nb::arg("ids"),
             "A copy of the stored vectors with these ids, as an (n, dim) float32 array in the order given; "
             "with Metric.Cosine they are the unit-length copies the index stores. Rebuild into another index "
             "with new.add(old.get_vectors(old.ids())). Raises ValueError on an unknown, deleted or repeated "
             "id, or if the index stores only PQ or SQ codes (see stores_vectors)")
        .def("dim", &vecengine::DenseIndex::dim, "Get vector dimension");
    m.attr("Index") = m.attr("DenseIndex");  // the name before 0.1.0a24

    nb::class_<vecengine::FlatIndex, vecengine::DenseIndex>(m, "FlatIndex")
        .def("__init__", [](vecengine::FlatIndex* self, std::size_t dim, bool custom_ids, vecengine::Metric metric) {
                 new (self) vecengine::FlatIndex(dim, 1024, custom_ids, metric);
             },
             nb::arg("dim"), nb::kw_only(), nb::arg("custom_ids") = false, nb::arg("metric") = vecengine::Metric::L2,
             "Construct FlatIndex with vector dimension; custom_ids=True makes add() take your own ids");

    nb::enum_<vecengine::PQDistance>(m, "PQDistance")
        .value("ADC", vecengine::PQDistance::ADC)
        .value("SDC", vecengine::PQDistance::SDC);

    nb::class_<vecengine::IVFIndex, vecengine::DenseIndex>(m, "IVFIndex")
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
                     "True only once they exist: PQ enabled, trained, and within the 256 MB cap")
        .def_prop_rw("filter_probe_scaling",
                     [](const vecengine::IVFIndex& self) { return read_locked(self, [&] { return self.filter_probe_scaling(); }); },
                     [](vecengine::IVFIndex& self, bool v) { write_locked(self, [&] { self.set_filter_probe_scaling(v); }); },
                     "Whether filtered queries probe more lists as the filter narrows, so recall stays near an "
                     "unfiltered query's (on by default). Off is faster but loses recall under narrow filters. "
                     "Not saved: a loaded index has it on");

    nb::class_<vecengine::HNSWIndex, vecengine::DenseIndex>(m, "HNSWIndex")
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
        .def_prop_rw("filter_ef_scaling",
                     [](const vecengine::HNSWIndex& self) { return read_locked(self, [&] { return self.filter_ef_scaling(); }); },
                     [](vecengine::HNSWIndex& self, bool v) { write_locked(self, [&] { self.set_filter_ef_scaling(v); }); },
                     "Whether filtered queries narrow the search width as the filter narrows, to max(k, ef_search * "
                     "sqrt(fraction allowed)), keeping recall near an unfiltered query's at a fraction of the cost "
                     "(on by default). Off uses the full ef_search: slower, higher recall. Not saved: a loaded index has it on")
        .def_prop_ro("M", [](const vecengine::HNSWIndex& self) { return self.M(); }, "Neighbors per node (fixed at construction)")
        .def_prop_ro("ef_construction", [](const vecengine::HNSWIndex& self) { return self.ef_construction(); },
                     "Search width used while inserting (fixed at construction)");

    // Holds `base` by reference; keep_alive ties base's lifetime to the
    // RefineIndex so Python can't free it first.
    nb::class_<vecengine::RefineIndex, vecengine::DenseIndex>(m, "RefineIndex")
        // The constructor checks base is empty; lock base so a concurrent
        // base.add() can't race that check.
        .def("__init__", [](vecengine::RefineIndex* self, vecengine::DenseIndex& base, std::size_t k_factor, bool custom_ids) {
                 write_locked(base, [&] { new (self) vecengine::RefineIndex(base, k_factor, custom_ids); });
             },
             nb::arg("base"), nb::arg("k_factor") = 10, nb::kw_only(), nb::arg("custom_ids") = false,
             nb::keep_alive<1, 2>(),
             "Re-rank an empty, trained approximate index with exact distances. Add vectors through this wrapper.")
        .def_prop_rw("k_factor",
                     [](const vecengine::RefineIndex& self) { return read_locked(self, [&] { return self.k_factor(); }); },
                     [](vecengine::RefineIndex& self, std::size_t v) { write_locked(self, [&] { self.set_k_factor(v); }); },
                     "Candidates fetched from base per result (k * k_factor)")
        .def_prop_ro("base", [](vecengine::RefineIndex& self) -> vecengine::DenseIndex& { return self.base(); },
                     nb::rv_policy::reference_internal,
                     "The wrapped approximate index, e.g. to change nprobe on a loaded RefineIndex");

    nb::class_<vecengine::SparseIndex, vecengine::IndexBase>(
        m, "SparseIndex",
        "An index of sparse vectors (BM25, SPLADE), scored by dot product with distance -(q . d); see "
        "InvertedIndex")
        .def("add", &sparse_add, nb::arg("data"), nb::arg("ids").none() = nb::none(), nb::kw_only(),
             nb::arg("metadata").none() = nb::none(),
             "Add the rows of a scipy.sparse matrix (CSR, or converted with tocsr()); a column is a term id in "
             "[0, 2^32), values become float32, repeated entries in a row are summed and zeros dropped. ids and "
             "metadata as in DenseIndex.add. Returns the vectors' ids")
        .def("query",
             [](const vecengine::SparseIndex& self, nb::handle query, std::size_t k, std::optional<IdsArg> allowed,
                std::optional<WhereArg> where) {
                 const OwnedRows rows = as_sparse_rows(query, "query");
                 if (rows.view().rows() != 1) throw std::invalid_argument("query must be one row; use batch_query for more");
                 return results_1d(sparse_search(self, rows, k, std::move(allowed), std::move(where))[0]);
             },
             nb::arg("query"), nb::arg("k"), nb::kw_only(), nb::arg("allowed").none() = nb::none(),
             nb::arg("where").none() = nb::none(),
             "The k vectors with the largest dot product with a one-row sparse matrix, largest first; returns 1D "
             "(ids, distances) with distance -(q . d). Only vectors sharing a nonzero term with the query are "
             "returned. allowed and where filter as in DenseIndex.query")
        .def("batch_query",
             [](const vecengine::SparseIndex& self, nb::handle queries, std::size_t k, std::optional<IdsArg> allowed,
                std::optional<WhereArg> where) {
                 const OwnedRows rows = as_sparse_rows(queries, "queries");
                 return results_padded(sparse_search(self, rows, k, std::move(allowed), std::move(where)), k);
             },
             nb::arg("queries"), nb::arg("k"), nb::kw_only(), nb::arg("allowed").none() = nb::none(),
             nb::arg("where").none() = nb::none(),
             "query() for each row of a sparse matrix, in parallel; returns (ids, distances) arrays of shape "
             "(num_queries, k), padded with -1 / inf");

    nb::class_<vecengine::InvertedIndex, vecengine::SparseIndex>(
        m, "InvertedIndex", "Exact search over sparse vectors: a posting list per term, with MaxScore pruning")
        .def("__init__", [](vecengine::InvertedIndex* self, bool custom_ids) { new (self) vecengine::InvertedIndex(custom_ids); },
             nb::kw_only(), nb::arg("custom_ids") = false,
             "An empty InvertedIndex. custom_ids=True makes add() take ids, as for the dense indexes")
        .def_prop_rw("pruning",
                     [](const vecengine::InvertedIndex& self) { return read_locked(self, [&] { return self.pruning(); }); },
                     [](vecengine::InvertedIndex& self, bool on) { write_locked(self, [&] { self.set_pruning(on); }); },
                     "Whether queries skip vectors that can't reach the top k (MaxScore; default True). Results are "
                     "the same either way, up to which of several vectors tied at the k-th distance is kept. Not saved")
        .def_prop_ro("terms", [](const vecengine::InvertedIndex& self) { return read_locked(self, [&] { return self.terms(); }); },
                     "Distinct terms with a posting list")
        .def_prop_ro("postings",
                     [](const vecengine::InvertedIndex& self) { return read_locked(self, [&] { return self.postings(); }); },
                     "Stored (term, vector) pairs, including deleted vectors' until compact()");

    nb::class_<vecengine::SparseApproxIndex, vecengine::SparseIndex>(
        m, "SparseApproxIndex",
        "Approximate search over sparse vectors (SPLADE): the query's heaviest terms and each list's heaviest "
        "entries find candidates, re-scored exactly from a forward index")
        .def("__init__",
             [](vecengine::SparseApproxIndex* self, bool custom_ids, std::size_t query_terms, std::size_t list_depth,
                std::size_t k_factor) {
                 new (self) vecengine::SparseApproxIndex(custom_ids, query_terms, list_depth, k_factor);
             },
             nb::kw_only(), nb::arg("custom_ids") = false, nb::arg("query_terms") = 10, nb::arg("list_depth") = 10000,
             nb::arg("k_factor") = 10,
             "An empty SparseApproxIndex. custom_ids=True makes add() take ids. The search settings are the "
             "properties of the same names. It also keeps each vector's own terms, doubling the memory")
        .def_prop_rw("query_terms",
                     [](const vecengine::SparseApproxIndex& self) { return read_locked(self, [&] { return self.query_terms(); }); },
                     [](vecengine::SparseApproxIndex& self, std::size_t n) { write_locked(self, [&] { self.set_query_terms(n); }); },
                     "How many of the query's largest values to use (0: all; default 10). Saved with the index")
        .def_prop_rw("list_depth",
                     [](const vecengine::SparseApproxIndex& self) { return read_locked(self, [&] { return self.list_depth(); }); },
                     [](vecengine::SparseApproxIndex& self, std::size_t n) { write_locked(self, [&] { self.set_list_depth(n); }); },
                     "How many of each query term's heaviest list entries to read, plus any added since the list "
                     "was last sorted (0: all; default 10000). Saved with the index")
        .def_prop_rw("k_factor",
                     [](const vecengine::SparseApproxIndex& self) { return read_locked(self, [&] { return self.k_factor(); }); },
                     [](vecengine::SparseApproxIndex& self, std::size_t n) { write_locked(self, [&] { self.set_k_factor(n); }); },
                     "The k * k_factor best partial scores are re-scored exactly with the whole query (default 10; "
                     "at least 1). Saved with the index")
        .def_prop_ro("terms", [](const vecengine::SparseApproxIndex& self) { return read_locked(self, [&] { return self.terms(); }); },
                     "Distinct terms with a posting list")
        .def_prop_ro("postings",
                     [](const vecengine::SparseApproxIndex& self) { return read_locked(self, [&] { return self.postings(); }); },
                     "Stored (term, vector) pairs, including deleted vectors' until compact()");

    nb::enum_<vecengine::IndexType>(m, "IndexType")
        .value("Flat", vecengine::IndexType::Flat)
        .value("IVF", vecengine::IndexType::IVF)
        .value("HNSW", vecengine::IndexType::HNSW)
        .value("Inverted", vecengine::IndexType::Inverted)
        .value("SparseApprox", vecengine::IndexType::SparseApprox);

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
        .def_rw("pruning", &vecengine::IndexOptions::pruning)
        .def_rw("query_terms", &vecengine::IndexOptions::query_terms)
        .def_rw("list_depth", &vecengine::IndexOptions::list_depth)
        .def_rw("k_factor", &vecengine::IndexOptions::k_factor)
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
          "Construct a FlatIndex, IVFIndex, HNSWIndex, InvertedIndex or SparseApproxIndex from an IndexType and "
          "IndexOptions (dim is unused for the sparse ones)");

    m.def("load", nb::overload_cast<const std::filesystem::path&>(&vecengine::load_index), nb::arg("path"),
          nb::call_guard<nb::gil_scoped_release>(),
          "Load an index saved with save(path); returns a FlatIndex, IVFIndex, HNSWIndex, RefineIndex, InvertedIndex or "
          "SparseApproxIndex");
}
