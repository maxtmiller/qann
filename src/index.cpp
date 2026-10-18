#include "vecengine/index.hpp"
#include "vecengine/id_map.hpp"
#include "parallel.hpp"

#include <cstddef>
#include <vector>
#include <span>
#include <cassert>
#include <stdexcept>
#include <cmath> 

namespace vecengine {

using std::size_t;
using std::vector;
using std::span;

namespace {

// Squared L2 between unit vectors is 2 * (1 - cos), so halving it gives the
// cosine distance.
void l2_to_cosine(vector<Neighbor>& results) {
    for (Neighbor& nb : results) nb.distance *= 0.5f;
}

} // namespace

void Index::add(span<const float> vec) {
    add_batch(vec, 1, nullptr);
}

void Index::add_batch(span<const float> vecs, size_t n, const int64_t* ids, const Metadata& metadata) {
    assert(vecs.size() == n * dim());
    // validate metadata first and append it only once the vectors are in, so a
    // throw anywhere leaves both unchanged
    metadata_.check(n, metadata);
    const size_t rows = slots();
    if (metric_ == Metric::L2) {
        add_batch_impl(vecs, n, ids);
    } else {
        const vector<float> unit = normalize_rows(vecs, n);
        add_batch_impl(unit, n, ids);
    }
    metadata_.append(rows, n, metadata);
}

vector<Neighbor> Index::query(span<const float> vec, size_t k) const {
    assert(vec.size() == dim());
    return std::move(query_prepared(vec, 1, k, nullptr, false)[0]);
}

vector<vector<Neighbor>> Index::query_batch(span<const float> queries, size_t num_queries, size_t k) const {
    assert(queries.size() == num_queries * dim());
    return query_prepared(queries, num_queries, k, nullptr, true);
}

vector<Neighbor> Index::query(span<const float> vec, size_t k, span<const int64_t> allowed) const {
    assert(vec.size() == dim());
    const detail::SlotFilter filter = id_map().filter(allowed);
    return std::move(query_prepared(vec, 1, k, &filter, false)[0]);
}

vector<vector<Neighbor>> Index::query_batch(span<const float> queries, size_t num_queries, size_t k,
                                            span<const int64_t> allowed) const {
    assert(queries.size() == num_queries * dim());
    const detail::SlotFilter filter = id_map().filter(allowed);
    return query_prepared(queries, num_queries, k, &filter, true);
}

vector<Neighbor> Index::query(span<const float> vec, size_t k, const Expr& where) const {
    assert(vec.size() == dim());
    const detail::SlotFilter filter = where_filter(where, nullptr);
    return std::move(query_prepared(vec, 1, k, &filter, false)[0]);
}

vector<Neighbor> Index::query(span<const float> vec, size_t k, const Expr& where, span<const int64_t> allowed) const {
    assert(vec.size() == dim());
    const detail::SlotFilter filter = where_filter(where, &allowed);
    return std::move(query_prepared(vec, 1, k, &filter, false)[0]);
}

vector<vector<Neighbor>> Index::query_batch(span<const float> queries, size_t num_queries, size_t k,
                                            const Expr& where) const {
    assert(queries.size() == num_queries * dim());
    const detail::SlotFilter filter = where_filter(where, nullptr);
    return query_prepared(queries, num_queries, k, &filter, true);
}

vector<vector<Neighbor>> Index::query_batch(span<const float> queries, size_t num_queries, size_t k,
                                            const Expr& where, span<const int64_t> allowed) const {
    assert(queries.size() == num_queries * dim());
    const detail::SlotFilter filter = where_filter(where, &allowed);
    return query_prepared(queries, num_queries, k, &filter, true);
}

detail::SlotFilter Index::where_filter(const Expr& where, const span<const int64_t>* allowed) const {
    vector<uint64_t> words = metadata_.evaluate(where, slots());
    if (allowed) {
        const vector<uint64_t> also = id_map().filter(*allowed).words;
        for (size_t w = 0; w < words.size(); ++w) words[w] &= also[w];
    }
    return id_map().live(std::move(words));
}

vector<vector<Neighbor>> Index::query_prepared(span<const float> queries, size_t num_queries, size_t k,
                                               const detail::SlotFilter* filter, bool batch) const {
    if (filter && filter->count == 0) return vector<vector<Neighbor>>(num_queries);

    auto run = [&](span<const float> q) {
        if (!batch) return vector<vector<Neighbor>>{query_impl(q, k, filter)};
        return query_batch_impl(q, num_queries, k, filter);
    };
    if (metric_ == Metric::L2) return run(queries);

    const vector<float> unit = normalize_rows(queries, num_queries);
    vector<vector<Neighbor>> results = run(unit);
    for (auto& row : results) l2_to_cosine(row);
    return results;
}

vector<vector<Neighbor>> Index::query_batch_impl(span<const float> queries, size_t num_queries, size_t k,
                                                 const detail::SlotFilter* filter) const {
    vector<vector<Neighbor>> results(num_queries);
    detail::parallel_for(num_queries, [&](size_t i) { results[i] = query_impl(queries.subspan(i * dim(), dim()), k, filter); });
    return results;
}

vector<float> Index::normalize_rows(span<const float> vecs, size_t n) const {
    size_t d = dim();
    vector<float> res;
    res.resize(n * d);

    for (size_t i = 0; i < n; ++i) {
        double sqr_sum = 0;
        for (size_t j = 0; j < d; ++j) {
            float curr = vecs[i * d + j];
            sqr_sum += double(curr) * curr;
        }
        double length = std::sqrt(sqr_sum);
        if (length == 0) throw std::invalid_argument("cosine metric needs non-zero vectors");
        float scale = static_cast<float>(1.0 / length);
        for (size_t j = 0; j < d; ++j) {
            res[i * d + j] = vecs[i * d + j] * scale;
        }
    }
    
    return res;
}

}
