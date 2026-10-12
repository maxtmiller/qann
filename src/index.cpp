#include "vecengine/index.hpp"
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

void Index::add_batch(span<const float> vecs, size_t n, const int64_t* ids) {
    assert(vecs.size() == n * dim());
    if (metric_ == Metric::L2) {
        add_batch_impl(vecs, n, ids);
        return;
    }
    const vector<float> unit = normalize_rows(vecs, n);
    add_batch_impl(unit, n, ids);
}

vector<Neighbor> Index::query(span<const float> vec, size_t k) const {
    assert(vec.size() == dim());
    if (metric_ == Metric::L2) return query_impl(vec, k);

    const vector<float> unit = normalize_rows(vec, 1);
    vector<Neighbor> results = query_impl(unit, k);
    l2_to_cosine(results);
    return results;
}

vector<vector<Neighbor>> Index::query_batch(span<const float> queries, size_t num_queries, size_t k) const {
    assert(queries.size() == num_queries * dim());
    if (metric_ == Metric::L2) return query_batch_impl(queries, num_queries, k);

    const vector<float> unit = normalize_rows(queries, num_queries);
    vector<vector<Neighbor>> results = query_batch_impl(unit, num_queries, k);
    for (auto& row : results) l2_to_cosine(row);
    return results;
}

vector<vector<Neighbor>> Index::query_batch_impl(span<const float> queries, size_t num_queries, size_t k) const {
    vector<vector<Neighbor>> results(num_queries);
    detail::parallel_for(num_queries, [&](size_t i) { results[i] = query_impl(queries.subspan(i * dim(), dim()), k); });
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
