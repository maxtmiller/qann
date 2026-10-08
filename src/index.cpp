#include "vecengine/index.hpp"
#include "parallel.hpp"

#include <cstddef>
#include <vector>
#include <span>
#include <cassert>
#include <stdexcept>

namespace vecengine {

using std::size_t;
using std::vector;
using std::span;

void Index::add_batch(span<const float> vecs, size_t n, const int64_t* ids) {
    assert(vecs.size() == n * dim());
    if (ids != nullptr) throw std::invalid_argument("Index::add_batch: ids require an index created with custom_ids=true");

    for (size_t i = 0; i < n; ++i) add(span<const float>(vecs.data() + i * dim(), dim()));
}

vector<vector<Neighbor>> Index::query_batch(span<const float> queries, size_t num_queries, size_t k) const {
    assert(queries.size() == num_queries * dim());

    vector<vector<Neighbor>> results(num_queries);
    detail::parallel_for(num_queries, [&](size_t i) { results[i] = query(span<const float>(queries.data() + i * dim(), dim()), k); });

    return results;
}

}
