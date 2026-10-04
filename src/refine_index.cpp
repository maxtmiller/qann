#include "vecengine/refine_index.hpp"
#include "vecengine/distance.hpp"

#include <cstddef>
#include <vector>
#include <span>
#include <cassert>
#include <stdexcept>
#include <queue>
#include <utility>
#include <algorithm>

namespace vecengine {

using std::size_t;
using std::vector;
using std::span;
using std::pair;

RefineIndex::RefineIndex(Index& base, size_t k_factor) : base_(base), k_factor_(k_factor) {
    if (base.size() != 0) throw std::invalid_argument("base index must be empty");
    if (k_factor == 0) throw std::invalid_argument("k_factor must be >= 1");
}

void RefineIndex::set_k_factor(size_t k_factor) {
    if (k_factor == 0) throw std::invalid_argument("k_factor must be >= 1");
    k_factor_ = k_factor;
}

void RefineIndex::add(span<const float> vec) {
    assert(vec.size() == dim());

    base_.add(vec);
    data_.insert(data_.end(), vec.begin(), vec.end());
    ++count_;

}

vector<Neighbor> RefineIndex::query(span<const float> vec, size_t k) const {
    assert(vec.size() == dim());

    vector<Neighbor> candidatesRaw = base_.query(vec, std::min(k * k_factor_, count_));
    vector<pair<float, size_t>> candidates;
    candidates.reserve(candidatesRaw.size());

    for (size_t i = 0; i < candidatesRaw.size(); ++i) {
        float dst = l2_distance(vec.data(), data_.data() + (candidatesRaw[i].index * dim()), dim());
        candidates.emplace_back(dst, candidatesRaw[i].index);
    }

    std::priority_queue<pair<float, size_t>, vector<pair<float, size_t>>, std::less<pair<float, size_t>>> maxHeap(candidates.begin(), candidates.end());
    while (maxHeap.size() > k) maxHeap.pop();

    const size_t n = maxHeap.size();
    vector<Neighbor> results(n);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        auto [distance, index] = maxHeap.top();
        results[i] = {index, distance};
        maxHeap.pop();
    }

    return results;
}

vector<vector<Neighbor>> RefineIndex::query_batch(span<const float> queries, size_t num_queries, size_t k) const {
    assert(queries.size() == num_queries * dim());

    vector<vector<Neighbor>> results;
    results.reserve(num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
        span<const float> row(queries.data() + i * dim(), dim());
        results.emplace_back(query(row, k));
    }

    return results;
}

} // namespace vecengine
