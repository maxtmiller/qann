#include "vecengine/flat_index.hpp"
#include "vecengine/distance.hpp"
#include <cstddef>
#include <vector>
#include <span>
#include <cassert>
#include <queue>


namespace vecengine {

using std::size_t;
using std::vector;
using std::span;
using std::pair;

FlatIndex::FlatIndex(size_t dim, size_t capacity): dim_(dim), count_(0)  {
    data_.reserve(dim_ * capacity);
}

void FlatIndex::add(span<const float> vec) {
    assert(vec.size() == dim_);
    data_.insert(data_.end(), vec.begin(), vec.end());
    ++count_;
}

vector<Neighbor> FlatIndex::query(span<const float> vec, size_t k) const {
    std::priority_queue<pair<float, size_t>, vector<pair<float, size_t>>, std::less<pair<float, size_t>>> maxHeap;
    for (size_t i = 0; i < count_; ++i) {
        const float* start = data_.data() + i * dim_;
        float dist = l2_distance(start, vec.data(), dim_);

        if (maxHeap.size() < k) {
            maxHeap.emplace(dist, i);
        } else if (dist < maxHeap.top().first) {
            maxHeap.pop();
            maxHeap.emplace(dist, i);
        }
    }

    const size_t n = maxHeap.size();
    vector<Neighbor> results(n);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        auto [dist, idx] = maxHeap.top();
        results[i] = {idx,dist};
        maxHeap.pop();
    }

    return results;
}

vector<vector<Neighbor>> FlatIndex::query_batch(span<const float> queries, size_t num_queries, size_t k) const {
    assert(queries.size() == num_queries * dim_);

    vector<vector<Neighbor>> results;
    results.reserve(num_queries);
    for (size_t i = 0; i < num_queries; ++i) {
        span<const float> row(queries.data() + i * dim_, dim_);
        results.emplace_back(query(row, k));
    }

    return results;
}


}
