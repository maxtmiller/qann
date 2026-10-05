#pragma once

#include "vecengine/index.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>
#include <cassert>

namespace vecengine::detail {

using std::size_t;
using std::pair;
using std::vector;

// Keeps the k smallest (distance, id) pairs seen. Candidates are appended to
// a buffer of capacity 2k; when it fills, nth_element keeps the best k and
// tightens the threshold. Callers filter with threshold() first, so most
// candidates cost one comparison and an accepted one costs one append.
class TopK {
public:
    explicit TopK(size_t k) : k_(k) {
        assert(k > 0);

        buf_.reserve(2 * k_);
    }

    // Distance a candidate must beat (strictly) to be kept; +inf until k
    // entries have been kept.
    float threshold() const { return worst_; }

    // Requires dist < threshold().
    void push(float dist, size_t id) {
        assert(dist < worst_);

        buf_.emplace_back(dist, id);
        if (buf_.size() >= 2 * k_) {
            std::nth_element(buf_.begin(), buf_.begin() + k_ - 1, buf_.end());
            buf_.resize(k_);
            worst_ = buf_[k_ - 1].first;
        }
    }

    // The kept entries, nearest first. Leaves the TopK empty.
    vector<Neighbor> take_sorted() {
        if (buf_.size() > k_) {
            std::nth_element(buf_.begin(), buf_.begin() + k_, buf_.end());
            buf_.resize(k_);
        }

        std::sort(buf_.begin(), buf_.end());
        vector<Neighbor> results;
        results.reserve(buf_.size());
        for (auto [dist, id] : buf_) results.emplace_back(id, dist);
        buf_.clear();

        return results;
    }

private:
    size_t k_;
    float worst_ = INFINITY;
    vector<pair<float, size_t>> buf_;
};

} // namespace vecengine::detail
