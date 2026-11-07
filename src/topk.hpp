#pragma once

#include "vecengine/index.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include <cassert>

namespace vecengine::detail {

using std::size_t;
using std::pair;
using std::vector;

// Keeps the k smallest (distance, id) pairs seen. Candidates are appended to
// a buffer of capacity 2k; when it fills, trim() keeps the best k and
// tightens the threshold. Callers filter with threshold() first, so most
// candidates cost one comparison and an accepted one costs one append.
// Of candidates tied at the k-th distance, the ones pushed first are kept, the
// same on every platform; take_sorted's final sort breaks ties by id.
class TopK {
public:
    // Keeps the best k; k must be >= 1.
    explicit TopK(size_t k) : k_(k) {
        assert(k > 0);

        dist_.reserve(2 * k_);
        id_.reserve(2 * k_);
    }

    // Distance a candidate must beat (strictly) to be kept; +inf until k
    // entries have been kept.
    float threshold() const { return worst_; }

    // How many entries are kept: at most k.
    size_t size() const { return std::min(dist_.size(), k_); }

    // Adds a candidate; requires dist < threshold().
    void push(float dist, size_t id) {
        assert(dist < worst_);

        dist_.push_back(dist);
        id_.push_back(id);
        if (dist_.size() >= 2 * k_) trim();
    }

    // The kept entries, nearest first. Leaves the TopK empty.
    vector<Neighbor> take_sorted() {
        if (dist_.size() > k_) trim();

        const size_t n = dist_.size();
        vector<Neighbor> results(n);
        if (std::all_of(id_.begin(), id_.end(), [](size_t id) { return id <= UINT32_MAX; })) {
            // key and id packed in one integer sort in the same order as the pairs,
            // about 2x faster to sort (one comparison each, without branches on libc++)
            vector<uint64_t> packed(n);
            for (size_t i = 0; i < n; ++i) packed[i] = uint64_t{key(dist_[i])} << 32 | id_[i];
            std::sort(packed.begin(), packed.end());
            for (size_t i = 0; i < n; ++i)
                results[i] = {static_cast<size_t>(packed[i] & UINT32_MAX), from_key(static_cast<uint32_t>(packed[i] >> 32))};
        } else {
            vector<pair<float, size_t>> buf(n);
            for (size_t i = 0; i < n; ++i) buf[i] = {dist_[i], id_[i]};
            std::sort(buf.begin(), buf.end());
            for (size_t i = 0; i < n; ++i) results[i] = {buf[i].second, buf[i].first};
        }
        dist_.clear();
        id_.clear();

        return results;
    }

private:
    // Maps a float to a uint32_t that sorts the same way.
    static uint32_t key(float f) {
        const uint32_t b = std::bit_cast<uint32_t>(f);
        return b ^ (static_cast<uint32_t>(static_cast<int32_t>(b) >> 31) | 0x80000000u);
    }

    // Inverse of key().
    static float from_key(uint32_t k) {
        return std::bit_cast<float>(k & 0x80000000u ? k ^ 0x80000000u : ~k);
    }

    // Keeps the k smallest entries, in push order. The k-th smallest key is
    // found one bit at a time, each bit a count over the keys that the compiler
    // vectorizes; on glove-25 fast-scan queries this was about 10x faster than
    // nth_element over the pairs, which was a fifth of the query.
    void trim() {
        const size_t n = dist_.size();
        keys_.resize(n);
        uint32_t lo = UINT32_MAX, hi = 0;
        for (size_t i = 0; i < n; ++i) {
            keys_[i] = key(dist_[i]);
            lo = std::min(lo, keys_[i]);
            hi = std::max(hi, keys_[i]);
        }
        auto count_le = [&](uint32_t t) {
            uint32_t c = 0;
            for (size_t i = 0; i < n; ++i) c += keys_[i] <= t;
            return c;
        };

        // t is the smallest key with at least k keys <= t; it shares the bits
        // above lo ^ hi's highest set bit with lo, so only the bits below are searched
        const int bits = std::bit_width(lo ^ hi);
        uint32_t t = bits == 32 ? 0 : lo & ~((1u << bits) - 1);
        for (int b = bits - 1; b >= 0; --b) {
            const uint32_t below = (1u << b) - 1;
            if (count_le(t | below) < k_) t |= 1u << b;
        }

        // every key below t, then keys equal to t in push order until there are k
        size_t ties = k_ - (t == 0 ? 0 : count_le(t - 1));
        size_t w = 0;
        for (size_t i = 0; i < n; ++i) {
            const bool tie = keys_[i] == t && ties > 0;
            dist_[w] = dist_[i];
            id_[w] = id_[i];
            w += keys_[i] < t || tie;
            ties -= tie;
        }
        dist_.resize(k_);
        id_.resize(k_);
        worst_ = from_key(t);
    }

    size_t k_;
    float worst_ = INFINITY;
    vector<float> dist_;
    vector<size_t> id_;
    vector<uint32_t> keys_;   // trim()'s scratch
};

} // namespace vecengine::detail
