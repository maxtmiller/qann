#pragma once

#include "vecengine/index.hpp"

#include <cstddef>
#include <vector>
#include <span>
#include <iosfwd>
#include <memory>

namespace vecengine {

using std::size_t;
using std::vector;
using std::span;

// Re-ranks an approximate index's results with exact distances.
// query() asks `base` for k * k_factor candidates, re-scores each against
// the raw vector stored here, and returns the exact top k.
//
// Only useful when `base` ranks with approximate distances (e.g. IVFIndex
// with PQ). Holds `base` by reference: the caller keeps it alive, trains it
// before the first add(), and can keep tuning it (nprobe, pq_distance).
class RefineIndex : public Index {
public:
    // base must be empty so IDs line up with this index's raw storage.
    explicit RefineIndex(Index& base, size_t k_factor = 10);

    void add(span<const float> vec) override;
    void add_batch(span<const float> vecs, size_t n) override;
    vector<Neighbor> query(span<const float> vec, size_t k) const override;

    void save(std::ostream& out) const override;
    static std::unique_ptr<RefineIndex> load_body(std::istream& in);

    size_t size() const noexcept override { return count_; }
    size_t dim() const noexcept override { return base_.dim(); }
    
    Index& base() noexcept { return base_; }
    const Index& base() const noexcept { return base_; }

    size_t k_factor() const noexcept { return k_factor_; }
    void set_k_factor(size_t k_factor);

private:
    RefineIndex(std::unique_ptr<Index> base, size_t k_factor)
        : owned_base_(std::move(base)), base_(*owned_base_), k_factor_(k_factor) {}

    std::unique_ptr<Index> owned_base_; // set on file load; nullptr when wrapping a caller-owned base.
    Index& base_; // approximate first-stage index
    size_t k_factor_; // candidates fetched per result: k * k_factor
    size_t count_ = 0; // num of vectors
    vector<float> data_; // raw vectors, row-major: count_ * dim
};

} // namespace vecengine
