#pragma once

#include <cstddef>
#include <vector>
#include <span>
#include <iosfwd>

namespace vecengine {

using std::size_t;
using std::vector;
using std::span;

struct Neighbor {
    size_t index;
    float distance;
};

// Common interface every index type implements.
// FlatIndex is exact brute-force; future types (IVF, HNSW) trade
// exactness for speed at scale but share this same surface.
//
// Invariant: add() assigns IDs in insertion order starting at 0, and
// Neighbor::index is that ID. RefineIndex relies on this to map a base
// index's results back to its own copy of the raw vectors.
class Index {
public:
    virtual ~Index() = default;

    virtual void add(span<const float> vec) = 0;
    virtual void add_batch(span<const float> vecs, size_t n);
    virtual vector<Neighbor> query(span<const float> vec, size_t k) const = 0;
    virtual vector<vector<Neighbor>> query_batch(span<const float> queries, size_t num_queries, size_t k) const;

    virtual void save(std::ostream& out) const = 0;

    virtual size_t size() const noexcept = 0;
    virtual size_t dim() const noexcept = 0;
};

} // namespace vecengine
