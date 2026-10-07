#pragma once

#include <cstddef>
#include <vector>
#include <span>
#include <iosfwd>
#include <shared_mutex>

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

    // Not used by the library itself, which is not thread-safe for writes.
    // Callers sharing an index across threads (the Python bindings) take it
    // shared for queries and exclusive for add/train/setters; for a
    // RefineIndex they lock it first, then its base.
    std::shared_mutex& mutex() const noexcept { return mutex_; }

private:
    mutable std::shared_mutex mutex_;
};

} // namespace vecengine
