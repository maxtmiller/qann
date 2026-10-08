#pragma once

#include <cstddef>
#include <vector>
#include <span>
#include <iosfwd>
#include <shared_mutex>
#include <cstdint>

namespace vecengine {

using std::size_t;
using std::vector;
using std::span;

// One query result: the vector's id and its squared L2 distance to the query.
struct Neighbor {
    size_t index;
    float distance;
};

// Common interface every index type implements.
//
// Every added vector gets a slot: its position in insertion order (0, 1, 2,
// ...), which never changes. Without custom ids a vector's id is its slot;
// with custom ids (set at construction) each add supplies the ids, and
// Neighbor::index reports them. remove() only marks slots deleted, so slots
// stay stable; RefineIndex relies on that to keep its raw vectors aligned
// with its base's slots.
class Index {
public:
    virtual ~Index() = default;

    // Adds one vector of dim() floats. Throws std::invalid_argument on an
    // index with custom ids, since there is no id to give it.
    virtual void add(span<const float> vec) = 0;

    // Adds n vectors (n * dim() floats, row-major). ids must hold n ids (>= 0,
    // unique) when the index uses custom ids and be nullptr otherwise. On a bad
    // id nothing is added. The default implementation calls add() per vector.
    virtual void add_batch(span<const float> vecs, size_t n, const int64_t* ids = nullptr);

    // Up to k nearest live vectors to vec, nearest first. Returns fewer than k
    // only when fewer are reachable (too few live vectors, or for IVF too few
    // in the probed lists).
    virtual vector<Neighbor> query(span<const float> vec, size_t k) const = 0;

    // query() for each of num_queries rows of queries, run in parallel.
    virtual vector<vector<Neighbor>> query_batch(span<const float> queries, size_t num_queries, size_t k) const;

    // Deletes the vectors with these ids; unknown and already deleted ids are
    // skipped. Returns how many were deleted. Their memory is kept, and
    // queries skip them.
    virtual size_t remove(span<const int64_t> ids) = 0;

    // Writes the index (header + body) so load_index() can rebuild it.
    virtual void save(std::ostream& out) const = 0;

    // Vectors ever added, including deleted ones (the next slot number).
    virtual size_t slots() const noexcept = 0;

    // Whether add_batch() takes user ids (fixed at construction).
    virtual bool custom_ids() const noexcept = 0;

    // Number of live (not deleted) vectors.
    virtual size_t size() const noexcept = 0;

    // Floats per vector.
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
