#pragma once

#include <cstddef>
#include <vector>
#include <span>
#include <iosfwd>
#include <shared_mutex>
#include <atomic>
#include <cstdint>

namespace vecengine {

namespace detail {
class IdMap;
struct SlotFilter;
}

using std::size_t;
using std::vector;
using std::span;

// One query result: the vector's id and its distance to the query under the
// index's metric (squared L2, or 1 - cosine similarity).
struct Neighbor {
    size_t index;
    float distance;
};

// How an index compares vectors. Values are part of the file format.
enum class Metric : uint8_t {
    L2 = 0,      // squared Euclidean distance
    Cosine = 1,  // 1 - cosine similarity, in [0, 2]: vectors are scaled to unit
                 // length on add and query, so squared L2 / 2 is the cosine distance
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

    // add, add_batch, query and query_batch prepare vectors for the metric
    // (scaling them to unit length for Cosine) exactly once, call the index's
    // *_impl, and turn squared L2 into the metric's distance on the way out.

    // Adds one vector of dim() floats. Throws std::invalid_argument on an
    // index with custom ids, since there is no id to give it.
    void add(span<const float> vec);

    // Adds n vectors (n * dim() floats, row-major). ids must hold n ids (>= 0,
    // unique) when the index uses custom ids and be nullptr otherwise. On a bad
    // id nothing is added. With Metric::Cosine, a zero vector throws
    // std::invalid_argument (it has no direction) and nothing is added.
    void add_batch(span<const float> vecs, size_t n, const int64_t* ids = nullptr);

    // Up to k nearest live vectors to vec, nearest first. Returns fewer than k
    // only when fewer are reachable (too few live vectors, or for IVF too few
    // in the probed lists).
    vector<Neighbor> query(span<const float> vec, size_t k) const;

    // query() for each of num_queries rows of queries, run in parallel.
    vector<vector<Neighbor>> query_batch(span<const float> queries, size_t num_queries, size_t k) const;

    // Filtered search: query() and query_batch() that return only vectors
    // whose id is in `allowed` (one set shared by every query of a batch).
    // Unknown, deleted and repeated ids in `allowed` are ignored; an empty set
    // returns no results. Fewer than k come back only when fewer than k
    // allowed vectors are live, or are reachable (HNSW). IVFIndex probes lists
    // beyond nprobe, nearest first, until it has found k allowed vectors.
    vector<Neighbor> query(span<const float> vec, size_t k, span<const int64_t> allowed) const;
    vector<vector<Neighbor>> query_batch(span<const float> queries, size_t num_queries, size_t k,
                                         span<const int64_t> allowed) const;

    // How this index compares vectors (fixed at construction).
    Metric metric() const noexcept { return metric_; }

    // Deletes the vectors with these ids; unknown and already deleted ids are
    // skipped. Returns how many were deleted. Their memory is kept, and
    // queries skip them.
    virtual size_t remove(span<const int64_t> ids) = 0;

    // Frees the storage of deleted vectors and returns how many slots were
    // reclaimed (0 when nothing is deleted). Ids stay the same: results,
    // remove() and later adds behave as before, with new ids continuing after
    // the highest ever assigned. Slots are renumbered, so slots() shrinks.
    virtual size_t compact() = 0;

    // Writes the index (header + body) so load_index() can rebuild it.
    virtual void save(std::ostream& out) const = 0;

    // Current slots: live vectors plus deleted ones not yet compacted away.
    virtual size_t slots() const noexcept = 0;

    // Whether every vector's id is its slot (auto ids, never compacted with
    // stable ids). RefineIndex requires this of its base.
    virtual bool ids_are_slots() const noexcept = 0;

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

    // Whether a RefineIndex currently wraps this index. While it does, the
    // Python add() and remove() refuse to run on this index directly: changes
    // must go through the RefineIndex to keep its slots aligned with this
    // index's. The C++ API doesn't enforce it, since RefineIndex itself calls
    // add_batch() and remove() here.
    bool wrapped() const noexcept { return wrapped_.load(); }

protected:
    explicit Index(Metric metric = Metric::L2) : metric_(metric) {}

    // A copy of the n rows of vecs (dim() floats each), each scaled to unit
    // length. Throws std::invalid_argument on a zero row. Used for
    // Metric::Cosine, by the public methods and by IVFIndex::train.
    vector<float> normalize_rows(span<const float> vecs, size_t n) const;

    Metric metric_; // set by the constructor, or by a load_body

    // Default: query_impl() for each row, in parallel. Protected so an
    // override can fall back to it.
    virtual vector<vector<Neighbor>> query_batch_impl(span<const float> queries, size_t num_queries, size_t k,
                                                      const detail::SlotFilter* filter) const;

private:
    friend class RefineIndex;  // sets and clears wrapped_; calls compact_as_base() and the *_impl methods

    // Shared by the public query() and query_batch() overloads: prepares the
    // queries for the metric, calls query_impl() (one query, batch false) or
    // query_batch_impl(), and converts distances. filter is nullptr for an
    // unfiltered query.
    vector<vector<Neighbor>> query_prepared(span<const float> queries, size_t num_queries, size_t k,
                                            const detail::SlotFilter* filter, bool batch) const;

    // This index's slot <-> id map, used to turn allowed ids into slots.
    virtual const detail::IdMap& id_map() const noexcept = 0;

    // What each index implements. Vectors arrive already prepared for the
    // metric, and distances go back as squared L2 between prepared vectors.
    // A non-null filter holds the live slots that may be returned (count > 0);
    // without one, every live slot may.
    virtual void add_batch_impl(span<const float> vecs, size_t n, const int64_t* ids) = 0;
    virtual vector<Neighbor> query_impl(span<const float> vec, size_t k, const detail::SlotFilter* filter) const = 0;

    // compact() for the base of a RefineIndex: same storage compaction, but
    // ids become the new slots so they keep matching the RefineIndex's slots.
    virtual size_t compact_as_base() = 0;

    mutable std::shared_mutex mutex_;
    std::atomic<bool> wrapped_{false};
};

} // namespace vecengine
