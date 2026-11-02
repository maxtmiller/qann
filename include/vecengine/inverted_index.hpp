#pragma once

#include "vecengine/id_map.hpp"
#include "vecengine/sparse_index.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

namespace vecengine {

using std::size_t;
using std::span;
using std::vector;

// Exact search over sparse vectors: a posting list per term. A query reads
// only its own terms' lists, and returns only vectors that share at least
// one nonzero term with it. With pruning on (the default), queries skip
// vectors that can't reach the top k; results are the same.
class InvertedIndex : public SparseIndex {
public:
    // custom_ids makes add_batch() take user ids.
    explicit InvertedIndex(bool custom_ids = false) : id_map_(custom_ids) {}

    // IndexBase overrides; see IndexBase for the contract of each.
    size_t remove(span<const int64_t> ids) override { return id_map_.remove(ids); }
    size_t compact() override;
    void save(std::ostream& out) const override;
    size_t slots() const noexcept override { return id_map_.slots(); }
    bool ids_are_slots() const noexcept override { return id_map_.ids_are_slots(); }
    bool custom_ids() const noexcept override { return id_map_.custom_ids(); }
    size_t size() const noexcept override { return id_map_.live(); }

    // Whether queries prune (default on; see query_impl). Off scores every
    // vector in the query terms' lists; results are the same either way, up
    // to which of several vectors tied at the k-th distance is kept. Not saved.
    bool pruning() const noexcept { return pruning_; }
    void set_pruning(bool on) noexcept { pruning_ = on; }

    // Distinct terms with a posting list, and stored (term, vector) pairs,
    // including those of deleted vectors not yet compacted away.
    size_t terms() const noexcept { return lists_.size(); }
    size_t postings() const noexcept;

    // Rebuilds an InvertedIndex from the body save() wrote, after
    // load_index() has read the header. Throws std::runtime_error on corrupt
    // data.
    static std::unique_ptr<InvertedIndex> load_body(std::istream& in, uint32_t version);

private:
    struct Posting {
        uint32_t slot;
        float weight;
    };

    // A term's postings, in increasing slot order, and the extremes of their
    // weights, which bound what the term can add to a score.
    struct List {
        vector<Posting> postings;
        float max_weight = 0;
        float min_weight = 0;

        // Sets max_weight and min_weight from postings.
        void bound();
    };

    const detail::IdMap& id_map() const noexcept override { return id_map_; }

    // SparseIndex implementations.
    void add_batch_impl(const SparseRows& rows, const int64_t* ids) override;
    size_t compact_as_base() override { return compact_impl(false); }

    // compact() and compact_as_base(): keep_ids as in IdMap::compact.
    size_t compact_impl(bool keep_ids);

    // One query, term at a time, its lists taken
    // from the largest possible contribution down. With pruning_, once the
    // lists left can't lift an unseen vector into the top k, they only add to
    // the vectors that can still get there (MaxScore's term-at-a-time form,
    // Turtle and Flood, 1995), found by searching the lists instead of
    // reading them. Both ways sum each score in the same order, so they
    // return the same distances.
    vector<Neighbor> query_impl(const Terms& terms, size_t k, const detail::SlotFilter* filter) const override;

    std::unordered_map<uint32_t, List> lists_;
    detail::IdMap id_map_;
    bool pruning_ = true;
};

} // namespace vecengine
