#pragma once

#include "vecengine/id_map.hpp"
#include "vecengine/sparse_index.hpp"
#include "vecengine/vector_store.hpp"

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
    // load_index() has read the header. With ctx->file set (format 11 on),
    // the posting lists are mapped from that file instead of read. Throws
    // std::runtime_error on corrupt data.
    static std::unique_ptr<InvertedIndex> load_body(std::istream& in, uint32_t version,
                                                    const detail::LoadContext* ctx = nullptr);

    // Whether the posting lists are read from a file (qann.load(path, mmap=True)) rather than held in RAM.
    bool mapped() const noexcept { return file_ != nullptr; }

private:
    using Posting = detail::SparsePosting;
    // A term's postings, in increasing slot order (mapped ones first: they hold the older slots).
    using List = detail::PostingList;

    const detail::IdMap& id_map() const noexcept override { return id_map_; }

    // SparseIndex implementations.
    void add_batch_impl(const SparseRows& rows, const int64_t* ids) override;
    size_t compact_as_base() override { return compact_impl(false); }

    // compact() and compact_as_base(): keep_ids as in IdMap::compact.
    size_t compact_impl(bool keep_ids);

    // compact_impl() for mapped lists, which can't be rewritten in place: writes the
    // compacted lists to a temporary file next to the loaded one and maps that instead.
    void compact_mapped(const vector<uint32_t>& remap);

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
    std::shared_ptr<const detail::MappedFile> file_; // what mapped lists point into
};

} // namespace vecengine
