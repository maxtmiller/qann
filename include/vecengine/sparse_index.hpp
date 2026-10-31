#pragma once

#include "vecengine/index.hpp"
#include "vecengine/id_map.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vecengine {

using std::size_t;
using std::span;
using std::vector;

// A batch of sparse vectors in CSR form: row i holds the (term, value) pairs
// indices[indptr[i] .. indptr[i + 1]) and values[...]. Terms are any uint32
// (a vocabulary id or a hashed feature); within a row they may come in any
// order and repeat (repeats are summed).
struct SparseRows {
    span<const uint64_t> indptr;  // rows() + 1 offsets, starting at 0, non-decreasing, ending at indices.size()
    span<const uint32_t> indices;
    span<const float> values;     // one per index

    size_t rows() const noexcept { return indptr.empty() ? 0 : indptr.size() - 1; }
};

// Exact inverted index over sparse vectors (issue #59), for keyword-style
// retrieval with BM25 or learned sparse weights (SPLADE) computed by the
// caller. Scores by inner product: a result's distance is -(q . d), so the
// largest dot product comes first, as with Metric::InnerProduct. A query
// reads only its own terms' posting lists, and returns only vectors that
// share at least one nonzero term with it. With pruning on (the default),
// queries skip vectors that can't reach the top k; results are the same.
//
// With a forward index, queries can instead run approximately, for learned
// sparse vectors whose queries have many terms: only the query's heaviest
// terms, only the heaviest entries of their lists, and the best candidates
// that finds re-scored exactly (see set_list_depth).
class SparseIndex : public IndexBase {
public:
    // custom_ids makes add_batch() take user ids. forward_index also keeps
    // each vector's own terms, which approximate queries need to re-score
    // candidates; it doubles the index's memory.
    explicit SparseIndex(bool custom_ids = false, bool forward_index = false)
        : id_map_(custom_ids), forward_(forward_index) {}

    // Adds rows.rows() vectors, with ids and metadata as in Index::add_batch.
    // Zero values are dropped. Throws std::invalid_argument, adding nothing,
    // on malformed rows (see SparseRows), bad ids or bad metadata. Returns the
    // vectors' ids, in order.
    vector<int64_t> add_batch(const SparseRows& rows, const int64_t* ids = nullptr, const Metadata& metadata = {});

    // Up to k vectors with the largest dot product with each query row,
    // largest first, run in parallel. Throws std::invalid_argument on
    // malformed rows.
    vector<vector<Neighbor>> query_batch(const SparseRows& queries, size_t k) const;

    // Filtered search, as in Index: only vectors whose id is in `allowed`,
    // whose metadata matches `where`, or both, can be returned.
    vector<vector<Neighbor>> query_batch(const SparseRows& queries, size_t k, span<const int64_t> allowed) const;
    vector<vector<Neighbor>> query_batch(const SparseRows& queries, size_t k, const Expr& where) const;
    vector<vector<Neighbor>> query_batch(const SparseRows& queries, size_t k, const Expr& where,
                                         span<const int64_t> allowed) const;

    // IndexBase overrides; see IndexBase for the contract of each.
    size_t remove(span<const int64_t> ids) override { return id_map_.remove(ids); }
    size_t compact() override;
    void save(std::ostream& out) const override;
    size_t slots() const noexcept override { return id_map_.slots(); }
    bool ids_are_slots() const noexcept override { return id_map_.ids_are_slots(); }
    bool custom_ids() const noexcept override { return id_map_.custom_ids(); }
    size_t size() const noexcept override { return id_map_.live(); }

    // Whether queries prune (default on; see search_one). Off scores every
    // vector in the query terms' lists; results are the same either way, up
    // to which of several vectors tied at the k-th distance is kept. Not saved.
    bool pruning() const noexcept { return pruning_; }
    void set_pruning(bool on) noexcept { pruning_ = on; }

    // Whether the index keeps a forward index (fixed at construction).
    bool forward_index() const noexcept { return forward_; }

    // Approximate search, used when query_terms or list_depth is nonzero
    // (both 0 by default: exact): each query reads only its query_terms
    // largest-valued terms (0: all), and in each of their lists only the
    // list_depth heaviest entries (0: all that are kept in weight order, at
    // most kHead) plus entries added since the list was last sorted. The
    // k * k_factor vectors with the best partial scores are then re-scored
    // exactly with the whole query, so each result's distance is its exact
    // one. The setters throw std::invalid_argument without a forward index
    // (for a nonzero value), for list_depth above kHead, or for k_factor 0.
    // Saved with the index.
    static constexpr size_t kHead = 16384;
    size_t query_terms() const noexcept { return query_terms_; }
    size_t list_depth() const noexcept { return list_depth_; }
    size_t k_factor() const noexcept { return k_factor_; }
    void set_query_terms(size_t n);
    void set_list_depth(size_t n);
    void set_k_factor(size_t n);

    // Distinct terms with a posting list, and stored (term, vector) pairs,
    // including those of deleted vectors not yet compacted away.
    size_t terms() const noexcept { return lists_.size(); }
    size_t postings() const noexcept;

    // Rebuilds a SparseIndex from the body save() wrote, after load_index()
    // has read the header. Throws std::runtime_error on corrupt data.
    static std::unique_ptr<SparseIndex> load_body(std::istream& in, uint32_t version);

private:
    struct Posting {
        uint32_t slot;
        float weight;
    };

    // A term's postings: postings[0, head) are the heaviest of
    // postings[0, sorted), heaviest first (ties by slot), at most kHead of
    // them; postings[head, end) are in slot order, so exact search reads long
    // lists almost sequentially. postings[sorted, end) were added since the
    // last split. The extremes of the weights bound what the term can add to
    // a score.
    struct List {
        vector<Posting> postings;
        size_t head = 0;
        size_t sorted = 0;
        float max_weight = 0;
        float min_weight = 0;

        // Sets max_weight and min_weight from postings.
        void bound();

        // Splits the list again once more than 64 entries, and more than an
        // eighth of the split ones (at most 4096), were added since the last
        // split, so reading the heaviest entries stays cheap as single vectors
        // arrive: the heaviest of the head and the new entries become the
        // head, and the rest merge into the slot-ordered part.
        void settle();
    };

    const detail::IdMap& id_map() const noexcept override { return id_map_; }

    using Terms = vector<std::pair<uint32_t, float>>;

    // query_batch() for every overload; filter is nullptr for an unfiltered query.
    vector<vector<Neighbor>> search(const SparseRows& queries, size_t k, const detail::SlotFilter* filter) const;

    // A query term's list, with the most (hi) and least (lo) it can add to a
    // score; both bound 0, a vector not in the list.
    struct QueryList {
        uint32_t term;
        float value;
        const List* list;
        double hi;
        double lo;
    };

    // The lists of the query's terms, in the order every search adds them:
    // largest hi first, ties to the shorter list. Summing in one order gives
    // the same float score whichever way a vector was found.
    vector<QueryList> query_lists(const Terms& terms) const;

    // One query (terms canonical, k >= 1), term at a time, its lists taken
    // from the largest possible contribution down. With pruning_, once the
    // lists left can't lift an unseen vector into the top k, they only add to
    // the vectors that can still get there (MaxScore's term-at-a-time form,
    // Turtle and Flood, 1995). Both ways sum each score in the same order, so
    // they return the same distances.
    vector<Neighbor> search_one(const Terms& terms, size_t k, const detail::SlotFilter* filter) const;

    // One approximate query (see set_list_depth).
    vector<Neighbor> search_approximate(const Terms& terms, size_t k, const detail::SlotFilter* filter) const;

    // Fills the forward index from lists_, for load_body() and compact().
    void rebuild_forward();

    std::unordered_map<uint32_t, List> lists_;
    detail::IdMap id_map_;
    bool pruning_ = true;
    bool forward_;
    size_t query_terms_ = 0;
    size_t list_depth_ = 0;
    size_t k_factor_ = 10;
    // forward index: slot s's terms (increasing) and weights are at [fwd_ptr_[s], fwd_ptr_[s + 1])
    vector<uint64_t> fwd_ptr_{0};
    vector<uint32_t> fwd_terms_;
    vector<float> fwd_weights_;
};

} // namespace vecengine
