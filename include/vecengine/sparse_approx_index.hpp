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

// Approximate search over sparse vectors, for learned sparse vectors
// (SPLADE) whose queries have dozens of terms (issue #59). A query reads
// only its query_terms largest-valued terms, and in each of their posting
// lists only the list_depth heaviest entries; the k * k_factor vectors with
// the best partial scores are then re-scored exactly from a forward index
// (each vector's own terms). Each result's distance is its exact one, the
// same as InvertedIndex returns; only which vectors are found is approximate.
// The forward index doubles the memory of the posting lists.
class SparseApproxIndex : public SparseIndex {
public:
    // custom_ids makes add_batch() take user ids. The search settings are
    // as for the setters below.
    explicit SparseApproxIndex(bool custom_ids = false, size_t query_terms = 10, size_t list_depth = 10000,
                               size_t k_factor = 10);

    // IndexBase overrides; see IndexBase for the contract of each.
    size_t remove(span<const int64_t> ids) override { return id_map_.remove(ids); }
    size_t compact() override;
    void save(std::ostream& out) const override;
    size_t slots() const noexcept override { return id_map_.slots(); }
    bool ids_are_slots() const noexcept override { return id_map_.ids_are_slots(); }
    bool custom_ids() const noexcept override { return id_map_.custom_ids(); }
    size_t size() const noexcept override { return id_map_.live(); }

    // The search settings, used by every query until changed, and saved
    // with the index. query_terms: how many of the query's largest values
    // to use (0: all). list_depth: how many of each list's heaviest entries
    // to read, plus those added since it was last sorted (0: all).
    // k_factor: k * k_factor candidates are re-scored; set_k_factor throws
    // std::invalid_argument on 0. With query_terms and list_depth 0 and a
    // large k_factor, results are exact.
    size_t query_terms() const noexcept { return query_terms_; }
    size_t list_depth() const noexcept { return list_depth_; }
    size_t k_factor() const noexcept { return k_factor_; }
    void set_query_terms(size_t n) noexcept { query_terms_ = n; }
    void set_list_depth(size_t n) noexcept { list_depth_ = n; }
    void set_k_factor(size_t n);

    // Distinct terms with a posting list, and stored (term, vector) pairs,
    // including those of deleted vectors not yet compacted away.
    size_t terms() const noexcept { return lists_.size(); }
    size_t postings() const noexcept;

    // Rebuilds a SparseApproxIndex from the body save() wrote, after
    // load_index() has read the header. Throws std::runtime_error on corrupt
    // data.
    static std::unique_ptr<SparseApproxIndex> load_body(std::istream& in, uint32_t version);

private:
    struct Posting {
        uint32_t slot;
        float weight;
    };

    // A term's postings: postings[0, sorted) heaviest first (ties by slot),
    // then the ones added since, in slot order. The extremes of the weights
    // order the query's terms when re-scoring.
    struct List {
        vector<Posting> postings;
        size_t sorted = 0;
        float max_weight = 0;
        float min_weight = 0;

        // Sets max_weight and min_weight from postings.
        void bound();

        // Sorts the entries added since into the rest once there are more
        // than 64 of them and more than an eighth of the sorted ones (at
        // most 4096), so reading the heaviest entries stays cheap as single
        // vectors arrive: O(list) per merge.
        void settle();
    };

    const detail::IdMap& id_map() const noexcept override { return id_map_; }

    // SparseIndex implementations.
    void add_batch_impl(const SparseRows& rows, const int64_t* ids) override;
    vector<Neighbor> query_impl(const Terms& terms, size_t k, const detail::SlotFilter* filter) const override;

    // Fills the forward index from lists_, for load_body() and compact().
    void rebuild_forward();

    std::unordered_map<uint32_t, List> lists_;
    detail::IdMap id_map_;
    size_t query_terms_;
    size_t list_depth_;
    size_t k_factor_;
    // forward index: slot s's terms (increasing) and weights are at [fwd_ptr_[s], fwd_ptr_[s + 1])
    vector<uint64_t> fwd_ptr_{0};
    vector<uint32_t> fwd_terms_;
    vector<float> fwd_weights_;
};

} // namespace vecengine
