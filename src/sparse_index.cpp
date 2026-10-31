#include "vecengine/sparse_index.hpp"
#include "parallel.hpp"
#include "serialize.hpp"
#include "topk.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace vecengine {

namespace {

using Term = std::pair<uint32_t, float>;

// Throws std::invalid_argument unless rows is well-formed CSR (see SparseRows).
void check_rows(const SparseRows& rows) {
    if (rows.indptr.empty()) throw std::invalid_argument("sparse rows need indptr with at least one entry");
    if (rows.indptr[0] != 0) throw std::invalid_argument("sparse indptr must start at 0");
    for (size_t i = 0; i < rows.rows(); ++i)
        if (rows.indptr[i + 1] < rows.indptr[i]) throw std::invalid_argument("sparse indptr must be non-decreasing");
    if (rows.indptr.back() != rows.indices.size() || rows.values.size() != rows.indices.size())
        throw std::invalid_argument("sparse indptr, indices and values must agree in length");
}

// Row i's terms sorted by term, repeats summed (in their original order) and
// zeros dropped.
void canonical_row(const SparseRows& rows, size_t i, vector<Term>& out) {
    out.clear();
    for (uint64_t j = rows.indptr[i]; j < rows.indptr[i + 1]; ++j) out.emplace_back(rows.indices[j], rows.values[j]);
    std::stable_sort(out.begin(), out.end(), [](const Term& a, const Term& b) { return a.first < b.first; });
    size_t w = 0;
    for (size_t r = 0; r < out.size(); ++r) {
        if (w > 0 && out[w - 1].first == out[r].first) out[w - 1].second += out[r].second;
        else out[w++] = out[r];
    }
    out.resize(w);
    std::erase_if(out, [](const Term& t) { return t.second == 0.0f; });
}

// Per-thread accumulators of search(), sized to the largest index searched
// and left all zero between queries.
struct Scratch {
    vector<float> scores;
    vector<uint8_t> seen;
    vector<uint32_t> touched;
    vector<Term> terms;
};

} // namespace

vector<int64_t> SparseIndex::add_batch(const SparseRows& rows, const int64_t* ids, const Metadata& metadata) {
    check_rows(rows);
    const size_t n = rows.rows();
    id_map_.check(n, ids);
    metadata_.check(n, metadata);

    const size_t first = slots();
    vector<Term> terms;
    for (size_t i = 0; i < n; ++i) {
        canonical_row(rows, i, terms);
        for (auto [term, value] : terms) lists_[term].push_back({static_cast<uint32_t>(first + i), value});
    }
    id_map_.add(n, ids);
    metadata_.append(first, n, metadata);

    vector<int64_t> added(n);
    for (size_t i = 0; i < n; ++i) added[i] = id_map_.label(static_cast<uint32_t>(first + i));
    return added;
}

vector<vector<Neighbor>> SparseIndex::query_batch(const SparseRows& queries, size_t k) const {
    return search(queries, k, nullptr);
}

vector<vector<Neighbor>> SparseIndex::query_batch(const SparseRows& queries, size_t k, span<const int64_t> allowed) const {
    const detail::SlotFilter filter = id_map_.filter(allowed);
    return search(queries, k, &filter);
}

vector<vector<Neighbor>> SparseIndex::query_batch(const SparseRows& queries, size_t k, const Expr& where) const {
    const detail::SlotFilter filter = where_filter(where, nullptr);
    return search(queries, k, &filter);
}

vector<vector<Neighbor>> SparseIndex::query_batch(const SparseRows& queries, size_t k, const Expr& where,
                                                  span<const int64_t> allowed) const {
    const detail::SlotFilter filter = where_filter(where, &allowed);
    return search(queries, k, &filter);
}

vector<vector<Neighbor>> SparseIndex::search(const SparseRows& queries, size_t k, const detail::SlotFilter* filter) const {
    check_rows(queries);
    const size_t n = queries.rows();
    vector<vector<Neighbor>> results(n);
    if (k == 0 || (filter && filter->count == 0)) return results;

    detail::parallel_for(n, [&](size_t i) {
        thread_local Scratch s;
        if (s.scores.size() < slots()) {
            s.scores.resize(slots(), 0.0f);
            s.seen.resize(slots(), 0);
        }
        // term at a time: add each query term's contribution to every vector in its list
        canonical_row(queries, i, s.terms);
        for (auto [term, value] : s.terms) {
            const auto it = lists_.find(term);
            if (it == lists_.end()) continue;
            for (const Posting& p : it->second) {
                if (!s.seen[p.slot]) {
                    s.seen[p.slot] = 1;
                    s.touched.push_back(p.slot);
                }
                s.scores[p.slot] += value * p.weight;
            }
        }
        detail::TopK top(k);
        for (uint32_t slot : s.touched) {
            const float dist = -s.scores[slot];
            const bool allowed = filter ? filter->test(slot) : !id_map_.is_deleted(slot);
            if (allowed && dist < top.threshold()) top.push(dist, static_cast<size_t>(id_map_.label(slot)));
            s.scores[slot] = 0.0f;
            s.seen[slot] = 0;
        }
        s.touched.clear();
        results[i] = top.take_sorted();
    });
    return results;
}

size_t SparseIndex::compact() {
    const vector<uint32_t> remap = id_map_.compact(true);
    if (remap.empty()) {
        metadata_.drop_unused_strings();
        return 0;
    }
    for (auto it = lists_.begin(); it != lists_.end();) {
        vector<Posting>& list = it->second;
        size_t w = 0;
        for (const Posting& p : list)  // remap keeps slot order, so the list stays sorted
            if (remap[p.slot] != detail::IdMap::kRemoved) list[w++] = {remap[p.slot], p.weight};
        list.resize(w);
        if (w == 0) {
            it = lists_.erase(it);
        } else {
            list.shrink_to_fit();
            ++it;
        }
    }
    metadata_.compact(remap);
    return remap.size() - id_map_.slots();
}

size_t SparseIndex::postings() const noexcept {
    size_t total = 0;
    for (const auto& [term, list] : lists_) total += list.size();
    return total;
}

void SparseIndex::save(std::ostream& out) const {
    using namespace detail;

    write_header(out, IndexKind::Sparse);
    write_pod(out, static_cast<uint64_t>(id_map_.slots()));
    write_pod(out, static_cast<uint64_t>(lists_.size()));
    vector<uint32_t> terms;  // sorted, so the same index always writes the same bytes
    terms.reserve(lists_.size());
    for (const auto& [term, list] : lists_) terms.push_back(term);
    std::sort(terms.begin(), terms.end());
    for (uint32_t term : terms) {
        write_pod(out, term);
        write_vec(out, lists_.at(term));
    }
    id_map_.save(out);
    metadata_.save(out);
}

std::unique_ptr<SparseIndex> SparseIndex::load_body(std::istream& in, uint32_t version) {
    using namespace detail;

    if (version < 9) throw std::runtime_error("corrupt file: SparseIndex in a format before version 9");
    const auto count = read_pod<uint64_t>(in);
    if (count > UINT32_MAX) throw std::runtime_error("corrupt file: invalid SparseIndex size");
    const auto nlists = read_pod<uint64_t>(in);

    auto index = std::make_unique<SparseIndex>();
    for (uint64_t l = 0; l < nlists; ++l) {
        const auto term = read_pod<uint32_t>(in);
        vector<Posting> list = read_vec<Posting>(in, count);
        if (list.empty()) throw std::runtime_error("corrupt file: empty SparseIndex posting list");
        for (size_t j = 0; j < list.size(); ++j)
            if (list[j].slot >= count || (j > 0 && list[j].slot <= list[j - 1].slot))
                throw std::runtime_error("corrupt file: SparseIndex posting list out of order");
        if (!index->lists_.emplace(term, std::move(list)).second)
            throw std::runtime_error("corrupt file: SparseIndex term appears twice");
    }
    index->id_map_ = IdMap::load(in, count, version);
    index->metadata_ = MetadataStore::load(in, count);
    return index;
}

} // namespace vecengine
