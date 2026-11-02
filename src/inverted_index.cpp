#include "vecengine/inverted_index.hpp"
#include "serialize.hpp"
#include "topk.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace vecengine {

namespace {

// Per-thread accumulators of query_impl(), sized to the largest index
// searched and left all zero between queries.
struct Scratch {
    vector<float> scores;
    vector<uint8_t> seen;  // kSeen, kKept while the slot is in the k-th score heap, kCandidate
    vector<uint32_t> touched;
    vector<uint32_t> candidates;
    vector<std::pair<double, uint32_t>> kept;  // min-heap of (lower bound, slot)
};

constexpr uint8_t kSeen = 1, kKept = 2, kCandidate = 4;
constexpr size_t kPruneRatio = 128;  // see InvertedIndex::query_impl

// The first posting in [at, end) with a slot >= slot: an exponential search
// from at, so a run of lookups in increasing slot order costs about
// log(gap) each.
template <class Posting>
const Posting* seek(const Posting* at, const Posting* end, uint32_t slot) {
    if (at == end || at->slot >= slot) return at;
    const size_t n = static_cast<size_t>(end - at);
    size_t hi = 1;
    while (hi < n && at[hi].slot < slot) hi *= 2;
    return std::lower_bound(at + hi / 2 + 1, at + std::min(hi + 1, n), slot,
                            [](const Posting& p, uint32_t s) { return p.slot < s; });
}

} // namespace

void InvertedIndex::add_batch_impl(const SparseRows& rows, const int64_t* ids) {
    const size_t n = rows.rows();
    id_map_.check(n, ids);
    const size_t first = slots();
    Terms terms;
    for (size_t i = 0; i < n; ++i) {
        canonical_row(rows, i, terms);
        for (auto [term, value] : terms) {
            List& list = lists_[term];
            if (list.postings.empty()) list.max_weight = list.min_weight = value;
            list.max_weight = std::max(list.max_weight, value);
            list.min_weight = std::min(list.min_weight, value);
            list.postings.push_back({static_cast<uint32_t>(first + i), value});
        }
    }
    id_map_.add(n, ids);
}

vector<Neighbor> InvertedIndex::query_impl(const Terms& terms, size_t k, const detail::SlotFilter* filter) const {
    // the query's lists, by the most each can add to a score (hi), largest
    // first; lo is the least (both bound 0, a vector not in the list)
    struct QueryList {
        const vector<Posting>* postings;
        float value;
        double hi;
        double lo;
    };
    vector<QueryList> lists;
    double mass = 0;  // the sum of |contribution| bounds, for the rounding margin
    size_t postings = 0;  // in the query's lists
    for (auto [term, value] : terms) {
        const auto it = lists_.find(term);
        if (it == lists_.end()) continue;
        const double a = double(value) * it->second.max_weight, b = double(value) * it->second.min_weight;
        lists.push_back({&it->second.postings, value, std::max({0.0, a, b}), std::min({0.0, a, b})});
        mass += std::max(std::abs(a), std::abs(b));
        postings += it->second.postings.size();
    }
    // ties go to the shorter list, leaving long ones for last, where pruning skips them
    std::stable_sort(lists.begin(), lists.end(), [](const QueryList& x, const QueryList& y) {
        return x.hi != y.hi ? x.hi > y.hi : x.postings->size() < y.postings->size();
    });
    // bounds and postings of lists[j, end) summed
    vector<double> rest_hi(lists.size() + 1, 0.0), rest_lo(lists.size() + 1, 0.0);
    vector<size_t> rest_postings(lists.size() + 1, 0);
    for (size_t j = lists.size(); j-- > 0;) {
        rest_hi[j] = rest_hi[j + 1] + lists[j].hi;
        rest_lo[j] = rest_lo[j + 1] + lists[j].lo;
        rest_postings[j] = rest_postings[j + 1] + lists[j].postings->size();
    }
    // Scores are float sums of up to lists.size() products, so every bound
    // is widened by this much: rounding never prunes a vector the full
    // scoring would keep.
    const double margin = 2.0 * mass * double(lists.size() + 2) * 1.2e-7;

    thread_local Scratch scratch;
    Scratch& s = scratch;  // one thread-local lookup, not one per use (each is a call on macOS)
    if (s.scores.size() < slots()) {
        s.scores.resize(slots(), 0.0f);
        s.seen.resize(slots(), 0);
    }
    auto allowed = [&](uint32_t slot) { return filter ? filter->test(slot) : !id_map_.is_deleted(slot); };

    // kth: a lower bound on the k-th best final score among allowed
    // vectors, from a min-heap of k distinct ones and a lower bound on each
    // (its partial score when it entered, plus the least the lists after that
    // can add), less the margin. Entries aren't raised as scores grow, so it
    // lags a little, but every entry stays a valid bound. -inf until k are in.
    auto& kept = s.kept;
    kept.clear();
    double kth = -std::numeric_limits<double>::infinity();
    const auto heap_order = [](const std::pair<double, uint32_t>& x, const std::pair<double, uint32_t>& y) {
        return x.first > y.first;
    };
    // Called when lower may beat the heap's minimum (always while it isn't full).
    auto offer = [&](uint32_t slot, double lower) {
        if ((s.seen[slot] & kKept) || !allowed(slot)) return;
        if (kept.size() == k) {
            if (lower <= kept.front().first) return;
            std::pop_heap(kept.begin(), kept.end(), heap_order);
            s.seen[kept.back().second] &= ~kKept;
            kept.pop_back();
        }
        kept.emplace_back(lower, slot);
        std::push_heap(kept.begin(), kept.end(), heap_order);
        s.seen[slot] |= kKept;
        if (kept.size() == k) kth = kept.front().first - margin;
    };

    // Read whole lists while a vector not yet seen could still make the top
    // k. Past that, the rest only add to candidates: vectors that could still
    // reach kth. Keeping kth costs more than pruning saves unless k is small
    // next to the vectors the lists reach (k <= postings / 128 measured best
    // on BM25 over BEIR SciFact and FiQA), so larger k reads every list.
    // Collecting the candidates costs a pass over the vectors seen, so it
    // stops being worth it once fewer postings are left than that; vectors
    // seen only grow and postings left only shrink, so pruning then ends.
    bool prune = pruning_ && k * kPruneRatio <= postings;
    vector<uint32_t>& candidates = s.candidates;
    candidates.clear();
    size_t j = 0;
    for (; j < lists.size(); ++j) {
        if (prune && rest_postings[j] <= s.touched.size()) prune = false;
        if (prune && rest_hi[j] + margin <= kth) break;
        const double after = rest_lo[j + 1];
        for (const Posting& p : *lists[j].postings) {
            if (!s.seen[p.slot]) {
                s.seen[p.slot] = kSeen;
                s.touched.push_back(p.slot);
            }
            float& score = s.scores[p.slot];
            score = std::fma(lists[j].value, p.weight, score);  // fused in every path, so they agree
            if (prune && double(score) + after > kth + margin) offer(p.slot, double(score) + after);
        }
    }
    // The rest only add to vectors that could still reach kth. One that
    // falls out isn't added to again, and its score stays below kth, so the
    // final TopK drops it. A list long next to the vectors seen is searched
    // for each candidate instead, once they are collected and sorted.
    bool listed = false;  // whether candidates holds the allowed ones that could reach kth, in slot order
    for (; j < lists.size(); ++j) {
        const double hi = rest_hi[j] + margin, after = rest_lo[j + 1];
        const auto viable = [&](uint32_t slot) { return double(s.scores[slot]) + hi > kth; };
        const auto add = [&](uint32_t slot, float weight) {
            float& score = s.scores[slot];
            score = std::fma(lists[j].value, weight, score);
            if (double(score) + after > kth + margin) offer(slot, double(score) + after);
        };
        const vector<Posting>& list = *lists[j].postings;
        if (!listed && s.touched.size() * 16 < list.size()) {
            for (uint32_t slot : s.touched) {
                if (allowed(slot) && viable(slot)) {
                    s.seen[slot] |= kCandidate;
                    candidates.push_back(slot);
                }
            }
            std::sort(candidates.begin(), candidates.end());
            listed = true;
        }
        if (listed && candidates.size() * 16 < list.size()) {
            std::erase_if(candidates, [&](uint32_t slot) { return !viable(slot); });
            const Posting* at = list.data();
            const Posting* end = at + list.size();
            for (uint32_t slot : candidates) {
                at = seek(at, end, slot);
                if (at == end) break;
                if (at->slot == slot) add(slot, (at++)->weight);
            }
        } else {
            const uint8_t flag = listed ? kCandidate : kSeen;
            for (const Posting& p : list)
                if ((s.seen[p.slot] & flag) && viable(p.slot)) add(p.slot, p.weight);
        }
    }

    detail::TopK top(k);
    for (uint32_t slot : s.touched) {
        const float dist = -s.scores[slot];
        if (allowed(slot) && dist < top.threshold()) top.push(dist, static_cast<size_t>(id_map_.label(slot)));
        s.scores[slot] = 0.0f;
        s.seen[slot] = 0;
    }
    s.touched.clear();
    return top.take_sorted();
}

size_t InvertedIndex::compact() { return compact_impl(true); }

size_t InvertedIndex::compact_impl(bool keep_ids) {
    const vector<uint32_t> remap = id_map_.compact(keep_ids);
    if (remap.empty()) {
        metadata_.drop_unused_strings();
        return 0;
    }
    for (auto it = lists_.begin(); it != lists_.end();) {
        vector<Posting>& list = it->second.postings;
        size_t w = 0;
        for (const Posting& p : list)  // remap keeps slot order, so the list stays sorted
            if (remap[p.slot] != detail::IdMap::kRemoved) list[w++] = {remap[p.slot], p.weight};
        list.resize(w);
        if (w == 0) {
            it = lists_.erase(it);
        } else {
            list.shrink_to_fit();
            it->second.bound();
            ++it;
        }
    }
    metadata_.compact(remap);
    return remap.size() - id_map_.slots();
}

size_t InvertedIndex::postings() const noexcept {
    size_t total = 0;
    for (const auto& [term, list] : lists_) total += list.postings.size();
    return total;
}

void InvertedIndex::List::bound() {
    max_weight = min_weight = postings.front().weight;
    for (const Posting& p : postings) {
        max_weight = std::max(max_weight, p.weight);
        min_weight = std::min(min_weight, p.weight);
    }
}

void InvertedIndex::save(std::ostream& out) const {
    using namespace detail;

    write_header(out, IndexKind::Inverted);
    write_pod(out, static_cast<uint64_t>(id_map_.slots()));
    write_pod(out, static_cast<uint64_t>(lists_.size()));
    vector<uint32_t> terms;  // sorted, so the same index always writes the same bytes
    terms.reserve(lists_.size());
    for (const auto& [term, list] : lists_) terms.push_back(term);
    std::sort(terms.begin(), terms.end());
    for (uint32_t term : terms) {
        write_pod(out, term);
        write_vec(out, lists_.at(term).postings);
    }
    id_map_.save(out);
    metadata_.save(out);
}

std::unique_ptr<InvertedIndex> InvertedIndex::load_body(std::istream& in, uint32_t version) {
    using namespace detail;

    if (version < 9) throw std::runtime_error("corrupt file: SparseIndex in a format before version 9");
    const auto count = read_pod<uint64_t>(in);
    if (count > UINT32_MAX) throw std::runtime_error("corrupt file: invalid SparseIndex size");
    const auto nlists = read_pod<uint64_t>(in);

    auto index = std::make_unique<InvertedIndex>();
    for (uint64_t l = 0; l < nlists; ++l) {
        const auto term = read_pod<uint32_t>(in);
        vector<Posting> list = read_vec<Posting>(in, count);
        if (list.empty()) throw std::runtime_error("corrupt file: empty SparseIndex posting list");
        for (size_t j = 0; j < list.size(); ++j)
            if (list[j].slot >= count || (j > 0 && list[j].slot <= list[j - 1].slot))
                throw std::runtime_error("corrupt file: SparseIndex posting list out of order");
        auto [it, added] = index->lists_.try_emplace(term);
        if (!added) throw std::runtime_error("corrupt file: SparseIndex term appears twice");
        it->second.postings = std::move(list);
        it->second.bound();
    }
    index->id_map_ = IdMap::load(in, count, version);
    index->metadata_ = MetadataStore::load(in, count);
    return index;
}

} // namespace vecengine
