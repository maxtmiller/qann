#include "vecengine/sparse_index.hpp"
#include "parallel.hpp"
#include "serialize.hpp"
#include "topk.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <functional>
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

// Per-thread accumulators of the searches, sized to the largest index
// searched and left all zero between queries.
struct Scratch {
    vector<float> scores;
    vector<uint8_t> seen;  // kSeen, plus kKept while the slot is in the k-th score heap
    vector<uint32_t> touched;
    vector<std::pair<double, uint32_t>> kept;  // min-heap of (lower bound, slot)
    vector<std::pair<float, uint32_t>> best;   // approximate candidates: (-partial score, slot)
    vector<Term> picked;                       // approximate query terms

    void fit(size_t slots) {
        if (scores.size() >= slots) return;
        scores.resize(slots, 0.0f);
        seen.resize(slots, 0);
    }
};

constexpr uint8_t kSeen = 1, kKept = 2;
constexpr size_t kPruneRatio = 128;  // see SparseIndex::search_one

} // namespace

vector<int64_t> SparseIndex::add_batch(const SparseRows& rows, const int64_t* ids, const Metadata& metadata) {
    check_rows(rows);
    const size_t n = rows.rows();
    id_map_.check(n, ids);
    metadata_.check(n, metadata);

    const size_t first = slots();
    vector<Term> terms;
    vector<List*> touched;
    for (size_t i = 0; i < n; ++i) {
        canonical_row(rows, i, terms);
        for (auto [term, value] : terms) {
            List& list = lists_[term];
            if (list.postings.size() == list.sorted) touched.push_back(&list);
            if (list.postings.empty()) list.max_weight = list.min_weight = value;
            list.max_weight = std::max(list.max_weight, value);
            list.min_weight = std::min(list.min_weight, value);
            list.postings.push_back({static_cast<uint32_t>(first + i), value});
            if (forward_) {
                fwd_terms_.push_back(term);
                fwd_weights_.push_back(value);
            }
        }
        if (forward_) fwd_ptr_.push_back(fwd_terms_.size());
    }
    for (List* list : touched) list->settle();
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
        thread_local Terms terms;
        canonical_row(queries, i, terms);
        results[i] = search_one(terms, k, filter);
    });
    return results;
}

vector<SparseIndex::QueryList> SparseIndex::query_lists(const Terms& terms) const {
    vector<QueryList> lists;
    for (auto [term, value] : terms) {
        const auto it = lists_.find(term);
        if (it == lists_.end()) continue;
        const double a = double(value) * it->second.max_weight, b = double(value) * it->second.min_weight;
        lists.push_back({term, value, &it->second, std::max({0.0, a, b}), std::min({0.0, a, b})});
    }
    std::stable_sort(lists.begin(), lists.end(), [](const QueryList& x, const QueryList& y) {
        return x.hi != y.hi ? x.hi > y.hi : x.list->postings.size() < y.list->postings.size();
    });
    return lists;
}

vector<Neighbor> SparseIndex::search_one(const Terms& terms, size_t k, const detail::SlotFilter* filter) const {
    if (query_terms_ || list_depth_) return search_approximate(terms, k, filter);
    // ties go to the shorter list (query_lists), leaving long ones for last, where pruning skips them
    const vector<QueryList> lists = query_lists(terms);
    double mass = 0;      // the sum of |contribution| bounds, for the rounding margin
    size_t postings = 0;  // in the query's lists
    for (const QueryList& l : lists) {
        mass += std::max(-l.lo, l.hi);
        postings += l.list->postings.size();
    }
    // bounds and postings of lists[j, end) summed
    vector<double> rest_hi(lists.size() + 1, 0.0), rest_lo(lists.size() + 1, 0.0);
    vector<size_t> rest_postings(lists.size() + 1, 0);
    for (size_t j = lists.size(); j-- > 0;) {
        rest_hi[j] = rest_hi[j + 1] + lists[j].hi;
        rest_lo[j] = rest_lo[j + 1] + lists[j].lo;
        rest_postings[j] = rest_postings[j + 1] + lists[j].list->postings.size();
    }
    // Scores are float sums of up to lists.size() products, so every bound
    // is widened by this much: rounding never prunes a vector the full
    // scoring would keep.
    const double margin = 2.0 * mass * double(lists.size() + 2) * 1.2e-7;

    thread_local Scratch scratch;
    Scratch& s = scratch;  // one thread-local lookup, not one per use (each is a call on macOS)
    s.fit(slots());
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
    // k. Past that, the rest only add to vectors that could still reach kth.
    // Keeping kth costs more than pruning saves unless k is small next to the
    // vectors the lists reach (k <= postings / 128 measured best on BM25 over
    // BEIR SciFact and FiQA), so larger k reads every list, and once fewer
    // postings are left than vectors seen; vectors seen only grow and
    // postings left only shrink, so pruning then ends.
    bool prune = pruning_ && k * kPruneRatio <= postings;
    size_t j = 0;
    for (; j < lists.size(); ++j) {
        if (prune && rest_postings[j] <= s.touched.size()) prune = false;
        if (prune && rest_hi[j] + margin <= kth) break;
        const double after = rest_lo[j + 1];
        for (const Posting& p : lists[j].list->postings) {
            if (!s.seen[p.slot]) {
                s.seen[p.slot] = kSeen;
                s.touched.push_back(p.slot);
            }
            float& score = s.scores[p.slot];
            score = std::fma(lists[j].value, p.weight, score);  // fused in every path, so they agree
            if (prune && double(score) + after > kth + margin) offer(p.slot, double(score) + after);
        }
    }
    // The rest only add to vectors that could still reach kth, testing a
    // flag per posting. One that falls out isn't added to again, and its
    // score stays below kth, so the final TopK drops it.
    for (; j < lists.size(); ++j) {
        const double hi = rest_hi[j] + margin, after = rest_lo[j + 1];
        for (const Posting& p : lists[j].list->postings) {
            if (!s.seen[p.slot] || double(s.scores[p.slot]) + hi <= kth) continue;
            float& score = s.scores[p.slot];
            score = std::fma(lists[j].value, p.weight, score);
            if (double(score) + after > kth + margin) offer(p.slot, double(score) + after);
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

vector<Neighbor> SparseIndex::search_approximate(const Terms& terms, size_t k, const detail::SlotFilter* filter) const {
    thread_local Scratch scratch;
    Scratch& s = scratch;
    s.fit(slots());
    auto allowed = [&](uint32_t slot) { return filter ? filter->test(slot) : !id_map_.is_deleted(slot); };

    // the query_terms largest values (ties to the smaller term)
    vector<Term>& picked = s.picked;
    picked.assign(terms.begin(), terms.end());
    if (query_terms_ && picked.size() > query_terms_) {
        std::nth_element(picked.begin(), picked.begin() + (query_terms_ - 1), picked.end(),
                         [](const Term& a, const Term& b) { return a.second != b.second ? a.second > b.second : a.first < b.first; });
        picked.resize(query_terms_);
    }
    // partial scores from the heaviest entries of their lists, and every unsorted one
    for (auto [term, value] : picked) {
        const auto it = lists_.find(term);
        if (it == lists_.end()) continue;
        const List& list = it->second;
        const size_t depth = list_depth_ ? std::min(list_depth_, list.head) : list.head;
        const auto read = [&](const Posting& p) {
            if (!s.seen[p.slot]) {
                s.seen[p.slot] = kSeen;
                s.touched.push_back(p.slot);
            }
            s.scores[p.slot] = std::fma(value, p.weight, s.scores[p.slot]);
        };
        std::for_each(list.postings.begin(), list.postings.begin() + depth, read);
        std::for_each(list.postings.begin() + list.sorted, list.postings.end(), read);
    }
    vector<std::pair<float, uint32_t>>& best = s.best;
    best.clear();
    for (uint32_t slot : s.touched) {
        if (allowed(slot)) best.emplace_back(-s.scores[slot], slot);
        s.scores[slot] = 0.0f;
        s.seen[slot] = 0;
    }
    s.touched.clear();
    const size_t keep = k > SIZE_MAX / k_factor_ ? SIZE_MAX : k * k_factor_;
    if (best.size() > keep) {
        std::nth_element(best.begin(), best.begin() + (keep - 1), best.end());
        best.resize(keep);
    }

    // re-score the candidates with the whole query, adding terms in the
    // order search_one does, so each gets its exact distance
    const vector<QueryList> lists = query_lists(terms);
    vector<std::pair<uint32_t, uint32_t>> by_term(lists.size());  // (term, position in lists), by term
    for (size_t j = 0; j < lists.size(); ++j) by_term[j] = {lists[j].term, static_cast<uint32_t>(j)};
    std::sort(by_term.begin(), by_term.end());
    vector<float> weight(lists.size(), 0.0f);  // the candidate's weight per list; 0 adds nothing
    detail::TopK top(k);
    for (auto [partial, slot] : best) {
        // merge the candidate's terms with the query's, both in term order
        size_t q = 0;
        for (uint64_t r = fwd_ptr_[slot]; r < fwd_ptr_[slot + 1] && q < by_term.size(); ++r) {
            while (q < by_term.size() && by_term[q].first < fwd_terms_[r]) ++q;
            if (q < by_term.size() && by_term[q].first == fwd_terms_[r]) weight[by_term[q].second] = fwd_weights_[r];
        }
        float score = 0;
        for (size_t j = 0; j < lists.size(); ++j) {
            score = std::fma(lists[j].value, weight[j], score);
            weight[j] = 0.0f;
        }
        if (-score < top.threshold()) top.push(-score, static_cast<size_t>(id_map_.label(slot)));
    }
    return top.take_sorted();
}

void SparseIndex::set_query_terms(size_t n) {
    if (n && !forward_) throw std::invalid_argument("approximate search needs a forward index (forward_index=True)");
    query_terms_ = n;
}

void SparseIndex::set_list_depth(size_t n) {
    if (n && !forward_) throw std::invalid_argument("approximate search needs a forward index (forward_index=True)");
    if (n > kHead) throw std::invalid_argument("list_depth is at most " + std::to_string(kHead));
    list_depth_ = n;
}

void SparseIndex::set_k_factor(size_t n) {
    if (n == 0) throw std::invalid_argument("k_factor must be at least 1");
    k_factor_ = n;
}

size_t SparseIndex::compact() {
    const vector<uint32_t> remap = id_map_.compact(true);
    if (remap.empty()) {
        metadata_.drop_unused_strings();
        return 0;
    }
    for (auto it = lists_.begin(); it != lists_.end();) {
        vector<Posting>& list = it->second.postings;
        size_t w = 0, head = 0, sorted = 0;
        for (size_t i = 0; i < list.size(); ++i) {  // remap keeps slot order, so every part stays in order
            if (remap[list[i].slot] == detail::IdMap::kRemoved) continue;
            list[w++] = {remap[list[i].slot], list[i].weight};
            if (i < it->second.head) head = w;
            if (i < it->second.sorted) sorted = w;
        }
        it->second.head = head;
        it->second.sorted = sorted;
        list.resize(w);
        if (w == 0) {
            it = lists_.erase(it);
        } else {
            list.shrink_to_fit();
            it->second.bound();
            ++it;
        }
    }
    if (forward_) rebuild_forward();
    metadata_.compact(remap);
    return remap.size() - id_map_.slots();
}

void SparseIndex::rebuild_forward() {
    fwd_ptr_.assign(slots() + 1, 0);
    for (const auto& [term, list] : lists_)
        for (const Posting& p : list.postings) ++fwd_ptr_[p.slot + 1];
    for (size_t i = 0; i < slots(); ++i) fwd_ptr_[i + 1] += fwd_ptr_[i];
    fwd_terms_.resize(fwd_ptr_.back());
    fwd_weights_.resize(fwd_ptr_.back());
    vector<uint32_t> terms;  // increasing, so each row's terms come out in order
    for (const auto& [term, list] : lists_) terms.push_back(term);
    std::sort(terms.begin(), terms.end());
    vector<uint64_t> at(fwd_ptr_.begin(), fwd_ptr_.end() - 1);
    for (uint32_t term : terms) {
        for (const Posting& p : lists_.at(term).postings) {
            fwd_terms_[at[p.slot]] = term;
            fwd_weights_[at[p.slot]++] = p.weight;
        }
    }
}

void SparseIndex::List::settle() {
    const size_t added = postings.size() - sorted;
    if (added <= std::max<size_t>(64, std::min<size_t>(sorted / 8, 4096))) return;
    const auto heavier = [](const Posting& a, const Posting& b) {
        return a.weight != b.weight ? a.weight > b.weight : a.slot < b.slot;
    };
    const auto by_slot = [](const Posting& a, const Posting& b) { return a.slot < b.slot; };
    // the head and the new entries compete for the head
    vector<Posting> contest(postings.begin(), postings.begin() + head);
    contest.insert(contest.end(), postings.begin() + sorted, postings.end());
    const size_t keep = std::min(contest.size(), kHead);
    std::nth_element(contest.begin(), contest.begin() + (keep - 1), contest.end(), heavier);
    std::sort(contest.begin(), contest.begin() + keep, heavier);
    std::sort(contest.begin() + keep, contest.end(), by_slot);
    // the losers merge into the slot-ordered rest
    vector<Posting> merged(postings.size());
    std::copy(contest.begin(), contest.begin() + keep, merged.begin());
    std::merge(postings.begin() + head, postings.begin() + sorted, contest.begin() + keep, contest.end(),
               merged.begin() + keep, by_slot);
    postings = std::move(merged);
    head = keep;
    sorted = postings.size();
}

size_t SparseIndex::postings() const noexcept {
    size_t total = 0;
    for (const auto& [term, list] : lists_) total += list.postings.size();
    return total;
}

void SparseIndex::List::bound() {
    max_weight = min_weight = postings.front().weight;
    for (const Posting& p : postings) {
        max_weight = std::max(max_weight, p.weight);
        min_weight = std::min(min_weight, p.weight);
    }
}

void SparseIndex::save(std::ostream& out) const {
    using namespace detail;

    write_header(out, IndexKind::Sparse);
    write_pod(out, static_cast<uint64_t>(id_map_.slots()));
    write_pod(out, static_cast<uint8_t>(forward_));
    write_pod(out, static_cast<uint64_t>(query_terms_));
    write_pod(out, static_cast<uint64_t>(list_depth_));
    write_pod(out, static_cast<uint64_t>(k_factor_));
    write_pod(out, static_cast<uint64_t>(lists_.size()));
    vector<uint32_t> terms;  // sorted, so the same index always writes the same bytes
    terms.reserve(lists_.size());
    for (const auto& [term, list] : lists_) terms.push_back(term);
    std::sort(terms.begin(), terms.end());
    for (uint32_t term : terms) {
        write_pod(out, term);
        write_pod(out, static_cast<uint64_t>(lists_.at(term).head));
        write_pod(out, static_cast<uint64_t>(lists_.at(term).sorted));
        write_vec(out, lists_.at(term).postings);
    }
    id_map_.save(out);
    metadata_.save(out);
}

std::unique_ptr<SparseIndex> SparseIndex::load_body(std::istream& in, uint32_t version) {
    using namespace detail;

    if (version < 9) throw std::runtime_error("corrupt file: SparseIndex in a format before version 9");
    const auto count = read_pod<uint64_t>(in);
    if (count > UINT32_MAX) throw std::runtime_error("corrupt file: invalid SparseIndex size");
    const auto forward = read_pod<uint8_t>(in);
    if (forward > 1) throw std::runtime_error("corrupt file: invalid SparseIndex forward flag");
    auto index = std::make_unique<SparseIndex>(false, forward == 1);
    try {
        index->set_query_terms(read_pod<uint64_t>(in));
        index->set_list_depth(read_pod<uint64_t>(in));
        index->set_k_factor(read_pod<uint64_t>(in));
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("corrupt file: invalid SparseIndex search settings");
    }
    const auto nlists = read_pod<uint64_t>(in);

    vector<uint32_t> last_list(count, UINT32_MAX);  // the list a slot last appeared in, to catch repeats
    const auto heavier = [](const Posting& a, const Posting& b) {
        return a.weight != b.weight ? a.weight > b.weight : a.slot < b.slot;
    };
    for (uint64_t l = 0; l < nlists; ++l) {
        const auto term = read_pod<uint32_t>(in);
        const auto head = read_pod<uint64_t>(in);
        const auto sorted = read_pod<uint64_t>(in);
        vector<Posting> list = read_vec<Posting>(in, count);
        if (list.empty() || head > kHead || head > sorted || sorted > list.size())
            throw std::runtime_error("corrupt file: invalid SparseIndex posting list");
        for (size_t j = 0; j < list.size(); ++j) {
            if (list[j].slot >= count || last_list[list[j].slot] == l)
                throw std::runtime_error("corrupt file: SparseIndex posting list repeats or exceeds a slot");
            last_list[list[j].slot] = static_cast<uint32_t>(l);
            // the head heaviest first, the rest in slot order
            const bool ordered = j == 0 || j == head ? true
                               : j < head         ? heavier(list[j - 1], list[j])
                                                  : list[j - 1].slot < list[j].slot;
            if (!ordered) throw std::runtime_error("corrupt file: SparseIndex posting list out of order");
        }
        auto [it, added] = index->lists_.try_emplace(term);
        if (!added) throw std::runtime_error("corrupt file: SparseIndex term appears twice");
        it->second.postings = std::move(list);
        it->second.head = head;
        it->second.sorted = sorted;
        it->second.bound();
    }
    index->id_map_ = IdMap::load(in, count, version);
    index->metadata_ = MetadataStore::load(in, count);
    if (index->forward_) index->rebuild_forward();
    return index;
}

} // namespace vecengine
