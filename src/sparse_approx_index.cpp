#include "vecengine/sparse_approx_index.hpp"
#include "serialize.hpp"
#include "sparse_lists.hpp"
#include "topk.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace vecengine {

namespace {

// Per-thread accumulators of query_impl(), sized to the largest index
// searched and left all zero between queries.
struct Scratch {
    vector<float> scores;
    vector<uint8_t> seen;
    vector<uint32_t> touched;
    vector<std::pair<float, uint32_t>> best;  // candidates: (-partial score, slot)
    vector<std::pair<uint32_t, float>> picked;  // the query terms read

    void fit(size_t slots) {
        if (scores.size() >= slots) return;
        scores.resize(slots, 0.0f);
        seen.resize(slots, 0);
    }
};

} // namespace

SparseApproxIndex::SparseApproxIndex(bool custom_ids, size_t query_terms, size_t list_depth, size_t k_factor)
    : id_map_(custom_ids), query_terms_(query_terms), list_depth_(list_depth), k_factor_(1) {
    set_k_factor(k_factor);
}

void SparseApproxIndex::set_k_factor(size_t n) {
    if (n == 0) throw std::invalid_argument("k_factor must be at least 1");
    k_factor_ = n;
}

void SparseApproxIndex::add_batch_impl(const SparseRows& rows, const int64_t* ids) {
    const size_t n = rows.rows();
    id_map_.check(n, ids);
    const size_t first = slots();
    Terms terms;
    vector<List*> touched;
    for (size_t i = 0; i < n; ++i) {
        canonical_row(rows, i, terms);
        for (auto [term, value] : terms) {
            List& list = lists_[term];
            if (list.size() == list.sorted) touched.push_back(&list);
            if (list.size() == 0) list.max_weight = list.min_weight = value;
            list.max_weight = std::max(list.max_weight, value);
            list.min_weight = std::min(list.min_weight, value);
            list.postings.push_back({static_cast<uint32_t>(first + i), value});
            fwd_ram_.push_back({term, value});
        }
        fwd_ptr_.push_back(fwd_mapped_.size() + fwd_ram_.size());
    }
    for (List* list : touched) list->settle();
    id_map_.add(n, ids);
}

vector<Neighbor> SparseApproxIndex::query_impl(const Terms& terms, size_t k, const detail::SlotFilter* filter) const {
    thread_local Scratch scratch;
    Scratch& s = scratch;  // one thread-local lookup, not one per use (each is a call on macOS)
    s.fit(slots());
    auto allowed = [&](uint32_t slot) { return filter ? filter->test(slot) : !id_map_.is_deleted(slot); };

    // the query_terms largest values (ties to the smaller term)
    auto& picked = s.picked;
    picked.assign(terms.begin(), terms.end());
    if (query_terms_ && picked.size() > query_terms_) {
        std::nth_element(picked.begin(), picked.begin() + (query_terms_ - 1), picked.end(), [](const auto& a, const auto& b) {
            return a.second != b.second ? a.second > b.second : a.first < b.first;
        });
        picked.resize(query_terms_);
    }
    // partial scores from the heaviest entries of their lists, and every unsorted one
    for (auto [term, value] : picked) {
        const auto it = lists_.find(term);
        if (it == lists_.end()) continue;
        const List& list = it->second;
        const size_t depth = list_depth_ ? std::min(list_depth_, list.sorted) : list.sorted;
        const auto read = [&](const Posting& p) {
            if (!s.seen[p.slot]) {
                s.seen[p.slot] = 1;
                s.touched.push_back(p.slot);
            }
            s.scores[p.slot] = std::fma(value, p.weight, s.scores[p.slot]);
        };
        list.each(0, depth, read);
        list.each(list.sorted, list.size(), read);
    }
    auto& best = s.best;
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
    // in slot order, so a tie at the k-th place goes to the lower slot whatever order the lists
    // and nth_element left the candidates in (a mapped list's unsorted tail differs from a RAM
    // list's); the forward rows are then read in file order too
    std::sort(best.begin(), best.end(), [](const auto& a, const auto& b) { return a.second < b.second; });

    // Re-score the candidates with the whole query. Terms are added in the
    // order InvertedIndex::query_impl adds its lists (the most a term can add
    // to a score first, ties to the shorter list), with std::fma, so each
    // distance is bit-identical to the exact one.
    struct QueryList {
        uint32_t term;
        float value;
        size_t length;
        double hi;
    };
    vector<QueryList> lists;
    for (auto [term, value] : terms) {
        const auto it = lists_.find(term);
        if (it == lists_.end()) continue;
        const double a = double(value) * it->second.max_weight, b = double(value) * it->second.min_weight;
        lists.push_back({term, value, it->second.size(), std::max({0.0, a, b})});
    }
    std::stable_sort(lists.begin(), lists.end(), [](const QueryList& x, const QueryList& y) {
        return x.hi != y.hi ? x.hi > y.hi : x.length < y.length;
    });
    vector<std::pair<uint32_t, uint32_t>> by_term(lists.size());  // (term, position in lists), by term
    for (size_t j = 0; j < lists.size(); ++j) by_term[j] = {lists[j].term, static_cast<uint32_t>(j)};
    std::sort(by_term.begin(), by_term.end());
    vector<float> weight(lists.size(), 0.0f);  // the candidate's weight per list; 0 adds nothing
    detail::TopK top(k);
    for (auto [partial, slot] : best) {
        // merge the candidate's terms with the query's, both in term order
        size_t q = 0;
        for (uint64_t r = fwd_ptr_[slot]; r < fwd_ptr_[slot + 1] && q < by_term.size(); ++r) {
            const FwdEntry& e = fwd(r);
            while (q < by_term.size() && by_term[q].first < e.term) ++q;
            if (q < by_term.size() && by_term[q].first == e.term) weight[by_term[q].second] = e.weight;
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

size_t SparseApproxIndex::compact() { return compact_impl(true); }

size_t SparseApproxIndex::compact_impl(bool keep_ids) {
    if (file_) {
        const vector<uint32_t> remap = id_map_.compact_remap();
        if (remap.empty()) {
            metadata_.drop_unused_strings();
            return 0;
        }
        compact_mapped(remap);  // may throw; changes nothing until it has written the new file
        id_map_.compact(keep_ids);
        metadata_.compact(remap);
        return remap.size() - id_map_.slots();
    }
    const vector<uint32_t> remap = id_map_.compact(keep_ids);
    if (remap.empty()) {
        metadata_.drop_unused_strings();
        return 0;
    }
    for (auto it = lists_.begin(); it != lists_.end();) {
        vector<Posting>& list = it->second.postings;
        size_t w = 0, sorted = 0;
        for (size_t i = 0; i < list.size(); ++i) {  // remap keeps slot order, so both parts stay in order
            if (remap[list[i].slot] == detail::IdMap::kRemoved) continue;
            list[w++] = {remap[list[i].slot], list[i].weight};
            if (i < it->second.sorted) sorted = w;
        }
        list.resize(w);
        it->second.sorted = sorted;
        if (w == 0) {
            it = lists_.erase(it);
        } else {
            list.shrink_to_fit();
            it->second.bound();
            ++it;
        }
    }
    rebuild_forward();
    metadata_.compact(remap);
    return remap.size() - id_map_.slots();
}

void SparseApproxIndex::compact_mapped(const vector<uint32_t>& remap) {
    using detail::IdMap;
    const vector<uint32_t> terms = detail::sorted_terms(lists_);
    vector<uint64_t> live(terms.size(), 0), live_sorted(terms.size(), 0);
    uint64_t total = 0;
    for (size_t i = 0; i < terms.size(); ++i) {
        const List& list = lists_.at(terms[i]);
        for (size_t j = 0; j < list.size(); ++j)
            if (remap[list[j].slot] != IdMap::kRemoved) {
                ++live[i];
                if (j < list.sorted) ++live_sorted[i];
            }
        total += live[i];
    }
    // the forward index keeps each live vector's row; only the row offsets change
    vector<uint64_t> ptr{0};
    for (size_t old = 0; old < remap.size(); ++old)
        if (remap[old] != IdMap::kRemoved) ptr.push_back(ptr.back() + (fwd_ptr_[old + 1] - fwd_ptr_[old]));
    uint64_t fwd_at = 0;  // file offset of the forward entries
    std::shared_ptr<const detail::MappedFile> file;
    if (total > 0) {
        file = detail::MappedFile::temporary(file_->path(), [&](std::ostream& out) {
            for (uint32_t term : terms) {
                const List& list = lists_.at(term);
                list.each(0, list.size(), [&](const Posting& p) {
                    if (remap[p.slot] == IdMap::kRemoved) return;
                    const Posting moved{remap[p.slot], p.weight};
                    out.write(reinterpret_cast<const char*>(&moved), sizeof moved);
                });
            }
            for (size_t old = 0; old < remap.size(); ++old) {
                if (remap[old] == IdMap::kRemoved) continue;
                for (uint64_t r = fwd_ptr_[old]; r < fwd_ptr_[old + 1]; ++r)
                    out.write(reinterpret_cast<const char*>(&fwd(r)), sizeof(FwdEntry));
            }
        });
        fwd_at = total * sizeof(Posting);
    }
    // the new file is written: from here nothing throws
    const auto* at = file ? reinterpret_cast<const Posting*>(file->data()) : nullptr;
    for (size_t i = 0; i < terms.size(); ++i) {
        auto it = lists_.find(terms[i]);
        if (live[i] == 0) {
            lists_.erase(it);
            continue;
        }
        List& list = it->second;
        list.mapped = span<const Posting>(at, live[i]);
        vector<Posting>().swap(list.postings);
        list.sorted = live_sorted[i];
        list.bound();
        at += live[i];
    }
    fwd_ptr_ = std::move(ptr);
    fwd_mapped_ = file ? span<const FwdEntry>(reinterpret_cast<const FwdEntry*>(file->data() + fwd_at), fwd_ptr_.back())
                       : span<const FwdEntry>();
    vector<FwdEntry>().swap(fwd_ram_);
    if (file) file->advise_random(file->data() + fwd_at, fwd_ptr_.back() * sizeof(FwdEntry));
    file_ = std::move(file);
}

void SparseApproxIndex::rebuild_forward() {
    fwd_ptr_.assign(slots() + 1, 0);
    for (const auto& [term, list] : lists_)
        list.each(0, list.size(), [&](const Posting& p) { ++fwd_ptr_[p.slot + 1]; });
    for (size_t i = 0; i < slots(); ++i) fwd_ptr_[i + 1] += fwd_ptr_[i];
    fwd_mapped_ = {};
    fwd_ram_.resize(fwd_ptr_.back());
    vector<uint64_t> at(fwd_ptr_.begin(), fwd_ptr_.end() - 1);
    for (uint32_t term : detail::sorted_terms(lists_))  // increasing, so each row's terms come out in order
        lists_.at(term).each(0, lists_.at(term).size(), [&](const Posting& p) { fwd_ram_[at[p.slot]++] = {term, p.weight}; });
}

size_t SparseApproxIndex::postings() const noexcept {
    size_t total = 0;
    for (const auto& [term, list] : lists_) total += list.size();
    return total;
}

void SparseApproxIndex::List::settle() {
    if (!mapped.empty()) return;
    const size_t added = postings.size() - sorted;
    if (added <= std::max<size_t>(64, std::min<size_t>(sorted / 8, 4096))) return;
    const auto heavier = [](const Posting& a, const Posting& b) {
        return a.weight != b.weight ? a.weight > b.weight : a.slot < b.slot;
    };
    std::sort(postings.begin() + sorted, postings.end(), heavier);
    std::inplace_merge(postings.begin(), postings.begin() + sorted, postings.end(), heavier);
    sorted = postings.size();
}

void SparseApproxIndex::save(std::ostream& out) const {
    using namespace detail;

    write_header(out, IndexKind::SparseApprox);
    write_pod(out, static_cast<uint64_t>(id_map_.slots()));
    write_pod(out, static_cast<uint64_t>(query_terms_));
    write_pod(out, static_cast<uint64_t>(list_depth_));
    write_pod(out, static_cast<uint64_t>(k_factor_));
    save_posting_lists(out, lists_, [](std::ostream& o, const List& list) { write_pod(o, static_cast<uint64_t>(list.sorted)); });
    // version 11: the forward index, its entries at a 64-byte offset so a load with mmap can point into it
    write_vec(out, fwd_ptr_);
    write_pod(out, static_cast<uint64_t>(fwd_ptr_.back()));
    write_pad(out, kPostingAlign);
    write_span(out, fwd_mapped_);
    write_span(out, span<const FwdEntry>(fwd_ram_));
    id_map_.save(out);
    metadata_.save(out);
}

std::unique_ptr<SparseApproxIndex> SparseApproxIndex::load_body(std::istream& in, uint32_t version, const detail::LoadContext* ctx) {
    using namespace detail;

    if (version < 9) throw std::runtime_error("corrupt file: SparseApproxIndex in a format before version 9");
    const auto count = read_pod<uint64_t>(in);
    if (count > UINT32_MAX) throw std::runtime_error("corrupt file: invalid SparseApproxIndex size");
    const auto query_terms = read_pod<uint64_t>(in);
    const auto list_depth = read_pod<uint64_t>(in);
    const auto k_factor = read_pod<uint64_t>(in);
    if (k_factor == 0) throw std::runtime_error("corrupt file: SparseApproxIndex k_factor 0");
    auto index = std::make_unique<SparseApproxIndex>(false, query_terms, list_depth, k_factor);
    if (version >= 11) {
        load_posting_lists(in, count, ctx, index->lists_, index->file_,
                           [](std::istream& i, List& list) { list.sorted = read_pod<uint64_t>(i); });
    } else {  // before 11: each list as its term, sorted and a write_vec, read into RAM
        const auto nlists = read_pod<uint64_t>(in);
        for (uint64_t l = 0; l < nlists; ++l) {
            const auto term = read_pod<uint32_t>(in);
            const auto sorted = read_pod<uint64_t>(in);
            vector<Posting> list = read_vec<Posting>(in, count);
            if (list.empty()) throw std::runtime_error("corrupt file: invalid SparseApproxIndex posting list");
            auto [it, added] = index->lists_.try_emplace(term);
            if (!added) throw std::runtime_error("corrupt file: SparseApproxIndex term appears twice");
            it->second.postings = std::move(list);
            it->second.sorted = sorted;
        }
    }

    // every entry is checked, mapped or not (a mapped load reads the file once here): queries
    // index arrays by slot, so a slot out of range would be out of bounds
    vector<uint32_t> last_list(count, UINT32_MAX);  // the list a slot last appeared in, to catch repeats
    vector<uint64_t> per_slot(count, 0);              // entries per slot, which the forward index must match
    const auto heavier = [](const Posting& a, const Posting& b) {
        return a.weight != b.weight ? a.weight > b.weight : a.slot < b.slot;
    };
    uint32_t l = 0;
    for (auto& [term, list] : index->lists_) {
        if (list.sorted > list.size()) throw std::runtime_error("corrupt file: invalid SparseApproxIndex posting list");
        for (size_t j = 0; j < list.size(); ++j) {
            const Posting& p = list[j];
            if (p.slot >= count || last_list[p.slot] == l)
                throw std::runtime_error("corrupt file: SparseApproxIndex posting list repeats or exceeds a slot");
            last_list[p.slot] = l;
            ++per_slot[p.slot];
            // heaviest first up to sorted, then slot order
            const bool ordered = j == 0 || j == list.sorted ? true
                               : j < list.sorted         ? heavier(list[j - 1], p)
                                                         : list[j - 1].slot < p.slot;
            if (!ordered) throw std::runtime_error("corrupt file: SparseApproxIndex posting list out of order");
        }
        list.bound();
        ++l;
    }

    if (version >= 11) {
        index->fwd_ptr_ = read_vec<uint64_t>(in, count + 1);
        if (index->fwd_ptr_.size() != count + 1 || index->fwd_ptr_[0] != 0)
            throw std::runtime_error("corrupt file: invalid SparseApproxIndex forward index");
        for (uint64_t slot = 0; slot < count; ++slot)
            if (index->fwd_ptr_[slot + 1] < index->fwd_ptr_[slot] ||
                index->fwd_ptr_[slot + 1] - index->fwd_ptr_[slot] != per_slot[slot])
                throw std::runtime_error("corrupt file: SparseApproxIndex forward index doesn't match its lists");
        const uint64_t entries = index->fwd_ptr_.back();
        if (read_pod<uint64_t>(in) != entries) throw std::runtime_error("corrupt file: invalid SparseApproxIndex forward index");
        read_pad(in, kPostingAlign);
        const char* fwd_at = nullptr;
        if (ctx && ctx->file && entries > 0) {
            fwd_at = map_next(in, *ctx->file, entries * sizeof(FwdEntry), alignof(FwdEntry));
            index->fwd_mapped_ = span<const FwdEntry>(reinterpret_cast<const FwdEntry*>(fwd_at), entries);
            index->file_ = ctx->file;
        } else {
            index->fwd_ram_.resize(entries);
            in.read(reinterpret_cast<char*>(index->fwd_ram_.data()), static_cast<std::streamsize>(entries * sizeof(FwdEntry)));
            if (!in) throw std::runtime_error("truncated file");
        }
        // each row's terms increasing, as the re-scoring merge needs
        for (uint64_t slot = 0; slot < count; ++slot)
            for (uint64_t r = index->fwd_ptr_[slot] + 1; r < index->fwd_ptr_[slot + 1]; ++r)
                if (index->fwd(r).term <= index->fwd(r - 1).term)
                    throw std::runtime_error("corrupt file: SparseApproxIndex forward index out of order");
        // only after the check above, which reads the rows in order and needs readahead: re-scoring
        // reads candidates' rows, scattered, where readahead would fetch rows nobody asked for
        if (fwd_at) ctx->file->advise_random(fwd_at, entries * sizeof(FwdEntry));
    }
    index->id_map_ = IdMap::load(in, count, version);
    index->metadata_ = MetadataStore::load(in, count);
    if (version < 11) index->rebuild_forward();
    return index;
}

} // namespace vecengine
