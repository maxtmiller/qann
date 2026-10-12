#include "vecengine/hnsw_index.hpp"
#include "vecengine/distance.hpp"
#include "serialize.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <queue>

namespace vecengine {

namespace {

// Tracks which nodes the current search has visited without clearing between
// searches: a node is visited when its tag equals the current generation.
// One per thread, since query_batch runs searches in parallel.
struct VisitedTags {
    vector<uint32_t> tags;
    uint32_t gen = 0;

    // Starts a new search over n nodes.
    void begin(size_t n) {
        if (tags.size() < n) tags.resize(n, 0);
        if (++gen == 0) {   // wrapped: old tags could match again
            std::fill(tags.begin(), tags.end(), 0);
            gen = 1;
        }
    }

    // Marks slot visited; returns false if it already was.
    bool visit(uint32_t slot) {
        if (tags[slot] == gen) return false;
        tags[slot] = gen;
        return true;
    }
};

thread_local VisitedTags t_visited;

} // namespace


HNSWIndex::HNSWIndex(size_t dim, size_t M, size_t ef_construction, bool custom_ids, std::optional<uint32_t> seed, Metric metric)
    : Index(metric), dim_(dim), M_(M), M0_(2 * M), ef_construction_(ef_construction),
      rng_state_(seed ? *seed : std::random_device{}()), id_map_(custom_ids) {
    if (dim == 0) throw std::invalid_argument("dim must be >= 1");
    if (M < 2) throw std::invalid_argument("M must be >= 2");
    if (ef_construction == 0) throw std::invalid_argument("ef_construction must be >= 1");
    level_mult_ = 1.0 / std::log(static_cast<double>(M));
}

void HNSWIndex::set_ef_search(size_t ef) {
    if (ef == 0) throw std::invalid_argument("ef_search must be >= 1");
    ef_search_ = ef;
}

void HNSWIndex::add_batch_impl(span<const float> vecs, size_t n, const int64_t* ids) {
    assert(vecs.size() == n * dim_);
    const uint32_t first = static_cast<uint32_t>(id_map_.slots());
    id_map_.add(n, ids);

    data_.insert(data_.end(), vecs.begin(), vecs.end());
    level0_.resize(id_map_.slots() * (M0_ + 1), 0);
    upper_.resize(id_map_.slots());
    levels_.resize(id_map_.slots(), 0);

    build(first, n);
}

vector<Neighbor> HNSWIndex::query_impl(span<const float> vec, size_t k) const {
    assert(vec.size() == dim_);
    if (max_level_ < 0 || k == 0) return {};

    // greedy descent: the nearest node on each upper level enters the next
    uint32_t ep = entry_;
    for (int level = max_level_; level > 0; --level) ep = greedy_closest(vec.data(), ep, level, false);

    vector<Neighbor> results;
    results.reserve(k);
    for (auto [dist, slot] : search_layer(vec.data(), {&ep, 1}, std::max(ef_search_, k), 0, true)) {
        results.push_back({static_cast<size_t>(id_map_.label(slot)), dist});
        if (results.size() == k) break;
    }
    return results;
}

size_t HNSWIndex::remove(span<const int64_t> ids) { return id_map_.remove(ids); }

size_t HNSWIndex::compact() { return compact_impl(true); }

size_t HNSWIndex::compact_as_base() { return compact_impl(false); }

size_t HNSWIndex::compact_impl(bool keep_ids) {
    auto remap = id_map_.compact(keep_ids);
    if (remap.size() == 0) return 0;

    size_t write_slot = 0;
    for (size_t i = 0; i < remap.size(); ++i) {
        uint32_t new_slot = remap[i];
        if (remap[i] == detail::IdMap::kRemoved) continue;

        if (new_slot != i) {
            float* src = data_.data() + i * dim();
            float* dst = data_.data() + new_slot * dim();
            std::copy_n(src, dim(), dst);
        }
        ++write_slot;
    }
    data_.resize(write_slot * dim());
    data_.shrink_to_fit();

    level0_.assign(write_slot * (M0_ + 1), 0);
    upper_.assign(write_slot, {});
    levels_.assign(write_slot, 0);
    max_level_ = -1;
    entry_ = 0;

    build(0, write_slot);

    return remap.size() - id_map_.slots();
}

void HNSWIndex::save(std::ostream& out) const {
    using namespace detail;

    write_header(out, IndexKind::HNSW);
    write_pod(out, static_cast<uint64_t>(dim_));
    write_pod(out, static_cast<uint64_t>(M_));
    write_pod(out, static_cast<uint64_t>(ef_construction_));
    write_pod(out, static_cast<uint64_t>(ef_search_));
    write_pod(out, static_cast<uint64_t>(id_map_.slots()));
    write_pod(out, static_cast<int32_t>(max_level_));
    write_pod(out, entry_);

    // the generator's state, so adds after a load draw the same levels as
    // adds without the save
    write_pod(out, rng_state_);

    write_vec(out, data_);
    write_vec(out, levels_);
    write_vec(out, level0_);
    for (size_t slot = 0; slot < upper_.size(); ++slot)
        for (const auto& list : upper_[slot]) write_vec(out, list);
    id_map_.save(out);
    write_pod(out, static_cast<uint8_t>(metric_));
}

std::unique_ptr<HNSWIndex> HNSWIndex::load_body(std::istream& in, uint32_t version) {
    using namespace detail;

    const auto dim = read_pod<uint64_t>(in);
    const auto M = read_pod<uint64_t>(in);
    const auto ef_construction = read_pod<uint64_t>(in);
    const auto ef_search = read_pod<uint64_t>(in);
    const auto slots = read_pod<uint64_t>(in);
    const auto max_level = read_pod<int32_t>(in);
    const auto entry = read_pod<uint32_t>(in);

    const auto corrupt = [](const char* what) { return std::runtime_error(std::string("corrupt file: ") + what); };
    if (dim == 0 || M < 2 || M > 4096 || ef_construction == 0 || ef_search == 0 || slots > UINT32_MAX ||
        slots > UINT64_MAX / dim)
        throw corrupt("invalid HNSWIndex shape");

    auto index = std::make_unique<HNSWIndex>(dim, M, ef_construction);
    index->set_ef_search(ef_search);

    if (version >= 5) {
        index->rng_state_ = read_pod<uint64_t>(in);
    } else {
        // version 4 stored std::mt19937's text form, which isn't portable;
        // continue from a hash of it (FNV-1a) so loading works everywhere
        const auto text = read_vec<char>(in, 1 << 16);
        uint64_t h = 0xCBF29CE484222325ull;
        for (char c : text) h = (h ^ static_cast<unsigned char>(c)) * 0x100000001B3ull;
        index->rng_state_ = h;
    }

    index->data_ = read_vec<float>(in, slots * dim);
    index->levels_ = read_vec<uint8_t>(in, slots);
    index->level0_ = read_vec<uint32_t>(in, slots * (index->M0_ + 1));
    if (index->data_.size() != slots * dim || index->levels_.size() != slots ||
        index->level0_.size() != slots * (index->M0_ + 1))
        throw corrupt("HNSWIndex length mismatch");

    // every neighbor list: within capacity and pointing at real slots
    const auto check_list = [&](span<const uint32_t> list, size_t cap) {
        if (list.size() > cap) throw corrupt("HNSWIndex neighbor list too long");
        for (uint32_t nb : list)
            if (nb >= slots) throw corrupt("HNSWIndex neighbor out of range");
    };
    int top = -1;
    index->upper_.resize(slots);
    for (uint32_t slot = 0; slot < slots; ++slot) {
        if (index->level0_[size_t{slot} * (index->M0_ + 1)] > index->M0_) throw corrupt("HNSWIndex neighbor list too long");
        check_list(index->neighbors(slot, 0), index->M0_);
        const int level = index->levels_[slot];
        top = std::max(top, level);
        index->upper_[slot].resize(static_cast<size_t>(level));
        for (auto& list : index->upper_[slot]) {
            list = read_vec<uint32_t>(in, M);
            check_list(list, M);
        }
    }
    if (max_level != top || (slots > 0 && (entry >= slots || index->levels_[entry] != top)))
        throw corrupt("invalid HNSWIndex entry point");
    index->max_level_ = max_level;
    index->entry_ = entry;

    index->id_map_ = IdMap::load(in, slots, version);
    if (version >= 5) index->metric_ = read_metric(in);
    return index;
}

vector<HNSWIndex::Candidate> HNSWIndex::search_layer(const float* q, span<const uint32_t> entry_points, size_t ef, int level, bool skip_deleted, bool locked) const {
    t_visited.begin(slots());
    vector<uint32_t> copy;   // a locked search's snapshot of the list being expanded
    if (locked) copy.reserve(capacity(level));
    vector<uint32_t> fresh;  // the expanded node's neighbors not visited yet
    fresh.reserve(capacity(level));

    std::priority_queue<Candidate, vector<Candidate>> maxHeap;
    std::priority_queue<Candidate, vector<Candidate>, std::greater<Candidate>> minHeap;

    // a node always joins minHeap to be expanded; it joins the results
    // (maxHeap) unless it is deleted and the caller skips deleted nodes
    auto admit = [&](float dist, uint32_t slot) {
        minHeap.emplace(dist, slot);
        if (skip_deleted && id_map_.is_deleted(slot)) return;
        maxHeap.emplace(dist, slot);
        if (maxHeap.size() > ef) maxHeap.pop();
    };

    for (uint32_t p : entry_points) {
        if (t_visited.visit(p)) admit(distance(q, p), p);
    }

    while (!minHeap.empty()) {
        const Candidate current = minHeap.top();
        if (maxHeap.size() >= ef && current.first > maxHeap.top().first) break;
        minHeap.pop();

        // the node expanded after this one is likely the new top: start loading its list now
        if (level == 0 && !minHeap.empty()) __builtin_prefetch(level0_.data() + size_t{minHeap.top().second} * (M0_ + 1));

        span<const uint32_t> list;
        if (locked) {
            // another thread may rewrite the list: copy it under its lock, then
            // compute distances with the lock released
            std::lock_guard<std::mutex> guard(lock_for(current.second));
            const span<const uint32_t> shared = neighbors(current.second, level);
            copy.assign(shared.begin(), shared.end());
            list = copy;
        } else {
            list = neighbors(current.second, level);
        }

        // first pass: find the unvisited neighbors and start loading their vectors, so
        // the memory reads overlap; second pass: compute distances, add those that
        // beat the worst result (or fill the results)
        fresh.clear();
        for (uint32_t n : list) {
            if (t_visited.visit(n)) {
                fresh.push_back(n);
                prefetch_vector(n);
            }
        }
        for (uint32_t n : fresh) {
            const float dist = distance(q, n);
            if (maxHeap.size() < ef || dist < maxHeap.top().first) admit(dist, n);
        }
    }

    // maxHeap pops worst first, so fill from the back
    vector<Candidate> res(maxHeap.size());
    for (size_t i = res.size(); i-- > 0;) {
        res[i] = maxHeap.top();
        maxHeap.pop();
    }

    return res;
}

uint32_t HNSWIndex::greedy_closest(const float* q, uint32_t ep, int level, bool locked) const {
    float best = distance(q, ep);
    vector<uint32_t> copy;
    for (bool moved = true; moved;) {
        moved = false;
        span<const uint32_t> list;
        if (locked) {
            std::lock_guard<std::mutex> guard(lock_for(ep));
            const span<const uint32_t> shared = neighbors(ep, level);
            copy.assign(shared.begin(), shared.end());
            list = copy;
        } else {
            list = neighbors(ep, level);
        }

        for (uint32_t n : list) prefetch_vector(n);
        const uint32_t from = ep;
        for (uint32_t n : list) {
            const float d = distance(q, n);
            if (d < best) {
                best = d;
                ep = n;
            }
        }
        moved = ep != from;
    }
    return ep;
}

vector<uint32_t> HNSWIndex::select_neighbors(vector<Candidate> candidates, size_t max_neighbors) const {
    std::sort(candidates.begin(), candidates.end());

    vector<uint32_t> res;
    res.reserve(max_neighbors);
    for (auto& c : candidates) { 
        if (res.size() == max_neighbors) break;
        bool keep = true;
        for (uint32_t r : res) {
            if (distance(vec(c.second), r) <= c.first) { keep = false; break; }
        }
        if (keep) res.emplace_back(c.second);
    }

    return res;
}

void HNSWIndex::build(uint32_t first, size_t n) {
    // levels are drawn here, in slot order, so a seed fixes them however the
    // inserts below are spread over threads
    for (size_t i = 0; i < n; ++i) {
        const int level = random_level();
        levels_[first + i] = static_cast<uint8_t>(level);
        upper_[first + i].resize(static_cast<size_t>(level));
    }

    // a small graph is built serially: vectors inserted at the same moment
    // cannot see each other, which costs recall while there are few nodes
    size_t done = 0;
    for (; done < n && first + done < kSerialBuild; ++done) insert(first + static_cast<uint32_t>(done));
    detail::parallel_for(n - done, [&](size_t i) { insert(first + static_cast<uint32_t>(done + i)); });
}

void HNSWIndex::insert(uint32_t slot) {
    const int level = levels_[slot];

    // read the entry point; a node that raises the top level keeps the lock for
    // its whole insert, so two threads never raise it at once (rare: ~1 in M^level)
    std::unique_lock<std::mutex> top(top_lock_);
    const int top_level = max_level_;
    uint32_t ep = entry_;
    if (top_level < 0) {   // first node
        entry_ = slot;
        max_level_ = level;
        return;
    }
    if (level <= top_level) top.unlock();

    // descend greedily to the new node's top level, then link it on each level
    // from there down, searching with ef_construction
    const float* q = vec(slot);
    for (int l = top_level; l > level; --l) ep = greedy_closest(q, ep, l, true);

    vector<uint32_t> entry_points{ep};
    for (int l = std::min(level, top_level); l >= 0; --l) {
        const vector<Candidate> found = search_layer(q, entry_points, ef_construction_, l, false, true);
        const vector<uint32_t> chosen = select_neighbors(found, M_);
        set_own_neighbors(slot, l, chosen);
        for (uint32_t nb : chosen) add_link(nb, slot, l);

        entry_points.clear();
        for (const Candidate& c : found) entry_points.push_back(c.second);
    }

    if (level > top_level) {   // still holding top_lock_
        max_level_ = level;
        entry_ = slot;
    }
}

void HNSWIndex::add_link(uint32_t node, uint32_t new_neighbor, int level) {
    std::lock_guard<std::mutex> guard(lock_for(node));
    const span<const uint32_t> current = neighbors(node, level);
    vector<uint32_t> list(current.begin(), current.end());
    if (list.size() < capacity(level)) {
        list.push_back(new_neighbor);
        set_neighbors(node, level, list);
        return;
    }

    // full: keep the best capacity(level) of the old neighbors plus the new one
    const float* base = vec(node);
    vector<Candidate> candidates;
    candidates.reserve(list.size() + 1);
    for (uint32_t nb : list) candidates.emplace_back(distance(base, nb), nb);
    candidates.emplace_back(distance(base, new_neighbor), new_neighbor);
    set_neighbors(node, level, select_neighbors(std::move(candidates), capacity(level)));
}

void HNSWIndex::set_own_neighbors(uint32_t slot, int level, const vector<uint32_t>& chosen) {
    std::lock_guard<std::mutex> guard(lock_for(slot));
    const span<const uint32_t> existing = neighbors(slot, level);
    if (existing.empty()) {
        set_neighbors(slot, level, chosen);
        return;
    }

    // other threads already linked into this level (they reached the node
    // through a level above): keep their links, add ours, re-select if over
    vector<uint32_t> merged(existing.begin(), existing.end());
    for (uint32_t nb : chosen)
        if (std::find(merged.begin(), merged.end(), nb) == merged.end()) merged.push_back(nb);
    if (merged.size() <= capacity(level)) {
        set_neighbors(slot, level, merged);
        return;
    }
    vector<Candidate> candidates;
    candidates.reserve(merged.size());
    for (uint32_t nb : merged) candidates.emplace_back(distance(vec(slot), nb), nb);
    set_neighbors(slot, level, select_neighbors(std::move(candidates), capacity(level)));
}

int HNSWIndex::random_level() {
    // u from the raw 32-bit draw, so the same seed gives the same levels on every platform
    const double u = (static_cast<double>(next_random()) + 1.0) / 4294967296.0;
    return std::min(static_cast<int>(-std::log(u) * level_mult_), 255);
}

span<const uint32_t> HNSWIndex::neighbors(uint32_t slot, int level) const {
    if (level == 0) {
        const uint32_t* row = level0_.data() + size_t{slot} * (M0_ + 1);
        return {row + 1, row[0]};
    }
    return upper_[slot][static_cast<size_t>(level - 1)];
}

void HNSWIndex::set_neighbors(uint32_t slot, int level, span<const uint32_t> list) {
    assert(list.size() <= capacity(level));
    if (level == 0) {
        uint32_t* row = level0_.data() + size_t{slot} * (M0_ + 1);
        row[0] = static_cast<uint32_t>(list.size());
        std::copy(list.begin(), list.end(), row + 1);
        return;
    }
    upper_[slot][static_cast<size_t>(level - 1)].assign(list.begin(), list.end());
}

float HNSWIndex::distance(const float* q, uint32_t slot) const { return l2_distance(q, vec(slot), dim_); }

} // namespace vecengine
