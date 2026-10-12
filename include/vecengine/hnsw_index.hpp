#pragma once

#include "vecengine/index.hpp"
#include "vecengine/id_map.hpp"
#include "vecengine/sq.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <span>
#include <optional>
#include <random>
#include <utility>
#include <iosfwd>
#include <memory>
#include <array>
#include <mutex>

namespace vecengine {

using std::size_t;
using std::vector;
using std::span;

// Hierarchical Navigable Small World graph (Malkov & Yashunin, 2016). Every
// vector is a node on level 0; a node also appears on levels 1..L with
// probability falling by 1/M per level. A query descends greedily through the
// sparse upper levels, then runs a beam search of width ef_search on level 0.
// By default vectors are stored in full, so distances are exact and no
// training is needed. With enable_sq() they are stored as 8- or 4-bit codes
// per dimension (see detail::SQ): 4x or 8x less vector memory and faster
// searches, with approximate distances that a RefineIndex can make exact again.
class HNSWIndex : public Index {
public:
    // dim floats per vector. M is the neighbors a node keeps per upper level
    // (2 * M on level 0); ef_construction the beam width used while inserting.
    // A seed makes the level draws reproducible, and the whole graph too when
    // vectors are inserted on one thread (set_num_threads(1)); with several
    // threads the inserts interleave differently from run to run.
    explicit HNSWIndex(size_t dim, size_t M = 16, size_t ef_construction = 200, bool custom_ids = false,
                       std::optional<uint32_t> seed = std::nullopt, Metric metric = Metric::L2);

    // Index overrides; see Index for the contract of each. add_batch inserts
    // the vectors on num_threads() threads.
    // add_batch throws std::logic_error with SQ enabled but not trained.
    size_t remove(span<const int64_t> ids) override;

    // Drops deleted vectors and rebuilds the graph from the live ones, in slot
    // order. Slow (a full rebuild), but repairing a graph in place is not
    // worth its complexity for occasional cleanup.
    size_t compact() override;

    void save(std::ostream& out) const override;

    // Rebuilds an HNSWIndex from the body save() wrote, after load_index() has
    // read the header. Throws std::runtime_error on corrupt data.
    static std::unique_ptr<HNSWIndex> load_body(std::istream& in, uint32_t version);

    size_t slots() const noexcept override { return id_map_.slots(); }
    bool custom_ids() const noexcept override { return id_map_.custom_ids(); }
    bool ids_are_slots() const noexcept override { return id_map_.ids_are_slots(); }

    size_t size() const noexcept override { return id_map_.live(); }
    size_t dim() const noexcept override { return dim_; }

    // Beam width of a query's level-0 search; queries use max(ef_search, k).
    // Higher finds more true neighbors and costs more time.
    size_t ef_search() const noexcept { return ef_search_; }
    void set_ef_search(size_t ef);

    size_t M() const noexcept { return M_; }
    size_t ef_construction() const noexcept { return ef_construction_; }

    // Stores vectors as scalar codes of `bits` (4 or 8; std::invalid_argument
    // otherwise) per dimension instead of floats. Must be called before
    // train() and before any add; throws std::logic_error otherwise. The graph
    // is built and searched with the codes' distances.
    void enable_sq(int bits = 8);

    // Bits per dimension of the codes: 4 or 8, or 0 without enable_sq().
    int sq_bits() const noexcept { return sq_bits_; }

    // Learns each dimension's SQ range from num_vectors (>= 1) sample
    // vectors; later vectors outside it are clamped to its ends, so the sample
    // should cover the data. Call once, after enable_sq() and before any add.
    // Throws std::logic_error without SQ, when already trained or after an
    // add, and std::invalid_argument when num_vectors is 0.
    void train(span<const float> vectors, size_t num_vectors);

    // Whether train() has run (always false without SQ).
    bool trained() const noexcept { return quant_.dim() > 0; }

private:
    // Index implementations: vectors already prepared for the metric.
    void add_batch_impl(span<const float> vecs, size_t n, const int64_t* ids) override;
    vector<Neighbor> query_impl(span<const float> vec, size_t k) const override;

    // (distance to the query, slot)
    using Candidate = std::pair<float, uint32_t>;

    size_t compact_as_base() override;

    // Shared by compact() and compact_as_base(): compacts id_map_ (keep_ids as
    // in IdMap::compact), moves the live vectors, rebuilds the graph and
    // returns the slots reclaimed.
    size_t compact_impl(bool keep_ids);

    // Up to ef nodes of `level` closest to q, nearest first, found by a beam
    // search that starts from entry_points. Deleted nodes are always expanded,
    // since they keep the graph connected; with skip_deleted they are left out
    // of the results, so a query's beam holds only live nodes. Inserts keep
    // them, as routes for the new node's links.
    // `locked` reads each neighbor list under its node's lock, for searches
    // that run while other threads insert; queries never do, so they skip it.
    vector<Candidate> search_layer(const float* q, span<const uint32_t> entry_points, size_t ef, int level,
                                   bool skip_deleted = false, bool locked = false) const;

    // The node of `level` closest to q, found by walking from ep to whichever
    // neighbor is closer until none is. The upper levels only need this one
    // node, so it skips search_layer's heaps. `locked` as in search_layer.
    uint32_t greedy_closest(const float* q, uint32_t ep, int level, bool locked) const;

    // Picks up to max_neighbors of candidates (distances to one base node) as
    // that node's neighbors with the paper's heuristic: a candidate is kept
    // only if it is closer to the base than to every candidate kept so far.
    vector<uint32_t> select_neighbors(vector<Candidate> candidates, size_t max_neighbors) const;

    // Draws the levels of slots [first, first + n) in order, then links them
    // into the graph: serially below kSerialBuild slots, then on num_threads()
    // threads. Their vectors and storage must already be in place, so nothing
    // reallocates while threads run.
    void build(uint32_t first, size_t n);

    // Links one slot into the graph; its level is already drawn. Safe to run
    // on several threads at once.
    void insert(uint32_t slot);

    // Adds a link from node to new_neighbor on level, re-selecting node's
    // neighbors with select_neighbors when its list is already full. Holds
    // node's lock throughout.
    void add_link(uint32_t node, uint32_t new_neighbor, int level);

    // Sets a new node's own neighbor list on level, under its lock. Other
    // threads may have linked into that level already (they can reach the node
    // through a level above before it gets here), so their links are merged in.
    void set_own_neighbors(uint32_t slot, int level, const vector<uint32_t>& chosen);

    // Draws a node's top level: floor(-ln(U) / ln(M)), U uniform in (0, 1].
    int random_level();

    // Neighbor list of slot on level, and its replacement.
    span<const uint32_t> neighbors(uint32_t slot, int level) const;
    void set_neighbors(uint32_t slot, int level, span<const uint32_t> list);

    // Max neighbors a node keeps on level.
    size_t capacity(int level) const noexcept { return level == 0 ? M0_ : M_; }

    // Squared L2 distance from q to the vector at slot (to its decoded code
    // with SQ).
    float distance(const float* q, uint32_t slot) const;

    // The vector at slot as dim() floats: a pointer into data_, or with SQ
    // its code decoded into buf (valid until buf is reused).
    const float* vector_of(uint32_t slot, vector<float>& buf) const;

    // Where slot's vector is stored and how many bytes it takes.
    const char* stored(uint32_t slot) const noexcept {
        return sq_bits_ ? reinterpret_cast<const char*>(codes_.data() + size_t{slot} * code_bytes_)
                        : reinterpret_cast<const char*>(data_.data() + size_t{slot} * dim_);
    }
    size_t stored_bytes() const noexcept { return sq_bits_ ? code_bytes_ : dim_ * sizeof(float); }

    // Asks the CPU to start loading slot's vector into cache. Searches are
    // limited by these random reads, so issuing several before using any lets
    // them overlap.
    void prefetch_vector(uint32_t slot) const noexcept {
        const char* p = stored(slot);
        const size_t bytes = stored_bytes();
        for (size_t b = 0; b < bytes; b += 64) __builtin_prefetch(p + b);
        __builtin_prefetch(p + bytes - 1);   // the last line, when the vector isn't 64-byte aligned
    }

    // The lock guarding slot's neighbor lists. Slots share a fixed pool of
    // locks; that is safe because a thread holds at most one at a time.
    std::mutex& lock_for(uint32_t slot) const noexcept { return link_locks_[slot % kLockStripes]; }

    size_t dim_;
    size_t M_, M0_;            // neighbors per node on upper levels / level 0
    size_t ef_construction_;
    size_t ef_search_ = 64;
    double level_mult_;        // 1 / ln(M)

    // Level draws come from splitmix64, whose whole state is this one number,
    // so files store it as 8 bytes. (std::mt19937's text form differs between
    // libc++ and libstdc++, which made 0.1.0a10 HNSW files platform-bound.)
    uint64_t rng_state_;
    uint32_t next_random() noexcept {
        uint64_t z = (rng_state_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return static_cast<uint32_t>((z ^ (z >> 31)) >> 32);
    }

    int sq_bits_ = 0;          // 0 (floats), 4 or 8
    size_t code_bytes_ = 0;    // with SQ: bytes of codes per slot
    detail::SQ quant_;         // trained ranges; empty until train()
    vector<float> data_;       // row-major, one row per slot; empty with SQ
    vector<uint8_t> codes_;    // with SQ: code_bytes_ per slot, row-major
    vector<uint32_t> level0_;  // per slot, M0_ + 1 entries: count, then neighbors
    vector<vector<vector<uint32_t>>> upper_; // upper_[slot][level - 1]: neighbors on levels >= 1
    vector<uint8_t> levels_;   // top level of each slot
    uint32_t entry_ = 0;       // a node on max_level_, where every search starts
    int max_level_ = -1;       // -1 while the graph is empty

    detail::IdMap id_map_;

    static constexpr size_t kSerialBuild = 1024;   // slots below this are inserted on one thread
    static constexpr size_t kLockStripes = 1024;
    mutable std::array<std::mutex, kLockStripes> link_locks_;
    std::mutex top_lock_;      // guards entry_ and max_level_ while inserting
};

} // namespace vecengine
