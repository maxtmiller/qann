// Study for issue #23: does storing each node's level-0 links next to its
// vector (one memory read per visited node) speed up HNSW queries over the
// current layout (links in level0_, vectors in data_ / codes_: two reads)?
//
// Loads an HNSWIndex file saved by qann (float or SQ codes, L2), builds both
// layouts from it, and runs the same search on each: greedy descent through
// the upper levels, then a level-0 beam search with the library's prefetching.
// Results are identical by construction; only memory placement differs.
//
//   cmake --build build --target vecengine_core
//   clang++ -std=c++20 -O3 -mcpu=native -Iinclude -Isrc benchmarks/hnsw_layout_bench.cpp \
//       build/libvecengine_core.a -framework Accelerate -o hnsw_layout_bench
//   ./hnsw_layout_bench index.qann benchmarks/data/sift [ef ...]
#include "vecengine/distance.hpp"
#include "vecengine/sq.hpp"
#include "serialize.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace vecengine;
using namespace vecengine::detail;
using clk = std::chrono::steady_clock;

struct Graph {
    size_t dim = 0, M0 = 0, slots = 0;
    int max_level = -1;
    uint32_t entry = 0;
    int bits = 0;                        // 0 = floats
    SQ quant;
    size_t vec_bytes = 0;                // bytes of one stored vector
    vector<uint8_t> vectors;             // current layout: slots * vec_bytes
    vector<uint32_t> level0;             // current layout: slots * (M0 + 1)
    vector<vector<vector<uint32_t>>> upper;
    vector<uint8_t> rows;                // interleaved layout: per slot, links then vector
    size_t row_bytes = 0;
};

static Graph load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    const Header h = read_header(in);
    if (h.kind != IndexKind::HNSW || h.version < 6) throw std::runtime_error("need a version-6 HNSWIndex file");
    Graph g;
    g.dim = read_pod<uint64_t>(in);
    const auto M = read_pod<uint64_t>(in);
    g.M0 = 2 * M;
    (void)read_pod<uint64_t>(in);   // ef_construction
    (void)read_pod<uint64_t>(in);   // ef_search
    g.slots = read_pod<uint64_t>(in);
    g.max_level = read_pod<int32_t>(in);
    g.entry = read_pod<uint32_t>(in);
    (void)read_pod<uint64_t>(in);   // generator state
    g.bits = read_pod<uint8_t>(in);
    if (g.bits) {
        g.quant.bits = g.bits;
        g.quant.vmin = read_vec<float>(in, g.dim);
        g.quant.step = read_vec<float>(in, g.dim);
        g.vectors = read_vec<uint8_t>(in, UINT64_MAX);
        g.vec_bytes = g.quant.code_bytes();
        (void)read_vec<float>(in, 0);
    } else {
        const auto data = read_vec<float>(in, g.slots * g.dim);
        g.vec_bytes = g.dim * sizeof(float);
        g.vectors.resize(data.size() * sizeof(float));
        std::memcpy(g.vectors.data(), data.data(), g.vectors.size());
    }
    const auto levels = read_vec<uint8_t>(in, g.slots);
    g.level0 = read_vec<uint32_t>(in, g.slots * (g.M0 + 1));
    g.upper.resize(g.slots);
    for (size_t s = 0; s < g.slots; ++s) {
        g.upper[s].resize(levels[s]);
        for (auto& list : g.upper[s]) list = read_vec<uint32_t>(in, M);
    }

    // interleaved: [count][M0 neighbors][vector], rows back to back like hnswlib
    const size_t links = (g.M0 + 1) * sizeof(uint32_t);
    g.row_bytes = links + g.vec_bytes;
    g.rows.resize(g.slots * g.row_bytes);
    for (size_t s = 0; s < g.slots; ++s) {
        std::memcpy(g.rows.data() + s * g.row_bytes, g.level0.data() + s * (g.M0 + 1), links);
        std::memcpy(g.rows.data() + s * g.row_bytes + links, g.vectors.data() + s * g.vec_bytes, g.vec_bytes);
    }
    return g;
}

static void prefetch(const void* p, size_t bytes) {
    const char* c = static_cast<const char*>(p);
    for (size_t b = 0; b < bytes; b += 64) __builtin_prefetch(c + b);
    __builtin_prefetch(c + bytes - 1);
}

// Where one layout keeps a slot's level-0 links and vector.
struct Separate {
    const Graph& g;
    const uint32_t* links(uint32_t s) const { return g.level0.data() + size_t{s} * (g.M0 + 1); }
    const uint8_t* vec(uint32_t s) const { return g.vectors.data() + size_t{s} * g.vec_bytes; }
    static constexpr const char* name = "separate";
};
struct Interleaved {
    const Graph& g;
    const uint32_t* links(uint32_t s) const {
        return reinterpret_cast<const uint32_t*>(g.rows.data() + size_t{s} * g.row_bytes);
    }
    const uint8_t* vec(uint32_t s) const { return g.rows.data() + size_t{s} * g.row_bytes + (g.M0 + 1) * 4; }
    static constexpr const char* name = "interleaved";
};

struct Stats { uint64_t dists = 0, expanded = 0; };

thread_local vector<uint32_t> t_tags;
thread_local uint32_t t_gen = 0;

template <class L>
static vector<uint32_t> search(const L& lay, const float* q, size_t ef, size_t k, Stats& st) {
    const Graph& g = lay.g;
    auto dist = [&](uint32_t s) {
        ++st.dists;
        return g.bits ? g.quant.distance(q, lay.vec(s))
                      : l2_distance(q, reinterpret_cast<const float*>(lay.vec(s)), g.dim);
    };

    // greedy descent through the upper levels (same lists in both layouts)
    uint32_t ep = g.entry;
    float best = dist(ep);
    for (int level = g.max_level; level > 0; --level) {
        for (bool moved = true; moved;) {
            moved = false;
            const auto& list = g.upper[ep][static_cast<size_t>(level - 1)];
            for (uint32_t n : list) prefetch(lay.vec(n), g.vec_bytes);
            const uint32_t from = ep;
            for (uint32_t n : list) {
                const float d = dist(n);
                if (d < best) { best = d; ep = n; }
            }
            moved = ep != from;
        }
    }

    // level-0 beam search, as HNSWIndex::search_layer
    if (t_tags.size() < g.slots) t_tags.assign(g.slots, 0);
    if (++t_gen == 0) { std::fill(t_tags.begin(), t_tags.end(), 0); t_gen = 1; }
    using C = std::pair<float, uint32_t>;
    std::priority_queue<C> results;
    std::priority_queue<C, vector<C>, std::greater<C>> frontier;
    t_tags[ep] = t_gen;
    results.emplace(best, ep);
    frontier.emplace(best, ep);
    vector<uint32_t> fresh;
    fresh.reserve(g.M0);
    while (!frontier.empty()) {
        const C cur = frontier.top();
        if (results.size() >= ef && cur.first > results.top().first) break;
        frontier.pop();
        ++st.expanded;
        if (!frontier.empty()) __builtin_prefetch(lay.links(frontier.top().second));
        const uint32_t* row = lay.links(cur.second);
        fresh.clear();
        for (uint32_t i = 1; i <= row[0]; ++i) {
            const uint32_t n = row[i];
            if (t_tags[n] == t_gen) continue;
            t_tags[n] = t_gen;
            fresh.push_back(n);
            prefetch(lay.vec(n), g.vec_bytes);
        }
        for (uint32_t n : fresh) {
            const float d = dist(n);
            if (results.size() < ef || d < results.top().first) {
                frontier.emplace(d, n);
                results.emplace(d, n);
                if (results.size() > ef) results.pop();
            }
        }
    }
    vector<C> all;
    while (!results.empty()) { all.push_back(results.top()); results.pop(); }
    std::reverse(all.begin(), all.end());
    vector<uint32_t> ids;
    for (size_t i = 0; i < std::min(k, all.size()); ++i) ids.push_back(all[i].second);
    return ids;
}

static vector<float> read_fvecs(const std::string& path, size_t& n, size_t& dim) {
    std::ifstream in(path, std::ios::binary);
    int32_t d;
    in.read(reinterpret_cast<char*>(&d), 4);
    in.seekg(0, std::ios::end);
    const size_t bytes = in.tellg();
    dim = d;
    n = bytes / (4 + 4 * dim);
    vector<float> out(n * dim);
    in.seekg(0);
    for (size_t i = 0; i < n; ++i) {
        in.seekg(4, std::ios::cur);
        in.read(reinterpret_cast<char*>(out.data() + i * dim), 4 * dim);
    }
    return out;
}

template <class L>
static void run(const L& lay, const vector<float>& qs, size_t nq, const vector<int32_t>& gt, size_t gt_k,
                size_t ef, unsigned threads, double& qps, double& recall, Stats& total) {
    const size_t k = 10;
    vector<vector<uint32_t>> res(nq);
    vector<Stats> stats(threads);
    std::atomic<size_t> next{0};
    const auto start = clk::now();
    vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t)
        pool.emplace_back([&, t] {
            for (size_t i; (i = next.fetch_add(1)) < nq;) res[i] = search(lay, qs.data() + i * lay.g.dim, ef, k, stats[t]);
        });
    for (auto& th : pool) th.join();
    qps = nq / std::chrono::duration<double>(clk::now() - start).count();
    size_t hit = 0;
    for (size_t i = 0; i < nq; ++i) {
        std::set<int32_t> truth(gt.begin() + i * gt_k, gt.begin() + i * gt_k + k);
        for (uint32_t id : res[i]) hit += truth.count(static_cast<int32_t>(id));
    }
    recall = static_cast<double>(hit) / (nq * k);
    total = {};
    for (auto& s : stats) { total.dists += s.dists; total.expanded += s.expanded; }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <hnsw.qann> <sift dir> [ef ...]\n", argv[0]);
        return 1;
    }
    const Graph g = load(argv[1]);
    const std::string dir = argv[2];
    size_t nq, qdim, gn, gk;
    const auto qs = read_fvecs(dir + "/sift_query.fvecs", nq, qdim);
    const auto gtf = read_fvecs(dir + "/sift_groundtruth.ivecs", gn, gk);
    vector<int32_t> gt(gtf.size());
    std::memcpy(gt.data(), gtf.data(), gt.size() * 4);
    vector<size_t> efs;
    for (int i = 3; i < argc; ++i) efs.push_back(std::strtoul(argv[i], nullptr, 10));
    if (efs.empty()) efs = {64, 256};

    std::printf("%zu slots, dim %zu, M0 %zu, %s, vector %zu B, link row %zu B, interleaved row %zu B\n", g.slots,
                g.dim, g.M0, g.bits ? (g.bits == 8 ? "SQ8" : "SQ4") : "floats", g.vec_bytes, (g.M0 + 1) * 4,
                g.row_bytes);
    const Separate sep{g};
    const Interleaved inter{g};
    const unsigned all = std::thread::hardware_concurrency();
    for (size_t ef : efs) {
        for (unsigned threads : {1u, all}) {
            // alternate the layouts and keep each one's best, so machine load hits both alike
            double best_sep = 0, best_int = 0, rs = 0, ri = 0;
            Stats ss, si;
            for (int round = 0; round < 4; ++round) {
                double q;
                run(sep, qs, nq, gt, gk, ef, threads, q, rs, ss);
                best_sep = std::max(best_sep, q);
                run(inter, qs, nq, gt, gk, ef, threads, q, ri, si);
                best_int = std::max(best_int, q);
            }
            std::printf("ef %4zu, %u thr: separate %7.0f qps, interleaved %7.0f qps (%.2fx) | recall %.4f / %.4f | "
                        "%.0f dists, %.0f expansions per query\n",
                        ef, threads, best_sep, best_int, best_int / best_sep, rs, ri,
                        double(ss.dists) / nq, double(ss.expanded) / nq);
        }
    }
}
