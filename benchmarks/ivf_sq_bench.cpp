// Study for issue #25: would SQ codes in IVF lists beat raw floats?
//
// Loads a float IVFIndex saved by qann (no PQ) and SIFT1M, then builds SQ
// codes for every list: 8-bit and 4-bit codes of the residual (vector minus
// its list's centroid, ranges learned from sift_learn residuals) and 8-bit
// codes of the vector itself. Every variant runs the same search: brute-force
// coarse step, then a sequential scan of the nprobe lists into a top-k, as
// IVFIndex::scan does for raw floats. Only the list storage and its distance
// differ. Variants alternate over 3 rounds, keeping each one's best.
//
//   cmake --build build --target vecengine_core
//   clang++ -std=c++20 -O3 -mcpu=native -Iinclude -Isrc benchmarks/ivf_sq_bench.cpp \
//       build/libvecengine_core.a -framework Accelerate -o ivf_sq_bench
//   ./ivf_sq_bench ivf.qann benchmarks/data/sift [nprobe ...]
#include "vecengine/distance.hpp"
#include "vecengine/index.hpp"
#include "vecengine/sq.hpp"
#include "serialize.hpp"
#include "topk.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace vecengine;
using namespace vecengine::detail;
using clk = std::chrono::steady_clock;

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

struct List {
    vector<uint32_t> ids;
    vector<float> vecs;
    vector<uint8_t> res8, res4, dir8;
};

struct Ivf {
    size_t dim = 0, nlist = 0;
    vector<float> centroids;
    vector<List> lists;
    SQ res8, res4, dir8;
};

static Ivf load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    const Header h = read_header(in);
    if (h.kind != IndexKind::IVF) throw std::runtime_error("need an IVFIndex file");
    Ivf ivf;
    ivf.dim = read_pod<uint64_t>(in);
    ivf.nlist = read_pod<uint64_t>(in);
    (void)read_pod<uint64_t>(in);   // nprobe
    const auto slots = read_pod<uint64_t>(in);
    (void)read_pod<uint8_t>(in);    // trained
    (void)read_pod<uint8_t>(in);    // pq_distance
    ivf.centroids = read_vec<float>(in, ivf.nlist * ivf.dim);
    if (read_pod<uint8_t>(in)) throw std::runtime_error("need an IVFIndex without PQ");
    ivf.lists.resize(ivf.nlist);
    for (auto& list : ivf.lists) {
        list.ids = read_vec<uint32_t>(in, slots);
        list.vecs = read_vec<float>(in, slots * ivf.dim);
    }
    return ivf;
}

// Index of the centroid nearest to v.
static size_t nearest(const Ivf& ivf, const float* v) {
    size_t best = 0;
    float bd = INFINITY;
    for (size_t c = 0; c < ivf.nlist; ++c) {
        const float d = l2_distance(v, ivf.centroids.data() + c * ivf.dim, ivf.dim);
        if (d < bd) { bd = d; best = c; }
    }
    return best;
}

static void build_codes(Ivf& ivf, const vector<float>& learn, size_t n_learn) {
    const size_t dim = ivf.dim;
    // residual ranges from the learn set, as train() would learn them
    vector<float> res(n_learn * dim);
    for (size_t i = 0; i < n_learn; ++i) {
        const float* v = learn.data() + i * dim;
        const float* c = ivf.centroids.data() + nearest(ivf, v) * dim;
        for (size_t d = 0; d < dim; ++d) res[i * dim + d] = v[d] - c[d];
    }
    ivf.res8.bits = 8;
    ivf.res8.train(res, n_learn, dim);
    ivf.res4.bits = 4;
    ivf.res4.train(res, n_learn, dim);
    ivf.dir8.bits = 8;
    ivf.dir8.train(learn, n_learn, dim);

    vector<float> r(dim);
    for (size_t l = 0; l < ivf.nlist; ++l) {
        List& list = ivf.lists[l];
        const size_t n = list.ids.size();
        const float* c = ivf.centroids.data() + l * dim;
        list.res8.resize(n * ivf.res8.code_bytes());
        list.res4.resize(n * ivf.res4.code_bytes());
        list.dir8.resize(n * ivf.dir8.code_bytes());
        for (size_t j = 0; j < n; ++j) {
            const float* v = list.vecs.data() + j * dim;
            for (size_t d = 0; d < dim; ++d) r[d] = v[d] - c[d];
            ivf.res8.encode(r.data(), list.res8.data() + j * ivf.res8.code_bytes());
            ivf.res4.encode(r.data(), list.res4.data() + j * ivf.res4.code_bytes());
            ivf.dir8.encode(v, list.dir8.data() + j * ivf.dir8.code_bytes());
        }
    }
}

enum Variant { kFloat, kRes8, kDir8, kRes4, kRes4R4, kVariants };
static const char* kNames[] = {"floats", "SQ8 residual", "SQ8 vector", "SQ4 residual", "SQ4 residual + R4"};

static vector<uint32_t> search(const Ivf& ivf, const vector<float>& raw, const float* q, size_t nprobe, size_t k,
                               Variant v) {
    const size_t dim = ivf.dim;
    vector<std::pair<float, uint32_t>> coarse(ivf.nlist);
    for (size_t c = 0; c < ivf.nlist; ++c)
        coarse[c] = {l2_distance(q, ivf.centroids.data() + c * dim, dim), static_cast<uint32_t>(c)};
    std::partial_sort(coarse.begin(), coarse.begin() + nprobe, coarse.end());

    const size_t keep = v == kRes4R4 ? 4 * k : k;
    TopK top(keep);
    vector<float> qr(dim);
    for (size_t p = 0; p < nprobe; ++p) {
        const List& list = ivf.lists[coarse[p].second];
        const size_t n = list.ids.size();
        const float* c = ivf.centroids.data() + coarse[p].second * dim;
        for (size_t d = 0; d < dim; ++d) qr[d] = q[d] - c[d];
        for (size_t j = 0; j < n; ++j) {
            float dist;
            switch (v) {
                case kFloat: dist = l2_distance(q, list.vecs.data() + j * dim, dim); break;
                case kRes8:  dist = ivf.res8.distance(qr.data(), list.res8.data() + j * ivf.res8.code_bytes()); break;
                case kDir8:  dist = ivf.dir8.distance(q, list.dir8.data() + j * ivf.dir8.code_bytes()); break;
                default:     dist = ivf.res4.distance(qr.data(), list.res4.data() + j * ivf.res4.code_bytes()); break;
            }
            if (dist < top.threshold()) top.push(dist, list.ids[j]);
        }
    }
    auto found = top.take_sorted();
    if (v == kRes4R4) {
        for (auto& nb : found) nb.distance = l2_distance(q, raw.data() + nb.index * dim, dim);
        std::sort(found.begin(), found.end(), [](const Neighbor& a, const Neighbor& b) { return a.distance < b.distance; });
    }
    vector<uint32_t> ids;
    for (size_t i = 0; i < std::min(k, found.size()); ++i) ids.push_back(static_cast<uint32_t>(found[i].index));
    return ids;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <ivf.qann> <sift dir> [nprobe ...]\n", argv[0]);
        return 1;
    }
    Ivf ivf = load(argv[1]);
    const std::string dir = argv[2];
    size_t nb, nl, nq, gk, d;
    const auto raw = read_fvecs(dir + "/sift_base.fvecs", nb, d);
    const auto learn = read_fvecs(dir + "/sift_learn.fvecs", nl, d);
    const auto qs = read_fvecs(dir + "/sift_query.fvecs", nq, d);
    const auto gtf = read_fvecs(dir + "/sift_groundtruth.ivecs", nq, gk);
    vector<int32_t> gt(gtf.size());
    std::memcpy(gt.data(), gtf.data(), gt.size() * 4);
    build_codes(ivf, learn, nl);
    vector<size_t> nprobes;
    for (int i = 3; i < argc; ++i) nprobes.push_back(std::strtoul(argv[i], nullptr, 10));
    if (nprobes.empty()) nprobes = {16, 64};

    const size_t bytes[] = {ivf.dim * 4, ivf.res8.code_bytes(), ivf.dir8.code_bytes(), ivf.res4.code_bytes(),
                            ivf.res4.code_bytes() + ivf.dim * 4};
    std::printf("nlist %zu, dim %zu, %zu vectors\n", ivf.nlist, ivf.dim, nb);
    const size_t k = 10;
    const unsigned all = std::thread::hardware_concurrency();
    for (size_t nprobe : nprobes) {
        for (unsigned threads : {1u, all}) {
            double best[kVariants] = {};
            double recall[kVariants] = {};
            for (int round = 0; round < 3; ++round) {
                for (int v = 0; v < kVariants; ++v) {
                    vector<vector<uint32_t>> res(nq);
                    std::atomic<size_t> next{0};
                    const auto start = clk::now();
                    vector<std::thread> pool;
                    for (unsigned t = 0; t < threads; ++t)
                        pool.emplace_back([&] {
                            for (size_t i; (i = next.fetch_add(1)) < nq;)
                                res[i] = search(ivf, raw, qs.data() + i * ivf.dim, nprobe, k, static_cast<Variant>(v));
                        });
                    for (auto& th : pool) th.join();
                    best[v] = std::max(best[v], nq / std::chrono::duration<double>(clk::now() - start).count());
                    size_t hit = 0;
                    for (size_t i = 0; i < nq; ++i) {
                        std::set<int32_t> truth(gt.begin() + i * gk, gt.begin() + i * gk + k);
                        for (uint32_t id : res[i]) hit += truth.count(static_cast<int32_t>(id));
                    }
                    recall[v] = static_cast<double>(hit) / (nq * k);
                }
            }
            std::printf("nprobe %zu, %u threads:\n", nprobe, threads);
            for (int v = 0; v < kVariants; ++v)
                std::printf("  %-20s %4zu B/vec  recall %.4f  %8.0f qps  (%.2fx floats)\n", kNames[v], bytes[v], recall[v],
                            best[v], best[v] / best[kFloat]);
        }
    }
}
