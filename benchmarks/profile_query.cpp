// Per-section query time breakdown on SIFT1M, single-threaded.
//
//   cmake -B build-prof -DCMAKE_BUILD_TYPE=Release -DVECENGINE_PROFILE=ON -DVECENGINE_BUILD_TESTS=OFF
//   cmake --build build-prof --target profile_query -j
//   ./build-prof/profile_query benchmarks/data/sift [nprobe]
//
// Builds plain IVF, IVF-PQ16 and IVF-PQ16 + re-ranking (k_factor 16) with
// seed 1, runs every query once to warm caches, then reports microseconds per
// query for each section timed in src/profile.hpp. Run on an idle machine.

#include "profile.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/refine_index.hpp"
#include "vecengine/threads.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace vecengine;

static std::vector<float> read_fvecs(const std::string& path, size_t dim, size_t& n) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(1);
    }
    n = static_cast<size_t>(f.tellg()) / (4 * (dim + 1));
    f.seekg(0);
    std::vector<float> out(n * dim);
    int32_t row_dim;
    for (size_t i = 0; i < n; ++i) {
        f.read(reinterpret_cast<char*>(&row_dim), 4);
        f.read(reinterpret_cast<char*>(out.data() + i * dim), 4 * dim);
    }
    return out;
}

static void report(const char* label, const Index& index, const std::vector<float>& queries, size_t nq) {
    for (double& s : detail::g_prof) s = 0;
    auto start = std::chrono::steady_clock::now();
    index.query_batch(queries, nq, 10);
    double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    const char* names[] = {"coarse search", "ADC table build", "ADC code scan", "raw vector scan", "result drain", "re-rank"};
    double accounted = 0;
    std::printf("\n%s: %.0f us/query (%.0f QPS)\n", label, total / nq * 1e6, nq / total);
    for (int i = 0; i < detail::kProfSlots; ++i) {
        double s = detail::g_prof[i];
        if (s == 0) continue;
        accounted += s;
        std::printf("  %-16s %7.1f us  %5.1f%%\n", names[i], s / nq * 1e6, 100 * s / total);
    }
    std::printf("  %-16s %7.1f us  %5.1f%%\n", "other", (total - accounted) / nq * 1e6, 100 * (total - accounted) / total);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <sift dir> [nprobe]\n", argv[0]);
        return 1;
    }
    const std::string dir = argv[1];
    const size_t nprobe = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 16;
    const size_t dim = 128, nlist = 1000, k_factor = 16;
    const uint32_t seed = 1;

    size_t nb, nl, nq;
    auto base = read_fvecs(dir + "/sift_base.fvecs", dim, nb);
    auto learn = read_fvecs(dir + "/sift_learn.fvecs", dim, nl);
    auto queries = read_fvecs(dir + "/sift_query.fvecs", dim, nq);

    IVFIndex ivf(dim, nlist, nprobe);
    ivf.train(learn, nl, 25, seed);
    ivf.add_batch(base, nb);

    IVFIndex pq(dim, nlist, nprobe);
    pq.enable_pq(16);
    pq.train(learn, nl, 25, seed);
    RefineIndex refine(pq, k_factor);
    refine.add_batch(base, nb);

    set_num_threads(1);
    ivf.query_batch(queries, nq, 10);
    pq.query_batch(queries, nq, 10);
    refine.query_batch(queries, nq, 10);

    std::printf("SIFT1M, nprobe %zu, %zu queries, 1 thread\n", nprobe, nq);
    report("plain IVF", ivf, queries, nq);
    report("IVF-PQ16", pq, queries, nq);
    report("IVF-PQ16 + R16", refine, queries, nq);
}
