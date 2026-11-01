// Per-section single-query profile of IVF fast-scan under a RefineIndex, the
// path ann-benchmarks and VIBE time: one DenseIndex::query per query, one thread.
//
//   cmake -B build-prof -DCMAKE_BUILD_TYPE=Release -DVECENGINE_PROFILE=ON -DVECENGINE_BUILD_TESTS=OFF
//   cmake --build build-prof --target profile_fastscan -j
//   ./build-prof/profile_fastscan <prefix> <nlist> <m> [nq] [nprobes] [k_factors] [rotate]
//
// nprobes and k_factors are comma-separated lists (default 8,16,32 and 8) and
// rotate is off, on or both (default both). For the high-recall end of the
// ann-benchmarks grid on fashion-mnist (issue #62):
//
//   ./build-prof/profile_fastscan fm 1024 196 2000 8,16,32,64,128,256 8,16,32,64 off
//
// <prefix>_train.f32 and <prefix>_test.f32 hold rows of float32 and
// <prefix>_neighbors.i32 the 10 true neighbors per test row as int32, e.g. from
// an ann-benchmarks file:
//
//   python3 -c "import h5py, sys; f = h5py.File(sys.argv[1]); p = sys.argv[2]
//   f['train'][:].astype('f4').tofile(p + '_train.f32'); f['test'][:].astype('f4').tofile(p + '_test.f32')
//   f['neighbors'][:, :10].astype('i4').tofile(p + '_neighbors.i32'); print(f['train'].shape)" data.hdf5 fm
//
// Builds the index once per rotate setting, then for every nprobe and k_factor
// prints one row: recall@10, microseconds per query and the time in each section
// timed in src/profile.hpp, per query (best of 3 runs).

#include "profile.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/refine_index.hpp"
#include "vecengine/threads.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace vecengine;
using clk = std::chrono::steady_clock;

template <typename T>
static std::vector<T> read_all(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(1);
    }
    std::vector<T> v(static_cast<size_t>(f.tellg()) / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
    return v;
}

static std::vector<size_t> parse_list(const char* s) {
    std::vector<size_t> out;
    std::stringstream in(s);
    for (std::string item; std::getline(in, item, ',');) out.push_back(std::stoul(item));
    return out;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <prefix> <nlist> <m> [nq] [nprobes] [k_factors] [off|on|both]\n", argv[0]);
        return 1;
    }
    const std::string prefix = argv[1];
    const size_t nlist = std::stoul(argv[2]), m = std::stoul(argv[3]);
    const auto train = read_all<float>(prefix + "_train.f32");
    auto queries = read_all<float>(prefix + "_test.f32");
    const auto truth = read_all<int32_t>(prefix + "_neighbors.i32");
    const size_t all_q = truth.size() / 10, d = queries.size() / all_q, n = train.size() / d;
    const size_t nq = std::min(all_q, argc > 4 ? std::stoul(argv[4]) : size_t{2000});
    const auto nprobes = parse_list(argc > 5 ? argv[5] : "8,16,32");
    const auto k_factors = parse_list(argc > 6 ? argv[6] : "8");
    const std::string rotate_arg = argc > 7 ? argv[7] : "both";
    std::vector<bool> rotates;
    if (rotate_arg != "on") rotates.push_back(false);
    if (rotate_arg != "off") rotates.push_back(true);
    std::printf("%zu x %zu, %zu queries, nlist %zu, m %zu; microseconds per query\n", n, d, nq, nlist, m);

    // Column order follows the slots in src/profile.hpp; "other" is the rest,
    // the rotation when on plus timer overhead.
    const char* names[] = {"coarse", "ADC", "scan", "raw", "drain", "rerank", "setup", "innerC", "perlist", "lutround"};
    std::printf("\n%-3s %7s %7s %8s %9s", "rot", "nprobe", "k_fac", "recall", "total");
    for (const char* name : names) std::printf(" %8s", name);
    std::printf(" %8s\n", "other");
    for (bool rotate : rotates) {
        set_num_threads(0);
        IVFIndex ivf(d, nlist);
        ivf.enable_pq(m, 16, true, rotate);
        ivf.train(train, n, 25, 1);
        RefineIndex refine(ivf, k_factors[0]);
        refine.add_batch(train, n);
        set_num_threads(1);
        for (size_t nprobe : nprobes) for (size_t k_factor : k_factors) {
            ivf.set_nprobe(nprobe);
            refine.set_k_factor(k_factor);
            std::vector<std::vector<Neighbor>> got(nq);
            auto run = [&] {
                const auto t0 = clk::now();
                for (size_t i = 0; i < nq; ++i) got[i] = refine.query({queries.data() + i * d, d}, 10);
                return std::chrono::duration<double>(clk::now() - t0).count();
            };
            run();   // warm caches
            double best = 1e9, best_prof[detail::kProfSlots] = {};
            for (int rep = 0; rep < 3; ++rep) {
                for (double& s : detail::g_prof) s = 0;
                const double t = run();
                if (t < best) {
                    best = t;
                    std::copy(std::begin(detail::g_prof), std::end(detail::g_prof), best_prof);
                }
            }
            size_t hits = 0;
            for (size_t i = 0; i < nq; ++i) {
                const std::set<int32_t> want(truth.begin() + i * 10, truth.begin() + i * 10 + 10);
                for (const auto& nb : got[i]) hits += want.count(static_cast<int32_t>(nb.index));
            }
            std::printf("%-3s %7zu %7zu %8.4f %9.1f", rotate ? "on" : "off", nprobe, k_factor,
                        double(hits) / (nq * 10), best / nq * 1e6);
            double acc = 0;
            for (int i = 0; i < detail::kProfSlots; ++i) {
                acc += best_prof[i];
                std::printf(" %8.1f", best_prof[i] / nq * 1e6);
            }
            std::printf(" %8.1f\n", (best - acc) / nq * 1e6);
            std::fflush(stdout);
        }
    }
}
