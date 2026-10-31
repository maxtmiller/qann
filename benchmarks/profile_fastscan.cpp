// Per-section single-query profile of IVF fast-scan under a RefineIndex, the
// path ann-benchmarks and VIBE time: one DenseIndex::query per query, one thread.
//
//   cmake -B build-prof -DCMAKE_BUILD_TYPE=Release -DVECENGINE_PROFILE=ON -DVECENGINE_BUILD_TESTS=OFF
//   cmake --build build-prof --target profile_fastscan -j
//   ./build-prof/profile_fastscan <prefix> <nlist> <m> [nq]
//
// <prefix>_train.f32 and <prefix>_test.f32 hold rows of float32 and
// <prefix>_neighbors.i32 the 10 true neighbors per test row as int32, e.g. from
// an ann-benchmarks file:
//
//   python3 -c "import h5py, sys; f = h5py.File(sys.argv[1]); p = sys.argv[2]
//   f['train'][:].astype('f4').tofile(p + '_train.f32'); f['test'][:].astype('f4').tofile(p + '_test.f32')
//   f['neighbors'][:, :10].astype('i4').tofile(p + '_neighbors.i32'); print(f['train'].shape)" data.hdf5 fm
//
// Builds the index with rotation on and off, then for nprobe 8, 16 and 32 at
// k_factor 8 reports recall@10, microseconds per query and the time in each
// section timed in src/profile.hpp (best of 3 runs).

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

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <prefix> <nlist> <m> [nq]\n", argv[0]);
        return 1;
    }
    const std::string prefix = argv[1];
    const size_t nlist = std::stoul(argv[2]), m = std::stoul(argv[3]);
    const auto train = read_all<float>(prefix + "_train.f32");
    auto queries = read_all<float>(prefix + "_test.f32");
    const auto truth = read_all<int32_t>(prefix + "_neighbors.i32");
    const size_t all_q = truth.size() / 10, d = queries.size() / all_q, n = train.size() / d;
    const size_t nq = std::min(all_q, argc > 4 ? std::stoul(argv[4]) : size_t{2000});
    std::printf("%zu x %zu, %zu queries, nlist %zu, m %zu\n", n, d, nq, nlist, m);

    const char* names[] = {"coarse step", "ADC table build", "code scan", "raw vector scan", "result drain", "re-rank",
                           "query setup", "inner table (C)", "per-list setup"};
    for (bool rotate : {false, true}) {
        set_num_threads(0);
        IVFIndex ivf(d, nlist);
        ivf.enable_pq(m, 16, true, rotate);
        ivf.train(train, n, 25, 1);
        RefineIndex refine(ivf, 8);
        refine.add_batch(train, n);
        set_num_threads(1);
        for (size_t nprobe : {8, 16, 32}) {
            ivf.set_nprobe(nprobe);
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
            std::printf("\nrotate=%d nprobe=%zu: recall@10 %.4f, %.1f us/query (%.0f QPS)\n", rotate, nprobe,
                        double(hits) / (nq * 10), best / nq * 1e6, nq / best);
            double acc = 0;
            for (int i = 0; i < detail::kProfSlots; ++i) {
                if (best_prof[i] == 0) continue;
                acc += best_prof[i];
                std::printf("  %-16s %7.1f us  %5.1f%%\n", names[i], best_prof[i] / nq * 1e6, 100 * best_prof[i] / best);
            }
            std::printf("  %-16s %7.1f us  %5.1f%%  (the rotation when on, and timer overhead)\n", "other",
                        (best - acc) / nq * 1e6, 100 * (best - acc) / best);
        }
    }
}
