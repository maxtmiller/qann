#include "vecengine/hnsw_index.hpp"
#include "vecengine/flat_index.hpp"
#include "vecengine/threads.hpp"
#include <cstdio>
#include <random>
#include <set>
#include <algorithm>
using namespace vecengine;
static std::vector<float> rnd(size_t n, size_t dim, unsigned seed) { std::mt19937 r(seed); std::normal_distribution<float> d(0, 1); std::vector<float> v(n * dim); for (auto& x : v) x = d(r); return v; }
static double rec(const std::vector<std::vector<Neighbor>>& g, const std::vector<std::vector<Neighbor>>& t) { double s = 0; for (size_t q = 0; q < t.size(); ++q) { std::set<size_t> ids; for (auto& nb : t[q]) ids.insert(nb.index); size_t h = 0; for (auto& nb : g[q]) h += ids.count(nb.index); s += double(h) / t[q].size(); } return s / t.size(); }
int main(int argc, char** argv) {
    const size_t n = 20000, dim = 32, nq = 200, k = 10; int threads = argc > 1 ? atoi(argv[1]) : 8;
    auto data = rnd(n, dim, 30), queries = rnd(nq, dim, 31);
    FlatIndex flat(dim); flat.add_batch(data, n); auto truth = flat.query_batch(queries, nq, k);
    set_num_threads(1); HNSWIndex serial(dim, 16, 100, false, 1); serial.add_batch(data, n); double sr = rec(serial.query_batch(queries, nq, k), truth);
    set_num_threads(threads); double mn = 1, sum = 0; int B = argc > 2 ? atoi(argv[2]) : 20; std::vector<double> rs;
    for (int b = 0; b < B; ++b) { HNSWIndex p(dim, 16, 100, false, 1); p.add_batch({data.data(), 5000 * dim}, 5000); p.add_batch({data.data() + 5000 * dim, (n - 5000) * dim}, n - 5000); double r = rec(p.query_batch(queries, nq, k), truth); mn = std::min(mn, r); sum += r; rs.push_back(r); }
    std::sort(rs.begin(), rs.end()); std::printf("lowest:"); for (int i = 0; i < 10 && i < (int)rs.size(); ++i) std::printf(" %.4f", rs[i]); std::printf("\n");
    std::printf("below threshold: %d of %d\n", (int)std::count_if(rs.begin(), rs.end(), [&](double r) { return r < sr - 0.02; }), B);
    std::printf("threads %d: serial %.4f parallel min %.4f mean %.4f (threshold %.4f)\n", threads, sr, mn, sum / B, sr - 0.02);
}
