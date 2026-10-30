// Picks the thresholds string filters use to choose between a pass over the
// column, a string's row list and its bitmap (#72 step 3, PR #90).
//
//   clang++ -O3 -std=c++20 -Iinclude benchmarks/string_filter_sweep.cpp build/libvecengine_core.a \
//       -framework Accelerate -o build/string_filter_sweep && ./build/string_filter_sweep
//
// For a column of 1M rows where one string covers a fraction p at random rows,
// times one evaluation of tag = 'x' by each path on its own: "scan" (the
// branch-free pass over codes that evaluate_leaf falls back to), "list" (set
// each listed row's bit; rows ascending as adds leave them, and shuffled as
// updates can), "bitmap" (OR the string's words) and "auto" (the real
// MetadataStore::evaluate, which picks one). Each includes allocating the
// result words, like the real call. Paths alternate in rounds and keep their
// best, so machine load hits all of them alike.

#include "vecengine/metadata.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace vecengine;
using std::vector;

namespace {

// Microseconds per call of fn, the best of `rounds` timed batches.
template <class F>
double time_us(F&& fn, size_t reps) {
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t r = 0; r < reps; ++r) fn();
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
}

volatile uint64_t g_sink;

void consume(const vector<uint64_t>& words) { g_sink = g_sink + words[words.size() / 2]; }

} // namespace

int main() {
    const size_t n = 1'000'000, words_n = (n + 63) / 64;
    const double fractions[] = {0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0};
    const int rounds = 7;
    std::mt19937 rng(1);

    std::printf("1M rows, one string at fraction p, us per evaluation (best of %d rounds)\n\n", rounds);
    std::printf("%8s %9s %9s %9s %9s %9s %9s   %s\n", "p", "rows", "scan", "list", "list-shuf", "bitmap", "auto",
                "per listed row vs per scanned row");

    for (double p : fractions) {
        std::bernoulli_distribution pick(p);
        vector<uint32_t> codes(n), rows;
        vector<uint8_t> valid(n, 1);
        vector<std::string> tag(n);
        for (size_t i = 0; i < n; ++i) {
            const bool x = pick(rng);
            codes[i] = x ? 1 : 0;
            tag[i] = x ? "x" : "y";
            if (x) rows.push_back(static_cast<uint32_t>(i));
        }
        vector<uint32_t> shuffled = rows;
        std::shuffle(shuffled.begin(), shuffled.end(), rng);
        vector<uint64_t> bitmap(words_n, 0);
        for (uint32_t s : rows) bitmap[s / 64] |= uint64_t{1} << (s % 64);
        const uint8_t match[2] = {0, 1};

        detail::MetadataStore store;
        store.append(0, n, {{"tag", tag}});
        const Expr where = Expr::compare("tag", Expr::Op::Eq, std::string("x"));

        auto scan = [&] {
            vector<uint64_t> words(words_n, 0);
            for (size_t w = 0; w < words_n; ++w) {
                const size_t first = w * 64, end = std::min(n, first + 64);
                uint64_t bits = 0;
                for (size_t i = first; i < end; ++i) bits |= uint64_t{bool(valid[i] && match[codes[i]])} << (i - first);
                words[w] = bits;
            }
            consume(words);
        };
        auto scatter = [&](const vector<uint32_t>& list) {
            vector<uint64_t> words(words_n, 0);
            for (uint32_t s : list) words[s / 64] |= uint64_t{1} << (s % 64);
            consume(words);
        };
        auto list = [&] { scatter(rows); };
        auto list_shuf = [&] { scatter(shuffled); };
        auto bits = [&] {
            vector<uint64_t> words(words_n, 0);
            for (size_t w = 0; w < words_n; ++w) words[w] |= bitmap[w];
            consume(words);
        };
        auto automatic = [&] { consume(store.evaluate(where, n)); };

        double best[5] = {1e30, 1e30, 1e30, 1e30, 1e30};
        for (int r = 0; r < rounds; ++r) {
            best[0] = std::min(best[0], time_us(scan, 20));
            best[1] = std::min(best[1], time_us(list, 20));
            best[2] = std::min(best[2], time_us(list_shuf, 20));
            best[3] = std::min(best[3], time_us(bits, 20));
            best[4] = std::min(best[4], time_us(automatic, 20));
        }
        // the constant evaluate_leaf needs: time per listed row over time per scanned row
        const double ratio = rows.empty() ? 0 : (best[1] / rows.size()) / (best[0] / n);
        std::printf("%8.4f %9zu %9.1f %9.1f %9.1f %9.1f %9.1f   %.1f\n", p, rows.size(), best[0], best[1], best[2],
                    best[3], best[4], ratio);
    }
}
