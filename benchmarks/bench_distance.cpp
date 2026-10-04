#include <benchmark/benchmark.h>
#include "vecengine/distance.hpp"
#include "vecengine/aligned_allocator.hpp"

#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Fixture: fill AVX2-aligned vectors with random float32 data
// ---------------------------------------------------------------------------

static void fill_random(vecengine::AVX2Vector<float>& v) {
    std::mt19937 rng{42};
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

template <std::size_t N>
struct VecPair {
    vecengine::AVX2Vector<float> a{N}, b{N};
    VecPair() { fill_random(a); fill_random(b); }
};

// ---------------------------------------------------------------------------
// L2 benchmarks
// ---------------------------------------------------------------------------

template <std::size_t N>
static void BM_L2_Scalar(benchmark::State& state) {
    VecPair<N> vp;
    for (auto _ : state)
        benchmark::DoNotOptimize(vecengine::l2_distance_scalar(vp.a.data(), vp.b.data(), N));
}

template <std::size_t N>
static void BM_L2_AVX2(benchmark::State& state) {
    VecPair<N> vp;
    for (auto _ : state)
        benchmark::DoNotOptimize(vecengine::l2_distance_avx2(vp.a.data(), vp.b.data(), N));
}

// ---------------------------------------------------------------------------
// Cosine benchmarks
// ---------------------------------------------------------------------------

template <std::size_t N>
static void BM_Cosine_Scalar(benchmark::State& state) {
    VecPair<N> vp;
    for (auto _ : state)
        benchmark::DoNotOptimize(vecengine::cosine_distance_scalar(vp.a.data(), vp.b.data(), N));
}

template <std::size_t N>
static void BM_Cosine_AVX2(benchmark::State& state) {
    VecPair<N> vp;
    for (auto _ : state)
        benchmark::DoNotOptimize(vecengine::cosine_distance_avx2(vp.a.data(), vp.b.data(), N));
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

BENCHMARK_TEMPLATE(BM_L2_Scalar,   128);
BENCHMARK_TEMPLATE(BM_L2_AVX2,     128);
BENCHMARK_TEMPLATE(BM_L2_Scalar,   512);
BENCHMARK_TEMPLATE(BM_L2_AVX2,     512);
BENCHMARK_TEMPLATE(BM_L2_Scalar,  1536); // common embedding dim
BENCHMARK_TEMPLATE(BM_L2_AVX2,   1536);

BENCHMARK_TEMPLATE(BM_Cosine_Scalar,  128);
BENCHMARK_TEMPLATE(BM_Cosine_AVX2,    128);
BENCHMARK_TEMPLATE(BM_Cosine_Scalar, 1536);
BENCHMARK_TEMPLATE(BM_Cosine_AVX2,   1536);

BENCHMARK_MAIN();
