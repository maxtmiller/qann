#pragma once
// Inline dispatch: selected at compile time via __AVX2__ macro.
// This header is included only by distance.hpp.

#include <cassert>
#include <span>

namespace vecengine {

// Squared L2 distance, using the fastest kernel this build supports.
inline float l2_distance(const float* a, const float* b, size_t n) {
#if defined(__AVX2__)
    return l2_distance_avx2(a, b, n);
#elif defined(__ARM_NEON)
    return l2_distance_neon(a, b, n);
#else
    return l2_distance_scalar(a, b, n);
#endif
}

// Cosine distance, using the fastest kernel this build supports.
inline float cosine_distance(const float* a, const float* b, size_t n) {
#if defined(__AVX2__)
    return cosine_distance_avx2(a, b, n);
#elif defined(__ARM_NEON)
    return cosine_distance_neon(a, b, n);
#else
    return cosine_distance_scalar(a, b, n);
#endif
}

// l2_distance over two equal-length spans.
inline float l2_distance(span<const float> a, span<const float> b) {
    assert(a.size() == b.size());
    return l2_distance(a.data(), b.data(), a.size());
}

// cosine_distance over two equal-length spans.
inline float cosine_distance(span<const float> a, span<const float> b) {
    assert(a.size() == b.size());
    return cosine_distance(a.data(), b.data(), a.size());
}

} // namespace vecengine
