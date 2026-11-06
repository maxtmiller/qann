#pragma once
// Inline dispatch: selected at compile time via the __AVX512F__, __AVX2__ and
// __ARM_NEON macros, and on other x86 builds between AVX-512 and AVX2 by
// detail::kAvx512 (a branch that always goes the same way).
// This header is included only by distance.hpp.

#include <cassert>
#include <span>

namespace vecengine {

// Squared L2 distance, using the fastest kernel this build supports.
inline float l2_distance(const float* a, const float* b, size_t n) {
#if defined(__AVX512F__)
    return l2_distance_avx512(a, b, n);
#elif defined(__AVX2__)
#  if VECENGINE_AVX512_DISPATCH
    if (detail::kAvx512) return l2_distance_avx512(a, b, n);
#  endif
    return l2_distance_avx2(a, b, n);
#elif defined(__ARM_NEON)
    return l2_distance_neon(a, b, n);
#else
    return l2_distance_scalar(a, b, n);
#endif
}

// Cosine distance, using the fastest kernel this build supports.
inline float cosine_distance(const float* a, const float* b, size_t n) {
#if defined(__AVX512F__)
    return cosine_distance_avx512(a, b, n);
#elif defined(__AVX2__)
#  if VECENGINE_AVX512_DISPATCH
    if (detail::kAvx512) return cosine_distance_avx512(a, b, n);
#  endif
    return cosine_distance_avx2(a, b, n);
#elif defined(__ARM_NEON)
    return cosine_distance_neon(a, b, n);
#else
    return cosine_distance_scalar(a, b, n);
#endif
}

// Inner product, using the fastest kernel this build supports.
inline float inner_product(const float* a, const float* b, size_t n) {
#if defined(__AVX512F__)
    return inner_product_avx512(a, b, n);
#elif defined(__AVX2__)
#  if VECENGINE_AVX512_DISPATCH
    if (detail::kAvx512) return inner_product_avx512(a, b, n);
#  endif
    return inner_product_avx2(a, b, n);
#elif defined(__ARM_NEON)
    return inner_product_neon(a, b, n);
#else
    return inner_product_scalar(a, b, n);
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

// inner_product over two equal-length spans.
inline float inner_product(span<const float> a, span<const float> b) {
    assert(a.size() == b.size());
    return inner_product(a.data(), b.data(), a.size());
}

} // namespace vecengine
