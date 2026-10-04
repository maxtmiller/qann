#include "vecengine/distance.hpp"

#include <cmath>
#include <numeric>

#if defined(__AVX2__)
#  include <immintrin.h>
#endif

#if defined(__ARM_NEON)
#  include <arm_neon.h>
#endif

namespace vecengine {

// ---------------------------------------------------------------------------
// Scalar reference implementations
// ---------------------------------------------------------------------------

float l2_distance_scalar(const float* a, const float* b, std::size_t n) {
    float acc = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        float diff = a[i] - b[i];
        acc += diff * diff;
    }
    return acc; // squared L2; caller may std::sqrt if needed
}

float cosine_distance_scalar(const float* a, const float* b, std::size_t n) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    const float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-10f) return 1.0f; // treat zero-magnitude vectors as maximally distant
    return 1.0f - (dot / denom);
}

// ---------------------------------------------------------------------------
// AVX2 + FMA implementations
// ---------------------------------------------------------------------------

#if defined(__AVX2__)

// Horizontal sum of an __m256 register.
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

float l2_distance_avx2(const float* a, const float* b, std::size_t n) {
    __m256 acc = _mm256_setzero_ps();
    std::size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        __m256 va   = _mm256_loadu_ps(a + i);
        __m256 vb   = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        // FMA: acc = diff * diff + acc
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }

    float result = hsum256(acc);

    // scalar tail
    for (; i < n; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

float cosine_distance_avx2(const float* a, const float* b, std::size_t n) {
    __m256 vdot   = _mm256_setzero_ps();
    __m256 vnorma = _mm256_setzero_ps();
    __m256 vnormb = _mm256_setzero_ps();
    std::size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vdot   = _mm256_fmadd_ps(va, vb, vdot);
        vnorma = _mm256_fmadd_ps(va, va, vnorma);
        vnormb = _mm256_fmadd_ps(vb, vb, vnormb);
    }

    float dot    = hsum256(vdot);
    float norm_a = hsum256(vnorma);
    float norm_b = hsum256(vnormb);

    // scalar tail
    for (; i < n; ++i) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    const float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-10f) return 1.0f;
    return 1.0f - (dot / denom);
}

#else

// Non-AVX2 build: forward to scalar so the TU still compiles cleanly.
float l2_distance_avx2(const float* a, const float* b, std::size_t n) { return l2_distance_scalar(a, b, n); }
float cosine_distance_avx2(const float* a, const float* b, std::size_t n) { return cosine_distance_scalar(a, b, n); }

#endif // __AVX2__


#if defined(__ARM_NEON)

// Horizontal sum of an float32x4_t register.
static inline float hsum128(float32x4_t v) {
    return vaddvq_f32(v);
}

float l2_distance_neon(const float* a, const float* b, std::size_t n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    std::size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t va   = vld1q_f32(a + i);
        float32x4_t vb   = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        // FMA: acc = diff * diff + acc
        acc = vfmaq_f32(acc, diff, diff);
    }

    float result = hsum128(acc);

    // scalar tail
    for (; i < n; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

float cosine_distance_neon(const float* a, const float* b, std::size_t n) {
    float32x4_t vdot   = vdupq_n_f32(0.0f);
    float32x4_t vnorma = vdupq_n_f32(0.0f);
    float32x4_t vnormb = vdupq_n_f32(0.0f);
    std::size_t i = 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vdot   = vfmaq_f32(vdot, va, vb);
        vnorma = vfmaq_f32(vnorma, va, va);
        vnormb = vfmaq_f32(vnormb, vb, vb);
    }

    float dot    = hsum128(vdot);
    float norm_a = hsum128(vnorma);
    float norm_b = hsum128(vnormb);

    // scalar tail
    for (; i < n; ++i) {
        dot    += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    const float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denom < 1e-10f) return 1.0f;
    return 1.0f - (dot / denom);
}

#else

// Non-ARM_NEON build: forward to scalar so the TU still compiles cleanly.
float l2_distance_neon(const float* a, const float* b, std::size_t n) { return l2_distance_scalar(a, b, n); }
float cosine_distance_neon(const float* a, const float* b, std::size_t n) { return cosine_distance_scalar(a, b, n); }

#endif // __ARM_NEON


} // namespace vecengine
