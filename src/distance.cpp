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

float inner_product_scalar(const float* a, const float* b, std::size_t n) {
    float acc = 0.0f;
    for (std::size_t i = 0; i < n; ++i) acc += a[i] * b[i];
    return acc;
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

// The kernels below keep several independent accumulators: with one, each FMA
// waits out the previous one's latency (about 4 cycles), capping long vectors
// at a quarter of the FMA throughput the loads could feed.

float l2_distance_avx2(const float* a, const float* b, std::size_t n) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    std::size_t i = 0;

    for (; i + 32 <= n; i += 32) {
        const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        const __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        const __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);
    }
    for (; i + 8 <= n; i += 8) {
        const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc0 = _mm256_fmadd_ps(d, d, acc0);
    }

    float result = hsum256(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));

    // scalar tail
    for (; i < n; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

float cosine_distance_avx2(const float* a, const float* b, std::size_t n) {
    // three sums per step already overlap, so two steps per iteration suffice
    __m256 vdot0 = _mm256_setzero_ps(), vnorma0 = _mm256_setzero_ps(), vnormb0 = _mm256_setzero_ps();
    __m256 vdot1 = _mm256_setzero_ps(), vnorma1 = _mm256_setzero_ps(), vnormb1 = _mm256_setzero_ps();
    std::size_t i = 0;

    for (; i + 16 <= n; i += 16) {
        const __m256 va0 = _mm256_loadu_ps(a + i), vb0 = _mm256_loadu_ps(b + i);
        const __m256 va1 = _mm256_loadu_ps(a + i + 8), vb1 = _mm256_loadu_ps(b + i + 8);
        vdot0   = _mm256_fmadd_ps(va0, vb0, vdot0);
        vnorma0 = _mm256_fmadd_ps(va0, va0, vnorma0);
        vnormb0 = _mm256_fmadd_ps(vb0, vb0, vnormb0);
        vdot1   = _mm256_fmadd_ps(va1, vb1, vdot1);
        vnorma1 = _mm256_fmadd_ps(va1, va1, vnorma1);
        vnormb1 = _mm256_fmadd_ps(vb1, vb1, vnormb1);
    }
    for (; i + 8 <= n; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i), vb = _mm256_loadu_ps(b + i);
        vdot0   = _mm256_fmadd_ps(va, vb, vdot0);
        vnorma0 = _mm256_fmadd_ps(va, va, vnorma0);
        vnormb0 = _mm256_fmadd_ps(vb, vb, vnormb0);
    }

    float dot    = hsum256(_mm256_add_ps(vdot0, vdot1));
    float norm_a = hsum256(_mm256_add_ps(vnorma0, vnorma1));
    float norm_b = hsum256(_mm256_add_ps(vnormb0, vnormb1));

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

float inner_product_avx2(const float* a, const float* b, std::size_t n) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
    }
    for (; i + 8 <= n; i += 8) acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);

    float result = hsum256(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}

#else

// Non-AVX2 build: forward to scalar so the TU still compiles cleanly.
float l2_distance_avx2(const float* a, const float* b, std::size_t n) { return l2_distance_scalar(a, b, n); }
float cosine_distance_avx2(const float* a, const float* b, std::size_t n) { return cosine_distance_scalar(a, b, n); }
float inner_product_avx2(const float* a, const float* b, std::size_t n) { return inner_product_scalar(a, b, n); }

#endif // __AVX2__


#if defined(__ARM_NEON)

// Horizontal sum of an float32x4_t register.
static inline float hsum128(float32x4_t v) {
    return vaddvq_f32(v);
}

float l2_distance_neon(const float* a, const float* b, std::size_t n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f), acc3 = vdupq_n_f32(0.0f);
    std::size_t i = 0;

    for (; i + 16 <= n; i += 16) {
        const float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        const float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        const float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        const float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
        acc0 = vfmaq_f32(acc0, d0, d0);
        acc1 = vfmaq_f32(acc1, d1, d1);
        acc2 = vfmaq_f32(acc2, d2, d2);
        acc3 = vfmaq_f32(acc3, d3, d3);
    }
    for (; i + 4 <= n; i += 4) {
        const float32x4_t d = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        acc0 = vfmaq_f32(acc0, d, d);
    }

    float result = hsum128(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));

    // scalar tail
    for (; i < n; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

float cosine_distance_neon(const float* a, const float* b, std::size_t n) {
    float32x4_t vdot0 = vdupq_n_f32(0.0f), vnorma0 = vdupq_n_f32(0.0f), vnormb0 = vdupq_n_f32(0.0f);
    float32x4_t vdot1 = vdupq_n_f32(0.0f), vnorma1 = vdupq_n_f32(0.0f), vnormb1 = vdupq_n_f32(0.0f);
    std::size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        const float32x4_t va0 = vld1q_f32(a + i), vb0 = vld1q_f32(b + i);
        const float32x4_t va1 = vld1q_f32(a + i + 4), vb1 = vld1q_f32(b + i + 4);
        vdot0   = vfmaq_f32(vdot0, va0, vb0);
        vnorma0 = vfmaq_f32(vnorma0, va0, va0);
        vnormb0 = vfmaq_f32(vnormb0, vb0, vb0);
        vdot1   = vfmaq_f32(vdot1, va1, vb1);
        vnorma1 = vfmaq_f32(vnorma1, va1, va1);
        vnormb1 = vfmaq_f32(vnormb1, vb1, vb1);
    }
    for (; i + 4 <= n; i += 4) {
        const float32x4_t va = vld1q_f32(a + i), vb = vld1q_f32(b + i);
        vdot0   = vfmaq_f32(vdot0, va, vb);
        vnorma0 = vfmaq_f32(vnorma0, va, va);
        vnormb0 = vfmaq_f32(vnormb0, vb, vb);
    }

    float dot    = hsum128(vaddq_f32(vdot0, vdot1));
    float norm_a = hsum128(vaddq_f32(vnorma0, vnorma1));
    float norm_b = hsum128(vaddq_f32(vnormb0, vnormb1));

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

float inner_product_neon(const float* a, const float* b, std::size_t n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f), acc3 = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    for (; i + 4 <= n; i += 4) acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));

    float result = hsum128(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}

#else

// Non-ARM_NEON build: forward to scalar so the TU still compiles cleanly.
float l2_distance_neon(const float* a, const float* b, std::size_t n) { return l2_distance_scalar(a, b, n); }
float cosine_distance_neon(const float* a, const float* b, std::size_t n) { return cosine_distance_scalar(a, b, n); }
float inner_product_neon(const float* a, const float* b, std::size_t n) { return inner_product_scalar(a, b, n); }

#endif // __ARM_NEON


} // namespace vecengine
