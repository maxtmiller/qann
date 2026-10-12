#include "vecengine/sq8.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

#if defined(__AVX2__)
#  include <immintrin.h>
#elif defined(__ARM_NEON)
#  include <arm_neon.h>
#endif

namespace vecengine::detail {

void SQ8::train(span<const float> vecs, size_t n, size_t dim) {
    assert(n >= 1 && vecs.size() == n * dim);
    vector<float> lo(vecs.begin(), vecs.begin() + dim);
    vector<float> hi = lo;
    for (size_t i = 1; i < n; ++i) {
        const float* row = vecs.data() + i * dim;
        for (size_t d = 0; d < dim; ++d) {
            lo[d] = std::min(lo[d], row[d]);
            hi[d] = std::max(hi[d], row[d]);
        }
    }
    step.resize(dim);
    for (size_t d = 0; d < dim; ++d) step[d] = (hi[d] - lo[d]) / 255.0f;
    vmin = std::move(lo);
}

void SQ8::encode(const float* vec, uint8_t* out) const noexcept {
    for (size_t d = 0; d < dim(); ++d) {
        const float c = step[d] > 0 ? std::round((vec[d] - vmin[d]) / step[d]) : 0.0f;
        out[d] = static_cast<uint8_t>(std::clamp(c, 0.0f, 255.0f));
    }
}

void SQ8::decode(const uint8_t* code, float* out) const noexcept {
    for (size_t d = 0; d < dim(); ++d) out[d] = vmin[d] + static_cast<float>(code[d]) * step[d];
}

float SQ8::distance(const float* q, const uint8_t* code) const noexcept {
    const size_t n = dim();
    const float* lo = vmin.data();
    const float* st = step.data();
    size_t d = 0;
    float sum = 0.0f;

#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; d + 8 <= n; d += 8) {
        const __m256i c32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(code + d)));
        const __m256 x = _mm256_fmadd_ps(_mm256_cvtepi32_ps(c32), _mm256_loadu_ps(st + d), _mm256_loadu_ps(lo + d));
        const __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(q + d), x);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }
    const __m128 half = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    const __m128 pair = _mm_add_ps(half, _mm_movehl_ps(half, half));
    sum = _mm_cvtss_f32(_mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 1)));
#elif defined(__ARM_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    // one 16-byte load feeds four groups of 4 lanes; two accumulators halve the FMA dependency chain
    auto step4 = [&](float32x4_t& acc, uint32x4_t c, size_t at) {
        const float32x4_t x = vfmaq_f32(vld1q_f32(lo + at), vcvtq_f32_u32(c), vld1q_f32(st + at));
        const float32x4_t diff = vsubq_f32(vld1q_f32(q + at), x);
        acc = vfmaq_f32(acc, diff, diff);
    };
    for (; d + 16 <= n; d += 16) {
        const uint8x16_t c8 = vld1q_u8(code + d);
        const uint16x8_t c16lo = vmovl_u8(vget_low_u8(c8));
        const uint16x8_t c16hi = vmovl_u8(vget_high_u8(c8));
        step4(acc0, vmovl_u16(vget_low_u16(c16lo)), d);
        step4(acc1, vmovl_u16(vget_high_u16(c16lo)), d + 4);
        step4(acc0, vmovl_u16(vget_low_u16(c16hi)), d + 8);
        step4(acc1, vmovl_u16(vget_high_u16(c16hi)), d + 12);
    }
    sum = vaddvq_f32(vaddq_f32(acc0, acc1));
#endif

    for (; d < n; ++d) {
        const float diff = q[d] - (lo[d] + static_cast<float>(code[d]) * st[d]);
        sum += diff * diff;
    }
    return sum;
}

} // namespace vecengine::detail
