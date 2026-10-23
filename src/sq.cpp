#include "vecengine/sq.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

#if defined(__AVX2__) || defined(__AVX512F__)
#  include <immintrin.h>
#elif defined(__ARM_NEON)
#  include <arm_neon.h>
#endif

namespace vecengine::detail {

namespace {

// The code of dimension d in a vector's codes.
inline uint8_t code_at(const uint8_t* code, size_t d, int bits) noexcept {
    if (bits == 8) return code[d];
    return static_cast<uint8_t>((code[d / 2] >> ((d & 1) * 4)) & 0x0F);
}

} // namespace

void SQ::train(span<const float> vecs, size_t n, size_t dim) {
    assert(n >= 1 && vecs.size() == n * dim && (bits == 4 || bits == 8));
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
    for (size_t d = 0; d < dim; ++d) step[d] = (hi[d] - lo[d]) / static_cast<float>(max_code());
    vmin = std::move(lo);
}

void SQ::encode(const float* vec, uint8_t* out) const noexcept {
    if (bits == 4) std::fill_n(out, code_bytes(), uint8_t{0});
    const float top = static_cast<float>(max_code());
    for (size_t d = 0; d < dim(); ++d) {
        const float c = step[d] > 0 ? std::round((vec[d] - vmin[d]) / step[d]) : 0.0f;
        const auto code = static_cast<uint8_t>(std::clamp(c, 0.0f, top));
        if (bits == 8) out[d] = code;
        else out[d / 2] |= static_cast<uint8_t>(code << ((d & 1) * 4));
    }
}

void SQ::decode(const uint8_t* code, float* out) const noexcept {
    for (size_t d = 0; d < dim(); ++d) out[d] = vmin[d] + static_cast<float>(code_at(code, d, bits)) * step[d];
}

float SQ::distance(const float* q, const uint8_t* code) const noexcept { return score<false>(q, code); }

float SQ::inner_product(const float* q, const uint8_t* code) const noexcept { return score<true>(q, code); }

template <bool InnerProduct>
float SQ::score(const float* q, const uint8_t* code) const noexcept {
    const size_t n = dim();
    const float* lo = vmin.data();
    const float* st = step.data();
    size_t d = 0;
    float sum = 0.0f;

    // The SIMD loops first spread 4-bit codes out to one byte per dimension,
    // then score bytes the same way for both widths.
#if defined(__AVX512F__)
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    // two accumulators halve the FMA dependency chain
    // scores the 16 dimensions at `at` from one byte code each in c16
    auto step16 = [&](__m512& acc, __m128i c16, size_t at) {
        const __m512 c = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(c16));
        const __m512 x = _mm512_fmadd_ps(c, _mm512_loadu_ps(st + at), _mm512_loadu_ps(lo + at));
        if constexpr (InnerProduct) {
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(q + at), x, acc);
        } else {
            const __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(q + at), x);
            acc = _mm512_fmadd_ps(diff, diff, acc);
        }
    };
    if (bits == 8) {
        for (; d + 32 <= n; d += 32) {
            step16(acc0, _mm_loadu_si128(reinterpret_cast<const __m128i*>(code + d)), d);
            step16(acc1, _mm_loadu_si128(reinterpret_cast<const __m128i*>(code + d + 16)), d + 16);
        }
        for (; d + 16 <= n; d += 16) step16(acc0, _mm_loadu_si128(reinterpret_cast<const __m128i*>(code + d)), d);
    } else {
        const __m128i nibble = _mm_set1_epi8(0x0F);
        for (; d + 32 <= n; d += 32) {
            const __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(code + d / 2));
            const __m128i low = _mm_and_si128(c, nibble);
            const __m128i high = _mm_and_si128(_mm_srli_epi16(c, 4), nibble);
            step16(acc0, _mm_unpacklo_epi8(low, high), d);
            step16(acc1, _mm_unpackhi_epi8(low, high), d + 16);
        }
    }
    sum = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    // scores the 8 dimensions at `at` from one byte code each, in the low 8 bytes of c8
    auto step8 = [&](__m128i c8, size_t at) {
        const __m256 c = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(c8));
        const __m256 x = _mm256_fmadd_ps(c, _mm256_loadu_ps(st + at), _mm256_loadu_ps(lo + at));
        if constexpr (InnerProduct) {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(q + at), x, acc);
        } else {
            const __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(q + at), x);
            acc = _mm256_fmadd_ps(diff, diff, acc);
        }
    };
    if (bits == 8) {
        for (; d + 8 <= n; d += 8) step8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(code + d)), d);
    } else {
        const __m128i nibble = _mm_set1_epi8(0x0F);
        for (; d + 16 <= n; d += 16) {
            const __m128i c = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(code + d / 2));
            const __m128i bytes = _mm_unpacklo_epi8(_mm_and_si128(c, nibble), _mm_and_si128(_mm_srli_epi16(c, 4), nibble));
            step8(bytes, d);
            step8(_mm_srli_si128(bytes, 8), d + 8);
        }
    }
    const __m128 half = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    const __m128 pair = _mm_add_ps(half, _mm_movehl_ps(half, half));
    sum = _mm_cvtss_f32(_mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 1)));
#elif defined(__ARM_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    // two accumulators halve the FMA dependency chain
    auto step4 = [&](float32x4_t& acc, uint32x4_t c, size_t at) {
        const float32x4_t x = vfmaq_f32(vld1q_f32(lo + at), vcvtq_f32_u32(c), vld1q_f32(st + at));
        if constexpr (InnerProduct) {
            acc = vfmaq_f32(acc, vld1q_f32(q + at), x);
        } else {
            const float32x4_t diff = vsubq_f32(vld1q_f32(q + at), x);
            acc = vfmaq_f32(acc, diff, diff);
        }
    };
    // scores the 16 dimensions at `at` from one byte code each
    auto step16 = [&](uint8x16_t c8, size_t at) {
        const uint16x8_t c16lo = vmovl_u8(vget_low_u8(c8));
        const uint16x8_t c16hi = vmovl_u8(vget_high_u8(c8));
        step4(acc0, vmovl_u16(vget_low_u16(c16lo)), at);
        step4(acc1, vmovl_u16(vget_high_u16(c16lo)), at + 4);
        step4(acc0, vmovl_u16(vget_low_u16(c16hi)), at + 8);
        step4(acc1, vmovl_u16(vget_high_u16(c16hi)), at + 12);
    };
    if (bits == 8) {
        for (; d + 16 <= n; d += 16) step16(vld1q_u8(code + d), d);
    } else {
        for (; d + 32 <= n; d += 32) {
            const uint8x16_t c = vld1q_u8(code + d / 2);
            const uint8x16_t low = vandq_u8(c, vdupq_n_u8(0x0F));
            const uint8x16_t high = vshrq_n_u8(c, 4);
            step16(vzip1q_u8(low, high), d);
            step16(vzip2q_u8(low, high), d + 16);
        }
    }
    sum = vaddvq_f32(vaddq_f32(acc0, acc1));
#endif

    for (; d < n; ++d) {
        const float x = lo[d] + static_cast<float>(code_at(code, d, bits)) * st[d];
        if constexpr (InnerProduct) {
            sum += q[d] * x;
        } else {
            const float diff = q[d] - x;
            sum += diff * diff;
        }
    }
    return sum;
}

} // namespace vecengine::detail
