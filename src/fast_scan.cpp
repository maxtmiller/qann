#include "fast_scan.hpp"

#include <stdexcept>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSSE3__)
#include <tmmintrin.h>
#endif

namespace vecengine::detail {

void quantize_lut_scalar(const float* table, size_t m, uint8_t* lut, float& scale, float& offset) {
    std::vector<float> min_s(m, std::numeric_limits<float>::max());
    float max_w = -1;
    offset = 0;
    for (size_t i = 0; i < m; ++i) {
        const float* pos = table + (i * 16);
        float lo = pos[0], hi = pos[0];
        for (size_t j = 1; j < 16; ++j) { 
            lo = std::min(lo, pos[j]); 
            hi = std::max(hi, pos[j]); 
        }
        min_s[i] = lo;
        offset += min_s[i];
        max_w = std::max(max_w, hi - min_s[i]);
    }

    scale = max_w > 0 ? kLutMax / max_w : 1;

    for (size_t i = 0; i < m; ++i) {
        const float* pos = table + (i * 16);
        for (size_t j = 0; j < 16; ++j) {
            lut[i * 16 + j] = static_cast<uint8_t>(std::lround((pos[j] - min_s[i]) * scale));
        }
    }
}

#if defined(__ARM_NEON)
// Same arithmetic as quantize_lut_scalar, 4 floats per instruction: min/max of
// a 16-entry row are 4 loads and two reductions, and vcvtaq rounds to nearest
// with ties away from zero exactly like std::lround, so the bytes are identical.
static void quantize_lut_neon(const float* table, size_t m, uint8_t* lut, float& scale, float& offset) {
    float row_min[kFsMaxSubspaces];
    float widest = 0.0f;
    offset = 0.0f;
    for (size_t i = 0; i < m; ++i) {
        const float* row = table + i * 16;
        const float32x4_t a = vld1q_f32(row), b = vld1q_f32(row + 4), c = vld1q_f32(row + 8), d = vld1q_f32(row + 12);
        const float lo = vminvq_f32(vminq_f32(vminq_f32(a, b), vminq_f32(c, d)));
        const float hi = vmaxvq_f32(vmaxq_f32(vmaxq_f32(a, b), vmaxq_f32(c, d)));
        row_min[i] = lo;
        offset += lo;
        widest = std::max(widest, hi - lo);
    }

    scale = widest > 0.0f ? kLutMax / widest : 1.0f;
    const float32x4_t vscale = vdupq_n_f32(scale);

    for (size_t i = 0; i < m; ++i) {
        const float* row = table + i * 16;
        const float32x4_t vmin = vdupq_n_f32(row_min[i]);
        uint32x4_t q[4];
        for (int k = 0; k < 4; ++k)
            q[k] = vcvtaq_u32_f32(vmulq_f32(vsubq_f32(vld1q_f32(row + 4 * k), vmin), vscale));
        const uint16x8_t lo16 = vcombine_u16(vqmovn_u32(q[0]), vqmovn_u32(q[1]));
        const uint16x8_t hi16 = vcombine_u16(vqmovn_u32(q[2]), vqmovn_u32(q[3]));
        vst1q_u8(lut + i * 16, vcombine_u8(vqmovn_u16(lo16), vqmovn_u16(hi16)));
    }
}
#endif

#if defined(__AVX2__) || defined(__SSSE3__)
static inline float hmin_ps(__m128 v) {
    v = _mm_min_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(_mm_min_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1))));
}

static inline float hmax_ps(__m128 v) {
    v = _mm_max_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(_mm_max_ps(v, _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1))));
}

#endif

#if defined(__AVX2__)
// quantize_lut_sse's arithmetic, 8 floats per instruction (a row is two registers).
static void quantize_lut_avx2(const float* table, size_t m, uint8_t* lut, float& scale, float& offset) {
    float row_min[kFsMaxSubspaces];
    float widest = 0.0f;
    offset = 0.0f;
    for (size_t i = 0; i < m; ++i) {
        const float* row = table + i * 16;
        const __m256 a = _mm256_loadu_ps(row), b = _mm256_loadu_ps(row + 8);
        const __m256 lo8 = _mm256_min_ps(a, b), hi8 = _mm256_max_ps(a, b);
        const float lo = hmin_ps(_mm_min_ps(_mm256_castps256_ps128(lo8), _mm256_extractf128_ps(lo8, 1)));
        const float hi = hmax_ps(_mm_max_ps(_mm256_castps256_ps128(hi8), _mm256_extractf128_ps(hi8, 1)));
        row_min[i] = lo;
        offset += lo;
        widest = std::max(widest, hi - lo);
    }

    scale = widest > 0.0f ? kLutMax / widest : 1.0f;
    const __m256 vscale = _mm256_set1_ps(scale), half = _mm256_set1_ps(0.5f);
    const __m256i one = _mm256_set1_epi32(1);

    for (size_t i = 0; i < m; ++i) {
        const float* row = table + i * 16;
        const __m256 vmin = _mm256_set1_ps(row_min[i]);
        __m256i q[2];
        for (int k = 0; k < 2; ++k) {
            const __m256 x = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(row + 8 * k), vmin), vscale);
            const __m256i t = _mm256_cvttps_epi32(x);
            const __m256 up = _mm256_cmp_ps(_mm256_sub_ps(x, _mm256_cvtepi32_ps(t)), half, _CMP_GE_OQ);
            q[k] = _mm256_add_epi32(t, _mm256_and_si256(_mm256_castps_si256(up), one));
        }
        // the 256-bit packs work per 128-bit lane, so pack the halves as SSE does
        const __m128i lo16 = _mm_packs_epi32(_mm256_castsi256_si128(q[0]), _mm256_extracti128_si256(q[0], 1));
        const __m128i hi16 = _mm_packs_epi32(_mm256_castsi256_si128(q[1]), _mm256_extracti128_si256(q[1], 1));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(lut + i * 16), _mm_packus_epi16(lo16, hi16));
    }
}
#elif defined(__SSSE3__)
// Same arithmetic as quantize_lut_scalar, 4 floats per instruction. SSE has no
// round-half-away-from-zero like std::lround, but every value here is >= 0:
// truncate, then add 1 when the (exact) remainder is >= 0.5, which gives the
// same bytes. floor(x + 0.5) would not: 0.49999997f + 0.5f rounds up to 1.
static void quantize_lut_sse(const float* table, size_t m, uint8_t* lut, float& scale, float& offset) {
    float row_min[kFsMaxSubspaces];
    float widest = 0.0f;
    offset = 0.0f;
    for (size_t i = 0; i < m; ++i) {
        const float* row = table + i * 16;
        const __m128 a = _mm_loadu_ps(row), b = _mm_loadu_ps(row + 4), c = _mm_loadu_ps(row + 8), d = _mm_loadu_ps(row + 12);
        const float lo = hmin_ps(_mm_min_ps(_mm_min_ps(a, b), _mm_min_ps(c, d)));
        const float hi = hmax_ps(_mm_max_ps(_mm_max_ps(a, b), _mm_max_ps(c, d)));
        row_min[i] = lo;
        offset += lo;
        widest = std::max(widest, hi - lo);
    }

    scale = widest > 0.0f ? kLutMax / widest : 1.0f;
    const __m128 vscale = _mm_set1_ps(scale), half = _mm_set1_ps(0.5f);
    const __m128i one = _mm_set1_epi32(1);

    for (size_t i = 0; i < m; ++i) {
        const float* row = table + i * 16;
        const __m128 vmin = _mm_set1_ps(row_min[i]);
        __m128i q[4];
        for (int k = 0; k < 4; ++k) {
            const __m128 x = _mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(row + 4 * k), vmin), vscale);
            const __m128i t = _mm_cvttps_epi32(x);
            const __m128 up = _mm_cmpge_ps(_mm_sub_ps(x, _mm_cvtepi32_ps(t)), half);
            q[k] = _mm_add_epi32(t, _mm_and_si128(_mm_castps_si128(up), one));
        }
        const __m128i lo16 = _mm_packs_epi32(q[0], q[1]), hi16 = _mm_packs_epi32(q[2], q[3]);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(lut + i * 16), _mm_packus_epi16(lo16, hi16));
    }
}
#endif

void quantize_lut(const float* table, size_t m, uint8_t* lut, float& scale, float& offset) {
#if defined(__ARM_NEON)
    if (m <= kFsMaxSubspaces) {   // row_min is on the stack
        quantize_lut_neon(table, m, lut, scale, offset);
        return;
    }
#elif defined(__AVX2__)
    if (m <= kFsMaxSubspaces) {
        quantize_lut_avx2(table, m, lut, scale, offset);
        return;
    }
#elif defined(__SSSE3__)
    if (m <= kFsMaxSubspaces) {
        quantize_lut_sse(table, m, lut, scale, offset);
        return;
    }
#endif
    quantize_lut_scalar(table, m, lut, scale, offset);
}

void fast_scan_block_scalar(const uint8_t* lut, const uint8_t* block, size_t m, uint16_t* out) {
    for (size_t v = 0; v < kFsBlock; ++v) {
        uint16_t sum = 0;
        for (size_t s = 0; s < m; ++s) sum = static_cast<uint16_t>(sum + lut[s * 16 + fs_get(block, v, s)]);
        out[v] = sum;
    }
}

#if defined(__ARM_NEON)
static uint32_t fast_scan_block_neon(const uint8_t* lut, const uint8_t* block, size_t m, uint16_t limit, uint16_t* out) {
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    uint16x8_t a0 = vdupq_n_u16(0), a1 = a0, a2 = a0, a3 = a0;
    for (size_t s = 0; s < m; s += 4) {
        // up to 4 subspaces are summed in bytes (entries <= kLutMax, so no
        // overflow) and widened once: the widening adds were the bottleneck
        const size_t group = std::min<size_t>(4, m - s);
        uint8x16_t lo = vdupq_n_u8(0), hi = lo;
        for (size_t g = 0; g < group; ++g) {
            const uint8x16_t t = vld1q_u8(lut + (s + g) * 16);
            const uint8x16_t c = vld1q_u8(block + (s + g) * 16);
            // one table lookup scores 16 vectors: low nibbles are vectors 0-15,
            // high nibbles vectors 16-31
            lo = vaddq_u8(lo, vqtbl1q_u8(t, vandq_u8(c, mask)));
            hi = vaddq_u8(hi, vqtbl1q_u8(t, vshrq_n_u8(c, 4)));
        }
        a0 = vaddw_u8(a0, vget_low_u8(lo));
        a1 = vaddw_high_u8(a1, lo);
        a2 = vaddw_u8(a2, vget_low_u8(hi));
        a3 = vaddw_high_u8(a3, hi);
    }
    vst1q_u16(out, a0);
    vst1q_u16(out + 8, a1);
    vst1q_u16(out + 16, a2);
    vst1q_u16(out + 24, a3);

    // lanes below limit are all-ones; keeping one weight per lane and summing
    // gives 8 mask bits per accumulator
    static const uint16_t kWeights[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    const uint16x8_t w = vld1q_u16(kWeights), l = vdupq_n_u16(limit);
    return static_cast<uint32_t>(vaddvq_u16(vandq_u16(vcltq_u16(a0, l), w))) |
           static_cast<uint32_t>(vaddvq_u16(vandq_u16(vcltq_u16(a1, l), w))) << 8 |
           static_cast<uint32_t>(vaddvq_u16(vandq_u16(vcltq_u16(a2, l), w))) << 16 |
           static_cast<uint32_t>(vaddvq_u16(vandq_u16(vcltq_u16(a3, l), w))) << 24;
}
#elif defined(__AVX2__)
// One 256-bit lookup scores 16 vectors on two subspaces: m is even and a
// subspace's lut row and codes are 16 bytes each, so loading 32 bytes puts
// subspace s in the low lane and s + 1 in the high lane. Each lane then sums 4
// subspaces in bytes before widening, as in the NEON kernel, and the lanes are
// added at the end.
static uint32_t fast_scan_block_avx2(const uint8_t* lut, const uint8_t* block, size_t m, uint16_t limit, uint16_t* out) {
    const __m256i mask = _mm256_set1_epi8(0x0F);
    const __m256i zero = _mm256_setzero_si256();
    __m256i a0 = zero, a1 = zero, a2 = zero, a3 = zero;
    for (size_t s = 0; s < m; s += 8) {
        const size_t end = std::min(m, s + 8);
        __m256i lo = zero, hi = zero;
        for (size_t g = s; g < end; g += 2) {
            const __m256i t = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut + g * 16));
            const __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block + g * 16));
            lo = _mm256_add_epi8(lo, _mm256_shuffle_epi8(t, _mm256_and_si256(c, mask)));                      // vectors 0-15
            hi = _mm256_add_epi8(hi, _mm256_shuffle_epi8(t, _mm256_and_si256(_mm256_srli_epi16(c, 4), mask))); // vectors 16-31
        }
        a0 = _mm256_add_epi16(a0, _mm256_unpacklo_epi8(lo, zero));
        a1 = _mm256_add_epi16(a1, _mm256_unpackhi_epi8(lo, zero));
        a2 = _mm256_add_epi16(a2, _mm256_unpacklo_epi8(hi, zero));
        a3 = _mm256_add_epi16(a3, _mm256_unpackhi_epi8(hi, zero));
    }
    auto lanes = [](__m256i a) { return _mm_add_epi16(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1)); };
    const __m128i s0 = lanes(a0), s1 = lanes(a1), s2 = lanes(a2), s3 = lanes(a3);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), s0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 8), s1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 16), s2);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 24), s3);

    // same unsigned compare and movemask as the SSSE3 kernel
    const __m128i bias = _mm_set1_epi16(static_cast<short>(0x8000));
    const __m128i l = _mm_xor_si128(_mm_set1_epi16(static_cast<short>(limit)), bias);
    auto below = [&](__m128i a) { return _mm_cmplt_epi16(_mm_xor_si128(a, bias), l); };
    const uint32_t m01 = static_cast<uint32_t>(_mm_movemask_epi8(_mm_packs_epi16(below(s0), below(s1))));
    const uint32_t m23 = static_cast<uint32_t>(_mm_movemask_epi8(_mm_packs_epi16(below(s2), below(s3))));
    return m01 | m23 << 16;
}
#elif defined(__SSSE3__)
static uint32_t fast_scan_block_ssse3(const uint8_t* lut, const uint8_t* block, size_t m, uint16_t limit, uint16_t* out) {
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i zero = _mm_setzero_si128();
    __m128i a0 = zero, a1 = zero, a2 = zero, a3 = zero;
    for (size_t s = 0; s < m; s += 4) {
        // same byte-sum grouping as the NEON kernel
        const size_t group = std::min<size_t>(4, m - s);
        __m128i lo = zero, hi = zero;
        for (size_t g = 0; g < group; ++g) {
            const __m128i t = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut + (s + g) * 16));
            const __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block + (s + g) * 16));
            lo = _mm_add_epi8(lo, _mm_shuffle_epi8(t, _mm_and_si128(c, mask)));                       // vectors 0-15
            hi = _mm_add_epi8(hi, _mm_shuffle_epi8(t, _mm_and_si128(_mm_srli_epi16(c, 4), mask)));    // vectors 16-31
        }
        a0 = _mm_add_epi16(a0, _mm_unpacklo_epi8(lo, zero));
        a1 = _mm_add_epi16(a1, _mm_unpackhi_epi8(lo, zero));
        a2 = _mm_add_epi16(a2, _mm_unpacklo_epi8(hi, zero));
        a3 = _mm_add_epi16(a3, _mm_unpackhi_epi8(hi, zero));
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), a0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 8), a1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 16), a2);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 24), a3);

    // SSE only compares signed 16-bit lanes; flipping the top bit maps unsigned
    // order onto signed order. Packing to bytes then gives one movemask bit per vector.
    const __m128i bias = _mm_set1_epi16(static_cast<short>(0x8000));
    const __m128i l = _mm_xor_si128(_mm_set1_epi16(static_cast<short>(limit)), bias);
    auto below = [&](__m128i a) { return _mm_cmplt_epi16(_mm_xor_si128(a, bias), l); };
    const uint32_t m01 = static_cast<uint32_t>(_mm_movemask_epi8(_mm_packs_epi16(below(a0), below(a1))));
    const uint32_t m23 = static_cast<uint32_t>(_mm_movemask_epi8(_mm_packs_epi16(below(a2), below(a3))));
    return m01 | m23 << 16;
}
#endif

void fast_scan_block(const uint8_t* lut, const uint8_t* block, size_t m, uint16_t* out) {
    fast_scan_block_mask(lut, block, m, 0, out);
}

uint32_t fast_scan_block_mask_scalar(const uint8_t* lut, const uint8_t* block, size_t m, uint16_t limit, uint16_t* out) {
    fast_scan_block_scalar(lut, block, m, out);
    uint32_t mask = 0;
    for (size_t v = 0; v < kFsBlock; ++v)
        if (out[v] < limit) mask |= 1u << v;
    return mask;
}

uint32_t fast_scan_block_mask(const uint8_t* lut, const uint8_t* block, size_t m, uint16_t limit, uint16_t* out) {
#if defined(__ARM_NEON)
    return fast_scan_block_neon(lut, block, m, limit, out);
#elif defined(__AVX2__)
    return fast_scan_block_avx2(lut, block, m, limit, out);
#elif defined(__SSSE3__)
    return fast_scan_block_ssse3(lut, block, m, limit, out);
#else
    return fast_scan_block_mask_scalar(lut, block, m, limit, out);
#endif
}

} // namespace vecengine::detail
