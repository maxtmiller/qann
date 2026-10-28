#include "fast_scan.hpp"

#include <stdexcept>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>

#if defined(__ARM_NEON)
#include <arm_neon.h>
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

void quantize_lut(const float* table, size_t m, uint8_t* lut, float& scale, float& offset) {
#if defined(__ARM_NEON)
    if (m <= kFsMaxSubspaces) {   // row_min is on the stack
        quantize_lut_neon(table, m, lut, scale, offset);
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
#elif defined(__SSSE3__)
    return fast_scan_block_ssse3(lut, block, m, limit, out);
#else
    return fast_scan_block_mask_scalar(lut, block, m, limit, out);
#endif
}

} // namespace vecengine::detail
