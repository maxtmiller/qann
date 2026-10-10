#pragma once

// PQ fast-scan: 4-bit PQ codes (16 centroids per subspace) scored 32 vectors
// at a time with a SIMD table lookup on 8-bit tables.
//
// Block layout: a list's codes are stored in blocks of kFsBlock = 32 vectors.
// A block is m * 16 bytes: for each subspace s, 16 bytes hold the 32 codes,
// byte j carrying vector j in its low nibble and vector j + 16 in its high
// nibble. The last block of a list is zero-padded; callers skip the padding
// entries using the list's real size.

#include <cstddef>
#include <cstdint>

namespace vecengine::detail {

using std::size_t;

inline constexpr size_t kFsBlock = 32; // vectors per block

// Bytes in one block for m subspaces.
inline constexpr size_t fs_block_bytes(size_t m) noexcept { return m * 16; }

// Writes the 4-bit code of subspace s for the vector at pos (0..31) of a block.
inline void fs_set(uint8_t* block, size_t pos, size_t s, uint8_t code) noexcept {
    uint8_t& b = block[s * 16 + (pos % 16)];
    b = pos < 16 ? static_cast<uint8_t>((b & 0xF0) | code) : static_cast<uint8_t>((b & 0x0F) | (code << 4));
}

// Reads the 4-bit code of subspace s for the vector at pos (0..31) of a block.
inline uint8_t fs_get(const uint8_t* block, size_t pos, size_t s) noexcept {
    const uint8_t b = block[s * 16 + (pos % 16)];
    return pos < 16 ? static_cast<uint8_t>(b & 0x0F) : static_cast<uint8_t>(b >> 4);
}

// Rounds an m x 16 float ADC table to bytes: lut[s * 16 + j] =
// round((table[s * 16 + j] - min_s) * scale), with min_s the row's minimum and
// one scale for all rows chosen so the widest row spans 0..255. A distance is
// then recovered as sum / scale + offset, where offset is the sum of the row
// minimums; the error is at most m / (2 * scale). A table whose rows are all
// constant gets scale 1.
void quantize_lut(const float* table, size_t m, uint8_t* lut, float& scale, float& offset);

// Portable version of quantize_lut; the NEON version must match it exactly.
void quantize_lut_scalar(const float* table, size_t m, uint8_t* lut, float& scale, float& offset);

// Sums of the lut entries picked by each of the block's 32 vectors: out[v] =
// sum over s of lut[s * 16 + code(v, s)]. m <= 256 keeps the sums within
// uint16_t. Uses NEON or SSSE3 when the build has them.
void fast_scan_block(const uint8_t* lut, const uint8_t* block, size_t m, uint16_t* out);

// Portable version of fast_scan_block; the SIMD versions must match it exactly.
void fast_scan_block_scalar(const uint8_t* lut, const uint8_t* block, size_t m, uint16_t* out);

// fast_scan_block that also compares the sums with limit while they are still
// in registers: bit v of the result is set when out[v] < limit.
uint32_t fast_scan_block_mask(const uint8_t* lut, const uint8_t* block, size_t m, uint16_t limit, uint16_t* out);

// Portable version of fast_scan_block_mask.
uint32_t fast_scan_block_mask_scalar(const uint8_t* lut, const uint8_t* block, size_t m, uint16_t limit, uint16_t* out);

} // namespace vecengine::detail
