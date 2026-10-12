#pragma once

// Scalar quantization: each dimension's trained range [vmin, vmin + max_code *
// step] is split into evenly spaced values, so a vector is stored as one small
// integer code per dimension. 8 bits give 256 values and one byte per
// dimension (4x smaller than floats); 4 bits give 16 values packed two per
// byte (8x smaller), dimension 2i in the low nibble of byte i and 2i + 1 in
// the high nibble. Values outside the range are clamped to its ends.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vecengine::detail {

using std::size_t;
using std::span;
using std::vector;

struct SQ {
    int bits = 8;        // 4 or 8
    vector<float> vmin;  // per dimension: the value of code 0
    vector<float> step;  // per dimension: the spacing between codes (0 when the dimension is constant)

    // Sets each dimension's range to the min and max of n training vectors
    // (dim floats each, row-major). n >= 1; bits must already be 4 or 8.
    void train(span<const float> vecs, size_t n, size_t dim);

    // Dimensions per vector (0 before training).
    size_t dim() const noexcept { return vmin.size(); }

    // The highest code: 15 or 255.
    int max_code() const noexcept { return (1 << bits) - 1; }

    // Bytes one vector's codes take: dim, or ceil(dim / 2) for 4 bits.
    size_t code_bytes() const noexcept { return bits == 8 ? dim() : (dim() + 1) / 2; }

    // Writes the code_bytes() bytes of vec (dim() floats) to out.
    void encode(const float* vec, uint8_t* out) const noexcept;

    // Writes the dim() floats that code stands for to out.
    void decode(const uint8_t* code, float* out) const noexcept;

    // Squared L2 distance from q (dim() floats) to the vector code stands for.
    float distance(const float* q, const uint8_t* code) const noexcept;
};

} // namespace vecengine::detail
