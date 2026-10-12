#pragma once

// 8-bit scalar quantization: each dimension's trained range [vmin, vmin +
// 255 * step] is split into 256 evenly spaced values, so a vector is stored as
// one byte per dimension (4x smaller than floats). Values outside the range
// are clamped to its ends.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vecengine::detail {

using std::size_t;
using std::span;
using std::vector;

struct SQ8 {
    vector<float> vmin;  // per dimension: the value of code 0
    vector<float> step;  // per dimension: the spacing between codes (0 when the dimension is constant)

    // Sets each dimension's range to the min and max of n training vectors
    // (dim floats each, row-major). n >= 1.
    void train(span<const float> vecs, size_t n, size_t dim);

    // Dimensions per vector (0 before training).
    size_t dim() const noexcept { return vmin.size(); }

    // Writes the dim() codes of vec (dim() floats) to out.
    void encode(const float* vec, uint8_t* out) const noexcept;

    // Writes the dim() floats that code stands for to out.
    void decode(const uint8_t* code, float* out) const noexcept;

    // Squared L2 distance from q (dim() floats) to the vector code stands for.
    float distance(const float* q, const uint8_t* code) const noexcept;
};

} // namespace vecengine::detail
