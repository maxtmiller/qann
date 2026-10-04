#pragma once

#include <cstddef>
#include <span>

namespace vecengine {

using std::size_t;
using std::span;

// All functions operate on float32 vectors of equal dimension `n`.
// Callers are responsible for ensuring a and b point to n valid floats.
// AVX2 paths require 32-byte alignment; scalar fallbacks have no alignment
// requirement.

// Squared Euclidean distance: sum((a[i] - b[i])^2)
float l2_distance_scalar(const float* a, const float* b, size_t n);
float l2_distance_avx2(const float* a, const float* b, size_t n);

// Cosine distance: 1 - (a . b) / (|a| * |b|)
float cosine_distance_scalar(const float* a, const float* b, size_t n);
float cosine_distance_avx2(const float* a, const float* b, size_t n);

// Dispatch wrappers: use AVX2 when available, else scalar.
inline float l2_distance(const float* a, const float* b, size_t n);
inline float cosine_distance(const float* a, const float* b, size_t n);

// std::span convenience overloads
inline float l2_distance(span<const float> a, span<const float> b);
inline float cosine_distance(span<const float> a, span<const float> b);

// 
float l2_distance_neon(const float* a, const float* b, std::size_t n);
float cosine_distance_neon(const float* a, const float* b, std::size_t n);

} // namespace vecengine

// Inline dispatch implementations
#include "distance_inl.hpp"
