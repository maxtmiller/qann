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

// Inner product: sum(a[i] * b[i])
float inner_product_scalar(const float* a, const float* b, size_t n);
float inner_product_avx2(const float* a, const float* b, size_t n);

// Dispatch wrappers: use AVX-512, AVX2 or NEON when compiled for them, else scalar.
inline float l2_distance(const float* a, const float* b, size_t n);
inline float cosine_distance(const float* a, const float* b, size_t n);
inline float inner_product(const float* a, const float* b, size_t n);

// std::span convenience overloads
inline float l2_distance(span<const float> a, span<const float> b);
inline float cosine_distance(span<const float> a, span<const float> b);
inline float inner_product(span<const float> a, span<const float> b);

// ARM NEON versions of the same three functions (used on Apple Silicon and
// aarch64 Linux).
float l2_distance_neon(const float* a, const float* b, std::size_t n);
float cosine_distance_neon(const float* a, const float* b, std::size_t n);
float inner_product_neon(const float* a, const float* b, std::size_t n);

// AVX-512 versions. Builds that target AVX-512 (VECENGINE_NATIVE on such a CPU)
// call them directly; other x86 builds with GCC or Clang compile them anyway and
// call them when the CPU has AVX-512 (detail::kAvx512), else the AVX2 ones.
float l2_distance_avx512(const float* a, const float* b, std::size_t n);
float cosine_distance_avx512(const float* a, const float* b, std::size_t n);
float inner_product_avx512(const float* a, const float* b, std::size_t n);

#if !defined(__AVX512F__) && (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#  define VECENGINE_AVX512_DISPATCH 1
namespace detail {
// Whether this CPU runs AVX-512, checked once at load. Reads false before
// then, which only means the AVX2 kernels.
extern const bool kAvx512;
} // namespace detail
#else
#  define VECENGINE_AVX512_DISPATCH 0
#endif

} // namespace vecengine

// Inline dispatch implementations
#include "distance_inl.hpp"
