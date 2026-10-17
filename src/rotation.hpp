#pragma once

// PCA-balanced rotation for IVF-PQ (issue #41): an orthogonal matrix R, learned
// from the training sample, that turns vectors into the data's principal axes
// and spreads those axes over the PQ subspaces so each gets a similar share of
// the variance. Rotations preserve L2 and cosine distances, so only PQ's
// quantization error changes.
//
// Convention used everywhere: R is d x d, row-major, and a row vector x is
// rotated as x * R, so column j of R is the direction that becomes output
// coordinate j.

#include <cstddef>
#include <span>
#include <vector>

namespace vecengine::detail {

using std::size_t;
using std::span;
using std::vector;

// Eigendecomposition of the symmetric d x d matrix a (row-major) by cyclic
// Jacobi rotations. Fills values with the d eigenvalues and vectors (d x d,
// row-major) with the matching unit eigenvectors as columns, in no particular
// order. a is taken by value since Jacobi overwrites its copy.
void symmetric_eigen(vector<double> a, size_t d, vector<double>& values, vector<double>& vectors);

// The rotation for PQ with m subspaces (d % m == 0), learned from n vectors
// of d floats: principal axes from the covariance, assigned largest
// eigenvalue first to the subspace with room whose sum of log eigenvalues is
// smallest. Returns R (d x d, row-major, orthogonal).
vector<float> balanced_pca_rotation(span<const float> vecs, size_t n, size_t d, size_t m);

// Whether R (d x d) satisfies R^T R = I to within 1e-3 per entry. Used to
// reject corrupt files; O(d^3).
bool is_orthogonal(span<const float> r, size_t d);

// out = in * R for n rows of d floats (in and out must not overlap), on
// num_threads() threads in blocks of 64 rows (one block runs on the caller, so
// a single query can rotate inside a parallel_for). Plain loops, not BLAS, so
// results are the same on every platform.
void rotate(span<const float> r, const float* in, size_t n, size_t d, float* out);

} // namespace vecengine::detail
