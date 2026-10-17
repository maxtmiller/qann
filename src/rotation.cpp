#include "rotation.hpp"

#include <stdexcept>

#ifdef VECENGINE_USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

namespace vecengine::detail {

void symmetric_eigen(vector<double> a, size_t d, vector<double>& values, vector<double>& vectors) {
    // TODO(#41): start from vectors = I; sweep every pair p < q, zeroing a[p][q]
    // with a Jacobi rotation (update rows/columns p and q of a, columns p and q
    // of vectors); stop when the squared off-diagonal sum is below ~1e-12 of
    // the total, or after ~50 sweeps. values = diagonal of a.
    (void)a; (void)d; (void)values; (void)vectors;
    throw std::logic_error("symmetric_eigen is not implemented yet");
}

vector<float> balanced_pca_rotation(span<const float> vecs, size_t n, size_t d, size_t m) {
    // TODO(#41):
    // 1. mean, then covariance sum (x - mean)^T (x - mean) / (n - 1), in double
    // 2. symmetric_eigen
    // 3. eigenvalues largest first, each into the subspace (d / m slots) with
    //    room and the smallest sum of log(max(lambda, 1e-12))
    // 4. column (subspace * d / m + position) of R = that eigenvector
    (void)vecs; (void)n; (void)d; (void)m;
    throw std::logic_error("balanced_pca_rotation is not implemented yet");
}

void rotate(span<const float> r, const float* in, size_t n, size_t d, float* out) {
    // TODO(#41): cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, d, d,
    // 1, in, d, r, d, 0, out, d) with BLAS; a plain triple loop otherwise.
    (void)r; (void)in; (void)n; (void)d; (void)out;
    throw std::logic_error("rotate is not implemented yet");
}

} // namespace vecengine::detail
