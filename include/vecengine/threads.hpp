#pragma once

#include <cstddef>

namespace vecengine {

using std::size_t;

// Process-wide thread count for parallel work (query_batch, add_batch PQ
// encoding, PQ training). 0 means one thread per core, the default; 1 runs
// everything serially on the calling thread. Safe to call while other
// threads are querying. Does not limit the BLAS library's own threads
// (VECLIB_MAXIMUM_THREADS / OPENBLAS_NUM_THREADS).
void set_num_threads(size_t n);

// The thread count parallel_for will use: the set value, or
// hardware_concurrency() (at least 1) when it is 0.
size_t num_threads();

} // namespace vecengine
