#pragma once

// Opt-in per-section query timers (CMake -DVECENGINE_PROFILE=ON). Without the
// option the macros compile to nothing. Totals are thread_local, so read them
// with set_num_threads(1) to have every query run on the calling thread.
// benchmarks/profile_query.cpp prints the breakdown.

#ifdef VECENGINE_PROFILE
#include <chrono>

namespace vecengine::detail {

enum ProfSlot { kProfCoarse, kProfTable, kProfCodeScan, kProfRawScan, kProfDrain, kProfRerank, kProfSlots };

inline thread_local double g_prof[kProfSlots] = {}; // seconds per slot

inline double prof_now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace vecengine::detail

#define VECENGINE_PROF_START(name) const double name = vecengine::detail::prof_now()
#define VECENGINE_PROF_STOP(name, slot) (vecengine::detail::g_prof[vecengine::detail::slot] += vecengine::detail::prof_now() - (name))
#else
#define VECENGINE_PROF_START(name) ((void)0)
#define VECENGINE_PROF_STOP(name, slot) ((void)0)
#endif
