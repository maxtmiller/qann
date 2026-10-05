#include "vecengine/threads.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <thread>

namespace vecengine {

static std::atomic<size_t> l_num_threads{0};

void set_num_threads(size_t n) {
    l_num_threads.store(n, std::memory_order_relaxed);
}

size_t num_threads() {
    size_t n = l_num_threads.load(std::memory_order_relaxed);
    if (n == 0) n = std::max<size_t>(std::thread::hardware_concurrency(), 1);
    return n;
}

} // namespace vecengine
