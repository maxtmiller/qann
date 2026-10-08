#pragma once

#include "vecengine/threads.hpp"

#include <atomic>
#include <thread>
#include <vector>
#include <exception>
#include <mutex>
#include <algorithm>

namespace vecengine::detail {

using std::size_t;

// Calls fn(i) for every i in [0, n), spread over num_threads() threads (the
// calling thread works too). Returns once all calls finish. If any call
// throws, the remaining work is skipped and the first exception is rethrown.
template <class F>
void parallel_for(size_t n, F&& fn) {

    size_t threads = std::min(n, vecengine::num_threads());
    if (threads <= 1) {
        for (size_t i = 0; i < n; ++i) fn(i);
        return;
    }

    std::atomic<size_t> next{0};
    std::exception_ptr err = nullptr;
    std::mutex err_mutex;

    auto worker = [&]() {
        while (true) {
            {
                std::lock_guard<std::mutex> lock(err_mutex);
                if (err != nullptr) break;
            }

            size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= n) break;

            try {
                fn(i);
            } catch (...) {
                std::lock_guard<std::mutex> lock(err_mutex);
                if (!err) {
                    err = std::current_exception();
                }
                break;
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(threads - 1);
    for (size_t t = 0; t < threads - 1; ++t) {
        workers.emplace_back(worker);
    }
    worker();

    for (auto& t : workers) if (t.joinable()) t.join();

    if (err) std::rethrow_exception(err);
}

} // namespace vecengine::detail
