#pragma once

#include <atomic>
#include <thread>
#include <vector>
#include <exception>
#include <mutex>
#include <algorithm>

namespace vecengine::detail {

using std::size_t;

template <class F>
void parallel_for(size_t n, F&& fn) {

    if (n == 1) fn(0);
    if (n <= 1) return;

    size_t num_threads = std::max(std::min(n, static_cast<size_t>(std::thread::hardware_concurrency())), size_t{1});

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
    workers.reserve(num_threads - 1);
    for (size_t t = 0; t < num_threads - 1; ++t) {
        workers.emplace_back(worker);
    }
    worker();

    for (auto& t : workers) if (t.joinable()) t.join();

    if (err) std::rethrow_exception(err);
}

} // namespace vecengine::detail
