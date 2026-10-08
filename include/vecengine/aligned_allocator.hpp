#pragma once

#include <cstdlib>
#include <cstddef>
#include <new>
#include <limits>
#include <vector>

namespace vecengine {

using std::size_t;
using std::ptrdiff_t;
using std::vector;

// std::allocator replacement whose memory starts on an Align-byte boundary,
// so SIMD loads can use aligned instructions.
// Alignment must be a power of two >= alignof(T).
// Use Align=32 for AVX2 (256-bit), Align=64 for AVX-512 (512-bit).
template <typename T, size_t Align = 32>
struct AlignedAllocator {
    using value_type = T;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    static_assert((Align & (Align - 1)) == 0, "Align must be a power of two");
    static_assert(Align >= alignof(T), "Align must be >= alignof(T)");

    AlignedAllocator() noexcept = default;

    // Converting copy, required by the allocator interface for rebinding.
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

    // Memory for n T's, Align-aligned. Throws std::bad_alloc on failure.
    [[nodiscard]] T* allocate(size_type n) {
        if (n > max_size())
            throw std::bad_array_new_length{};

        const size_type bytes = n * sizeof(T);
        // Round up to a multiple of Align so the total allocation is aligned.
        const size_type aligned_bytes = (bytes + Align - 1) & ~(Align - 1);

        void* ptr = nullptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(aligned_bytes, Align);
        if (!ptr) throw std::bad_alloc{};
#else
        // POSIX: size must be a multiple of alignment.
        if (::posix_memalign(&ptr, Align, aligned_bytes) != 0)
            throw std::bad_alloc{};
#endif
        return static_cast<T*>(ptr);
    }

    // Frees memory from allocate().
    void deallocate(T* ptr, size_type /*n*/) noexcept {
#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        ::free(ptr);
#endif
    }

    // Largest n allocate() accepts.
    [[nodiscard]] size_type max_size() const noexcept {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }

    template <typename U>
    struct rebind { using other = AlignedAllocator<U, Align>; };
};

// Allocators are stateless, so any two compare equal.
template <typename T, typename U, size_t A>
bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept { return true; }

template <typename T, typename U, size_t A>
bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept { return false; }

// Convenience aliases
template <typename T>
using AVX2Vector = vector<T, AlignedAllocator<T, 32>>;

template <typename T>
using AVX512Vector = vector<T, AlignedAllocator<T, 64>>;

} // namespace vecengine
