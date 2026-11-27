#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <memory>
#include <span>
#include <vector>

namespace vecengine::detail {

using std::size_t;
using std::span;
using std::vector;

// A whole file mapped read-only, with the kernel's default readahead. Unmapped
// when the last owner releases it. The file must not be truncated
// while mapped; replacing it by rename (as save_index does) is safe.
class MappedFile {
public:
    // Throws std::runtime_error if the file can't be opened or mapped.
    explicit MappedFile(const std::filesystem::path& path);

    // A new file next to `near`, filled by write(out) and mapped, then deleted, so it lasts only as
    // long as the map. Throws std::runtime_error if it can't be written (nothing is left behind).
    static std::shared_ptr<const MappedFile> temporary(const std::filesystem::path& near,
                                                       const std::function<void(std::ostream&)>& write);
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const char* data() const noexcept { return data_; }
    size_t size() const noexcept { return size_; }
    // The file mapped, or for a temporary one the file it was made next to.
    const std::filesystem::path& path() const noexcept { return path_; }

    // Turns readahead off (MADV_RANDOM) for the pages of [begin, begin + bytes), within the map.
    void advise_random(const char* begin, size_t bytes) const;

private:
    const char* data_ = nullptr;
    size_t size_ = 0;
    std::filesystem::path path_;
};

// What load_body() needs beyond the stream. A set `file` (the file being
// read) makes a RefineIndex map its raw vectors from it instead of reading
// them in.
struct LoadContext {
    std::shared_ptr<const MappedFile> file;
};

// Writes a u32 pad length and that many zero bytes, so the stream position (the
// file offset when saving to a file) becomes a multiple of align.
void write_pad(std::ostream& out, uint64_t align);

// Skips what write_pad wrote; throws std::runtime_error if the pad is >= align or truncated.
void read_pad(std::istream& in, uint64_t align);

// The next `bytes` of `in`, which must be reading `file`, as a pointer into the map; seeks
// past them. Throws std::runtime_error if they run past the file or the offset isn't a
// multiple of align.
const char* map_next(std::istream& in, const MappedFile& file, uint64_t bytes, size_t align);

// Allocates bytes; on Linux a block of 2 MB or more is 2 MB aligned and advised
// MADV_HUGEPAGE, since most distros only give huge pages to memory that asks.
void* huge_page_alloc(size_t bytes);
// Frees what huge_page_alloc(bytes) returned.
void huge_page_free(void* p, size_t bytes) noexcept;

// Allocator for large buffers read at random rows: re-ranking a 768-d set
// touches one 3 KB row per candidate across gigabytes, and huge pages cut that
// cost about 25% on landmark-dino (x86), mostly page walks.
template <class T>
struct HugePageAllocator {
    using value_type = T;
    HugePageAllocator() noexcept = default;
    template <class U>
    HugePageAllocator(const HugePageAllocator<U>&) noexcept {}
    T* allocate(size_t n) { return static_cast<T*>(huge_page_alloc(n * sizeof(T))); }
    void deallocate(T* p, size_t n) noexcept { huge_page_free(p, n * sizeof(T)); }
    template <class U>
    bool operator==(const HugePageAllocator<U>&) const noexcept { return true; }
};

// A RefineIndex's raw vectors, one row of dim floats per slot. Rows live in
// RAM, or, for an index loaded with mmap, in the mapped file plus a RAM tail
// for rows added since. A mapped store can't move its rows, so compact()
// rewrites a slot -> row table instead, and save() writes rows in slot order.
class VectorStore {
public:
    explicit VectorStore(size_t dim = 0) : dim_(dim) {}

    // Slots held (rows reachable through row()).
    size_t slots() const noexcept;
    bool mapped() const noexcept { return file_ != nullptr; }

    // The row of `slot` (< slots()).
    const float* row(uint32_t slot) const noexcept {
        const size_t r = table_.empty() ? slot : table_[slot];
        return r < mapped_rows_ ? mapped_ + r * dim_ : owned_.data() + (r - mapped_rows_) * dim_;
    }

    // Appends rows (vecs.size() a multiple of dim) as the next slots.
    void append(span<const float> vecs);

    // remap as returned by IdMap::compact: each old slot's new slot, or IdMap::kRemoved.
    // In RAM rows are moved down; mapped, only the slot -> row table changes and
    // removed rows stay in the file until the next save.
    void compact(span<const uint32_t> remap, size_t new_slots);

    // Writes the element count, padding up to a 64 KB file offset, then the rows in slot order.
    void save(std::ostream& out) const;

    // Reads what save() wrote (or, before version 10, a plain write_vec), for `slots` rows of
    // dim floats. With ctx->file the rows are mapped from that file rather than read; that needs
    // version 10. Throws std::runtime_error on corrupt data.
    static VectorStore load(std::istream& in, uint32_t version, uint64_t slots, size_t dim, const LoadContext* ctx);

private:
    size_t dim_;
    std::shared_ptr<const MappedFile> file_;
    const float* mapped_ = nullptr; // into file_, mapped_rows_ * dim_ floats
    size_t mapped_rows_ = 0;
    vector<float, HugePageAllocator<float>> owned_; // rows after the mapped ones (all rows when not mapped)
    vector<uint32_t> table_;        // slot -> row once a mapped store is compacted; empty means identity
};

} // namespace vecengine::detail
