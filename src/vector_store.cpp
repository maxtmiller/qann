#include "vecengine/vector_store.hpp"
#include "vecengine/id_map.hpp"
#include "serialize.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>

namespace vecengine::detail {

namespace {

// File offset the rows start at: a multiple of every page size in use (4 KB x86,
// 16 KB macOS arm64, up to 64 KB on some Linux arm64 kernels).
constexpr uint64_t kRowAlign = 64 * 1024;

std::runtime_error sys_error(const std::string& what, const std::filesystem::path& path) {
    return std::runtime_error("cannot " + what + " " + path.string() + ": " + std::strerror(errno));
}

} // namespace

MappedFile::MappedFile(const std::filesystem::path& path) : path_(path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw sys_error("open", path);
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        const auto err = sys_error("stat", path);
        ::close(fd);
        throw err;
    }
    size_ = static_cast<size_t>(st.st_size);
    if (size_ == 0) {
        ::close(fd);
        throw std::runtime_error("not a qann index file");
    }
    void* p = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) throw sys_error("map", path);
    data_ = static_cast<const char*>(p);
}

void MappedFile::advise_random(const char* begin, size_t bytes) const {
    static const uintptr_t page = static_cast<uintptr_t>(::sysconf(_SC_PAGESIZE));
    const uintptr_t start = reinterpret_cast<uintptr_t>(begin) & ~(page - 1);
    ::madvise(reinterpret_cast<void*>(start), reinterpret_cast<uintptr_t>(begin) + bytes - start, MADV_RANDOM);
}

void write_pad(std::ostream& out, uint64_t align) {
    // The pad is computed from the stream position, which is the file offset when
    // saving to a file; a stream that can't report it (-1) gets no padding.
    const auto pos = static_cast<int64_t>(out.tellp());
    const uint64_t after = pos < 0 ? 0 : static_cast<uint64_t>(pos) + sizeof(uint32_t);
    const auto pad = static_cast<uint32_t>(pos < 0 ? 0 : (align - after % align) % align);
    write_pod(out, pad);
    static const char zeros[4096] = {};
    for (uint32_t left = pad; left > 0;) {
        const uint32_t chunk = std::min<uint32_t>(left, sizeof(zeros));
        out.write(zeros, chunk);
        left -= chunk;
    }
}

void read_pad(std::istream& in, uint64_t align) {
    const auto pad = read_pod<uint32_t>(in);
    if (pad >= align) throw std::runtime_error("corrupt file: invalid padding");
    in.ignore(pad);
    if (!in || in.gcount() != static_cast<std::streamsize>(pad)) throw std::runtime_error("truncated file");
}

const char* map_next(std::istream& in, const MappedFile& file, uint64_t bytes, size_t align) {
    const auto offset = static_cast<int64_t>(in.tellg());
    if (offset < 0 || static_cast<uint64_t>(offset) > file.size() || bytes > file.size() - static_cast<uint64_t>(offset))
        throw std::runtime_error("truncated file");
    if (static_cast<uint64_t>(offset) % align != 0) throw std::runtime_error("corrupt file: misaligned data");
    in.seekg(static_cast<std::streamoff>(offset + static_cast<int64_t>(bytes)));
    return file.data() + offset;
}

std::shared_ptr<const MappedFile> MappedFile::temporary(const std::filesystem::path& near,
                                                        const std::function<void(std::ostream&)>& write) {
    std::string name = (near.parent_path() / (near.filename().string() + ".compact-XXXXXX")).string();
    const int fd = ::mkstemp(name.data());
    if (fd < 0) throw sys_error("create a temporary file for compaction next to", near);
    ::close(fd);
    try {
        {
            std::ofstream out(name, std::ios::binary | std::ios::trunc);
            write(out);
            out.close();
            if (!out) throw std::runtime_error("failed to write " + name);
        }
        auto file = std::make_shared<const MappedFile>(name);
        std::filesystem::remove(name);  // the map keeps the data until it goes
        return file;
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(name, ignored);
        throw;
    }
}

MappedFile::~MappedFile() {
    ::munmap(const_cast<char*>(data_), size_);
}

size_t VectorStore::slots() const noexcept {
    return table_.empty() ? mapped_rows_ + owned_.size() / dim_ : table_.size();
}

void VectorStore::append(span<const float> vecs) {
    if (!table_.empty()) {
        const size_t first = mapped_rows_ + owned_.size() / dim_;
        for (size_t r = 0; r < vecs.size() / dim_; ++r) table_.push_back(static_cast<uint32_t>(first + r));
    }
    owned_.insert(owned_.end(), vecs.begin(), vecs.end());
}

void VectorStore::compact(span<const uint32_t> remap, size_t new_slots) {
    if (!mapped()) {
        for (size_t i = 0; i < remap.size(); ++i) {
            if (remap[i] == IdMap::kRemoved || remap[i] == i) continue;
            std::copy_n(owned_.data() + i * dim_, dim_, owned_.data() + size_t{remap[i]} * dim_);
        }
        owned_.resize(new_slots * dim_);
        owned_.shrink_to_fit();
        return;
    }
    if (new_slots == 0) {
        *this = VectorStore(dim_);
        return;
    }
    vector<uint32_t> table(new_slots);
    for (size_t i = 0; i < remap.size(); ++i)
        if (remap[i] != IdMap::kRemoved) table[remap[i]] = table_.empty() ? static_cast<uint32_t>(i) : table_[i];
    table_ = std::move(table);
}

void VectorStore::save(std::ostream& out) const {
    const size_t n = slots();
    write_pod(out, static_cast<uint64_t>(n * dim_));
    write_pad(out, kRowAlign);

    if (!mapped()) {
        out.write(reinterpret_cast<const char*>(owned_.data()), static_cast<std::streamsize>(n * dim_ * sizeof(float)));
        return;
    }
    for (size_t s = 0; s < n; ++s)
        out.write(reinterpret_cast<const char*>(row(static_cast<uint32_t>(s))),
                  static_cast<std::streamsize>(dim_ * sizeof(float)));
}

VectorStore VectorStore::load(std::istream& in, uint32_t version, uint64_t slots, size_t dim, const LoadContext* ctx) {
    VectorStore store(dim);
    const uint64_t len = read_pod<uint64_t>(in);
    if (len != slots * dim) throw std::runtime_error("corrupt file: unexpected number of vectors");

    if (version >= 10) read_pad(in, kRowAlign);

    const bool map = ctx && ctx->file;
    if (map && version < 10)
        throw std::runtime_error("mmap needs a file saved in format 10 or later; load this one without mmap and save it again");
    if (map && len > 0) {
        const uint64_t bytes = len * sizeof(float);
        const char* rows = map_next(in, *ctx->file, bytes, alignof(float));
        // re-ranking reads scattered rows, so readahead would only fetch rows nobody asked for
        ctx->file->advise_random(rows, bytes);
        store.file_ = ctx->file;
        store.mapped_ = reinterpret_cast<const float*>(rows);
        store.mapped_rows_ = slots;
        return store;
    }

    store.owned_.resize(len);
    if (len > 0) in.read(reinterpret_cast<char*>(store.owned_.data()), static_cast<std::streamsize>(len * sizeof(float)));
    if (!in) throw std::runtime_error("truncated file");
    return store;
}

} // namespace vecengine::detail
