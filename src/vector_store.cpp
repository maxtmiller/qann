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

MappedFile::MappedFile(const std::filesystem::path& path) {
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
    ::madvise(p, size_, MADV_RANDOM);
    data_ = static_cast<const char*>(p);
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

    // The pad is computed from the stream position, which is the file offset when
    // saving to a file; a stream that can't report it (-1) gets no padding.
    const auto pos = static_cast<int64_t>(out.tellp());
    const uint64_t after = pos < 0 ? 0 : static_cast<uint64_t>(pos) + sizeof(uint32_t);
    const auto pad = static_cast<uint32_t>(pos < 0 ? 0 : (kRowAlign - after % kRowAlign) % kRowAlign);
    write_pod(out, pad);
    static const char zeros[4096] = {};
    for (uint32_t left = pad; left > 0;) {
        const uint32_t chunk = std::min<uint32_t>(left, sizeof(zeros));
        out.write(zeros, chunk);
        left -= chunk;
    }

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

    if (version >= 10) {
        const auto pad = read_pod<uint32_t>(in);
        if (pad >= kRowAlign) throw std::runtime_error("corrupt file: invalid vector padding");
        in.ignore(pad);
        if (!in || in.gcount() != static_cast<std::streamsize>(pad)) throw std::runtime_error("truncated file");
    }

    const bool map = ctx && ctx->file;
    if (map && version < 10)
        throw std::runtime_error("mmap needs a file saved in format 10 or later; load this one without mmap and save it again");
    if (map && len > 0) {
        const auto offset = static_cast<int64_t>(in.tellg());
        const MappedFile& file = *ctx->file;
        if (offset < 0 || static_cast<uint64_t>(offset) > file.size() ||
            len > (file.size() - static_cast<uint64_t>(offset)) / sizeof(float))
            throw std::runtime_error("truncated file");
        if (offset % alignof(float) != 0) throw std::runtime_error("corrupt file: misaligned vectors");
        store.file_ = ctx->file;
        store.mapped_ = reinterpret_cast<const float*>(file.data() + offset);
        store.mapped_rows_ = slots;
        in.seekg(static_cast<std::streamoff>(offset + static_cast<int64_t>(len * sizeof(float))));
        return store;
    }

    store.owned_.resize(len);
    if (len > 0) in.read(reinterpret_cast<char*>(store.owned_.data()), static_cast<std::streamsize>(len * sizeof(float)));
    if (!in) throw std::runtime_error("truncated file");
    return store;
}

} // namespace vecengine::detail
