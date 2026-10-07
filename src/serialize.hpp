#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace vecengine::detail {

using std::size_t;
using std::vector;

// Files are written in native byte order; every supported target is little-endian.
static_assert(std::endian::native == std::endian::little, "qann index files assume a little-endian host");

// "QANN" read as a little-endian uint32.
inline constexpr uint32_t kMagic = 0x4E4E4151;
inline constexpr uint32_t kFormatVersion = 1;

// Type tag stored in each header. Values are part of the file format: never
// renumber or reuse them, only append.
enum class IndexKind : uint8_t {
    Flat = 1,
    IVF = 2,
    Refine = 3,
};

// Writes a trivially copyable value as raw bytes.
template <class T>
void write_pod(std::ostream& out, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);

    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

// Reads a value written by write_pod; throws std::runtime_error if the
// stream ends early.
template <class T>
T read_pod(std::istream& in) {
    static_assert(std::is_trivially_copyable_v<T>);

    T val;
    in.read(reinterpret_cast<char*>(&val), sizeof(T));
    if (!in) throw std::runtime_error("truncated file");
    return val;
}

// Writes a uint64 element count followed by the elements.
template <class T>
void write_vec(std::ostream& out, const vector<T>& v) {
    static_assert(std::is_trivially_copyable_v<T>);

    const uint64_t size = v.size();
    write_pod(out, size);
    if (size > 0) out.write(reinterpret_cast<const char*>(v.data()), sizeof(T) * size);
}

// Reads a vector written by write_vec. Throws std::runtime_error if the
// stored count exceeds max_len (guards against allocating from a corrupt
// count) or the stream ends early. Callers pass the size they expect, e.g.
// count * dim.
template <class T>
vector<T> read_vec(std::istream& in, uint64_t max_len) {
    static_assert(std::is_trivially_copyable_v<T>);

    uint64_t size = read_pod<uint64_t>(in);
    if (size > max_len) throw std::runtime_error("corrupt file: vector length exceeds expected size");

    vector<T> v(size);
    if (size > 0) in.read(reinterpret_cast<char*>(v.data()), sizeof(T) * size);
    if (!in) throw std::runtime_error("truncated file");

    return v;
}

// Writes magic, format version and the type tag.
inline void write_header(std::ostream& out, IndexKind kind) {
    write_pod(out, kMagic);
    write_pod(out, kFormatVersion);
    write_pod(out, static_cast<uint8_t>(kind));
}

// Reads and validates a header written by write_header. Throws
// std::runtime_error on a wrong magic, an unsupported version or an unknown
// type tag.
inline IndexKind read_header(std::istream& in) {

    if (read_pod<uint32_t>(in) != kMagic) throw std::runtime_error("not a qann index file");

    uint32_t version = read_pod<uint32_t>(in);
    if (version != kFormatVersion) throw std::runtime_error("unsupported format version");

    uint8_t kind = read_pod<uint8_t>(in);
    switch (kind) {
        case static_cast<uint8_t>(IndexKind::Flat):
            return IndexKind::Flat;
        case static_cast<uint8_t>(IndexKind::IVF):
            return IndexKind::IVF;
        case static_cast<uint8_t>(IndexKind::Refine):
            return IndexKind::Refine;
        default:
            throw std::runtime_error("unknown index kind"); 
    }
}

} // namespace vecengine::detail
