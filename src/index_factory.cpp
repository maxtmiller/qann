#include "vecengine/index_factory.hpp"
#include "vecengine/flat_index.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/refine_index.hpp"
#include "serialize.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace vecengine {

std::unique_ptr<Index> make_index(IndexType type, size_t dim, const IndexOptions& opts) {
    switch (type) {
        case IndexType::Flat:
            return std::make_unique<FlatIndex>(dim, opts.capacity);
        case IndexType::IVF: {
            auto idx = std::make_unique<IVFIndex>(dim, opts.nlist, opts.nprobe);
            if (opts.pq_subspaces > 0) idx->enable_pq(opts.pq_subspaces, opts.pq_centroids);
            return idx;
        }
    }
    throw std::invalid_argument("unknown IndexType");
}

std::unique_ptr<Index> load_index(std::istream& in) {
    switch (detail::read_header(in)) {
        case detail::IndexKind::Flat:   return FlatIndex::load_body(in);
        case detail::IndexKind::IVF:    return IVFIndex::load_body(in);
        case detail::IndexKind::Refine: return RefineIndex::load_body(in);
    }
    throw std::runtime_error("unknown index kind");
}

void save_index(const Index& index, const std::filesystem::path& path) {
    std::filesystem::path tmp = path;
    tmp += ".tmp";

    try {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot open " + tmp.string() + " for writing");
        index.save(out);
        out.close();
        if (!out) throw std::runtime_error("failed to write " + tmp.string());
        std::filesystem::rename(tmp, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(tmp, ignored);
        throw;
    }
}

std::unique_ptr<Index> load_index(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path.string() + " for reading");

    std::unique_ptr<Index> index;
    try {
        index = load_index(in);
    } catch (const std::invalid_argument& e) {
        // Constructors and setters reject bad sizes (e.g. nlist, nprobe) with
        // invalid_argument; from a file those mean corruption.
        throw std::runtime_error(std::string("corrupt file: ") + e.what());
    }

    if (in.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error("corrupt file: unexpected data after the index");
    return index;
}

} // namespace vecengine
