#include "vecengine/index_factory.hpp"
#include "vecengine/flat_index.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/hnsw_index.hpp"
#include "vecengine/refine_index.hpp"
#include "vecengine/sparse_index.hpp"
#include "serialize.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace vecengine {

std::unique_ptr<Index> make_index(IndexType type, size_t dim, const IndexOptions& opts) {
    switch (type) {
        case IndexType::Flat:
            return std::make_unique<FlatIndex>(dim, opts.capacity, opts.custom_ids, opts.metric);
        case IndexType::IVF: {
            auto idx = std::make_unique<IVFIndex>(dim, opts.nlist, opts.nprobe, opts.custom_ids, opts.metric);
            if (opts.pq_subspaces > 0) idx->enable_pq(opts.pq_subspaces, opts.pq_centroids, false, opts.pq_rotate);
            if (opts.ivf_sq_bits) idx->enable_sq(opts.ivf_sq_bits);
            return idx;
        }
        case IndexType::HNSW: {
            auto idx = std::make_unique<HNSWIndex>(dim, opts.hnsw_m, opts.ef_construction, opts.custom_ids, std::nullopt,
                                                    opts.metric);
            idx->set_ef_search(opts.ef_search);
            if (opts.hnsw_sq_bits) idx->enable_sq(opts.hnsw_sq_bits);
            return idx;
        }
    }
    throw std::invalid_argument("unknown IndexType");
}

std::unique_ptr<IndexBase> load_index(std::istream& in) {
    const detail::Header header = detail::read_header(in);
    switch (header.kind) {
        case detail::IndexKind::Flat:   return FlatIndex::load_body(in, header.version);
        case detail::IndexKind::IVF:    return IVFIndex::load_body(in, header.version);
        case detail::IndexKind::Refine: return RefineIndex::load_body(in, header.version);
        case detail::IndexKind::HNSW:   return HNSWIndex::load_body(in, header.version);
        case detail::IndexKind::Sparse: return SparseIndex::load_body(in, header.version);
    }
    throw std::runtime_error("unknown index kind");
}

void save_index(const IndexBase& index, const std::filesystem::path& path) {
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

std::unique_ptr<IndexBase> load_index(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path.string() + " for reading");

    std::unique_ptr<IndexBase> index;
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
