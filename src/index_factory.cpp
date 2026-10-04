#include "vecengine/index_factory.hpp"
#include "vecengine/flat_index.hpp"
#include "vecengine/ivf_index.hpp"

#include <stdexcept>

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

} // namespace vecengine
