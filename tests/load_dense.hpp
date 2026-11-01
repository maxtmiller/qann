#pragma once

#include <catch2/catch_test_macros.hpp>

#include "vecengine/index_factory.hpp"

#include <memory>

// make_index() for a dense index type.
inline std::unique_ptr<vecengine::DenseIndex> make_dense(vecengine::IndexType type, size_t dim,
                                                         const vecengine::IndexOptions& opts = {}) {
    std::unique_ptr<vecengine::IndexBase> made = vecengine::make_index(type, dim, opts);
    REQUIRE(dynamic_cast<vecengine::DenseIndex*>(made.get()) != nullptr);
    return std::unique_ptr<vecengine::DenseIndex>(static_cast<vecengine::DenseIndex*>(made.release()));
}

// load_index() for a stream or path holding a dense index.
template <class Source>
std::unique_ptr<vecengine::DenseIndex> load_dense(Source&& source) {
    std::unique_ptr<vecengine::IndexBase> loaded = vecengine::load_index(source);
    REQUIRE(dynamic_cast<vecengine::DenseIndex*>(loaded.get()) != nullptr);
    return std::unique_ptr<vecengine::DenseIndex>(static_cast<vecengine::DenseIndex*>(loaded.release()));
}
