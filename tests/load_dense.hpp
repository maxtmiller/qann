#pragma once

#include <catch2/catch_test_macros.hpp>

#include "vecengine/index_factory.hpp"

#include <memory>

// load_index() for a stream or path holding a dense index.
template <class Source>
std::unique_ptr<vecengine::Index> load_dense(Source&& source) {
    std::unique_ptr<vecengine::IndexBase> loaded = vecengine::load_index(source);
    REQUIRE(dynamic_cast<vecengine::Index*>(loaded.get()) != nullptr);
    return std::unique_ptr<vecengine::Index>(static_cast<vecengine::Index*>(loaded.release()));
}
