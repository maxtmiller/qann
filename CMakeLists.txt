cmake_minimum_required(VERSION 3.20)
project(vecengine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Dependencies via FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.3
)

FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.8.3
)

# Must be set before FetchContent_MakeAvailable to suppress GoogleTest download
set(BENCHMARK_ENABLE_TESTING    OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_INSTALL_DOCS      OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG        v1.9.2
)

FetchContent_MakeAvailable(Catch2 benchmark nanobind)

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
add_compile_options(
  $<$<CONFIG:Release>:-O3>
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall>
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wextra>
)

# AVX2+FMA are x86-only; skip silently on arm64 (Apple Silicon uses NEON).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686")
  add_compile_options(
    $<$<CONFIG:Release>:-mavx2>
    $<$<CONFIG:Release>:-mfma>
  )
endif()

# ---------------------------------------------------------------------------
# Core static library
# ---------------------------------------------------------------------------
add_library(vecengine_core STATIC
  src/distance.cpp
  src/flat_index.cpp
  src/ivf_index.cpp
  src/pq.cpp
  src/kmeans.cpp
  src/refine_index.cpp
  src/index_factory.cpp
)

target_include_directories(vecengine_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(vecengine_core PUBLIC cxx_std_20)

# Accelerate's cblas_sgemm runs on Apple's matrix unit; k-means assignment uses
# it when available. PUBLIC link: anything linking the static lib needs it too.
if(APPLE)
  target_link_libraries(vecengine_core PUBLIC "-framework Accelerate")
  target_compile_definitions(vecengine_core PRIVATE VECENGINE_USE_ACCELERATE ACCELERATE_NEW_LAPACK)
endif()

# ---------------------------------------------------------------------------
# Unit tests (Catch2 v3)
# ---------------------------------------------------------------------------
add_executable(test_vecengine
  tests/test_distance.cpp
)

target_link_libraries(test_vecengine PRIVATE
  vecengine_core
  Catch2::Catch2WithMain
)

include(CTest)
include(Catch)
catch_discover_tests(test_vecengine)

# ---------------------------------------------------------------------------
# Microbenchmarks (Google Benchmark)
# ---------------------------------------------------------------------------
add_executable(bench_vecengine
  benchmarks/bench_distance.cpp
)

target_link_libraries(bench_vecengine PRIVATE
  vecengine_core
  benchmark::benchmark
)

# ---------------------------------------------------------------------------
# Python bindings (nanobind)
# ---------------------------------------------------------------------------
find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

nanobind_add_module(vecengine_py
  bindings/python_bindings.cpp
)

target_link_libraries(vecengine_py PRIVATE vecengine_core)

set_target_properties(vecengine_py PROPERTIES
  OUTPUT_NAME "vecengine"
)
