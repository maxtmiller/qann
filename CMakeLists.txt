cmake_minimum_required(VERSION 3.20)
project(vecengine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Dependencies via FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)

# Off for Python wheel builds (pyproject.toml) so they skip fetching and
# compiling Catch2 and Google Benchmark.
option(VECENGINE_BUILD_TESTS "Build C++ tests and microbenchmarks" ON)
# Per-section query timers (src/profile.hpp) and the profile_query tool.
# Adds timer calls to the query path, so keep it off for normal builds.
option(VECENGINE_PROFILE "Build with query section timers and profile_query" OFF)

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
  GIT_TAG        v3.1.0
)

if(VECENGINE_BUILD_TESTS)
  FetchContent_MakeAvailable(Catch2 benchmark nanobind)
else()
  FetchContent_MakeAvailable(nanobind)
endif()

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
  src/index.cpp
  src/threads.cpp
  src/flat_index.cpp
  src/ivf_index.cpp
  src/pq.cpp
  src/kmeans.cpp
  src/refine_index.cpp
  src/index_factory.cpp
  src/id_map.cpp
)

target_include_directories(vecengine_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(vecengine_core PUBLIC cxx_std_20)

# The static lib is linked into the Python shared module, which Linux requires
# to be position-independent code (macOS builds PIC by default).
set_target_properties(vecengine_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(VECENGINE_PROFILE)
  target_compile_definitions(vecengine_core PUBLIC VECENGINE_PROFILE)
  add_executable(profile_query benchmarks/profile_query.cpp)
  target_include_directories(profile_query PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(profile_query PRIVATE vecengine_core)
endif()

# k-means assignment uses cblas_sgemm when available: Accelerate on Apple (runs
# on the matrix unit), otherwise any BLAS that ships cblas.h (e.g. OpenBLAS).
# Without one it falls back to a scalar scan. PUBLIC link: anything linking the
# static lib needs it too.
if(APPLE)
  target_link_libraries(vecengine_core PUBLIC "-framework Accelerate")
  target_compile_definitions(vecengine_core PRIVATE VECENGINE_USE_BLAS ACCELERATE_NEW_LAPACK)
else()
  find_package(BLAS)
  find_path(CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas)
  if(BLAS_FOUND AND CBLAS_INCLUDE_DIR)
    message(STATUS "vecengine: using BLAS (${BLAS_LIBRARIES}) with ${CBLAS_INCLUDE_DIR}/cblas.h")
    target_link_libraries(vecengine_core PUBLIC BLAS::BLAS)
    target_include_directories(vecengine_core PRIVATE ${CBLAS_INCLUDE_DIR})
    target_compile_definitions(vecengine_core PRIVATE VECENGINE_USE_BLAS)
  else()
    message(STATUS "vecengine: no BLAS with cblas.h found; k-means assignment uses the scalar fallback")
  endif()
endif()

if(VECENGINE_BUILD_TESTS)
# ---------------------------------------------------------------------------
# Unit tests (Catch2 v3)
# ---------------------------------------------------------------------------
add_executable(test_vecengine
  tests/test_distance.cpp
  tests/test_serialize.cpp
  tests/test_id_map.cpp
)

target_link_libraries(test_vecengine PRIVATE
  vecengine_core
  Catch2::Catch2WithMain
)

# Lets tests reach internal headers like parallel.hpp.
target_include_directories(test_vecengine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
# Index files saved by older releases, for format compatibility tests.
target_compile_definitions(test_vecengine PRIVATE QANN_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")

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
endif()

# ---------------------------------------------------------------------------
# Python bindings (nanobind)
# ---------------------------------------------------------------------------
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

nanobind_add_module(vecengine_py
  bindings/python_bindings.cpp
)

target_link_libraries(vecengine_py PRIVATE vecengine_core)

set_target_properties(vecengine_py PROPERTIES
  OUTPUT_NAME "qann"
)

# Type stubs (qann.pyi) for editors and type checkers, generated by importing
# the freshly built module, so they always match the bindings.
nanobind_add_stub(qann_stub
  MODULE qann
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/qann.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:vecengine_py>
  DEPENDS vecengine_py
)

# scikit-build-core sets SKBUILD; the module goes at the wheel root so
# `import qann` finds it. Type checkers only read installed stubs from a
# package, so the stubs ship as the PEP 561 stub package qann-stubs.
if(SKBUILD)
  install(TARGETS vecengine_py LIBRARY DESTINATION .)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/qann.pyi DESTINATION qann-stubs RENAME __init__.pyi)
  target_compile_definitions(vecengine_py PRIVATE QANN_VERSION="${SKBUILD_PROJECT_VERSION_FULL}")
else()
  target_compile_definitions(vecengine_py PRIVATE QANN_VERSION="dev")
endif()
