#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build}"
BUILD_TYPE="${2:-Release}"

echo "==> Configuring ($BUILD_TYPE) in $BUILD_DIR/"
cmake -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo "==> Building"
cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)"

echo "==> Running tests"
ctest --test-dir "$BUILD_DIR" --output-on-failure

echo "==> Done. Binaries in $BUILD_DIR/"
echo "    benchmarks: ./$BUILD_DIR/bench_vecengine"
