#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
NPROC=$(nproc)

echo "Building HeteroSched runtime..."

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=native

make -j"$NPROC"

cd ..

echo ""
echo "Build complete: ${BUILD_DIR}/heterosched_bench"
echo "Run with: ./${BUILD_DIR}/heterosched_bench --trace benchmarks/traces/mixed_trace.json --output results.csv"
