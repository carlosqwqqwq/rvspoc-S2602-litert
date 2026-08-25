#!/usr/bin/env bash
set -eu

src_root=${1:-"$(cd "$(dirname "$0")/.." && pwd)"}
build_root=${2:-"$src_root/build-rv64gcv"}

cmake -S "$src_root/tflite" -B "$build_root" \
  -DCMAKE_TOOLCHAIN_FILE="$src_root/toolchains/riscv64-linux-gnu.cmake" \
  -DTENSORFLOW_SOURCE_DIR="$src_root/third_party/tensorflow" \
  -DTFLITE_HOST_TOOLS_DIR=/usr/bin \
  -DProtobuf_PROTOC_EXECUTABLE=/usr/bin/protoc \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-march=rv64gcv -mabi=lp64d" \
  -DTFLITE_ENABLE_BENCHMARK_MODEL=ON \
  -DTFLITE_ENABLE_XNNPACK=OFF \
  -DTFLITE_ENABLE_GPU=OFF
cmake --build "$build_root" --target benchmark_model --parallel 8
