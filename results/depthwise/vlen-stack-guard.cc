// SPDX-License-Identifier: Apache-2.0

#include "tflite/kernels/internal/optimized/rvv_optimized_ops.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace tflite::rvv_optimized_ops;

int main() {
  constexpr int float_depth = 80;
  std::vector<float> float_input(float_depth);
  std::vector<float> float_output(float_depth, 0.0f);
  for (int i = 0; i < float_depth; ++i) {
    float_input[i] = static_cast<float>(i - 37);
  }
  MaxPoolChannels<float>(
      float_input.data(), float_output.data(), float_depth, 1, 1, 1, 1, 1, 1,
      1, 1, 0, 0, 1, -1000.0f, -1000.0f, 1000.0f);
  for (int i = 0; i < float_depth; ++i) {
    if (float_output[i] != float_input[i]) {
      std::fprintf(stderr, "float mismatch at %d\n", i);
      return 1;
    }
  }

  constexpr int int8_depth = 128;
  std::vector<int8_t> int8_input(int8_depth);
  std::vector<int8_t> int8_output(int8_depth, 0);
  for (int i = 0; i < int8_depth; ++i) {
    int8_input[i] = static_cast<int8_t>(i - 64);
  }
  MaxPoolChannels<int8_t>(
      int8_input.data(), int8_output.data(), int8_depth, 1, 1, 1, 1, 1, 1,
      1, 1, 0, 0, 1, -128, -128, 127);
  if (int8_output != int8_input) {
    std::fprintf(stderr, "int8 mismatch\n");
    return 2;
  }

  std::puts("RVV_VLEN_GUARD_PASS");
  return 0;
}
