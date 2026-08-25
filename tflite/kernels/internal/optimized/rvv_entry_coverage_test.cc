/* Copyright 2026 The LiteRT RVV contributors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "tflite/kernels/cpu_backend_context.h"
#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/optimized/depthwiseconv_multithread.h"
#include "tflite/kernels/internal/optimized/optimized_ops.h"
#if defined(__riscv_vector)
#include "tflite/kernels/internal/optimized/rvv_tensor_utils.h"
#endif
#include "tflite/kernels/internal/reference/depthwiseconv_float.h"
#include "tflite/kernels/internal/reference/reference_ops.h"

namespace tflite {
namespace {

constexpr int kN = 37;

RuntimeShape Shape(int n = kN) { return RuntimeShape({n}); }

ArithmeticParams FloatParams() {
  ArithmeticParams p = {};
  p.float_activation_min = -2.0f;
  p.float_activation_max = 2.0f;
  return p;
}

ArithmeticParams QuantParams() {
  ArithmeticParams p = {};
  p.input1_offset = -11;
  p.input2_offset = 7;
  p.output_offset = 123;
  p.output_multiplier = 1 << 30;
  p.output_shift = 1;
  p.left_shift = 0;
  p.input1_multiplier = 1 << 30;
  p.input1_shift = 1;
  p.input2_multiplier = 1 << 30;
  p.input2_shift = 1;
  p.quantized_activation_min = 0;
  p.quantized_activation_max = 255;
  return p;
}

template <typename T>
void ExpectSame(const std::vector<T> &got, const std::vector<T> &want) {
  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < got.size(); ++i)
    EXPECT_EQ(got[i], want[i]) << i;
}

void ExpectNear(const std::vector<float> &got, const std::vector<float> &want) {
  ASSERT_EQ(got.size(), want.size());
  for (size_t i = 0; i < got.size(); ++i)
    EXPECT_NEAR(got[i], want[i], 1e-5f) << i;
}

TEST(RvvEntryCoverageTest, FloatElementwiseActivationAndArg) {
  const RuntimeShape shape = Shape();
  const ArithmeticParams params = FloatParams();
  std::vector<float> a(kN), b(kN), alpha(kN), got(kN), want(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = (i - 18) * .25f;
    b[i] = (17 - i) * .125f;
    alpha[i] = .1f + (i % 5) * .05f;
  }

  optimized_ops::AddElementwise(kN, params, a.data(), b.data(), got.data());
  reference_ops::Add(params, shape, a.data(), shape, b.data(), shape,
                     want.data());
  ExpectNear(got, want);
  optimized_ops::MulElementwise(kN, params, a.data(), b.data(), got.data());
  reference_ops::Mul(params, shape, a.data(), shape, b.data(), shape,
                     want.data());
  ExpectNear(got, want);

  optimized_ops::AddScalarBroadcast(kN, params, 1.25f, b.data(), got.data());
  for (int i = 0; i < kN; ++i)
    want[i] = std::clamp(1.25f + b[i], -2.0f, 2.0f);
  ExpectNear(got, want);
  optimized_ops::MulSimpleBroadcast(kN, params, 1.5f, b.data(), got.data());
  for (int i = 0; i < kN; ++i)
    want[i] = std::clamp(1.5f * b[i], -2.0f, 2.0f);
  ExpectNear(got, want);

  optimized_ops::HardSwish(shape, a.data(), shape, got.data());
  reference_ops::HardSwish(shape, a.data(), shape, want.data());
  ExpectNear(got, want);
  optimized_ops::PReluScalarBroadcast(kN, params, .2f, a.data(), got.data());
  for (int i = 0; i < kN; ++i)
    want[i] = a[i] >= 0 ? a[i] : .2f * a[i];
  ExpectNear(got, want);
  optimized_ops::PReluElementWise(kN, params, alpha.data(), a.data(),
                                  got.data());
  for (int i = 0; i < kN; ++i)
    want[i] = a[i] >= 0 ? a[i] : alpha[i] * a[i];
  ExpectNear(got, want);

  a[9] = -99.0f;
  a[23] = 99.0f;
  EXPECT_EQ(optimized_ops::ArgMinVector(a.data(), kN), 9);
  EXPECT_EQ(optimized_ops::ArgMaxVector(a.data(), kN), 23);
}

TEST(RvvEntryCoverageTest, QuantizedElementwiseConversionAndMinMax) {
  const RuntimeShape shape = Shape();
  const ArithmeticParams params = QuantParams();
  std::vector<uint8_t> a(kN), b(kN), got(kN), want(kN);
  std::vector<int8_t> s8(kN), t8(kN), got8(kN);
  std::vector<int32_t> i32(kN), j32(kN), got32(kN);
  for (int i = 0; i < kN; ++i) {
    a[i] = static_cast<uint8_t>(i * 7);
    b[i] = static_cast<uint8_t>(255 - i * 3);
    s8[i] = static_cast<int8_t>(i - 18);
    t8[i] = static_cast<int8_t>(18 - i);
    i32[i] = i - 18;
    j32[i] = i % 7 - 3;
  }

  optimized_ops::AddElementwise(kN, params, a.data(), b.data(), got.data());
  reference_ops::AddElementwise<uint8_t>(kN, params, a.data(), b.data(),
                                         want.data());
  ExpectSame(got, want);
  optimized_ops::AddScalarBroadcast(kN, params, a[3], b.data(), got.data());
  reference_ops::AddScalarBroadcast(kN, params, a[3], b.data(), want.data());
  ExpectSame(got, want);
  optimized_ops::MulElementwise(kN, params, a.data(), b.data(), got.data());
  reference_ops::MulElementwise(kN, params, a.data(), b.data(), want.data());
  ExpectSame(got, want);
  optimized_ops::MulSimpleBroadcast(kN, params, a[3], b.data(), got.data());
  for (int i = 0; i < kN; ++i) {
    int32_t value = MultiplyByQuantizedMultiplier(
        (params.input1_offset + a[3]) * (params.input2_offset + b[i]),
        params.output_multiplier, params.output_shift);
    want[i] = static_cast<uint8_t>(std::clamp(params.output_offset + value,
                                              params.quantized_activation_min,
                                              params.quantized_activation_max));
  }
  ExpectSame(got, want);

  ArithmeticParams int_params = params;
  int_params.quantized_activation_min = -100;
  int_params.quantized_activation_max = 100;
  optimized_ops::MulElementwise(kN, int_params, i32.data(), j32.data(),
                                got32.data());
  for (int i = 0; i < kN; ++i)
    EXPECT_EQ(got32[i], std::clamp(i32[i] * j32[i], -100, 100));

  optimized_ops::MaximumElementwise(kN, params, s8.data(), t8.data(),
                                    got8.data());
  for (int i = 0; i < kN; ++i)
    EXPECT_EQ(got8[i], std::max(s8[i], t8[i]));
  optimized_ops::MinimumScalarBroadcast(kN, params, s8[2], t8.data(),
                                        got8.data());
  for (int i = 0; i < kN; ++i)
    EXPECT_EQ(got8[i], std::min(s8[2], t8[i]));

  std::vector<int32_t> scratch = i32;
  optimized_ops::Quantize(1 << 30, 1, kN, 128, 0, 255, scratch.data(),
                          got.data());
  for (int i = 0; i < kN; ++i) {
    int32_t value = MultiplyByQuantizedMultiplier(scratch[i], 1 << 30, 1) + 128;
    want[i] = static_cast<uint8_t>(std::clamp(value, 0, 255));
  }
  ExpectSame(got, want);

  optimized_ops::Requantize<int8_t, uint8_t>(s8.data(), kN, 1 << 30, 1, -3, 127,
                                             got.data());
  reference_ops::Requantize<int8_t, uint8_t>(s8.data(), kN, 1 << 30, 1, -3, 127,
                                             want.data());
  ExpectSame(got, want);
  optimized_ops::Requantize<uint8_t, int8_t>(a.data(), kN, 1 << 30, 1, 127, -3,
                                             got8.data());
  std::vector<int8_t> want8(kN);
  reference_ops::Requantize<uint8_t, int8_t>(a.data(), kN, 1 << 30, 1, 127, -3,
                                             want8.data());
  ExpectSame(got8, want8);

  DequantizationParams dequant = {};
  dequant.scale = .25;
  dequant.zero_point = 127;
  std::vector<float> gotf(kN), wantf(kN);
  optimized_ops::Dequantize(dequant, shape, a.data(), shape, gotf.data());
  reference_ops::Dequantize(dequant, shape, a.data(), shape, wantf.data());
  ExpectNear(gotf, wantf);

  QuantizationParams affine = {};
  affine.scale = .25;
  affine.zero_point = 3;
  std::vector<float> input(kN);
  for (int i = 0; i < kN; ++i)
    input[i] = (i - 18) * .25f;
  optimized_ops::AffineQuantize(affine, shape, input.data(), shape,
                                got8.data());
  reference_ops::AffineQuantize(affine, shape, input.data(), shape,
                                want8.data());
  ExpectSame(got8, want8);
}

TEST(RvvEntryCoverageTest, PoolAndFixedPointActivation) {
  PoolParams pool = {};
  pool.stride_height = 1;
  pool.stride_width = 1;
  pool.filter_height = 2;
  pool.filter_width = 2;
  pool.quantized_activation_min = 0;
  pool.quantized_activation_max = 255;
  const RuntimeShape input_shape({1, 3, 3, 5});
  const RuntimeShape output_shape({1, 2, 2, 5});
  std::vector<uint8_t> input(45), got(20), want(20);
  for (int i = 0; i < 45; ++i)
    input[i] = static_cast<uint8_t>(i * 5);
  ASSERT_TRUE(optimized_ops::AveragePool(pool, input_shape, input.data(),
                                         output_shape, got.data()));
  ASSERT_TRUE(reference_ops::AveragePool(pool, input_shape, input.data(),
                                         output_shape, want.data()));
  ExpectSame(got, want);
  optimized_ops::MaxPool(pool, input_shape, input.data(), output_shape,
                         got.data());
  reference_ops::MaxPool(pool, input_shape, input.data(), output_shape,
                         want.data());
  ExpectSame(got, want);

  TanhParams tanh_params = {};
  tanh_params.input_left_shift = 0;
  LogisticParams logistic_params = {};
  std::vector<int16_t> fixed_input(kN), fixed_got(kN), fixed_want(kN);
  for (int i = 0; i < kN; ++i)
    fixed_input[i] = static_cast<int16_t>((i - 18) * 997);
  optimized_ops::Tanh(tanh_params, Shape(), fixed_input.data(), Shape(),
                      fixed_got.data());
  reference_ops::Tanh(tanh_params, Shape(), fixed_input.data(), Shape(),
                      fixed_want.data());
  ExpectSame(fixed_got, fixed_want);
  optimized_ops::Logistic(logistic_params, Shape(), fixed_input.data(), Shape(),
                          fixed_got.data());
  reference_ops::Logistic(logistic_params, Shape(), fixed_input.data(), Shape(),
                          fixed_want.data());
  ExpectSame(fixed_got, fixed_want);
}

TEST(RvvEntryCoverageTest, ShuffledFullyConnected) {
  FullyConnectedParams params = {};
  params.output_multiplier = 1 << 30;
  params.output_shift = 1;
  params.quantized_activation_min = -32768;
  params.quantized_activation_max = 32767;
  const RuntimeShape input_shape({1, 16});
  const RuntimeShape weights_shape({4, 16});
  const RuntimeShape bias_shape({4});
  const RuntimeShape output_shape({1, 4});
  std::vector<uint8_t> input(16), weights(64), workspace(16);
  std::vector<int32_t> bias({1, -2, 3, -4});
  std::vector<int16_t> got(4);
  for (int i = 0; i < 16; ++i)
    input[i] = static_cast<uint8_t>(i * 9 + 3);
  for (int i = 0; i < 64; ++i)
    weights[i] = static_cast<uint8_t>(static_cast<int8_t>((i % 7) - 3));
  CpuBackendContext context;
  context.SetMaxNumThreads(1);
  optimized_ops::ShuffledFullyConnected(params, input_shape, input.data(),
                                        weights_shape, weights.data(),
                                        bias_shape, bias.data(), output_shape,
                                        got.data(), workspace.data(), &context);
  for (int row = 0; row < 4; ++row) {
    int32_t acc = bias[row];
    for (int col = 0; col < 16; ++col) {
      const int8_t x = static_cast<int8_t>(input[col] ^ 0x80);
      const int8_t w = static_cast<int8_t>(weights[row * 16 + col]);
      acc += static_cast<int32_t>(x) * w;
    }
    acc = MultiplyByQuantizedMultiplier(acc, params.output_multiplier,
                                        params.output_shift);
    EXPECT_EQ(got[row], static_cast<int16_t>(std::clamp(acc, -32768, 32767)));
  }

  const RuntimeShape no_bias_shape({0});
  std::vector<int16_t> no_bias_got(4);
  optimized_ops::ShuffledFullyConnected(
      params, input_shape, input.data(), weights_shape, weights.data(),
      no_bias_shape, nullptr, output_shape, no_bias_got.data(),
      workspace.data(), &context);
  for (int row = 0; row < 4; ++row) {
    int32_t acc = 0;
    for (int col = 0; col < 16; ++col) {
      const int8_t x = static_cast<int8_t>(input[col] ^ 0x80);
      const int8_t w = static_cast<int8_t>(weights[row * 16 + col]);
      acc += static_cast<int32_t>(x) * w;
    }
    acc = MultiplyByQuantizedMultiplier(acc, params.output_multiplier,
                                        params.output_shift);
    EXPECT_EQ(no_bias_got[row],
              static_cast<int16_t>(std::clamp(acc, -32768, 32767)));
  }
}

TEST(RvvEntryCoverageTest, HotspotDepthwiseAndGemm) {
  DepthwiseParams params = {};
  params.padding_type = PaddingType::kSame;
  params.padding_values.width = 1;
  params.padding_values.height = 1;
  params.stride_width = 1;
  params.stride_height = 1;
  params.dilation_width_factor = 1;
  params.dilation_height_factor = 1;
  params.depth_multiplier = 1;
  params.float_activation_min = -1.5f;
  params.float_activation_max = 1.5f;
  const RuntimeShape input_shape({1, 5, 7, 17});
  const RuntimeShape filter_shape({1, 3, 3, 17});
  const RuntimeShape bias_shape({1, 1, 1, 17});
  const RuntimeShape output_shape({1, 5, 7, 17});
  std::vector<float> input(5 * 7 * 17), filter(3 * 3 * 17), bias(17),
      got(input.size()), want(input.size());
  for (int i = 0; i < static_cast<int>(input.size()); ++i)
    input[i] = (i % 17 - 8) * .17f;
  for (int i = 0; i < static_cast<int>(filter.size()); ++i)
    filter[i] = (i % 7 - 3) * .13f;
  for (int i = 0; i < static_cast<int>(bias.size()); ++i)
    bias[i] = (i - 2) * .1f;
  CpuBackendContext context;
  context.SetMaxNumThreads(1);
  optimized_ops::DepthwiseConv<float, float>(
      params, input_shape, input.data(), filter_shape, filter.data(),
      bias_shape, bias.data(), output_shape, got.data(), &context);
  reference_ops::DepthwiseConv(params, input_shape, input.data(), filter_shape,
                               filter.data(), bias_shape, bias.data(),
                               output_shape, want.data());
  ExpectNear(got, want);

  optimized_ops::DepthwiseConv<float, float>(
      params, input_shape, input.data(), filter_shape, filter.data(),
      bias_shape, nullptr, output_shape, got.data(), &context);
  reference_ops::DepthwiseConv(params, input_shape, input.data(), filter_shape,
                               filter.data(), bias_shape, nullptr, output_shape,
                               want.data());
  ExpectNear(got, want);

  DepthwiseParams pair_params = params;
  pair_params.stride_width = 2;
  pair_params.padding_values.width = 2;
  pair_params.padding_values.height = 2;
  pair_params.dilation_width_factor = 2;
  pair_params.dilation_height_factor = 2;
  const RuntimeShape pair_input_shape({1, 9, 11, 17});
  const RuntimeShape pair_filter_shape({1, 3, 3, 17});
  const RuntimeShape pair_bias_shape({1, 1, 1, 17});
  const RuntimeShape pair_output_shape({1, 9, 6, 17});
  std::vector<float> pair_input(9 * 11 * 17), pair_filter(3 * 3 * 17),
      pair_bias(17), pair_got(9 * 6 * 17), pair_want(pair_got.size());
  for (int i = 0; i < static_cast<int>(pair_input.size()); ++i)
    pair_input[i] = (i % 23 - 11) * .09f;
  for (int i = 0; i < static_cast<int>(pair_filter.size()); ++i)
    pair_filter[i] = (i % 5 - 2) * .11f;
  for (int i = 0; i < static_cast<int>(pair_bias.size()); ++i)
    pair_bias[i] = (i - 3) * .07f;
  optimized_ops::DepthwiseConv<float, float>(
      pair_params, pair_input_shape, pair_input.data(), pair_filter_shape,
      pair_filter.data(), pair_bias_shape, pair_bias.data(), pair_output_shape,
      pair_got.data(), &context);
  reference_ops::DepthwiseConv(
      pair_params, pair_input_shape, pair_input.data(), pair_filter_shape,
      pair_filter.data(), pair_bias_shape, pair_bias.data(), pair_output_shape,
      pair_want.data());
  ExpectNear(pair_got, pair_want);

#if defined(__riscv_vector)
  constexpr int kRows = 5;
  constexpr int kDepth = 19;
  constexpr int kCols = 3;
  std::vector<float> lhs(kRows * kDepth), rhs(kCols * kDepth), gemm_bias(kRows);
  std::vector<float> gemm_got(kRows * kCols), gemm_want(kRows * kCols);
  for (int i = 0; i < static_cast<int>(lhs.size()); ++i)
    lhs[i] = (i % 13 - 6) * .11f;
  for (int i = 0; i < static_cast<int>(rhs.size()); ++i)
    rhs[i] = (i % 9 - 4) * .07f;
  for (int i = 0; i < static_cast<int>(gemm_bias.size()); ++i)
    gemm_bias[i] = (i - 2) * .03f;
  rvv_optimized_ops::RvvGemmFloat(lhs.data(), kRows, kDepth, rhs.data(), kCols,
                                  gemm_got.data(), gemm_bias.data(), -1.0f,
                                  1.0f);
  for (int col = 0; col < kCols; ++col) {
    for (int row = 0; row < kRows; ++row) {
      float sum = gemm_bias[row];
      for (int depth = 0; depth < kDepth; ++depth)
        sum += lhs[row * kDepth + depth] * rhs[col * kDepth + depth];
      gemm_want[row + col * kRows] = std::clamp(sum, -1.0f, 1.0f);
    }
  }
  ExpectNear(gemm_got, gemm_want);

  constexpr int kTileRows = 17;
  constexpr int kTileDepth = 65;
  constexpr int kTileCols = 32;
  std::vector<float> tile_lhs(kTileRows * kTileDepth);
  std::vector<float> tile_rhs(kTileCols * kTileDepth);
  std::vector<float> tile_bias(kTileRows);
  std::vector<float> tile_got(kTileRows * kTileCols);
  std::vector<float> tile_want(tile_got.size());
  for (int i = 0; i < static_cast<int>(tile_lhs.size()); ++i)
    tile_lhs[i] = (i % 19 - 9) * .07f;
  for (int i = 0; i < static_cast<int>(tile_rhs.size()); ++i)
    tile_rhs[i] = (i % 11 - 5) * .05f;
  for (int i = 0; i < kTileRows; ++i) tile_bias[i] = (i - 4) * .03f;
  rvv_optimized_ops::RvvGemmFloat(
      tile_lhs.data(), kTileRows, kTileDepth, tile_rhs.data(), kTileCols,
      tile_got.data(), tile_bias.data(), -1.0f, 1.0f);
  for (int col = 0; col < kTileCols; ++col) {
    for (int row = 0; row < kTileRows; ++row) {
      float sum = tile_bias[row];
      for (int depth = 0; depth < kTileDepth; ++depth)
        sum += tile_lhs[row * kTileDepth + depth] *
               tile_rhs[col * kTileDepth + depth];
      tile_want[row + col * kTileRows] = std::clamp(sum, -1.0f, 1.0f);
    }
  }
  ExpectNear(tile_got, tile_want);

  constexpr int kSmallRows = 8, kSmallDepth = 97, kSmallCols = 9;
  std::vector<float> small_lhs(kSmallRows * kSmallDepth),
      small_rhs(kSmallCols * kSmallDepth), small_got(kSmallRows * kSmallCols),
      small_want(small_got.size());
  for (int i = 0; i < static_cast<int>(small_lhs.size()); ++i)
    small_lhs[i] = (i % 17 - 8) * .03125f;
  for (int i = 0; i < static_cast<int>(small_rhs.size()); ++i)
    small_rhs[i] = (i % 13 - 6) * .0625f;
  rvv_optimized_ops::RvvGemmFloat(
      small_lhs.data(), kSmallRows, kSmallDepth, small_rhs.data(),
      kSmallCols, small_got.data(), nullptr, -1.0f, 1.0f);
  for (int col = 0; col < kSmallCols; ++col) {
    for (int row = 0; row < kSmallRows; ++row) {
      float sum = 0.0f;
      for (int depth = 0; depth < kSmallDepth; ++depth)
        sum += small_lhs[row * kSmallDepth + depth] *
               small_rhs[col * kSmallDepth + depth];
      small_want[row + col * kSmallRows] = std::clamp(sum, -1.0f, 1.0f);
    }
  }
  ExpectNear(small_got, small_want);

  constexpr int kQRows = 5;
  constexpr int kQDepth = 17;
  constexpr int kQCols = 3;
  std::vector<uint8_t> qlhs(kQRows * kQDepth), qrhs(kQCols * kQDepth);
  std::vector<uint8_t> qgot(kQRows * kQCols), qwant(kQRows * kQCols);
  std::vector<int32_t> qbias(kQRows);
  for (int i = 0; i < static_cast<int>(qlhs.size()); ++i)
    qlhs[i] = (i * 13 + 7) & 255;
  for (int i = 0; i < static_cast<int>(qrhs.size()); ++i)
    qrhs[i] = (i * 17 + 3) & 255;
  for (int i = 0; i < static_cast<int>(qbias.size()); ++i)
    qbias[i] = i - 2;
  rvv_optimized_ops::RvvGemmQuantized<uint8_t, uint8_t, uint8_t>(
      qlhs.data(), kQRows, kQDepth, 123, qrhs.data(), kQCols, 117, qgot.data(),
      qbias.data(), 1 << 30, 1, nullptr, nullptr, 121, 0, 255);
  for (int col = 0; col < kQCols; ++col) {
    for (int row = 0; row < kQRows; ++row) {
      int32_t sum = qbias[row];
      for (int depth = 0; depth < kQDepth; ++depth) {
        sum += (static_cast<int32_t>(qlhs[row * kQDepth + depth]) - 123) *
               (static_cast<int32_t>(qrhs[col * kQDepth + depth]) - 117);
      }
      const int32_t value =
          MultiplyByQuantizedMultiplier(sum, 1 << 30, 1) + 121;
      qwant[row + col * kQRows] =
          static_cast<uint8_t>(std::clamp(value, 0, 255));
    }
  }
  ExpectSame(qgot, qwant);

  constexpr int kI8Rows = 8, kI8Depth = 17, kI8Cols = 9;
  std::vector<int8_t> i8lhs(kI8Rows * kI8Depth), i8rhs(kI8Cols * kI8Depth);
  std::vector<int8_t> i8got(kI8Rows * kI8Cols), i8want(i8got.size());
  std::vector<int32_t> i8multipliers(kI8Rows, 1 << 30);
  std::vector<int> i8shifts(kI8Rows, -1);
  for (int i = 0; i < static_cast<int>(i8lhs.size()); ++i)
    i8lhs[i] = static_cast<int8_t>(i % 29 - 14);
  for (int i = 0; i < static_cast<int>(i8rhs.size()); ++i)
    i8rhs[i] = static_cast<int8_t>(i % 23 - 11);
  rvv_optimized_ops::RvvGemmQuantized<int8_t, int8_t, int8_t>(
      i8lhs.data(), kI8Rows, kI8Depth, 3, i8rhs.data(), kI8Cols, -4,
      i8got.data(), nullptr, 1 << 30, -1, i8multipliers.data(),
      i8shifts.data(), 2, -128, 127);
  for (int col = 0; col < kI8Cols; ++col) {
    for (int row = 0; row < kI8Rows; ++row) {
      int32_t sum = 0;
      for (int depth = 0; depth < kI8Depth; ++depth)
        sum += (static_cast<int>(i8lhs[row * kI8Depth + depth]) - 3) *
               (static_cast<int>(i8rhs[col * kI8Depth + depth]) + 4);
      const int32_t value =
          rvv_optimized_ops::MultiplyByRuyQuantizedMultiplier(sum, 1 << 30,
                                                               -1) +
          2;
      i8want[row + col * kI8Rows] =
          static_cast<int8_t>(std::clamp(value, -128, 127));
    }
  }
  ExpectSame(i8got, i8want);

  constexpr int kI8WideCols = 65;
  std::vector<int8_t> wide_rhs(kI8WideCols * kI8Depth);
  std::vector<int8_t> wide_got(kI8Rows * kI8WideCols), wide_want(wide_got.size());
  for (int i = 0; i < static_cast<int>(wide_rhs.size()); ++i)
    wide_rhs[i] = static_cast<int8_t>(i % 19 - 9);
  rvv_optimized_ops::RvvGemmQuantized<int8_t, int8_t, int8_t>(
      i8lhs.data(), kI8Rows, kI8Depth, 3, wide_rhs.data(), kI8WideCols, -4,
      wide_got.data(), nullptr, 1 << 30, -1, i8multipliers.data(),
      i8shifts.data(), 2, -128, 127);
  for (int col = 0; col < kI8WideCols; ++col) {
    for (int row = 0; row < kI8Rows; ++row) {
      int32_t sum = 0;
      for (int depth = 0; depth < kI8Depth; ++depth)
        sum += (static_cast<int>(i8lhs[row * kI8Depth + depth]) - 3) *
               (static_cast<int>(wide_rhs[col * kI8Depth + depth]) + 4);
      const int32_t value =
          rvv_optimized_ops::MultiplyByRuyQuantizedMultiplier(sum, 1 << 30,
                                                               -1) +
          2;
      wide_want[row + col * kI8Rows] =
          static_cast<int8_t>(std::clamp(value, -128, 127));
    }
  }
  ExpectSame(wide_got, wide_want);
#endif
}

#if defined(__riscv_vector)
TEST(RvvEntryCoverageTest, Int8DepthwisePairStrideAndDilation) {
  constexpr int kHeight = 9, kWidth = 11, kDepth = 17, kFilter = 3;
  constexpr int kOutputWidth = 6, kStride = 2, kDilation = 2, kPad = 2;
  std::vector<int8_t> input(kHeight * kWidth * kDepth);
  std::vector<int8_t> filter(kFilter * kFilter * kDepth);
  std::vector<int8_t> got(kHeight * kOutputWidth * kDepth), want(got.size());
  std::vector<int32_t> bias(kDepth), multipliers(kDepth, 1 << 30),
      shifts(kDepth, 1);
  for (int i = 0; i < static_cast<int>(input.size()); ++i)
    input[i] = static_cast<int8_t>((i * 13) % 127 - 63);
  for (int i = 0; i < static_cast<int>(filter.size()); ++i)
    filter[i] = static_cast<int8_t>((i * 7) % 31 - 15);
  for (int i = 0; i < kDepth; ++i) bias[i] = i * 17 - 100;
  rvv_optimized_ops::DepthwiseConvInt8PerChannel(
      input.data(), filter.data(), bias.data(), got.data(), 1, kHeight, kWidth,
      kDepth, kFilter, kFilter, kHeight, kOutputWidth, kDepth, 1, 1, kStride,
      kPad, kPad, kDilation, kDilation, 3, 5, multipliers.data(), shifts.data(),
      -128, 127, 0, kHeight, 1);
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kOutputWidth; ++x) {
      const int y0 = y - kPad;
      const int x0 = x * kStride - kPad;
      const int fy0 = std::max(0, (-y0 + kDilation - 1) / kDilation);
      const int fy1 = std::min(kFilter,
                               (kHeight - y0 + kDilation - 1) / kDilation);
      const int fx0 = std::max(0, (-x0 + kDilation - 1) / kDilation);
      const int fx1 = std::min(kFilter,
                               (kWidth - x0 + kDilation - 1) / kDilation);
      for (int c = 0; c < kDepth; ++c) {
        int32_t sum = bias[c];
        for (int fy = fy0; fy < fy1; ++fy)
          for (int fx = fx0; fx < fx1; ++fx)
            sum += (static_cast<int32_t>(
                        input[((y0 + fy * kDilation) * kWidth +
                               x0 + fx * kDilation) * kDepth + c]) +
                    3) * filter[(fy * kFilter + fx) * kDepth + c];
        const int32_t value = MultiplyByQuantizedMultiplier(
            sum, multipliers[c], shifts[c]) + 5;
        want[(y * kOutputWidth + x) * kDepth + c] =
            static_cast<int8_t>(std::clamp(value, -128, 127));
      }
    }
  }
  ExpectSame(got, want);
}

TEST(RvvEntryCoverageTest, Uint8DepthwisePairStrideAndDilation) {
  constexpr int kHeight = 9, kWidth = 11, kDepth = 17, kFilter = 3;
  constexpr int kOutputWidth = 6, kStride = 2, kDilation = 2, kPad = 2;
  constexpr int32_t kInputOffset = -123, kFilterOffset = -117;
  std::vector<uint8_t> input(kHeight * kWidth * kDepth);
  std::vector<uint8_t> filter(kFilter * kFilter * kDepth);
  std::vector<uint8_t> got(kHeight * kOutputWidth * kDepth), want(got.size());
  std::vector<int32_t> bias(kDepth);
  for (int i = 0; i < static_cast<int>(input.size()); ++i)
    input[i] = (i * 29 + 7) & 255;
  for (int i = 0; i < static_cast<int>(filter.size()); ++i)
    filter[i] = (i * 13 + 3) & 255;
  for (int i = 0; i < kDepth; ++i) bias[i] = i * 17 - 91;
  rvv_optimized_ops::DepthwiseConvUint8(
      input.data(), filter.data(), bias.data(), got.data(), 1, kHeight, kWidth,
      kDepth, kFilter, kFilter, kHeight, kOutputWidth, kDepth, 1, 1, kStride,
      kPad, kPad, kDilation, kDilation, kInputOffset, kFilterOffset, 121,
      1 << 30, 1, 0, 255, 0, kHeight, 1);
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kOutputWidth; ++x) {
      const int y0 = y - kPad;
      const int x0 = x * kStride - kPad;
      const int fy0 = std::max(0, (-y0 + kDilation - 1) / kDilation);
      const int fy1 = std::min(
          kFilter, (kHeight - y0 + kDilation - 1) / kDilation);
      const int fx0 = std::max(0, (-x0 + kDilation - 1) / kDilation);
      const int fx1 = std::min(
          kFilter, (kWidth - x0 + kDilation - 1) / kDilation);
      for (int c = 0; c < kDepth; ++c) {
        int32_t sum = bias[c];
        for (int fy = fy0; fy < fy1; ++fy)
          for (int fx = fx0; fx < fx1; ++fx)
            sum += (static_cast<int32_t>(
                        input[((y0 + fy * kDilation) * kWidth +
                               x0 + fx * kDilation) * kDepth + c]) +
                    kInputOffset) *
                   (static_cast<int32_t>(
                        filter[(fy * kFilter + fx) * kDepth + c]) +
                    kFilterOffset);
        const int32_t value = MultiplyByQuantizedMultiplier(sum, 1 << 30, 1) + 121;
        want[(y * kOutputWidth + x) * kDepth + c] =
            static_cast<uint8_t>(std::clamp(value, 0, 255));
      }
    }
  }
  ExpectSame(got, want);
}
#endif

// The Python audit owns the full 61-row source-to-route accounting. This
// binary intentionally keeps executable checks at route-family granularity so
// the same target remains useful for both rv64gc and RVV builds.

#if defined(__riscv_vector)
TEST(RvvEntryCoverageTest, VlenIsRuntimeVisible) {
  EXPECT_GE(tensor_utils::RvvGetVlenBits(), 128);
}
#endif

} // namespace
} // namespace tflite
