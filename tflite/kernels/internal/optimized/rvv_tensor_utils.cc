/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "tflite/kernels/internal/optimized/rvv_tensor_utils.h"

#if defined(__riscv_vector)

#include <stddef.h>

#include <algorithm>

#include <riscv_vector.h>

namespace tflite {
namespace tensor_utils {

namespace {

// Stack scratch paths use 64-lane buffers; vector-only paths keep full VLEN.
constexpr size_t kMaxStackVectorLanes = 64;

size_t RvvSetVlE8M1Stack(size_t avl) {
  return __riscv_vsetvl_e8m1(std::min(avl, kMaxStackVectorLanes));
}

size_t RvvSetVlE16M1Stack(size_t avl) {
  return __riscv_vsetvl_e16m1(std::min(avl, kMaxStackVectorLanes));
}

vint32m2_t RvvRoundingDivideByPOTVector(vint32m2_t value, int exponent,
                                        size_t vl) {
  if (exponent <= 0) {
    return value;
  }
  const int32_t mask = static_cast<int32_t>((INT64_C(1) << exponent) - 1);
  const vint32m2_t remainder = __riscv_vand_vx_i32m2(value, mask, vl);
  const vint32m2_t shifted = __riscv_vsra_vx_i32m2(value, exponent, vl);
  const vbool16_t negative = __riscv_vmslt_vx_i32m2_b16(value, 0, vl);
  const vint32m2_t threshold = __riscv_vadd_vv_i32m2(
      __riscv_vmv_v_x_i32m2(mask >> 1, vl),
      __riscv_vmerge_vxm_i32m2(__riscv_vmv_v_x_i32m2(0, vl), 1, negative, vl),
      vl);
  const vbool16_t round_up =
      __riscv_vmsgt_vv_i32m2_b16(remainder, threshold, vl);
  return __riscv_vadd_vv_i32m2(
      shifted,
      __riscv_vmerge_vxm_i32m2(__riscv_vmv_v_x_i32m2(0, vl), 1, round_up, vl),
      vl);
}

#if TFLITE_SINGLE_ROUNDING
// Keep the optional single-rounding contract bit-exact while avoiding the
// historical vector-to-stack-to-scalar lane loop.
vint32m2_t RvvMultiplyByQuantizedMultiplierVector(vint32m2_t value,
                                                  int32_t multiplier, int shift,
                                                  size_t vl) {
  const int64_t total_shift = 31 - shift;
  const int64_t round = INT64_C(1) << (total_shift - 1);
  const vint64m4_t product = __riscv_vwmul_vx_i64m4(value, multiplier, vl);
  const vint64m4_t rounded = __riscv_vadd_vx_i64m4(product, round, vl);
  const vint64m4_t shifted = __riscv_vsra_vx_i64m4(rounded, total_shift, vl);
  return __riscv_vncvt_x_x_w_i32m2(shifted, vl);
}
#else
vint32m2_t RvvMultiplyByQuantizedMultiplierVector(vint32m2_t value,
                                                  int32_t multiplier, int shift,
                                                  size_t vl) {
  const int left_shift = shift > 0 ? shift : 0;
  const int right_shift = shift > 0 ? 0 : -shift;
  const vint32m2_t shifted =
      left_shift == 0 ? value : __riscv_vsll_vx_i32m2(value, left_shift, vl);
  const vint64m4_t product = __riscv_vwmul_vx_i64m4(shifted, multiplier, vl);
  const vbool16_t negative = __riscv_vmslt_vx_i64m4_b16(product, 0, vl);
  const vint64m4_t nudge = __riscv_vmerge_vxm_i64m4(
      __riscv_vmv_v_x_i64m4(INT64_C(1) << 30, vl),
      1 - (INT64_C(1) << 30), negative, vl);
  const vint64m4_t nudged = __riscv_vadd_vv_i64m4(product, nudge, vl);
  const vint64m4_t quotient_floor = __riscv_vsra_vx_i64m4(nudged, 31, vl);
  const vint64m4_t remainder =
      __riscv_vand_vx_i64m4(nudged, (INT64_C(1) << 31) - 1, vl);
  const vbool16_t truncation_correction = __riscv_vmand_mm_b16(
      __riscv_vmslt_vx_i64m4_b16(nudged, 0, vl),
      __riscv_vmsne_vx_i64m4_b16(remainder, 0, vl), vl);
  const vint64m4_t quotient = __riscv_vadd_vv_i64m4(
      quotient_floor,
      __riscv_vmerge_vxm_i64m4(
          __riscv_vmv_v_x_i64m4(0, vl), 1, truncation_correction, vl),
      vl);
  vint32m2_t result = __riscv_vnsra_wx_i32m2(quotient, 0, vl);
  if (multiplier == INT32_MIN) {
    const vbool16_t special =
        __riscv_vmseq_vx_i32m2_b16(shifted, INT32_MIN, vl);
    result = __riscv_vmerge_vxm_i32m2(result, INT32_MAX, special, vl);
  }
  return RvvRoundingDivideByPOTVector(result, right_shift, vl);
}
#endif

void RvvMinMaxWithZero(const float *values, int size, float *min_value,
                       float *max_value) {
  // Seed each reduction with zero to match NeonMinMax's zero-inclusive range.
  float min_result = 0.0f;
  float max_result = 0.0f;
  for (int offset = 0; offset < size;) {
    const size_t vl = __riscv_vsetvl_e32m1(size - offset);
    const vfloat32m1_t chunk =
        __riscv_vle32_v_f32m1(values + offset, vl);
    const vfloat32m1_t min_seed =
        __riscv_vfmv_s_f_f32m1(min_result, vl);
    const vfloat32m1_t max_seed =
        __riscv_vfmv_s_f_f32m1(max_result, vl);
    min_result = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmin_vs_f32m1_f32m1(chunk, min_seed, vl));
    max_result = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmax_vs_f32m1_f32m1(chunk, max_seed, vl));
    offset += static_cast<int>(vl);
  }
  *min_value = min_result;
  *max_value = max_result;
}

float RvvDotProduct(const float *vector1, const float *vector2, int v_size) {
  if (v_size <= 0) {
    return 0.0f;
  }

  float dot_product = 0.0f;
  for (size_t col = 0; col < static_cast<size_t>(v_size);) {
    const size_t remaining = static_cast<size_t>(v_size) - col;
    const size_t vl = __riscv_vsetvl_e32m1(remaining);
    const vfloat32m1_t vector1_values =
        __riscv_vle32_v_f32m1(vector1 + col, vl);
    const vfloat32m1_t vector2_values =
        __riscv_vle32_v_f32m1(vector2 + col, vl);
    const vfloat32m1_t products =
        __riscv_vfmul_vv_f32m1(vector1_values, vector2_values, vl);
    const vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0.0f, vl);
    const vfloat32m1_t sum =
        __riscv_vfredosum_vs_f32m1_f32m1(products, zero, vl);
    dot_product += __riscv_vfmv_f_s_f32m1_f32(sum);
    col += vl;
  }
  return dot_product;
}

float RvvFloatSum(const float *vector, int v_size) {
  if (v_size <= 0) {
    return 0.0f;
  }

  float sum = 0.0f;
  for (size_t col = 0; col < static_cast<size_t>(v_size);) {
    const size_t remaining = static_cast<size_t>(v_size) - col;
    const size_t vl = __riscv_vsetvl_e32m1(remaining);
    const vfloat32m1_t values = __riscv_vle32_v_f32m1(vector + col, vl);
    const vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0.0f, vl);
    const vfloat32m1_t reduced =
        __riscv_vfredosum_vs_f32m1_f32m1(values, zero, vl);
    sum += __riscv_vfmv_f_s_f32m1_f32(reduced);
    col += vl;
  }
  return sum;
}

float RvvSquaredDifferenceSum(const float *vector, int v_size, float mean) {
  if (v_size <= 0) {
    return 0.0f;
  }

  float sum = 0.0f;
  for (size_t col = 0; col < static_cast<size_t>(v_size);) {
    const size_t remaining = static_cast<size_t>(v_size) - col;
    const size_t vl = __riscv_vsetvl_e32m1(remaining);
    const vfloat32m1_t values = __riscv_vle32_v_f32m1(vector + col, vl);
    const vfloat32m1_t differences = __riscv_vfsub_vf_f32m1(values, mean, vl);
    const vfloat32m1_t squares =
        __riscv_vfmul_vv_f32m1(differences, differences, vl);
    const vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0.0f, vl);
    const vfloat32m1_t reduced =
        __riscv_vfredosum_vs_f32m1_f32m1(squares, zero, vl);
    sum += __riscv_vfmv_f_s_f32m1_f32(reduced);
    col += vl;
  }
  return sum;
}

int32_t RvvInt8DotProduct(const int8_t *vector1, const int8_t *vector2,
                          int v_size) {
  if (v_size <= 0) {
    return 0;
  }

  int32_t dot_product = 0;
  for (size_t col = 0; col < static_cast<size_t>(v_size);) {
    const size_t remaining = static_cast<size_t>(v_size) - col;
    const size_t vl = __riscv_vsetvl_e8m1(remaining);
    const vint8m1_t vector1_values = __riscv_vle8_v_i8m1(vector1 + col, vl);
    const vint8m1_t vector2_values = __riscv_vle8_v_i8m1(vector2 + col, vl);
    const vint16m2_t products16 =
        __riscv_vwmul_vv_i16m2(vector1_values, vector2_values, vl);
    const vint32m4_t products32 = __riscv_vsext_vf2_i32m4(products16, vl);
    const vint32m1_t zero = __riscv_vmv_s_x_i32m1(0, vl);
    const vint32m1_t sum = __riscv_vredsum_vs_i32m4_i32m1(products32, zero, vl);
    dot_product += __riscv_vmv_x_s_i32m1_i32(sum);
    col += vl;
  }
  return dot_product;
}

int32_t RvvInt8Sum(const int8_t *vector, int v_size) {
  if (v_size <= 0) {
    return 0;
  }

  int32_t sum = 0;
  for (size_t col = 0; col < static_cast<size_t>(v_size);) {
    const size_t remaining = static_cast<size_t>(v_size) - col;
    const size_t vl = __riscv_vsetvl_e8m1(remaining);
    const vint8m1_t values = __riscv_vle8_v_i8m1(vector + col, vl);
    const vint16m2_t values16 = __riscv_vsext_vf2_i16m2(values, vl);
    const vint32m4_t values32 = __riscv_vsext_vf2_i32m4(values16, vl);
    const vint32m1_t zero = __riscv_vmv_s_x_i32m1(0, vl);
    const vint32m1_t reduced =
        __riscv_vredsum_vs_i32m4_i32m1(values32, zero, vl);
    sum += __riscv_vmv_x_s_i32m1_i32(reduced);
    col += vl;
  }
  return sum;
}

int32_t RvvSaturatingRoundingDoublingHighMul(int32_t a, int32_t b) {
  if (a == b && a == INT32_MIN) {
    return INT32_MAX;
  }
  const int64_t product = static_cast<int64_t>(a) * b;
  const int64_t nudge =
      product >= 0 ? (INT64_C(1) << 30) : (1 - (INT64_C(1) << 30));
  return static_cast<int32_t>((product + nudge) / (INT64_C(1) << 31));
}

int32_t RvvRoundingDivideByPOT(int32_t value, int exponent) {
  if (exponent == 0) {
    return value;
  }
  const int32_t mask = static_cast<int32_t>((INT64_C(1) << exponent) - 1);
  const int32_t remainder = value & mask;
  const int32_t threshold = (mask >> 1) + (value < 0 ? 1 : 0);
  return (value >> exponent) + (remainder > threshold ? 1 : 0);
}

int32_t RvvMultiplyByQuantizedMultiplier(int32_t value,
                                         int32_t quantized_multiplier,
                                         int shift) {
#if TFLITE_SINGLE_ROUNDING
  const int right_shift = shift < -1 ? shift : -1;
  const int left_shift = shift - right_shift;
  const int64_t total_shift = 31 - right_shift;
  const int64_t round = INT64_C(1) << (total_shift - 1);
  const int64_t product = static_cast<int64_t>(value) * quantized_multiplier *
                          (INT64_C(1) << left_shift);
  return static_cast<int32_t>((product + round) >> total_shift);
#else
  const int left_shift = shift > 0 ? shift : 0;
  const int right_shift = shift > 0 ? 0 : -shift;
  const int32_t shifted_value = value * (1 << left_shift);
  return RvvRoundingDivideByPOT(
      RvvSaturatingRoundingDoublingHighMul(shifted_value, quantized_multiplier),
      right_shift);
#endif
}

int16_t RvvRescaleConstant16(int32_t value) {
  return static_cast<int16_t>(RvvRoundingDivideByPOT(value, 16));
}
// The fixed-point activation helpers below keep the scalar arithmetic
// contract, but carry a whole RVV register through the same operations.  The
// previous implementation loaded a vector and immediately spilled it to a
// lane array; that made Logistic/Tanh vector entry points scalar in practice.
vint16m2_t RvvRoundingDivideByPOT16Vector(vint16m2_t value, int exponent,
                                          size_t vl) {
  if (exponent <= 0) {
    return value;
  }
  const int32_t mask = (1 << exponent) - 1;
  const vint32m4_t value32 = __riscv_vsext_vf2_i32m4(value, vl);
  const vint32m4_t remainder = __riscv_vand_vx_i32m4(value32, mask, vl);
  const vint32m4_t shifted = __riscv_vsra_vx_i32m4(value32, exponent, vl);
  const vint32m4_t sign = __riscv_vsra_vx_i32m4(value32, 31, vl);
  const vbool8_t negative =
      __riscv_vmsne_vx_i32m4_b8(sign, 0, vl);
  const vint32m4_t threshold = __riscv_vadd_vv_i32m4(
      __riscv_vmv_v_x_i32m4(mask >> 1, vl),
      __riscv_vmerge_vxm_i32m4(__riscv_vmv_v_x_i32m4(0, vl), 1, negative, vl),
      vl);
  const vbool8_t round_up =
      __riscv_vmsgt_vv_i32m4_b8(remainder, threshold, vl);
  const vint32m4_t rounded = __riscv_vadd_vv_i32m4(
      shifted,
      __riscv_vmerge_vvm_i32m4(__riscv_vmv_v_x_i32m4(0, vl),
                               __riscv_vmv_v_x_i32m4(1, vl), round_up, vl),
      vl);
  return __riscv_vncvt_x_x_w_i16m2(rounded, vl);
}

vint16m2_t RvvSaturatingRoundingDoublingHighMul16Vector(
    vint16m2_t a, vint16m2_t b, size_t vl) {
  const vint32m4_t product = __riscv_vwmul_vv_i32m4(a, b, vl);
  const vint32m4_t sign = __riscv_vsra_vx_i32m4(product, 31, vl);
  const vbool8_t negative = __riscv_vmsne_vx_i32m4_b8(sign, 0, vl);
  const vint32m4_t nudge = __riscv_vmerge_vxm_i32m4(
      __riscv_vmv_v_x_i32m4(1 << 14, vl), 1 - (1 << 14), negative, vl);
  const vint32m4_t nudged = __riscv_vadd_vv_i32m4(product, nudge, vl);
  const vint32m4_t quotient_floor = __riscv_vsra_vx_i32m4(nudged, 15, vl);
  const vint32m4_t remainder = __riscv_vand_vx_i32m4(nudged, 0x7fff, vl);
  const vbool8_t truncation_correction = __riscv_vmand_mm_b8(
      __riscv_vmsne_vx_i32m4_b8(sign, 0, vl),
      __riscv_vmsne_vx_i32m4_b8(remainder, 0, vl), vl);
  const vint32m4_t quotient = __riscv_vadd_vv_i32m4(
      quotient_floor,
      __riscv_vmerge_vvm_i32m4(__riscv_vmv_v_x_i32m4(0, vl),
                               __riscv_vmv_v_x_i32m4(1, vl),
                               truncation_correction, vl),
      vl);
  vint16m2_t result = __riscv_vncvt_x_x_w_i16m2(
      __riscv_vmax_vx_i32m4(
          __riscv_vmin_vx_i32m4(quotient, INT16_MAX, vl), INT16_MIN, vl),
      vl);
  const vbool8_t min_product = __riscv_vmand_mm_b8(
      __riscv_vmseq_vx_i16m2_b8(a, INT16_MIN, vl),
      __riscv_vmseq_vx_i16m2_b8(b, INT16_MIN, vl), vl);
  return __riscv_vmerge_vxm_i16m2(result, INT16_MAX, min_product, vl);
}

vint16m2_t RvvFixedPointMul16Vector(vint16m2_t value, int16_t multiplier,
                                   size_t vl) {
  return RvvSaturatingRoundingDoublingHighMul16Vector(
      value, __riscv_vmv_v_x_i16m2(multiplier, vl), vl);
}

vint16m2_t RvvFixedPointMul16Vector(vint16m2_t a, vint16m2_t b, size_t vl) {
  return RvvSaturatingRoundingDoublingHighMul16Vector(a, b, vl);
}

vint16m2_t RvvSaturatingRoundingMultiplyByPOT16Vector(vint16m2_t value,
                                                      int exponent,
                                                      size_t vl) {
  if (exponent == 0) {
    return value;
  }
  if (exponent < 0) {
    return RvvRoundingDivideByPOT16Vector(value, -exponent, vl);
  }
  const vint32m4_t widened = __riscv_vsext_vf2_i32m4(value, vl);
  const vint32m4_t shifted = __riscv_vsll_vx_i32m4(widened, exponent, vl);
  const vint32m4_t clamped = __riscv_vmax_vx_i32m4(
      __riscv_vmin_vx_i32m4(shifted, INT16_MAX, vl), INT16_MIN, vl);
  return __riscv_vncvt_x_x_w_i16m2(clamped, vl);
}

vint16m2_t RvvRescale16Vector(vint16m2_t value, int source_integer_bits,
                              int destination_integer_bits, size_t vl) {
  return RvvSaturatingRoundingMultiplyByPOT16Vector(
      value, source_integer_bits - destination_integer_bits, vl);
}

vint16m2_t RvvSaturatingAdd16Vector(vint16m2_t a, vint16m2_t b, size_t vl) {
  const vint32m4_t sum = __riscv_vadd_vv_i32m4(
      __riscv_vsext_vf2_i32m4(a, vl), __riscv_vsext_vf2_i32m4(b, vl), vl);
  return __riscv_vncvt_x_x_w_i16m2(
      __riscv_vmax_vx_i32m4(
          __riscv_vmin_vx_i32m4(sum, INT16_MAX, vl), INT16_MIN, vl),
      vl);
}

vint16m2_t RvvRoundingHalfSum16Vector(vint16m2_t a, vint16m2_t b,
                                     size_t vl) {
  const vint32m4_t sum = __riscv_vadd_vv_i32m4(
      __riscv_vsext_vf2_i32m4(a, vl), __riscv_vsext_vf2_i32m4(b, vl), vl);
  const vint32m4_t half = __riscv_vsra_vx_i32m4(sum, 1, vl);
  const vbool8_t odd = __riscv_vmsne_vx_i32m4_b8(
      __riscv_vand_vx_i32m4(sum, 1, vl), 0, vl);
  const vbool8_t nonnegative = __riscv_vmseq_vx_i32m4_b8(
      __riscv_vsra_vx_i32m4(sum, 31, vl), 0, vl);
  const vbool8_t round_up = __riscv_vmand_mm_b8(odd, nonnegative, vl);
  return __riscv_vncvt_x_x_w_i16m2(
      __riscv_vadd_vv_i32m4(
          half,
          __riscv_vmerge_vvm_i32m4(__riscv_vmv_v_x_i32m4(0, vl),
                                   __riscv_vmv_v_x_i32m4(1, vl), round_up, vl),
          vl),
      vl);
}

vint16m2_t RvvAdd16Vector(vint16m2_t a, vint16m2_t b, size_t vl) {
  return __riscv_vadd_vv_i16m2(a, b, vl);
}

vint16m2_t RvvExpOnInterval16Vector(vint16m2_t a, size_t vl) {
  const int16_t constant_term = RvvRescaleConstant16(1895147668);
  const int16_t constant_one_third = RvvRescaleConstant16(715827883);
  const vint16m2_t x = __riscv_vadd_vx_i16m2(a, 1 << 12, vl);
  const vint16m2_t x2 = RvvFixedPointMul16Vector(x, x, vl);
  const vint16m2_t x3 = RvvFixedPointMul16Vector(x2, x, vl);
  const vint16m2_t x4 = RvvFixedPointMul16Vector(x2, x2, vl);
  const vint16m2_t x4_over_4 =
      RvvSaturatingRoundingMultiplyByPOT16Vector(x4, -2, vl);
  const vint16m2_t polynomial = RvvSaturatingRoundingMultiplyByPOT16Vector(
      RvvAdd16Vector(
          RvvFixedPointMul16Vector(
              RvvAdd16Vector(x4_over_4, x3, vl), constant_one_third, vl),
          x2, vl),
      -1, vl);
  return RvvSaturatingAdd16Vector(
      __riscv_vmv_v_x_i16m2(constant_term, vl),
      RvvFixedPointMul16Vector(
          __riscv_vmv_v_x_i16m2(constant_term, vl),
          RvvAdd16Vector(x, polynomial, vl), vl),
      vl);
}

vint16m2_t RvvExpOnNegative16Vector(vint16m2_t a, int integer_bits,
                                   size_t vl) {
  const int fractional_bits = 15 - integer_bits;
  const int16_t one_quarter = static_cast<int16_t>(1 << (fractional_bits - 2));
  const int16_t mask = static_cast<int16_t>(one_quarter - 1);
  const vint16m2_t a_mod = __riscv_vsub_vx_i16m2(
      __riscv_vand_vx_i16m2(a, mask, vl), one_quarter, vl);
  vint16m2_t result = RvvExpOnInterval16Vector(
      RvvRescale16Vector(a_mod, integer_bits, 0, vl), vl);
  const vint16m2_t remainder = __riscv_vsub_vv_i16m2(a_mod, a, vl);
  constexpr int kExponents[] = {-2, -1, 0, 1, 2, 3, 4};
  constexpr int32_t kMultipliers[] = {1672461947, 1302514674, 790015084,
                                      290630308, 39332535, 720401, 242};
  for (int i = 0; i < 7; ++i) {
    if (integer_bits > kExponents[i]) {
      const int shift = fractional_bits + kExponents[i];
      const int16_t bit = static_cast<int16_t>(1 << shift);
      const vbool8_t selected = __riscv_vmsne_vx_i16m2_b8(
          __riscv_vand_vx_i16m2(remainder, bit, vl), 0, vl);
      const vint16m2_t multiplied = RvvFixedPointMul16Vector(
          result, RvvRescaleConstant16(kMultipliers[i]), vl);
      result = __riscv_vmerge_vvm_i16m2(result, multiplied, selected, vl);
    }
  }
  if (integer_bits > 5) {
    const int clamp_bits = 36 - integer_bits;
    const int16_t clamp =
        RvvRescaleConstant16(-(1 << clamp_bits));
    const vbool8_t clamped = __riscv_vmslt_vx_i16m2_b8(a, clamp, vl);
    result = __riscv_vmerge_vvm_i16m2(
        result, __riscv_vmv_v_x_i16m2(0, vl), clamped, vl);
  }
  const vbool8_t zero = __riscv_vmseq_vx_i16m2_b8(a, 0, vl);
  return __riscv_vmerge_vxm_i16m2(result, INT16_MAX, zero, vl);
}

vint16m2_t RvvOneOverOnePlusX16Vector(vint16m2_t a, size_t vl) {
  const vint16m2_t half_denominator = RvvRoundingHalfSum16Vector(
      a, __riscv_vmv_v_x_i16m2(INT16_MAX, vl), vl);
  const int16_t constant_48_over_17 = RvvRescaleConstant16(1515870810);
  const int16_t constant_neg_32_over_17 = RvvRescaleConstant16(-1010580540);
  vint16m2_t x = __riscv_vadd_vx_i16m2(
      RvvFixedPointMul16Vector(half_denominator, constant_neg_32_over_17, vl),
      constant_48_over_17, vl);
  for (int i = 0; i < 3; ++i) {
    const vint16m2_t half_denominator_times_x =
        RvvFixedPointMul16Vector(half_denominator, x, vl);
    const vint16m2_t one_minus_half_denominator_times_x = __riscv_vsub_vv_i16m2(
        __riscv_vmv_v_x_i16m2(1 << 13, vl), half_denominator_times_x, vl);
    const vint16m2_t correction = RvvRescale16Vector(
        RvvFixedPointMul16Vector(x, one_minus_half_denominator_times_x, vl),
        4, 2, vl);
    x = __riscv_vadd_vv_i16m2(x, correction, vl);
  }
  return RvvRescale16Vector(x, 1, 0, vl);
}

vint16m2_t RvvOneMinusXOverOnePlusX16Vector(vint16m2_t a, size_t vl) {
  const vint16m2_t half_denominator = RvvRoundingHalfSum16Vector(
      a, __riscv_vmv_v_x_i16m2(INT16_MAX, vl), vl);
  const int16_t constant_48_over_17 = RvvRescaleConstant16(1515870810);
  const int16_t constant_neg_32_over_17 = RvvRescaleConstant16(-1010580540);
  vint16m2_t x = __riscv_vadd_vx_i16m2(
      RvvFixedPointMul16Vector(half_denominator, constant_neg_32_over_17, vl),
      constant_48_over_17, vl);
  for (int i = 0; i < 3; ++i) {
    const vint16m2_t half_denominator_times_x =
        RvvFixedPointMul16Vector(half_denominator, x, vl);
    const vint16m2_t one_minus_half_denominator_times_x = __riscv_vsub_vv_i16m2(
        __riscv_vmv_v_x_i16m2(1 << 13, vl), half_denominator_times_x, vl);
    const vint16m2_t correction = RvvRescale16Vector(
        RvvFixedPointMul16Vector(x, one_minus_half_denominator_times_x, vl),
        4, 2, vl);
    x = __riscv_vadd_vv_i16m2(x, correction, vl);
  }
  return RvvRescale16Vector(
      __riscv_vsub_vx_i16m2(x, 1 << 13, vl), 2, 0, vl);
}

vint16m2_t RvvLogistic16Vector(vint16m2_t input, int integer_bits,
                              size_t vl) {
  const vint16m2_t zero_value = __riscv_vmv_v_x_i16m2(0, vl);
  const vbool8_t negative = __riscv_vmslt_vx_i16m2_b8(input, 0, vl);
  const vint16m2_t absolute = __riscv_vmerge_vvm_i16m2(
      input, __riscv_vsub_vv_i16m2(zero_value, input, vl), negative, vl);
  const vint16m2_t positive = RvvOneOverOnePlusX16Vector(
      RvvExpOnNegative16Vector(__riscv_vsub_vv_i16m2(zero_value, absolute, vl),
                               integer_bits, vl),
      vl);
  const vint16m2_t negative_result = __riscv_vsub_vv_i16m2(
      __riscv_vmv_v_x_i16m2(INT16_MAX, vl), positive, vl);
  const vbool8_t positive_input =
      __riscv_vmsgt_vx_i16m2_b8(input, 0, vl);
  vint16m2_t result = __riscv_vmerge_vvm_i16m2(
      negative_result, positive, positive_input, vl);
  const vbool8_t zero_input = __riscv_vmseq_vx_i16m2_b8(input, 0, vl);
  return __riscv_vmerge_vxm_i16m2(
      result, RvvRescaleConstant16(1 << 30), zero_input, vl);
}

vint16m2_t RvvTanh16Vector(vint16m2_t input, int integer_bits, size_t vl) {
  const vint16m2_t zero_value = __riscv_vmv_v_x_i16m2(0, vl);
  const vbool8_t negative = __riscv_vmslt_vx_i16m2_b8(input, 0, vl);
  const vint16m2_t absolute = __riscv_vmerge_vvm_i16m2(
      input, __riscv_vsub_vv_i16m2(zero_value, input, vl), negative, vl);
  const vint16m2_t positive = RvvOneMinusXOverOnePlusX16Vector(
      RvvExpOnNegative16Vector(__riscv_vsub_vv_i16m2(zero_value, absolute, vl),
                               integer_bits + 1, vl),
      vl);
  const vint16m2_t negative_result =
      __riscv_vsub_vv_i16m2(zero_value, positive, vl);
  vint16m2_t result = __riscv_vmerge_vvm_i16m2(
      positive, negative_result, negative, vl);
  const vbool8_t zero_input = __riscv_vmseq_vx_i16m2_b8(input, 0, vl);
  return __riscv_vmerge_vxm_i16m2(result, 0, zero_input, vl);
}

int RvvCountLeadingZeros(uint32_t value) {
  return value == 0 ? 32 : __builtin_clz(value);
}

int32_t RvvSaturatingRoundingMultiplyByPOT32(int32_t value, int exponent) {
  if (exponent == 0) {
    return value;
  }
  if (exponent < 0) {
    return RvvRoundingDivideByPOT(value, -exponent);
  }
  const int64_t shifted =
      static_cast<int64_t>(value) * (INT64_C(1) << exponent);
  if (shifted < INT32_MIN) {
    return INT32_MIN;
  }
  if (shifted > INT32_MAX) {
    return INT32_MAX;
  }
  return static_cast<int32_t>(shifted);
}

int32_t RvvFixedPointMul32(int32_t a, int32_t b) {
  return RvvSaturatingRoundingDoublingHighMul(a, b);
}

void RvvGetInvSqrtQuantizedMultiplierExp(int32_t input, int reverse_shift,
                                         int32_t *output_inv_sqrt,
                                         int *output_shift) {
  if (input <= 1) {
    *output_inv_sqrt = INT32_MAX;
    *output_shift = 0;
    return;
  }
  *output_shift = 11;
  while (input >= (1 << 29)) {
    input /= 4;
    ++*output_shift;
  }
  const unsigned max_left_shift_bits =
      static_cast<unsigned>(
          RvvCountLeadingZeros(static_cast<uint32_t>(input))) -
      1;
  const unsigned max_left_shift_bit_pairs = max_left_shift_bits / 2;
  const unsigned left_shift_bit_pairs = max_left_shift_bit_pairs - 1;
  *output_shift -= static_cast<int>(left_shift_bit_pairs);
  input <<= 2 * left_shift_bit_pairs;

  const int32_t fixedpoint_input = input >> 1;
  const int32_t fixedpoint_half_input =
      RvvSaturatingRoundingMultiplyByPOT32(fixedpoint_input, -1);
  const int32_t fixedpoint_half_three = (1 << 28) + (1 << 27);
  int32_t x = 1 << 28;
  for (int i = 0; i < 5; ++i) {
    const int32_t x2 = RvvFixedPointMul32(x, x);
    const int32_t x3 = RvvFixedPointMul32(x2, x);
    const int32_t x3_rescaled = RvvSaturatingRoundingMultiplyByPOT32(x3, 6);
    const int32_t term1 = RvvFixedPointMul32(fixedpoint_half_three, x);
    const int32_t term2 =
        RvvFixedPointMul32(fixedpoint_half_input, x3_rescaled);
    x = RvvSaturatingRoundingMultiplyByPOT32(term1 - term2, 3);
  }
  x = RvvFixedPointMul32(x, 1518500250);
  *output_inv_sqrt = x;
  if (*output_shift < 0) {
    *output_inv_sqrt =
        RvvSaturatingRoundingMultiplyByPOT32(*output_inv_sqrt, -*output_shift);
    *output_shift = 0;
  }
  *output_shift *= reverse_shift;
}

template <typename Output>
void RvvMatrixBatchVectorMultiplyAccumulateImpl(
    const int8_t *input, const int32_t *bias,
    const int8_t *input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    Output *output) {
  const int32_t output_min = sizeof(Output) == sizeof(int8_t) ? -128 : -32768;
  const int32_t output_max = sizeof(Output) == sizeof(int8_t) ? 127 : 32767;
  for (int32_t batch = 0; batch < n_batch; ++batch) {
    const int8_t *input_in_batch = input + batch * n_input;
    for (int32_t row = 0; row < n_output; ++row) {
      const int32_t bias_value = bias != nullptr ? bias[row] : 0;
      int32_t value =
          bias_value + RvvInt8DotProduct(input_to_gate_weights + row * n_input,
                                         input_in_batch, n_input);
      value = RvvMultiplyByQuantizedMultiplier(value, multiplier, shift);
      value += output_zp + output[batch * n_output + row];
      if (value < output_min) {
        value = output_min;
      } else if (value > output_max) {
        value = output_max;
      }
      output[batch * n_output + row] = static_cast<Output>(value);
    }
  }
}

void RvvQuantizeFloatRange(const float *values, int size, float scale,
                           int32_t offset, int32_t quantized_min,
                           int32_t quantized_max, int8_t *quantized_values) {
  for (size_t i = 0; i < static_cast<size_t>(size);) {
    const size_t remaining = static_cast<size_t>(size) - i;
    const size_t vl = __riscv_vsetvl_e32m4(remaining);
    const vfloat32m4_t input = __riscv_vle32_v_f32m4(values + i, vl);
    const vfloat32m4_t scaled = __riscv_vfmul_vf_f32m4(input, scale, vl);
    const vfloat32m4_t half = __riscv_vfmv_v_f_f32m4(0.5f, vl);
    const vfloat32m4_t signed_half =
        __riscv_vfsgnj_vv_f32m4(half, scaled, vl);
    const vfloat32m4_t rounded_input =
        __riscv_vfadd_vv_f32m4(scaled, signed_half, vl);
    vint32m4_t rounded = __riscv_vfcvt_rtz_x_f_v_i32m4(rounded_input, vl);
    rounded = __riscv_vadd_vx_i32m4(rounded, offset, vl);
    rounded = __riscv_vmax_vx_i32m4(rounded, quantized_min, vl);
    rounded = __riscv_vmin_vx_i32m4(rounded, quantized_max, vl);
    const vint16m2_t rounded16 =
        __riscv_vncvt_x_x_w_i16m2(rounded, vl);
    const vint8m1_t quantized =
        __riscv_vncvt_x_x_w_i8m1(rounded16, vl);
    __riscv_vse8_v_i8m1(quantized_values + i, quantized, vl);
    i += vl;
  }
}

} // namespace

void RvvApplyLayerNorm(const int16_t *input, const int16_t *layer_norm_weights,
                       const int32_t *bias, int32_t layer_norm_scale_a,
                       int32_t layer_norm_scale_b, int32_t variance_limit,
                       int n_batch, int n_input, int16_t *output) {
  if (n_batch <= 0 || n_input <= 0) {
    return;
  }

  const int32_t temp = 1048576 / n_input;
  for (int batch = 0; batch < n_batch; ++batch) {
    const int16_t *input_in_batch = input + batch * n_input;
    int64_t sum = 0;
    int64_t sum_sq = 0;
    for (size_t i = 0; i < static_cast<size_t>(n_input);) {
      const size_t remaining = static_cast<size_t>(n_input) - i;
      const size_t vl = __riscv_vsetvl_e16m1(remaining);
      const vint16m1_t values = __riscv_vle16_v_i16m1(input_in_batch + i, vl);
      const vint32m2_t values32 = __riscv_vsext_vf2_i32m2(values, vl);
      const vint32m1_t zero = __riscv_vmv_s_x_i32m1(0, vl);
      const vint32m1_t sum_vector =
          __riscv_vredsum_vs_i32m2_i32m1(values32, zero, vl);
      const vint32m2_t squares = __riscv_vwmul_vv_i32m2(values, values, vl);
      const vint32m1_t sum_sq_vector =
          __riscv_vredsum_vs_i32m2_i32m1(squares, zero, vl);
      sum += __riscv_vmv_x_s_i32m1_i32(sum_vector);
      sum_sq += __riscv_vmv_x_s_i32m1_i32(sum_sq_vector);
      i += vl;
    }

    const int32_t mean = static_cast<int32_t>(sum * 1024 / n_input);
    const int64_t variance = sum_sq * temp - static_cast<int64_t>(mean) * mean;
    int32_t variance2 = static_cast<int32_t>(variance / 1048576);
    if (variance2 < 1) {
      variance2 = variance_limit;
    }
    int32_t stddev_inverse_a;
    int stddev_inverse_b;
    RvvGetInvSqrtQuantizedMultiplierExp(variance2, -1, &stddev_inverse_a,
                                        &stddev_inverse_b);

    int16_t *output_in_batch = output + batch * n_input;
    for (size_t i = 0; i < static_cast<size_t>(n_input);) {
      const size_t remaining = static_cast<size_t>(n_input) - i;
      const size_t vl = RvvSetVlE16M1Stack(remaining);
      const vint16m1_t values = __riscv_vle16_v_i16m1(input_in_batch + i, vl);
      const vint32m2_t values32 = __riscv_vsext_vf2_i32m2(values, vl);
      const vint32m2_t shifted = __riscv_vsub_vx_i32m2(
          __riscv_vsll_vx_i32m2(values32, 10, vl), mean, vl);
      const vint32m2_t rescaled = RvvMultiplyByQuantizedMultiplierVector(
          shifted, stddev_inverse_a, stddev_inverse_b, vl);
      const vint16m1_t weights =
          __riscv_vle16_v_i16m1(layer_norm_weights + i, vl);
      const vint32m2_t weights32 = __riscv_vsext_vf2_i32m2(weights, vl);
      const vint32m2_t bias_values = __riscv_vle32_v_i32m2(bias + i, vl);
      const vint64m4_t val3 = __riscv_vadd_vv_i64m4(
          __riscv_vwmul_vv_i64m4(rescaled, weights32, vl),
          __riscv_vsext_vf2_i64m4(bias_values, vl), vl);
      const vbool16_t positive = __riscv_vmsgt_vx_i64m4_b16(val3, 0, vl);
      const vint64m4_t nudge = __riscv_vmerge_vxm_i64m4(
          __riscv_vmv_v_x_i64m4(-512, vl), 512, positive, vl);
      const vint64m4_t nudged = __riscv_vadd_vv_i64m4(val3, nudge, vl);
      const vint64m4_t quotient_floor =
          __riscv_vsra_vx_i64m4(nudged, 10, vl);
      const vint64m4_t remainder =
          __riscv_vand_vx_i64m4(nudged, 1023, vl);
      const vbool16_t truncation_correction = __riscv_vmand_mm_b16(
          __riscv_vmslt_vx_i64m4_b16(nudged, 0, vl),
          __riscv_vmsne_vx_i64m4_b16(remainder, 0, vl), vl);
      const vint64m4_t quotient = __riscv_vadd_vv_i64m4(
          quotient_floor,
          __riscv_vmerge_vvm_i64m4(__riscv_vmv_v_x_i64m4(0, vl),
                                   __riscv_vmv_v_x_i64m4(1, vl),
                                   truncation_correction, vl),
          vl);
      const vint32m2_t val4 = __riscv_vncvt_x_x_w_i32m2(
          __riscv_vmax_vx_i64m4(
              __riscv_vmin_vx_i64m4(quotient, INT32_MAX, vl), INT32_MIN, vl),
          vl);
      const vint32m2_t val5 = RvvMultiplyByQuantizedMultiplierVector(
          val4, layer_norm_scale_a, layer_norm_scale_b + 12, vl);
      const vint32m2_t clamped = __riscv_vmax_vx_i32m2(
          __riscv_vmin_vx_i32m2(val5, INT16_MAX, vl), INT16_MIN, vl);
      __riscv_vse16_v_i16m1(output_in_batch + i,
                            __riscv_vncvt_x_x_w_i16m1(clamped, vl), vl);
      i += vl;
    }
  }
}

void RvvApplySigmoid(const int16_t *input, int32_t n_batch, int32_t n_input,
                     int16_t *output) {
  if (n_batch <= 0 || n_input <= 0) {
    return;
  }
  const size_t total = static_cast<size_t>(n_batch) * n_input;
  for (size_t i = 0; i < total;) {
    const size_t remaining = total - i;
    const size_t vl = __riscv_vsetvl_e16m2(remaining);
    const vint16m2_t values = __riscv_vle16_v_i16m2(input + i, vl);
        __riscv_vse16_v_i16m2(output + i,
                              RvvLogistic16Vector(values, 3, vl), vl);
    i += vl;
  }
}

void RvvApplyTanh(int32_t integer_bits, const int16_t *input, int32_t n_batch,
                  int32_t n_input, int16_t *output) {
  if (integer_bits < 0 || integer_bits > 6 || n_batch <= 0 || n_input <= 0) {
    return;
  }
  const size_t total = static_cast<size_t>(n_batch) * n_input;
  for (size_t i = 0; i < total;) {
    const size_t remaining = total - i;
    const size_t vl = __riscv_vsetvl_e16m2(remaining);
    const vint16m2_t values = __riscv_vle16_v_i16m2(input + i, vl);
    __riscv_vse16_v_i16m2(
        output + i, RvvTanh16Vector(values, integer_bits, vl), vl);
    i += vl;
  }
}

void RvvApplyTanhWithInputLeftShift(int32_t integer_bits,
                                    int32_t input_left_shift,
                                    const int16_t *input, int32_t n_batch,
                                    int32_t n_input, int16_t *output) {
  if (input_left_shift < 0 || input_left_shift > 1 || integer_bits < 0 ||
      integer_bits > 6 || n_batch <= 0 || n_input <= 0) {
    return;
  }
  const size_t total = static_cast<size_t>(n_batch) * n_input;
  for (size_t i = 0; i < total;) {
    const size_t remaining = total - i;
    const size_t vl = __riscv_vsetvl_e16m2(remaining);
    const vint16m2_t values = __riscv_vle16_v_i16m2(input + i, vl);
    const vint16m2_t shifted =
        RvvSaturatingRoundingMultiplyByPOT16Vector(values, input_left_shift,
                                                   vl);
    __riscv_vse16_v_i16m2(output + i,
                          RvvTanh16Vector(shifted, integer_bits, vl), vl);
    i += vl;
  }
}

void RvvTanh16BitPrecisionUint8(const uint8_t *input, int size,
                                int32_t input_zero_point,
                                int32_t input_range_radius,
                                int16_t input_multiplier,
                                int16_t input_left_shift, uint8_t *output) {
  for (int offset = 0; offset < size;) {
    const size_t vl = RvvSetVlE8M1Stack(size - offset);
    const vint16m2_t centered = __riscv_vsub_vx_i16m2(
        __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(
            __riscv_vle8_v_u8m1(input + offset, vl), vl)),
        input_zero_point, vl);
    const vint16m2_t rescaled = RvvFixedPointMul16Vector(
        RvvSaturatingRoundingMultiplyByPOT16Vector(centered, input_left_shift,
                                                   vl),
        input_multiplier, vl);
    vint16m2_t result = __riscv_vadd_vx_i16m2(
        RvvRoundingDivideByPOT16Vector(RvvTanh16Vector(rescaled, 4, vl), 8,
                                       vl),
        128, vl);
    result = __riscv_vmax_vx_i16m2(
        __riscv_vmin_vx_i16m2(result, 255, vl), 0, vl);
    const vbool8_t below =
        __riscv_vmslt_vx_i16m2_b8(centered, -input_range_radius, vl);
    const vbool8_t above =
        __riscv_vmsgt_vx_i16m2_b8(centered, input_range_radius, vl);
    result = __riscv_vmerge_vxm_i16m2(result, 0, below, vl);
    result = __riscv_vmerge_vxm_i16m2(result, 255, above, vl);
    const vuint8m1_t narrowed = __riscv_vncvt_x_x_w_u8m1(
        __riscv_vreinterpret_v_i16m2_u16m2(result), vl);
    __riscv_vse8_v_u8m1(output + offset, narrowed, vl);
    offset += static_cast<int>(vl);
  }
}

void RvvTanh16BitPrecisionInt8(const int8_t *input, int size,
                               int32_t input_zero_point,
                               int32_t input_range_radius,
                               int16_t input_multiplier,
                               int16_t input_left_shift, int8_t *output) {
  for (int offset = 0; offset < size;) {
    const size_t vl = RvvSetVlE8M1Stack(size - offset);
    const vint16m2_t centered = __riscv_vsub_vx_i16m2(
        __riscv_vsext_vf2_i16m2(__riscv_vle8_v_i8m1(input + offset, vl), vl),
        input_zero_point, vl);
    const vint16m2_t rescaled = RvvFixedPointMul16Vector(
        RvvSaturatingRoundingMultiplyByPOT16Vector(centered, input_left_shift,
                                                   vl),
        input_multiplier, vl);
    vint16m2_t result = RvvRoundingDivideByPOT16Vector(
        RvvTanh16Vector(rescaled, 4, vl), 8, vl);
    result = __riscv_vmax_vx_i16m2(
        __riscv_vmin_vx_i16m2(result, 127, vl), -128, vl);
    const vbool8_t below =
        __riscv_vmslt_vx_i16m2_b8(centered, -input_range_radius + 1, vl);
    const vbool8_t above =
        __riscv_vmsgt_vx_i16m2_b8(centered, input_range_radius - 1, vl);
    // Apply the upper endpoint first so an invalid zero radius preserves the
    // scalar path's lower-endpoint precedence at centered == 0.
    result = __riscv_vmerge_vxm_i16m2(result, 127, above, vl);
    result = __riscv_vmerge_vxm_i16m2(result, -128, below, vl);
    __riscv_vse8_v_i8m1(
        output + offset, __riscv_vncvt_x_x_w_i8m1(result, vl), vl);
    offset += static_cast<int>(vl);
  }
}

void RvvLogistic16BitPrecisionUint8(const uint8_t *input, int size,
                                    int32_t input_zero_point,
                                    int32_t input_range_radius,
                                    int32_t input_multiplier,
                                    int16_t input_left_shift, uint8_t *output) {
  for (int offset = 0; offset < size;) {
    const size_t vl = RvvSetVlE8M1Stack(size - offset);
    const vint16m2_t centered = __riscv_vsub_vx_i16m2(
        __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(
            __riscv_vle8_v_u8m1(input + offset, vl), vl)),
        input_zero_point, vl);
    const vint16m2_t rescaled = RvvFixedPointMul16Vector(
        RvvSaturatingRoundingMultiplyByPOT16Vector(centered, input_left_shift,
                                                   vl),
        static_cast<int16_t>(input_multiplier), vl);
    vint16m2_t result = __riscv_vadd_vx_i16m2(
        RvvRoundingDivideByPOT16Vector(
            RvvLogistic16Vector(rescaled, 4, vl), 7, vl),
        0, vl);
    result = __riscv_vmax_vx_i16m2(
        __riscv_vmin_vx_i16m2(result, 255, vl), 0, vl);
    const vbool8_t below =
        __riscv_vmslt_vx_i16m2_b8(centered, -input_range_radius, vl);
    const vbool8_t above =
        __riscv_vmsgt_vx_i16m2_b8(centered, input_range_radius, vl);
    result = __riscv_vmerge_vxm_i16m2(result, 0, below, vl);
    result = __riscv_vmerge_vxm_i16m2(result, 255, above, vl);
    __riscv_vse8_v_u8m1(
        output + offset,
        __riscv_vncvt_x_x_w_u8m1(__riscv_vreinterpret_v_i16m2_u16m2(result),
                                 vl),
        vl);
    offset += static_cast<int>(vl);
  }
}

void RvvLogistic16BitPrecisionInt8(const int8_t *input, int size,
                                   int32_t input_zero_point,
                                   int32_t input_range_radius,
                                   int32_t input_multiplier,
                                   int16_t input_left_shift, int8_t *output) {
  for (int offset = 0; offset < size;) {
    const size_t vl = RvvSetVlE8M1Stack(size - offset);
    const vint16m2_t centered = __riscv_vsub_vx_i16m2(
        __riscv_vsext_vf2_i16m2(__riscv_vle8_v_i8m1(input + offset, vl), vl),
        input_zero_point, vl);
    const vint16m2_t rescaled = RvvFixedPointMul16Vector(
        RvvSaturatingRoundingMultiplyByPOT16Vector(centered, input_left_shift,
                                                   vl),
        static_cast<int16_t>(input_multiplier), vl);
    vint16m2_t result = __riscv_vsub_vx_i16m2(
        RvvRoundingDivideByPOT16Vector(
            RvvLogistic16Vector(rescaled, 4, vl), 7, vl),
        128, vl);
    result = __riscv_vmax_vx_i16m2(
        __riscv_vmin_vx_i16m2(result, 127, vl), -128, vl);
    const vbool8_t below =
        __riscv_vmslt_vx_i16m2_b8(centered, -input_range_radius, vl);
    const vbool8_t above =
        __riscv_vmsgt_vx_i16m2_b8(centered, input_range_radius, vl);
    result = __riscv_vmerge_vxm_i16m2(result, -128, below, vl);
    result = __riscv_vmerge_vxm_i16m2(result, 127, above, vl);
    __riscv_vse8_v_i8m1(
        output + offset, __riscv_vncvt_x_x_w_i8m1(result, vl), vl);
    offset += static_cast<int>(vl);
  }
}

void RvvLstmCellQuantized(const int16_t *activ_temp, const int16_t *prev_state,
                          int outer_size, int output_depth,
                          int state_integer_bits, int16_t *output_state,
                          uint8_t *output_activation) {
  if (outer_size <= 0 || output_depth <= 0 || state_integer_bits < 0 ||
      state_integer_bits > 15) {
    return;
  }
  for (int batch = 0; batch < outer_size; ++batch) {
    const int16_t *batch_activ = activ_temp + batch * 4 * output_depth;
    const int16_t *batch_prev_state = prev_state + batch * output_depth;
    int16_t *batch_output_state = output_state + batch * output_depth;
    uint8_t *batch_output_activation = output_activation + batch * output_depth;
    for (int offset = 0; offset < output_depth;) {
      const size_t vl = __riscv_vsetvl_e16m2(output_depth - offset);
      const vint16m2_t input_gate =
          __riscv_vle16_v_i16m2(batch_activ + offset, vl);
      const vint16m2_t modulation_gate = __riscv_vle16_v_i16m2(
          batch_activ + output_depth + offset, vl);
      const vint16m2_t forget_gate = __riscv_vle16_v_i16m2(
          batch_activ + 2 * output_depth + offset, vl);
      const vint16m2_t output_gate = __riscv_vle16_v_i16m2(
          batch_activ + 3 * output_depth + offset, vl);
      const vint16m2_t previous =
          __riscv_vle16_v_i16m2(batch_prev_state + offset, vl);
      const vint16m2_t input_gate_output =
          RvvLogistic16Vector(input_gate, 3, vl);
      const vint16m2_t modulation_gate_output =
          RvvTanh16Vector(modulation_gate, 3, vl);
      const vint16m2_t forget_gate_output =
          RvvLogistic16Vector(forget_gate, 3, vl);
      const vint16m2_t output_gate_output =
          RvvLogistic16Vector(output_gate, 3, vl);
      const vint16m2_t input_times_modulation =
          RvvSaturatingRoundingDoublingHighMul16Vector(
              input_gate_output, modulation_gate_output, vl);
      const vint16m2_t previous_times_forget =
          RvvSaturatingRoundingDoublingHighMul16Vector(
              forget_gate_output, previous, vl);
      const vint16m2_t rescaled_input = RvvRescale16Vector(
          input_times_modulation, 0, state_integer_bits, vl);
      const vint32m4_t state_sum = __riscv_vadd_vv_i32m4(
          __riscv_vsext_vf2_i32m4(rescaled_input, vl),
          __riscv_vsext_vf2_i32m4(previous_times_forget, vl), vl);
      const vint16m2_t new_state = __riscv_vncvt_x_x_w_i16m2(
          __riscv_vmax_vx_i32m4(
              __riscv_vmin_vx_i32m4(state_sum, INT16_MAX, vl), INT16_MIN, vl),
          vl);
      const vint16m2_t new_state_f3 =
          RvvRescale16Vector(new_state, state_integer_bits, 3, vl);
      const vint16m2_t output_tanh = RvvTanh16Vector(new_state_f3, 3, vl);
      const vint16m2_t output_product =
          RvvSaturatingRoundingDoublingHighMul16Vector(
              output_gate_output, output_tanh, vl);
      vint16m2_t rescaled_output =
          RvvRoundingDivideByPOT16Vector(output_product, 8, vl);
      rescaled_output = __riscv_vmax_vx_i16m2(
          __riscv_vmin_vx_i16m2(rescaled_output, 127, vl), -128, vl);
      __riscv_vse16_v_i16m2(batch_output_state + offset, new_state, vl);
      const vint16m2_t activation =
          __riscv_vadd_vx_i16m2(rescaled_output, 128, vl);
      __riscv_vse8_v_u8m1(
          batch_output_activation + offset,
          __riscv_vncvt_x_x_w_u8m1(
              __riscv_vreinterpret_v_i16m2_u16m2(activation), vl),
          vl);
      offset += static_cast<int>(vl);
    }
  }
}

void RvvMatrixBatchVectorMultiplyAccumulate(const float *matrix, int m_rows,
                                            int m_cols, const float *vector,
                                            int n_batch, float *result) {
  for (int batch = 0; batch < n_batch; ++batch) {
    const float *vector_in_batch = vector + batch * m_cols;
    float *result_in_batch = result + batch * m_rows;

    for (int row = 0; row < m_rows; ++row) {
      const float *matrix_row = matrix + row * m_cols;
      result_in_batch[row] +=
          RvvDotProduct(matrix_row, vector_in_batch, m_cols);
    }
  }
}

void RvvMatrixBatchVectorMultiplyAccumulate(const int8_t *matrix, int m_rows,
                                            int m_cols, const int8_t *vectors,
                                            const float *scaling_factors,
                                            int n_batch, float *result) {
  for (int batch = 0; batch < n_batch; ++batch) {
    const int8_t *vector_in_batch = vectors + batch * m_cols;
    float *result_in_batch = result + batch * m_rows;
    for (int row = 0; row < m_rows; ++row) {
      const int8_t *matrix_row = matrix + row * m_cols;
      result_in_batch[row] +=
          RvvInt8DotProduct(matrix_row, vector_in_batch, m_cols) *
          scaling_factors[batch];
    }
  }
}

void RvvMatrixBatchVectorMultiplyAccumulate(const int8_t *matrix, int m_rows,
                                            int m_cols, const int8_t *vectors,
                                            const float *scaling_factors,
                                            int n_batch, int32_t *scratch,
                                            float *result,
                                            CpuBackendContext *context) {
  (void)context;
  for (int batch = 0; batch < n_batch; ++batch) {
    const int8_t *vector_in_batch = vectors + batch * m_cols;
    int32_t *scratch_in_batch = scratch + batch * m_rows;
    float *result_in_batch = result + batch * m_rows;
    for (int row = 0; row < m_rows; ++row) {
      const int32_t dot_product =
          RvvInt8DotProduct(matrix + row * m_cols, vector_in_batch, m_cols);
      scratch_in_batch[row] = dot_product;
      result_in_batch[row] += dot_product * scaling_factors[batch];
    }
  }
}

void RvvMatrixBatchVectorMultiplyAccumulate(
    const int8_t *matrix, int m_rows, int m_cols, const int8_t *vectors,
    const float *scaling_factors, int n_batch, float *result,
    const float *per_channel_scale, const int32_t *input_offset,
    int32_t *scratch, int32_t *row_sums, bool *compute_row_sums,
    CpuBackendContext *context) {
  (void)scratch;
  (void)context;
  if (input_offset == nullptr) {
    RvvMatrixBatchVectorMultiplyAccumulate(matrix, m_rows, m_cols, vectors,
                                           scaling_factors, n_batch, result);
    return;
  }

  if (compute_row_sums == nullptr || *compute_row_sums) {
    for (int row = 0; row < m_rows; ++row) {
      row_sums[row] = RvvInt8Sum(matrix + row * m_cols, m_cols);
    }
    if (compute_row_sums != nullptr) {
      *compute_row_sums = false;
    }
  }

  for (int batch = 0; batch < n_batch; ++batch) {
    const int8_t *vector_in_batch = vectors + batch * m_cols;
    float *result_in_batch = result + batch * m_rows;
    for (int row = 0; row < m_rows; ++row) {
      int32_t dot_product =
          RvvInt8DotProduct(matrix + row * m_cols, vector_in_batch, m_cols);
      dot_product -= row_sums[row] * input_offset[batch];
      float scale = scaling_factors[batch];
      if (per_channel_scale != nullptr) {
        scale *= per_channel_scale[row];
      }
      result_in_batch[row] += dot_product * scale;
    }
  }
}

void RvvMatrixScalarMultiplyAccumulate(const int8_t *matrix, int32_t scalar,
                                       int32_t n_row, int32_t n_col,
                                       int32_t *output) {
  for (int32_t row = 0; row < n_row; ++row) {
    output[row] += RvvInt8Sum(matrix + row * n_col, n_col) * scalar;
  }
}

void RvvSparseMatrixBatchVectorMultiplyAccumulate1x4(
    const float *matrix, const int32_t *segments, const int32_t *indices,
    int m_rows, int m_cols, const float *vector, int n_batch, float *result) {
  constexpr int kBlockSize = 4;
  for (int batch = 0; batch < n_batch; ++batch) {
    const float *matrix_ptr = matrix;
    const float *vector_in_batch = vector + batch * m_cols;
    for (int row = 0; row < m_rows; ++row) {
      float dot_product = 0.0f;
      for (int block = segments[row]; block < segments[row + 1]; ++block) {
        dot_product += RvvDotProduct(
            matrix_ptr, vector_in_batch + indices[block] * kBlockSize,
            kBlockSize);
        matrix_ptr += kBlockSize;
      }
      result[batch * m_rows + row] += dot_product;
    }
  }
}

void RvvSparseMatrixBatchVectorMultiplyAccumulate(const float *matrix,
                                                  const uint8_t *ledger,
                                                  int m_rows, int m_cols,
                                                  const float *vector,
                                                  int n_batch, float *result) {
  constexpr int kBlockSize = 16;
  for (int batch = 0; batch < n_batch; ++batch) {
    const float *matrix_ptr = matrix;
    const uint8_t *ledger_ptr = ledger;
    const float *vector_in_batch = vector + batch * m_cols;
    for (int row = 0; row < m_rows; ++row) {
      float dot_product = 0.0f;
      const int nonzero_blocks = *ledger_ptr++;
      for (int block = 0; block < nonzero_blocks; ++block) {
        dot_product += RvvDotProduct(
            matrix_ptr, vector_in_batch + *ledger_ptr++ * kBlockSize,
            kBlockSize);
        matrix_ptr += kBlockSize;
      }
      result[batch * m_rows + row] += dot_product;
    }
  }
}

void RvvSparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t *matrix, const uint8_t *ledger, int m_rows, int m_cols,
    const int8_t *vectors, const float *scaling_factors, int n_batch,
    float *result, const float *per_channel_scale) {
  constexpr int kBlockSize = 16;
  for (int batch = 0; batch < n_batch; ++batch) {
    const int8_t *matrix_ptr = matrix;
    const uint8_t *ledger_ptr = ledger;
    const int8_t *vector_in_batch = vectors + batch * m_cols;
    for (int row = 0; row < m_rows; ++row) {
      int32_t dot_product = 0;
      const int nonzero_blocks = *ledger_ptr++;
      for (int block = 0; block < nonzero_blocks; ++block) {
        dot_product += RvvInt8DotProduct(
            matrix_ptr, vector_in_batch + *ledger_ptr++ * kBlockSize,
            kBlockSize);
        matrix_ptr += kBlockSize;
      }
      float scale = scaling_factors[batch];
      if (per_channel_scale != nullptr) {
        scale *= per_channel_scale[row];
      }
      result[batch * m_rows + row] += dot_product * scale;
    }
  }
}

void RvvSparseMatrixBatchVectorMultiplyAccumulate1x16(
    const int8_t *matrix, const int32_t *segments, const int32_t *indices,
    int m_rows, int m_cols, const int8_t *vector, const int32_t *bias_vector,
    int n_batch, int32_t input_offset, int32_t output_multiplier,
    int32_t output_shift, const int32_t *per_channel_scale,
    const int32_t *per_channel_shift, int32_t output_offset,
    int32_t output_activation_min, int32_t output_activation_max,
    int8_t *result) {
  constexpr int kBlockSize = 16;
  for (int batch = 0; batch < n_batch; ++batch) {
    const int8_t *matrix_ptr = matrix;
    const int8_t *vector_in_batch = vector + batch * m_cols;
    for (int row = 0; row < m_rows; ++row) {
      int32_t dot_product = 0;
      int32_t matrix_row_sum = 0;
      for (int block = segments[row]; block < segments[row + 1]; ++block) {
        dot_product += RvvInt8DotProduct(
            matrix_ptr, vector_in_batch + indices[block] * kBlockSize,
            kBlockSize);
        matrix_row_sum += RvvInt8Sum(matrix_ptr, kBlockSize);
        matrix_ptr += kBlockSize;
      }
      const int32_t bias = bias_vector != nullptr ? bias_vector[row] : 0;
      dot_product += bias + input_offset * matrix_row_sum;
      const int32_t multiplier = per_channel_scale != nullptr
                                     ? per_channel_scale[row]
                                     : output_multiplier;
      const int32_t shift =
          per_channel_shift != nullptr ? per_channel_shift[row] : output_shift;
      dot_product =
          RvvMultiplyByQuantizedMultiplier(dot_product, multiplier, shift);
      dot_product += output_offset;
      if (dot_product < output_activation_min) {
        dot_product = output_activation_min;
      } else if (dot_product > output_activation_max) {
        dot_product = output_activation_max;
      }
      result[batch * m_rows + row] = static_cast<int8_t>(dot_product);
    }
  }
}

void RvvMatrixBatchVectorMultiplyAccumulate(
    const int8_t *input, const int32_t *bias,
    const int8_t *input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t *scratch, int16_t *output, CpuBackendContext *context) {
  (void)scratch;
  (void)context;
  RvvMatrixBatchVectorMultiplyAccumulateImpl(
      input, bias, input_to_gate_weights, multiplier, shift, n_batch, n_input,
      n_output, output_zp, output);
}

void RvvMatrixBatchVectorMultiplyAccumulate(
    const int8_t *input, const int32_t *bias,
    const int8_t *input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t *scratch, int8_t *output, CpuBackendContext *context) {
  (void)scratch;
  (void)context;
  RvvMatrixBatchVectorMultiplyAccumulateImpl(
      input, bias, input_to_gate_weights, multiplier, shift, n_batch, n_input,
      n_output, output_zp, output);
}

float RvvVectorVectorDotProduct(const float *vector1, const float *vector2,
                                int v_size) {
  return RvvDotProduct(vector1, vector2, v_size);
}

void RvvCwiseAdd(const int16_t *input_1, const int16_t *input_2, int n_batch,
                 int n_input, int16_t *output) {
  if (n_batch <= 0 || n_input <= 0) {
    return;
  }

  for (int batch = 0; batch < n_batch; ++batch) {
    const size_t batch_offset = static_cast<size_t>(batch) * n_input;
    for (size_t i = 0; i < static_cast<size_t>(n_input);) {
      const size_t remaining = static_cast<size_t>(n_input) - i;
      const size_t vl = __riscv_vsetvl_e16m1(remaining);
      const vint16m1_t values_1 =
          __riscv_vle16_v_i16m1(input_1 + batch_offset + i, vl);
      const vint16m1_t values_2 =
          __riscv_vle16_v_i16m1(input_2 + batch_offset + i, vl);
      const vint16m1_t sum = __riscv_vsadd_vv_i16m1(values_1, values_2, vl);
      __riscv_vse16_v_i16m1(output + batch_offset + i, sum, vl);
      i += vl;
    }
  }
}

void RvvCwiseMul(const int16_t *input_1, const int16_t *input_2, int n_batch,
                 int n_input, int shift, int16_t *output) {
  if (n_batch <= 0 || n_input <= 0) {
    return;
  }
  for (int batch = 0; batch < n_batch; ++batch) {
    const size_t batch_offset = static_cast<size_t>(batch) * n_input;
    for (size_t i = 0; i < static_cast<size_t>(n_input);) {
      const size_t remaining = static_cast<size_t>(n_input) - i;
      const size_t vl = __riscv_vsetvl_e16m1(remaining);
      const vint16m1_t values_1 =
          __riscv_vle16_v_i16m1(input_1 + batch_offset + i, vl);
      const vint16m1_t values_2 =
          __riscv_vle16_v_i16m1(input_2 + batch_offset + i, vl);
      const vint32m2_t products =
          __riscv_vwmul_vv_i32m2(values_1, values_2, vl);
      const vint32m2_t rounded =
          RvvRoundingDivideByPOTVector(products, shift, vl);
      __riscv_vse16_v_i16m1(
          output + batch_offset + i,
          __riscv_vnsra_wx_i16m1(rounded, 0, vl), vl);
      i += vl;
    }
  }
}

void RvvCwiseMul(const int16_t *input_1, const int16_t *input_2,
                 int32_t multiplier, int32_t shift, int32_t n_batch,
                 int32_t n_input, int32_t output_zp, int8_t *output) {
  if (n_batch <= 0 || n_input <= 0) {
    return;
  }
#if TFLITE_SINGLE_ROUNDING
  for (int32_t batch = 0; batch < n_batch; ++batch) {
    const size_t batch_offset = static_cast<size_t>(batch) * n_input;
    for (size_t i = 0; i < static_cast<size_t>(n_input);) {
      const size_t remaining = static_cast<size_t>(n_input) - i;
      const size_t vl = __riscv_vsetvl_e16m1(remaining);
      const vint16m1_t values_1 =
          __riscv_vle16_v_i16m1(input_1 + batch_offset + i, vl);
      const vint16m1_t values_2 =
          __riscv_vle16_v_i16m1(input_2 + batch_offset + i, vl);
      const vint32m2_t products =
          __riscv_vwmul_vv_i32m2(values_1, values_2, vl);
      vint32m2_t values = RvvMultiplyByQuantizedMultiplierVector(
          products, multiplier, shift, vl);
      values = __riscv_vadd_vx_i32m2(values, output_zp, vl);
      values = __riscv_vmax_vx_i32m2(values, -128, vl);
      values = __riscv_vmin_vx_i32m2(values, 127, vl);
      const vint16m1_t values16 = __riscv_vncvt_x_x_w_i16m1(values, vl);
      __riscv_vse8_v_i8mf2(output + batch_offset + i,
                           __riscv_vncvt_x_x_w_i8mf2(values16, vl), vl);
      i += vl;
    }
  }
#else
  for (int32_t batch = 0; batch < n_batch; ++batch) {
    const size_t batch_offset = static_cast<size_t>(batch) * n_input;
    for (size_t i = 0; i < static_cast<size_t>(n_input);) {
      const size_t remaining = static_cast<size_t>(n_input) - i;
      const size_t vl = __riscv_vsetvl_e16m1(remaining);
      const vint16m1_t values_1 =
          __riscv_vle16_v_i16m1(input_1 + batch_offset + i, vl);
      const vint16m1_t values_2 =
          __riscv_vle16_v_i16m1(input_2 + batch_offset + i, vl);
      const vint32m2_t products =
          __riscv_vwmul_vv_i32m2(values_1, values_2, vl);
      vint32m2_t values = RvvMultiplyByQuantizedMultiplierVector(
          products, multiplier, shift, vl);
      values = __riscv_vadd_vx_i32m2(values, output_zp, vl);
      values = __riscv_vmax_vx_i32m2(values, -128, vl);
      values = __riscv_vmin_vx_i32m2(values, 127, vl);
      const vint16m1_t values16 = __riscv_vncvt_x_x_w_i16m1(values, vl);
      __riscv_vse8_v_i8mf2(output + batch_offset + i,
                           __riscv_vncvt_x_x_w_i8mf2(values16, vl), vl);
      i += vl;
    }
  }
#endif
}

void RvvCwiseClipping(float *vector, int v_size, float clipping_value) {
  if (v_size <= 0) {
    return;
  }

  const float min_value = -clipping_value;
  for (size_t i = 0; i < static_cast<size_t>(v_size);) {
    const size_t remaining = static_cast<size_t>(v_size) - i;
    const size_t vl = __riscv_vsetvl_e32m1(remaining);
    vfloat32m1_t values = __riscv_vle32_v_f32m1(vector + i, vl);
    values = __riscv_vfmin_vf_f32m1(values, clipping_value, vl);
    values = __riscv_vfmax_vf_f32m1(values, min_value, vl);
    __riscv_vse32_v_f32m1(vector + i, values, vl);
    i += vl;
  }
}

void RvvCwiseClipping(int16_t *vector, int v_size, int16_t clipping_value) {
  if (v_size <= 0) {
    return;
  }

  const int16_t min_value = -clipping_value;
  for (size_t i = 0; i < static_cast<size_t>(v_size);) {
    const size_t remaining = static_cast<size_t>(v_size) - i;
    const size_t vl = __riscv_vsetvl_e16m1(remaining);
    vint16m1_t values = __riscv_vle16_v_i16m1(vector + i, vl);
    values = __riscv_vmin_vx_i16m1(values, clipping_value, vl);
    values = __riscv_vmax_vx_i16m1(values, min_value, vl);
    __riscv_vse16_v_i16m1(vector + i, values, vl);
    i += vl;
  }
}

void RvvCwiseClipping(int8_t *vector, int v_size, int8_t clipping_value) {
  if (v_size <= 0) {
    return;
  }

  const int8_t min_value = -clipping_value;
  for (size_t i = 0; i < static_cast<size_t>(v_size);) {
    const size_t remaining = static_cast<size_t>(v_size) - i;
    const size_t vl = __riscv_vsetvl_e8m1(remaining);
    vint8m1_t values = __riscv_vle8_v_i8m1(vector + i, vl);
    values = __riscv_vmin_vx_i8m1(values, clipping_value, vl);
    values = __riscv_vmax_vx_i8m1(values, min_value, vl);
    __riscv_vse8_v_i8m1(vector + i, values, vl);
    i += vl;
  }
}

void RvvVectorBatchVectorCwiseProductAccumulate(const int16_t *vector,
                                                 int v_size,
                                                 const int16_t *batch_vector,
                                                 int n_batch, int32_t multiplier,
                                                 int shift, int16_t *result) {
  if (v_size <= 0 || n_batch <= 0) {
    return;
  }
#if TFLITE_SINGLE_ROUNDING
  for (int batch = 0; batch < n_batch; ++batch) {
    const int16_t *batch_values = batch_vector + batch * v_size;
    int16_t *result_values = result + batch * v_size;
    for (size_t i = 0; i < static_cast<size_t>(v_size);) {
      const size_t remaining = static_cast<size_t>(v_size) - i;
      const size_t vl = __riscv_vsetvl_e16m1(remaining);
      const vint16m1_t vector_values = __riscv_vle16_v_i16m1(vector + i, vl);
      const vint16m1_t batch_values_vector =
          __riscv_vle16_v_i16m1(batch_values + i, vl);
      const vint32m2_t products =
          __riscv_vwmul_vv_i32m2(vector_values, batch_values_vector, vl);
      vint32m2_t values = RvvMultiplyByQuantizedMultiplierVector(
          products, multiplier, shift, vl);
      values = __riscv_vadd_vv_i32m2(
          values,
          __riscv_vsext_vf2_i32m2(__riscv_vle16_v_i16m1(result_values + i, vl),
                                  vl),
          vl);
      values = __riscv_vmax_vx_i32m2(values, INT16_MIN, vl);
      values = __riscv_vmin_vx_i32m2(values, INT16_MAX, vl);
      __riscv_vse16_v_i16m1(result_values + i,
                            __riscv_vncvt_x_x_w_i16m1(values, vl), vl);
      i += vl;
    }
  }
#else
  for (int batch = 0; batch < n_batch; ++batch) {
    const int16_t *batch_values = batch_vector + batch * v_size;
    int16_t *result_values = result + batch * v_size;
    for (size_t i = 0; i < static_cast<size_t>(v_size);) {
      const size_t remaining = static_cast<size_t>(v_size) - i;
      const size_t vl = __riscv_vsetvl_e16m1(remaining);
      const vint16m1_t vector_values = __riscv_vle16_v_i16m1(vector + i, vl);
      const vint16m1_t batch_values_vector =
          __riscv_vle16_v_i16m1(batch_values + i, vl);
      const vint32m2_t products =
          __riscv_vwmul_vv_i32m2(vector_values, batch_values_vector, vl);
      vint32m2_t values = RvvMultiplyByQuantizedMultiplierVector(
          products, multiplier, shift, vl);
      values = __riscv_vadd_vv_i32m2(
          values,
          __riscv_vsext_vf2_i32m2(__riscv_vle16_v_i16m1(result_values + i, vl),
                                  vl),
          vl);
      values = __riscv_vmax_vx_i32m2(values, INT16_MIN, vl);
      values = __riscv_vmin_vx_i32m2(values, INT16_MAX, vl);
      __riscv_vse16_v_i16m1(result_values + i,
                            __riscv_vncvt_x_x_w_i16m1(values, vl), vl);
      i += vl;
    }
  }
#endif
}

void RvvSub1Vector(const float *vector, int v_size, float *result) {
  if (v_size <= 0) {
    return;
  }

  for (size_t i = 0; i < static_cast<size_t>(v_size);) {
    const size_t remaining = static_cast<size_t>(v_size) - i;
    const size_t vl = __riscv_vsetvl_e32m1(remaining);
    const vfloat32m1_t values = __riscv_vle32_v_f32m1(vector + i, vl);
    const vfloat32m1_t subtracted = __riscv_vfrsub_vf_f32m1(values, 1.0f, vl);
    __riscv_vse32_v_f32m1(result + i, subtracted, vl);
    i += vl;
  }
}

void RvvSub1Vector(const int16_t *vector, int v_size, int16_t *result) {
  if (v_size <= 0) {
    return;
  }

  for (size_t i = 0; i < static_cast<size_t>(v_size);) {
    const size_t remaining = static_cast<size_t>(v_size) - i;
    const size_t vl = __riscv_vsetvl_e16m1(remaining);
    const vint16m1_t values = __riscv_vle16_v_i16m1(vector + i, vl);
    const vint16m1_t result_values = __riscv_vxor_vx_i16m1(values, 32767, vl);
    __riscv_vse16_v_i16m1(result + i, result_values, vl);
    i += vl;
  }
}

void RvvReductionSumVector(const float *input_vector, float *output_vector,
                           int output_size, int reduction_size) {
  for (int output = 0; output < output_size; ++output) {
    output_vector[output] =
        RvvFloatSum(input_vector + output * reduction_size, reduction_size);
  }
}

void RvvReductionSumVector(const int8_t *input_vector, int32_t *output_vector,
                           int output_size, int reduction_size) {
  for (int output = 0; output < output_size; ++output) {
    output_vector[output] =
        RvvInt8Sum(input_vector + output * reduction_size, reduction_size);
  }
}

void RvvMeanStddevNormalization(const float *input_vector, float *output_vector,
                                int v_size, int n_batch) {
  if (v_size <= 0 || n_batch <= 0) {
    return;
  }

  constexpr float kNormalizationConstant = 1e-8f;
  for (int batch = 0; batch < n_batch; ++batch) {
    const float *input = input_vector + batch * v_size;
    float *output = output_vector + batch * v_size;
    const float mean = RvvFloatSum(input, v_size) / v_size;
    const float variance =
        RvvSquaredDifferenceSum(input, v_size, mean) / v_size;
    const float stddev_inv =
        1.0f / __builtin_sqrtf(variance + kNormalizationConstant);

    for (size_t i = 0; i < static_cast<size_t>(v_size);) {
      const size_t remaining = static_cast<size_t>(v_size) - i;
      const size_t vl = __riscv_vsetvl_e32m1(remaining);
      const vfloat32m1_t values = __riscv_vle32_v_f32m1(input + i, vl);
      const vfloat32m1_t differences = __riscv_vfsub_vf_f32m1(values, mean, vl);
      const vfloat32m1_t normalized =
          __riscv_vfmul_vf_f32m1(differences, stddev_inv, vl);
      __riscv_vse32_v_f32m1(output + i, normalized, vl);
      i += vl;
    }
  }
}

void RvvVectorScalarMultiply(const int8_t *vector, int v_size, float scale,
                             float *result) {
  if (v_size <= 0) {
    return;
  }

  for (size_t i = 0; i < static_cast<size_t>(v_size);) {
    const size_t remaining = static_cast<size_t>(v_size) - i;
    const size_t vl = __riscv_vsetvl_e8m1(remaining);
    const vint8m1_t values = __riscv_vle8_v_i8m1(vector + i, vl);
    const vint32m4_t values_i32 = __riscv_vsext_vf4_i32m4(values, vl);
    const vfloat32m4_t values_f32 = __riscv_vfcvt_f_x_v_f32m4(values_i32, vl);
    const vfloat32m4_t scaled = __riscv_vfmul_vf_f32m4(values_f32, scale, vl);
    __riscv_vse32_v_f32m4(result + i, scaled, vl);
    i += vl;
  }
}

void RvvSymmetricQuantizeFloats(const float *values, int size,
                                int8_t *quantized_values, float *min_value,
                                float *max_value, float *scaling_factor) {
  if (size <= 0) {
    return;
  }

  float min = values[0];
  float max = values[0];
  for (int i = 1; i < size; ++i) {
    if (values[i] < min) {
      min = values[i];
    }
    if (values[i] > max) {
      max = values[i];
    }
  }
  *min_value = min;
  *max_value = max;
  RvvSymmetricQuantizeFloats(values, size, quantized_values, min, max,
                             scaling_factor);
}

void RvvSymmetricQuantizeFloats(const float *values, int size,
                                int8_t *quantized_values, float min_value,
                                float max_value, float *scaling_factor) {
  constexpr int32_t kScale = 127;
  const float abs_min = __builtin_fabsf(min_value);
  const float abs_max = __builtin_fabsf(max_value);
  const float range = abs_min > abs_max ? abs_min : abs_max;
  if (range == 0.0f) {
    for (int i = 0; i < size; ++i) {
      quantized_values[i] = 0;
    }
    *scaling_factor = 1.0f;
    return;
  }

  *scaling_factor = range / kScale;
  RvvQuantizeFloatRange(values, size, kScale / range, 0, -kScale, kScale,
                        quantized_values);
}

void RvvAsymmetricQuantizeFloats(const float *values, int size,
                                 int8_t *quantized_values,
                                 float *scaling_factor, int32_t *offset) {
  constexpr int32_t kMinScale = -128;
  constexpr int32_t kMaxScale = 127;
  if (size <= 0) {
    *scaling_factor = 1.0f;
    *offset = 0;
    return;
  }

  float rmin;
  float rmax;
  RvvMinMaxWithZero(values, size, &rmin, &rmax);

  if (rmin == rmax) {
    for (int i = 0; i < size; ++i) {
      quantized_values[i] = 0;
    }
    *scaling_factor = 1.0f;
    *offset = 0;
    return;
  }

  const double scale = (static_cast<double>(rmax) - rmin) /
                       (static_cast<double>(kMaxScale) - kMinScale);
  const double zero_point_from_min =
      kMinScale - static_cast<double>(rmin) / scale;
  const double zero_point_from_max =
      kMaxScale - static_cast<double>(rmax) / scale;
  const double zero_point_from_min_error =
      __builtin_fabs(static_cast<double>(kMinScale)) +
      __builtin_fabs(static_cast<double>(rmin) / scale);
  const double zero_point_from_max_error =
      __builtin_fabs(static_cast<double>(kMaxScale)) +
      __builtin_fabs(static_cast<double>(rmax) / scale);
  const double zero_point_double =
      zero_point_from_min_error < zero_point_from_max_error
          ? zero_point_from_min
          : zero_point_from_max;
  if (zero_point_double <= kMinScale) {
    *offset = kMinScale;
  } else if (zero_point_double >= kMaxScale) {
    *offset = kMaxScale;
  } else {
    *offset = static_cast<int32_t>(__builtin_round(zero_point_double));
  }
  *scaling_factor = static_cast<float>(scale);
  RvvQuantizeFloatRange(values, size, 1.0f / *scaling_factor, *offset,
                        kMinScale, kMaxScale, quantized_values);
}

bool RvvIsZeroVector(const float *vector, int v_size) {
  if (v_size <= 0) {
    return true;
  }

  for (size_t i = 0; i < static_cast<size_t>(v_size);) {
    const size_t remaining = static_cast<size_t>(v_size) - i;
    const size_t vl = __riscv_vsetvl_e32m1(remaining);
    const vfloat32m1_t values = __riscv_vle32_v_f32m1(vector + i, vl);
    const vbool32_t zero_mask = __riscv_vmfeq_vf_f32m1_b32(values, 0.0f, vl);
    if (__riscv_vcpop_m_b32(zero_mask, vl) != vl) {
      return false;
    }
    i += vl;
  }
  return true;
}

bool RvvIsZeroVector(const int8_t *vector, int v_size) {
  if (v_size <= 0) {
    return true;
  }

  for (size_t i = 0; i < static_cast<size_t>(v_size);) {
    const size_t remaining = static_cast<size_t>(v_size) - i;
    const size_t vl = __riscv_vsetvl_e8m1(remaining);
    const vint8m1_t values = __riscv_vle8_v_i8m1(vector + i, vl);
    const vbool8_t zero_mask = __riscv_vmseq_vx_i8m1_b8(values, 0, vl);
    if (__riscv_vcpop_m_b8(zero_mask, vl) != vl) {
      return false;
    }
    i += vl;
  }
  return true;
}

} // namespace tensor_utils
} // namespace tflite

#endif // defined(__riscv_vector)
