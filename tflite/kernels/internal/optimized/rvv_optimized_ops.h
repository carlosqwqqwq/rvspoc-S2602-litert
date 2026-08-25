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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_OPTIMIZED_OPS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_OPTIMIZED_OPS_H_

#if defined(__riscv_vector)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <riscv_vector.h>

#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/quantization_util.h"
#include "tflite/kernels/internal/optimized/rvv_tensor_utils.h"

namespace tflite {
namespace rvv_optimized_ops {

// The fixed-size lane scratch buffers below hold at most 64 active lanes.
// Only stack-buffer paths use RvvSetVlE8M1Stack; vector-only paths retain the
// full implementation-defined VLEN.
constexpr size_t kMaxStackVectorLanes = 64;

inline size_t RvvSetVlE8M1(size_t avl) {
  return __riscv_vsetvl_e8m1(avl);
}

inline size_t RvvSetVlE8M1Stack(size_t avl) {
  return __riscv_vsetvl_e8m1(std::min(avl, kMaxStackVectorLanes));
}

// Per-channel quantization has a fixed-size scalar fallback.  Keep that
// fallback within its stack buffer, while allowing the vector path to use the
// full implementation-defined e32m4 VLMAX on wider VLEN hardware.
inline size_t RvvSetVlE32M4ForQuantize(size_t avl, bool vectorized) {
  return __riscv_vsetvl_e32m4(
      vectorized ? avl : std::min(avl, kMaxStackVectorLanes));
}

inline vint32m4_t RvvVectorizedQuantize(vint32m4_t value,
                                        int32_t quantized_multiplier,
                                        int shift, size_t vl);

// Ruy's quantized GEMM backend uses a single rounded fixed-point shift.  RVV
// GEMM bypasses Ruy, so it must preserve that backend contract instead of
// using LiteRT's default double-rounding helper.
inline int32_t MultiplyByRuyQuantizedMultiplier(int32_t value,
                                                 int32_t multiplier,
                                                 int shift) {
  const int total_shift = 31 - shift;
  const int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
  const int64_t scaled = static_cast<int64_t>(value) * multiplier + round;
  return static_cast<int32_t>(scaled >> total_shift);
}

template <typename InputScalar, typename OutputScalar>
inline void RvvMeanChannels(const InputScalar* input, int batches, int height,
                            int width, int depth, int start_depth,
                            int end_depth, int32_t multiplier, int shift,
                            int32_t bias, OutputScalar* output,
                            int output_batch_stride) {
  const int32_t min_value =
      static_cast<int32_t>(std::numeric_limits<OutputScalar>::min());
  const int32_t max_value =
      static_cast<int32_t>(std::numeric_limits<OutputScalar>::max());
  const int input_batch_stride = height * width * depth;
  const int spatial_stride = width * depth;

  for (int batch = 0; batch < batches; ++batch) {
    for (int channel = start_depth; channel < end_depth;) {
      const size_t vl =
          RvvSetVlE8M1(static_cast<size_t>(end_depth - channel));
      vint32m4_t sums = __riscv_vmv_v_x_i32m4(0, vl);
      for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
          const InputScalar* values =
              input + batch * input_batch_stride + row * spatial_stride +
              col * depth + channel;
          if constexpr (std::is_same_v<InputScalar, int8_t>) {
            const vint16m2_t values16 = __riscv_vsext_vf2_i16m2(
                __riscv_vle8_v_i8m1(values, vl), vl);
            sums = __riscv_vadd_vv_i32m4(
                sums, __riscv_vsext_vf2_i32m4(values16, vl), vl);
          } else {
            const vuint16m2_t values16 = __riscv_vzext_vf2_u16m2(
                __riscv_vle8_v_u8m1(values, vl), vl);
            const vuint32m4_t values32 =
                __riscv_vzext_vf2_u32m4(values16, vl);
            sums = __riscv_vadd_vv_i32m4(
                sums, __riscv_vreinterpret_v_u32m4_i32m4(values32), vl);
          }
        }
      }

      vint32m4_t value =
          RvvVectorizedQuantize(sums, multiplier, shift, vl);
      value = __riscv_vadd_vx_i32m4(value, bias, vl);
      value = __riscv_vmax_vx_i32m4(value, min_value, vl);
      value = __riscv_vmin_vx_i32m4(value, max_value, vl);
      if constexpr (std::is_same_v<OutputScalar, int8_t>) {
        const vint16m2_t value16 =
            __riscv_vncvt_x_x_w_i16m2(value, vl);
        __riscv_vse8_v_i8m1(
            reinterpret_cast<int8_t*>(output + batch * output_batch_stride +
                                      channel),
            __riscv_vncvt_x_x_w_i8m1(value16, vl), vl);
      } else {
        const vuint16m2_t value16 = __riscv_vncvt_x_x_w_u16m2(
            __riscv_vreinterpret_v_i32m4_u32m4(value), vl);
        __riscv_vse8_v_u8m1(
            reinterpret_cast<uint8_t*>(output + batch * output_batch_stride +
                                       channel),
            __riscv_vncvt_x_x_w_u8m1(value16, vl), vl);
      }
      channel += static_cast<int>(vl);
    }
  }
}

inline void RvvLookupTable(const uint8_t* input, int size, const uint8_t* table,
                           uint8_t* output) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = RvvSetVlE8M1(size - offset);
    const vuint8m1_t indices = __riscv_vle8_v_u8m1(input + offset, vl);
    __riscv_vse8_v_u8m1(output + offset,
                        __riscv_vluxei8_v_u8m1(table, indices, vl), vl);
    offset += static_cast<int>(vl);
  }
}

inline void RvvLookupTable(const int8_t* input, int size, const int8_t* table,
                           int8_t* output) {
  RvvLookupTable(reinterpret_cast<const uint8_t*>(input), size,
                 reinterpret_cast<const uint8_t*>(table),
                 reinterpret_cast<uint8_t*>(output));
}

// Applies the quantized multiplier used by LiteRT's integer binary ops.
inline vint32m4_t RvvVectorizedQuantize(vint32m4_t value,
                                        int32_t quantized_multiplier,
                                        int shift, size_t vl) {
#if defined(TFLITE_SINGLE_ROUNDING) && TFLITE_SINGLE_ROUNDING
  // Bit-exact single-rounding MultiplyByQuantizedMultiplier (the contract
  // used when TFLITE_SINGLE_ROUNDING is enabled):
  //   total_shift = 31 - shift;
  //   round = 1 << (total_shift - 1);
  //   result = (x * qmul + round) >> total_shift;
  // Computed in 64-bit vectors so the 32x32 product and the round add cannot
  // lose precision; the quantized result is guaranteed to fit int32.
  const int64_t total_shift = 31 - shift;
  const int64_t round = INT64_C(1) << (total_shift - 1);
  vint64m8_t ab = __riscv_vwmul_vx_i64m8(value, quantized_multiplier, vl);
  vint64m8_t acc = __riscv_vadd_vx_i64m8(ab, round, vl);
  acc = __riscv_vsra_vx_i64m8(acc, total_shift, vl);
  return __riscv_vncvt_x_x_w_i32m4(acc, vl);
#else
  // The default LiteRT contract is gemmlowp double-rounding. The first
  // stage uses a widened product; the explicit correction converts RVV's
  // arithmetic-floor shift to C++'s signed division toward zero for negative
  // non-integral products. The second stage follows RoundingDivideByPOT.
  const int left_shift = shift > 0 ? shift : 0;
  const int right_shift = shift > 0 ? 0 : -shift;
  if (left_shift != 0) {
    value = __riscv_vsll_vx_i32m4(value, left_shift, vl);
  }

  // RVV vsmul has the same saturating, signed-rounding high-multiply contract
  // as gemmlowp's SaturatingRoundingDoublingHighMul, including the INT_MIN
  // corner case.  Keep the explicit POT correction below: vssra rounds with
  // a different negative tie rule than LiteRT's RoundingDivideByPOT.
  vint32m4_t result =
      __riscv_vsmul_vx_i32m4(value, quantized_multiplier, vl);
  if (right_shift == 0) return result;

  const uint32_t mask_value = right_shift == 31
                                  ? 0x7fffffffU
                                  : ((uint32_t{1} << right_shift) - 1);
  const vint32m4_t remainder = __riscv_vand_vx_i32m4(
      result, static_cast<int32_t>(mask_value), vl);
  const vint32m4_t threshold = __riscv_vadd_vv_i32m4(
      __riscv_vmv_v_x_i32m4(static_cast<int32_t>(mask_value >> 1), vl),
      __riscv_vand_vx_i32m4(__riscv_vsra_vx_i32m4(result, 31, vl), 1, vl),
      vl);
  const vbool8_t round_up =
      __riscv_vmsgt_vv_i32m4_b8(remainder, threshold, vl);
  const vint32m4_t increment = __riscv_vmerge_vvm_i32m4(
      __riscv_vmv_v_x_i32m4(0, vl), __riscv_vmv_v_x_i32m4(1, vl), round_up,
      vl);
  return __riscv_vadd_vv_i32m4(
      __riscv_vsra_vx_i32m4(result, right_shift, vl), increment, vl);
#endif
}

// Per-channel form used by depthwise output. Keep the same double-rounding
// sequence as RvvVectorizedQuantize, but load the multiplier and right/left
// shifts per lane. The single-rounding contract uses a separate 64-bit vector
// path below because its total shift is also per lane.
inline vint32m4_t RvvVectorizedQuantizePerChannel(
    vint32m4_t value, const int32_t* quantized_multipliers,
    const int32_t* shifts, size_t vl, bool use_vectorized) {
#if defined(TFLITE_SINGLE_ROUNDING) && TFLITE_SINGLE_ROUNDING
  (void)use_vectorized;
  const vint32m4_t multipliers =
      __riscv_vle32_v_i32m4(quantized_multipliers, vl);
  const vint32m4_t shift = __riscv_vle32_v_i32m4(shifts, vl);
  const vint64m8_t shift64 = __riscv_vsext_vf2_i64m8(shift, vl);
  const vint64m8_t total_shift = __riscv_vsub_vv_i64m8(
      __riscv_vmv_v_x_i64m8(31, vl), shift64, vl);
  const vint64m8_t total_shift_minus_one =
      __riscv_vsub_vx_i64m8(total_shift, 1, vl);
  const vuint64m8_t total_shift_u =
      __riscv_vreinterpret_v_i64m8_u64m8(total_shift);
  const vuint64m8_t total_shift_minus_one_u =
      __riscv_vreinterpret_v_i64m8_u64m8(total_shift_minus_one);
  const vint64m8_t product = __riscv_vwmul_vv_i64m8(value, multipliers, vl);
  const vint64m8_t round = __riscv_vsll_vv_i64m8(
      __riscv_vmv_v_x_i64m8(1, vl), total_shift_minus_one_u, vl);
  const vint64m8_t rounded = __riscv_vadd_vv_i64m8(product, round, vl);
  const vint64m8_t shifted =
      __riscv_vsra_vv_i64m8(rounded, total_shift_u, vl);
  return __riscv_vncvt_x_x_w_i32m4(shifted, vl);
#else
  // The RVV widening multiply/narrow sequence is exact on the target
  // VLEN>=256 path. Keep VLEN=128 on the existing scalar contract for the
  // rare positive-left-shift overflow boundary; this still leaves the
  // depthwise accumulation vectorized and avoids a one-LSB portability gap.
  // The caller supplies the immutable VLEN class so hot loops do not reread
  // the vlenb CSR for every channel block.
  if (!use_vectorized) {
    int32_t values[kMaxStackVectorLanes];
    int32_t result[kMaxStackVectorLanes];
    __riscv_vse32_v_i32m4(values, value, vl);
    for (size_t lane = 0; lane < vl; ++lane) {
      result[lane] = MultiplyByQuantizedMultiplier(
          values[lane], quantized_multipliers[lane], shifts[lane]);
    }
    return __riscv_vle32_v_i32m4(result, vl);
  }

  const vint32m4_t multipliers =
      __riscv_vle32_v_i32m4(quantized_multipliers, vl);
  const vint32m4_t shift = __riscv_vle32_v_i32m4(shifts, vl);
  const vint32m4_t zero = __riscv_vmv_v_x_i32m4(0, vl);
  const vint32m4_t left_shift = __riscv_vmax_vv_i32m4(shift, zero, vl);
  value = __riscv_vsll_vv_i32m4(
      value, __riscv_vreinterpret_v_i32m4_u32m4(left_shift), vl);

  // vsmul.vv is bit-exact for the signed high multiply above and avoids the
  // i64m8 product/sign/remainder sequence in the depthwise hot path.
  const vint32m4_t rounded =
      __riscv_vsmul_vv_i32m4(value, multipliers, vl);

  // RoundingDivideByPOT with a per-lane right shift. Constructing the mask
  // as (1 << shift) - 1 also gives the required 0x7fffffff at shift=31.
  const vint32m4_t right_shift_signed =
      __riscv_vmax_vv_i32m4(__riscv_vsub_vv_i32m4(zero, shift, vl), zero, vl);
  const vuint32m4_t right_shift =
      __riscv_vreinterpret_v_i32m4_u32m4(right_shift_signed);
  const vint32m4_t mask = __riscv_vsub_vv_i32m4(
      __riscv_vsll_vv_i32m4(__riscv_vmv_v_x_i32m4(1, vl), right_shift, vl),
      __riscv_vmv_v_x_i32m4(1, vl), vl);
  const vint32m4_t remainder =
      __riscv_vand_vv_i32m4(rounded, mask, vl);
  const vint32m4_t threshold = __riscv_vadd_vv_i32m4(
      __riscv_vsra_vx_i32m4(mask, 1, vl),
      __riscv_vand_vx_i32m4(__riscv_vsra_vx_i32m4(rounded, 31, vl), 1, vl),
      vl);
  const vbool8_t round_up =
      __riscv_vmsgt_vv_i32m4_b8(remainder, threshold, vl);
  const vint32m4_t increment = __riscv_vmerge_vvm_i32m4(
      __riscv_vmv_v_x_i32m4(0, vl), __riscv_vmv_v_x_i32m4(1, vl), round_up,
      vl);
  return __riscv_vadd_vv_i32m4(
      __riscv_vsra_vv_i32m4(rounded, right_shift, vl), increment, vl);
#endif
}

// FP32 sigmoid via exp2 decomposition: exp(x) = 2^(x*log2e) with a
// polynomial for the fractional part and an integer exponent built through
// bit manipulation. Matches the Eigen scalar_logistic_op<float> semantics
// (saturating at large |x|) while vectorizing across arbitrary VLEN.
inline void RvvLogisticFloatM2(const float* input, float* output, int size) {
  constexpr float kLog2e = 1.4426950408889634f;
  constexpr float kC1 = 0.6931471805599453f;
  constexpr float kC2 = 0.2402265069591007f;
  constexpr float kC3 = 0.05550410866482158f;
  constexpr float kC4 = 0.009618129107628477f;
  constexpr float kC5 = 0.0013333558146428443f;
  int offset = 0;
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e32m2(size - offset);
    const vfloat32m2_t x = __riscv_vle32_v_f32m2(input + offset, vl);
    const vfloat32m2_t y = __riscv_vfmul_vf_f32m2(x, -kLog2e, vl);
    const vfloat32m2_t half = __riscv_vfmv_v_f_f32m2(0.5f, vl);
    const vfloat32m2_t y_rounded = __riscv_vfadd_vv_f32m2(
        y, __riscv_vfsgnj_vv_f32m2(half, y, vl), vl);
    vint32m2_t n = __riscv_vfcvt_rtz_x_f_v_i32m2(y_rounded, vl);
    const vfloat32m2_t nf = __riscv_vfcvt_f_x_v_f32m2(n, vl);
    const vfloat32m2_t f = __riscv_vfsub_vv_f32m2(y, nf, vl);
    n = __riscv_vmax_vx_i32m2(n, -126, vl);
    n = __riscv_vmin_vx_i32m2(n, 127, vl);
    const vint32m2_t exp_bits = __riscv_vsll_vx_i32m2(
        __riscv_vadd_vx_i32m2(n, 127, vl), 23, vl);
    const vfloat32m2_t pow2n =
        __riscv_vreinterpret_v_i32m2_f32m2(exp_bits);
    vfloat32m2_t p = __riscv_vfmul_vf_f32m2(f, kC5, vl);
    p = __riscv_vfadd_vf_f32m2(p, kC4, vl);
    p = __riscv_vfmul_vv_f32m2(p, f, vl);
    p = __riscv_vfadd_vf_f32m2(p, kC3, vl);
    p = __riscv_vfmul_vv_f32m2(p, f, vl);
    p = __riscv_vfadd_vf_f32m2(p, kC2, vl);
    p = __riscv_vfmul_vv_f32m2(p, f, vl);
    p = __riscv_vfadd_vf_f32m2(p, kC1, vl);
    p = __riscv_vfmul_vv_f32m2(p, f, vl);
    p = __riscv_vfadd_vf_f32m2(p, 1.0f, vl);
    const vfloat32m2_t exp_neg_x =
        __riscv_vfmul_vv_f32m2(pow2n, p, vl);
    const vfloat32m2_t one = __riscv_vfmv_v_f_f32m2(1.0f, vl);
    const vfloat32m2_t denom = __riscv_vfadd_vv_f32m2(exp_neg_x, one, vl);
    const vfloat32m2_t result = __riscv_vfdiv_vv_f32m2(one, denom, vl);
    __riscv_vse32_v_f32m2(output + offset, result, vl);
    offset += static_cast<int>(vl);
  }
}

inline void RvvLogisticFloat(const float* input, float* output, int size) {
  // VLEN=256 m2 regresses on the QEMU A/B; reserve the wider path for
  // VLEN=512, where repeated EfficientDet FP32 A/B is positive.
  if (tensor_utils::RvvVlenClassBits() >= 512) {
    RvvLogisticFloatM2(input, output, size);
    return;
  }
  constexpr float kLog2e = 1.4426950408889634f;  // log2(e)
  // Degree-5 minimax-ish approximation of 2^f for f in [-0.5, 0.5].
  constexpr float kC1 = 0.6931471805599453f;   // ln(2)
  constexpr float kC2 = 0.2402265069591007f;   // ln(2)^2 / 2
  constexpr float kC3 = 0.05550410866482158f;  // ln(2)^3 / 6
  constexpr float kC4 = 0.009618129107628477f;
  constexpr float kC5 = 0.0013333558146428443f;
  int offset = 0;
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e32m1(size - offset);
    const vfloat32m1_t x =
        __riscv_vle32_v_f32m1(input + offset, vl);
    // exp(-x) = 2^(-x*log2e). Build n = round(-x*log2e), f = frac part.
    const vfloat32m1_t y = __riscv_vfmul_vf_f32m1(x, -kLog2e, vl);
    // Round-to-nearest (ties away) via truncation after +/-0.5.
    const vfloat32m1_t half =
        __riscv_vfmv_v_f_f32m1(0.5f, vl);
    const vfloat32m1_t y_rounded = __riscv_vfadd_vv_f32m1(
        y,
        __riscv_vfsgnj_vv_f32m1(half, y, vl),  // copysign(0.5, y)
        vl);
    vint32m1_t n = __riscv_vfcvt_rtz_x_f_v_i32m1(y_rounded, vl);
    const vfloat32m1_t nf = __riscv_vfcvt_f_x_v_f32m1(n, vl);
    const vfloat32m1_t f = __riscv_vfsub_vv_f32m1(y, nf, vl);
    // Clamp the exponent so the bit-built 2^n stays in normal float range.
    n = __riscv_vmax_vx_i32m1(n, -126, vl);
    n = __riscv_vmin_vx_i32m1(n, 127, vl);
    // 2^n via IEEE-754 bit construction: exponent field = n + 127.
    const vint32m1_t exp_bits =
        __riscv_vsll_vx_i32m1(__riscv_vadd_vx_i32m1(n, 127, vl), 23, vl);
    const vfloat32m1_t pow2n =
        __riscv_vreinterpret_v_i32m1_f32m1(exp_bits);
    // Polynomial for 2^f.
    vfloat32m1_t p = __riscv_vfmul_vf_f32m1(f, kC5, vl);
    p = __riscv_vfadd_vf_f32m1(p, kC4, vl);
    p = __riscv_vfmul_vv_f32m1(p, f, vl);
    p = __riscv_vfadd_vf_f32m1(p, kC3, vl);
    p = __riscv_vfmul_vv_f32m1(p, f, vl);
    p = __riscv_vfadd_vf_f32m1(p, kC2, vl);
    p = __riscv_vfmul_vv_f32m1(p, f, vl);
    p = __riscv_vfadd_vf_f32m1(p, kC1, vl);
    p = __riscv_vfmul_vv_f32m1(p, f, vl);
    p = __riscv_vfadd_vf_f32m1(p, 1.0f, vl);
    const vfloat32m1_t exp_neg_x =
        __riscv_vfmul_vv_f32m1(pow2n, p, vl);
    // sigmoid(x) = 1 / (1 + exp(-x)).
    const vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    const vfloat32m1_t denom = __riscv_vfadd_vv_f32m1(exp_neg_x, one, vl);
    const vfloat32m1_t result = __riscv_vfdiv_vv_f32m1(one, denom, vl);
    __riscv_vse32_v_f32m1(output + offset, result, vl);
    offset += static_cast<int>(vl);
  }
}

template <int kRows>
inline void RvvGemmFloatChannelTile(const float* lhs, int lhs_rows, int depth,
                                    const float* rhs, int cols, float* dst,
                                    const float* bias, float clamp_min,
                                    float clamp_max, int row_block) {
  // _tu keeps earlier K chunks in inactive lanes of the final partial chunk.
  const size_t vlmax = __riscv_vsetvl_e32m2(depth);
  const size_t seed_vl = __riscv_vsetvl_e32m1(1);
  const vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0.0f, seed_vl);
  const float* lhs0 = lhs + row_block * depth;
  const float* lhs1 = kRows > 1 ? lhs0 + depth : lhs0;
  const float* lhs2 = kRows > 2 ? lhs1 + depth : lhs1;
  const float* lhs3 = kRows > 3 ? lhs2 + depth : lhs2;
  const float b0 = bias == nullptr ? 0.0f : bias[row_block];
  const float b1 = kRows > 1 && bias != nullptr ? bias[row_block + 1] : 0.0f;
  const float b2 = kRows > 2 && bias != nullptr ? bias[row_block + 2] : 0.0f;
  const float b3 = kRows > 3 && bias != nullptr ? bias[row_block + 3] : 0.0f;
  for (int col = 0; col < cols; ++col) {
    vfloat32m2_t acc0 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t acc1 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t acc2 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t acc3 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    const float* rhs_col = rhs + col * depth;
    int offset = 0;
    while (offset < depth) {
      const size_t remaining = depth - offset;
      const size_t vl = remaining < vlmax
                            ? __riscv_vsetvl_e32m2(remaining)
                            : vlmax;
      const vfloat32m2_t rhs_v = __riscv_vle32_v_f32m2(rhs_col + offset, vl);
      acc0 = __riscv_vfmacc_vv_f32m2_tu(
          acc0, __riscv_vle32_v_f32m2(lhs0 + offset, vl), rhs_v, vl);
      if constexpr (kRows > 1)
        acc1 = __riscv_vfmacc_vv_f32m2_tu(
            acc1, __riscv_vle32_v_f32m2(lhs1 + offset, vl), rhs_v, vl);
      if constexpr (kRows > 2)
        acc2 = __riscv_vfmacc_vv_f32m2_tu(
            acc2, __riscv_vle32_v_f32m2(lhs2 + offset, vl), rhs_v, vl);
      if constexpr (kRows > 3)
        acc3 = __riscv_vfmacc_vv_f32m2_tu(
            acc3, __riscv_vle32_v_f32m2(lhs3 + offset, vl), rhs_v, vl);
      offset += static_cast<int>(vl);
    }
    const float sum0 = b0 + __riscv_vfmv_f_s_f32m1_f32(
                                  __riscv_vfredusum_vs_f32m2_f32m1(
                                      acc0, zero, vlmax));
    dst[row_block + col * lhs_rows] =
        std::max(clamp_min, std::min(clamp_max, sum0));
    if constexpr (kRows > 1) {
      const float sum1 = b1 + __riscv_vfmv_f_s_f32m1_f32(
                                    __riscv_vfredusum_vs_f32m2_f32m1(
                                        acc1, zero, vlmax));
      dst[row_block + 1 + col * lhs_rows] =
          std::max(clamp_min, std::min(clamp_max, sum1));
    }
    if constexpr (kRows > 2) {
      const float sum2 = b2 + __riscv_vfmv_f_s_f32m1_f32(
                                    __riscv_vfredusum_vs_f32m2_f32m1(
                                        acc2, zero, vlmax));
      dst[row_block + 2 + col * lhs_rows] =
          std::max(clamp_min, std::min(clamp_max, sum2));
    }
    if constexpr (kRows > 3) {
      const float sum3 = b3 + __riscv_vfmv_f_s_f32m1_f32(
                                    __riscv_vfredusum_vs_f32m2_f32m1(
                                        acc3, zero, vlmax));
      dst[row_block + 3 + col * lhs_rows] =
          std::max(clamp_min, std::min(clamp_max, sum3));
    }
  }
}

inline void RvvGemmFloatChannelTiled(const float* lhs, int lhs_rows, int depth,
                                     const float* rhs, int cols, float* dst,
                                     const float* bias, float clamp_min,
                                     float clamp_max) {
  const int full_rows = lhs_rows & ~3;
  for (int row_block = 0; row_block < full_rows; row_block += 4)
    RvvGemmFloatChannelTile<4>(lhs, lhs_rows, depth, rhs, cols, dst, bias,
                               clamp_min, clamp_max, row_block);
  for (int row = full_rows; row < lhs_rows; ++row)
    RvvGemmFloatChannelTile<1>(lhs, lhs_rows, depth, rhs, cols, dst, bias,
                               clamp_min, clamp_max, row);
}

inline void RvvGemmFloat2x2(const float* lhs, int lhs_rows, int depth,
                            const float* rhs, int cols, float* dst,
                            const float* bias, float clamp_min,
                            float clamp_max) {
  const size_t vlmax = __riscv_vsetvl_e32m2(depth);
  const vfloat32m1_t zero =
      __riscv_vfmv_s_f_f32m1(0.0f, __riscv_vsetvl_e32m1(1));
  const auto clamp = [&](float value) {
    return std::max(clamp_min, std::min(clamp_max, value));
  };
  const auto run_single = [&](int row, int col) {
    vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    for (int offset = 0; offset < depth;) {
      const size_t vl = depth - offset < vlmax
                            ? __riscv_vsetvl_e32m2(depth - offset)
                            : vlmax;
      acc = __riscv_vfmacc_vv_f32m2_tu(
          acc, __riscv_vle32_v_f32m2(lhs + row * depth + offset, vl),
          __riscv_vle32_v_f32m2(rhs + col * depth + offset, vl), vl);
      offset += static_cast<int>(vl);
    }
    float value = bias == nullptr ? 0.0f : bias[row];
    value += __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m2_f32m1(acc, zero, vlmax));
    dst[row + col * lhs_rows] = clamp(value);
  };
  const auto run_pair = [&](int row, int col) {
    vfloat32m2_t acc00 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t acc01 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t acc10 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    vfloat32m2_t acc11 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
    for (int offset = 0; offset < depth;) {
      const size_t vl = depth - offset < vlmax
                            ? __riscv_vsetvl_e32m2(depth - offset)
                            : vlmax;
      const vfloat32m2_t lhs0 =
          __riscv_vle32_v_f32m2(lhs + row * depth + offset, vl);
      const vfloat32m2_t lhs1 =
          __riscv_vle32_v_f32m2(lhs + (row + 1) * depth + offset, vl);
      const vfloat32m2_t rhs0 =
          __riscv_vle32_v_f32m2(rhs + col * depth + offset, vl);
      const vfloat32m2_t rhs1 =
          __riscv_vle32_v_f32m2(rhs + (col + 1) * depth + offset, vl);
      acc00 = __riscv_vfmacc_vv_f32m2_tu(acc00, lhs0, rhs0, vl);
      acc01 = __riscv_vfmacc_vv_f32m2_tu(acc01, lhs1, rhs0, vl);
      acc10 = __riscv_vfmacc_vv_f32m2_tu(acc10, lhs0, rhs1, vl);
      acc11 = __riscv_vfmacc_vv_f32m2_tu(acc11, lhs1, rhs1, vl);
      offset += static_cast<int>(vl);
    }
    const auto reduce = [&](vfloat32m2_t acc, int out_row) {
      float value = bias == nullptr ? 0.0f : bias[out_row];
      value += __riscv_vfmv_f_s_f32m1_f32(
          __riscv_vfredusum_vs_f32m2_f32m1(acc, zero, vlmax));
      return clamp(value);
    };
    dst[row + col * lhs_rows] = reduce(acc00, row);
    dst[row + 1 + col * lhs_rows] = reduce(acc01, row + 1);
    dst[row + (col + 1) * lhs_rows] = reduce(acc10, row);
    dst[row + 1 + (col + 1) * lhs_rows] = reduce(acc11, row + 1);
  };
  for (int row = 0; row + 1 < lhs_rows; row += 2)
    for (int col = 0; col + 1 < cols; col += 2) run_pair(row, col);
  if (cols & 1)
    for (int row = 0; row + 1 < lhs_rows; row += 2) {
      run_single(row, cols - 1);
      run_single(row + 1, cols - 1);
    }
  if (lhs_rows & 1)
    for (int col = 0; col < cols; ++col) run_single(lhs_rows - 1, col);
}
inline void RvvGemmFloat(const float* lhs, int lhs_rows, int depth,
                         const float* rhs, int cols, float* dst,
                         const float* bias, float clamp_min,
                         float clamp_max) {
  if (tensor_utils::RvvVlenClassBits() >= 256 && lhs_rows >= 8 &&
      depth >= 96 && cols >= 2 && cols <= 64) {
    RvvGemmFloat2x2(lhs, lhs_rows, depth, rhs, cols, dst, bias, clamp_min,
                    clamp_max);
    return;
  }
  // Use the unordered reduction to enable the hardware tree; FP32 inference
  // tolerates its reassociated accumulation.
#if !defined(TFLITE_RVV_GEMM_ROWOUTER)
  if (depth < 16) {
    for (int col = 0; col < cols; ++col) {
      for (int row = 0; row < lhs_rows; ++row) {
        float sum = bias == nullptr ? 0.0f : bias[row];
        int offset = 0;
        for (; offset + 3 < depth; offset += 4) {
          sum += lhs[row * depth + offset] * rhs[col * depth + offset];
          sum += lhs[row * depth + offset + 1] * rhs[col * depth + offset + 1];
          sum += lhs[row * depth + offset + 2] * rhs[col * depth + offset + 2];
          sum += lhs[row * depth + offset + 3] * rhs[col * depth + offset + 3];
        }
        for (; offset < depth; ++offset)
          sum += lhs[row * depth + offset] * rhs[col * depth + offset];
        dst[row + col * lhs_rows] =
            std::max(clamp_min, std::min(clamp_max, sum));
      }
    }
    return;
  }
#endif

  // Share each RHS vector across four output channels for large products. The
  // tail block handles 1-3 remaining rows; small products keep the simpler
  // single-output path below.
#if !defined(TFLITE_RVV_GEMM_ROWOUTER)
  if (tensor_utils::RvvVlenClassBits() >= 256 && depth >= 16 && lhs_rows >= 4 &&
      static_cast<int64_t>(lhs_rows) * depth * cols >= 32768) {
    RvvGemmFloatChannelTiled(lhs, lhs_rows, depth, rhs, cols, dst, bias,
                             clamp_min, clamp_max);
    return;
  }
#endif

  // Experimental row-outer variant: hoist the lhs row vector load out of the
  // column loop so each lhs row is loaded from memory exactly once and reused
  // across all columns, and process kColBlock output columns per pass to
  // amortize the reduction. On VLEN>=256 use m2
  // registers; VLEN=128 keeps m1. Disabled by default: measured regression on
  // Spacemit X60 (FP32 GEMM is vfmacc-throughput-bound, extra register
  // pressure from multiple accumulators hurts), kept for QEMU functional
  // verification and for A/B retesting on other RV64GCV hardware.
#if defined(TFLITE_RVV_GEMM_ROWOUTER)
  constexpr int kColBlock = 2;
  if (tensor_utils::RvvVlenClassBits() >= 256) {
    const size_t vlmax = __riscv_vsetvl_e32m2(depth);
    for (int row = 0; row < lhs_rows; ++row) {
      int col = 0;
      for (; col + kColBlock <= cols; col += kColBlock) {
        vfloat32m2_t accum0 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
        vfloat32m2_t accum1 = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
        int depth_offset = 0;
        while (depth_offset < depth) {
          const size_t remaining = depth - depth_offset;
          const size_t vl = remaining < vlmax
                                ? __riscv_vsetvl_e32m2(remaining)
                                : vlmax;
          const vfloat32m2_t lhs_v = __riscv_vle32_v_f32m2(
              lhs + row * depth + depth_offset, vl);
          accum0 = __riscv_vfmacc_vv_f32m2_tu(
              accum0, lhs_v,
              __riscv_vle32_v_f32m2(rhs + (col + 0) * depth + depth_offset, vl),
              vl);
          accum1 = __riscv_vfmacc_vv_f32m2_tu(
              accum1, lhs_v,
              __riscv_vle32_v_f32m2(rhs + (col + 1) * depth + depth_offset, vl),
              vl);
          depth_offset += static_cast<int>(vl);
        }
        const vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0.0f, vlmax);
        float sum0 = bias == nullptr ? 0.0f : bias[row];
        float sum1 = sum0;
        sum0 += __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m2_f32m1(accum0, zero, vlmax));
        sum1 += __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m2_f32m1(accum1, zero, vlmax));
        dst[row + (col + 0) * lhs_rows] =
            std::max(clamp_min, std::min(clamp_max, sum0));
        dst[row + (col + 1) * lhs_rows] =
            std::max(clamp_min, std::min(clamp_max, sum1));
      }
      for (; col < cols; ++col) {
        float sum = bias == nullptr ? 0.0f : bias[row];
        vfloat32m2_t accum = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
        int depth_offset = 0;
        while (depth_offset < depth) {
          const size_t remaining = depth - depth_offset;
          const size_t vl = remaining < vlmax
                                ? __riscv_vsetvl_e32m2(remaining)
                                : vlmax;
          accum = __riscv_vfmacc_vv_f32m2_tu(
              accum, __riscv_vle32_v_f32m2(lhs + row * depth + depth_offset,
                                           vl),
              __riscv_vle32_v_f32m2(rhs + col * depth + depth_offset, vl), vl);
          depth_offset += static_cast<int>(vl);
        }
        const vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0.0f, vlmax);
        sum += __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m2_f32m1(accum, zero, vlmax));
        dst[row + col * lhs_rows] =
            std::max(clamp_min, std::min(clamp_max, sum));
      }
    }
    return;
  } else {
    const size_t vlmax = __riscv_vsetvl_e32m1(depth);
    for (int row = 0; row < lhs_rows; ++row) {
      int col = 0;
      for (; col + kColBlock <= cols; col += kColBlock) {
        vfloat32m1_t accum0 = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
        vfloat32m1_t accum1 = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
        int depth_offset = 0;
        while (depth_offset < depth) {
          const size_t remaining = depth - depth_offset;
          const size_t vl = remaining < vlmax
                                ? __riscv_vsetvl_e32m1(remaining)
                                : vlmax;
          const vfloat32m1_t lhs_v = __riscv_vle32_v_f32m1(
              lhs + row * depth + depth_offset, vl);
          accum0 = __riscv_vfmacc_vv_f32m1_tu(
              accum0, lhs_v,
              __riscv_vle32_v_f32m1(rhs + (col + 0) * depth + depth_offset, vl),
              vl);
          accum1 = __riscv_vfmacc_vv_f32m1_tu(
              accum1, lhs_v,
              __riscv_vle32_v_f32m1(rhs + (col + 1) * depth + depth_offset, vl),
              vl);
          depth_offset += static_cast<int>(vl);
        }
        const vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0.0f, vlmax);
        float sum0 = bias == nullptr ? 0.0f : bias[row];
        float sum1 = sum0;
        sum0 += __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m1_f32m1(accum0, zero, vlmax));
        sum1 += __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m1_f32m1(accum1, zero, vlmax));
        dst[row + (col + 0) * lhs_rows] =
            std::max(clamp_min, std::min(clamp_max, sum0));
        dst[row + (col + 1) * lhs_rows] =
            std::max(clamp_min, std::min(clamp_max, sum1));
      }
      for (; col < cols; ++col) {
        float sum = bias == nullptr ? 0.0f : bias[row];
        vfloat32m1_t accum = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
        int depth_offset = 0;
        while (depth_offset < depth) {
          const size_t remaining = depth - depth_offset;
          const size_t vl = remaining < vlmax
                                ? __riscv_vsetvl_e32m1(remaining)
                                : vlmax;
          accum = __riscv_vfmacc_vv_f32m1_tu(
              accum, __riscv_vle32_v_f32m1(lhs + row * depth + depth_offset,
                                           vl),
              __riscv_vle32_v_f32m1(rhs + col * depth + depth_offset, vl), vl);
          depth_offset += static_cast<int>(vl);
        }
        const vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0.0f, vlmax);
        sum += __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m1_f32m1(accum, zero, vlmax));
        dst[row + col * lhs_rows] =
            std::max(clamp_min, std::min(clamp_max, sum));
      }
    }
    return;
  }
#endif  // TFLITE_RVV_GEMM_ROWOUTER

  // Runtime dispatch on hardware VLEN: on VLEN>=256 use a wider register
  // group (m2) so each vector iteration covers more elements and reduces
  // loop/instruction overhead; VLEN=128 keeps the m1 path that best fits the
  // 128-bit registers.
  if (tensor_utils::RvvVlenClassBits() >= 256) {
    const size_t vlmax = __riscv_vsetvl_e32m2(depth);
    for (int col = 0; col < cols; ++col) {
      for (int row = 0; row < lhs_rows; ++row) {
        float sum = bias == nullptr ? 0.0f : bias[row];
        vfloat32m2_t accum = __riscv_vfmv_v_f_f32m2(0.0f, vlmax);
        int depth_offset = 0;
        while (depth_offset < depth) {
          const size_t remaining = depth - depth_offset;
          const size_t vl = remaining < vlmax
                                ? __riscv_vsetvl_e32m2(remaining)
                                : vlmax;
          accum = __riscv_vfmacc_vv_f32m2_tu(
              accum, __riscv_vle32_v_f32m2(lhs + row * depth + depth_offset,
                                           vl),
              __riscv_vle32_v_f32m2(rhs + col * depth + depth_offset, vl), vl);
          depth_offset += static_cast<int>(vl);
        }
        const vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0.0f, vlmax);
        const vfloat32m1_t reduced =
            __riscv_vfredusum_vs_f32m2_f32m1(accum, zero, vlmax);
        sum += __riscv_vfmv_f_s_f32m1_f32(reduced);
        dst[row + col * lhs_rows] =
            std::max(clamp_min, std::min(clamp_max, sum));
      }
    }
  } else {
    const size_t vlmax = __riscv_vsetvl_e32m1(depth);
    for (int col = 0; col < cols; ++col) {
      for (int row = 0; row < lhs_rows; ++row) {
        float sum = bias == nullptr ? 0.0f : bias[row];
        vfloat32m1_t accum = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
        int depth_offset = 0;
        while (depth_offset < depth) {
          const size_t remaining = depth - depth_offset;
          const size_t vl = remaining < vlmax
                                ? __riscv_vsetvl_e32m1(remaining)
                                : vlmax;
          accum = __riscv_vfmacc_vv_f32m1_tu(
              accum, __riscv_vle32_v_f32m1(lhs + row * depth + depth_offset,
                                           vl),
              __riscv_vle32_v_f32m1(rhs + col * depth + depth_offset, vl), vl);
          depth_offset += static_cast<int>(vl);
        }
        const vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0.0f, vlmax);
        const vfloat32m1_t reduced =
            __riscv_vfredusum_vs_f32m1_f32m1(accum, zero, vlmax);
        sum += __riscv_vfmv_f_s_f32m1_f32(reduced);
        dst[row + col * lhs_rows] =
            std::max(clamp_min, std::min(clamp_max, sum));
      }
    }
  }
}

inline void RvvGemmQuantized2x2Int8(
    const int8_t* lhs, int lhs_rows, int depth, int lhs_zero_point,
    const int8_t* rhs, int cols, int rhs_zero_point, int8_t* dst,
    const int32_t* bias, int32_t multiplier, int multiplier_shift,
    const int32_t* multiplier_perchannel, const int* shift_perchannel,
    int32_t output_zero_point, int8_t clamp_min, int8_t clamp_max) {
  const size_t vlmax = __riscv_vsetvl_e8m1(depth);
  // Keep the full-depth VL across complete chunks; only a short tail needs a
  // new vsetvl, which removes one CSR setup per chunk on pointwise GEMMs.
  const auto quantize = [&](int32_t sum, int row) {
    const int32_t row_multiplier = multiplier_perchannel == nullptr
                                       ? multiplier
                                       : multiplier_perchannel[row];
    const int row_shift = shift_perchannel == nullptr
                              ? multiplier_shift
                              : shift_perchannel[row];
    int32_t value = MultiplyByRuyQuantizedMultiplier(
        sum, row_multiplier, row_shift);
    value += output_zero_point;
    return std::max(static_cast<int32_t>(clamp_min),
                    std::min(static_cast<int32_t>(clamp_max), value));
  };
  const auto load = [&](const int8_t* data, int offset, size_t vl) {
    vint16m2_t value = __riscv_vsext_vf2_i16m2(
        __riscv_vle8_v_i8m1(data + offset, vl), vl);
    return value;
  };
  const auto run_single = [&](int row, int col) {
    vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vlmax);
    int32_t sum = bias == nullptr ? 0 : bias[row];
    for (int offset = 0; offset < depth;) {
      const size_t vl = depth - offset < vlmax
                            ? __riscv_vsetvl_e8m1(depth - offset)
                            : vlmax;
      vint16m2_t lhs16 = load(lhs + row * depth, offset, vl);
      vint16m2_t rhs16 = load(rhs + col * depth, offset, vl);
      if (lhs_zero_point != 0)
        lhs16 = __riscv_vsub_vx_i16m2(lhs16, lhs_zero_point, vl);
      if (rhs_zero_point != 0)
        rhs16 = __riscv_vsub_vx_i16m2(rhs16, rhs_zero_point, vl);
      acc = __riscv_vwmacc_vv_i32m4_tu(acc, lhs16, rhs16, vl);
      offset += static_cast<int>(vl);
    }
    const vint32m1_t zero = __riscv_vmv_s_x_i32m1(0, vlmax);
    sum += __riscv_vmv_x_s_i32m1_i32(
        __riscv_vredsum_vs_i32m4_i32m1(acc, zero, vlmax));
    dst[row + col * lhs_rows] = static_cast<int8_t>(quantize(sum, row));
  };
  const auto run_pair = [&](int row, int col) {
    vint32m4_t acc00 = __riscv_vmv_v_x_i32m4(0, vlmax);
    vint32m4_t acc01 = __riscv_vmv_v_x_i32m4(0, vlmax);
    vint32m4_t acc10 = __riscv_vmv_v_x_i32m4(0, vlmax);
    vint32m4_t acc11 = __riscv_vmv_v_x_i32m4(0, vlmax);
    for (int offset = 0; offset < depth;) {
      const size_t vl = depth - offset < vlmax
                            ? __riscv_vsetvl_e8m1(depth - offset)
                            : vlmax;
      vint16m2_t lhs0 = load(lhs + row * depth, offset, vl);
      vint16m2_t lhs1 = load(lhs + (row + 1) * depth, offset, vl);
      vint16m2_t rhs0 = load(rhs + col * depth, offset, vl);
      vint16m2_t rhs1 = load(rhs + (col + 1) * depth, offset, vl);
      if (lhs_zero_point != 0) {
        lhs0 = __riscv_vsub_vx_i16m2(lhs0, lhs_zero_point, vl);
        lhs1 = __riscv_vsub_vx_i16m2(lhs1, lhs_zero_point, vl);
      }
      if (rhs_zero_point != 0) {
        rhs0 = __riscv_vsub_vx_i16m2(rhs0, rhs_zero_point, vl);
        rhs1 = __riscv_vsub_vx_i16m2(rhs1, rhs_zero_point, vl);
      }
      acc00 = __riscv_vwmacc_vv_i32m4_tu(acc00, lhs0, rhs0, vl);
      acc01 = __riscv_vwmacc_vv_i32m4_tu(acc01, lhs1, rhs0, vl);
      acc10 = __riscv_vwmacc_vv_i32m4_tu(acc10, lhs0, rhs1, vl);
      acc11 = __riscv_vwmacc_vv_i32m4_tu(acc11, lhs1, rhs1, vl);
      offset += static_cast<int>(vl);
    }
    const vint32m1_t zero = __riscv_vmv_s_x_i32m1(0, vlmax);
    const auto reduce = [&](vint32m4_t acc, int out_row) {
      int32_t sum = bias == nullptr ? 0 : bias[out_row];
      sum += __riscv_vmv_x_s_i32m1_i32(
          __riscv_vredsum_vs_i32m4_i32m1(acc, zero, vlmax));
      return static_cast<int8_t>(quantize(sum, out_row));
    };
    dst[row + col * lhs_rows] = reduce(acc00, row);
    dst[row + 1 + col * lhs_rows] = reduce(acc01, row + 1);
    dst[row + (col + 1) * lhs_rows] = reduce(acc10, row);
    dst[row + 1 + (col + 1) * lhs_rows] = reduce(acc11, row + 1);
  };
  // When K fits in one vector, keep four lhs and two rhs vectors live and
  // emit eight outputs before reloading either operand.
  if (depth <= vlmax) {
    const vint32m1_t zero = __riscv_vmv_s_x_i32m1(0, vlmax);
    const auto emit = [&](vint16m2_t lhs_v, vint16m2_t rhs_v, int row,
                          int col) {
      vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vlmax);
      acc = __riscv_vwmacc_vv_i32m4_tu(acc, lhs_v, rhs_v, vlmax);
      int32_t sum = bias == nullptr ? 0 : bias[row];
      sum += __riscv_vmv_x_s_i32m1_i32(
          __riscv_vredsum_vs_i32m4_i32m1(acc, zero, vlmax));
      dst[row + col * lhs_rows] = static_cast<int8_t>(quantize(sum, row));
    };
    const auto run_four_by_two = [&](int row, int col) {
      vint16m2_t lhs0 = load(lhs + row * depth, 0, vlmax);
      vint16m2_t lhs1 = load(lhs + (row + 1) * depth, 0, vlmax);
      vint16m2_t lhs2 = load(lhs + (row + 2) * depth, 0, vlmax);
      vint16m2_t lhs3 = load(lhs + (row + 3) * depth, 0, vlmax);
      vint16m2_t rhs0 = load(rhs + col * depth, 0, vlmax);
      vint16m2_t rhs1 = load(rhs + (col + 1) * depth, 0, vlmax);
      if (lhs_zero_point != 0) {
        lhs0 = __riscv_vsub_vx_i16m2(lhs0, lhs_zero_point, vlmax);
        lhs1 = __riscv_vsub_vx_i16m2(lhs1, lhs_zero_point, vlmax);
        lhs2 = __riscv_vsub_vx_i16m2(lhs2, lhs_zero_point, vlmax);
        lhs3 = __riscv_vsub_vx_i16m2(lhs3, lhs_zero_point, vlmax);
      }
      if (rhs_zero_point != 0) {
        rhs0 = __riscv_vsub_vx_i16m2(rhs0, rhs_zero_point, vlmax);
        rhs1 = __riscv_vsub_vx_i16m2(rhs1, rhs_zero_point, vlmax);
      }
      emit(lhs0, rhs0, row, col);
      emit(lhs1, rhs0, row + 1, col);
      emit(lhs2, rhs0, row + 2, col);
      emit(lhs3, rhs0, row + 3, col);
      emit(lhs0, rhs1, row, col + 1);
      emit(lhs1, rhs1, row + 1, col + 1);
      emit(lhs2, rhs1, row + 2, col + 1);
      emit(lhs3, rhs1, row + 3, col + 1);
    };
    const int full_rows = lhs_rows & ~3;
    const int full_cols = cols & ~1;
    for (int row = 0; row < full_rows; row += 4)
      for (int col = 0; col < full_cols; col += 2)
        run_four_by_two(row, col);
    for (int row = full_rows; row < lhs_rows; ++row)
      for (int col = 0; col < cols; ++col) run_single(row, col);
    if (cols & 1)
      for (int row = 0; row < full_rows; ++row) run_single(row, cols - 1);
    return;
  }
  for (int row = 0; row + 1 < lhs_rows; row += 2)
    for (int col = 0; col + 1 < cols; col += 2) run_pair(row, col);
  if (cols & 1)
    for (int row = 0; row + 1 < lhs_rows; row += 2)
      run_single(row, cols - 1), run_single(row + 1, cols - 1);
  if (lhs_rows & 1)
    for (int col = 0; col < cols; ++col) run_single(lhs_rows - 1, col);
}
template <typename LhsScalar, typename RhsScalar, typename DstScalar>
inline void RvvGemmQuantized(
    const LhsScalar* lhs, int lhs_rows, int depth, int lhs_zero_point,
    const RhsScalar* rhs, int cols, int rhs_zero_point, DstScalar* dst,
    const int32_t* bias, int32_t multiplier, int multiplier_shift,
    const int32_t* multiplier_perchannel, const int* shift_perchannel,
    int32_t output_zero_point, DstScalar clamp_min, DstScalar clamp_max) {
  if constexpr (std::is_same_v<LhsScalar, int8_t> &&
                std::is_same_v<RhsScalar, int8_t> &&
                std::is_same_v<DstScalar, int8_t>) {
    if (tensor_utils::RvvVlenClassBits() >= 256 && lhs_rows >= 8 &&
        depth >= 16 && cols >= 2) {
      RvvGemmQuantized2x2Int8(
          lhs, lhs_rows, depth, lhs_zero_point, rhs, cols, rhs_zero_point,
          dst, bias, multiplier, multiplier_shift, multiplier_perchannel,
          shift_perchannel, output_zero_point, clamp_min, clamp_max);
      return;
    }
  }
  const auto quantize = [&](int32_t sum, int row) {
    const int32_t row_multiplier = multiplier_perchannel == nullptr
                                       ? multiplier
                                       : multiplier_perchannel[row];
    const int row_shift = shift_perchannel == nullptr
                              ? multiplier_shift
                              : shift_perchannel[row];
    int32_t value;
    if constexpr (std::is_same_v<LhsScalar, uint8_t> &&
                  std::is_same_v<RhsScalar, uint8_t>) {
      // Non-Ruy uint8 GEMM uses LiteRT's gemmlowp-compatible multiplier.
      value = MultiplyByQuantizedMultiplier(sum, row_multiplier, row_shift);
    } else {
      // Ruy-backed int8 GEMM uses the single-rounding contract above.
      value = MultiplyByRuyQuantizedMultiplier(sum, row_multiplier, row_shift);
    }
    value += output_zero_point;
    return std::max(static_cast<int32_t>(clamp_min),
                    std::min(static_cast<int32_t>(clamp_max), value));
  };

  // Keep the m1 -> m2 -> m4 register-group chain for all VLENs. On VLEN=256
  // hardware (e.g. Spacemit X60) a wider m8 accumulator exhausts the vector
  // register file and *regresses* INT8 GEMM (measured 0.63-0.94x on k1), so
  // the quantized path intentionally does not use VLEN-based widening. The
  // FP32 path does dispatch on VLEN (m2 for VLEN>=256) where it measured a
  // 1.04-1.29x win on k1.
  // Row-blocked GEMM: for a fixed output column, process kRowBlock rows at a
  // time so the rhs column vector (load + zero-point subtraction + widening)
  // is amortized across rows. This reduces per-output-element vsetvl /
  // load / reduce overhead and measurably speeds up the CONV_2D pointwise
  // GEMMs that dominate the benchmark models.
  constexpr int kRowBlock = 4;
  // Accumulator initialization restores the full VL for each block; keep it
  // for complete depth chunks and only reprogram VL for a tail chunk.
  // depth and VLEN are invariant for this GEMM; reuse the lane count for all
  // row blocks and scalar tails instead of rereading the vlenb-derived VL.
  const size_t vlmax = __riscv_vsetvl_e8m1(depth);
  // Keep the accumulation vectorized, but use scalar output requantization:
  // A210 A/B measured the four-lane vector path 20.3% slower (141.5 ms to
  // 170.1 ms at seven threads) despite identical output bytes.
  constexpr bool use_vectorized_4row_quantize = false;
  for (int col = 0; col < cols; ++col) {
    int row = 0;
    for (; row + kRowBlock <= lhs_rows; row += kRowBlock) {
      vint32m4_t accum0 = __riscv_vmv_v_x_i32m4(0, vlmax);
      vint32m4_t accum1 = __riscv_vmv_v_x_i32m4(0, vlmax);
      vint32m4_t accum2 = __riscv_vmv_v_x_i32m4(0, vlmax);
      vint32m4_t accum3 = __riscv_vmv_v_x_i32m4(0, vlmax);
      int32_t sum0 = bias == nullptr ? 0 : bias[row + 0];
      int32_t sum1 = bias == nullptr ? 0 : bias[row + 1];
      int32_t sum2 = bias == nullptr ? 0 : bias[row + 2];
      int32_t sum3 = bias == nullptr ? 0 : bias[row + 3];
      int depth_offset = 0;
      while (depth_offset < depth) {
        const size_t remaining = depth - depth_offset;
        const size_t vl = remaining < vlmax
                              ? __riscv_vsetvl_e8m1(remaining)
                              : vlmax;
        vint16m2_t rhs16;
        if constexpr (std::is_same_v<RhsScalar, int8_t>) {
          rhs16 = __riscv_vsext_vf2_i16m2(
              __riscv_vle8_v_i8m1(rhs + col * depth + depth_offset, vl), vl);
          if (rhs_zero_point != 0) {
            rhs16 = __riscv_vsub_vx_i16m2(rhs16, rhs_zero_point, vl);
          }
        } else {
          rhs16 = __riscv_vreinterpret_v_u16m2_i16m2(
              __riscv_vsub_vx_u16m2(
                  __riscv_vzext_vf2_u16m2(
                      __riscv_vle8_v_u8m1(rhs + col * depth + depth_offset,
                                          vl),
                      vl),
                  static_cast<uint16_t>(rhs_zero_point), vl));
        }
        vint16m2_t lhs16_0;
        vint16m2_t lhs16_1;
        vint16m2_t lhs16_2;
        vint16m2_t lhs16_3;
        if constexpr (std::is_same_v<LhsScalar, int8_t>) {
          lhs16_0 = __riscv_vsext_vf2_i16m2(
              __riscv_vle8_v_i8m1(lhs + row * depth + depth_offset, vl), vl);
          lhs16_1 = __riscv_vsext_vf2_i16m2(
              __riscv_vle8_v_i8m1(lhs + (row + 1) * depth + depth_offset, vl),
              vl);
          lhs16_2 = __riscv_vsext_vf2_i16m2(
              __riscv_vle8_v_i8m1(lhs + (row + 2) * depth + depth_offset, vl),
              vl);
          lhs16_3 = __riscv_vsext_vf2_i16m2(
              __riscv_vle8_v_i8m1(lhs + (row + 3) * depth + depth_offset, vl),
              vl);
          if (lhs_zero_point != 0) {
            lhs16_0 = __riscv_vsub_vx_i16m2(lhs16_0, lhs_zero_point, vl);
            lhs16_1 = __riscv_vsub_vx_i16m2(lhs16_1, lhs_zero_point, vl);
            lhs16_2 = __riscv_vsub_vx_i16m2(lhs16_2, lhs_zero_point, vl);
            lhs16_3 = __riscv_vsub_vx_i16m2(lhs16_3, lhs_zero_point, vl);
          }
        } else {
          lhs16_0 = __riscv_vreinterpret_v_u16m2_i16m2(
              __riscv_vsub_vx_u16m2(
                  __riscv_vzext_vf2_u16m2(
                      __riscv_vle8_v_u8m1(
                          lhs + row * depth + depth_offset, vl),
                      vl),
                  static_cast<uint16_t>(lhs_zero_point), vl));
          lhs16_1 = __riscv_vreinterpret_v_u16m2_i16m2(
              __riscv_vsub_vx_u16m2(
                  __riscv_vzext_vf2_u16m2(
                      __riscv_vle8_v_u8m1(
                          lhs + (row + 1) * depth + depth_offset, vl),
                      vl),
                  static_cast<uint16_t>(lhs_zero_point), vl));
          lhs16_2 = __riscv_vreinterpret_v_u16m2_i16m2(
              __riscv_vsub_vx_u16m2(
                  __riscv_vzext_vf2_u16m2(
                      __riscv_vle8_v_u8m1(
                          lhs + (row + 2) * depth + depth_offset, vl),
                      vl),
                  static_cast<uint16_t>(lhs_zero_point), vl));
          lhs16_3 = __riscv_vreinterpret_v_u16m2_i16m2(
              __riscv_vsub_vx_u16m2(
                  __riscv_vzext_vf2_u16m2(
                      __riscv_vle8_v_u8m1(
                          lhs + (row + 3) * depth + depth_offset, vl),
                      vl),
                  static_cast<uint16_t>(lhs_zero_point), vl));
        }
        accum0 = __riscv_vwmacc_vv_i32m4_tu(accum0, lhs16_0, rhs16, vl);
        accum1 = __riscv_vwmacc_vv_i32m4_tu(accum1, lhs16_1, rhs16, vl);
        accum2 = __riscv_vwmacc_vv_i32m4_tu(accum2, lhs16_2, rhs16, vl);
        accum3 = __riscv_vwmacc_vv_i32m4_tu(accum3, lhs16_3, rhs16, vl);
        depth_offset += static_cast<int>(vl);
      }
      // vredsum consumes only seed lane 0; vmv.s.x avoids an m1 fill and
      // the extra m1/m4 VTYPE transitions of vmv.v.x.
      const vint32m1_t zero32 = __riscv_vmv_s_x_i32m1(0, vlmax);
      const vint32m1_t reduced0 =
          __riscv_vredsum_vs_i32m4_i32m1(accum0, zero32, vlmax);
      const vint32m1_t reduced1 =
          __riscv_vredsum_vs_i32m4_i32m1(accum1, zero32, vlmax);
      const vint32m1_t reduced2 =
          __riscv_vredsum_vs_i32m4_i32m1(accum2, zero32, vlmax);
      const vint32m1_t reduced3 =
          __riscv_vredsum_vs_i32m4_i32m1(accum3, zero32, vlmax);
      sum0 += __riscv_vmv_x_s_i32m1_i32(reduced0);
      sum1 += __riscv_vmv_x_s_i32m1_i32(reduced1);
      sum2 += __riscv_vmv_x_s_i32m1_i32(reduced2);
      sum3 += __riscv_vmv_x_s_i32m1_i32(reduced3);

      if constexpr (std::is_same_v<LhsScalar, uint8_t> &&
                    std::is_same_v<RhsScalar, uint8_t>) {
        if (use_vectorized_4row_quantize) {
          const int32_t sums[4] = {sum0, sum1, sum2, sum3};
          vint32m4_t value = __riscv_vle32_v_i32m4(sums, 4);
          if (multiplier_perchannel == nullptr) {
            value = RvvVectorizedQuantize(value, multiplier, multiplier_shift,
                                          4);
          } else {
            value = RvvVectorizedQuantizePerChannel(
                value, multiplier_perchannel + row,
                reinterpret_cast<const int32_t*>(shift_perchannel + row), 4,
                true);
          }
          value = __riscv_vadd_vx_i32m4(value, output_zero_point, 4);
          value = __riscv_vmax_vx_i32m4(value, clamp_min, 4);
          value = __riscv_vmin_vx_i32m4(value, clamp_max, 4);
          DstScalar* output = dst + row + col * lhs_rows;
          if constexpr (std::is_same_v<DstScalar, uint8_t>) {
            const vuint16m2_t value16 = __riscv_vncvt_x_x_w_u16m2(
                __riscv_vreinterpret_v_i32m4_u32m4(value), 4);
            __riscv_vse8_v_u8m1(
                reinterpret_cast<uint8_t*>(output),
                __riscv_vncvt_x_x_w_u8m1(value16, 4), 4);
          } else {
            const vint16m2_t value16 =
                __riscv_vncvt_x_x_w_i16m2(value, 4);
            __riscv_vse8_v_i8m1(
                reinterpret_cast<int8_t*>(output),
                __riscv_vncvt_x_x_w_i8m1(value16, 4), 4);
          }
          continue;
        }
      }
      dst[row + 0 + col * lhs_rows] =
          static_cast<DstScalar>(quantize(sum0, row + 0));
      dst[row + 1 + col * lhs_rows] =
          static_cast<DstScalar>(quantize(sum1, row + 1));
      dst[row + 2 + col * lhs_rows] =
          static_cast<DstScalar>(quantize(sum2, row + 2));
      dst[row + 3 + col * lhs_rows] =
          static_cast<DstScalar>(quantize(sum3, row + 3));
    }
    for (; row < lhs_rows; ++row) {
      int32_t sum = bias == nullptr ? 0 : bias[row];
      vint32m4_t accum = __riscv_vmv_v_x_i32m4(0, vlmax);
      int depth_offset = 0;
      while (depth_offset < depth) {
        const size_t remaining = depth - depth_offset;
        const size_t vl = remaining < vlmax
                              ? __riscv_vsetvl_e8m1(remaining)
                              : vlmax;
        vint16m2_t lhs16;
        vint16m2_t rhs16;
        if constexpr (std::is_same_v<LhsScalar, int8_t>) {
          lhs16 = __riscv_vsext_vf2_i16m2(
              __riscv_vle8_v_i8m1(lhs + row * depth + depth_offset, vl), vl);
          if (lhs_zero_point != 0) {
            lhs16 = __riscv_vsub_vx_i16m2(lhs16, lhs_zero_point, vl);
          }
        } else {
          lhs16 = __riscv_vreinterpret_v_u16m2_i16m2(
              __riscv_vsub_vx_u16m2(
                  __riscv_vzext_vf2_u16m2(
                      __riscv_vle8_v_u8m1(lhs + row * depth + depth_offset,
                                          vl),
                      vl),
                  static_cast<uint16_t>(lhs_zero_point), vl));
        }
        if constexpr (std::is_same_v<RhsScalar, int8_t>) {
          rhs16 = __riscv_vsext_vf2_i16m2(
              __riscv_vle8_v_i8m1(rhs + col * depth + depth_offset, vl), vl);
          if (rhs_zero_point != 0) {
            rhs16 = __riscv_vsub_vx_i16m2(rhs16, rhs_zero_point, vl);
          }
        } else {
          rhs16 = __riscv_vreinterpret_v_u16m2_i16m2(
              __riscv_vsub_vx_u16m2(
                  __riscv_vzext_vf2_u16m2(
                      __riscv_vle8_v_u8m1(rhs + col * depth + depth_offset,
                                          vl),
                      vl),
                  static_cast<uint16_t>(rhs_zero_point), vl));
        }
        accum = __riscv_vwmacc_vv_i32m4_tu(accum, lhs16, rhs16, vl);
        depth_offset += static_cast<int>(vl);
      }
      const vint32m1_t zero32 = __riscv_vmv_s_x_i32m1(0, vlmax);
      const vint32m1_t reduced =
          __riscv_vredsum_vs_i32m4_i32m1(accum, zero32, vlmax);
      sum += __riscv_vmv_x_s_i32m1_i32(reduced);
      dst[row + col * lhs_rows] =
          static_cast<DstScalar>(quantize(sum, row));
    }
  }
}

inline void AddInt32(const int32_t* input1, const int32_t* input2,
                     int32_t* output, int size, int32_t activation_min,
                     int32_t activation_max) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e32m1(size - offset);
    vint32m1_t value = __riscv_vadd_vv_i32m1(
        __riscv_vle32_v_i32m1(input1 + offset, vl),
        __riscv_vle32_v_i32m1(input2 + offset, vl), vl);
    value = __riscv_vmax_vx_i32m1(value, activation_min, vl);
    value = __riscv_vmin_vx_i32m1(value, activation_max, vl);
    __riscv_vse32_v_i32m1(output + offset, value, vl);
    offset += static_cast<int>(vl);
  }
}

inline void AddScalarInt32(const int32_t* input, int32_t broadcast,
                           int32_t* output, int size, int32_t activation_min,
                           int32_t activation_max) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e32m1(size - offset);
    vint32m1_t value = __riscv_vadd_vx_i32m1(
        __riscv_vle32_v_i32m1(input + offset, vl), broadcast, vl);
    value = __riscv_vmax_vx_i32m1(value, activation_min, vl);
    value = __riscv_vmin_vx_i32m1(value, activation_max, vl);
    __riscv_vse32_v_i32m1(output + offset, value, vl);
    offset += static_cast<int>(vl);
  }
}

inline void MulInt32(const int32_t* input1, const int32_t* input2,
                     int32_t* output, int size, int32_t activation_min,
                     int32_t activation_max) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e32m1(size - offset);
    vint32m1_t value = __riscv_vmul_vv_i32m1(
        __riscv_vle32_v_i32m1(input1 + offset, vl),
        __riscv_vle32_v_i32m1(input2 + offset, vl), vl);
    value = __riscv_vmax_vx_i32m1(value, activation_min, vl);
    value = __riscv_vmin_vx_i32m1(value, activation_max, vl);
    __riscv_vse32_v_i32m1(output + offset, value, vl);
    offset += static_cast<int>(vl);
  }
}

inline void MulScalarInt32(const int32_t* input, int32_t broadcast,
                           int32_t* output, int size, int32_t activation_min,
                           int32_t activation_max) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e32m1(size - offset);
    vint32m1_t value = __riscv_vmul_vx_i32m1(
        __riscv_vle32_v_i32m1(input + offset, vl), broadcast, vl);
    value = __riscv_vmax_vx_i32m1(value, activation_min, vl);
    value = __riscv_vmin_vx_i32m1(value, activation_max, vl);
    __riscv_vse32_v_i32m1(output + offset, value, vl);
    offset += static_cast<int>(vl);
  }
}

// The RVV quantized arithmetic uses RvvVectorizedQuantize so its fixed-point
// rounding remains aligned with the scalar path while byte loads/stores,
// widening, arithmetic and clamping stay vectorized. Non-RVV callers retain
// the original scalar implementations in optimized_ops.h.
template <bool kMultiply>
inline void QuantizedBinaryUint8(const uint8_t* input1, const uint8_t* input2,
                                 uint8_t* output, int size, int32_t offset1,
                                 int32_t offset2, int32_t left_shift,
                                 int32_t multiplier1, int32_t shift1,
                                 int32_t multiplier2, int32_t shift2,
                                 int32_t output_multiplier,
                                 int32_t output_shift, int32_t output_offset,
                                 int32_t activation_min,
                                 int32_t activation_max) {
  const bool reuse_vl = tensor_utils::RvvVlenClassBits() >= 256;
  const size_t vlmax = reuse_vl ? RvvSetVlE8M1(size) : 0;
  int offset = 0;
  while (offset < size) {
    const size_t vl = !reuse_vl || size - offset < vlmax
                          ? RvvSetVlE8M1(size - offset)
                          : vlmax;
    // Pure vector path: widen to i32, add offsets, apply scale via the
    // bit-exact vectorized quantizer, combine, quantize the sum, add the
    // output offset, clamp and narrow back to uint8.
    const vuint16m2_t in1_16 = __riscv_vzext_vf2_u16m2(
        __riscv_vle8_v_u8m1(input1 + offset, vl), vl);
    const vuint16m2_t in2_16 = __riscv_vzext_vf2_u16m2(
        __riscv_vle8_v_u8m1(input2 + offset, vl), vl);
    const vint32m4_t v1 = __riscv_vreinterpret_v_u32m4_i32m4(
        __riscv_vzext_vf2_u32m4(in1_16, vl));
    const vint32m4_t v2 = __riscv_vreinterpret_v_u32m4_i32m4(
        __riscv_vzext_vf2_u32m4(in2_16, vl));
    vint32m4_t value1 = v1;
    vint32m4_t value2 = v2;
    if (offset1 != 0) {
      value1 = __riscv_vadd_vx_i32m4(value1, offset1, vl);
    }
    if (offset2 != 0) {
      value2 = __riscv_vadd_vx_i32m4(value2, offset2, vl);
    }
    if (left_shift != 0) {
      value1 = __riscv_vsll_vx_i32m4(value1, left_shift, vl);
      value2 = __riscv_vsll_vx_i32m4(value2, left_shift, vl);
    }
    vint32m4_t scaled1 =
        RvvVectorizedQuantize(value1, multiplier1, shift1, vl);
    vint32m4_t scaled2 =
        RvvVectorizedQuantize(value2, multiplier2, shift2, vl);
    vint32m4_t raw =
        kMultiply ? __riscv_vmul_vv_i32m4(scaled1, scaled2, vl)
                  : __riscv_vadd_vv_i32m4(scaled1, scaled2, vl);
    vint32m4_t result =
        RvvVectorizedQuantize(raw, output_multiplier, output_shift, vl);
    if (output_offset != 0) {
      result = __riscv_vadd_vx_i32m4(result, output_offset, vl);
    }
    result = __riscv_vmax_vx_i32m4(result, activation_min, vl);
    result = __riscv_vmin_vx_i32m4(result, activation_max, vl);
    const vuint16m2_t out16 = __riscv_vncvt_x_x_w_u16m2(
        __riscv_vreinterpret_v_i32m4_u32m4(result), vl);
    const vuint8m1_t out8 = __riscv_vncvt_x_x_w_u8m1(out16, vl);
    __riscv_vse8_v_u8m1(output + offset, out8, vl);
    offset += static_cast<int>(vl);
  }
}

template <bool kMultiply>
inline void QuantizedBinaryInt8(const int8_t* input1, const int8_t* input2,
                                int8_t* output, int size, int32_t offset1,
                                int32_t offset2, int32_t left_shift,
                                int32_t multiplier1, int32_t shift1,
                                int32_t multiplier2, int32_t shift2,
                                 int32_t output_multiplier,
                                 int32_t output_shift, int32_t output_offset,
                                 int32_t activation_min,
                                 int32_t activation_max) {
  const bool reuse_vl = tensor_utils::RvvVlenClassBits() >= 256;
  const size_t vlmax = reuse_vl ? RvvSetVlE8M1(size) : 0;
  int offset = 0;
  while (offset < size) {
    const size_t vl = !reuse_vl || size - offset < vlmax
                          ? RvvSetVlE8M1(size - offset)
                          : vlmax;
    // Pure vector path, mirroring QuantizedBinaryUint8.
    const vint32m4_t v1 = __riscv_vsext_vf4_i32m4(
        __riscv_vle8_v_i8m1(input1 + offset, vl), vl);
    const vint32m4_t v2 = __riscv_vsext_vf4_i32m4(
        __riscv_vle8_v_i8m1(input2 + offset, vl), vl);
    vint32m4_t value1 = v1;
    vint32m4_t value2 = v2;
    if (offset1 != 0) {
      value1 = __riscv_vadd_vx_i32m4(value1, offset1, vl);
    }
    if (offset2 != 0) {
      value2 = __riscv_vadd_vx_i32m4(value2, offset2, vl);
    }
    if (left_shift != 0) {
      value1 = __riscv_vsll_vx_i32m4(value1, left_shift, vl);
      value2 = __riscv_vsll_vx_i32m4(value2, left_shift, vl);
    }
    vint32m4_t scaled1 =
        RvvVectorizedQuantize(value1, multiplier1, shift1, vl);
    vint32m4_t scaled2 =
        RvvVectorizedQuantize(value2, multiplier2, shift2, vl);
    vint32m4_t raw =
        kMultiply ? __riscv_vmul_vv_i32m4(scaled1, scaled2, vl)
                  : __riscv_vadd_vv_i32m4(scaled1, scaled2, vl);
    vint32m4_t result =
        RvvVectorizedQuantize(raw, output_multiplier, output_shift, vl);
    if (output_offset != 0) {
      result = __riscv_vadd_vx_i32m4(result, output_offset, vl);
    }
    result = __riscv_vmax_vx_i32m4(result, activation_min, vl);
    result = __riscv_vmin_vx_i32m4(result, activation_max, vl);
    const vint16m2_t out16 = __riscv_vncvt_x_x_w_i16m2(result, vl);
    const vint8m1_t out8 = __riscv_vncvt_x_x_w_i8m1(out16, vl);
    __riscv_vse8_v_i8m1(output + offset, out8, vl);
    offset += static_cast<int>(vl);
  }
}

inline void MulUint8(const uint8_t* input1, const uint8_t* input2,
                     uint8_t* output, int size, int32_t input1_offset,
                     int32_t input2_offset, int32_t output_multiplier,
                     int32_t output_shift, int32_t output_offset,
                     int32_t activation_min, int32_t activation_max) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = RvvSetVlE8M1(size - offset);
    const vuint16m2_t input1_16 = __riscv_vzext_vf2_u16m2(
        __riscv_vle8_v_u8m1(input1 + offset, vl), vl);
    const vuint16m2_t input2_16 = __riscv_vzext_vf2_u16m2(
        __riscv_vle8_v_u8m1(input2 + offset, vl), vl);
    vint32m4_t value1 = __riscv_vreinterpret_v_u32m4_i32m4(
        __riscv_vzext_vf2_u32m4(input1_16, vl));
    vint32m4_t value2 = __riscv_vreinterpret_v_u32m4_i32m4(
        __riscv_vzext_vf2_u32m4(input2_16, vl));
    if (input1_offset != 0) {
      value1 = __riscv_vadd_vx_i32m4(value1, input1_offset, vl);
    }
    if (input2_offset != 0) {
      value2 = __riscv_vadd_vx_i32m4(value2, input2_offset, vl);
    }
    vint32m4_t result = __riscv_vmul_vv_i32m4(value1, value2, vl);
    result = RvvVectorizedQuantize(result, output_multiplier, output_shift, vl);
    if (output_offset != 0) {
      result = __riscv_vadd_vx_i32m4(result, output_offset, vl);
    }
    result = __riscv_vmax_vx_i32m4(result, activation_min, vl);
    result = __riscv_vmin_vx_i32m4(result, activation_max, vl);
    const vuint16m2_t out16 = __riscv_vncvt_x_x_w_u16m2(
        __riscv_vreinterpret_v_i32m4_u32m4(result), vl);
    __riscv_vse8_v_u8m1(
        output + offset, __riscv_vncvt_x_x_w_u8m1(out16, vl), vl);
    offset += static_cast<int>(vl);
  }
}

inline void MulScalarUint8(const uint8_t* input, uint8_t broadcast,
                           uint8_t* output, int size, int32_t broadcast_offset,
                           int32_t input_offset,
                           int32_t output_multiplier, int32_t output_shift,
                           int32_t output_offset, int32_t activation_min,
                           int32_t activation_max) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = RvvSetVlE8M1(size - offset);
    const vuint16m2_t input16 = __riscv_vzext_vf2_u16m2(
        __riscv_vle8_v_u8m1(input + offset, vl), vl);
    vint32m4_t value = __riscv_vreinterpret_v_u32m4_i32m4(
        __riscv_vzext_vf2_u32m4(input16, vl));
    if (input_offset != 0) {
      value = __riscv_vadd_vx_i32m4(value, input_offset, vl);
    }
    value = __riscv_vmul_vx_i32m4(
        value, static_cast<int32_t>(broadcast) + broadcast_offset, vl);
    value = RvvVectorizedQuantize(value, output_multiplier, output_shift, vl);
    if (output_offset != 0) {
      value = __riscv_vadd_vx_i32m4(value, output_offset, vl);
    }
    value = __riscv_vmax_vx_i32m4(value, activation_min, vl);
    value = __riscv_vmin_vx_i32m4(value, activation_max, vl);
    const vuint16m2_t out16 = __riscv_vncvt_x_x_w_u16m2(
        __riscv_vreinterpret_v_i32m4_u32m4(value), vl);
    __riscv_vse8_v_u8m1(
        output + offset, __riscv_vncvt_x_x_w_u8m1(out16, vl), vl);
    offset += static_cast<int>(vl);
  }
}

inline void AddScalarUint8(int size, uint8_t broadcast, const uint8_t* input,
                           uint8_t* output, int32_t broadcast_offset,
                           int32_t input_offset, int32_t left_shift,
                           int32_t broadcast_multiplier,
                           int32_t broadcast_shift, int32_t input_multiplier,
                           int32_t input_shift, int32_t output_multiplier,
                           int32_t output_shift, int32_t output_offset,
                           int32_t activation_min, int32_t activation_max) {
  const int32_t broadcast_scaled =
      MultiplyByQuantizedMultiplierSmallerThanOneExp(
          (static_cast<int32_t>(broadcast) + broadcast_offset) *
              (1 << left_shift),
          broadcast_multiplier, broadcast_shift);
  const bool reuse_vl = tensor_utils::RvvVlenClassBits() >= 256;
  const size_t vlmax = reuse_vl ? RvvSetVlE8M1(size) : 0;
  int offset = 0;
  while (offset < size) {
    const size_t vl = !reuse_vl || size - offset < vlmax
                          ? RvvSetVlE8M1(size - offset)
                          : vlmax;
    const vuint16m2_t input16 = __riscv_vzext_vf2_u16m2(
        __riscv_vle8_v_u8m1(input + offset, vl), vl);
    const vint32m4_t input32 = __riscv_vreinterpret_v_u32m4_i32m4(
        __riscv_vzext_vf2_u32m4(input16, vl));
    vint32m4_t value = input32;
    if (input_offset != 0) {
      value = __riscv_vadd_vx_i32m4(value, input_offset, vl);
    }
    if (left_shift != 0) {
      value = __riscv_vsll_vx_i32m4(value, left_shift, vl);
    }
    value = RvvVectorizedQuantize(value, input_multiplier, input_shift, vl);
    value = __riscv_vadd_vx_i32m4(value, broadcast_scaled, vl);
    value = RvvVectorizedQuantize(value, output_multiplier, output_shift, vl);
    if (output_offset != 0) {
      value = __riscv_vadd_vx_i32m4(value, output_offset, vl);
    }
    value = __riscv_vmax_vx_i32m4(value, activation_min, vl);
    value = __riscv_vmin_vx_i32m4(value, activation_max, vl);
    const vuint16m2_t out16 = __riscv_vncvt_x_x_w_u16m2(
        __riscv_vreinterpret_v_i32m4_u32m4(value), vl);
    __riscv_vse8_v_u8m1(
        output + offset, __riscv_vncvt_x_x_w_u8m1(out16, vl), vl);
    offset += static_cast<int>(vl);
  }
}

inline void AddScalarInt8(int size, int8_t broadcast, const int8_t* input,
                          int8_t* output, int32_t broadcast_offset,
                          int32_t input_offset, int32_t left_shift,
                          int32_t broadcast_multiplier,
                          int32_t broadcast_shift, int32_t input_multiplier,
                          int32_t input_shift, int32_t output_multiplier,
                          int32_t output_shift, int32_t output_offset,
                          int32_t activation_min, int32_t activation_max) {
  const int32_t broadcast_scaled =
      MultiplyByQuantizedMultiplierSmallerThanOneExp(
          (static_cast<int32_t>(broadcast) + broadcast_offset) *
              (1 << left_shift),
          broadcast_multiplier, broadcast_shift);
  const bool reuse_vl = tensor_utils::RvvVlenClassBits() >= 256;
  const size_t vlmax = reuse_vl ? RvvSetVlE8M1(size) : 0;
  int offset = 0;
  while (offset < size) {
    const size_t vl = !reuse_vl || size - offset < vlmax
                          ? RvvSetVlE8M1(size - offset)
                          : vlmax;
    const vint32m4_t value0 = __riscv_vsext_vf4_i32m4(
        __riscv_vle8_v_i8m1(input + offset, vl), vl);
    vint32m4_t value = value0;
    if (input_offset != 0) {
      value = __riscv_vadd_vx_i32m4(value, input_offset, vl);
    }
    if (left_shift != 0) {
      value = __riscv_vsll_vx_i32m4(value, left_shift, vl);
    }
    value = RvvVectorizedQuantize(value, input_multiplier, input_shift, vl);
    value = __riscv_vadd_vx_i32m4(value, broadcast_scaled, vl);
    value = RvvVectorizedQuantize(value, output_multiplier, output_shift, vl);
    if (output_offset != 0) {
      value = __riscv_vadd_vx_i32m4(value, output_offset, vl);
    }
    value = __riscv_vmax_vx_i32m4(value, activation_min, vl);
    value = __riscv_vmin_vx_i32m4(value, activation_max, vl);
    const vint16m2_t out16 = __riscv_vncvt_x_x_w_i16m2(value, vl);
    __riscv_vse8_v_i8m1(
        output + offset, __riscv_vncvt_x_x_w_i8m1(out16, vl), vl);
    offset += static_cast<int>(vl);
  }
}

inline void MulInt8(const int8_t* input1, const int8_t* input2, int8_t* output,
                    int size, int32_t input1_offset, int32_t input2_offset,
                    int32_t output_multiplier, int32_t output_shift,
                    int32_t output_offset, int32_t activation_min,
                    int32_t activation_max) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = RvvSetVlE8M1(size - offset);
    vint32m4_t value1 = __riscv_vsext_vf4_i32m4(
        __riscv_vle8_v_i8m1(input1 + offset, vl), vl);
    vint32m4_t value2 = __riscv_vsext_vf4_i32m4(
        __riscv_vle8_v_i8m1(input2 + offset, vl), vl);
    if (input1_offset != 0) {
      value1 = __riscv_vadd_vx_i32m4(value1, input1_offset, vl);
    }
    if (input2_offset != 0) {
      value2 = __riscv_vadd_vx_i32m4(value2, input2_offset, vl);
    }
    vint32m4_t result = __riscv_vmul_vv_i32m4(value1, value2, vl);
    result = RvvVectorizedQuantize(result, output_multiplier, output_shift, vl);
    if (output_offset != 0) {
      result = __riscv_vadd_vx_i32m4(result, output_offset, vl);
    }
    result = __riscv_vmax_vx_i32m4(result, activation_min, vl);
    result = __riscv_vmin_vx_i32m4(result, activation_max, vl);
    const vint16m2_t out16 = __riscv_vncvt_x_x_w_i16m2(result, vl);
    __riscv_vse8_v_i8m1(
        output + offset, __riscv_vncvt_x_x_w_i8m1(out16, vl), vl);
    offset += static_cast<int>(vl);
  }
}

inline void MulScalarInt8(const int8_t* input, int8_t broadcast, int8_t* output,
                          int size, int32_t broadcast_offset,
                          int32_t input_offset, int32_t output_multiplier,
                          int32_t output_shift, int32_t output_offset,
                          int32_t activation_min, int32_t activation_max) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = RvvSetVlE8M1(size - offset);
    vint32m4_t value = __riscv_vsext_vf4_i32m4(
        __riscv_vle8_v_i8m1(input + offset, vl), vl);
    if (input_offset != 0) {
      value = __riscv_vadd_vx_i32m4(value, input_offset, vl);
    }
    value = __riscv_vmul_vx_i32m4(
        value, static_cast<int32_t>(broadcast) + broadcast_offset, vl);
    value = RvvVectorizedQuantize(value, output_multiplier, output_shift, vl);
    if (output_offset != 0) {
      value = __riscv_vadd_vx_i32m4(value, output_offset, vl);
    }
    value = __riscv_vmax_vx_i32m4(value, activation_min, vl);
    value = __riscv_vmin_vx_i32m4(value, activation_max, vl);
    const vint16m2_t out16 = __riscv_vncvt_x_x_w_i16m2(value, vl);
    __riscv_vse8_v_i8m1(
        output + offset, __riscv_vncvt_x_x_w_i8m1(out16, vl), vl);
    offset += static_cast<int>(vl);
  }
}

template <typename Input, typename Output>
inline void Requantize(const Input* input, int size, int32_t multiplier,
                       int32_t shift, int32_t input_zero_point,
                       int32_t output_zero_point, Output* output) {
  constexpr int32_t kMin = std::numeric_limits<Output>::min();
  constexpr int32_t kMax = std::numeric_limits<Output>::max();
  int offset = 0;
  while (offset < size) {
    const size_t vl = RvvSetVlE8M1(size - offset);
    if constexpr (std::is_same<Input, int8_t>::value) {
      vint32m4_t value = __riscv_vsext_vf4_i32m4(
          __riscv_vle8_v_i8m1(input + offset, vl), vl);
      if (input_zero_point != 0) {
        value = __riscv_vadd_vx_i32m4(value, -input_zero_point, vl);
      }
      value = RvvVectorizedQuantize(value, multiplier, shift, vl);
      if (output_zero_point != 0) {
        value = __riscv_vadd_vx_i32m4(value, output_zero_point, vl);
      }
      value = __riscv_vmax_vx_i32m4(value, kMin, vl);
      value = __riscv_vmin_vx_i32m4(value, kMax, vl);
      if constexpr (std::is_same<Output, int8_t>::value) {
        const vint16m2_t out16 = __riscv_vncvt_x_x_w_i16m2(value, vl);
        __riscv_vse8_v_i8m1(
            output + offset, __riscv_vncvt_x_x_w_i8m1(out16, vl), vl);
      } else {
        const vuint16m2_t out16 = __riscv_vncvt_x_x_w_u16m2(
            __riscv_vreinterpret_v_i32m4_u32m4(value), vl);
        __riscv_vse8_v_u8m1(
            reinterpret_cast<uint8_t*>(output) + offset,
            __riscv_vncvt_x_x_w_u8m1(out16, vl), vl);
      }
    } else {
      vint32m4_t value = __riscv_vreinterpret_v_u32m4_i32m4(
          __riscv_vzext_vf2_u32m4(
              __riscv_vzext_vf2_u16m2(
                  __riscv_vle8_v_u8m1(input + offset, vl), vl),
              vl));
      if (input_zero_point != 0) {
        value = __riscv_vadd_vx_i32m4(value, -input_zero_point, vl);
      }
      value = RvvVectorizedQuantize(value, multiplier, shift, vl);
      if (output_zero_point != 0) {
        value = __riscv_vadd_vx_i32m4(value, output_zero_point, vl);
      }
      value = __riscv_vmax_vx_i32m4(value, kMin, vl);
      value = __riscv_vmin_vx_i32m4(value, kMax, vl);
      if constexpr (std::is_same<Output, int8_t>::value) {
        const vint16m2_t out16 = __riscv_vncvt_x_x_w_i16m2(value, vl);
        __riscv_vse8_v_i8m1(
            reinterpret_cast<int8_t*>(output) + offset,
            __riscv_vncvt_x_x_w_i8m1(out16, vl), vl);
      } else {
        const vuint16m2_t out16 = __riscv_vncvt_x_x_w_u16m2(
            __riscv_vreinterpret_v_i32m4_u32m4(value), vl);
        __riscv_vse8_v_u8m1(
            reinterpret_cast<uint8_t*>(output) + offset,
            __riscv_vncvt_x_x_w_u8m1(out16, vl), vl);
      }
    }
    offset += static_cast<int>(vl);
  }
}

template <typename Input>
inline void Dequantize(const Input* input, int size, int32_t zero_point,
                       float scale, float* output) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e8m1(size - offset);
    if constexpr (std::is_same<Input, int8_t>::value) {
      const vint8m1_t values = __riscv_vle8_v_i8m1(input + offset, vl);
      const vint16m2_t values16 = __riscv_vsext_vf2_i16m2(values, vl);
      const vint32m4_t values32 = __riscv_vsext_vf2_i32m4(values16, vl);
      vfloat32m4_t result = __riscv_vfcvt_f_x_v_f32m4(values32, vl);
      if (zero_point != 0) {
        result = __riscv_vfsub_vf_f32m4(
            result, static_cast<float>(zero_point), vl);
      }
      result = __riscv_vfmul_vf_f32m4(result, scale, vl);
      __riscv_vse32_v_f32m4(output + offset, result, vl);
    } else {
      const vuint8m1_t values = __riscv_vle8_v_u8m1(input + offset, vl);
      const vuint16m2_t values16 = __riscv_vzext_vf2_u16m2(values, vl);
      const vuint32m4_t values32 = __riscv_vzext_vf2_u32m4(values16, vl);
      vfloat32m4_t result = __riscv_vfcvt_f_xu_v_f32m4(values32, vl);
      if (zero_point != 0) {
        result = __riscv_vfsub_vf_f32m4(
            result, static_cast<float>(zero_point), vl);
      }
      result = __riscv_vfmul_vf_f32m4(result, scale, vl);
      __riscv_vse32_v_f32m4(output + offset, result, vl);
    }
    offset += static_cast<int>(vl);
  }
}

inline void DequantizeInt16(const int16_t* input, int size, int32_t zero_point,
                            float scale, float* output) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e16m1(size - offset);
    const vint16m1_t values = __riscv_vle16_v_i16m1(input + offset, vl);
    const vint32m2_t values32 = __riscv_vsext_vf2_i32m2(values, vl);
    vfloat32m2_t result = __riscv_vfcvt_f_x_v_f32m2(values32, vl);
    if (zero_point != 0) {
      result = __riscv_vfsub_vf_f32m2(
          result, static_cast<float>(zero_point), vl);
    }
    result = __riscv_vfmul_vf_f32m2(result, scale, vl);
    __riscv_vse32_v_f32m2(output + offset, result, vl);
    offset += static_cast<int>(vl);
  }
}

template <typename Output>
inline void AffineQuantize(const float* input, int size, float inverse_scale,
                           int32_t zero_point, int32_t min_value,
                           int32_t max_value, Output* output) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e32m4(size - offset);
    const vfloat32m4_t values = __riscv_vle32_v_f32m4(input + offset, vl);
    const vfloat32m4_t scaled =
        __riscv_vfmul_vf_f32m4(values, inverse_scale, vl);
    const vfloat32m4_t half = __riscv_vfmv_v_f_f32m4(0.5f, vl);
    const vfloat32m4_t signed_half =
        __riscv_vfsgnj_vv_f32m4(half, scaled, vl);
    const vfloat32m4_t rounded_input =
        __riscv_vfadd_vv_f32m4(scaled, signed_half, vl);
    const vint32m4_t rounded =
        __riscv_vfcvt_rtz_x_f_v_i32m4(rounded_input, vl);
    vint32m4_t shifted = rounded;
    if (zero_point != 0) {
      shifted = __riscv_vadd_vx_i32m4(shifted, zero_point, vl);
    }
    const vint32m4_t clamped = __riscv_vmax_vx_i32m4(
        __riscv_vmin_vx_i32m4(shifted, max_value, vl), min_value, vl);
    if constexpr (std::is_same<Output, int8_t>::value) {
      const vint16m2_t out16 = __riscv_vncvt_x_x_w_i16m2(clamped, vl);
      const vint8m1_t out8 = __riscv_vncvt_x_x_w_i8m1(out16, vl);
      __riscv_vse8_v_i8m1(reinterpret_cast<int8_t*>(output) + offset, out8,
                          vl);
    } else if constexpr (std::is_same<Output, int16_t>::value) {
      const vint16m2_t out16 = __riscv_vncvt_x_x_w_i16m2(clamped, vl);
      __riscv_vse16_v_i16m2(reinterpret_cast<int16_t*>(output) + offset, out16,
                            vl);
    } else {
      const vuint16m2_t out16 = __riscv_vncvt_x_x_w_u16m2(
          __riscv_vreinterpret_v_i32m4_u32m4(clamped), vl);
      const vuint8m1_t out8 = __riscv_vncvt_x_x_w_u8m1(out16, vl);
      __riscv_vse8_v_u8m1(reinterpret_cast<uint8_t*>(output) + offset, out8,
                          vl);
    }
    offset += static_cast<int>(vl);
  }
}

template <typename Output>
inline void QuantizeUniform(const int32_t* input, int size, int32_t multiplier,
                            int32_t shift, int32_t output_zero_point,
                            int32_t output_min, int32_t output_max,
                            Output* output) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e32m4(size - offset);
    const vint32m4_t values = __riscv_vle32_v_i32m4(input + offset, vl);
    vint32m4_t value = RvvVectorizedQuantize(values, multiplier, shift, vl);
    if (output_zero_point != 0) {
      value = __riscv_vadd_vx_i32m4(value, output_zero_point, vl);
    }
    value = __riscv_vmax_vx_i32m4(value, output_min, vl);
    value = __riscv_vmin_vx_i32m4(value, output_max, vl);
    if constexpr (std::is_same<Output, int8_t>::value) {
      const vint16m2_t out16 = __riscv_vncvt_x_x_w_i16m2(value, vl);
      const vint8m1_t out8 = __riscv_vncvt_x_x_w_i8m1(out16, vl);
      __riscv_vse8_v_i8m1(reinterpret_cast<int8_t*>(output) + offset, out8,
                          vl);
    } else {
      const vuint16m2_t out16 = __riscv_vncvt_x_x_w_u16m2(
          __riscv_vreinterpret_v_i32m4_u32m4(value), vl);
      const vuint8m1_t out8 = __riscv_vncvt_x_x_w_u8m1(out16, vl);
      __riscv_vse8_v_u8m1(reinterpret_cast<uint8_t*>(output) + offset, out8,
                          vl);
    }
    offset += static_cast<int>(vl);
  }
}

template <typename Output>
inline void QuantizePerChannel(const int32_t* multiplier, const int32_t* shift,
                               int channel_size, int total_size,
                               int32_t output_zero_point, int32_t output_min,
                               int32_t output_max, const int32_t* input,
                               Output* output) {
#if defined(TFLITE_SINGLE_ROUNDING) && TFLITE_SINGLE_ROUNDING
  const bool use_vectorized_quantize = tensor_utils::RvvVlenClassBits() >= 256;
#else
  // VLEN=128 uses the scalar contract for the positive-left-shift boundary;
  // VLEN>=256 uses the exact widened RVV sequence.
  const bool use_vectorized_quantize = tensor_utils::RvvVlenClassBits() >= 256;
#endif
  for (int row = 0; row < total_size / channel_size; ++row) {
    int channel = 0;
    while (channel < channel_size) {
      const size_t vl = RvvSetVlE32M4ForQuantize(
          channel_size - channel, use_vectorized_quantize);
      vint32m4_t value = RvvVectorizedQuantizePerChannel(
          __riscv_vle32_v_i32m4(input + row * channel_size + channel, vl),
          multiplier + channel, shift + channel, vl,
          use_vectorized_quantize);
      if (output_zero_point != 0) {
        value = __riscv_vadd_vx_i32m4(value, output_zero_point, vl);
      }
      value = __riscv_vmax_vx_i32m4(value, output_min, vl);
      value = __riscv_vmin_vx_i32m4(value, output_max, vl);
      if constexpr (std::is_same<Output, int8_t>::value) {
        const vint16m2_t out16 = __riscv_vncvt_x_x_w_i16m2(value, vl);
        __riscv_vse8_v_i8m1(reinterpret_cast<int8_t*>(output) +
                                row * channel_size + channel,
                            __riscv_vncvt_x_x_w_i8m1(out16, vl), vl);
      } else if constexpr (std::is_same<Output, int16_t>::value) {
        const vint16m2_t out16 = __riscv_vncvt_x_x_w_i16m2(value, vl);
        __riscv_vse16_v_i16m2(reinterpret_cast<int16_t*>(output) +
                                  row * channel_size + channel,
                              out16, vl);
      } else {
        const vuint16m2_t out16 = __riscv_vncvt_x_x_w_u16m2(
            __riscv_vreinterpret_v_i32m4_u32m4(value), vl);
        __riscv_vse8_v_u8m1(reinterpret_cast<uint8_t*>(output) +
                                row * channel_size + channel,
                            __riscv_vncvt_x_x_w_u8m1(out16, vl), vl);
      }
      channel += static_cast<int>(vl);
    }
  }
}

template <typename T>
inline void VectorCopy(const T* input, T* output, int size) {
  if constexpr (sizeof(T) == 1) {
    int offset = 0;
    while (offset < size) {
      const size_t vl = __riscv_vsetvl_e8m1(size - offset);
      const vuint8m1_t values = __riscv_vle8_v_u8m1(
          reinterpret_cast<const uint8_t*>(input) + offset, vl);
      __riscv_vse8_v_u8m1(reinterpret_cast<uint8_t*>(output) + offset, values,
                          vl);
      offset += static_cast<int>(vl);
    }
  } else if constexpr (sizeof(T) == 2) {
    int offset = 0;
    while (offset < size) {
      const size_t vl = __riscv_vsetvl_e16m1(size - offset);
      const vint16m1_t values = __riscv_vle16_v_i16m1(
          reinterpret_cast<const int16_t*>(input) + offset, vl);
      __riscv_vse16_v_i16m1(reinterpret_cast<int16_t*>(output) + offset,
                            values, vl);
      offset += static_cast<int>(vl);
    }
  } else {
    int offset = 0;
    while (offset < size) {
      const size_t vl = __riscv_vsetvl_e32m1(size - offset);
      const vint32m1_t values = __riscv_vle32_v_i32m1(
          reinterpret_cast<const int32_t*>(input) + offset, vl);
      __riscv_vse32_v_i32m1(reinterpret_cast<int32_t*>(output) + offset,
                            values, vl);
      offset += static_cast<int>(vl);
    }
  }
}

inline int ArgMinFloat(const float* input, int size) {
  if (size <= 0) return 0;
  float min_value = input[0];
  int min_index = 0;
  int offset = 0;
  float values[16];
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e32m1(
        std::min<size_t>(size - offset, 16));
    __riscv_vse32_v_f32m1(values,
                          __riscv_vle32_v_f32m1(input + offset, vl), vl);
    for (size_t lane = 0; lane < vl; ++lane) {
      if (values[lane] < min_value) {
        min_value = values[lane];
        min_index = offset + static_cast<int>(lane);
      }
    }
    offset += static_cast<int>(vl);
  }
  return min_index;
}

inline int ArgMaxFloat(const float* input, int size) {
  if (size <= 0) return 0;
  float max_value = input[0];
  int max_index = 0;
  int offset = 0;
  float values[16];
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e32m1(
        std::min<size_t>(size - offset, 16));
    __riscv_vse32_v_f32m1(values,
                          __riscv_vle32_v_f32m1(input + offset, vl), vl);
    for (size_t lane = 0; lane < vl; ++lane) {
      if (values[lane] > max_value) {
        max_value = values[lane];
        max_index = offset + static_cast<int>(lane);
      }
    }
    offset += static_cast<int>(vl);
  }
  return max_index;
}

template <typename T>
inline int ArgMaxInteger(const T* input, int size) {
  if (size <= 0) return 0;
  T max_value = input[0];
  int max_index = 0;
  int offset = 0;
  T values[64];
  while (offset < size) {
    const size_t vl = RvvSetVlE8M1Stack(size - offset);
    if constexpr (std::is_same<T, int8_t>::value) {
      __riscv_vse8_v_i8m1(reinterpret_cast<int8_t*>(values),
                          __riscv_vle8_v_i8m1(input + offset, vl), vl);
    } else {
      __riscv_vse8_v_u8m1(reinterpret_cast<uint8_t*>(values),
                          __riscv_vle8_v_u8m1(input + offset, vl), vl);
    }
    for (size_t lane = 0; lane < vl; ++lane) {
      if (values[lane] > max_value) {
        max_value = values[lane];
        max_index = offset + static_cast<int>(lane);
      }
    }
    offset += static_cast<int>(vl);
  }
  return max_index;
}

// Vectorized row-copy RESIZE_NEAREST_NEIGHBOR.  The nearest-neighbor mapping
// is computed once per output pixel and each output row is then copied with
// RVV byte loads/stores, which is the dominant cost when depth >= VLEN.
template <typename T>
inline void RvvResizeNearestNeighbor(const T* input, T* output, int batches,
                                     int input_height, int input_width,
                                     int output_height, int output_width,
                                     int depth, int32_t height_scale,
                                     int32_t width_scale) {
  const int col_offset = depth;
  const int row_offset = input_width * depth;
  const int batch_offset = input_height * row_offset;
  const T* input_ptr = input;
  T* output_ptr = output;
  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      const int in_y =
          std::min((y * height_scale) >> 16, input_height - 1);
      const T* y_input_ptr = input_ptr + in_y * row_offset;
      for (int x = 0; x < output_width; ++x) {
        const int in_x =
            std::min((x * width_scale) >> 16, input_width - 1);
        const T* x_input_ptr = y_input_ptr + in_x * col_offset;
        int c = 0;
        while (c < depth) {
          if constexpr (std::is_same<T, uint8_t>::value) {
            const size_t vl = RvvSetVlE8M1(depth - c);
            __riscv_vse8_v_u8m1(
                reinterpret_cast<uint8_t*>(output_ptr + c),
                __riscv_vle8_v_u8m1(
                reinterpret_cast<const uint8_t*>(x_input_ptr + c), vl),
                vl);
            c += static_cast<int>(vl);
          } else if constexpr (std::is_same<T, int8_t>::value) {
            const size_t vl = RvvSetVlE8M1(depth - c);
            __riscv_vse8_v_i8m1(
                reinterpret_cast<int8_t*>(output_ptr + c),
                __riscv_vle8_v_i8m1(
                reinterpret_cast<const int8_t*>(x_input_ptr + c), vl),
                vl);
            c += static_cast<int>(vl);
          } else {
            const size_t vl32 = __riscv_vsetvl_e32m1(depth - c);
            __riscv_vse32_v_i32m1(
                output_ptr + c,
                __riscv_vle32_v_i32m1(x_input_ptr + c, vl32), vl32);
            c += static_cast<int>(vl32);
          }
        }
        output_ptr += depth;
      }
    }
    input_ptr += batch_offset;
  }
}

template <typename T>
inline void MaxPoolChannels(const T* input, T* output, int depth,
                            int input_height, int input_width,
                            int output_height, int output_width, int stride_h,
                            int stride_w, int filter_h, int filter_w,
                            int pad_h, int pad_w, int batches, T initial_value,
                            T activation_min, T activation_max) {
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = out_x * stride_w - pad_w;
        const int in_y_origin = out_y * stride_h - pad_h;
        const int filter_x_start = std::max(0, -in_x_origin);
        const int filter_x_end = std::min(filter_w, input_width - in_x_origin);
        const int filter_y_start = std::max(0, -in_y_origin);
        const int filter_y_end = std::min(filter_h, input_height - in_y_origin);
        T* out = output +
                 ((batch * output_height + out_y) * output_width + out_x) *
                     depth;
        int channel = 0;
        while (channel < depth) {
          size_t vl;
          if constexpr (std::is_same<T, uint8_t>::value ||
                        std::is_same<T, int8_t>::value) {
            vl = RvvSetVlE8M1(depth - channel);
          } else {
            vl = __riscv_vsetvl_e32m1(depth - channel);
          }
          if constexpr (std::is_same<T, uint8_t>::value) {
            vuint8m1_t value = __riscv_vmv_v_x_u8m1(initial_value, vl);
            for (int fy = filter_y_start; fy < filter_y_end; ++fy) {
              for (int fx = filter_x_start; fx < filter_x_end; ++fx) {
                const T* in = input +
                    ((batch * input_height + in_y_origin + fy) * input_width +
                     in_x_origin + fx) *
                        depth + channel;
                value = __riscv_vmaxu_vv_u8m1(
                    value, __riscv_vle8_v_u8m1(in, vl), vl);
              }
            }
            value = __riscv_vmaxu_vx_u8m1(value, activation_min, vl);
            value = __riscv_vminu_vx_u8m1(value, activation_max, vl);
            __riscv_vse8_v_u8m1(out + channel, value, vl);
          } else if constexpr (std::is_same<T, int8_t>::value) {
            vint8m1_t value = __riscv_vmv_v_x_i8m1(initial_value, vl);
            for (int fy = filter_y_start; fy < filter_y_end; ++fy) {
              for (int fx = filter_x_start; fx < filter_x_end; ++fx) {
                const T* in = input +
                    ((batch * input_height + in_y_origin + fy) * input_width +
                     in_x_origin + fx) *
                        depth + channel;
                value = __riscv_vmax_vv_i8m1(
                    value, __riscv_vle8_v_i8m1(in, vl), vl);
              }
            }
            value = __riscv_vmax_vx_i8m1(value, activation_min, vl);
            value = __riscv_vmin_vx_i8m1(value, activation_max, vl);
            __riscv_vse8_v_i8m1(out + channel, value, vl);
          } else {
            vfloat32m1_t value = __riscv_vfmv_v_f_f32m1(initial_value, vl);
            for (int fy = filter_y_start; fy < filter_y_end; ++fy) {
              for (int fx = filter_x_start; fx < filter_x_end; ++fx) {
                const T* in = input +
                    ((batch * input_height + in_y_origin + fy) * input_width +
                     in_x_origin + fx) *
                        depth + channel;
                value = __riscv_vfmax_vv_f32m1(
                    value, __riscv_vle32_v_f32m1(in, vl), vl);
              }
            }
            value = __riscv_vfmax_vf_f32m1(value, activation_min, vl);
            value = __riscv_vfmin_vf_f32m1(value, activation_max, vl);
            __riscv_vse32_v_f32m1(out + channel, value, vl);
          }
          channel += static_cast<int>(vl);
        }
      }
    }
  }
}

inline bool AveragePoolUint8(const uint8_t* input, uint8_t* output, int depth,
                             int input_height, int input_width,
                             int output_height, int output_width, int stride_h,
                             int stride_w, int filter_h, int filter_w,
                             int pad_h, int pad_w, int batches, int min_value,
                             int max_value) {
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = out_x * stride_w - pad_w;
        const int in_y_origin = out_y * stride_h - pad_h;
        const int filter_x_start = std::max(0, -in_x_origin);
        const int filter_x_end = std::min(filter_w, input_width - in_x_origin);
        const int filter_y_start = std::max(0, -in_y_origin);
        const int filter_y_end = std::min(filter_h, input_height - in_y_origin);
        const int filter_count = (filter_x_end - filter_x_start) *
                                 (filter_y_end - filter_y_start);
        if (filter_count <= 0) return false;
        uint8_t* out = output +
                       ((batch * output_height + out_y) * output_width + out_x) *
                           depth;
        int channel = 0;
        while (channel < depth) {
          const size_t vl = RvvSetVlE8M1Stack(depth - channel);
          vuint32m4_t sums = __riscv_vmv_v_x_u32m4(0, vl);
          for (int fy = filter_y_start; fy < filter_y_end; ++fy) {
            for (int fx = filter_x_start; fx < filter_x_end; ++fx) {
              const uint8_t* in = input +
                  ((batch * input_height + in_y_origin + fy) * input_width +
                   in_x_origin + fx) *
                      depth + channel;
              const vuint16m2_t values16 = __riscv_vzext_vf2_u16m2(
                  __riscv_vle8_v_u8m1(in, vl), vl);
              sums = __riscv_vadd_vv_u32m4(
                  sums, __riscv_vzext_vf2_u32m4(values16, vl), vl);
            }
          }
          uint32_t sums_scalar[64];
          __riscv_vse32_v_u32m4(sums_scalar, sums, vl);
          for (size_t lane = 0; lane < vl; ++lane) {
            int32_t value = static_cast<int32_t>(
                (sums_scalar[lane] + filter_count / 2) / filter_count);
            value = std::max(min_value, std::min(max_value, value));
            out[channel + lane] = static_cast<uint8_t>(value);
          }
          channel += static_cast<int>(vl);
        }
      }
    }
  }
  return true;
}

inline bool AveragePoolInt8(const int8_t* input, int8_t* output, int depth,
                            int input_height, int input_width,
                            int output_height, int output_width, int stride_h,
                            int stride_w, int filter_h, int filter_w,
                            int pad_h, int pad_w, int batches, int min_value,
                            int max_value) {
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = out_x * stride_w - pad_w;
        const int in_y_origin = out_y * stride_h - pad_h;
        const int filter_x_start = std::max(0, -in_x_origin);
        const int filter_x_end = std::min(filter_w, input_width - in_x_origin);
        const int filter_y_start = std::max(0, -in_y_origin);
        const int filter_y_end = std::min(filter_h, input_height - in_y_origin);
        const int filter_count = (filter_x_end - filter_x_start) *
                                 (filter_y_end - filter_y_start);
        if (filter_count <= 0) return false;
        int8_t* out = output +
                      ((batch * output_height + out_y) * output_width + out_x) *
                          depth;
        int channel = 0;
        while (channel < depth) {
          const size_t vl = RvvSetVlE8M1Stack(depth - channel);
          vint32m4_t sums = __riscv_vmv_v_x_i32m4(0, vl);
          for (int fy = filter_y_start; fy < filter_y_end; ++fy) {
            for (int fx = filter_x_start; fx < filter_x_end; ++fx) {
              const int8_t* in = input +
                  ((batch * input_height + in_y_origin + fy) * input_width +
                   in_x_origin + fx) *
                      depth + channel;
              const vint16m2_t values16 = __riscv_vsext_vf2_i16m2(
                  __riscv_vle8_v_i8m1(in, vl), vl);
              sums = __riscv_vadd_vv_i32m4(
                  sums, __riscv_vsext_vf2_i32m4(values16, vl), vl);
            }
          }
          int32_t sums_scalar[64];
          __riscv_vse32_v_i32m4(sums_scalar, sums, vl);
          for (size_t lane = 0; lane < vl; ++lane) {
            const int32_t sum = sums_scalar[lane];
            int32_t value = sum >= 0 ? (sum + filter_count / 2) / filter_count
                                     : (sum - filter_count / 2) / filter_count;
            value = std::max(min_value, std::min(max_value, value));
            out[channel + lane] = static_cast<int8_t>(value);
          }
          channel += static_cast<int>(vl);
        }
      }
    }
  }
  return true;
}

// Reuse each filter vector for two adjacent output pixels when their x-window
// is identical; the caller keeps all boundary and VLEN fallbacks.
inline void RvvDepthwiseConvFloatPixelPair(
    const float* input, const float* filter, const float* bias, float* output0,
    float* output1, int input_width, int input_depth, int output_depth,
    int filter_width, int filter_y_start, int filter_y_end,
    int filter_x_start, int filter_x_end, int in_y, int in_x,
    int dilation_height, int input_filter_step, int input_pixel_step,
    float activation_min, float activation_max, size_t vlmax) {
  int channel = 0;
  while (channel < output_depth) {
    const size_t remaining = output_depth - channel;
    const size_t vl = remaining < vlmax ? __riscv_vsetvl_e32m2(remaining)
                                        : vlmax;
    vfloat32m2_t accum0 =
        bias == nullptr ? __riscv_vfmv_v_f_f32m2(0.0f, vl)
                        : __riscv_vle32_v_f32m2(bias + channel, vl);
    vfloat32m2_t accum1 =
        bias == nullptr ? __riscv_vfmv_v_f_f32m2(0.0f, vl)
                        : __riscv_vle32_v_f32m2(bias + channel, vl);
    for (int fy = filter_y_start; fy < filter_y_end; ++fy) {
      const int input_y = in_y + fy * dilation_height;
      for (int fx = filter_x_start; fx < filter_x_end; ++fx) {
        const float* weights =
            filter + (fy * filter_width + fx) * output_depth + channel;
        const float* input0 =
            input + (input_y * input_width + in_x) * input_depth +
            fx * input_filter_step + channel;
        const float* input1 = input0 + input_pixel_step;
        const vfloat32m2_t weights_v = __riscv_vle32_v_f32m2(weights, vl);
        accum0 = __riscv_vfmacc_vv_f32m2(
            accum0, weights_v, __riscv_vle32_v_f32m2(input0, vl), vl);
        accum1 = __riscv_vfmacc_vv_f32m2(
            accum1, weights_v, __riscv_vle32_v_f32m2(input1, vl), vl);
      }
    }
    accum0 = __riscv_vfmax_vf_f32m2(accum0, activation_min, vl);
    accum0 = __riscv_vfmin_vf_f32m2(accum0, activation_max, vl);
    accum1 = __riscv_vfmax_vf_f32m2(accum1, activation_min, vl);
    accum1 = __riscv_vfmin_vf_f32m2(accum1, activation_max, vl);
    __riscv_vse32_v_f32m2(output0 + channel, accum0, vl);
    __riscv_vse32_v_f32m2(output1 + channel, accum1, vl);
    channel += static_cast<int>(vl);
  }
}

inline void DepthwiseConvFloat(
    const float* input, const float* filter, const float* bias, float* output,
    int batches, int input_height, int input_width, int input_depth,
    int filter_height, int filter_width, int output_height, int output_width,
    int output_depth, int depth_multiplier, int stride_height, int stride_width,
    int pad_height, int pad_width, int dilation_height, int dilation_width,
    float activation_min, float activation_max, int thread_start,
    int thread_end, int thread_dim) {
  const int batch_start = thread_dim == 0 ? thread_start : 0;
  const int batch_end = thread_dim == 0 ? thread_end : batches;
  const int row_start = thread_dim == 1 ? thread_start : 0;
  const int row_end = thread_dim == 1 ? thread_end : output_height;
  const bool use_m2 = tensor_utils::RvvVlenClassBits() >= 256;
  const size_t vlmax = depth_multiplier == 1
                           ? (use_m2 ? __riscv_vsetvl_e32m2(output_depth)
                                     : __riscv_vsetvl_e32m1(output_depth))
                           : __riscv_vsetvl_e32m1(depth_multiplier);
  for (int batch = batch_start; batch < batch_end; ++batch) {
    for (int out_y = row_start; out_y < row_end; ++out_y) {
      const int in_y_origin = out_y * stride_height - pad_height;
      const int filter_y_start = dilation_height == 1
                                     ? std::max(0, -in_y_origin)
                                     : std::max(
                                           0, (-in_y_origin +
                                                   dilation_height - 1) /
                                                  dilation_height);
      const int filter_y_end = dilation_height == 1
                                   ? std::min(filter_height,
                                              input_height - in_y_origin)
                                   : std::min(
                                         filter_height,
                                         (input_height - in_y_origin +
                                          dilation_height - 1) /
                                             dilation_height);
      for (int out_x = 0; out_x < output_width;) {
        const int in_x_origin = out_x * stride_width - pad_width;
        const int filter_x_start = dilation_width == 1
                                       ? std::max(0, -in_x_origin)
                                       : std::max(
                                             0, (-in_x_origin +
                                                     dilation_width - 1) /
                                                    dilation_width);
        const int filter_x_end = dilation_width == 1
                                     ? std::min(filter_width,
                                                input_width - in_x_origin)
                                     : std::min(
                                           filter_width,
                                           (input_width - in_x_origin +
                                            dilation_width - 1) /
                                               dilation_width);
        float* out = output +
                     ((batch * output_height + out_y) * output_width + out_x) *
                         output_depth;
        if (depth_multiplier == 1 && use_m2 && out_x + 1 < output_width) {
          const int next_in_x_origin = (out_x + 1) * stride_width - pad_width;
          const int next_filter_x_start =
              dilation_width == 1
                  ? std::max(0, -next_in_x_origin)
                  : std::max(
                        0, (-next_in_x_origin + dilation_width - 1) /
                               dilation_width);
          const int next_filter_x_end =
              dilation_width == 1
                  ? std::min(filter_width, input_width - next_in_x_origin)
                  : std::min(
                        filter_width,
                        (input_width - next_in_x_origin + dilation_width - 1) /
                            dilation_width);
          if (filter_x_start == next_filter_x_start &&
              filter_x_end == next_filter_x_end) {
            float* next_out = out + output_depth;
            RvvDepthwiseConvFloatPixelPair(
                input + batch * input_height * input_width * input_depth,
                filter, bias, out, next_out, input_width, input_depth,
                output_depth, filter_width, filter_y_start, filter_y_end,
                filter_x_start, filter_x_end, in_y_origin, in_x_origin,
                dilation_height, dilation_width * input_depth,
                stride_width * input_depth, activation_min, activation_max,
                vlmax);
            out_x += 2;
            continue;
          }
        }
        if (depth_multiplier == 1) {
          // Mirror the FP32 GEMM VLEN dispatch: on VLEN>=256 the wider m2
          // register group halves the loop/instruction overhead per channel
          // element and measured 1.07-1.33x on k1 (Spacemit X60); VLEN=128
          // keeps the m1 path.
          if (use_m2) {
            int channel = 0;
            while (channel < output_depth) {
              const size_t remaining = output_depth - channel;
              const size_t vl = remaining < vlmax
                                    ? __riscv_vsetvl_e32m2(remaining)
                                    : vlmax;
              vfloat32m2_t accum = bias == nullptr
                                        ? __riscv_vfmv_v_f_f32m2(0.0f, vl)
                                        : __riscv_vle32_v_f32m2(bias + channel, vl);
              for (int fy = filter_y_start; fy < filter_y_end; ++fy) {
                const int in_y = in_y_origin + fy * dilation_height;
                for (int fx = filter_x_start; fx < filter_x_end; ++fx) {
                  const int in_x = in_x_origin + fx * dilation_width;
                  const float* in = input +
                      ((batch * input_height + in_y) * input_width + in_x) *
                          input_depth + channel;
                  const float* weights =
                      filter + (fy * filter_width + fx) * output_depth +
                      channel;
                  accum = __riscv_vfmacc_vv_f32m2_tu(
                      accum, __riscv_vle32_v_f32m2(weights, vl),
                      __riscv_vle32_v_f32m2(in, vl), vl);
                }
              }
              accum = __riscv_vfmax_vf_f32m2(accum, activation_min, vl);
              accum = __riscv_vfmin_vf_f32m2(accum, activation_max, vl);
              __riscv_vse32_v_f32m2(out + channel, accum, vl);
              channel += static_cast<int>(vl);
            }
          } else {
            int channel = 0;
            while (channel < output_depth) {
              const size_t remaining = output_depth - channel;
              const size_t vl = remaining < vlmax
                                    ? __riscv_vsetvl_e32m1(remaining)
                                    : vlmax;
              vfloat32m1_t accum = bias == nullptr
                                        ? __riscv_vfmv_v_f_f32m1(0.0f, vl)
                                        : __riscv_vle32_v_f32m1(bias + channel, vl);
              for (int fy = filter_y_start; fy < filter_y_end; ++fy) {
                const int in_y = in_y_origin + fy * dilation_height;
                for (int fx = filter_x_start; fx < filter_x_end; ++fx) {
                  const int in_x = in_x_origin + fx * dilation_width;
                  const float* in = input +
                      ((batch * input_height + in_y) * input_width + in_x) *
                          input_depth + channel;
                  const float* weights =
                      filter + (fy * filter_width + fx) * output_depth +
                      channel;
                  accum = __riscv_vfmacc_vv_f32m1_tu(
                      accum, __riscv_vle32_v_f32m1(weights, vl),
                      __riscv_vle32_v_f32m1(in, vl), vl);
                }
              }
              accum = __riscv_vfmax_vf_f32m1(accum, activation_min, vl);
              accum = __riscv_vfmin_vf_f32m1(accum, activation_max, vl);
              __riscv_vse32_v_f32m1(out + channel, accum, vl);
              channel += static_cast<int>(vl);
            }
          }
        } else {
          // For depth_multiplier > 1, output channels for one input channel
          // are contiguous. Vectorize that fan-out instead of re-reading the
          // same input scalar for every output channel.
          for (int input_channel = 0; input_channel < input_depth;
               ++input_channel) {
            int multiplier = 0;
            while (multiplier < depth_multiplier) {
              const size_t remaining = depth_multiplier - multiplier;
              const size_t vl = remaining < vlmax
                                    ? __riscv_vsetvl_e32m1(remaining)
                                    : vlmax;
              const int output_channel =
                  input_channel * depth_multiplier + multiplier;
              vfloat32m1_t accum =
                  bias == nullptr
                      ? __riscv_vfmv_v_f_f32m1(0.0f, vl)
                      : __riscv_vle32_v_f32m1(bias + output_channel, vl);
              for (int fy = filter_y_start; fy < filter_y_end; ++fy) {
                const int in_y = in_y_origin + fy * dilation_height;
                for (int fx = filter_x_start; fx < filter_x_end; ++fx) {
                  const int in_x = in_x_origin + fx * dilation_width;
                  const float input_value =
                      input[((batch * input_height + in_y) * input_width +
                             in_x) * input_depth + input_channel];
                  const float* weights =
                      filter + (fy * filter_width + fx) * output_depth +
                      output_channel;
                  accum = __riscv_vfmacc_vf_f32m1(
                      accum, input_value,
                      __riscv_vle32_v_f32m1(weights, vl), vl);
                }
              }
              accum = __riscv_vfmax_vf_f32m1(accum, activation_min, vl);
              accum = __riscv_vfmin_vf_f32m1(accum, activation_max, vl);
              __riscv_vse32_v_f32m1(out + output_channel, accum, vl);
              multiplier += static_cast<int>(vl);
            }
          }
        }
        ++out_x;
      }
    }
  }
}

inline void RvvDepthwiseConvUint8PixelPair(
    const uint8_t* input, const uint8_t* filter, const int32_t* bias,
    uint8_t* output0, uint8_t* output1, int input_width, int input_depth,
    int output_depth, int filter_width, int filter_y_start, int filter_y_end,
    int filter_x_start, int filter_x_end, int in_y, int in_x,
    int dilation_height, int input_filter_step, int input_pixel_step,
    int32_t input_offset, int32_t filter_offset, int32_t output_offset,
    int32_t output_multiplier, int output_shift, int32_t activation_min,
    int32_t activation_max, size_t vlmax) {
  int channel = 0;
  while (channel < output_depth) {
    const size_t remaining = output_depth - channel;
    const size_t vl = remaining < vlmax
                          ? __riscv_vsetvl_e32m4(remaining)
                          : vlmax;
    vint32m4_t accum0 = bias == nullptr
                            ? __riscv_vmv_v_x_i32m4(0, vl)
                            : __riscv_vle32_v_i32m4(bias + channel, vl);
    vint32m4_t accum1 = accum0;
    for (int fy = filter_y_start; fy < filter_y_end; ++fy) {
      const int input_y = in_y + fy * dilation_height;
      for (int fx = filter_x_start; fx < filter_x_end; ++fx) {
        const uint8_t* weights =
            filter + (fy * filter_width + fx) * output_depth + channel;
        const uint8_t* input0 =
            input + (input_y * input_width + in_x) * input_depth +
            fx * input_filter_step + channel;
        const uint8_t* input1 = input0 + input_pixel_step;
        vuint16m2_t weights16_u = __riscv_vzext_vf2_u16m2(
            __riscv_vle8_v_u8m1(weights, vl), vl);
        vuint16m2_t input16_u0 = __riscv_vzext_vf2_u16m2(
            __riscv_vle8_v_u8m1(input0, vl), vl);
        vuint16m2_t input16_u1 = __riscv_vzext_vf2_u16m2(
            __riscv_vle8_v_u8m1(input1, vl), vl);
        if (filter_offset != 0) {
          weights16_u = __riscv_vadd_vx_u16m2(
              weights16_u, static_cast<uint16_t>(filter_offset), vl);
        }
        if (input_offset != 0) {
          input16_u0 = __riscv_vadd_vx_u16m2(
              input16_u0, static_cast<uint16_t>(input_offset), vl);
          input16_u1 = __riscv_vadd_vx_u16m2(
              input16_u1, static_cast<uint16_t>(input_offset), vl);
        }
        const vint16m2_t weights16 =
            __riscv_vreinterpret_v_u16m2_i16m2(weights16_u);
        accum0 = __riscv_vwmacc_vv_i32m4(
            accum0, __riscv_vreinterpret_v_u16m2_i16m2(input16_u0), weights16,
            vl);
        accum1 = __riscv_vwmacc_vv_i32m4(
            accum1, __riscv_vreinterpret_v_u16m2_i16m2(input16_u1), weights16,
            vl);
      }
    }
    vint32m4_t value0 =
        RvvVectorizedQuantize(accum0, output_multiplier, output_shift, vl);
    vint32m4_t value1 =
        RvvVectorizedQuantize(accum1, output_multiplier, output_shift, vl);
    if (output_offset != 0) {
      value0 = __riscv_vadd_vx_i32m4(value0, output_offset, vl);
      value1 = __riscv_vadd_vx_i32m4(value1, output_offset, vl);
    }
    value0 = __riscv_vmax_vx_i32m4(value0, activation_min, vl);
    value0 = __riscv_vmin_vx_i32m4(value0, activation_max, vl);
    value1 = __riscv_vmax_vx_i32m4(value1, activation_min, vl);
    value1 = __riscv_vmin_vx_i32m4(value1, activation_max, vl);
    const vuint16m2_t value16_0 = __riscv_vncvt_x_x_w_u16m2(
        __riscv_vreinterpret_v_i32m4_u32m4(value0), vl);
    const vuint16m2_t value16_1 = __riscv_vncvt_x_x_w_u16m2(
        __riscv_vreinterpret_v_i32m4_u32m4(value1), vl);
    __riscv_vse8_v_u8m1(output0 + channel,
                        __riscv_vncvt_x_x_w_u8m1(value16_0, vl), vl);
    __riscv_vse8_v_u8m1(output1 + channel,
                        __riscv_vncvt_x_x_w_u8m1(value16_1, vl), vl);
    channel += static_cast<int>(vl);
  }
}

inline void DepthwiseConvUint8(
    const uint8_t* input, const uint8_t* filter, const int32_t* bias,
    uint8_t* output, int batches, int input_height, int input_width,
    int input_depth, int filter_height, int filter_width, int output_height,
    int output_width, int output_depth, int depth_multiplier, int stride_height,
    int stride_width, int pad_height, int pad_width, int dilation_height,
    int dilation_width, int32_t input_offset, int32_t filter_offset,
    int32_t output_offset, int32_t output_multiplier, int output_shift,
    int32_t activation_min, int32_t activation_max, int thread_start,
    int thread_end, int thread_dim) {
  const int batch_start = thread_dim == 0 ? thread_start : 0;
  const int batch_end = thread_dim == 0 ? thread_end : batches;
  const int row_start = thread_dim == 1 ? thread_start : 0;
  const int row_end = thread_dim == 1 ? thread_end : output_height;
  const bool use_pair = tensor_utils::RvvVlenClassBits() >= 128;
  // e8m1 and e32m4 have the same VLMAX, so each output-channel block can
  // reuse its e32m4 VL for byte loads instead of issuing a second vsetvl.
  if (depth_multiplier == 1) {
    const size_t vlmax = __riscv_vsetvl_e32m4(output_depth);
    // Channel-vectorized path (mirrors DepthwiseConvInt8PerChannel): uint8
    // input/filter are zero-extended to i16, shifted by the offsets, and
    // accumulated with vwmacc into an i32 vector per output pixel. No
    // stack-buffer round trip inside the filter loop.
    for (int batch = batch_start; batch < batch_end; ++batch) {
      for (int out_y = row_start; out_y < row_end; ++out_y) {
        const int in_y_origin = out_y * stride_height - pad_height;
        const int filter_y_start = dilation_height == 1
                                       ? std::max(0, -in_y_origin)
                                       : std::max(
                                             0, (-in_y_origin +
                                                     dilation_height - 1) /
                                                    dilation_height);
        const int filter_y_end = dilation_height == 1
                                     ? std::min(filter_height,
                                                input_height - in_y_origin)
                                     : std::min(
                                           filter_height,
                                           (input_height - in_y_origin +
                                            dilation_height - 1) /
                                               dilation_height);
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = out_x * stride_width - pad_width;
          const int filter_x_start = dilation_width == 1
                                         ? std::max(0, -in_x_origin)
                                         : std::max(
                                               0, (-in_x_origin +
                                                       dilation_width - 1) /
                                                      dilation_width);
        const int filter_x_end = dilation_width == 1
                                     ? std::min(filter_width,
                                                input_width - in_x_origin)
                                     : std::min(
                                           filter_width,
                                           (input_width - in_x_origin +
                                            dilation_width - 1) /
                                               dilation_width);
          uint8_t* out = output +
                         ((batch * output_height + out_y) * output_width +
                          out_x) * output_depth;
          if (use_pair && out_x + 1 < output_width) {
            const int next_in_x_origin =
                (out_x + 1) * stride_width - pad_width;
            const int next_filter_x_start =
                dilation_width == 1
                    ? std::max(0, -next_in_x_origin)
                    : std::max(
                          0, (-next_in_x_origin + dilation_width - 1) /
                                 dilation_width);
            const int next_filter_x_end =
                dilation_width == 1
                    ? std::min(filter_width, input_width - next_in_x_origin)
                    : std::min(
                          filter_width,
                          (input_width - next_in_x_origin + dilation_width - 1) /
                              dilation_width);
            if (filter_x_start == next_filter_x_start &&
                filter_x_end == next_filter_x_end) {
              RvvDepthwiseConvUint8PixelPair(
                  input + batch * input_height * input_width * input_depth,
                  filter, bias, out, out + output_depth, input_width,
                  input_depth, output_depth, filter_width, filter_y_start,
                  filter_y_end, filter_x_start, filter_x_end, in_y_origin,
                  in_x_origin, dilation_height, dilation_width * input_depth,
                  stride_width * input_depth, input_offset, filter_offset,
                  output_offset, output_multiplier, output_shift,
                  activation_min, activation_max, vlmax);
              ++out_x;
              continue;
            }
          }
           int channel = 0;
           while (channel < output_depth) {
             const size_t remaining = output_depth - channel;
             const size_t vl = remaining < vlmax
                                   ? __riscv_vsetvl_e32m4(remaining)
                                   : vlmax;
            vint32m4_t accum = bias == nullptr
                                   ? __riscv_vmv_v_x_i32m4(0, vl)
                                   : __riscv_vle32_v_i32m4(bias + channel, vl);
              for (int fy = filter_y_start; fy < filter_y_end; ++fy) {
                const int in_y = in_y_origin + fy * dilation_height;
                for (int fx = filter_x_start; fx < filter_x_end; ++fx) {
                  const int in_x = in_x_origin + fx * dilation_width;
                  const uint8_t* in = input +
                    ((batch * input_height + in_y) * input_width + in_x) *
                        input_depth + channel;
                const uint8_t* w = filter +
                    (fy * filter_width + fx) * output_depth + channel;
                const size_t vl8 = vl;
                // uint8 quantized values use non-positive zero-point offsets;
                // widen to i16 then add so the offset keeps its signedness.
                vuint16m2_t in16_u = __riscv_vzext_vf2_u16m2(
                    __riscv_vle8_v_u8m1(in, vl8), vl8);
                if (input_offset != 0) {
                  in16_u = __riscv_vadd_vx_u16m2(
                      in16_u, static_cast<uint16_t>(input_offset), vl8);
                }
                const vint16m2_t in16 =
                    __riscv_vreinterpret_v_u16m2_i16m2(in16_u);
                vuint16m2_t w16_u = __riscv_vzext_vf2_u16m2(
                    __riscv_vle8_v_u8m1(w, vl8), vl8);
                if (filter_offset != 0) {
                  w16_u = __riscv_vadd_vx_u16m2(
                      w16_u, static_cast<uint16_t>(filter_offset), vl8);
                }
                const vint16m2_t w16 =
                    __riscv_vreinterpret_v_u16m2_i16m2(w16_u);
                accum = __riscv_vwmacc_vv_i32m4(accum, in16, w16, vl8);
              }
            }
            vint32m4_t value = RvvVectorizedQuantize(
                accum, output_multiplier, output_shift, vl);
            if (output_offset != 0) {
              value = __riscv_vadd_vx_i32m4(value, output_offset, vl);
            }
            value = __riscv_vmax_vx_i32m4(value, activation_min, vl);
            value = __riscv_vmin_vx_i32m4(value, activation_max, vl);
            const vuint16m2_t out16 = __riscv_vncvt_x_x_w_u16m2(
                __riscv_vreinterpret_v_i32m4_u32m4(value), vl);
            const vuint8m1_t out8 = __riscv_vncvt_x_x_w_u8m1(out16, vl);
            __riscv_vse8_v_u8m1(out + channel, out8, vl);
            channel += static_cast<int>(vl);
          }
        }
      }
    }
  } else {
    const size_t vlmax = __riscv_vsetvl_e32m4(depth_multiplier);
    // For depth_multiplier > 1, vectorize the contiguous fan-out for each
    // input channel. The common output multiplier lets the quantization and
    // narrowing stay vectorized after accumulation.
    for (int batch = batch_start; batch < batch_end; ++batch) {
      for (int out_y = row_start; out_y < row_end; ++out_y) {
        const int in_y_origin = out_y * stride_height - pad_height;
        const int filter_y_start = dilation_height == 1
                                       ? std::max(0, -in_y_origin)
                                       : std::max(
                                             0, (-in_y_origin +
                                                     dilation_height - 1) /
                                                    dilation_height);
        const int filter_y_end = dilation_height == 1
                                     ? std::min(filter_height,
                                                input_height - in_y_origin)
                                     : std::min(
                                           filter_height,
                                           (input_height - in_y_origin +
                                            dilation_height - 1) /
                                               dilation_height);
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = out_x * stride_width - pad_width;
          const int filter_x_start = dilation_width == 1
                                         ? std::max(0, -in_x_origin)
                                         : std::max(
                                               0, (-in_x_origin +
                                                       dilation_width - 1) /
                                                      dilation_width);
          const int filter_x_end = dilation_width == 1
                                       ? std::min(filter_width,
                                                  input_width - in_x_origin)
                                       : std::min(
                                             filter_width,
                                             (input_width - in_x_origin +
                                              dilation_width - 1) /
                                                 dilation_width);
          uint8_t* out = output +
                         ((batch * output_height + out_y) * output_width +
                          out_x) * output_depth;
          for (int input_channel = 0; input_channel < input_depth;
               ++input_channel) {
            int multiplier = 0;
            while (multiplier < depth_multiplier) {
              const size_t remaining = depth_multiplier - multiplier;
              const size_t vl = remaining < vlmax
                                    ? __riscv_vsetvl_e32m4(remaining)
                                    : vlmax;
              const int output_channel =
                  input_channel * depth_multiplier + multiplier;
              vint32m4_t accum =
                  bias == nullptr
                      ? __riscv_vmv_v_x_i32m4(0, vl)
                      : __riscv_vle32_v_i32m4(bias + output_channel, vl);
              for (int fy = filter_y_start; fy < filter_y_end; ++fy) {
                const int in_y = in_y_origin + fy * dilation_height;
                for (int fx = filter_x_start; fx < filter_x_end; ++fx) {
                  const int in_x = in_x_origin + fx * dilation_width;
                  const int16_t input_value = static_cast<int16_t>(
                      static_cast<int32_t>(
                          input[((batch * input_height + in_y) * input_width +
                                 in_x) * input_depth + input_channel]) +
                      input_offset);
                  const uint8_t* weights =
                      filter + (fy * filter_width + fx) * output_depth +
                      output_channel;
                  const size_t vl8 = vl;
                  vuint16m2_t weights16_u = __riscv_vzext_vf2_u16m2(
                      __riscv_vle8_v_u8m1(weights, vl8), vl8);
                  if (filter_offset != 0) {
                    weights16_u = __riscv_vadd_vx_u16m2(
                        weights16_u, static_cast<uint16_t>(filter_offset),
                        vl8);
                  }
                  const vint16m2_t weights16 =
                      __riscv_vreinterpret_v_u16m2_i16m2(weights16_u);
                  accum = __riscv_vwmacc_vx_i32m4(accum, input_value,
                                                  weights16, vl8);
                }
              }
              vint32m4_t value = RvvVectorizedQuantize(
                  accum, output_multiplier, output_shift, vl);
              if (output_offset != 0) {
                value = __riscv_vadd_vx_i32m4(value, output_offset, vl);
              }
              value = __riscv_vmax_vx_i32m4(value, activation_min, vl);
              value = __riscv_vmin_vx_i32m4(value, activation_max, vl);
              const vuint16m2_t out16 = __riscv_vncvt_x_x_w_u16m2(
                  __riscv_vreinterpret_v_i32m4_u32m4(value), vl);
              const vuint8m1_t out8 = __riscv_vncvt_x_x_w_u8m1(out16, vl);
              __riscv_vse8_v_u8m1(out + output_channel, out8, vl);
              multiplier += static_cast<int>(vl);
            }
          }
        }
      }
    }
  }
}

// Reuse one per-channel filter vector for two adjacent pixels. The caller
// only enters when both clipped filter windows are identical.
inline void RvvDepthwiseConvInt8PixelPair(
    const int8_t* input, const int8_t* filter, const int32_t* bias,
    int8_t* output0, int8_t* output1, int input_width, int input_depth,
    int output_depth, int filter_width, int filter_y_start, int filter_y_end,
    int filter_x_start, int filter_x_end, int in_y, int in_x,
    int dilation_height, int input_filter_step, int input_pixel_step,
    int32_t input_offset, int32_t output_offset, const int32_t* multipliers,
    const int32_t* shifts, int32_t activation_min, int32_t activation_max,
    bool use_vectorized_quantize, size_t vlmax) {
  int channel = 0;
  while (channel < output_depth) {
    const size_t remaining = output_depth - channel;
    const size_t vl = remaining < vlmax
                          ? RvvSetVlE32M4ForQuantize(
                                remaining, use_vectorized_quantize)
                          : vlmax;
    vint32m4_t accum0 = bias == nullptr
                            ? __riscv_vmv_v_x_i32m4(0, vl)
                            : __riscv_vle32_v_i32m4(bias + channel, vl);
    vint32m4_t accum1 = accum0;
    for (int fy = filter_y_start; fy < filter_y_end; ++fy) {
      const int input_y = in_y + fy * dilation_height;
      for (int fx = filter_x_start; fx < filter_x_end; ++fx) {
        const int8_t* weights =
            filter + (fy * filter_width + fx) * output_depth + channel;
        const int8_t* input0 =
            input + (input_y * input_width + in_x) * input_depth +
            fx * input_filter_step + channel;
        const int8_t* input1 = input0 + input_pixel_step;
        const vint16m2_t weights16 = __riscv_vsext_vf2_i16m2(
            __riscv_vle8_v_i8m1(weights, vl), vl);
        vint16m2_t input16_0 = __riscv_vsext_vf2_i16m2(
            __riscv_vle8_v_i8m1(input0, vl), vl);
        vint16m2_t input16_1 = __riscv_vsext_vf2_i16m2(
            __riscv_vle8_v_i8m1(input1, vl), vl);
        if (input_offset != 0) {
          input16_0 = __riscv_vadd_vx_i16m2(input16_0, input_offset, vl);
          input16_1 = __riscv_vadd_vx_i16m2(input16_1, input_offset, vl);
        }
        accum0 = __riscv_vwmacc_vv_i32m4(accum0, input16_0, weights16, vl);
        accum1 = __riscv_vwmacc_vv_i32m4(accum1, input16_1, weights16, vl);
      }
    }
    vint32m4_t value0 = RvvVectorizedQuantizePerChannel(
        accum0, multipliers + channel, shifts + channel, vl,
        use_vectorized_quantize);
    vint32m4_t value1 = RvvVectorizedQuantizePerChannel(
        accum1, multipliers + channel, shifts + channel, vl,
        use_vectorized_quantize);
    if (output_offset != 0) {
      value0 = __riscv_vadd_vx_i32m4(value0, output_offset, vl);
      value1 = __riscv_vadd_vx_i32m4(value1, output_offset, vl);
    }
    value0 = __riscv_vmax_vx_i32m4(value0, activation_min, vl);
    value0 = __riscv_vmin_vx_i32m4(value0, activation_max, vl);
    value1 = __riscv_vmax_vx_i32m4(value1, activation_min, vl);
    value1 = __riscv_vmin_vx_i32m4(value1, activation_max, vl);
    const vint16m2_t value16_0 = __riscv_vncvt_x_x_w_i16m2(value0, vl);
    const vint16m2_t value16_1 = __riscv_vncvt_x_x_w_i16m2(value1, vl);
    __riscv_vse8_v_i8m1(output0 + channel,
                        __riscv_vncvt_x_x_w_i8m1(value16_0, vl), vl);
    __riscv_vse8_v_i8m1(output1 + channel,
                        __riscv_vncvt_x_x_w_i8m1(value16_1, vl), vl);
    channel += static_cast<int>(vl);
  }
}

inline void DepthwiseConvInt8PerChannel(
    const int8_t* input, const int8_t* filter, const int32_t* bias,
    int8_t* output, int batches, int input_height, int input_width,
    int input_depth, int filter_height, int filter_width, int output_height,
    int output_width, int output_depth, int depth_multiplier, int stride_height,
    int stride_width, int pad_height, int pad_width, int dilation_height,
    int dilation_width, int32_t input_offset, int32_t output_offset,
    const int32_t* multipliers, const int32_t* shifts, int32_t activation_min,
    int32_t activation_max, int thread_start, int thread_end, int thread_dim) {
  const int batch_start = thread_dim == 0 ? thread_start : 0;
  const int batch_end = thread_dim == 0 ? thread_end : batches;
  const int row_start = thread_dim == 1 ? thread_start : 0;
  const int row_end = thread_dim == 1 ? thread_end : output_height;
  const bool use_vlen_m2 = tensor_utils::RvvVlenClassBits() >= 256;
#if !defined(TFLITE_SINGLE_ROUNDING) || !TFLITE_SINGLE_ROUNDING
  const bool use_vectorized_quantize = use_vlen_m2;
#else
  constexpr bool use_vectorized_quantize = true;
#endif
  // The two-pixel accumulator raises register pressure on the A/B target;
  // the single-pixel channel path is faster on VLEN=256 and 512.
  constexpr bool use_pair = false;
  // e8m1 and e32m4 have the same VLMAX, so each output-channel block can
  // reuse its e32m4 VL for byte loads instead of issuing a second vsetvl.
  if (depth_multiplier == 1) {
    const size_t vlmax = RvvSetVlE32M4ForQuantize(
        output_depth, use_vectorized_quantize);
    // Channel-vectorized path: for each output pixel, accumulate over the
    // filter taps with per-channel vector FMAs (int8 input x int8 filter ->
    // int32 accumulator). Each output channel is independent, so the whole
    // channel dimension is processed in VLEN-sized vectors.
    for (int batch = batch_start; batch < batch_end; ++batch) {
      for (int out_y = row_start; out_y < row_end; ++out_y) {
        const int in_y_origin = out_y * stride_height - pad_height;
        const int filter_y_start = dilation_height == 1
                                       ? std::max(0, -in_y_origin)
                                       : std::max(
                                             0, (-in_y_origin +
                                                     dilation_height - 1) /
                                                    dilation_height);
        const int filter_y_end = dilation_height == 1
                                     ? std::min(filter_height,
                                                input_height - in_y_origin)
                                     : std::min(
                                           filter_height,
                                           (input_height - in_y_origin +
                                            dilation_height - 1) /
                                               dilation_height);
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = out_x * stride_width - pad_width;
          const int filter_x_start = dilation_width == 1
                                         ? std::max(0, -in_x_origin)
                                         : std::max(
                                               0, (-in_x_origin +
                                                       dilation_width - 1) /
                                                      dilation_width);
          const int filter_x_end = dilation_width == 1
                                       ? std::min(filter_width,
                                                  input_width - in_x_origin)
                                       : std::min(
                                             filter_width,
                                             (input_width - in_x_origin +
                                              dilation_width - 1) /
                                                 dilation_width);
          int8_t* out = output +
                        ((batch * output_height + out_y) * output_width +
                         out_x) * output_depth;
          if (use_pair && out_x + 1 < output_width) {
            const int next_in_x_origin =
                (out_x + 1) * stride_width - pad_width;
            const int next_filter_x_start =
                dilation_width == 1
                    ? std::max(0, -next_in_x_origin)
                    : std::max(
                          0, (-next_in_x_origin + dilation_width - 1) /
                                 dilation_width);
            const int next_filter_x_end =
                dilation_width == 1
                    ? std::min(filter_width, input_width - next_in_x_origin)
                    : std::min(
                          filter_width,
                          (input_width - next_in_x_origin + dilation_width - 1) /
                              dilation_width);
            if (filter_x_start == next_filter_x_start &&
                filter_x_end == next_filter_x_end) {
              RvvDepthwiseConvInt8PixelPair(
                  input + batch * input_height * input_width * input_depth,
                  filter, bias, out, out + output_depth, input_width,
                  input_depth, output_depth, filter_width, filter_y_start,
                  filter_y_end, filter_x_start, filter_x_end, in_y_origin,
                  in_x_origin, dilation_height, dilation_width * input_depth,
                  stride_width * input_depth, input_offset, output_offset,
                  multipliers, shifts, activation_min, activation_max,
                  use_vectorized_quantize, vlmax);
              ++out_x;
              continue;
            }
          }
           int channel = 0;
           while (channel < output_depth) {
             const size_t remaining = output_depth - channel;
             const size_t vl = remaining < vlmax
                                   ? RvvSetVlE32M4ForQuantize(
                                         remaining, use_vectorized_quantize)
                                   : vlmax;
            vint32m4_t accum = bias == nullptr
                                   ? __riscv_vmv_v_x_i32m4(0, vl)
                                   : __riscv_vle32_v_i32m4(bias + channel, vl);
              for (int fy = filter_y_start; fy < filter_y_end; ++fy) {
                const int in_y = in_y_origin + fy * dilation_height;
                for (int fx = filter_x_start; fx < filter_x_end; ++fx) {
                  const int in_x = in_x_origin + fx * dilation_width;
                  const int8_t* in = input +
                    ((batch * input_height + in_y) * input_width + in_x) *
                        input_depth + channel;
                const int8_t* w = filter +
                    (fy * filter_width + fx) * output_depth + channel;
                const size_t vl8 = vl;
                vint16m2_t in16 = __riscv_vsext_vf2_i16m2(
                    __riscv_vle8_v_i8m1(in, vl8), vl8);
                vint16m2_t w16 = __riscv_vsext_vf2_i16m2(
                    __riscv_vle8_v_i8m1(w, vl8), vl8);
                if (input_offset != 0) {
                  in16 = __riscv_vadd_vx_i16m2(in16, input_offset, vl8);
                }
                accum = __riscv_vwmacc_vv_i32m4(accum, in16, w16, vl8);
              }
            }
            vint32m4_t value = RvvVectorizedQuantizePerChannel(
                accum, multipliers + channel, shifts + channel, vl,
                use_vectorized_quantize);
            if (output_offset != 0) {
              value = __riscv_vadd_vx_i32m4(value, output_offset, vl);
            }
            value = __riscv_vmax_vx_i32m4(value, activation_min, vl);
            value = __riscv_vmin_vx_i32m4(value, activation_max, vl);
            const vint16m2_t out16 =
                __riscv_vncvt_x_x_w_i16m2(value, vl);
            const vint8m1_t out8 = __riscv_vncvt_x_x_w_i8m1(out16, vl);
            __riscv_vse8_v_i8m1(out + channel, out8, vl);
            channel += static_cast<int>(vl);
          }
        }
      }
    }
  } else {
    const size_t vlmax = RvvSetVlE32M4ForQuantize(
        depth_multiplier, use_vectorized_quantize);
    // For depth_multiplier > 1, vectorize the contiguous fan-out for each
    // input channel. Both rounding contracts stay in the vector path.
    for (int batch = batch_start; batch < batch_end; ++batch) {
      for (int out_y = row_start; out_y < row_end; ++out_y) {
        const int in_y_origin = out_y * stride_height - pad_height;
        const int filter_y_start = dilation_height == 1
                                       ? std::max(0, -in_y_origin)
                                       : std::max(
                                             0, (-in_y_origin +
                                                     dilation_height - 1) /
                                                    dilation_height);
        const int filter_y_end = dilation_height == 1
                                     ? std::min(filter_height,
                                                input_height - in_y_origin)
                                     : std::min(
                                           filter_height,
                                           (input_height - in_y_origin +
                                            dilation_height - 1) /
                                               dilation_height);
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = out_x * stride_width - pad_width;
          const int filter_x_start = dilation_width == 1
                                         ? std::max(0, -in_x_origin)
                                         : std::max(
                                               0, (-in_x_origin +
                                                       dilation_width - 1) /
                                                      dilation_width);
          const int filter_x_end = dilation_width == 1
                                       ? std::min(filter_width,
                                                  input_width - in_x_origin)
                                       : std::min(
                                             filter_width,
                                             (input_width - in_x_origin +
                                              dilation_width - 1) /
                                                 dilation_width);
          int8_t* out = output +
                        ((batch * output_height + out_y) * output_width +
                         out_x) * output_depth;
          for (int input_channel = 0; input_channel < input_depth;
               ++input_channel) {
            int multiplier = 0;
            while (multiplier < depth_multiplier) {
              const size_t remaining = depth_multiplier - multiplier;
              const size_t vl = remaining < vlmax
                                    ? RvvSetVlE32M4ForQuantize(
                                          remaining, use_vectorized_quantize)
                                    : vlmax;
              const int output_channel =
                  input_channel * depth_multiplier + multiplier;
              vint32m4_t accum =
                  bias == nullptr
                      ? __riscv_vmv_v_x_i32m4(0, vl)
                      : __riscv_vle32_v_i32m4(bias + output_channel, vl);
              for (int fy = filter_y_start; fy < filter_y_end; ++fy) {
                const int in_y = in_y_origin + fy * dilation_height;
                for (int fx = filter_x_start; fx < filter_x_end; ++fx) {
                  const int in_x = in_x_origin + fx * dilation_width;
                  const int16_t input_value = static_cast<int16_t>(
                      static_cast<int32_t>(
                          input[((batch * input_height + in_y) * input_width +
                                 in_x) * input_depth + input_channel]) +
                      input_offset);
                  const int8_t* weights =
                      filter + (fy * filter_width + fx) * output_depth +
                      output_channel;
                  const size_t vl8 = vl;
                  const vint16m2_t weights16 = __riscv_vsext_vf2_i16m2(
                      __riscv_vle8_v_i8m1(weights, vl8), vl8);
                  accum = __riscv_vwmacc_vx_i32m4(accum, input_value,
                                                  weights16, vl8);
                }
              }
              vint32m4_t value = RvvVectorizedQuantizePerChannel(
                  accum, multipliers + output_channel, shifts + output_channel,
                  vl, use_vectorized_quantize);
              if (output_offset != 0) {
                value = __riscv_vadd_vx_i32m4(value, output_offset, vl);
              }
              value = __riscv_vmax_vx_i32m4(value, activation_min, vl);
              value = __riscv_vmin_vx_i32m4(value, activation_max, vl);
              const vint16m2_t out16 =
                  __riscv_vncvt_x_x_w_i16m2(value, vl);
              const vint8m1_t out8 = __riscv_vncvt_x_x_w_i8m1(out16, vl);
              __riscv_vse8_v_i8m1(out + output_channel, out8, vl);
              multiplier += static_cast<int>(vl);
            }
          }
        }
      }
    }
  }
}

inline void ShuffledFullyConnectedWorker(
    const uint8_t* shuffled_input, const int8_t* shuffled_weights, int batches,
    int output_depth, int output_stride, int accum_depth,
    const int32_t* bias, int32_t output_multiplier, int output_shift,
    int16_t* output) {
  for (int batch = 0; batch < batches; ++batch) {
    for (int row = 0; row < output_depth; ++row) {
      int32_t accumulated = 0;
      for (int depth = 0; depth < accum_depth; ) {
        const int block = std::min(16, accum_depth - depth);
        const size_t vl = RvvSetVlE8M1(block);
        const int weight_block = (row / 4) * accum_depth * 4 +
                                 (depth / 16) * 64 + (row % 4) * 16 +
                                 (depth % 16);
        const int input_block =
            (batches == 1) ? depth : (depth / 16) * 64 + batch * 16;
        vint8m1_t weights = __riscv_vle8_v_i8m1(
            shuffled_weights + weight_block, vl);
        vint8m1_t values = __riscv_vle8_v_i8m1(
            reinterpret_cast<const int8_t*>(shuffled_input + input_block), vl);
        vint16m2_t products = __riscv_vwmul_vv_i16m2(weights, values, vl);
        vint32m4_t products32 = __riscv_vsext_vf2_i32m4(products, vl);
        vint32m1_t zero = __riscv_vmv_s_x_i32m1(0, vl);
        vint32m1_t partial =
            __riscv_vredsum_vs_i32m4_i32m1(products32, zero, vl);
        accumulated += __riscv_vmv_x_s_i32m1_i32(partial);
        depth += static_cast<int>(vl);
      }
      int32_t value = MultiplyByQuantizedMultiplier(
          accumulated + (bias == nullptr ? 0 : bias[row]), output_multiplier,
          output_shift);
      value = std::max(-32768, std::min(32767, value));
      output[batch * output_stride + row] = static_cast<int16_t>(value);
    }
  }
}

// C++ signed division truncates toward zero.  RVV's arithmetic right shift
// floors negative values, so add one only for a negative, non-integral lane.
inline vint32m4_t RvvTruncDivideByPOT32(vint32m4_t value, int exponent,
                                        size_t vl) {
  if (exponent <= 0) return value;
  const uint32_t mask_u = exponent == 31
                              ? 0x7fffffffU
                              : ((uint32_t{1} << exponent) - 1);
  const int32_t mask = static_cast<int32_t>(mask_u);
  const vint32m4_t floor = __riscv_vsra_vx_i32m4(value, exponent, vl);
  const vint32m4_t remainder = __riscv_vand_vx_i32m4(value, mask, vl);
  const vbool8_t negative = __riscv_vmsne_vx_i32m4_b8(
      __riscv_vsra_vx_i32m4(value, 31, vl), 0, vl);
  const vbool8_t non_integral = __riscv_vmsne_vx_i32m4_b8(
      remainder, 0, vl);
  const vbool8_t correction =
      __riscv_vmand_mm_b8(negative, non_integral, vl);
  return __riscv_vadd_vv_i32m4(
      floor,
      __riscv_vmerge_vvm_i32m4(__riscv_vmv_v_x_i32m4(0, vl),
                               __riscv_vmv_v_x_i32m4(1, vl), correction, vl),
      vl);
}

inline vint32m4_t RvvRoundingDivideByPOT32(vint32m4_t value, int exponent,
                                           size_t vl) {
  if (exponent <= 0) return value;
  const uint32_t mask_u = exponent == 31
                              ? 0x7fffffffU
                              : ((uint32_t{1} << exponent) - 1);
  const int32_t mask = static_cast<int32_t>(mask_u);
  const vint32m4_t floor = __riscv_vsra_vx_i32m4(value, exponent, vl);
  const vint32m4_t remainder = __riscv_vand_vx_i32m4(value, mask, vl);
  const vbool8_t negative = __riscv_vmsne_vx_i32m4_b8(
      __riscv_vsra_vx_i32m4(value, 31, vl), 0, vl);
  const vint32m4_t threshold = __riscv_vadd_vv_i32m4(
      __riscv_vmv_v_x_i32m4(mask >> 1, vl),
      __riscv_vmerge_vxm_i32m4(__riscv_vmv_v_x_i32m4(0, vl), 1, negative, vl),
      vl);
  const vbool8_t round_up =
      __riscv_vmsgt_vv_i32m4_b8(remainder, threshold, vl);
  return __riscv_vadd_vv_i32m4(
      floor,
      __riscv_vmerge_vvm_i32m4(__riscv_vmv_v_x_i32m4(0, vl),
                               __riscv_vmv_v_x_i32m4(1, vl), round_up, vl),
      vl);
}

// Exact scalar contract for (value * multiplier + nudge) / 32768.  The
// nudge is selected from value, matching the reference HardSwish code rather
// than from the product sign.
inline vint32m4_t RvvHardSwishMultiplyQ15(vint16m2_t value,
                                          int16_t multiplier, size_t vl) {
  const vint32m4_t value32 = __riscv_vsext_vf2_i32m4(value, vl);
  const vint32m4_t product =
      __riscv_vwmul_vx_i32m4(value, multiplier, vl);
  const vbool8_t negative = __riscv_vmsne_vx_i32m4_b8(
      __riscv_vsra_vx_i32m4(value32, 31, vl), 0, vl);
  const vint32m4_t nudged = __riscv_vadd_vv_i32m4(
      product,
      __riscv_vmerge_vxm_i32m4(__riscv_vmv_v_x_i32m4(1 << 14, vl),
                               1 - (1 << 14), negative, vl),
      vl);
  return RvvTruncDivideByPOT32(nudged, 15, vl);
}

template <typename T>
inline void HardSwishQuantized(const T* input, T* output, int size,
                               int input_zero_point,
                               int output_zero_point,
                               int output_multiplier_exponent,
                               int output_multiplier_fixedpoint,
                               int reluish_multiplier_exponent,
                               int reluish_multiplier_fixedpoint) {
  int offset = 0;
  while (offset < size) {
    // e8m1, e16m2 and e32m4 have the same lane count.  No stack scratch is
    // needed: the whole HardSwish fixed-point pipeline stays in registers.
    const size_t vl = RvvSetVlE8M1(size - offset);
    vint16m2_t input_value;
    if constexpr (std::is_same<T, int8_t>::value) {
      input_value = __riscv_vsext_vf2_i16m2(
          __riscv_vle8_v_i8m1(input + offset, vl), vl);
    } else {
      input_value = __riscv_vreinterpret_v_u16m2_i16m2(
          __riscv_vzext_vf2_u16m2(
              __riscv_vle8_v_u8m1(input + offset, vl), vl));
    }
    input_value = __riscv_vsub_vx_i16m2(
        input_value, static_cast<int16_t>(input_zero_point), vl);
    const vint16m2_t hires = __riscv_vsll_vx_i16m2(input_value, 7, vl);

    const vint16m2_t preshift = __riscv_vncvt_x_x_w_i16m2(
        RvvTruncDivideByPOT32(
            __riscv_vwmul_vx_i32m4(
                hires, static_cast<int16_t>(output_multiplier_fixedpoint), vl),
            15, vl),
        vl);

    vint16m2_t reluish = hires;
    if (reluish_multiplier_exponent > 0) {
      vint32m4_t shifted = __riscv_vsll_vx_i32m4(
          __riscv_vsext_vf2_i32m4(reluish, vl),
          reluish_multiplier_exponent - 1, vl);
      shifted = __riscv_vmax_vx_i32m4(
          __riscv_vmin_vx_i32m4(shifted, INT16_MAX, vl), INT16_MIN, vl);
      reluish = __riscv_vncvt_x_x_w_i16m2(shifted, vl);
    }
    reluish = __riscv_vncvt_x_x_w_i16m2(
        RvvHardSwishMultiplyQ15(
            reluish, static_cast<int16_t>(reluish_multiplier_fixedpoint), vl),
        vl);
    if (reluish_multiplier_exponent > 0) {
      vint32m4_t shifted = __riscv_vsll_vx_i32m4(
          __riscv_vsext_vf2_i32m4(reluish, vl), 1, vl);
      shifted = __riscv_vmax_vx_i32m4(
          __riscv_vmin_vx_i32m4(shifted, INT16_MAX, vl), INT16_MIN, vl);
      reluish = __riscv_vncvt_x_x_w_i16m2(shifted, vl);
    } else if (reluish_multiplier_exponent < 0) {
      const int shift = -reluish_multiplier_exponent;
      const vint32m4_t reluish32 = __riscv_vsext_vf2_i32m4(reluish, vl);
      reluish = __riscv_vncvt_x_x_w_i16m2(
          RvvRoundingDivideByPOT32(reluish32, shift, vl), vl);
    }
    reluish = __riscv_vncvt_x_x_w_i16m2(
        __riscv_vsra_vx_i32m4(
            __riscv_vadd_vx_i32m4(
                __riscv_vsext_vf2_i32m4(reluish, vl), 1 << 15, vl),
            1, vl),
        vl);

    vint16m2_t product = __riscv_vncvt_x_x_w_i16m2(
        RvvTruncDivideByPOT32(
            __riscv_vwmul_vv_i32m4(reluish, preshift, vl), 15, vl),
        vl);
    if (output_multiplier_exponent < 0) {
      const int shift = -output_multiplier_exponent;
      const vint32m4_t product32 = __riscv_vsext_vf2_i32m4(product, vl);
      product = __riscv_vncvt_x_x_w_i16m2(
          RvvRoundingDivideByPOT32(product32, shift, vl), vl);
    } else if (output_multiplier_exponent > 0) {
      product = __riscv_vncvt_x_x_w_i16m2(
          __riscv_vsll_vx_i32m4(
              __riscv_vsext_vf2_i32m4(product, vl),
              output_multiplier_exponent, vl),
          vl);
    }

    vint32m4_t result = __riscv_vadd_vx_i32m4(
        __riscv_vsext_vf2_i32m4(product, vl), output_zero_point, vl);
    if constexpr (std::is_same<T, int8_t>::value) {
      result = __riscv_vmax_vx_i32m4(
          __riscv_vmin_vx_i32m4(result, INT8_MAX, vl), INT8_MIN, vl);
      const vint16m2_t result16 =
          __riscv_vncvt_x_x_w_i16m2(result, vl);
      __riscv_vse8_v_i8m1(output + offset,
                          __riscv_vncvt_x_x_w_i8m1(result16, vl), vl);
    } else {
      result = __riscv_vmax_vx_i32m4(
          __riscv_vmin_vx_i32m4(result, UINT8_MAX, vl), 0, vl);
      const vuint16m2_t result16 = __riscv_vreinterpret_v_i16m2_u16m2(
          __riscv_vncvt_x_x_w_i16m2(result, vl));
      __riscv_vse8_v_u8m1(output + offset,
                          __riscv_vncvt_x_x_w_u8m1(result16, vl), vl);
    }
    offset += static_cast<int>(vl);
  }
}

template <typename In, typename Out>
inline void SoftmaxInt8Lut(const In* input, Out* output, int outer_size,
                          int last_dim, const uint8_t* table1,
                          const uint8_t* table2, float output_scale,
                          int32_t output_zero_point) {
  constexpr int32_t output_min = std::numeric_limits<Out>::min();
  constexpr int32_t output_max = std::numeric_limits<Out>::max();
  const uint8_t input_offset =
      std::is_same<In, int8_t>::value ? static_cast<uint8_t>(0x80) : 0;
  for (int row = 0; row < outer_size; ++row) {
    const In* row_input = input + row * last_dim;
    Out* row_output = output + row * last_dim;
    int32_t max_value = 0;
    for (int offset = 0; offset < last_dim;) {
      const size_t vl = RvvSetVlE8M1(last_dim - offset);
      const vuint8m1_t values = __riscv_vle8_v_u8m1(
          reinterpret_cast<const uint8_t*>(row_input + offset), vl);
      const vuint8m1_t centered =
          __riscv_vxor_vx_u8m1(values, input_offset, vl);
      const vuint8m1_t max_init = __riscv_vmv_s_x_u8m1(0, vl);
      const vuint8m1_t max_vector =
          __riscv_vredmaxu_vs_u8m1_u8m1(centered, max_init, vl);
      max_value = std::max(
          max_value,
          static_cast<int32_t>(__riscv_vmv_x_s_u8m1_u8(max_vector)));
      offset += static_cast<int>(vl);
    }
    const uint8_t table_offset = static_cast<uint8_t>(255 - max_value);
    int32_t sum_exp = 0;
    for (int offset = 0; offset < last_dim;) {
      const size_t vl = RvvSetVlE8M1(last_dim - offset);
      const vuint8m1_t centered = __riscv_vxor_vx_u8m1(
          __riscv_vle8_v_u8m1(
              reinterpret_cast<const uint8_t*>(row_input + offset), vl),
          input_offset, vl);
      const vuint8m1_t indices8 =
          __riscv_vadd_vx_u8m1(centered, table_offset, vl);
      const vuint8m1_t table1_values =
          __riscv_vluxei8_v_u8m1(table1, indices8, vl);
      const vuint8m1_t table2_values =
          __riscv_vluxei8_v_u8m1(table2, indices8, vl);
      vuint32m4_t exp_values = __riscv_vzext_vf2_u32m4(
          __riscv_vzext_vf2_u16m2(table1_values, vl), vl);
      exp_values = __riscv_vsll_vx_u32m4(exp_values, 8, vl);
      exp_values = __riscv_vadd_vv_u32m4(
          exp_values,
          __riscv_vzext_vf2_u32m4(
              __riscv_vzext_vf2_u16m2(table2_values, vl), vl),
          vl);
      const vuint32m1_t sum_init = __riscv_vmv_s_x_u32m1(0, vl);
      const vuint32m1_t sum_vector =
          __riscv_vredsum_vs_u32m4_u32m1(exp_values, sum_init, vl);
      sum_exp += static_cast<int32_t>(
          __riscv_vmv_x_s_u32m1_u32(sum_vector));
      offset += static_cast<int>(vl);
    }
    int32_t multiplier;
    int shift;
    QuantizeMultiplier(1.0 / (sum_exp * output_scale), &multiplier, &shift);
    for (int offset = 0; offset < last_dim;) {
      const size_t vl = RvvSetVlE8M1(last_dim - offset);
      const vuint8m1_t centered = __riscv_vxor_vx_u8m1(
          __riscv_vle8_v_u8m1(
              reinterpret_cast<const uint8_t*>(row_input + offset), vl),
          input_offset, vl);
      const vuint8m1_t indices8 =
          __riscv_vadd_vx_u8m1(centered, table_offset, vl);
      const vuint8m1_t table1_values =
          __riscv_vluxei8_v_u8m1(table1, indices8, vl);
      const vuint8m1_t table2_values =
          __riscv_vluxei8_v_u8m1(table2, indices8, vl);
      vuint32m4_t exp_values = __riscv_vzext_vf2_u32m4(
          __riscv_vzext_vf2_u16m2(table1_values, vl), vl);
      exp_values = __riscv_vsll_vx_u32m4(exp_values, 8, vl);
      exp_values = __riscv_vadd_vv_u32m4(
          exp_values,
          __riscv_vzext_vf2_u32m4(
              __riscv_vzext_vf2_u16m2(table2_values, vl), vl),
          vl);
      vint32m4_t result = RvvVectorizedQuantize(
          __riscv_vreinterpret_v_u32m4_i32m4(exp_values), multiplier, shift,
          vl);
      if (output_zero_point != 0) {
        result = __riscv_vadd_vx_i32m4(result, output_zero_point, vl);
      }
      result = __riscv_vmax_vx_i32m4(result, output_min, vl);
      result = __riscv_vmin_vx_i32m4(result, output_max, vl);
      if constexpr (std::is_same_v<Out, int8_t>) {
        const vint16m2_t result16 =
            __riscv_vncvt_x_x_w_i16m2(result, vl);
        __riscv_vse8_v_i8m1(
            reinterpret_cast<int8_t*>(row_output + offset),
            __riscv_vncvt_x_x_w_i8m1(result16, vl), vl);
      } else {
        const vuint16m2_t result16 = __riscv_vncvt_x_x_w_u16m2(
            __riscv_vreinterpret_v_i32m4_u32m4(result), vl);
        __riscv_vse8_v_u8m1(
            reinterpret_cast<uint8_t*>(row_output + offset),
            __riscv_vncvt_x_x_w_u8m1(result16, vl), vl);
      }
      offset += static_cast<int>(vl);
    }
  }
}

inline void AddFloat(const float* input1, const float* input2, float* output,
                     int size, float activation_min, float activation_max) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e32m1(size - offset);
    vfloat32m1_t value = __riscv_vle32_v_f32m1(input1 + offset, vl);
    value = __riscv_vfadd_vv_f32m1(
        value, __riscv_vle32_v_f32m1(input2 + offset, vl), vl);
    value = __riscv_vfmax_vf_f32m1(value, activation_min, vl);
    value = __riscv_vfmin_vf_f32m1(value, activation_max, vl);
    __riscv_vse32_v_f32m1(output + offset, value, vl);
    offset += static_cast<int>(vl);
  }
}

inline void AddScalarFloat(const float* input, float broadcast, float* output,
                           int size, float activation_min,
                           float activation_max) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e32m1(size - offset);
    vfloat32m1_t value = __riscv_vle32_v_f32m1(input + offset, vl);
    value = __riscv_vfadd_vf_f32m1(value, broadcast, vl);
    value = __riscv_vfmax_vf_f32m1(value, activation_min, vl);
    value = __riscv_vfmin_vf_f32m1(value, activation_max, vl);
    __riscv_vse32_v_f32m1(output + offset, value, vl);
    offset += static_cast<int>(vl);
  }
}

inline void MulFloat(const float* input1, const float* input2, float* output,
                     int size, float activation_min, float activation_max) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e32m1(size - offset);
    vfloat32m1_t value = __riscv_vfmul_vv_f32m1(
        __riscv_vle32_v_f32m1(input1 + offset, vl),
        __riscv_vle32_v_f32m1(input2 + offset, vl), vl);
    value = __riscv_vfmax_vf_f32m1(value, activation_min, vl);
    value = __riscv_vfmin_vf_f32m1(value, activation_max, vl);
    __riscv_vse32_v_f32m1(output + offset, value, vl);
    offset += static_cast<int>(vl);
  }
}

inline void MulScalarFloat(const float* input, float broadcast, float* output,
                           int size, float activation_min,
                           float activation_max) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e32m1(size - offset);
    vfloat32m1_t value = __riscv_vfmul_vf_f32m1(
        __riscv_vle32_v_f32m1(input + offset, vl), broadcast, vl);
    value = __riscv_vfmax_vf_f32m1(value, activation_min, vl);
    value = __riscv_vfmin_vf_f32m1(value, activation_max, vl);
    __riscv_vse32_v_f32m1(output + offset, value, vl);
    offset += static_cast<int>(vl);
  }
}

inline void PReluScalarFloat(const float* input, float alpha, float* output,
                             int size) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e32m1(size - offset);
    const vfloat32m1_t value = __riscv_vle32_v_f32m1(input + offset, vl);
    const vfloat32m1_t scaled = __riscv_vfmul_vf_f32m1(value, alpha, vl);
    const vbool32_t nonnegative = __riscv_vmfge_vf_f32m1_b32(value, 0.0f, vl);
    const vfloat32m1_t result =
        __riscv_vmerge_vvm_f32m1(scaled, value, nonnegative, vl);
    __riscv_vse32_v_f32m1(output + offset, result, vl);
    offset += static_cast<int>(vl);
  }
}

inline void PReluElementwiseFloat(const float* alpha, const float* input,
                                  float* output, int size) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e32m1(size - offset);
    const vfloat32m1_t value = __riscv_vle32_v_f32m1(input + offset, vl);
    const vfloat32m1_t scaled = __riscv_vfmul_vv_f32m1(
        value, __riscv_vle32_v_f32m1(alpha + offset, vl), vl);
    const vbool32_t nonnegative = __riscv_vmfge_vf_f32m1_b32(value, 0.0f, vl);
    const vfloat32m1_t result =
        __riscv_vmerge_vvm_f32m1(scaled, value, nonnegative, vl);
    __riscv_vse32_v_f32m1(output + offset, result, vl);
    offset += static_cast<int>(vl);
  }
}

inline void HardSwishFloat(const float* input, float* output, int size) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e32m1(size - offset);
    vfloat32m1_t value = __riscv_vle32_v_f32m1(input + offset, vl);
    vfloat32m1_t gate = __riscv_vfadd_vf_f32m1(value, 3.0f, vl);
    gate = __riscv_vfmax_vf_f32m1(gate, 0.0f, vl);
    gate = __riscv_vfmin_vf_f32m1(gate, 6.0f, vl);
    value = __riscv_vfmul_vv_f32m1(value, gate, vl);
    value = __riscv_vfmul_vf_f32m1(value, 1.0f / 6.0f, vl);
    __riscv_vse32_v_f32m1(output + offset, value, vl);
    offset += static_cast<int>(vl);
  }
}

inline void MaximumInt8(const int8_t* input1, const int8_t* input2,
                        int8_t* output, int size) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e8m1(size - offset);
    const vint8m1_t value = __riscv_vmax_vv_i8m1(
        __riscv_vle8_v_i8m1(input1 + offset, vl),
        __riscv_vle8_v_i8m1(input2 + offset, vl), vl);
    __riscv_vse8_v_i8m1(output + offset, value, vl);
    offset += static_cast<int>(vl);
  }
}

inline void MaximumScalarInt8(const int8_t* input, int8_t broadcast,
                              int8_t* output, int size) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e8m1(size - offset);
    const vint8m1_t value = __riscv_vmax_vx_i8m1(
        __riscv_vle8_v_i8m1(input + offset, vl), broadcast, vl);
    __riscv_vse8_v_i8m1(output + offset, value, vl);
    offset += static_cast<int>(vl);
  }
}

inline void MinimumInt8(const int8_t* input1, const int8_t* input2,
                        int8_t* output, int size) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e8m1(size - offset);
    const vint8m1_t value = __riscv_vmin_vv_i8m1(
        __riscv_vle8_v_i8m1(input1 + offset, vl),
        __riscv_vle8_v_i8m1(input2 + offset, vl), vl);
    __riscv_vse8_v_i8m1(output + offset, value, vl);
    offset += static_cast<int>(vl);
  }
}

inline void MinimumScalarInt8(const int8_t* input, int8_t broadcast,
                              int8_t* output, int size) {
  int offset = 0;
  while (offset < size) {
    const size_t vl = __riscv_vsetvl_e8m1(size - offset);
    const vint8m1_t value = __riscv_vmin_vx_i8m1(
        __riscv_vle8_v_i8m1(input + offset, vl), broadcast, vl);
    __riscv_vse8_v_i8m1(output + offset, value, vl);
    offset += static_cast<int>(vl);
  }
}

}  // namespace rvv_optimized_ops
}  // namespace tflite

#endif  // defined(__riscv_vector)

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_OPTIMIZED_OPS_H_
