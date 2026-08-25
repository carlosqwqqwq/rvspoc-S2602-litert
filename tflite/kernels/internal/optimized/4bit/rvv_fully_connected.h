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

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_4BIT_RVV_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_4BIT_RVV_FULLY_CONNECTED_H_

#if defined(FC_4BIT_RVV) && defined(__riscv_vector)

#include <riscv_vector.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "tflite/kernels/internal/optimized/4bit/fully_connected_common.h"
#include "tflite/kernels/internal/optimized/4bit/fully_connected_reference_impl.h"

namespace tflite {
namespace optimized_4bit {

inline int GetMaxSupportedRows() { return 4; }

inline void PackInner(const int8_t* src, uint8_t* box, int src_rows,
                      int src_cols, int outer_row, int outer_col,
                      int outer_rows, int outer_cols, int inner_rows,
                      int inner_cols) {
  const int width = inner_rows;
  const int depth = inner_cols;
  const int real_depth = depth / 2;
  const int real_src_cols = src_cols / 2;
  const int row = outer_row * inner_rows;
  const int col = outer_col * inner_cols;
  const int src_width = std::min(width, src_rows - row);
  const int src_depth = std::min(depth, src_cols - col);

  // The production kernel consumes the 4x32 layout. Preserve the reference
  // contract for partial or non-production blocks, including its 0x77 pad.
  if (depth != 32 || src_depth != depth || src_width <= 0) {
    ReferencePackInner(src, box, src_rows, src_cols, outer_row, outer_col,
                       outer_rows, outer_cols, inner_rows, inner_cols);
    return;
  }

  const int real_col = col / 2;
  for (int m = 0; m < src_width; ++m) {
    const int8_t* src_data =
        src + (row + m) * real_src_cols + real_col;
    uint8_t* row_box = box + m * real_depth;
    for (int offset = 0; offset < 8;) {
      const size_t vl = __riscv_vsetvl_e8m1(8 - offset);
      const vuint8m1_t packed1 = __riscv_vle8_v_u8m1(
          reinterpret_cast<const uint8_t*>(src_data + offset), vl);
      const vuint8m1_t packed2 =
          __riscv_vle8_v_u8m1(
              reinterpret_cast<const uint8_t*>(src_data + 8 + offset), vl);

      // Arithmetic shifts reproduce upper()/lower() and adding seven maps
      // signed int4 values to the unsigned nibbles expected by the kernel.
      const vint8m1_t packed1_i =
          __riscv_vreinterpret_v_u8m1_i8m1(packed1);
      const vint8m1_t packed2_i =
          __riscv_vreinterpret_v_u8m1_i8m1(packed2);
      const vint8m1_t upper1 = __riscv_vsra_vx_i8m1(packed1_i, 4, vl);
      const vint8m1_t upper2 = __riscv_vsra_vx_i8m1(packed2_i, 4, vl);
      const vint8m1_t lower1 = __riscv_vsra_vx_i8m1(
          __riscv_vreinterpret_v_u8m1_i8m1(
              __riscv_vsll_vx_u8m1(packed1, 4, vl)),
          4, vl);
      const vint8m1_t lower2 = __riscv_vsra_vx_i8m1(
          __riscv_vreinterpret_v_u8m1_i8m1(
              __riscv_vsll_vx_u8m1(packed2, 4, vl)),
          4, vl);
      const vuint8m1_t upper1_code = __riscv_vreinterpret_v_i8m1_u8m1(
          __riscv_vadd_vx_i8m1(upper1, 7, vl));
      const vuint8m1_t upper2_code = __riscv_vreinterpret_v_i8m1_u8m1(
          __riscv_vadd_vx_i8m1(upper2, 7, vl));
      const vuint8m1_t lower1_code = __riscv_vreinterpret_v_i8m1_u8m1(
          __riscv_vadd_vx_i8m1(lower1, 7, vl));
      const vuint8m1_t lower2_code = __riscv_vreinterpret_v_i8m1_u8m1(
          __riscv_vadd_vx_i8m1(lower2, 7, vl));
      const vuint8m1_t lower_merged = __riscv_vor_vv_u8m1(
          __riscv_vsll_vx_u8m1(lower1_code, 4, vl), lower2_code, vl);
      const vuint8m1_t upper_merged = __riscv_vor_vv_u8m1(
          __riscv_vsll_vx_u8m1(upper1_code, 4, vl), upper2_code, vl);

      uint8_t lower_values[8];
      uint8_t upper_values[8];
      __riscv_vse8_v_u8m1(lower_values, lower_merged, vl);
      __riscv_vse8_v_u8m1(upper_values, upper_merged, vl);
      for (size_t lane = 0; lane < vl; ++lane) {
        const int output_offset = 2 * (offset + static_cast<int>(lane));
        row_box[output_offset] = lower_values[lane];
        row_box[output_offset + 1] = upper_values[lane];
      }
      offset += static_cast<int>(vl);
    }
  }
}

inline void Prepack(uint8_t* dest, const int8_t* tensor, int layout_rows,
                    int layout_cols, int src_rows, int src_cols, int width,
                    int depth) {
  if (depth != 32 || !(width == 1 || width == 2 || width == 4)) {
    ReferencePrepack(dest, tensor, layout_rows, layout_cols, src_rows, src_cols,
                     width, depth);
    return;
  }

  const size_t size = static_cast<size_t>(layout_rows) * layout_cols / 2;
  std::memset(dest, static_cast<uint8_t>(0x77), size);
  const int outer_cols = layout_cols / depth;
  const int outer_rows = layout_rows / width;
  for (int outer_row = 0; outer_row < outer_rows; ++outer_row) {
    for (int outer_col = 0; outer_col < outer_cols; ++outer_col) {
      const int cluster_index = outer_row * outer_cols + outer_col;
      uint8_t* box = dest + cluster_index * (depth / 2) * width;
      PackInner(tensor, box, src_rows, src_cols, outer_row, outer_col,
                outer_rows, outer_cols, width, depth);
    }
  }
}

inline void BatchQuantizeFloats4Bit(const float* float_data_ptr, int n_batch,
                                    int n_data, int8_t* quantized_data_ptr,
                                    float* scaling_factors, int width,
                                    int depth, int32_t* input_offsets) {
#if defined(__riscv_vector)
  // The production 4bit path uses 4x32 blocks. Keep other layouts on the
  // reference implementation rather than silently changing their packing
  // contract.
  if (depth != 32 ||
      !(width == 1 || width == 2 || width == 4)) {
    ReferenceBatchQuantizeFloats4Bit(
        float_data_ptr, n_batch, n_data, quantized_data_ptr, scaling_factors,
        width, depth, input_offsets);
    return;
  }

  const int layout_rows = (n_batch + (width - 1)) & ~(width - 1);
  const int layout_cols = (n_data + (depth - 1)) & ~(depth - 1);
  std::memset(quantized_data_ptr, 0,
              sizeof(int8_t) * layout_rows * layout_cols);
  std::memset(input_offsets, 0, sizeof(int32_t) * layout_rows);

  const int outer_rows = layout_rows / width;
  const int outer_cols = layout_cols / depth;
  for (int outer_row = 0; outer_row < outer_rows; ++outer_row) {
    float scale[4] = {};
    const int row = outer_row * width;
    for (int w = 0; w < width && row + w < n_batch; ++w) {
      const float* row_data = float_data_ptr + (row + w) * n_data;
      float scale_denom = 0.0f;
      for (int offset = 0; offset < n_data;) {
        const size_t vl = __riscv_vsetvl_e32m4(n_data - offset);
        const vfloat32m4_t values = __riscv_vle32_v_f32m4(
            row_data + offset, vl);
        const vfloat32m4_t absolute = __riscv_vfabs_v_f32m4(values, vl);
        const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        const vfloat32m1_t maximum =
            __riscv_vfredmax_vs_f32m4_f32m1(absolute, zero, vl);
        scale_denom = std::max(
            scale_denom, __riscv_vfmv_f_s_f32m1_f32(maximum));
        offset += static_cast<int>(vl);
      }
      if (scale_denom == 0.0f) {
        scale_denom = 127.0f;
      }
      scale[w] = 127.0f / scale_denom;
      scaling_factors[row + w] = scale_denom / 127.0f;
    }

    for (int outer_col = 0; outer_col < outer_cols; ++outer_col) {
      const int col = outer_col * depth;
      const int src_width = std::min(width, n_batch - row);
      const int src_depth = std::min(depth, n_data - col);
      const int cluster_index = outer_row * outer_cols + outer_col;
      int8_t* box = quantized_data_ptr + cluster_index * depth * width;
      for (int w = 0; w < src_width; ++w) {
        const float* source = float_data_ptr + (row + w) * n_data + col;
        int8_t* destination = box + w * depth;
        int32_t row_sum = 0;
        for (int offset = 0; offset < src_depth;) {
          const size_t vl = __riscv_vsetvl_e32m4(src_depth - offset);
          const vfloat32m4_t values = __riscv_vle32_v_f32m4(
              source + offset, vl);
          const vfloat32m4_t scaled =
              __riscv_vfmul_vf_f32m4(values, scale[w], vl);
          const vfloat32m4_t half =
              __riscv_vfmv_v_f_f32m4(0.5f, vl);
          const vfloat32m4_t signed_half =
              __riscv_vfsgnj_vv_f32m4(half, scaled, vl);
          const vfloat32m4_t rounded_input =
              __riscv_vfadd_vv_f32m4(scaled, signed_half, vl);
          vint32m4_t rounded =
              __riscv_vfcvt_rtz_x_f_v_i32m4(rounded_input, vl);
          rounded = __riscv_vmax_vx_i32m4(rounded, -127, vl);
          rounded = __riscv_vmin_vx_i32m4(rounded, 127, vl);

          const vint32m1_t zero = __riscv_vmv_s_x_i32m1(0, vl);
          const vint32m1_t sum =
              __riscv_vredsum_vs_i32m4_i32m1(rounded, zero, vl);
          row_sum += __riscv_vmv_x_s_i32m1_i32(sum);

          const vint16m2_t rounded16 =
              __riscv_vncvt_x_x_w_i16m2(rounded, vl);
          const vint8m1_t quantized =
              __riscv_vncvt_x_x_w_i8m1(rounded16, vl);
          __riscv_vse8_v_i8m1(destination + offset, quantized, vl);
          offset += static_cast<int>(vl);
        }
        input_offsets[row + w] += row_sum;
      }
    }
  }
  for (int row = 0; row < layout_rows; ++row) {
    input_offsets[row] *= zero_point_4bit;
  }
  return;
#endif
  ReferenceBatchQuantizeFloats4Bit(
      float_data_ptr, n_batch, n_data, quantized_data_ptr, scaling_factors,
      width, depth, input_offsets);
}

inline void AssignBiasAndComputeOffsets(const int32_t* input_offsets,
                                        const float* batch_scales,
                                        float* filter_scales,
                                        const float* bias_ptr,
                                        float* output_ptr, int output_depth,
                                        int batch_size) {
#if defined(__riscv_vector)
  for (int batch = 0; batch < batch_size; ++batch) {
    const float value = static_cast<float>(input_offsets[batch]) *
                        batch_scales[batch];
    for (int offset = 0; offset < output_depth;) {
      const size_t vl = __riscv_vsetvl_e32m1(output_depth - offset);
      vfloat32m1_t result = __riscv_vfmul_vf_f32m1(
          __riscv_vle32_v_f32m1(filter_scales + offset, vl), value, vl);
      if (bias_ptr != nullptr) {
        result = __riscv_vfadd_vv_f32m1(
            result, __riscv_vle32_v_f32m1(bias_ptr + offset, vl), vl);
      }
      __riscv_vse32_v_f32m1(output_ptr + offset, result, vl);
      offset += static_cast<int>(vl);
    }
    output_ptr += output_depth;
  }
  return;
#endif
  ReferenceAssignBiasAndComputeOffsets(
      input_offsets, batch_scales, filter_scales, bias_ptr, output_ptr,
      output_depth, batch_size);
}

template <int Depth, int Width>
void Unpack(float* output_ptr, const int32_t* dst, int batch_size,
            int num_units, const float* scaling_factors,
            const float* filter_scales, int dst_layout_rows,
            int dst_layout_cols) {
  ReferenceUnpack<Depth, Width>(output_ptr, dst, batch_size, num_units,
                                scaling_factors, filter_scales,
                                dst_layout_rows, dst_layout_cols);
}

template <int Width>
inline void Unpack4BitWidth(float* output_ptr, const int32_t* dst,
                            int batch_size, int num_units,
                            const float* scaling_factors,
                            const float* filter_scales, int dst_layout_rows,
                            int dst_layout_cols) {
  const int outer_rows = dst_layout_rows / Width;
  const int outer_cols = dst_layout_cols / 4;
  for (int outer_col = 0; outer_col < outer_cols; ++outer_col) {
    const int unit = outer_col * 4;
    const int remaining_units =
        std::max(0, std::min(num_units - unit, 4));
    if (remaining_units == 0) {
      continue;
    }
    const size_t vl = __riscv_vsetvl_e32m1(remaining_units);
    const vfloat32m1_t filter_values =
        __riscv_vle32_v_f32m1(filter_scales + unit, vl);
    for (int outer_row = 0; outer_row < outer_rows; ++outer_row) {
      const int batch = outer_row * Width;
      const int remaining_width = std::min(batch_size - batch, Width);
      if (remaining_width <= 0) {
        continue;
      }
      const int cluster_index = outer_col * outer_rows + outer_row;
      const int32_t* dst_ptr = dst + cluster_index * 4 * Width;
      float* output_row = output_ptr + batch * num_units + unit;
      const float* scale = scaling_factors + batch;
      for (int w = 0; w < remaining_width; ++w, ++scale) {
        vfloat32m1_t values = __riscv_vfcvt_f_x_v_f32m1(
            __riscv_vle32_v_i32m1(dst_ptr, vl), vl);
        values = __riscv_vfmul_vf_f32m1(values, *scale, vl);
        values = __riscv_vfmul_vv_f32m1(values, filter_values, vl);
        values = __riscv_vfadd_vv_f32m1(
            values, __riscv_vle32_v_f32m1(output_row, vl), vl);
        __riscv_vse32_v_f32m1(output_row, values, vl);
        dst_ptr += 4;
        output_row += num_units;
      }
    }
  }
}

template <>
inline void Unpack<4, 1>(float* output_ptr, const int32_t* dst, int batch_size,
                         int num_units, const float* scaling_factors,
                         const float* filter_scales, int dst_layout_rows,
                         int dst_layout_cols) {
#if defined(__riscv_vector)
  (void)batch_size;
  const int outer_rows = dst_layout_rows;
  const int outer_cols = dst_layout_cols / 4;
  const int32_t* dst_ptr = dst;
  int unit = 0;
  for (int outer_col = 0; outer_col < outer_cols;
       ++outer_col, unit += 4) {
    float* row_output = output_ptr + unit;
    const int len = std::min(num_units - unit, 4);
    const float* scale_ptr = scaling_factors;
    const size_t vl = len > 0 ? __riscv_vsetvl_e32m1(len) : 0;
    const vfloat32m1_t filter_values =
        len > 0 ? __riscv_vle32_v_f32m1(filter_scales + unit, vl)
                : __riscv_vfmv_v_f_f32m1(0.0f, 0);
    for (int outer_row = 0; outer_row < outer_rows; ++outer_row) {
      if (len > 0) {
        vfloat32m1_t values = __riscv_vfcvt_f_x_v_f32m1(
            __riscv_vle32_v_i32m1(dst_ptr, vl), vl);
        values = __riscv_vfmul_vf_f32m1(values, *scale_ptr, vl);
        values = __riscv_vfmul_vv_f32m1(values, filter_values, vl);
        values = __riscv_vfadd_vv_f32m1(
            __riscv_vle32_v_f32m1(row_output, vl), values, vl);
        __riscv_vse32_v_f32m1(row_output, values, vl);
      }
      dst_ptr += 4;
      scale_ptr += 1;
      row_output += num_units;
    }
  }
  return;
#else
  ReferenceUnpack<4, 1>(output_ptr, dst, batch_size, num_units,
                        scaling_factors, filter_scales, dst_layout_rows,
                        dst_layout_cols);
#endif
}

template <>
inline void Unpack<4, 2>(float* output_ptr, const int32_t* dst, int batch_size,
                         int num_units, const float* scaling_factors,
                         const float* filter_scales, int dst_layout_rows,
                         int dst_layout_cols) {
#if defined(__riscv_vector)
  Unpack4BitWidth<2>(output_ptr, dst, batch_size, num_units, scaling_factors,
                     filter_scales, dst_layout_rows, dst_layout_cols);
#else
  ReferenceUnpack<4, 2>(output_ptr, dst, batch_size, num_units,
                        scaling_factors, filter_scales, dst_layout_rows,
                        dst_layout_cols);
#endif
}

template <>
inline void Unpack<4, 4>(float* output_ptr, const int32_t* dst, int batch_size,
                         int num_units, const float* scaling_factors,
                         const float* filter_scales, int dst_layout_rows,
                         int dst_layout_cols) {
#if defined(__riscv_vector)
  Unpack4BitWidth<4>(output_ptr, dst, batch_size, num_units, scaling_factors,
                     filter_scales, dst_layout_rows, dst_layout_cols);
#else
  ReferenceUnpack<4, 4>(output_ptr, dst, batch_size, num_units,
                        scaling_factors, filter_scales, dst_layout_rows,
                        dst_layout_cols);
#endif
}

template <int RowsLeft, int RowsRight, int Cols>
void RunKernel(const uint8_t* lhs, const int8_t* rhs, int32_t* dst,
               int lhs_layout_rows, int lhs_layout_cols, int rhs_layout_rows,
               int rhs_layout_cols, int dst_layout_rows, int dst_layout_cols) {
  ReferenceRunKernel<RowsLeft, RowsRight, Cols>(
      lhs, rhs, dst, lhs_layout_rows, lhs_layout_cols, rhs_layout_rows,
      rhs_layout_cols, dst_layout_rows, dst_layout_cols);
}

// RVV implementation for the 4x1/4x2/4x4 configurations selected by
// GetMaxSupportedRows. The prepacked lhs stores four rows of 32 unsigned int4
// values as 16 bytes per row: high nibbles are the first 16 values and low
// nibbles the second 16. Keeping an int32 vector accumulator across depth
// blocks preserves the reference accumulation width while reducing each output
// only once.
template <int RowsRight>
inline void RunKernel4x32(
    const uint8_t* lhs, const int8_t* rhs, int32_t* dst,
    int lhs_layout_rows, int lhs_layout_cols, int rhs_layout_rows,
    int rhs_layout_cols, int dst_layout_rows, int dst_layout_cols) {
  static_assert(RowsRight == 1 || RowsRight == 2 || RowsRight == 4,
                "unsupported 4bit RVV width");
  const int clamped_end_row = std::min(lhs_layout_rows, dst_layout_cols);
  const int clamped_end_col = std::min(rhs_layout_rows, dst_layout_rows);
  const int outer_rows = (clamped_end_row + 3) / 4;
  const int outer_cols =
      (clamped_end_col + RowsRight - 1) / RowsRight;
  const int depth = std::min(lhs_layout_cols / 32, rhs_layout_cols / 32);
  const size_t vl = __riscv_vsetvl_e8m1(16);
  int32_t* element = dst;

  for (int outer_row = 0; outer_row < outer_rows; ++outer_row) {
    const uint8_t* lhs_cluster =
        lhs + outer_row * 4 * (lhs_layout_cols / 2);
    for (int outer_col = 0; outer_col < outer_cols; ++outer_col) {
      const int8_t* rhs_cluster =
          rhs + outer_col * RowsRight * rhs_layout_cols;
      int32_t results[RowsRight][4] = {};
      for (int row = 0; row < 4; ++row) {
        vint32m4_t accum0 = __riscv_vmv_v_x_i32m4(0, vl);
        vint32m4_t accum1 = __riscv_vmv_v_x_i32m4(0, vl);
        vint32m4_t accum2 = __riscv_vmv_v_x_i32m4(0, vl);
        vint32m4_t accum3 = __riscv_vmv_v_x_i32m4(0, vl);
        for (int block = 0; block < depth; ++block) {
          const uint8_t* lhs_block = lhs_cluster + block * 4 * 16;
          const uint8_t* packed = lhs_block + row * 16;
          const vuint8m1_t packed_values =
              __riscv_vle8_v_u8m1(packed, vl);
          const vuint8m1_t high_nibbles =
              __riscv_vsrl_vx_u8m1(packed_values, 4, vl);
          const vuint8m1_t low_nibbles =
              __riscv_vand_vx_u8m1(packed_values, 0x0f, vl);
          const vint16m2_t high_values = __riscv_vsext_vf2_i16m2(
              __riscv_vreinterpret_v_u8m1_i8m1(high_nibbles), vl);
          const vint16m2_t low_values = __riscv_vsext_vf2_i16m2(
              __riscv_vreinterpret_v_u8m1_i8m1(low_nibbles), vl);
          auto accumulate_rhs = [&](vint32m4_t accumulator,
                                     int right) -> vint32m4_t {
            const int8_t* rhs_block =
                rhs_cluster + block * RowsRight * 32 + right * 32;
            const vint16m2_t rhs_low = __riscv_vsext_vf2_i16m2(
                __riscv_vle8_v_i8m1(rhs_block, vl), vl);
            const vint16m2_t rhs_high = __riscv_vsext_vf2_i16m2(
                __riscv_vle8_v_i8m1(rhs_block + 16, vl), vl);
            accumulator = __riscv_vwmacc_vv_i32m4(
                accumulator, high_values, rhs_low, vl);
            return __riscv_vwmacc_vv_i32m4(
                accumulator, low_values, rhs_high, vl);
          };
          accum0 = accumulate_rhs(accum0, 0);
          if constexpr (RowsRight >= 2) {
            accum1 = accumulate_rhs(accum1, 1);
          }
          if constexpr (RowsRight >= 4) {
            accum2 = accumulate_rhs(accum2, 2);
            accum3 = accumulate_rhs(accum3, 3);
          }
        }
        const vint32m1_t zero = __riscv_vmv_s_x_i32m1(0, vl);
        results[0][row] = __riscv_vmv_x_s_i32m1_i32(
            __riscv_vredsum_vs_i32m4_i32m1(accum0, zero, vl));
        if constexpr (RowsRight >= 2) {
          results[1][row] = __riscv_vmv_x_s_i32m1_i32(
              __riscv_vredsum_vs_i32m4_i32m1(accum1, zero, vl));
        }
        if constexpr (RowsRight >= 4) {
          results[2][row] = __riscv_vmv_x_s_i32m1_i32(
              __riscv_vredsum_vs_i32m4_i32m1(accum2, zero, vl));
          results[3][row] = __riscv_vmv_x_s_i32m1_i32(
              __riscv_vredsum_vs_i32m4_i32m1(accum3, zero, vl));
        }
      }
      for (int right = 0; right < RowsRight; ++right) {
        for (int row = 0; row < 4; ++row) {
          *element++ = results[right][row];
        }
      }
    }
  }
}

template <>
inline void RunKernel<4, 1, 32>(
    const uint8_t* lhs, const int8_t* rhs, int32_t* dst,
    int lhs_layout_rows, int lhs_layout_cols, int rhs_layout_rows,
    int rhs_layout_cols, int dst_layout_rows, int dst_layout_cols) {
  RunKernel4x32<1>(lhs, rhs, dst, lhs_layout_rows, lhs_layout_cols,
                   rhs_layout_rows, rhs_layout_cols, dst_layout_rows,
                   dst_layout_cols);
}

template <>
inline void RunKernel<4, 2, 32>(
    const uint8_t* lhs, const int8_t* rhs, int32_t* dst,
    int lhs_layout_rows, int lhs_layout_cols, int rhs_layout_rows,
    int rhs_layout_cols, int dst_layout_rows, int dst_layout_cols) {
  RunKernel4x32<2>(lhs, rhs, dst, lhs_layout_rows, lhs_layout_cols,
                   rhs_layout_rows, rhs_layout_cols, dst_layout_rows,
                   dst_layout_cols);
}

template <>
inline void RunKernel<4, 4, 32>(
    const uint8_t* lhs, const int8_t* rhs, int32_t* dst,
    int lhs_layout_rows, int lhs_layout_cols, int rhs_layout_rows,
    int rhs_layout_cols, int dst_layout_rows, int dst_layout_cols) {
  RunKernel4x32<4>(lhs, rhs, dst, lhs_layout_rows, lhs_layout_cols,
                   rhs_layout_rows, rhs_layout_cols, dst_layout_rows,
                   dst_layout_cols);
}

inline void RunAndUnpack(int rhs_width, const uint8_t* lhs, const int8_t* rhs,
                         int32_t* dst, int output_depth, int batch_size,
                         int lhs_layout_rows, int lhs_layout_cols,
                         int rhs_layout_rows, int rhs_layout_cols,
                         int dst_layout_rows, int dst_layout_cols,
                         float* output_ptr, const float* scaling_factors,
                         const float* filter_scales) {
  if (rhs_width >= 4) {
    RunKernel<4, 4, 32>(lhs, rhs, dst, lhs_layout_rows, lhs_layout_cols,
                       rhs_layout_rows, rhs_layout_cols, dst_layout_rows,
                       dst_layout_cols);
    Unpack<4, 4>(output_ptr, dst, batch_size, output_depth, scaling_factors,
                 filter_scales, dst_layout_rows, dst_layout_cols);
    return;
  }
  if (rhs_width >= 2) {
    RunKernel<4, 2, 32>(lhs, rhs, dst, lhs_layout_rows, lhs_layout_cols,
                       rhs_layout_rows, rhs_layout_cols, dst_layout_rows,
                       dst_layout_cols);
    Unpack<4, 2>(output_ptr, dst, batch_size, output_depth, scaling_factors,
                 filter_scales, dst_layout_rows, dst_layout_cols);
    return;
  }
  RunKernel<4, 1, 32>(lhs, rhs, dst, lhs_layout_rows, lhs_layout_cols,
                      rhs_layout_rows, rhs_layout_cols, dst_layout_rows,
                      dst_layout_cols);
  Unpack<4, 1>(output_ptr, dst, batch_size, output_depth, scaling_factors,
               filter_scales, dst_layout_rows, dst_layout_cols);
}

}  // namespace optimized_4bit
}  // namespace tflite

#endif  // defined(FC_4BIT_RVV) && defined(__riscv_vector)
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_4BIT_RVV_FULLY_CONNECTED_H_
