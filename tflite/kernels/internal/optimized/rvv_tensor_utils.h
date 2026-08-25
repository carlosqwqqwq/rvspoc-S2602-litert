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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_TENSOR_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_TENSOR_UTILS_H_

#include <cstddef>
#include <stdint.h>

namespace tflite {
class CpuBackendContext;
}

#if defined(__riscv_vector)

namespace tflite {
namespace tensor_utils {

// Returns the hardware vector length in bits (VLEN) by reading the `vlenb`
// CSR. Used for runtime dispatch / vector-length adaptive tuning: callers can
// choose kernel policies (e.g. register group / unroll factors) based on
// whether the hardware runs VLEN=128, 256 or 512 bits.
inline int RvvGetVlenBits() {
  size_t vlenb = 0;
  __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
  return static_cast<int>(vlenb * 8);
}

// Convenience classifier: returns 128, 256 or 512 rounded down to the nearest
// supported band (any other VLEN maps to the closest lower supported band,
// with a floor of 128). The exact value is available from RvvGetVlenBits().
inline int RvvVlenClassBits() {
  const int vlen_bits = RvvGetVlenBits();
  if (vlen_bits >= 512) return 512;
  if (vlen_bits >= 256) return 256;
  return 128;
}

void RvvMatrixBatchVectorMultiplyAccumulate(const float *matrix, int m_rows,
                                            int m_cols, const float *vector,
                                            int n_batch, float *result);

void RvvMatrixBatchVectorMultiplyAccumulate(const int8_t *matrix, int m_rows,
                                            int m_cols, const int8_t *vectors,
                                            const float *scaling_factors,
                                            int n_batch, float *result);

void RvvMatrixBatchVectorMultiplyAccumulate(const int8_t *matrix, int m_rows,
                                            int m_cols, const int8_t *vectors,
                                            const float *scaling_factors,
                                            int n_batch, int32_t *scratch,
                                            float *result,
                                            CpuBackendContext *context);

void RvvMatrixBatchVectorMultiplyAccumulate(
    const int8_t *matrix, int m_rows, int m_cols, const int8_t *vectors,
    const float *scaling_factors, int n_batch, float *result,
    const float *per_channel_scale, const int32_t *input_offset,
    int32_t *scratch, int32_t *row_sums, bool *compute_row_sums,
    CpuBackendContext *context);

void RvvMatrixScalarMultiplyAccumulate(const int8_t *matrix, int32_t scalar,
                                       int32_t n_row, int32_t n_col,
                                       int32_t *output);

void RvvSparseMatrixBatchVectorMultiplyAccumulate1x4(
    const float *matrix, const int32_t *segments, const int32_t *indices,
    int m_rows, int m_cols, const float *vector, int n_batch, float *result);

void RvvSparseMatrixBatchVectorMultiplyAccumulate(const float *matrix,
                                                  const uint8_t *ledger,
                                                  int m_rows, int m_cols,
                                                  const float *vector,
                                                  int n_batch, float *result);

void RvvSparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t *matrix, const uint8_t *ledger, int m_rows, int m_cols,
    const int8_t *vectors, const float *scaling_factors, int n_batch,
    float *result, const float *per_channel_scale);

void RvvSparseMatrixBatchVectorMultiplyAccumulate1x16(
    const int8_t *matrix, const int32_t *segments, const int32_t *indices,
    int m_rows, int m_cols, const int8_t *vector, const int32_t *bias_vector,
    int n_batch, int32_t input_offset, int32_t output_multiplier,
    int32_t output_shift, const int32_t *per_channel_scale,
    const int32_t *per_channel_shift, int32_t output_offset,
    int32_t output_activation_min, int32_t output_activation_max,
    int8_t *result);

void RvvMatrixBatchVectorMultiplyAccumulate(
    const int8_t *input, const int32_t *bias,
    const int8_t *input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t *scratch, int16_t *output, CpuBackendContext *context);

void RvvMatrixBatchVectorMultiplyAccumulate(
    const int8_t *input, const int32_t *bias,
    const int8_t *input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t *scratch, int8_t *output, CpuBackendContext *context);

float RvvVectorVectorDotProduct(const float *vector1, const float *vector2,
                                int v_size);

void RvvCwiseAdd(const int16_t *input_1, const int16_t *input_2, int n_batch,
                 int n_input, int16_t *output);

void RvvCwiseMul(const int16_t *input_1, const int16_t *input_2, int n_batch,
                 int n_input, int shift, int16_t *output);

void RvvCwiseMul(const int16_t *input_1, const int16_t *input_2,
                 int32_t multiplier, int32_t shift, int32_t n_batch,
                 int32_t n_input, int32_t output_zp, int8_t *output);

void RvvCwiseClipping(float *vector, int v_size, float clipping_value);

void RvvCwiseClipping(int16_t *vector, int v_size, int16_t clipping_value);

void RvvCwiseClipping(int8_t *vector, int v_size, int8_t clipping_value);

void RvvVectorBatchVectorCwiseProductAccumulate(const int16_t *vector,
                                                int v_size,
                                                const int16_t *batch_vector,
                                                int n_batch, int32_t multiplier,
                                                int shift, int16_t *result);

void RvvSub1Vector(const float *vector, int v_size, float *result);

void RvvSub1Vector(const int16_t *vector, int v_size, int16_t *result);

void RvvReductionSumVector(const float *input_vector, float *output_vector,
                           int output_size, int reduction_size);

void RvvReductionSumVector(const int8_t *input_vector, int32_t *output_vector,
                           int output_size, int reduction_size);

void RvvMeanStddevNormalization(const float *input_vector, float *output_vector,
                                int v_size, int n_batch);

void RvvVectorScalarMultiply(const int8_t *vector, int v_size, float scale,
                             float *result);

void RvvSymmetricQuantizeFloats(const float *values, int size,
                                int8_t *quantized_values, float *min_value,
                                float *max_value, float *scaling_factor);

void RvvSymmetricQuantizeFloats(const float *values, int size,
                                int8_t *quantized_values, float min_value,
                                float max_value, float *scaling_factor);

void RvvAsymmetricQuantizeFloats(const float *values, int size,
                                 int8_t *quantized_values,
                                 float *scaling_factor, int32_t *offset);

bool RvvIsZeroVector(const float *vector, int v_size);

bool RvvIsZeroVector(const int8_t *vector, int v_size);

void RvvApplyLayerNorm(const int16_t *input, const int16_t *layer_norm_weights,
                       const int32_t *bias, int32_t layer_norm_scale_a,
                       int32_t layer_norm_scale_b, int32_t variance_limit,
                       int n_batch, int n_input, int16_t *output);

void RvvApplySigmoid(const int16_t *input, int32_t n_batch, int32_t n_input,
                     int16_t *output);

void RvvApplyTanh(int32_t integer_bits, const int16_t *input, int32_t n_batch,
                  int32_t n_input, int16_t *output);

void RvvApplyTanhWithInputLeftShift(int32_t integer_bits,
                                    int32_t input_left_shift,
                                    const int16_t *input, int32_t n_batch,
                                    int32_t n_input, int16_t *output);

void RvvTanh16BitPrecisionUint8(const uint8_t *input, int size,
                                int32_t input_zero_point,
                                int32_t input_range_radius,
                                int16_t input_multiplier,
                                int16_t input_left_shift, uint8_t *output);
void RvvTanh16BitPrecisionInt8(const int8_t *input, int size,
                               int32_t input_zero_point,
                               int32_t input_range_radius,
                               int16_t input_multiplier,
                               int16_t input_left_shift, int8_t *output);
void RvvLogistic16BitPrecisionUint8(const uint8_t *input, int size,
                                    int32_t input_zero_point,
                                    int32_t input_range_radius,
                                    int32_t input_multiplier,
                                    int16_t input_left_shift, uint8_t *output);
void RvvLogistic16BitPrecisionInt8(const int8_t *input, int size,
                                   int32_t input_zero_point,
                                   int32_t input_range_radius,
                                   int32_t input_multiplier,
                                   int16_t input_left_shift, int8_t *output);

void RvvLstmCellQuantized(const int16_t *activ_temp, const int16_t *prev_state,
                          int outer_size, int output_depth, int state_integer_bits,
                          int16_t *output_state, uint8_t *output_activation);

} // namespace tensor_utils
} // namespace tflite

#endif // defined(__riscv_vector)

#endif // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RVV_TENSOR_UTILS_H_
