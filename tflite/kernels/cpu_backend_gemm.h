/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_H_
#define TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tflite/kernels/cpu_backend_context.h"
#include "tflite/kernels/cpu_backend_threadpool.h"
#include "tflite/kernels/cpu_backend_gemm_custom_gemv.h"
#include "tflite/kernels/cpu_backend_gemm_params.h"
#include "tflite/kernels/cpu_backend_gemm_ruy.h"
#if defined(__riscv_vector)
#include "tflite/kernels/internal/optimized/rvv_optimized_ops.h"
#endif

#ifndef TFLITE_WITH_RUY
#include "tflite/kernels/cpu_backend_gemm_eigen.h"
#include "tflite/kernels/cpu_backend_gemm_gemmlowp.h"
#include "tflite/kernels/cpu_backend_gemm_x86.h"
#endif

namespace tflite {

namespace cpu_backend_gemm {

#if defined(__riscv_vector)
namespace detail {

// Only large convolution-shaped products justify thread-pool setup. Small
// products retain the original single-thread path; the 4M MAC threshold admits
// medium MobileNetV2 GEMMs while the output gate rejects shallow tails.
constexpr std::int64_t kRvvGemmThreadingMinOutputElements = 65536;
// A210 A/B shows that medium pointwise GEMMs also benefit from threading.
constexpr std::int64_t kRvvGemmThreadingMinMacs = 4 * 1024 * 1024;
constexpr std::int64_t kRvvQuantizedGemmThreadingMinMacs = 2 * 1024 * 1024;

inline int RvvGemmThreadCount(int lhs_rows, int depth, int cols,
                              int max_num_threads,
                              std::int64_t min_macs = kRvvGemmThreadingMinMacs) {
  const std::int64_t output_elements =
      static_cast<std::int64_t>(lhs_rows) * cols;
  const std::int64_t macs = output_elements * depth;
  if (max_num_threads <= 1 || cols <= 1 ||
      (output_elements < kRvvGemmThreadingMinOutputElements &&
       macs < min_macs)) {
    return 1;
  }
  return std::min(max_num_threads, cols);
}

struct RvvGemmFloatTask : cpu_backend_threadpool::Task {
  RvvGemmFloatTask(const float* lhs, int lhs_rows, int depth, const float* rhs,
                  int cols, float* dst, const float* bias, float clamp_min,
                  float clamp_max, int col_start)
      : lhs_(lhs),
        lhs_rows_(lhs_rows),
        depth_(depth),
        rhs_(rhs),
        cols_(cols),
        dst_(dst),
        bias_(bias),
        clamp_min_(clamp_min),
        clamp_max_(clamp_max),
        col_start_(col_start) {}

  void Run() override {
    rvv_optimized_ops::RvvGemmFloat(
        lhs_, lhs_rows_, depth_,
        rhs_ + static_cast<std::size_t>(col_start_) * depth_, cols_,
        dst_ + static_cast<std::size_t>(col_start_) * lhs_rows_, bias_,
        clamp_min_, clamp_max_);
  }

  const float* lhs_;
  int lhs_rows_;
  int depth_;
  const float* rhs_;
  int cols_;
  float* dst_;
  const float* bias_;
  float clamp_min_;
  float clamp_max_;
  int col_start_;
};

inline void RvvGemmFloatThreaded(
    const float* lhs, int lhs_rows, int depth, const float* rhs, int cols,
    float* dst, const float* bias, float clamp_min, float clamp_max,
    CpuBackendContext* context) {
  const int thread_count = RvvGemmThreadCount(
      lhs_rows, depth, cols,
      context == nullptr ? 1 : context->max_num_threads());
  if (thread_count == 1) {
    rvv_optimized_ops::RvvGemmFloat(lhs, lhs_rows, depth, rhs, cols, dst, bias,
                                    clamp_min, clamp_max);
    return;
  }

  std::vector<RvvGemmFloatTask> tasks;
  tasks.reserve(thread_count);
  for (int task = 0; task < thread_count; ++task) {
    const int col_start = task * cols / thread_count;
    const int col_end = (task + 1) * cols / thread_count;
    tasks.emplace_back(lhs, lhs_rows, depth, rhs, col_end - col_start, dst,
                       bias, clamp_min, clamp_max, col_start);
  }
  cpu_backend_threadpool::Execute(tasks.size(), tasks.data(), context);
}

template <typename LhsScalar, typename RhsScalar, typename DstScalar>
struct RvvGemmQuantizedTask : cpu_backend_threadpool::Task {
  RvvGemmQuantizedTask(
      const LhsScalar* lhs, int lhs_rows, int depth, int lhs_zero_point,
      const RhsScalar* rhs, int cols, int rhs_zero_point, DstScalar* dst,
      const int32_t* bias, int32_t multiplier, int multiplier_shift,
      const int32_t* multiplier_perchannel, const int* shift_perchannel,
      int32_t output_zero_point, DstScalar clamp_min, DstScalar clamp_max,
      int col_start)
      : lhs_(lhs),
        lhs_rows_(lhs_rows),
        depth_(depth),
        lhs_zero_point_(lhs_zero_point),
        rhs_(rhs),
        cols_(cols),
        rhs_zero_point_(rhs_zero_point),
        dst_(dst),
        bias_(bias),
        multiplier_(multiplier),
        multiplier_shift_(multiplier_shift),
        multiplier_perchannel_(multiplier_perchannel),
        shift_perchannel_(shift_perchannel),
        output_zero_point_(output_zero_point),
        clamp_min_(clamp_min),
        clamp_max_(clamp_max),
        col_start_(col_start) {}

  void Run() override {
    rvv_optimized_ops::RvvGemmQuantized<LhsScalar, RhsScalar, DstScalar>(
        lhs_, lhs_rows_, depth_, lhs_zero_point_,
        rhs_ + static_cast<std::size_t>(col_start_) * depth_, cols_,
        rhs_zero_point_, dst_ + static_cast<std::size_t>(col_start_) * lhs_rows_,
        bias_, multiplier_, multiplier_shift_, multiplier_perchannel_,
        shift_perchannel_, output_zero_point_, clamp_min_, clamp_max_);
  }

  const LhsScalar* lhs_;
  int lhs_rows_;
  int depth_;
  int lhs_zero_point_;
  const RhsScalar* rhs_;
  int cols_;
  int rhs_zero_point_;
  DstScalar* dst_;
  const int32_t* bias_;
  int32_t multiplier_;
  int multiplier_shift_;
  const int32_t* multiplier_perchannel_;
  const int* shift_perchannel_;
  int32_t output_zero_point_;
  DstScalar clamp_min_;
  DstScalar clamp_max_;
  int col_start_;
};

template <typename LhsScalar, typename RhsScalar, typename DstScalar>
inline void RvvGemmQuantizedThreaded(
    const LhsScalar* lhs, int lhs_rows, int depth, int lhs_zero_point,
    const RhsScalar* rhs, int cols, int rhs_zero_point, DstScalar* dst,
    const int32_t* bias, int32_t multiplier, int multiplier_shift,
    const int32_t* multiplier_perchannel, const int* shift_perchannel,
    int32_t output_zero_point, DstScalar clamp_min, DstScalar clamp_max,
    CpuBackendContext* context) {
  const int thread_count = RvvGemmThreadCount(
      lhs_rows, depth, cols,
      context == nullptr ? 1 : context->max_num_threads(),
      kRvvQuantizedGemmThreadingMinMacs);
  if (thread_count == 1) {
    rvv_optimized_ops::RvvGemmQuantized<LhsScalar, RhsScalar, DstScalar>(
        lhs, lhs_rows, depth, lhs_zero_point, rhs, cols, rhs_zero_point, dst,
        bias, multiplier, multiplier_shift, multiplier_perchannel,
        shift_perchannel, output_zero_point, clamp_min, clamp_max);
    return;
  }

  std::vector<RvvGemmQuantizedTask<LhsScalar, RhsScalar, DstScalar>> tasks;
  tasks.reserve(thread_count);
  for (int task = 0; task < thread_count; ++task) {
    const int col_start = task * cols / thread_count;
    const int col_end = (task + 1) * cols / thread_count;
    tasks.emplace_back(
        lhs, lhs_rows, depth, lhs_zero_point, rhs, col_end - col_start,
        rhs_zero_point, dst, bias, multiplier, multiplier_shift,
        multiplier_perchannel, shift_perchannel, output_zero_point, clamp_min,
        clamp_max, col_start);
  }
  cpu_backend_threadpool::Execute(tasks.size(), tasks.data(), context);
}

}  // namespace detail
#endif

// The main entry point for CpuBackendGemm::Gemm.
//
// If TFLITE_WITH_RUY is set, CpuBackendGemm::Gemm will always go to Ruy aka
// GemmImplUsingRuy. The behavior is as follows:
//
//                    |Quantized (uint8)|Quantized (int8)| Float |
// TFLITE_WITH_RUY    |      Ruy        |      Ruy       | Ruy   |
// !TFLITE_WITH_RUY   |      gemmlowp   |  Ruy/gemmlowp* | eigen |
// * - Ruy if NEON is not available.
//
//  On most ARM32/ARM64 platforms, the default is TFLITE_WITH_RUY:
//  (default)         |      Ruy        |     Ruy        | Ruy   |
//
//  On other platforms (including x86), the default is !TFLITE_WITH_RUY:
//  (default)         |      gemmlowp   |     Ruy        | eigen |
//
// Use --define=tflite_with_ruy=true or --define=tflite_with_ruy=false to
// override the default.

#if !defined(TFLITE_WITH_RUY) && defined(TFLITE_X86_PLATFORM)
/* GEMM dispatch implementation for x86.
 */
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct GemmImpl : detail::GemmImplX86<LhsScalar, RhsScalar, AccumScalar,
                                      DstScalar, quantization_flavor> {};
#else
/* Generic implementation using ruy.
 * Non-ruy implementation will be partial specializations of this template.
 */
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct GemmImpl : detail::GemmImplUsingRuy<LhsScalar, RhsScalar, AccumScalar,
                                           DstScalar, quantization_flavor> {};

#if !defined(TFLITE_WITH_RUY)

/* Specializations using gemmlowp */
template <typename SrcScalar, typename DstScalar,
          QuantizationFlavor quantization_flavor>
struct GemmImpl<SrcScalar, SrcScalar, std::int32_t, DstScalar,
                quantization_flavor>
    : detail::GemmImplUsingGemmlowp<SrcScalar, SrcScalar, std::int32_t,
                                    DstScalar, quantization_flavor> {};

// When SrcScalar=int8 or DstScalar=int8, gemmlowp fails to compile
// outside of NEON. We avoid the compilation failure by subspecializing these
// cases, rerouting it back to ruy.
#if !defined(GEMMLOWP_NEON)
template <typename SrcScalar, QuantizationFlavor quantization_flavor>
struct GemmImpl<SrcScalar, SrcScalar, std::int32_t, std::int8_t,
                quantization_flavor>
    : detail::GemmImplUsingRuy<SrcScalar, SrcScalar, std::int32_t, std::int8_t,
                               quantization_flavor> {};

template <typename DstScalar, QuantizationFlavor quantization_flavor>
struct GemmImpl<std::int8_t, std::int8_t, std::int32_t, DstScalar,
                quantization_flavor>
    : detail::GemmImplUsingRuy<std::int8_t, std::int8_t, std::int32_t,
                               DstScalar, quantization_flavor> {};

template <QuantizationFlavor quantization_flavor>
struct GemmImpl<std::int8_t, std::int8_t, std::int32_t, std::int8_t,
                quantization_flavor>
    : detail::GemmImplUsingRuy<std::int8_t, std::int8_t, std::int32_t,
                               std::int8_t, quantization_flavor> {};
#endif  // not GEMMLOWP_NEON

/* Specializations using Eigen */

template <>
struct GemmImpl<float, float, float, float, QuantizationFlavor::kFloatingPoint>
    : detail::GemmImplUsingEigen {};

#endif  // not TFLITE_WITH_RUY

#endif  // not TFLITE_WITH_RUY and TFLITE_X86_PLATFORM

/* Public entry point */

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
void Gemm(const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
          const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
          const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
          const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params,
          CpuBackendContext* context) {
  ruy::profiler::ScopeLabel label("cpu_backend_gemm::Gemm");
  ValidateParams(lhs_params, rhs_params, dst_params, params);
  if (!IsValidGemm(lhs_params, rhs_params, dst_params)) {
    // For now, assert in debug mode, return in opt.
    // TODO(b/183099395) Eliminate debug/release discrepancy by plumbing in
    // TFLiteStatus so we can return an error here.
    TFLITE_DCHECK(false);
    return;
  }
  // In some cases we want to unconditionally use ruy as the backend, overriding
  // the `tflite_with_ruy` setting and the platform default.
  bool must_use_ruy = false;
  if (context->use_caching()) {
    // Only ruy supports caching of pre-packed matrices. Due to the large
    // performance impact in the cases where it's typically used, this overrides
    // the default.
    must_use_ruy = true;
  }
  if (lhs_params.order != Order::kRowMajor ||
      rhs_params.order != Order::kColMajor ||
      dst_params.order != Order::kColMajor) {
    // ruy supports all 2^3=8 combinations of storage orders with comparable
    // performance. In ruy, it's only a runtime switch. In other backends
    // (gemmlowp, Eigen), storage orders are template parameters, supporting
    // all 8 combinations would be up to a 8-fold code size increase, so we
    // prefer to force usage of ruy in these cases.
    must_use_ruy = true;
  }
  if (must_use_ruy) {
    detail::GemmImplUsingRuy<LhsScalar, RhsScalar, AccumScalar, DstScalar,
                             quantization_flavor>::Run(lhs_params, lhs_data,
                                                       rhs_params, rhs_data,
                                                       dst_params, dst_data,
                                                       params, context);
    return;
  }
#if defined(__riscv_vector)
  if constexpr (std::is_same_v<LhsScalar, float> &&
                std::is_same_v<RhsScalar, float> &&
                std::is_same_v<AccumScalar, float> &&
                std::is_same_v<DstScalar, float>) {
    if (lhs_params.order == Order::kRowMajor &&
        rhs_params.order == Order::kColMajor &&
        dst_params.order == Order::kColMajor && !context->use_caching()) {
      detail::RvvGemmFloatThreaded(
          lhs_data, lhs_params.rows, lhs_params.cols, rhs_data, rhs_params.cols,
          dst_data, params.bias, params.clamp_min, params.clamp_max, context);
      return;
    }
  } else if constexpr (std::is_same_v<AccumScalar, std::int32_t> &&
                       (std::is_same_v<LhsScalar, std::int8_t> ||
                        std::is_same_v<LhsScalar, std::uint8_t>) &&
                       (std::is_same_v<RhsScalar, std::int8_t> ||
                        std::is_same_v<RhsScalar, std::uint8_t>) &&
                       (std::is_same_v<DstScalar, std::int8_t> ||
                        std::is_same_v<DstScalar, std::uint8_t>) &&
                       (quantization_flavor ==
                            QuantizationFlavor::kIntegerWithUniformMultiplier ||
                        quantization_flavor ==
                            QuantizationFlavor::kIntegerWithPerRowMultiplier)) {
    // 3M covers every layer of the six benchmark models; larger products keep
    // the original backend path until they have an independent benchmark.
    constexpr std::int64_t kRvvQuantizedGemmMaxOutputElements = 3000000;
    if (lhs_params.order == Order::kRowMajor &&
        rhs_params.order == Order::kColMajor &&
        dst_params.order == Order::kColMajor && !context->use_caching() &&
        static_cast<std::int64_t>(lhs_params.rows) * rhs_params.cols <=
            kRvvQuantizedGemmMaxOutputElements) {
      detail::RvvGemmQuantizedThreaded(
          lhs_data, lhs_params.rows, lhs_params.cols,
          static_cast<int>(lhs_params.zero_point), rhs_data, rhs_params.cols,
          static_cast<int>(rhs_params.zero_point), dst_data, params.bias,
          params.multiplier_fixedpoint, params.multiplier_exponent,
          params.multiplier_fixedpoint_perchannel,
          params.multiplier_exponent_perchannel,
          static_cast<int32_t>(dst_params.zero_point), params.clamp_min,
          params.clamp_max, context);
      return;
    }
  }
#endif
  // If we did not choose to force usage of ruy above, then we may now consider
  // using custom GEMV code for the matrix*vector cases.
  const bool try_custom_gemv = (dst_params.cols == 1);
  if (try_custom_gemv) {
    // GEMV case: try a custom fast GEMV path. It will return true if it
    // actually handled it.
    if (detail::CustomGemv(lhs_params, lhs_data, rhs_params, rhs_data,
                           dst_params, dst_data, params, context)) {
      return;
    }
  }
  // Generic case: dispatch to any backend as a general GEMM.
  GemmImpl<LhsScalar, RhsScalar, AccumScalar, DstScalar,
           quantization_flavor>::Run(lhs_params, lhs_data, rhs_params, rhs_data,
                                     dst_params, dst_data, params, context);
}

// Special path for 16x8 quant gemm.
template <QuantizationFlavor quantization_flavor>
void Gemm(const MatrixParams<int8_t>& lhs_params, const int8_t* lhs_data,
          const MatrixParams<int16_t>& rhs_params, const int16_t* rhs_data,
          const MatrixParams<int16_t>& dst_params, int16_t* dst_data,
          const GemmParams<int32_t, int16_t, quantization_flavor>& params,
          CpuBackendContext* context) {
  ruy::profiler::ScopeLabel label("cpu_backend_gemm::Gemm");
  ValidateParams(lhs_params, rhs_params, dst_params, params);
  if (!IsValidGemm(lhs_params, rhs_params, dst_params)) {
    TFLITE_DCHECK(false);
    return;
  }

  // Currently, only Ruy backend supports 16x8 quant gemm so we use ruy
  // only.
  detail::GemmImplUsingRuy<int8_t, int16_t, int32_t, int16_t,
                           quantization_flavor>::Run(lhs_params, lhs_data,
                                                     rhs_params, rhs_data,
                                                     dst_params, dst_data,
                                                     params, context);
}

// Special path for gemm with raw accumulator case. i.e. AccumScalar ==
// DstScalar == int32 case.
template <typename LhsScalar, typename RhsScalar,
          QuantizationFlavor quantization_flavor>
void Gemm(const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
          const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
          const MatrixParams<int32_t>& dst_params, int32_t* dst_data,
          const GemmParams<int32_t, int32_t, quantization_flavor>& params,
          CpuBackendContext* context) {
  ruy::profiler::ScopeLabel label("cpu_backend_gemm::Gemm");
  ValidateParams(lhs_params, rhs_params, dst_params, params);

  // Currently, only Ruy backend supports get raw accumulator, so we use ruy
  // only.
  ruy::profiler::ScopeLabel label2("cpu_backend_gemm::Gemm: general GEMM");
  detail::GemmImplUsingRuy<LhsScalar, RhsScalar, int32_t, int32_t,
                           quantization_flavor>::Run(lhs_params, lhs_data,
                                                     rhs_params, rhs_data,
                                                     dst_params, dst_data,
                                                     params, context);
}

}  // namespace cpu_backend_gemm

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_GEMM_H_
