// SPDX-License-Identifier: Apache-2.0

#include <array>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/optimized/rvv_optimized_ops.h"

namespace {

struct Shape {
  int batches;
  int input_height;
  int input_width;
  int input_depth;
  int filter_height;
  int filter_width;
  int output_height;
  int output_width;
  int depth_multiplier;
  int stride_height;
  int stride_width;
  int pad_height;
  int pad_width;
  int dilation_height;
  int dilation_width;

  int output_depth() const { return input_depth * depth_multiplier; }
  size_t input_size() const {
    return static_cast<size_t>(batches) * input_height * input_width *
           input_depth;
  }
  size_t output_size() const {
    return static_cast<size_t>(batches) * output_height * output_width *
           output_depth();
  }
  size_t filter_size() const {
    return static_cast<size_t>(filter_height) * filter_width * output_depth();
  }
};

int InputIndex(const Shape& s, int batch, int y, int x, int channel) {
  return (((batch * s.input_height + y) * s.input_width + x) *
          s.input_depth) +
         channel;
}

int OutputIndex(const Shape& s, int batch, int y, int x, int channel) {
  return (((batch * s.output_height + y) * s.output_width + x) *
          s.output_depth()) +
         channel;
}

int FilterIndex(const Shape& s, int y, int x, int channel) {
  return ((y * s.filter_width + x) * s.output_depth()) + channel;
}

void ReferenceFloat(const Shape& s, const std::vector<float>& input,
                    const std::vector<float>& filter,
                    const std::vector<float>& bias, float activation_min,
                    float activation_max, std::vector<float>* output) {
  output->assign(s.output_size(), 0.0f);
  for (int batch = 0; batch < s.batches; ++batch) {
    for (int out_y = 0; out_y < s.output_height; ++out_y) {
      const int in_y_origin = out_y * s.stride_height - s.pad_height;
      for (int out_x = 0; out_x < s.output_width; ++out_x) {
        const int in_x_origin = out_x * s.stride_width - s.pad_width;
        for (int output_channel = 0; output_channel < s.output_depth();
             ++output_channel) {
          const int input_channel = output_channel / s.depth_multiplier;
          float value = bias[output_channel];
          for (int fy = 0; fy < s.filter_height; ++fy) {
            const int in_y = in_y_origin + fy * s.dilation_height;
            if (in_y < 0 || in_y >= s.input_height) continue;
            for (int fx = 0; fx < s.filter_width; ++fx) {
              const int in_x = in_x_origin + fx * s.dilation_width;
              if (in_x < 0 || in_x >= s.input_width) continue;
              value += input[InputIndex(s, batch, in_y, in_x, input_channel)] *
                       filter[FilterIndex(s, fy, fx, output_channel)];
            }
          }
          (*output)[OutputIndex(s, batch, out_y, out_x, output_channel)] =
              std::max(activation_min, std::min(activation_max, value));
        }
      }
    }
  }
}

template <typename T>
T Clamp(int32_t value, int32_t min_value, int32_t max_value) {
  return static_cast<T>(std::max(min_value, std::min(max_value, value)));
}

void ReferenceUint8(const Shape& s, const std::vector<uint8_t>& input,
                    const std::vector<uint8_t>& filter,
                    const std::vector<int32_t>& bias, int32_t input_offset,
                    int32_t filter_offset, int32_t output_offset,
                    int32_t output_multiplier, int output_shift,
                    int32_t activation_min, int32_t activation_max,
                    std::vector<uint8_t>* output) {
  output->assign(s.output_size(), 0);
  for (int batch = 0; batch < s.batches; ++batch) {
    for (int out_y = 0; out_y < s.output_height; ++out_y) {
      const int in_y_origin = out_y * s.stride_height - s.pad_height;
      for (int out_x = 0; out_x < s.output_width; ++out_x) {
        const int in_x_origin = out_x * s.stride_width - s.pad_width;
        for (int output_channel = 0; output_channel < s.output_depth();
             ++output_channel) {
          const int input_channel = output_channel / s.depth_multiplier;
          int32_t value = bias[output_channel];
          for (int fy = 0; fy < s.filter_height; ++fy) {
            const int in_y = in_y_origin + fy * s.dilation_height;
            if (in_y < 0 || in_y >= s.input_height) continue;
            for (int fx = 0; fx < s.filter_width; ++fx) {
              const int in_x = in_x_origin + fx * s.dilation_width;
              if (in_x < 0 || in_x >= s.input_width) continue;
              value +=
                  (static_cast<int32_t>(
                       input[InputIndex(s, batch, in_y, in_x, input_channel)]) +
                   input_offset) *
                  (static_cast<int32_t>(
                       filter[FilterIndex(s, fy, fx, output_channel)]) +
                   filter_offset);
            }
          }
          value = tflite::MultiplyByQuantizedMultiplier(
                      value, output_multiplier, output_shift) +
                  output_offset;
          (*output)[OutputIndex(s, batch, out_y, out_x, output_channel)] =
              Clamp<uint8_t>(value, activation_min, activation_max);
        }
      }
    }
  }
}

void ReferenceInt8(const Shape& s, const std::vector<int8_t>& input,
                   const std::vector<int8_t>& filter,
                   const std::vector<int32_t>& bias, int32_t input_offset,
                   int32_t output_offset, const std::vector<int32_t>&
                       multipliers, const std::vector<int32_t>& shifts,
                   int32_t activation_min, int32_t activation_max,
                   std::vector<int8_t>* output) {
  output->assign(s.output_size(), 0);
  for (int batch = 0; batch < s.batches; ++batch) {
    for (int out_y = 0; out_y < s.output_height; ++out_y) {
      const int in_y_origin = out_y * s.stride_height - s.pad_height;
      for (int out_x = 0; out_x < s.output_width; ++out_x) {
        const int in_x_origin = out_x * s.stride_width - s.pad_width;
        for (int output_channel = 0; output_channel < s.output_depth();
             ++output_channel) {
          const int input_channel = output_channel / s.depth_multiplier;
          int32_t value = bias[output_channel];
          for (int fy = 0; fy < s.filter_height; ++fy) {
            const int in_y = in_y_origin + fy * s.dilation_height;
            if (in_y < 0 || in_y >= s.input_height) continue;
            for (int fx = 0; fx < s.filter_width; ++fx) {
              const int in_x = in_x_origin + fx * s.dilation_width;
              if (in_x < 0 || in_x >= s.input_width) continue;
              value +=
                  (static_cast<int32_t>(
                       input[InputIndex(s, batch, in_y, in_x, input_channel)]) +
                   input_offset) *
                  static_cast<int32_t>(
                      filter[FilterIndex(s, fy, fx, output_channel)]);
            }
          }
          value = tflite::MultiplyByQuantizedMultiplier(
                      value, multipliers[output_channel],
                      shifts[output_channel]) +
                  output_offset;
          (*output)[OutputIndex(s, batch, out_y, out_x, output_channel)] =
              Clamp<int8_t>(value, activation_min, activation_max);
        }
      }
    }
  }
}

bool CheckFloat(const Shape& s, const std::vector<float>& input,
                const std::vector<float>& filter,
                const std::vector<float>& bias) {
  std::vector<float> expected;
  std::vector<float> actual(s.output_size(), -123.0f);
  ReferenceFloat(s, input, filter, bias, -0.65f, 0.73f, &expected);
  tflite::rvv_optimized_ops::DepthwiseConvFloat(
      input.data(), filter.data(), bias.data(), actual.data(), s.batches,
      s.input_height, s.input_width, s.input_depth, s.filter_height,
      s.filter_width, s.output_height, s.output_width, s.output_depth(),
      s.depth_multiplier, s.stride_height, s.stride_width, s.pad_height,
      s.pad_width, s.dilation_height, s.dilation_width, -0.65f, 0.73f, 0,
      s.batches, 0);
  float max_error = 0.0f;
  for (size_t i = 0; i < expected.size(); ++i) {
    max_error = std::max(max_error, std::abs(expected[i] - actual[i]));
    if (max_error > 1.0e-5f) {
      std::cerr << "FP32 mismatch multiplier=" << s.depth_multiplier
                << " index=" << i << " expected=" << expected[i]
                << " actual=" << actual[i] << "\n";
      return false;
    }
  }

  std::vector<float> split(s.output_size(), -123.0f);
  tflite::rvv_optimized_ops::DepthwiseConvFloat(
      input.data(), filter.data(), bias.data(), split.data(), s.batches,
      s.input_height, s.input_width, s.input_depth, s.filter_height,
      s.filter_width, s.output_height, s.output_width, s.output_depth(),
      s.depth_multiplier, s.stride_height, s.stride_width, s.pad_height,
      s.pad_width, s.dilation_height, s.dilation_width, -0.65f, 0.73f, 1,
      s.output_height, 1);
  for (int batch = 0; batch < s.batches; ++batch) {
    for (int y = 1; y < s.output_height; ++y) {
      for (int x = 0; x < s.output_width; ++x) {
        for (int channel = 0; channel < s.output_depth(); ++channel) {
          const size_t index = OutputIndex(s, batch, y, x, channel);
          if (std::abs(expected[index] - split[index]) > 1.0e-5f) {
            std::cerr << "FP32 split mismatch multiplier="
                      << s.depth_multiplier << " index=" << index << "\n";
            return false;
          }
        }
      }
    }
  }
  std::cout << "fp32 multiplier=" << s.depth_multiplier
            << " max_abs_error=" << max_error << "\n";
  return true;
}

bool CheckUint8(const Shape& s, const std::vector<uint8_t>& input,
                const std::vector<uint8_t>& filter,
                const std::vector<int32_t>& bias) {
  constexpr int32_t kOutputOffset = 9;
  constexpr int32_t kOutputMultiplier = 1073741824;
  constexpr int kOutputShift = 1;
  constexpr int32_t kActivationMin = 7;
  constexpr int32_t kActivationMax = 241;
  const std::array<std::pair<int32_t, int32_t>, 3> offset_cases = {{
      {-13, -121}, {0, 0}, {-255, -255}}};
  for (const auto& offsets : offset_cases) {
    const int32_t input_offset = offsets.first;
    const int32_t filter_offset = offsets.second;
    std::vector<uint8_t> expected;
    std::vector<uint8_t> actual(s.output_size(), 0xA5);
    ReferenceUint8(s, input, filter, bias, input_offset, filter_offset,
                   kOutputOffset, kOutputMultiplier, kOutputShift,
                   kActivationMin, kActivationMax, &expected);
    tflite::rvv_optimized_ops::DepthwiseConvUint8(
        input.data(), filter.data(), bias.data(), actual.data(), s.batches,
        s.input_height, s.input_width, s.input_depth, s.filter_height,
        s.filter_width, s.output_height, s.output_width, s.output_depth(),
        s.depth_multiplier, s.stride_height, s.stride_width, s.pad_height,
        s.pad_width, s.dilation_height, s.dilation_width, input_offset,
        filter_offset, kOutputOffset, kOutputMultiplier, kOutputShift,
        kActivationMin, kActivationMax, 0, s.batches, 0);
    for (size_t i = 0; i < expected.size(); ++i) {
      if (expected[i] != actual[i]) {
        std::cerr << "UINT8 mismatch multiplier=" << s.depth_multiplier
                  << " offsets=" << input_offset << "," << filter_offset
                  << " index=" << i << " expected="
                  << static_cast<int>(expected[i]) << " actual="
                  << static_cast<int>(actual[i]) << "\n";
        return false;
      }
    }

    std::vector<uint8_t> split(s.output_size(), 0xA5);
    tflite::rvv_optimized_ops::DepthwiseConvUint8(
        input.data(), filter.data(), bias.data(), split.data(), s.batches,
        s.input_height, s.input_width, s.input_depth, s.filter_height,
        s.filter_width, s.output_height, s.output_width, s.output_depth(),
        s.depth_multiplier, s.stride_height, s.stride_width, s.pad_height,
        s.pad_width, s.dilation_height, s.dilation_width, input_offset,
        filter_offset, kOutputOffset, kOutputMultiplier, kOutputShift,
        kActivationMin, kActivationMax, 1, s.output_height, 1);
    for (int batch = 0; batch < s.batches; ++batch) {
      for (int y = 1; y < s.output_height; ++y) {
        for (int x = 0; x < s.output_width; ++x) {
          for (int channel = 0; channel < s.output_depth(); ++channel) {
            const size_t index = OutputIndex(s, batch, y, x, channel);
            if (expected[index] != split[index]) {
              std::cerr << "UINT8 split mismatch multiplier="
                        << s.depth_multiplier << " offsets=" << input_offset
                        << "," << filter_offset << " index=" << index
                        << "\n";
              return false;
            }
          }
        }
      }
    }
    std::cout << "uint8 multiplier=" << s.depth_multiplier << " offsets="
              << input_offset << "," << filter_offset << " exact=1\n";
  }
  return true;
}

bool CheckInt8(const Shape& s, const std::vector<int8_t>& input,
               const std::vector<int8_t>& filter,
               const std::vector<int32_t>& bias) {
  constexpr int32_t kOutputOffset = 5;
  constexpr int32_t kActivationMin = -99;
  constexpr int32_t kActivationMax = 101;
  const std::vector<int32_t> multipliers(s.output_depth(), 1073741824);
  const std::vector<int32_t> shifts(s.output_depth(), 1);
  const std::array<int32_t, 3> input_offsets = {-128, -3, 0};
  for (const int32_t input_offset : input_offsets) {
    std::vector<int8_t> expected;
    std::vector<int8_t> actual(s.output_size(), 0x5A);
    ReferenceInt8(s, input, filter, bias, input_offset, kOutputOffset,
                  multipliers, shifts, kActivationMin, kActivationMax,
                  &expected);
    tflite::rvv_optimized_ops::DepthwiseConvInt8PerChannel(
        input.data(), filter.data(), bias.data(), actual.data(), s.batches,
        s.input_height, s.input_width, s.input_depth, s.filter_height,
        s.filter_width, s.output_height, s.output_width, s.output_depth(),
        s.depth_multiplier, s.stride_height, s.stride_width, s.pad_height,
        s.pad_width, s.dilation_height, s.dilation_width, input_offset,
        kOutputOffset, multipliers.data(), shifts.data(), kActivationMin,
        kActivationMax, 0, s.batches, 0);
    for (size_t i = 0; i < expected.size(); ++i) {
      if (expected[i] != actual[i]) {
        std::cerr << "INT8 mismatch multiplier=" << s.depth_multiplier
                  << " input_offset=" << input_offset << " index=" << i
                  << " expected=" << static_cast<int>(expected[i])
                  << " actual=" << static_cast<int>(actual[i]) << "\n";
        return false;
      }
    }

    std::vector<int8_t> split(s.output_size(), 0x5A);
    tflite::rvv_optimized_ops::DepthwiseConvInt8PerChannel(
        input.data(), filter.data(), bias.data(), split.data(), s.batches,
        s.input_height, s.input_width, s.input_depth, s.filter_height,
        s.filter_width, s.output_height, s.output_width, s.output_depth(),
        s.depth_multiplier, s.stride_height, s.stride_width, s.pad_height,
        s.pad_width, s.dilation_height, s.dilation_width, input_offset,
        kOutputOffset, multipliers.data(), shifts.data(), kActivationMin,
        kActivationMax, 1, s.output_height, 1);
    for (int batch = 0; batch < s.batches; ++batch) {
      for (int y = 1; y < s.output_height; ++y) {
        for (int x = 0; x < s.output_width; ++x) {
          for (int channel = 0; channel < s.output_depth(); ++channel) {
            const size_t index = OutputIndex(s, batch, y, x, channel);
            if (expected[index] != split[index]) {
              std::cerr << "INT8 split mismatch multiplier="
                        << s.depth_multiplier << " input_offset="
                        << input_offset << " index=" << index << "\n";
              return false;
            }
          }
        }
      }
    }
    std::cout << "int8 multiplier=" << s.depth_multiplier << " input_offset="
              << input_offset << " exact=1\n";
  }
  return true;
}

int RunCase(const Shape& s, std::mt19937* rng) {
  std::uniform_real_distribution<float> float_distribution(-1.0f, 1.0f);
  std::uniform_int_distribution<int> uint8_distribution(0, 255);
  std::uniform_int_distribution<int> int8_distribution(-127, 127);
  std::uniform_int_distribution<int32_t> bias_distribution(-100, 100);

  std::vector<float> float_input(s.input_size());
  std::vector<float> float_filter(s.filter_size());
  std::vector<float> float_bias(s.output_depth());
  for (float& value : float_input) value = float_distribution(*rng);
  for (float& value : float_filter) value = float_distribution(*rng);
  for (float& value : float_bias) value = float_distribution(*rng);

  std::vector<uint8_t> uint8_input(s.input_size());
  std::vector<uint8_t> uint8_filter(s.filter_size());
  std::vector<int32_t> uint8_bias(s.output_depth());
  for (uint8_t& value : uint8_input) value = uint8_distribution(*rng);
  for (uint8_t& value : uint8_filter) value = uint8_distribution(*rng);
  for (int32_t& value : uint8_bias) value = bias_distribution(*rng);

  std::vector<int8_t> int8_input(s.input_size());
  std::vector<int8_t> int8_filter(s.filter_size());
  std::vector<int32_t> int8_bias(s.output_depth());
  for (int8_t& value : int8_input) value = int8_distribution(*rng);
  for (int8_t& value : int8_filter) value = int8_distribution(*rng);
  for (int32_t& value : int8_bias) value = bias_distribution(*rng);

  return CheckFloat(s, float_input, float_filter, float_bias) &&
                 CheckUint8(s, uint8_input, uint8_filter, uint8_bias) &&
                 CheckInt8(s, int8_input, int8_filter, int8_bias)
             ? 0
             : 1;
}

}  // namespace

int main() {
  const std::vector<Shape> cases = {
      {2, 4, 5, 3, 3, 2, 4, 3, 2, 1, 2, 1, 0, 1, 1},
      {1, 5, 4, 5, 2, 3, 3, 5, 3, 2, 1, 0, 1, 1, 2},
      {2, 3, 3, 7, 3, 3, 2, 2, 5, 1, 1, 1, 1, 1, 1},
      {1, 3, 4, 2, 3, 2, 3, 4, 17, 1, 1, 1, 0, 1, 1},
  };
  std::mt19937 rng(2602);
  for (const Shape& shape : cases) {
    if (RunCase(shape, &rng) != 0) return 1;
  }
  std::cout << "DEPTHWISE_MULTIPLIER_DIFFERENTIAL_PASS cases=" << cases.size()
            << "\n";
  return 0;
}
