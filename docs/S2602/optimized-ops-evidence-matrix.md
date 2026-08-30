<!-- SPDX-License-Identifier: Apache-2.0 -->

# S2602 61 个 ARM 优化入口逐项证据矩阵

绑定源码：LiteRT v2.1.4，基线 `ea79caffdd0f52cd44f203674f18a16a3cb861ad`；提交源码为本仓库根目录。

本表把 61 个实际 ARM 保护入口逐项列出。`RVV-DIRECT`/`RVV-SHARED` 是静态路由状态，不等于入口级验收通过；三档 VLEN、`rv64gc` 回退和 scalar differential 需要逐入口补证。已归档的 137 项官方 kernel suite 是独立的通用回归证据，不能替代本表的 61 行动态证据。

| # | ARM 入口 | RVV 路由 | 静态口径 | VLEN128/256/512 | GC fallback | scalar diff | 当前放行 |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | `ShuffledFullyConnectedWorkerImpl` (431) | `rvv_optimized_ops::ShuffledFullyConnectedWorker` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 2 | `ShuffledFullyConnected` (602) | `rvv_optimized_ops::ShuffledFullyConnectedWorker` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 3 | `RoundToNearest` (888) | `rvv_optimized_ops::RvvVectorizedQuantize` | `RVV-SHARED` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 4 | `RoundToNearestUnsigned` (904) | `rvv_optimized_ops::RvvVectorizedQuantize` | `RVV-SHARED` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 5 | `AddElementwise`（FP32）(1476) | `rvv_optimized_ops::AddFloat` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 6 | `AddElementwise`（UINT8）(1539) | `rvv_optimized_ops::QuantizedBinaryUint8<false>` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 7 | `AddScalarBroadcast`（UINT8）(1631) | `rvv_optimized_ops::AddScalarUint8` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 8 | `AddScalarBroadcast`（FP32）(1734) | `rvv_optimized_ops::AddScalarFloat` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 9 | `MulElementwise`（FP32）(1896) | `rvv_optimized_ops::MulFloat` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 10 | `MulElementwise`（INT32）(1953) | `rvv_optimized_ops::MulInt32` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 11 | `MulElementwise`（UINT8）(2114) | `rvv_optimized_ops::MulUint8` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 12 | `MulSimpleBroadcast`（UINT8）(2190) | `rvv_optimized_ops::MulScalarUint8` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 13 | `MulSimpleBroadcast`（FP32）(2261) | `rvv_optimized_ops::MulScalarFloat` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 14 | `LstmCell<StateIntegerBits>` (2689) | `tensor_utils::RvvLstmCellQuantized` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 15 | `AveragePool`（UINT8）(2970) | `rvv_optimized_ops::AveragePoolUint8` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 16 | `MaxPool`（UINT8）(3231) | `rvv_optimized_ops::MaxPoolChannels<uint8_t>` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 17 | `StoreValue`（INT8）(3682) | `rvv_optimized_ops::SoftmaxInt8Lut` | `RVV-SHARED` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 18 | `StoreValue`（UINT8）(3694) | `rvv_optimized_ops::SoftmaxInt8Lut` | `RVV-SHARED` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 19 | `Logistic`（INT16）(3993) | `tensor_utils::RvvApplySigmoid` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 20 | `Tanh`（INT16）(4107) | `tensor_utils::RvvApplyTanhWithInputLeftShift` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 21 | `Quantize` (5127) | `rvv_optimized_ops::QuantizeUniform` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 22 | `Quantize`（per-channel，single rounding）(5192) | `rvv_optimized_ops::QuantizePerChannel` + `RvvVectorizedQuantizePerChannel` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 23 | `Quantize`（per-channel，single rounding）(5282) | `rvv_optimized_ops::QuantizePerChannel` + `RvvVectorizedQuantizePerChannel` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 24 | `Quantize`（per-channel，double rounding）(5372) | `rvv_optimized_ops::QuantizePerChannel` + `RvvVectorizedQuantizePerChannel` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 25 | `Quantize`（per-channel，double rounding）(5463) | `rvv_optimized_ops::QuantizePerChannel` + `RvvVectorizedQuantizePerChannel` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 26 | `Requantize<int8_t, uint8_t>` (5729) | `rvv_optimized_ops::Requantize` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 27 | `Requantize<uint8_t, int8_t>` (5816) | `rvv_optimized_ops::Requantize` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 28 | `Requantize<int8_t, int8_t>` (5894) | `rvv_optimized_ops::Requantize` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 29 | `Requantize<uint8_t, uint8_t>` (5973) | `rvv_optimized_ops::Requantize` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 30 | `Dequantize`（int8）(6370) | `rvv_optimized_ops::Dequantize` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 31 | `Dequantize`（uint8）(6410) | `rvv_optimized_ops::Dequantize` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 32 | `Dequantize`（int16）(6449) | `rvv_optimized_ops::DequantizeInt16` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 33 | `AffineQuantize`（int8）(6502) | `rvv_optimized_ops::AffineQuantize<int8_t>` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 34 | `AffineQuantize`（uint8）(6560) | `rvv_optimized_ops::AffineQuantize<uint8_t>` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 35 | `AffineQuantize`（int16）(6618) | `rvv_optimized_ops::AffineQuantize<int16_t>` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 36 | `HardSwish`（FP32）(6057) | `rvv_optimized_ops::HardSwishFloat` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 37 | `SaturateAndStore`（uint8）(6118) | `rvv_optimized_ops::HardSwishQuantized` 的向量 narrowing helper | `RVV-SHARED` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 38 | `SaturateAndStore`（int8）(6125) | `rvv_optimized_ops::HardSwishQuantized` 的向量 narrowing helper | `RVV-SHARED` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 39 | `HardSwish<T>`（量化）(6134) | `rvv_optimized_ops::HardSwishQuantized` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 40 | `ScaleWithNewZeroPoint` (6356) | `rvv_optimized_ops::Dequantize` 共享输入转换路径 | `RVV-SHARED` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 41 | `SaturatingRounding` (6677) | 16-bit RVV fixed-point helper pipeline | `RVV-SHARED` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 42 | `FixedPoint4Logistic` (6699) | `tensor_utils::RvvLogistic16BitPrecision*` 共享 helper | `RVV-SHARED` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 43 | `FixedPoint4Tanh` (6729) | `tensor_utils::RvvTanh16BitPrecision*` 共享 helper | `RVV-SHARED` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 44 | `CalculateUnsignedClampingWithRangeBitMasks` (6757) | 16-bit RVV narrowing/clamp helper | `RVV-SHARED` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 45 | `CalculateSignedClampingWithRangeBitMasks` (6778) | 16-bit RVV narrowing/clamp helper | `RVV-SHARED` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 46 | `ClampWithRangeAndStore`（uint8）(6799) | 16-bit RVV narrowing/clamp helper | `RVV-SHARED` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 47 | `ClampWithRangeAndStore`（int8）(6806) | 16-bit RVV narrowing/clamp helper | `RVV-SHARED` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 48 | `Tanh16bitPrecision`（uint8）(6818) | `tensor_utils::RvvTanh16BitPrecisionUint8` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 49 | `Tanh16bitPrecision`（int8）(6927) | `tensor_utils::RvvTanh16BitPrecisionInt8` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 50 | `Logistic16bitPrecision`（uint8）(7022) | `tensor_utils::RvvLogistic16BitPrecisionUint8` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 51 | `Logistic16bitPrecision`（int8）(7115) | `tensor_utils::RvvLogistic16BitPrecisionInt8` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 52 | `MaximumElementwise` (7230) | `rvv_optimized_ops::MaximumInt8` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 53 | `MaximumScalarBroadcast` (7251) | `rvv_optimized_ops::MaximumScalarInt8` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 54 | `MinimumElementwise` (7274) | `rvv_optimized_ops::MinimumInt8` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 55 | `MinimumScalarBroadcast` (7295) | `rvv_optimized_ops::MinimumScalarInt8` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 56 | `PReluScalarBroadcast` (7395) | `rvv_optimized_ops::PReluScalarFloat` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 57 | `PReluElementWise` (7444) | `rvv_optimized_ops::PReluElementwiseFloat` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 58 | `ArgMinVector<float>` (7548) | `rvv_optimized_ops::ArgMinFloat` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 59 | `ArgMaxVector<float>` (7604) | `rvv_optimized_ops::ArgMaxFloat` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 60 | `ArgMaxVector<int8_t>` (7660) | `rvv_optimized_ops::ArgMaxInteger<int8_t>` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |
| 61 | `ArgMaxVector<uint8_t>` (7709) | `rvv_optimized_ops::ArgMaxInteger<uint8_t>` | `RVV-DIRECT` | 待入口级 | 待入口级 | 待入口级 | PENDING-ENTRY |

## 统计

- 实际 ARM 分母：61；严格 90% 门槛：55。
- 静态路由：`RVV-DIRECT 47/61`、`RVV-SHARED 14/61`；若组委会接受 shared helper 归并，静态路由可描述为 61/61，但当前仍不宣称严格 90% 动态覆盖。
- 提交内可复核的通用证据：`results/official-kernel-suite/vlen128-summary`、`vlen256-summary`、`vlen512-summary`（三档均 137/137）。
- 放行规则：补齐每行的调用者/算子、输入 shape/dtype、VLEN 128/256/512、GC fallback、scalar/reference 最大误差、运行次数与原始日志后，才能把 `PENDING-ENTRY` 改为 `PASS`。
