<!-- SPDX-License-Identifier: Apache-2.0 -->

# S2602 Depthwise 58 个 ARM 模板特化审计

基线：LiteRT v2.1.4，commit `ea79caffdd0f52cd44f203674f18a16a3cb861ad`。这里把 ARM Neon 源码中的每个 `template <>` 特化逐项列出，不把“通用 RVV 循环”误写成 58 个同名模板的复制实现。

## 逐 spec 索引

状态含义：表中旧标签 `RVV-GENERIC`（等价于 `RVV-GENERIC-M1`）表示该特化的 `depth_multiplier=1` 组合由通用 RVV strip-mining 路径覆盖；`RVV-GENERIC-M>1` 表示当前通用入口对固定输入通道的 multiplier fan-out 做连续向量化；两者均不声称是 ARM 模板的一比一代码搬运。非 RVV 编译仍走原标量回退。

| ID | 基线文件 | Neon 特化 | 行 | RVV 状态 |
| --- | --- | --- | ---: | --- |
| F-01 | `depthwiseconv_float.h` | `FloatDepthwiseConvKernel<false,8,1>` | 35 | RVV-GENERIC |
| F-02 | 同上 | `<false,2,1>` | 96 | RVV-GENERIC |
| F-03 | 同上 | `<true,0,1>` | 179 | RVV-GENERIC |
| F-04 | 同上 | `<true,0,8>` | 250 | RVV-GENERIC-M>1 |
| F-05 | 同上 | `<true,0,2>` | 320 | RVV-GENERIC-M>1 |
| F-06 | 同上 | `<true,3,2>` | 426 | RVV-GENERIC-M>1 |
| F-07 | 同上 | `<true,3,4>` | 459 | RVV-GENERIC-M>1 |
| F-08 | 同上 | `<true,1,8>` | 494 | RVV-GENERIC-M>1 |
| F-09 | 同上 | `<true,1,32>` | 527 | RVV-GENERIC-M>1 |
| F-10 | 同上 | `<true,1,20>` | 579 | RVV-GENERIC-M>1 |
| F-11 | 同上 | `<true,0,16>` | 619 | RVV-GENERIC-M>1 |
| F-12 | 同上 | `<true,8,1>` | 657 | RVV-GENERIC |
| F-13 | 同上 | `<true,2,1>` | 693 | RVV-GENERIC |
| F-14 | 同上 | `<true,4,1>` | 740 | RVV-GENERIC |
| I-01 | `integer_ops/depthwise_conv.h` | `QuantizedDepthwiseConvKernel<true,8,2>` | 46 | RVV-GENERIC-M>1 |
| I-02 | 同上 | `<false,8,1>` | 92 | RVV-GENERIC |
| I-03 | 同上 | `<false,4,2>` | 159 | RVV-GENERIC-M>1 |
| I-04 | 同上 | `<false,2,8>` | 227 | RVV-GENERIC-M>1 |
| I-05 | 同上 | `<false,2,2>` | 301 | RVV-GENERIC-M>1 |
| I-06 | 同上 | `<false,2,1>` | 364 | RVV-GENERIC |
| I-07 | 同上 | `<false,1,2>` | 474 | RVV-GENERIC-M>1 |
| I-08 | 同上 | `<false,1,4>` | 532 | RVV-GENERIC-M>1 |
| I-09 | 同上 | `<false,4,1>` | 624 | RVV-GENERIC |
| I-10 | 同上 | `<false,4,4>` | 691 | RVV-GENERIC-M>1 |
| I-11 | 同上 | `<true,0,3>` | 774 | RVV-GENERIC-M>1 |
| I-12 | 同上 | `<true,0,2>` | 857 | RVV-GENERIC-M>1 |
| I-13 | 同上 | `<true,0,1>` | 920 | RVV-GENERIC |
| I-14 | 同上 | `<true,16,1>` | 1001 | RVV-GENERIC |
| I-15 | 同上 | `<true,8,1>` | 1052 | RVV-GENERIC |
| I-16 | 同上 | `<true,1,16>` | 1085 | RVV-GENERIC-M>1 |
| I-17 | 同上 | `<true,1,32>` | 1126 | RVV-GENERIC-M>1 |
| I-18 | 同上 | `<true,1,20>` | 1178 | RVV-GENERIC-M>1 |
| I-19 | 同上 | `<true,1,8>` | 1224 | RVV-GENERIC-M>1 |
| I-20 | 同上 | `<true,2,1>` | 1255 | RVV-GENERIC |
| I-21 | 同上 | `<true,4,1>` | 1314 | RVV-GENERIC |
| I-22 | 同上 | `<false,12,1>` | 1374 | RVV-GENERIC |
| U-01 | `depthwiseconv_uint8.h` | `QuantizedDepthwiseConvKernel<true,8,2>` | 42 | RVV-GENERIC-M>1 |
| U-02 | 同上 | `<false,8,1>` | 89 | RVV-GENERIC |
| U-03 | 同上 | `<false,4,2>` | 157 | RVV-GENERIC-M>1 |
| U-04 | 同上 | `<false,2,8>` | 227 | RVV-GENERIC-M>1 |
| U-05 | 同上 | `<false,2,2>` | 304 | RVV-GENERIC-M>1 |
| U-06 | 同上 | `<false,2,1>` | 370 | RVV-GENERIC |
| U-07 | 同上 | `<false,1,2>` | 484 | RVV-GENERIC-M>1 |
| U-08 | 同上 | `<false,1,4>` | 544 | RVV-GENERIC-M>1 |
| U-09 | 同上 | `<false,4,1>` | 639 | RVV-GENERIC |
| U-10 | 同上 | `<false,4,4>` | 709 | RVV-GENERIC-M>1 |
| U-11 | 同上 | `<true,0,3>` | 794 | RVV-GENERIC-M>1 |
| U-12 | 同上 | `<true,0,2>` | 880 | RVV-GENERIC-M>1 |
| U-13 | 同上 | `<true,0,1>` | 945 | RVV-GENERIC |
| U-14 | 同上 | `<true,16,1>` | 1068 | RVV-GENERIC |
| U-15 | 同上 | `<true,8,1>` | 1122 | RVV-GENERIC |
| U-16 | 同上 | `<true,1,16>` | 1156 | RVV-GENERIC-M>1 |
| U-17 | 同上 | `<true,1,32>` | 1200 | RVV-GENERIC-M>1 |
| U-18 | 同上 | `<true,1,20>` | 1256 | RVV-GENERIC-M>1 |
| U-19 | 同上 | `<true,1,8>` | 1305 | RVV-GENERIC-M>1 |
| U-20 | 同上 | `<true,2,1>` | 1337 | RVV-GENERIC |
| U-21 | 同上 | `<true,4,1>` | 1400 | RVV-GENERIC |
| U-22 | 同上 | `<false,12,1>` | 1464 | RVV-GENERIC |

## 汇总与证据边界

| 文件 | ARM Neon 特化数 | `depth_multiplier=1` | `depth_multiplier>1` | 当前 RVV结论 |
| --- | ---: | ---: | ---: | --- |
| FP32 `depthwiseconv_float.h` | 14 | 6 | 8 | 14 个由通用 RVV 入口覆盖（M=1 通道路径 + M>1 fan-out 路径） |
| INT8 `integer_ops/depthwise_conv.h` | 22 | 9 | 13 | 22 个由通用 RVV 入口覆盖（M=1 通道路径 + M>1 fan-out 路径） |
| UINT8 `depthwiseconv_uint8.h` | 22 | 9 | 13 | 22 个由通用 RVV 入口覆盖（M=1 通道路径 + M>1 fan-out 路径） |
| **合计** | **58** | **24** | **34** | **58/58 call-site family 路由至通用 RVV；不是 58 个独立专用微内核** |

六个冻结模型的 depthwise 实际都是 `depth_multiplier=1`；固定输入下 QEMU VLEN=128/256/512 六模型共 18/18 通过，MobileNet 量化输出最大差异 1 LSB、FP32 模型输出最大绝对误差 `4.888e-6`。M>1 另由独立 scalar reference 覆盖 M=2/3/5/17、VLEN=128/256/512/1024、行切分与零点边界，四档均通过，证据见 `results/depthwise/`；模型证据见 `results/model-differential/`，官方 kernel suite 三档各 137/137 见 `results/official-kernel-suite/`。

因此本表采用分层统计口径：功能组口径为六模型实际命中的 27/27；实际 61 个 ARM 保护函数入口和这 58 个 ARM 模板特化分别统计，历史 63 行入口表中的两个 Pow 函数不属于 ARM 分母。generic RVV 已覆盖该 depthwise family 的 M=1/M>1 layout；58 个模板特化均映射到对应通用入口，非 RVV 目标仍保留标量回退。
