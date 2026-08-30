<!-- SPDX-License-Identifier: Apache-2.0 -->

# 固定输入

输入均为连续 little-endian tensor 数据，供模型回归和板卡运行脚本使用。

| 文件 | shape | 元素类型 |
| --- | --- | --- |
| `mobilenet-fp32.input` | `1x224x224x3` | FP32 |
| `mobilenet-u8.input` | `1x224x224x3` | UINT8 |
| `efficientdet-fp32.input` | `1x320x320x3` | FP32 |
| `efficientdet-u8.input` | `1x320x320x3` | UINT8 |

对应文件直接作为输入载荷传给复现实验脚本；模型、输入、线程数和 VLEN 的组合
由 `SUBMISSION.md` 中的命令固定。
