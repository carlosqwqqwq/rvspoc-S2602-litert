<!-- SPDX-License-Identifier: Apache-2.0 -->

# 固定输入

输入均为连续 little-endian tensor 数据，供 `scripts/run-model-differential.sh`
和 benchmark 命令使用。

| 文件 | shape | 元素类型 |
| --- | --- | --- |
| `mobilenet-fp32.input` | `1x224x224x3` | FP32 |
| `mobilenet-u8.input` | `1x224x224x3` | UINT8 |
| `efficientdet-fp32.input` | `1x320x320x3` | FP32 |
| `efficientdet-u8.input` | `1x320x320x3` | UINT8 |

脚本通过 `--input_layer_value_files=input:<file>` 传入文件，并使用
`--input_layer=input` 与表中 shape。模型、输入、线程数和 VLEN 的组合由
`SUBMISSION.md` 中的命令固定。
