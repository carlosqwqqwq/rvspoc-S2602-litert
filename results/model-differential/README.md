<!-- SPDX-License-Identifier: Apache-2.0 -->

# 六模型精度差分

本目录保存 MobileNetV1、MobileNetV2、EfficientDet-Lite0 六种 FP32/INT8 组合
的 GC scalar 输出和 GCV 输出。每个模型有 VLEN=128/256/512 三份 GCV raw，
共 `18/18` 个模型-VLEN 组合。

从仓库根目录执行：

```bash
python3 scripts/compare-model-differential.py results/model-differential
```

脚本按 FP32 `1e-5` 最大绝对误差和 INT8 `1 LSB` 最大有符号数值差进行检查。
