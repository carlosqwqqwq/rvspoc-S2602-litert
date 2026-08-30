<!-- SPDX-License-Identifier: Apache-2.0 -->

# 六模型精度差分

本目录保存 MobileNetV1、MobileNetV2、EfficientDet-Lite0 六种 FP32/量化组合
的 GC scalar 输出和 GCV 输出。每个模型有 VLEN=128/256/512 三份 GCV raw，
共 `18/18` 个模型-VLEN 组合。

从仓库根目录使用提交内 ELF 和固定输入重新生成 raw：

```bash
bash scripts/run-model-differential.sh
```

再执行差分检查：

```bash
python3 scripts/compare-model-differential.py results/model-differential
```

脚本按输出 tensor 类型检查：FP32 使用 `1e-5` 最大绝对误差，MobileNet
量化输出使用 UINT8 的 `1 LSB`，EfficientDet-Lite0 量化模型的输出使用 FP32。
本次归档结果为：MobileNetV1 量化输出最大差异 1 LSB，MobileNetV2 为 0，
EfficientDet-Lite0 量化模型输出最大绝对误差为 0。
