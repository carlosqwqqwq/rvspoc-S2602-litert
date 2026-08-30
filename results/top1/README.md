<!-- SPDX-License-Identifier: Apache-2.0 -->

# 200 张真实图像 Top-1 回归

本目录保存固定 200 张 Imagenette 图像对应的四个 MobileNet 模型参考输出、
RVV 输出、标签表、逐模型退出记录和两个统计脚本，共 800 次推理。四个模型
均为 `200/200` argmax 一致，Top-1 差异均为 `0.00` 个百分点。

解压输出归档后执行：

```bash
tar -xzf rv-raw-current-v256.tar.gz
tar -xzf x86-ref2-raw.tar.gz
python3 compare-top1.py .
python3 evaluate-top1.py .
```

`ground-truth.tsv` 固定样本与标签对应关系，`top1-accuracy.tsv` 保存最终统计。
