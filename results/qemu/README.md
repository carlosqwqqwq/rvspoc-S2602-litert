<!-- SPDX-License-Identifier: Apache-2.0 -->

# 当前 GCV/GC QEMU A/B

`SUMMARY.tsv` 保存 EfficientDet-Lite0 INT8、同一模型和输入、QEMU RVV 1.0
VLEN=256、8 线程、warmup=3、正式 10 次的 GCV/GC 对照。两个 raw 文件为
对应输出，表中包含 avg、p50、p95、std、p95/avg、FPS、初始化和 footprint。

QEMU 结果用于回归和相对比较；A210 的 110 ms 指标需在有效 A210 slot 上重测。
