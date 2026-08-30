<!-- SPDX-License-Identifier: Apache-2.0 -->

# 当前 GCV/GC QEMU A/B

`SUMMARY.tsv` 保存 MobileNetV1、MobileNetV2、EfficientDet-Lite0 五个模型变体
在同一模型和固定输入、QEMU RVV 1.0 VLEN=256、8 线程、warmup=3、配置
`num_runs=10` 下的 GCV/GC 对照。EfficientDet 的两个 raw 文件为同一组输入
对应的输出；表中包含 avg、p50、p95、std、p95/avg、FPS、初始化和 footprint。

QEMU 结果用于回归和相对比较；A210 的 110 ms 指标需在有效 A210 slot 上重测。
