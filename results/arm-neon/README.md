<!-- SPDX-License-Identifier: Apache-2.0 -->

# Raspberry Pi 5 ARM Neon 对照

`summary.tsv` 保存 Raspberry Pi 5（Cortex-A76、aarch64）4 线程、warmup=3、
配置 `num_runs=10` 的六模型 Neon 延迟、尾延迟、标准差和 footprint，作为 ARM
Neon 工程对照数据；`runs` 列记录 benchmark 实际完成的迭代次数。
