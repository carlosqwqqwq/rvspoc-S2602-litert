<!-- SPDX-License-Identifier: Apache-2.0 -->

# Depthwise multiplier 回归

本目录保存 multiplier=2/3/5/17 组合在 VLEN=128/256/512/1024 下的独立回归
源码、边界探针、日志和结果表，覆盖行切分、通道尾部及 zero-point 边界。
这些材料与 `docs/S2602/depthwise-template-audit.md` 的 58 个 ARM 特化审计对应。
