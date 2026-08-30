<!-- SPDX-License-Identifier: Apache-2.0 -->

# EfficientDet-Lite0 INT8 算子热点

`efficientdet-int8-vlen256.csv` 是 GCV、QEMU RVV 1.0 VLEN=256、8 线程、warmup=3、
正式 3 次的算子 profile。`CONV_2D` 与 `DEPTHWISE_CONV_2D` 合计占 `93.5596%`，
用于确定 RVV 优化优先级。
