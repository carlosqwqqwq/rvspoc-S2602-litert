<!-- SPDX-License-Identifier: Apache-2.0 -->

# S2602 提交说明

本说明对应 LiteRT v2.1.4 的 RV64GCV/RVV 1.0 适配。源码补丁以
`ea79caffdd0f52cd44f203674f18a16a3cb861ad` 为基线，构建产物和实验数据均
通过 SHA-256 绑定。本文给出一条完整复现路径；不依赖其他说明文件才能理解
提交内容。

## 结果摘要

| 项目 | 结果 | 口径 |
| --- | --- | --- |
| RV64GCV 编译与运行 | 已生成并运行验证 ELF | CMake Release，`-march=rv64gcv -mabi=lp64d` |
| RV64GC 回退编译与运行 | 已生成并运行验证 ELF | 同一源码，`-march=rv64gc -mabi=lp64d` |
| VLEN 适配 | 128/256/512 | 运行时 `vsetvl`，保留尾块和标量回退 |
| ARM 优化入口静态审计 | 61/61 | 47 个 direct 路由、14 个 shared 路由 |
| Depthwise 特化审计 | 58/58 | FP32 14、UINT8 22、INT8 22，通用 RVV 路径覆盖 |
| 官方 kernel suite | 137/137 × 3 档 VLEN | 三档日志均为 PASS |
| 六模型差分归档 | 18/18 | MobileNetV1/V2、EfficientDet-Lite0，FP32/INT8；记录按对应 ELF 哈希绑定 |
| FP32 算子差分 | 最大绝对误差 `3.576e-7` | 阈值 `1e-5` |
| INT8 算子差分 | 最大差异 `0` | 阈值 `1 LSB` |

### 当前候选的可复核性能

以下是当前 GCV 与 GC 标量构建在同一 QEMU 命令、同一模型、同一线程数下的
10 次正式测量。QEMU 用于回归和相对比较，不能代替 A210 的 110 ms 评分。

| 构建 | 模型 | VLEN | 线程 | avg (ms) | p50 (ms) | p95 (ms) | std (ms) | p95/avg | FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GCV | EfficientDet-Lite0 INT8 | 256 | 8 | 3340.270 | 3201.113 | 3995.759 | 333.136 | 1.196 | 0.299377 |
| GC scalar | EfficientDet-Lite0 INT8 | — | 8 | 7477.056 | 7372.711 | 8363.659 | 415.871 | 1.119 | 0.133742 |

相对 GC scalar：avg `2.238×`、p50 `2.303×`、p95 `2.093×`。当前 GCV
初始化 `133.474 ms`、overall footprint `19.3242 MB`；GC scalar 初始化
`116.023 ms`、overall footprint `20.3477 MB`。这些绝对延迟来自 QEMU，
不用于声称满足 110 ms。

### ARM Neon 工程对照

Raspberry Pi 5（Cortex-A76，aarch64，4 线程）的独立 Neon 参考测量如下，
用于满足 ARM Neon 对照要求；两种硬件不同，不将该表作为 A210 排名依据。

| 模型 | Neon avg (ms) | p50 (ms) | p95 (ms) | std (ms) | footprint (MB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MobileNetV1 FP32 | 27.426 | 27.415 | 28.330 | 0.433 | 41.84 |
| MobileNetV1 INT8 | 11.661 | 11.602 | 11.863 | 0.248 | 7.58 |
| MobileNetV2 FP32 | 23.346 | 23.330 | 23.791 | 0.245 | 38.11 |
| MobileNetV2 INT8 | 9.372 | 9.316 | 9.644 | 0.239 | 7.55 |
| EfficientDet-Lite0 FP32 | 72.162 | 72.119 | 74.353 | 1.037 | 48.17 |
| EfficientDet-Lite0 INT8 | 31.706 | 31.690 | 31.782 | 0.077 | 19.05 |

### Top-1 证据

冻结的 200 张真实图像、4 个 MobileNet 模型共 800 次推理全部正常退出，
RVV 与参考构建的 argmax 为 `800/800` 一致；四个模型的 Top-1 差异均为
`0.00` 个百分点。该证据带有独立运行的源码和 ELF 哈希，当前候选的整网
数值正确性以六模型差分归档和当前 EfficientDet 输出逐项复核为准。

### A210 与 110 ms 状态

当前 A210 slot 未完成 BatchMode 验收，本轮没有把历史板卡数据写成当前成绩。
因此：

- 当前提交材料包含可在 A210 上直接运行的 GCV/GC benchmark 和完整指标命令；
- 110 ms、A210 footprint、initialization、avg/p50/p95/std 和 FPS 必须用当前
  GCV ELF 在有效 slot 上重新测量；
- 现有历史参考中 EfficientDet-Lite0 INT8 为 `208.567 ms`，该数字绑定旧
  ELF，只用于定位剩余验证项，不代表本版本结果。

## 实现范围

### 构建与平台

- 增加 `riscv64-linux-gnu` 与 `riscv64-unknown-linux-gnu` CMake toolchain；
- `__riscv_vector` 条件编译隔离 RVV 路径，RV64GC 可在无 RVV 环境构建；
- 使用 CMake、Release、`-mabi=lp64d`，关闭尚无上游 RVV 后端的 XNNPACK/GPU；
- 保留 LiteRT 原始布局、量化契约和 Apache-2.0 许可。

### 热点内核

- FP32 GEMM 使用 RVV 向量累加和按列线程分片；
- INT8/UINT8 GEMM 使用 VLEN 自适应 `vsetvl`、2×2 向量块、K 尾块、奇数行列、
  bias、per-channel multiplier/shift 和 zero-point 修正；
- INT8 对称滤波器路径把输入 zero-point 修正折叠为行和，主循环采用 RVV
  widening multiply-accumulate，并保留通用路径；
- FP32、INT8、UINT8 depthwise 覆盖 depth multiplier=1 与大于 1、padding、
  stride、dilation、相邻像素复用和尾部通道；
- elementwise、pooling、resize、reduce、activation、quantize/dequantize、
  4-bit fully-connected 和 tensor utility 提供 RVV 路由，非 RVV 编译回到
  原始实现。

热点 profile 显示 EfficientDet-Lite0 INT8 中 `CONV_2D` 约占 79.6%、
`DEPTHWISE_CONV_2D` 约占 13.9%，合计约 93.6%；因此实现优先投入 GEMM/CONV
和 depthwise，而不是低占比 elementwise 算子。

## 构建

### 依赖

在 Linux、Docker 或 A210 Linux 环境准备：

- CMake 3.16 或更高版本；
- `riscv64-linux-gnu-gcc/g++/ar/ranlib`，或官方工具链对应的
  `riscv64-unknown-linux-gnu-*`；
- `flatc`、`protoc`、GNU make 或 Ninja；
- 仅在 QEMU 验证时需要 `qemu-riscv64`。

### RV64GCV 与 RV64GC

```bash
git submodule update --init --recursive

bash scripts/build-riscv.sh "$PWD" "$PWD/build-rv64gcv" rv64gcv
bash scripts/build-riscv.sh "$PWD" "$PWD/build-rv64gc" rv64gc
```

产物为：

```text
build-rv64gcv/tools/benchmark/benchmark_model
build-rv64gc/tools/benchmark/benchmark_model
```

使用 `riscv64-unknown-linux-gnu` 时，将 toolchain 文件替换为：

```bash
cmake -S tflite -B build-rv64gcv-unknown \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/toolchains/riscv64-unknown-linux-gnu.cmake" \
  -DTENSORFLOW_SOURCE_DIR="$PWD/third_party/tensorflow" \
  -DTFLITE_HOST_TOOLS_DIR=/usr/bin \
  -DProtobuf_PROTOC_EXECUTABLE=/usr/bin/protoc \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-march=rv64gcv -mabi=lp64d" \
  -DTFLITE_ENABLE_BENCHMARK_MODEL=ON \
  -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_GPU=OFF
cmake --build build-rv64gcv-unknown --target benchmark_model --parallel 8
```

### QEMU benchmark

```bash
qemu-riscv64 -L /usr/riscv64-linux-gnu \
  -cpu rv64,v=true,vlen=256,vext_spec=v1.0 \
  build-rv64gcv/tools/benchmark/benchmark_model \
  --graph=models/efficientdet_lite0_int8.tflite \
  --num_threads=8 --warmup_runs=3 --num_runs=10 \
  --min_secs=0 --max_secs=300 --enable_op_profiling=false
```

将 `vlen` 改为 `128` 或 `512` 可验证运行时向量长度分派；将 ELF 换成
`build-rv64gc/tools/benchmark/benchmark_model` 可得到标量参考。

## 精度与覆盖验证

模型差分目录需要包含同一输入、同一模型下的 GC raw 和 GCV raw：

```bash
python3 scripts/compare-model-differential.py results/model-differential-current
```

覆盖审计：

```bash
python3 scripts/audit-entry-coverage.py \
  --source-root "$PWD/tflite" \
  --matrix "$PWD/docs/S2602/optimized-ops-evidence-matrix.md" \
  --suite-root "$PWD/results/official-kernel-suite" \
  --depthwise-inventory "$PWD/docs/S2602/depthwise-template-audit.md" \
  --require-suite
```

官方 kernel suite：

```bash
bash scripts/run-official-suite.sh <build-root> 128 <out-dir>
bash scripts/run-official-suite.sh <build-root> 256 <out-dir>
bash scripts/run-official-suite.sh <build-root> 512 <out-dir>
```

## 提交边界与许可

进入 PR 的内容为 LiteRT 源码、必要的 CMake/toolchain、构建与验证脚本、
覆盖矩阵、AI 披露和本说明。构建目录、缓存、QEMU 临时文件、实验 raw、
本地容器信息和隐藏版本库均不属于提交内容；`.gitignore` 已覆盖常见本地
实验产物。

源码按 Apache-2.0 发布。RVV 路径依据 LiteRT 原始 ARM 实现的算法与量化
契约自主完成，未使用第三方 RISC-V LiteRT/TFLite 移植代码。AI 辅助方式、
人工复核边界和比例见 `AI-DISCLOSURE.md`。
