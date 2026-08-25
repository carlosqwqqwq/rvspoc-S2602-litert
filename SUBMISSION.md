<!-- SPDX-License-Identifier: Apache-2.0 -->

# RVSPOC S2602 提交说明（优化候选）

状态：2026-08-26 本地待提交候选，源码绑定当前本地 commit（执行 `git rev-parse HEAD` 获取唯一 hash）。没有创建 PR、没有 push；本文只记录当前源码和已实际运行的验证，不把历史二进制当作本轮结果。

## 1. 绑定信息

- 基线：LiteRT v2.1.4，`ea79caffdd0f52cd44f203674f18a16a3cb861ad`。
- TensorFlow 子仓库：`2adc36c677a558b93a454705059baac2b0cdf5a3`。
- 当前候选源码已在 Docker 中生成 `benchmark_model` ELF，SHA256=`996cd7e735c5eebcb6711d8cddfa8f69e7b48d589b39c3379b4bd19fa648c1f3`；该 ELF 对应当前本地工作树中的代码内容，提交前仍应在目标环境按第 3 节重建并重新计算 SHA256。
- A210 当前不可用，因此本文件不宣称本轮改动已经完成 A210 benchmark，也不宣称 110 ms 已达标。
- 本轮可复核的 QEMU 入口回归和性能筛选结果见第 4 节；它们不能替代 A210 真机数据。

## 2. 实现摘要

- RVV 实现通过 `#ifdef __riscv_vector` 与通用代码隔离；RV64GC 和无 RVV 构建保留标量回退。
- RVV 1.0 路径使用 `vsetvl` 自适应 VLEN，已验证 128/256/512 bit。
- 标准布局 FP32 GEMM 在 VLEN≥256、大矩阵且至少有 4 个输出通道时使用原创四通道共享 RHS 内核；`K<16` 走标量展开，其他形状保留 RVV 单输出回退，避免小矩阵和寄存器压力回归；大矩阵 batch=1 也可复用 RHS。
- FP32 及精确 int8 GEMM 增加 VLEN≥256 的 2×2 输出微内核：FP32 门控为 `rows≥8、K≥96、2≤N≤64`，int8 门控为 `rows≥8、K≥16、N≥2`；其他类型、FP32 其他形状和 VLEN=128 自动回退到既有路径。
- INT8 GEMM 在 `K≤e8m1 VLMAX` 时增加独立的 4 行×2 列单向量路径，一次加载四个 lhs 与两个 rhs 向量并复用到八个输出；K 尾块、奇数行列和 VLEN=128 仍走既有路径。
- FP32 GEMM 四通道内核同时处理最后 1–3 个输出行，覆盖非 4 倍数的真实卷积矩阵；深度尾块仍由 `vsetvl` 处理。
- FP32/量化 GEMM 的分段累加使用 tail-undisturbed RVV FMA，K 尾块按 RVV 规范保留已有累加 lane。
- FP32 depthwise 在 VLEN≥256、`depth_multiplier=1` 且相邻像素过滤范围一致时成对计算相邻输出，按实际 stride/dilation 计算输入地址并复用 filter 向量；边界和其他 VLEN 走已有回退。
- INT8 per-channel depthwise 最终采用单像素通道块向量化路径，避免双像素累加的寄存器压力；QEMU VLEN=256 的 `28×28×64` 当前 shape probe 为 `19.075 ms`，三档 VLEN 整模输出逐字节一致。
- UINT8 depthwise 在同一窗口条件下复用 filter 向量并并行计算两个像素；VLEN=128/256/512 均启用，实际通道块仍由 `vsetvl` 处理。
- FP32/INT8 GEMM 的空间列按精确均衡区间分片，非整除列数也会创建完整的非空任务集合。
- 覆盖 FP32/INT8 GEMM、DepthwiseConv、量化/反量化、激活、reduce、pooling、resize、tensor utils 和 4-bit FC 等 ARM Neon/SVE 对应路径。
- 默认 gemmlowp double-rounding 量化使用 `vsmul.vx`；逐通道路径使用 `vsmul.vv`，并保留舍入修正。
- 当前增量在大尺寸、同形状、非广播 INT8 ADD 上启用保守任务分片：元素数至少 65536 且线程数大于 1 时切分不重叠输出区间；小张量、广播和非 RVV 构建不改变原路径。
- GEMM 线程门控经过分类型 A/B：FP32 保留 `4M` MAC 门槛，量化 GEMM 使用 `2M` MAC 门槛；这样覆盖中小型 INT8 CONV，同时避免 FP32 EfficientDet 回归。
- 非广播 INT8/UINT8 ADD 在 VLEN≥256 时预设并复用 e8m1 VLMAX，仅深度尾块重新设置 VL；VLEN=128 保留逐块设置 VL。长窗口 helper A/B 在 VLEN=256/512 分别为 `1.0145×/1.0240×`，三档 VLEN 精度均 `0 mismatch`。
- scalar-broadcast INT8 ADD 同样复用 e8m1 VLMAX；同公式 helper A/B 在 VLEN=256 约 `3.0%`、VLEN=512 约 `0.9%` 加速，三档 VLEN 均 `0 mismatch`。
- Depthwise 与 shuffled Fully Connected 的可选 bias 在 scalar 和 RVV 路径均按零累加；入口回归测试覆盖空 bias，带 bias 的冻结 benchmark 路径不变。

## 3. 构建

在 Linux/Docker 或有效 A210 环境执行，不在 Windows 控制面直接运行 RISC-V ELF：

```bash
cmake -S tflite -B build-rv64gcv \
  -DCMAKE_TOOLCHAIN_FILE=toolchains/riscv64-linux-gnu.cmake \
  -DTENSORFLOW_SOURCE_DIR="$PWD/third_party/tensorflow" \
  -DTFLITE_HOST_TOOLS_DIR=/usr/bin \
  -DProtobuf_PROTOC_EXECUTABLE=/usr/bin/protoc \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-march=rv64gcv -mabi=lp64d" \
  -DTFLITE_ENABLE_BENCHMARK_MODEL=ON \
  -DTFLITE_ENABLE_XNNPACK=OFF -DTFLITE_ENABLE_GPU=OFF
cmake --build build-rv64gcv --target benchmark_model --parallel 8
```

RV64GC scalar 对照将 `-march=rv64gcv` 改为 `-march=rv64gc`。完整入口为 `scripts/build-riscv.sh`。

QEMU 运行示例（模型路径按实际下载位置替换）：

```bash
qemu-riscv64 -L /usr/riscv64-linux-gnu \
  -cpu rv64,v=true,vlen=256,vext_spec=v1.0 \
  build-rv64gcv/tools/benchmark/benchmark_model \
  --graph=<MODEL>.tflite --num_threads=8 --warmup_runs=1 --num_runs=5
```

启用 `-DTFLITE_KERNEL_TEST=ON` 时，另需传入宿主机原生编译的 `-DTFLITE_HOST_OPTION_WRITER_GENERATOR=/path/to/option_writer_generator`；该工具只在构建期运行，不能使用 RV64GCV 目标二进制代替。

## 4. 当前验证

- 随提交包提供的官方 kernel suite 摘要：`results/official-kernel-suite/vlen128-summary`、`vlen256-summary`、`vlen512-summary`，三档均 137/137 PASS、0 FAIL；这是 2026-08-23 的独立回归快照，不能替代本轮 A210 整模重测。
- 本轮直接 helper 回归：GEMM `17×65×32` 在 VLEN=128/256/512 最大绝对误差 `1.78814e-7`；新 INT8 2×2 内核的 `rows=8,K=65,N=65`（完整块+尾块、非零 zero-point、逐通道 multiplier）三档均 `mismatches=0`；FP32/INT8/UINT8 depthwise 的 stride=2、dilation=2 边界用例在 VLEN=128/256/512 均通过 scalar 对照，UINT8 双像素路径的 `5×7×17` 用例三档均 `mismatches=0`。
- QEMU 固定形状直接 helper 微基准：FP32 depthwise `28×28×64` 双像素路径存档约 `12.09 ms`；INT8 per-channel depthwise `28×28×64` 当前单像素路径约 `19.075 ms`；UINT8 depthwise `28×28×64` 同一基准在 VLEN=128 为 `9.530→7.615 ms`（1.25×）、VLEN=256 为 `7.738→6.327 ms`（1.22×）；49×512×1024 FP32 GEMM 共享 RHS 路径存档约 `510.0 ms`，同形状单输出回退约 `851.4 ms`（单次两轮方向性筛选，约 `1.67x`）。这些均不是 A210 整模成绩，冻结前需用最终源码重测。
- 新增 2×2 GEMM 的同源 helper A/B：VLEN=256 下，int8 代表性小 N 形状约 `1.10–1.40×`，FP32（`K≥96`）代表性形状约 `1.03–1.25×`；VLEN=128 自动回退，VLEN=512 保持逐元素一致并在目标形状获益。FP32 最大误差为 `0`，int8 与 scalar 逐元素 `0 mismatch`；这些是方向性微基准，不是 A210 整模成绩。
- int8 大空间 shape bench 覆盖 EfficientDet 的代表 `(rows,K,N)`，VLEN=256/512 均逐元素 `0 mismatch`，2×2 相对既有 4×1 路径约 `1.23–4.67×`；加入 4×2 路径后，与插入前二进制交替 5 轮 `num_runs=3` 均值为 `2890.79→2845.89 ms`（约 `1.55%`），再启用量化 GEMM `2M` 线程门槛后，最终源码与旧候选交替 3 轮 `num_runs=5` 整模均值 `3102.32→2756.68 ms`（约 `11.1%`），EfficientDet INT8 输出 SHA256=`729545205046f1cf96340374a76991ab3a65747b9cf7f9e008db579356a82f4c`。最终源码单次 profile 的 CONV/Depthwise/ADD 聚合为 `2090.330/349.504/57.979 ms`。profile 是单次方向性快照，整模交替均值作为主要结论；这是模拟器结果，不能替代 A210 最终成绩。
- `vsmul.vv` 探针覆盖 8313 组边界和确定性随机输入，三档 VLEN 均 `mismatches=0`。
- VLEN=256、8 线程六模型输出一致性通过：FP32 最大误差 `7.823e-08`，INT8 `0 LSB`。
- VLEN=512 的 FP32 Logistic e32m2 分派已进入冻结证据 GCV ELF；EfficientDet FP32 五轮 QEMU A/B 均值约提升 6.7%，中位数约提升 5.1%；VLEN=128/256 保留原 m1 路径。
- QEMU VLEN=256、8 线程 ADD A/B 长窗口：MobileNetV1 INT8 `1.69830e6→1.67892e6` us，MobileNetV2 INT8 `1.39072e6→1.30194e6` us，EfficientDet-Lite0 INT8 `4.01407e6→3.72572e6` us。该表用于优化筛选，不是 A210 最终成绩。
- 当前源码完整链路 smoke：候选 ELF（SHA256=`996cd7e735c5eebcb6711d8cddfa8f69e7b48d589b39c3379b4bd19fa648c1f3`）运行仓库内 MobileNetV2 FP32 `mobilenet_v2_fp32.tflite`，单线程、warmup=1、正式 1 次，QEMU VLEN=256 avg 为 `5542.91 ms`，初始化 `78.731 ms`、overall footprint `24.4453 MB`，正常退出；该小样本 QEMU 结果只证明当前 ELF 可运行和 VLEN 分派，不替代官方三模型或 A210 的 avg/p50/p95/110ms 验收。
- 同一 smoke 导出的 1001 个 FP32 输出值在三档 VLEN 均无元素超过 `1e-5`：128↔256 最大绝对差 `2.3841858e-7`，256↔512 最大绝对差 `3.5762787e-7`。输出 raw 只保存在本地 ignored build cache，不进入提交包。
- `rvv_entry_coverage_test.cc` 已分别用 `-march=rv64gcv` 与 `-march=rv64gc` 完成编译检查；完整 GoogleTest 目标还需要宿主侧 `option_writer_generator`，其前置命令已在第 3 节保留。

入口覆盖测试：新增 `tflite/kernels/internal/optimized/rvv_entry_coverage_test.cc`，用 scalar/reference 结果校验 FC（含空 bias shuffled-FC）、FP32/INT8/UINT8 17 通道 depthwise（含 stride/dilation）、elementwise、量化、requantize、dequantize、pool、定点激活和 argmin/argmax 等代表路径；`scripts/audit-entry-coverage.py` 对 61 行矩阵、58 个 depthwise template 和三档 suite 摘要做静态审计。静态 PASS 不等同于 61 行动态 VLEN/fallback/diff 证明。

入口测试构建和运行：在启用 `-DTFLITE_KERNEL_TEST=ON` 的交叉编译目录中，构建目标名为 `internal-optimized-rvv_entry_coverage_test`；使用 `qemu-riscv64` 将 `vlen=128/256/512` 分别运行，并用 `-march=rv64gc` 运行 scalar fallback。

复现差分：

```bash
python3 scripts/compare-model-differential.py <model-differential-result-dir>
bash scripts/run-official-suite.sh <kernel-test-build> 256 <result-dir>
```

## 5. 覆盖与边界

静态审计为 61 个实际 ARM 保护入口（47 direct + 14 shared），六模型实际命中 27 个功能组，Depthwise 58 个 Neon template 特化有逐项清单。`IntegerExponentPow`/`BroadcastPow4D` 是新增 RVV-only 路径，不进入 ARM 分母。实现保留 scalar fallback；静态 61/61 和代表路径测试不单独等同于严格 90% 动态入口证明。

覆盖审计命令（在应用补丁后的源码 checkout 中执行）：

```bash
python3 scripts/audit-entry-coverage.py \
  --source-root "$PWD/tflite" \
  --matrix "$PWD/docs/S2602/optimized-ops-evidence-matrix.md" \
  --suite-root "$PWD/results/official-kernel-suite" \
  --depthwise-inventory "$PWD/docs/S2602/depthwise-template-audit.md"
```

矩阵、suite 摘要和 depthwise inventory 均随提交包提供；若源码 hash 不一致，报告仍保留 `STATIC-PASS`，不升级为动态入口 PASS。

历史 A210 六模型数据与当前源码快照不一致，只能作为待复测基线；当前候选仅完成 QEMU 验证，尚未完成真机验收。提交前必须在 A210 重新构建当前源码，报告 avg/p50/p95/std、p95/avg、FPS、footprint、initialization，并逐模型核对 `≤110 ms`。

## 6. 许可与 AI 披露

代码按 Apache-2.0 发布。RVV 逻辑依据 LiteRT 原始 ARM 内核算法契约自主适配，未直接搬运第三方 RISC-V LiteRT 移植代码。AI 辅助使用范围、人工审阅和验证方式见 `AI-DISCLOSURE.md`。
