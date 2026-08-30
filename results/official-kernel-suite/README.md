<!-- SPDX-License-Identifier: Apache-2.0 -->

# 官方 kernel suite 归档

三档 RVV VLEN 的目标日志和逐目标摘要如下：

| VLEN | 目标数 | PASS | FAIL |
| ---: | ---: | ---: | ---: |
| 128 | 137 | 137 | 0 |
| 256 | 137 | 137 | 0 |
| 512 | 137 | 137 | 0 |

`vlen*-runner.log` 保存完整 PASS/FAIL 记录，`vlen*-summary` 保存逐目标摘要。
从新构建重复运行：

```bash
bash scripts/run-official-suite.sh <build-root> 128 <out-dir>
bash scripts/run-official-suite.sh <build-root> 256 <out-dir>
bash scripts/run-official-suite.sh <build-root> 512 <out-dir>
```
