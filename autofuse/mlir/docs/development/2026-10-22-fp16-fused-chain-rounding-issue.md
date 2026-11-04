# FP16 融合链与 eager 中间舍入语义差异

> 2026-10-23：逐元素 NPU 专项已撤下。下文命令与结果为历史记录；复现需使用撤下前的提交 `0706a660dee787142366e65c43c7abe409f1aa89`。

## 状态

2026-10-22 记录，暂缓修改。后续再决定调整业务计算语义，还是调整测试 oracle / 精度标准。
当前不修改升精度逻辑、测试容差或失败判定，不将这些失败改记为通过。

## 现象与范围

NPU `elewise_matrix.py` 在固定输入 seed=0、原有 `rtol=atol=1e-3` 下，96 项中 89 项通过、7 项数值失败。
失败均来自 FP16 多算子融合链：

| 形状 | 失败用例 |
|---|---|
| `(16,32)` | `chain_diamond`、`chain_long` |
| `(16,37)` | `chain_diamond`、`chain_long` |
| `(2048,128)` | `chain_mul_add`、`chain_diamond`、`chain_long` |

修改前 `c0b1acd8` 与 scratch 修改后 `203633ac` 的失败名单、不匹配元素数、最大绝对/相对误差及报告位置完全一致。
该现象不是此次 `life_time_axis_id` / `scratch_roles` 修改引入的回归。

## 已验证的原因

测试的 CPU eager oracle 逐算子产生 FP16 结果，包含各算子之间的 FP16 舍入。
设备侧融合前端将输入升为 FP32，链内保留 FP32 中间计算，最后转回 FP16。

设备环境中的 torch_npu 实现可追溯到：

- `_inductor/ascendc/npu.py`：输入加载使用 `upcast_dtype=True`。
- `_inductor/ascendc/asc_overrides.py` 的 `to_dtype`：将 FP16/BF16 的目标计算类型映射为 FP32。
- 实际 HintGraph 已包含输入 FP16→FP32 Cast 和输出 FP32→FP16 Cast；精度策略发生在 MLIR lowering 之前。

额外探针覆盖三种融合链 × 三个形状，共 9 项：所有 NPU 输出均与“相同 FP16 输入转 FP32，整条链计算完后再转 FP16”的 CPU 结果逐元素完全相同。
这 9 项均不与逐算子 FP16 eager 逐位相同，其中上述 7 项超过既有容差。
该证据支持中间舍入语义差异，未显示 scratch 复用或搬运覆盖错误。
此结论限于已验证的用例，不能据此认定所有 FP16 差异都可接受。

## 待决策事项

- 是否要求融合路径保留逐算子的 FP16 舍入语义；若要求，应在前端保留对应舍入边界。
- 是否允许 FP32 中间计算；若允许，需要明确 oracle、容差及 eager 偏差的记录标准。
- 不直接放宽全局容差；须先确定精度契约，并保留能够识别真实数值错误的验证。

## 复现与证据

在已配置的 NPU 测试环境中运行（沿用原测试入口）：

```bash
python3 autofuse/mlir/test/e2e/npu/elewise_matrix.py \
  --group chain --dtypes f16 --mode mlir --steps 0 \
  --work-dir /tmp/fp16-chain-rounding
```

本次本地证据位于仓库根目录下的以下路径；这些是未纳入版本库的运行产物：

- `build/scratch-npu-203633ac/comparison/comparison.json`：修改前后逐项比较。
- `build/scratch-npu-203633ac/comparison/before/elewise_matrix.log`、`after/elewise_matrix.log`：原始日志。
- `build/scratch-npu-203633ac/investigation/fp16_metrics.jsonl`：9 项升精度 oracle 对比。
- `build/scratch-npu-203633ac/investigation/fp16_*.pt`：输入、实际输出与两种参考输出。
- `build/scratch-npu-203633ac/investigation/frontend/`：设备环境相关前端源码快照。
