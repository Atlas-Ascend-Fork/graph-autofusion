# SK 算子融合断图问题分析

所需产物：`sk_meta`、`static_kernel_compile_outputs`。

## 1. 先确认实际融合范围

按 PID、模型和时间关联同一次运行，比较原始 scope、最终融合节点和更新后的图。优先检查 `sk_fusion_fail_reasons.log`、`sk_scope_split.log`、`sk_fused_nodes.log`，必要时结合 `super_kernel.log`、图 JSON 和 task queue。文件名及字段大小写可能随版本变化；找不到固定名称时，在当前产物中搜索 `breakReason`、`triggerNode`、`syncAllNodes` 等字段。

区分以下情况：

- 节点不可融合：检查具体节点的绑定解析、`isFusible` 和失败原因。
- 节点均可融合，但一个 scope 被拆成多个成功融合的 scope：查找切分 pass 和核数、同步或死锁约束，不把它等同于算子不支持 SK。
- scope 外节点标记 `NOT_IN_SCOPE`：核对其位置，不将它归为 scope 内算子的融合失败。

日志中的节点总数可能包含 scope 标记和 placeholder；报告实际算子 node ID、每组融合范围及入口核数，避免混用节点计数。结合 stdout、执行和同步记录，区分保护性拆分与设备执行失败。

## 2. 常见原因汇总

1. 静态编译失败或未命中，参考 [静态编译问题分析](static-compile.md)。
2. 不可融合算子影响，例如 AICPU、TBE、Triton 算子等。
3. batch mode 算子的核数约束冲突。在 `sk_fusion_fail_reason.log` 中搜索 `breakReason=SYNCALL_OP_DROP`，查看 `triggerNode`（触发当前断开的节点）、`SyncAllNodes`（可能产生影响的全核同步算子）和 `detail`（全核同步算子核数与 SK 核数之间的具体冲突）。
3. 自定义 `<<<>>>` 算子未完成 `SK_BIND` 等适配。

   自定义算子提前编译时，需要完成 `SK_BIND` 等 SK 相关适配，才能使能 SK。按 [核函数直调算子适配说明](https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/guide/programming_guide/advanced_programming/super_kernel/kernel_direct_call_adaptation.md) 检查适配情况。


## 3. 不支持融合的算子

在对应芯片版本中，部分算子为非 AscendC 实现，或不支持静态编译，因此不支持 SK 融合。可考虑替换为支持融合的 AscendC 算子。

以下 built-in 算子列表提供参考：

| 算子名称（OpType） | 芯片版本 | 不支持原因 | 规避方式 |
| --- | --- | --- | --- |
| `ForeachCopy` | A5 | CANN 9.2.0 版本之前不支持，CANN 9.2.0 版本已放开静态编译 | NA |
| `Range` | A5 | CANN 9.2.0 版本之前不支持，CANN 9.2.0 版本已放开静态编译 | NA |
| `Cast` | A3 | 非 AscendC 算子 | AscendC 替换算子，算子自定义开发 AscendC 版本的算子 |
| `Add` | A3 | 非 AscendC 算子 | 寻找 AscendC 替换算子，替换接口为 `torch._foreach_add` |
| `Abs` | A3 | 非 AscendC 算子 | 寻找 AscendC 替换算子 |
| `Sub` | A3 | 非 AscendC 算子 | 寻找 AscendC 替换算子 |
| `Mul` | A3 | 非 AscendC 算子 | 寻找 AscendC 替换算子，算子自定义开发 AscendC 版本的算子 |
| `Muls` | A3 | 非 AscendC 算子 | 寻找 AscendC 替换算子 |
| `Neg` | A3 | 非 AscendC 算子 | 寻找 AscendC 替换算子 |
| `Rsqrt` | A3 | 非 AscendC 算子 | 寻找 AscendC 替换算子 |
| `LogicalOr` | A3 | 非 AscendC 算子 | 寻找 AscendC 替换算子 |
| `LogicalNot` | A3 | 非 AscendC 算子 | 寻找 AscendC 替换算子 |
| `Equal` | A3 | 非 AscendC 算子 | 寻找 AscendC 替换算子 |
| `Less` | A3 | 非 AscendC 算子 | 寻找 AscendC 替换算子 |
| `GreaterEqual` | A3 | 非 AscendC 算子 | 寻找 AscendC 替换算子 |
| `MatMulV2` | A3 | 非 AscendC 算子 | 使用 `MatMulV3` 算子进行替换，替换方式：算子添加白名单，使对应的 shape 走到对应的算子分支 |
| `BatchMatMulV2` | A3 | 非 AscendC 算子 | 寻找 AscendC 替换算子 |
| `GatherV2` | A3 | 非 AscendC 算子 | 静态编译黑名单，NA |
| `SelectV2` | A3 | 非 AscendC 算子 | 静态编译黑名单，NA |
| `Tile` | A3 | 非 AscendC 算子 | 寻找 AscendC 替换算子 |
| `Fill` | A3 | 非 AscendC 算子 | 寻找 AscendC 替换算子 |
| `MaskedFill` | A3 | 非 AscendC 算子 | 寻找 AscendC 替换算子 |
| `Sigmoid` | A3 | 非 AscendC 算子 | 寻找 AscendC 替换算子 |
| `LayerNormV3` | A3 | 存疑 | NA |
| `TensorMove` | A3 | 非 AscendC 算子 | 寻找 AscendC 替换算子 |
| `ReduceMean` | A3 | 非 AscendC 算子 | 寻找 AscendC 替换算子 |
| `ReduceSum` | A3 | 非 AscendC 算子 | NA |
| `Slice` | A3 | 非 AscendC 算子 | 寻找 AscendC 替换算子 |
| `AsStrided` | A3 | 非 AscendC 算子 | 寻找 AscendC 替换算子 |
| `Transpose` | A3 | 非 AscendC 算子 | 寻找 AscendC 替换算子 |
| `FusedInferAttentionScore` | A3 | 动态 tiling 算子，不支持 | NA |
