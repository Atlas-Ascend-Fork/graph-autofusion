# 案例：QuantBatchMatmulV3 从 MIX_1_1 进入 SK MIX_1_2 后卡死

## 1. 问题入口与现场材料

用户提供开启 SuperKernel 后运行卡死的现场，开始时并不知道卡死发生在框架侧还是子算子侧。

分析从本次运行产物开始，先定位具体子算子，再确认触发问题的 MIX 组合。下文已摘录关键证据，无需访问原现场文件，也不依赖用户提供正常版本、Git diff 或已知修复。

```text
<用户提供的现场目录>
├── 运行日志、debug/run plog、dump
├── sk_meta/<PID>/<model>/
│   ├── super_kernel.log
│   ├── sk_fused_nodes.log
│   ├── sk_task_queue.json
│   └── bin_files/
├── static_kernel_compile_outputs/
└── 本次编译信息
```

本案例包含两个证据范围：原始多算子模型用于描述问题入口和缩小范围；后续单算子最小实验用于确认子算子内的等待指令，以及 MIX_1_1 进入 MIX_1_2 的触发条件。两者分开记录，不将最小实验的结果直接当作原模型所有卡死的结论。以下地址、核数和 flag ID 均为对应实验的观测值。

## 2. 从产物判断卡死位置，再缩小到子算子

先按 PID、时间、设备、model 和 scope 关联 `sk_meta`、plog、静态编译产物与实际二进制，确认异常发生在融合后的设备执行阶段。

在 SK 异常信息中搜索：

```text
[PrintExceptionDumpInfoArray] kernelName
```

结合 C/V 核符号和 PC 映射，判断观察到的卡死位置：C、V 核均落在 SK 框架时，先检查框架执行及调度；有核明确落入子算子时，转入该子算子的同步分析。`No sub kernel matched` 只表示没有匹配到子算子，单独出现时不足以证明 PC 必在框架中；还需结合框架地址范围或反汇编。

原始多算子现场中，采样到的 C/V 核停在 `sk_entry` 的 `WAIT_FLAG_DEV`，并报告 `No sub kernel matched`。此时只能确认观察到的是框架等待，尚不能归因到某个 QuantBatchMatmulV3 内部指令。

为缩小范围，优先使用 `debug_per_op_max_core_num=1` 排查算子核数、核间同步及 SK 适配的影响，检查其实际生效结果和自动联动的核间检查配置。该选项不是“只使用 1 个核”，参见 [SuperKernel 调试选项说明](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md#table2)。

拆分后的融合成员、任务队列和静态编译信息将排查目标收敛到 QuantBatchMatmulV3。原现场相关字段整理为：

```text
容器类型：MIX_1_2，numBlocks=24
子算子类型：MIX_1_1，taskRatio=[1,1]，numBlocks=8
needMixKernelSplit=1
isScheModeOn=0
```

这些字段提示需要检查 MIX_1_1 子算子进入 MIX_1_2 的适配，但比例不同本身还不是错误证据。容器的 MIX block 数与子任务的 block 数也不能混为同一个启动范围。

原模型缺少完整原始静态产物，尚不能闭合其具体故障指令。于是根据已观察到的算子和 MIX 组合，构建单算子最小实验继续验证；这是定位过程中设计的实验，不是用户预先提供的正常用例。

## 3. 通过 PC 和反汇编定位到 QBMMv3 子算子

后续构建的最小模型只有一个 QuantBatchMatmulV3。先确认静态编译成功、SK entry 实际生成并执行，避免将静态编译失败后的回退误报为 SK 通过。

从该次异常产物中取得实际执行的 QBMM 二进制，使用匹配的 CANN 反汇编工具。以下命令中的环境和文件路径按现场替换：

```bash
"$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/llvm-objdump" --version
"$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/llvm-objdump" -d /path/to/runtime_qbmm.o
```

一组有效异常 PC 的映射为：

```text
采样核         = AIV core 45
CurrentPC      = 0x12c044323430
QBMM ELF 基址  = 0x12c04430d000
代码偏移        = 0x16430
入口变体        = AIV middle_split2
对应指令        = WAIT_FLAG_DEV #15
```

融合成员、函数符号、静态编译产物和 PC 映射共同确认：**卡死位置在 QuantBatchMatmulV3 子算子内，属于核间同步等待。** 外层任务名为 `sk_entry_*`，不能据此把该次异常归到框架侧。

此时已定位到具体算子及同步类型，下一步核对该算子进入 SK 后的 MIX 类型和执行范围。

## 4. 确认 MIX_1_1 进入 SK MIX_1_2 是触发条件

异常组的任务队列和融合信息记录：

```text
SK 容器：MIX_1_2
QBMMv3 子算子：MIX_1_1
taskRatio=[1,1]
needMixKernelSplit=1
```

这些信息将排查方向收敛到“1:1 算子进入 SK 1:2”的适配问题。仅有比例不同还不足以确认问题，因此设计容器比例对照实验。

保持输入值、shape、dtype、stride、子算子 block 数和实验框架库一致，通过实验开关选择容器，得到：

| 容器 | 子算子 | 调试配置 | 结果 |
| --- | --- | --- | --- |
| MIX_1_1，16 AIC / 16 AIV | MIX_1_1，16 个 block | per-op、sync-all、trace、cross-check 均关闭 | 执行通过，最大绝对误差 0 |
| MIX_1_2，16 AIC / 32 AIV | MIX_1_1，16 个 block | 同上 | 设备同步超时，无有效输出 |

选择容器的开关属于本案例的实验框架修改，不是 CANN 公开配置。两组对照是定位后设计的验证，不要求用户预先知道正常版本。

该实验表明：在这个最小现场中，同一个 MIX_1_1 QBMMv3 在 MIX_1_1 容器内执行正常，进入 MIX_1_2 容器后卡死；关闭显式 debug 选项后仍能触发，不能将其简单归因为 per-op 调试选项。

## 5. 定位结论与边界

**定位结论：** 问题已收敛到 QuantBatchMatmulV3 子算子，其 MIX_1_1 变体进入 SK MIX_1_2 场景时出现核间同步卡死。依据是子算子内部的异常 PC、实际 MIX 配置，以及只改变容器比例的对照结果。

本案例到此完成定位，具体实现层的缺陷及修复不在本案例中展开。不能仅凭 `WAIT_FLAG_DEV #15` 判断是哪处通知或等待实现有误。

原始多算子模型尚缺完整原始静态产物，以上最小实验不能证明原模型所有卡死均由同一原因造成，也不能推广到所有 QBMMv3 tiling、布局、SoC 或 CANN 版本。

## 6. 可复用的定位方法

先从产物判断观察到的卡死在框架还是子算子，通过调试选项缩小范围，再用实际 PC 和反汇编确认具体算子与同步类型。随后核对容器和子算子的 MIX 类型，用控制变量实验确认 1:1 进入 1:2 是否触发问题。

区分两层证据：PC 映射回答“卡在哪个算子”，容器比例对照回答“什么场景触发”。仅看到 MIX 类型不同，不能直接认定算子适配有问题。
