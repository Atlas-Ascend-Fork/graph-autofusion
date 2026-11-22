# 案例：条件执行全核同步导致参与核不完整

## 1. 问题入口与现场材料

用户提供一个运行失败的自定义 Ascend C 加法算子现场：通过 `add_custom<<<...>>>` 启动核函数，注册为 `torch.ops.ascendc_ops.add_custom`，并使用 `torch.compile(..., backend="npugraph_ex")` 开启静态核编译和 SuperKernel。

开始分析时，只有当前问题现场，不预设问题类型或根因。定位使用运行产物、实际二进制和当前源码，不要求正常版本、Git diff 或历史对照。

下面保留定位所需的证据摘录，阅读本案例无需访问原现场文件。文件名用于说明证据来源；复用方法时，应从用户当前现场获取对应材料，不能要求这些历史文件存在。

涉及的材料类型为：

```text
<用户提供的现场目录>
├── 运行日志
├── plog
├── sk_meta/<PID>/<model>/
│   ├── super_kernel.log
│   ├── sk_fused_nodes.log
│   ├── sk_task_queue.json
│   └── bin_files/
│       ├── add_custom_*.o
│       └── sk_entry_*.o
├── static_kernel_compile_outputs/
└── 当前算子源码
```

下文中的地址和配置是本案例的观测值，不是其他现场的搜索条件或预设值。

## 2. 从运行产物确定问题类型

### 2.1 关联本次运行

| 项目 | 本次运行记录 |
| --- | --- |
| 环境 | CANN `9.3.0`、Ascend910B4 |
| 模型、scope | `model_47_1`、scope `3` / `custom_add_sk` |
| 故障任务 | stream `156`、task `3`、`sk_entry_aiv_op_trace` |
| 融合成员 | 仅 `add_custom`，原图 nodeId `3` |
| 执行配置 | `AIV_ONLY`、11 个 AIV block、`isScheModeOn=1` |

先按 PID、时间和模型关联同一次运行。环境信息来自 plog，融合成员来自 `sk_fused_nodes.log`，执行核数和调度配置来自 `super_kernel.log` 与 `sk_task_queue.json`。关键字段摘录如下：

```text
funcName:add_custom, kernelType:AIV_ONLY, numBlocks:11,
cubeNum:0, vecNum:11, isScheModeOn:1, resolvedNum:4

Set SK launch cfg: nodeId=3, isScheModeOn=1
```

### 2.2 检查静态编译路径和失败阶段

静态编译输出目录为空，但 `bin_files` 中已经存在自定义算子和 SK 框架二进制。SK 日志记录了绑定表及四个 SK 子函数入口的成功解析。

目标算子采用 `<<<>>>` 直调加 `SK_BIND` 路径，因此不能仅凭编译目录为空或二进制缺少 `static_kernel_` 前缀，判断为 ACLNN 静态编译未命中。

plog 显示：融合后的模型已生成，随后在 14:48、14:51、14:54 连续报告等待未完成，14:54:24 收到设备异常。运行日志显示错误在执行模型后的 `torch.npu.synchronize()` 暴露。

```text
SynchronizeExecutedTask: report three minutes timeout!
...
An exception callback message is received.
```

由此进入 SK 卡死分析分支：已发生设备执行阶段的持续等待，尚未据通用错误码判断具体根因。

## 3. 判断卡死发生在框架侧还是子算子侧

在 `sk_meta` 中搜索：

```text
[PrintExceptionDumpInfoArray] kernelName
```

异常信息中的关键字段为：

```text
coreType   : AIV
kernelName : add_custom
```

PC 匹配记录进一步将异常 AIV core 20 定位到 `add_custom` 的 `entry[1]`，符号为：

```text
_Z13add_custom_skILj1EEvPK13AddCustomArgsP18SkSystemArgsCompat
```

因此转入子算子侧排查。外层任务名为 `sk_entry_aiv_op_trace`，并不表示阻塞指令就在 SK 框架中。

本次 SK 日志已记录以下设置：

```text
OptionName: debug_per_op_max_core_num, set value: 1
OptionName: debug_cross_core_sync_check, set value: 1
[DEBUG_PER_OP_MAX_CORE_NUM] auto-enabled DEBUG_CROSS_CORE_SYNC_CHECK
```

当前是这些选项开启后仍失败的现场，不需要将相同设置当作一次新实验，也没有据此证明关闭选项时的行为。

## 4. 反汇编实际运行二进制，判断同步类型

使用与实际运行二进制匹配的 CANN 工具。命令模板如下，环境变量和目标文件路径应替换为当前现场的实际值：

```bash
"$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/llvm-objdump" --version
"$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/llvm-objdump" -d /path/to/runtime_kernel.o
```

本次使用的工具报告 LLVM `15.0.5`，能够识别目标二进制的 `elf64-hiipu` 格式。

plog 中记录的 PC、修正基址及据此计算的偏移为：

```text
CurrentPC       = 0x12c0414006d8
fixedStartPC    = 0x12c041400000
代码偏移         = 0x6d8
entry[1] 偏移   = 0x568
函数内偏移       = 0x170
```

这里使用的是子算子的修正基址，不能直接减去异常中最初报告的 `StartPC=0x12c04141ec00`。

PC 附近的实际反汇编为：

```text
6ac: BAR.ALL
6b0: MOV X0, BLOCKID
6b4: SIGNEXT.s32 X0, X0
6b8: MOV X1, SUBBLOCKDIM
6bc: MUL.s64 X0, X0, X1
6c0: MOV X1, SUBBLOCKID
6c4: NEG.s64 X1, X1
6c8: JUMPCMP.EQ.s64 #1756, X0, X1  ; 目标 0x6dc
6cc: BAR.ALL
6d0: MOV X0, #3585
6d4: SET_CROSS_CORE.MTE3 X0
6d8: WAIT_FLAG_DEV #14
6dc: RET
```

异常 PC 正好对应 `WAIT_FLAG_DEV #14`，按核间同步方向排查。其前面的分支允许部分 block 绕过发送和等待指令，直接返回，需要进一步关联源码确认参与条件。

## 5. 读取当前源码，检查同步参与者

当前算子源码中，SK 子函数的尾部为：

```cpp
op.Process();
AscendC::PipeBarrier<PIPE_ALL>();
if (AscendC::GetBlockIdx() != 0) {
    AscendC::SyncAll<true>();
}
```

检查与本次运行匹配的 CANN 包中 `SyncAllImpl<true>()` 实现，确认 `SYNC_AIV_ONLY_ALL=14`，对应执行路径包含：

```cpp
ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_AIV_ONLY_ALL));
wait_flag_dev(SYNC_AIV_ONLY_ALL);
```

它与异常位置的发送和等待指令一致。结合源码与反汇编可见：

1. 本次 SK 和子算子均按 11 个 AIV block 执行。
2. 逻辑 block 0 在条件分支中跳过全核同步，不发送这次同步所需的通知。
3. 其余 10 个 block 执行全核同步，无法等齐该轮所需的参与者。

为核对构建产物与运行二进制，将归档 ELF 的整个 `.text` 与构建扩展 `.so` 中的代码比较，确认 3304 字节逐字节一致。源码尾部条件也与实际反汇编控制流吻合。

## 6. 用核状态作旁证，排除误判

SK 日志的全核状态表中，11 个有执行记录的 AIV 槽位表现为：

- 1 个 `opState=4`，记录 SK 入口执行完成。
- 10 个 `opState=2`，记录仍在执行 `add_custom`。

该分布与一个 block 跳过同步、其余 block 等待的机制吻合。但这些运行中槽位的 PC 为零，不能据此断言它们全部停在 `0x6d8`；异常核编号也不能直接当作逻辑 block ID。

同时核对以下线索：

- 入口已有 `__schedmode__(1)`，日志中的子算子与 SK launch 均为 `isScheModeOn=1`。本次证据指向同步参与条件，应继续检查子算子实现。
- `507011` 是执行失败向同步接口传播的结果，不能单独解释为什么等待。
- `EE1014` 记录的是设备异常后的 dump 元信息解析失败，时间晚于持续等待，不能作为此前等待的原因。

## 7. 定位结论与待执行验证

**已确认的位置和类型：** `add_custom_sk<1> + 0x170` 的 `WAIT_FLAG_DEV #14`，属于子算子侧核间同步等待。

**高置信度原因：** SK 子函数对 `SyncAll<true>()` 使用了排除逻辑 block 0 的条件，导致参与全核同步的 block 不完整。源码控制流、实际指令、11-block 配置和核状态分布共同支持该判断。

**尚未完成：** 没有修改源码、重新构建或执行设备验证，因此不能记为“修复已验证”。

最小验证方案是保存现场后，仅去掉包围同步的条件，让所有参与 block 执行：

```cpp
op.Process();
AscendC::PipeBarrier<PIPE_ALL>();
AscendC::SyncAll<true>();
```

保持输入、11 个 block、调度模式和 SK/debug 配置不变，重新构建并加载扩展，检查新二进制是否生效、任务是否执行完成，以及现有 `torch.allclose` 正确性检查是否通过。此方案是定位后提出的验证，不是用户预先提供的正常版本。

## 8. 可复用的定位方法

从未分类的失败现场开始，依次完成：运行关联与执行路径检查 → 根据异常 `kernelName` 和 PC 区分框架与子算子 → 用实际运行二进制识别同步类型 → 检查当前源码中的参与条件 → 用核状态交叉验证 → 提出单变量实验。

遇到核间等待时，既要核对调度模式和执行核数，也要检查是否有核通过分支或提前返回绕过了同一轮同步；每个判断仍需由当前现场证据支持。
