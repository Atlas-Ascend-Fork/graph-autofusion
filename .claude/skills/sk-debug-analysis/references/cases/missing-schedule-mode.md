# 案例：全核同步但未配置 schedule mode

## 1. 问题入口与现场材料

用户提供一个自定义 Ascend C `add_custom` 算子的运行失败现场：通过 `<<<>>>` 启动核函数，使用 `SK_BIND` 提供 SK 子函数入口，在开启 SuperKernel 后发生设备等待。

开始时只知道运行失败，不预设同步类型或缺少何种声明。分析使用现场产物和当前问题源码，不依赖正常版本、Git diff 或历史对照。

本案例已保留关键证据摘录，阅读时无需访问原现场文件。下文的文件名用于说明证据来源；分析其他问题时，应从用户当前现场获取对应材料。

```text
<用户提供的现场目录>
├── 运行日志、plog
├── sk_meta/<PID>/<model>/
│   ├── super_kernel.log
│   ├── sk_fused_nodes.log
│   ├── sk_task_queue.json
│   ├── sk_mdl_updated.json
│   └── bin_files/
│       ├── add_custom_*.o
│       └── sk_entry_*.o
├── static_kernel_compile_outputs/
└── 当前算子源码
```

下文的地址、核数和配置均为本案例的观测值，不能作为其他现场的预设条件。

## 2. 从运行产物确定问题类型

先按 PID、时间、设备和模型关联同一次运行。plog 显示融合和模型更新已完成，随后设备同步持续等待，最终报告设备异常；运行日志出现 `507011` 和 `timeout or trap error`，没有输出校验成功记录。

实际 `bin_files` 中存在自定义算子和 SK 框架二进制，SK 日志记录了绑定表和四个 SK 子函数入口的成功解析。静态编译输出目录虽然为空，但目标算子是自定义 `<<<>>>` / `SK_BIND` 路径，不能据此判定为 ACLNN 静态编译失败或未命中。

因此进入 SK 卡死分析分支，继续定位设备执行中的等待位置。

## 3. 判断卡死发生在框架侧还是子算子侧

在 SK 异常信息中搜索：

```text
[PrintExceptionDumpInfoArray] kernelName
```

关键字段为：

```text
coreType   : AIV
kernelName : add_custom
```

PC 匹配记录进一步确认异常位置属于 `add_custom` 的 `entry[1]`，即：

```text
_Z13add_custom_skILj1EEvPK13AddCustomArgsP18SkSystemArgsCompat
```

据此进入子算子侧排查。外层故障任务名为 `sk_entry_mix12_op_trace`，不能仅凭外层任务名判断为框架卡死。

同时检查实际生效的调试配置，日志摘录为：

```text
OptionName: debug_per_op_max_core_num, set value: 1
OptionName: debug_cross_core_sync_check, set value: 1
[DEBUG_PER_OP_MAX_CORE_NUM] auto-enabled DEBUG_CROSS_CORE_SYNC_CHECK
```

这说明当前是调试选项已开启后仍失败的现场。应继续分析已有产物，不能把再次设置相同选项算作新验证，也不能据此推断关闭选项后的行为。

## 4. 反汇编实际二进制，定位等待指令

从本次 `sk_meta` 的 `bin_files` 选择实际运行的 `add_custom` 二进制，使用与其匹配的 CANN 反汇编工具。以下命令中的环境和文件路径按现场替换：

```bash
"$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/llvm-objdump" --version
"$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/llvm-objdump" -d /path/to/runtime_kernel.o
```

将异常 PC 与日志中的子算子加载基址、符号入口关联，得到：

```text
CurrentPC       = 0x12c0414006a0
加载基址         = 0x12c041400000
代码偏移         = 0x6a0
entry[1] 偏移   = 0x54c
函数内偏移       = 0x154
```

对应反汇编片段为：

```text
690: BAR.ALL
694: BAR.ALL
698: MOV X0, #3585
69c: SET_CROSS_CORE.MTE3 X0
6a0: WAIT_FLAG_DEV #14
6a4: RET
```

至此可确认异常 PC 位于 `add_custom_sk<1> + 0x154` 的核间等待指令。仅看产物，还不能确定为什么该轮同步无法完成。

## 5. 从调度配置和核数形成候选假设

从融合日志、任务队列和更新后的模型中提取以下信息。表内为相关字段的整理，不是完整文件：

| 证据来源 | 关键记录 |
| --- | --- |
| 子算子记录 | `funcName:add_custom`、`kernelType:AIV_ONLY`、`numBlocks:11`、`isScheModeOn:0` |
| SK 调度记录 | `entryType=MIX_1_2`、`isScheModeOn=0` |
| SK launch 配置 | `isScheModeOn=0`、`attrCount=0` |
| 任务队列 AIC 范围 | `numBlocks=20` |
| 任务队列 AIV 范围 | `numBlocks=40` |
| AIV 的 FUNC 子任务 | `numBlocks=11` |

外层 SK 的 20 个 MIX block 对应 20 AIC、40 AIV；子算子实际任务只有 11 个 AIV block。结合核间等待指令，可将“同步参与范围与调度范围不匹配”列为候选，但不能仅凭 11 与 40 不同就确认错误。

## 6. 关联当前源码，解释同步范围不匹配

### 6.1 检查入口声明和 SK 子函数

当前问题源码的入口声明没有配置调度模式，声明关键部分为：

```cpp
extern "C" __global__ __vector__ void add_custom(...)
```

对应 SK 子函数在处理结束后仍执行：

```cpp
op.Process();
AscendC::PipeBarrier<PIPE_ALL>();
AscendC::SyncAll<true>();
```

这与日志中的调度属性关闭相符。匹配的 CANN 同步实现使用 `SYNC_AIV_ONLY_ALL=14`，相关路径为：

```cpp
ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_AIV_ONLY_ALL));
wait_flag_dev(SYNC_AIV_ONLY_ALL);
```

该实现与实际二进制中的发送指令和 `WAIT_FLAG_DEV #14` 对应。

### 6.2 检查 SK 如何选择执行范围

在与运行版本对应的 SK 实现中，是否要求核数精确匹配的条件为：

```cpp
bool RequiresExactCoreMatch() const {
    return IsScheModeOn() && !GetNodeInfos().kernelInfos.capBits.blockDimScaleUp;
}
```

本次 `IsScheModeOn()` 为 `false`，`blockDimScaleUp` 也为 `false`，因此未进入精确匹配路径。在当前 `debug_per_op_max_core_num` 配置下，框架选用了设备最大 MIX 范围，实际任务队列证实为 20 AIC、40 AIV。

设备端分发 FUNC 子任务的关键条件为：

```cpp
auto blockId = AscendC::GetBlockIdx();
if (blockId < task->numBlocks) {
    // 调用对应 SK 子函数
}
```

因此，只有 11 个 AIV 调用该子函数并执行全核同步，其余 29 个 AIV 跳过子函数。给子函数传入 `skNumBlocks=11` 用于任务计算，并不会自动将这里的硬件全核同步改成仅等待这 11 个参与者。

至此形成完整解释：入口未配置匹配的调度模式 → 未约束 SK 核数精确匹配 → 外层运行范围扩大 → 仅部分 AIV 执行全核同步 → 等待无法完成。

上述调度选择逻辑是本案例运行版本的实现，分析其他版本时需要重新核对，不能直接套用核数和条件。

## 7. 用核状态交叉验证，说明证据边界

全核状态记录中：

- 11 个 AIV 槽位仍在执行 `add_custom`。
- 29 个 AIV 槽位已有 SK 入口执行完成记录。

该分布与任务分发和同步参与范围的分析一致。但运行中槽位的 PC 为零，不能声称这 11 个核全部停在同一条指令；明确的指令定位来自有有效 PC 的异常核。

`507011` 仅说明执行失败传播至同步接口；`EE1014` 是设备异常后缺少 `RT_FUNCTION_TYPE_DFX_TYPE` 元信息引起的 dump 解析错误，影响采集，但不能解释此前的持续等待。

## 8. 定位结论与待执行验证

**已确认：** 设备执行阶段持续等待；异常 PC 位于 `add_custom` 的 `WAIT_FLAG_DEV #14`；调度属性为关闭；外层 AIV 范围与子任务范围分别为 40 和 11。

**高置信度原因：** 入口未配置与全核同步匹配的 schedule mode，导致 SK 调度范围与实际执行同步的核范围不一致。当前源码、运行配置、指令和核状态共同支持该判断。

**尚未完成：** 没有补充调度声明后的设备重跑结果，不能记为修复已验证。

根据当前源码和该版本的调度语义，提出最小验证：只为入口补充 `__schedmode__(1)`，保持输入、`BLOCK_NUM`、`SyncAll` 和 debug 配置不变，重新构建并运行。预期检查：

1. 运行日志记录 `isScheModeOn=1`，SK launch 设置对应调度属性。
2. 此例的纯 AIV、11-block 子算子取得精确的 11 AIV 执行范围。
3. 任务执行完成，不再发生该核间等待超时。
4. 输出通过正确性检查。

这是定位后提出的单变量实验，不是来自已知正常版本。上述结果尚未执行或观察到，具体接口支持仍应以现场芯片和 CANN 版本为准。

## 9. 可复用的定位方法

从运行失败现象出发，先定位到子算子中的核间等待，再关联入口属性、外层 SK 范围、子任务分发和同步实现，解释为什么等待所需的参与者没有到齐。

复用时应核对当前源码与实际运行二进制的关联，不以“源码没有某个声明”或“两个核数不同”单独下结论。是否修复仍需通过针对性实验和输出正确性检查验证。
