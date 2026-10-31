# 案例：Process 返回后的重复 MTE2→V 等待

## 1. 问题入口与现场材料

用户提供一个自定义 Ascend C `add_custom` 算子的运行失败现场：通过 `<<<>>>` 启动核函数，使用 `SK_BIND` 提供 SK 子函数入口，在开启 SuperKernel 后运行失败。

分析从当前问题产物开始，不预设同步类型或根因，不依赖正常版本、Git diff 或历史对照。下文已摘录定位所需的关键证据，无需访问原现场文件；文件名仅用于说明证据来源。

```text
<用户提供的现场目录>
├── 运行日志、plog
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

地址、核数和模型等信息是本案例的观测值。分析其他现场时，应重新关联该次运行，不能将这些值作为固定搜索条件。

## 2. 从运行产物确定问题类型

按 PID、时间、设备和模型关联运行日志、plog 和 `sk_meta`。日志显示模型完成融合后进入设备执行，随后多次报告等待超时，最终发生 AIV 设备异常。主机侧在 `torch.npu.synchronize()` 报告 `507011`，尚未完成输出正确性检查。

从融合记录与任务队列整理得到：

| 项目 | 现场记录 |
| --- | --- |
| 环境 | CANN `9.3.0`、Ascend910B4 |
| 模型、scope | `model_49_1`、scope `3` / `custom_add_sk` |
| 故障任务 | stream `292`、task `3`、`sk_entry_aiv_op_trace` |
| 融合成员 | 仅 `add_custom`，原图 nodeId `3` |
| 子算子 | `AIV_ONLY`、11 个 AIV block、`isScheModeOn=1` |
| SK 执行范围 | 0 AIC、11 AIV，FUNC 子任务同样为 11 个 block |

`bin_files` 中已取得自定义算子和 SK 入口二进制。该算子采用 `<<<>>>` / `SK_BIND` 路径，不能仅凭静态编译输出目录为空或缺少 `static_kernel_` 前缀，判断为 ACLNN 静态编译问题。

据执行阶段的等待和异常记录，进入 SK 卡死分析分支。

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

异常 AIV core 37 的 PC 匹配到 `add_custom` 的 `entry[2]`，符号为：

```text
_Z13add_custom_skILj2EEvPK13AddCustomArgsP18SkSystemArgsCompat
```

因此转入子算子侧排查，不能仅凭外层任务名中的 `sk_entry_*` 归责框架。

SK 日志还记录：

```text
OptionName: debug_per_op_max_core_num, set value: 1
OptionName: debug_cross_core_sync_check, set value: 1
[DEBUG_PER_OP_MAX_CORE_NUM] auto-enabled DEBUG_CROSS_CORE_SYNC_CHECK
```

本次已经开启该调试选项且仍然失败，应继续检查子算子同步。

## 4. 反汇编实际运行二进制

从本次 `bin_files` 取得实际运行的 `add_custom` 二进制，使用匹配的 CANN 工具。以下命令中的环境和目标路径应替换为当前现场的实际值：

```bash
"$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/llvm-objdump" --version
"$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/llvm-objdump" -d /path/to/runtime_kernel.o
```

plog 记录的修正基址和异常 PC 可换算为：

```text
CurrentPC       = 0x12c04140090c
fixedStartPC    = 0x12c041400000
代码偏移         = 0x90c
entry[2] 偏移   = 0x7c0
函数内偏移       = 0x14c
```

实际反汇编片段为：

```text
904: MOV X0, #0
908: WAIT_FLAG.MTE2.V X0
90c: BAR.ALL
910: BAR.ALL
914: MOV X0, #3585
918: SET_CROSS_CORE.MTE3 X0
91c: WAIT_FLAG_DEV #14
920: RET
```

**异常 PC 直接对应的是 `BAR.ALL`，不是前面的 `WAIT_FLAG.MTE2.V`，也不是后面的 `WAIT_FLAG_DEV #14`。** 按核内同步方向继续排查，并保留整个相邻同步序列，不能把附近出现跨核等待指令当作本次已经卡在跨核等待的证据。

## 5. 关联当前源码，检查事件生产与消费

SK 子函数的尾部结构为：

```cpp
op.Process();
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
AscendC::PipeBarrier<PIPE_ALL>();
AscendC::SyncAll<true>();
```

继续查看 `KernelAdd::Process()`，其循环中在数据搬运后已经执行以下配对。这里只摘录相关部分：

```cpp
AscendC::DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
AscendC::DataCopyPad(yLocal, yGm_[offset], copyParams, padParams);

AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
```

沿执行路径检查可见：

1. `Process()` 内部每轮对 `MTE2_V/EVENT_ID0` 置位并等待。
2. 最后一轮的等待也在函数内部完成。
3. `Process()` 返回后，又等待同一方向、同一 ID 的事件，中间没有新的对应 `SetFlag`。
4. 实际二进制中同样可见尾部额外的 `WAIT_FLAG.MTE2.V`，紧接现场 PC 对应的 `BAR.ALL`。

因此，将函数返回后的重复等待列为高置信度原因：子算子尾部没有对应的新事件生产者，核内事件配对不完整。

## 6. 定位结论与待执行验证

**已确认：** 异常位于 `add_custom_sk<2> + 0x14c` 的 `BAR.ALL`；其前方存在尾部 `WAIT_FLAG.MTE2.V`，当前源码在 `Process()` 已完成事件配对后再次等待同一事件。

**高置信度原因：** `Process()` 返回后额外执行 `WaitFlag<MTE2_V>(EVENT_ID0)`，没有新的对应置位，导致核内同步异常，与尾部同步序列附近的设备卡死现场吻合。

最小验证是保存现场后，仅删除 `Process()` 返回后的额外等待，得到以下待验证的尾部结构：

```cpp
op.Process();
AscendC::PipeBarrier<PIPE_ALL>();
AscendC::SyncAll<true>();
```

保持 `Process()` 内部事件配对、输入、block 数和 SK/debug 配置不变，重新构建并加载扩展，检查：

1. 新运行二进制中的尾部额外等待是否已移除。
2. 任务是否执行完成，不再出现本次设备超时。
3. 输出是否通过正确性检查。

## 7. 可复用的定位方法

从产物中确认子算子和异常 PC，检查 PC 前后的完整同步序列，再沿当前源码的函数调用和返回路径核对事件生产者、消费者及 ID。

尤其要区分两个判断：PC 对应哪条指令是定位事实；哪处事件不匹配导致无法继续，是需要源码证据和实验支持的故障解释。函数内部已经等待过的事件，返回后再次等待时必须检查是否有新的对应置位。
