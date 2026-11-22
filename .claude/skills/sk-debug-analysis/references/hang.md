# SK 卡死问题分析

SK 卡死问题优先结合 `sk_meta`、`static_kernel_compile_outputs` 和 plog 分析。纯卡死需要保留足够的模型超时时间，否则可能无法产生 AiCore 超时现场，建议等待时间为 20 分钟以上。

## 1. 常见原因

SK 卡死通常与算子没有正确适配 SK 有关，优先检查以下方向：

1. 使用全核同步但没有配置匹配的 `schedule_mode`；
2. `MIX_1_1` 算子没有自动适配 `MIX_1_2` 场景，参考[SuperKernel 算子适配要求](https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/guide/programming_guide/advanced_programming/super_kernel/operator_adaptation.md)；
3. 算子核间同步异常，例如 `CrossCoreSetFlag` 冗余或 flag ID 错误，参考[CrossCoreSetFlag 文档](https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreSetFlag_ISASI.md)；
4. 算子核内同步异常，例如 `SetFlag/WaitFlag` 冗余或不匹配，参考[SetFlag/WaitFlag 文档](https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/SIMD-API/basic_api/sync_control/intra_core_sync/SetFlag_WaitFlag_ISASI.md)。

算子同步异常在 SK 场景下不一定表现为当前算子卡死，也可能表现为其他算子卡死、精度异常或框架卡死。

## 2. 判断卡死类型

SuperKernel 主要判断由特性引起的 AiCore Error。

如果是其他核先发生非 timeout 问题并引出 timeout，通常会较快报告，应先按具体报错类型处理。

通过 `sk_meta` 和异常日志判断卡死发生在 SK 框架还是子算子。搜索：

```text
[PrintExceptionDumpInfoArray] kernelName
```

判断规则：

- 参与执行的核均停在 SK 框架中时，判断为框架侧卡死；
- 若有核停在对应子算子中，则转入该子算子侧排查。

## 3. 框架卡死

### 3.1 使用 debug 选项缩小范围

优先使用 [`debug_per_op_max_core_num`](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md#table2)，排查算子核间同步和算子使能 SK 对现象的影响。

如果卡死现象消失，说明问题与框架融合、调度或相关 debug 路径有关，应继续交给对应框架组件排查。

### 3.2 仍然卡死时检查指令

如果仍然卡死，转入算子同步排查。先使用实际运行二进制进行反汇编，再根据卡死指令分类。

### 3.3 反汇编方法

1. 从 `sk_meta/<PID>/<model>/bin_files` 选择与异常任务、entry 和 PID 对应的 `.o`，不要直接使用其他目录中的同名文件。
2. 使用与该二进制的 SoC/CANN 环境匹配的工具。CANN 环境通常提供以下工具，先检查帮助和版本，再选择能识别目标 ELF 的一个：

   ```bash
   "$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/llvm-objdump" --version
   "$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/llvm-objdump" --help
   "$ASCEND_HOME_PATH/tools/msobjdump/npu-objdump" --help
   ```

3. 对目标 `.o` 反汇编并保存输出；常用形式如下，具体选项以本地工具帮助为准：

   ```bash
   llvm-objdump -d /path/to/kernel.o > /path/to/kernel.o.dis
   # 或使用匹配版本的 npu-objdump
   npu-objdump -d /path/to/kernel.o > /path/to/kernel.o.dis
   ```

4. 将异常日志中的 `CurrentPC` 减去日志记录的加载基址或 `fixedStartPC`，得到代码偏移；再用符号表和反汇编输出定位函数、函数内偏移及相邻指令。不能把设备绝对 PC 直接当作文件偏移。
5. 记录二进制路径、SHA256、工具路径、工具版本、完整命令、加载基址、PC 换算结果和反汇编片段，保证定位可复核。

如果工具无法识别 ELF、架构与环境不匹配或 PC 无法换算，应先报告映射缺口，不要根据指令名称猜测卡死类型。

随后按指令分类：

- `WAIT_FLAG.X.X` 或 `BAR.ALL`：核内同步方向，排查算子核内同步；
- `WAIT_FLAG_DEV.X` 或 `WAIT_INTRA_BLOCKI.X`：核间同步方向，排查算子核间同步；
- 其他指令：保留现场，交给对应开发人员继续定位。

## 4. 子算子卡死

### 4.1 先使用 debug 选项

优先使用 [`debug_per_op_max_core_num`](https://gitcode.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/advanced/superkernel.md#table2)，排查算子核间同步影响。

如果属于核间同步问题，优先检查：

1. 使用全核同步但没有配置 `schedule_mode`；
2. `MIX_1_1` 算子没有自动适配 `MIX_1_2` 场景。

如果开启该选项后不再卡死，说明触发条件发生了变化，还需要结合运行配置和算子同步实现继续判断，不能直接据此确认根因。

### 4.2 通过反汇编分类

对实际子算子二进制按第 3.3 节的方法反汇编：

- `WAIT_FLAG.X.X` 或 `BAR.ALL`：核内同步卡死。可以通过 `scope None` 排除前序算子，逐步缩小范围，直到问题消失，从而定位可能存在核内同步冗余的算子。重点检查 `SetFlag/WaitFlag` 是否匹配使用；不匹配可能引发未定义行为。
- `WAIT_FLAG_DEV.X` 或 `WAIT_INTRA_BLOCKI.X`：核间同步卡死。重点检查核间同步接口、参与核和 flag 配对。
- 其他指令：保留反汇编、PC 和运行现场，交给对应开发人员继续定位。

如果卡在 C 核，可以优先排除融合范围内位于该算子之前的 MIX/C 算子；这只是缩小范围的方法，不能据此排除其他算子。后续如需使用 msSanitizer，应先确认当前版本是否支持对应同步检查能力。

## 5. 结论要求

报告中区分：

- 卡死类型：框架侧或子算子侧；
- 卡死指令：核内同步、核间同步或其他指令；
- 当前排查程度：现象已确认、位置已确认、候选原因或根因已验证。

仅有 timeout 或 debug 后现象消失时，不要直接写成最终根因。
