---
name: superkernel-debug-analysis
description: 用于分析使能昇腾 CANN SuperKernel 后出现的问题，支持用户已明确问题类型或仅提供异常现场的情况。基于 sk_meta、编译产物、plog、profiling 和相关源码判断问题类型并定位原因，覆盖静态编译、卡死、AiCore 超时、融合断图、精度和性能问题。
---

# SuperKernel 问题定位

面向使能 SK 后遇到问题的用户。用户已明确问题类型时，优先进入对应定位分支，并结合本次运行产物核实；问题类型尚不明确时，根据用户描述和产物完成分类，再选择定位路径。使能 SK 后出现异常不等于根因已经确定在 SK 框架。

输入路径由用户提供或从当前工作区发现，不依赖本 skill 的安装位置、固定源码仓库或示例用例。

## 适用问题与所需产物

问题类型已明确时，按下表获取对应产物；尚不明确时，先查看已有问题现场，优先获取 `sk_meta`、`static_kernel_compile_outputs` 和 `log/plog`，分类后再按需补采。

| 问题类型 | 优先产物 |
|---|---|
| 静态编译问题 | `sk_meta`、`static_kernel_compile_outputs` |
| SK 卡死问题 | `sk_meta`、`static_kernel_compile_outputs`、`plog` |
| SK 融合断图问题 | `sk_meta`、`static_kernel_compile_outputs` |
| 精度问题 | `sk_meta`、`static_kernel_compile_outputs`、`plog`、`profiling` |
| 性能劣化问题 | 未开 SK 的 profiling、包含 SK profiling 的 `sk_meta`、开启 SK 的 profiling |

缺少某类产物时，继续分析已有材料，并明确缺失项会阻塞哪个判断。

## 产物采集

需要补采或重新运行时，先保存已有现场，再按现场环境设置相应选项。

### 设置环境变量

```bash
export ASCEND_PROCESS_LOG_PATH=./log       # 指定 plog 保存路径
export ASCEND_GLOBAL_LOG_LEVEL=0           # 保留 static_kernel_compile_outputs
export ASCEND_OP_COMPILE_SAVE_KERNEL_META=1 # 收集 sk_meta

# 算子融入 SK 后性能劣化、需要 SK profiling 时按需开启；值的单位和支持范围以对应版本为准
# export ASCEND_PROF_SK_ON=64
# 需要通过 PTA 侧日志分析静态编译指令等问题时按需开启
# export TORCH_COMPILE_DEBUG=1
# plog 因日志轮转而缺失时，增加保留文件数，最大 1000
# export ASCEND_HOST_LOG_FILE_NUM=1000
```

### 运行与打包

执行用户提供的实际运行命令，运行结束后在产物所在目录打包：

```bash
tar -zcvf "output-$(date +%Y%m%d-%H%M).tar.gz" ./sk_meta ./static_kernel_compile_outputs ./log
```

打包路径按现场实际目录调整。其他产物按需加入，例如 profiling，以及寄存器相关定位所需的 `atrace`等。

### 设备侧诊断配置

需要检查或调整 singlecommit、设备日志级别时，可使用以下命令。`x` 替换为目标设备 ID；配置修改按定位需要和已有授权执行，不作为每次采集的默认操作。

```bash
# 查询 signal 和日志状态
msnpureport config --get -d x
# 开启 singlecommit
msnpureport config --set -d x --singlecommit 1
# 关闭 singlecommit
msnpureport config --set -d x --singlecommit 0
# 将目标设备的全局日志级别设为 error
msnpureport -g error -d x
```

## 通用定位流程

开始前，如果用户没有提供运行产物，先询问问题现场或产物根目录路径；问题类型已明确时，按上表询问对应产物路径。如果只提供了部分产物，继续分析已有材料，并说明缺失项对判断的影响。

1. **核对实际运行产物与静态编译产物是否一致。** 检查本次运行对应的 `sk_meta` 下 `bin_files` 与本次静态编译产物是否一致；如果不一致，优先按 [静态编译问题](references/static-compile.md) 分析。
2. **确定当前问题类型。** 用户已明确类型时，优先读取对应参考文件，并结合产物核实；尚未明确时，从用户描述和产物中识别报错或异常，判断应进入静态编译、卡死、融合断图、精度或性能分支。分类依据不足时，说明尚不能分类及需要补充的证据，不要求用户自行判定。后续步骤按判断结果选择执行。
3. **分析 SK 卡死问题。** 先判断卡死发生在框架侧还是子算子侧：框架卡死优先使用各类 debug 选项定位；子算子卡死根据反汇编结果逐项排查。具体步骤见 [SK卡死问题分析](references/hang.md)。
4. **分析 SK 算子融合断图问题。** 出现融合范围被拆分或预期节点未融合时，按 [融合断图](references/graph-break.md) 中的步骤排查断图原因。
5. **分析精度问题。** 出现开启 SK 后输出异常或精度不符合预期时，按 [精度问题](references/precision.md) 中的步骤定位。
6. **分析性能劣化或收益不达预期问题。** 按 [性能问题](references/performance.md) 中的步骤，将性能问题精确归因到对应的 SK 融合算子。

案例位于 `references/cases/`，只有在当前问题与某个案例的现象和证据路径确实相近时才读取；案例用于补充分析方法，不能替代当前运行产物、当前版本源码和当前二进制的证据。

## 输出要求

报告至少包含：

1. 异常发生阶段、位置和当前定位程度；
2. 支持关键判断的产物证据；
3. 候选原因及支持、反对证据；
4. 证据缺口和补采方式；
5. 下一步最小验证及结果解释。

只有在排除主要替代解释并完成针对性验证后，才称为“已确认根因”；仅凭超时、通用错误码或调试选项改变现象，只能确认现象或缩小范围。
