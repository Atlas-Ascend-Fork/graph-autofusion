# 案例：SwigluGroupQuant 算子精度问题定位

## 1. 问题入口与证据范围

Recipe DeepSeek V4.1 极低时延模型在一次代码改动后出现模型吐字异常：关闭 SuperKernel 正常，开启 SuperKernel 异常。原模型现场报告为 A5、CANN 9.2.0 930 版本；后续使用 custom op `custom::npu_swiglu_group_quant`（算子类型 `SwigluGroupQuant`）构建单算子实验，在 A5、CANN 9.3.0 上提高复现率并完成根因验证。

本案例分为两层证据：模型级开关和 scope 二分用于锁定算子及触发阶段；单算子 replay、插桩、反汇编和二进制变体用于确认源码缺陷。合成输入不是 DeepSeek 真实 tensor，因此单算子修复结果不能代替原模型回归。

## 2. 从模型异常缩小到子算子 SK 编译

首先保持模型和输入不变，对比 SK 开关，确认异常只在开启 SK 时出现。随后按精度流程执行三项测试。

### 2.1 逐算子 SK 诊断

开启 `debug_per_op_max_core_num` 后原精度问题仍存在，同时 `FusedMatMul` 因核内同步多 `set` 少 `wait` 触发 trap。该 trap 是诊断选项暴露的独立问题：将该算子排除融合后继续二分，不能把执行失败当作原精度问题的原因。

### 2.2 加强子算子间同步

开启 `debug_sync_all` 后精度问题仍存在，说明加强 SK 子算子之间的同步没有消除异常。该结果不能排除子算子内部同步缺陷。

### 2.3 空 scope 与逐算子关闭 SK 子编译

继续二分 scope，直至 scope begin/end 之间不包含任何算子。目标算子在 scope 外执行普通入口时仍有精度问题，因此实际执行 SK 融合不是必要条件。

空 scope 仍保留 SK 子算子编译流程：普通版本、SK 子入口和绑定信息会进入合并产物。通过在编译入口按算子类型关闭 `ascendc_enable_super_kernel`，逐个排查候选算子，最终得到：

```text
SwigluGroupQuant 开启 SK 子算子编译：异常
仅关闭 SwigluGroupQuant SK 子算子编译：正常
```

这一步把问题缩小到 SwigluGroupQuant 的 SK 生成、合并产物或相关执行条件，尚未证明 SK 子入口本身出错。

## 3. 建立单算子 replay 精度反馈环

单算子输入元数据为：

| 项目 | 配置 |
|---|---|
| `x` | BF16、ND、`[21,640]` |
| `group_index` | int64、ND、`[128,2]` |
| 属性 | `group_list_type=2`、`quant_mode=1`、`dst_type=36`、`round_scale=true`、`clamp_limit=10.0` |
| 有效输出 | `y`、`scale_out`；`output_origin=false` 时不比较 `y_origin` |

测试使用多 seed、多个数值范围、常量和边界输入，连续 replay 不同输入。调用前将输出预填为哨兵，并保存每次输入、eager 参考和实际输出。

异常不是分散的数值误差，而是部分 `y` 和 `scale_out` 行完整保留前一次输入的对应行或哨兵。历史空 scope 运行出现 73/216 次异常；正式源码对照中原版再次出现 61/216，表明问题具有时序随机性，不能把某一次 `0/N` 当作永久安全。

## 4. 核对 scope、实际入口和二进制

scope 内与空 scope 两组最终 ELF 的完整哈希不同，但 `.text` 和 `.rodata` 一致；完整文件差异位于元数据、符号表和字符串表。执行路径不同：

```text
scope 内：更新图调用 sk_entry_aiv，再进入 middle/split 子入口
空 scope：更新图保留目标 __kernel0 普通任务
```

不融合时，runtime 发射的是普通 `__kernel0`，不是误发给 SK 子入口；但该普通入口位于 SK 合并 ELF 中。后续对普通静态包和 SK 合并包进一步确认：

```text
普通静态包：普通入口位于 0x0
SK 合并包：普通入口位于 0x5b3c
```

两条普通路径的计算代码及归约函数机器码相同，只有 tiling 地址重定位立即数字段随位置改变。runtime 加载的程序段、实际发射 PC 和 ELF 符号相符，没有证据支持“调用错入口”或“加载错二进制”。

## 5. 从整行旧值定位到 group sum

### 5.1 先将运行中的 kernel 映射回具体源码

在检查中间值前，先证明分析的是本次运行实际执行的 kernel，而不是源码目录中某个未被使用的副本。这里采用以下映射链：

```text
runtime 实际发射入口
  -> sk_meta/<pid>/<model>/bin_files 中的 SwigluGroupQuant ELF
  -> static_kernel_compile_outputs 中同一编译批次的 ELF 和算子 JSON
  -> tiling key=2100
  -> swiglu_group_quant.cpp 的 MX quant 分支
  -> SwigluMxQuantPerf::Init
  -> VFProcessGroupIndexPair
  -> swiglu_group_quant_base.h 中的两次 DataCopy
```

具体做法如下：

1. 从 runtime/plog 或更新图确认本次实际执行的目标算子、入口类型和安装路径；计算 `sk_meta/.../bin_files/` 中目标 `.o` 的 SHA-256，并与本次 `static_kernel_compile_outputs` 生成的目标 `.o` 比对。哈希不一致时，不能把后续源码分析与本次运行直接对应起来。
2. 读取同一编译批次的 `SwigluGroupQuant_*.json`。本用例的 `tiling_key=2100`，对应 `swiglu_group_quant.cpp` 中的 `MX_QUANT_TILING_KEY` 分支；因此执行的是 `SwigluMxQuantPerf<..., false>`，而不是 block quant、MXFP4 或 `y_origin` 分支。
3. 沿设备入口的源码调用链向下追踪：`swiglu_group_quant.cpp` 调用 `SwigluMxQuantPerf::Init`；当 `group_list_type=2` 时，`Init` 调用 `VFProcessGroupIndexPair` 处理 `[group_id,count]` 表。该函数定义在 `swiglu_group_quant_base.h`，这才是检查 group sum 写回的具体源码位置。
4. 如果编译产物保留了 `*_norm_kernel.cpp`、`norm.i` 或编译命令日志，可用它们核对本次预处理后的 include 和模板实例；它们用于确认编译时采用的源码副本。若这些中间文件缺失，仍需依靠 ELF 哈希、tiling JSON、源码分支和反汇编建立映射，不能仅凭目录名认定源码对应关系。

这一步的目标是确定“运行的哪一个 kernel、走源码中的哪一个模板分支、最终落到哪一个函数”。它本身还没有证明该函数有缺陷；只有完成映射后，才能把整行漏写现象与该函数产生的 group sum 联系起来。

### 5.2 从整行旧值反推 group sum

源码入口确定后，再根据错误形态选择检查位置。`y` 和 `scale_out` 同一行都保留旧值，优先考虑该行的计算或搬出没有执行，而不是先追查 SwiGLU 的浮点公式。沿 `SwigluMxQuantPerf::Process` 检查处理范围和提前返回，找到 `GetBlockIdx() >= usedCoreNums` 时直接返回的分支；再回到 `Init` 追踪 `usedCoreNums` 和每核行数的来源：

```text
groupSumLocal.GetValue(0)
  -> realBs
  -> rowOfFormerBlock / usedCoreNums / 每核循环范围
  -> Process 是否返回、哪些行会被写回
```

本用例前 21 个 count 为 1，其余为 0，正确 group sum 应为 21。由此将 `groupSumLocal` 选为插桩检查的候选控制量，再向上追踪其生产者 `VFProcessGroupIndexPair`。此时“sum 出错导致漏写”仍是假设，需要后续逐逻辑 block 的记录和单变量对照验证，不能仅凭整行旧值直接断定归约错误。

源码 `VFProcessGroupIndexPair` 对 `[group_id,count]` 表的 count 列求和，将结果作为后续有效行数计算的输入。该函数使用同一 UB 地址的两种类型视图：

```cpp
__local_mem__ int64_t* yLocalAddr =
    (__local_mem__ int64_t*)yLocal.GetPhyAddr();
__local_mem__ int32_t* yLocalAddr32 =
    (__local_mem__ int32_t*)yLocalAddr;
```

函数尾部连续执行：

```cpp
DataCopy(yLocalAddr, zeroReg64, pregMerge64);
DataCopy(yLocalAddr32, sum, pregMerge);
```

第一条从地址 `A` 写 8 字节零 `[A,A+8)`，第二条从同一地址写 4 字节归约和 `[A,A+4)`。期望最终得到低 32 位 `sum`、高 32 位零，但两次 vector store 之间没有 `VEC_STORE -> VEC_STORE` 顺序约束。

实际运行二进制中也存在两个同基址的连续 `VSTI`，中间没有 `SMEM_BAR.VST_VST`。后续插桩记录每个逻辑 block 的 group sum，异常行集合与读取到 `sum=0` 的逻辑 block 集合严格一致；延迟后再次读取仍为零。

`SwigluMxQuantPerf::Init` 使用该 sum 计算 `realBs`、各核处理范围和 `usedCoreNums`。错误读到零后，部分逻辑核提前返回，本次输出行没有写回；graph replay 复用输出缓冲区时便表现为保留前一次结果。

## 6. 单变量实验建立因果关系

围绕“重叠写缺少完成顺序”进行以下对照：

| 变体 | 结果 | 解释 |
|---|---:|---|
| 原始归约 VF | 可复现异常 | 保留原始问题 |
| 强制读取的 sum 为本用例正确值 21 | 0/40 | 确认错误 sum 是下游漏写的必要条件；不是通用修复 |
| 同地址、同指令数，插入普通 predicate 指令 | 13/40 | 仅增加指令或延迟不能消除异常 |
| 同地址、同指令数，换成 `SMEM_BAR.VST_VST` | 0/40 | 建立写写顺序后通过 |
| 第一条 64 位 store 也写入正确 sum，保留两次 store | 0/80 | 两次写的低 32 位不再冲突后通过 |

普通指令和屏障探针具有相同调用长度，完整 ELF 只在新增指令上不同。这一区分了“布局或延迟偶然规避”与“建立正确依赖”。

## 7. 布局、DCCI 和 VF 重排是触发因素，不是根因

SK 编译改变最终 ELF 中代码块的布局。只移动普通代码或归约 VF 的位置，可以改变异常概率；给普通入口增加 `aligned(512)` 或在合并阶段加入 `--HiIPU-disable-vf-reorder`，也能使已测普通入口路径不再复现。

但这些改动没有增加两次 store 的顺序约束。实际 SK 融合对照中，默认链接和禁止 VF 重排的包均为 9/60 次异常，证明禁止重排不能修复 SK 子入口中的同类缺陷。

同样，删除 `dcci_disable_on_kernel=[".*"]` 的早期实验曾使有限样本通过，但只清除实际绑定中的 disable-DCCI 能力位后仍出现漏写。DCCI 和代码布局会改变执行条件及暴露概率，现有证据不足以将它们认定为根因。

## 8. 源码修复及多路径回归

在两次写之间增加明确的 vector-store 屏障：

```cpp
DataCopy(yLocalAddr, zeroReg64, pregMerge64);
AscendC::MicroAPI::LocalMemBar<
    AscendC::MicroAPI::MemType::VEC_STORE,
    AscendC::MicroAPI::MemType::VEC_STORE>();
DataCopy(yLocalAddr32, sum, pregMerge);
```

重新编译后，反汇编确认对应普通入口和 SK 子入口副本均包含 `SMEM_BAR.VST_VST`。已完成的回归包括：

| 路径 | 修复后结果 |
|---|---:|
| 空 scope、SK 子编译开启、普通入口 replay | 0/216 |
| SK 合并包普通入口直接静态调用 | 0/40 |
| 初次完整 SK 图执行 | 0/15 |
| 后续实际 SK 融合入口复测 | 0/40 |

普通静态路径原本就未观察到异常，不能表述为“屏障将普通静态路径从失败修好”。正确结论是：屏障为所有编译副本补充了原先缺失的顺序保证，并在已测普通入口和实际 SK 子入口路径中消除了异常。

## 9. 定位结论与边界

**已确认根因：** custom op `SwigluGroupQuant` 的 `VFProcessGroupIndexPair` 对同一 UB 地址执行部分重叠的 vector store，缺少 `VEC_STORE -> VEC_STORE` 顺序约束；错误 group sum 导致部分逻辑核跳过输出写回。

**触发条件：** SK 子算子编译、合并布局、入口地址、DCCI 和其他执行时序条件会改变缺陷暴露概率，但不是正确性修复。

**边界：** 根因和修复已在 A5、指定 BF16 `[21,640]` 合成输入及相关静态/SK 路径验证；尚需在原始模型环境用真实 tensor 回归，并覆盖其他 shape、属性、量化分支和可选输出。该实现来自 Recipe custom op，不能据此宣称 CANN 标准算子存在相同缺陷。

## 10. 可复用方法

本案例的通用定位链路是：

```text
模型 SK 开关对照
  -> 三项分类测试
  -> 空 scope 排除“实际融合是必要条件”
  -> 按算子二分 SK 子编译
  -> 多输入 replay + 哨兵识别整行未写回
  -> 核对实际二进制和入口
  -> 找到第一个错误控制量
  -> 源码与反汇编确认重叠访问
  -> 等指令数、单变量因果实验
  -> 普通入口和真实 SK 子入口共同回归
```

复用时关注证据方法，不套用本案例的固定地址、复现率、DCCI 配置或屏障类型。其他算子的整行漏写可能来自不同的计数、索引、搬运或同步缺陷。
