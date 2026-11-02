# 案例：batch mode 核数冲突导致融合范围拆分

## 1. 问题入口与证据范围

同一命名 scope 中有四次自定义 `add_custom` 调用，预期进入一个 SuperKernel，实际生成两个 SK；输出及图重放校验通过。这里的断图指 SK 融合范围拆分，不代表 Python 图捕获失败或设备执行失败。

下面摘录真实现场的关键证据，并区分仅分析异常产物、补充源码和完成对照验证三个阶段。阅读案例无需访问原现场；复用时应根据当前现场证据判断原因。核数、节点 ID 和字段含义均受本次运行版本约束。

## 2. 仅从异常产物定位直接机制

先按 PID、模型和时间关联同一次运行，再建立以下证据链：

| 产物 | 本次观测 | 判断 |
| --- | --- | --- |
| `sk_fusion_fail_reasons.log` | 原始 scope 为 `[3,4,5,6]`；最终为 `[3,4]`、`[5,6]`，两组均 `scopeStatus=SUCCESS` | 四个业务节点都已融合，融合范围被拆分 |
| `sk_fused_nodes.log` | 四个节点均为 `AIV_ONLY`、`isScheModeOn:1`、`resolvedNum:4`；前两节点 `vecNum:4`，后两节点 `vecNum:8`；两个 SK 入口分别为 block 4 和 block 8 | 两组使用不同 Vector 核数；绑定入口已解析 |
| `super_kernel.log` | `bindMap size=1`、`disableScheMode=0`、`blockDimScaleUp=0`；`debug_per_op_max_core_num` 为 0 | 调度模式开启，不支持 blockDim 扩展；排除逐算子调试模式强制拆分 |
| `sk_scope_split.log` | `InitialScopeSplitPass` 后为一个 scope，`ScheModeKernelSplitPass` 后为两个 | 定位到发生拆分的 pass |
| 原始图、更新图及 stdout | 原始图 11 节点；更新为两个 SK 入口及一个 scope 外 default 节点；输出及图重放校验通过 | 原 scope 的四个实际算子与最终两组相符，未观察到执行失败 |

初始切分日志中的 scope 有 10 个节点，包含标记和 placeholder；原始图的 11 个节点还包含 scope 外节点。不能将这些总数当作待融合业务算子数，也不能把 scope 外节点的 `NOT_IN_SCOPE` 算作四个 `add_custom` 的失败。

`sk_scope_split.log` 给出的直接触发记录如下，省略时间和日志前缀：

```text
breakReason=SYNCALL_OP_DROP, triggerNode=5, triggerStream=0,
syncAllNodes=[3,4],
detail="beforeSk=AIV_ONLY(blocks=4,C=0,V=4), candidateSk=AIV_ONLY(blocks=8,C=0,V=8), mismatch=ScheMode V(expected=4,actual=8)"
```

该记录与节点配置及最终融合组共同证实：**node 5 加入时，候选 SK 的 V=8 与已有节点要求的 V=4 冲突，导致 scope 拆分。** 这是图优化阶段的直接机制。此时尚未读取源码，不能解释发射核数的上游来源，也不能仅据 `syncAllNodes` 字段断言源码调用了 `SyncAll`。

同一记录的通用说明写“syncall 算子核数应大于等于其他算子”，但具体 `detail` 明确是 `ScheMode V` 精确匹配失败；不能据通用说明建议单纯增大融合核数。

## 3. 补充源码，追溯核数配置

结合当前算子实现，可取得三类补充证据。以下为关键片段，省略无关参数和函数体：

```cpp
// 入口声明
extern "C" __global__ __vector__ __schedmode__(1) void add_custom(...);

// SK 子函数在 Process 和流水屏障后执行
AscendC::SyncAll<true>();

// 绑定四个 SK 子函数入口
SK_BIND(add_custom, 4, add_custom_sk<0>, add_custom_sk<1>, add_custom_sk<2>, add_custom_sk<3>);

// host 发射核数来自调用参数 blocks
add_custom<<<static_cast<uint32_t>(blocks), nullptr, aclStream>>>(...);
```

调用侧固定前两次 `blocks=4`，后两次由同一个参数控制，本次设为 8。输入均为固定的 `[8,2048]`、FP16，核数差异来自 host 发射参数；batch mode 指入口调度模式，与输入的 batch 维度不同。核函数按传入核数划分计算范围，host 检查只允许本例支持的 4 或 8。

本例走预编译扩展的 `<<<>>>` / `SK_BIND` 直调路径。现场只有扩展构建日志、`.so` 和运行 `.o`，缺少 ACLNN `.run` 或实际 `ts..._outputs` 内容不构成本次断图的原因；即使外层开启 `static_kernel_compile`、stdout 提示开始静态编译，也要按实际调用路径判断。

调度模式和不允许 blockDim 扩展的绑定信息，与本次日志的精确核数约束一致。通用判定及跨版本边界见 [batch mode 核数约束冲突](../graph-break.md#3-batch-mode-核数约束冲突)；不把本例的 capability 数值或源码条件套用到其他版本。

## 4. 已完成的单因素对照

三个新进程始终开启 SK，保持同一扩展 `.so`、同一算子二进制、相同输入、batch mode、同步逻辑及其他选项，仅改变后两次调用的 host 发射核数：

| 运行 | 四次调用的 blocks | 最终融合范围 | 输出及图重放 |
| --- | --- | --- | --- |
| 基线 | `[4,4,4,4]` | `[3,4,5,6]`，一个 block 4 的 SK | 通过 |
| 冲突配置 | `[4,4,8,8]` | `[3,4]`、`[5,6]`，分别为 block 4、8 | 通过 |
| 恢复配置 | `[4,4,4,4]` | `[3,4,5,6]`，一个 block 4 的 SK | 通过 |

三次运行保留的扩展 `.so` SHA-256 相同，`bin_files` 中实际 `add_custom` 的 `.o` SHA-256 也相同，支持排除更换算子二进制这一替代解释。基线与恢复运行均没有本次 `SYNCALL_OP_DROP` 记录。

结合切分日志与对照结果，确认本例 host 发射核数差异触发 batch mode 精确核数约束冲突，造成一个 scope 拆成两个成功融合组。该结果体现框架对不兼容配置的保护性拆分，不能作为框架缺陷、算子不支持 SK 或执行卡死的证据；也未测量性能影响。

## 5. 可复用方法与判断边界

- **仅有异常产物：** 可确认切分阶段、具体核数冲突和最终融合组；发射参数来源尚未知，不妨碍报告直接机制。
- **补充源码：** 核对入口调度声明、绑定能力、同步逻辑、host 参数来源及实际调用路径。不能因案例使用 `SyncAll` 就认定当前算子也使用。
- **获准重跑后：** 在算子支持的配置内，仅统一冲突核数，检查切分原因消失、目标融合组恢复和输出正确；保留原配置便于复验。未开启 SK 的对照不是验证此机制的前提。
