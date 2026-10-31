# AutoFuse 架构介绍

## 系统架构总览

AutoFuse 是 CANN 生态中面向昇腾系列芯片的自动算子融合组件。它接收 GE、Inductor 等图编译组件经图转换、Lowering 和融合范围判定后产出的融合子图及统一 IR，在已确定的融合范围内完成调度优化、Tiling 求解与代码生成，最终输出高性能的 AscendC 融合算子。通过将多个原本独立执行的算子融合为单一 Kernel，AutoFuse 有效减少了中间结果的 GM 读写、Kernel 启动次数以及 Host-Device 调度开销，从而显著提升昇腾 NPU 上的模型执行性能。

整体逻辑结构如下图所示：
![image.png](https://raw.gitcode.com/user-images/assets/9919791/97d497fa-3976-4d40-92bc-ed58c3d53ef0/image.png 'image.png')

如图所示，自动融合方案基于昇腾 NPU 底层统一的 AscendLoopIR（面向 AscendC 编程语言建模的 IR）以及配套的 Schedule 和代码生成等能力，构建了两条融合实现路径：

- **GE 路径**：基于昇腾自研的 GE 框架，侧重昇腾 NPU 亲和性，由 GE 完成符号化、Lowering 和融合范围判断，AutoFuse 负责后端调度、切分和代码生成。
- **Inductor 路径**：对接 PyTorch Inductor，侧重生态适配，复用 Inductor 的融合范围识别能力，后端处理仍由 AutoFuse 完成。

## 解决什么问题

### 面临的问题

在 AI 芯片上，大量算子独立执行会带来以下挑战：

- **Memory Bound 制约性能**：AI 芯片的算力通常高于访存带宽。当模型由大量小算子构成时，每个中间结果都需要写入并重新读取全局内存，内存搬运开销可能超过计算本身，导致计算单元等待数据。
- **手写融合算子成本高**：以推荐类模型为例，其结构变化快、算子数量多，人工手写融合算子的 Kernel/Tiling 代码工作量大、开发周期长，难以跟上模型迭代速度。

### 融合方案选型

针对上述问题，业界存在三种主流算子融合方案，对比如下：

| 方案                       | 优点                                             | 缺点                                                                                                                  |
| -------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| **手写融合算子**     | 性能上限最高                                     | 仅适合高价值固定 pattern，高度定制；开发工作量大，模型变化后维护成本高                                                |
| **Pattern 匹配融合** | 稳定可控，工程落地容易，性能不错                 | 融合范围有限，只能处理已知 pattern，失效几率高，效果难以跨模型复制；会催生海量的 pattern 识别 pass 与特定 kernel 实现 |
| **JIT 自动融合**     | 泛化能力强，可覆盖大量小算子链，适合模型快速演进 | 依赖统一的 IR 表达，性能建模难度大                                                                                    |

AutoFuse 选择 **JIT 自动融合** 方案：优先考虑**泛化能力**，通过统一 IR 和编译期自动决策覆盖大量算子组合，适应模型快速迭代需求，并通过调度、切分和代码生成等技术提升融合后的执行性能。

## 关键技术方案

AutoFuse 按计算特征将网络算子分为两类：一类是 Elemwise、Broadcast 和 View 类（Transpose、Slice、Split）等基础计算类型；另一类是 Reduce、Concat 和 MatMul 等在基础计算类型上扩展融合能力的计算类型。这意味着，各类扩展融合能力都需要支持与基础计算类型进行融合。

### 支持的算子类型

下表列出了主要支持的算子类型及其对应的计算单元：

| 算子类型            | 说明                                                                              | 计算单元   | 典型算子示例                    |
| :------------------ | :-------------------------------------------------------------------------------- | :--------- | :------------------------------ |
| **Elemwise**  | 逐元素计算，每个输出元素与输入元素一一对应                                        | Vector     | Add、Mul、Abs、Exp、Relu、Cast  |
| **Broadcast** | 广播计算，将较小 Shape 的数据沿广播轴扩展，再执行逐元素计算                       | Vector     | BroadcastTo、BiasAdd            |
| **View**      | 视图变换，改变数据的逻辑形状、轴序或切分方式                                      | MTE/Vector | Transpose、Slice、Split         |
| **Reduce**    | 规约计算，沿指定轴对多个元素进行聚合                                              | Vector     | ReduceSum、ReduceMax、ReduceMin |
| **泛 Norm**   | 由同轴 Reduce、Broadcast 和 Elemwise 等计算组合形成的归一化计算模式，并非单一算子 | Vector     | LayerNorm、RMSNorm              |
| **Concat**    | 拼接计算，沿指定轴将多个 Tensor 拼接为一个 Tensor                                 | MTE/Vector | Concat                          |
| **MatMul**    | 矩阵计算，包括矩阵乘和卷积等                                                      | Cube       | MatMul                          |

### 支持的融合能力

AutoFuse 当前主要支持 VV 和 CV 两类融合：

| 融合类型 | 可融合算子类型 |
| :------- | :------------- |
| **VV 融合**（Vector + Vector） | 支持 Elemwise、Broadcast、View（包括 Transpose、Slice 和 Split）、Reduce、Concat 等 Vector 类算子的融合。 |
| **CV 融合**（Cube + Vector） | 支持 Cube 类算子与 Vector 类算子的融合。 |

## 前端适配

前端适配负责将 PyTorch、TensorFlow 等主流深度学习框架的模型图转换为 AutoFuse 可处理的图 IR。当前主要有两条接入路径：

- **GE 路径**：在线场景通过 TorchAir 或 TensorFlow Adapter 将 AtenIR、GraphDef 等图表示转换为 AscendIR，进入 GE 图编译流程；离线场景通过 ATC 内置 Parser 将 TensorFlow、ONNX 等格式的模型解析为 AscendIR。相关说明请参见 [GE 项目](https://gitcode.com/cann/ge)。
- **Inductor 路径**：将 PyTorch 的 AtenIR 转换为 InductorIR，走 PyTorch Inductor 路径。其中，`torch-npu` 模块的相关说明请参见 [PyTorch 项目](https://gitcode.com/Ascend/pytorch)；`inductor-npu-ext` 模块的相关说明请参见 [TorchAir 项目](https://gitcode.com/Ascend/torchair/tree/master/experimental/_inductor_npu_ext)。

## Graph Engine

Graph Engine 作为 AutoFuse 的前端，负责对 AscendIR 图进行符号化推导、Lowering 和 CanFuse 融合条件判断，确定可融合的算子范围以及融合后的计算表达。

| 步骤                          | 职责                                                                                                               | 相关资料                                                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| **符号化**              | 用符号表达算子 Shape，增强动态 Shape 处理能力，提供化简、推导和 Guard 功能，为后续循环轴合并和内存优化提供关键信息 | [官方资料](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/graphdevg/autofuse_1_0006.html) |
| **Lowering**            | 将高层级 AscendIR 转换为低层级 AscendLoopIR，以贴近 AscendC 语义表达计算逻辑，确定融合结构和数据依赖               | [官方资料](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/graphdevg/autofuse_1_0012.html) |
| **CanFuse（融合策略）** | 从语义正确性、是否可表达和资源预算三层确定融合边界                                                                 | [官方资料](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/graphdevg/autofuse_1_0018.html) |

## AutoFuse

AutoFuse 作为上层 GE 或 Inductor 的后端，是自动融合编译的核心。它接收已经确定的融合范围，通过 Schedule、Codegen 和 Auto Tiling 三个核心模块完成融合 Kernel 的调度、代码生成和切分策略求解，并借助 AscendC API 提供算子接口支持。

| 模块                  | 职责                                                                                          | 相关资料                                                |
| --------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| **Schedule**    | 调度策略生成：计算重排、循环合并、并行优化、内存优化和多模板生成                              | [模块说明](../design/features/schedule.md)               |
| **Codegen**     | 代码生成：解析调度图，生成 Host 侧和 Device 侧代码                                            | [模块说明](../design/features/codegen.md)                |
| **Auto Tiling** | Tiling 求解：在 UB 约束下求解 Tile 大小和分核策略，评估切分方案性能，选择合适的模板和切分策略 | [模块说明](../design/features/auto_tiling.md)            |
| **AscendC API** | 提供 Vector 计算、Cube 计算、数据搬运和类型转换等 API                                         | [源码参考](../../../../autofuse/v35/ascendc/api_regbase) |

## 编译与运行

AutoFuse 生成的 Host 和 Device 源码由 **BiSheng 编译器**进一步编译为 Host 侧共享库和 Device 侧 Kernel 二进制。运行时，上层框架根据实际输入准备 Tiling 参数并启动融合 Kernel，**Runtime** 负责设备资源管理、Kernel 下发和执行。对于原本由多个 Kernel 完成的算子链，融合后通常可以减少 Kernel 启动次数和中间 Tensor 的全局内存读写，从而降低数据搬运和调度开销，提升**昇腾 NPU** 的硬件资源利用率和模型执行性能。

## 启用方式

按照两条融合实现路径启用：

- **GE 路径**：[GE 路径下使能 AutoFuse（TensorFlow）](./enable_autofuse_ge_tensorflow.md)。该文档以 TensorFlow 为例，介绍依赖版本、`AUTOFUSE_FLAGS` 配置、环境变量和运行用例。PyTorch 框架经 TorchAir 接入 GE 路径的方法，参见 [GE 路径下使能 AutoFuse（PyTorch）](./enable_autofuse_ge_pytorch.md)，该文档同时演示了动态 Shape 场景。
- **Inductor 路径**：[Inductor 路径下使能 AutoFuse](./enable_autofuse_inductor.md)。该文档以 PyTorch 为例，介绍依赖版本、`torch.compile` 配置、环境变量和运行用例。

## 项目结构

```text
graph-autofusion/
├── autofuse                     # AutoFuse 核心代码
│   ├── ascir                    # AscIR 算子元信息、内建算子与注册能力
│   ├── graph_metadef            # 图、节点、Tensor、属性及符号表达式等基础定义
│   ├── inc                      # 对外公共头文件及融合相关接口
│   ├── optimize                 # 图优化、任务切分、自动调度与内存规划
│   ├── att                      # Auto Tiling、候选方案求解及 Host Tiling 生成
│   ├── codegen                  # Kernel、Tiling Data、Host Tiling 与 InferShape 代码生成
│   ├── ascendc                  # 代码生成使用的 AscendC API 定义与扩展
│   ├── compiler                 # Python/C++ 接口、Host/Device 编译及产物发布
│   ├── common                   # 日志、配置、平台上下文和公共工具
│   ├── v35                      # 昇腾 950 芯片相关优化
│   ├── examples                 # PyTorch、TensorFlow 等接入与运行样例
│   ├── tests                    # 单元测试、系统测试及端到端测试
│   └── tools                    # 辅助工具脚本
├── super_kernel                 # 独立的 SuperKernel 融合组件
├── docs                         # Graph-autofusion 项目文档
├── cmake                        # 公共 CMake 脚本与依赖配置
└── scripts                      # 环境检查、测试、打包和安装脚本
```

## 支持产品

自动融合特性仅在如下产品型号支持：

| 系列       | 产品型号                                      |
| ---------- | --------------------------------------------- |
| Ascend 950 | Ascend 950PR / Ascend 950DT                   |
| Atlas A3   | Atlas A3 训练系列产品 / Atlas A3 推理系列产品 |
| Atlas A2   | Atlas A2 训练系列产品 / Atlas A2 推理系列产品 |

> 以上支持列表随 CANN 版本迭代可能扩展，最新信息请以 [昇腾官网](https://www.hiascend.com/) 发布的各版本产品支持信息为准。

## 相关资料

- [AutoFuse 概述（官网资料）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/graphdevg/docs/zh/user_guides/graph_dev/autofuse/overview.md)
- [Autofuse 组件说明（含多类型算子融合用例）](../../../../autofuse/README.md)
