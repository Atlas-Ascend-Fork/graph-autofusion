# Codegen

## 1. 特性背景

### 1.1 Codegen 在 Autofuse 流水线中的位置

Autofuse 的编译流水线为：图优化（Optimize）→ 调度（Schedule）→ 代码生成（Codegen）→ 性能调优（ATT/Auto Tiling）。其中：

- **Schedule** 负责对融合后的计算图做切分决策（UB 切分、核间切分、多模板），输出 ImplGraph；
- **Codegen** 解析 ImplGraph，生成实际可执行的代码，包括在 Host 侧（Host Code）和 Device 侧（Kernel Code）运行的代码；
- **ATT** 负责在运行前计算 tiling 的具体取值（tiling_key、各切分轴 size 等），填充到 Codegen 生成的数据结构中。

Codegen 的核心设计思想是：**Codegen 只生成代码的"形"，不决定 tiling 的"值"**。代码中所有与切分相关的量（tiling_key、轴 size、循环次数）都以 tiling_data 变量的形式占位，具体数值由 ATT 在 Host 侧运行时计算并传入，从而使同一份生成代码可以适配不同的 shape 和硬件资源条件。

### 1.2 设计目标

- **一处调度，多处适配**：Schedule 针对不同切分策略生成多个 ImplGraph 模板，Codegen 将其全部生成并通过 tiling_key 分发，运行时按实际 shape 选择最优模板；
- **生成代码自包含**：kernel 源文件内联所有依赖的 AscendC API 源码，可独立编译；
- **平台可扩展**：通过 ApiCall 注册机制与平台扩展层，支持不同计算类别（Vector/Cube/MicroAPI）和不同 SoC 版本的算子接入，Kernel 生成主流程保持平台无关。

## 2. 用户使用场景

| 场景 | 说明 |
| --- | --- |
| 静态 shape 融合编译 | TF/ATC 前端接入，shape 编译期已知，Codegen 生成寄存器 tiling_key 分发的多模板 kernel，运行时零解析开销 |
| 动态 shape 融合编译 | shape 编译期未知，tiling 在 Host 侧运行时计算，Codegen 生成运行时解析 tiling_data 的分发结构 |
| PyTorch Inductor 接入 | 通过 `GenerateForInductor` 路径生成 Inductor 专用接口，支持静态 shape 全 const tiling 前移和 TopN tiling 候选选择 |
| CV（Cube/Vector）融合 | Matmul 等算子与 Vector 算子融合，Codegen 生成多 group CV kernel 与配套 tiling wrapper |
| PGO tiling 搜索 | 生成独立的 tiling 搜索可执行程序，在真实硬件上实测各 tiling 候选性能，保存最优结果供编译期使用 |

## 3. 特殊背景及限制

本节描述 Codegen 与相邻组件协作时的约束，这些约束大部分无法从单个模块的代码中直接看出，是模块间协作的隐式契约。

### 3.1 与 Schedule 的契约（ImplGraph）

ImplGraph 是 Schedule 与 Codegen 之间唯一的接口，Codegen 不做任何调度决策，隐含约束包括：

- **轴完备性**：ImplGraph 上的轴必须构成完整的切分树（ORIGINAL → TILE/BLOCK 的 OUT/IN 配对）。Codegen 按 BLOCK_OUT（核间并行，不生成 for）、BLOCK_IN（生成核内偏移）、其余轴（生成 for 循环）三类处理，缺失配对会导致生成的循环结构不完整；
- **节点信息完备性**：每个节点的 repeats/strides/vectorized_strides 必须与 axis 维度一致，dtype 组合必须在 API 支持范围内，Codegen 生成前会做校验（`CheckGraphValidity`），校验失败直接终止编译。当前 Broadcast 的 int64 dtype 暂不在校验范围内（豁免项）；
- **表达式可翻译性**：图上的 shape/offset 表达式（`ge::Expression`）必须能翻译为"tiling 变量 + 循环变量"的组合。Codegen 不支持运行时无法求值的表达式，例如依赖 Host 侧随机性的表达式无法出现在轴定义中。

### 3.2 与 ATT/Auto Tiling 的职责边界

- **Codegen 决定结构，ATT 决定取值**：`AutofuseTilingData` 的字段（哪些轴需要 size 变量、tiling_key、block_dim）由 Codegen 根据切分策略确定，但字段的具体数值由 ATT 计算。两侧对 TilingData 结构的定义必须一致，该一致性由两者共用同一份 TilingData 生成代码保证，任何一侧单独修改字段都会导致运行时错位；
- **性能建模在 ATT 侧**：Codegen 在组装 API 参数时会计算 `outer_call_count`（API 外层调用次数）等信息写入节点参数，Auto Tiling 的性能公式（耗时 ≈ API 调用次数 × 单次耗时）以此为输入做 tiling 候选评估。性能公式本体不在 Codegen 中，Codegen 新增 API 参数时需确认 ATT 侧公式是否依赖该参数；
- **PGO/TopN 的兜底限制**：tiling_key 数量超过上限（10000）时 PGO 搜索自动关闭，回退到常规 tiling 生成；tiling_key 数量超过单编译单元上限时，kernel 会自动拆分为多个编译单元分别编译，两处上限不同，独立生效。

### 3.3 与外部 tiling 库的依赖

Host 侧 tiling 函数的实际生成逻辑位于外部 gen_tiling 库中，Codegen 通过函数指针（`TilingLibCodegenFunc`，指向 gen_tiling.so 的 `CodegenTiling` 符号）回调生成，再由 Codegen 负责组装头文件、拼接翻译单元。这意味着：

- tiling 代码生成能力的变更可能涉及 gen_tiling 库的同步更新，两者需配套发布；
- gen_tiling 库缺失或符号不匹配时，tiling 生成失败，与 kernel 生成是独立的两条失败路径。

### 3.4 平台（SoC 版本）约束

- 不同 SoC 版本（如 3510、5102、9201、9202）的指令集和 API 形态存在差异，版本差异通过 IR 注册表按 SoC 版本路由到不同的实现（V2 实现），Codegen 主流程不感知具体芯片；
- 新增 SoC 版本支持时，需同时提供该版本的 ApiCall 实现和 API 源码模板注册，不能复用其他版本的模板；
- Kernel 文件中调用的 AscendC API 源码由 `AscendCApiRegistry` 在构建期从源码头文件打包为字符串注册，生成时内联。API 模板更新后必须重新编译 Codegen 模块，否则生成结果仍是旧模板。

### 3.5 其他限制

- **IO 数量限制**：单个融合算子的输入输出数超过 64 时，kernel 形参自动切换为 list_tensor 形式，超出该数量可能触发编译失败；
- **队列深度与 UB 容量**：UB 内存分配（队列深度、double buffer）由 ImplGraph 上的 mem/que 信息给出，Codegen 忠实翻译不做事后调整，UB 超限的校验发生在 Schedule 阶段；
- **Cube 算子的 tiling 依赖官方实现**：Matmul/Conv 的 tiling 通过 wrapper 调用官方 matmul tiling 接口（`AutofuseDoCubeMatMulTiling`），Codegen 不自行实现 Cube tiling 算法。

## 4. 对外接口

### 4.1 生成产物接口（运行时可见）

Codegen 产出的代码以 C 接口符号对外提供，供运行时框架调用：

| 符号 | 侧别 | 说明 |
| --- | --- | --- |
| `AutofuseTiling`（GetTiling 系列） | Host | tiling 主函数，根据 shape 计算 tiling_key、各轴切分大小、workspace 大小 |
| `InferShape` | Host | 形状推断函数，输入符号表达式上下文，输出各 tensor 形状 |
| `GetKernelBin(std::vector<char>&)` | Host | 返回已编译 kernel 的二进制内容，供运行时直接加载，免去运行期编译 |
| 核函数（`extern "C" __global__ __aicore__`） | Device | 按 tiling_key 分发多个模板函数的核函数入口 |
| `AutofuseTilingData` | 共享 | Host/Device 共享的 tiling 数据结构，字段值由 ATT 填充 |

### 4.2 内部扩展接口（编译期）

| 接口 | 说明 |
| --- | --- |
| `Codegen::Generate / GenerateForInductor` | 总入口，输入 Schedule 的 `FusedScheduledResult`，输出 `CodegenResult`（proto/tiling_data/tiling/kernel/infer_shape） |
| `Codegen::GenerateTilingData / GenerateTiling / GenerateKernel / GenerateInferShape / GenGetKernelAndJson` | 各产物的独立生成接口，可单独调用 |
| `ApiCallRegister<T>` | ApiCall 静态自注册，新增算子只需实现 `ApiCall` 子类并注册，主流程自动发现 |
| `MicroApiCallRegister<T>` | MicroAPI 子工厂注册（v35 平台扩展层内部使用） |
| `AscendCApiRegistry::RegisterApi / GetFileContent` | API 源码模板注册与查询，生成 kernel 时按节点依赖内联 |

## 5. 整体架构

Codegen 由门面类 `Codegen`（autofuse/codegen/codegen.h）对外提供统一入口，一次生成产出五类结果（`CodegenResult`）：

| 产物 | 说明 | 生成入口 |
|------|------|----------|
| proto | 序列化的调度结果等元信息 | `Codegen::Generate` |
| tiling_data | `AutofuseTilingData` 结构体定义 | `Codegen::GenerateTilingData` |
| tiling | Host 侧 tiling 函数源码 | `Codegen::GenerateTiling` |
| kernel | Device 侧核函数源码 | `Codegen::GenerateKernel` |
| infer_shape | Host 侧形状推断函数源码 | `Codegen::GenerateInferShape` |

整体架构分层如下：

```text
┌─────────────────────────────────────────────────────────────┐
│                 入口层（Codegen 门面）                        │
│   Generate / GenerateForInductor / GenerateTiling / ...      │
├─────────────────────────────────────────────────────────────┤
│                 Kernel 生成层                                │
│   Kernel（图解析 → Loop 树 → 循环与 API 调用生成）             │
│   tiling_key 多模板分发 / API 源码内联                        │
├─────────────────────────────────────────────────────────────┤
│                 ApiCall 框架（核心扩展点）                    │
│   ApiCall 基类 + ApiCallFactory 注册工厂                      │
│   ┌──────────────┐ ┌─────────────────────────────────────┐  │
│   │ 主框架内置    │ │ v35 平台扩展层                       │  │
│   │ Load/Store/  │ │ reg_api_call（RegBase）              │  │
│   │ elewise/     │ │ cube_api_call（Matmul/Conv2D）       │  │
│   │ reduce/      │ │ micro_api_call（MicroAPI）           │  │
│   │ transpose... │ │ vec_func_call（VectorFunc）          │  │
│   └──────────────┘ └─────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                 Host 生成层                                  │
│   TilingData / TilingLib（含 PGO、Inductor TopN、CV tiling）  │
│   InfershapeGen / GenGetKernelAndJson                        │
├─────────────────────────────────────────────────────────────┤
│                 支撑层                                       │
│   AscendCApiRegistry（API 模板内联）                          │
│   ExpressionConvertStruct（表达式翻译）                       │
│   CheckGraphValidity（图校验） / ApiParamDump（调试）          │
└─────────────────────────────────────────────────────────────┘
```

生成顺序为：先 Kernel，再 TilingData、Tiling、InferShape，最后可选执行 PGO 生成。Tiling 部分的实际生成由 `TilingLib` 通过函数指针回调外部 gen_tiling 库完成，Codegen 负责组装与拼接各翻译单元。

## 6. 核心实现

### 6.1 输入：ImplGraph

Codegen 的输入是 Schedule 生成的 ImplGraph，该图包含了 shape 信息、节点信息、轴信息等生成代码的基本要素。ImplGraph 示例如下：

<div style="text-align: center;">
<img src="../../figures/codegen_implgraph_example.png" alt="ImplGraph 示例" style="width: 10%; max-width: 800px;">
</div>

图上的具体信息如下所示：

```text
// size描述计算量大小
Sizes:
  z0z1t_size: VAR
  z0z1Tb_size: VAR

// axis描述轴信息，包括大小、类型、对齐方式等
Axis:
  z0(0) : 200, ORIGINAL, align: -1, allow_oversize_axis: 0, allow_unaligned_tail: 1
  z1(1) : 200, ORIGINAL, align: -1, allow_oversize_axis: 0, allow_unaligned_tail: 1
  z0z1(2) : 40000, ORIGINAL, align: -1, allow_oversize_axis: 0, allow_unaligned_tail: 1  // ORIGINAL代表未切分的原始轴
  z0z1T(3) : Ceiling((40000 / (z0z1t_size))), TILE_OUT, from: {z0z1, }, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1
  z0z1t(4) : z0z1t_size, TILE_IN, from: {z0z1, }, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1  // TILE_IN代表UB切分的内轴
  z0z1TB(5) : Ceiling((Ceiling((40000 / (z0z1t_size))) / (z0z1Tb_size))), BLOCK_OUT, from: {z0z1T, }, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1
  z0z1Tb(6) : z0z1Tb_size, BLOCK_IN, from: {z0z1T, }, align: 1, allow_oversize_axis: 0, allow_unaligned_tail: 1  // BLOCK_IN代表核间切分的内轴

// node，每个节点会生成kernel代码中的api调用
Nodes:
......
  abs_test/gather_0: Load (1)
......
  abs_test/abs_0: Abs (2)
    .axis = {z0z1TB, z0z1Tb, z0z1t, }
    .loop_axis = z0z1Tb
    .api:
      .compute_type = elewise
      .type = Compute
      .unit = Vector
    .x = abs_test/gather_0.y
    .y.dtype = float32
    .y.axis = {z0z1TB, z0z1Tb, z0z1t, }
    .y.repeats = {(40000 / (z0z1Tb_size * z0z1t_size)), z0z1Tb_size, z0z1t_size, }
    .y.strides = {(z0z1Tb_size * z0z1t_size), z0z1t_size, 1, }
    .y.vectorized_axis = {z0z1t, }  // 向量化轴代表UB内计算的数据量对应的轴
    .y.vectorized_strides = {1, }
    .y.mem:
      .tensor_id = 3
      .alloc_type = Queue
      .hardware = UB
      .position = TPosition::VECOUT
    .y.que:
      .id = 1
      .depth = 2
      .buf_num = 2
      .reuse_id = 1
  abs_test/store: Store (3)
......
```

根据 ImplGraph 解析，以下示例代码展示了 Codegen 生成 DataCopy API 的过程，可以看出，核心流程包括解析图上的切分策略，组装 API 的入参，生成在 Device 侧执行的代码。

```cpp
ss << "DataCopyPadExtend(" << ub << ", " << gm << "[" << gm_offset << " + " << tpipe.tiler.Size(api_attr.offset)
   << "], " << dma_param.block_count << ", " << dma_param.block_len << ", " << dma_param.src_stride << ", "
   << dma_param.dst_stride << ");" << std::endl;
```

### 6.2 Kernel 代码生成

Kernel 生成由 `Kernel` 类（autofuse/codegen/codegen_kernel.h）完成，一张 ImplGraph 对应一个 Kernel 生成器，整体分为解析和生成两个阶段。

#### 6.2.1 图解析与合法性校验

`Kernel::ParseGraph` 解析 ImplGraph 前首先调用 `CheckGraphValidity`（autofuse/codegen/codegen_graph_check.cpp）对图做合法性校验，包括：

- **dtype 校验**：收集节点输入 dtype 组合，校验 API 是否支持；
- **repeats/strides 校验**：校验 axis、repeats、strides、vectorized_strides 维度一致性；
- **节点校验**：逐节点调用 IR codegen 实现的 `IsNodeValid`。

解析过程中，Kernel 将图上的轴注册进 `Tiler`（tiling 数据变量映射）、将 tensor/队列注册进 `TPipe`（UB 内存与队列管理），并按节点序列构造出 `Loop` 树（autofuse/codegen/codegen_kernel_loop.cpp），树中每个节点对应一个 ApiCall 对象。

#### 6.2.2 循环与 API 调用生成

`Loop::Generate` 按 ImplGraph 的切分策略递归生成循环结构：

- **BLOCK_OUT 轴**（核间并行轴）：只生成 size/actual_size 计算，不生成 for 循环，通过 blockidx 隐式并行；
- **BLOCK_IN 轴**：生成 `block_dim_offset` 偏移计算，确定当前核处理的数据分片；
- **其余轴**：生成 `for (axis = 0; axis < loop_size; axis++)` 循环，并在循环体内生成尾块（tail）的实际 size 计算与缓存使能条件。

循环体内每遇到一个计算节点，按固定骨架生成代码：`WaitInputs → AllocOutputs → API 调用 → SyncOutputs → FreeInputs → FreeUnusedOutputs`，即围绕 API 调用完成 UB 队列的等待、分配、同步与释放，保证 MTE 搬运与 Vector 计算的正确流水。

#### 6.2.3 多模板 tiling_key 分发

与传统算子代码类似，针对不同的切分策略，Schedule 会生成多个模板，对应不同的 ImplGraph，Codegen 需要依次解析这些模板，生成模板函数，并在核函数入口处根据 tiling_key 调用不同的模板函数。示例如下：

```cpp
extern "C" __global__ __aicore__ void abs_test(GM_ADDR abs_test_Data_0, GM_ADDR abs_test_Output_0, GM_ADDR workspace, AutofuseTilingData param) {
    const AutofuseTilingData t;
    if (t.tiling_key == 0) {
        abs_test_0_general_0_nil_2_nil(abs_test_Data_0, abs_test_Output_0, workspace, t);
    } else if (t.tiling_key == 1) {
        abs_test_0_general_0_nil_2_nil_unaligned(abs_test_Data_0, abs_test_Output_0, workspace, t);
    }
}
```

上面 AutofuseTilingData 结构体示例如下，其中的成员变量具体值，包括 tiling_key 等由 ATT 计算给出：

```cpp
BEGIN_TILING_DATA_DEF_T(AutofuseTilingData)
  const uint32_t block_dim = 40;
  const uint32_t corenum = 0;
  const uint32_t ub_size = 196352;
  const uint32_t hbm_size = 0;
  const uint32_t tiling_key = 0;
  const uint32_t z0z1z2z3t_size = 17968;
  const uint32_t z0z1z2z3Tb_size = 57;
  const uint32_t q0_size = 96;
  const uint32_t q1_size = 35936;
  const uint32_t q2_size = 35936;
  const uint32_t b0_size = 35936;
  const uint32_t tmp_tbuf_size = 8192;
END_TILING_DATA_DEF_T;
```

不同的模板函数，生成的 for 循环以及每次处理的数据量各不相同，示例如下：

- **tiling_key=0**

对 z1 轴进行切分，每次在 UB 中完成计算的向量化轴为 z1t。

```text
for (int z0z1Tb = 0; z0z1Tb < z0z1Tb_loop_size; z0z1Tb++) {
    Abs(y_local[0], xlocal[0], z1t_actual_size);
}
```

- **tiling_key=1**

对 z0 轴进行切分，每次在 UB 中完成计算的向量化轴为 z0t 以及 z1。

```text
for (int z0Tb = 0; z0Tb < z0Tb_loop_size; z0Tb++) {
    Abs(y_local[0], xlocal[0], z0t_actual_size * z1_axis_size);
}
```

`Kernel::GenKernelFuncByTilingKey` 根据场景选择不同的分发实现：

- **寄存器 tiling_key**（静态 shape）：将 tiling_key 写入寄存器，每个调度模板生成一个 kernel 函数，入口处生成 `if (t.tiling_key == N) { ... }` 分发链；当 group 内多模板需要并行时插入 `SyncAll()` 同步；
- **解析 tiling_data**（动态 shape / Inductor）：运行时解析 tiling_data 结构选择分支；
- **CV 融合**：生成多 group 的 CV kernel 函数；
- **模板数超限**：tiling_key 数量超过单编译单元上限时，自动拆分为多个编译单元（PackingFunc）分别编译。

#### 6.2.4 API 源码内联

Kernel 文件中调用的 AscendC API（如 `DataCopyPadExtend`、`Abs` 等）的实现源码，由 `AscendCApiRegistry`（autofuse/codegen/ascendc_api_registry.cpp）统一管理。构建期将各 API 头文件打包为源码字符串注册进单例表，生成 kernel 时按节点声明的头文件依赖，将 API 源码字符串直接内联到 kernel 文件中，使每个 kernel 源文件自包含、可独立编译。

### 6.3 Host Code 生成

#### 6.3.1 tiling_data 生成

`TilingData::Generate`（autofuse/codegen/codegen_tiling_data.cpp）生成 `AutofuseTilingData` 结构体定义，字段包括 tiling_key、block_dim、workspace 大小、各切分轴 size 等；多 ScheduleGroup 场景生成统一的包装结构。对于静态 shape 的 Inductor 场景，还会生成全 const 初值版本，将 tiling 计算前移到编译期。

workspace 字节数与字节偏移均使用 `uint64_t`，单块超过 2 GiB、累计超过 4 GiB 时不会截断。直接 tiling ABI 使用 `uint64_t *workspace_size`；Host 侧累加与 512B 对齐使用受检计算，溢出时返回失败。非 Inductor 直接 tiling 在累计 workspace 后保留既有 16 MiB reserved workspace 语义。

#### 6.3.2 tiling_func 生成

Host 侧 tiling 函数由 `TilingLib` 组织生成（autofuse/codegen/codegen_tiling.cpp），包括 `AutofuseGetTilingSize`（估算 tiling 所需内存）、`AutofuseTiling`（主 tiling 函数，根据 shape 计算 tiling_key 和各轴切分大小）、workspace 计算等函数。除常规生成外，tiling 侧还支持：

- **PGO 生成**：生成独立的 tiling 搜索可执行程序，在真实硬件上遍历 tiling 候选、采集性能并保存最优结果；
- **Inductor TopN 选择**：结合性能建模公式（由 ATT 提供，耗时 ≈ API 调用次数 × 单次耗时）与实测 profiling，对 tiling 候选做建模评估与 TopN 筛选；
- **CV 融合 tiling**：针对 Cube/Vector 融合场景生成专用 tiling 函数与辅助接口。

#### 6.3.3 infer_shape 生成

`InfershapeGen::GenInferShapeFunc`（autofuse/codegen/codegen_infershape.cpp）生成 `extern "C" InferShape` 函数：解析各输出 shape 表达式中的符号变量，生成逐维 AppendDim 的推断代码；对含除法取整的表达式额外生成四舍五入与精度校验逻辑，保证动态 shape 下形状推断与 kernel 实际计算一致。

#### 6.3.4 get_kernel 生成

`Codegen::GenGetKernelAndJson` 读取已编译的 kernel 二进制（ota），生成 `extern "C" void GetKernelBin(std::vector<char>&)` 函数，将 Device 代码以二进制形式嵌入 Host 侧源码，供运行时直接加载，免去运行期编译开销。

### 6.4 ApiCall 框架

ImplGraph 上每个节点最终都会转化为一次 API 调用，这一转换由 ApiCall 框架承载，是 Codegen 的核心扩展点。

#### 6.4.1 基类与生成链路

所有 ApiCall 继承自基类 `ApiCall`（autofuse/codegen/codegen_kernel_loop.h），核心生成链路为：

```text
Init(node)                 // 收集 unit、输入输出 tensor、tmp_buf 信息
  → ParseAttr()            // 子类覆写，解析算子属性
  → BuildApiParam()        // 组装 CodegenApiParam（循环轴、offset、cal_count 等）
  → GenerateApiCallString()// 按 api_param 拼接 API 调用字符串
  → PostProcess()          // 生成收尾同步代码
```

其中 `CodegenApiParam` 是 API 入参的中间表示，将图上的符号表达式（`ge::Expression`）经 `ExpressionConvertStruct`（autofuse/codegen/expression_convert_struct.cpp）翻译为 tiling 变量、循环变量的 C++ 表达式组合，例如 `cal_count`、偏移量、循环 size 等。

基类还提供 UB 生命周期辅助接口（`WaitInputs/AllocOutputs/SyncOutputs/FreeInputs` 等），处理共享队列复用与 MTE2/MTE3 同步；子类通常只需覆写 `BuildApiParam` 和 `Generate` 即可接入新算子。

#### 6.4.2 工厂与注册机制

ApiCall 的创建采用静态自注册工厂模式（autofuse/codegen/api_call/utils/api_call_factory.h）：

- 各 ApiCall 实现类在文件尾定义静态 `ApiCallRegister<T>` 对象，向单例工厂 `ApiCallFactory` 注册"类名 → 构造函数"；
- IR 定义侧（ascir codegen impl）通过 `GetApiCallName()` 声明节点对应的 ApiCall 类名；
- Kernel 生成遍历节点时，按类名从工厂创建 ApiCall 对象，主生成流程无需感知具体算子实现，新增算子只需增加实现类并注册。

#### 6.4.3 内置 ApiCall

主框架（autofuse/codegen/api_call/）内置的 ApiCall 按计算类别组织：

| 类别 | ApiCall |
|------|---------|
| 数据搬运 | `LoadApiCall`、`StoreApiCall` |
| 一元计算 | `UnaryApiCall`、`CastApiCall`、`RsqrtApiCall`、`LogicalNotApiCall`、`UnaryBitWidthChangeApiCall` 等 |
| 二元计算 | `BinaryApiCall`、`AxpyApiCall`、`TrueDivApiCall`、`PowApiCall`、`LeakyReluApiCall` 等 |
| 比较与选择 | `CompareApiCall`、`WhereApiCall`、`ClipByValueApiCall` |
| 归约 | `ReduceApiCall` |
| 数据重排 | `TransposeApiCall`、`BroadcastApiCall`、`ConcatApiCall`、`GatherApiCall` |
| 其他 | `PadApiCall`、`RemovePadApiCall`、`Ub2UbApiCall` 等 |

### 6.5 v35 平台扩展层

不同 SoC 版本的硬件指令集和 API 形态存在差异，Codegen 通过平台扩展层（autofuse/v35/codegen/）承载版本特定实现。扩展层完全复用主框架的基类与注册机制：各实现类同样继承 `ApiCall` 并静态注册进 `ApiCallFactory`，由 SoC 版本路由在 IR 注册表层面按芯片版本选择实现，Kernel 生成流程本身保持平台无关。

v35 扩展层包含四类实现：

#### 6.5.1 reg_api_call：RegBase 风格 API

面向 RegTensor 指令形态的向量计算与数据搬运 API，是 v35 的主体，包括 `BinaryApiCallV2`、`CastV2ApiCall`、`CompareV2ApiCall`、`LoadRegApiCall`、`StoreRegApiCall`、`NddmaApiCall`、`IndirectLoadRegApiCall`、`BroadcastRegApiCall`、`ConcatRegApiCall`、`SplitRegApiCall`、`RegReduceApiCall`、`TransposeRegApiCall`、`GatherRegApiCall`、`WhereRegApiCall`、`SoftmaxApiCall` 等约 24 个类。相比主框架版本，V2 系列 API 具备更强的参数化能力（如轴合并 merge_axes），并会在 `BuildApiParam` 中计算 `outer_call_count`（API 外层调用次数），该值直接对接 Auto Tiling 的性能公式建模（耗时 ≈ 调用次数 × 单次耗时），作为 tiling 候选评估的输入。

对应的 API 源码模板（约 100 个 RegBase API）由构建期从源码头文件自动打包为字符串，注册进 `AscendCApiRegistry`（autofuse/v35/codegen/ascendc_reg_base_api_register.cpp）。

#### 6.5.2 cube_api_call：Cube 算子 API

`MatmulApiCall`、`Conv2DApiCall` 生成 Matmul/Conv 等 Cube 算子的 AscendC API 调用。Cube 单元调用除生成调用点外，还会生成模板函数定义（`GenerateFuncDefinition`）和宏展开（`GenerateMacro`），并通过 tiling wrapper 对接官方 matmul tiling 实现。相关 Cube API 模板（matmul、batch_matmul、conv2d 及其 dynamic/tiling_key/pingpong 变体）注册见 autofuse/v35/codegen/ascendc_cube_api_register.cpp。

#### 6.5.3 micro_api_call：MicroAPI

MicroAPI 是最低一级的代码形态，直接生成 `AscendC::MicroAPI::*` 指令级调用（如 `LoadAlign`、`StoreAlign`）与 `AscendC::Reg::*` 寄存器操作（如 `Pack`、`UnPack`、`MaskPack`），由独立的 `MicroApiCall` 基类与 `MicroApiCallFactory` 工厂承载（不继承主框架 `ApiCall`），包括 `MicroLoadApiCall`、`MicroStoreApiCall`、`MicroCastApiCall`、`MicroCompareApiCall`、`MicroWhereApiCall`、`MicroBinaryScalarApiCall`、`MicroScalarBroadcastApiCall` 等。

#### 6.5.4 vec_func_call：VectorFunc

`VfCall` 将 ImplGraph 中的子图封装为一个独立的 vf 函数，实现"多次循环 → 一次函数调用"的结构优化。其内部持有 `VFLoop` 循环树，遍历子图节点时通过 MicroAPI 工厂为每个节点创建 MicroApiCall，形成两层结构：

```text
VfCall（主框架 ApiCall，生成函数定义与调用点）
  └── VFLoop（vf 内循环树）
        └── MicroApiCall（指令级 API 调用）
```

该机制同时支持 CV-UB 融合场景的 kernel 生成。

### 6.6 辅助机制

- **API 参数 dump**：`CodegenApiParam::DumpGraphApiParams`（autofuse/codegen/codegen_api_param_dump.cpp）可将每张 ImplGraph 中所有节点的 API 入参序列化为文本文件，用于调试和对比生成的调用参数，开发新 ApiCall 时可通过该工具核验参数组装是否正确。
- **表达式转换**：`ExpressionItem/CombinedExpression` 提供符号表达式到 C++ 代码的统一渲染，并附带工厂函数（`ExprItemFactory/CombinedExprFactory`）用于组装常见表达式模式（如 `SizeMinusActualSize` 尾块计算）。

<a id="template-semantic-naming"></a>

### 6.7 新增模板的语义命名规范

新增模板、实现策略、API 选择或回退分支，以及影响名称的分组、图改写和 reuse 映射变更，必须同步适配 `semantic_name`。仅调整 tile、核数、dtype 或 buffer 数量且未改变实现分类时，说明沿用既有名称的依据，不为数值组合新增分类。

名称格式如下，方括号表示可选部分，`|` 分隔多个 API feature，`+` 分隔成功转换：

```text
<family>/<strategy>[@<role>][#<api-feature>|...][+<transform>+...]
Reduce/RCore@Phase1
Reduce/Common@Norm#AR|NoReuseSrc
Concat/UB#OneAxis
IndirectLoad/SIMT@Strided#Offset32
```

`family` 表示实现家族，`strategy` 表示实现策略，`role` 表示逻辑阶段或职责，`api-feature` 表示最终 API 实现特征，`transform` 表示已成功的图或搬运转换。数值参数保留在原 tiling 信息中。协议使用 `semantic_schema=1`，字段和名称不加 `v2` 后缀。

- **统一入口**：词表统一定义在 `autofuse/common/common_utils.h` 的 `ascgen_utils::template_semantic` 中；枚举到名称的映射也使用公共 helper。实际选择点只引用公共常量，保持原选择条件。`common_utils.cpp` 的 `NormalizeTemplateSemanticName()` 在生成 `ModelInfo` 时规范化一次。ATT、TopN、summary 和离线工具消费规范值，不各建 role 表、重新分类或另写模板选型算法；不得从 Torch 算子名、测试标签、图名或生成源码猜测实现。
- **采集与传递**：Schedule 按[模板语义信息传递规则](schedule.md#template-semantic-propagation)记录策略、组级角色和成功转换；Codegen 在最终 API 选择成功点通过 `CodegenApiParam.semantic_feature` 或现有 `SetSemanticFeature` 记录 feature。回退后清理或替换旧信息，不能沿用失败分支的标签。
- **身份与生命周期**：名称必须对应当前调用、候选及完整的 graph/result/group/key/subcase/template 身份。reuse 后恢复当前逻辑身份；最终选择快照随胜出 tiling data 一起提交，不能只按 `group_id` 或共享 primary 最后一次选择推断。cache/PGO 无精确快照时按当前 key 回退，不借用历史状态，不补做求解。
- **不确定信息**：仅在运行时分支未定、独立 `GenerateTiling` 缺少最终 API 信息等明确条件下使用父名或 `Adaptive`，并给出条件及测试。身份匹配不唯一标 `ambiguous`，信息缺失标 `unavailable`；不得用默认父名或 `Adaptive` 掩盖接入遗漏。
- **行为与性能**：观测不改变公开 ABI、tiling/cache 数据、候选选择或真实求解异常；诊断失败仅降级观测。INFO 关闭时不做新增的诊断字符串和快照容器构造，不新增求解、图扫描、IO 或 kernel launch。沿用原功能及编译、tiling、kernel 性能 gate，不放宽阈值；INFO 关闭时测性能，开启时单独报告观测开销，未执行项明确列出。

新增模板或分支的设计及 PR 说明必须给出分类或继承理由、真实采集点及复制/失效/回退时机、完整身份与 reuse 处理、不确定状态，以及测试入口、实际分支、期望名称和代码/产物版本。不适用项说明原因。

复用现有 UT 和 backend/生成代码执行测试，验证实际 API 或分支、完整身份、名称及状态的对应关系，不能只检查非空字符串或文件存在。按本次变更覆盖复制、失败、回退、多 group/reuse（含跨 graph/result 同号 group）、同 key 多 subcase、cache/PGO 和 INFO 关闭及诊断失败等受影响边界。名称适配、设备执行、精度和性能分别记录；TopN 候选覆盖不等于设备执行或精度通过。

以上为开发和评审要求。模板注册项与名称测试的自动对应 CI 检查仍待实施，不维护另一份全量模板清单，也不能宣称当前 CI 已能自动拦截所有遗漏。

## 7. 关键文件索引

| 文件路径 | 职责 |
| --- | --- |
| `autofuse/codegen/codegen.h` / `codegen.cpp` | Codegen 门面类，总入口与产物组装 |
| `autofuse/codegen/codegen_kernel.h` / `codegen_kernel.cpp` | Kernel 生成器、Tiler/TPipe 数据模型、tiling_key 分发 |
| `autofuse/codegen/codegen_kernel_loop.h` / `codegen_kernel_loop.cpp` | Loop 循环树、ApiCall 基类与生成骨架 |
| `autofuse/codegen/codegen_tiling*.cpp` | Host 侧 tiling 生成（含 PGO、Inductor TopN、CV tiling） |
| `autofuse/codegen/codegen_tiling_data.cpp` | AutofuseTilingData 结构体生成 |
| `autofuse/codegen/codegen_infershape.cpp` | InferShape 函数生成 |
| `autofuse/codegen/codegen_graph_check.cpp` | ImplGraph 合法性校验 |
| `autofuse/codegen/expression_convert_struct.cpp` | 符号表达式到 C++ 表达式转换 |
| `autofuse/codegen/codegen_api_param_dump.cpp` | API 入参 dump 调试工具 |
| `autofuse/codegen/ascendc_api_registry.cpp` | AscendC API 源码模板注册表 |
| `autofuse/codegen/api_call/` | 主框架内置 ApiCall 实现（datacopy/elewise/reduce/transpose/gather/broadcast/concat） |
| `autofuse/v35/codegen/reg_api_call/` | RegBase 风格 V2 ApiCall（平台扩展层主体） |
| `autofuse/v35/codegen/cube_api_call/` | Matmul/Conv2D Cube ApiCall |
| `autofuse/v35/codegen/micro_api_call/` | MicroAPI 指令级 ApiCall 及独立工厂 |
| `autofuse/v35/codegen/vec_func_call/` | VectorFunc 子图封装（VfCall/VFLoop） |
| `autofuse/v35/codegen/ascendc_reg_base_api_register.cpp` | RegBase API 模板注册 |
| `autofuse/v35/codegen/ascendc_cube_api_register.cpp` | Cube API 模板注册 |

## 相关链接

- 返回 [AutoFuse 架构介绍](../../introduction/architecture.md)
