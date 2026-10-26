# Schedule

自动融合的 Schedule 模块是连接计算定义与高效代码生成的核心组件，其核心能力在于对计算过程进行灵活的重排与优化：在不改变计算语义的前提下，通过计算重排、调度原语调用等方式灵活调整计算实现，以获得更佳的性能。优化手段包括循环合并、解空间生成、并行优化、内存优化及多模板生成等。

当用户通过 AscIR 定义一张描述 Scalar 计算逻辑的 HintGraph 后，Schedule 模块会基于硬件特性对计算进行调度优化，生成多张表达切分及内存关系的 ImplGraph，为 Codegen 及 Auto Tiling 模块提供基础，支撑其生成高性能 Kernel 代码。

## 循环合并

循环合并是一种重要的循环变换技术，其核心作用是通过重构循环结构，减少内存访问次数、降低控制开销、提升数据局部性，并为后续优化铺路，最终在不改变计算结果的前提下提升程序执行效率。

例如，假设有如下两层循环:

```text
for i in range(N):
    for j in range(M):
        C[i][j] = A[i][j] + B[i][j]
```

Add 是逐元素（element-wise）操作，所以可以合并成一个线性循环：

```text
for fused in range(N * M):
    i = fused // M
    j = fused % M
    C[i][j] = A[i][j] + B[i][j]
```

对应到 AscIR 表达如下：

```python
z0 = graph.create_axis("z0", N)
z1 = graph.create_axis("z1", M)

Load load0("load_0");
load0.x = data0.y;
load0.attr.sched.axis = {z0.id, z1.id};
load0.y.axis = {z0.id, z1.id};
load0.y.repeats = {N, M};
load0.y.strides = {M, 1};

Load load1("load_1");
load1.x = data1.y;
load1.attr.sched.axis = {z0.id, z1.id};
load1.y.axis = {z0.id, z1.id};
load1.y.repeats = {N, M};
load1.y.strides = {M, 1};

Add add("add");
add.x1 = load0.y;
add.x2 = load1.y;
add.attr.sched.axis = {z0.id, z1.id};
add.y.axis = {z0.id, z1.id};
add.y.repeats = {N, M};
add.y.strides = {M, 1};
```

循环合并后：

```python
Load load0("load_0");
load0.x = data0.y;
load0.attr.sched.axis = {z0z1.id};
load0.y.axis = {z0z1.id};
load0.y.repeats = {N * M};
load0.y.strides = {1};

Load load1("load_1");
load1.x = data1.y;
load1.attr.sched.axis = {z0z1.id};
load1.y.axis = {z0z1.id};
load1.y.repeats = {N * M};
load1.y.strides = {1};

Add add("add");
add.x1 = load0.y;
add.x2 = load1.y;
add.attr.sched.axis = {z0z1.id};
add.y.axis = {z0z1.id};
add.y.repeats = {N * M};
add.y.strides = {1};
```

## 生成 TilingCase

在自动融合技术中，tiling 解空间生成是实现高效计算调度的关键环节，其核心目标是为复杂计算任务提供多样化的分块（tiling）策略选项，以便后续优化器从中筛选出最优解。简单来说，tiling 解空间生成的过程可以理解为对输入数据或计算任务进行"分块可能性"的系统枚举，每一个解空间被称为一个 TilingCase。

切分方式的设计与算子实现特性紧密关联，为了实现对多样化算子的系统性切分策略枚举，首先基于算子的实现特性将其抽象为 9 种 compute_type（如下图中的 compute 类和 view 类算子）。同一 compute_type 类别的算子具有相似的计算逻辑与数据访问模式，因此可以共享一套 tiling 切分策略框架。

<div style="text-align: center;">
<img src="../../figures/schedule_tiling_strategy.png" alt="Tiling 切分策略" style="width: 50%; max-width: 800px;">
</div>

为具象化这一策略框架，我们对算子的轴进行归一化分组，将所有轴统一划分为（xgroup、ygroup、rgroup）三个维度集合，具体定义如下：

- **xgroup**：专为 Concat 等视图类算子设计的分组。以 Concat 为例，会以 Concat 轴进行划分，将 Concat 轴前的轴划分到 xgroup，将 Concat 轴及其以后的轴划分到 ygroup。
- **ygroup**：对应 Elementwise、Broadcast 等类型算子的循环轴分组。
- **rgroup**：Reduce 类操作通常对 reduce 轴有特殊的切分要求，因此会将所有 reduce 轴单独放入 rgroup。

> **说明**：引入 xgroup、ygroup、rgroup 的核心原因是为了支持复杂场景下的"双切分"需求。例如，在包含 reduce 混合的计算图中，ygroup 控制 elementwise 的循环切分，rgroup 中的轴控制 reduce 方向的循环切分。

在完成单算子的轴分组后，需通过预设的合并规则（Merge）对计算图中所有算子的分组策略进行合并。合并结果将作为适用于全图所有算子的统一切分策略，为后续解空间生成提供基础。

通过上述分组与合并机制，可实现两大核心功能：

- 筛选出适用于计算图中所有节点的切分方式，形成有效的 TilingCase。
- 通过判断不同 AscGraph 的 tiling 分组是否能成功合并，验证两张图的可融合性。

以下面 case 为例，介绍通过 tiling 分组合并生成 TilingCase 的原理：

```python
z0 = graph.create_axis("z0", s0)
z1 = graph.create_axis("z1", s1)

data = ascir.ops.Data('data', graph)
data.y.dtype = ascir.dtypes.float32

# 声明 Load 算子
load = ascir.ops.Load('load')
load.attr.sched.axis = [z0, z1]  # 调度轴
load.x = data.y
load.y.axis = [z0, z1]  # tensor的输出轴
load.y.size = [s0, s1]  # tensor输出大小
load.y.strides = [s1, 1]  # tensor的输出步长

# 声明 Abs 算子
abs = ascir.ops.Abs('abs')
abs.attr.sched.axis = [z0, z1]
abs.x = load.y
abs.y.axis = [z0, z1]
abs.y.size = [s0, s1]
abs.y.strides = [s1, 1]

# 声明 Max 算子
max = ascir.ops.Max('max')
max.attr.sched.axis = [z0, z1]
max.x = abs.y
max.y.axis = [z0, z1]
max.y.size = [s0, 1]  # 对z1轴进行reduce操作
max.y.strides = [1, 0]
```

abs 是一个 elewise 算子，每个轴在计算上并无差异，因此只要内存连续，可以将多根轴合并成一根轴进行切分：例如下图所示，先在轴上做 block 切分，15 被分到 3 个 block 上，block0 为紫色的部分，在 block0 内再进行 tiling 分块，此时 tiling 块未占满 block0 分配的部分，因此还需要在 block 内加一个 for 循环。

<div style="text-align: center;">
<img src="../../figures/schedule_elewise_split.png" alt="elewise 类算子切分" style="width: 45%; max-width: 800px;">
</div>

reduce 类的切分较为复杂，实现上需要双切分：行方向上是 elewise 轴，列方向上是 reduce 轴；首先在行方向上分 block，block 内写循环，然后在列方向上再加一层 for 循环。

<div style="text-align: center;">
<img src="../../figures/schedule_reduce_split.png" alt="reduce 类算子切分" style="width: 45%; max-width: 800px;">
</div>

通过 TilingGroup 的合并规则:

```text
()(z0,z1)()  Merge  ()(z0)(z1)  =>  ()(z0)(z1)
```

将 abs 算子原本同属 ygroup 的 (z0，z1) 轴进行拆分，其中 z1 轴被调整至 rgroup，这一调整的核心目的是使 abs 算子与后续的 reduce 类型算子保持统一的切分策略。

## 并行优化

### 循环拆分

循环拆分的核心作用是通过引入新的循环层级，明确划分出适合并行的外层循环和适合向量化的内层循环。

针对每个 TilingCase，会将 xgroup、ygroup、rgroup 中存在的轴切分成 ub_out 和 ub_in：

<div style="text-align: center;">
<img src="../../figures/schedule_loop_split.png" alt="循环拆分" style="width: 50%; max-width: 800px;">
</div>

例如：{z0, z1, z2}，切分在 z1 上，就把 z1 切分成 z1T 和 z1t，z1T 就是 ub_out，z1t 就是 ub_in。

### 向量化

向量化是利用硬件 SIMD（Single-Instruction Multiple-Data stream processing，单指令流多数据流）单元提升数据并行计算效率的关键技术，通过将单元素操作转化为向量操作，显著减少指令执行次数并提高硬件利用率。

例如，对于如下循环：

```text
for (int i = 0; i < 256; i++) {
    c[i] = a[i] + b[i];
}
```

非向量化执行需要 256 条加法指令，向量化后只需要一条加法指令。

在每个组内选择一根轴作为 ub 切分轴后，会将 ub_in 及其内轴作为向量化轴。此时，由于分组是按照 xyr 来生成的，按照这个顺序生成的向量化轴，轴序与内存排布不一致会引入非连续搬运，因此需要根据输出的轴序进行重排列。

<div style="text-align: center;">
<img src="../../figures/schedule_vectorize_reorder.png" alt="向量化重排列" style="width: 60%; max-width: 800px;">
</div>

例如，输出 tensor 轴序是 (a, b, c, d)，轴分组是 (a, c),(b, d)()，按照这个顺序生成的向量化轴分组是 (a_in, c, b_in, d)，需要调整成 (a_in, b_in, c, d)。

> **说明**：Schedule 阶段全图统一设置了相同的向量化轴，对于部分 API 来说，由于指令等限制，并不能将所有向量化轴都进行向量化处理。此时，需要 CodeGen 阶段对无法向量化处理的轴进行外抛 for 循环处理。
>
> 例如向量化轴为 [z1,z2,z3]，相当于 3 层循环，如果指令支持 3 层循环，则可以写成 `vector[z1,z2,z3]`；但如果指令只能支持两层循环，则需要 CodeGen 给出如下的代码：
>
> ```text
> for (i in z1) {
>     vector[z2,z3]
> }
> ```

### 循环合并、循环绑核

循环合并与循环绑核二者常常配合优化，循环切分阶段在每个分组内都产生了 ub_out 的轴。

1. 将所有非 reduce 轴合并为一根轴，所有 reduce 轴合并为一根轴以减少循环嵌套层数。
2. 对合并后的循环进行拆分，得到外层和内层。
3. 将拆分后的外层循环绑定到多个 block 上以实现并行，内层通过循环进行消化。

<div style="text-align: center;">
<img src="../../figures/schedule_loop_merge_bindcore.png" alt="循环合并与循环绑核" style="width: 60%; max-width: 800px;">
</div>

例如：{z0, z1T, z1t, z2}，会先把 z0、z1T 合轴成 z0z1T，再在 z0z1T 上进行多核切分，由于 z0z1T 可能会超过参与计算的逻辑 AI Core 核数，因此在分完核后还要多一层循环，z0z1T 会被进一步拆成 z0z1TB 和 z0z1Tb，值的具体大小由 Auto Tiling 在 Tiling 阶段计算得出。

## 内存优化

目前主要依据节点引用关系进行内存复用，为了提升复用效果，会尽量将占用大小相近的内存分配到同一个 group 内，然后在 group 内进行复用。

内存复用伪代码如下：

```python
for (node in all nodes) {
    for (output in node.outputs) {
        # 标记tensor的依赖数
        output->sch.depends = output->anchor->GetPeerInDataNodesSize();
        # try reuse from free queue
    }
    for (input in node.inputs) {
        input->sch.depends--;
        if (input->sch.depends == 0) {
            Enqueue(input->opt.reuse_id); # 标记为freeTensor，可以被后续节点复用
        }
    }
}
```

对于部分 API 来说，输出可以直接复用输入，针对这一类 API，可以采用 Inplace 复用，即输出直接复用输入内存。如下图所示，Inplace 复用前需要 3 块内存：

<div style="text-align: center;">
<img src="../../figures/schedule_mem_reuse_before.png" alt="内存复用前" style="width: 25%; max-width: 800px;">
</div>

Inplace 复用后只需要 2 块内存：

<div style="text-align: center;">
<img src="../../figures/schedule_mem_reuse_after.png" alt="内存复用后" style="width: 25%; max-width: 800px;">
</div>

## 多模板生成

针对一张计算图，可能存在多种实现方式。以尾轴 concat 为例，可以在 UB 上将多个小包做 ub_concat 先组成大包再完整搬出，也可以直接转成非连续搬运在 GM（Global Memory，全局内存）上完成重排。前者在小 shape 场景可以显著提高 MTE（Memory Transfer Engine，AI Core 的数据传递引擎）搬运效率，从而获得更好的性能优势。但 ub_concat 也存在需要内轴全载的限制，导致某些场景下无法使用。在 Schedule 阶段无法确定选择哪个模板时，通常会生成一个适用于任意 shape 的通用模板，以及特定场景下的性能优化模板，由 Auto Tiling 模块在 tiling 阶段决定具体使用哪个模板。

- UB concat 模板：

<div style="text-align: center;">
<img src="../../figures/schedule_ub_concat_template.png" alt="UB concat 模板" style="width: 35%; max-width: 800px;">
</div>

- 改图模板：

<div style="text-align: center;">
<img src="../../figures/schedule_graph_rewrite_template.png" alt="改图模板" style="width: 35%; max-width: 800px;">
</div>

<a id="template-semantic-propagation"></a>

## 新增模板的语义信息传递

新增模板及影响实现分类的变更必须遵循 [Codegen 语义命名规范](codegen.md#template-semantic-naming)。Schedule 负责记录真实调度结果，不推测只有最终 Codegen API 才能确认的 feature。

- 在实际 generator/策略选择点写入 `ScheduleTask.semantic_name`，传递到 `ScheduledResult.semantic_name`；继续复用已有图标签携带必要的角色和转换信息。
- role 只作用于受影响的逻辑 group；图级父策略不得覆盖 `Phase1`、`Phase2`、`Norm` 等组级职责。多 group 可有不同名称，不要求整图名称一致。
- 图或搬运转换成功后再记 transform；失败不得污染原图或其它候选，复制/派生必须保留对应标签，后续回退必须反映最终结果。
- 保留完整 graph/result/group/key/subcase/template 归属；reuse 恢复当前逻辑身份，不以局部 `group_id` 或物理 primary 作为唯一标识。候选私有选择状态随最终 tiling data 一起提交。
- PR 说明给出分类或继承依据及上述信息来源，复用现有测试验证本次受影响的选择、复制、失败、回退和多 group/reuse 边界；最终 API feature 的适配按 Codegen 规范完成。

## 相关链接

- 返回 [AutoFuse 架构介绍](../../introduction/architecture.md)
