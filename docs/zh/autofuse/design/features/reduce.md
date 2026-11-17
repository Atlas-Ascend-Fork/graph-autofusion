# Reduce算子

## Reduce算子简介

### 1. 什么是Reduce算子

Reduce（Reduction/规约）算子是一类沿指定轴对输入张量进行聚合计算的算子。它在指定轴上遍历所有元素，用一个二元操作（如加法、取最大值）逐步将多个值"归约为"一个值，使输出在该轴上的维度消失（或变为1）。

**典型示例**：

```
输入 x: shape [2, 3] = [[1, 2, 3],
                          [4, 5, 6]]

ReduceSum(x, axis=1, keepdims=False) → shape [2] = [6, 15]
ReduceMax(x, axis=0, keepdims=False) → shape [3] = [4, 5, 6]
ReduceMean(x, axis=None, keepdims=False) → shape [] = 3.5   # 全轴规约
```

### 2. Reduce算子在框架中的接口形式

#### TensorFlow接口

TensorFlow将各类Reduce算子统一为 `tf.reduce_*` 系列，接口形式高度一致：

| 接口 | 计算语义 |
|------|---------|
| `tf.reduce_sum(input_tensor, axis, keepdims)` | 求和 |
| `tf.reduce_mean(input_tensor, axis, keepdims)` | 求均值 |
| `tf.reduce_max(input_tensor, axis, keepdims)` | 取最大值 |
| `tf.reduce_min(input_tensor, axis, keepdims)` | 取最小值 |
| `tf.reduce_prod(input_tensor, axis, keepdims)` | 求连乘 |
| `tf.reduce_all(input_tensor, axis, keepdims)` | 逻辑全与（bool） |
| `tf.reduce_any(input_tensor, axis, keepdims)` | 逻辑全或（bool） |

**共同参数**：

- `input_tensor`：输入张量
- `axis`：规约轴，可为 `None`（全轴规约）、整数、整数列表或元组
- `keepdims`：是否保留规约轴（默认 `False`，轴消失；`True` 时轴变为1）

#### PyTorch接口

PyTorch同样提供 `torch.*` 系列Reduce函数，并额外提供 `dim` 和 `out` 参数：

| 接口 | 计算语义 |
|------|---------|
| `torch.sum(input, dim, keepdim)` | 求和 |
| `torch.mean(input, dim, keepdim)` | 求均值（仅浮点） |
| `torch.max(input, dim, keepdim)` | 取最大值（dim指定时返回值+索引） |
| `torch.min(input, dim, keepdim)` | 取最小值（dim指定时返回值+索引） |
| `torch.prod(input, dim, keepdim)` | 求连乘 |
| `torch.all(input, dim, keepdim)` | 逻辑全与（bool） |
| `torch.any(input, dim, keepdim)` | 逻辑全或（bool） |
| `torch.argmax(input, dim, keepdim)` | 取最大值索引 |
| `torch.argmin(input, dim, keepdim)` | 取最小值索引 |

**与TF的主要区别**：

- 参数名不同：`axis` → `dim`
- `torch.max/min` 在指定dim时会同时返回 `(values, indices)`，TF需额外调用 `tf.argmax/argmin`
- PyTorch的 `mean` 仅支持浮点类型，整型需先转换

#### GE中的Reduce算子类型

GE在Ascend IR层面定义了以下Reduce算子类型，并与框架算子对应：

| GE算子类型 | AscIR融合层类型 | 对应框架算子 |
|-----------|---------------|------------|
| ReduceSum | Sum | tf.reduce_sum / torch.sum |
| ReduceMean | Mean | tf.reduce_mean / torch.mean |
| ReduceMax | Max | tf.reduce_max / torch.max |
| ReduceMin | Min | tf.reduce_min / torch.min |
| ReduceProd | Prod | tf.reduce_prod / torch.prod |
| ReduceAll | All | tf.reduce_all / torch.all |
| ReduceAny | Any | tf.reduce_any / torch.any |
| ArgMax | ArgMax | tf.argmax / torch.argmax |

此外，GE还有带"D"后缀的变体（ReduceSumD、ReduceMaxD等），其axes信息通过属性传递而非第二个输入张量。

#### Prod数据类型支持

AscIR V2路径下，Prod支持`int8`、`int16`、`int32`、`int64`和`float32`：

| 数据类型 | 执行类型 | 说明 |
|---------|---------|------|
| `int8` | `int16` | 输入加宽为`int16`执行，输出按低位截断回绕为`int8` |
| `int16` | `int16` | 直接执行 |
| `int32` | `int32` | 直接执行 |
| `int64` | `int64` | 使用b64双寄存器分支执行 |
| `float32` | `float32` | 直接执行 |

`int8`执行结果按低位截断回绕到`int8`。例如`100 * 100 = 10000`输出为`16`，而不是饱和值`127`。

`bool`和`uint8`本期不支持。3510类架构的`Reg::Mul`不支持8位类型，相关输入保持原有不融合行为。

#### Any/All数据类型支持

AscIR V2路径下，Any/All支持`bool`、`uint8`、`int8`、`int16`、`int32`、`int64`和`float32`：

| 数据类型 | 执行类型 | 说明 |
|---------|---------|------|
| `bool` | `uint8` | 按0/1位模式零拷贝重解释执行 |
| `uint8` | `uint8` | 直接执行 |
| `int8` | `int8` | 直接执行 |
| `int16` | `int16` | 直接执行 |
| `int32` | `int32` | 直接执行 |
| `int64` | `int64` | 使用b64双寄存器分支执行 |
| `float32` | `float32` | 直接执行 |

逻辑输入需满足0/1值域：非bool输入进入Any/All前应先完成非零归一化。`uint16`、`uint64`、`fp16`、`bf16`和`fp64`不注册，相关输入保持原有不融合行为。

### 3. 计算逻辑

Reduce算子的核心计算逻辑可统一描述为：

```
output[i] = init_value ⊙ x[i, j₁, j₂, ..., jₖ]  (沿规约轴j遍历所有元素)
```

其中：

- `⊙` 是二元聚合操作（`+` for Sum/Mean, `max` for Max, `min` for Min, `×` for Prod, `&&` for All, `||` for Any）
- `init_value` 是聚合的初始值（Sum→0, Max→-∞, Min→+∞, Prod→1, All→True, Any→False）
- `i` 是非规约轴（A轴）上的索引，`j` 是规约轴（R轴）上的索引
- Mean 的计算为 `Sum / R轴元素总数`，即先求和再除以规约轴长度

**ArgMax的特殊逻辑**：

```
output[i] = argmax_j(x[i, j])  # 返回使x[i,j]最大的索引j，而非值本身
```

### 4. 与Element-wise算子的本质区别

| 维度 | Element-wise算子 | Reduce算子 |
|------|-----------------|-----------|
| **轴变换** | 轴不变，输出shape = 输入shape | 轴减少，规约轴消失或变为1 |
| **数据量** | 输出数据量 = 输入数据量 | 输出数据量 < 输入数据量（沿R轴压缩） |
| **计算依赖** | 每个输出元素仅依赖同一位置的输入元素 | 每个输出元素依赖R轴上**所有**输入元素 |
| **并行模式** | 完全独立，每个元素可单独计算 | A轴可并行，R轴必须串行聚合 |
| **访存特征** | 1读1写，无中间数据 | 多读1写，需要中间累加buffer |
| **融合角色** | 可向前/向后融合，可水平融合 | 只能向后融合elementwise（最多3个），不支持水平融合 |
| **硬件对齐** | 无特殊对齐限制 | 尾轴必须32B对齐；仅支持AR或RA模式 |
| **初始值** | 无需 | 必须有初始值（0、-∞、+∞、1等） |

**核心差异总结**：Element-wise算子是"逐元素变换"，输入输出一一对应，天然可并行；Reduce算子是"跨元素聚合"，一个输出依赖多个输入，存在R轴上的串行依赖和累加过程，这使得它在并行调度、内存布局、融合策略上都需要特殊处理。

---

## 自动融合中的核心概念

### 1. A轴与R轴

Reduce算子将输入张量的轴分为两类：

- **A轴（Axis/Active轴）**：不被规约的轴，在输出中保留。A轴上的元素可以独立并行计算——不同的A轴索引对应不同的输出元素，彼此无依赖关系。
- **R轴（Reduce轴）**：被规约的轴，在输出中消失（`keepdims=False`）或变为1（`keepdims=True`）。R轴上的所有元素必须聚合为一个值，存在串行依赖。

**示例**：
```
输入: shape [B, M, K]，对 axis=2 做 ReduceSum

A轴 = {B, M}   → 输出中保留，shape 中的 B 和 M 维度不变
R轴 = {K}       → 输出中被消除

输出: shape [B, M]，每个 output[b, m] = sum(x[b, m, :])  // K个元素求和
```

**多轴规约**：当 `axis` 指定多个轴时，所有指定轴均为R轴，其余为A轴。例如 `ReduceSum(x, axis=[0,2])` 对 shape `[B, M, K]` 操作，R轴={B, K}, A轴={M}，输出 shape `[M]`。

**Norm-like判断**：自动融合中，R轴总大小 ≤ 32 且 A轴总大小 ≥ 128 的Reduce被称为 **Norm-like Reduce**，这类场景有专门的优化路径（AllLoad模板，R轴一次性加载）。

### 2. 向量化轴与"全局一套轴"原则

#### 向量化轴

在自动融合中，每个Tensor有两组轴描述：

- **全轴（axis）**：Tensor逻辑上包含的所有轴，以及对应的 `repeats`（轴大小）和 `strides`（轴步长）
- **向量化轴（vectorized_axis）**：**UB（Unified Buffer）中实际存储的数据所对应的轴**——是全轴的子集

多核切分后，一个Tensor在单个核的UB中可能只保留了部分轴的数据（block内轴），block外轴通过循环展开来遍历。`vectorized_axis` 描述的就是UB中实际驻留的那些轴。

**向量化步长（vectorized_strides）**：与 `vectorized_axis` 一一对应，表示在UB内按向量化轴索引时的步长：
- `stride = 0`：该轴是广播轴，在UB中不占存储空间（数据沿该轴复制，无需独立存储）
- `stride ≠ 0`：该轴在UB中有实际数据，索引时需要跳过 stride 个元素

步长从尾到头累积计算：尾轴 stride=1（或对齐后的1），中间轴若为广播轴则 stride=0，否则 stride=前面所有轴的累积大小。

#### 全局一套轴原则

自动融合的基础原则：**融合后的所有算子必须共享同一套轴ID空间**。

融合前，各算子有独立的轴ID。融合时通过轴映射将所有算子统一到一套轴ID：
- **垂直融合**：前序节点输出轴 → 后序节点输入轴，按轴大小值相等进行匹配
- **水平融合**：相同输入的两个节点之间，数据轴按轴大小匹配
- 匹配完成后，刷新所有节点的轴属性为统一ID

#### Reduce在全局一套轴中的特殊性

Element-wise算子满足"轴不变"，天然与前后算子共享同一套轴。Reduce算子则不同：
- 输入的R轴在输出中消失（或变为1），导致输出轴集合是输入轴集合的子集
- 融合时需要将Reduce的输入轴映射到后续elementwise算子的轴，R轴需通过广播（stride=0）恢复为与后续算子一致的轴体系
- 这也是为什么Reduce只支持向后融合elementwise：后续算子的轴体系是Reduce输出的子集或等集，映射可行；向前融合则可能因轴数量不匹配而无法统一

### 3. AscendC接口

AscendC是昇腾AI处理器的底层编程接口，提供直接操作硬件计算单元的C++ API。自动融合的CodeGen模块最终生成的内核代码就是AscendC代码。

#### 核心特点

- **LocalTensor**：所有AscendC API操作的数据都在 `LocalTensor<T>` 上，这是UB上的局部内存，大小有限，数据需从GM搬运到UB后才能计算
- **模板化**：大部分API以模板形式提供，参数包括数据类型 `T`、模式 `pattern`、是否复用源 `isReuseSource` 等
- **显式同步**：数据搬运和计算之间需要通过 PipeBarrier 等屏障指令显式同步

#### Reduce相关AscendC接口

自动融合为Reduce算子封装了以下AscendC接口：

| 接口 | 功能 |
|------|------|
| `WholeReduceSumAdapt` | 沿last轴求和适配层 |
| `ReduceLast` | 沿最后一个轴做通用reduce（可指定聚合函数） |
| `ReduceSumInt32` | int32专用reduce sum |
| `ReduceInit` | 填充padding初值（Reduce前初始化） |
| `ReduceProdExtend` | 连乘reduce（支持AR/RA双模式） |
| `ReduceMaxExtend` | int32专用reduce max |
| `ArgMaxWithValueExtend` | ArgMax（同时输出值和索引） |

**ReduceInit的padding初值**（每种Reduce类型有对应的初始值）：

| Reduce类型 | padding值 |
|-----------|----------|
| Sum / Mean | 0 |
| Max | -INF |
| Min | +INF |
| Prod | 1 |
| All | True |
| Any | False |

#### AR/RA模式

AscendC的Reduce API只支持两种数据排布模式：

- **AR模式**：数据布局为 `[A轴..., R轴]`，沿最内层（尾轴）做reduce。例如 `[M, K]` 沿K轴reduce，每行的K个元素连续排列，逐行求和
- **RA模式**：数据布局为 `[R轴..., A轴]`，沿最外层（首轴）做reduce。例如 `[K, N]` 沿K轴reduce，N个元素跨K行分布，逐列求和

**判断方式**：通过输出Tensor的向量化步长判断——尾轴 stride=0 表示尾轴是R轴（被消除），即AR模式；尾轴 stride≠0 表示尾轴是A轴（保留），即RA模式。

**AR模式需要ReduceInit**：沿尾轴reduce时，尾轴大小可能不是32B对齐的，需要先填充padding初值到非对齐区域再执行reduce。RA模式沿首轴reduce，不需要padding初始化。

### 4. 对齐

#### 什么是对齐

昇腾AI处理器的向量计算单元要求数据访问地址以 **32字节（32B）** 对齐，这是DMA搬运和向量指令的基本硬件约束。32B是1个Block的大小，是所有向量操作的原子访问粒度。

不同数据类型下的对齐元素数：

| 数据类型 | 单元素大小 | 32B对齐元素数 |
|---------|----------|-------------|
| float32 | 4B | 8 |
| float16 | 2B | 16 |
| int32 | 4B | 8 |
| int8 | 1B | 32 |

**含义**：一个float32的Tensor，尾轴元素数必须是8的倍数，才能被向量单元高效访问。

#### Reduce场景的对齐处理

Reduce是对齐处理最复杂的场景之一，主要涉及以下策略：

**1. 尾轴合轴**：当尾轴元素数不足32B对齐时，将倒数第二轴与尾轴合并，使新尾轴满足对齐要求。

**2. ReduceInit填充padding**：AR模式下，R轴大小不满足32B对齐时，在UB中填充对应类型的padding初值（如Sum填0、Max填-INF），使数据在对齐后的范围内可以正确规约。

**3. 对齐传播**：Reduce输出的A轴需要反向传播对齐要求到输入端，确保输入的尾轴满足32B对齐。

**4. RemovePad**：对于非连续排布的数据，自动融合会插入 RemovePad 节点，在reduce前将数据重新紧凑排列，消除padding间隙。

## Reduce特殊处理的根源

自动融合中所有针对Reduce的特殊处理，都源自两个方面的限制：**Reduce算子本身的计算特性** 和 **底层AscendC接口的约束**。理解这两个根源，就能将后面所有模块的Reduce处理逻辑串联起来。

### 限制一：Reduce算子本身的计算特性

#### 1. 经过Reduce之后轴变少

Reduce沿指定轴聚合后，R轴在输出中消失（或变为1），输出shape是输入shape的子集。这带来了以下连锁影响：

- **轴映射困难**：Element-wise算子输入输出轴不变，天然适配"全局一套轴"。Reduce输出轴少了，后续算子若要与Reduce融合，R轴必须以广播形式（stride=0）重新出现在统一的轴体系中，增加了映射的复杂度
- **融合方向受限**：Reduce只能向后融合elementwise，不能向前融合——因为Reduce输出的轴集合是输入的子集，向后融合时后续算子的轴可以通过广播补齐，但向前融合时前序算子的轴集合更大，无法将多出的轴映射到Reduce的输出
- **不支持水平融合**：水平融合要求两个算子共享同一输入并拥有相同的轴集合，但Reduce与其他算子的输出轴数量不同，无法统一

#### 2. R轴要都算完才能出结果

Reduce的每个输出元素依赖R轴上所有输入元素的聚合结果，不存在部分计算即可输出的可能。这带来了：

- **两阶段执行**：R轴较大时无法一次性在UB中完成规约，需要分多次加载R轴数据并逐次累加——第一阶段沿R轴分块累加，第二阶段合并中间结果得到最终值
- **串行依赖**：R轴上的计算必须串行执行，无法像elementwise那样完全并行。A轴可以并行，但同一个A轴位置上的R轴遍历必须顺序完成
- **中间buffer需求**：两阶段执行需要在UB中分配临时buffer存储中间累加结果，增加了UB空间的规划复杂度
- **ArgMax更复杂**：ArgMax不仅要追踪最大值，还要追踪最大值的索引，两阶段合并时索引和值必须一起传递，比单纯数值规约多一倍中间数据

### 限制二：底层AscendC接口的约束

#### 1. 仅支持AR或RA形式的入参

AscendC的Reduce API要求输入数据排布必须满足两种模式之一：要么A轴在前R轴在后（AR），要么R轴在前A轴在后（RA）。不支持其他轴顺序（如RAA、ARA等混合排布）。这带来了：

- **可能需要插入Transpose**：如果原始数据既不是AR也不是RA排列，自动融合需要在Reduce前插入Transpose将轴重排为AR或RA
- **AR/RA选择影响执行策略**：AR模式沿尾轴规约，数据在UB中连续排列时效率更高；RA模式沿首轴规约，适合R轴一次性加载的Norm-like场景
- **合轴必须保持AR/RA顺序**：尾轴合轴（将小尾轴与相邻轴合并以满足32B对齐）时，合轴后的新轴顺序仍需满足AR或RA，不能破坏模式约束

#### 2. 尾轴要32B对齐

AscendC向量指令以32B为原子访问粒度，要求操作数据的尾轴大小必须是32B对齐的整数倍。这带来了：

- **尾轴合轴**：当尾轴元素数不足32B对齐时，需要将尾轴与倒数第二轴合并，使新尾轴满足对齐要求
- **ReduceInit填充padding**：AR模式下沿尾轴规约时，如果R轴（尾轴）大小不满足32B对齐，需要在UB中填充padding初值到对齐边界。不同Reduce类型的padding初值不同（Sum→0, Max→-INF, Prod→1等），以保证padding不影响规约结果
- **对齐传播**：Reduce输出的A轴对齐要求需要反向传播到输入端，确保整个融合子图的尾轴都满足32B对齐
- **RemovePad**：非连续排布的数据在规约前需要紧凑排列，消除对齐带来的间隙

### 根源与处理的对应关系

将两条根源与具体的处理措施对应起来：

| 根源限制 | 具体表现 | 衍生的处理措施 |
|---------|---------|-------------|
| 轴变少 | 输出轴是输入轴的子集 | 融合方向受限（只向后）、不支持水平融合、R轴广播恢复 |
| 轴变少 | 轴映射困难 | 全局一套轴中的特殊映射策略 |
| R轴串行 | 全部R轴算完才出结果 | 两阶段执行（Common/RCore模板）、中间buffer分配 |
| R轴串行 | ArgMax需追踪索引 | ArgMax两阶段（Phase1局部最大+索引，Phase2合并） |
| 仅AR/RA | 不支持其他轴顺序 | 插入Transpose、合轴保持AR/RA约束 |
| 仅AR/RA | 模式选择影响效率 | Norm-like判断→AllLoad模板（R轴一次性加载） |
| 32B对齐 | 尾轴不足对齐因子 | 尾轴合轴 |
| 32B对齐 | AR模式R轴非对齐 | ReduceInit填充padding |
| 32B对齐 | 对齐要求需全局满足 | 对齐反向传播、RemovePad |

后续模块（Lowering、CanFuse、Schedule、CodeGen）的具体实现，本质上都是在不同阶段应对上述限制：Lowering识别Reduce并标记类型，CanFuse判断能否在轴变少的约束下融合，Schedule规划两阶段执行策略和R轴切分方式，CodeGen生成满足AR/RA和32B对齐的AscendC代码。

## 模块处理详解

### Lowering模块

#### 1. 功能概述

Lowering模块将GE计算图中的Reduce算子转换为AscIR中间表示，生成规约计算的计算单元。

#### 2. 主要接口

**`StoreReduction` API**：
```cpp
loop::StoreReduction(loop::ReduceType::SUM, node->GetOutDataAnchor(0), z,
                     {batch_size, m_size, k_size}, {2});  // 规约轴索引
```

**ReduceType类型**：
- `SUM` → Sum/Mean算子
- `MAX` → Max算子
- `MIN` → Min算子
- `PROD` → Prod算子

#### 3. 实验性功能开关

```cpp
if (meta->type == FuseType::kReduction &&
    !ge::AutoFuseConfig::LoweringConfig().experimental_lowering_reduce) {
    GELOGI("Drop lower result... you can enable it by setting "
           "AUTOFUSE_FLAGS=\"--autofuse_enable_pass=reduce\"");
    meta->type = FuseType::kExtern;  // 转换为外部调用
}
```

**开启方式**：
```bash
export AUTOFUSE_FLAGS="--autofuse_enable_pass=reduce"
```

#### 4. MatMul到Reduce的Lowering

当MatMul满足特定条件时，可lowering为Reduce：

```cpp
// [BS, M, K] * [BS, K, 1] -> [BS, M, 1]
auto x = loop::Load(x_anchor);
auto y = loop::Load(y_anchor);
y = transpose_b ? y : loop::Permute(y, {0, 2, 1});  // [BS, K, 1] -> [BS, 1, K]
y = loop::Broadcast(y, {batch_size, Symbol(1), k_size}, {batch_size, m_size, k_size});
const auto z = loop::Mul(x, y);  // [BS, M, K] * [BS, M, K]
loop::StoreReduction(loop::ReduceType::SUM, node->GetOutDataAnchor(0), z,
                     {batch_size, m_size, k_size}, {2});  // K轴规约
```

**条件检查**：
- `n_size == 1`（MatMul的N维度为1）
- `k_size <= max_k_for_vectorize_mm`（K维度不超过阈值）
- `transpose_a == false`（A矩阵不转置）

---

### CanFuse模块

#### 1. 功能概述

CanFuse模块判断Reduce算子能否与其他算子融合，决定融合范围。

#### 2. ReduceFusionStrategy核心逻辑

```cpp
bool ReduceFusionStrategy::CanFuse(const NodePtr &node1, const NodePtr &node2) {
  // 1. 检查并初始化norm-like reduce状态
  CheckAndInitReduceAllLoadState(node1, attr1, node1_desc);
  CheckAndInitReduceAllLoadState(node2, attr2, node2_desc);

  // 2. 检查norm-like状态是否允许融合
  if (attr1->GetReduceAllLoadState() != REDUCE_ALL_LOAD_NOT_ALL &&
      attr2->GetReduceAllLoadState() != REDUCE_ALL_LOAD_NOT_ALL) {
    return true;
  }

  // 3. Reduce不支持水平融合
  if (BackendUtils::IsHorizontal(node1, node2)) {
    GELOGI("Reduce cannot fuse horizontally");
    return false;
  }

  // 4. Reduce只能向后融合elementwise节点
  if (attr1->HasFuseType(loop::FuseType::kReduction)) {
    if (!BackendUtils::IsOnlyPointwise(node2)) {
      GELOGI("Reduce can only backward fuse with elementwise");
      return false;
    }
    // 5. 最多向后融合3个elementwise
    if (attr1->GetReduceFusedElementwiseNodeNum() +
        BackendUtils::GetComputeNodeNumInAscgraph(node2) >
        config.max_reduce_can_fuse_elementwise_nums) {
      GELOGI("Reduce can only backward fuse with at most 3 elementwise");
      return false;
    }
  }
  return true;
}
```

#### 3. 融合规则总结

| 规则 | 说明 |
|------|------|
| **不支持水平融合** | Reduce之间、Reduce与其他算子不能水平融合 |
| **只能向后融合** | Reduce后最多融合3个elementwise节点 |
| **norm-like优先** | 满足norm-like条件的Reduce有特殊融合路径 |

#### 4. Norm-like Reduce检查

```cpp
bool CheckReduceNodeNormLike(const ge::AscNodePtr &asc_node) {
  constexpr int64_t kThresholdTR = 32;    // R轴阈值（实际上由计算得出）
  constexpr int64_t kThresholdTA = 128;   // A轴阈值（实际上>=16）

  // 计算R轴和A轴总大小
  int64_t r_axis_total_size = 1;
  int64_t a_axis_total_size = 1;
  CalculateRAxisTotalSize(*input_attr_ptr, output.attr,
                          r_axis_total_size, a_axis_total_size);

  // 检查条件
  // R轴 <= 65536 (实际threshold根据计算调整)
  // A轴 >= 16 (实际threshold根据计算调整)
  if (r_axis_total_size > kThresholdTR || a_axis_total_size < kThresholdTA) {
    return false;
  }
  return true;
}
```

**Norm-like Reduce定义**：
- R轴（规约轴）总大小较小（≤65536）
- A轴（非规约轴）总大小较大（≥16）
- 典型场景：LayerNorm、Softmax等

---

### Schedule模块

#### 1. 功能概述

Schedule模块为Reduce算子生成调度方案，决定R轴如何切分、分核策略等。

#### 2. Reduce模板类型

```cpp
// reduce_schedule_case_generator.h
enum class ReduceTemplateType {
  kCommon,    // 通用模板：R轴切分，多阶段执行
  kAllLoad,   // 全载模板：一次性加载所有R轴数据
  kRCore      // R轴分核模板：R轴分到多个核执行
};
```

#### 3. 模板选择逻辑

```cpp
Status ReducePartitionCaseGenerator::GeneratorTask(...) {
  bool is_norm_like_reduce = optimize::NormLikeReduceChecker::IsNormLikeReduceGraph(optimize_graph);

  if (is_norm_like_reduce) {
    // Norm-like场景：仅生成AllLoad模板
    GELOGI("Graph satisfies norm-like reduce, only generate AllLoad tasks");
    GE_CHK_STATUS_RET(GeneratorAllLoadTask(optimize_graph, tasks));
  } else {
    // 非Norm-like场景：生成所有模板类型
    GE_CHK_STATUS_RET(GeneratorGeneralTask(optimize_graph, tasks));    // Common
    GE_CHK_STATUS_RET(GeneratorRCoreTask(optimize_graph, tasks));      // RCore
    GE_CHK_STATUS_RET(GeneratorAllLoadTask(optimize_graph, tasks));    // AllLoad
  }
}
```

#### 4. 三种模板详解

| 模板类型 | 适用场景 | 执行方式 | 优缺点 |
|---------|---------|---------|--------|
| **Common** | R轴较大、通用场景 | R轴切分，两阶段执行 | 模板数多，ATT选择复杂 |
| **AllLoad** | Norm-like（R轴≤65536，A轴≥16） | R轴一次性加载，单阶段 | 性能好，模板数少 |
| **RCore** | R轴可分核 | R轴分到多核，两阶段 | 多核并行，需workspace |

#### 5. R轴切分逻辑

```cpp
// 1. Reduce后融合切分
GE_CHK_STATUS_RET(ReducePartitionPostFusion(optimize_graph));

// 2. 按环路起点、终点进行norm切分
GE_CHK_STATUS_RET(PartitionNorm(optimize_graph, loop_start_end));

// 3. Reduce多引用结构切分
GE_CHK_STATUS_RET(ReducePartitionMultipleCitations(optimize_graph));
```

**R轴切分影响**：
- 轴切分碎 → 模板数多 → ATT选择困难
- 无法像elementwise那样合轴计算
- 需要特殊处理尾轴对齐

---

### CodeGen模块

#### 1. 功能概述

CodeGen模块生成Reduce算子的Ascend C代码，处理AR/RA模式、尾轴对齐等。

#### 2. ReduceApiCall核心逻辑

```cpp
Status ReduceApiCall::Generate(...) {
  // 1. 获取reduce类型和指令类型
  auto &[type_value, instr_type] = reduce_type_map.find(this->api_name_);

  // 2. 判断AR/RA模式
  std::string reduce_pattern;
  GetIsArAndPattern(x, y, x.isAr, reduce_pattern);

  // 3. 生成dtype名称（ArgMax特殊处理）
  GE_CHK_STATUS_RET(GetDtypeNameForReduce(this->api_name_, x, y, dtype_name));

  // 4. 尾轴合轴（满足32B对齐）
  ReduceMergedSizeCodeGen(tpipe, ss, x, y);
  ReduceDimACodeGen(x, this->api_name_, ss);

  // 5. 生成ReduceInit代码
  ReduceInitCodeGen(x, y, type_value, ss, tpipe, dtype_name);

  // 6. 生成Reduce计算代码
  if (!IsNeedMultiReduce(tpipe.tiler, x, y, current_axis.back())) {
    // 单次Reduce
    ss << "Reduce" << new_api_name << "<" << dtype_name << ", " << reduce_pattern << ", false>"
       << "(y[offset], x[offset], tmp_buf, tmp_reduce_shape, true);";
  } else {
    // 多次Reduce（R轴分块）
    // 需要中间结果累积
  }
}
```

#### 3. AR/RA模式代码生成

```cpp
void GetIsArAndPattern(const Tensor &x, const Tensor &y, bool &isAr, std::string &reduce_pattern) {
  isAr = (y.vectorized_strides.back() == 0);  // 向量化stride为0表示AR模式

  const std::map<bool, std::string> reduce_pattern_map = {
    {true, "AR"},   // A轴在前，R轴在后
    {false, "RA"}   // R轴在前，A轴在后
  };
  reduce_pattern = reduce_pattern_map[isAr];
}
```

**AR模式特征**：
- `vectorized_strides.back() == 0`：最后轴（R轴）stride为0
- A轴被向量化，R轴在内部循环

**RA模式特征**：
- `vectorized_strides.back() != 0`：R轴被向量化
- R轴在外部循环，A轴在内部

#### 4. 尾轴对齐处理

```cpp
void ReduceDimACodeGen(const Tensor &x, const std::string &api_name, std::stringstream &ss) {
  // 检查尾轴是否满足32B对齐
  size_t last_dim_size = x.last_axis_size;
  size_t aligned_elements = 32 / sizeof(dtype);  // 对齐元素数

  if (last_dim_size < aligned_elements) {
    // 尾轴不足，需要合轴
    ss << "first_actual = " << merged_axis_size << ";";
    ss << "last = " << last_dim_aligned_size << ";";
  } else {
    // 尾轴已满足
    ss << "first_actual = " << first_axis_size << ";";
    ss << "last = " << last_dim_size << ";";
  }
}
```

#### 5. ArgMax多阶段处理

ArgMax在R轴较大时需要两阶段处理：

```cpp
// Phase1: 分块计算局部最大值和索引
ss << "ArgMaxWithValueExtend<int64_t, dtype, pattern>"
   << "(tmp_index, tmp_value, x[offset], tmp_buf, shape);";

// 累加offset（用于索引计算）
if (x.isAr) {
  ss << "accumulated_offset += vectorized_axis.actual_size;";
} else {
  ss << "accumulated_offset += first_actual;";
}

// Phase2: 合并所有块的局部结果
ss << "UpdateMaxIndexAndValue<dtype>(tmp_index, tmp_value, y[0], saved_value, offset, tmp_buf);";
```

---

### ATT模块

#### 1. 功能概述

ATT模块进行模板搜索和选择，为Reduce生成最优的tiling参数。

#### 2. R轴识别

```cpp
void AttUtils::CollectReduceAxisNames(const ge::AscNodePtr &node,
                                      std::set<std::string> &reduce_axis_orig_names) {
  // 遍历节点，收集R轴名称
  for (auto &dim : node_info.dims) {
    if (dim->is_reduce_axis) {
      reduce_axis_orig_names.insert(dim->name);
    }
  }
}
```

#### 3. Reduce轴标记

```cpp
bool CheckAndMarkReduceSplitAxis(SubAxis *axis,
                                 const std::set<std::string> &reduce_axis_orig_names) {
  if (reduce_axis_orig_names.find(orig_name) != reduce_axis_orig_names.end()) {
    axis->is_reduce_split_axis = true;  // 标记为R轴切分轴
    return true;
  }
  return false;
}
```

**R轴切分特征**：
- `is_reduce_split_axis = true`：该轴是R轴切分轴
- 影响tiling策略和分核方案

#### 4. A轴查找

```cpp
SubAxis *FindAAxis(NodeInfo &node_info) {
  std::set<std::string> reduce_split_axis_names = CollectReduceSplitAxisNames();

  // 从右向左查找非R轴（A轴）
  for (auto it = node_info.vectorized_axes.rbegin(); it != node_info.vectorized_axes.rend(); ++it) {
    SubAxis *sub_axis = *it;

    // 跳过R轴和B轴
    if (ShouldSkipAxis(sub_axis, reduce_split_axis_names)) {
      continue;
    }

    // 找到第一个非R轴即为A轴
    return sub_axis;
  }
  return nullptr;
}
```

#### 5. R轴切分与性能

**ATT中的特殊处理**：
```cpp
// 1. 若切分轴同时为R轴或非A轴，使用性能公式除以切分轴循环次数
// 2. 若切分轴同时是A轴，且R轴相关循环轴轮次为1，除以B切分轴循环次数
```

---

## 典型场景分析

### 场景1：LayerNorm (Norm-like Reduce)

**算子序列**：`Load → Mean → Sub → Div → Store`

**特点**：
- R轴较小（如1024）
- A轴较大（如4096）
- 满足norm-like条件

**处理流程**：
1. **Lowering**: Mean → StoreReduction(SUM)
2. **CanFuse**： 全部融合为单子图
3. **Schedule**： 选择AllLoad模板
4. **CodeGen**： AR模式，尾轴满足32B对齐
5. **ATT**： R轴不分核，一次性计算

### 场景2：ArgMax大R轴

**算子序列**：`Load → ArgMax → Store`

**特点**：
- R轴很大（如100000）
- 需要分阶段处理

**处理流程**：
1. **Lowering**: ArgMax → StoreReduction(MAX)
2. **CanFuse**： 单子图
3. **Schedule**： 选择RCore模板（R轴分核）
4. **CodeGen**： 两阶段处理
   - Phase1: 分块计算局部最大值
   - Phase2: 合并所有块的局部结果
5. **ATT**： R轴分核tiling

### 场景3：Reduce + Elementwise融合

**算子序列**：`Load → Sum → Add → Mul → Store`

**特点**：
- Reduce后接2个elementwise
- 满足融合条件（≤3个）

**处理流程**：
1. **Lowering**: Sum → StoreReduction(SUM)
2. **CanFuse**： 允许向后融合Add和Mul
3. **Schedule**： Common模板，两阶段
   - Stage1: Sum规约 → workspace
   - Stage2: Add + Mul计算
4. **CodeGen**： RA模式可能需要Transpose
5. **ATT**： 需要选择最优tiling

---

## 常见问题定位

### 问题1：R轴切分导致模板数过多

**现象**：融合后模板数远超elementwise融合

**原因**：
- Reduce特殊R轴限制
- 尾轴对齐要求导致轴不能合轴
- AR/RA模式需要Transpose

**定位**：
1. 查看Schedule模块生成的模板数
2. 分析R轴切分策略
3. 检查是否满足norm-like条件

### 问题2：尾轴32B对齐失败

**现象**：Reduce API调用失败或性能劣化

**原因**：
- 尾轴元素数不足对齐要求
- dtype计算错误

**定位**：
1. 检查`ReduceDimACodeGen`输出
2. 验证`last`变量是否满足对齐
3. 查看是否触发合轴逻辑

### 问题3：AR/RA模式选择错误

**现象**：Transpose多余或缺失

**原因**：
- `isAr`判断逻辑错误
- 向量化轴stride计算错误

**定位**：
1. 检查`GetIsArAndPattern`输出
2. 验证`vectorized_strides.back()`值
3. 分析Tensor的向量化轴配置

### 问题4：融合范围不符合预期

**现象**：Reduce未融合或融合过多elementwise

**原因**：
- `max_reduce_can_fuse_elementwise_nums`配置
- Norm-like检查失败

**定位**：
1. 查看CanFuse日志
2. 检查`GetReduceAllLoadState`状态
3. 验证Norm-like条件（R轴、A轴大小）

---

## 附录：关键代码位置

| 模块 | 文件 | 关键函数 |
|------|------|---------|
| Lowering | `autofuse/lowering/asc_lowerer/loop_api.cpp` | `StoreReduction` |
| Lowering | `autofuse/lowering/op_lowering_impl/lowering_impl.cpp` | `TryLowerMatMulToReduce` |
| CanFuse | `autofuse/can_fuse/strategy/reduce_fusion_strategy.cpp` | `ReduceFusionStrategy::CanFuse` |
| Schedule | `optimize/task_generator/reduce_schedule_case_generator.cpp` | `GeneratorTask` |
| Schedule | `optimize/norm_like_reduce_checker.cpp` | `IsNormLikeReduceGraph` |
| CodeGen | `codegen/api_call/reduce/reduce_api_call.cpp` | `ReduceApiCall::Generate` |
| CodeGen | `codegen/api_call/reduce/reduce_api_call_base.cpp` | `GetIsArAndPattern`, `ReduceDimACodeGen` |
| ATT | `att/util/att_utils.cpp` | `CollectReduceAxisNames` |
| ATT | `att/gen_model_info/expr_gen/generate_tiling_expr.cpp` | `FindAAxis` |

## 相关链接

- 返回 [AutoFuse 架构介绍](../../introduction/architecture.md)
