# 简介

## 目的

让 mem 侧的分段广播收下**尾轴带填充**的目的地 tile（片上行距被补到 32B 而行长更短），把探针拓扑切片里那几格 `'afir.broadcast' op has no supported Ascend implementation` 变成 ok，并保持 `FAILED` / `DRIFT` 两个桶恒为 0。

目标读者：改 `conversion/AFIRToAscendC/Implementations/Broadcast.cpp` 与广播能力判据的人。

## 范围

**包含**：mem 侧 `classifySegments` 的窗口稠密判据、`SegmentedBroadcastEmitter::emitWindowCall` 发出的那一条 `Broadcast(...)` 调用、对应的 lit 与 CPU E2E。

**不包含**：

- reg 侧 `classifyExtendSegments`（它**已经**收了这个形状，本次照搬它的判据，不改它）。
- 形态 B（最内那一段就是广播段）带填充的情形，reg 侧也拒，本次继续拒。
- `afir.nddma`、队列绑定（T3 的 `BindingConflict`）、非队列资源布局，都是另外的线。

涉及组件：Autofuse Codegen/Backend（MLIR 侧），测试框架。不涉及 SuperKernel、打包、Python 接口。

# 总体概述

## 软件概述

### 项目介绍

`afir.broadcast` 在 MLIR codegen 里分两档：源在每根向量化轴上都是 1 的走 `Duplicate`，其余走分段广播（`broadcast.tensor.segmented`，发 `broadcast.h` 的包装 `Broadcast(...)`，与 legacy 的 `BroadcastOneAxis` 逐字同一个调用）。分段广播把最内两段交给一条两段窗口调用，更外侧的段变 `scf.for`。

本次动的是分段广播**收不收**的判据里的一条，以及那条窗口调用的两个实参。

### 产品环境介绍

只经由 `libaihac_codegen.so` 生成 AscendC 源码，不新增 runtime / aclrt 调用，不改对外接口。发出去的调用仍是 `broadcast.h` 的那条包装，由它自己按 dtype 分流。

## 软件功能

目的地 tile 的尾轴（最内那根向量化轴）行距被补到 32B、与行长不等时，分段广播仍然收下，并按**行距**而不是行长发窗口调用。

## 设计约束

- 拒收只能拒在 capability 层。pattern 返回 `failure()` 会变成 `failed to legalize`、整图硬失败，而仓库把「`DRIFT` 恒为 0」当不变量。capability 与 pattern 必须问同一个函数（`analyzeSegmentedBroadcast` 各算一遍、不缓存，判据是同一个）。
- AscendC 的 `Broadcast` 包装要求尾轴按 32B 对齐（见「假设和依赖关系」第 2 条）。
- 不改 ABI/API，不硬编码芯片类型。
- 稠密布局上发出来的 Op 必须**逐字不变**（本次是放宽，不是改写既有行为）。

## 假设和依赖关系

1. **目的地的填充位是 don't care。** 依据：`afir.store` 读这块 tile 时 UB 侧用 `vectorized_strides`（行距 8/16）、GM 侧用 `strides`（行长 5/7/8），逐行搬运时跳过填充。同一条假设已被既有的「相邻广播段」那一档依赖（`emitBlockCopy` 连填充一起复制，注释里写明填充位 don't care）。
2. **行距恒是 32B 的整数倍。** 实测 `vectorized_strides` 的次内一位恒为 `32 / sizeof(T)`（f32 → 8，f16 → 16）。所以按行距发调用**同时满足**了 AscendC 包装对尾轴 32B 对齐的要求——这是本方案比「按行长发、另想办法补对齐」更可取的主要理由（参见 [[broadcast-32b-tail-invariant]] 记的那条：违反该不变量是静默算错，codegen 侧没有检查）。
3. **形态 A 下按行距读源，读到的越界元素只会落进目的地的填充位。** 形态 A 的源只有一行（外段源 stride 为 0），行长 = 内段 extent，其资源按 32B 对齐分配，所以读满一个行距仍在该资源之内。读到的多余元素被写进目的地填充位，而填充位 don't care。**待确认项**：若将来源 tile 与别的成员共享一个非队列资源且紧邻排布，这几个元素会读到邻居的数据——仍然不影响正确性（只落进 don't care 位），但不再是「读自己的填充」。本次按「正确性不受影响」接受，并在代码注释里写明这条推理。

# 需求分析与设计

## 整体介绍

数据流不变：`collapseVectorizedAxes` 出分段 → `classifySegments` 判形状 → `materializeVectorizedSegments` 把窗口之外的段变循环 → `SegmentedBroadcastEmitter` 发调用。

本次只在 `classifySegments` 里把「目的地在窗口里稠密」这一条从**等值要求**改成**取行距**，并把行距带到 `emitWindowCall`。

## 功能需求

### 功能需求 1：窗口稠密判据改成取行距

1. **介绍**

   原判据要求内段目的地 stride 为 1 **且**外段目的地 stride 等于内段 extent（稠密）。后一半在尾轴有填充时不成立，于是整个形状被拒。改成：外段目的地 stride 即**行距** `P`；`P` 与内段 extent 可证相等时行为完全不变，不相等时记下 `P`，并且**只在形态 A（最内段不是广播段）下收**。

2. **输入**

   `afir::VectorPlan`（`collapseVectorizedAxes` 的结果），段数 ≥ 2。关键输入是 `plan.segments[count-2].strides[destination]`（行距）与 `plan.segments[count-1].extent`（行长）。

3. **处理**

   - 内段目的地 stride 仍必须可证为 1（这一条不放宽：它说的是「一行之内连续」）。
   - 取 `P = outer.strides[destination]`。`provablyEqual(P, inner.extent)` 成立 → 稠密，`pitch` 留空。
   - 不成立 → 真有填充：若 `broadcastIsInner`（形态 B）则 `failure()`（调用方翻成 unsupported、整图回落）；否则记 `shape.pitch = P`。
   - 证不了相等时按「不相等」走（保守方向：要么拒收，要么按行距发——两者都不会算错）。

4. **输出**

   `BroadcastSegmentation` 多一个 `pitch` 字段（`ExprAttr`，空表示稠密）。capability 与 pattern 拿到同一个结果。

### 功能需求 2：窗口调用按行距发

1. **介绍**

   形态 A 的调用是 `Broadcast(dst, src, 1, Z, 0, K, Z, 0, tmp)`，`Z` 是内段 extent、`K` 是外段 extent。有填充时把两处 `Z` 都换成行距 `P`。

2. **输入**

   `shape_.pitch`（空则沿用 `extentOf(count - 1)`）。

3. **处理**

   `pitch` 非空时用 `afir::materializeStrideValue` 物化出 `P`，替换 `src_k` 与 `dst_k`。两处必须一起换：包装要求 `srcShape[i] == dstShape[i]`，只换一边对不上重载。`dst_k` 在包装里身兼行长与行距两职（`BroadcastFirstDim` 的 `dst_inner_offset`），所以换成行距之后行与行落在 `0, P, 2P, …`，正是带填充的真实布局。

   **不给 `P` 加尾块修正**：尾块只让最后一趟的行长变短，不改行距，而 `P` 在这里当的就是行距。这与 reg 侧「只有最外那一段带尾块修正」同一条规则。

3. **输出**

   一条 `ascendc.broadcast_dim3`，稠密时与此前逐字相同。外层广播段的整块复制（`emitBlockCopy`）**不改**——它取的本来就是目的地 stride，已经是行距正确的。

## 非功能需求

### 可维护性

判据与 reg 侧 `classifyExtendSegments` 的 `shape.pitch` 一段**形状相同、理由相同**，两处注释互相指认。新增字段一个（`BroadcastSegmentation::pitch`），不新增抽象。

### 可测试性

lit 钉下降产物（发的是行距还是行长），CPU E2E 给数值证据，探针量可用面。形态 B 带填充那一格继续由 capability 拒，lit 反例钉它仍被拒。

### 可移植性

不新增平台判断。mem 侧与 reg 侧各自保留自己的判据函数。

### 可靠性

唯一的新分支是「填充 + 形态 B」→ `failure()`，走既有的 unsupported 回落路径，无资源申请、无生命周期变化。

### 特性交叉影响

按 `docs/guidelines/cross_feature_check.md` 逐项：

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| SuperKernel Python 接口 | 不适用 | 不碰 `super_kernel/` |
| SuperKernel C++/AOT 接口 | 不适用 | 不改对外符号与动态库行为 |
| Autofuse 图优化 | 不适用 | 不改 ASCIR 图结构与 pass 时序；只读 descriptor |
| Autofuse Codegen/Backend | **适用** | 改分段广播的收取判据与一条窗口调用的两个实参；稠密布局上产物逐字不变，带填充时多收下一批形状 |
| AscendC API / Runtime 交互 | **适用** | 仍发 `broadcast.h` 那条包装，不新增 runtime 调用；按行距发**满足**包装对尾轴 32B 对齐的要求 |
| Python/C++ 混合绑定 | 不适用 | 不碰 `py_module/` |
| 构建与打包 | 不适用 | 不改 CMake / build.sh / 产物布局（新增 E2E 格子只动 `test/e2e/cpu/cases/*/CMakeLists.txt`） |
| 测试与覆盖率 | **适用** | 新增 lit（正例 + 形态 B 拒收反例）与 CPU E2E 格子 |
| 性能与日志 | **适用** | 见下，判据多一次符号比较；不新增日志 |
| 兼容性 | 不适用 | 无对外接口、配置、产物命名变化 |

## 性能

### 编译时长

`classifySegments` 每个 `afir.broadcast` 多一次 `provablyEqual`（行距 vs 行长）。文本相同先短路，绝大多数是字面量对字面量，省掉 SymEngine 往返。原判据本来也调一次同样的 `provablyEqual`，所以净增约为 0。

### 执行性能

带填充时发出的 `Broadcast` 处理的元素数从行长变成行距（f32 行长 5 → 8，f16 行长 8 → 16），每行多算至多 `32B - 1` 个元素，落在 don't care 位。相对的是此前整图回落 legacy，所以这条路径的对照是 legacy 而不是「更快的 MLIR 实现」；legacy 在同一形状上发的也是整块对齐的调用。

### 内存和产物大小

`BroadcastSegmentation` 多一个 `ExprAttr`（栈上、随分析结束释放）。无常驻内存、无产物大小变化。

## 接口设计

### 新增/修改接口描述

无对外接口。文件内部：`BroadcastSegmentation` 增加 `afir::ExprAttr pitch`（空 = 稠密）。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|--------|----------|----------|------|
| 接口说明 | 是否需要接口评审 | 否 | 无对外接口 |
| 接口说明 | 是否需要补充文档 | 是 | 本文档 + 代码注释 |
| 接口兼容 | 行为是否兼容 | 是 | 稠密布局产物逐字不变；带填充从拒收变成支持 |
| 接口兼容 | ABI/API 是否兼容 | 否 | 不涉及 |
| 接口约束 | 约束不满足时是否清晰报错 | 是 | 形态 B 带填充走 capability unsupported、整图回落 |
| 接口测试 | 是否需要独立接口用例 | 是 | lit 正例 + 反例，CPU E2E 数值 |

## 软件设计

### 关键数据结构

`BroadcastSegmentation::pitch`（`afir::ExprAttr`）：窗口里一行到下一行的目的地步长。空表示与行长相等（稠密），非空表示真有填充、窗口调用要按它发。生命周期随 `classifySegments` 的返回值，归调用方所有。

### 关键技术/算法

`P = segments[count-2].strides[destination]`，`Z = segments[count-1].extent`。`provablyEqual(P, Z)` 为真即稠密。判据复杂度不变（一次符号比较）。

### 流程设计

主流程见功能需求 1、2。异常流程：形态 B 带填充 → `failure()` → capability unsupported → 整图回落 legacy。无需回滚或清理（分析阶段不建 IR；`emitWindowCall` 建 IR 之前判据已经过）。

### 对子模块的修改

- `conversion/AFIRToAscendC/Implementations/Broadcast.cpp`：`BroadcastSegmentation` 加字段；`classifySegments` 的窗口判据；`SegmentedBroadcastEmitter::emitWindowCall`。
- `test/lit/conversion/`：新增正例与形态 B 反例。
- `test/e2e/cpu/cases/load_broadcast_segmented_store/`（及级联用例）：新增带填充的形状格。

### 错误处理

#### 系统错误

`materializeStrideValue` 物化不出行距 → `failure()`。它在 `emitWindowCall` 里，而那时判据已过，所以这是「本该能物化却没能」——按既有写法报 match 失败，不吞掉。

#### 接口错误

无对外错误码。能力不足经 `CapabilityShortfall` 标记走回落，诊断里带判据名。

## 安全检查

按 `docs/guidelines/编码红线.md`：

- **敏感信息 / 公网地址 / 芯片类型硬编码**：无新增。注释与 stable id 不出现芯片名。
- **外部输入作索引、长度、偏移**：行距来自调度给的 descriptor，不是用户输入；它作为 `Broadcast` 的长度实参，而「它恒是 32B 的整数倍、且不小于行长」是「假设和依赖关系」第 2、3 条，由判据里 `provablyEqual` 之外的那条不变量承担——**本次不新增运行期校验，因为符号表达式在编译期证不出不等式**；这一点与既有 reg 侧同口径。
- **整数溢出 / 反转 / 除 0**：不新增算术，行距是现成的表达式。
- **资源释放 / 异常分支**：不申请资源。
- **ABI/API 兼容**：不涉及。
- **图改写等价性与确定性**：判据是对 descriptor 的纯函数，无随机、无哈希序遍历，同一输入恒给同一答案。
- **runtime / AscendC 生命周期**：不新增调用；发的包装与此前同一条。

## 兼容性检查

不影响脚本参数、配置文件、安装目录、动态库、Python 包、旧产物或用户代码。稠密布局上生成的 kernel 文本逐字不变（由 lit 的既有用例保证）。

## 测试设计

### 测试边界

入口：`af-opt --afir-plan-candidate-lowering-pipeline`（lit）、`check-e2e-cpu-*`（数值）、`af-capability-probe --topo`（可用面）。出口：生成的 AscendC 文本与 CPU 模拟器的输出缓冲。不打桩。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|----------|
| 功能 | 形态 A 带填充：窗口调用的 `src_k` / `dst_k` 发的是行距 | lit 钉 `broadcast_dim3` 的实参 | lit |
| 功能 | 形态 A 带填充的数值正确（含填充位不被读回） | CPU E2E，输出初始化成哨兵 | E2E |
| 兼容性 | 稠密布局产物逐字不变 | 既有 lit 全绿 | lit |
| 异常 | 形态 B 带填充仍被拒、整图回落 | lit 反例 + 探针该格仍 unsupported | lit |
| 特性交叉 | 级联图（广播结果进下游计算）在带填充形状上可用 | 既有级联 E2E 加格子 | E2E |
| 性能 | 编译时长无明显变化 | 探针 112 张图耗时对比 | 手工验证 |

变异验证（按 [[mutation-not-red-triage]]）：把判据改回等值要求，正例 lit 必须变红；把形态 B 那条拒收删掉，反例 lit 必须变红。反例一格只违反一条判据，用 `--split-input-file`，CHECK 用计数式。

### 测试命令

```bash
make -C build af-opt -j 6
bash ./mlir-test.sh
make -C build af-capability-probe -j 6
```

## 验收标准

1. 探针 `--topo` 的 `FAILED` 与 `DRIFT` 两桶仍为 0。
2. 形态 A 带填充那几格从 `capability: no implementation` 变 ok；形态 B 带填充那几格仍是 capability 回落，并说明原因。
3. `bash ./mlir-test.sh` 四个目标全绿。
4. 新增 lit（正例 + 反例，含变异验证结论写进文件头）与 CPU E2E 格子（输出用哨兵初始化）。
5. 稠密布局上生成的 kernel 文本不变。

# 实测结果（2026-10-12）

## 探针 `--topo`（112 张图）

`FAILED` 0、`DRIFT` 0，两条不变量都保住。按拓扑：

| 拓扑 | 改前 ok / no-impl | 改后 ok / no-impl |
|------|-------------------|-------------------|
| T1 `brc → store` | 4 / 6 | **8 / 2** |
| T2 `brc → abs` | 4 / 6 | **8 / 2** |
| T5 `brc` 作分叉点 | 4 / 6 | **8 / 2** |
| T6 `brc → cast` | 4 / 6 | **8 / 2** |
| T7 | 4 / 6 | **8 / 2** |
| T4 两步广播 | 1 / 1 | **2 / 0** |
| T3 `brc` 作二元 operand | 2 / 8 | 2 / 8（**未变**，见下） |

每个拓扑剩下的 2 格都是 `[2,3,5,1] → [2,3,5,7]`（`inner=brc`，形态 B 带填充），按设计继续由
capability 拒、整图回落。

## 一条此前的判断是错的：尾轴象限**不是** T3 的上游

> 当前实现更新：不同 tensor ID 的 queue 成员按各自对齐后的物理大小打包，允许不同布局；
> bridge 不再要求独立成员的完整 tensor view 相同。资源属性冲突、成员 descriptor 非法、
> 同一 tensor 的不兼容重复视图仍拒绝。以下拒收记录为修复前现象；解除该限制不代表
> tile 内 inline broadcast 已受支持，也不保证整张图的全部候选都能通过。

接任务时写进 brief 的说法是「T3 那 6 格是尾轴象限的下游，收掉它就解开」。**实测不成立**：
收掉之后 T3 那 6 格一格没动，它们卡在更早的一道闸——

```
UnsupportedOperation: node 'load0' result resource domain queue buffer_id=0 reuse_id=0
carries a broadcast tile and a full-shape tile on one resource
```

也就是「一条队列上同时挂着广播 tile 与全形状 tile」那条资源判据（2026-10-12 由
`42337250` 那条线引入的命名拒收）。要收 T3 得先解它，与本文档这条判据无关。

## 已知瑕疵

形态 A 带填充时，窗口内段的 extent 仍被 `materializeVectorizedSegments` 物化出来，而调用
改发行距之后它没人用了，于是生成的 kernel 里多一个 `constexpr uint32_t c8_idx = 8;`。
它是 `constexpr`、无运行期代价。**不在本次范围内去掉**：那要改
`materializeVectorizedSegments` 的契约（它按定义物化窗口每一段的 extent），比需求要求的
改动面大。

## 验收

1. 探针 `FAILED` / `DRIFT` 均为 0 ✓
2. 形态 A 带填充从 `capability: no implementation` 变 ok（5 个拓扑各 +4 格，T4 +1 格）；
   形态 B 带填充仍回落并说明原因 ✓
3. `bash ./mlir-test.sh` 四目标全绿，lit **195/195** ✓
4. 新增 lit `afir-broadcast-tail-pitch.mlir`（含三条变异验证的实测结论）、
   反例 `afir-broadcast-unsupported.mlir` 第二片改成形态 B 带填充、
   CPU E2E `load_broadcast_segmented_store_1x5`（`[1,5] → [33,5]`，输出用 `-999` 哨兵初始化）✓
5. 稠密布局产物不变（既有 lit 与 E2E 全绿）✓
