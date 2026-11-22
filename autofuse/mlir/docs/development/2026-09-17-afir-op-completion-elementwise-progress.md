# AFIR Op 补齐（elementwise 切片）：进度与剩余任务

方案在 [AFIR Op 补齐：elementwise 切片开发方案](../design/2026-09-17-afir-op-completion-elementwise-slice-design.md)。
本文不重复方案，只记三件事：**做到哪了**、**下一步做什么**、**哪些地方不能照方案直接往下走**。
新 session 从这里起步，需要判据和取舍的来由时再回方案查。

---

## 从这里开始

**op 接完了。** 比较族六个（eq / ne / lt / le / gt / ge）是最后一拨，这一轮接通两侧；
47 个共有 elementwise 节点类型里只剩 3 个非 op。**下一步见「下一步挑哪个（比较族之后
重排）」——剩的三件都不是「再接一个 op」**：~~前置 D（一条队列上不能混 dtype）~~
**已解**、~~L-4（swap 方向的标量）~~ **已解**、~~E3 剩余档（`ScalarData` / `IndexExpr`）~~
**已解**。~~另有断头路 `Broadcast`~~ **常量源一档已接通**（三轴 / 四轴那两档仍未接通）。待办清单以
[遗留问题清单](2026-09-22-afir-elementwise-remaining-issues.md) 为准。

**比较族的卡点写着三件新机制，核完只剩一件。** 文档一直写「`<T, dim, CMPMODE>` +
brace-init 数组 + second-to-last stride」：second-to-last stride 只有 `dim == 2` 读得到
而这条路只发连续形态，brace-init 数组 `Cast` 那一轮已经做掉，**真正新的只有「同一个
`CMPMODE` 在 mem 侧是运行期实参、在 reg 侧是模板实参」**。现场在「比较族：已接通」
一节；Rsqrt / LogicalNot、Cast、Pow、Tanh / Erf、TrueDiv、Axpy 在它上面六节。

**这是本文那条「文档里的卡点是待验证的断言」第三次兑现**（前两次是偏差 6 与前置 B）。
三次的形状一样：结论写在文档里、来源在另一个目录，没人回去对过。

这份文档 1500 多行，是三块拼起来的，**按用途跳着读，不要通读**：

| 你要做什么 | 读哪几节 |
|---|---|
| **接着接一个 op** | 「重算」（当前数字与还剩哪些）→ 目标 op 自己那一节 →「接着做需要知道的」（改动点、验证与提交口径） |
| **撞上机制卡点** | 「三条跨波次的前置」（前置 A / B / C / D，**A 与 B 已做，B 那一条原先的说法是错的**）、「挂起的 op 与理由」 |
| **回查某个已做的 op 为什么那么写** | 「BitwiseAnd」「FloorDiv」「Remainder」「E6 后三波」「Axpy」「TrueDiv」「Tanh / Erf」「Pow」「Cast」「Rsqrt / LogicalNot」「比较族」十一节，每节记一条被顶翻的假设，后面的 op 直接用得上 |
| **回查合入前改了什么** | 「合入前的五项收口」「E5 收尾与三条被推翻的判断」「pyasc 同步后的重新评估」——**存档，日常不用读** |
| **与本切片无关** | 「VF 融合的现场」、「本轮 PR 的边界」里的 E8 取消与 ATT tiling 搁置 |

两条贯穿全文、每次都要记得的：

- **数字一律按末尾「清点口径」的命令重算**，别照抄正文——每一波都会动，正文里的是快照。
- **方案文档里的旧判断不作数**，已被推翻多次（偏差 6、前置 C、ODS 还缺几个，都错过）。
  本文与方案冲突时以本文为准，本文与代码冲突时以代码为准。
  **本文自己的「XX 表达不了 / 产不出来」也不作数**——偏差 6 那一条挂了半年，
  验证只花了写一个 `.mlir` 过一遍 `ascir-translate` 的工夫。先验再信。

---

## 已落地

`./mlir-test.sh` 全绿。按内容分组，一条 commit 一件完整的事，
合入前的收口按归属并进了各自的 commit 而不另立一条。

| commit | 波次 | 内容 |
|---|---|---|
| 地基 | E0 | 扩展 API 通道、tmp buffer 操作数与三段划分、`afir.abs` 纵向探针 |
| 直通档 | E1 / E1.5 | Exp / Ln / Relu / Sqrt / Mul / Maximum / Minimum（两侧同实现） |
| E2 波次 | E2 | Neg / Reciprocal / Sigmoid（扩展头 + 前导通道 + `ApiFamily`） |
| 谓词族 | E4 | Isnan / IsFinite / IsInf（结果 dtype 与输入不同） |
| 二元路径 | E2.5 | `ElementwiseApiFamily.h`、三段划分，附 E3 查证结果 |
| E3 第一档 | E3-1 | `Scalar` 源 + 四个 op 的 tensor-scalar 形态 |
| 队列复用链 | — | `e0f8e849`：一条队列上多个 reuse 组各发一次 alloc，而 `init_queue` 只取 slot_count 最大值，三节点串一条队列就是 3 > 2、运行期报 exceed limits、输出全 0。改成 legacy 的 `reuse_from` 链（`AllocOutputs`），整条链一次 alloc |
| pyasc 同步 | — | `fe96b79d`：`api/ascendc` 同步上游到 `b44c97b`。结论见下「pyasc 同步后的重新评估」 |
| E5 / E6 各 op | E5 / E6 | Select / Where / ClipByValue / LeakyRelu / BitwiseAnd / FloorDiv / Remainder / Sub / Div / Gelu / Sign / LogicalAnd / LogicalOr / Axpy / TrueDiv，一波一到两个 op，各自的现场写在下面对应小节 |
| Math 家族 | — | Tanh / Erf。**这一波真正的产出是推翻偏差 6**（它不是卡点），ascendc dialect 与 CodeEmitter 一行没改，见下「Tanh / Erf：已接通」 |
| Pow | — | Pow。两侧各建一个 def，与 `Remainder` 同形；没有新机制，见下「Pow：已接通」 |
| Cast | — | Cast。**这一波有新机制**：ODS 基类拆出「不声明 `InferTypeOpInterface`」的一档，emitter 加 brace-init 数组形态，见下「Cast：已接通」 |
| 前置 B + Rsqrt / LogicalNot | — | 预先填好 1 的常量 block（`Implementations/ConstBlockTensor.{h,cpp}`），两个 op 跟着接通两侧。**这一波顺带订正了前置 B 的说法**，见下「Rsqrt / LogicalNot：已接通」 |
| E2E 形状 + 搬运对齐 | — | 44 条 `[128]` 注册全部换成 2~4 轴（每轴 [1, 512] 随机），**换完当场红 26 条，根因是裸 `AscendC::DataCopy` 的 32B 对齐**。Load / Store 改为一律发 `DataCopyPadExtend`（与 legacy 一致），`TargetFeature::DataCopyPadExtend` 整条删除。见下「E2E 形状换掉 2~4 轴」 |
| 比较族 | — | Eq / Ne / Lt / Le / Gt / Ge 六个一拨接通两侧。**这一波真正的产出是把「新调用形态」那条卡点核干净**：三件里两件不存在，剩的一件是「同一个 `CMPMODE` 两侧落在不同的位置」，见下「比较族：已接通」 |

**当前覆盖面**：bridge registry 46 行，**46 行全是 `Both`**——`Cast` 接通之后表里再没有
只有 tensor 形态的一行了，`KeepsBufferFormAllowlistExplicit` 那条断言跟着换了说法
（见下「Cast：已接通」）；集合相等仍由 `BridgeOperationRegistryTest` 断言。
elementwise 两条路共注册 **104 个 implementation（一元 32、二元 60、三元 12）**；CPU E2E
**46 个用例目录 / 49 个注册用例**。

**末尾「清点口径」那段 python 数不到三元那 12 个**：`registerSelection` 把 id 拼出来
（`(op + ".tensor.extend").str()`），不是字面量，正则扫不到。原先那个「86」也因此只算了
一元加二元两条路。三元要按 `registerTernary` 的调用点数：`select` / `where` / `clip_by_value`
各 4 个（tensor / 标量 × mem / reg）。

**二元那个数这一轮从 30 跳到 54**，一拨加了 24 个：比较族六个 op 各占四个
（两侧 × {tensor-tensor, tensor-scalar}）。**op 数与 implementation 数不同步**，
挑下一批时别拿后者估工作量。

**后面这两个数原先混成了一个。** 「28 个 CPU E2E 用例」数的是
`test/e2e/cpu/cases/` 下的目录数，而 `add_mlir_e2e_cpu_case` 的行数比它多 2——
`load_add_store` 一个目录注册了三个形状。跑起来的是后者，所以两个都写出来。

上面这三个数每一波都会动，别照抄——重算的命令在末尾「清点口径」。
**「一元 19」那个旧数是错的，不是过期**：Gelu 与 Sign 接通时漏改了，从此一直少 3。

**47 个共有 elementwise 节点类型里接通了 44 个**：Add / Mul / Maximum / Minimum /
Abs / Exp / Ln / Relu / Sqrt / Neg / Reciprocal / Sigmoid / Isnan / IsFinite / IsInf /
Select / Where / MaskedFill / ClipByValue / LeakyRelu / BitwiseAnd / FloorDiv /
Remainder / Sub / Div / Gelu / Sign / LogicalAnd / LogicalOr / Axpy / TrueDiv /
Tanh / Erf / Pow / Cast / Rsqrt / LogicalNot / Eq / Ne / Lt / Le / Gt / Ge / `Scalar`。
`Broadcast` 的常量源一档（legacy 的 `Duplicate` 分支）2026-09-23 接通，
`BufferFormSetIsCoveredByAscendCImplementations` 的豁免名单已清空；三轴 / 四轴
`Broadcast(...)` 那两档仍未接通，前置是逐轴的 `TransferPlan`（清单 L-7）。

**这一段与下面「重算」那张表的口径不同**：这里算 `Scalar` 不算 `Broadcast`，那张表反过来。
两处各按各的口径算，别互相抄。

---

## 合入前的五项收口（已完成）

拆 PR 之前的一轮收口，1 / 2 / 3 / 5 已按归属并进上表对应的 commit。第一项本来只是补个
用例，结果炸出一个静默的越界写。

### 1. 两块 tmp buffer 被并排放进只够一块的资源里　**真 bug，已修**

E0 检视留下的风险「同一个 tmp id 跨节点复用」此前无人验证：带 tmp buffer 的 E2E 全是单
计算节点，唯一的多节点用例 `load_exp_sqrt_store` 两个节点都不要 tmp buffer。补上
`load_abs_reciprocal_store`（mem-based 一侧两个都走包装、都要 tmp）之后当场红了，
**结果全 0**。

机制与预想的不同，也更难发现：

- 预想是 `life_time_axis_id` 不同 → `registerResource` 报 conflicting descriptors →
  生成失败。实际两者相同，这一关放行了。
- 真正出错的是**布局**。`materializeLocalGroup` 把同一 canonical 资源的成员**依次累加
  偏移**并排摆放，而 ATT 的尺寸聚合是**跨节点取 max**（`buf_occupy_expr.cpp`：各节点内
  Σ、节点之间 max）——`b0_size` 只按一个节点算过一次。于是
  `InitBuffer(tbuf, b0_size)` 之后取 `tbuf.Get()[b0_size]`，第二块 tmp buffer 整个落在
  资源之外。
- **没有任何断言拦得住**：`resource member offset must not overflow` 只查算术溢出，
  不查偏移小于尺寸。

修法：`CanonicalGroupKey` 加 `exclusiveMemberOrdinal`，tmp buffer 各自成组、各自从 0 开始
（`CanonicalMaterialization.cpp` 按三段划分判断哪个操作数是 tmp buffer，不另立判据）。这与
legacy 的「先后复用同一块」逐字对上。同时加一条硬检查：tmp buffer 的 `view.offset` 非 0
直接报 `InvalidBinding`，把静默越界变成诊断。

用例钉的是「两个调用拿到同一个 tmp buffer 张量、只有一个 TBuf」——把修改回退，两个实参
立刻不同，用例红。

### 2. 家族判据漏查前导头　**已补**

`checkFamilyMatchesHeader` 只比对 `getExtensionHeader()`，E2 后加的
`getExtensionPreludeHeader()`（mem `utils.h` / reg `utils_reg_base.h`）没人管。判据提成
`headerMatchesFamily`，两个头名都过一遍。今天不可达（声明前导的四个 op 全在 mem 侧），
但 E5/E6 一上来就是成对的 reg-base 包装。

**顺带补上这条判据本身的用例**（`ElementwiseApiFamilyTest`）。它此前一个用例都没有——
「写反了就把整张注册表拆掉」这种判据，只要没人写反过，删成 `return success()` 也不会有
一处变红。这正是本轮检视清单里「每条新判据是否真有用例会因为删掉它而变红」那一条。

### 3. 删掉 `afir.tmp_buffer_ids`　**已删**

E7 的收尾项之一，提前做掉。**做法与方案设想的不同**：原以为可以让 GenTilingData 顺着
use-def 去读 tmp buffer 的 `memref.alloc`——不行，`AFIRMemoryAllocate` 在它之前就把那些
alloc 删掉了（`CanonicalRewrite.cpp`），`collectCanonicalResourceIds` 在物化之后也取不到。
改成在 bridge 收集时就把 tmp buffer 的 id 并进 `afir.buffer_ids`：两者本来就是同一个 id
空间、同一套 `b{id}_size` 命名，合表之后 GenTilingData 那条特判自然消失，属性、常量、
DTO 字段一起删干净。

### 4. 验收条件 6 / 7　**已测**

**条件 7（`load_add_store` 端到端编译耗时回退 ≤ 5%）：+0.88%，通过。**
`upstream/af-mlir` 建了一份 worktree 做基线，两边各跑 7 次同一个 generator：

| | 中位数 | min | max |
|---|---|---|---|
| 基线 `upstream/af-mlir`（`0af07117`） | 1.140s | 1.130 | 1.240 |
| HEAD | 1.150s | 1.140 | 1.160 |

噪声 σ≈0.007s，5% 阈值是 0.057s——这个测量分辨得出真回退，不是「反正都在误差里」。

**条件 6：**

- `AF_ENABLE_MLIR_CODEGEN` 关闭时 legacy 输出不变——**按构造成立，但没有拿到逐字节
  对照**。本轮对 legacy 的改动只有一条：`ascendc_api_registry.cpp` 往
  `AscendCApiRegistry` 加了一项 `{"utils.h", ...}`。它对 legacy 不可达：
  legacy 只按 `impl->LoadApiHeaderFiles()` 返回的名字查表、从不遍历这张 map
  （`codegen_kernel.cpp:3729`），而 v1 那 58 个头名里**没有 `utils.h`**（reg 侧用的是
  `utils_reg_base.h`），legacy 的 `utils.h` 走 `IncludeAndDefines` 那条无条件内联的路。
  `utils.h` 本来就在 `autofuse/ascendc/api/CMakeLists.txt` 的生成清单第一行，没有新增
  构建产物。
  想做的逐字节对照做不成：CPU E2E 的 generator 在 `AF_ENABLE_MLIR_CODEGEN=0` 下
  `GenerateForInductor` 直接失败——**基线上也一样失败**，是这个 harness 本来就不支持
  legacy 路线，与本轮无关。真要逐字节比，得走 legacy 自己的测试套。
- `build.sh --pkg -j 8` 与 `build.sh --pkg --no-autofuse -j 8`：**都失败，但与本轮无关，
  基线上逐字复现**。失败点是 `super_kernel/CMakeFiles/superkernel_whl`，报
  `/usr/bin/python3: No module named build`——CMake custom command 里的解释器装不到
  `build` 模块（交互 shell 里 import 得到，说明是那条命令的环境）。`super_kernel/` 本轮
  零改动，`upstream/af-mlir` 的 worktree 上跑同一条命令报同一行。是环境依赖缺失，
  不是回退；要让这条真正转绿得先把那个 Python 依赖补上。

### 5. 删掉没有实例的三元 ODS base

`AFIR_BufferizableTernaryOpBase` 与 `AFIR_BufferizableTernaryElementwiseOp` 自 E0-3 起
就没有任何 def 用到。**没有实例的 tablegen class 不会被展开，也就从来没被检查过**——
留在 PR 里等于带进去一段没验证的代码，第一次跑到是在 E5，而那时它还得为三元的 tmp buffer
与家族判据改一遍。整段挪到 E5 再加。

---

## 本轮 PR 的边界

两件事已定，**不在本轮做**，也不要因为它们把本轮的口径改宽：

### E8 取消（reg-based 独有的 81 个）

方案 E1～E8 表里的 E8 一行作废，**但作废的理由与当初写的不是一回事**。当初写的是
「上游 PyAsc 已补齐 RegBased micro api，这 81 个的供给不再需要在本仓批量手写注册」。
`fe96b79d` 同步到 `b44c97b` 之后按实际供给重新评估，那句话不成立，详见下一节。

对验收没有影响：验收条件 1 说的是**共有的 47 个**，本来就不含这 81 个。但风险表
**R-1 对这 81 个的描述是错的**——它写「全落 D3 最简单一档（标准 API、无 tmp）」，
实际对着矩阵数：**64 个要 tmp、55 个引扩展头、49 个 API 名带 `Extend`**，调用形态
有 12 种（现有实现只覆盖 `UnaryApiCall` / `UnaryApiTmpCall` / `BinaryApiTmpCall` 三种；
`UnaryTemplateAttrApiTmpCall` 8 个、`UnaryDoubleOutputApiTmpCall`（Frexp）、三个
`*ToInt` 都要新形态）。真正「无 tmp 又无扩展头」的只有 7 个：BitwiseNot / BitwiseOr /
BitwiseXor / FloorToInt / LShift / Log10 / RShift。**这批要真开，先按这张账重估工作量，
别照 R-1 那一行。**

### ~~ATT tiling 拒收含 `Scalar` 节点的图~~　**这一节整个是错的**

**已实测：`load_add_scalar_store` 建出来一次就过，数值正确。** 标量族现在有 CPU E2E 了。
详情、埋点与「为什么会误判」见
[遗留问题清单](2026-09-22-afir-elementwise-remaining-issues.md) 的 L-1。

下面原文保留，因为它是一份完整的反面教材——**排查工具失效被记成了被测对象失效**：

> - 现象：`codegen.GenerateForInductor` 在 tiling 阶段失败。
>   `codegen_tiling.cpp:872` 的 `codegen_func_`（ATT 的 `CodegenTiling`）返回 false，
>   ATT 侧不留任何日志，只有 GE 侧三行 `ErrorNo: 4294967295`。
> - 时序：失败发生在 **MLIR bridge 之前**，`CheckGeneratedKernel` 根本没跑到。
> - 复现：`ASCEND_GLOBAL_LOG_LEVEL=3 ASCEND_SLOG_PRINT_TO_STDOUT=1 ./build/.../MlirE2eCpu_<case>_generator`

三处错：

1. 这个 harness 的 `codegen_func_` **不是 ATT**，是 `test/e2e/cpu/support/TilingStub.cpp`
   的桩，无条件 `return false`。任何回落到 legacy tiling 的图都报那一行。
2. 时序反了。`GenerateForInductor` 第一件事就是 `RunMlirCodegen`，只有它返回
   `kNotSupported` 才走到 tiling。
3. 那条复现命令缺 `AF_ENABLE_MLIR_CODEGEN=1` 与 cmake 烘进去的 `LD_LIBRARY_PATH`，
   两者缺一个 MLIR 路径就谢绝——**本来是绿的用例也会报同一行**。
   **E2E 一律走 `check-e2e-cpu-<用例名>` 目标。**

---

## pyasc 同步后的重新评估

`fe96b79d` 把 `api/ascendc` 同步到上游 `b44c97b`。下面三条是对着同步后的代码重新数出来的，
推翻了此前几处判断。

### reg-based api 就是 micro api，不是两条路

`v35/ascendc/api_regbase/*.h` 里的包装**本身就是用微 API 写的**：`log1p.h` 的函数体整段是
`Reg::RegTensor` / `Reg::MaskReg` / `Reg::CreateMask` / `Reg::UpdateMask` / `Reg::LoadAlign`
写在 `__simd_vf__` 里，外面套一层 `Log1p(dst, src, calCount)` 的 LocalTensor 入口。
同步进来的 `OpReg.td` 发的 `AscendC::Reg::*` 是同一个命名空间。

所以 **AFIR 从来不需要表达微 API** 才能用这批包装：建一个 `ascendc.xxx_extend_reg_base`、
emit 出张量级调用、微 API 的函数体作为头文本由 `AscendCApiRegistry` 内联进 kernel——
`neg_extend_reg_base` 现在走的就是这条路。`OpReg.td` 解锁的是另一件事：**在 MLIR 里直接
生成微 API**。两者是两个切入点，不是替代关系。

### E8 那 81 个的供给，同步几乎没动

| | 个数 | |
|---|---|---|
| 甲 无扩展头 + pyasc 已有同名 op | 24 | Acos / Asin / Cosh / Digamma / Erfc / Lgamma / Tan / Xor / BitwiseOr…… |
| 乙 无扩展头 + pyasc 无 op | 2 | FloorToInt / Fma |
| 丙 引扩展头，要在 fork 加包装 op | 55 | 头文本 182 个全在 `ascendc_reg_base_api_register.cpp` 里；缺的只是 `ascendc.*` 的 td def，今天整个 fork 只有 `neg_extend_reg_base` 一个 |

同步对这张表的贡献是 **Log2 / Log10 / Mod 三个**，其余 27 个同步前就在。

~~**甲那 24 个看着现成，其实全被偏差 6 卡着**~~——**这句话是错的，见下一条**。偏差 6 不是闸，
甲那 24 个也不在验收的 47 个里；它们现成不现成要单独查，本文没查过。

### ~~偏差 6（`isReuseSource`）是 MathLibraryOp 全家的总闸~~　**整条推翻了，它根本不是闸**

**下面这一段是订正。原文写的是「dialect 里没有东西产得出会被 emit 成编译期常量的值」，
因此判定 `UnaryMathOp` / `BinaryMathOp` 全家（Tanh / Erf 加 E8 甲组 24 个，「26 个一起
解锁」）都被这一条挡着。这句话是错的，当场验了。**

事实是 `arith.constant false : i1` 走 `printConstantOp`（`Target/AscendC/Common.cpp:19`）
就打成 `constexpr bool`：那里除 f16 外一律先打 `constexpr `，再由 `emitIntegerType`
把 i1 打成 `bool`。而 `constexpr bool` 正是一个合法的非类型模板实参。手写一段
`"ascendc.tanh"(%dst, %src, %n, %reuse)` 过 `ascir-translate` 得到的是：

```c++
constexpr bool c0_i1 = false;
AscendC::Tanh<float, c0_i1>(v1, v2, c128_i32);
```

**所以既不需要给 Math 家族加一组不带该模板实参的 op，也不需要改 CodeEmitter。**
lowering 里物化一个 `arith.constant false` 就够了——`afir.tanh` / `afir.erf` 这一轮
就是这么接通的，见下「Tanh / Erf：已接通」。

顺带把挂在这条上的两个数也订正掉：

- 「26 个一起解锁」不成立。**E8 甲组那 24 个从来不是被这一条挡着的**，它们不在验收的
  47 个里（见上「E8 取消」与「本轮 PR 的边界」），本轮也不顺手接。这一条真正影响的
  就是 Tanh / Erf 两个。
- 原文说「同步没动这两行」是对的，但不重要——那两行本来就没错。

---

## VF 融合的现场（**独立 PR，不在 elementwise 切片范围内**）

写在这里是为了不被重新发明一遍。**VF 融合不是 MLIR 要做的 pass，legacy 在 schedule
阶段已经做完了**：

```
optimize::Optimizer::Optimize
  └─ BufQueAllocator::ProcessSingleImplGraph        (optimize/buffer_allocate/buf_que_allocator.cpp:907)
       └─ platform.PartitionSubFunctions(impl_graph)
            └─ VectorFuncPartitioner::Partition()   (v35/optimize/partition/vector_func_partitioner.cpp:465)
                 InitClusters → MergeClusters → SortClusters → BuildSubgraphs
       └─ 然后才是 AllocBufQueForSingleImplGraph
```

在 buffer / queue 分配**之前**，所以「融合决策拿不到、UB 已经分配完了」不是问题。

- **产出形状**：一个 cluster 变成一个 `af::ascir_op::VectorFunc` 节点
  （`<graph>_VfNode_<n>`），带属性 `sub_graph_name`，指向一张真的 AscGraph 子图
  （`<graph>_VfSubgraph_<n>`，`impl_graph_.AddSubGraph` 挂进去，继承父图的 axis 列表）。
- **触发条件**：`PartitionSubFunctions` 是平台钩子，`PlatformV1::PartitionSubFunctions`
  是空实现（`optimize/platform/v1/platformv1.cpp:30`），只有 PlatformV2（soc `3510` /
  `5102` / `9202`）真跑。
- **覆盖面**：`AscIrCodegen::IsVectorFunctionSupported` 基类返回 false，v2 里 23 个 op
  覆写为 true——Add / Sub / Mul / Div / TrueDiv / Maximum / Minimum / Abs / Exp / Ln /
  Sqrt / Rsqrt / Neg / Relu / Cast / Broadcast / Compare / Where / LeakyRelu / LogicalOr /
  BitwiseNot / Arange / IndirectLoad。分区器另有收紧：尾轴 stride≠1 不融、图里有 reduce
  时 Cast 不参与、`kMinVfNodesNum = 2`（单节点不融，Arange singleton 与 brc-inline 例外）。
- **下游都通了**：ATT 侧 `VectorFunctionGraphParser` 解析子节点、`vf_perf_utils` 按 micro
  api 条数算 perf；codegen 侧 `v35/codegen/micro_api_call/` 12 个 emitter 出 `AscendC::Reg::*`。
- **MLIR 侧是零**：`mlir/` 里除扫描脚本和文档外没有一行提到 `VectorFunc`，bridge registry
  也没有它。方案第 27 行「VectorFunc / Micro / SIMT 标量路径搁置（仅 reg-based）」就是它，
  是有意排除的。

**为什么它不影响 E5 / E6**：没进 VF 区域的节点仍然留在父图里、走普通 v2 codegen 发张量级
reg-base 调用（`BuildSubgraphs` 里 `cluster->Nodes().size() < kMinVfNodesNum` 那一支直接
`continue`）。所以**两份 implementation 不会因为 VF 变成死代码**，E5 / E6 该怎么写就怎么写。
反过来，E5 / E6 接通的 op 正好是 VF 那一版 PR 的输入。

**接手时的起点**：`tests/v35/st/backend_e2e_v2/vf_scalar_fusion_comprehensive_test/` 是现成
模板（`RuntimeStubV2` 切 3510 → `optimizer.Optimize` → `codegen.Generate` → 比对期望
kernel），另有 `tests/v35/ut/optimize/test_vf_partition.cpp` 与
`tests/v35/st/optimize/vector_func_st.cpp`。注意 **MLIR 的 E2E harness 把 soc 钉死在
`Ascend910B1`**（`mlir/test/e2e/cpu/CMakeLists.txt:9`），走的是 V1，VF 永远不触发；
第一步的探针别走 MLIR E2E，直接起一个纯 optimize 的 spike 把分区后的 impl graph dump 出来。

第一个要回答的问题是结构性的：**`VectorFunc` 是带子图的节点，而 bridge today 的假设是
「一个 AscGraph 节点 → 一个 AFIR op」**。AFIR 要么得有一个带 region 的 op，要么 bridge
把子图摊平、另想办法保住「这几个节点是一个区域」。

---

## E5 收尾与三条被推翻的判断

E5 的 Select / Where 之后还剩三件，都做完了或查清了。**其中两条当初写下的卡点，
对着代码核过之后不成立**——记在这里是因为它们已经被当成「前置」写进了后面几波的计划。

### ClipByValue　**已接通两侧**

两侧同名 `ClipByValue`，都是连续 tile 形态、没有循环形态：mem
`ClipByValue(dst, x, min, max, calCount, tmp_buf)`（要 tmp），reg 同名但少 tmp（转发给
`Clamp`）。每侧只有一个扩展头，两份头文本本来就在各自 registry 里，**不需要前置 A**。

三元 ODS 因此分成两族，差别只有一条：第一个操作数进不进 `AllElementTypesMatch`。
选择族的它是 ui8 掩码、不进；clip 族的它是 x、与 min / max 和结果同 dtype、要进。躯干提成
`AFIR_BufferizableTernaryElementwiseBase` 收一个 trait 列表，两族各剩一个类头。捎带把
`AFIR_TernaryOpBase` / `AFIR_TernaryOp` / `AFIR_TernaryMathOp` 删了——clip 是它们唯一的
实例，换基类之后一个实例都不剩（同收口第 5 条那条规则）。

`clipbyvalue.h` 原有重载是 `(tmp_buf, calCount)`，与这族其余包装相反。给它补了一个
反向的转发重载，让 MLIR 这条路上只有一种顺序、与选择族共用一个 emitter；原重载没动，
legacy 生成的 kernel 文本一个字不变。

### MaskedFill reg 侧　**当初写的卡点是错的，而且这个 op 在正式流程里到不了**

E5 的 commit message 里写「那块 UB 是留给标量形态铺 block 张量用的」。**不是。** 三条：

- block 张量有自己的一块 32 字节 TBuf（`TPipe::BlkTensorAllocAndInit`，对每个 `Scalar`
  节点发 `InitBuffer(<name>_tbuf, 32)`），与节点的 tmp buffer 是两块 UB。
- `WhereRegApiCall` 的 .cpp 与 .h 里 `tmp` 出现 **0 次**（五个分支都没有），
  `where_reg_base.h` 的五个 `WhereExtend` 重载也都没有 tmp 形参。那块 UB 分了没人用。
- 它是从 v1 抄来的残留：v1 的 `SelectAscIrCodegenImpl` 基类本来就有 `CalcSelectTmpSize`，
  而 `CalcMaskedFillTmpSize` 的函数体就是 `return CalcSelectTmpSize(node);`——那个 override
  在 v1 里就是恒等的。v2 父类不再要 tmp，override 被原样搬了过来。

更要紧的一条：`MaskedFillInputReorderPass` 对**每一个** MaskedFill 节点无条件改写成
`Select`（唯一跳过条件是输入数 < 3，没有任何能力判据），注册在 V1 / V2 两个 pass runner 里，
跑在 `Optimizer::Optimize` 的最前面——早于调度、早于 buffer 分配、早于 bridge。所以
`MaskedFillAscIrCodegenImpl{,V2}` 两侧都是不可达代码，legacy 里唯一那个 case
（`tests/st/backend_e2e/isinf_maskedfill_test`）跑到 codegen 时节点已经是 Select 了。

**推论**：reg 侧不阻塞任何东西，也不可能靠 E2E 证明；E5 注册的 mem 一侧同样只有手写
MLIR 到得了。要收口是 legacy 侧删掉那个 override（顺带删 v1 那个恒等的），AFIR 注册成
`TakesTmpBuffer=false`——收益是去掉一块没人用的 UB，不是「接通一个 op」。
**验收条件 1 里 MaskedFill 这一项要重新定义**：它不可能「端到端」。

### 「标量形态那套调用」不是那三个的前置

当初写的是「`Where<true,true>` + second-to-last stride 是 MaskedFill reg 侧、比较族、
TrueDiv 的共同前置」。三项都不成立，而且它本身是**两件事**被并在了一起：

- **循环（2D）形态**：`outer_repeats` 非空才走，`Where<false,false>` 的普通张量形态也走
  这一支，跟标量无关。这才是「连续 tile + 元素个数」说不出的那套。
- **标量铺 block 张量**：`<true,*>` 加 `local_blk_tensor_of_X[0]`，只在循环形态里出现。
  连续形态下标量直接打成 C++ 标量（`GenerateNoLoopCase`，5 实参），那是 E3 已经做掉的
  tensor-scalar 形态。

比较族不需要它：`CompareV2ApiCall` 在连续形态下发的是
`CompareExtend<T, 1, CMPMODE::X>(y[off], x1[off], x2[off], {size}, {1}, {1})`——dim=1、
stride 数组全是字面量 1，今天的模型说得出。它真正要的是「模板实参带 CMPMODE + 实参是
brace-init 数组」这一个新 emitter 形状。TrueDiv 同理：tensor-tensor 是连续形态，
今天的模型说得出。

**这一段里「TrueDiv 的新东西是显式 dtype 模板实参」那句后来证伪了。** legacy 打的是
`TrueDivExtend<T, Ty>(y, x1, x2, tmp, size)`，但 `printApiExtensionBinary` 一个模板实参
都不打，照样编得过——两个模板形参都从实参推得出来（`dst` 是 `LocalTensor<U>`、两个 src
是 `LocalTensor<T>`）。**接通它一行 emitter 都没改。** 别照着这一段去给 `Pow` / `Cast`
预留「显式模板实参」那个形状，先看包装签名推不推得出来。

**所以循环形态应该从「E5 / 比较族的前置」里摘出来单独排**，它谁也不挡。

### 顺带修掉的三个 bug

都是写用例撞出来的，不是通读代码找出来的。

1. **`where.h` 缺 `utils.h` 前导**（E5 引入，真 bug）。`where.h` 与 `duplicate.h` 都用
   `KernelUtils::` 和 `MAX_REPEAT_TIME`，只有 `utils.h` 声明，而 `WhereContiguousOp` 没
   声明前导——生成的 kernel 编不过。选择族当时只有 lit（比对调用文本，不编译内联进来的
   头），所以一路绿到底；是新加的 clip E2E 在 `clipbyvalue.h` 上撞出来的同一条，两处一起修。
2. **`AscGraphBuilder::Select` 的 `follow_index`**（测试框架）。它按默认值 0 从**掩码**
   继承输出 tensor 属性，于是 select 节点的输出 dtype 成了 ui8。改成跟第二个输入。
3. **`AddConfiguredOp`**（测试框架，新增）。带自有 `ir_attr` 的算子（LeakyRelu 的
   `negative_slope`）过不了通用 `Op<>` 通道，加一个建节点前作用在 op 上的回调。

---

## E2E 形状换掉 2~4 轴，以及它撞出来的搬运对齐缺陷（2026-09-23）

**这一波没接任何 op，产出是一个在 NPU 上静默算错的缺陷。**

### 1. 做了什么

44 条 `[128]` 注册（`ctest -N` 里带 `x128` 后缀的那些）全部换成 **2~4 轴、每轴取
[1, 512] 里的随机值**。约束两条：

- **种子固定、一次抽定、写成 CMake 里的字面量。** 不是每次跑随机——形状每次都变的用例
  红了没法复现，等于没有用例。
- **元素总数约束在 [2048, 262144]。** 下界保证一趟切出来的 count 多于一个 repeat
  （f32 64 个 / ui8 256 个），上界让最慢的用例仍在秒级（对照：既有的 `128x128x128`
  是 2M 元素、2.5 秒）。每轴按对数均匀抽，所以 4 轴的形状也能落在这个区间里。

注册名跟着改成「目录名 + 形状后缀」（`load_add_store_1x67x209`），与既有的
`load_add_store_128x23x5` 同形。**形状只在 CMakeLists 里**，用例源码从生成的
`CaseConfig.h` 读，所以 44 条里绝大多数一行 C++ 都不用改。动到 C++ 的一共 7 个用例目录，逐条有由头：
`load_exp_sqrt_store`（通查式负断言，见下第 4 条）、`load_add_index_expr_dynamic_store`
（建图与期望值，见下第 5 条）、`load_abs_reciprocal_store`（本地的 `CountOccurrences` 挪进
共用的 `E2eCase.h`，因为第二个用例要用它），另外 4 个（`load_add_index_expr_store` /
`load_add_scalar_store` / `load_bitwise_and_store` / `load_leaky_relu_store`）只是注释里
写着 `[128]` 或「这个形状探不到」，形状一换就不成立了。

### 1.5 轴会被融成一根再切，这是**正常且正确**的

调度把所有轴融成一根再切，生成的 kernel 只有**一层**循环，TilingData 的字段名把这件事写在
脸上：`load_isnan_store_14x4x161x4` 是 `z0z1z2z3t_size`，`load_add3_store_3x5x2x382` 也是。

**融轴发生在 legacy 的 optimizer 里，MLIR 没见过未融的形态**：`optimize.cpp:898` 调
`impl_graph.MergeAxis`，`optimize.cpp:965` 还专门 dump 一个 `AfterMergeAxis` 阶段。
这也是 pointwise 的通行做法（Inductor 把 pointwise 压成一个 `xnumel`；TVM / Halide 的规范
schedule 就是 `fuse` → `split` → `vectorize`；XLA 的 elementwise fusion 直接发线性下标）。
道理：elementwise 没有数据复用、各操作数 layout 一致，迭代顺序自由，一根扁平轴给的是最宽
的向量与最省的下标算术。**不要试图让 elementwise 保留多层循环。**

由此，这一波覆盖到的那根轴是「**融轴后总量切不齐**」——也就是撞出下面那个对齐缺陷的那件
事。而清单 L-7 里「`TransferPlan` 只建模一根向量化轴」那条限制**本来就不属于 elementwise**：
它只在融轴不合法时出现（广播某个操作数 stride 0、转置 / 非连续 slice、沿内轴的 reduce、
以及为复用而切的 matmul / conv）。别拿这一波去覆盖它，也别因为它没被覆盖就觉得这一波少做了事。

顺带白拿一件：`load_neg_reg_base_store_506x1x14x4` 融出来的是 `z0z2z3t_size`——**尺寸为 1 的
那根轴整根消失了**。几条形状里带 1 的用例现在钉住了这个行为，以前一条都没有。

### 2. 换完当场红 26 条，全是同一个根因

`AscendC::DataCopy` 要求 `count * sizeof(T)` 32B 对齐：

```
[ERROR][Block_2][AIV_0][kernel.cpp:830][DataCopyCheck] ... count * sizeof(T) must be 32B
align, current count value is 4659. In NPU mode, no error is reported. The value is
rounded down by 32B.
```

而 tile 的元素个数是运行期的 TilingData 字段，**尾块与整块走同一条语句**
（`v68 = v67 ? v37 : v11`），凡是总量切不齐的图，尾块就不对齐。
**NPU 上不报错，按 32B 向下取整——尾巴静默丢掉。**

`[128]` 把它藏得很严：f32 的 128 个元素是 512B，恒对齐。**这与 L-2 是同一种盲**，但后果
更重：L-2 是一个属性的缺省极性，这一条是每一张 mem 侧图的搬运。

**判据是 per-dtype、而且在 block 切分之后才定，看总量看不出来。** 44 条里总量不是 8 的倍数
（f32 的 32B = 8 个元素）的正好 26 条，红的也正好 26 条，但**不是同一批**：
`load_isnan_store_14x4x161x4` 的总量 36064 是 8 的倍数却照样红，因为它的输出是 ui8，
32B = 32 个元素；`load_bitwise_and_store_56x73` 是 i16，要 16 个。所以挑形状时别去算
「总量除得尽吗」——**每个张量按自己的 dtype 算，而且要算的是切分之后尾块的那个 count**。

### 3. 修法：一律发 `DataCopyPadExtend`

`DataMovement.cpp` 原先注册四个 implementation（`load.gm_ub` / `load.gm_ub.data_copy_pad_extend`
及 store 的两个），由 `DataMovementSelectionPolicy` 按 `TargetFeature::DataCopyPadExtend`
二选一，而那个 feature 按 NpuArch 推导、2201 = Unavailable。改成每个 op 一个
implementation，一律发 wrapper，与 legacy 的 `load_api_call.cpp` / `store_api_call.cpp`
一致——那两处也没有第二条路。

**没有按「count 对不对齐」分流**：lowering 这一刻看不出运行期的 count 对不对齐，
真按形状分流就又留下一条只在某些形状上走到的路，正是这一条缺陷本来的形状。

**性能**：与 legacy 持平，不是回退。`DataCopyPadExtend` 展开成 `AscendC::DataCopyPad`，
`blockCount = 1`、`blockLen = count * sizeof(T)`，仍是一趟连续搬运；legacy 的 v1 / v2 Load 与
Store 本来就一律走它，所以 MLIR 这条路从「比 legacy 少一次 pad 处理」变成「与 legacy 逐字
相同」。原先那条裸 `DataCopy` 快的那点，买的是对齐形状下的正确结果加不对齐形状下的错结果。

跟着删掉的：`TargetFeature::DataCopyPadExtend` 整条（枚举、按 arch 的回落、loader 的名字
表）。它只有这一个消费者，而它的判断本身是错的——`DataCopyPadExtend` 定义在 **mem 侧**的
`autofuse/ascendc/api/datacopy.h`，只是 `AscendC::DataCopyPad` 的模板包装，头里没有任何
arch 判断。两个只覆盖这个 feature 的 UT 改成打 `RegBasedApiFamily`（那一项此前**零直接
覆盖**，删了就是净减）。

### 4. 三条值得记住的

- **正确性不靠推理。** reg 侧那半边一直走的就是 `DataCopyPadExtend`：换形状之后 4 条
  reg-base 用例（`load_cast_reg_base_store_255x3x335` 等，形状比 mem 侧那些更刁钻）
  **一条没红**。修法在改之前就已经有半边在生产里跑着。
- **「会牵连很多」这种估计要 grep 一遍再写。** 清单 L-7 与广播设计文档都写着「放开这个门
  会改动现有 40 多个 mem 侧 E2E 与一批 lit 的 Load / Store 文本」，所以判成「另立项」。
  实际要改的是 **2 个 lit 文件**，E2E 的 `CheckGeneratedKernel` 一个都没查搬运文本。
  差了一个数量级，而这个错误估计把一个静默算错的缺陷推迟了一轮。
- **数「几个文件受影响」时，两种拼法都要搜。** 上面那个「2 个」第一次数成了 1 个：只
  grep 了 `DataCopy`（**发出去的 C++ 文本**），漏了 `data_copy_l2`（**方言 op 名**）。
  漏掉的那个是 `demo/afir_pipeline_sync_canonical.mlir`，它的 `FINAL` 钉的是 lowering 的
  产物，全量跑起来才红。**同一件事在这个仓里有三种写法**：`DataCopyPadExtend`（C++）、
  `data_copy_pad_extend`（方言 op）、`.data_copy_pad_extend`（implementation id）。
  另有 `demo/ascendc_vector_target.mlir` 直接手写 `ascendc.data_copy_l2` 做方言往返，
  那个 op 仍在方言里、那条用例照旧绿——**不产出它不等于删掉它**。
- **通查式的负断言会被无关改动打破，而它本来想说的是别的事。**
  `load_exp_sqrt_store` 断言 kernel 里「一个 `Extend<` 都没有、一个 `__ASCENDC_API_` 都
  没有」，想说的是「exp / sqrt 不走包装」，但写成了全 kernel 通查，于是搬运一改就红。
  改成数个数（`Extend<` 两处 = load 与 store，`#define __ASCENDC_API_` 一处 = `datacopy.h`），
  强度不降。**写负断言时把主语写清楚：是这个 op 没有，还是整个 kernel 没有。**

### 5. 验证

`bash ./mlir-test.sh` 四个目标全绿：`check-ut` / `check-lit`（162 条 100%）/ `check-st` /
`check-e2e`，退出码 0。CPU E2E **47 / 47**，含 4 条 reg-base 与 3 条既有形状。

**两层都验过，不是只看绿**：

- 修之前 26 条红，报的是 `DataCopyCheck`；修之后同样的形状全绿。**红的那一轮就是埋点**
  ——它证明这批形状确实走到了那条判据，不是「换了形状但什么都没多覆盖」。
- reg 侧 4 条在改产品代码**之前**就是绿的（它们本来走 wrapper），改完照旧绿：说明改动没有
  动到本来对的那半边。

### 6. `load_add_index_expr_dynamic_store` 顺带改成真多轴

这条用例原先只登记一个 size var，值填成元素总数，形状换成多轴也还是一根轴。改成**每根轴
各登记一个符号**（`s0` / `s1` / ...），`g_e2eRuntimeShapeArgs` 按轴给值。表达式仍是
`2 * s0`，于是它现在钉的是「多个运行期尺寸字段 + 轴不参与替换」——只有一根轴时后半句
与「没有轴」看不出区别。KernelTest 里那个写死的 `kRuntimeExtent = 128` 改成从形状表读。

生效的证据看生成的 tiling 入口签名，不看用例绿不绿：

```cpp
extern "C" int64_t AutofuseTiling(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, ...)
  return AutofuseTiling(24, 16, 2, 3, tiling, workspaceSize, blockDim, nullptr);
```

四个 `uint32_t` 形参，按轴给值。**退化成一根轴的话这里只会有一个形参，而用例照样绿。**

---

## 标量形态的三种发射规则（2026-09-24）

七个 op 的标量形态一起接通：**Sub / TrueDiv（只 mem）、Div / Pow / ClipByValue / Select /
Where（两侧）**。这一波的产出不是七个 op，是把「标量形态怎么发射」分成了三条规则——
在这之前只有第一条，而按第一条去做另外两族会**静默算错**。

| 规则 | op | legacy 发的调用 | 为什么是这一条 |
|---|---|---|---|
| 一、tile 挪到第一位，方向丢掉 | add / mul / maximum / minimum / eq / ne / logical_and / logical_or | `Adds(dst, tile, s, cnt)` | 可交换，交换完逐字相同 |
| 二、tile 挪到第一位，**方向落到 bool 模板实参** | sub / div / truediv | `Subs<T, IS_SCALAR_LATTER>(dst, tile, s, tmp, cnt)` | 不可交换，`a - s` 与 `s - a` 靠那一位区分 |
| 三、**位置原样保留** | pow / select / where / clip_by_value | `Pow(dst, a, b, tmp, cnt)`、`Where(dst, mask, a, b, cnt, tmp)` | 供给是**同名重载**，标量在哪位由 C++ 重载决议挑，没有模板实参承载方向 |

**规则三是这一波最容易做错的地方。** 已有的 `emissionOrder` 恒把有形状的操作数挪到第一个
实参位——对规则一、二是对的，对规则三会把 `s ^ x` 发成 `x ^ s`：**两个都是合法 C++、都编得过**，
只是算的不是同一个东西。所以规则三另起一个 pattern（`BinaryScalarInPlaceConversionPattern`），
`emissionOrder` 在那里只用来校验形态与找出哪个是 tile，发射用 AFIR 原次序。

### 改动面

- **AFIR ODS**：`sub` / `div` / `truediv` / `pow` 加 `scalarOperand = "any"`；三元两族加一位
  `scalarInputs`（只放宽后两个输入，第一个恒是 tile）。`"any"` 那段注释跟着订正——它原先写
  着「只给可交换的 op」，现在说的是「两边都可以是标量，方向怎么发是 lowering 的事」。
- **fork td**：规则二要 4 个新 def（`Subs` / `Divs` / `TrueDivExtends` / `DivExtends`，都带
  `isScalarLatter` 属性）；**规则三一个新 def 都不用建**，只把可能是标量的操作数从
  `AscendC_LocalTensor` 放宽成 `AnyType`。
- **emitter**：规则二一条新躯干 `printApiExtensionScalarBinary`（模板实参必须写出来，方向那
  一位跳不过前面的 dtype）；规则三复用原有躯干。
- **conversion**：两个新 pattern（方向 / 原次序）、三元的 `getTensorTypes` 与
  `materializeOperandTiles` 改成按「是不是 tile」逐个判。

### 核出来的四件事

1. **`Sub` / `TrueDiv` 的 reg 侧不接，是查过的**：`sub_reg_base.h` 里 `SubExtends` 供给在，
   但 legacy v2 的 `IsScalarInputSupported` 返回 false（矩阵 `reg_scalar` 列就是它的摘要），
   调度产不出那种图。接了是一个谁也走不到的 implementation。
2. **`Select` / `Where` 的标量格此前两份文档都没记过。** 矩阵上是
   `is_scalar_list[0] == false`，不是 `no`——后两个输入各自可以是标量，只有 mask 不行。
   L-4 的「范围之外」那条列了五个 op，漏了这两个。
3. **`queryScalarInput` 里那条卡点是假的。** 原文写着标量形态「走的是另一套调用形态
   （`Where<true, true>` 加 second-to-last stride）」。`where.h` 的三个标量重载（头里自己标
   着「场景 1/2/3」）是 `template <typename T> Where(dst, mask, src0, src1, size, tmp)`，与
   tensor 形态同名同位，没有 bool 模板实参也没有 stride。**又一次「文档说卡住了」不成立**
   ——这次的载体不是设计文档，是产品代码里的一条注释，比文档更容易被当成事实。
4. **clip 的三个标量重载缺一个实参序，只有 E2E 抓得到。** 头里场景 2/3/4 是
   `(…, tmp_buf, calCount)`，而这条路上这一族的实参序是 `(…, calCount, tmp_buf)`
   （场景 1 早先补过一个转发重载统一它）。六份 lit **全绿**——它们比的是发出来的文本，而
   文本是对的；红的是 device 编译器：`no matching function for call to ClipByValue(...)`。
   又一次「lit 绿证明不了 kernel 编得过」，按场景 1 的同一个做法补了三个转发重载。
5. **`ClipByValue` 的 `x` 不能是标量，虽然矩阵那一格是 `any`。** 头里只有「三个全是标量」
   那一个重载收标量的 `src_x`，而它没有 count / tmp 形参。矩阵的 `any` 是
   `IsScalarInputSupported` 的摘要，**它比头里真实存在的重载宽**——legacy 自己在那个组合上
   也会编不过。按头写，不按矩阵写。

### 验证

- lit 六份：`afir-scalar-binary-direction-api.mlir`（三个 op × 两个方向，模板实参与 id）、
  `...-reg-base-api.mlir`、`afir-pow-scalar-api.mlir`（两个方向的实参位置）、
  `afir-ternary-scalar-api.mlir`（三种标量组合）、`...-reg-base-api.mlir`、
  `afir-ternary-scalar-invalid.mlir`（第一个操作数是标量要被 ODS 拒）。
- CPU E2E 两条，各钉一条规则的数值：`load_scalar_sub_store`（`2.5 - x`，方向那一位写反就
  变成 `x - 2.5`，逐元素红）、`load_clip_by_value_scalar_store`（`clip(x, -1, 1)`，上下界不
  对称，位置错就红）。
- 埋点验过：把方向那一位钉死成 `true`，lit 当场红。

---

## 剩余五个 op 的 E2E 数值覆盖，以及它撞出的循环生成缺陷（2026-09-24）

上一节接通七个 op、只留了两条 E2E（`sub` 与 `clip_by_value`）。**剩下五个补齐**：
`div` / `truediv`（规则二）、`pow` / `select` / `where`（规则三），一个 op 一条，全部
mem 侧、`Ascend910B1`。至此七个 op 各有一条数值用例。

### 1. 五条用例各钉什么

| 用例 | 图 | 钉的是 |
|---|---|---|
| `load_scalar_div_store` | `4.0 / x` | 方向位 `false`。`Divs<T, false>` 与 `Subs<T, false>` **不共用代码**：方向在 `scalar_div.h` 里落成 `BinaryRepeatParams` 的四个 stride 与两处实参次序，那一整段只有 `false` 才走到 |
| `load_truediv_scalar_store` | `x / 4`，i32 → f32 | 方向位 `true`，**缺省那一侧的第一条数值用例**；外加 i32 支：tile 先 `Cast` 成 f32、标量按 `U` 强转，cast 结果与 `Divs` 的 `Duplicate` 共用同一块 tmp 分两段 |
| `load_scalar_pow_store` | `2.0 ^ x` | 实参**位置**。`x ^ 2` 与 `2 ^ x` 都是合法 C++、都编得过，lit 那一层的文本也正常 |
| `load_select_scalar_store` | `isnan(p) ? 3.0 : v` | 标量在 then 位（`where.h` 场景 2） |
| `load_where_scalar_store` | `isnan(p) ? v : 7.0` | 标量在 else 位（`where.h` 场景 3），**也是 `afir.where` 的第一条 E2E** |

`select` 与 `where` 在这条路上发的是同一个调用，差别只在 implementation id，kernel 文本
看不出来——**所以两个 op 各留一条**，理由与 `load_div_store` / `load_sub_store` 分两份一样。

数值取法：`div` 取 ±2 的整数次幂、`truediv` 取 [-9, 9] 的整数、选择族取 0.25 的奇数倍，
商与结果在 f32 里都精确，逐元素比相等；只有 `pow` 比相对量（设备上算的是 `exp(x·ln2)`，
口径抄 `load_pow_store`）。每条的取值都让「发反了」的那个函数与期望**处处不同**，
`pow` 那一档七个点里只有 `x = 2` 相同。

### 2. 撞出来的缺陷：循环生成只提升 `memref.alloc`，不提升标量常量

**`select` / `where` 两条在 codegen 就红**，报的是 `operand #1/#2 does not dominate this
use`，连 kernel 都没生成。

- **根因**：`LoopGeneratePass` 把循环建在**第一个 AFIR Op 的位置**上，然后把调度出去的 Op
  搬进 `scf.for`。`hoistLoopInvariantAllocations` 只把 `memref.alloc` 那条链提到循环之前，
  而 bridge 是按 AscGraph 的节点次序**就地**建 `Scalar` / `IndexExpr` 的 `arith.constant`
  的。节点排在第一个搬运之后时，常量就留在了循环**之后**，用它的 Op 已经在循环体里。
- **为什么此前没人撞上**：`load_add_scalar_store` / `load_scalar_sub_store` /
  `load_clip_by_value_scalar_store` 的图都只有一个 `Load`，标量节点排在它之前，常量正好
  落在第一个 AFIR Op 之前。**这一批的掩码由图里 `Isnan` 现算**，标量夹在两个计算之间，
  才第一次走到后面那个位置。
- **修法**（`LoopGenerate.cpp`）：把提升的种子从「操作数是 `memref.alloc`」改成
  「操作数的定义方**不是**调度出去的 Op」。`memref.alloc` 只是其中最常见的一类；
  这条改法覆盖的是「调度出去的 Op 读到的每一个循环不变值」，不是再加一个特判。
  递归收集那一支原样不动——它撞上调度结果时照旧报错，只把消息里「buffer sizes」的说法
  放宽。
- **判据**：lit `afir-loop-generate-scalar-hoist.mlir`（常量故意写在第一个 `afir.load`
  之后，检查它被打印在 `scf.for` 之前、而 `afir.add` 在循环体里用它）+ 两条 E2E。
  **三处都验过会变红**：改之前 lit 直接报 dominance 错，两条 E2E 在 codegen 阶段失败。

### 3. 埋点

三处，各自让对应的用例**在数值上**变红（为了确认红的是数值而不是 `CheckGeneratedKernel`
的文本断言，验证时把五条用例的文本断言临时摘掉）：

- 方向位钉死 `true` → `load_scalar_div_store` 红（`0.0625` 对 `16`）；钉死 `false` →
  `load_truediv_scalar_store` 红（`-0.44444445` 对 `-2.25`）。**两个方向各要一次埋点**：
  钉死成其中一侧时，本来就在那一侧的那条用例照样绿。
- 规则三的发射次序改成 `{1, 0}` → `load_scalar_pow_store` 红。
- 三元的 `source0` / `source1` 对调 → `load_select_scalar_store` 与
  `load_where_scalar_store` 红。

CPU E2E 这一批之后是 **51 个目录 / 54 个注册用例**，其中 4 条按 `PLATFORM reg_base` 跑
（reg 侧补齐后的数字见下一节）。

---

## reg 侧标量形态补齐（2026-09-24）

**此前 `*.tensor_scalar.*_reg_base` 的 12 个 implementation 一个都没有被任何编译器看过。**
四条 reg-base 用例（cast / sign / eq / neg，L-13 补的）全是 tensor 形态，标量形态那半边
只有 lit。这正是 L-13 自己的教训——**lit 比的是发出来的文本，包装头是整段内联进 kernel
的**，少一个前导、少一个重载都要到 device 编译器那一步才炸。两份文档都没记这个洞。

### 1. 按 def 补，不按 op 补

reg 侧的标量 implementation 落在 **6 个不同的 AscendC def** 上，一个 def 一条用例：

| 用例 | def | 与 mem 侧的差别 |
|---|---|---|
| `load_scalar_div_reg_base_store` | `DivExtendsRegBaseOp` | 同样带方向位，但**不收 tmp**（mem 侧的 `Divs` 要一块放 `Duplicate` 出来的 block），也不要 `utils.h` |
| `load_scalar_pow_reg_base_store` | `PowRegBaseOp` | 与 tensor 形态同一个 def；两份 `pow.h` 的 include guard **撞名**，所以钉 reg 独有的 `pow_config` / `PowerAlgo` |
| `load_select_scalar_reg_base_store` | `WhereContiguousRegBaseOp` | 包装不同名（`Where` → `WhereExtend`）、不收 tmp、不内联 `duplicate.h`；掩码那一路 reg 侧发的是标准 `AscendC::IsNan` |
| `load_clip_by_value_scalar_reg_base_store` | `ClipByValueContiguousRegBaseOp` | guard 又一次撞名，钉 reg 独有的 `Clamp` / mem 独有的 `AscendC::ClampMax`；实参序少一个 tmp |
| `load_eq_scalar_reg_base_store` | `CompareScalarExtendContiguousRegBaseOp` | 与 `load_eq_reg_base_store` 成对：那条走 tensor 的 `CompareExtend`，这条走**另一个函数** `CompareScalarExtend`。六个比较 op 共用这一个 def |
| `load_logical_and_scalar_reg_base_store` | `LogicalAndExtendsRegBaseOp` | 这一族 reg 侧**只有 `logical_and` 有标量形态**（`logical_or` 没注册）；不收 tmp、不要 `utils.h`，两条都与 mem 侧相反 |

前四条与各自的 mem 侧用例**同一张图、同一份数值断言，差别只有 soc**，照
`load_neg_reg_base_store` 的做法。后两条是新图：mem 侧本来也没有这两个的标量用例。

### 2. 核出来的三件

1. **两侧的 tmp 契约有三处是反的**：`div` / `logical_and` / `clip` 的 mem 侧要 tmp、reg 侧
   不要。调度那一侧按 NpuArch 分流算 UB，两边对不上时 `checkTmpBuffer` 当场判 unsupported
   ——所以这几条同时也是「换 soc 的三处都改对了」的判据。
2. **`select` 的 reg 侧是 `TmpBufferUse::Unused`**：调度照样分了一块，`WhereExtend` 的签名
   里没有那个形参，于是 kernel 里有一块取出来、`ReinterpretCast` 过、**没有任何调用读它**
   的 `TBuf`。与 `load_eq_reg_base_store` 同一个形状，它不该被优化掉。
3. **guard 撞名有两处**（`pow` / `clipbyvalue`），钉头文件那一套判据在这两个 op 上直接失效
   ——只能改钉两份头各自独有的函数名。写这类判据前先 dump 一眼实际输出。

### 3. 埋点

去掉 `load_clip_by_value_scalar_reg_base_store` 的 `PLATFORM reg_base`（也就是让它跑回
mem 侧），`Case.cpp` 的三条断言同时红：实参序那条、`Clamp` 那条、`TBuf` 不出现那条。
**这一条埋点管的是整批**——它证明这些用例真的在分辨两侧，而不是「换没换 soc 都绿」。

数值那一侧不需要另做埋点：前四条的数值断言与各自的 mem 侧用例逐字相同，而上一节的三处
埋点打在**两侧共用的** conversion 代码上。

CPU E2E 现在是 **57 个目录 / 60 个注册用例**，其中 **10 条**按 `PLATFORM reg_base` 跑。

---

## 剩余任务

**不是只剩 E5 / E6 / E7。** 下面这张表是 E5 之前的口径，**现在的实际数字见本节末尾的
「重算」**：E5 做完、LeakyRelu / BitwiseAnd / FloorDiv / Remainder 接通之后是 47 里接通
24（另有 Broadcast 挂在白名单上但当时仍是断头路），剩 20 个 op 加 3 个非 op。

按 op 清点，47 个共有 elementwise 里还有 **31 个**没接通，
E5 与 E6 的清单只盖住其中 14 个：

| 去处 | op | 个数 |
|---|---|---|
| E5（方案已列） | Select / Where / ClipByValue / MaskedFill | 4 |
| E6（方案已列） | Broadcast / Cast / Div / TrueDiv / Pow / FloorDiv / Remainder / Axpy / Gelu / LeakyRelu | 10 |
| **E6（方案漏列，本次补进）** | **Sub / BitwiseAnd / Sign** | **3** |
| E3 剩余档 | `ScalarData` / `IndexExpr` | 2 |
| **挂起、至今没有指定波次** | Rsqrt / LogicalNot / Tanh / Erf / Eq / Ne / Lt / Le / Gt / Ge / LogicalAnd / LogicalOr | **12** |

三个漏列的来由：

- **Sub**：方案里 E3-4 和偏差都写着「压在 Sub / Div 的 tensor-tensor 形态之后（E6）」，
  但 E6 那一行的清单里只有 Div 没有 Sub。是漏写，不是另有安排。
- **BitwiseAnd**：D2，mem `BitwiseAndExtend` + `bitwise_and.h` + tmp，reg `BitwiseAnd`
  无头无 tmp——与已经做完的 Reciprocal 同形，是剩下这批里最便宜的一个。
- **Sign**：D2，两侧都走包装且都要 tmp（`cast.h;sign.h` vs `cast.h;sign_reg_base.h`），
  卡在下面那条「一个 op 两个扩展头」。

所以顺序上：**E3 剩余档 → E5 → E6 → E7**，其中 12 个挂起的 op 必须在 E7 之前各自找到
落点，否则验收条件 1（47 个全部端到端）达不到。

### 重算（比较族之后）

对着 `ascir_op_lowering_matrix.csv`（`family == elementwise`）与
`AFIR_BRIDGE_OPERATION_REGISTRY` 的 `Both` 列重新做的差集，不是人工数的。
**这张表是 elementwise 的一个划分，三行加起来必须是 47**——不是 bridge registry 的行数，
那个数在上面「已落地」里，口径不一样。

| | 个数 |
|---|---|
| buffer 形态已接通 | 44（含 Broadcast，2026-09-23 起不再是断头路） |
| 还没接通的 op | **0** |
| 非 op（`Scalar` / `ScalarData` / `IndexExpr`） | 3 |

**op 这一列清零了。** 比较族六个是最后一拨，接通之后 47 个 elementwise 节点类型里
只剩 3 个非 op（`Scalar` 已接通，`ScalarData` / `IndexExpr` 是 E3 剩余档）。

**这不等于 elementwise 收尾了**，还欠三件，各自有落点：

- ~~**前置 D**（一条队列上不能混 dtype）~~ **已解**（清单 L-6）。判据去掉 dtype，
  字段清单同时从两份收成一份。顺带推翻了本文档原先的说法：挡住的不是
  `compare → select`（那张图一直是绿的），而是**掩码自己从 GM 搬进来**的那种形状；
- ~~**L-4**（swap 方向的标量）~~ **已解**：`RhsForm` 换成 `ScalarOperand{None,Rhs,Any}`，
  8 个可交换 op 的那半张格子接上了。原先估的三处改动少一处——`is_scalar_latter` 只发给
  Div / Sub / TrueDiv，与这 8 个无关；
- ~~**`Broadcast` 仍是断头路**~~ **已接通常量源一档**（2026-09-23，豁免名单清空，
  设计见 `2026-09-23-afir-broadcast-design.md`）。剩的是三轴 / 四轴 `Broadcast(...)`
  那两档，前置是逐轴的 `TransferPlan`；`ui8` 那一位比 legacy 窄，由
  `ElementTypeMatrixTest` 的 `kKnownNarrower` 快照表钉死。清单 L-7。

**「照样板再走一遍就能接通」的 op 一个都不剩了**——这句从 `Cast` 那一轮起就成立，
比较族是最后一个印证它的：那条卡点写着三件新机制，核完只剩一件（见下「比较族：
已接通」），**但那一件得先核过才知道**，账不是照着文档估出来的。

**剩下这些里，两侧供给都真正可用的是 0 个。** 每一个都至少要先补一处 fork 侧 td def
（TrueDiv 是最后一个只缺一处的——它的 reg 侧连新 def 都不用建，直接复用了
`AscendC_DivExtendRegBaseOp`）。
BitwiseAnd / FloorDiv / Remainder 是头三个按这条账做完的（见下），两侧各补了一个 def，
正好是「四处起步」的样板。

这个数字比第一次算出来的严：第一次只拿矩阵的 `mem_api_name` / `reg_api_name` 去 grep td
里的 apiName 字符串，结果把 Axpy / LogicalAnd / Pow / Gelu 四个误判成「有供给」。两处错：

- **apiName 撞名**。`defm Axpy : VecScalarL012Op<"axpy", "Axpy">` 给的是标准的
  `AscendC::Axpy`，而矩阵两侧要的都是 `AxpyExtend`（带 tmp、引 `axpy.h`）。名字沾边，
  函数不是同一个。
- **有 def 不等于 def 被展开**。见下一节。

重算要**同时**排除这两种，办法是：从 `Ops.td` 传递展开 include 图、只在真正到得了的文本里
找 apiName。~~并把 `MathLibraryOp` 那一族单独标出来（偏差 6 挡着）~~——**这一句作废**：
偏差 6 不是卡点，`MathLibraryOp` 的 def 照用（`tanh` / `erf` 用的就是它），
不必单列一族。

**E6 按这张账重估：剩下的 op 个个都要补 fork 侧 td def**（原写三条结构性前置：偏差 6
是假卡点，前置 B 那一轮做掉了，前置 D 已解——而且它挡的范围比这里写的小，见清单 L-6）。
**这一波不是「再来一轮三处改动」，是「四处起步」。**

### ~~有 9 个 td 文件从来没被 `Ops.td` 包含过~~　**七个死文件已于 2026-09-24 随 L-17 删除**

> 下面这一节记的是当时的分析，路径按当时的位置写。那七个 `*Ext.td` 已经不在仓里了，
> 门禁 `check_ascendc_td_reachable.py` 的豁免表现在只剩 `Doc.td`。

`Basic/` 下的七个 `*Ext.td`——`OpVecBinaryExt` / `OpVecCastExt` / `OpVecCompareExt` /
`OpVecDataMoveExt` / `OpVecReduceExt` / `OpVecScalarExt` / `OpVecTernaryExt`——自最初集成起
就不在 `Ops.td` 的 include 图里（另两个是 `Ops.td` 自己和 `Doc.td`）。**里面 41 个 def
一个都没有被展开**，也就从来没被 tablegen 检查过。

E5 已经吃过一次这个亏的边：`where_extend` 就在 `OpVecTernaryExt.td` 里，看着像现成供给，
实际既没被展开、建模的又是另一个 overload。

**不要把这七个文件当成现成供给**。抽查过的两个都对不上：

| 文件里的 def | 它声明的 | 实际要的 |
|---|---|---|
| `AscendC_CompareExtendOp` | apiName `"Compare"`，带 `AscFunc` → 打成 `AscendC::Compare` | 全局的 `CompareExtend<T, dim, CMPMODE::EQ>(...)` |
| `where_extend` | 12 操作数的循环形态 | 连续形态的 5/6 操作数 |

所以比较族的「缺 reg 侧供给」是真缺，不是「文件没 include」那么便宜。把这七个文件接进来
既救不了场，还会一次性引入 41 个没编译过的 def。**要补就在已经被包含的文件里新增
（E5 加 `where_contiguous` 就是这么做的），别去动那七个文件。**

这条已经有门禁了：`check_ascendc_td_reachable.py`（pre-commit 的 `ascendc-td-reachable`）
断言 `Dialect/Asc/IR/` 下每个 td 都从 `Ops.td` 传递可达，那七个加上 `Doc.td` 列在脚本的
豁免表里——**列在那里是为了让它们可数、有来由，不是为了放行**。脚本同时校验豁免表本身：
文件被接进来了、或者被删了，却忘了从表里拿掉，一样拦。

这条 hook 配的是 `always_run: true`，不是 `files:`。**写 `files:` 会静默失效**：它要看的
目录整个在仓库级 `exclude` 里（`mlir/api/ascendc/` fork 自上游，任何 hook 都不许碰它），
而 `exclude` 无条件生效，`files:` 被过滤成空之后 pre-commit 只打一行「no files to
check」就跳过。门禁最初就是这么加的，白设了两个提交才发现——**加针对这个目录的门禁一律
用 `always_run`**。

### BitwiseAnd：「四处起步」的第一个样板

按上面那张账做的第一个 op。它与一元的 Reciprocal 同形（一侧标准调用、另一侧走包装且
要 tmp），差别全在「两侧的 def 都得现写」，正好把这一波的改动面量出来：

| 改动处 | 内容 |
|---|---|
| AFIR td | `AFIR_BinaryMathOp` → `AFIR_BufferizableBinaryElementwiseOp`；手写的 `verify` / `AFIR_INFER_BROADCAST` 跟着删掉 |
| **fork td（mem）** | `AscendC_BitwiseAndExtendOp`，扩展头 `bitwise_and.h`，收 tmp |
| **fork td（reg）** | `AscendC_BitwiseAndL2Op`，标准 `AscendC::BitwiseAnd`，不收 tmp |
| 发射 | `printApiExtensionBinary`（新增，二元一族的第一个扩展包装躯干）+ 两处 `Translation.cpp` 登记 |
| conversion | 两个 implementation，`ElementDomain` 新增 |
| bridge | 注册表一行 |

两件值得单记的事：

- **元素类型的取值范围成了每个 implementation 自己的一件事。** 在此之前二元路径只有
  一个全局 `isSupportedElementType`（f16 / f32 / i16 / i32）。位模式这一族的集合与它
  既不相等也不包含：浮点只在算术那边，`ui8` / `ui16` 只在位这边。于是加了
  `ElementDomain`，与一元路径的 `ResultElementType` 同一个写法。
- **`BitwiseAndExtend` 的 tmp buffer 是个没人用的形参。** 包装体里一次都没引用它
  （`bitwise_and.h`），legacy 仍按 `CalcDefaultTmpSize` 给它留了一块。这里跟着留——
  少给一个实参就调不上。这是继 MaskedFill 之后第二个这样的形参，两处的来由不同：
  MaskedFill 那个是复制粘贴的残留，这个是包装自己签名里就有。

`load_bitwise_and_store` 还是**第一个非浮点的 CPU E2E**：到 LeakyRelu 为止这条 harness
上两端的 GM 张量全是 f32（谓词族的 ui8 只在图中间出现）。「元素类型不是 f32」在 tiling、
搬运、kernel 签名上到底成不成立，到这个用例才第一次被端到端走过。

### FloorDiv：两侧同名，差的是头、tmp 和元素类型

紧接着的第二个。它是**二元里的 `neg`**——两张供给表给的是同一个函数名 `FloorDivExtend`，
只有扩展头不同，所以照 `AscendC_NegExtendOpBase` 写成一个 base 加两个 def。与 neg 不同的
是这一对还差两件事，两件都落在 implementation 上：

| | mem（`floor_div.tensor.extend`） | reg（`floor_div.tensor.extend_reg_base`） |
|---|---|---|
| 扩展头 | `floor_div.h` + `utils.h` 前导 | `floor_div_reg_base.h` |
| tmp buffer | 要（装 `Div` 的中间商再交给 `Floor`） | 不要（寄存器里做完） |
| 元素类型 | f16 / f32 | f16 / f32 / i16 / i32 |

**元素类型这一行是新出现的情况：同一个 AFIR op 的两个 implementation 落在不同的族。**
于是 `ElementDomain` 添了 `Float`（Div / TrueDiv / Pow / Gelu 之后都用得上），ODS 那一层
写两侧的并集。

顺带订正了一个 ODS 错误：`AFIR_FloorDivOp` 原来写的是 `[I16, I32]`——与 mem 侧**完全不
相交**、又是 reg 侧的真子集，两边都对不上。它没被发现是因为这个 op 当时还没有任何
implementation，ODS 的类型表在那之前谁也没有去比对过供给。**剩下的 op 里可能还有同类的
错，接的时候要拿 legacy 的 `REG_ASC_IR` 逐个对一遍，别信 ODS 现有的那一行。**

`load_floor_div_store` 是第一个把扩展头的**前导**也一起编出来的二元用例：`floor_div.h`
用了 `MAX_REPEAT_TIME`，只有 `utils.h` 声明。lit 比对的是调用文本、不编译内联进来的头，
所以这条只有 E2E 拦得住——`where.h` 当初正是这么漏过去的。

### Remainder：第一个现建 ODS 的 op，以及「元素类型族」这个抽象的报废

三个里最整齐的一个：两侧同名 `RemainderExtend`、**都要一块 tmp buffer**，只有扩展头和
收的元素类型不同，于是连 `arguments` 都进了 base，两个 def 各剩一行。

两件值得单记的事。

**一、它是本切片第一个 ODS 上原本就没有、现建的 op**（在方案的「ODS 还缺 3 个」名单
上）。此前接通的每一个，`afir.*` 的定义都已经在，改的是「换个 base 类」。所以从 pyasc
的 `Remainder` 节点一路到 kernel 里那行调用，中间每一段都是第一次走，
`load_remainder_store` 也就比前几个用例多兜住一层。

**二、`ElementDomain` 那个枚举接到第三个 op 就报废了，改成了位掩码。**

它是接 BitwiseAnd 时加的，想给反复出现的那几个集合各起一个名字。到 Remainder 手上有
五个集合：

| | 收的元素类型 |
|---|---|
| add / mul / maximum / minimum | {f16, f32, i16, i32} |
| bitwise_and（两侧） | {ui8, i16, ui16, i32} |
| floor_div mem | {f16, f32} |
| floor_div reg | {f16, f32, i16, i32} |
| remainder mem | {f32, i32} |
| remainder reg | {f16, f32, i32} |

**两两都不是包含关系，也没有共同的概念。** 再起名字只能叫 `FloatOrI32`——那不是名字，
是把集合又拼写了一遍。所以换成位掩码，注册处直接写出那几种类型：

```cpp
registry, "remainder.tensor.extend", ApiFamily::MemBased, kF32 | kI32
```

这样还有一个好处：**每一处的取值都看得见是从 legacy 的 `REG_ASC_IR` 抄来的**，不是推出
来的。接着做的人对一遍那张表就行，不用先弄懂某个族名当初为什么那么划。

顺带记一条 reg 侧的怪事：`remainder_reg_base.h` 里 `RemainderExtend` 有 4 参和 5 参两个
重载，legacy 发的是 5 参那个，而它给的 tmp buffer 尺寸是 **0**（`CalcVoidTmpSizeV2`）。
那块 buffer 不是拿来算的，是拿来挑重载的。所以 AFIR 这边两侧都得是
`TakesTmpBuffer=true`——按「reg 侧不要 tmp」的直觉写会挑到另一个重载去。

### E6 后三波：Sub / Div、Gelu / Sign、LogicalAnd / LogicalOr

三波各自顶翻了一个此前够用的假设，机制都落在共用代码里，所以单列出来——**接着做的人
会直接用到这三样**。

**Sub / Div：`TakesTmpBuffer` 那一位不够用了。** 它原先同时说两件事——capability 收不收
tmp 操作数、建 target op 时传不传那个实参——两者从来一致。`CalcBinaryApiTmpSize` 对
sub / div 的 tensor-tensor 形态也分了一块（32 字节），而这条路发的是标准的
`AscendC::Sub` / `AscendC::Div`，四个实参，没有它的位置；真正用它的是标量形态的
`Subs` / `Divs`。改成三态 `TmpBufferUse { None, Passed, Unused }`。收下它是必须的：
bridge 已经把它挂上了 AFIR op，capability 答「不要」会直接判成 unsupported。

**Gelu / Sign：家族判据改成只查一个方向。** 原先那条判据有两个前提，`sign` 把两个都
推翻了——`cast.h` 两侧是同一个文件（两张注册表灌同一个 `AscendCApiRegistry` 单例），
所以「逐项都要对」不成立；`utils_reg_base.h` 不是 `utils.h` 的 reg 版本而是另一个文件，
reg 侧的 `sign_reg_base.h` 要的就是 mem 那份 `utils.h`，所以「两张列表各判各的」也不成立。
现在只留唯一还查得动的方向：mem 侧不许点名任何 `_reg_base.h`，reg 侧至少要点名一个，
中立的头两侧通用。同一波里一元的元素类型也从一个写死的全局判据改成 per-implementation。

**LogicalAnd / LogicalOr：二元的 `sameElementType` 语义就此定死。** 这是二元一族里第一对
结果 dtype 与输入无关的 op（两侧 `T2` 都只有 `uint8`）。清掉那一位**只放开 `result` 那
一项**，两个输入之间仍然相绑——整组一起丢的话 `logical_and %a: f16, %b: i32` 能过 ODS，
要到 device 编译器才因为模板实参推导冲突报错。另外两条现场：

- 它是第一对 apiName 与 mnemonic 对不上的 def（mem 侧发裸名 `LogicalAnd`，reg 侧发
  `LogicalAndExtend`），也是第一个带 `ApiExtensionOpInterface` 的 `VecScalarOp`；
- **mem 侧的包装不归一化**：`logical.h` 把两个输入 `Cast` 成 half 之后直接取乘积（and）
  / 最大值（or），喂 4 和 4 出来的是 16。前提是输入已经是 0 / 1 真值，legacy 自己的 UT
  也是按 `std::rand() % 2` 喂的。E2E 按同一个契约写，别拿任意非零值去断言 1。

### Axpy：已接通（tier D1 第一个）

**它撞的不是供给，是家族判据。** `headersMatchFamily` 对 reg 侧的要求是「至少点名一个
`_reg_base.h`」（否则这个 implementation 够不着自己那张供给表），而 `Any` 那一档写死了
「扩展包装总是出自某一张表，不可能两张表通用」直接返回 false。Axpy 两侧是**同一个**包装
`AxpyExtend` 加**同一个**头 `axpy.h`，`api_regbase/` 下没有 axpy——三种注册法全堵死：
`Any` 撞那一行，`RegBased` 点不出 reg 头，`MemBased` 只盖得住一半。

**做法：把 D1 在 MLIR 这一层说出来。** `ApiFamily` 加了第四档 `Shared`：允许声明扩展头，
但**一个 `_reg_base.h` 都不许点名**（与 mem 侧同一条——两侧共用的那份包装住在 `api/` 下）；
`checkApiFamily` 里与 `Any` 一样不问 feature。**`Any` 保持「标准调用、无扩展头」原样、
仍然全查**，没有去放宽它：放宽的话 `logical.h` 那种 mem 专有的包装也能被注册成两侧通用。
反方向（reg 侧混进一个 mem 专有包装）查不了，但那是这条判据本来就承认查不了的，
见上「Gelu / Sign」那一段。

**这是矩阵 tier D1 第一次到 MLIR 手上。** D1＝「两侧共用扩展头」，elementwise 里共 5 个：

| D1 op | 共用的头 | 今天为什么没撞上 |
|---|---|---|
| Add / Mul / Maximum / Minimum | `scalar_add.h` / `scalar_mul.h` / `scalar_maximum.h` / `scalar_minimum.h` | 这四个的 D1 落在 **scalar⊕scalar** 形态（`Adds(dst, x, y)` 两个实参都是标量，算一次再 `Duplicate`），MLIR 还没接。已接通的 `add.tensor_scalar` 走的是标准 `AscendC::Adds`、没有扩展头，判据不触发 |
| Axpy | `axpy.h` | 第一个接的，**也是唯一没有「退回标准调用」这条后路的**——`AscendC::Axpy` 是 `dst += scalar*src`，与 `AxpyExtend` 的 `dst = src0 + alpha*src1` 不是一个函数，顶不上 |

这五个在 `autofuse/v35/ascendc/api_regbase/` 下**一个都没有**。不是 legacy 漏写了文件，
是那些文件没有内容可写：reg 侧根本没有一份不同的实现。所以另外四个接 scalar⊕scalar
形态（E3 第 1 档 / L-4 附近）的时候会一起撞上同一条，**`Shared` 这一档不是为 Axpy 一个加的**。

**两条否掉的路，连同理由一起记下来，别再走一遍：**

- **给 legacy 加 `api_regbase/axpy.h`。** `_reg_base.h` 是 `api_regbase/CMakeLists.txt:103`
  那个 `foreach` 从 `api_regbase/<name>.h` 改名生成的，加一个文件机制上很便宜。否掉是因为
  `ApiFamily` 这根轴建模的**就是平台差异**，而 Axpy 两侧在 legacy 里是同一段代码——
  apiName、apiCall、tmp 函数 `CalcAxpyTmpSize`、头四样全同（`v1_ascir_codegen_impl.h:1982`
  与 `v2_ascir_codegen_impl.h:4884` 调的是同一个函数），区别只有注册到哪张 soc 版本表上。
  加那个文件是**伪造一个不存在的平台差异**去满足一条专门用来查平台差异的判据，而且按上表
  要做五遍，每遍都复制一份 body（axpy 的 half 分支是 15 行 cast/切段/`CAST_RINT`，
  没有可转发的对象，不像 `api_regbase/sub.h` 那种三行转发给 `AscendC::Sub`）。
- **拿 `neg.h` 当先例。** 不成立，方向是反的：两份 `neg.h` 都是 `54d11b57`（最初迁移）
  带进来的，legacy v1 返回 `neg.h`、v2 返回 `neg_reg_base.h`——**legacy 本来就把 Neg
  建模成两个实现**，本仓 E2 加的 td def 只是镜像了一个已经存在的区分。Axpy 是 legacy
  只有一个实现、要往 legacy 里加第二个。

**`OpVecBinaryExt.td:39` 的 `AscendC_AxpyExtendOp` 用不了，是两条原因**（原写三条，
其中「是 `MathLibraryOp`、撞偏差 6」那条不成立，偏差 6 是假卡点）：不在
`Ops.td` 的 include 图里，**而且它的操作数顺序就是错的**
——写的是 `$alpha, $count, $tmpBuf`，而包装的真实签名是
`AxpyExtend(dst, src_0, src_1, alpha, tmp_buf, count)`，tmp 在 count 前面。

**于是新 def 的 mnemonic 叫 `axpy_extend_contiguous`，不叫 `axpy_extend`**：那个用不了的
def 已经占了后一个名字。今天不冲突（那个文件从来不被展开），但它哪天被接进来就会撞名，
而撞名的两个里能用的只有新的这个。这与 E5 加 `where_contiguous` 是同一件事的第二次——
**在这七个 `*Ext.td` 里现写 def 之前先 grep 一遍要用的名字**，撞上了就改名，别指望
「反正那个文件没被 include」。

改动清单（已落地）：

| 改动处 | 内容 |
|---|---|
| 判据 | `ElementwiseApiFamily.h/.cpp`：`ApiFamily` 加 `Shared`；`headersMatchFamily` 与 `checkApiFamily` 各接一处 |
| fork td | `OpVecBinary.td` 现写的 `AscendC_AxpyExtendContiguousOp`，6 操作数 `dst, src0, src1, alpha, sharedTmpBuffer, calCount`，`ApiExtensionHeader<["axpy.h"]>`，非 `MathLibraryOp`，无前导头 |
| 发射 | `printApiExtensionBinary` 加了一个 `Value extraScalar`，打在 src1 之后、tmp 之前；原有 4 个 overload 的调用点补 `/*extraScalar=*/Value()`。`Translation.cpp` 的 `PrintableOpTypes` 加一行 |
| AFIR ODS | `AFIR_BufferizableBinaryOpBase` 与 `AFIR_BufferizableBinaryElementwiseOp` 补 `extraAttrs`（一元那边 LeakyRelu 那轮已经有了）；`AFIR_AxpyOp`：`[F16, F32]`、`sameElementType=1`、`allowScalarRhs=0`、`extraAttrs=(ins F32Attr:$alpha)` |
| conversion | 新增 `BinaryScalarAttrConversionPattern`（二元＋属性实参＋tmp）与 `registerBinaryScalarAttr`；一个 implementation `axpy.tensor_tensor`，`ApiFamily::Shared`、`kF16\|kF32`、tmp 恒收 |
| bridge | 注册表加 `M("Axpy", afir::AxpyOp, BridgeOpForm::Both)`；`negative_slope` 与 `alpha` 收成一张具名属性表 `getFloatIrAttrMappings()`（AscGraph 名 → AFIR 名），`ScheduleToAfir` 与 `AfirBuilder` 两处各读同一行，`NodeInfo::negative_slope` 改名 `float_ir_attr` |
| 覆盖集 | `BridgeOperationRegistryTest` 32→33 并加名，另加 `FloatIrAttrRowsNameABridgeableOperation` 把属性表钉在操作表上；`ScheduleToAfirTest.KeepsBufferFormAllowlistExplicit` 加一行 `Axpy` |
| 用例 | lit `afir-axpy-extension-api.mlir`（两个 target 同一组 CHECK，钉 alpha 的实参位置与「恒 f32」）；E2E `load_axpy_store`（f32、alpha=0.5、断言 `y = x1 + 0.5*x2`）；builder 加 `Axpy(name, in1, in2, alpha)` |

几条查过的口径：

- **alpha 常量恒建 f32**，不跟 tile 元素类型走。包装形参是 `const float alpha`。
  **这与 LeakyRelu 恰好相反**——那边的包装是 `const T&`，给个 f32 常量配
  `LocalTensor<half>` 会让 `T` 同时推成两种。照抄 `UnaryScalarAttrConversionPattern`
  那段就错了，而且错了**没有任何一层会报**：`AxpyExtend` 的 `T` 由两个 tile 定死，
  alpha 是独立的隐式转换，f16 常量照样编得过、只是多一次舍入。lit 里 f16 那一份钉的
  就是这条。
- **`axpy.h` 不要 `utils.h` 前导**：它没用 `MAX_REPEAT_TIME` / `Rational`。里面的裸
  `LocalTensor<float>` / `RoundMode` 由 `AddKernelIncludes` 在有扩展头时发的
  `using namespace AscendC;` 兜住。E2E 里有一条反向断言（kernel 里**不该**出现
  `__ASCENDC_API_UTILS_H__`）——多声明一个前导头不报错、只白白内联一份文本。
- **dtype 取 `{f16, f32}`**：v1 的 `T` 就是这两个；v2 多出 bf16 / u64 / i64，但 AFIR 这一层
  的 `ElementTypes` 只有 f16/f32/i16/i32/ui8/ui16，拼不出来。所以一个 implementation 覆两侧
  不丢东西。
- **tmp 恒收**：`CalcAxpyTmpSize` 两个分支都分一块（f16 按输入算，其余固定 8192）。
  f32 走的 `else` 分支**其实一次没用**它——与 `BitwiseAndExtend` 同一种情况，形参在签名里，
  少一个实参就调不上。所以这个形状没有 `TmpBufferUse` 那根轴，只有一种取值。
- **没有标量形态**：矩阵的 `mem_scalar` / `reg_scalar` 两格都是 `no`。

**留下的一个洞**：`load_axpy_store` 只做 f32，而**真用那块 tmp 的是 f16 分支**（切两段
做 `Cast` / `CAST_RINT`）。所以这份用例证明的是「tmp buffer 被正确传进去了」，不是
「那块 tmp 被正确用了」。补一份 `load_axpy_half_store` 就能覆到——没做，记在「已知覆盖
空洞」。

**顺带查清的一条，对所有钉常量的用例都成立**：生成的 C++ 里浮点字面量打的是**定长科学
计数**形式（`(float)5.000000000e-01`），不是 `(float)0.5`。写 `(float)0.5` 直接对不上——
Axpy 这一轮第一次写 lit 就是栽在这里。

反过来，`afir-leaky-relu-scalar-attr.mlir` 里那条
`CPP: constexpr float {{.*}} = (float)1.25` 之所以一直是绿的，是因为 emitter 打的
`(float)1.250000000e-01`（也就是 **0.125**）**以它为前缀**——FileCheck 是子串匹配。
**那条 CPP check 没有钉住值**：把斜率改成 1.25 或 12.5 它照样绿。同一份文件里的 IR check
（`arith.constant 1.250000e-01 : f32`）是完整的，值本身有人钉；漏的只是「C++ 侧打出来的
是不是同一个数」。Axpy 的 lit 与 E2E 都按完整字面量写了，`leaky_relu` 那条没动
（不在本次范围）。

### TrueDiv：已接通，以及「一个 AscendC def 服务两个 AFIR op」

**这个 op 全部的意义就是 i32 那一行。** `true_div.h` 的 `TrueDivExtend` 里
`else` 分支字面上就是 `AscendC::Div(dst, src1, src2, size)`，reg 侧的 `reg_api_name`
写的干脆就是 `DivExtend`——也就是 `afir.div` 的 reg implementation 今天已经在发的那个
调用。只有 `if constexpr (IsSameType<T, int32_t>)` 那一支不同：先把两个 src `Cast`
成 f32（`CAST_RINT`）再除。所以**不收 i32 的话 `afir.truediv` 两侧都是 `afir.div` 的
逐字别名**，用例全绿而覆盖面一个没涨。

改动点与预想的一样少（mem 侧一个 td def），但有三件事是这一轮才知道的。

**一、reg 侧一个 def 都没新建——直接点了 `AscendC_DivExtendRegBaseOp`。** v2 的
`TrueDivAscIrCodegenImplV2` 与 `DivAscIrCodegenImplV2` 逐行同形（apiName `DivExtend`、
头 `div_reg_base.h`、`BinaryApiCall`、不收 tmp），没有第二份实现可建。**这是第一次
一个 AscendC op def 同时服务两个 AFIR op 的 implementation**，与 tier D1 不是一回事：
D1 是同一个包装两侧通用（`ApiFamily::Shared`），这里是两个不同的包装，只是 reg 那一个
恰好已经有人建过。id 上仍然是 `truediv.` 打头，按 AFIR op 取名而不是按 AscendC op。

**二、换基类要删两处手写代码，不只是改那一行。** `AFIR_BinaryMathOp` →
`AFIR_BufferizableBinaryElementwiseOp` 之后，`Ops/Elementwise.cpp` 里原先手写的
`TrueDivOp::verify()` 与 `AFIR_INFER_BROADCAST(TrueDivOp)` 两处都要拿掉：前者与生成的
`verify()` 重复定义（链接期才报，编译期一声不吭），后者声明的是
`inferReturnTypeComponents`，而新基类走的是 `inferReturnTypes`，编译期报「no
declaration matches」。`gelu` / `sign` 那一轮（`4b2f2588`）删的就是同样两处，只是当时
没写下来。

**三、结果 dtype 那条规则有两个读点，写成了一个函数。** `ResultElementType` 加第三档
`SameOrF32ForI32`（没有改成 (输入, 结果) 对表——那要动全部 47 个注册点），而 ODS 那边的
`logicalResultElementType` 也要说同一件事。两处各写一遍的话错法是静默的：一边推出 f32、
另一边只认「目的地与输入同 dtype」，op 建得出来但永远选不出 implementation。所以规则
落在 `afir::elementwise::trueDivResultElementType` 一个函数里，两处都调它——与 shape
推导那条「verifier 和 inferReturnType 共用一个推导函数」同一个理由。

**元素类型两侧不同，这次是真的不同**：mem 侧 `{f16, f32, i32}`，reg 侧 `{f16, f32}`
（v2 的 `kTrueDivTypePairs` 第三行是 bf16，AFIR 不建模）。`floor_div` / `remainder`
那两对也是两侧各说一半，但它们差的是「谁多收几种」；这一对差的是**这个 op 存在的理由
本身**只在一侧。

**用例上多出来的一条坑**：`load_truediv_store` 的 `CheckGeneratedKernel` 里想断言
「reg 侧那份没进来」，**不能写 `kernel.find("DivExtend(") == npos`**——`TrueDivExtend(`
含着它，那条断言永远红。按 include guard 写（`__ASCENDC_REGBASE_API_DIV_H__`），
与 `load_div_store` / `load_sub_store` 同一个办法。

lit 两份（mem / reg 各一，mem 那份带 i32→f32 的函数）都测不出前导头 `utils.h`——
`scalar_div.h` 用了 `MAX_REPEAT_TIME` 与 `KernelUtils::`，少了它 IR 照建、
`ascir-translate` 照打那行调用，只有 `load_truediv_store` 真去编 kernel 才炸。
这已经是第五次（`where.h` / `sign.h` / `logical.h` / `floor_div.h` 之后）。

### Tanh / Erf：已接通，以及「偏差 6 是假卡点」

**这一轮真正的产出是把偏差 6 那一节推翻。** 它写了半年「dialect 里没有东西产得出会被
emit 成编译期常量的值」，据此判定 `MathLibraryOp` 全家都动不了；实际 `arith.constant
false` 经 `printConstantOp` 就打成 `constexpr bool`，正好是模板实参位上用得了的东西。
验证只要手写一个 `"ascendc.tanh"(...)` 过一遍 `ascir-translate`——**方案与本文里所有
「XX 表达不了 / 产不出来」的断言都该先这么验一遍再信**，这是本文第四条被顶翻的假设。

于是这一轮**没有动 ascendc dialect 的 op 建模，也没有动 CodeEmitter**，只在 lowering
里多物化一个常量。四件现场：

**一、这一对的方向与前面所有两侧 op 都是反的。** `sigmoid` / `abs` / `reciprocal` /
谓词族的标准调用都在 **reg** 侧，包装在 mem 侧；`tanh` / `erf` 反过来——标准的
`AscendC::Tanh` / `AscendC::Erf` 在 **mem** 侧且不引任何扩展头，包装
（`TanhExtend` / `ErfExtend`，`tanh_reg_base.h` / `erf_reg_base.h`）在 reg 侧。
照着 `sigmoid` 猜方向会把两个 `ApiFamily` 写反，而写反了**没有任何一层会报错**——
那正是 `checkFamilyMatchesHeader` 当初要拦的东西，可它拦不了这一对：见下一条。

**二、`ApiFamily::MemBased` 而一个扩展头都不带，是第一次。** `headersMatchFamily` 对
「无头」的处理是**直接放行**（`hasAnyHeader` 两个都空就返回 true，根本不看 family），
所以 mem 侧那个 implementation 过得去。`abs.tensor` / `sigmoid.tensor` 那几个
`RegBased` 且无头的走的是同一条路径，只是换了一边——**这条判据在「无头」上是哑的，
两侧都一样，别以为它还在替你把关。** 也别因为无头就写成 `Any`：`Any` 的语义是「两张表
给的是同一个调用」，而这里 reg 侧是另一个函数。

**三、`isReuseSource` 打出来的文本与 legacy 不同，语义相同。** legacy 的
`UnaryApiTmpCall`（`codegen/api_call/elewise/unary_api_tmp_call.cpp`）**一个模板实参都
不打**，发的是 `Tanh(y[off], x[off], tmpbuf_N, size);`，靠 AscendC 自己的默认值
`isReuseSource = false`；我们发的是 `AscendC::Tanh<float, c0_i1>(...)`。
~~**与「已知覆盖空洞」第 0 条的 `isSetMask` 同一类偏差**，是已被接受的先例。~~
**这个类比不成立，别再引它**：`isSetMask` 后来查出是语义不同（见遗留问题清单 L-2），
不是「文本不同、语义相同」。这一条本身仍然成立，但理由要换成它自己的：
`isReuseSource` 在 ODS 上是**操作数**不是属性，conversion 现建一个 `false` 常量塞进去
——与 AscendC 声明上的默认值 `false` 同侧，不存在「不写就落到另一侧」的问题。

**四、`MaterializesReuseSource` 是一元那条路的第四根轴，但它不独立。** 今天这一族的
实例都要 tmp buffer，「不要 tmp 却要 isReuseSource」一个实例都没有，所以那一支干脆
不写，由 `static_assert` 挡住——**别先摆一条没人走的分支在那里**。

另外两条踩在别人身上的：

- **换基类要连手写 `verify()` 一起删。** `AFIR_UnaryMathOp`（tensor-only）换成
  `AFIR_BufferizableUnaryElementwiseOp` 之后，`Ops/Elementwise.cpp` 里那两个
  `ErfOp::verify()` / `TanhOp::verify()` 就与 ODS 生成的重了——**不删是链接期报重复
  定义，编译期一声不吭**。一元**没有** `AFIR_INFER_BROADCAST` 那一行要删，那是二元
  `AFIR_BinaryMathOp` 才有的（TrueDiv 那轮踩过）。
- **新建的 ascendc op 要在两处登记打印器**，不是一处：`Target/Asc/Basic/VecUnary.h` 的
  重载集合 **和** `Target/AscendC/Translation.cpp` 的 dispatch 列表。只加前一处的表现是
  `ascir-translate` 报 `op unable to find printer for op`，lit 当场红——这条不会漏到
  合入后，但会浪费一轮构建。

**mem 侧最后一个实参不带 `static_cast<uint32_t>`。** 那个强转出自
`printApiExtensionUnary`（包装那条路），Math 一族的打印器不加，直接把
`local_tensor.get_size` 的 `int32_t` 传给形参 `const uint32_t calCount`。隐式转换，
编得过（`load_tanh_store` 证的就是这条加上 `constexpr bool` 当模板实参）。
**写 lit 的 CPP check 时别照抄 sign / remainder 那几份的形状。**

### Pow：已接通，以及「两侧收的类型差得最远」

**这一轮没有新机制。** 形状与 `Remainder` 逐字同形：两张供给表同名、都要一块 tmp
buffer、只有扩展头不同（`pow.h` vs `pow_reg_base.h`），所以 `arguments` 进 base、
两个 def 各自只剩一行头。改动面就是「四处起步」那张样板表，一处没多。
**它是剩下的 op 里最后一个能这么走的**——`Cast` 要重写一条覆盖断言，另外 8 个各挡在
一条新机制后面。

四件值得记下来的：

**一、apiName 是裸名 `Pow`，而 `AscendC::Pow` 根本不存在。** 标准调用叫 `Power`
（`Adv/Math.td` 里的 `def PowerOp : BinaryMathOp<"power", "Power">`），两份包装转发给的
都是它；包装存在的理由是 dtype 分流与标量形态的展开。所以 mnemonic 上也没用
`_extend` 那套后缀，直接 `pow` / `pow_reg_base`——**别照 `remainder_extend` 的样子取名**，
那个 op 的 apiName 真的带 `Extend`。与 mem 侧的 `LogicalAnd`、选择族的 `ClipByValue`
同一种情况：走扩展头但名字是裸的，id 上仍按 `_reg_base` 区分两侧。

**二、两侧的元素类型是这一批里差得最远的一对。** mem 侧 legacy 的 `REG_ASC_IR(Pow)`
是 {i32, f16, f32}，reg 侧那一版是 {i8, ui8, ui16, ui32, i16, i32, f16, f32, bf16}。
AFIR 建模得出来的是其中六个，于是 ODS 的并集有 `[F16, F32, I16, I32, UI8, UI16]`
——**比 `bitwise_and` 那一族还宽一档**，而 mem 侧只认其中一半。reg 侧那份直接写成
`kArithmeticElementTypes | kUI8 | kUI16`。

**三、两份 `pow.h` 的 include guard 撞名，e2e 的「进来的是哪一侧」不能比 guard。**
两个文件都写 `__ASCENDC_API_POW_H__`（`remainder` 那一对不是这样，reg 那份是
`__ASCENDC_API_REGBASE_REMAINDER_H`）。`load_pow_store` 改钉 reg 侧独有的
`PowerAlgo::DOUBLE_FLOAT_TECH` 不出现——与 `load_sign_store` 同一个办法。

**四、前导头 `utils.h`，两侧都要，这是第六次。** 两份 `pow.h` 都用了
`KernelUtils::BlkSize`，reg 那份还用了 `KernelUtils::BlkAlign` 与
`IsIntegerWithinUint32Magnitude`，三个符号都只有 **mem 的** `utils.h` 声明
（`utils_reg_base.h` 是 micro-API 的 `CastTrait` 常量表，一个都没有，与 `sign` 那一对
同一条）。少了它两份 lit 全绿而 kernel 编不过。`where.h` / `sign.h` / `logical.h` /
`floor_div.h` / `scalar_div.h` 之后，**这条已经连续六个 op 都成立了——新接一个走包装的
op，先翻头文件里有没有 `KernelUtils::` / `MAX_REPEAT_TIME` / `Rational`，别等 e2e 报。**

顺带删掉的：`AFIR_INFER_BROADCAST` 这个宏没有实例了。`pow` 是最后一个用它的 op，
换基类之后结果类型由生成的 `inferReturnTypes` 给，宏和 `#undef` 一起删。逻辑族那个
（`AFIR_INFER_BROADCAST_LOGICAL`）还在，它推的结果恒为 `ui8`。

**标量形态没接**：legacy **两侧**都收标量（矩阵上 mem 那一格是
`!std::all_of(is_scalar_list...)`，也就是「只拒全标量」，reg 那一格是 `any`），两侧都要
L-4 的 swap 方向。按矩阵的 `mem_scalar` / `reg_scalar` 两列数，**至少一侧写着 `any`
的有四个：Sub / Div / TrueDiv / Pow**（加 ClipByValue 两侧都是 `any`）。**这几个不是
L-4 那半张格子**：`any` 说的是 legacy 两边都收标量，而它们不可交换，要的是
`is_scalar_latter` 那第二种调用形状。L-4 做完它们一个都没接通，值得单开一条。

### Cast：已接通，以及挂起表那一行为什么低估了它

**这一轮不是「两个 def + 换基类」。** 挂起表上那一行只说对了「两侧供给都缺」；
真正的代价在下面三件事上——一件是那一行写错了，两件是它根本没提。
三件都出自同一个毛病：**照着矩阵那几列推，没翻代码**。

**一、mem 侧发的不是裸名 `Cast`，是 `CastExtend`。** 矩阵的 `mem_api_name` 写的是
`Cast`，挂起表照抄了它，于是那一行写着「mem 要 fork 自己那个裸名 `Cast`」。
**那是注册名，不是发出去的名字**：`CastApiCall::Generate` 第一行就是
`(void)this->api_name_;`，函数名是硬编码的 `CastExtend(`。
**`GetApiName()` 与 emitter 实际打出来的名字可以不一致**，这一族里 `CastApiCall` /
`CastV2ApiCall` 两个都是——查供给时别只看矩阵那一列，要翻对应的 `*_api_call.cpp`。

**二、reg 侧要一套新的 emitter 形状：brace-init 数组。** 签名是

```
template <typename InT, typename OutT, uint8_t dim, AscendC::RoundMode round_mode>
void CastExtend(const LocalTensor<OutT>& dst, const LocalTensor<InT>& src,
                const uint32_t (&output_dims)[dim], const uint32_t (&output_stride)[dim],
                const uint32_t (&input_stride)[dim]);
```

`dim` 由 brace-init 的长度推出来，连续形态下恒为 1、两个 stride 恒为 1，所以那三个
实参不进 `arguments`，由 emitter 打字面量——「连续」写进助记符里
（`cast_extend_contiguous_reg_base`），与 `where_contiguous` 同一个办法。
**这正是文档里一直挂在比较族名下的那个形状**（「模板实参带 CMPMODE + 实参是 brace-init
数组」），Cast 把其中一半先兑现了。

也因此**不打显式模板实参**：`InT` / `OutT` 从两个 LocalTensor 推得出来，但 `dim` 只能
从 brace-init 推——写出 `CastExtend<InT, OutT>` 就得把 `dim` 也写出来，否则模板实参的
位置对不上。mem 侧那半照旧走 `printApiExtensionUnary`（打一个 `<InT>`）。

**三、`InferTypeOpInterface` 装不下 cast，ODS 基类得拆。** 这是本轮最费事的一处。
`AFIR_BufferizableUnaryElementwiseOp` 捆着 `DeclareOpInterfaceMethods<InferTypeOpInterface>`,
而 `inferReturnTypes` 只拿得到操作数：谓词族推 `ui8`（常量）、`truediv` 推「i32 就
f32」（输入的函数），**cast 的结果 dtype 两者都不是**，它是调用方写在 op 上的那一个。

**声明了却推不准比不声明更坏，而且是静默的**：`verifyInferredResultTypes` 拿
`isCompatibleReturnTypes` 对结果，而这一族的那个只比形状
（`AFIR_ElementwiseShapeMethods`，`verifyCompatibleShape`），于是一个错的元素类型
一路不报。所以 `AFIR_BufferizableElementwiseMethods` 拆成三段
（verify / common / infer），cast 只取 common，自己拼一个 `verify()`，
`AFIR_BufferizableElementwiseTraits` 也拆出不带 `InferTypeOpInterface` 的
`AFIR_BufferizableDestinationStyleTraits`。**buffer 形态其实推得出来**（结果恒等于
destination），但两种形态共用同一个方法，为能推的那一半声明整条接口，换来的是另一半
静默出错。

另外四条现场：

- **(输入, 输出) 对表有两张，不是一张。** mem 侧 36 对、reg 侧 43 对（legacy 那张 44
  对里的 `{f32, bool}` 不建模），两边互有对方没有的行——mem 独有 i4 那一族 3 对，
  reg 独有 i8 / i64 那两族 10 对。**三个读点**（ODS 的 `verify()` 查并集 46 对、两个
  implementation 各查自己那一半）共用 `Ops/Cast.cpp` 里的同一个查表，与
  `trueDivResultElementType` 同一条。
- **ODS 那两个元素类型列表原先各漏一个**：输入列表少 `UI16`（而 mem 表里有
  `ui16 -> i16`），结果列表少 `UI64`（而 reg 表里有 `i64 -> ui64`）。没被发现是因为
  这个 op 当时没有任何 implementation——与 `floor_div` 那一轮同一种。现在两边都写成
  同一个全集，**列表不再是判据**，判据是那两张对表。
- **`{f32, f32}` 那一行要单独挡掉。** 它在 mem 的对表里（留给 VF 融合的），而 legacy 的
  `CastApiCall` 开头就 `GE_ASSERT_TRUE(x.dtype != y.dtype)`。capability 判据里因此加了
  一条「源与目的地必须不同 dtype」，光查对表会发出一条 legacy 永远不会发的调用。
- **`KeepsBufferFormAllowlistExplicit` 换了说法，不是换了一行。** 那条断言原先靠一个
  反例立住——表里有一行是 `Tensor` 形态（最后是 `Cast`）。Cast 接通之后 38 行全是
  `Both`，再挑哪一行写 `EXPECT_FALSE` 都是假的。改成钉住「白名单只认表里写下的名字」：
  拼错的（`"cast"`）、legacy 里真有而这一层没登记的（`"Rsqrt"`）、彻底没有的，
  三种都必须答 false。挡的是同一件事，而且不再依赖表里恰好有一行反例。

**e2e 里「没走标准调用」这件事根本钉不了，两种写法都试过、都当场红。**
`AscendC::Cast(` 不行——包装最后转发的就是它；`AscendC::Cast<` 也不行——`cast.h` 里
`CastExtendWithMaskMode` 打的是 `AscendC::Cast<OutT, InT, false>(...)`。**整段头文本是
内联进 kernel 的，凡是头里出现过的字面量都不能拿来写负例。**
与 `load_truediv_store` 那条 `DivExtend(` 同一类坑（那次是 `TrueDivExtend(` 含着它），
只是这次含它的是内联进来的头文本。**写「某某不出现」之前，先 grep 一遍要内联的那个
头文件**；钉不了就交给 lit（IR 里建的是哪个 op、选中的是哪个 id），e2e 只钉正面证据。

**没有标量形态**：对照表两侧的 `*_scalar` 格都是 `no`，这一个不是 L-4 欠的账。

### Rsqrt / LogicalNot：已接通，以及「前置 B 原先说反了方向」

这一波的产出有两件：一条机制（前置 B）和两个 op。**机制那一件真正花时间的不是写代码，
是先证明文档写的不是它。**

#### 1. 「一个 op 两块 tmp buffer」不成立，去问调度就知道

挂起表与前置 B 两处都写着「`RsqrtExtend` / `LogicalNot` 比其余包装多一块 tmp buffer，
当前模型表达不了」。查证只要两个文件：`ascir/reg_func/rsqrt.cpp` 与
`ascir/reg_func/logical_not.cpp`——两个 `CalcXxxTmpSize` 各返回**一块**，AscGraph 上没有
第二块。那块预先填好 1 的张量是 legacy 在 codegen 里凭一张静态表现造的，与调度无关，
口径、去重方式与 UB 预算写在上面「前置 B」。

**照原说法动手的代价**：`AFIR_Bufferizable{Unary,Binary,Ternary}OpBase` 三个基类的
`Optional<AFIR_TmpBufferType>` 要变 `Variadic`、`getNumTmpBuffers()` 要变成真计数、
`makeOperandSegmentSizes` 要多一段、`AfirBuilder` 要凭空造出第二块的 `b{id}_size`（而
ATT 不会给这个字段）、`CanonicalGroupKey` 的 `exclusiveMemberOrdinal` 要再分一档。
**实际改动是 0 处**——AFIR 那一层一个字没动。

这是本文开头那条「本文自己的『XX 表达不了』也不作数」第二次兑现（第一次是偏差 6）。
两次的形状一样：**结论写在文档里、来源在另一个目录，没人回去对过**。

#### 2. 机制：`ConstBlockTensor.{h,cpp}`，约 100 行

`getOrCreateConstBlockTensor(rewriter, op, elementType, value)`：在 `func.func` 里按
`ascendc.const_block` 这条认领标记找现成的那一块，没有就在序言里造四条——
`ascendc.tbuf` + `pipe.init_buffer(32)` + `tbuf.get_tensor` + `duplicate_l2`，末尾补一条
`PipeBarrier<PIPE_V>`。

三个决定值得单记：

- **认领标记的取值是那个常量本身**（`FloatAttr`，类型就是这块张量的元素类型），所以一条
  属性同时说清「值是几」与「什么 dtype」，去重查的就是它。另立一张 (值, dtype) 的表是
  第二个真源。
- **位置是入口块的 `ascendc.pipe` 之后；函数里没有 pipe 就先建一个。** 后一支不是防御
  性代码，是 lit 用例的常态：只喂 `!ascendc.local_tensor` 参数的那些函数里
  `MemoryAllocate` 一个资源都没建，自然也没有 pipe。这块常量的字节数是编译期的 32、
  不引用任何 TilingData 符号，所以不必排在 `afir.tiling_param` 之后。
- **那条 `PipeBarrier<PIPE_V>` 不能省**：`Duplicate` 与随后读它的包装都在 V 上，是一次
  RAW 竞争。legacy 的 `BlkTensorAllocAndInit` 也是紧跟一条；`GenDuplicateBufAlloc`
  那条路没有，但那是它自己的账。

**唯一的真源是 pattern 的模板实参** `ConstBlockOnes`（`None` / `F32` / `F16`），
与 `TakesTmpBuffer` 同一个办法：同一个实参既决定建 op 时的实参个数，也决定要不要去造
那一块。没有按 op 名查的第二张表。

#### 3. 那一块的 dtype 与正在算的 dtype 无关

`RsqrtExtend` 收 `LocalTensor<float>`，`LogicalNot` 收 `LocalTensor<half>`——**写死在包装
的形参上**。f16 的 `rsqrt` 用的仍然是 float 那一块（包装一律先把中间结果转成 float 再
`Div`），f32 的 `logical_not` 用的仍然是 half 那一块。按 dst / src 的 dtype 去造，device
编译器当场报重载对不上，而 lit 测不出来（它不编译内联进来的文本）——`load_logical_not_store`
是拦这一条的那个用例。

#### 4. LogicalNot 是一元里第一个两侧结果 dtype 关系不同的 op

| | mem 侧（v1） | reg 侧（v2） |
|---|---|---|
| 包装 | `LogicalNot(dst, src, blk, tmp, size)`，**裸名** | `LogicalNotExtend(dst, src, count)` |
| 头 | `logical_not.h` | `logical_not_reg_base.h` |
| 结果 dtype | 与输入相同（`kLogicalNotTypePairs` 逐行 `T -> T`） | 恒为 `ui8`（逐行 `T -> ui8`） |
| tmp buffer | 要一块 | 不要 |
| 常量 block | 要（`half`） | 不要 |

**方案的偏差 7 说「它的结果 dtype 是 uint8 而输入不是」——那只对 reg 侧成立。**
两侧关系不同，`InferTypeOpInterface` 就推不出来（它只拿得到操作数，说不出「看选中的是
哪一侧」），于是 ODS 走 `afir.cast` 那条路：新建 `AFIR_BufferizableLogicalNotOp`，只取
`bufferizableElementwiseCommonMethodDef`，`verify()` 自己拼形态检查 +
`verifyLogicalNotPair`（并集两行：结果等于输入，或者结果是 `ui8`）。逐侧那一半由
lowering 的 `ResultElementType` 说——mem 侧 `Same`，reg 侧新加的 `Uint8Logical`。

**`Uint8Logical` 与谓词族的 `Uint8Predicate` 不能合并**：谓词族把输入钉死成浮点、不读
`elementTypes`（`isnan` 对 int32 没有意义），而这一档收的正是整数与浮点混着的那一串。
合并就要么把谓词族放宽到整数，要么把这一档收窄到浮点。

裸名那一件也值得记：mem 侧发出去的是 `LogicalNot<T>(...)`，与 `AscendC::LogicalNot`
同名而不是同一个函数——包装是全局的、不带限定，两者靠实参个数分开（5 个 vs 3 个）。
legacy 的 `LogicalNotAscIrCodegenImpl::GetApiName` 写的也是这个裸名。**与 `Pow` 是同一类
（裸名、`AscendC::` 里另有同名或近名的东西），不是 `CastExtend` 那一类（矩阵那一列是
注册名不是发出去的名字）。**

#### 5. Rsqrt 两侧差得比一元里任何一对都远

reg 侧是标准调用 `AscendC::Rsqrt(dst, src, count)`——不引扩展头、不要 tmp buffer、也不要
常量 block；mem 侧走 `RsqrtExtend`，三样全要。**`RsqrtL2Op` 早就在 `OpVecUnary.td` 里**
（`defm Rsqrt : UnaryL012Op<"rsqrt", "Rsqrt">`），所以这一侧一行 td 都不用加，前置 C 表上
「只缺 mem 侧」那一行是对的。

ODS 那一行照 `4b2f2588` 的先例从 `AFIR_UnaryMathOp` 换成
`AFIR_BufferizableUnaryElementwiseOp`，手写的 `verify()` 一并删掉。

#### 6. 改动面：六处，比「四处起步」多两处

1. `api/.../Basic/OpVecUnary.td`：三个 def（`RsqrtExtendOp` / `LogicalNotExtendOp` /
   `LogicalNotExtendRegBaseOp`）加一个共用基类；
2. `Target/Asc/Basic/VecUnary.h` 的重载集合 + `Target/AscendC/Translation.cpp` 的
   dispatch 列表（`printApiExtensionUnary` 多收一个 `constBlock`，排在 tmp buffer **之前**，
   与包装的形参顺序一致）；
3. `dialect/AFIR/IR/Ops/Elementwise.td`：`rsqrt` 换基类 + `logical_not` 新建一个基类；
4. `dialect/AFIR/IR/Ops/Elementwise.cpp`：删两个手写 `verify()`，加 `verifyLogicalNotPair`；
5. `Implementations/ConstBlockTensor.{h,cpp}`（新文件）+ `UnaryElementwise.cpp` 的
   `ConstBlockOnes` 与四个注册；
6. `bridge/AfirBuilder/AfirBuilder.cpp` 两行。

出口：lit 四份（`afir-rsqrt-{extension,stock}-api.mlir`、
`afir-logical-not-{extension,reg-base-extension}-api.mlir`，其中 rsqrt 那份第二个函数专钉
**一个 kernel 里两个 op 共用一块常量**），CPU E2E 两份（`load_rsqrt_store` /
`load_logical_not_store`）。

**e2e 的输入不能取常数**：`load_rsqrt_store` 全喂同一个值的话，「算 `1/sqrt(x)`」与
「照搬那块常量」这两种结果分不开，而「常量没被 `Duplicate`」正是要拦的错法之一。
`load_logical_not_store` 同理要零与非零都有——全非零的话「结果恒为 0」也能过。

### 比较族：已接通，以及「新调用形态」核完只剩一件

六个 op（eq / ne / lt / le / gt / ge）一拨接通两侧。**这一波真正的产出是把挂了整篇文档的
那条卡点核干净**：它写着「一套新调用形态：`<T, dim, CMPMODE>` + brace-init 数组 +
second-to-last stride」，对着两个 `*_api_call.cpp` 与两份头核完，剩下的只有一件。

#### 1. 三件里两件不存在，第三件是真的

- **second-to-last stride 到不了这条路**。`compare_reg_base.h` 的
  `CompareExtend<InT, dim, mode, OutT>` 里 `static_assert(dim == 1 || dim == 2)`,
  两个 stride 只有 `dim == 2` 才读得到；mem 侧对应的是 `compare_v2.h` 那一支
  （`outer_repeats_size != 0`）。这条路只发连续形态，两侧都够不着——与 `cast` /
  `where` / `clipbyvalue` 同一个理由。
- **brace-init 数组 `Cast` 那一轮已经做掉了**（`CastExtendContiguousRegBaseOp` 的
  emitter）。只有一处不能照抄：**这一族的数组是 `const uint16_t (&)[dim]`，
  `cast_reg_base.h` 是 `uint32_t`**。写成 `{1U}` 是窄化初始化，device 编译器当场报错而
  lit 全绿。
- **真正新的那一件是：同一个 `CMPMODE` 在两侧落在不同的位置。** mem 侧是第四个运行期
  实参，reg 侧是第三个模板实参。写在同一个 op 上分不开，所以是两个 def 两个 emitter。

#### 2. 第一次「一个 AscendC def 服务六个 AFIR op」

六个 op 的 ODS **一字不差**——差别只有 `CMPMODE`，而那不是 AFIR 这一层的东西：它由
「是哪个 op」定死。于是：

| | 个数 |
|---|---|
| AFIR ODS | 一个 `AFIR_CompareOp` 类加六行 |
| AscendC def | 4（两侧 × {tensor-tensor, tensor-scalar}） |
| implementation | 24（6 op × 2 侧 × 2 形态） |

`CMPMODE` 走**模板实参**而不是 mapping 字段：注册处写岔了当场编不过，写成字段的话只会
在运行期发出一条比错的调用。legacy 那边也是一份 emitter 服务六个节点类型
（`CompareApiCall` / `CompareV2ApiCall` 把 `GetApiName()` 直接拼进 `CMPMODE::`）。

#### 3. mem 侧的 tmp buffer 排在 count **之后**

`CompareExtend(dst, src0, src1, mode, cal_cnt, tmp_buf)`——这条路上其余每一个包装都是
tmp 在 count 之前，`printApiExtensionBinary` 就是按那个次序写死的，所以这一族不能复用
它。**次序写反了没有哪一层会报**：两个位置上都是 `LocalTensor`，IR 与 C++ 都成立，
只有算出来的数不对。拦它的是 `load_eq_store` 那条逐元素断言与 lit 的 CPP 那条 check。

reg 侧反过来，一块 tmp 都不收，而 `GetCompareSizeV2` 照样给这个节点分 32 字节——
`TmpBufferUse::Unused` 那一档，这是 `sub` / `div` 之外第二处用上第三档的地方。

#### 4. 原先那六行 ODS 里有四行是错的

换基类时才发现：三行写了 `F64`、四行写了 `I64`、`lt` 少了 `I32`。两侧的 legacy 注册里
从来没有 `float64`；`int64` 两侧都有，但 AFIR 这一层的 `ElementTypes` 位掩码拼不出它。
六行各写各的、四行是错的却一直没人发现，**是因为这个 op 当时没有任何 implementation**
——与 `floor_div` / `cast` 那两轮同一种。ODS 那一行不接实现的话，谁都不会去对它。

今天六个 op 收的是同一组：mem 侧 {f16, f32, i32}，reg 侧 {f16, f32, i16, i32, ui8, ui16}。
`lt` 在 legacy 的 mem 侧比其余五个少一个 i64，而那个本来就在建模范围外，所以不必逐行分。

#### 5. 换基类连带删掉的

`AFIR_BinaryHomoMathOp` 的最后一批实例就是这六行，换到
`AFIR_BufferizableBinaryElementwiseOp` 之后，**三个 tensor-only 的 base
（`AFIR_BinaryOpBase` / `AFIR_BinaryOp` / `AFIR_BinaryHomoOp`）、三个 math 族
（`AFIR_UnaryMathOp` / `AFIR_BinaryMathOp` / `AFIR_BinaryHomoMathOp`）、薄封装
`AFIR_UnaryOp`** 一个实例都不剩，一并删掉（`AFIR_UnaryOpBase` 留着，`Reduce.td` 还在
用它）。`Ops/Elementwise.cpp` 里跟着删掉六个手写 `verify()` 与
`AFIR_INFER_BROADCAST_LOGICAL` 那个宏——**这是「换基类要连带删两处手写代码」唯一会
报出来的地方**：只改 td 的话生成的声明与手写定义对不上，当场编不过。

比较族的结果恒为 `ui8`、是个常量，所以**推得出来**，照常留在
`AFIR_BufferizableBinaryElementwiseOp` 上，不必像 `cast` / `logical_not` 那样另起基类。

#### 6. 改动面：六处

1. `api/.../Basic/OpVecBinary.td`：四个 def；
2. `Target/Asc/Basic/VecBinary.h` 的重载集合（四个）+ `VecBinary.cpp` 两个 body
   （`printApiExtensionCompare` / `...CompareRegBase`）+ `Target/AscendC/Translation.cpp`
   的 dispatch 列表；
3. `dialect/AFIR/IR/Ops/Elementwise.td`：一个类加六行，另删三个 math 族基类；
4. `dialect/AFIR/IR/AFIROpBase.td`：删四个没有实例的 base；
5. `dialect/AFIR/IR/Ops/Elementwise.cpp`：删六个 `verify()` 与一个宏；
6. `bridge/AfirBuilder/AfirBuilder.cpp` 六行 + `Implementations/BinaryElementwise.cpp`
   一个 pattern、一个注册模板、24 行注册。

出口：lit 两份（`afir-compare-{extension,reg-base-extension}-api.mlir`，各按
`cmpMode = N` 钉三个模式，两侧各一个标量形态），CPU E2E 一份（`load_eq_store`）。
测试侧另有三处跟着改：`BridgeOperationRegistryTest` 的覆盖集 40 → 46 并加六个名字、
`ScheduleToAfirTest.KeepsBufferFormAllowlistExplicit` 的反例换成 `LogicalXor`
（`Eq` 接通了，站这个位置的名字要挑一个还没接、而且**越像已接的那些越好**的），
`easy_asc_graph` 加六个 builder 方法。

**e2e 只做 `eq` 一个不是抽样图省事**：六个 op 在 MLIR 这一侧逐字共用同一条路，
唯一的差别 `CMPMODE` 由 lit 按属性值钉。`load_eq_store` 要的是另外两件只有真编一遍
kernel 才拦得住的事——`utils.h` 前导（`compare.h` 用了 `MAX_REPEAT_TIME(S)` 与
`KernelUtils::` 的七个成员）和上面第 3 条的实参次序。输入取 f32 不取 i32：
i32 的 `EQ` 在包装里走的就是通用那一支（`CompareExtendInt32` 只接 GT/GE/LE/LT/NE，
`EQ` 被那个 `if` 排除在外），f32 与它同一条路而且不必绕开那条 `ASSERT`。

#### 7. 两件没跟着做的

- ~~**swap 方向的标量**（`eq` / `ne` 两侧都有 `swap(2nd)`）仍然不接，那是 L-4。~~
  **已接**：`eq` / `ne` 的 `scalarOperand` 是 `"any"`，`lt` / `le` / `gt` / `ge` 留
  `"rhs"`（不可交换）。
- ~~**前置 D（一条队列上不能混 dtype）仍然没解。**~~ **已解，而且这里写的范围是错的。**
  「比较族产出 ui8、下游接 select 的图里，中间张量与两边的 f32 落在同一条队列上」
  ——实测不是这样：`Eq` 的结果是计算中间量，落在 `TBuf<VECCALC>` 上，`compare → select`
  一直是绿的（现在有 E2E `load_eq_select_store`）。真正撞上那条限制的是掩码自己从 GM
  搬进来（`load_mask_select_store`）。详情见清单 L-6。

### 三条跨波次的前置，比单个 op 都重要

（前置 A 已经做掉，见 `00cda3a0`：扩展头改成列表、家族判据逐项对。**前置 B 这一轮做掉
了，但它原先的说法是错的**，改写见下。前置 C 仍在，另外新添一条 D。）

#### 前置 A：一个 op 要说得出**多个**扩展头

`ApiExtensionHeader<header, prelude>`（`Base.td:92`）今天只生成
`getExtensionHeader()` 单数形式。**剩下 31 个 op 里有 12 个引用两个头**：

| op | mem 侧 | reg 侧 |
|---|---|---|
| Broadcast（三轴 / 四轴那两档；已接通的 `Duplicate` 一档发的是标准调用，不引扩展头） | `broadcast.h` + `duplicate.h` | `broadcast_reg_base.h` + `duplicate.h` |
| Select / Where / MaskedFill | `duplicate.h` + `where.h` | `where_reg_base.h` |
| ~~Eq / Ne / Lt / Le / Gt / Ge~~（已接通） | `compare.h` + `compare_v2.h`（另要 `utils.h` 前导） | `compare_reg_base.h` |
| ~~Sign~~（已接通） | `cast.h` + `sign.h` | `cast.h` + `sign_reg_base.h` |
| ~~TrueDiv~~（已接通） | `scalar_div.h` + `true_div.h` | `div_reg_base.h` |

**方案里两处说法都不准**：E6 说 Broadcast「是唯一引用两个扩展头的 op」，偏差 8 说比较族
「是第一个」——实际是 12 个，而且**最先撞上的是 E5**（Select / Where / MaskedFill 三个
都要 `duplicate.h` + `where.h`）。这条不是比较族的专属卡点，是 **E5 的前置**。

**别拿 `prelude` 顶替**。`prelude` 是为「包装头不是自足翻译单元」造的（`utils.h`），
语义是「要先于 header 内联」，且单数——用它装第二个业务头，这个 op 就再也说不出自己
还需要 `utils.h`，而失败只会炸在 device 编译器上、报的是内联进来那段文本的行号
（E2 已经吃过一次这个亏）。改法是把两处都改成列表，注册期那条
「家族 ↔ 扩展头名」的对照（`checkFamilyMatchesHeader`）跟着改成逐项对。

#### 前置 B：lowering 能造一块**预先填好 1** 的常量 block　**已做**

**这一条原先的标题是「一个 op 可以有不止一块 tmp buffer」，它是错的。** 记在这里不是
存档——错的是判断的方向，而那个方向会把改动引到 AFIR 的操作数三段划分、ODS 的三个
bufferizable 基类、`AttrSizedOperandSegments`、bridge 的 `makeOperandSegmentSizes` 上去，
一处也不该动。

**推翻它的是调度那一侧**：`CalcRsqrtTmpSize` 与 `CalcLogicalNotTmpSize`
（`ascir/reg_func/`）各只返回**一块** tmp buffer，AscGraph 上没有第二块，`node->attr.
tmp_buffers` 里也只有一项。那块预先填好 1 的张量在图上根本不存在——它是 legacy 在
**codegen** 里凭一张按节点类型查的静态表现造的：

- `common_utils.cpp` 的 `GetApiExtractDupSet` / `api_extract_dup_map`：
  `{Rsqrt -> {1, float}}`、`{LogicalNot -> {1, half}}`，**按 (值, dtype) 去重**；
- `codegen_kernel.cpp` 的 `GenDuplicateBufAlloc`：每个去重后的 pair 造一个 32 字节的
  `TBuf` + 一条 `Duplicate`，名字是 `local_blk_tensor_of_<dtype>_<value>`，
  **整个 kernel 共用一份**；
- UB 预算也是按这个口径记的：`CalcExtraTmpBufForAscGraph` 把它们算进 ATT 的
  `tuning_space_->builtin_tmp_buffer`。MLIR 走的是同一份 ATT，所以预算**已经算过了**，
  这一层只要照同一个口径造出来就对得上。一个 op 一块会超预算，而超出去的症状是运行期
  UB 越界，不是编译报错。

于是真正要补的机制是「lowering 能在 kernel 序言里造一块按 (值, dtype) 共用的常量」，
落点是 `conversion/AFIRToAscendC/Implementations/ConstBlockTensor.{h,cpp}`
（**约 100 行，没有新 pass、没有新 AFIR op、没有 ODS 改动**）。它与 tmp buffer 的区别
一句话说得清：tmp buffer 是调度分配的、未初始化的草稿纸，尺寸来自 ATT 的聚合；这块是
一个 32 字节的常量，元素类型由**包装的形参**钉死（`RsqrtExtend` 收
`LocalTensor<float>`，`LogicalNot` 收 `LocalTensor<half>`），与 dst / src 的 dtype 无关。

现场与三种错法见上「Rsqrt / LogicalNot：已接通」。

#### ~~前置 C~~：fork 侧的 `ascendc.*` 供给（**已清空**）

**这条方案里完全没有，但它直接决定每个 op 的实际工作量。** 下面这张表覆盖的是还没接通
的 op（口径与命令见本节末尾）：

| 状态 | op | 个数 |
|---|---|---|
| 两侧供给都齐 | （无） | 0 |
| 只缺 mem 侧 | （无） | 0 |
| 两侧都缺 | （无） | 0 |

**最后那一行 `Broadcast` 是 2026-09-23 划掉的，而且不是靠新建 def 划掉的——它本来就不该
在表上**：`Duplicate` 一档要的 `DuplicateL2Op` / `LocalTensorGetValueOp` / `SetFlagOp` /
`WaitFlagOp` 全是现成的，连三轴 / 四轴那一档要的 `BroadcastL2Op`
（`Autofuse/OpVecBroadcast.td`，当时在 `Basic/OpVecBroadcastExt.td:29`）也已经有。原先记的「两侧各新建两个 def」是照着扩展头
名单估的，没去点过供给表。

**这张表在 Tanh / Erf 那一轮漏改了一整行**（`只缺 reg 侧 | Erf / Tanh | 2`），两个 op
早就接通、reg 侧的 def 也建了，表上还挂着；`Pow` 与 `Cast` 同理，是后两轮补掉的；
`Rsqrt`（原「只缺 mem 侧」那一行）与 `LogicalNot` 是这一轮划掉的——**`Rsqrt` 那一行
是对的**：reg 侧点的 `RsqrtL2Op` 早就在 `OpVecUnary.td` 里，这一轮只新建了 mem 侧的
`RsqrtExtendOp`；`LogicalNot` 两侧各新建一个。
**接通一个 op 就要回来划掉它这一行**——这张表说的是「还剩的工作量」，留着已完成的行
会让下一批的挑选整个错位（`Axpy` 就是这么被当成最便宜的那个的，见下）。

**另外这张表只答「有没有 td def」，不答「要不要新 emitter」。** `Cast` 与 `LogicalNot`
两侧供给都缺、当初在这张表上同一行，实际代价差着一条 brace-init 数组形态。挑下一批之前
还要翻一遍对应的 `*_api_call.cpp`——**`GetApiName()` 与 emitter 实际打出来的东西可以
不一致**，`CastApiCall` 就是（见上「Cast：已接通」第一条）。

**这张表上一版是错的，错法是系统性的：把「fork 里有这个 C++ 头」当成了「有 `ascendc.*`
供给」。** 两者不是一回事——`autofuse/ascendc/api/` 下 `axpy.h` / `compare.h` /
`broadcast.h` / `pow.h` / `true_div.h` / `logical_not.h` / `rsqrt.h` 全都在，但它们**一个
对应的 td def 都没有**。今天 td 里引得到的业务头只有 `cast.h` / `duplicate.h` /
`logical.h` / `sign.h` 等寥寥几个，而且**全是本仓自己加的**。

被它坑到的四个（Axpy / LogicalAnd / Pow / Gelu）在上面「重算」那一节已经点名。
**Axpy 那一行曾经漏改、一直挂在「两侧供给都齐」**，谁照它挑下一批都会把 Axpy 当成最
便宜的那个；接通 Axpy 时确认了它两侧都缺（`axpy.h` 在 fork 里，td def 是这一轮现写的），
现在它已经不在这张表上了。

两条与直觉相反、值得单记的：

- **Axpy 在矩阵上是 D1**（两侧同 apiName `AxpyExtend`、同头 `axpy.h`、同 tmp 函数
  `CalcAxpyTmpSize`），也就是**只要一个 implementation**，比 D2 那批省一半。但「便宜」
  到此为止：**它真正的卡点是家族判据，不是供给**——原先 `ApiFamily` 那三档说不出「两侧
  共用同一个扩展包装」，三种注册法全堵死。**已解：加了第四档 `Shared`**，现场见下
  「Axpy：已接通（tier D1 第一个）」一节。
- **TrueDiv 的 reg 侧已经被 Sub / Div 那一轮顺带补上了。** 它要的是 `DivExtend` +
  `div_reg_base.h`，正是那轮加的 `DivExtendRegBaseOp`。所以它从「两侧都缺」降成了
  「只缺 mem 侧」，是这批里唯一因为别人做事而变便宜的一个。**已接通**（接通时确认
  reg 侧连一行都不用改，直接点那个 def），现在它已经不在这张表上了。

整个 fork 里 reg-base 扩展包装起初只有一个 `neg_extend_reg_base`（本仓 E2 自己加的）；
此后 E5 加了 `where_contiguous{,_reg_base}` 与 `clipbyvalue_contiguous{,_reg_base}`，
E6 陆续加了 bitwise_and / floor_div / remainder / sub / div / sign / logical 各自的那份。

**重算的办法（别再 grep td 文本，那正是上一版错的原因）**：td 文本里的 def 可能在没被
`Ops.td` 包含的文件里，grep 得到但从来不会被展开。拿**生成产物**当真源——
`build/.../Asc/IR/AscendCOps.h.inc` 里只有真正展开的 op，每个都带
`getAPIName()` 与 `getExtensionHeaders()`。判据是：矩阵那一侧的 `*_headers` 为空就要
一个同 apiName 且**不带**扩展头的 op（标准调用），非空就要一个同 apiName 且声明了这些头
的 op。

#### ~~前置 D：一条队列上不能混 dtype~~　**已解**（见清单 L-6）

`CheckResourceCompatibility`（`ScheduleToAfir.cpp`）**曾**要求同一个
`(domain, buffer_id, reuse_id)` 下的所有张量 dtype 逐字相同。而 legacy 的调度会把
不同 dtype 的张量放到同一条队列上——队列就是一块字节，`oneBufSize` 取成员的 max
（`codegen_kernel.cpp` 对 `que_buf_num_value` 取 max，遍历整条队列）。AFIR 这边没建模
「成员尺寸不同、取 max」，只有「全都一样」。

**撞上它的场景**：select 的 ui8 掩码与 f32 值三个一起 Load，三个 Load 落在同一条
队列上，当场 `BindingConflict`。绕开的办法是让掩码在图里现算（`load_select_store` 就是
这么写的）。

**已解**：判据去掉 dtype 就够——片上资源的字节数由 ATT 填进 `q<id>_size`
（descriptor 上是 `size_from_resource`），descriptor 上没有一处依赖 dtype。字段清单同时
从两份收成一份（`afir::canonicalResourceFactsAgree`）。从 GM 搬掩码的那张图现在有
E2E：`load_mask_select_store`。详情、负例与埋点见清单 L-6。

**顺带推翻了本节原先写的一句**：「它挡着比较族的下游用法」——`compare → select`
**改动之前就是绿的**。`Eq` 的 ui8 结果是计算中间量，落在 `TBuf<VECCALC>` 上，搬入分在
两条 `TQue<VECIN>` 上，掩码与浮点从来没共用过队列。那张图现在也有 E2E
（`load_eq_select_store`），留着当这句错断言的凭据。

**原先这里写的是「接比较族之前要先解这条」——这句作废，顺序反过来也成立。** 比较族
这一轮先接通了：这条路发得出正确的调用，`load_eq_store`（下游直接 store）从 AscGraph
一路绿到逐元素断言。

两条推论：

- **「一个 op 的改动点 ≤ 3 处」那条验收只对已做的 16 个成立**——它们恰好都有供给。
  剩下这批大部分要**四处**，多出来的一处是 `api/ascendc/include/ascir/Dialect/Asc/IR/`
  下的 td def。验收表述要改成「3 处 AFIR 侧 + 上游供给缺时另加 fork 一处」。
- **方案说「LogicalAnd / LogicalOr / BitwiseAnd 是剩下这批里最便宜的」不成立**，这三个
  都缺一侧供给。真正最便宜的是 **LeakyRelu**：D0、两侧同名同调用、无扩展头、无 tmp、
  ODS 已有、`leaky_relu_l2` 供给齐。唯一的新形状是「ODS 属性（`negative_slop`）变成 API
  实参」——当前一元 pattern 只会传 dst / src / [tmp] / count。

### E3 剩余三档（按依赖顺序）

1. ~~**`scalar_tensor` 方向**（`afir.add %scalar, %tile`）~~ **已接通**（清单 L-4）。
   **原先这条写错了落点**：不需要在 AFIR 上声明交换律，也没有加 `Commutative` trait。
   换的是一根轴——ODS 的 `allowScalarRhs`（一位）变成 `scalarOperand`（`"none"` /
   `"rhs"` / `"any"`），conversion 侧对应 `ScalarOperand{None,Rhs,Any}`，发射顺序由
   `emissionOrder` 一处算出、capability 与 pattern 共用。这 8 个 op 可交换，交换方向
   发出来的调用**逐字相同**，所以**没有增加注册点**，选中的仍是 `<op>.tensor_scalar`。
   E2E 是 `load_scalar_add_store`。
2. ~~**`ScalarData`（运行期标量）**~~ **已接通（2026-09-23）**。
   **原先这条写错了**：通道不是 `afir.tiling_param`，是 **kernel 形参 ABI**——
   `REG_ASC_IR(ScalarData)` 只有 `index`，与 `Data` 同一个序号空间，legacy 的
   `codegen_kernel.cpp` 把两者排在同一个形参循环里，`ScalarData` 排成按值传的标量形参。
   改走 tiling 会让 launch 签名与 legacy 不一致。改动地图与三条设计判断见清单 L-9。
3. ~~**`IndexExpr`**~~ **已接通（2026-09-23）**。
   **原先这条也写错了**：它不需要归纳变量。legacy 把 `IndexExpr` 与 `Scalar` 收进同一张
   `kernel.constant_tensors`，由 `AppendConstTensorInit` 在 kernel 顶部物化，而那一处跑在
   `root_loop.Generate` **之前**；渲染只把 size var 换成 TilingData 字段，轴不参与。
   落点与 `Scalar` 同一处，形状也不特殊。常量与符号两种形态都接通了，后者走
   `materializeRuntimeSize`，并顺带添了套件里第一条动态形状 E2E。详见清单 L-9。
4. ~~**Sub / Div 的标量形态**~~ **已接通（2026-09-24）**，而且这一档比原先写的大：
   连同 `TrueDiv` / `Pow` / `ClipByValue` / `Select` / `Where` 一共七个 op。
   见下「标量形态的三种发射规则」。

（第 1 档只是操作数形态，不新增 op；第 2、3 档各对应上表里的一个 op。）

### ODS 不缺了

**`AFIR_AxpyOp` 这一轮现建（`Elementwise.td`），这条至此清零。** 在它之前：`Remainder`
在 E6 那一轮现建，`MaskedFill` 一直就在、只是不在 `Elementwise.td` 而在 `Condition.td`
（`AFIR_TernaryConditionOp`）——某一版按单个文件 grep，把它算成了缺。方案写的是缺 5 个：
`IsInf` 已在 E4 补上，`ClipByValue` 一直就在（tensor 形态，只是当时没有 buffer 形态）。

**op 已经全部接通，这一节留作判据。** 三个标量源不是 op，不计。

**「ODS 在」不等于「ODS 对」**：`afir.cast` 的两个元素类型列表各漏一个
（输入少 `UI16`、结果少 `UI64`），`afir.floor_div` 原先写的集合与 mem 侧完全不相交。
**比较族是这条判据最后也是最狠的一次兑现**：六行里四行是错的——三行写了 `F64`
（两侧 legacy 从来没有 `float64`）、四行写了 `I64`、`lt` 少了 `I32`。六个 op 的 ODS
本该一字不差，六行各写各的却没人发现。

**没有 implementation 的 op，它的 ODS 谁也没对过。** 接一个之前先拿它的 `REG_ASC_IR`
核一遍，别默认那几行是查证过的。

### E5 / E6 / E7

| 波次 | 内容 | 备注 |
|---|---|---|
| E5 | ~~Select / Where / ClipByValue / MaskedFill~~ **已完成** | 见下「E5 收尾」。MaskedFill 只接 mem 一侧，reg 一侧的结论改了——见那一节 |
| E6 | ~~**Broadcast 接通断头路**~~（常量源一档 2026-09-23 接通，见清单 L-7） + ~~Cast~~ + Sub / Div / ~~TrueDiv~~ / ~~Pow~~ / FloorDiv / Remainder / BitwiseAnd / Sign / ~~Axpy~~ / Gelu / LeakyRelu / LogicalAnd / LogicalOr | Broadcast 原记「四项全差」是错的——供给全在、`asc_graph_builder.h` 的方法也在，真卡点在别处（见清单 L-7）；已接通的 `Duplicate` 一档发标准调用，用不着前置 A，三轴 / 四轴那两档才要；Sign / TrueDiv 要前置 A（都已做）；ODS 缺 Axpy / Remainder；这一批里大部分要先补 fork 侧供给（见「前置 C」），**建议把那批 td def 一次性加齐再逐个接通**，而不是每个 op 各加各的——形状一致，分散做等于重复同样的判断十几遍 |
| E7 | ~~收尾~~ **这一波是空的** | 四项全部在合入前的整改里做掉了，见下 |

**E7 已经没有内容了，这一行是过期的。** 四项逐条核过：`afir.tmp_buffer_ids` 与
`GenTilingData` 特判在收口第 3 条删干净了（今天全仓 grep 只剩文档与注释里的提及，
`GenTilingData.cpp` 那个函数上方还留着一句「这里因此只有两张表，没有第三条分支」）；
C4 负例换 `afir.sum` 已做；覆盖矩阵重跑无漂移；编译耗时已测（+0.88%）。方案文档
第 344 行已经是划掉状态，本文件这一行漏改了。

**与 VF 融合的关系**：无。VF 是独立 PR，两边要动的文件不重叠，理由见上「VF 融合的现场」。
先补 op 再做融合是对的顺序——E5 / E6 接通的 op 正好是 VF 那一版的输入。

---

## 挂起的 op 与理由

这些**不是漏了**，是查证后有具体原因推迟的。动它们之前先读方案里对应的偏差条目。
**这张表空了**：最后一行（比较族 6 个）这一轮接通了。整张表留着不删，因为它记的不是
「还欠什么」，而是**每一条卡点最后是怎么了结的**——七行里有四行的了结方式是「卡点的
说法本身是错的」（前置 B、偏差 6、Cast 的「裸名」、比较族的三件新机制里的两件）。

**这张表原先只盖住 10 个，`Cast` / `Pow` / `TrueDiv` 一行都没有**——它们在方案里挂在 E6，
而 E6 其余的都做完了，于是它们既不在「已落地」也不在这张表上，谁清点都会漏。后来补上了。

| op | 卡点 | 解法 |
|---|---|---|
| ~~Rsqrt、LogicalNot~~ | ~~包装多一个预先填好 1 的 block 张量，「一个 op 至多一块 tmp buffer」表达不了（偏差 5 / 10）~~——**卡点的说法是错的**：调度给这两个 op 分的 tmp buffer 一直只有一块，那块常量在 AscGraph 上不存在 | **已接通**（`ConstBlockTensor.{h,cpp}`，AFIR 一个字没动），见上「Rsqrt / LogicalNot：已接通」与改写后的「前置 B」 |
| ~~Tanh / Erf~~ | ~~偏差 6 的 `isReuseSource`~~——**这条卡点是假的**：`arith.constant false` 本来就 emit 成 `constexpr bool`，能当模板实参 | **已接通**（lowering 物化一个 `arith.constant false`，ascendc dialect 与 CodeEmitter 一行没改），见上「偏差 6……整条推翻了」与下「Tanh / Erf：已接通」 |
| ~~Eq / Ne / Lt / Le / Gt / Ge~~ | ~~两条：① 多一套调用形态，`<T, CMPMODE>` vs `<T, dim, CMPMODE>`，`dim` 由 tile 轴结构推出，运行时参数带 second-to-last stride，当前「连续 tile + 元素个数」表达不了；② 两个扩展头（偏差 8）~~——**① 里的两件不成立**：second-to-last stride 只有 `dim == 2` 读得到，这条路只发连续形态；brace-init 数组 `Cast` 那一轮做掉了 | **已接通**（四个 def、24 个 implementation，真正新的只有「同一个 `CMPMODE` 两侧落在不同的位置」），见上「比较族：已接通」。② 随前置 A 一起解掉了 |
| LogicalAnd / LogicalOr | 本身干净（两侧都是包装，名字与 tmp 需求不同），原本只是等二元路径改造——**E2.5 已经解掉**（偏差 9） | **可以直接排进 E6**，没有别的阻塞 |
| ~~Axpy~~ | 家族判据说不出「两侧共用同一个扩展包装」（矩阵 tier D1，elementwise 里 5 个）；另有二元 + 属性实参 + tmp 的发射形状 | **已接通**（`ApiFamily::Shared`），见上「Axpy：已接通（tier D1 第一个）」 |
| ~~TrueDiv~~ | 结果 dtype 是**一张 (输入, 输出) 对表**，不是输入的函数：mem 侧 `{f16→f16, f32→f32, **i32→f32**}`，reg 侧 `{f16→f16, f32→f32, bf16→bf16}`。`ResultElementType` 那根轴原先的两个取值（`Same` / `Uint8Logical`）都说不出 i32→f32 这一行 | **已接通**（`ResultElementType::SameOrF32ForI32`，mem 侧新建一个 def、reg 侧复用 `DivExtendRegBaseOp`），见上「TrueDiv：已接通」 |
| ~~Pow~~ | ~~两侧供给都缺、两侧都要 tmp、`reg_scalar = any` 撞 L-4~~ | **已接通**（两个 def + 换基类，与 `Remainder` 逐字同形，没有新机制），见上「Pow：已接通」。标量形态 2026-09-24 接通，用的是**同一个 def**（同名重载），见下「标量形态的三种发射规则」 |
| ~~Cast~~ | ~~两侧供给都缺、ODS 不同基、bridge 表里唯一的 `Tensor` 行~~。**「裸名 `Cast`」那句是错的**：mem 侧发的是 `CastExtend`，矩阵那一列是注册名不是发出去的名字 | **已接通**，但不是「两个 def + 换基类」——还要 brace-init 数组 emitter 与 ODS 基类拆档，见上「Cast：已接通」 |

**`LogicalAnd` / `LogicalOr` / `LeakyRelu` 那几句「最便宜」的话已经过期**（三个都接通了），
删掉了。下面这段是按 TrueDiv 之后的现场重新排的。

### 下一步挑哪个（比较族之后重排）

**没有 op 可挑了。** 47 个共有 elementwise 节点类型里，op 那一列已经清零，剩下的三件
都不是「再接一个 op」：

| | 范围 | 先要的那件事 |
|---|---|---|
| ~~前置 D~~ **已解** | 掩码从 GM 搬入的图（不是 compare → select，那张图一直是绿的） | 一条队列上不能混 dtype。判据去掉 dtype，字段清单从两份收成一份；见清单 L-6 |
| ~~L-4~~ **已解** | 8 个 op 的标量格 | `RhsForm` 换成 `ScalarOperand{None,Rhs,Any}`；**没有增加注册点**——8 个 op 全可交换，交换方向发的是同一个调用、选中的还是同一个 `<op>.tensor_scalar`。详情见清单 L-4 |
| E3 剩余档 | `ScalarData` / `IndexExpr` | 两个非 op，依赖顺序见「E3 剩余三档」 |

**这三件都不像前面那些 op 一样能靠「照样板再走一遍」推进**：前两件各自会改到跨 op 的
一根轴（队列的 dtype 建模、`RhsForm`），所以照旧是**先做机制、再一次接通一批**，不要
混进某个 op 的提交里。前置 B 那一轮把机制、两个 op、文档订正放进同一条提交，是因为
机制只有约 100 行且没有第二个消费者——这三件都不是那个体量。

**挑之前先核卡点，这条这一轮又兑现了一次**：比较族的卡点写着三件新机制，核完只剩
一件。核的办法是读两个 `*_api_call.cpp` 与两份头，半天的事，而照着文档估出来的工作量
是它的三倍。

**挑之前先做的两件事**（这一轮都兑现了，值得照着走）：

1. **去查证卡点本身**。前置 B 的说法是错的，证据在 `ascir/reg_func/` 两个 30 行的文件
   里。偏差 6 也是这样。**「文档说表达不了」不是事实，是一条待验证的断言。**
2. **换基类之前先问结果类型推不推得出来**。`AFIR_Bufferizable{Unary,Binary}ElementwiseOp`
   捆着 `InferTypeOpInterface`，而它只拿得到操作数。`cast` 推不出来（结果是写在 op 上
   的），`logical_not` 也推不出来（两侧关系不同）——这两个都得另起一个基类，只取
   `bufferizableElementwiseCommonMethodDef`，`verify()` 自己拼。**声明了却推不准是静默
   的**，这一族的 `isCompatibleReturnTypes` 只比形状。比较族的结果恒为 ui8，是常量，
   推得出来。

**换基类顺带删手写代码那一条仍然成立**：`AFIR_INFER_BROADCAST` 这个宏在 `pow` 换完
之后一个实例都不剩、已经整个删掉了；`Rsqrt` / `LogicalNot` 是一元，只有手写的
`verify()` 那一处（`Ops/Elementwise.cpp`），这一轮删掉了。

---

## 已知覆盖空洞

写下来是为了不被误认为「已经验过」。

0. ~~**`isSetMask` 与 legacy 不一致**~~ **已修，见遗留问题清单 L-2。这条当初的定性错了
   两处，两处都值得记住。**

   - 写的是「两种取值都编得过，差别在 API 内部还设不设 mask」，**轻描淡写了**：`false`
     时 API 按 `repeatTimes = 1` 直发一条指令，`count` 根本没被读。一趟算多于一个
     repeat 就开始错。
   - 写的是「CPU 模拟器与 NPU 在 mask 上不一定同构，要上 NPU 查」，**不成立**：CPU
     模拟器编的就是同一份 `dav_c220` 头，一模一样地错。当时探不到的真正原因是
     **所有用例的形状都是 `[128]`**，切完一趟的 count 不超过一个 repeat（f32 是 64 个）
     ——两种取值在这个范围内结果相同。换成 `[128, 23, 5]` 当场红。

   **教训**：写「CPU 上验不了」之前，先确认 CPU 那一侧编的是不是同一份实现；是的话，
   问题就不在「验不了」，而在用例的形状。

**E0 检视留的那条风险（同一个 tmp id 跨节点复用）已经不在这张表上了**——
`load_abs_reciprocal_store` 把它验了，也把它炸出来的那个 bug 修了，见上「合入前的五项
整改」第 1 条。

1. ~~**`materializeOperandTiles` 的 `ScheduledContiguous` 分支对标量的处理**~~ **已补**
   （L-1）：`load_add_scalar_store`。原先写「因为上面搁置的 ATT tiling 问题没有 E2E」
   ——那条搁置是误判，见上。埋点验过（改坏解析出来的标量值，逐元素断言当场红）。
2. ~~**`getBufferizedElementwiseEffects` 里标量不挂 `Read` effect`**~~ **已补**（L-3，
   `0fc93e1f`）：`AFIRElementwiseContractTest.ScalarOperandCarriesNoReadEffect` 直接断言
   buffer 形态的 effect 列表——tile 一个 `Read`、目的地一个 `Write`、标量一条都没有，
   个数一起钉住。埋点验过：删掉那条 `contributesShape` 判据，该用例红。
   当初难的地方留作记录：它的可观测后果是访存排序，而标量是 `Pure` 的 `arith.constant`，
   多挂一个 Read 在今天这些图上不改变任何调度结果——所以走 UT 直接断言 effect 列表，
   而不是去找一张「标量的位置会影响重排」的图。

3. ~~**`AxpyExtend` 的 f16 分支没有用例走到。**~~ **已补**（L-10）：`load_axpy_half_store`。
   `load_axpy_store` 只做 f32，而 f32 走的是包装里那条**一次都没用 tmp buffer** 的
   `else` 分支；真正用那块 tmp 的是 f16 分支（把 tmp 切两段、两个输入 `Cast` 成 float
   算完再 `CAST_RINT` 回来）。新用例取的值在 half 里全部精确可表示，所以逐元素比相等
   而不是比容差——容差一放宽，「tmp 切错段」这种偏一点的错就正好漏过去。
   **它也是整个 CPU E2E 套件里第一个 f16 用例**（此前只有 f32 / i32 / i16 / ui8）。
   顺带钉住「alpha 恒是 f32、不跟 tile 走」——f32 那份里两者恰好一致，写反了看不出来。

   **验证这条断言不是空的，用的不是「把 `CalcAxpyTmpSize` 的 f16 尺寸改小」**：那条路
   走不通，`b{id}_size` 有 8 KB 下限（`asc_graph_ub_expr_builder.cpp` 的
   `max(Σ, 8192)`），128 个 f16 元素算出来的 1024 字节根本够不着下限，改小了也不变。
   真正有效的埋点是改 `api/axpy.h` 里 f16 分支的 tmp 切分偏移（`offset` 置 0，两段重
   叠），用例当场红。

4. ~~**C++ 侧的浮点字面量 check 有一类是假的。**~~ **已修**（L-11）。emitter 打的是定长
   科学计数（`(float)1.250000000e-01`），而 FileCheck 是子串匹配，于是
   `afir-leaky-relu-scalar-attr.mlir` 里那条 `CPP: ... = (float)1.25` 只匹配上了前缀——
   **斜率改成 1.25 它照样绿**。现在两条都写全了字面量，并把变量名绑进调用；f32 与 f16
   打出来的位数还不一样（`e-01` 前面一个九位、一个五位），所以两条各写各的。把斜率改成
   1.25 当场红，这一条验过。全仓扫过一遍，同类写法只有这一处。
   **新写钉常量的 check 时先看一眼实际输出。**

5. ~~**ODS 的 `logicalResultElementType` 整条轴没有任何用例走到。**~~ **已补**（L-12，
   `0fc93e1f`）：`AFIRElementwiseContractTest` 四条，拿 op 自己的操作数 / 属性去调静态的
   `inferReturnTypes`，**跳过 assembly 里写着的那个结果类型**——这是让「没有 destination」
   那一支唯一走得到的办法。四条里两条是对照组（`truediv` 的 f32 输入必须推出 f32、
   `add` 必须推出输入的元素类型），缺了它们判据是虚的。埋点验过。
   当初的分析留作记录：这条轴只在 `inferReturnTypes` 的「没有 destination」那一支被读，
   而 tensor 形态的 op 两条建法都不走那一支——assembly 里结果类型是写出来的，
   `AfirBuilder` 的 `createLogicalNodeOp` 传的是 `TypeRange{resultTy}`。
   **这不是 TrueDiv 带进来的**，是这条轴从逻辑族起就这样；比较族换基类之后
   这条轴上的代码从「六份手写」变成「一行 ODS」，用户从两个变成八个。
   `truediv` 的 lowering 那一侧不在此列——`SameOrF32ForI32` 有
   `afir-truediv-extension-api.mlir` 的 i32 函数钉着，改回 `Same` 当场红。

6. **比较族 reg 侧的元素个数会被静默截断成 16 位。** 发出去的是
   `CompareExtend<T, 1, MODE>(dst, s0, s1, {static_cast<uint16_t>(n)}, {1}, {1})`，
   而 `api_regbase/compare.h` 的形参就是 `const uint16_t (&)[dim]`，所以 `n > 65535`
   时高位丢掉、算错的元素数，不报任何错。
   **今天到不了，所以不修**：一块 dst（ui8）加两块 src 再算上双缓冲，910B / 950 的 UB
   预算把单 tile 元素数压在 65535 以下；legacy 的 `compare_v2_api_call.cpp` 在同一个
   分支上也是这么写的（`static_cast<uint16_t>(actual_size)`），我们没有比它更松。
   记在这里是因为**这是一条会随 UB 变大而复活的假设**，而且失败形态是算错数不是报错。
   UB 预算变化、或者哪天这条路开始发 `dim == 2` 形态时，回来看这一条。

---

## 遗留问题（合入后一起解决）

> **这张表已经不是待办清单了。** 2026-09-22 的全量评审把它连同散在正文的「已知覆盖空洞」
> 「下一步挑哪个」一起重排进
> [elementwise 切片：遗留问题清单](2026-09-22-afir-elementwise-remaining-issues.md)——
> 那份是**唯一工作清单**（L-1～L-20，分六个阶段，每条带证据与完成判据）。
> 下面这张表原样保留，作为 L-1～L-5 的详情来源；**开工请从新清单进**。

每条各自的详情写在上面或下面，这里只做一张单子。**L-1 ~ L-4 用户已定：本轮都不动，
最后一起解**；**L-5 是前置 B 这一轮新记的，还没定去处**。

| # | 问题 | 详情在 | 解开之后 |
|---|---|---|---|
| ~~L-1~~ **不存在** | ~~ATT tiling 拒收含 `Scalar` 节点的图~~ **这一条是假的**：`load_add_scalar_store` 一次就过。原先的失败是手工跑 generator 时 MLIR 路径谢绝、落到 always-false 的 tiling 桩上 | 上同名小节（原文保留作反面教材）+ 清单 L-1 | 已补 `load_add_scalar_store`，标量族有 CPU E2E 了 |
| ~~L-2~~ **已修** | ~~`isSetMask` 一律打成 `0`~~ 根因是**两个缺省极性相反**：`UnitAttr` 不写 = `false`，AscendC 声明上是 `= true`。改 `VecScalarL2Op` 的缺省为 `DefaultValuedAttr<BoolAttr, "true">` | 上「已知覆盖空洞」第 0 条（含当初两处定性错误）+ 清单 L-2 | ~~**要在 NPU 上查**~~ **不用上机**：CPU 模拟器编的是同一份 `dav_c220` 头，把形状放大到一趟算多于一个 repeat 就红。已补 `load_add_scalar_store_128x23x5` |
| L-3 **半解** | 第一支（`materializeOperandTiles` 的 `ScheduledContiguous` 标量分支）已被 `load_add_scalar_store` 走到；**第二支（`getBufferizedElementwiseEffects` 里标量不挂 `Read` effect）仍然没有人钉**，已实测删掉它 lit 与 UT 全绿 | 上「已知覆盖空洞」第 1、2 条 | 第二支要直接在 UT 里断言 effect 列表 |
| L-4 | **标量只表达得出「在第二个操作数」**。矩阵上 8 个 op 的标量格是 `2nd+swap(2nd)`，交换之后那半张格子 MLIR 整个没有 | 下「L-4：swap 方向的标量」 | 8 个 op 的标量覆盖补齐；`RhsForm` 那根轴换掉之后，`registerScalarBinary` 那几行也要跟着重写 |
| L-5 | 常量 block 的 32 字节 **ATT 预留、MLIR 分配**，两边各自实现同一条「按 (值, dtype) 去重」的规则，**没有共享断言**。今天两边的数相等，靠的是两边都只认 `{1, float}` / `{1, half}` 这两对 | 下「L-5：常量 block 的预留与分配分在两个目录」 | 加一块常量不再要同时改两个目录，也不会再有「MLIR 多分配了一块而 ATT 不知道」这种只在运行期露出来的失败 |

~~L-1 与 L-2 互不相干~~ **L-1 已确认不存在**，所以 L-3 不再等任何东西，L-2 也不再有
「先做哪个」的问题。L-4、L-5 与其余各条都不相干。

### L-4：swap 方向的标量

**这不是哪一波带进来的洞，是一直就在。** `add` / `mul` / `maximum` / `minimum` 从 E3-1
接通标量形态起就只收「标量在第二个」，到 `logical_and` / `logical_or` 才被看见。

范围（对着 `ascir_op_lowering_matrix.csv` 的 `mem_scalar` / `reg_scalar` 两列数，8 个 op、
14 格）：

| op | 带 swap 的侧 |
|---|---|
| Add / Maximum / Minimum / Eq / Ne / LogicalAnd | 两侧 |
| Mul / LogicalOr | 只有 mem 侧 |

**今天的行为是干净回落，不是错编**，这一条要紧：bridge 的 `buildComputeNode` 按图序取
操作数，标量在前就建出 `afir.add %scalar, %tile`；而 ODS 上 `allowScalarRhs` 放宽的只有
rhs，lhs 仍然钉死在 shaped type 上，于是 verifier 当场拒收、整条 MLIR codegen bail 掉、
图回落 legacy。`afir-scalar-binary-invalid.mlir` 末尾两条负例钉住这个状态（一条 `add`、
一条 `logical_and`，两个一起钉是因为这不是逻辑族自己的事）。

要做的时候得动三处，**都不是加一行注册能解决的**：

- ODS：`allowScalarRhs` 那一位要变成「哪个操作数可以是标量」，lhs 的类型约束跟着放宽，
  同时还得挡住「两个都是标量」；
- conversion：`RhsForm` 这根轴要变成 `ScalarOperand { None, Lhs, Rhs }`，
  `getTensorTypes` 与 `materializeOperandTiles` 的分流跟着改；
- 发射：legacy 那边 `binary_api_call.cpp` 的 `is_scalar_latter` 模板实参是随交换翻转的
  （`Sub` / `Div` 那条分支），换轴的时候要一起对。

**会改到 4 个已接通 op 的行为**（Add / Mul / Maximum / Minimum），所以它要自己一个提交
加一轮完整验证，不能塞进某个 op 的接通里。

**上面那张表里的 Eq / Ne 这一轮接通了，但只接了「标量在第二个」那一半**（ODS 上
`allowScalarRhs = 1`，注册了 `*.tensor_scalar` 两侧共四个 implementation）。所以换轴时
要一起对的 op 比原先多了六个——比较族六个都带 `allowScalarRhs`，虽然带 swap 的只有
Eq / Ne 两个。

### L-5：常量 block 的预留与分配分在两个目录

**这一条不是「该把它挪进 `MemoryAllocate`」。** 挪不挪都不影响它——ATT 那边的预留是从
legacy 的 `api_extract_dup_map` 算出来的，换个 pass 放也还是两处各算各的。

（顺带把「为什么不挪」记下来，省得下一个人再推一遍：`MemoryAllocate` 跑在
`SelectImplementation` **之前**，拿不到「选中的是哪一侧」，只能按 op 类型分配——于是
reg 侧会多出一块没人用的 `Duplicate`。legacy 正是这么干的，它那张表也是按节点类型查的；
照抄不难，但换来的是 reg 侧 IR 里一条死代码，而省下的只是把四行 op 从 pattern 挪个地方。）

要紧的是这个**形状**：分配侧与预留侧分在两个目录，两边没有共享的断言——与 `AGENTS.md`
3.1 记的那条 NPU 0x507035 越界同一类（那次是 ATT 按单 tile 分配、下降按多 tile 取用）。
今天两边的数相等，靠的是两边**各自**实现了同一条规则：

- ATT：`CalcExtraTmpBufForAscGraph` 按 `api_extract_dup_map` 去重，结果进
  `tuning_space_->builtin_tmp_buffer`，再由 `buf_occupy_expr.cpp:128` 与 `b{id}_size`、
  队列一起求和算进 UB 占用。**这一条查证过**，不是推的——`SummaryBufferOccup` 那一行就在
  `tmp_buffer` 那个循环下面；
- MLIR：`ConstBlockTensor.cpp` 按 `ascendc.const_block` 这条认领标记去重，每块 32 字节。

**会失效的那一天很具体**：给 `ConstBlockOnes` 加一个枚举值（比如某个包装要一块填 0 的
half），MLIR 多分配 32 字节，而 ATT 那张表里没有这一行、不会多预留。失败是运行期 UB
越界，不是编译报错，**也不会有任何用例变红**——今天两块常量的 e2e 都只证明「这一块是
对的」，证明不了「没有多出来的一块」。

所以这一轮做的不是修它，是**把警告放到要改的那一行上**：`ConstBlockOnes` 的枚举上写了
「加枚举值之前先看 legacy 的 `api_extract_dup_map`」。真要解开有两条路，都不在本轮：
ATT 把这个数发成一个 TilingData 字段供 MLIR 引用（今天它刻意不发，见
`tuning_space.h:261` 的注释「不需要申请TilingData」），或者两边共用一张表。

---

## 接着做需要知道的

### 一个 elementwise op 的改动点

方案的验收条件之一是「新增一个 op 的改动点 ≤ 3 处」。AFIR 侧确实是三处：

1. `dialect/AFIR/IR/Ops/Elementwise.td` 一行；
2. `conversion/AFIRToAscendC/Implementations/{Unary,Binary}Elementwise.cpp` 注册一到两个
   implementation；
3. `bridge/AfirBuilder/AfirBuilder.cpp` 的 `AFIR_BRIDGE_OPERATION_REGISTRY` 一行
   （改完要同步 `BridgeOperationRegistryTest` 的覆盖集数字，它断言集合相等）。

**但剩下的 op 大部分还有第四处**：`api/ascendc/include/ascir/Dialect/Asc/IR/` 下要先加
`ascendc.*` 的 td def，因为上游没有那个包装（31 个里 26 个至少缺一侧，见「前置 C」）。
已做的 16 个之所以是三处，是因为它们恰好都有供给——别把那当成常态。

**而这第四处自己是两处**：新建一个 `ascendc.*` op，打印器要在 **`Target/Asc/Basic/*.h`
的重载集合**和 **`Target/AscendC/Translation.cpp` 的 dispatch 列表**里各登记一次。
只加前一处的表现是 `ascir-translate` 报 `op unable to find printer for op`，
lit 当场红（Tanh / Erf 那轮踩过）。

**第一处（ODS 那一行）也可能不是一行。** 换基类之前先问：**这个 op 的结果类型推得出来
吗？** `AFIR_BufferizableUnaryElementwiseOp` / `...BinaryElementwiseOp` 捆着
`InferTypeOpInterface`，而它只拿得到操作数。推不出来就得像 `afir.cast` 那样另起一个
基类（只取 `AFIR_BufferizableDestinationStyleTraits` 与
`bufferizableElementwiseCommonMethodDef`）——**声明了却推不准是静默的**，因为这一族的
`isCompatibleReturnTypes` 只比形状。同理，op 还要查一张对表的话，`verify()` 也得自己
拼，不能用共享那一段。

### 每个 implementation 由三件**互相独立**的事定死

绑在一起过一次就会在某个 op 上答反，详见 `UnaryElementwise.cpp` 文件头：

- 建哪个 AscendC op（模板实参）；
- 属于哪张扩展 API 供给表（`ApiFamily`，注册期由 `checkFamilyMatchesHeader` 与扩展头名
  对照，对不上则整张注册表建不起来）；
- 要不要 tmp buffer（`TakesTmpBuffer`，同一个模板实参同时决定 capability 判据和建 op 的
  实参个数，所以两者不可能对不上）。

### 「标量是输入但不是 tile」要在六处分别说出来

见方案 E3 段。前四处是写代码时想到的，后两处（`CanonicalBindings::recordArray`、
`CanonicalFunctionMemoryAllocator`）是补调度形态用例时被报错逼出来的——
这类「每个操作数都是一块缓冲」的隐含假设不容易靠通读找全，**加新的操作数形态时优先
补一个过整条流水的用例，让它们自己报出来**。

### 验证与提交

- 验证一律 `./mlir-test.sh`（仓库根目录，不接参数，自带 LLVM/CANN 环境）。
- 提交一律 `SKIP=oat-check`，且要**两遍**：第一遍 clang-format 改文件后失败，
  `git add -A` 再提一次。
- 不要 `pre-commit run --all-files`，会改写约 390 个无关文件。

### 协作方式

本轮沿用的工作法（用户指定）：一个波次做完后，**新开一个辅助 session 做代码检视**，
主 session 确认/修复后提交。检视要点按本轮的经验排：判据是否被悄悄放宽、两条路共用
代码后有没有行为变化、**每条新判据是否真有用例会因为删掉它而变红**。
本轮检视最有价值的一条不是找出 bug，而是逼出了两个没人想到的隐含假设。

### 清点口径

上面「重算」那张表里的数是对着
[ascir_op_lowering_matrix.csv](../guides/ascir_op_lowering_matrix.csv)（`family ==
elementwise`）与 `AFIR_BRIDGE_OPERATION_REGISTRY` 做差集算出来的，不是人工数的。
改完一波之后重算一遍比通读方案可靠——**Sub / BitwiseAnd / Sign 这三个就是这么翻出来的：
它们在方案的任何一条波次清单和偏差条目里都不出现。**

这张矩阵扫的是 **legacy**（`ascir/generator` 的 v1/v2 impl），与 pyasc 无关，所以 pyasc
同步不会让它过期。**「fork 侧有没有 `ascendc.*` 供给」是另一张表**，要拿矩阵的
`mem_api_name` / `reg_api_name` 去比对 `api/ascendc` 里 td def 的 apiName（`def X : Y<"助记符",
"ApiName">` 的第二个字符串），两边对不上就是缺供给。

**「共注册几个 implementation」也别人工数。** 一元那条路有 `registerPredicate`，
一行注册两个，肉眼数必错——「一元 19」就是这么错的，Gelu 与 Sign 接通时漏改，从此
一直少 3。下面这段按「字面 id 的个数 + `registerPredicate` 的行数」算（后者一行两个）：

```bash
python3 -c '
import re
for path in ("Unary", "Binary"):
    f = "autofuse/mlir/conversion/AFIRToAscendC/Implementations/%sElementwise.cpp" % path
    s = open(f).read()
    ids = re.findall(r"registry,\s*\"([a-z0-9_.]+)\"", s)
    pred = re.findall(r"registerPredicate<[^>]*>\(registry,\s*\"[a-z0-9_]+\"\)", s)
    print(path, len(ids) + len(pred))
'
```

**「CPU E2E 几个用例」也有两个口径**，别混：`test/e2e/cpu/cases/` 的目录数与
`add_mlir_e2e_cpu_case` 的行数不相等（`load_add_store` 一个目录三个形状）。

```bash
ls autofuse/mlir/test/e2e/cpu/cases | wc -l                                  # 目录数
grep -rh add_mlir_e2e_cpu_case autofuse/mlir/test/e2e/cpu/cases/*/CMakeLists.txt | wc -l  # 注册用例数
```

```bash
# 矩阵（legacy 侧）
python3 autofuse/mlir/tools/af-mlir-tools/scan_ascir_op_lowering_matrix.py
# pyasc 清单（fork 侧），同步之后要重跑，docs/guides 里那份快照现在是 E0 时的
python3 autofuse/mlir/tools/af-mlir-tools/scan_pyasc_coverage.py --format csv \
  --output autofuse/mlir/docs/guides/pyasc_ascendc_op_inventory.csv
```

~~**`docs/guides/pyasc_ascendc_op_inventory.csv` 这份快照停在 E0，`fe96b79d` 之后没重跑**
（同步新增 75 个 op、删 2 个）。哪一波要用它之前先重新生成。~~ **这条已过期**：后来加了
pre-commit 的 `pyasc-op-inventory` 门禁（`always_run`，`--check`），改了 td 而忘了重跑会被
拦住，所以快照现在恒是最新的。**门禁只 check 不写**——它报出 diff 之后要自己跑一遍
（去掉 `--check`）再提交。
