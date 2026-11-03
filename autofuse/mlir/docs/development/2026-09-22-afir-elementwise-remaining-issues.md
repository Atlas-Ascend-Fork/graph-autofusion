# AFIR Op 补齐（elementwise 切片）：遗留问题清单

> 2026-09-22 · 进行中 · 上游：[elementwise 切片开发方案](../design/2026-09-17-afir-op-completion-elementwise-slice-design.md)、
> [进度与剩余任务](2026-09-17-afir-op-completion-elementwise-progress.md)
>
> 这份文档在「比较族接通」（`ff865292`）之后的一次全量代码评审里重排，**取代进度文档
> 「遗留问题（合入后一起解决）」那张表**作为后续修复的唯一工作清单。进度文档里那一节
> 以及散在正文各处的「已知覆盖空洞」「下一步挑哪个」仍然是**详情来源**，但不再是待办
> 清单——开工时从这里往回查。

## 2026-10-20 补充：Broadcast + Reduce 融合覆盖缺口（待定位）

状态：**仅记录，尚未修复**。本轮确认 8 个配置中仅 1 个真正生成 MLIR kernel，另 7 个在
生产入口回落 legacy。不能将数值全通过解读为 MLIR 全覆盖。

### 复现范围与结果

- 源码提交：`e01e80abc0ea6172307fdef7385a83da78912a4b`；设备报告 SoC：`Ascend910_9362`。
- dtype：float32；静态二维图，经真实 optimizer 生成 schedule；ATT 配置为
  `--force_schedule_result=0`，未强制具体 tiling case。
- AR 归约 `dim=1`，RA 归约 `dim=0`。前置广播的 bias 沿 A 轴扩展：AR bias 为 `(1,R)`，
  RA bias 为 `(R,1)`。后置广播使用 `keepdim=True`，把归约结果广播回 x 的形状。
- 原生 NPU kernel 验证：ATT 数据在本地 CPU 构建中生成后回放，不是远端 Python/Inductor
  全链路验证。每个配置跑直接 legacy 基线和请求 MLIR 的生产路线，各用两类合成输入，
  每类输入首次及重复 launch 各一次。

| 问题 | 图与输入形状 | 实际路线 | 已确认诊断 |
|---|---|---|---|
| BR-1 | `sum(x + bias)`：AR `(16,32)`、`(16,37)`；RA `(32,16)`、`(37,16)` | 4 项 legacy fallback | `cannot build the cache guard for exec_condition cache_block_split_fused_brc_axis` |
| 正向对照 | `x - sum(x, keepdim=True)`：AR `(16,32)` | MLIR | 无能力不足诊断，真机数值正确 |
| BR-2 | 同上：AR `(16,37)` | legacy fallback | `'afir.broadcast' op has no supported Ascend implementation` |
| BR-3 | 同上：RA `(32,16)`、`(37,16)` | 2 项 legacy fallback | `copy_from_load` 的 `queue buffer_id=0 reuse_id=0` 同时承载 broadcast tile 和 full-shape tile |

共 **64 次 launch**：4 次实际 MLIR、28 次请求 MLIR 后的 legacy fallback、32 次直接 legacy。
数学参考错误为 0；32 组路线输出比较、32 组首次/重复比较均一致。
实际路线由 `RunMlirCodegen` 状态及生产入口 kernel 与直接对应路线 kernel 的比较确认；
源码比较仅归一化同进程递增的局部 `_mask_list` 编号。

### 当前判断与待定位项

- **BR-1：执行条件 lowering。** 在 `ExecConditionPass.cpp` 构造广播缓存 guard 时拒绝；
  待核对 fused broadcast axis 的调度信息、轴绑定与 guard 推导。尚未确定具体修复点。
- **BR-2：Broadcast 实现匹配。** 对齐形状通过，尾块形状拒绝；待定位布局、对齐或实现匹配
  条件中的具体限制，不能仅凭诊断断定高阶 Broadcast API 无法处理。
- **BR-3：资源视图建模。** 待对照 legacy 如何表达并使用共享 queue/reuse 资源。必须先
  证明共享和生命周期语义，再决定调整 bridge 表达还是资源校验；不能直接放宽校验。
- 三类失败均发生于编译期，当前没有 Reduce 归约执行错误的证据。初步归于广播与调度、
  资源建模的衔接缺口，而非已证实的 Reduce 算法缺陷。
- 修复验收需使用原始合法图和同一份 schedule，验证 `RunMlirCodegen == kSucceeded`，
  并完成 legacy/MLIR/数学参考的首次及重复 NPU 对照。不得通过删除候选、修改参考值或
  接受 fallback 来将这些格子标为 MLIR 通过。

### 证据与探针注意事项

- 本地目录（仓库根目录下，未纳入 Git）：`build/broadcast-reduce-e01e80ab-npu/`。
  `results.json` 为分项结果；`provenance.json` 记录源码提交、codegen 库和探针哈希；
  `Case.cpp`、`prepare.py`、`source/main.cpp` 为构图、生成与设备执行脚本；
  `generate/probe*/case0/generate.log` 含实际路线证明和诊断；`evidence/` 保存设备日志、
  退出码、源码及二进制哈希。
- 远端目录：`/mnt/workspace/agent/broadcast-reduce-e01e80ab-npu-20261020`，证据保留。
- 现有 CPU 生成器的 `TilingStub` 固定返回失败，不能用它判断 legacy 回落后的正确性。
  本探针使用临时适配库调用真实 ATT `GenTilingImplAutoFuseV3`，并在 host wrapper 中适配
  legacy ATT 的 64 位 workspace 参数；未修改生产代码。
- 通用 NPU runner 只打印请求的 `backend_route` 不足以证明实际使用 MLIR，后续覆盖报告
  必须继续区分 MLIR 成功与 legacy fallback。

## 从这里开始

| 你要做什么 | 读哪几节 |
|---|---|
| **挑一件事做** | 「修复顺序」→ 那一条的 `L-x` 详情 |
| **想知道某条为什么在这儿** | 该条的「证据」一栏，全部带 `文件:行` |
| **想知道什么算做完** | 该条的「完成判据」一栏——**每条都要有一个会变红的东西**，不接受「读过代码觉得对」 |
| **回查这批改动本身** | 进度文档的各 op 小节；本文档不重复 |

五条纪律，写在最前面：

1. **「文档说卡住了」不是事实，是待验证的断言。** 前置 B、偏差 6 都是这么被推翻的。
   本清单里凡是写「卡点」的，先去核卡点。**L-6 又兑现了一次，而且这次错的是本清单
   自己**：它写的复现图（`compare → select`）改动之前就是绿的，真正撞上那条限制的是
   另一个形状。**L-2 是第四次，「要机器」不成立**：那条在 CPU 模拟器上就变红，只要把
   形状放大到一趟算多于一个 repeat。写「要机器」之前，先说清楚**为什么现有 harness
   到不了**——说不清就是没核。
2. **判据有两份编码时，分开写就会分开错。** 本清单里 L-5、L-8、L-14 全是这个形状。
3. **「lit 绿」证明不了「kernel 编得过」。** L-13 曾是这条的最大一笔欠账，已还上——
   去掉一个前导头之后 device 编译器报的是**内联进 kernel 的那段头文本的行号**，
   与设计文档描述过两次的失败形状一模一样。
4. **排查工具失效会被记成被测对象失效。** L-1 整条是这么来的：手工跑 E2E generator 少了
   两个环境变量，MLIR 路径谢绝、落到 always-false 的 tiling 桩上，报出来的行号被当成
   「ATT 拒收 `Scalar`」写进了文档。**E2E 一律走 `check-e2e-cpu-<用例名>` 目标**（用例名 = 目录名 + 形状后缀，`ctest -N` 可列出当前全部名字）。
5. **缺省反了的接口，修调用方是修不完的。** L-2 的根因不是「少设了一个属性」，而是
   MLIR 的 `UnitAttr`（不写 = `false`）与 AscendC 的 `bool isSetMask = true` 两个缺省
   极性相反，**危险的那一侧正好是什么都不写时落到的那一侧**。见到「这个属性今天没人设」
   先别急着去设它——先看不设它对应的是哪一侧。

---

## 基线（2026-09-22，`ff865292`）

- `bash ./mlir-test.sh` 四个目标全绿：`check-ut` / `check-lit` / `check-st` / `check-e2e`，退出码 0。
- 47 个共有 elementwise 节点类型：**44 个 op 在 bridge 白名单里且有 conversion 实现**
  （`Broadcast` 的常量源一档 2026-09-23 接通，豁免名单已清空，其余各档见 L-7），
  3 个非 op（`Scalar` / `ScalarData` / `IndexExpr` 均已接通，见 L-9）。
- **`MaskedFill` 不在这条路上，也不该在**（2026-09-23 起）：`MaskedFillInputReorderPass`
  在调度之前就把它改写成 `Select`，所以 bridge 看不到这个节点类型。它的 ODS、bridge 行与
  mem 侧 implementation 原先都在，但一条用例都走不到，已一并删除。数字因此从 44 变 43。
  见 L-8。
- CPU E2E **44 个目录 / 47 个注册用例**。默认 soc 仍是 `Ascend910B1`（mem-based 一侧），
  **其中 4 个用例按 `PLATFORM reg_base` 跑在 `Ascend950PR_9599` / NpuArch 3510 上**（L-13 补的）。
  **这两个数别混**：`load_add_store` 一个目录注册三条、`load_add_scalar_store` 注册两条。
  原先这里写「44 个用例」，数的是目录，而下一条写「44 条里只有 3 条不是 `[128]`」，
  数的是注册——两句拼在一起就成了「41 条 `[128]`」，而注册里 `[128]` 的实际是 44 条。
- ~~**形状几乎全是 `[128]`**~~ **已整套换掉（2026-09-23）**：44 条 `[128]` 注册全部改成
  **2~4 轴、每轴取 [1, 512] 里的随机值**（种子固定、一次抽定写成字面量，红了能复现），
  元素总数约束在 [2048, 262144]，最慢的用例仍在秒级。**换完当场红 26 条，全是同一个
  根因**——裸 `AscendC::DataCopy` 的 32B 对齐，见 L-7 的 (c)。也就是说这个形状不只藏住了
  L-2，还藏住了一个**在 NPU 上静默算错**的缺陷。挑判据时那句「在 `[128]` 上走不到吗」
  从此不用问了，换成：**新用例不要用一根轴、也不要用只含 2 的幂的形状**，那样每趟
  count 恒对齐，尾块那一类判据照样探不到。
- **轴会被融成一根再切，这是正常的**：融轴在 legacy 的 optimizer 里（`optimize.cpp:898`
  的 `MergeAxis`，还有 `AfterMergeAxis` 这个 dump 阶段），MLIR 没见过未融的形态；pointwise
  在业界也都这么做。所以这些形状覆盖的是「融轴后总量切不齐」。L-7 里「`TransferPlan`
  只建模一根向量化轴」那条**不属于 elementwise**——它只在融轴不合法时出现（广播的
  stride 0、转置 / 非连续 slice、沿内轴的 reduce），别拿这一波当它的覆盖。
- **「接通」= 这条路发得出正确的调用**，不等于「所有用得到它的图都能跑」（L-6），
  也不等于「两侧都能跑」（L-8、L-13）。

---

## 修复顺序

分六个阶段。阶段内可并行，跨阶段的依赖在每条的「依赖」栏里写死。

| 阶段 | 内容 | 为什么是这个位置 |
|---|---|---|
| ~~**A　零依赖小修**~~ **已完成** | L-10、L-11、L-15、L-16 | 见下，四条都验过埋点。原先的判断「合起来一两个提交」成立，但 L-15 比预想的大一圈：补一元判据时发现二元那条负例也是假的 |
| ~~**B　reg-base 实测**~~ **已完成** | L-13 | 四条 reg-base CPU E2E 跑通（模拟器确实在 `dav_3510` 上），加 mem 侧最后一个洞 `load_sigmoid_store`。两层埋点都验过 |
| ~~**C　结构性**~~ **已完成** | L-6、L-4 | 两条各自改到一根跨 op 的轴（队列的 dtype 建模、`RhsForm` → `ScalarOperand`）。L-6 的实际改动比预想小得多（判据去掉一个字段）；L-4 少了一处（`is_scalar_latter` 不在范围内，8 个 op 全可交换、发的是同一个调用） |
| ~~**D　外部依赖**~~ **已清空——两条都没有外部依赖** | ~~L-1（legacy ATT）~~ **不存在**、~~L-2（要上 NPU）~~ **已修，没上机** | **两条都是「文档说卡住了」被核掉。** L-1 那张图一次就过，原先记的失败是手工跑 generator 时 MLIR 路径谢绝、落到 always-false 的 tiling 桩上；L-2 在 CPU 模拟器上就复现了，原先探不到只是因为**所有用例的形状都太小** |
| **E　补齐与加固** | ~~L-14~~、~~L-12~~、~~L-3~~、~~L-8~~、~~L-7~~、~~L-9~~、L-5 | 依赖前面几阶段，或者本身就是「防下一个人犯错」而不是「今天有错」。加固四项已在 `0fc93e1f` 做掉，五处埋点逐条验过；L-8 后半的结论是**把 `masked_fill` 整个从这条路上删掉**（节点到不了，留着是死代码）。**L-7 与 L-9 的完成判据在 2026-09-23 这一轮被订正，原说法都不成立，订正后两条都已落地**：L-9 的 `ScalarData` 走的是 kernel 形参不是 TilingData 字段；L-7 记的三条前置有两条不成立，「`TransferPlan` 只建模一根向量化轴」只挡住三轴 / 四轴那一档，常量源的 `Duplicate` 一档已接通，真正挡住 CPU E2E 的是计划选择 / `exec_condition` / `DataCopy` 对齐这三件原先没记的事，其中 `DataCopy` 对齐这一件 2026-09-23 已修，且它根本不是广播独有的 |
| **F　可维护性** | ~~L-17~~、L-18、L-19、L-20 | 不阻塞任何 op，但决定下一批 op 贵不贵。**L-17 已解（2026-09-24）**：定制内容收进 `Autofuse/` 子树，上游文件的差异从约 1500 行降到 217 行，其中只有 51 行是 autofuse 接缝。L-18 第 1 条性价比最高 |

---

## 阶段 A：零依赖小修

### ~~L-10~~　`AxpyExtend` 的 f16 分支没有任何用例走到　**已修**

- **现象**：`load_axpy_store` 只做 f32，而 f32 走的是包装里**一次都没用 tmp buffer** 的
  `else` 分支。真正用那块 tmp 的是 f16 分支（把 tmp 切两段、两个输入 `Cast` 成 float
  算完再 `CAST_RINT` 回来）。
- **所以现有用例证明的是**「tmp buffer 被正确传进去了」，不是「那块 tmp 被正确用了」
  ——包括 `CalcAxpyTmpSize` 的 f16 那支算出来的尺寸够不够。
- **证据**：[Implementations/BinaryElementwise.cpp](../../conversion/AFIRToAscendC/Implementations/BinaryElementwise.cpp)
  的 `registerBinaryScalarAttr<AxpyOp, ...>(registry, "axpy.tensor_tensor", kF16 | kF32)`；
  `autofuse/ascendc/api/axpy.h` 的 `if constexpr (IsSameType<T, half>)` 分支。
- **落地**：`test/e2e/cpu/cases/load_axpy_half_store`。取的值在 half 里全部精确可表示
  （lhs ∈ [-3,3] 整数、rhs ∈ {2,4,6,8,10}、alpha = 0.5），所以逐元素比相等而不是比容差
  ——容差一放宽，「tmp 切错段」这种偏一点的错就正好漏过去。
  **它也是整个 CPU E2E 套件里第一个 f16 用例**（此前只有 f32 / i32 / i16 / ui8），
  半精度这条路（ATT tiling 的 dtype、GM 上 2 字节步长、`Cast` 往返）在这一层第一次走通。
  顺带钉住「alpha 恒是 f32、不跟 tile 走」——f32 那份里两者恰好一致，写反了看不出来。
- **原先写的完成判据是错的，记下来**：「把 `CalcAxpyTmpSize` 的 f16 尺寸改小一半」这条
  路走不通。`b{id}_size` 有 8 KB 下限（`asc_graph_ub_expr_builder.cpp` 的
  `max(Σ, 8192)`），128 个 f16 元素算出来的 1024 字节根本够不着下限，改小了最终值不变。
  真正有效的埋点是改 `api/axpy.h` 里 f16 分支的 tmp 切分偏移（`offset` 置 0，两段重叠）
  ——**已验，用例当场红**。

### ~~L-11~~　C++ 侧的浮点字面量 CHECK 有一类是假的　**已修**

- **现象**：emitter 打的是定长科学计数（`(float)1.250000000e-01`），而 FileCheck 是子串
  匹配，于是 `afir-leaky-relu-scalar-attr.mlir` 里那条 `CPP: ... = (float)1.25` 只匹配上了
  前缀——**斜率改成 1.25 或 12.5 它照样绿**。
- 同一份文件的 IR check 是完整的，所以值本身有人钉，漏的是「C++ 侧打出来的是不是同一个数」。
- **证据**：`autofuse/mlir/test/lit/conversion/afir-leaky-relu-scalar-attr.mlir`。Axpy 那一轮
  的 lit 与 E2E 都按完整字面量写了，可以照抄。
- **落地**：两条 CHECK 都写全了字面量，并把变量名绑进调用
  （`AscendC::LeakyRelu<float, 0>({{[^,]*}}, {{[^,]*}}, [[SLOPE]],`）。f32 与 f16 打出来的
  位数不一样（`e-01` 前面一个九位、一个五位），所以两条各写各的、不能共用一个模式。
  **埋点验过**：把输入里的斜率改成 `1.250000e+00`，用例当场红。
- 全仓扫过一遍 `test/lit/`，同类写法只有这一处。
- **纪律**：**新写钉常量的 check 时先看一眼实际输出**，别照抄那一条。

### ~~L-15~~　一元路径缺一条二元 / 三元都有的判据　**已修**

- **先澄清一件事**：`Identity` 是 `TransferPlanKind` 的一个取值，指「op 上没有 `sched`
  属性、或者 `buildScheduledPlan` 失败，且 indexing map 是 identity / 缺省」的那条路
  （[Analysis/TransferPlan.cpp:174](../../dialect/AFIR/Analysis/TransferPlan.cpp:174)）。
  **生产路径走的是 `ScheduledContiguous`，那一档对 rank 没有任何限制**；下面说的只是
  未调度那条路。
- **现象**：`Identity` 计划下，二元与三元 capability 要求所有 tile `rank == 1`
  （[BinaryElementwise.cpp](../../conversion/AFIRToAscendC/Implementations/BinaryElementwise.cpp)、
  [TernaryElementwise.cpp](../../conversion/AFIRToAscendC/Implementations/TernaryElementwise.cpp)），
  这是**有意的**（`afir-to-ascendc-binary-l2-invalid.mlir` 里有一条名为
  `unsupported_add_rank` 的负例——但见下，那条负例其实是假的）。
  **一元路径没有这一条**：[UnaryElementwise.cpp:174](../../conversion/AFIRToAscendC/Implementations/UnaryElementwise.cpp:174)
  拿到 plan 只判了 `failed()` 就把它丢了。
- **失败形态**：`afir.abs` 收一块 `2x8xf16` 会被判成 supported 并下降，发出去的 count 是
  `GetSize()`（全部元素数）；同一份形状的 `afir.add` 当场报 unsupported。三条路号称同构，
  这一处不同构，而且松的是没有负例的那一条。
- **落地**：走的是 (a)——一元补上同一条判据。但做的过程中翻出**第二件事，比第一件更要紧**：
  `afir-to-ascendc-binary-l2-invalid.mlir` 里那条 `unsupported_add_rank` **根本没有钉住秩**。
  那份 IR 一个 descriptor 都不写，于是先被 `hasCanonicalDescriptors` 判成
  MissingInformation，报出来的错与秩无关；把二元的秩判据整条删掉它照样绿。同一份文件里的
  `unsupported_add_type`（f64）也是同样的毛病。
- 所以负例重做成一份 `afir-elementwise-identity-rank-invalid.mlir`，一元 / 二元 / 三元
  三条一起钉，descriptor 三件写全。**挑 op 的判据是「只有秩这一条能触发」**：一元用
  `afir.exp`、二元用 `afir.add`（家族 `Any`、不要 tmp、收 f16，前面三条判据全过），
  三元用 `afir.select` 并把 tmp 一起喂上。换成 `afir.abs` 就不行——910B1 上它会先被
  tmp buffer 那条拦掉，秩的判据删光也照样绿。`unsupported_add_type` 顺手补上了 descriptor。
- **三条都验过埋点**：分别删掉一元 / 二元 / 三元的秩判据，用例当场红。
- **这不是本批 26 个提交引入的**，`8fcd2ebe` 之前就这样。

### ~~L-16~~　三条 nit　**已处理**

1. ~~`printApiExtensionBinary` 的 `op` 形参没用也没 `(void)op`~~ **已加**，并写明为什么
   这个形参还留着（与同文件的两个 compare body 保持签名一致）。
2. ~~[check_ascendc_td_reachable.py](../../tools/af-mlir-tools/check_ascendc_td_reachable.py)
   在 `DIALECT_DIR` 不存在时 `return 0`~~ **已改成 `return 1`** 并打出当前工作目录。
   原先那样写，脚本一旦被 CI 从别处调用，门禁会**静默通过**——和它自己文档里批评的
   「`files:` 被 `exclude` 过滤成空之后静默跳过」是同一种失败形状。
   已验：从仓库根跑退出 0，从 `/tmp` 跑退出 1 并说明原因。
3. 比较族 reg 侧发的是 `{static_cast<uint16_t>(count)}`（`CompareExtend` 的
   `const uint16_t (&)[dim]`），count > 65535 会静默截断。**今天到不了**：算上 dst、两块
   src 与双缓冲，910B / 950 的 UB 预算把单 tile 元素数压在 65535 以下；legacy 的
   `compare_v2_api_call.cpp` 同一分支也是这么写的。**按判断不改**，已记进进度文档
   「已知覆盖空洞」第 6 条——它是个会随 UB 变大而复活的假设，失败形态是算错数不是报错。

---

## 阶段 B：reg-base 实测

### ~~L-13~~　**reg-base 那半边的 kernel 从来没有被任何编译器看过**　**已解**

这是这批改动里最大的未验证面，比本清单其余任何一条都重要。

- **现象**：CPU E2E 的 soc 固定 `Ascend910B1`，也就是 mem-based 一侧；lit 不注入
  provider、`#include "x.h"` 原样保留；`test/st/` 下只有 bridge 用例。于是这 26 个提交接通的
  **15 个 `*_reg_base.h` 的内联文本，一行都没有被编译过**：

  `neg` `sign` `tanh` `erf` `floor_div` `remainder` `pow` `sub` `div` `logical`
  `logical_not` `compare` `cast` `where` `clipbyvalue`（均为 `_reg_base.h` 后缀的注册 key）

- **为什么这条要紧**：这正是 E2 那一轮吃过两次亏的形状——「include 照发、key 照解析、
  文本照内联，最后炸在 device 编译器上，报的是内联进来那段头文本的行号」。今天 reg 侧
  前导头的判断（`sign_reg_base.h` / `pow_reg_base.h` 要 mem 的 `utils.h`，其余不要）
  **全部是读出来的，不是编出来的**。评审时我逐个核对了两侧 25 个头的
  `MAX_REPEAT_TIME` / `KernelUtils` / `Rational` 用法，与 td 里的 `preludes` 声明一处不差；
  但 `api_regbase/compare.h` 里 `CreateMask` / `UpdateMask` / `RegTensor` / `Std::` /
  `MicroAPI::` 全是不带限定的名字，靠 `AddKernelIncludes` 发的那条 `using namespace AscendC;`
  兜着——这条链没有任何用例走过。
- **mem 侧同类的洞只剩一个**：`sigmoid.h`。除它之外，mem 侧每一个在用的扩展头都至少有
  一个 CPU E2E 编过它（`abs` `reciprocal` `neg` `isnan/isfinite/is_inf` `bitwise_and`
  `floor_div` `remainder` `logical` `cast` `sign` `true_div`+`scalar_div` `pow` `axpy`
  `rsqrt` `logical_not` `compare`+`compare_v2` `clipbyvalue` `duplicate`+`where` `utils`）。
  **`sigmoid.h` 偏偏是当初暴露「前导头」这个问题的那个 op**，它自己的头文本却没人编过。
- **可行性（已查）**：`Ascend950PR_9599` 两边都有——
  profile [Profiles/platform_config/Ascend950PR_9599.ini](../../target/Ascend/Profiles/platform_config/Ascend950PR_9599.ini)
  里 `NpuArch=3510`（即 `RegBasedApiFamily` 为 `Available`），本机 CANN 也有
  `tools/tikicpulib/lib/Ascend950PR_9599`。所以**看起来一条 reg-base 的 CPU E2E 是做得出来的**，
  不必退到 compile-only。但 ATT tiling / 数值模拟那一侧没验过，**这条本身要先实测**。
- **落地**：五个用例，CPU 模拟跑得通，**没有退到 compile-only**。运行日志里是
  `simulator/dav_3510/lib/config.json`，模拟器确实在 3510 架构上。

  | 用例 | 覆盖的形状 | 实际发出的调用 |
  |---|---|---|
  | `load_sigmoid_store` | mem 侧最后一个没编过的头 | `SigmoidExtend<float>(dst, src, tmp, count)` |
  | `load_neg_reg_base_store` | 通路本身：纯转发、无前导、无 tmp | `NegExtend<float>(v67, v66, static_cast<uint32_t>(v59))` |
  | `load_sign_reg_base_store` | **前导头可以是中立的**：reg 侧的包装用 mem 的 `utils.h` | `SignExtend<float>(v79, v78, v45, ...)` |
  | `load_eq_reg_base_store` | **micro-API 一族**：`CreateMask` / `RegTensor` / `Std::` 靠 `using namespace AscendC;` | `CompareExtend<float, 1, CMPMODE::EQ>(..., {static_cast<uint16_t>(v87)}, {static_cast<uint16_t>(1)}, {static_cast<uint16_t>(1)})` |
  | `load_cast_reg_base_store` | **brace-init 数组**发射路径 | `CastExtend(v67, v66, {static_cast<uint32_t>(v59)}, {1U}, {1U})` |

- **换平台要同时改三处，少一处就不是真的换了**，这条写进了
  [AddCpuCase.cmake](../../test/e2e/cpu/AddCpuCase.cmake) 的注释：

  1. `CodegenOptions::soc_version` —— MLIR 侧查 TargetProfile，决定 `RegBasedApiFamily`，
     也就是选哪一侧的 implementation；
  2. **NpuArch 字符串** —— legacy 那一半全部按它分流。`REG_ASC_IR(...).Impl(...)` 注册的
     键就是 `"2201"` / `"3510"`，optimizer 的 `PlatformFactory`、ATT 的 tmp 尺寸函数也都查
     它。只改第 1 处的话，调度算的还是 v1 口径的 tmp buffer，而 MLIR 选的是 reg 侧实现，
     `checkTmpBuffer` 当场判 unsupported；
  3. tikicpulib 目标 —— CPU 模拟器按芯片分库。

  第 2 处走 `PlatformContext::SetPlatformInfo`，**不动共用的 runtime stub**（它由整个测试
  套件共用）。**不能用 `SetPlatform`**：后者只写 `soc_ver` 就把 `initialized_` 置了位，
  `aiv_num` / `ub_size` / `l2_size` 全留在 0，调度会拿到一台没有 UB 的机器。

- **顺带看见了两件此前只是推断的事**：
  - **`TmpBufferUse::Unused` 那一档第一次看得见**。`load_eq_reg_base_store` 的 kernel 里
    `TBuf` 计数是 1——`GetCompareSizeV2` 照样分了 32 字节，而 reg 侧的 `CompareExtend`
    收不下它。那块没人读的 `TBuf` **没有被优化掉**，这正是需要的：ATT 的 UB 预算按分了
    它算。
  - `afir.cast` 两侧的 tmp 差别成立：reg 侧 `TBuf` 计数 0，mem 侧要一块。
- **断言是看过实际输出才写死的**（L-11 那条纪律的直接应用）：先用宽松断言编出来，dump
  生成的 kernel，再按实际文本收紧，包括 `{static_cast<uint16_t>(1)}` 与 `{1U}` 这种
  「写成另一种就编不过」的细节——比较族收 `uint16_t`、cast 收 `uint32_t`，不是同一个。
- **两层埋点都验过**：去掉 `AscendC_SignExtendRegBaseOp` 的 `["utils.h"]` 前导，
  `load_sign_reg_base_store` 先在 `CheckGeneratedKernel` 那条断言上红；把那条断言也去掉
  之后，**device 编译器**接着报错。也就是说这条「reg 侧的包装需要 mem 的 `utils.h`」
  从「读出来的」变成了「编译器验过的」。
- **还没覆盖到的**：剩下 11 个 `*_reg_base.h`（`tanh` `erf` `floor_div` `remainder` `pow`
  `sub` `div` `logical` `logical_not` `where` `clipbyvalue`）仍然没有编译型用例。它们的
  形状都落在上面四类里（无前导转发 / 要 mem 前导 / micro-API / brace-init），所以风险
  比原先低一个量级，但不等于零。**新接 reg 侧的 op 时顺手加一条 `PLATFORM reg_base`
  用例**，现在这是一行 cmake 的事。

---

## 阶段 C：结构性

### ~~L-6~~　前置 D：一条队列上不能混 dtype　**已解**

- **现象**：比较族六个、逻辑族两个、谓词族三个的产出都是 `ui8`。当它与 `f32`/`f16`
  落在同一条队列上时，`CheckResourceCompatibility` 当场 `BindingConflict`，整个 kernel
  生成失败。
- **本条写的复现图是错的，记下来**：`compare → select` **改动之前就是绿的**，从 AscGraph
  一路绿到逐元素断言。原因在调度——`Eq` 的 ui8 结果是计算中间量，落在
  `TBuf<VECCALC>` 上；四块搬入分在两条 `TQue<VECIN>` 上，掩码与浮点从来没共用过队列。
  「比较族最常见的下游用法一个都跑不了」这句不成立。
- **真正撞上它的形状**：**掩码自己从 GM 搬进来**——三块搬入（ui8 + f32 + f32）落在同一条
  `TQue<VECIN>` 上。这正是进度文档「前置 D」原本写的那个场景，也是
  `load_select_store` 文件头记的那条绕法所绕开的东西；本清单成文时把它改写成
  `compare → select`，改错了。
- **落地**：判据去掉 dtype 就够——队列就是一块字节（legacy `oneBufSize` 取成员的 max），
  而片上资源的字节数由 ATT 填进 `q<id>_size`（descriptor 上是 `size_from_resource`），
  **descriptor 上没有一处依赖 dtype**。
  - 两处判据合成一处：字段清单现在只在
    [ResourceIds.h](../../dialect/AFIR/Analysis/ResourceIds.h) 的
    `canonicalResourceFactsAgree`，bridge 的 `CheckResourceCompatibility` 与 conversion 的
    `registerResource` 各查它一遍。不参与比较的三类（`slot_count` 取 max、`reuse_id` 是
    资源身份的一部分、张量 dtype）连理由一起写在那个声明上。
  - 张量视图那几个字段（shape / strides / axis / vectorized_*）**留在 bridge**：它们不是
    资源字段，而是「同一条队列上的张量今天必然是同一个 tile」这条调度前提。
  - 新用例：`load_mask_select_store`（ui8 掩码从 GM 搬入，改动前红）、
    `load_eq_select_store`（compare → select，改动前就绿，作为那句错断言的凭据留下）。
  - 负例两条，两侧各一：bridge 的
    `SharesOneQueueAcrossDtypesButRejectsIncompatibleDescriptors`（同一个图跑三遍，只差
    第二个成员的 dtype 与 `queue_depth`：前两遍建得出模块、第三遍 `BindingConflict`），
    conversion 的 `afir_memory_allocate_shared_queue_conflict.mlir`。
  - **埋点验过两层**：把 dtype 相等判据放回 bridge，UT 的混 dtype 那一遍与
    `load_mask_select_store` 双双当场红；把 `queueDepth` 从那份共用清单里删掉，两条负例
    双双当场红——**同一处改动让两侧都红，这就是「合成一处」的凭据**。
  - 混 dtype 的**正例不写成 lit**：conversion 那一侧的判据只看 descriptor，而 descriptor
    上从来没有 dtype，所以一份混 dtype 的 lit 正例改动前也绿——是假的。理由记在那条负例
    的文件头。

### ~~L-4~~　标量只表达得出「在第二个操作数」　**已解**

- **现象**：矩阵上 8 个 op 的标量格是 `2nd + swap(2nd)`，交换之后那半张格子 MLIR 整个没有。
- **前提核过，是真的**：`ScalarBroadcastOptimizationPass`（legacy 支持这半张格子的办法，
  step7 `SwapInputIndex` 物理交换图上的两个输入）**只对 `Broadcast` 节点触发**。直接建
  一个 `Scalar` 节点的图它不管，图原样到 MLIR，ODS 当场拒收
  （`operand #0 must be Tensor or MemRef of ... but got 'f32'`）。
- **计划里有两处是错的，记下来**：
  1. **`is_scalar_latter` 不在这一条的范围内。** [binary_api_call.cpp:76](../../../codegen/api_call/elewise/binary_api_call.cpp:76)
     只对 `Div` / `Sub` / `DivExtend` / `SubExtend` 发它，其余走 `else` 分支、发的是
     **不带模板实参**的 `Adds(y, x1, (T)scalar, count)`。8 个 swap op 全在 `else` 里。
  2. **8 个 swap op 全是可交换的**（add / mul / maximum / minimum / eq / ne /
     logical_and / logical_or），这正是 legacy 只给它们覆写
     `IsScalarInputSupportedIfExchangeInputs` 而不给 lt / le / gt / ge 覆写的原因。所以
     交换方向发出来的调用**逐字相同**，发射侧要做的只是「把 tile 放到第一个实参上」。
- **落地**（换轴，不是在 bridge 里归一化——`Div` / `Sub` / `TrueDiv` 那三个 `any` 档将来
  接通时不可交换，真的需要第二种调用形状，那时这根轴已经在了）：
  - ODS：`allowScalarRhs`（一位）→ `scalarOperand`（三值字符串 `"none"` / `"rhs"` /
    `"any"`）。`AFIR_BufferizableBinaryOpBase` 多一个 `lhsType`。
  - conversion：`RhsForm{Tensor,Scalar}` → `ScalarOperand{None,Rhs,Any}`，**三个取值与 ODS
    那三个字符串一一对应**。新增 `emissionOrder(form, lhsIsTile, rhsIsTile)`：形态校验与
    发射顺序同一个函数算，capability 与 pattern 查的是同一份，「两个都是 tile / 两个都是
    标量 / 形态对不上」三种走法一次挡掉。
  - `materializeOperandTiles` 不再按模板实参分流，改收 `emissionOrder` 算好的那一份——
    pattern 为了校验形态已经算过一次，再算一遍就是第二份编码。
  - verifier：「两个都是标量」ODS 挡不住（类型约束是逐个操作数的），落在
    `verifyBufferizedElementwise` 的「至少一个输入有形状」。`inferLogicalReturnType` 里
    「第一个输入一定有形状」那句跟着改成「取有形状的那一个」——那里原先写着「等做的时候
    这里要一起改」。
  - **`registerScalarBinary` / `registerCompare` 没有增加注册点**：交换方向不是第二个
    implementation，选中的 id 仍是 `<op>.tensor_scalar`，由 SELECT 那条 check 钉住。
- **用例**：
  - lit 正例 `afir-scalar-binary-swapped-api.mlir`——8 个 op 各一个函数，三条 RUN 分别钉
    IR（tile 在前）、CPP（emitter 的实参序）、SELECT（id 与不交换时相同）。
  - `afir-scalar-binary-invalid.mlir` 末尾两条负例按判据重做：一条换成「两个都是标量」
    （verifier 负例），一条换成「`lt` 标量在左仍然拒收」（ODS 负例，钉住「可交换」这个
    前提没被顺手放宽成全部）。
  - CPU E2E `load_scalar_add_store`，与 `load_add_scalar_store` 成对：两份图只差操作数
    次序，发出来的 kernel 逐字相同。
- **三层埋点都验过**：`emissionOrder` 的 `Any` 那一档不再交换 → 交换方向的 lit 与 E2E
  双双当场红（E2E 报的是 `no supported Ascend implementation`）；去掉「至少一个输入有
  形状」→ 「两个都是标量」那条负例当场红；把 `lt` 的 `scalarOperand` 改成 `"any"` →
  第五条负例当场红。
- ~~**范围之外**：`Div` / `Sub` / `TrueDiv` / `ClipByValue` / `Pow` 一个标量形态都没接~~
  **已接通（2026-09-24）**，而且这条当初**漏了 `Select` / `Where`**：它们的
  `mem_scalar` / `reg_scalar` 是 `is_scalar_list[0] == false`，不是 `no`——后两个输入各自
  可以是标量，只有 mask 不行。七个 op 一起接的，详情见进度文档「标量形态的三种发射规则」。
  **按 legacy 的 `IsScalarInputSupported` 缩了两处范围**：`Sub` / `TrueDiv` 的 reg 侧那一格
  是 `no`，虽然 `sub_reg_base.h` 里 `SubExtends` 供给是在的——调度产不出那种图，接了是死代码。

## 阶段 D：外部依赖

### ~~L-1~~　ATT tiling 拒收含 `Scalar` 节点的图　**不存在**

**这一条是假的。** `load_add_scalar_store` 建出来一次就过，数值正确——`Adds<float, 0>`
加 `constexpr float c2_5_f32 = (float)2.500000000e+00;`，整个标量族现在有 E2E 了。

原先记的现场有两处错，都指向同一个操作失误：

1. **`codegen_func_` 不是 ATT。** 这个 harness 的 `tiling_lib_path` 指的是
   [TilingStub.cpp](../../test/e2e/cpu/support/TilingStub.cpp)，它的 `CodegenTiling`
   **无条件 `return false`**。所以任何回落到 legacy tiling 的图都会报
   `codegen_tiling.cpp:872`——那是 harness 的属性，与 `Scalar`、与 ATT 都无关。
2. **时序反了。** [codegen.cpp:399](../../../codegen/codegen.cpp:399)：`GenerateForInductor`
   第一件事就是 `RunMlirCodegen`，只有它返回 `kNotSupported` 才走到 tiling。所以真实链条
   是「MLIR 路径先谢绝，才落到那个桩上」，不是「tiling 比 bridge 早」。
3. **原先给的复现命令自带这个故障**：手工跑 generator，既没有 `AF_ENABLE_MLIR_CODEGEN=1`
   也没有 cmake 烘进去的 `LD_LIBRARY_PATH`（后者会解析到 CANN 装的那份不带
   `ENABLE_AUTOFUSE_MLIR` 的 `libaihac_codegen.so`）。缺任一个都让 MLIR 路径谢绝。
   **这条同样会让本来是绿的用例报同一行 872**，本轮先这么误判过一次
   `compare → select`。所以：**一律用 `check-e2e-cpu-<用例名>` 目标跑。**

- **埋点验过**：把 `buildScalarConstant` 解析出来的浮点值乘 2，逐元素断言当场红——
  这条链（AscGraph 的 `Scalar` 节点 → 有类型的常量 → API 实参）确实被走到。
- **纪律**：这是「文档说卡住了不是事实」的第三个实例，而且**排查工具本身的失效被记成了
  被测对象的失效**。下次写「已确认在 X 一侧」之前，先确认 X 在这个 harness 里真的是 X。

### ~~L-2~~　`isSetMask` 一律打成 `0`　**已修（`DefaultValuedAttr<BoolAttr, "true">`）**

**不需要 NPU。这条在 CPU 模拟器上就能变红**，原先探不到只是因为所有用例的形状都太小。

- **现象**（成立）：`printIsSetMaskTemplate` 给所有 VecScalar L2 op 打 `<T, isSetMask>`
  两个模板实参，而 `isSetMask` 原本是个 `UnitAttr`，没人设，于是一律打成 `0`。
- **根因不是「少设了一个属性」，是两个缺省的极性相反。** AscendC 声明上是
  `bool isSetMask = true`；MLIR 的 `UnitAttr` 缺省（属性不写）是 `false`。危险的那一侧
  正好是「什么都不写」时落到的那一侧——**这类 bug 修调用方是修不完的，要修缺省**。
- **`false` 不是坏取值，是「由谁设 mask」的分工**，这一点原先记错了。读
  [dav_c220/kernel_operator_vec_binary_scalar_impl.h] 的 Level 2 分支（m200 / c100 同形）：
  取 `true` 时 API 自己 `SetMaskCount()` + `SetMask(0, count)` + 收尾复位；取 `false` 时
  API 整段跳过，按 `repeatTimes = 1` 直发一条指令，**靠调用方事先把 mask 置成 count
  模式并设好 count**。所以 `false` 是个有用的优化——一串同 count 的 L2 调用可以把那四条
  提到循环外——只是有前置条件，而今天没有任何 pass 建立那个前置条件。
- **两个取值实测（910B / `dav_c220`，tikicpulib，128 个 f32 一趟）**：

  | 调用 | 错的元素 | 第一个错的下标 |
  |---|---|---|
  | `Adds(dst, src, s, 128)`（缺省 `true`） | 0 | — |
  | `Adds<float, 0>(...)`，mask 未设 | 64 | 64 |
  | `Adds<float, 0>(...)`，调用方先设 count mask | 0 | — |

  第三行是关键：它证明 `false` 有正当用途，所以**修法不能是把这个模板实参屏蔽掉**。
- **影响面**：`add/mul/maximum/minimum` 的 `tensor_scalar` 一族（`Adds` / `Muls` / `Maxs` /
  `Mins`）加 `leaky_relu`。reg-base（`dav_3510`）那一侧的 L2 `AddsImpl` **压根不读
  `isSetMask`**，所以这条只影响 mem 一侧——也就是默认的 `Ascend910B1`。
- **改动**：[Base.td](../../api/ascendc/include/ascir/Dialect/Asc/IR/Base.td) 的
  `VecScalarL2Op`，`UnitAttr:$isSetMask` → `DefaultValuedAttr<BoolAttr, "true">:$isSetMask`。
  一处，覆盖 9 个 VecScalar L2 op，新增 conversion 点不会漏；旋钮保留，想用那个优化的
  pass 显式写 `isSetMask = false` 即可。emitter 一个字没动。
  - **为什么不改 emitter（让 L2 不打模板实参、与 legacy 逐字一致）**：那会把 `false`
    彻底封死，上表第三行说明它有正当用途。`DuplicateL2Op` 的 ODS 里干脆没有这个参数，
    所以它的 emitter 不打是对的，**不能照抄到这里**。
  - **为什么不在 conversion 里逐处 `setAttr`**：缺省仍停在危险侧，漏一处就静默回到 `0`
    ——正是 L-14 那个形状。
  - **代价**：`Base.td` 是上游文件，这给 L-17 的上游 diff 加了 1 行（+ 注释）。L-17 做完后
    这条仍然留在那里——改的是上游 op 的 ODS，TableGen 不能 reopen class，外移不了。
- **完成判据（已满足）**：`load_add_scalar_store_128x23x5`——**修复前 14720 个元素错
  14528 个、第一个错的是下标 64；修复后全绿**。另有 7 条 lit / E2E 的文本 CHECK 从
  `<T, 0>` 改钉 `<T, 1>`，缺省被改回 `UnitAttr` 时它们会红。
- **顺带记下、不在本条范围**：`VecScalarL0Op` / `VecScalarL1Op` 的 `isSetMask` 仍是
  `UnitAttr`，同样的极性反了。今天没有任何 conversion 建这两类 op，够不着；等有人要用
  L0/L1 时一起改。

[dav_c220/kernel_operator_vec_binary_scalar_impl.h]: CANN 安装目录下
`compiler/tikcpp/tikcfw/impl/dav_c220/kernel_operator_vec_binary_scalar_impl.h`

---

## 阶段 E：补齐与加固

### ~~L-7~~　`Broadcast` 仍是断头路　**各档均已接通（2026-09-23 常量源，2026-09-29 分段与 reg 侧）**

它曾是 bridge buffer 形态白名单里唯一没有 conversion 实现的 op，含它的图整张回落 legacy。
**legacy 的 `Duplicate` 一档已接通**，两侧各一个 implementation
（`broadcast.tensor.duplicate` / `..._reg_base`，`Implementations/Broadcast.cpp`），
`BufferFormSetIsCoveredByAscendCImplementations` 的豁免名单已清空。设计与全部证据见
`2026-09-23-afir-broadcast-design.md`。

**分段那几档（含多广播段）与 reg 侧的 `BroadcastExtend` 随后也接通了**，卡点是向量化轴的
单轴假设，解法是合轴：见 `2026-09-28-afir-vectorized-axis-collapse-design.md`（那份文档里
「测试覆盖现状」一节列了各档今天的覆盖，以及还剩的两处：reg 侧没有 E2E、整块复制那一趟的上限）。

**这一节原先记的三条前置，两条不成立，「真卡点」也只说对一半。逐条订正：**

- **不成立①：「两侧供给都要现建 td def（前置 C 剩的最后一行）」。** 这一档一个 def 都不用
  建：`AscendC_DuplicateL2Op`（`Basic/OpVecDuplicate.td:37`）、`LocalTensorGetValueOp`
  （`Core/Tensor.td:252`）、`SetFlagOp` / `WaitFlagOp` 与 `MTE2_S` / `V_S` 两档枚举全是现成的。
  连三轴 / 四轴那一档要的 `AscendC_BroadcastL2Op` 也已经有
  （`Autofuse/OpVecBroadcast.td`，L-17 之前在 `Basic/OpVecBroadcastExt.td:29`）。**前置 C 里 `Broadcast` 那一行应当划掉。**
- **不成立②：「reg 侧 emitter 必须能在调用前先打两条局部 `const uint32_t[N]` 声明」。**
  `Autofuse/VecBroadcast.cpp` 的手写 emitter 已经会打。
- **不成立③：「`asc_graph_builder.h` 要加 `Broadcast` 方法」。** `:105` 起本来就有三个重载。
- **说对一半：`TransferPlan` 只建模一根向量化轴。** 属实
  （`ScheduledTileIndices` 只给 `globalOffset` + 一个 `count`），但**它只挡住三轴 / 四轴那一档**。
  一根向量化轴时 legacy 走得到的分支只有 `Duplicate` 一条：`BroadcastApiCall::Generate` 先查
  `IsBroadcastConstantTensor(x)`，它要求 x 的**每一根**向量化轴 repeat 都是 1；只有一根轴时
  「这根轴被广播」与它等价，于是必然先命中 `Duplicate`，`broadcast_num == 1` 的三轴调用永远
  够不着。**「一根向量化轴」与「`Duplicate` 一档」是同一条边界。**

**原先没记、而真正挡住 CPU E2E 的是另外四件事**（证据与探针数据见设计文档 2.4）。这一档因此
**没有 CPU E2E**——不是省略，是今天没有一条广播图跑得到执行那一步，补齐 `TransferPlan` 之后
仍然如此。四条各自比 L-7 大，应各自立项：

> **订正（2026-09-29 重测）**：`Duplicate` 那一档在 (c) 修掉之后跑得通，已补两条 CPU E2E
> （`load_broadcast_duplicate_store_128` / `_8x128`）；分段那一档仍卡在 (a)，而 (a) 的真实形态是
> **一个 stage 里模板不唯一**（`B0Y0` / `B0Y1`），不是 plan 不唯一。**(b) 不是卡点**——
> 强制选中 `B0Y0` 之后两种形态都跑通且算得对，忽略 `exec_condition` 只是漏掉一次优化。详见
> `2026-09-28-afir-vectorized-axis-collapse-design.md` 的「广播 CPU E2E 重测（2026-09-29）」
> 与其后的「遗留问题清单（2026-09-29 立）」：B-1（模板选择）、B-3（`segments.empty()`
> 静默算错）与 B-5（把 GM 偏移统一成逐轴式子）当天已修，**形态 B 因此有了两条 CPU E2E**；
> B-7（逐轴偏移的强度削减）与 B-2（`MTE2 → MTE3` 同步）
> 也已修——**形态 A 与形态 B 都有了 CPU E2E**，而既有 62 条用例的生成 kernel 与改动前逐字相同。
> B-6（多模板 `tiling_key` 重复）查清后是输入不成立、诊断不清，已补在 bridge 侧报错。清单清空。

- **(a) 计划选择：只要有 2 个以上 plan 就选不出来。** 广播是第一个把 tiling 变成真选择题的
  op（探针八种形状里七种有 2~3 个 plan，无广播的对照组恒为 1）。plan 不唯一就落到
  `SelectWinnersFromModels` 的 ATT 代价模型，而真实模型里有 ATT 求解器自己的未知量
  （埋点原文：`((15.89 * block_dim) + 882.09 + load0_Load_AIV_MTE2_contrib)`），代不进去，
  一律报 `ATT did not provide a complete, constant-cost plan for subgraph 0`。
  `AttCostEvaluatorTest` 喂的全是合成的常量 `ModelInfo`，**这条路在真实图上一次都没跑通过**。
- ~~**(b) `exec_condition` 整个没建模。**~~ 广播图留下来的 plan 的 Load 与 Broadcast 都带
  `exec_condition = cache_block_split_fused_brc_axis`；`autofuse/mlir` 全树 grep
  `exec_condition` / `ExecuteCondition` 零命中，bridge 既不读也不拒，照搬会静默丢掉这个条件。
  **已修（2026-09-30）**：它进了 `afir.sched`，守卫由新 pass `afir-exec-condition` 包成
  `scf.if`。落点与验收见 `2026-09-28-afir-vectorized-axis-collapse-design.md` 的「下一步」
  第 1 条。
- ~~**(c) 唯一能到 MLIR codegen 的那个形状，Load 发不出合法的搬运。**~~ **已修（2026-09-23）**，
  而且**范围比这一条写的大得多：它不是广播独有的，是 mem 侧每一张图的问题。**
  `AscendC::DataCopy` 要求元素数 × sizeof(T) 32B 对齐（CPU 模拟器：`count * sizeof(T) must
  be 32B align ... rounded down by 32B`），而 tile 的元素个数是运行期的 TilingData 字段，
  **尾块与整块走同一条语句**——凡是总量切不齐的图，尾块就不对齐。NPU 上不报错，按 32B
  向下取整，**尾巴静默丢掉**。原先探不到只因为所有用例的形状都是 `[128]`（f32 512B，恒对齐）；
  44 条换成 2~4 轴当场红 26 条。
  - 修法：`DataMovement.cpp` 一律发 `DataCopyPadExtend`，与 legacy 的 `load_api_call.cpp` /
    `store_api_call.cpp` 一致——那两处也没有第二条路。**不按「count 对不对齐」分流**：
    lowering 这一刻看不出运行期的 count 对不对齐。
  - `TargetFeature::DataCopyPadExtend` 整条删掉（枚举、按 arch 的回落、loader 的名字表）。
    它只有一个消费者就是这个门，而门写的「reg-base 家族才有这个 wrapper」本身是错的。
    两个只覆盖这个 feature 的 UT 改成打 `RegBasedApiFamily`——那一项此前**零直接覆盖**。
  - **原先估的代价「40 多个 mem 侧 E2E 与一批 lit 的 Load / Store 文本」是错的**：实际要改
    2 个 lit 文件（`afir-api-extension-include.mlir` 查 C++ 文本、
    `demo/afir_pipeline_sync_canonical.mlir` 查方言 op），E2E 的 `CheckGeneratedKernel`
    一个都没查。**估代价之前先 grep 一遍，别按「感觉会牵连很多」写；而且两种拼法都要搜**
    ——第一次数成 1 个，就是只 grep 了 `DataCopy` 没 grep `data_copy_l2`。
  - 正确性不靠推理：reg 侧那半边一直走的就是 `DataCopyPadExtend`，改形状之后 4 条
    reg-base 用例（`load_cast_reg_base_store_255x3x335` 等，形状比 mem 侧那些更刁钻）
    **一条没红**。
- **(d) reg 侧同样到不了**：同一张图 plan 唯一，但模板有两个（多出一个 `nddma`），
  `HasUniqueCandidatePerSubgraph` 要求每个 stage 恰好一个模板，一样落到 (a) 那条死路。

**顺带修掉的一处潜伏缺陷**：`materializeScheduledTileIndices` 原先把向量化 stride 当成恒为 1，
对 stride 0 的 tile（常量源 Load 正是这一种）算出的是带迭代偏移、元素数取 `zt_size` 的搬运——
越界且算错。只是在本次之前没有任何实现能把这种 tile 送到那里，所以一直没红。已补上 stride 0
的分支，stride 既非 0 也非 1 的从「按 1 算」改为报失败、回落 legacy。

**剩下的**：三轴 / 四轴 `Broadcast(...)` 那两档仍未接通，前置是逐轴的 `TransferPlan`；
`ui8` 这一位注册处比 legacy 窄（legacy 走 `duplicate.h` 的包装重载，不是标准
`AscendC::Duplicate`），已由 `ElementTypeMatrixTest` 的 `kKnownNarrower` 快照表钉死，
补齐它要另起一个带扩展头的 implementation。**在 (a)(c) 之前做逐轴 `TransferPlan`，
是给一条走不通的路铺砖。**

### ~~L-8~~　`MaskedFill` 只有 mem 一侧，而交叉断言是 per-op 不是 per-family　**已了结**（断言 `0fc93e1f`，`masked_fill` 删除见下）

这一条有两半，第二半比第一半重要。

- **第一半：卡点的说法是错的，而且错了两层（2026-09-23 核实）。**

  **原先的说法**：只注册了 `masked_fill.tensor.extend`（mem）；reg 侧接不上是因为 legacy 在
  那一侧给它算了一块 tmp buffer（`MaskedFillAscIrCodegenImplV2::CalcTmpBufSize`），而它继承的
  `WhereRegApiCall` 在连续 tile 形态下发 5 个实参、没有 tmp 的位置，所以要先把标量形态那套
  调用接进来。

  **这两层，进度文档早就写了。** `2026-09-17-afir-op-completion-elementwise-progress.md`
  的「MaskedFill reg 侧　**当初写的卡点是错的，而且这个 op 在正式流程里到不了**」一节
  （第 369 行起）把两条都记全了，还多一条本清单没有的：`CalcMaskedFillTmpSize` 的函数体
  就是 `return CalcSelectTmpSize(node)`，v1 基类本来就有这个尺寸函数，那个 override 从 v1
  起就是恒等的，v2 父类不再要 tmp 时被原样搬了过来——**是复制粘贴残留，不是「两侧都要
  tmp」的证据**。本条当初是照着 E5 的 commit message 写的，没有回查那一节，于是两份文档
  互相矛盾地并存了一个月。**挑一条开工前先 grep 进度文档里这个 op 的名字。**

  **第一层错**：就算前提成立，机制也是现成的。`BinaryElementwise.cpp` 的
  `TmpBufferUse` 早就有第三档 `Unused`（「调度分了、调用不收」），是 Sub / Div 顶出来的，
  比较族的 reg 侧今天也在用。三元路径的 `registerTernary` 还是个两态的
  `bool TakesTmpBuffer`——**这正是本清单纪律第 2 条「判据有两份编码时，分开写就会分开错」
  的又一例**。所以「要先接标量形态」不成立，需要的只是把那一档搬到三条路共用。

  **第二层错，而且这一层决定了结论**：`MaskedFill` 这个节点类型**到不了这条路**。
  `optimize/graph_pass/masked_fill_input_reorder_pass.cpp` 的 `MaskedFillInputReorderPass`
  无条件把每个 `MaskedFill` 节点改写成 `Select`（`RunPass` 里没有任何 guard），而它挂在
  `Optimizer::GraphPass` 上（`optimize/optimize.cpp:954`），**跑在 `GenerateTasks` /
  `AutoScheduler` 之前**，v1 / v2 两个 pass runner 都注册了它。bridge 消费的是调度结果，
  所以它看到的图里一个 `MaskedFill` 都没有。

  连带的两件事：`MaskedFillAscIrCodegenImplV2::CalcTmpBufSize` 在 codegen 期也不会被调到
  （那时节点已经是 `Select`，而 `Select` 在 v2 上用的是不覆写 tmp 的
  `WhereAscIrCodegenImplV2`）；已经注册的 mem 侧 `masked_fill.tensor.extend` 同样是死的
  ——`afir.masked_fill` 在全部 lit 用例里**一次都没出现过**。

  **所以补 reg 侧是给一条不会被执行的路写代码。** 结论不是「补另一侧」也不是「记进豁免
  表」，而是**把已有的那半边也删掉**：`AFIR_MaskedFillOp`（`Ops/Condition.td`）、bridge
  registry 那一行、mem 侧的 `masked_fill.tensor.extend` 注册，三处一起删（bridge 注释里
  写的「加一个 op 意味着三处」，删也是同样三处）。`BufferFormSetIsTheProductionCoverage`
  的计数与名单跟着从 44 改到 43。
  `Ops/Condition.td` 上留了一段说明为什么这一族没有 `masked_fill`，以及改写哪天变成有
  条件时要一起加回来的是哪三处。
  `scan_ascir_op_lowering_matrix.py` 的族划分里**保留** `MaskedFill`：那张表记的是
  legacy 的供给，与 MLIR 接不接它无关。
- **legacy 侧还剩一处，本轮不动**：进度文档那一节建议顺带删掉
  `MaskedFillAscIrCodegenImplV2::CalcTmpBufSize`（以及 v1 那个恒等的）——那块 UB 分了没人
  用。它在 `autofuse/v35/` 与 `autofuse/ascir/` 下，有自己的直接用例
  （`tests/ut/ascir/reg_func/test_reg_func_masked_fill.cpp`），而 `mlir-test.sh` 跑不到那套。
  换句话说它是另一个仓区、另一套验证，不属于本条的边界。
- **第二半（断言接不住）**：[BridgeOperationRegistryTest.cpp:92](../../test/ut/bridge/BridgeOperationRegistryTest.cpp:92)
  的 `BufferFormSetIsCoveredByAscendCImplementations` 查的是
  `registry.containsSourceOp(sourceOp)`——**只要任意一侧注册了就算通过**。于是：
  - `Broadcast`（两侧都没有）被抓住了，必须写进显式豁免名单，豁免名单本身还有反向断言；
  - `MaskedFill`（只有一侧）从这条断言下面**整个漏过去**，没有任何东西记录它是半连通的。

  危害不是今天的 `masked_fill`（文档写了），而是**这条断言的语义比它宣称的弱**：下一个 op
  只注册了一侧（很容易发生——两侧就是两行 `registerXxx`，漏一行编译照过）不会有任何用例变红，
  而它在另一半目标上就是不可用。
- **完成判据**：
  1. ~~断言升级成 per-family~~ **已做**。`elementwise::recordRegisteredFamily` 在六个
     `registerXxx` 叶子函数里各记一行 `{afir op name, stable id, ApiFamily}`，新断言
     `EveryElementwiseOpCoversBothApiFamilies` 据此要求每个 op 要么有 `Any` / `Shared`，
     要么 mem / reg 各至少一个；`masked_fill` 进显式的 `{op → 缺哪一侧 + 理由}` 表，表本身
     有反向断言。旁表记在注册函数里而不是 `checkFamilyMatchesHeader` 里：后者只拿得到
     `TargetOp`，拿不到 afir op 名与 stable id。
  2. ~~`masked_fill` 的 reg 侧接通，或者以「缺 reg 侧」的身份进那张表~~ **改为：两侧都
     不留，整个 op 从这条路上删除**。见上面第一半。豁免表本身留着（今天是空的），它是
     这条断言唯一的出口。
- **留给下一个人的两件事**（都不在本条范围内，不要顺手做）：
  - 三元路径的 `bool TakesTmpBuffer` 与二元路径的三态 `TmpBufferUse` 仍然是两份编码。
    今天一元 + 三元里没有任何 op 落在 `Unused` 那一档（拿
    `ascir_op_lowering_matrix.csv` 的 `*_tmp_func` 列逐个对过，只有 MaskedFill 的 reg 侧
    一行，而它到不了），所以合并它属于没有需求的改动；第一个真的落在那一档的三元 op
    出现时再合。
  - ~~`afir.masked_fill` 的 bridge 行与 mem 侧注册今天都是死代码~~ **已删**，见上。
- **两条断言是接力**：一侧都没有的 op 根本不出现在旁表里，由
  `BufferFormSetIsCoveredByAscendCImplementations` 的豁免名单接住；一侧有一侧没有的，
  由新断言接住。
- **已知的剩余粒度问题**：新断言是 per-op 而不是 per-form，所以
  `logical_or.tensor_scalar` 只有 mem 侧这件事它查不出来（`logical_or` 的 tensor-tensor
  形态两侧都在）。那一条是 legacy 判据本身就不让 reg 侧选中，理由写在注册处。
- **埋点验过**：删掉 `neg` 的 reg 侧注册后，**只有**新断言红，
  `BufferFormSetIsCoveredByAscendCImplementations` 照样绿——这正是本条说的那个洞。
- **依赖**：第 1 条无依赖（已做）；第 2 条压在标量形态那套调用（与 L-4 同一批机制）之后。

### ~~L-9　E3 剩余档：`ScalarData` / `IndexExpr`~~　**两档均已接通（2026-09-23）**

这两个是「非 op」，不是「再接一个 op」。~~另有第四档 Sub / Div 的标量形态（走
`scalar_sub.h` / `scalar_div.h`，只有 mem 侧有），压在 L-4 之后。~~ **第四档已接通
（2026-09-24）**，而且不只 Sub / Div：七个 op 一起做的，见 L-4 的「范围之外」。
**详情**：进度文档「E3 剩余三档（按依赖顺序）」。

两档今天都会在 bridge 被干净地拒掉（`AppendTemplateNode` 那条「Schedule buffer-form
bridge does not support node ... of type ...」），不是静默错值。

#### ~~`ScalarData`：通道不是 `afir.tiling_param`，是 kernel 形参~~　**已接通（2026-09-23）**

**落地（2026-09-23）。** 改动地图，每行一处读点：

| 环节 | 落点 | 做了什么 |
|---|---|---|
| ABI 词表 | `AFIREnums.td` | `#afir.kernel_arg` 加 `scalar_input`（值 4，**追加在末尾**：`underlyingType` 是 `uint8_t`，插中间会改动已有档的编码） |
| 身份 / 键 / 排序 | `AscendCKernelABI.{h,cpp}` | `KernelArgumentKind::ScalarInput`；`getKernelArgumentKey` 与 `Input` **同键**、`getKernelArgumentPermutation` 与 `Input` **同秩** |
| 三道闸门 | `AscendCKernelABI.cpp` ×2、`AFIRVerifierUtils.cpp` | 按角色放行，各配反向判据；`arg_descs` 那处复用现成的 `allowsScalars` 档 |
| 资源绑定 / 签名物化 | `CanonicalBindings.cpp`、`AFIRKernelBodySignature.cpp` | 标量跳过 `buffer_desc`；`cast<MemRefType>` 改成有条件转换（原先遇到标量是 assert 失败，不是诊断） |
| bridge | `ScheduleToAfir.cpp`、`AfirBuilder.{h,cpp}` | `IoTensor` 加 `by_value`；形参类型取元素类型；`arg_descs` 占一格退化视图 |
| entry / launch | `GenKernelEntry.cpp`、`GenAggregate*.cpp`、`AutofuseLaunchEmitter.cpp` | 角色原样写回；host 签名按值、调用处不 cast |

三条当时定下、值得留着的判断：

1. **同键同秩是 C2 的唯一编码点。** `Data` 与 `ScalarData` 共用一个 `index` 空间，所以
   同 index 上出现两者是冲突（报 `duplicate argument identity input:N`），而排序分秩会把
   标量整体排到指针形参之后、与 legacy 的交错顺序对不上。
2. **新增角色而不是按类型（非 `ShapedType`）反推。** 两条路都能放行标量，差别在失败形态：
   按类型反推时，一个**本该是 GM 张量**的参数被写成标量类型会被静默收下，而没有任何用例
   会红——`KernelBodySignatureTest` 的 `RejectsScalarDataArgumentOnAlreadyMaterializedAbi`
   钉的正是这种形状。代价是多改 4 处 `switch`，但加枚举档编译器会逐个指出来。
3. **C 类型名不要引入第三张表。** legacy 的 `DtypeName`（`common_utils.cpp:1247`）与 MLIR 的
   `CodeEmitter::emitIntegerType` / `emitFloatType` 对涉及的每种 dtype 给出的字符串**已经
   逐字相同**，`emitCType` 直接用即可。

**conversion 层一个字没改**——二元路径判标量只看类型（`emissionOrder` 的入参是
`isa<ascendc::LocalTensorType>`），块参数与 `arith.constant` 同路。发出来的是
`AscendC::Adds<float, 1>(dst, src, v2, count)`，`v2` 就是那个块参数。

**判据**：CPU E2E `load_add_scalar_data_store`（形参类型序列 + launch 签名 + 数值）、
lit `afir-scalar-input-kernel-arg.mlir`、bridge UT 两条、conversion UT 四条。
**五处埋点逐条验过**（去掉折秩 / 标量当 tile / launch 不分流 / 删两条反向判据 / 键不共用），
其中 launch 那处在 E2E 与 lit 上各验一次。

**原先记的卡点里有一条自己也错了**：说「单标量的 E2E 测不出排序回归、必须要标量夹在
两个 `Data` 中间」——不成立。`ScalarInput` 的枚举值追加在末尾（值 4），比 `Output` /
`Workspace` 都大，不折秩时单标量也会被排到它们之后，E2E 当场红（实测，参数序列变成
`{ptr, ptr, u8*, float, tiling}`）。「夹在中间」那条形状仍然补了一条 UT，但理由是
**照出标量与指针之间按 index 排先后**，不是「E2E 照不出」。

下面是接通前的调研记录，留作依据。

- **原先的说法**：「通道是现成的 `afir.tiling_param "<字段名>"`，缺的是 AscGraph 的 index
  到 tiling 字段名的对应关系。」**不成立。**
- `REG_ASC_IR(ScalarData)`（`ascir/generator/ascir_builtin_ops_v1.cpp:85`）只有一个属性
  `Attr<int64_t>("index")`，与 `Data` 的 `index` **是同一个索引空间**。legacy 的
  `codegen/codegen_kernel.cpp:2196` 起把 `Data` 与 `ScalarData` 放在同一个循环里按 `index`
  排 kernel 形参：`Data` 排成 `GM_ADDR`，`ScalarData` 排成
  `Variable(Type(dtype_name), input_name)`——**一个按值传的标量形参**，不是 TilingData 字段。
- 而 `afir.tiling_param` 读的是 TilingData 结构体的字段（降下去是 `tiling.q0_size`，见
  `Ops/Tiling.td`）。两者是两套 ABI，host 侧不一样。改走 tiling 会让 launch 签名与 legacy
  不一致，而 `CodegenOrchestrationTest.AutofuseLaunchSignatureMatchesLegacy` 正钉着这一条。
- **所以工作量不是一张名字对照表，是一次跨四处的 ABI 改动**：bridge 要按 `index` 建一个
  非 memref 的 kernel 形参，并在 `handleMarkerNode` 里加一条 `ScalarData` 分支（照 `Data`
  那条）；kernel body 签名这一关**今天是明确拒绝的**——
  `test/ut/conversion/KernelBodySignatureTest.cpp:192` 的
  `RejectsScalarDataArgumentOnAlreadyMaterializedAbi` 钉的就是「非 memref 形参一律报
  must be a GM memref」；emitter 与 launch 两侧都要按 legacy 的排法发出那个标量形参。

#### ~~`IndexExpr`：原说法成立~~　**常量形态已接通（2026-09-23）；原说法不成立**

- **「要在 `LoopGenerate` 之后才有归纳变量」这条是错的**，三条证据：
  1. legacy 把 `IndexExpr` 与 `Scalar` 收进**同一张** `kernel.constant_tensors`
     （`codegen_kernel.cpp:2247`），由 `AppendConstTensorInit` 一起发；
  2. 渲染用的 `Tiler::Size`（`:816`）只拿 `sizes` 做替换，而 `AddSizeVar`（`:694`）
     往里放的只有 `s0 → t->s0` 这类 TilingData 字段——**轴进的是 `axis_map`，不在替换
     表里**；
  3. 发射位置在循环之前：`codegen_kernel.cpp:2436` 起是 `GlobalTensorInit` →
     `LocalTensorQueBufAlloc` → **然后才是** `root_loop.Generate`。

  **一个归纳变量都用不上。** 它就是「值写成 size var 表达式的 `Scalar`」，落点与
  `Scalar` 同一处。实测也证实了：`IndexExpr → Add` 的图调度得出来，六个节点
  `Data / IndexExpr / Load / Add / Store / Output`，没有插入任何东西。
- **ODS 不用改。** 这条路上不产生 `afir.index_expr`——就像 `Scalar` 不产生
  `afir.scalar`、而是折成 `arith.constant` 一样。`AFIR_IndexExprOp`（`Ops/Data.td:29`）
  一个属性都没有、因而作为 DSL 构造表达不了自己，这是**既有的**缺口，不由这条引入，
  也不在这条的范围内。
- **落地**：两种形态都接通了，由表达式里有没有自由符号分流。

  | 形态 | 走哪条 | 用例 |
  |---|---|---|
  | 常量（`"2"`） | 直接建一个有类型的常量，与 `Scalar` 完全同路 | `load_add_index_expr_store` |
  | 带自由符号（`"2 * s0"`） | `materializeRuntimeSize`：符号取成 `afir.tiling_param`、表达式成为 `afir.symbolic_eval`，落在**入口块的尺寸序言**里 | `load_add_index_expr_dynamic_store` |

  两种形态**都不占 kernel 形参**——这是它与 `ScalarData` 的分界。
- 符号形态算出来的是 `index`，而它要当元素类型的标量用，所以跟着一次转换：整数一次
  `index_cast`，浮点再跟一次 `sitofp`（`sitofp` 不收 `index`）。取 i64 中转而不是 i32，
  免得表达式的值超出 i32 时静默截断。legacy 那边是 `constexpr <dtype> k = <整数表达式>;`
  靠 C++ 的隐式转换，这里显式写出来。
- **这是套件里第一条动态形状 E2E。** 此前 43 条用例的形状全是字面量
  （`AscGraphBuilder::Loops` 把 `128` 建成 `Symbol(128)`，一个 size var 都不产生），
  所以顺带补了两处：`AscGraphBuilder::SizeVar` 登记运行期尺寸符号；E2E 的
  `AutofuseE2eInitTiling` 包装能带运行期尺寸实参（`g_e2eRuntimeShapeArgs`）——动态形状的
  `AutofuseTiling` 会在 `tiling` 之前多出每个 size var 一个 `uint32_t` 形参，少传一个是
  签名对不上、编译报错，不会悄悄算错。静态用例那张表是空的，拼出来与从前逐字相同。
- **判据**：两条 CPU E2E、lit `afir-index-expr-scalar.mlir`（钉运行期读字段 + 两次转换
  相邻两行）、bridge UT 两条。三处埋点验过：折秩不适用，验的是**把符号表达式折成常量**
  ——E2E、lit、bridge UT 三层各自变红。

#### 完成判据（订正）

- 两档**各自先要一个结论**，写清楚接入点在哪：`ScalarData` 是 kernel 形参 ABI 怎么跨
  bridge / 签名 / emitter / launch 对齐；`IndexExpr` 是折叠放进 `LoopGenerate` 之后的哪个
  pass、`afir.index_expr` 的 ODS 怎么承载 `expr`。
- 然后才是各自的 bridge UT + lit + CPU E2E。**不要把两档放进同一条提交**：它们只是都叫
  「标量源」，接入点完全不同。

### ~~L-14~~　「忘了声明 `ApiExtensionOpInterface`」这条判据防不住　**已修（`0fc93e1f`）**

**先说结论：这不是忘了加，放行是有意的；漏的是没人写下来的那半边。**

- `headersMatchFamily`（[ElementwiseApiFamily.h:158](../../conversion/AFIRToAscendC/Implementations/ElementwiseApiFamily.h:158)）
  在扩展头与前导头**两个列表都空**时无条件返回 true，不看 family。这个分支从最早的
  `headerMatchesFamily` 就在，注释写着「空名字（标准调用、无前导）对任何家族都成立」
  ——它是为了让 `abs.tensor`（`RegBased` + 标准 `AscendC::Abs`）这类过得去，**必需**。
  今天这样的 implementation 有 12 个，我逐个对过 `ascir_op_lowering_matrix.csv`，
  **每一个在那一侧的 `*_headers` 列都确实是空的**：
  `abs` `reciprocal` `sigmoid` `isnan` `isfinite` `isinf` `rsqrt` `bitwise_and` 的 reg 侧，
  `tanh` `erf` `sub` `div` 的 mem 侧（`sub`/`div` 那两个 `scalar_*.h` 是**标量形态**的头，
  与 tensor-tensor 这条路无关）。**没有一个 op 今天是错的。**
- **漏的是**：「ODS 忘了写 `ApiExtensionOpInterface`」的表现与上面 12 个**完全一样**
  ——`ApiExtensionHeaderOf` 的 SFINAE 退回主模板、返回空、判据放行、`AddKernelIncludes`
  一个 include 都不发，最后炸在 device 编译器上。而注册表那条「对不上就整张建不起来」的
  保护，恰恰在这个方向上是哑的。设计文档 E0-2 / E2 讲这条判据时讲的是「写反了」，
  没讲「漏写了」。
- **所以这是加固，不是修 bug**，放在阶段 E。
- **完成判据**：注册处能说出「我确实不带头」这句话，且说错要当场失败。
- **实际做法与原提议不同。** 原提议是给 `registerUnary` / `registerBinary` /
  `registerTernary` 加 `bool StockCall = false` 模板实参。那三个模板的实参列表前面已经有
  `TakesTmpBuffer` / `MaterializesReuseSource` / `ConstBlock`，而 12 个 stock 注册点用的全是
  默认值，追加尾部模板实参意味着每处要把前面几个默认值手写出来。改成按 `TargetOp` 类型的
  一张名单：`ElementwiseStockCalls.h` 列出**全部 26 个**标准调用（12 个单侧的，加 14 个
  家族恒为 `Any` 的），`checkFamilyMatchesHeader` 要求
  `headers.empty() == kIsStockCall<TargetOp>`。判据等价、两个方向都会红，零注册点改动，
  而且 26 条能一眼扫完——本条当初也是把它们当一组对着 CSV 核的。
- **埋点验过（两个方向）**：摘掉 `AscendC_TrueDivExtendOp` 的 `ApiExtensionOpInterface`
  后，**不加这条判据时 `ScheduleToAfirTest` 的 69 个用例全绿**（这就是本条说的「哑」），
  加上判据后注册表建不起来、5 个用例红。挑 `TrueDivExtendOp` 而不是 `AbsExtendOp` 是因为
  后者在 `CodegenOrchestrationTest` 那张手抄名单上，会先在编译期被拦下，测不到这条判据。

### L-5　常量 block 的预留与分配分在两个目录

- **形状**：分配侧与预留侧分在两个目录，两边没有共享断言——与 `AGENTS.md` 3.1 记的那条
  NPU 0x507035 越界同一类。今天两边的数相等，靠的是两边**各自**实现了同一条
  「按 (值, dtype) 去重」的规则：
  - ATT：`common_utils.cpp` 的 `GetApiExtractDupSet` / `api_extract_dup_map`
    （`{LogicalNot → {1, half}}`、`{Rsqrt → {1, float}}`），按**节点类型**查，进
    `tuning_space_->builtin_tmp_buffer`，再由 `buf_occupy_expr.cpp:128` 算进 UB 占用；
  - MLIR：[Implementations/ConstBlockTensor.cpp](../../conversion/AFIRToAscendC/Implementations/ConstBlockTensor.cpp)
    按 `ascendc.const_block` 这条认领标记去重，每块 32 字节，**只在 mem 侧造**。
- **今天不会错**：ATT 按节点类型恒预留、MLIR 按选中的实现才分配，所以
  「MLIR 分配 ≤ ATT 预留」恒成立（reg 侧 ATT 白留 32 字节，无害）。
- **会失效的那一天很具体**：给 `ConstBlockOnes` 加一个枚举值（比如某个包装要一块填 0 的
  half），MLIR 多分配 32 字节而 ATT 那张表里没有这一行。失败是运行期 UB 越界，不是编译
  报错，**也不会有任何用例变红**。
- **本轮评审补的一条**：MLIR 的去重是 **per-`func.func`** 的 IR walk，ATT 的是
  **per-graph**。今天一个 kernel 一个 func 所以相等；多 func 的那天 MLIR 会多分配。
- **现状**：`ConstBlockOnes` 的枚举上已经写了「加枚举值之前先看 legacy 的
  `api_extract_dup_map`」。**本轮不改这个做法。**
- **完成判据**（真要解开时，两条路选一）：ATT 把这个数发成一个 TilingData 字段供 MLIR
  引用（今天它刻意不发，见 `tuning_space.h:261`），或者两边共用一张表。

### ~~L-12~~　`logicalResultElementType` 整条轴零覆盖　**已补（`0fc93e1f`）**

- 它只在 `inferReturnTypes` 里、且只在「没有 destination」那一支被读；而 tensor 形态的 op
  两条建法都不走那一支——assembly 里结果类型是写出来的，`AfirBuilder` 的
  `createLogicalNodeOp` 传的是 `TypeRange{resultTy}`。
- 于是逻辑族的 `ui8` 与 `truediv` 的 `trueDivResultElementType` 都是**写对了但没人验**。
  **这不是 TrueDiv 带进来的**，是这条轴从逻辑族起就这样；**比较族这一轮又给它加了六个
  用户**（`AFIR_CompareOp` 里那一行 `ui8`），换基类把原先六份手写的
  `inferReturnTypeComponents` 删掉了，这条轴上的代码从「六份手写」变成「一行 ODS」，
  **但覆盖没变，还是零**。
- `truediv` 的 lowering 那一侧不在此列——`SameOrF32ForI32` 有
  `afir-truediv-extension-api.mlir` 的 i32 函数钉着，改回 `Same` 当场红。
- **完成判据**：直接调 `TrueDivOp::inferReturnTypes` / `EqOp::inferReturnTypes` 的 UT
  （不给 destination），把 `logicalResultElementType` 改错当场红。
- **已补**：`test/ut/dialect/AFIRElementwiseContractTest.cpp` 四条。拿 op 自己的操作数 /
  属性 / properties 去调静态的 `inferReturnTypes`，**跳过 assembly 里写着的那个结果类型**
  ——这是让「没有 destination」那一支唯一走得到的办法。
- 四条里两条是对照组，缺了它们判据是虚的：`truediv` 的 f32 输入必须推出 f32（否则
  `trueDivResultElementType` 写成恒返回 f32 也能过），`add` 必须推出输入的元素类型
  （否则把传空的那一支也改成 `ui8` 不会红）。
- **埋点验过**：把 `trueDivResultElementType` 改成恒取输入类型、把 `AFIR_CompareOp` 的
  `ui8` 改成传空，各自对应的用例红，两条对照组不红。

### ~~L-3~~　两支标量分支没有用例到得了　**已解（`0fc93e1f`）**

- **第一支已解**：`materializeOperandTiles` 的 `ScheduledContiguous` 标量分支（两块 tile
  加块内偏移、标量不加）现在由 `load_add_scalar_store` 走到，埋点验过（见 L-1）。
- **第二支已补**：`AFIRElementwiseContractTest.ScalarOperandCarriesNoReadEffect` 直接断言
  buffer 形态的 effect 列表——tile 一个 `Read`、目的地一个 `Write`、标量一条都没有。个数
  一起钉住：只查「标量没有 effect」的话，把整个循环删掉也能过。埋点验过：删掉那条
  `contributesShape` 判据，该用例红。下面是当初的分析，留作记录。
- **第二支原先没有**：`getBufferizedElementwiseEffects` 里「标量不挂 `Read` effect」。
  把那条 `contributesShape` 判据删掉（标量也挂上 Read），`load_add_scalar_store`、
  **全部 156 条 lit、AFIR 方言 UT 全都照样绿**——已实测。
- **难点在哪**：这一支的可观测后果是访存排序，而标量是 `arith.constant`（`Pure`），多挂
  一个 Read effect 在今天这些图上不改变任何调度结果。要钉住它得找一张「标量的位置会影响
  重排」的图，或者退一步在 UT 里直接断言 effect 列表。**后者更实在**，也就一条 UT。
- **依赖**：无（L-1 已确认不存在）。

---

## 阶段 F：可维护性（2026-09-22 第二轮评审新增）

前面 L-1～L-16 都是「这一处有问题」。这一阶段的四条是「照现在这个做法走下去，下一批 op
会越来越贵」，性质不同，所以单列一个阶段。**它们不阻塞任何 op 的接通**，但越晚做代价越大。

### ~~L-17~~　`api/ascendc` 的定制改动收进独立子树　**已解（2026-09-24）**

- **已做**：定制内容整体移进三个 `Autofuse/` 子目录
  （`include/ascir/Dialect/Asc/IR/Autofuse/`、`include/ascir/Target/Asc/Autofuse/`、
  `lib/Target/AscendC/Autofuse/`）。子树的布局、接缝清单和「补一个 op 要改哪几行」
  写在 `api/ascendc/README.md` 第 4 节，不在这里重复。
- **基线的说法要订正。** 下面写的「16 个上游文件、+1030 行」量的是 `fe96b79d..HEAD`，
  也就是**上一次同步之后**新加的那一截；同步那一次保留下来的定制没算进去。拿本地
  上游 `b44c97b` 原件逐文件对，实际是 **50 个文件有差异**，其中 14 个纯粹是 CRLF
  （上游是 CRLF，本仓是 LF），扣掉换行和行尾空白后 **36 个文件、约 1500 行**。
- **结果**：上游文件的差异从约 1500 行降到 **217 行**，其中
  - **71 行是拷贝方案本身**（`bin/CMakeLists.txt` 去掉 `ascir-opt` / `ascir-lsp`，
    `lib/Target/AscendC/CMakeLists.txt` 加 `MLIRLLVMDialect`）；
  - **95 行是 LLVM 版本适配**（`applyPatternsAndFoldGreedily` → `applyPatternsGreedily`、
    `builder.create<X>` → `X::create(builder, ...)`、`TableGenMain` 多一个实参、
    删掉会触发 `-Wunused-variable` 的 `auto& os` 等）——**这些本来就不是「定制」**，
    同步时按上游新写法取舍，搬不走也不该搬；
  - **51 行才是 autofuse 的接缝**，见下。
  原先的完成判据「上游文件的改动行数 ≤ 5」是照着那份不完整的基线写的，做不到也不该做到：
  上面前两类占 166 行，与本条要解决的问题无关。
- **接缝比预估的三处多，而且第 3 处写错了。** 实际是 6 个文件 10 行的「挂接」加 3 处
  「搬不走」：
  - 挂接：`IR/Ops.td`（1 行 include）、`IR/Core/Attributes.td`（1 行 include，枚举）、
    `IR/CMakeLists.txt`（3 行，给 interface 开自己的 tablegen 入口）、
    `IR/Asc.h` 与 `lib/Dialect/Asc/IR/OpInterfaces.cpp`（各 1 行 include 生成物）、
    `Translation.cpp`（include + `printModulePrelude` + 第二个 `addCases`）、
    `External/Arith.cpp`（include + 1 行改用 `getResultBitWidth`）。
  - **interface 不能像 op 那样靠 include 搭车**：`-gen-op-interface-decls`
    只发**主文件里定义**的 interface（`OpInterfacesGen.cpp` 的
    `getAllInterfaceDefinitions` 按 `getMainFileID()` 显式过滤），所以
    `Autofuse/Interfaces.td` 必须是自己一个 tablegen 入口——上面那 5 行由此而来。
    op（`-gen-op-decls`）和枚举（`-gen-enum-decls`）没有这个过滤，各 1 行 include 就够。
  - 原第 3 处「给**上游已有的 op** 挂 `ApiExtensionOpInterface`（今天只有
    `DataCopyPadExtendOp`）」**不成立**：`DataCopyPadExtendOp` 是 autofuse 自己加的 def，
    跟着 def 一起搬走了。真正搬不走的是另外三处——`IR/Base.td` 的
    `VecScalarL2Op::isSetMask` 缺省值（改的是**上游 op** 的 ODS，TableGen 不能 reopen
    class）、`CodeEmitter.h` 的两个 `getOrCreateName` 重载（成员函数，生成的 emitter
    按 `emitter.` 调）、`bin/ascir-translate.cpp` 注册 affine / bufferization。
- **`BitwiseAndL2Op` 那条预判是对的**：在 `Autofuse/VecBinary.h` 写一个
  `LogicalResultForT<Op, BitwiseAndL2Op>` 的重载，上游那个模板对它 SFINAE 失败，不二义，
  上游头文件的类型列表恢复原样。
- **顺带修掉一处没人知道的失真。** `Base.td` 里 autofuse 加的 `ApiExtensionHeader` 带一个
  `code decl = [{ ... }]`，`scan_pyasc_coverage.py` 的括号配平在那里断掉，于是 `Base.td`
  后面 35 个 multiclass 全认不出来，`defm Abs : UnaryL012Op<...>` 只记成一行
  `ascendc.abs` 而不是 `abs_l0/l1/l2`。定制搬走后 `Base.td` 恢复上游原样，快照从 581 行
  涨到 654 行。表现是「少几十行」不是报错，所以此前没人发现。
- **7 个从未被 include 的 `*Ext.td` 已删**（395 行、41 个 def，自最初集成起没有一个被展开过）。
  `check_ascendc_td_reachable.py` 的豁免表因此只剩 `Doc.td`。连带影响：`concat` family 归零
  ——原先记的 18 个 data move op 全部来自 `OpVecDataMoveExt.td`，量的是死文本；
  `pyasc_ascendc_op_inventory.csv` 从 654 行回到 613 行。
- **验证**：`mlir-test.sh` 全绿（check-ut / check-lit / check-st / check-e2e，169 条 lit 100%）；
  `check_ascendc_td_reachable.py`、`pyasc-op-inventory` 两条门禁通过。

下面是当初的分析，留作记录。


- **现状与它自己的约定冲突。** 目录 README 写着「已拷贝（原始目录结构、**文件内容保持
  不变**，方便未来对照上游 diff）」，而从上一次同步（`fe96b79d`，上游 `b44c97b`）到现在，
  16 个上游文件被改过，共 +1030 行。最集中的一个已经反客为主：

  | 文件 | 全文件行数 | 其中 autofuse 新增 | 占比 |
  |---|---|---|---|
  | `Basic/OpVecBinary.td` | 386 | 345 | **89%** |
  | `Basic/OpVecUnary.td` | 214 | 129 | 60% |
  | `Basic/OpVecVconv.td` | 134 | 42 | 31% |
  | `Basic/OpVecCmpsel.td` | 289 | 80 | 28% |
  | `Target/Asc/Basic/VecBinary.h` | 245 | 123 | 50% |

  上一次全量同步已经手工解了 **13 处冲突**；按现在的增速，下一次会显著更多。
- **可行性（已查证，成本比预期低）**：把定制内容整体挪进 `Autofuse/` 子树后，
  留在上游文件里的接缝只有三处：

  1. `IR/Ops.td` 末尾一行 `include "Autofuse/Ops.td"`（在 `Base.td` 之后即可；
     `Base.td` 自己 include 了 `Interfaces.td`，依赖满足）；
  2. `lib/Target/AscendC/Translation.cpp` 两行：一行 `#include`，一行
     `addCases(typeSwitch, autofuse::PrintableOpTypes{}, callback);`
     ——**不用动那个巨型 tuple**，`addCases` 本来就可以调两次；
  3. 给**上游已有的 op** 挂 `ApiExtensionOpInterface` 的那一处
     （今天只有 `DataCopyPadExtendOp`）。tablegen 不能 reopen def，这一行无法外移。

  **CMake 零改动**：`lib/Target/AscendC/CMakeLists.txt` 已经是 `GLOB_RECURSE *.cpp`。
- **上游 op 的 emitter 也不必再改上游文件**：`BitwiseAndL2Op` 这类「上游有 def、没有
  emitter」的情况，今天是往上游 `printOperation` 的 `LogicalResultForT<...>` 列表里加一个
  类型；改成在 `Autofuse/` 自己写一个 `LogicalResultForT<Op, BitwiseAndL2Op>` 的重载即可
  ——上游那个模板对它 SFINAE 失败，只有我们这个可行，不会二义。
- **一个额外好处**：我们的 def 和上游的 def 共用同一个 def 名字空间，上游哪天新增同名
  `AscendC_PowOp`，tablegen 会直接报 `def already exists`。这是**响亮**的失败，比今天
  「同步时靠人眼在 diff 里认出来」强。
- **完成判据**：`git diff <上游基线> -- api/ascendc` 里，上游文件的改动行数 ≤ 5；
  `mlir-test.sh` 全绿；README 的「文件内容保持不变」那句话重新成立，并补一节说明
  `Autofuse/` 子树与三处接缝。
- **代价**：约 1030 行的机械搬移 + 10 个新文件 + 3 处接缝，主要风险在 td include 次序与
  emitter 重载解析，都能被现有用例接住。**建议排在下一次上游同步之前**，否则那次同步要
  先付一遍冲突成本再付搬移成本。

### ~~L-18~~　元素类型的校验分散在三到四处，而且**唯一的真源不可机读**　**已解（2026-10-27 核）**

- **今天一个 op 的 dtype 事实要写几遍**（以两侧收不同集合的 `floor_div` 为例）：

  | # | 位置 | 写的是什么 |
  |---|---|---|
  | 1 | AFIR ODS 的 `elementTypes` / `resultTypes` 列表 | 两侧的**并集**（AFIR 能不能表达） |
  | 2 | AFIR ODS 的 `sameElementType` 位 | 绑哪一组 `AllElementTypesMatch` |
  | 3 | AFIR ODS 的 `logicalResultElementType` | 结果 dtype（结果与输入不同的族才写） |
  | 4 | conversion 注册处的 `elementTypes` 位掩码 | **每侧各一份**（这一侧供不供得起） |
  | 5 | conversion 的 `ResultElementType` 档 | 结果与输入的关系（`Same` / `Uint8*` / `CastPair` / `SameOrF32ForI32`） |
  | 6 | `Ops/Cast.cpp` 的对表 | 只有 `cast` / `logical_not` 这种「逐行对表」的 op |

  AscendC 方言那一层**不参与**——`AscendC_LocalTensor` 不约束元素类型
  （`Core/Types.td:71`），emitter 只拿元素类型去填模板实参，也不校验。这一点是对的，
  不要往那边加。
- **哪些是合理的**：#1 与 #4 说的不是同一件事（「AFIR 能不能表达」vs「这一侧供不供得起」），
  分开写有道理；但 #1 **可以由 #4 机械求并集**得到，现在是手写第三遍。
- **哪些是真重复**：#2 / #3 / #5 说的都是「结果 dtype 与输入的关系」，分在 ODS 两处和
  conversion 一处。`trueDivResultElementType` 已经是「一条规则两个读点共用一个函数」的
  范本，但只有 truediv 用上了；比较族、逻辑族、谓词族、cast、logical_not 仍然是各写各的。
  **L-12 说这条轴覆盖为零，正是这种重复没人验的直接后果。**
- **最要紧的一条：dtype 是这条流水线上唯一没有机器真源的事实。**
  `ascir_op_lowering_matrix.csv` 有 API 名、扩展头、tmp 函数、标量位置四列，**唯独没有
  dtype 列**。于是每个 op 的两侧集合都是人读 `REG_ASC_IR` 抄出来的，抄错不会有任何用例
  变红——`floor_div` 那一轮「订正它的元素类型集合」、比较族那一轮「原先六行 ODS 里有四行
  是错的」，都是这么来的，而且都是**事后**发现。
- **位掩码只有 6 个位**（`kF16/kF32/kI16/kI32/kUI8/kUI16`），表达不了 bf16 / i8 / i64 /
  ui32 / ui64 / bool。代码里因此散着十几处「AFIR 建模不出来」的注释。加一种 dtype 要改
  枚举并复核**所有**注册点。
- **完成判据**（按性价比排序，做到第 1 条收益最大）：
  1. ~~`scan_ascir_op_lowering_matrix.py` 多抽两列 `mem_dtypes` / `reg_dtypes`……~~
     **已做**，见下面「第 1 条的落地」。
  2. ~~「结果 dtype 与输入的关系」收成一个共享函数/枚举……~~ **已做（2026-09-23）**。
     `AFIRElementwiseSupport.h` 的 `ResultElementRule`（`Same` / `Uint8` /
     `SameOrF32ForI32` / `CastPair`）加 `inferResultElementType` /
     `matchesResultElementRule`，ODS 与 conversion 共用它。实测收掉的：

     | | 之前 | 之后 |
     |---|---|---|
     | ODS 逐 op 手写的结果类型代码块 | 4 | **0** |
     | conversion 里的本地 `ResultElementType` 枚举 | 2 | **0** |
     | ODS 传 `sameElementType` 的 def | 10 | **0** |

     `sameElementType` 那一位不再单独传，由 `!eq(resultRule, "Same")` 推出来——两者本来
     就是同一件事的两种说法（每个传 `0` 的 def 恰好都是非 `Same` 规则）。
     `Uint8Predicate` 与 `Uint8Logical` 也合并成一档 `Uint8`：两者说的结果 dtype 是同一个,
     差别只在输入怎么查，而谓词族现在显式注册 `kFloatElementTypes`，那条写死的判据成了
     同一个答案的第二份编码。三元那条路仍用 `bit sameElementType`——它的 op 全是 `Same`,
     没有规则可收。
     **埋点验过**：把共用规则里的 `ui8` 改成 `i8`，一处改动同时打红 1 个 dialect UT
     （ODS 推导那个读点）与 10 条 lit（conversion capability 那个读点）。
     L-12 的 UT 在 `0fc93e1f` 已补。
  3. ~~ODS 的并集由两侧集合生成，而不是手写第三遍~~
     **这条写反了方向，2026-09-23 改写**：应该是**断言，不是推导**——保留手写的 ODS，
     加一条「**每一侧的注册集合 ⊆ ODS**」。四条理由：

     1. **推导会把依赖方向倒过来。** ODS 在 `dialect/AFIR`，注册处在
        `conversion/AFIRToAscendC`。让 IR 的合法性由某个后端的实现表推出来，等于让方言
        依赖 lowering，而 `check_afir_layering.py` 挡的正是这个（「afir 不依赖 afplan，
        **也不依赖后端方言**」，脚本注释写明理由：AFIR 是可对外公开的 kernel DSL）。今天
        只有一个后端所以看不出来；有第二个后端、或某个 target profile 少一种 dtype 的
        那天，「这个 op 能不能写成 ui8」会取决于编进了哪些后端。
     2. **ODS 说的是「这个 op 有没有意义」，不是「今天有没有人实现」。** `abs` 对 ui8 的
        语义是明确的，与谁实现了无关。推导的话 IR 的合法集合会随实现增删伸缩——昨天合法
        的图今天非法。平台无关门禁那句「平台差异属于 lowering，在它之上只能有并集」说的
        就是这件事。
     3. **推导会抹掉「表达得出来但还没接」这一档。** capability 层存在的全部意义就是回答
        它（`areElementTypesSupported` 不收就报 `UnsupportedType`，图干净回落）。ODS 一旦
        推导出来，这种情况变成 verifier 报错，报在错误的层上。
     4. **迁移到 MLIR 之后只会更糟。** legacy 消失后，「这个 op 收哪些 dtype」的真源必须
        落在 AFIR 自己身上。ODS 若是从注册处推出来的，迁移完就再没有任何一处独立记录 op
        的语义，只剩「我们碰巧实现了什么」。

     **判据是 `⊆` 不是 `=`**：ODS 允许比并集宽，那一段就是「表达得出来、还没接」。
     这条断言正好能抓住 2026-09-23 清掉的那 17 处——`exp` 注册了 i32 而它的 ODS 是
     `[F16, F32]`，那一位是句没人读的假话，当时是靠人肉对 ODS 发现的。
  4. 位掩码补齐 bf16 / i8 / i64 / ui32 / ui64，或改成类型谓词。**这一条可以最后做**，
     它只影响覆盖面上限，不影响正确性。

#### 第 1 条的落地（2026-09-23）

真源与对账是**两段接力**，缺哪一段另一段都白做：

- **表 ↔ legacy**：扫描脚本从 `REG_ASC_IR` 多抽 `mem_dtypes` / `reg_dtypes` 两列
  （含 `MakeT1List(kXxxPairs)` 那种要再查一张 pair 数组的写法；选择族取的是**数据**输入
  那个符号，不是 ui8 掩码），并加 `--check`，配一条 pre-commit 门禁
  `ascir-op-lowering-matrix`。没有这段，下一段就成了「把注册处钉在一份过期文件上」。
- **注册处 ↔ 表**：新用例 `test/ut/conversion/ElementTypeMatrixTest.cpp`。判据是
  **`注册处 ⊇ legacy 可建模部分`**，不是逐字相等——两个方向的后果不对称：
  - **少写了是真损失**：在一个 legacy 收的类型上答 unsupported，整张图静默回落 legacy，
    没有任何用例会红；
  - **多写了不是**：不在 legacy `REG_ASC_IR` 列表里的 dtype，它的 AscGraph 节点在调度之前
    就被 legacy 拒了，根本到不了 bridge。所以「比 legacy 宽」既伤不到人也用不上，将来
    AFIR 真能多支持一种时这条判据不该拦着。

  `Any` / `Shared` 只有一个掩码供两侧用，它要满足的下界取两列的**交集**（并集里窄侧没有
  的那部分，那一侧本来就发不出来）；`MemBased` / `RegBased` 各看自己那一列。

顺带补齐了一处「三条路分开写」：三元路径的元素类型集合原先写死在
`TernaryElementwise.cpp` 一个本地函数里，注册处看不见、也就无法对账，现在与一元 / 二元
同形——集合写在注册处，判据用共用的那个带掩码的重载。取值不变。

**这条断言第一次跑就照出 28 处不一致，同一轮全部清完**，分三类：

| 类 | 条数 | 形状 | 处理 |
|---|---|---|---|
| A | 17 | 注册处比 **ODS 和 legacy 都宽** | 收窄 |
| B1 | 4 | 注册处比 **ODS 和 legacy 都窄** | 放宽 |
| B2 | 10 | **legacy 比 AFIR 宽**（ODS 表达不到） | ODS 与注册处一起放宽 |

- **A 的根因是一个缺省实参**：`registerUnary` 原先有个叫 `kLegacyUnaryElementTypes`
  （= `f16|f32|i32`）的缺省，于是 `exp` / `ln` / `sqrt` / `sigmoid` / `reciprocal` /
  谓词族 / `clip_by_value` 这些 legacy 只收 f16 / f32 的 op 也一并领到了 i32。它们的 ODS
  本来就拒 i32，那一位是句**没人读的假话**，不是覆盖面。缺省实参已去掉——集合是每个 op
  自己的事实，不该有缺省。
- **B1**：选择族漏了 i16。ODS 的 `AFIR_SelectElementTypes` 有、legacy 两侧也有，只有注册处
  没写。补上就是实打实的覆盖面。
- **B2 是这轮真正扩了覆盖面的那一批**：reg 侧的 `abs`（+i16/ui8）、`neg`（+i16）、`sub` /
  `floor_div`（+ui8/ui16）、逻辑族（+ui16）、`where`（+ui8/ui16）、`clip_by_value`
  （+i16/i32/ui8/ui16）。**ODS 取两侧并集，每一侧收哪些由各自的注册处说**——这正是 L-18
  开头列的 #1 与 #4 的分工。`where` 与 `select` 的 ODS 表因此拆成两张：两者 mem 侧相同，
  但 reg 侧只有 `Where` 多收 ui8 / ui16，共用一张会把 `select` 也放宽到它供不出来的类型上。
- 逻辑族原先两侧共用一个集合，注释写着「这是 AFIR 的建模上限，不是 legacy 两侧真的一样」
  ——**ui16 恰恰是建模得出来的那一个**，于是被那句话一起挡掉了。

清完之后用例里没有已知差异表：下一个真的比 legacy 窄的 implementation 来时再加，加法写在
用例注释里。

#### 第 1 条没有减少「dtype 判断的处数」，这一点要说清楚

本条开头列的六处 dtype 说法，**第 1 条一处都没合并**。实测（`f2472df3` → 第 1 条做完）：

| | 之前 | 之后 |
|---|---|---|
| 注册处写出 dtype 集合的行 | 46 | **62** |
| ODS 手写的元素类型列表 | 45 | 45 |

**处数是涨的。** 涨的那 16 行来自去掉 `registerUnary` 的缺省实参——原先二十来个注册点靠
缺省免写，而那个缺省正是 28 处差异里 17 处的根因。把事实从「隐着」变成「写出来」，代价
就是这些行。真正变少的只有两处，都不是重复而是**藏**：三元那个写死的
`isSupportedElementType(Type)`，和那个缺省实参。

**第 1 条买到的是可验证性，不是更少的重复。** 减少重复是第 2 条的事（#2 / #3 / #5 三处
都在说「结果 dtype 与输入的关系」），它一处没动。要「少几处 dtype 判断」，下一步该做的是
第 2 条，不是第 3 条。

四个埋点验过：注册处写宽→UT 红；改齐但忘删快照行→UT 红并提示删；legacy 改了→门禁红；
表改了→UT 红。

> **阶段 F 到此挂起（2026-09-23）。** L-18 第 1、2 条已做，第 3 条的方向已订正；第 4 条
> （位掩码补齐 bf16 / i8 / i64 / ui32 / ui64）与 L-19、L-20 一并搁置，先回主线把核心功能
> 做完。挂起不等于作废：第 3 条是这一轮新引入的手工不变量——ODS 与两侧集合的关系今天没有
> 任何东西盯着——它是这几条里最该先回来的一条。

### L-19　新增一个 op 的成本，七成在测试样板上　**判据 1 已做（2026-10-27），判据 2 未动**

- **实测**（四个「没有新机制」的 op 提交，按产品代码 / 测试 / 文档拆分）：

  | 提交 | 产品代码 | 测试 | 文档 |
  |---|---|---|---|
  | `61860dc1` Pow | 7 文件 79 行 | 10 文件 326 行 | 88 行 |
  | `2423bba2` Remainder | 6 文件 110 行 | 9 文件 323 行 | 48 行 |
  | `f591fb32` BitwiseAnd | 8 文件 132 行 | 9 文件 328 行 | 37 行 |
  | `50b31cf2` TrueDiv | 8 文件 135 行 | 10 文件 415 行 | 105 行 |

  产品代码那 79～135 行里还有三到四成是注释：Pow 实际只有 **40 行代码**，而且其中一处是
  **删掉** 26 行手写的 `verify()` / `inferReturnTypeComponents`。
- **所以「新增一个 op 工作量巨大」这个判断，对产品代码不成立，对测试成立。** 一个 CPU
  E2E 用例 4 个文件 215 行，其中 `CMakeLists.txt`（14 行）与 `Case.h`（30 行）是逐字样板，
  `Case.cpp` 里建图那一段也高度雷同；真正一个 op 独有的只有 `KernelTest.cpp` 的数值参考。
- **另外两处每加一个 op 都要手改的计数**：`BridgeOperationRegistryTest` 里写死的
  `EXPECT_EQ(bufferForm.size(), 46u)` 和它下面那张名字列表。这两处是**故意**要求手改的
  （「增长必须是一次经过评审的改动」），合理，保留。
- **完成判据**：
  1. E2E 样板收成一个 CMake 函数 + 一个头文件模板，新增用例只写 `Case.cpp` 的建图段与
     `KernelTest.cpp` 的数值参考。目标：一个新 op 的 E2E 从 215 行降到 100 行以内；
  2. `asc_graph_builder.h` 的 builder 方法能由一张表生成（今天每个 op 手写一个）。
- **不做的**：不要为了省行数去掉 lit 的两个 `--target` 正例——那一对是「差异」的唯一证据，
  E5 / E2 两轮都靠它抓到过写反的家族。

### L-20　中文注释的行文约定

- 2026-09-22 已做：**机械标点统一**（82 处，全部是纯注释行）——中文语境下的
  `, ; :` 与成对括号改全角，反引号里的代码片段和 `(输入, 输出)` / `(值, dtype)` 这类记号
  式写法保持半角；另外重写了六个文件头的行文
  （`{Unary,Binary,Ternary}Elementwise.cpp`、`ElementwiseApiFamily.h`、
  `ConstBlockTensor.h`、`AFIRElementwiseSupport.h`）。
- **约定**（后面写注释照这个走，也是剩余优化的目标）：
  1. 一句话说一件事。超过两行的长句拆开；
  2. `——` 每段最多用一次，其余场合改用冒号、分号或另起一句；
  3. 少用「那一档」「这一条」「那一半」这类无指代的代词，直接点名；
  4. 中文语境里的标点一律全角，反引号内的代码原样；
  5. 保留「先说结论、再说为什么、最后说错了会怎样」这个结构——它是这批注释最有价值的
     部分，不要在优化行文时删掉。
- **剩余**：`Ops/Elementwise.td`（257 行中文）、`BinaryElementwise.cpp` 正文、
  `Basic/OpVecBinary.td`（133 行）三处的段落还没逐段重写。**不急**，建议随各自 op 的
  下一次改动顺手做，不要单开一个纯改注释的大提交。

---

## 已核销 / 明确不做

| 项 | 裁决 | 说明 |
|---|---|---|
| `clipbyvalue.h` 新增重载改掉了 legacy 内联进 kernel 的头文本 | **加注释，不回退** | 这个转发重载只有 MLIR 路径在用；legacy 的 `ClipByValueApiCall` 仍按原顺序发调用，**行为**一个字不变，但这个头是整段内联进 kernel 的（`ascendc_api_registry` 的 `clipbyvalue.h` → `build/.../clipbyvalue_str.h:47`），所以 legacy 生成的 kernel 源码里会多出那十行。**验收条件 6「逐字节不变」按「语义不变、文本多一个重载」理解**。已在 [autofuse/ascendc/api/clipbyvalue.h](../../../ascendc/api/clipbyvalue.h) 的注释里写明 |
| `afir.cast` 同 dtype 过得了 verifier、选不出实现 | **保留，不改** | 查过 legacy：`ascir_builtin_ops_v1.cpp:188` 的 `kCastTypePairs` 第一行就是 `{DT_FLOAT, DT_FLOAT}`，而 `cast_api_call.cpp:54` 有 `GE_ASSERT_TRUE(x.dtype != y.dtype)`——**legacy 自己就是「类型表收、codegen 拒」这个两层结构**。MLIR 这一侧（ODS verify 查并集、lowering 的 `CastPair` 档要求 `source != destination`）与它同形。那一行在 legacy 表里为什么存在（注释猜是给 VF 融合留的）没有结论，**不在本切片解决** |
| 偏差 5 / 6 / 8（常量 block、`isReuseSource`、比较族的「新调用形态」） | **已推翻并接通** | 三条卡点核完，两条不存在、一条是「同一个 `CMPMODE` 两侧落在不同位置」。详情见进度文档对应小节 |
| E8（reg-based 独有的 81 个） | **取消** | 与 E1～E7 的验收脱钩，pyasc 同步后另行评估 |
| E7 收尾 | **已空** | 四项全部在合入前的整改里做掉了 |

---

## 附：这一轮评审确认为「对」的地方

写下来是为了后面的人不用再核第二遍。

- 比较族两侧的发射形态与 legacy 的 `compare_api_call.cpp` / `compare_v2_api_call.cpp` 的
  `outer_repeats_size == 0` 非 CV 分支**逐字一致**，包括 reg 侧的
  `{static_cast<uint16_t>(n)}, {1}, {1}` 与 mem 侧「tmp 在 count 之后」。
- `RsqrtExtend<T>(dst, src, blk<float>, tmp, size)` / `LogicalNot<T>(dst, src, blk<half>, tmp, size)`
  的形参顺序与常量 block 的 dtype，与 `api/rsqrt.h:14` / `api/logical_not.h:89` 一致。
- **两侧 25 个扩展头的前导声明逐个核对过**，`MAX_REPEAT_TIME` / `KernelUtils` / `Rational`
  的用法与 td 里的 `preludes` 一处不差、无多余也无遗漏（`remainder.h` 用的是
  `AscendC::MAX_REPEAT_TIMES`，不是 autofuse 的那个，确实不该声明前导）。
- 所有在用的注册表 key（15 个 `*_reg_base.h` + mem 侧全部）在
  `ascendc_api_registry.cpp` / `ascendc_reg_base_api_register.cpp` 里都解析得到。
- 「多个扩展头按字典序发」那条推理成立：每个 op 的列表自身有序 + 全局按字典序合并
  ⇒ 所有 op 的两两依赖约束都保持；`CheckDeclarationOrderIsEmittable` 把这个前提钉住了。
- 比较族六个 op 的 `CMPMODE` 映射**每个都被钉过一次**：mem 侧 lit 钉 EQ / LT / GE，
  reg 侧 lit 钉 NE / GT / LE，合起来六个 op → 六个模式的对应关系无一遗漏。
