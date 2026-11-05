# 转置泛化测试设计（P1-P5，legacy 场景对齐）

> 上游设计：[2026-10-19-afir-transpose-e2e-design.md](2026-10-19-afir-transpose-e2e-design.md)（第一/二部分已交付，
> 其「扩展步骤 P1-P5」章节为本设计的输入）。本文档以**测试**为主体，包含验证暴露的 MLIR schema/ATT 适配及空 include 修复；本阶段是
> 「逐形态验证并登记能力缺口」。本轮用户明确只交付 legacy Inductor 迁移，TF 排除；
> 该迁移清单已验证通过，P1–P5 历史扩展不等同于本次必交清单。最新结果见
> [Inductor 迁移验收](2026-10-29-afir-transpose-generalization-test-design.md#inductor-用例迁移验收)。
> 文中 legacy 行号为 2026-10-29 快照，以符号名为准。

# 简介

## 目的

为泛化路线 P1-P5（消除路线无补齐形态、elementwise 链融合、多 transpose、交互形态、上板）给出
可执行的用例设计与验收口径。核心原则：**用例对齐 legacy 的测试场景**——legacy 有用例的形态直接
锚定其形状与断言层级；legacy 没有用例的形态用「同图 legacy 产物对照」补锚（见方法论）。

## 范围

- 包含：P1-P5 的 CPU E2E 用例（mem 档为主）、lit 合成矩阵补充、reg 档对齐口径、a2/a3 上板脚本
  与判据、每图固定循环与验收标准。
- 不包含：legacy ATT/v35 公共生成器修改；验证暴露的 MLIR 适配边界缺陷随本轮修复。nddma broadcast
  折叠用例族（属基线 broadcast 特性，A7 旧归因已由新基线复核撤销）；reg 档 e2e 数值验收（上游设计验收第 9 条已关闭：
  legacy 该层无数值锚点）。

# 总体概述

## 软件概述

### 项目介绍

转置/Pad 迁移（上游设计第一/二部分）已完成：`afir.transpose`/`afir.pad` 方言与 bridge、
跨步 load/store 段几何、`TransposeExtend`/`ConfusionTranspose` 两族下降、RemovePad/Pad、
allocator 修复。泛化路线验证调度器已有能力在新形态下的 MLIR 下降正确性。

### 产品环境介绍

与上游设计相同：MLIR e2e CPU 框架（`test/e2e/cpu`，tikicpulib 仿真）、lit
（`test/lit`）、e2e NPU 脚本（`test/e2e/npu`，npussh + torch 对拍）。

### 软件功能

P1-P5 五类形态的验证用例与判据；每图走固定循环（dump 调度 → legacy 同图产物对照 → 数值绿 →
补进 lit 矩阵）。

## 设计约束

- **两套 legacy 测试域不可混淆**（实测版图，2026-10-29）：
  - **a2/a3 域（mem 档，910B1 仿真）**＝`tests/st/`：`backend_e2e/`（全管线数值，如
    `sub_transpose_abs_test`）、`codegen/e2e/`（codegen 期望件，如
    `load_transpose_store_expect_code`）、`device_validation/`（上板框架，profile
    `ascend910_9362`）。
  - **v35 域（reg 档，950PR 仿真）**＝`tests/v35/`：`st/optimize/optimize_st_v2.cpp`（调度层
    断言）、`st/backend_e2e_v2/`（全管线，转置仅 `transpose_bool_test` 且数值腿 GTEST_SKIP）、
    `ut/ascendc/api_regbase/test_transpose.cpp`（TransposeExtend API 数值 UT）。
- 本特性零触碰 v35/ATT（上游设计红线）；用例只经公开构图接口建图。
- reg 档（950PR）数值不作为验收门（上游设计验收第 9 条口径）。

## 假设和依赖关系

- 调度规则事实（P0）沿用上游设计记录：场景 1/3（尾轴转置或尾轴小）→ 保留 + UB 重排；场景 2
  （非尾轴转置且尾轴大）→ 消除、折进 load；多 transpose 全推到 load；存在 Reduce 时原图消除
  全部 transpose。现入口 `platformv2.cpp:104`（`TransposeFusionCaseGeneratorV2::GeneratorTask`）、
  折叠机制 `ascendc_ir.cc:584`（`DoApplyTensorAxisReorder`）。
- a2/a3 真机可用（npussh）；950 无实机。
- **待确认**：P4 的 broadcast+transpose 图是否会被调度成 brc→Nddma 折叠。启动时先 dump
  调度确认，并与 legacy 同图对照；新基线已修复 Store GM 偏移且启用基线 nddma 用例，
  不再仅凭命中该路径判为 A7 阻塞。发现实际差异时按证据确定归属。

# 需求分析与设计

## 整体介绍

### 2026-10-31 迁移盘点与执行修订

本轮基于 `8b6f5ba6` 复核。原 MLIR `LegacyGraphGuardSt.cpp` 的搬迁集合不包含转置图，
不能以 elementwise/broadcast 已搬迁推断 transpose 已覆盖。以下区分图迁移、生成代码验证
与数值执行，不把调度 ST 或 API UT 宣称为整图数值证据。

| legacy 域与入口 | 原始形态 | MLIR 落点 | 验证层级 |
|---|---|---|---|
| a2/a3 `backend_e2e/sub_transpose_abs_test` | `[32,64,512]`，sub→transpose `[1,0,2]`→abs | 新 `sub_transpose_abs` CPU E2E、同名 `LegacyGraphGuardMemSt`、NPU 脚本 | 生成看护 + CPU 数值；真机待执行 |
| a2/a3 `pgo_sub_transpose_abs_test` | 与上一条同图，PGO 选择方式不同 | 共享图迁移；PGO 不以普通执行替代验证 | 图覆盖已补，PGO 独立记录 |
| a2/a3 `codegen/e2e/load_transpose_store_expect_code` | `[2,37,23]`，021 | 已有 `load_transpose_store_2x37x23` | 全管线 CPU 数值 |
| v35 `NddmaCaseTranspose021OutputWithSingleRef` | `[32,64,16]`，021 | 新同名 `LegacyGraphGuardRegSt` | 有效严格模式预检通过，待当前构建全量复验 |
| v35 `LoadCastTransposeCase` | `[64,64]`，f16→f32→10 | 新同名 `LegacyGraphGuardRegSt` | 有效严格模式预检通过，待当前构建全量复验 |
| v35 `LoadGEWhereTransposeCase` | `[41,54,38,55]`，GE→Where→0312 | 新同名 `LegacyGraphGuardRegSt` | 预检暴露 Ge VF 缺口，已补共用映射及数值用例，待当前构建严格复验 |
| v35 `TransposeTwoAxisSplitCaseNeedInputAlign` | `[347,15,49]`，转置+broadcast+mul | `LegacyGraphGuardRegSt.*CandidateBoundary` 保留原图 | 有效严格模式下候选拒收；正向迁移未完成，不能以 legacy optimizer 成功替代 |
| v35 `backend_e2e_v2/transpose_bool_test` | bool `[4,8]`，数值腿 GTEST_SKIP | 明确记录未支持边界，不换成 f32 后宣称完成同用例迁移 | `afir-transpose-dtype-boundaries.mlir` 明确验证 ODS 拒收 |
| 两域 AscendC `test_transpose.cpp` | API 参数枚举及数值 | 对照 API 特化面补 MLIR 发射/仿真覆盖 | 不等同于 backend 图迁移 |

**P1 更正**：legacy `sub_transpose_abs_test` 的 perm 为 `[1,0,2]`，不是 `[0,2,1]`。
本轮直接搬入同链、同 perm、同形状，避免只借形状却没有覆盖消除路线。另加 `[5,37,23]`
作为同图非对齐泛化格；已有 021 和 2D 保留路线不移除。

**已复现的实现缺口**：reg Compact stride 在尾块应按 legacy 的
`GetVectorizedStrideActualSizeFlag` 选择 ActualSize。修复复用现有实际尺寸绑定，新增
`afir-transpose-compact-tail.mlir` 钉同一 actual SSA 同时用于 dims 和 stride。

**执行环境状态**：v35 严格限制 CPU 仿真；a2/a3 真机复用 NpuRunner，按版本一致性要求
部署后执行。当前 `npussh list` 明确返回“尚未登录”，故新增真机脚本不计为已验证。
所有“待运行”项目只有取得本轮输出后才销号。任务截止时间为 2026-11-01 19:00（UTC+8）。

每个 P 步 = 一组新 e2e 用例 + lit 补充 + legacy 对照。用例落位沿用现有 harness 惯例：
mem 档进 `test/e2e/cpu/cases/load_transpose_*`（默认平台 910B1，**数值验收**）；reg 档用
`PLATFORM reg_base`（950PR，**发射文本对齐**，数值为加强非验收）；上板用 `test/e2e/npu/*.py`。

**方法论（每图固定循环，继承上游设计）**：新增 CPU E2E → `AUTOFUSE_DFX_FLAGS` dump 调度 →
**与 legacy 同图产物对照**（用 legacy `backend_e2e` 同款生成器跑同一张图，参数逐字比对——
legacy 没有现成用例的形态靠这一步补锚）→ 有缺口按上游设计同流程补 → 数值绿 → 新形态补进 lit
合成用例矩阵。**事实先行，不硬按预期实现**。

## 功能需求

### 功能需求 1：P1 消除路线·无补齐形态（场景 2）

1. 介绍：非尾轴转置且尾轴大时调度把转置折进 load（无 Pad、无 UB 重排的纯粹跨步折叠），验证
   段几何物化的一般性。
2. 输入：legacy `sub -> transpose[1,0,2] -> abs`，形状 `[32,64,512]`；补 `[5,37,23]` 尾块。
3. 处理：mem 档 e2e 数值 + 与 legacy `backend_e2e` 同图产物对照；reg 档 kernel 文本断言两类
   模板（沿用 `CaseRegBase.cpp` 口径）。
4. 输出：`sub_transpose_abs_32x64x512`、`sub_transpose_abs_5x37x23`，按真实输出
   `[B,A,C]` 的坐标 `(b,a,c)` 映射回输入 `(a,b,c)`。Store/Output 显式使用转置后的视图，
   不传会被构图器补回原 shape 的空 repeats。是否消除转置以最终调度产物为准，不从形状推定。

   legacy 锚：a2/a3 域 `tests/st/backend_e2e/sub_transpose_abs_test`（`{32,64,512}`，链上带
   sub/abs，ICPU 数值 0-diff）；v35 域 `optimize_st_v2.cpp` 的 `NddmaCaseTranspose021OutputWithSingleRef`
   （`{32,64,16}`，纯形态、断言 impl 图出 Nddma）。

### 功能需求 2：P2 elementwise 链融合

1. 介绍：transpose 与上游 elementwise 链同 kernel 融合；消除路线对整条前驱路径做轴序重排。
2. 输入：单目链（cast→transpose）与双输入链（transpose→add）。
3. 处理：mem 档 e2e 数值 + legacy 对照；reg 档 lit 补 cast+transpose 消除形态发射锚。
4. 输出：新增 e2e `cast_transpose_64x64`（f16→f32 `{64,64}`，**对齐 v35 域
   `LoadCastTransposeCase` 的形态**）与 `transpose_add_2x37x23` / `transpose_add_4x16x32`（双输入，形状取
   `{2,37,23}`/`{4,16,32}`——**对齐既有 NPU 脚本** `test/e2e/npu/load_transpose_add_store.py`，
   非对齐形状顺带覆盖尾块路径）。lit 补链形态合成用例。

   legacy 锚：a2/a3 域 `sub_transpose_abs_test` + `pgo_sub_transpose_abs_test`（链形态全管线
   数值）；v35 域 `LoadCastTransposeCase`（cast+transpose 调度层）、`LoadGEWhereTransposeCase`
   （`{41,54,38,55}` GE+Where+transpose 调度层，可选作进阶档）。

### 功能需求 3：P3 多 transpose 级联

1. 介绍：级联两个 transpose，验证「全部推到 load」的折叠形态。
2. 输入：`load -> transpose(p1) -> transpose(p2) -> store`，p1/p2 复合仍为合法轴重排。
3. 处理：legacy **无现成用例**——按方法论用 `backend_e2e` 同图生成 legacy 对照产物逐字比对。
4. 输出：新增 `transpose_chain_2x3x4x5`，perm `[0,2,1,3]` 后 `[0,3,1,2]`，最终轴序
   `[0,3,2,1]`；独立输出坐标 oracle 验证完整数值。调度规则出处：上游设计 P0 事实（多 transpose 不生成保留模板）。

### 功能需求 4：P4 交互形态

1. 介绍：broadcast + transpose（广播轴消隐 `ShouldIgnoreDataCopyZeroAxis` 实战）；Reduce 冲突
   （存在 Reduce 时原图消除全部 transpose 的整图形态）。
2. 输入：broadcast→transpose 链；transpose+reduce 同图。
3. 处理：先 dump 调度确认形态归属（转置消除路径 vs brc→Nddma 折叠路径），再对照 legacy；
   沿用新基线 nddma 实现，不以旧 A7 归因预先挂起。
   legacy 无组合形态现成用例（broadcast、reduce 各自的族都在），用同图对照补锚。
4. 输出：mem 档 `broadcast_transpose_8x37x23`：`(1,37,23)` brc→`(8,37,23)`→021；
   `transpose_sum_2x37x23`：021→sum(末轴)，输出 `[2,23,1]`。均有独立 oracle；
   legacy 同图参数对照与真机结果需单独记录，不能用 CPU oracle 结果替代。

   legacy 锚：v35 域 `NddmaCaseBrc*` 族（broadcast×nddma 调度层 6+ 条）、`ReduceNeedAlignment`；
   a2/a3 域 brc 族（`broadcast_force_merge` 等）与 reduce 族（`load_rsum_store` 等）的图模式。

### 功能需求 5：P5 上板

1. 介绍：P1-P4 每类挑代表图，a2/a3 走上板同套标准。
2. 输入：代表图（P1 消除形态、P2 add 链、P3 级联、P4 交互各一）。
3. 处理：`test/e2e/npu/*.py` 脚本（npussh，torch 2.10；纯 permute 会被降级为 extern
   `aten.clone`，须用复合图——沿用 `load_transpose_add_store.py` 的注释口径）；legacy vs MLIR
   两路对拍（接管证据独立核实，非 fallback 假绿）。
4. 输出：新增/复用脚本 `sub_transpose_abs.py`、`load_transpose_add_store.py`、
   `transpose_chain.py`、`broadcast_transpose.py`、`transpose_sum.py`；判据 bit 级等价 + p50（launcher 逐次同步，3 轮 × 500 iters，
   best-of-3，劣化 >5% 不通过）。已有脚本：`load_transpose_store.py`（v1）、
   `load_transpose_add_store.py`（P2 代表图现成）。

## 非功能需求

### 可维护性

用例目录/命名沿用现有 `load_transpose_*` 惯例；每条用例的 CMakeLists 注释写明 legacy 锚
（哪个域、哪个用例/形状）与「本条覆盖不到什么」，防止绿被当成覆盖到了。

### 可测试性

全部用例可经 `ctest`/`llvm-lit`/python 脚本独立执行；调度 dump 经 `AUTOFUSE_DFX_FLAGS`。

### 可移植性

不引入新平台依赖；reg 档仅用既有 950PR 仿真库。

### 可靠性

无产品代码改动；用例数值断言沿用既有 KernelTest 风格（纯搬运/加法 0-diff）。

### 特性交叉影响

按 [cross_feature_check.md](../../../../docs/guidelines/cross_feature_check.md) 逐项：

| 场景 | 适用性 | 分析说明 |
|------|--------|----------|
| SuperKernel Python 接口 | 不适用 | 不触碰 super_kernel/ |
| SuperKernel C++/AOT 接口 | 不适用 | 不触碰 |
| Autofuse 图优化 | 不适用 | 零触碰 v35/ATT；若发现缺口另立需求再评估 |
| Autofuse Codegen/Backend | 适用 | MLIR opaque API 字段保留 stage owner，并在 ATT 返回文本边界限定访问；空扩展头列表返回空集合 |
| AscendC API / Runtime 交互 | 不适用 | 无新增 runtime 调用 |
| Python/C++ 混合绑定 | 不适用 | NPU 脚本沿用既有 NpuRunner |
| 构建与打包 | 适用（轻） | 新增 e2e 用例目录与 add_subdirectory 行；`--no-autofuse` 路径不受影响 |
| 测试与覆盖率 | 适用 | 本设计主体；见测试设计 |
| 性能与日志 | 适用（轻） | CI 时长评估见性能节 |
| 兼容性 | 适用 | 不改 legacy ATT/v35、外部 API/ABI；修正 MLIR 生成访问路径和测试图的输出视图 |

## 性能

### 编译时长

本轮新增 9 个 CPU E2E 实例（SubTransposeAbs 2 个、P2–P4 6 个、GE 谓词 1 个）。增量构建每例生成并
编译独立 kernel；公共 `Base.td` 的修复会触发 AscendC 方言及相关 translation 单元重编译，
不能把此次重编译时长直接视为稳态编译开销。所有构建限制 `-j 8`。
MLIR adapter 的 owner 集合构建为 O(F log F)，字段文本扫描最坏 O(F×S)，F 为 schema
字段数、S 为生成文本长度；仅编译期执行，不在 kernel/launch 热路径调用。未新增日志。

### 执行性能

P5 上板 p50 判据见功能需求 5（≤1.05）。opaque owner 适配只改变 host 编译前的字段
拼写，不新增运行期调用；Compact stride 的 actual-size 运算属于正确性修复。当前无可用
a2/a3 登录态，运行性能尚未复验，不能用结构分析替代 p50 结果。

### 内存和产物大小

未新增交付文件、全局缓存或常驻状态；新增测试二进制和局部编译期集合。API tiling
仍使用 CANN 官方结构，不复制布局或改变已有对外 ABI；产物大小未单独量化。

## 接口设计

无新增/修改产品接口。测试侧沿用 `add_mlir_e2e_cpu_case` / lit / `NpuRunner.run` 既有入口。

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|--------|----------|----------|------|
| 接口说明 | 是否需要接口评审 | 否 | 无产品接口变化 |
| 接口说明 | 是否需要补充文档 | 否 | 本文档即测试设计 |
| 接口兼容 | 行为是否兼容 | 是 | 保持合法图语义；修正字段路径、Compact stride 及错误的测试视图 |
| 接口兼容 | ABI/API 是否兼容 | 否 | 无接口变化 |
| 接口约束 | 约束不满足时是否清晰报错 | 否 | 沿用既有判据 |
| 接口测试 | 是否需要独立接口用例 | 是（内部） | adapter 路径适配、schema owner 及 API 字段契约 UT |

## 软件设计

### 关键数据结构

沿用 `TilingFieldSchema` 的 `(path,name,type)` 字段身份：opaque API 字段的 path 始终
保留 aggregate stage owner；scalar 在单 stage 投影下仍为空 path。字段集合和文本仅在
编译调用内存活，不新增全局状态。用例侧沿用 `AscGraphBuilder`、`CaseConfig`、`ShapeConfig`。

### 关键技术/算法

legacy 同图产物对照：从同一份 optimizer 产出的 `FusedScheduledResult`，分别调用
MLIR 与 legacy `Codegen::GenerateKernel` 生成 kernel，比较搬运/计算 API 与参数表达式。
MLIR 先生成，避免 legacy 的参数补全影响其输入；符号名/SSA 编号或合法分段不同需说明，
不能把整份源码字节不同直接当成语义不同。独立 CPU oracle 验证数值；真机 tiling、接管
及性能结果单独记录。探针只放 `/tmp`，不新增测试侧 codegen 路由或产品开关。

### 流程设计

每图固定循环（见整体介绍）；发现缺口的升级路径：dump 定位 → legacy 对齐 → 上游设计同流程补
（需求条目 + lit/UT）。

### 对子模块的修改

| 目录 | 修改点 |
|------|--------|
| `autofuse/mlir/test/e2e/cpu/cases/` | 新增 `sub_transpose_abs`、`transpose_generalization`；复用 EQ/Where fixture 增加 GE 数值格 |
| `autofuse/mlir/common/tiling_schema/`、`bridge/AttSchemaAdapter/` | opaque owner 路径及根级访问适配，完整 stage 参数保持相对访问 |
| `autofuse/mlir/api/ascendc/.../Autofuse/Base.td` | 空扩展头列表返回空集合，避免空 include |
| `autofuse/mlir/conversion/AFIRToAscendC/` | Compact stride、canonical API 字段校验及复用比较谓词映射补 GE |
| `autofuse/mlir/test/e2e/cpu/CMakeLists.txt` | 新增 add_subdirectory 行 |
| `autofuse/mlir/test/lit/conversion/` | 新增/扩展合成 lit（每形态一条） |
| `autofuse/mlir/test/e2e/npu/` | 新增上板脚本（P1/P3/P4 代表图；P2 复用既有） |
| `autofuse/mlir/docs/development/` | 本文档 + 上游设计的泛化章节状态更新 |

### 错误处理

#### 系统错误

用例失败即 ctest 红，无特殊处理；上板失败按 npussh 既有报错口径。

#### 接口错误

无新增公开接口。metadata 的多 owner 裸访问明确返回失败，不静默选第一项；
这类 schema 不一致不标为 ATT 能力不足，不通过回落掩盖。

## 安全检查

已按 [编码红线.md](../../../../docs/guidelines/编码红线.md) 检查：不新增敏感信息、
外部 API/ABI、全局资源或图重写。owner 匹配使用 schema 的稳定字段身份，不按容器
偶然顺序选择；有歧义时明确失败。测试索引仅用于固定注册形状，GM 缓冲由 RAII 或
既有 fixture 释放。Ge 复用现有谓词/Select 数据流，覆盖相等及两侧分支。

## 兼容性检查

不改已有对外脚本参数和交付 ABI；修正既有测试图的输出视图与失效严格开关。当前基线
E2E/lit/UT/ST 全量作为回归门，不能沿用旧基线 125/269 数字。

## 测试设计

### 测试边界

入口：e2e ctest（数值）、lit（发射/调度文本）、npu 脚本（上板）。出口：数值 0-diff（纯
搬运/加法）、kernel 文本断言（模板归属）、bit 级 + p50（上板）。打桩：无（真调度真 codegen）。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|----------|------------|----------|----------|
| 功能-P1 | 消除路线无补齐（{32,64,512}） | e2e 数值 + legacy 同图对照 | E2E |
| 功能-P2 | cast→transpose（{64,64} f16→f32） | e2e 数值 + lit | E2E/lit |
| 功能-P2 | transpose→add 双输入（{2,37,23}/{4,16,32}） | e2e 数值（尾块覆盖） | E2E |
| 功能-P3 | 双 transpose 级联（4D） | e2e 数值 + legacy 同图对照 | E2E |
| 功能-P4 | broadcast→transpose | e2e 数值 + dump 确认形态归属 | E2E |
| 功能-P4 | transpose + reduce 同图 | e2e 数值 + legacy 同图对照 | E2E |
| 功能-各 P | 每形态 lit 合成用例 | lit（发射文本锚） | lit |
| 异常-P4 | 图被调度成 brc→Nddma 折叠且验证失败 | dump 与 legacy 同图对照，依据实际差异记录归属 | E2E |
| 性能-P5 | 代表图 p50 不劣化 | npussh 上板，best-of-3 ≤1.05 | Benchmark |
| 兼容性 | 当前基线全部 E2E/UT/ST/lit 不退化 | 全量回归，记录本轮实际数量 | E2E/UT/ST/lit |
| 特性交叉 | 接管证据（非 fallback 假绿） | kernel 名 / dlsym 执行链核实 | E2E |

### 测试命令

```bash
# 容器内（CPU 仿真）
cmake --build /workspace/build/autofuse/mlir/test/e2e -j 8
ctest --test-dir /workspace/build/autofuse/mlir/test/e2e -j 8          # 期望全绿（含新用例）
/opt/llvm/bin/llvm-lit -j 8 /workspace/build/autofuse/mlir/test/lit    # 期望全绿
ctest --test-dir /workspace/build/autofuse/mlir/test/ut -j 8
# a2/a3 上板（npussh 登录后）
python3 autofuse/mlir/test/e2e/npu/load_transpose_add_store.py          # P2 代表图
```

## 验收标准

1. P1-P4 每类 mem 档 e2e 数值绿；每形态有 lit 合成用例。
2. P3/P4 的 legacy 同图产物对照记录在案（参数逐字比对通过或差异已解释）。
3. P5：每类代表图 a2/a3 上板 legacy/MLIR bit 级一致 + p50 ≤1.05，接管证据独立核实。
4. 当前基线全部 E2E/lit/UT/ST 全量不退化；旧基线 125/269 仅作历史记录，不作为本轮数量。
5. 每条新用例的 CMakeLists 注释写明 legacy 锚与覆盖边界；上游设计文档泛化章节状态同步。
6. 全程零触碰 v35/ATT；发现缺口按上游设计同流程另立，不在本范围内硬修。

---

### 设计文档检查结果
- [x] 跨特性交叉影响：已按 cross_feature_check.md 逐项检查，Codegen、兼容性、测试、构建及性能适用；SuperKernel、legacy ATT/v35 与绑定接口无改动
- [x] 编码红线：已检查字段所有权、ABI/API、资源生命周期及确定性；多 owner 裸访问显式拒绝，不静默选首项
- [x] 测试设计：P1-P5 用例/方法/类型见测试用例设计表，含异常归属确认与兼容性回归
- [x] 性能评估：已区分公共 TableGen 重编译、编译期文本适配与运行期；上板 p50 尚待验证


## Inductor 用例迁移验收


按实际入口审计 `autofuse/tests`：直接 Inductor C++ 调用及其共享构图 helper 中，
Transpose 专项有下列 4 项；Python `Autofuser.codegen` 最终调用 `GenerateForInductor`，
UT/ST 各含同一组 3 项。两份 Python 构图 AST 一致，迁移为 3 个共用测试。
Matmul 的 `transpose_x1/transpose_x2` 属性属于 Matmul，不列为 Transpose op 迁移。

| legacy 来源 | MLIR 迁移落点 | 验证结果 |
|---|---|---|
| `TestCodegenTiling.GetTilingDataReprShouldKeepSingleGroupApiTilingFieldInDeclarationOrder` | `LegacyInductorTransposeTest` 同名用例；native GenTilingData + repr emitter | 通过 |
| `TestCodegenTiling.GetTilingDataReprShouldKeepMultiGroupApiTilingFieldInDeclarationOrder` | 同上，多 stage canonical 元数据 | 通过 |
| `TestCodegenTiling.SplitHeaderApiTilingSourceShouldIncludeApiHeaders` | 同名 MLIR ATT adapter 用例，保留原始 interleaved Transpose 构图 | 通过 |
| `TestCodegenTiling.DisabledInductorPgoShouldOverrideAttPgoEnvironment` | 同名 adapter 用例；另加真实 native Inductor 集成测试 | 通过 |
| Python UT/ST `TestAutofuseLoadTransposeStore.test_construct_graph` | `test/ut/python/test_legacy_inductor_transpose.py` | 通过 |
| Python UT/ST `TestAutofuseLoadTransposeStore.test_autofuse_backend` | 同上，保留 f16 `[100,200,300]`、102、Store offset=1 和 axis_map；严格 MLIR 生成 | 通过 |
| Python UT/ST `TestAutofuseLoadTransposeStore.test_autofuse_backend_faild_dump_graph` | 同上，保留无效图并检查 `Optimize fail` | 通过 |

C++ 新 suite 为 5/5（4 个迁移项 + 1 个 native 集成项），Python 为 3/3。
Python 迁移去掉旧正向测试吞掉 `RuntimeError` 的行为，并要求产物有 `_subgraph0_plan` 和
`GetTilingDataRepr`；临时 overlay 及 dump 均只存放本次测试的临时目录。
绑定测试目标复用生产绑定源码并链接生产 `aihac_codegen`，不使用 legacy UT 的独立 codegen/ATT 库。

迁移发现并补齐：native host tiling 原先没有 `GetTilingDataRepr`；新增表示函数从 header 使用的同一份
canonical schema 递归发射，保留完整嵌套身份与声明顺序，不回扫 legacy 图，也不新增字段词表。
GenTilingData 改为每个模板依次收集 size var/API，最后收集共享 q/b，避免将 s1 移到前一模板的 API 前。
原 legacy 空占位 group 在 native fixture 中用空字段 stage 表示，测试仍覆盖多组嵌套及同一字段顺序。
native Inductor 暂仅有 modeled tiling ABI，显式禁止 ATT 环境变量隐式启用其尚未接入的 legacy PGO helper。

补强验证：两个既有转置 CPU 数值实例通过，且实际调用新 repr 函数验证空指针、嵌套 API 字段及顺序。
本机 910B3 上，隔离 Torch 2.10.0 / torch_npu 2.10.0.post7 与当前源码 overlay 后，
`load_transpose_add_store.py --mode both --steps 0` 两种输入形状的 legacy/MLIR 路线共 4/4 数值通过，
MLIR 路线有严格接管标记，容差保持 rtol=0、atol=0。这里只记录正确性，不宣称性能或 a3 已验证。
未覆盖上板的原 Python 图沿用 legacy 的“构图/生成”验收层级，不把它写成设备数值通过。

本轮最终回归：完整 UT 14/14 个 CTest 项通过；完整 lit 284/284 通过。
上述原始用例合计 10 个来源测试实例（C++ 4、Python UT/ST 各 3），去重为 7 个行为测试；
另补 1 个 native PGO 集成测试，共 8 个直接迁移/集成断言用例全部通过。
此前 GE/Cast 等借用 optimizer 的组合探索不纳入该清单，不以本轮通过宣称这些额外问题已经修复。


## 泛化测试的未闭环项

以下是补充泛化探索中的已知问题，不属于 legacy Inductor 的 7 项迁移验收，但不能宣称已修复：

- Cast→Transpose：CPU 数值执行在 GM→UB 搬运阶段需要 256 字节，分配仅 128 字节；尚未证明属于 Cast 算术或 Transpose API 缺陷。
- Transpose chain：CPU 数值不一致。
- Broadcast→Transpose：元素数量证明阶段拒收。
- Transpose→Sum：实现选择阶段拒收。
- GE→Where→Transpose：完整严格 ST 仍存在 stride 适配失败。

最新全量 UT/lit 与上述泛化 CPU/ST 的验证集合不同；不得据 UT/lit 全绿宣称全部泛化用例通过。910B3 的 4 项验证仅为两种形状的 legacy/MLIR Transpose→Add 正确性验证，不包含性能、a3 或上述失败组合。

后续泛化缺口的原因分析、修复方案与验证计划集中在[泛化缺口修复设计](2026-11-02-afir-transpose-generalization-fix-design.md)。该方案待实现，不改变本文已验收的 Inductor 迁移结论。
