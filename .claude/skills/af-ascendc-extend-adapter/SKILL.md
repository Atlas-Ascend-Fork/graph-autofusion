---
name: af-ascendc-extend-adapter
description: |
  graph-autofusion 自动融合 AscendC API Extend 改造辅助：参考单算子实现新增或修改 regbase Extend，并将 V2 codegen 适配到 Extend 接口；含 SIMT 实现的 impl 级内存预留（GetSimtDcacheSize）与精度提升黑名单回落配套。
  **必须触发的场景**：用户提到修改 ascendc api、适配 ascendc api、AscendC API Extend 改造、regbase Extend、参考单算子实现改造算子、把调用官方 AscendC API 改为调用 XxxExtend、为 Extend 增加/修改 SIMT 路径、精度黑名单/回落官方 API，或提到 PR #2096/#2109/#2121/#2141/#2305/#2409/#2421/#2877/#2902 中同类改造。
---

# Skill: af-ascendc-extend-adapter

辅助在 graph-autofusion 中完成"参考单算子实现，新增或改造 AscendC API regbase Extend，并适配 V2 codegen"的全流程。本 skill 处于持续完善中，遇到未覆盖的场景应记录并补充。

---

## 触发场景

用户请求包含以下任一意图时，必须使用本 skill：

- 修改 ascendc api、适配 ascendc api、AscendC API Extend 改造。
- 参考单算子（ops-math / ops-nn）实现，改造自动融合仓内的算子实现。
- 把 V2 codegen 对官方 AscendC API 的调用切换为 `XxxExtend`（regbase 内联实现）。
- 为已有 Extend 修改实现（整体替换或仅按 dtype 替换某条路径）。
- 涉及 SIMT 路径的 Extend 新增/修改（必须处理 DcacheSize 内存预留）。
- 高精度 Extend 替换官方 API 后需要性能保护（精度提升黑名单回落）。

不适用：纯 ASCIR 注册/新增算子流程走 `af-reg-ascir`；性能公式建模走 `af-perf-modeler`；编译构建执行走 `af-build-runner`。

---

## 素材与路径约定

- **单算子参考仓库**（优先由用户提供对应单算子及其位置；未提供时按下述规则定位并回报候选）：
  - https://gitcode.com/cann/ops-math（数学类，目录形如 `math/<op>`）
  - https://gitcode.com/cann/ops-nn（神经网络/激活类，目录形如 `activation/<op>`、`nn/<op>`）
  - 单算子 kernel 实现一般在 `<op>/op_kernel/arch35/` 下（DAG 头文件 + def cpp）。
  - **定位规则（硬性）**：
    1. 必须同时搜索 ops-math 与 ops-nn，不得取第一个命中；数学/激活类算子也可能只在 ops-nn（如 Erfinv 正式实现在 `ops-nn/activation/erfinv`）。
    2. **禁止使用 `experimental/` 目录下的实现作为参考**（可能标注 AI 生成、非正式交付）；只以正式目录 + `op_host/*_def.cpp` + aclnn 接口确认为准。
    3. 同名算子存在多处时优先 `op_kernel/arch35/` 正式实现，并回报用户“候选路径 / 选择理由”。
  - 反例：Erfinv 的 `ops-math/experimental/math/erf_inv/op_kernel/erf_inv.h` 不是正式单算子，不能作为参考。
- **自动融合仓内路径一律使用仓内相对路径**，关键落点：

| 路径 | 作用 |
|---|---|
| `autofuse/v35/ascir/generator/ascir_builtin_ops_v2.cpp` | ASCIR 注册表，前置检查 `REG_ASC_IR(OpName)` 是否存在 |
| `autofuse/v35/ascendc/api_regbase/<op>.h` | regbase Extend 实现（`XxxExtend` / `XxxSimtCompute` / `XxxImplVF`） |
| `autofuse/v35/ascir/generator/v2_ascir_codegen_impl.h` | Codegen impl：`GetApiName` / `GetApiCallName` / `LoadApiHeaderFiles` / `IncludeApiHeaderFiles` / `GetConversionDtype` / `CalcTmpBufSize` / `GetSimtDcacheSize`（SIMT 实现声明 d-cache） |
| `autofuse/inc/graph_metadef/graph/ascendc_ir/ascir_registry.h` | `AscIrCodegen` 基类虚方法 `GetSimtDcacheSize(node)`，默认返回 0（非 SIMT/未覆写继承默认值） |
| `autofuse/v35/codegen/ascendc_reg_base_api_register.cpp` | `*_reg_base.h` 字符串声明与 `api_to_file` 注册 |
| `autofuse/v35/ascendc/api_regbase/CMakeLists.txt` | regbase 头文件生成 raw-string |
| `autofuse/v35/ascir/reg_func/default_reg_func_v2.h` | `CalcVoidTmpSizeV2` 等 tmp buffer 计算 |
| `autofuse/common/ascgraph_info_complete.cpp` | API 信息补全：`CompleteElewiseApiInfo`/`CompleteGatherApiInfo` → `CompleteSimtDcacheSize` 询问当前架构选中 impl（`kOpTypeToComputeType`/`kComputeTypeToCompleter` 两级查表调用链）；**不在此处硬编码 d-cache 算子列表** |
| `autofuse/v35/ascir/generator/v2_ascir_codegen_impl.h` 内 `IsXxxInBlacklist()`（基于 `af::pre_process::PreProcessConfig::Instance().IsInImprovePrecisionBlacklist`） | 精度提升黑名单判定（现成参考：Sqrt/Rsqrt/Reciprocal） |
| `autofuse/mlir/api/ascendc/include/ascir/Dialect/Asc/IR/Autofuse/OpVecUnary.td` | MLIR 侧 apiName 映射；**由 MLIR 维护者维护，legacy 适配 PR 默认不要修改** |
| `autofuse/mlir/test/lit/conversion/afir-reg-only-math-api.mlir` | MLIR 生成代码期望；**默认不要修改** |
| `autofuse/mlir/docs/guides/ascir_op_lowering_matrix.csv` 与 `.md` | lowering 对照表；**默认不要修改** |
| `docs/zh/autofuse/introduction/environment_variables.md` 与 `docs/en/autofuse/introduction/environment_variables.md` | 精度提升黑名单环境变量文档（中英文都要同步） |
| `autofuse/tests/v35/ut/ascendc/api_regbase/test_<op>.cpp` | regbase UT（ICPU） |
| `autofuse/tests/v35/st/backend_e2e_v2/<case>_test/` | backend E2E |
| `autofuse/tests/framework/share_graph/` | share graph 构图 |
| `scripts/test/run_autofuse_test.sh` | 统一测试入口 |

---

## 总体流程

```text
Step 0 信息收集 → Step 1 前置检查 → Step 2 按情况分流执行 → Step 3 SIMT 专项检查 → Step 4 验证
```

### Step 0: 信息收集

一次性向用户确认：

1. **算子名**（PascalCase，如 FloorDiv）与意图（写 Extend / 改 Extend 实现 / 仅适配）。
2. **单算子参考**：来自 ops-math 还是 ops-nn，具体单算子目录或 PR；未提供时必须先在两个仓库搜索正式目录（排除 `experimental/`）并回报候选，不得凭文件名猜测或使用实验实现。
3. **参考范围**：整体替换实现，还是仅按 dtype/类型替换部分路径（如仅 int64），必须由用户明确。
4. **是否需要精度黑名单回落**：高精度 Extend 替换官方快速 API 时默认应提供（见"扩展场景"），由用户确认。
5. 是否需要同步补测试（默认需要）。

### Step 1: 前置检查（不可跳过）

1. **ASCIR 检查**：在 `autofuse/v35/ascir/generator/ascir_builtin_ops_v2.cpp` 检索 `REG_ASC_IR(OpName)`。
   - 不存在 → **明确告知用户该算子不支持**，流程终止；不得绕过注册表自行适配。
   - 存在 → 记录注册的 dtype 列表（后续修改不得超出该范围）。
2. **Extend 现状检查**（按判定口径，不能只看文件是否存在）：
   - 同时检查三件事：`autofuse/v35/ascendc/api_regbase/<op>.h` 中是否存在 `XxxExtend` 函数；`v2_ascir_codegen_impl.h` 的 `GetApiName` 是否返回 `XxxExtend`；`LoadApiHeaderFiles` 是否返回对应 `*_reg_base.h`。
   - **三者都满足**才算“已有 Extend”（情况二/三）；文件存在但函数或 `GetApiName` 是裸名 `Xxx` → **属于情况一（尚未适配 Extend）**，必须补 `XxxExtend` 并切换 `GetApiName`，不能只改实现。
   - 裸名合法例外（如 `Pow`/`Exp2`/`Log1p`）必须有仓内文档或代码注释明确说明；采用例外时必须在交付说明中写明原因与同步点。
   - 同时检查是否已覆写 `GetSimtDcacheSize`（SIMT 实现应为非 0，非 SIMT 保持基类默认 0）。
   - **若把裸名改为 `XxxExtend`**：legacy 侧必须同步 `GetApiName`、`LoadApiHeaderFiles`、UT/E2E 断言；**MLIR 侧文件由专人维护，默认不要改**。若 MLIR 现有映射仍固定旧名 `Xxx`，优先在 regbase 头保留 `Xxx` 兼容转发（注释说明 legacy 用 `XxxExtend`），或在 PR 中注明需 MLIR 维护者同步；只有与 MLIR 维护者确认后才修改 `.td`/lit/lowering matrix。
3. **注册现状检查**：`ascendc_reg_base_api_register.cpp` 的 `api_to_file` 是否已有 `{"<op>_reg_base.h", ...}`。

依据 2 的结果分流到 Step 2 的三种情况。

### Step 2: 三种情况

#### 情况一：仓内没有对应 Extend（或只有裸名实现）—— 参考单算子写 Extend 并适配

典型案例：PR #2305 中 Rsqrt/Reciprocal/Ceil/Acosh/Cosh（codegen 原先直接调官方 AscendC API，后切换为 `XxxExtend`）。

> 判定口径：`api_regbase/<op>.h` 存在但只有裸名 `Xxx`、且 `GetApiName` 返回裸名，也走本情况，不能按“已有 Extend”只改算法。

最小修改面：

1. **写 Extend**：在 `autofuse/v35/ascendc/api_regbase/` 新增/修改 `<op>.h`，参考单算子 `op_kernel/arch35/` 的实现逻辑。**按单算子参考决定实现形态**：
   - 向量高精度/特殊语义实现（MicroAPI/Reg 寄存器级）：如 Sqrt 的 `PRECISION_0ULP_FTZ_FALSE` 精确开方、Round 的 SIMD 实现 + BF16 NaN 规范化（PR #2421）——不涉及 SIMT，无需 Step 3 的 DcacheSize。
   - SIMT 逐元素实现：单算子 `asc_vf_call<Xxx_1<T>>(dim3(1024), ...)` 映射为 `AscendC::Simt::VF_CALL<XxxSimtCompute<T>>(AscendC::Simt::Dim3(N), ...)`，走 Step 3 全部配套。
2. **签名与 ApiCall 匹配**（否则生成 kernel 编译失败）：

| ApiCall | Extend 签名 | tmp 要求 |
|---|---|---|
| `UnaryApiCall` | `(dst, src, size)` | 无 |
| `UnaryApiTmpCall` | `(dst, src, tmp_buf, size)` | 需 `CalcVoidTmpSizeV2`（或专用 `CalcXxxTmpSizeV2`） |
| `BinaryApiCallV2` | `(dst, src1, src2, size)` | 无 |
| `BinaryApiTmpCall` | `(dst, src1, src2, tmp_buf, size)` | 需 `CalcVoidTmpSizeV2` |

   无 tmp 需求却切 `*TmpCall` 时，用 `CalcVoidTmpSizeV2`（0 大小占位），Extend 提供 tmp_buf 参数但不使用。

3. **codegen 适配**（`v2_ascir_codegen_impl.h`）：
   - `GetApiName`：`Xxx` → `XxxExtend`。
   - `GetApiCallName`：按签名是否带 tmp 切换（见上表）。
   - `LoadApiHeaderFiles`：返回 `{"<op>_reg_base.h"}`（项目内 regbase 头，实现源码内联进生成 kernel）。
   - `IncludeApiHeaderFiles`：Extend 内部调用的 CANN 官方头（如 `basic_api/reg_compute/kernel_reg_compute_intf.h`）；SIMT 时另见 Step 3。
   - `GetSimtDcacheSize`：Extend 为 SIMT 时在该 V2 impl 覆写并返回 `kSimtDcacheSize`（见 Step 3 第 6 项）；非 SIMT 不覆写，继承基类默认 0。
4. **注册**：`ascendc_reg_base_api_register.cpp` 声明 `kAscendcXxxRegBaseStr` 并加入 `api_to_file` map；确认 regbase CMake 生成 raw-string 头。
5. 若注册 dtype 中存在 Extend 不支持的类型，同步删减 `ascir_builtin_ops_v2.cpp` 注册 dtype，或走 `GetConversionDtype` cast（PR #2305 中 Reciprocal 删除 INT64/UINT64 即此情形）。

#### 情况二：已有 Extend —— 参考单算子修改 Extend 实现

典型案例：PR #2305 中 Remainder/Fmod/Mod 浮点路径改 Simt `fmodf` 精确算法；PR #2877 中 FloorDiv 仅 int64 切 SIMT。

> 先按 Step 1 确认确实“已有 Extend”（存在 `XxxExtend` 且 `GetApiName` 指向它）；文件存在但为裸名时不要按本情况处理。

1. **范围控制**：整体替换，还是仅替换某 dtype/路径（用户指定）。
   - 按类型替换用 `if constexpr (AscendC::IsSameType<T, int64_t>::value)` 分支，其余 dtype 保持原实现不动。
   - 单算子的语义修正逻辑（如 floor 语义 quot-1 修正、精度算法）照搬，同时保持本仓既有行为约定（如除 0 输出与原向量实现对齐）。
2. **行为一致性**：同算子不同 dtype 的边界行为（除 0、NaN/Inf）尽量对齐原实现；无法对齐时在方案中说明。
3. 修改后走 Step 3 检查 SIMT，走 Step 4 验证。

#### 情况三：已有 Extend —— 用户仅要求适配

典型案例：PR #2305 六算子 codegen 切换；PR #2096 中 Sinh/Asinh 接入已有 `SinhExtend`/`AsinhExtend`；PR #2421 中 Round 接线（本体 round.h 由独立 PR #2424 先行合入，本 PR 仅带 codegen 接线——**算子本体与 codegen 接线可分 PR 合入**的先例）。

1. 执行情况一中的第 3、4 步（codegen 适配 + 注册确认），不动 Extend 实现本体。
2. 适配前先读 Extend 实现确认：签名（是否带 tmp_buf）、是否 SIMT（走 Step 3）、支持的 dtype（是否需要删注册或 cast）。
3. **只改默认路径，保留既有旁路**：VfCall 融合、Inplace、MicroApi、SIMT 标量（`IsSimtScalarSupported`/`GenerateSimtScalarExpr`）等既有路径不动（PR #2421 Sqrt 即如此；Sqrt 还移除了自定义 scalar 方法改由基类 `SimtFloatUnaryAscIrCodegenImplV2` 的 `GetSimtScalarApiName` 等价提供）。

#### 扩展场景：高精度 Extend + 精度提升黑名单双路径

适用：用高精度 Extend（对齐单算子 0ULP/精确算法）替换官方快速 API 后，性能可能回退，需要可配置回落。典型案例：PR #2409（Reciprocal，develop 侧原 PR #2396）、PR #2421（Sqrt）。

在 codegen impl 中新增私有判定 `IsXxxInBlacklist()`（`af::pre_process::PreProcessConfig::Instance().IsInImprovePrecisionBlacklist("Xxx")`，支持配置 `Xxx` 或 `all` 通配；现成参考：`SqrtAscIrCodegenImplV2`/`ReciprocalAscIrCodegenImplV2`），并在五个方法中条件返回两条路径：

| 方法 | 非黑名单（高精度 Extend） | 黑名单（回落官方 API） |
|---|---|---|
| `GetApiCallName` | 按 Extend 签名选（带 tmp_buf → `UnaryApiTmpCall`，如 Reciprocal；不带 → `UnaryApiCall`，如 Sqrt） | 官方 API 对应形式（通常 `UnaryApiCall`） |
| `GetApiName` | `XxxExtend` | `Xxx`（Vector adv_api） |
| `LoadApiHeaderFiles` | `{"<op>_reg_base.h"}` | 空（回落官方无需 regbase 头） |
| `IncludeApiHeaderFiles` | Extend 依赖的官方头 | 与非黑名单共用一套（两种路径头依赖取并集，参考 Sqrt 实现） |
| `CalcTmpBufSize` | 仅 TmpCall 形式需要 `CalcVoidTmpSizeV2(node)` | 无 tmp |

配套：
- 文档同步：`docs/zh/autofuse/introduction/environment_variables.md` 与英文版补黑名单条目（同算子可与既有条目合并为一行）。
- codegen UT 覆盖四类用例：默认 Extend 路径、配置 `Xxx` 命中回落、`all` 通配回落、平台解析到 V2 impl（参考 PR #2409 的 `test_codegen_reciprocal_reg_api_call.cpp`）。
- MicroApi/Inplace/向量融合等既有旁路不受黑名单影响，保持不动（Sqrt 即如此）。

---

### Step 3: SIMT 专项（三种情况通用，Extend 含 `AscendC::Simt::VF_CALL` 时必须执行）

判定：Extend 实现或新写的路径使用了 `__simt_vf__` / `AscendC::Simt::VF_CALL`。

必须完成以下配套（第 1、2、6、8 项为硬性要求，3-5、7 项按实现内容适用）：

1. **regbase 写法模板**：

```cpp
constexpr uint32_t XXX_THREAD_NUM = 1024;

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(XXX_THREAD_NUM) inline void XxxSimtCompute(__ubuf__ T *dst,
                                                                                __ubuf__ T *src1, /* ... */
                                                                                const int64_t total_num) {
  for (int64_t i = threadIdx.x; i < total_num; i += blockDim.x) {
    // 参考单算子 __simt_vf__ 计算函数的逐元素逻辑
  }
}

template <typename T>
__aicore__ inline void XxxExtend(/* 签名与 ApiCall 匹配 */) {
  AscendC::Simt::VF_CALL<XxxSimtCompute<T>>(AscendC::Simt::Dim3(XXX_THREAD_NUM), (__ubuf__ T *)dst.GetPhyAddr(),
                                            (__ubuf__ T *)src1.GetPhyAddr(), /* ... */ size);
}
```

2. **codegen 头**（按最小依赖原则选择）：
   - 优先 `"simt_api/cpp/kernel_simt_utils.h"`（`VF_CALL`/`Dim3` 声明）+ 用到 libm 函数时 `"simt_api/math_functions.h"`——PR #2121 已将 Sin/Cos/Acos/Asin 从重量级 `kernel_simt_intf.h` 精简到这两个头（`kernel_simt_intf.h` 会拉入 atomic/cast/cmp/print/warp/bessel 全家桶）。
   - 仅当确需 `Simt::` 封装接口时才用 `kernel_simt_intf.h`（PR #2096 早期模式，后续演进已替代）。
   - `threadIdx`/`blockDim`/`__simt_vf__`/`LAUNCH_BOUND` 为编译器内建，无需专门头。

3. **SimtCompute 内核直接调用标准数学函数**（PR #2109/#2121 演进方向）：`sinf/cosf/tanf/acosf/asinf/atanhf/sqrtf/rintf/fmodf` 等直接调用，而不是 `Simt::Sin` 等封装。注意语义对齐：如 Round 用 `rintf` 保持官方 round-half-to-even（PR #2421）。

4. **dtype 转换**：上述数学函数仅接受 float → `GetConversionDtype` 需含 `{DT_FLOAT16 -> DT_FLOAT}`（BF16→FLOAT 通常已有；PR #2096/#2421 均补过此项）。

5. **libm 精度陷阱与分段修正**（PR #2421 Atanh 案例）：`atanhf = ln((1+x)/(1-x))` 在零附近丢有效位 → `|x| < 0.125` 改用泰勒展开 `x + x^3/3 + x^5/5 + x^7/7`。**对齐单算子参考实现的分段算法，勿默认信任 libm 公式的全域精度**。

6. **单算子实现确认（动手前必做）**：阅读单算子 `op_kernel/arch35/` 的实际调用（如 `erfinvf`/`tanf`/`fmodf`）及其 DAG 组合（Cast→自定义算子→Cast），确认算法、分段和特殊值处理；不要用其他仓库或 `experimental/` 的同名实现替代。Erfinv 应采用 ops-nn `activation/erfinv` 的 SIMT `erfinvf`，而不是 ops-math `experimental` 里的多项式实现。

7. **精度容差设定**：期望值必须按实际 float32 输入计算（`erfinv(double(float32(x)))` 后再舍入到 float32），不能用十进制字面量的高精度结果；近 ±1 导数大，字面量表示误差会被放大（Erfinv `0.9999F` 约 2.9e-5）。容差按单算子算法的实测误差设定：可用 host 脚本按单算子源码逐位仿真，Erfinv 最大绝对误差约 5.4e-7，`1e-5` 约有 20 倍余量；无上板证据时沿用 `1e-4`（与 i0/bessel 等特殊函数 ST 一致），禁止盲目收紧。

8. **DcacheSize 内存预留**（防 SIMT 多线程访问踩内存，40KB）：d-cache 是**实现（impl）属性**，由当前架构选中的 codegen impl 声明，**禁止**在 `ascgraph_info_complete.cpp` 按算子名硬编码（PR #2902 已删除硬编码列表及 `kSimtDcacheSize`）。
   - 在 `v2_ascir_codegen_impl.h` 的对应 V2 impl 覆写：
     ```cpp
     [[nodiscard]] int64_t GetSimtDcacheSize([[maybe_unused]] const AscNode &node) const override {
       return kSimtDcacheSize;
     }
     ```
     - **全部 dtype 走 SIMT** → 无条件返回 `kSimtDcacheSize`（Remainder/Fmod/Mod/Gather/Sin/Cos/Asin/Acos/Tan/Atanh/Expm1 模式，PR #2902）。
     - **仅部分 dtype 走 SIMT** → 条件返回，如 FloorDiv 仅 int64：
       ```cpp
       return GetSimtInputDtype(node) == DT_INT64 ? kSimtDcacheSize : 0;
       ```
     - **向量实现**（Sqrt 0ULP/Round SIMD）不覆写，继承基类默认 0。
   - 常量 `kSimtDcacheSize = 40 * 1024` 定义在 `v2_ascir_codegen_impl.h`；基类虚方法 `AscIrCodegen::GetSimtDcacheSize` 在 `ascir_registry.h`，默认返回 0。
   - 调用链：`CompleteElewiseApiInfo`/`CompleteGatherApiInfo` → `CompleteSimtDcacheSize(node)` → `GetAscIrCodegenImpl(node->GetType())` 取当前 SoC 选中的 impl → 仅 `GetSimtDcacheSize > 0` 才 `SetDcacheSize`。因此**架构感知自动完成**：A5/3510 选中 v2 impl 打标记；A3（2201）选中 v1 实现继承默认 0，不打标记，既避免启动参数被解释为 `cce_launch_config.smdesc`（`void*`）导致设备编译失败，也避免 tiling 白扣 40KB UB。
   - 机制：DcacheSize 参与 UB 表达式构建与 codegen 的 dynamic ub_size 扣减（`ub_size - dcache`），是图级预留——图内出现一个 SIMT 节点即整图预留。
   - 若该算子类型不在 `kOpTypeToComputeType` map 中，仍需补注册为 `kComputeElewise`（如 Tan/Atanh，PR #2096；FloorDiv 属存量注册），否则走不到 `CompleteElewiseApiInfo`，d-cache 也不会生效。注意这只决定“能否补全信息”，预留大小仍由 impl 声明。
   - 注意：向量实现的 Extend（Sqrt 0ULP/Round）不设 DcacheSize；以最终合入代码为准，勿轻信 PR 描述文本（#2421 早期版本曾为 SIMT，终版改向量实现）。

验证锚点：v2 平台 E2E 生成器断言 `tiling.find("40960")`；生成 kernel 断言 `XxxSimtCompute` 内联与 simt 头 include。A3/2201 平台反向断言：无 40960，启动语句为 `<<<blockDim, nullptr, stream>>>`（PR #2902 回归用例 `CompleteApiInfoSkipsDcacheSizeOnV1Platform`）。涉及 d-cache 的 optimize UT 需显式 `SetPlatform("3510")`（v2，断言 40KB）或 `SetPlatform("2201")`（A3，断言不打标记），不能依赖默认平台（PR #2902 同步修正了 Gather/Expm1 既有用例）。

---

### Step 4: 验证

1. **编译**（必须限制并行度）：

```bash
cmake --build build --target aihac_codegen -j 8
```

2. **regbase UT**（dav_3510 CPU 仿真；整数与多数 SIMT 标量可本地验证，必须实际运行 `--gtest_filter` 并记录结果；旧环境或个别超越函数可能返回 0/报 `exec unknown instr`，此时才标注需上板或 CI 验证并给出证据）：

```bash
cmake --build build --target test_ascendc_api_v35 -j 8
./build/autofuse/tests/v35/ut/ascendc/test_ascendc_api_v35 --gtest_filter='*Xxx*'
```

   实测 Erfinv SIMT UT 在本地 dav_3510 仿真器 3 个用例全部通过；不要因“ICPU 浮点恒 0”的旧结论跳过 UT。

3. **E2E**：修改 Extend 行为或适配切换必须补 backend E2E（`<op>_<dtype>_store_test` 三件套 + share graph + 父 CMake + `scripts/test/run_autofuse_test.sh` 目标），参考 `autofuse/tests/v35/st/backend_e2e_v2/remainder_int32_store_test/` 结构；生成器断言 Extend 名、Simt 函数名、simt include、40960 预留（仅 v2 平台；A3 平台断言无 40960）；expect kernel 用独立参考实现比对，禁止用被测 API 自证；容差按单算子算法特性设定（见 Step 3 第 7 项）。**生成器注意**：backend E2E 生成器常用 `RemoveSubDirInclude()`，它会删除 `simt_api/...` 头；若 API 头直接调用 SIMT scalar（如 `erfinvf`），必须在 `#include "kernel_operator.h"` 之后补回 `#include "simt_api/math_functions.h"`，并断言 codegen 输出包含该头，否则生成 kernel 编译报 `'xxx' was not declared`（PR #3047）。本地可构建 `<case>_e2e_v2` 目标验证生成 kernel 编译；SIMT E2E 数值在部分本地模拟器会报 `scalar_pv.cc exec unknown instr`（现有 `cos_bf16_test_e2e_v2` 同样失败），此时以 CI/上板结果为准，不能据此改实现。
4. **生成 kernel 检查**：确认 Extend 源码内联、tmp_buf 参数与 ApiCall 匹配、include 正确（缺官方头只在真实 Device 编译暴露）。
5. **回归**：同算子其它 dtype 的既有 UT/E2E 必须回归通过（确保按类型修改未破坏其它路径）。
6. `clang-format -i` 所有修改的 C++ 文件。
7. **环境不可用时的最小验证**（缺 protoc/protobuf、无 NPU 等）：至少编译到测试 object/目标；条件允许时用 host 脚本按单算子算法复算关键点（含边界）；在交付说明中明确“未上板/未跑全量”并列出待验证项，禁止声称已完成数值验证。
8. **单算子对照说明**：在 PR/交付说明中写明参考的具体单算子路径与算法（例如 `ops-nn/activation/erfinv` → `erfinvf`），并说明 Extend 与其在 dtype、cast、特殊值上的对齐关系；禁止只写“参考单算子”而不给路径。
9. **MLIR 兼容说明**：若改名或改签名影响 MLIR 现有 apiName 映射，必须在交付说明中写明兼容策略（regbase 头保留旧名转发，或已与 MLIR 维护者确认同步），不得静默改动 MLIR 文件。

---

## 风险清单

- Extend 签名与 ApiCall 参数不匹配 → 生成 kernel 编译失败。
- 只看 `api_regbase/<op>.h` 是否存在就判为“已有 Extend”，导致 codegen 仍调裸名 → 实际未适配，PR 返工。
- 参考 `experimental/` 或 AI 生成实现作为单算子 → 算法/语义错误（Erfinv 曾误用 ops-math `experimental/math/erf_inv`）。
- 未经确认直接修改 MLIR `OpVecUnary.td`/lit/lowering matrix → 与 MLIR 维护者冲突；应保留兼容别名或交由 MLIR 维护者同步。
- 裸名改为 `XxxExtend` 后未在 regbase 头保留兼容转发，导致 MLIR 现有映射生成旧名调用 → MLIR 路径编译失败。
- 把十进制字面量的 erfinv 结果当作期望值 → 近 ±1 处字面量表示误差被导数放大（`0.9999F` 约 2.9e-5），误判为算法误差。
- 容差未按单算子算法实测误差设定 → 过紧 CI 失败，过松掩盖精度问题。
- E2E 生成器 `RemoveSubDirInclude` 删掉 `simt_api/math_functions.h`，SIMT scalar（如 `erfinvf`）未声明 → 生成 kernel 编译失败（PR #3047）。
- SIMT 实现未在对应 V2 impl 覆写 `GetSimtDcacheSize`（或误把覆写放到基类/v1 impl）→ v2 平台不预留，上板可能踩内存/数值错误，且本地 ICPU 难以暴露。
- 在 `ascgraph_info_complete.cpp` 按算子名硬编码 DcacheSize → A3 上 v1 实现也被打标记，启动参数被当作 `cce_launch_config.smdesc`（`void*`）导致设备编译失败且白扣 40KB UB（PR #2902 根因）。
- 算子类型不在 `kOpTypeToComputeType` → 走不到 `CompleteElewiseApiInfo`，d-cache 不生效且调度单元错乱。
- regbase 头未在 `ascendc_reg_base_api_register.cpp` 注册 → 生成 kernel 找不到内联源码。
- `IncludeApiHeaderFiles` 漏官方头/simt 头，或引入超出需要的重量级 simt 头（`kernel_simt_intf.h`）→ 后者增加 Device 编译负担。
- 按类型修改时误动其它 dtype 路径 → 回归缺失导致隐性破坏。
- libm 函数全域精度陷阱（如 `atanhf` 零附近丢位）未按单算子分段修正 → 精度不达标；ICPU 上浮点超越函数输出恒 0 无法发现。
- 黑名单回落只改了部分方法（如漏 `CalcTmpBufSize`/`LoadApiHeaderFiles`）→ 回落路径生成 kernel 编译失败；文档中英文遗漏同步。
- ATT perf 公式未同步 SIMT（VF 指令建模 vs SIMT 线程模型）→ 估算偏差；当前仓内既有模式允许暂不更新，但必须在交付说明中标注，并建议后续用 af-perf-modeler 建模。

---

## 案例索引

| PR | 情况 | 要点 |
|---|---|---|
| #2096 | 情况三 + SIMT | Tan/Atanh 写 Simt Extend 并适配；Sinh/Asinh 接入已有 Extend；DcacheSize 40KB、`kOpTypeToComputeType` 补 Tan/Atanh 注册、`FLOAT16→FLOAT` 转换、早期 `kernel_simt_intf.h` 头模式 |
| #2109 | 情况二 | Tan/Asin/Atanh SimtCompute 由 `Simt::Xxx` 封装改为直接调用 `tanf/asinf/atanhf` |
| #2121 | 情况二 | Sin/Cos/Acos SimtCompute 直接调用 `sinf/cosf/acosf`；SIMT 头精简为 `kernel_simt_utils.h` + `math_functions.h`（替代 `kernel_simt_intf.h`） |
| #2141 | 跨分支同步 | #2096/#2109/#2121 三个提交原样 cherry-pick 到 9.2.0 分支（保留原始提交信息，diff 校验） |
| #2305 | 情况一 + 情况二 | Rsqrt/Reciprocal/Ceil/Acosh/Cosh/Tanh codegen 从官方 API 切 `XxxExtend`（ApiCall/TmpBuf/注册表；Reciprocal 删 INT64/UINT64 注册）；Remainder/Fmod/Mod 浮点改 Simt fmodf 精确算法，Dcache 从仅 INT32 扩到全部 dtype |
| #2409 | 扩展场景（9.2.0 同步） | Reciprocal 适配 Extend + 精度黑名单五方法双路径回落官方 API（develop 原型 #2396）；codegen UT 四类用例（默认/黑名单/all 通配/平台解析） |
| #2421 | 情况一 + 扩展场景 | Sqrt 0ULP 高精度**向量** Extend（`PRECISION_0ULP_FTZ_FALSE`，非 SIMT 不设 Dcache）+ 黑名单回落；Round 接线（本体由 #2424 先行合入，BF16 NaN 规范化 SIMD 实现）；Atanh 泰勒展开分段修正对齐单算子；黑名单文档中英文同步 |
| #2877 | 情况二（按类型）+ SIMT | FloorDiv 仅 int64 参考单算子 `FloorDivInt_1` 切 SIMT；DcacheSize 按 dtype 条件预留；regbase UT + E2E 全链路验证 |
| #2902 | 机制重构 | SIMT DcacheSize 从 common 层硬编码改为 impl 属性：`AscIrCodegen::GetSimtDcacheSize` 默认 0，12 个 v2 SIMT impl 声明 40KB（FloorDiv 仅 int64）；A3 选中 v1 impl 不预留，修复 `<<<blockDim, <int>, stream>>>` 被当作 `smdesc` 的编译失败 |
| #3047 | 情况一（裸名误判）+ SIMT | Erfinv 原文件存在但只有裸名 `Erfinv`，曾被误判为“已有 Extend”；最终参考 ops-nn `activation/erfinv`（SIMT `erfinvf`）重写 `ErfinvExtend`，同步 legacy codegen/UT/E2E 并覆写 `GetSimtDcacheSize`；**不改 MLIR**，在 regbase 头保留 `Erfinv` 兼容转发；E2E 期望值按 float32 输入计算，生成器在 `RemoveSubDirInclude` 后补回 `simt_api/math_functions.h` |

持续完善本表：完成新的 Extend 改造后，将 PR、情况、差异点追加至此。
