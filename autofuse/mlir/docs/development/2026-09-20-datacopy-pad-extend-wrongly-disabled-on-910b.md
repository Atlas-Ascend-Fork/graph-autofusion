# 910B 被误判为不支持 DataCopyPadExtend

> 状态：已关闭，且本文记的改法**已被更彻底的做法取代**。关闭范围是 V2201/910B
> capability 误判；二维寻址方案仍按 S1～S5 分期实施。
>
> 本文第 5 节的改法是「把 `TargetFeature::DataCopyPadExtend` 在 V2201 上改判成
> `Available`」，也就是保留这一档 feature、只修它的取值。上游 `6dcac571` 走得更远：
> 既然两张供给表都提供这个 wrapper、legacy 两侧都无条件发它，那么**这一档 feature
> 本身就不该存在**——feature、按它分流的第二个 implementation（`load.gm_ub.
> data_copy_pad_extend`）、以及选不中时回落的裸 `AscendC::DataCopy` 路径，一并删除，
> 今天一律发 `DataCopyPadExtend`。
>
> 诊断部分（第 1～4 节）不受影响：根因与证据链完全成立，`AscendTargetProfile.cpp`
> 里那条「这里曾经还有一项 `DataCopyPadExtend`」的注释记的就是同一个根因。只有第 5
> 节的改法和第 6 节的验证对象换掉了。随之失效的还有当时配套的两个用例
> （`afir-transfer-no-pad-extend-rejected.mlir` 与 `Inputs/AscendNoPadExtend.ini`）：
> 它们测的是回落路径，被测对象已不存在，已删除。

## 1 结论先说

`TargetFeature::DataCopyPadExtend` 在 V2201（910B）上被判成 `Unavailable`，**这个判定是错的**。

两张扩展 API 供给表都提供这个 wrapper，六个函数参数完全一致；legacy 的 `LoadApiCall` /
`StoreApiCall` 对所有目标无条件发这个调用，910B 上一直是这么跑的。

后果是 910B 上的搬运退回 `AscendC::DataCopy`（只有一个长度参数）：

- **带 stride 的二维搬运无法表达** —— reduce 切 R 轴之后 Load 的 tile 是二维的，
  在 910B 上没有可用的发射形态。这是二维寻址的**硬阻塞**。
- **非对齐尾部被截断** —— legacy 发的是 `DataCopyPad`（`isPad = true`），MLIR 发的是
  `DataCopy`。37 个 i32（148B）的 CPU 仿真已确认两者对有效尾部的行为不同；
  pad 位的具体填充值尚不能由该模拟器确认，见第 4 节。

本文记录问题定位与修复结果。910B 的 wrapper 能力已由
`test_datacopy_pad_extend.cpp` 在 CANN 9.2.0 `tikicpulib_ascend910B1` 环境验证；
GM→UB、UB→GM 两个方向的多 block、非零 stride 用例均通过。随后在 910B 设备
`Ascend910_9362` 上验证了双向非零 stride 与 37 个 i32 的非对齐尾部，见第 4 节。

## 2 判定在哪，怎么来的

### 2.1 位置

`autofuse/mlir/target/Ascend/AscendTargetProfile.cpp:113-119`：

```cpp
case TargetFeature::DataCopyPadExtend:
  // reg-base 家族才有 `DataCopyPadExtend` 这个 wrapper。
  return identity_.npuArch == NpuArch::V3510   ? FeatureState::Available
         : identity_.npuArch == NpuArch::V2201 ? FeatureState::Unavailable
                                               : FeatureState::Unknown;
```

配套的 enum 注释（`autofuse/mlir/target/Ascend/AscendTargetProfile.h:76-80`）：

> The v35 reg-base DataCopyPadExtend wrapper is available on the V3510 instruction family.

这张表是该 feature **今天唯一的来源**：`TargetProfileLoader` 的 `ReadFeatures` 只读
`support_bf16` 与 `support_inf_nan`，profile 文件无法覆盖它。

### 2.2 三个 commit 的沿革

| commit | 作者 / 日期 | 做了什么 |
|---|---|---|
| `df138bc8` | liyuewei，2026-08-29 | **引入判定**。原注释写明这是「profile 里还没有这个字段，先按指令族兜底」的**临时回落** |
| `988c9b81` | benjoe，2026-09-17 | 抽成 `getFeatureStateFromArch`，首次写下「reg-base 家族才有这个 wrapper」 |
| `399f314c` | benjoe，2026-09-17 | 重写外层注释 |

值得单独指出：**后两次重写让论据比原来更糟**。

- `df138bc8`：「先按指令族兜底」—— 自认是回落，没有断言事实
- `988c9b81`（外层）：「`DataCopyPadExtend = Available` 的意思是**要用扩展 wrapper**」
  —— 读作**策略选择**，这个说法站得住
- `399f314c`（外层）：「一个问**这个 wrapper 在不在**」—— 转成**事实断言**，而该事实为假

内层那句「reg-base 家族才有」自 `988c9b81` 起未变。三次经手，**没有一次打开过
`autofuse/ascendc/api/datacopy.h`**。

## 3 证据

### 3.1 两张表都有，同名同签名

| | mem-based（910B 走这张） | reg-based（V3510） |
|---|---|---|
| 文件 | `autofuse/ascendc/api/datacopy.h:14` | `autofuse/v35/ascendc/api_regbase/datacopy.h:14` |
| 签名 | `DataCopyPadExtend<T>(dst, src, block_count, block_len, src_stride, dst_stride)` | `DataCopyPadExtend<T, PaddingMode mode = Normal>(`**同样 6 个参数**`)` |
| 落到 | `AscendC::DataCopyPad` | `AscendC::DataCopyPad<T, mode>` |
| 重载 | 2（UB←GM、GM←UB） | 4（各方向多一个 `LoopModeParams` 重载） |

reg-base 那份只是多了一个**有默认值的模板参数**和一个额外重载。调用形式完全兼容。

`autofuse/CMakeLists.txt:69` 的 `include_directories(${CODE_ROOT_DIR}/ascendc/api)`
说明 mem-based 那张表就是主构建的全局 include 路径。

### 3.2 legacy 在 910B 上无条件发这个调用

v1（mem-based / 910B）的 codegen 声明：

```cpp
// autofuse/ascir/generator/v1_ascir_codegen_impl.h:98-108
class LoadAscIrCodegenImpl : public AscIrCodegen {
  std::string GetApiCallName() const override { return "LoadApiCall"; }
  std::vector<std::string> LoadApiHeaderFiles(bool) const override { return {"datacopy.h"}; }
```

而 `LoadApiCall::Generate` 里：

```cpp
// autofuse/codegen/api_call/datacopy/load_api_call.cpp:48
ss << "DataCopyPadExtend(" << ub << ", " << gm << "[" << gm_offset << " + " ...
```

**没有任何 arch 分支。** Store 同理（`store_api_call.cpp:64`，`v1_ascir_codegen_impl.h:1632-1642`）。

### 3.3 v35 同时提供普通 reg lowering 与 micro lowering

v35 的 Load/Store 同时声明两套路径：

- 普通 reg lowering 是 `LoadRegApiCall` / `StoreRegApiCall`，API 名为
  `DataCopyPadExtend`，扩展头为 `datacopy_reg_base.h`；对应实现和注册分别位于
  `reg_load_api_call.cpp` 与 `reg_store_api_call.cpp`。
- micro/vector-function lowering 是 `MicroLoadApiCall` / `MicroStoreApiCall`，API 名为
  `Load` / `Store`。

存在 micro 路径不代表普通 reg 路径不存在。这里原先把路径选择与实现供给混为一谈；
正确结论是 mem-based 与 reg-base 的普通搬运路径都具备 `DataCopyPadExtend` wrapper。

### 3.4 lowering matrix 是生成的实现声明快照，不是能力判定

`autofuse/mlir/docs/guides/ascir_op_lowering_matrix.csv:94,161` 的
`LoadRegApiCall` / `StoreRegApiCall` + `datacopy_reg_base.h` 与当前源码一致。
该文件由 `scan_ascir_op_lowering_matrix.py` 从 v1/v2 codegen 类的 `GetApiName`、
`GetApiCallName`、扩展头、tmp buffer 和 scalar 约束生成；重新生成结果与仓库文件完全一致。
它不参与运行时 feature gate，也不描述 normal/micro 路径的最终选择。

需要修正的陈旧记录是 `OpDataCopy.td:117-122`：其 op 注释称
`AscendC_DataCopyPadExtendOp` 是
  「v35 reg-base wrapper operation」，「its target emitter prints the wrapper from
  api_regbase/datacopy.h」—— 同一个误解。而它声明的头文件名
  `ApiExtensionHeader<"datacopy.h">` 在两张表里**重名**，光看名字分不出走的是哪一份。

## 4 影响

### 4.1 二维搬运无法表达（确定）

修复前，910B 选中的是另一套实现（`DataMovement.cpp:271-276` 注册了两条：
`load.gm_ub` 与 `load.gm_ub.data_copy_pad_extend`），发
`ascendc.data_copy_l2` → `AscendC::DataCopy`（`OpDataCopy.td:86`），**只有一个长度参数**。

reduce 切 R 轴后 Load 的 tile 是 `[z0t, z1t]` 二维，需要
`blockCount` / `blockLen` / 两个 stride 四个参数。910B 上没有承载它的发射形态。

**这是修复前 reduce 二维寻址在 910B 上的硬阻塞**，也是把这条从二维方案里拆出来
单独修复的理由。

### 4.2 非对齐尾部：有效数据差异与 9362 实机读回

`autofuse/ascendc/api/datacopy.h:24` 传的是 `pad_params = {true, 0, 0, 0}`，即
`isPad = true`；`AscendC::DataCopy` 按 32B 块搬，没有这个语义。

原先注册的 CPU E2E 用例使用 `[128]`，f32/i32 下 512 字节天然 32B 对齐。
现新增 37 个 i32（148B，补齐到 160B）的 910B tikicpu API case：

- `DataCopyPadExtend` 的 37 个有效元素全部与 GM 输入一致。
- 普通 `DataCopy(dst, src, 37)` 在 tikicpu 中报错：count 必须 32B 对齐；
  诊断明确称 NPU 模式不会报错，而是向下取整到 32B。case 用合法的
  `DataCopy(dst, src, 32)` 验证对照路径只保留前 32 个元素，余下 5 个有效值
  保持初始化的 sentinel。实际硬件截断行为仍未在真机复核。
- 本次模拟中，补齐的 3 个 pad 位读到 `1` 而非预期的 `0`；生成的 CCE 包含
  `set_mov_pad_val`，但 npuchk 日志将它标成 `unsupported`。因此测试只断言
  **37 个有效值不丢失**，不将模拟器的 pad 填充值视为硬件语义。

2026-09-21 经 `npussh` 在 CANN 9.2.0、`aclrtGetSocName() == Ascend910_9362`
的实机（ACL 逻辑设备 0）运行独立 AscendC 用例
`autofuse/tests/st/ascendc/api/datacopy_pad_extend/`，使用仓库中的 mem-based
`datacopy.h`：

- GM 输入前 37 个 i32 为 `1..37`，后 3 个为 `-12345`；UB 先填 `-12345`。
  `DataCopyPadExtend(dst, src, 1, 37, 0, 0)` 输出前 37 个全部为 `1..37`。
- 合法的对照调用 `DataCopy(dst, src, 32)` 输出前 32 个为 `1..32`，
  索引 32..39 仍为 `-12345`。没有在硬件上直接调用非法的
  `DataCopy(dst, src, 37)`，因此不能把此对照描述成该调用的实机截断结果。
- 本次 9362 实机读回的 pad 索引 37..39 都为 `1`，与 tikicpu 观察一致，
  但不据此断言所有 dtype、参数组合或其他设备的 pad 填充值。有效元素与对照断言通过。
- 同目录的 stride 用例以 3 个 block、每块 8 个 i32、间隔 3 个 i32 验证
  GM→UB 与 UB→GM；连续区数据及 GM 间隔 sentinel 均符合预期，双向断言通过。

当前部署清单将 `Ascend910_9362` 归为 910B 设备；因此上述结果构成 910B 实机证据。
记录中仍保留 runtime 返回的精确 SoC 名，避免把结论无边界地外推到其他 profile。

## 5 改法与已知代价

### 5.1 判定本身

```cpp
case TargetFeature::DataCopyPadExtend:
  // 两张供给表都提供这个 wrapper，六个函数参数一致（reg-base 那份多一个有默认值的
  // PaddingMode 模板参数和一个 LoopModeParams 重载）。legacy 的 LoadApiCall /
  // StoreApiCall 对所有目标无条件发这个调用，910B 上一直是这么跑的。
  return identity_.npuArch == NpuArch::V3510 || identity_.npuArch == NpuArch::V2201
             ? FeatureState::Available
             : FeatureState::Unknown;
```

同时改掉 `AscendTargetProfile.h:76-80` 与 `OpDataCopy.td:117-124` 的注释。
`ascir_op_lowering_matrix.csv` 是与源码一致的生成物，不修改。

### 5.2 wrapper 保持不变：沿用既有命名空间前导

```cpp
// autofuse/ascendc/api/datacopy.h:17, 32
uint32_t align_num = ONE_BLK_SIZE / sizeof(T);
```

该头来自 `autofuse/ascendc/api/` 上游，既有约定是调用方先引入 `AscendC` 命名空间。
MLIR 的 `AddKernelIncludes` 在存在扩展头时，会先发出 `using namespace AscendC;`，再由
扩展 API inliner 展开 `datacopy.h`；生成的 kernel 已验证这个顺序。legacy codegen 也
沿用同一前导。因此 `ONE_BLK_SIZE` 是否带限定与 910B feature gate 无关。

独立 API UT/ST 同样把 `using namespace AscendC;` 放在 `datacopy.h` 之前。本次不把
“头文件可脱离既有前导独立 include”扩进 feature 修复，也不修改上游同步文件。

### 5.3 MLIR CPU E2E 的扩展头处理

CPU E2E 的 `AddKernelIncludes` 先记录 `datacopy.h` 依赖，后续扩展 API inliner 将
wrapper 头文本直接嵌入生成的 `kernel.cpp`，因此独立 target 不需要额外的
`${CODE_ROOT_DIR}/ascendc/api` include 路径。生成代码编译验证了这一机制。

### 5.4 连带变更的测试

910B 是默认目标，打开之后搬运的发射形态整体改变：

- `autofuse/mlir/test/ut/target/AscendTargetProfileTest.cpp` 已将 V2201 的
  `Unavailable` 预期改为 `Available`，并检查 `hasFeature`。
- `afir_pipeline_sync_canonical` 与 `afir-api-extension-include` 的 lit 预期已更新，
  覆盖 IR 中的 `data_copy_pad_extend` 及生成 C++ 中的 `datacopy.h`/wrapper 调用。

这个体量是把本条与二维寻址分开做的主要理由：**"打开 API"和"接二维"的失败应该能分开定位。**

### 5.5 include 机制：已不需要处理

历史设计文档（`2026-09-17-afir-op-completion-elementwise-slice-design.md:100-102`）记录过
「`Translation.cpp:76` 在见到 `DataCopyPadExtendOp` 时硬编码发 `#include "datacopy.h"`，
今天无人解析」。**该记录已陈旧**：当前 `DataCopyPadExtendOp` 通过
`ApiExtensionOpInterface` 声明头文件（`OpDataCopy.td:131`），由
`AddKernelIncludes.cpp:79` 收集并排在 `kernel_operator.h` 之后。这一条不需要再改。

## 6 修复与验证结果

- `AscendTargetProfile` 现在将 V2201 与 V3510 都判为 `Available`；未知家族仍为
  `Unknown`，显式 profile 覆盖语义不变。
- mem-based `datacopy.h` 保持上游原样；MLIR 与独立用例均遵守其既有命名空间前导。
- MLIR CPU E2E 使用既有扩展头内联机制，无须修改 target include 路径。
- `OpDataCopy.td` 已改为描述两套 API 家族共享的 target wrapper。
- 新增 910B AscendC API UT，覆盖 GM→UB、UB→GM、多 block、非零 stride 与
  37 个 i32 的非对齐尾部。
- CANN 9.2.0 `tikicpulib_ascend910B1` 下三条 API UT 均通过；
  `TargetProfileTest.*:TargetProfileLoaderTest.*` 共 41 条 MLIR UT 与两条受影响的
  lit 均通过。
- `check-e2e-cpu-load_add_storex128` 通过，生成代码成功内联 wrapper、编译并
  在 tikicpu 上得到正确结果。
- `Ascend910_9362` 实机独立用例通过，37 个有效值完整，32 元素对照路径
  只覆盖前 32 个；三个 pad 位本次读回均为 `1`（见 4.2）。

仍未覆盖：

- **没有查 `DataCopyPadExtend` 在 910B 硬件上的行为边界**，例如 `blockCount` /
  stride 的取值上限。当前实机用例证明了基本的非零 stride 与非对齐尾部，不覆盖
  极限参数组合。
- **MLIR 生成的非零 stride kernel 全链路属于二维寻址方案 S3 的验收项，尚未完成。**
  这不影响关闭“910B capability 被误判”本身，但二维寻址总任务仍处于实施中。

关闭依据：能力事实、feature 推断、两目标发射、910B CPU 仿真和 910B 实机 API 行为
均已形成证据闭环；剩余项是 API 极限边界与二维寻址后续阶段，不再属于本缺陷的关闭条件。
