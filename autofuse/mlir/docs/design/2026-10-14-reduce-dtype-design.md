# 简介

## 目的

补齐普通 legacy memory-based 路径已有、AFIR 已能表达的 reduce dtype，供实现与检视使用。

## 范围

Sum 新增 i32；Max/Min 新增 f16；七类 op 的 f32 行为保持。排除 ArgMax、v35 regbase、Mean 跨轮累加和布局放宽。

# 总体概述

## 软件概述

### 项目介绍

AFIR ODS 已接受上述类型，但 Reduce capability 和 rewrite 仅接受 f32。

### 产品环境介绍

在 CANN 9.2.0 memory-based API 上验证。Max/Min 高阶 API 接受 half；整数 Sum 使用 Autofuse 的 ReduceSumInt32 扩展头。

## 软件功能

支持新 dtype 的 AR/RA 下降、按类型计算对齐、既有跨轮合并与代码发射。

## 设计约束

复用调度器 scratch、既有二维布局判据和 ApiExtension 接口；不通过 float 转换实现整数和，不修改外部 ABI。

## 假设和依赖关系

输入、输出类型遵守 AFIR verifier 的一致性约束。资源规划沿用 legacy CalcReduceTmpSize；目标需支持既有 SIMD memory-based 执行形式。

# 需求分析与设计

## 整体介绍

按 op 判 dtype → 按元素宽度证明物理布局 → 选择 CANN 或 Autofuse API → 既有同步和跨轮合并 → 收集扩展头并发射。

## 功能需求

### 功能需求 1：按算子支持 dtype

1. 介绍：Sum 接受 f32/i32，Max/Min 接受 f32/f16，其余四类保持 f32。
2. 输入：规范 buffer reduce、descriptor、调度器 scratch。
3. 处理：能力查询与 rewrite 共用类型判据；i32 Sum 选择独立的 ReduceSumInt32Op。
4. 输出：i32 发未限定的 ReduceSumInt32；half 发 AscendC::ReduceMax/ReduceMin；不支持的组合拒绝。

### 功能需求 2：half padding 正确性

1. 介绍：原掩码按 8 元素块展开，只适用于 32 位类型；half 每块 16 元素且一次 repeat 使用两个 mask 字。
2. 输入：AR 有效长度与 nominal stride，元素宽度。
3. 处理：按 32B/sizeof(T) 展开 padding 位，half 同时设置高、低掩码字；保留 remainder 和 repeat 分批策略。half 的无穷值使用 CANN NumericLimits 构造，避免 float 到 half 的饱和转换破坏单位元。
4. 输出：有效元素不变，所有 padding 写入 Max/Min 单位元。

## 非功能需求

### 可维护性

整数 API 独立建模，使用现有 ApiExtensionHeader 声明 reduce.h 与 utils.h 前导；不在 include pass 添加 dtype 特判。

### 可测试性

lit 覆盖 dtype 下降、类型相关行距拒绝和发射；CPU gtest 验证 padding 的真实内存结果。

### 可移植性

沿用现有 target capability；无芯片字符串分支。底层 API 支持范围以本次验证版本为准。

### 可靠性

保留 shape 溢出检查、布局证明、scratch 角色、同步和生命周期；新整数 op 校验输入输出均为 i32。

### 特性交叉影响

| 场景 | 结论 |
|---|---|
| SuperKernel Python、C++/AOT | 不适用，不修改 SuperKernel |
| Autofuse 图优化 | 不修改图优化；更多 dtype 候选可以下降 |
| Autofuse Codegen/Backend | 适用，增加整数扩展 API 发射 |
| AscendC API/Runtime | 适用，复用现有 API 和 scratch；修正共享 half padding |
| Python/C++ 混合绑定 | 不修改绑定接口 |
| 构建与打包 | 无新外部依赖；复用扩展头收集与内联 |
| 测试与覆盖率 | 增加 conversion、emitter、CPU padding 回归 |
| 性能与日志 | 无新日志；无新增运行期分配或 dtype 转换 |
| 兼容性 | 原 f32 行为保持；扩展 AFIR 可下降范围 |

## 性能

### 编译时长

增加常数成本的 dtype 分支及一个 op；首次构建需重生成 AscendC ODS。

### 执行性能

直接使用 legacy 算法。padding mask 的循环次数为编译期元素宽度决定的 4 或 8 次，不新增 kernel launch。

### 内存和产物大小

不新增 runtime 缓存。i32 kernel 内联已有 reduce.h、utils.h；scratch 使用调度器现有规划。

## 接口设计

### 新增/修改接口描述

新增内部 ascendc.reduce_sum_int32，操作数与二维 reduce 族相同；输入输出为 LocalTensor<i32>，scratch 为既有工作区，shape 必须为两项。通过 ApiExtension 声明头依赖。

### 接口检查项

| 检查项 | 子检查项 | 是否涉及 | 说明 |
|---|---|---|---|
| 接口说明 | 接口评审 | 内部 | 与二维族同形 |
| 接口说明 | 文档 | 是 | 本文 |
| 接口兼容 | 行为 | 是 | 原支持类型保持 |
| 接口兼容 | ABI/API | 否 | 无外部接口改动 |
| 接口约束 | 报错 | 是 | 类型与 shape verifier |
| 接口测试 | 独立用例 | 是 | emitter lit |

## 软件设计

### 关键数据结构

沿用 ReducePlan、PhysicalReduceShape；新增整数 API op，无新持久状态。

### 关键技术/算法

对齐数为 32/元素字节数；类型白名单按 op；half padding mask 每 16 位重复，32 位类型每 8 位重复。

### 流程设计

保留全部 pass 时序；capability 拒绝非法 dtype；rewrite 选择实际 API；ApiExtension 提供头依赖。

### 对子模块的修改

Reduce.cpp 修改类型判据与整数 API 选择；AscendC ODS/verifier/printer 增加整数 op；MLIR 私有 mlir_reduce_init.h 修复 half mask；lit 与 MLIR helper UT 增加验证。

### 错误处理

#### 系统错误

沿用编译失败及现有 pass failure 路径，无新系统资源。

#### 接口错误

非法 dtype、错误 shape 或不可证明的 stride 在生成 kernel 前拒绝。

## 安全检查

已读编码红线。无凭据、平台硬编码、新分配或 ABI 改动；mask 移位小于 64，支持元素宽度为 16/32 位；保留原有长度溢出检查和同步。

## 兼容性检查

不改变脚本参数、安装目录、Python 接口和调度格式。共享 legacy helper 保持上游内容，修正只用于 MLIR 私有 helper；须验证 MLIR f32 回归及 legacy 原有用例。

## 测试设计

### 测试边界

AFIR → AscendC conversion 与 AscendC → C++ 发射分别检查；CPU gtest 检查真实 padding 内容，覆盖完整 repeat 与余数行。测试产物放 build/codex-reduce-dtype。

### 测试用例设计

| 测试类别 | 关键测试项 | 测试方法 | 用例类型 |
|---|---|---|---|
| 功能 | Sum/i32、Max/Min/f16 | conversion + emitter | lit |
| 异常 | f16 使用 f32 pitch | 必须拒绝 | lit |
| 性能 | 无额外 cast/分配 | 生成代码检视 | 手工 |
| 兼容性 | 原 reduce 族 | reduce lit 回归 | UT |
| 特性交叉 | half padding、原 f32 尾块 | tikicpulib 数值验证 | gtest |

### 测试命令

容器入口设置 AF_MLIR_DEV_IMAGE=autofuse-mlir-dev:arm64、AF_CANN_VOLUME=af-cann-910、AF_CANN_INSTALL_PREFIX=/usr/local/Ascend/cann-9.2.0。

```bash
cmake --build build --target af-opt ascir-translate -j 8
/opt/llvm/bin/llvm-lit -j 8 -v --filter=reduce build/autofuse/mlir/test/lit
```

不重新配置 build/；CPU helper gtest 可使用既有 CPU E2E flags/link 参数单独编译 test_reduce_init.cpp，运行 TestApiReduceInitHalf.* 和 InnerAndOuterTails/*。

## 验收标准

新增 dtype 成功下降并发射正确 API；half 使用 16 元素对齐并拒绝 8 元素行距；CPU padding 不覆盖有效元素且填满尾部；既有 reduce lit 与 f32 helper 回归通过。真机性能不由这些测试宣称覆盖。


### 本次验证结果

- af-opt、ascir-translate 构建成功，全部 22 条 reduce lit 通过。
- 54 条 padding gtest 通过，包括两个 half 掩码字与余数行，以及既有 f32 尾块回归。
- 实际发射的 C++ 在 CPU 模拟器上通过 6 个有限值用例：i32 Sum、half Max/Min 各覆盖 AR/RA。整数输入大于 2^24，结果保持精确。
- 特殊值探针没有通过：half Max 的全负无穷、half Min 的全正无穷在 CANN 9.2.0 CPU 模拟路径返回 ±65504。绕过 MLIR 发射函数、直接调用 legacy 所用的 CANN API，同样复现四个失败。padding 自身仍正确保留真正无穷值。后续真机验证见下条；本次不增加特殊值运行时补偿。
- 通过 npussh 在 DevEnvC_1AtLk / Ascend910_9362 / CANN 9.2.0 上运行当前 AFIR 下降产生的设备函数，10/10 通过：i32 Sum 的 AR/RA（大于 2^24 的精确整数），f16 Max/Min 的 AR/RA 有限值与无穷值。A=9、R=17，源 padding 用污染值初始化；所有生成代码和 helper 的 SHA256 与本地一致。
- half 无穷值在该真机返回正确的 ±inf，CPU 模拟器的 ±65504 现象没有复现。本次为归约内核设备验证，不覆盖 PyTorch 前端、ATT 选优、跨核两阶段或性能评估。
- 日志与临时数值程序保存在 build/codex-reduce-dtype；真机结果与原始日志在 npu/result.md 和 npu/device-logs/。

### 2026-10-16 实现边界更新

half/int32 padding 修正已迁入 MLIR 私有的
`mlir/api/ascendc/extensions/mlir_reduce_init.h`，发射使用 `autofuse_mlir::ReduceInit`。
共享 `ascendc/api/reduce_init.h` 与其 legacy UT 恢复上游内容，关闭 MLIR 时不应用这些修正。
原新增数值测试迁入 `mlir/test/e2e/cpu/probes/ReduceInitTest.cpp`，与原版 legacy UT 分别运行。
资源及回归边界参见 ArgReduce 设计文档的「2026-10-16：隔离 legacy 行为的实现边界」。
