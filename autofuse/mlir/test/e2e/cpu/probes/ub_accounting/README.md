# MLIR UB 计算链路核查（2026-10-19）

范围为当前工作区的调度资源 → ATT UB 约束与 tiling 字段 → bridge → MLIR 资源物化及 lowering。
本轮只取证，未修改生产实现、公共 helper 或发布门禁。新增探针使用已有 lit 输入的副本构造容量不一致场景，
不代表正常 scheduler/ATT 已产生这些输入，也没有执行其设备 kernel。

## 结论

尚未找到正常 ATT → MLIR 路径新增 UB 漏计的实测反例；已确认 MLIR 有两处容量校验缺口。
两种有问题的输入不仅通过 `afir-memory-allocate`，也通过 `afir-to-ascendc-pipeline`，各返回 0。

| 输入 | 所属资源容量 | 成员需求 / 总分配 | memory-allocate | 完整 lowering |
|---|---:|---:|---:|---:|
| 正常对照 | queue 单槽 64B | 两个 16xf16 成员，共 64B | 0 | 0 |
| 单槽不足 | queue 单槽 32B | 两个 16xf16 成员，共 64B | 0 | 0 |
| 总额超限 | 目标 UB 196608B | 两槽 × 131072B = 262144B | 0 | 0 |
| 零长度反向对照 | queue 单槽 0B | 成员共 64B | 1 | 1 |

### 成员范围没有与资源容量比较

`conversion/AFIRToAscendC/MemoryAllocate/CanonicalMaterialization.cpp::materializeLocalMember`
为 queue 成员累加偏移，检查整数加法溢出，但没有检查 `offset + memberBytes <= slotBytes`。
单槽不足的产物保留了 `init_queue(..., 1, 32)`、`subindex(raw, 32)` 以及针对第二成员的
`data_copy_pad_extend`。第二成员起点已经到达资源末尾，lowering 仍然成功。

非队列资源同样没有独立核对视图跨度：当前 `materializeLocalMemberSize` 的非队列分支直接返回资源大小，
而 `materializeLocalMember` 在构造非队列视图后直接返回。本轮动态探针钉住的是 queue 路径，
没有把上述非队列静态观察算作独立的设备越界结论。

### 最终 InitBuffer 总额没有与 UB 上限比较

`conversion/AFIRToAscendC/MemoryAllocate/CanonicalResources.cpp::createCanonicalResources`
逐资源发出 `InitBuffer`，各自检查正数和对齐溢出，但没有求和再与目标 UB 容量比较。
总额超限的输入显式携带 `#afir.target<..., ub_size = 196608>`，最终仍生成两槽、每槽 131072B 的队列。

生产 tiling 入口会先扣除 256B 保留量，并由 ATT 求解 UB 约束。这是现有的生产容量保障；
本探针走直接 AFIR 输入，刻意绕开 ATT 以检查下游能否发现不一致。
因此应将结果表述为“MLIR 缺少容量交叉核验”，不能表述为“ATT 已经漏算 64KB”或“真机必然发生了越界”。

## 原因追踪

本节依据当前源码、`git show` / `git log -S` 和已有设计记录；未编译历史版本。
历史定位说明检查何时已缺失，不推断提交者的主观意图。

### 成员边界：分配容量与切片需求没有形成校验关系

- 在 `49e46f35`（`feat: complete AFIR canonical IR and lowering`）版本的
  `AFIRMemoryAllocate.cpp::materializeLocalMember` 中，已经只有偏移加法溢出检查，
  没有与所属资源容量比较。该版本的 `createCanonicalResources` 也只逐资源物化，不累计 UB 总额。
- 当前 `createQueueResource` 把对齐后的容量交给 `TPipeInitQueueOp`；成员物化侧的
  `LocalSlab` 仅携带 `raw`、`rawType`、`offset` 和 `queue`，没有容量 SSA 值。
  queue 成员的实际跨度另由 vectorized repeats/strides 算出。两条计算链没有比较点，
  所以“整数没有溢出”成立时，仍可能“成员越过分配末尾”。
- `b9f830a0` 将 UB 分配尺寸改为读取 ATT 的 `q<ID>_size` / `b<ID>_size`，修正按整个
  逻辑张量分配的问题；`3174947a` 又改为 `size_from_resource`，避免 descriptor 引错资源字段。
  这些改动保障尺寸来源及资源身份，没有补上容量与成员需求的比较，也不是删除该检查的提交。
- 既有 `afir_memory_allocate_shared_queue_members.mlir` 于 `b9f830a0` 加入，覆盖的是
  64B 槽中两个 32B 成员的正确偏移。该正向用例无法区分“偏移正确”与“容量也经过检查”。
  架构设计 `2026-09-09-afir-public-dsl-architecture-design.md` 的异常测试矩阵明确列有
  “越界 view”；因此这里应视为尚未落实的容量验证，而不是设计明确允许越界。

### UB 总额：依赖 ATT 的预算没有与最终 lowering 分配交叉核验

- 初始 canonical allocator 已采用逐资源物化结构。正常路径的容量约束在 host ATT 侧求解，
  MLIR 消费资源尺寸；`3174947a` 的提交说明与当前 `materializeResourceSize` 注释明确记录了这一职责划分。
  直接 AFIR 也能用常量 `size_bytes_expr` 构造资源，却没有额外的总容量检查来弥补绕开 ATT 的路径。
- `AFIRToAscendCPipeline.cpp` 中 MemoryAllocate 先于 SelectImplementation；后续 lowering
  还能生成 `ConstBlockTensor`。因此仅统计 MemoryAllocate 当时的资源也不等于最终全部分配。
  当前常量分配与 ATT 预留规则匹配，但两者分别维护。
- `2026-09-17-afir-op-completion-elementwise-progress.md` 已记录这项分离：ATT 的额外常量预算
  与 MLIR 的 32B 常量块没有共享断言，并举例说明新增常量而未同步 ATT 可能漏预算。
  这支持“缺少最终交叉核验”的原因判断，不代表该假设场景已经在当前版本发生。

### 与 legacy 及近期改动的关系

legacy 默认 ATT 路径的 `codegen_kernel.cpp::InitTQueBuffers` / `InitTBufBuffer` 也直接消费
`t->q<ID>_size` / `t->b<ID>_size`；`TensorAlloc` 直接按 `que_share_offset` 切片，
这些函数不生成上述容量比较。因此尚不能把两项缺口称为 MLIR 相对 legacy 的预算回归。
这里比较的是 codegen 行为，未据此断言 CANN SDK 内部完全没有运行时检查。

`e1ec36a0` 将 allocator 拆成多个文件，提交说明明确为方法体原样搬运；检查缺失在拆分前已存在。
`42337250` 后续调整非队列成员的 offset 处理，也不是本轮 queue 反例的起点。
综合判断：两项问题是资源物化阶段沿用上游预算可信假设、但未补齐下游容量验证的既有缺口，
不是近期 ArgMax 收敛改动导致，也没有证据表明正常 ATT 流程当前已经漏算。

## 当前核对一致的部分

- `common/ub_expr/asc_graph_ub_expr_builder.cpp` 对 tensor 字节数做 32B 对齐，按资源 ID 聚合需求，
  queue 计入槽数，tmp 资源施加 8192B 下限，额外常量与 API 保留区计入 UB 表达式。
- `att/gen_model_info/gen_model_info.cpp::RefreshCommonUbExprContext` 把该表达式写回 ATT 约束及资源字段。
- `bridge/ScheduleToAfir/ScheduleToAfir.cpp::AppendTensorFacts` 对 UB 使用 `sizeFromResource`；
  `AfirBuilder.cpp::makeTmpBufferAllocation` 同样引用 `b<ID>_size`，没有重新按逻辑全张量大小计费。
- MLIR 按空间、queue/TBuf 类型和 buffer ID 合并物理资源；queue 槽数取最大值，alignment 聚合。
- lowering 新建的 `Rsqrt` f32 常量 1 和 `LogicalNot` f16 常量 1 各占 32B、按类型和值去重，
  与 `common/common_utils.cpp::GetApiExtractDupSet` 的 ATT 额外预算相符。
  这两处分别维护，目前匹配，但没有把预算与最终分配绑定起来的自动核对。
- 重新运行既有 ArgMax 同调度审计：8 格、12 个候选的资源字段与角色匹配；8 个选中 tiling
  的尺寸均为正且 32B 对齐，总 UB 分配在预算内。该覆盖范围见历史提交 `f5efee6a` 中的 `argmax_same_schedule/README.md`。

此前小 tile 真机探针证明的公共 ArgMax helper 越过源 buffer 的写入仍成立；
源访问边界与 scratch/总 UB 预算是不同检查项，8KB tmp 下限无法证明源 buffer 访问安全。

## 复现与证据

在仓库根目录的现有开发容器运行：

```sh
python3 autofuse/mlir/test/e2e/cpu/probes/ub_accounting/run.py
```

Mac 可使用：

```sh
AF_CANN_VOLUME=af-cann-910 \
AF_CANN_INSTALL_PREFIX=/usr/local/Ascend/cann-9.2.0 \
bash autofuse/mlir/scripts/enter_dev_container.sh -- \
python3 autofuse/mlir/test/e2e/cpu/probes/ub_accounting/run.py
```

产物位于 `build/ub-accounting-20261019/`，包括四个输入、两阶段输出与 stderr，以及记录退出码、
分配量、输入 fixture 哈希和现有断言的 `summary.json`。脚本返回 0 只表示正常对照与零长度反向对照符合预期，
其他病例的接收状态以 `summary.json` 为准；不表示容量安全检查全部通过。

本轮使用现有 `af-opt`（LLVM 21.1.8，构建时间晚于所核查资源物化源码），未重编实现。
该测试以常量 `size_bytes_expr` 显式构造反例，不等价于已经覆盖动态形状、所有算子 API 内部 scratch、
所有生命周期组合或所有芯片，也没有通过设备执行验证异常的表现形式。

### 代码检视检查结果

- [x] 编码红线：读取 `docs/guidelines/编码红线.md`，本轮重点核查尺寸、偏移、溢出与容量边界；上述缺口已记录。
- [x] 跨特性交叉影响：读取 `cross_feature_check.md`，涉及 Autofuse codegen/tiling/AscendC 资源链；本轮仅增加诊断探针，无生产 API/ABI、SuperKernel、绑定或交付变更。
- [x] 贡献规范：读取 `CONTRIBUTING.md`；未创建提交、PR 或发布产物。
- [x] 代码格式：读取 `.clang-format`，本轮未修改 C++；Python 探针通过语法检查。
- [x] Pre-commit：已确认配置要求；未执行全库自动改写，执行 `git diff --check`。

同时参考根目录与 `autofuse/mlir/AGENTS.md`、MLIR README 及既有同调度验证记录。
