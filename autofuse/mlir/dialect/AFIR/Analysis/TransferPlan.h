/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_TRANSFERPLAN_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_TRANSFERPLAN_H

#include <cstdint>

#include "dialect/AFIR/Analysis/ComputeExtent.h"

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class Operation;
class OpBuilder;
namespace func {
class FuncOp;
}  // namespace func
}  // namespace mlir

namespace mlir::afir {

// Identity 表示整块访问；Scheduled 按调度视图处理，可包含非连续维度。
enum class TransferPlanKind { Identity, Scheduled };

// 缓存守卫里循环轴与切分轴的维度关系，按两者的**原始轴集合**（`from` 链的闭包）比较：
//   - `SameDimension`：循环轴的集合覆盖切分轴的集合，循环推进就会推进切分轴那一维。合轴
//     （Merged）之后再分块的调度也落在这一支——合出来的轴同时含那一维。
//   - `OtherDimension`：两个集合不相交，循环跑的是另一维，切分轴那一维的 tile 序号在本循环
//     内不变。
//   - `Unresolved`：切分轴取不到、`from` 链断了，或两个集合部分重叠。守卫据此返回失败。
enum class GuardLoopRelation { Unresolved, SameDimension, OtherDimension };

// tile 外的一根调度轴及其 GM 元素步长，用于计算起点偏移。
struct TransferOuterAxis {
  int64_t axis = -1;
  ExprAttr globalStride;
};

// 搬运循环嵌套里的一层：循环 `extent` 次，每走一步 GM 指针前进 `globalStride`、UB 指针
// 前进 `localStride`。
// `TransferPlan::dims` 外到内排开就是这段嵌套本身：
//     for (i = 0; i < dims[0].extent; i++)
//       for (j = 0; j < dims[1].extent; j++)
//         ub[i * dims[0].localStride  + j * dims[1].localStride]
//          = gm[i * dims[0].globalStride + j * dims[1].globalStride];
// 一层有两个 stride，是因为同一批数据在两侧的间隔不同；
// Load 与 Store 共用，方向由两端类型决定。
struct TransferDim {
  ExprAttr extent;
  ExprAttr globalStride;
  ExprAttr localStride;
};

struct TransferPlan {
  TransferPlanKind kind = TransferPlanKind::Identity;
  // GM 轴表中不属于局部 vectorized_axis 的轴，保持原序；逐项计算 index × stride。
  SmallVector<TransferOuterAxis, 0> outerAxes;
  // 有效视图，外到内且非空：最内两项供 DMA，其余项生成外循环。
  SmallVector<TransferDim, 0> dims;
  // 分段尾块修正与执行守卫使用的可选轴，不参与 GM 偏移计算。
  // 无法解析时保留 -1；消费者须按无对应轴处理或返回失败，不得作为真实轴索引。
  int64_t tileInnerAxis = -1;
  int64_t loopAxis = -1;
  // 执行守卫用的块轴对：取其原始轴集合**覆盖** `tileOuterAxis` 原始轴集合的那一对，与
  // `loopAxis` 是哪根轴无关。解析不出时两者都保留 -1。
  int64_t blockOuterAxis = -1;
  int64_t blockInnerAxis = -1;
  int64_t tileOuterAxis = -1;
  // `loopAxis` 与 `tileOuterAxis` 的维度关系。fused 守卫按它分两种谓词形态，见
  // `materializeExecConditionGuard`；`Unresolved` 时守卫返回失败，不得挑一种形态。
  GuardLoopRelation guardLoopRelation = GuardLoopRelation::Unresolved;
};

// DMA 参数与两侧偏移。globalOffset 包含 GM tile 起点和 tile 内部行偏移；
// UB tile 从 buffer 基址开始，localOffset 仅包含 tile 内部行偏移。
// 一维时 blockCount = 1，两侧行间距为 0；不需要外循环时 localOffset = 0。
struct ScheduledTileIndices {
  Value globalOffset;
  Value localOffset;
  Value blockCount;
  Value blockLen;
  // 行间距 = 外层 stride − 实际 blockLen，单位为元素；API 负责字节/32B 块换算。
  Value globalStride;
  Value localStride;
};

// 单轴的标称/实际范围，供 Reduce 按语义组合 A/R 维；独立于搬运的双侧连续性合轴。
struct ScheduledAxisExtent {
  Value nominal;
  Value actual;
};

// Automatic：有结果取 result，无结果取 operand；Operand 显式选择输入视图。
enum class TransferPlanSide { Automatic, Operand };

// 构造搬运计划；失败不得回落 identity，调度路径不适用时仍须通过 identity 检查。
FailureOr<TransferPlan> buildTransferPlan(Operation *op, TransferPlanSide side = TransferPlanSide::Automatic);

// 检查 load/store 的 UB 行布局是否能由 block DMA 表达。可证行重叠或步距被截断时
// 返回 false，未知符号返回 true。Store 零间隙放行；不检查 GM 侧，不适用于 transpose。
bool blockDmaSegmentGeometryIsSound(Operation *op, const TransferPlan &plan);

// 按指定局部 descriptor 构造计划，不修改 op 属性。
FailureOr<TransferPlan> buildTransferPlanForDescriptor(Operation *op, TensorDescAttr localDescriptor);

// 局部视图取自 TensorDescAttr：Load 取结果，Store 取操作数；`Implementations/` 侧的
// 形态判据（`Transpose.cpp` / `Pad.cpp`）与计划构造取同一份事实。
FailureOr<TensorDescAttr> getLocalDescriptor(Operation *op, TransferPlanSide side);

// 图上的轴表（`afir.asc_graph` 的 axes），按轴 id 索引；取不到（无 func 或无属性）失败。
FailureOr<llvm::DenseMap<int64_t, AxisAttr>> collectGraphAxes(Operation *op);

// 两个符号表达式的相等性证明。Unknown 表示不能证明，不表示不相等。
detail::TruthValue proveEqual(ExprAttr lhs, ExprAttr rhs);

// 回调的 builder 位于搬运插入点，indices 中的值在该作用域可用。
using ScheduledTransferBody = llvm::function_ref<LogicalResult(OpBuilder &, const ScheduledTileIndices &)>;

// 最内两项供 DMA，其余项生成嵌套循环，在最内层调用 emitBody。
// 两项及以内直接调用；循环不变量在循环外物化，行偏移在循环内计算。
// compact 表示 UB 行紧凑存储；连续后缀的 stride 随尾块实际尺寸变化。
LogicalResult emitScheduledTransfer(Operation *op, OpBuilder &builder, const TransferPlan &plan,
                                    ScheduledTransferBody emitBody, bool compact = false);
// 按 ActualSize 物化最多二维的有效范围，保持外到内顺序，不生成 DMA 偏移。
FailureOr<SmallVector<Value, 2>> materializeScheduledTransferDims(Operation *op, OpBuilder &builder,
                                                                  const TransferPlan &plan);
// 物化局部物理跨度：范围取 actual，stride 取 nominal 布局。
FailureOr<Value> materializeScheduledLocalSpan(Operation *op, OpBuilder &builder, const TransferPlan &plan);
// 物化最内有效维的标称范围，供不随尾块缩小的 UB 布局与 ReduceInit 使用。
FailureOr<Value> materializeNominalInnermostExtent(Operation *op, OpBuilder &builder, const TransferPlan &plan);
// 按给定轴序物化标称/实际范围；有尾块的轴缺少 ActualSize 绑定时失败。
FailureOr<SmallVector<ScheduledAxisExtent, 4>> materializeScheduledAxisExtents(Operation *op, OpBuilder &builder,
                                                                               ArrayRef<int64_t> axes);

FailureOr<Value> materializeTailAwareTileSize(Operation *op, OpBuilder &builder, const TransferPlan &plan);
// 物化 GM tile 起始元素偏移，不限制局部视图的段数。
FailureOr<Value> materializeScheduledGlobalOffset(Operation *op, OpBuilder &builder, const TransferPlan &plan);
// 解析调度轴在 GM 侧的单位元素步长，必要时沿 from 追溯原始轴；无法解析则失败。
// UB 连续段不保证 GM 连续，消费者必须另行校验段内 GM 连续性。
FailureOr<ExprAttr> globalAxisUnitStride(Operation *op, int64_t axisId);
// 检查局部视图可否进行一维连续访问：至多一段，非退化段须具有单位局部 stride。
// 零段表示单元素；局部 descriptor 必须参与合轴，不能借用其他张量的结论。
// capability 与 lowering 共用；仅用于要求连续访问的 API，不限制带行间距的 DMA。
LogicalResult localTileIsSingleSegment(Operation *op);
FailureOr<Value> materializeExecConditionGuard(Operation *op, OpBuilder &builder, const TransferPlan &plan,
                                               ExecCondition condition);

}  // namespace mlir::afir

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_ANALYSIS_TRANSFERPLAN_H
