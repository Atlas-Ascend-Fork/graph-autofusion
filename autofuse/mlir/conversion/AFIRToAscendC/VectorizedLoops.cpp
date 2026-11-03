/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/VectorizedLoops.h"

#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::ascend::kernel {

FailureOr<VectorizedWindow> materializeVectorizedSegments(Operation *op, OpBuilder &rewriter,
                                                          const afir::VectorPlan &plan,
                                                          const afir::TransferPlan &transferPlan, ArrayRef<Value> tiles,
                                                          unsigned apiWindowSize) {
  // 一段都没有：每根向量化轴的 extent 都是 1（`collapseVectorizedAxes` 已把它们剔掉），
  // 这个 op 本趟只有**一个元素**的工作量。窗口因此是「不建循环、偏移 0、长度 1」。
  //
  // 这不是把一个空计划圆成一个窗口，而是搬运侧早就收下的同一档形状：
  // `materializeScheduledTileIndices`（`TransferPlan.cpp`）在空段上取的也是常量 1。真实
  // 来源是广播轴被切分、又与外轴合成一根块轴的那个模板（`exec_condition =
  // cache_block_split_fused_brc_axis`）——调度把 elementwise 排到 `afir.broadcast`
  // **之前**，于是它作用在广播前那块 `axis = [z0, z1T, z1t], repeats = [n, 1, 1],
  // vectorized_axis = [z1t]` 的 tile 上，唯一那根向量化轴的 repeat 是 1。同一个 op 周围的
  // `afir.load` 与 `afir.broadcast` 的 `Duplicate` 一档走的就是搬运侧那一档，三者对这块
  // tile 的说法因此一致：本趟一个元素，局部偏移 0。
  //
  // **只收 `apiWindowSize == 1`。** 要两段以上窗口的调用方（分段广播、`RemovePad`）问的是
  // 「这几段在 shape 数组里各占多少」，空段答不出来，那时仍旧报失败。
  const bool singleElementWindow = plan.segments.empty();
  if (op == nullptr || apiWindowSize == 0 || tiles.size() != plan.tensors.size()) {
    return failure();
  }
  if (singleElementWindow ? apiWindowSize != 1 : plan.segments.size() < apiWindowSize) {
    return failure();
  }
  const Location loc = op->getLoc();

  // 分段范围沿用搬运的 ActualSize 绑定，支持切分轴位于 tile 内任意位置。
  auto segmentExtent = [&](const afir::VectorSegment &segment) -> FailureOr<Value> {
    if (transferPlan.kind != afir::TransferPlanKind::Scheduled) {
      return afir::materializeSegmentExtent(op, rewriter, segment, -1, {});
    }
    return afir::materializeActualSegmentExtent(op, rewriter, segment);
  };

  const size_t loopCount = singleElementWindow ? 0 : plan.segments.size() - apiWindowSize;

  // 外层各段：一段一层循环。循环上界要在建循环**之前**算好，否则它会落进循环体里，
  // 既支配不了循环头，也白算一遍。
  SmallVector<Value> inductionVars;
  inductionVars.reserve(loopCount);
  for (size_t index = 0; index < loopCount; ++index) {
    FailureOr<Value> upperBound = segmentExtent(plan.segments[index]);
    if (failed(upperBound)) {
      return failure();
    }
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto loop = rewriter.create<scf::ForOp>(loc, zero, *upperBound, one);
    inductionVars.push_back(loop.getInductionVar());
    // 插入点移进循环体、落在终结符之前。下一层循环与最终的 API 调用都建在这里。
    rewriter.setInsertionPoint(loop.getBody()->getTerminator());
  }

  VectorizedWindow window;

  // 每个张量偏移 `Σ iv_j × stride_j`，与 legacy `CalcInnerOffset` 同一个算式。
  //
  // 没有外层循环时这个和是空的，偏移落成一个常量 0，仍然发一条 subindex——今天单段路径
  // 发的就是「常量 0 + subindex」，保持逐字相同。
  window.tiles.reserve(tiles.size());
  for (auto [tensorIndex, tile] : llvm::enumerate(tiles)) {
    Value offset;
    for (size_t segmentIndex = 0; segmentIndex < loopCount; ++segmentIndex) {
      const afir::VectorSegment &segment = plan.segments[segmentIndex];
      if (tensorIndex >= segment.strides.size()) {
        return failure();
      }
      FailureOr<Value> stride = afir::materializeStrideValue(op, rewriter, segment.strides[tensorIndex]);
      if (failed(stride)) {
        return failure();
      }
      Value term = rewriter.create<arith::MulIOp>(loc, inductionVars[segmentIndex], *stride).getResult();
      offset = offset ? rewriter.create<arith::AddIOp>(loc, offset, term).getResult() : term;
    }
    if (!offset) {
      offset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    }
    window.tiles.push_back(rewriter.create<ascendc::LocalTensorSubIndexOp>(loc, tile.getType(), tile, offset));
  }

  // 最内 `apiWindowSize` 段的元素数。单元素窗口没有段，长度是常量 1。
  window.extents.reserve(apiWindowSize);
  if (singleElementWindow) {
    window.extents.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
    return window;
  }
  for (size_t index = loopCount; index < plan.segments.size(); ++index) {
    FailureOr<Value> extent = segmentExtent(plan.segments[index]);
    if (failed(extent)) {
      return failure();
    }
    window.extents.push_back(*extent);
  }
  return window;
}

FailureOr<SmallVector<Value>> orderTilesForPlan(const afir::VectorPlan &plan, ArrayRef<Value> inputs,
                                                Value destination) {
  SmallVector<Value> ordered;
  ordered.reserve(plan.tensors.size());
  for (const afir::VectorTensor &tensor : plan.tensors) {
    if (tensor.side == afir::VectorTensor::Side::Destination) {
      if (tensor.index != 0 || !destination) {
        return failure();
      }
      ordered.push_back(destination);
      continue;
    }
    if (tensor.index >= inputs.size()) {
      return failure();
    }
    ordered.push_back(inputs[tensor.index]);
  }
  return ordered;
}

}  // namespace mlir::ascend::kernel
