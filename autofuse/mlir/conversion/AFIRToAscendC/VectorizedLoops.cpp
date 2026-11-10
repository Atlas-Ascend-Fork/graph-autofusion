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
                                                          const afir::VectorPlan &plan, afir::ExtentSource extentSource,
                                                          ArrayRef<Value> tiles, unsigned apiWindowSize) {
  // 空 segments 表示所有向量化轴 extent 均为 1。apiWindowSize 为 1 时返回
  // 单元素窗口：不建循环，局部偏移为 0，长度为 1，与搬运侧 materializeScheduledTileIndices 一致。
  // 这会出现在广播轴切分并与外轴合为块轴的模板中：elementwise 位于 broadcast 之前，
  // 其 tile 的 vectorized_axis repeat 为 1。周围的 load 和 Duplicate 广播也按单元素处理。
  // 需要两段及以上窗口的调用方（如分段广播、RemovePad）仍失败，因为空计划不能提供
  // 各段的 shape 范围。
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
    if (extentSource == afir::ExtentSource::Nominal) {
      return afir::materializeSegmentExtent(op, rewriter, segment, -1, {});
    }
    return afir::materializeActualSegmentExtent(op, rewriter, segment);
  };

  const size_t loopCount = singleElementWindow ? 0 : plan.segments.size() - apiWindowSize;

  // 外层各段：一段一层循环。循环上界要在建循环之前算好，否则它会落进循环体里，
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
  // 没有外层循环时这个和是空的，偏移落成一个常量 0，仍然生成一条 subindex——今天单段路径
  // 生成的就是「常量 0 + subindex」，保持逐字相同。
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
