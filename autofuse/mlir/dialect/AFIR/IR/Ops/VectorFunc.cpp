/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dialect/AFIR/IR/AFIR.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::afir {

VectorFuncYieldOp VectorFuncOp::getYield() {
  return cast<VectorFuncYieldOp>(getBodyBlock().getTerminator());
}

// region 内的算子由 `AFIRVectorFuncToRegBase` 一手下降（寄存器形态那张表），所以 AFIR 这边
// 的物化与改写 pass 都不往里走。判据集中在这一个函数里，而不是每个 pass 各写一次
// `getParentOfType`：那样加一个 pass 就会漏一处，而漏掉的那一处不会报错，只会在产物里
// 多出一段本不该有的东西。
bool isInsideVectorFunc(Operation *op) {
  return op != nullptr && op->getParentOfType<VectorFuncOp>() != nullptr;
}

// 这个 verifier **只查结构**：边界与 region 两侧对不对得上。
//
// 不查的东西要说清楚，否则下一个人会往里加：
//   - 「这一期只支持一个输出」「只支持 f32/f16」「region 内这个 op 有没有寄存器形态」
//     全是**能力**问题。verifier 拒收是硬失败，而能力不足必须干净回落给 legacy，
//     所以那些判据在能力层（`AFIRVectorFuncToRegBase`），不在这里。
//   - 形状不查：lane 这一维在 AFIR 里不出现，region 内的 tensor 是「一段 lane」而不是
//     整块 tile，两侧的 shape 本来就不该相等。能核的只有元素类型。
LogicalResult VectorFuncOp::verify() {
  Block &body = getBodyBlock();

  if (body.getNumArguments() != getSrcs().size()) {
    return emitOpError() << kInvalidDescriptorDiagnostic << ": body takes " << body.getNumArguments()
                         << " arguments but the operation has " << getSrcs().size()
                         << " srcs; they are paired by index";
  }
  for (auto [index, pair] : llvm::enumerate(llvm::zip(getSrcs(), body.getArguments()))) {
    const Type srcElement = cast<ShapedType>(std::get<0>(pair).getType()).getElementType();
    const Type argType = std::get<1>(pair).getType();
    const Type argElement = isa<ShapedType>(argType) ? cast<ShapedType>(argType).getElementType() : argType;
    if (srcElement != argElement) {
      return emitOpError() << kInvalidDescriptorDiagnostic << ": src #" << index << " has element type " << srcElement
                           << " but body argument #" << index << " has " << argElement;
    }
  }

  VectorFuncYieldOp yield = getYield();
  if (yield.getResults().size() != getDsts().size()) {
    return emitOpError() << kInvalidDescriptorDiagnostic << ": body yields " << yield.getResults().size()
                         << " values but the operation has " << getDsts().size() << " dsts; they are paired by index";
  }
  for (auto [index, pair] : llvm::enumerate(llvm::zip(getDsts(), yield.getResults()))) {
    const Type dstElement = cast<ShapedType>(std::get<0>(pair).getType()).getElementType();
    const Type yieldedType = std::get<1>(pair).getType();
    const Type yieldedElement =
        isa<ShapedType>(yieldedType) ? cast<ShapedType>(yieldedType).getElementType() : yieldedType;
    if (dstElement != yieldedElement) {
      return emitOpError() << kInvalidDescriptorDiagnostic << ": dst #" << index << " has element type " << dstElement
                           << " but the yielded value #" << index << " has " << yieldedElement;
    }
  }

  // `dsts` 同时作为结果产出，所以两侧数目与类型必须一致——消费者拿的是结果值。
  if (getResults().size() != getDsts().size()) {
    return emitOpError() << kInvalidDescriptorDiagnostic << ": has " << getResults().size() << " results but "
                         << getDsts().size() << " dsts; each dst is also produced as a result";
  }
  for (auto [index, pair] : llvm::enumerate(llvm::zip(getDsts(), getResults()))) {
    if (std::get<0>(pair).getType() != std::get<1>(pair).getType()) {
      return emitOpError() << kInvalidDescriptorDiagnostic << ": dst #" << index << " and result #" << index
                           << " must have the same type";
    }
  }

  // 嵌套的 VF 今天的分区器不产（它在一张 impl graph 上跑一趟，子图不再进分区）。
  // 判据仍然要在：**没有判据的「不会发生」会在它第一次发生时变成静默的错产物**，
  // 而这一条在结构上就能拒，不必等到下降。
  WalkResult nested = body.walk([&](VectorFuncOp inner) { return WalkResult::interrupt(); });
  if (nested.wasInterrupted()) {
    return emitOpError() << kInvalidDescriptorDiagnostic << ": nested afir.vector_func is not representable";
  }

  return success();
}

}  // namespace mlir::afir
