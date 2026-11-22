/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conversion/AFIRToAscendC/MemoryAllocate/CanonicalMemoryAllocator.h"

#include "conversion/AFIRToAscendC/AFIRToAscendCPasses.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include "dialect/AFIR/IR/AFIR.h"
#include "dialect/AFIR/IR/SymbolicBuilder.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"
#include "dialect/AFIR/Transforms/Symbolic/SymbolicTransformDetail.h"
#include "conversion/AFIRToAscendC/AscendCKernelABI.h"
#include "ascir/Dialect/Asc/IR/Asc.h"
#include "conversion/AFIRToAscendC/TilingSchema.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::afir;
using namespace mlir::ascend::kernel;

namespace mlir::ascend::kernel::detail {

Value CanonicalFunctionMemoryAllocator::materializedValue(Value value) const {
  Value root = resolveAlias(value);
  auto it = values.find(root);
  return it == values.end() ? Value() : it->second.materialized;
}

SmallVector<Operation *> CanonicalFunctionMemoryAllocator::collectCanonicalOperations() {
  SmallVector<Operation *> operations;
  func.walk([&](Operation *operation) {
    // region 内不进这一层。 `afir.vector_func` 的 region 里是 tensor 形态的算子：它们
    // 没有存储要物化（VF 的中间结果在寄存器里），而它们同样挂着 descriptor，所以不挡住
    // 就会被当成待物化的算子、去找一个根本不存在的 materialized 值。算子自己（边界是
    // 真的 UB 张量）照常物化，只是不往里走。
    if (afir::isInsideVectorFunc(operation)) {
      return;
    }
    if (isAfirOperation(operation) && !isa<afir::SymbolicEvalOp>(operation) && hasCanonicalAttrs(operation)) {
      operations.push_back(operation);
    }
  });
  return operations;
}

LogicalResult CanonicalFunctionMemoryAllocator::rewriteCanonicalOperation(Operation *operation) {
  SmallVector<Value> operands;
  operands.reserve(operation->getNumOperands());
  for (Value operand : operation->getOperands()) {
    // 标量操作数不经过这一层：它没有 memref 要换成 `!ascendc.local_tensor`，本来就是
    // 一个值，原样带过去。（`arith.constant` 之类的定义点也不在这个分配器的视野里。）
    if (!isa<ShapedType>(operand.getType())) {
      operands.push_back(operand);
      continue;
    }
    Value mapped = materializedValue(operand);
    if (!mapped) {
      return operation->emitError("canonical operand has no materialized AscendC value");
    }
    operands.push_back(mapped);
  }
  SmallVector<Type> resultTypes;
  for (Value result : operation->getResults()) {
    Value mapped = materializedValue(result);
    if (!mapped) {
      return operation->emitError("canonical result has no materialized AscendC value");
    }
    resultTypes.push_back(mapped.getType());
  }
  OperationState state(operation->getLoc(), operation->getName());
  state.addOperands(operands);
  state.addTypes(resultTypes);
  state.addAttributes(operation->getAttrs());
  // region 要跟着搬过来。 这一层是按操作数与结果类型重建算子的，而 `OperationState`
  // 不会自己把 region 带过去——带 region 的算子（`afir.vector_func`）会被重建成一个空壳，
  // 而空壳的失败方式取决于它的 trait：`SingleBlockImplicitTerminator` 会在下面那次
  // `verify` 上报「区域为空」，而不是报「谁把它清掉了」。
  state.addRegions({});
  for (size_t index = 0; index < operation->getNumRegions(); ++index) {
    state.addRegion();
  }
  OpBuilder builder(operation);
  Operation *replacement = builder.create(state);
  for (auto [oldRegion, newRegion] : llvm::zip(operation->getRegions(), replacement->getRegions())) {
    newRegion.takeBody(oldRegion);
  }
  if (failed(verify(replacement))) {
    replacement->erase();
    return operation->emitError("canonical AFIR operation failed verification after AscendC materialization");
  }
  newOperations.push_back(replacement);
  for (auto [oldResult, newResult] : llvm::zip(operation->getResults(), replacement->getResults())) {
    replacements.emplace_back(oldResult, newResult);
  }
  return success();
}

void CanonicalFunctionMemoryAllocator::eraseCanonicalOperations(ArrayRef<Operation *> operations) {
  for (auto [oldValue, newValue] : replacements) {
    oldValue.replaceAllUsesWith(newValue);
  }
  for (Operation *operation : llvm::reverse(operations)) {
    operation->erase();
  }
  SmallVector<Operation *> deadAllocs;
  func.walk([&](memref::AllocOp alloc) {
    if (alloc->use_empty()) {
      deadAllocs.push_back(alloc.getOperation());
    }
  });
  for (Operation *alloc : deadAllocs) {
    alloc->erase();
  }
}

LogicalResult CanonicalFunctionMemoryAllocator::rewriteCanonicalOperations() {
  SmallVector<Operation *> operations = collectCanonicalOperations();
  for (Operation *operation : operations) {
    if (failed(rewriteCanonicalOperation(operation))) {
      return failure();
    }
  }
  eraseCanonicalOperations(operations);
  return success();
}

LogicalResult CanonicalFunctionMemoryAllocator::lowerSymbolicExpressions() {
  RewritePatternSet patterns(context);
  afir::detail::populateSymbolicLoweringPatterns(patterns);

  // 贪心重写会删除无 SSA 使用者的纯算子，但 axis_binding 属性不计为 SSA use。
  // 部分 actual_size 被循环上界使用，其他向量化轴绑定仅供后续按属性查询，不能被 DCE。
  // 否则搬运下降会丢失 generatePairedInnerActualSize 产生的实际长度。
  SmallVector<Operation *> rewritable;
  func.walk([&](Operation *operation) {
    if (operation != func.getOperation() && !operation->hasAttr(afir::kAxisBindingAttrName)) {
      rewritable.push_back(operation);
    }
  });
  if (rewritable.empty()) {
    return success();
  }
  GreedyRewriteConfig config;
  config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
  return applyOpPatternsGreedily(rewritable, FrozenRewritePatternSet(std::move(patterns)), config);
}

}  // namespace mlir::ascend::kernel::detail
