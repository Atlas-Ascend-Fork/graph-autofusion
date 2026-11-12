/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dialect/AFIR/Analysis/ComputeExtent.h"
#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::afir {
// 收集尺寸/步长表达式中的自由符号。
LogicalResult collectFreeSymbols(MLIRContext *context, ExprAttr expression, llvm::SetVector<StringRef> &symbols) {
  if (!expression) {
    return success();
  }
  auto parsed = Expression::Parse(expression.getValue().getValue());
  if (!parsed) {
    return failure();
  }
  auto free = detail::ExpressionAccess::GetFreeSymbols(parsed.value());
  if (!free) {
    return failure();
  }
  for (const std::string &symbol : free.value()) {
    // StringRef 不能指向这个临时 std::string，交给 context 的字符串池持有。
    symbols.insert(StringAttr::get(context, symbol).getValue());
  }
  return success();
}

// 通过 axis_binding 恢复生成后 SSA 值的轴身份和角色。
Value findAxisValue(func::FuncOp func, Operation *user, int64_t axisId, afir::AxisValueRole role) {
  DominanceInfo dominance(func);
  Value result;
  func.walk([&](Operation *candidate) {
    if (result || candidate == user || !dominance.dominates(candidate, user)) {
      return;
    }
    if (!afir::hasAxisBinding(candidate, axisId, role)) {
      return;
    }
    if (candidate->getNumResults() != 0) {
      result = candidate->getResult(0);
    }
  });
  return result;
}

// 先按名字收集函数入口块已有的 tiling 参数。descriptor 里首次出现的符号由
// `materializeExpression` 用调用方的 builder 补建，以便 dialect conversion 跟踪并下降。
SymbolBindingMap collectTilingParamBindings(func::FuncOp func) {
  SymbolBindingMap bindings;
  for (Operation &operation : func.getBody().front()) {
    auto param = dyn_cast<TilingParamOp>(operation);
    if (!param) {
      continue;
    }
    FailureOr<ExprAttr> symbol = makeExprAttr(func.getContext(), param.getSymbol());
    if (succeeded(symbol)) {
      bindings.insert({*symbol, param.getResult()});
    }
  }
  return bindings;
}

// 字面量 stride 直接物化为常量。
FailureOr<int64_t> getLiteralStride(ExprAttr expression) {
  if (!expression) {
    return failure();
  }
  int64_t value = 0;
  if (expression.getValue().getValue().trim().getAsInteger(10, value)) {
    return failure();
  }
  return value;
}

// 仅切分 inner 轴可能有尾块；Original、Merged 和切分 outer 使用标称范围。
static bool axisCanHaveTail(AxisAttr axis) {
  const bool isSplitInner =
      axis.getAxisType() == afir::AxisType::TileInner || axis.getAxisType() == afir::AxisType::BlockInner;
  return isSplitInner && axis.getSplitPairOtherId() >= 0;
}

// 将 extent 中的轴尺寸符号替换为 ActualSize，等价于 legacy ActualSize(expr)。
// 可能有尾块的轴缺少绑定时必须失败，不得用标称范围代替。
FailureOr<SymbolBindingMap> collectExtentBindings(func::FuncOp func, Operation *op, AscGraphAttrGroupsAttr graph,
                                                  ArrayRef<ExprAttr> extents) {
  SymbolBindingMap bindings = collectTilingParamBindings(func);
  llvm::SetVector<StringRef> boundToActual;
  llvm::SetVector<StringRef> needsActual;
  for (AxisAttr axis : graph.getAxes()) {
    // 仅替换裸尺寸符号；无尾块的派生 outer 表达式不参与替换。
    if (!axis.getSize()) {
      continue;
    }
    const StringRef symbol = axis.getSize().getValue().getValue();
    if (axisCanHaveTail(axis)) {
      needsActual.insert(symbol);
    }
    if (Value actual = findAxisValue(func, op, axis.getId(), afir::AxisValueRole::ActualSize)) {
      bindings[axis.getSize()] = actual;
      boundToActual.insert(symbol);
    }
  }

  llvm::SetVector<StringRef> used;
  for (ExprAttr extent : extents) {
    if (failed(collectFreeSymbols(op->getContext(), extent, used))) {
      return failure();
    }
  }
  for (StringRef symbol : used) {
    if (needsActual.contains(symbol) && !boundToActual.contains(symbol)) {
      return failure();
    }
  }
  return bindings;
}

// 表达式 -> index 值。字面量直接建常量，符号走 `materializeSymbolicExpr`。
FailureOr<Value> materializeExpression(OpBuilder &builder, Location location, ExprAttr expression,
                                       const SymbolBindingMap &bindings) {
  FailureOr<int64_t> literal = getLiteralStride(expression);
  if (succeeded(literal)) {
    return builder.create<arith::ConstantIndexOp>(location, *literal).getResult();
  }
  llvm::SetVector<StringRef> symbols;
  if (failed(collectFreeSymbols(builder.getContext(), expression, symbols))) {
    return failure();
  }
  SymbolBindingMap resolved = bindings;
  for (StringRef symbol : symbols) {
    FailureOr<ExprAttr> key = makeExprAttr(builder.getContext(), symbol);
    if (failed(key)) {
      return failure();
    }
    if (resolved.count(*key)) {
      continue;
    }
    Operation *parent = builder.getInsertionBlock()->getParentOp();
    auto func = dyn_cast<func::FuncOp>(parent);
    if (!func) {
      func = parent->getParentOfType<func::FuncOp>();
    }
    if (!func) {
      return failure();
    }
    resolved[*key] = getOrCreateTilingParam(builder, func.getBody().front(), symbol).getResult();
  }
  return materializeSymbolicExpr(builder, location, expression, resolved, builder.getIndexType());
}

FailureOr<Value> materializeSegmentExtent(Operation *op, OpBuilder &builder, const VectorSegment &segment,
                                          int64_t scheduledAxis, Value tailAwareSize) {
  if (op == nullptr || segment.axes.size() != segment.axisExtents.size() || segment.axes.empty()) {
    return failure();
  }
  Value extent;
  for (auto [axisId, axisExtent] : llvm::zip(segment.axes, segment.axisExtents)) {
    Value factor;
    if (tailAwareSize && axisId == scheduledAxis) {
      factor = tailAwareSize;
    } else {
      FailureOr<Value> materialized = materializeExtentValue(op, builder, axisExtent);
      if (failed(materialized)) {
        return failure();
      }
      factor = *materialized;
    }
    extent = extent ? builder.create<arith::MulIOp>(op->getLoc(), extent, factor).getResult() : factor;
  }
  return extent;
}

FailureOr<Value> materializeActualSegmentExtent(Operation *op, OpBuilder &builder, const VectorSegment &segment) {
  auto func = op->getParentOfType<func::FuncOp>();
  auto graph = func ? func->getAttrOfType<AscGraphAttrGroupsAttr>(kAscGraphAttrName) : AscGraphAttrGroupsAttr();
  if (!graph || segment.axisExtents.empty()) {
    return failure();
  }
  auto bindings = collectExtentBindings(func, op, graph, segment.axisExtents);
  if (failed(bindings)) {
    return failure();
  }
  Value result;
  for (ExprAttr extent : segment.axisExtents) {
    auto value = materializeExpression(builder, op->getLoc(), extent, *bindings);
    if (failed(value)) {
      return failure();
    }
    result = result ? builder.create<arith::MulIOp>(op->getLoc(), result, *value).getResult() : *value;
  }
  return result;
}

FailureOr<SmallVector<ScheduledAxisExtent, 4>> materializeScheduledAxisExtents(Operation *op, OpBuilder &builder,
                                                                               ArrayRef<int64_t> axes) {
  if (axes.empty()) {
    return failure();
  }
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func) {
    return failure();
  }
  auto graph = func->getAttrOfType<AscGraphAttrGroupsAttr>(afir::kAscGraphAttrName);
  if (!graph) {
    return failure();
  }

  llvm::DenseMap<int64_t, AxisAttr> axesById;
  for (AxisAttr axis : graph.getAxes()) {
    axesById.try_emplace(axis.getId(), axis);
  }
  SmallVector<ExprAttr, 4> requested;
  requested.reserve(axes.size());
  for (int64_t axisId : axes) {
    auto found = axesById.find(axisId);
    if (found == axesById.end() || !found->second.getSize()) {
      return failure();
    }
    requested.push_back(found->second.getSize());
  }

  const SymbolBindingMap nominalBindings = collectTilingParamBindings(func);
  FailureOr<SymbolBindingMap> actualBindings = collectExtentBindings(func, op, graph, requested);
  if (failed(actualBindings)) {
    return failure();
  }
  SmallVector<ScheduledAxisExtent, 4> values;
  values.reserve(requested.size());
  for (ExprAttr extent : requested) {
    FailureOr<Value> nominal = materializeExpression(builder, op->getLoc(), extent, nominalBindings);
    FailureOr<Value> actual = materializeExpression(builder, op->getLoc(), extent, *actualBindings);
    if (failed(nominal) || failed(actual)) {
      return failure();
    }
    values.push_back(ScheduledAxisExtent{*nominal, *actual});
  }
  return values;
}

}  // namespace mlir::afir
