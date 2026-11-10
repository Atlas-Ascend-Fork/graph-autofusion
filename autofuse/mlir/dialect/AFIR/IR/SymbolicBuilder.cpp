/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dialect/AFIR/IR/SymbolicBuilder.h"

#include <string>
#include <vector>

#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::afir {

FailureOr<ExprAttr> makeExprAttr(MLIRContext *context, StringRef canonicalExpression) {
  ExprAttr attr = ExprAttr::getChecked([&]() { return emitError(UnknownLoc::get(context)); }, context,
                                       StringAttr::get(context, canonicalExpression));
  return attr ? FailureOr<ExprAttr>(attr) : FailureOr<ExprAttr>(failure());
}

FailureOr<ExprAttr> multiplyExprs(MLIRContext *context, ExprAttr lhs, ExprAttr rhs) {
  if (!lhs || !rhs) {
    return failure();
  }
  Result<std::string> product = MultiplyExpressions(lhs.getValue().getValue(), rhs.getValue().getValue());
  if (!product) {
    return failure();
  }
  return makeExprAttr(context, product.value());
}

FailureOr<ExprAttr> alignUpExpr(MLIRContext *context, ExprAttr extent, int64_t alignNum) {
  if (!extent || alignNum <= 0) {
    return failure();
  }
  if (alignNum == 1) {
    return extent;
  }
  // 模板按 `alignNum` 现场拼出来，但拼进去的只有字面量，符号位仍是占位符——这与
  // 「拼字符串造表达式」不同：`Rational(1 , N)` 与 `N * ...` 是序列化器对这个式子的原样
  // 输出，所以模板本身是规范的，Parse 不会把它判成非规范。
  const std::string pattern =
      "(" + std::to_string(alignNum) + " * Ceiling((Rational(1 , " + std::to_string(alignNum) + ") * __arg0)))";
  Result<Expression> aligned = Expression::Parse(pattern);
  if (!aligned) {
    return failure();
  }
  Result<Expression> substituted =
      detail::ExpressionAccess::Substitute(aligned.value(), {{"__arg0", extent.getValue().getValue().str()}});
  if (!substituted) {
    return failure();
  }
  Result<Expression> simplified = substituted.value().Simplify();
  if (!simplified) {
    return failure();
  }
  return makeExprAttr(context, simplified.value().Serialize());
}

bool provablyContiguous(MLIRContext *context, ExprAttr outerStride, ExprAttr innerStride, ExprAttr innerExtent) {
  if (!outerStride) {
    return false;
  }
  FailureOr<ExprAttr> expected = multiplyExprs(context, innerStride, innerExtent);
  if (failed(expected)) {
    return false;
  }
  StringRef outerText = outerStride.getValue().getValue();
  StringRef expectedText = expected->getValue().getValue();
  return outerText == expectedText || ProveExpressionsEqual(outerText, expectedText);
}

FailureOr<Value> materializeSymbolicExpr(OpBuilder &builder, Location location, ExprAttr expression,
                                         const SymbolBindingMap &bindings, Type resultType) {
  if (auto existing = bindings.find(expression); existing != bindings.end()) {
    return existing->second;
  }
  Result<Expression> parsed = Expression::Parse(expression.getValue().getValue());
  if (!parsed) {
    emitError(location) << "cannot materialize invalid symbolic expression: " << parsed.error().message;
    return failure();
  }
  Result<detail::NormalizedExpression> normalized = detail::ExpressionAccess::NormalizeFreeSymbols(parsed.value());
  if (!normalized) {
    emitError(location) << "cannot normalize symbolic expression: " << normalized.error().message;
    return failure();
  }

  SmallVector<Value> operands;
  operands.reserve(normalized.value().symbols.size());
  for (const std::string &symbol : normalized.value().symbols) {
    FailureOr<ExprAttr> symbolAttr = makeExprAttr(builder.getContext(), symbol);
    if (failed(symbolAttr)) {
      return failure();
    }
    auto binding = bindings.find(*symbolAttr);
    if (binding == bindings.end()) {
      emitError(location) << "unbound symbol '" << symbol << "' while materializing "
                          << expression.getValue().getValue();
      return failure();
    }
    operands.push_back(binding->second);
  }
  if (!resultType) {
    resultType = operands.empty() ? builder.getIndexType() : operands.front().getType();
  }
  FailureOr<ExprAttr> normalizedAttr = makeExprAttr(builder.getContext(), normalized.value().expression.Serialize());
  if (failed(normalizedAttr)) {
    return failure();
  }

  return builder
      .create<SymbolicEvalOp>(location, resultType, operands, *normalizedAttr, DenseI64ArrayAttr(), UnitAttr(),
                              UnitAttr())
      .getResult();
}

namespace {

// 入口块里那一份 `afir.tiling_param`，没有就返回空。
//
// 只认入口块。嵌套区域里同名的值支配不了入口块之后的全部消费者，复用它就是在制造一个
// 只在某个循环体内成立的支配关系。
TilingParamOp findTilingParam(Block &entry, StringRef symbol) {
  for (Operation &op : entry) {
    auto existing = dyn_cast<TilingParamOp>(op);
    if (existing && existing.getSymbol() == symbol) {
      return existing;
    }
  }
  return {};
}

}  // namespace

TilingParamOp getOrCreateTilingParam(Block &entry, StringRef symbol) {
  OpBuilder builder(&entry, entry.begin());
  return getOrCreateTilingParam(builder, entry, symbol);
}

TilingParamOp getOrCreateTilingParam(OpBuilder &builder, Block &entry, StringRef symbol) {
  if (TilingParamOp existing = findTilingParam(entry, symbol)) {
    return existing;
  }
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&entry);
  Location location = entry.getParentOp() ? entry.getParentOp()->getLoc() : builder.getUnknownLoc();
  return builder.create<TilingParamOp>(location, builder.getIndexType(), symbol);
}

namespace {

// 尺寸序言的末尾：块首那一段连续的 `afir.tiling_param` / `afir.symbolic_eval` 之后。
//
// 判据写成「块首的连续一段」而不是「全块最后一个同类 Op」：函数体中间也会有
// `afir.symbolic_eval`（循环生成算轴尺寸时就地产的那些），把插入点挪到它们之后就会
// 跳过一整段本该被支配的代码。
Block::iterator sizePrologueEnd(Block &entry) {
  Block::iterator position = entry.begin();
  while (position != entry.end() && isa<TilingParamOp, SymbolicEvalOp>(*position)) {
    ++position;
  }
  return position;
}

}  // namespace

FailureOr<Value> materializeRuntimeSize(Block &entry, Location location, ExprAttr expression) {
  Result<Expression> parsed = Expression::Parse(expression.getValue().getValue());
  if (!parsed) {
    emitError(location) << "cannot materialize invalid runtime size expression: " << parsed.error().message;
    return failure();
  }
  Result<std::vector<std::string>> symbols = detail::ExpressionAccess::GetFreeSymbols(parsed.value());
  if (!symbols) {
    emitError(location) << "cannot resolve runtime size expression symbols: " << symbols.error().message;
    return failure();
  }

  SymbolBindingMap bindings;
  for (const std::string &symbol : symbols.value()) {
    FailureOr<ExprAttr> symbolAttr = makeExprAttr(expression.getContext(), symbol);
    if (failed(symbolAttr)) {
      return failure();
    }
    bindings[*symbolAttr] = getOrCreateTilingParam(entry, symbol).getResult();
  }
  // 插入点在新建的那批 `afir.tiling_param` 之后取，顺序才对：它们各自插在块首。
  OpBuilder builder(&entry, sizePrologueEnd(entry));
  return materializeSymbolicExpr(builder, location, expression, bindings, builder.getIndexType());
}

FailureOr<Value> materializeSizeAt(OpBuilder &builder, Block &entry, Location location, ExprAttr expression) {
  Result<Expression> parsed = Expression::Parse(expression.getValue().getValue());
  if (!parsed) {
    emitError(location) << "cannot materialize invalid size expression: " << parsed.error().message;
    return failure();
  }
  Result<std::vector<std::string>> symbols = detail::ExpressionAccess::GetFreeSymbols(parsed.value());
  if (!symbols) {
    emitError(location) << "cannot resolve size expression symbols: " << symbols.error().message;
    return failure();
  }
  SymbolBindingMap bindings;
  for (const std::string &symbol : symbols.value()) {
    FailureOr<ExprAttr> symbolAttr = makeExprAttr(expression.getContext(), symbol);
    if (failed(symbolAttr)) {
      return failure();
    }
    // 用调用方的 builder 建，不自己开一个。 新建的那条 `afir.tiling_param` 落在入口块
    // 顶部（插入点由 `getOrCreateTilingParam` 自己挪、用完还原），但建它的是同一个
    // `ConversionPatternRewriter`，所以它进得了 dialect conversion 的建档：要么被
    // legalize，要么随 pattern 回滚。自己开一个 `OpBuilder` 建出来的那条谁都看不见，
    // `applyFullConversion` 连它非法都不知道，会把它原样留在下降完的 IR 里。
    bindings[*symbolAttr] = getOrCreateTilingParam(builder, entry, symbol).getResult();
  }
  return materializeSymbolicExpr(builder, location, expression, bindings, builder.getIndexType());
}

namespace {
// `materializeExtentValue` / `materializeStrideValue` 的共同部分：字面量走常量，符号走
// `materializeSizeAt`（自由符号绑到入口块的 `afir.tiling_param`）。差别只有字面量的下界。
FailureOr<Value> materializeDescriptorValue(Operation *op, OpBuilder &builder, ExprAttr expression,
                                            int64_t minimumLiteral) {
  if (op == nullptr || !expression) {
    return failure();
  }
  int64_t literal = 0;
  if (!expression.getValue().getValue().getAsInteger(10, literal)) {
    if (literal < minimumLiteral) {
      return failure();
    }
    return builder.create<arith::ConstantIndexOp>(op->getLoc(), literal).getResult();
  }
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func || func.getBody().empty()) {
    return failure();
  }
  return materializeSizeAt(builder, func.front(), op->getLoc(), expression);
}
}  // namespace

FailureOr<Value> materializeExtentValue(Operation *op, OpBuilder &builder, ExprAttr expression) {
  return materializeDescriptorValue(op, builder, expression, /*minimumLiteral=*/1);
}

FailureOr<Value> materializeStrideValue(Operation *op, OpBuilder &builder, ExprAttr expression) {
  return materializeDescriptorValue(op, builder, expression, /*minimumLiteral=*/0);
}

}  // namespace mlir::afir
