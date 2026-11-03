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
#include "dialect/AFIR/Transforms/Symbolic/SymbolicTransformDetail.h"

#include <functional>
#include <string>

#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir::afir {
namespace {

TEST(SymbolicTransformDetailTest, ParsesCanonicalBooleanLiteralsStrictly) {
  FailureOr<bool> trueValue = detail::parseBoolean({detail::NodeKind::Boolean, "True", {}});
  FailureOr<bool> falseValue = detail::parseBoolean({detail::NodeKind::Boolean, "False", {}});
  ASSERT_TRUE(succeeded(trueValue));
  EXPECT_TRUE(*trueValue);
  ASSERT_TRUE(succeeded(falseValue));
  EXPECT_FALSE(*falseValue);
  EXPECT_TRUE(failed(detail::parseBoolean({detail::NodeKind::Boolean, "true", {}})));
  EXPECT_TRUE(failed(detail::parseBoolean({detail::NodeKind::Symbol, "True", {}})));
}

std::string CaptureDiagnostic(MLIRContext &context, const std::function<void()> &action) {
  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &entry) {
    diagnostic += entry.str();
    return success();
  });
  action();
  return diagnostic;
}

detail::ExpressionNode MakeIntegerPower(int64_t exponent) {
  return {detail::NodeKind::Pow,
          "",
          {{detail::NodeKind::Symbol, "__arg0", {}}, {detail::NodeKind::Integer, std::to_string(exponent), {}}}};
}

SymbolicEvalOp MakeSymbolicEvalOp(OpBuilder &builder, Block &entry, bool overflowChecked) {
  FailureOr<ExprAttr> expression = makeExprAttr(builder.getContext(), "(__arg0 + __arg1)");
  if (failed(expression)) {
    ADD_FAILURE() << "failed to build the placeholder expression";
    return SymbolicEvalOp();
  }
  return builder.create<SymbolicEvalOp>(
      builder.getUnknownLoc(), builder.getIndexType(), ValueRange{entry.getArgument(0), entry.getArgument(1)},
      *expression, DenseI64ArrayAttr(), overflowChecked ? UnitAttr::get(builder.getContext()) : UnitAttr(), UnitAttr());
}

class AFIRSymbolicTest : public testing::Test {
 protected:
  void SetUp() override {
    context.loadDialect<AFIRDialect, func::FuncDialect>();
    OpBuilder builder(&context);
    module = ModuleOp::create(builder.getUnknownLoc());
    function = func::FuncOp::create(builder.getUnknownLoc(), "host_tiling",
                                    builder.getFunctionType({builder.getIndexType(), builder.getIndexType()}, {}));
    module->push_back(function);
    entry = function.addEntryBlock();
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
  func::FuncOp function;
  Block *entry = nullptr;
};

TEST_F(AFIRSymbolicTest, MaterializesNormalizedExpressionWithStableOperandOrder) {
  OpBuilder builder(entry, entry->end());
  FailureOr<ExprAttr> expression = makeExprAttr(&context, "Max((s0 + s2), s1)");
  FailureOr<ExprAttr> s0 = makeExprAttr(&context, "s0");
  FailureOr<ExprAttr> s1 = makeExprAttr(&context, "s1");
  FailureOr<ExprAttr> s2 = makeExprAttr(&context, "s2");
  ASSERT_TRUE(succeeded(expression) && succeeded(s0) && succeeded(s1) && succeeded(s2));

  SymbolBindingMap bindings;
  bindings.insert({*s2, entry->getArgument(0)});
  bindings.insert({*s0, entry->getArgument(1)});
  bindings.insert({*s1, entry->getArgument(0)});
  FailureOr<Value> value = materializeSymbolicExpr(builder, builder.getUnknownLoc(), *expression, bindings);
  ASSERT_TRUE(succeeded(value));

  auto op = value->getDefiningOp<SymbolicEvalOp>();
  ASSERT_TRUE(op);
  EXPECT_EQ(op.getExpr().getValue().getValue(), "Max((__arg0 + __arg2), __arg1)");
  ASSERT_EQ(op.getInputs().size(), 3U);
  EXPECT_EQ(op.getInputs()[0], entry->getArgument(1));
  EXPECT_EQ(op.getInputs()[1], entry->getArgument(0));
  EXPECT_EQ(op.getInputs()[2], entry->getArgument(0));
}

TEST_F(AFIRSymbolicTest, ReusesExactExpressionBinding) {
  OpBuilder builder(entry, entry->end());
  FailureOr<ExprAttr> expression = makeExprAttr(&context, "(s0 + s1)");
  ASSERT_TRUE(succeeded(expression));
  SymbolBindingMap bindings;
  bindings.insert({*expression, entry->getArgument(0)});

  FailureOr<Value> value = materializeSymbolicExpr(builder, builder.getUnknownLoc(), *expression, bindings);
  ASSERT_TRUE(succeeded(value));
  EXPECT_EQ(*value, entry->getArgument(0));
  EXPECT_TRUE(entry->empty());
}

TEST_F(AFIRSymbolicTest, RejectsUnboundSymbol) {
  OpBuilder builder(entry, entry->end());
  FailureOr<ExprAttr> expression = makeExprAttr(&context, "(s0 + s1)");
  FailureOr<ExprAttr> s0 = makeExprAttr(&context, "s0");
  ASSERT_TRUE(succeeded(expression) && succeeded(s0));
  SymbolBindingMap bindings;
  bindings.insert({*s0, entry->getArgument(0)});

  ScopedDiagnosticHandler diagnostics(&context, [](Diagnostic &) {});
  EXPECT_TRUE(failed(materializeSymbolicExpr(builder, builder.getUnknownLoc(), *expression, bindings)));
}

TEST_F(AFIRSymbolicTest, RejectsReservedPlaceholderInSourceExpression) {
  OpBuilder builder(entry, entry->end());
  FailureOr<ExprAttr> expression = makeExprAttr(&context, "(__arg0 + s0)");
  ASSERT_TRUE(succeeded(expression));
  ScopedDiagnosticHandler diagnostics(&context, [](Diagnostic &) {});
  EXPECT_TRUE(failed(materializeSymbolicExpr(builder, builder.getUnknownLoc(), *expression, {})));
}

// Canonical serialization expands positive integer powers into multiplication, so these Pow nodes
// cannot be produced by the textual grammar today. The node-level seam still pins the contract for
// any future producer that constructs the tree before serialization.
TEST_F(AFIRSymbolicTest, RejectsUncheckedIntegerPowerNode) {
  OpBuilder builder(entry, entry->end());
  SymbolicEvalOp op = MakeSymbolicEvalOp(builder, *entry, false);
  const std::string diagnostic =
      CaptureDiagnostic(context, [&] { (void)detail::verifySymbolicIndexSafety(MakeIntegerPower(2), op); });
  EXPECT_NE(diagnostic.find("integer power requires an explicit overflow_checked proof"), std::string::npos)
      << diagnostic;
}

TEST_F(AFIRSymbolicTest, RejectsIntegerPowerNodeAboveLoweringLimit) {
  OpBuilder builder(entry, entry->end());
  SymbolicEvalOp op = MakeSymbolicEvalOp(builder, *entry, true);
  const std::string diagnostic = CaptureDiagnostic(context, [&] {
    (void)detail::verifySymbolicIndexSafety(MakeIntegerPower(detail::kMaxSymbolicPowExponent + 1), op);
  });
  EXPECT_NE(diagnostic.find("integer power exponent exceeds the lowering limit of 8"), std::string::npos) << diagnostic;
}

}  // namespace
}  // namespace mlir::afir
