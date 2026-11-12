/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// `matchBroadcastInlineOperands` 的覆盖：行广播（brc_inline）形态判据，与 legacy
// `IsGeneralizeBrcInlineScene`（`common/common_utils.cpp:856`）逐步对齐。设计见
// docs/development/2026-11-10-afir-broadcast-inline-design.md 功能需求 1。

#include "conversion/AFIRToAscendC/Implementations/ElementwiseTransferPlan.h"

#include <optional>
#include <string>

#include "dialect/AFIR/IR/AFIR.h"

#include "ascir/Dialect/Asc/IR/Asc.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

namespace mlir::afir {
using namespace mlir::ascend::kernel;
namespace {

void LoadDialects(MLIRContext &context) {
  context.getOrLoadDialect<afir::AFIRDialect>();
  context.getOrLoadDialect<ascendc::AscendCDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
}

Operation *FindOperation(ModuleOp module, llvm::StringRef operationName) {
  Operation *found = nullptr;
  module.walk([&](Operation *operation) {
    if (operation->getName().getStringRef() == operationName) {
      found = operation;
    }
  });
  return found;
}

// 两个输入与目的地的 descriptor 自己拼好传进来，不复用 `BinaryElementwiseL2CapabilityTest`
// 的 `ParseBufferizedBinary`——那个helper给的是裸 `role = <input>` descriptor（没有
// `axis`/`vectorized_axis`/`repeats`），这个判据需要真实的轴表。
OwningOpRef<ModuleOp> ParseBrcInlineCandidate(MLIRContext &context, llvm::StringRef operand0Descriptor,
                                              llvm::StringRef operand1Descriptor,
                                              llvm::StringRef destinationDescriptor) {
  std::string ir =
      ("module {\n"
       "  func.func @test(%lhs: !ascendc.local_tensor<?xf16>, %rhs: !ascendc.local_tensor<?xf16>, "
       "%destination: !ascendc.local_tensor<?xf16>) {\n"
       "    %result = afir.add %lhs, %rhs outs(%destination) {\n"
       "      afir.operand_descs = [" +
       operand0Descriptor.str() + ", " + operand1Descriptor.str() +
       "],\n"
       "      afir.destination_descs = [" +
       destinationDescriptor.str() +
       "]\n"
       "    } : (!ascendc.local_tensor<?xf16>, !ascendc.local_tensor<?xf16>), "
       "!ascendc.local_tensor<?xf16> -> !ascendc.local_tensor<?xf16>\n"
       "    return\n"
       "  }\n"
       "}\n");
  return parseSourceString<ModuleOp>(ir, &context);
}

// 形态：`(1, s1) + (s0, s1)`——广播侧外层轴（axis 0）repeat 恒为 1、步长为 0（这一轴上
// 数据不变）；内层轴（axis 1）两侧同宽 `s1`。两侧与目的地共用同一张轴表 `[0, 1]`，向量化
// 轴也是 `[0, 1]`——与 `afir-to-ascendc-scheduled-transfer.mlir` 等既有 lit 用例同一种
// 两轴形态，不是另起一套命名。
constexpr llvm::StringLiteral kBroadcastDescriptor =
    R"AF(#afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"1">, #afir.expr<"s1">], strides = [#afir.expr<"0">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"0">, #afir.expr<"1">], role = <input>>)AF";
constexpr llvm::StringLiteral kOtherDescriptor =
    R"AF(#afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"s0">, #afir.expr<"s1">], strides = [#afir.expr<"s1">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"s1">, #afir.expr<"1">], role = <input>>)AF";
constexpr llvm::StringLiteral kDestinationDescriptor =
    R"AF(#afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"s0">, #afir.expr<"s1">], strides = [#afir.expr<"s1">, #afir.expr<"1">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"s1">, #afir.expr<"1">], role = <output>>)AF";
// 广播轴挪到内层（axis 1）而不是外层（axis 0）——与 `kBroadcastDescriptor` 相比只是
// 「哪一轴是 1」换了位置，专用于「两侧各在不同轴上广播」这条反例，不代表另一种真实形态。
constexpr llvm::StringLiteral kBroadcastOnInnerAxisDescriptor =
    R"AF(#afir.tensor_desc<axis = [0, 1], repeats = [#afir.expr<"s0">, #afir.expr<"1">], strides = [#afir.expr<"s1">, #afir.expr<"0">], vectorized_axis = [0, 1], vectorized_strides = [#afir.expr<"s1">, #afir.expr<"0">], role = <input>>)AF";

TEST(ElementwiseTransferPlanTest, MatchesRowBroadcastOnTheFirstOperand) {
  MLIRContext context;
  LoadDialects(context);
  OwningOpRef<ModuleOp> module =
      ParseBrcInlineCandidate(context, kBroadcastDescriptor, kOtherDescriptor, kDestinationDescriptor);
  ASSERT_TRUE(module);
  Operation *add = FindOperation(*module, "afir.add");
  ASSERT_NE(add, nullptr);

  std::optional<elementwise::BroadcastInlineOperands> match = elementwise::matchBroadcastInlineOperands(add);
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->broadcastOperandIndex, 0U);
  EXPECT_EQ(match->outerRepeat.getValue().getValue(), "s0");
  EXPECT_EQ(match->innerRepeat.getValue().getValue(), "s1");
  EXPECT_EQ(match->firstAxisVStride.getValue().getValue(), "s1");
}

TEST(ElementwiseTransferPlanTest, MatchesRowBroadcastOnTheSecondOperand) {
  MLIRContext context;
  LoadDialects(context);
  // 与上一条对调：广播侧现在是第二个操作数。
  OwningOpRef<ModuleOp> module =
      ParseBrcInlineCandidate(context, kOtherDescriptor, kBroadcastDescriptor, kDestinationDescriptor);
  ASSERT_TRUE(module);
  Operation *add = FindOperation(*module, "afir.add");
  ASSERT_NE(add, nullptr);

  std::optional<elementwise::BroadcastInlineOperands> match = elementwise::matchBroadcastInlineOperands(add);
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->broadcastOperandIndex, 1U);
  EXPECT_EQ(match->outerRepeat.getValue().getValue(), "s0");
  EXPECT_EQ(match->innerRepeat.getValue().getValue(), "s1");
  EXPECT_EQ(match->firstAxisVStride.getValue().getValue(), "s1");
}

// 成对的反例：两侧同形（都没有广播轴）必须返回不支持——否则「判据只看一侧」这类退化
// 实现也能让上一条用例通过。
TEST(ElementwiseTransferPlanTest, RejectsSameShapedOperands) {
  MLIRContext context;
  LoadDialects(context);
  OwningOpRef<ModuleOp> module =
      ParseBrcInlineCandidate(context, kOtherDescriptor, kOtherDescriptor, kDestinationDescriptor);
  ASSERT_TRUE(module);
  Operation *add = FindOperation(*module, "afir.add");
  ASSERT_NE(add, nullptr);

  EXPECT_FALSE(elementwise::matchBroadcastInlineOperands(add).has_value());
}

// 另一组反例：两侧都带广播轴——legacy 与这里都返回不支持，互斥条件缺一半也会把这一条
// 放过去。
TEST(ElementwiseTransferPlanTest, RejectsBothOperandsHavingABroadcastAxis) {
  MLIRContext context;
  LoadDialects(context);
  // 第一个操作数在外层轴（axis 0）广播，第二个在内层轴（axis 1）广播——两侧各有一根
  // 广播轴，谁都不是纯粹的「另一侧」。
  OwningOpRef<ModuleOp> module =
      ParseBrcInlineCandidate(context, kBroadcastDescriptor, kBroadcastOnInnerAxisDescriptor, kDestinationDescriptor);
  ASSERT_TRUE(module);
  Operation *add = FindOperation(*module, "afir.add");
  ASSERT_NE(add, nullptr);

  EXPECT_FALSE(elementwise::matchBroadcastInlineOperands(add).has_value());
}

}  // namespace
}  // namespace mlir::afir
