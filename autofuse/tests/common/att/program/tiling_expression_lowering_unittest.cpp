/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <cstring>
#include <initializer_list>
#include "program/tiling_expression_lowering.h"
namespace att {
namespace {
using Kind = TilingProgramIRExpressionKind;
using Type = TilingProgramIRValueType;
using Op = TilingProgramIROpcode;
using Status = TilingExpressionStatus;
using Definition = TilingProgramIRExpressionDefinition;
TilingProgramIRExpression Node(Kind kind, Type type, Op op = Op::kNone, std::initializer_list<uint32_t> args = {}) {
  TilingProgramIRExpression node;
  node.kind = kind;
  node.value_type = type;
  node.opcode = op;
  node.arg_count = args.size();
  std::copy(args.begin(), args.end(), node.args.begin());
  return node;
}
TilingProgramIRExpression Constant(uint32_t id, Type type = Type::kInt64) {
  auto node = Node(Kind::kConstant, type);
  node.constant_id = id;
  return node;
}
TilingProgramIRExpression Symbol(uint32_t id, Type type) {
  auto node = Node(Kind::kSymbol, type);
  node.symbol_id = id;
  return node;
}
struct Context {
  std::vector<TilingProgramIRValue> values;
  std::vector<uint32_t> reads;
};
TilingProgramValueStatus Read(void *opaque, uint32_t slot, TilingProgramIRValue *value) {
  auto &context = *static_cast<Context *>(opaque);
  context.reads.push_back(slot);
  if (slot >= context.values.size()) {
    return TilingProgramValueStatus::kInvalidArgument;
  }
  *value = context.values[slot];
  return TilingProgramValueStatus::kOk;
}
std::shared_ptr<const TilingProgramIRCompiledExpression> Compile(const Definition &definition,
                                                                 TilingProgramIRExpressionCode *saved = nullptr) {
  TilingProgramIRExpressionCode code;
  EXPECT_EQ(LowerTilingProgramIRExpression(definition, &code).status, Status::kOk);
  std::shared_ptr<const TilingProgramIRCompiledExpression> compiled;
  EXPECT_EQ(BuildTilingProgramIRExpression(code, &compiled).status, Status::kOk);
  if (saved != nullptr) {
    *saved = code;
  }
  return compiled;
}
TilingExpressionError Evaluate(const TilingProgramIRCompiledExpression &compiled, Context &context,
                               TilingProgramIRValue &value) {
  std::array<TilingProgramIRValue, kTilingExpressionMaxStack> scratch;
  return EvaluateTilingProgramIRExpression(compiled, &context, Read, scratch.data(), scratch.size(), &value);
}
Definition Conditional() {
  Definition d;
  d.constants = {{Type::kInt64, 1}, {Type::kInt64, 0}, {Type::kInt64, 7}};
  d.input_types = {Type::kBoolean};
  d.nodes = {Symbol(0, Type::kBoolean),
             Constant(0),
             Constant(1),
             Node(Kind::kOperation, Type::kInt64, Op::kDiv, {1, 2}),
             Constant(2),
             Node(Kind::kConditional, Type::kInt64, Op::kNone, {0, 3, 4})};
  return d;
}
TilingExpressionLoweringError Reject(const Definition &definition, Status status = Status::kInvalidProgram) {
  TilingProgramIRExpressionCode output;
  output.result_type = Type::kUint8;
  output.constants = {{Type::kUint8, 99}};
  output.instructions.resize(1);
  output.instructions[0].symbol_id = 88;
  auto error = LowerTilingProgramIRExpression(definition, &output);
  EXPECT_EQ(error.status, status);
  EXPECT_EQ(output.result_type, Type::kUint8);
  EXPECT_EQ(output.constants.size(), 1U);
  EXPECT_EQ(output.constants[0].raw_bits, 99U);
  EXPECT_EQ(output.instructions.size(), 1U);
  EXPECT_EQ(output.instructions[0].symbol_id, 88U);
  return error;
}
}  // namespace
TEST(TilingExpressionLoweringTest, ConditionalDoesNotFoldOrEvaluateUnselectedDivision) {
  auto definition = Conditional();
  TilingProgramIRExpressionCode code;
  auto compiled = Compile(definition, &code);
  ASSERT_NE(compiled, nullptr);
  EXPECT_EQ(code.instructions.size(), 7U);
  Context context{{{Type::kBoolean, 0}}, {}};
  TilingProgramIRValue value{Type::kInt64, 99};
  EXPECT_EQ(Evaluate(*compiled, context, value).status, Status::kOk);
  EXPECT_EQ(value.raw_bits, 7U);
  context.values[0].raw_bits = 1;
  value.raw_bits = 99;
  EXPECT_EQ(Evaluate(*compiled, context, value).status, Status::kEvaluationError);
  EXPECT_EQ(value.raw_bits, 99U);
}
TEST(TilingExpressionLoweringTest, LogicalControlFlowMatchesNativeTruthAndReads) {
  for (Op op : {Op::kLogicalAnd, Op::kLogicalOr}) {
    Definition d;
    d.input_types = {Type::kBoolean, Type::kBoolean};
    d.nodes = {Symbol(0, Type::kBoolean), Symbol(1, Type::kBoolean),
               Node(Kind::kOperation, Type::kBoolean, op, {0, 1})};
    auto compiled = Compile(d);
    ASSERT_NE(compiled, nullptr);
    for (bool left : {false, true}) {
      for (bool right : {false, true}) {
        std::vector<uint32_t> reads;
        auto lhs = [&] {
          reads.push_back(0);
          return left;
        };
        auto rhs = [&] {
          reads.push_back(1);
          return right;
        };
        const bool native = op == Op::kLogicalAnd ? (lhs() && rhs()) : (lhs() || rhs());
        Context context{{{Type::kBoolean, left ? 1U : 0U}, {Type::kBoolean, right ? 1U : 0U}}, {}};
        TilingProgramIRValue value;
        ASSERT_EQ(Evaluate(*compiled, context, value).status, Status::kOk);
        EXPECT_EQ(value.raw_bits, native ? 1U : 0U);
        EXPECT_EQ(context.reads, reads);
      }
    }
  }
}
TEST(TilingExpressionLoweringTest, SharedNodesRemainAtEachOccurrenceInsideBranches) {
  Definition d;
  d.input_types = {Type::kBoolean, Type::kInt64};
  d.constants = {{Type::kInt64, 0}};
  d.nodes = {Symbol(0, Type::kBoolean), Symbol(1, Type::kInt64), Node(Kind::kOperation, Type::kInt64, Op::kAdd, {1, 1}),
             Constant(0), Node(Kind::kConditional, Type::kInt64, Op::kNone, {0, 2, 3})};
  auto compiled = Compile(d);
  ASSERT_NE(compiled, nullptr);
  for (bool condition : {false, true}) {
    Context context{{{Type::kBoolean, condition ? 1U : 0U}, {Type::kInt64, 9}}, {}};
    TilingProgramIRValue value;
    EXPECT_EQ(Evaluate(*compiled, context, value).status, Status::kOk);
    EXPECT_EQ(value.raw_bits, condition ? 18U : 0U);
    EXPECT_EQ(context.reads, condition ? (std::vector<uint32_t>{0, 1, 1}) : (std::vector<uint32_t>{0}));
  }
}
TEST(TilingExpressionLoweringTest, NativeTernaryRemainsEagerAndConvertsResult) {
  auto d = Conditional();
  d.nodes.back().kind = Kind::kOperation;
  d.nodes.back().opcode = Op::kTernary;
  d.nodes.back().value_type = Type::kFloat64;
  auto compiled = Compile(d);
  ASSERT_NE(compiled, nullptr);
  Context context{{{Type::kBoolean, 0}}, {}};
  TilingProgramIRValue value{Type::kInt64, 99};
  EXPECT_EQ(Evaluate(*compiled, context, value).status, Status::kEvaluationError);
  EXPECT_EQ(value.raw_bits, 99U);
  d.constants[1].raw_bits = 2;
  compiled = Compile(d);
  ASSERT_NE(compiled, nullptr);
  ASSERT_EQ(Evaluate(*compiled, context, value).status, Status::kOk);
  EXPECT_EQ(value.type, Type::kFloat64);
  EXPECT_EQ(value.raw_bits, UINT64_C(0x401c000000000000));
}
TEST(TilingExpressionLoweringTest, ExplicitCastsAndFloat32StepsAreNotReassociated) {
  Definition d;
  d.constants = {{Type::kFloat32, UINT64_C(0x4b800000)}, {Type::kFloat32, UINT64_C(0x3f800000)}};
  d.nodes = {Constant(0, Type::kFloat32), Constant(1, Type::kFloat32),
             Node(Kind::kOperation, Type::kFloat32, Op::kAdd, {0, 1}),
             Node(Kind::kOperation, Type::kFloat32, Op::kSub, {2, 0}),
             Node(Kind::kCast, Type::kFloat64, Op::kNone, {3})};
  auto compiled = Compile(d);
  ASSERT_NE(compiled, nullptr);
  Context context;
  TilingProgramIRValue value;
  ASSERT_EQ(Evaluate(*compiled, context, value).status, Status::kOk);
  EXPECT_EQ(value.type, Type::kFloat64);
  EXPECT_EQ(value.raw_bits, 0U);
  d.nodes.back().value_type = Type::kInt64;
  compiled = Compile(d);
  ASSERT_NE(compiled, nullptr);
  ASSERT_EQ(Evaluate(*compiled, context, value).status, Status::kOk);
  EXPECT_EQ(value.type, Type::kInt64);
}
TEST(TilingExpressionLoweringTest, RejectsCyclesForwardReferencesAndUnreachableNodes) {
  auto d = Conditional();
  d.nodes[3].args[0] = 3;
  EXPECT_EQ(Reject(d).node, 3U);
  d = Conditional();
  d.nodes[3].args[0] = 5;
  Reject(d);
  d = Conditional();
  d.nodes.back().args[1] = 4;
  Reject(d);
  d = Conditional();
  d.nodes.back().args[0] = UINT32_MAX;
  Reject(d);
}
TEST(TilingExpressionLoweringTest, RejectsWrongTypesAritiesAndNoncanonicalFields) {
  auto d = Conditional();
  d.nodes[3].value_type = Type::kUint64;
  Reject(d);
  d = Conditional();
  d.nodes.back().args[0] = 1;
  Reject(d);
  d = Conditional();
  d.nodes[1].symbol_id = 1;
  Reject(d);
  d = Conditional();
  d.nodes[1].args[2] = 1;
  Reject(d);
  d = Conditional();
  d.nodes[3].arg_count = 4;
  Reject(d);
  d = Conditional();
  d.nodes[3].opcode = Op::kAlign;
  Reject(d, Status::kUnsupported);
  d = Conditional();
  d.nodes[3].opcode = Op::kPow;
  Reject(d);
  d = Conditional();
  d.nodes.back().kind = Kind::kCast;
  Reject(d);
  d = Conditional();
  d.nodes[1].value_type = static_cast<Type>(255);
  Reject(d, Status::kUnsupported);
}
TEST(TilingExpressionLoweringTest, ValidatesTablesAndPreservesOutputOnFailure) {
  auto d = Conditional();
  d.constants[0] = {Type::kBoolean, 2};
  Reject(d);
  d = Conditional();
  d.input_types[0] = static_cast<Type>(255);
  Reject(d, Status::kUnsupported);
  d = Conditional();
  d.nodes[1].constant_id = 99;
  Reject(d);
  d = Conditional();
  d.nodes[0].symbol_id = 99;
  Reject(d);
  d = Conditional();
  d.constants.resize(kTilingExpressionMaxItems + 1);
  Reject(d);
  d = Conditional();
  d.input_types.resize(kTilingExpressionMaxItems + 1);
  Reject(d);
  d.nodes.clear();
  Reject(d);
  EXPECT_EQ(LowerTilingProgramIRExpression(Conditional(), nullptr).status, Status::kInvalidArgument);
}
TEST(TilingExpressionLoweringTest, DeepUnaryChainUsesBoundedExplicitTraversal) {
  Definition d;
  d.constants = {{Type::kBoolean, 0}};
  d.nodes.push_back(Constant(0, Type::kBoolean));
  for (uint32_t id = 1; id < kTilingExpressionMaxItems; ++id) {
    d.nodes.push_back(Node(Kind::kOperation, Type::kBoolean, Op::kNot, {id - 1}));
  }
  auto compiled = Compile(d);
  ASSERT_NE(compiled, nullptr);
  EXPECT_EQ(GetTilingExpressionScratchElements(*compiled), 1U);
  Context context;
  TilingProgramIRValue value;
  EXPECT_EQ(Evaluate(*compiled, context, value).status, Status::kOk);
  EXPECT_EQ(value.raw_bits, 1U);
  d.nodes.push_back(Node(Kind::kOperation, Type::kBoolean, Op::kNot, {65535}));
  Reject(d);
}
TEST(TilingExpressionLoweringTest, SharedDagExpansionIsBoundedBeforePublication) {
  Definition d;
  d.constants = {{Type::kInt64, 1}};
  d.nodes.push_back(Constant(0));
  for (uint32_t id = 1; id <= 15; ++id) {
    d.nodes.push_back(Node(Kind::kOperation, Type::kInt64, Op::kAdd, {id - 1, id - 1}));
  }
  TilingProgramIRExpressionCode code;
  auto compiled = Compile(d, &code);
  ASSERT_NE(compiled, nullptr);
  EXPECT_EQ(code.instructions.size(), 65535U);
  Context context;
  TilingProgramIRValue value;
  EXPECT_EQ(Evaluate(*compiled, context, value).status, Status::kOk);
  EXPECT_EQ(value.raw_bits, 32768U);
  d.nodes.push_back(Node(Kind::kOperation, Type::kInt64, Op::kAdd, {15, 15}));
  Reject(d);
}
TEST(TilingExpressionLoweringTest, CommonVerifierRejectsExcessiveOperandStackWithSourceNode) {
  Definition d;
  d.constants = {{Type::kInt64, 1}};
  d.nodes.push_back(Constant(0));
  for (uint32_t i = 0; i < kTilingExpressionMaxStack; ++i) {
    const uint32_t previous = d.nodes.size() - 1;
    d.nodes.push_back(Constant(0));
    d.nodes.push_back(Node(Kind::kOperation, Type::kInt64, Op::kAdd, {previous + 1, previous}));
  }
  auto error = Reject(d);
  EXPECT_EQ(error.code_error.status, Status::kInvalidProgram);
  EXPECT_NE(error.code_error.instruction, UINT32_MAX);
  EXPECT_NE(error.node, UINT32_MAX);
}
TEST(TilingExpressionLoweringTest, PublishedCodeOwnsItsTables) {
  auto d = Conditional();
  auto compiled = Compile(d);
  ASSERT_NE(compiled, nullptr);
  d.nodes.clear();
  d.constants.clear();
  d.input_types.clear();
  Context context{{{Type::kBoolean, 0}}, {}};
  TilingProgramIRValue value;
  ASSERT_EQ(Evaluate(*compiled, context, value).status, Status::kOk);
  EXPECT_EQ(value.raw_bits, 7U);
}

TEST(TilingExpressionLoweringTest, NestedLogicalAndConditionalMatchNativeSelectedReads) {
  Definition d;
  d.input_types.assign(4, Type::kBoolean);
  d.nodes = {Symbol(0, Type::kBoolean),
             Symbol(1, Type::kBoolean),
             Symbol(2, Type::kBoolean),
             Symbol(3, Type::kBoolean),
             Node(Kind::kConditional, Type::kBoolean, Op::kNone, {1, 2, 3}),
             Node(Kind::kOperation, Type::kBoolean, Op::kLogicalAnd, {0, 4})};
  auto compiled = Compile(d);
  ASSERT_NE(compiled, nullptr);
  for (uint32_t bits = 0; bits < 16; ++bits) {
    Context context;
    for (uint32_t slot = 0; slot < 4; ++slot) {
      context.values.push_back({Type::kBoolean, (bits >> slot) & 1U});
    }
    std::vector<uint32_t> reads;
    auto read = [&](uint32_t slot) {
      reads.push_back(slot);
      return ((bits >> slot) & 1U) != 0;
    };
    const bool expected = read(0) && (read(1) ? read(2) : read(3));
    TilingProgramIRValue value;
    ASSERT_EQ(Evaluate(*compiled, context, value).status, Status::kOk);
    EXPECT_EQ(value.raw_bits, expected ? 1U : 0U);
    EXPECT_EQ(context.reads, reads);
  }
}
}  // namespace att
