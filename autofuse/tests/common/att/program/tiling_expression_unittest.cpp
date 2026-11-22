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
#include <future>
#include <array>
#include "program/tiling_expression.h"
namespace att {
namespace {
using Kind = TilingProgramIRInstructionKind;
using Type = TilingProgramIRValueType;
using Status = TilingExpressionStatus;
TilingExpressionError Evaluate(const TilingProgramIRCompiledExpression &code, void *context,
                               TilingExpressionReadSlot read_slot, TilingProgramIRValue *output) {
  std::array<TilingProgramIRValue, kTilingExpressionMaxStack> scratch;
  return EvaluateTilingProgramIRExpression(code, context, read_slot, scratch.data(), scratch.size(), output);
}

TilingProgramIRInstruction Inst(Kind kind, uint32_t operand = 0) {
  TilingProgramIRInstruction i;
  i.kind = kind;
  switch (kind) {
    case Kind::kConstant:
      i.constant_id = operand;
      break;
    case Kind::kSymbol:
      i.symbol_id = operand;
      break;
    case Kind::kJump:
    case Kind::kJumpIfFalse:
    case Kind::kJumpIfTrue:
      i.jump = operand;
      break;
    case Kind::kPushBoolean:
      i.boolean_value = operand != 0;
      break;
    default:
      break;
  }
  return i;
}
TilingProgramIRInstruction Binary(TilingProgramIROpcode op) {
  auto i = Inst(Kind::kOperation);
  i.opcode = op;
  i.arg_count = 2;
  return i;
}
TilingProgramIRExpressionCode Conditional() {
  TilingProgramIRExpressionCode c;
  c.constants = {{Type::kInt64, 1}, {Type::kInt64, 0}, {Type::kInt64, 7}};
  c.input_types = {Type::kBoolean};
  c.instructions = {Inst(Kind::kSymbol),      Inst(Kind::kJumpIfFalse, 6),         Inst(Kind::kConstant),
                    Inst(Kind::kConstant, 1), Binary(TilingProgramIROpcode::kDiv), Inst(Kind::kJump, 7),
                    Inst(Kind::kConstant, 2)};
  return c;
}
struct Context {
  std::vector<TilingProgramIRValue> values;
  std::vector<uint32_t> reads;
  uint32_t fail = UINT32_MAX;
};
TilingProgramValueStatus Read(void *opaque, uint32_t slot, TilingProgramIRValue *out) {
  auto &c = *static_cast<Context *>(opaque);
  c.reads.push_back(slot);
  if (slot == c.fail || slot >= c.values.size()) {
    return TilingProgramValueStatus::kInvalidValue;
  }
  *out = c.values[slot];
  return TilingProgramValueStatus::kOk;
}
std::shared_ptr<const TilingProgramIRCompiledExpression> Compile(const TilingProgramIRExpressionCode &c) {
  std::shared_ptr<const TilingProgramIRCompiledExpression> out;
  EXPECT_EQ(BuildTilingProgramIRExpression(c, &out).status, Status::kOk);
  return out;
}
void Invalid(const TilingProgramIRExpressionCode &c, Status status = Status::kInvalidProgram) {
  auto sentinel = Compile(Conditional());
  ASSERT_NE(sentinel, nullptr);
  auto saved = sentinel;
  EXPECT_EQ(BuildTilingProgramIRExpression(c, &sentinel).status, status);
  EXPECT_EQ(sentinel, saved);
}
}  // namespace
TEST(TilingExpressionTest, UnselectedDivisionDoesNotExecuteAndFirstErrorIsRetained) {
  auto code = Compile(Conditional());
  ASSERT_NE(code, nullptr);
  Context c{{{Type::kBoolean, 0}}, {}, UINT32_MAX};
  TilingProgramIRValue out{Type::kUint32, 99};
  EXPECT_EQ(Evaluate(*code, &c, Read, &out).status, Status::kOk);
  EXPECT_EQ(out.raw_bits, 7U);
  c.values[0].raw_bits = 1;
  out = {Type::kUint32, 99};
  auto error = Evaluate(*code, &c, Read, &out);
  EXPECT_EQ(error.status, Status::kEvaluationError);
  EXPECT_EQ(error.instruction, 4U);
  EXPECT_EQ(error.value_status, TilingProgramValueStatus::kArithmeticError);
  EXPECT_EQ(out.raw_bits, 99U);
  EXPECT_EQ(out.type, Type::kUint32);
  c.values[0].raw_bits = 0;
  EXPECT_EQ(Evaluate(*code, &c, Read, &out).status, Status::kOk);
}
TEST(TilingExpressionTest, LogicalAndOrReadOnlyRequiredSlots) {
  for (bool is_and : {true, false}) {
    TilingProgramIRExpressionCode c;
    c.result_type = Type::kBoolean;
    c.input_types = {Type::kBoolean, Type::kBoolean};
    auto jump = is_and ? Kind::kJumpIfFalse : Kind::kJumpIfTrue;
    c.instructions = {
        Inst(Kind::kSymbol, 0),           Inst(jump, 6),        Inst(Kind::kSymbol, 1),           Inst(jump, 6),
        Inst(Kind::kPushBoolean, is_and), Inst(Kind::kJump, 7), Inst(Kind::kPushBoolean, !is_and)};
    auto code = Compile(c);
    ASSERT_NE(code, nullptr);
    Context ctx{{{Type::kBoolean, is_and ? 0U : 1U}}, {}, 1};
    TilingProgramIRValue out;
    EXPECT_EQ(Evaluate(*code, &ctx, Read, &out).status, Status::kOk);
    EXPECT_EQ(ctx.reads, (std::vector<uint32_t>{0}));
    EXPECT_EQ(out.raw_bits, is_and ? 0U : 1U);
    ctx.values[0].raw_bits = is_and ? 1U : 0U;
    ctx.reads.clear();
    out = {Type::kInt64, 99};
    auto error = Evaluate(*code, &ctx, Read, &out);
    EXPECT_EQ(error.status, Status::kInputError);
    EXPECT_EQ(error.instruction, 2U);
    EXPECT_EQ(ctx.reads, (std::vector<uint32_t>{0, 1}));
    EXPECT_EQ(out.raw_bits, 99U);
  }
}
TEST(TilingExpressionTest, SuccessfulBranchesMatchNativeControlFlow) {
  for (bool is_and : {true, false}) {
    TilingProgramIRExpressionCode c;
    c.result_type = Type::kBoolean;
    c.input_types = {Type::kBoolean, Type::kBoolean};
    const auto jump = is_and ? Kind::kJumpIfFalse : Kind::kJumpIfTrue;
    c.instructions = {
        Inst(Kind::kSymbol, 0),           Inst(jump, 6),        Inst(Kind::kSymbol, 1),           Inst(jump, 6),
        Inst(Kind::kPushBoolean, is_and), Inst(Kind::kJump, 7), Inst(Kind::kPushBoolean, !is_and)};
    auto code = Compile(c);
    ASSERT_NE(code, nullptr);
    for (bool a : {false, true}) {
      for (bool b : {false, true}) {
        std::vector<uint32_t> native_reads;
        auto first = [&] {
          native_reads.push_back(0);
          return a;
        };
        auto second = [&] {
          native_reads.push_back(1);
          return b;
        };
        const bool native = is_and ? (first() && second()) : (first() || second());
        Context ctx{{{Type::kBoolean, a ? 1U : 0U}, {Type::kBoolean, b ? 1U : 0U}}, {}, UINT32_MAX};
        TilingProgramIRValue output;
        EXPECT_EQ(Evaluate(*code, &ctx, Read, &output).status, Status::kOk);
        EXPECT_EQ(output.raw_bits, native ? 1U : 0U);
        EXPECT_EQ(ctx.reads, native_reads);
      }
    }
  }
  auto c = Conditional();
  c.constants[1].raw_bits = 2;
  auto code = Compile(c);
  ASSERT_NE(code, nullptr);
  for (bool condition : {false, true}) {
    Context ctx{{{Type::kBoolean, condition ? 1U : 0U}}, {}, UINT32_MAX};
    TilingProgramIRValue output;
    ASSERT_EQ(Evaluate(*code, &ctx, Read, &output).status, Status::kOk);
    EXPECT_EQ(output.raw_bits, static_cast<uint64_t>(condition ? 1 / 2 : 7));
  }
}

TEST(TilingExpressionTest, ValidatesEveryBranchAndJoinBeforeExecution) {
  auto c = Conditional();
  c.instructions[4].arg_count = 1;
  Invalid(c);
  c = Conditional();
  c.constants[2].type = Type::kUint64;
  Invalid(c);
  c = Conditional();
  c.instructions[3].constant_id = 99;
  Invalid(c);
  c = Conditional();
  c.instructions[4].opcode = static_cast<TilingProgramIROpcode>(65535);
  Invalid(c, Status::kUnsupported);
  c = Conditional();
  c.instructions[4].opcode = TilingProgramIROpcode::kLogicalAnd;
  Invalid(c, Status::kUnsupported);
  c = Conditional();
  c.instructions[4].kind = Kind::kReciprocal;
  Invalid(c, Status::kUnsupported);
}
TEST(TilingExpressionTest, RejectsBackEdgesOutOfRangeUnreachableAndUnusedFields) {
  auto c = Conditional();
  c.instructions[1].jump = 0;
  Invalid(c);
  c = Conditional();
  c.instructions[1].jump = 99;
  Invalid(c);
  c = Conditional();
  c.instructions[0] = Inst(Kind::kJump, 6);
  Invalid(c);
  c = Conditional();
  c.instructions[0].arg_count = 1;
  Invalid(c);
  c = Conditional();
  c.result_type = Type::kBoolean;
  Invalid(c);
  c = Conditional();
  c.instructions[0] = Inst(Kind::kConstant);
  Invalid(c);
  c = Conditional();
  c.instructions.clear();
  Invalid(c);
}
TEST(TilingExpressionTest, EnforcesResourceAndValueBounds) {
  auto c = Conditional();
  c.instructions.resize(kTilingExpressionMaxItems + 1);
  Invalid(c);
  c = Conditional();
  c.constants.resize(kTilingExpressionMaxItems + 1);
  Invalid(c);
  c = Conditional();
  c.input_types.resize(kTilingExpressionMaxItems + 1, Type::kBoolean);
  Invalid(c);
  c = Conditional();
  c.instructions.assign(kTilingExpressionMaxStack + 1, Inst(Kind::kConstant));
  Invalid(c);
  c = Conditional();
  c.constants[0] = {Type::kBoolean, 2};
  Invalid(c);
  c = Conditional();
  c.input_types[0] = static_cast<Type>(255);
  Invalid(c, Status::kUnsupported);
}
TEST(TilingExpressionTest, OwnedCodeAndConcurrentCallsAreIndependent) {
  auto c = Conditional();
  auto code = Compile(c);
  ASSERT_NE(code, nullptr);
  c.instructions.clear();
  c.constants.clear();
  c.input_types.clear();
  std::vector<std::future<bool>> tasks;
  for (size_t thread = 0; thread < 4; ++thread) {
    tasks.push_back(std::async(std::launch::async, [code] {
      Context context{{{Type::kBoolean, 0}}, {}, UINT32_MAX};
      TilingProgramIRValue out;
      for (int i = 0; i < 100; ++i) {
        if (Evaluate(*code, &context, Read, &out).status != Status::kOk || out.raw_bits != 7) {
          return false;
        }
      }
      return context.reads.size() == 100;
    }));
  }
  for (auto &task : tasks) {
    EXPECT_TRUE(task.get());
  }
}
TEST(TilingExpressionTest, ExplicitCastAndRuntimeInputValidation) {
  auto c = Conditional();
  c.input_types = {Type::kInt64};
  auto cast = Inst(Kind::kCast);
  cast.value_type = Type::kBoolean;
  c.instructions.insert(c.instructions.begin() + 1, cast);
  c.instructions[2].jump = 7;
  c.instructions[6].jump = 8;
  auto code = Compile(c);
  ASSERT_NE(code, nullptr);
  Context ctx{{{Type::kInt64, 0}}, {}, UINT32_MAX};
  TilingProgramIRValue out;
  EXPECT_EQ(Evaluate(*code, &ctx, Read, &out).status, Status::kOk);
  EXPECT_EQ(out.raw_bits, 7U);
  ctx.values[0].type = Type::kUint64;
  out = {Type::kInt64, 99};
  auto e = Evaluate(*code, &ctx, Read, &out);
  EXPECT_EQ(e.status, Status::kInputError);
  EXPECT_EQ(e.instruction, 0U);
  EXPECT_EQ(out.raw_bits, 99U);
  EXPECT_EQ(Evaluate(*code, nullptr, nullptr, &out).status, Status::kInputError);
  EXPECT_EQ(Evaluate(*code, &ctx, Read, nullptr).status, Status::kInvalidArgument);
  EXPECT_EQ(BuildTilingProgramIRExpression(c, nullptr).status, Status::kInvalidArgument);
}

TEST(TilingExpressionTest, ScratchQueryAndCapacityFailureHaveNoSideEffects) {
  auto code = Compile(Conditional());
  ASSERT_NE(code, nullptr);
  EXPECT_EQ(GetTilingExpressionScratchElements(*code), 2U);
  std::array<TilingProgramIRValue, 4> scratch;
  for (auto &v : scratch) {
    v = {Type::kUint64, 123};
  }
  Context ctx{{{Type::kBoolean, 0}}, {}, UINT32_MAX};
  TilingProgramIRValue output{Type::kInt64, 99};
  for (size_t capacity : {0U, 1U}) {
    auto error = EvaluateTilingProgramIRExpression(*code, &ctx, Read, scratch.data() + 1, capacity, &output);
    EXPECT_EQ(error.status, Status::kBufferTooSmall);
    EXPECT_EQ(error.instruction, UINT32_MAX);
    EXPECT_EQ(output.raw_bits, 99U);
    EXPECT_TRUE(ctx.reads.empty());
    for (const auto &v : scratch) {
      EXPECT_EQ(v.raw_bits, 123U);
      EXPECT_EQ(v.type, Type::kUint64);
    }
  }
  EXPECT_EQ(EvaluateTilingProgramIRExpression(*code, &ctx, Read, nullptr, 2, &output).status, Status::kInvalidArgument);
  EXPECT_EQ(EvaluateTilingProgramIRExpression(*code, &ctx, Read, nullptr, 0, &output).status, Status::kBufferTooSmall);
  EXPECT_TRUE(ctx.reads.empty());
}
TEST(TilingExpressionTest, OverlappingOutputIsRejectedBeforeScratchWrites) {
  auto code = Compile(Conditional());
  ASSERT_NE(code, nullptr);
  std::array<TilingProgramIRValue, 4> scratch;
  for (auto &v : scratch) {
    v = {Type::kInt64, 99};
  }
  Context ctx{{{Type::kBoolean, 0}}, {}, UINT32_MAX};
  for (size_t offset : {1U, 2U}) {
    EXPECT_EQ(
        EvaluateTilingProgramIRExpression(*code, &ctx, Read, scratch.data() + 1, 2, scratch.data() + offset).status,
        Status::kInvalidArgument);
    for (auto &v : scratch) {
      EXPECT_EQ(v.raw_bits, 99U);
    }
  }
  EXPECT_TRUE(ctx.reads.empty());
}
TEST(TilingExpressionTest, ScratchCanBeReusedAfterSuccessAndFailureWithoutLeakingValues) {
  auto code = Compile(Conditional());
  ASSERT_NE(code, nullptr);
  std::array<TilingProgramIRValue, 4> scratch;
  for (auto &v : scratch) {
    v = {Type::kUint64, 123};
  }
  Context ctx{{{Type::kBoolean, 0}}, {}, UINT32_MAX};
  TilingProgramIRValue output;
  for (bool fail : {false, true, false}) {
    ctx.values[0].raw_bits = fail ? 1 : 0;
    output = {Type::kInt64, 99};
    EXPECT_EQ(EvaluateTilingProgramIRExpression(*code, &ctx, Read, scratch.data() + 1, 2, &output).status,
              fail ? Status::kEvaluationError : Status::kOk);
    EXPECT_EQ(output.raw_bits, fail ? 99U : 7U);
    EXPECT_EQ(scratch.front().raw_bits, 123U);
    EXPECT_EQ(scratch.back().raw_bits, 123U);
  }
}
TEST(TilingExpressionTest, MaximumValidatedStackFitsExactCapacity) {
  TilingProgramIRExpressionCode c;
  c.constants = {{Type::kInt64, 1}};
  c.instructions.assign(kTilingExpressionMaxStack, Inst(Kind::kConstant));
  for (size_t i = 1; i < kTilingExpressionMaxStack; ++i) {
    c.instructions.push_back(Binary(TilingProgramIROpcode::kAdd));
  }
  auto code = Compile(c);
  ASSERT_NE(code, nullptr);
  EXPECT_EQ(GetTilingExpressionScratchElements(*code), kTilingExpressionMaxStack);
  std::array<TilingProgramIRValue, kTilingExpressionMaxStack + 2> scratch;
  scratch.front() = {Type::kInt64, 77};
  scratch.back() = {Type::kInt64, 88};
  TilingProgramIRValue output;
  ASSERT_EQ(
      EvaluateTilingProgramIRExpression(*code, nullptr, nullptr, scratch.data() + 1, kTilingExpressionMaxStack, &output)
          .status,
      Status::kOk);
  EXPECT_EQ(output.raw_bits, kTilingExpressionMaxStack);
  EXPECT_EQ(scratch.front().raw_bits, 77U);
  EXPECT_EQ(scratch.back().raw_bits, 88U);
}

TEST(TilingExpressionTest, UnaryInstructionUsesDeclaredArityAndResultType) {
  TilingProgramIRExpressionCode c;
  c.constants = {{Type::kFloat32, UINT64_C(0x3f800000)}};
  auto unary = Binary(TilingProgramIROpcode::kLog);
  unary.arg_count = 1;
  c.instructions = {Inst(Kind::kConstant), unary};
  c.result_type = Type::kFloat64;
  auto code = Compile(c);
  ASSERT_NE(code, nullptr);
  EXPECT_EQ(GetTilingExpressionScratchElements(*code), 1U);
  TilingProgramIRValue output;
  ASSERT_EQ(Evaluate(*code, nullptr, nullptr, &output).status, Status::kOk);
  EXPECT_EQ(output.type, Type::kFloat64);
  EXPECT_EQ(output.raw_bits, 0U);
  c.result_type = Type::kFloat32;
  Invalid(c);
  c.result_type = Type::kFloat64;
  c.instructions.insert(c.instructions.begin(), Inst(Kind::kConstant));
  c.instructions.back().arg_count = 2;
  Invalid(c);
}
TEST(TilingExpressionTest, UnaryFailureReportsInstructionAndPreservesCallerOutput) {
  TilingProgramIRExpressionCode c;
  c.constants = {{Type::kInt64, UINT64_C(0x8000000000000000)}};
  auto unary = Binary(TilingProgramIROpcode::kNeg);
  unary.arg_count = 1;
  c.instructions = {Inst(Kind::kConstant), unary};
  auto code = Compile(c);
  ASSERT_NE(code, nullptr);
  TilingProgramIRValue output{Type::kUint64, 99};
  auto error = Evaluate(*code, nullptr, nullptr, &output);
  EXPECT_EQ(error.status, Status::kEvaluationError);
  EXPECT_EQ(error.instruction, 1U);
  EXPECT_EQ(error.value_status, TilingProgramValueStatus::kArithmeticError);
  EXPECT_EQ(output.type, Type::kUint64);
  EXPECT_EQ(output.raw_bits, 99U);
}

TEST(TilingExpressionTest, NativeTernaryReadsBothValuesAndProducesDouble) {
  TilingProgramIRExpressionCode c;
  c.input_types = {Type::kBoolean, Type::kInt64, Type::kInt64};
  c.result_type = Type::kFloat64;
  auto ternary = Binary(TilingProgramIROpcode::kTernary);
  ternary.arg_count = 3;
  c.instructions = {Inst(Kind::kSymbol, 0), Inst(Kind::kSymbol, 1), Inst(Kind::kSymbol, 2), ternary};
  auto code = Compile(c);
  ASSERT_NE(code, nullptr);
  EXPECT_EQ(GetTilingExpressionScratchElements(*code), 3U);
  for (bool condition : {false, true}) {
    Context ctx{{{Type::kBoolean, condition ? 1U : 0U}, {Type::kInt64, 1}, {Type::kInt64, 2}}, {}, UINT32_MAX};
    TilingProgramIRValue output;
    ASSERT_EQ(Evaluate(*code, &ctx, Read, &output).status, Status::kOk);
    EXPECT_EQ(ctx.reads, (std::vector<uint32_t>{0, 1, 2}));
    EXPECT_EQ(output.type, Type::kFloat64);
    EXPECT_EQ(output.raw_bits, condition ? UINT64_C(0x3ff0000000000000) : UINT64_C(0x4000000000000000));
    ctx.reads.clear();
    ctx.fail = condition ? 2U : 1U;
    output = {Type::kInt64, 99};
    EXPECT_EQ(Evaluate(*code, &ctx, Read, &output).status, Status::kInputError);
    EXPECT_EQ(output.raw_bits, 99U);
  }
  c.input_types[0] = Type::kInt64;
  Invalid(c);
  c.input_types[0] = Type::kBoolean;
  c.input_types[2] = Type::kUint64;
  Invalid(c);
  c.input_types[2] = Type::kInt64;
  c.instructions.back().opcode = TilingProgramIROpcode::kAdd;
  Invalid(c);
}
TEST(TilingExpressionTest, EagerTernaryCannotHideUnselectedArithmeticFailure) {
  TilingProgramIRExpressionCode c;
  c.constants = {{Type::kInt64, 7}, {Type::kInt64, 1}, {Type::kInt64, 0}};
  c.result_type = Type::kFloat64;
  auto ternary = Binary(TilingProgramIROpcode::kTernary);
  ternary.arg_count = 3;
  c.instructions = {Inst(Kind::kPushBoolean, 1),
                    Inst(Kind::kConstant, 0),
                    Inst(Kind::kConstant, 1),
                    Inst(Kind::kConstant, 2),
                    Binary(TilingProgramIROpcode::kDiv),
                    ternary};
  auto code = Compile(c);
  ASSERT_NE(code, nullptr);
  TilingProgramIRValue output{Type::kInt64, 99};
  const auto error = Evaluate(*code, nullptr, nullptr, &output);
  EXPECT_EQ(error.status, Status::kEvaluationError);
  EXPECT_EQ(error.instruction, 4U);
  EXPECT_EQ(output.raw_bits, 99U);
  c.result_type = Type::kInt64;
  Invalid(c);
}
}  // namespace att
