/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "graph/symbolizer/symbolic.h"
#include "graph/symbolizer/symbol_checker.h"
#include "attribute_group/attr_group_shape_env.h"
#include "util/mem_utils.h"
#include "expression/testcase/source_stub.h"

namespace af {
namespace {
using namespace sym;
using ge::GraphInputShapeSourceStub;

class UtestExpression : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

// SymEngine 规范打印会重排负系数 Add 项：语义相等但字节不等的合法串（如来自
// 不同打印路径的 guard 序列化串）经重解析+Compare 接受（曾因字节不等被拒返回
// 空表达式，上层 guard 集合比较器解引用空指针断言崩溃）
TEST_F(UtestExpression, Deserialize_SemanticEqualReorderedStr) {
  auto s0 = Symbol("s0");
  auto s1 = Symbol("s1");
  const auto parsed = Expression::Deserialize("-(2 * s1) + s0");
  EXPECT_EQ(parsed.IsValid(), true);
  EXPECT_EQ(parsed, s0 - Mul(Symbol(2), s1));
}

// 打印侧将 Pow(E, x) 输出为 Exp(x)、Pow(x, 1/2) 输出为 Sqrt(x)：含这两种形态的
// 序列化串经 Deserialize 还原为等价的 Pow 表达式，序列化-反序列化往返闭环
TEST_F(UtestExpression, ExpSqrtSerializeAndDeserialize_RoundTrip) {
  auto s0 = Symbol("s0");
  auto s1 = Symbol("s1");

  const auto exp_expr = sym::Exp(s0);
  const std::string exp_str(exp_expr.Serialize().get());
  EXPECT_EQ(exp_str, "Exp(s0)");
  EXPECT_EQ(Expression::Deserialize(exp_str.c_str()), exp_expr);

  const auto sqrt_expr = sym::Pow(s1, sym::Div(Symbol(1), Symbol(2)));
  const std::string sqrt_str(sqrt_expr.Serialize().get());
  EXPECT_EQ(sqrt_str, "Sqrt(s1)");
  EXPECT_EQ(Expression::Deserialize(sqrt_str.c_str()), sqrt_expr);

  const auto guard = Eq(exp_expr, sqrt_expr);
  const std::string guard_str(guard.Serialize().get());
  EXPECT_EQ(guard_str, "ExpectEq(Exp(s0), Sqrt(s1))");
  EXPECT_EQ(Expression::Deserialize(guard_str.c_str()), guard);
}

// float 常量(如 1e-8)经 RealDouble 打印为科学计数法(9.99999993922529e-09)：解析侧需识别
// e 指数，保证序列化-反序列化往返闭环（Range/Cast 等算子的 value 仍需支持 float）
TEST_F(UtestExpression, ScientificNotationSerializeAndDeserialize_RoundTrip) {
  auto s52 = Symbol("s52");
  const auto expr = Add(Symbol(9.99999993922529e-09), s52);
  const std::string expr_str(expr.Serialize().get());
  EXPECT_EQ(expr_str, "(9.99999993922529e-09 + s52)");
  EXPECT_EQ(Expression::Deserialize(expr_str.c_str()), expr);

  // 无小数点/正指数/单位数指数的科学计数法均需被正确解析为浮点
  EXPECT_EQ(Expression::Parse("2e-3"), Symbol(2e-3));
  EXPECT_EQ(Expression::Parse("2E3"), Symbol(2e3));
  EXPECT_EQ(Expression::Parse("1e+20"), Symbol(1e20));
  EXPECT_EQ(Expression::Parse("9.99999993922529e-9"), Symbol(9.99999993922529e-9));
}

class AttributeGroupShapeEnvUt : public testing::Test {};

// 间接环：已有 s3 == Min(2, s2) 的替换（变量 s3 的替换根为复合表达式 Min(2, s2)）时，
// 再建立 s2 == (s0+s3)*Ceil(s1) 会闭合出 s2 -> 复合 -> s3 -> Min(2, s2) 的替换链环；
// 检测命中后跳过该条替换（s2 不进入替换集合），既有替换保持不变
TEST_F(AttributeGroupShapeEnvUt, CheckReplacementCycleIndirectTest) {
  ShapeEnvAttr shape_env;
  SetCurShapeEnvContext(&shape_env);
  Symbol s0 = shape_env.CreateSymbol(1, MakeShared<GraphInputShapeSourceStub>(0, 0));
  Symbol s1 = shape_env.CreateSymbol(1, MakeShared<GraphInputShapeSourceStub>(0, 1));
  Symbol s2 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 2));
  Symbol s3 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 3));

  EXPECT_EQ(EXPECT_SYMBOL_EQ(s3, sym::Min(Symbol(2), s2)), true);
  EXPECT_NE(shape_env.replacements_.find(s3), shape_env.replacements_.end());

  EXPECT_EQ(EXPECT_SYMBOL_EQ((s0 + s3) * sym::Ceiling(s1), s2), true);
  EXPECT_EQ(shape_env.replacements_.find(s2), shape_env.replacements_.end());

  SetCurShapeEnvContext(nullptr);
}
}  // namespace
}  // namespace af
