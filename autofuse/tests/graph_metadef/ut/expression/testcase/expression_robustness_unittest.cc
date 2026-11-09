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
#include "graph/symbolizer/symbolic.h"
#include "graph/symbolizer/symbol_checker.h"
#include "graph/symbolizer/symbolic_utils.h"
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

// 生产-消费链路：符号化推导产生 Rational 系数 guard（如 ConcatV2D 对上游半维输入执行
// ASSERT_SYMBOL_EQ((1/2)*s2, 680)，实际场景 s2=1360），下游经 StaticCheckEq 消费。
// guard 规范化需将有理系数通分：680 == (1/2)*s2 应归一为 1360 == s2，
// 使消费侧等价形态（s2 == 1360、整数倍缩放 256*s2 == 348160）均可命中
TEST_F(UtestExpression, RationalGuardProduceAndConsume) {
  ShapeEnvAttr producer;
  SetCurShapeEnvContext(&producer);
  Symbol s2 = producer.CreateSymbol(1360, MakeShared<GraphInputShapeSourceStub>(0, 0));
  EXPECT_EQ(EXPECT_SYMBOL_EQ(sym::Rational(1, 2) * s2, Symbol(680)), true);

  const auto producer_infos = producer.GetAllSymbolCheckInfos();
  ASSERT_EQ(producer_infos.size(), 1UL);
  EXPECT_EQ(std::string(producer_infos[0].expr.Serialize().get()), "ExpectEq(1360, s0)");

  // 消费侧（跨 env，仅注入 guard，对齐跨进程仅携带 guard 集合的消费形态）
  ShapeEnvAttr consumer;
  SetCurShapeEnvContext(&consumer);
  const auto guard_expr = Expression::Deserialize(producer_infos[0].expr.Serialize().get());
  ASSERT_EQ(guard_expr.IsValid(), true);
  EXPECT_EQ(consumer.AppendSymbolCheckInfo(guard_expr, __FILE__, __LINE__), GRAPH_SUCCESS);

  EXPECT_EQ(SymbolicUtils::StaticCheckEq(s2, Symbol(1360)), TriBool::kTrue);
  EXPECT_EQ(SymbolicUtils::StaticCheckEq(s2 * Symbol(256), Symbol(348160)), TriBool::kTrue);
  SetCurShapeEnvContext(nullptr);
}

// 生产侧同 env 消费：guard 与替换（replacement）建立后，静态查询直接可判定
TEST_F(UtestExpression, RationalGuardSameEnvConsume) {
  ShapeEnvAttr producer;
  SetCurShapeEnvContext(&producer);
  Symbol s2 = producer.CreateSymbol(1360, MakeShared<GraphInputShapeSourceStub>(0, 0));
  EXPECT_EQ(EXPECT_SYMBOL_EQ(sym::Rational(1, 2) * s2, Symbol(680)), true);

  EXPECT_EQ(SymbolicUtils::StaticCheckEq(s2, Symbol(1360)), TriBool::kTrue);
  EXPECT_EQ(SymbolicUtils::StaticCheckEq(s2 * Symbol(256), Symbol(348160)), TriBool::kTrue);
  SetCurShapeEnvContext(nullptr);
}

// 已有整数系数 GCD 归一化行为回归：2*s0 == 200 与 s0 == 100 互相命中
TEST_F(UtestExpression, IntegerGuardRegression) {
  ShapeEnvAttr producer;
  SetCurShapeEnvContext(&producer);
  Symbol s0 = producer.CreateSymbol(100, MakeShared<GraphInputShapeSourceStub>(0, 0));
  EXPECT_EQ(EXPECT_SYMBOL_EQ(s0 * Symbol(2), Symbol(200)), true);

  const auto producer_infos = producer.GetAllSymbolCheckInfos();
  ASSERT_EQ(producer_infos.size(), 1UL);
  EXPECT_EQ(std::string(producer_infos[0].expr.Serialize().get()), "ExpectEq(100, s0)");

  ShapeEnvAttr consumer;
  SetCurShapeEnvContext(&consumer);
  const auto guard_expr = Expression::Deserialize(producer_infos[0].expr.Serialize().get());
  ASSERT_EQ(guard_expr.IsValid(), true);
  EXPECT_EQ(consumer.AppendSymbolCheckInfo(guard_expr, __FILE__, __LINE__), GRAPH_SUCCESS);
  EXPECT_EQ(SymbolicUtils::StaticCheckEq(s0, Symbol(100)), TriBool::kTrue);
  SetCurShapeEnvContext(nullptr);
}

// 通分仅对正整数分母生效：符号分母（如 2/s0）经记录路径规范化后保持既有形态
TEST_F(UtestExpression, SymbolDenominatorNotNormalized) {
  ShapeEnvAttr producer;
  SetCurShapeEnvContext(&producer);
  Symbol s0 = producer.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  EXPECT_EQ(EXPECT_SYMBOL_EQ(sym::Div(Symbol(2), s0), Symbol(1)), true);

  const auto producer_infos = producer.GetAllSymbolCheckInfos();
  ASSERT_EQ(producer_infos.size(), 1UL);
  EXPECT_EQ(std::string(producer_infos[0].expr.Serialize().get()), "ExpectEq((2 / (s0)), 1)");
  SetCurShapeEnvContext(nullptr);
}

// 比较方向保持：分母为正时通分不翻转 Lt/Le 语义（翻转为 ExpectLt(200, s0) 则断言失败）
TEST_F(UtestExpression, RationalGuardLtLeDirection) {
  ShapeEnvAttr producer;
  SetCurShapeEnvContext(&producer);
  Symbol s0 = producer.CreateSymbol(100, MakeShared<GraphInputShapeSourceStub>(0, 0));
  EXPECT_EQ(EXPECT_SYMBOL_LT(sym::Rational(1, 2) * s0, Symbol(100)), true);

  const auto producer_infos = producer.GetAllSymbolCheckInfos();
  ASSERT_EQ(producer_infos.size(), 1UL);
  EXPECT_EQ(std::string(producer_infos[0].expr.Serialize().get()), "ExpectLt(s0, 200)");
  SetCurShapeEnvContext(nullptr);
}

TEST_F(UtestExpression, RationalGuardLeDirection) {
  ShapeEnvAttr producer;
  SetCurShapeEnvContext(&producer);
  Symbol s0 = producer.CreateSymbol(200, MakeShared<GraphInputShapeSourceStub>(0, 0));
  EXPECT_EQ(EXPECT_SYMBOL_LE(sym::Rational(1, 2) * s0, Symbol(100)), true);

  const auto producer_infos = producer.GetAllSymbolCheckInfos();
  ASSERT_EQ(producer_infos.size(), 1UL);
  EXPECT_EQ(std::string(producer_infos[0].expr.Serialize().get()), "ExpectLe(s0, 200)");
  SetCurShapeEnvContext(nullptr);
}
}  // namespace
}  // namespace af
