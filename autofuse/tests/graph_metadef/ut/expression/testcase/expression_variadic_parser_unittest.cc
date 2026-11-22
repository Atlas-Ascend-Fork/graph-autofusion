/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "graph/symbolizer/symbolic.h"

namespace af {
namespace {
using namespace sym;

class UtestExpressionVariadicParser : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

// 形状 guard 会把多个条件合成为变参 LogicAnd/LogicOr（symbol_checker.h 已声明该形态），打印端
// DefaultLogicalAndPrinter/DefaultLogicalOrPrinter/PrintArgs 也按变参输出，解析端须按 N 参收参，
// 否则解析三参表达式在 Eat(kRparen) 处断言失败（与 ge 仓 graph_metadef 副本同步）
TEST_F(UtestExpressionVariadicParser, ParseNaryLogicalAndOr) {
  const char *nary_and = "LogicAnd(ExpectNe(-1, s101), ExpectNe(-1, s401), ExpectNe(0, s401))";
  const auto nary_and_expr = Expression::Parse(nary_and);
  EXPECT_NE(nary_and_expr.impl_, nullptr);
  EXPECT_EQ(std::string(nary_and_expr.Serialize().get()), nary_and);

  const char *nary_or = "LogicOr(ExpectEq(s6, s7), ExpectLt(s0, s1), ExpectLt(s3, s2), ExpectNe(s4, s5))";
  const auto nary_or_expr = Expression::Parse(nary_or);
  EXPECT_NE(nary_or_expr.impl_, nullptr);
  EXPECT_EQ(std::string(nary_or_expr.Serialize().get()), nary_or);
}

// Max/Min 打印端同为变参（DefaultMaxPrinter/DefaultMinPrinter），仅有二元构造，解析端按从左折叠
TEST_F(UtestExpressionVariadicParser, ParseNaryMaxMin) {
  const auto nary_max = Expression::Parse("Max(s0, s1, s2)");
  EXPECT_NE(nary_max.impl_, nullptr);
  EXPECT_EQ(nary_max, Max(Max(Symbol("s0"), Symbol("s1")), Symbol("s2")));

  const auto nary_min = Expression::Parse("Min(s0, s1, s2)");
  EXPECT_NE(nary_min.impl_, nullptr);
  EXPECT_EQ(nary_min, Min(Min(Symbol("s0"), Symbol("s1")), Symbol("s2")));
}

// 变参语义下单参 LogicAnd/LogicOr 合法，解析结果与单元素构造版折叠一致
TEST_F(UtestExpressionVariadicParser, ParseSingleArgLogicalAndOr) {
  const auto single_and = Expression::Parse("LogicAnd(ExpectEq(s0, s1))");
  EXPECT_NE(single_and.impl_, nullptr);
  EXPECT_EQ(std::string(single_and.Serialize().get()), "ExpectEq(s0, s1)");

  const auto single_or = Expression::Parse("LogicOr(ExpectEq(s0, s1))");
  EXPECT_NE(single_or.impl_, nullptr);
  EXPECT_EQ(std::string(single_or.Serialize().get()), "ExpectEq(s0, s1)");
}
}  // namespace
}  // namespace af
