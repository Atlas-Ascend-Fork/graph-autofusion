/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dialect/AFIR/Support/SymbolicExpression.h"
#include "dialect/AFIR/Support/SymbolicExpressionDetail.h"

#include <array>
#include <string>
#include <string_view>

#include "gtest/gtest.h"

namespace mlir::afir {
namespace {

struct CanonicalCase {
  std::string_view name;
  std::string_view expression;
};

constexpr std::array<CanonicalCase, 28> kLegacyCanonicalCorpus = {{
    {"integer", "1"},
    {"double", "1.5"},
    {"symbol", "s0"},
    {"add", "(1 + s0)"},
    {"subtract", "(s0 - s1)"},
    {"multiply", "(s0 * s1)"},
    {"divide", "(s0 / (s1))"},
    {"max", "Max(s0, s1)"},
    {"min", "Min(s0, s1)"},
    {"negative_power", "Pow(s0, -1)"},
    {"mod", "Mod(s0, s1)"},
    {"log", "Log(s0)"},
    {"ceiling", "Ceiling(s0)"},
    {"floor", "Floor(s0)"},
    {"abs", "Abs(s0)"},
    {"rational", "Rational(1 , 2)"},
    {"equal", "ExpectEq(s0, s1)"},
    {"not_equal", "ExpectNe(s0, s1)"},
    {"less_equal", "ExpectLe(s0, s1)"},
    {"less_than", "ExpectLt(s0, s1)"},
    {"logic_and", "LogicAnd(ExpectLt(s0, s1), ExpectLt(s1, s2))"},
    {"logic_or", "LogicOr(ExpectEq(s0, s1), ExpectLt(s1, s2))"},
    {"boolean_true", "True"},
    {"boolean_false", "False"},
    {"dynamic_shape", "(8 * Ceiling((Rational(1 , 8) * s1)))"},
    {"workspace", "Max(8192, Min(65536, (64 * s0)))"},
    {"tiling", "Ceiling((s0 / (tile_size)))"},
    {"guard", "LogicAnd(ExpectLe(1, s0), ExpectLt(s0, 65536))"},
}};

TEST(ExpressionTest, ParsesAndSerializesLegacyCanonicalCorpus) {
  for (const CanonicalCase &testCase : kLegacyCanonicalCorpus) {
    SCOPED_TRACE(testCase.name);
    Result<Expression> parsed = Expression::Parse(testCase.expression);
    ASSERT_TRUE(parsed) << parsed.error().message;
    EXPECT_EQ(parsed.value().Serialize(), testCase.expression);
  }
}

TEST(ExpressionTest, SerializationIsDeterministicAcrossRepeatedParses) {
  constexpr std::string_view expression = "Max((s0 + s1), (s0 * s2))";
  std::string expected;
  for (int iteration = 0; iteration < 20; ++iteration) {
    Result<Expression> parsed = Expression::Parse(expression);
    ASSERT_TRUE(parsed) << parsed.error().message;
    if (iteration == 0) {
      expected = parsed.value().Serialize();
    }
    EXPECT_EQ(parsed.value().Serialize(), expected);
  }
}

TEST(ExpressionTest, SimplifyProducesAParseableCanonicalExpression) {
  Result<Expression> parsed = Expression::Parse("((1 + s1) * s0)");
  ASSERT_TRUE(parsed) << parsed.error().message;

  Result<Expression> simplified = parsed.value().Simplify();
  ASSERT_TRUE(simplified) << simplified.error().message;
  const std::string serialized = simplified.value().Serialize();
  Result<Expression> reparsed = Expression::Parse(serialized);
  ASSERT_TRUE(reparsed) << reparsed.error().message;
  EXPECT_EQ(reparsed.value().Serialize(), serialized);
}

TEST(ExpressionTest, RejectsInvalidAndNonCanonicalInput) {
  constexpr std::array<std::string_view, 15> invalidExpressions = {
      "",   "s0 + 1", "(s0 + )", "Max()", "Max(s0)", "Max(s0,s1)", "Unknown(s0)", "Rational(s0, 2)", "LogicAnd(s0, s1)",
      "1.", ".1",     "s0 ? s1", "(s0",   "s0)",     "Pow(s0, 2)",
  };
  for (std::string_view expression : invalidExpressions) {
    SCOPED_TRACE(expression);
    EXPECT_FALSE(Expression::Parse(expression));
  }
}

TEST(ExpressionTest, ReservedFunctionsRequireArgumentLists) {
  constexpr std::array<std::string_view, 15> reservedFunctions = {
      "Max",      "Min",      "Pow",      "Mod",      "Log",      "Ceiling",  "Floor",   "Abs",
      "Rational", "ExpectEq", "ExpectNe", "ExpectLe", "ExpectLt", "LogicAnd", "LogicOr",
  };
  for (std::string_view function : reservedFunctions) {
    SCOPED_TRACE(function);
    Result<Expression> parsed = Expression::Parse(function);
    ASSERT_FALSE(parsed);
    EXPECT_EQ(parsed.error().code, ErrorCode::InvalidArgument);
    EXPECT_NE(parsed.error().message.find("requires an argument list"), std::string::npos);
  }
}

TEST(ExpressionTest, EnforcesSerializedByteLimit) {
  Result<Expression> parsed = Expression::Parse(std::string(64U * 1024U + 1U, 's'));
  ASSERT_FALSE(parsed);
  EXPECT_EQ(parsed.error().code, ErrorCode::ResourceExhausted);
}

TEST(ExpressionTest, EnforcesAstDepthLimit) {
  std::string expression = "s0";
  for (int depth = 0; depth < 130; ++depth) {
    expression = "(" + expression + ")";
  }
  Result<Expression> parsed = Expression::Parse(expression);
  ASSERT_FALSE(parsed);
  EXPECT_EQ(parsed.error().code, ErrorCode::ResourceExhausted);
}

TEST(ExpressionTest, EnforcesNodeCountLimit) {
  std::string expression = "s0";
  for (int index = 0; index < 4100; ++index) {
    expression += " + s0";
  }
  Result<Expression> parsed = Expression::Parse(expression);
  ASSERT_FALSE(parsed);
  EXPECT_EQ(parsed.error().code, ErrorCode::ResourceExhausted);
}

TEST(ExpressionTest, EnforcesFreeSymbolCountLimit) {
  std::string expression = "s0";
  for (int index = 1; index <= 256; ++index) {
    expression += " + s" + std::to_string(index);
  }
  Result<Expression> parsed = Expression::Parse(expression);
  ASSERT_FALSE(parsed);
  EXPECT_EQ(parsed.error().code, ErrorCode::ResourceExhausted);
}

TEST(ExpressionTest, ReportsNumericOverflowAsInvalidArgument) {
  Result<Expression> parsed = Expression::Parse("9223372036854775808");
  ASSERT_FALSE(parsed);
  EXPECT_EQ(parsed.error().code, ErrorCode::InvalidArgument);
  EXPECT_NE(parsed.error().message.find("out of range"), std::string::npos);
}

TEST(ExpressionTest, ConvertsSymEngineFailuresToErrors) {
  Result<Expression> parsed = Expression::Parse("Rational(1, 0)");
  ASSERT_FALSE(parsed);
  EXPECT_EQ(parsed.error().code, ErrorCode::InvalidArgument);
}

TEST(ExpressionTest, InternalTraversalReturnsStableFreeSymbolsAndTree) {
  Result<Expression> parsed = Expression::Parse("Max((s0 + s2), s1)");
  ASSERT_TRUE(parsed) << parsed.error().message;

  Result<std::vector<std::string>> symbols = detail::ExpressionAccess::GetFreeSymbols(parsed.value());
  ASSERT_TRUE(symbols) << symbols.error().message;
  EXPECT_EQ(symbols.value(), (std::vector<std::string>{"s0", "s1", "s2"}));

  Result<detail::ExpressionNode> tree = detail::ExpressionAccess::GetTree(parsed.value());
  ASSERT_TRUE(tree) << tree.error().message;
  EXPECT_EQ(tree.value().kind, detail::NodeKind::Max);
  EXPECT_EQ(tree.value().arguments.size(), 2U);
}

TEST(ExpressionTest, InternalNormalizationUsesStablePlaceholders) {
  Result<Expression> parsed = Expression::Parse("Max((s0 + s2), s1)");
  ASSERT_TRUE(parsed) << parsed.error().message;

  Result<detail::NormalizedExpression> normalized = detail::ExpressionAccess::NormalizeFreeSymbols(parsed.value());
  ASSERT_TRUE(normalized) << normalized.error().message;
  EXPECT_EQ(normalized.value().symbols, (std::vector<std::string>{"s0", "s1", "s2"}));
  EXPECT_EQ(normalized.value().expression.Serialize(), "Max((__arg0 + __arg2), __arg1)");
}

TEST(ExpressionTest, InternalNormalizationRejectsPlaceholderCollision) {
  Result<Expression> parsed = Expression::Parse("(__arg0 + s0)");
  ASSERT_TRUE(parsed) << parsed.error().message;
  Result<detail::NormalizedExpression> normalized = detail::ExpressionAccess::NormalizeFreeSymbols(parsed.value());
  ASSERT_FALSE(normalized);
  EXPECT_EQ(normalized.error().code, ErrorCode::InvalidArgument);
}

TEST(ExpressionTest, InternalSubstitutionAndEvaluationUseExplicitBindings) {
  Result<Expression> parsed = Expression::Parse("((2 * s1) + s0)");
  ASSERT_TRUE(parsed) << parsed.error().message;
  Result<Expression> substituted = detail::ExpressionAccess::Substitute(parsed.value(), {{"s0", "1"}, {"s1", "2"}});
  ASSERT_TRUE(substituted) << substituted.error().message;
  EXPECT_EQ(substituted.value().Serialize(), "5");

  Result<double> evaluated = detail::ExpressionAccess::Evaluate(parsed.value(), {{"s0", 1.0}, {"s1", 2.0}});
  ASSERT_TRUE(evaluated) << evaluated.error().message;
  EXPECT_DOUBLE_EQ(evaluated.value(), 5.0);
  EXPECT_FALSE(detail::ExpressionAccess::Evaluate(parsed.value(), {{"s0", 1.0}}));
}

TEST(ExpressionTest, InternalProofIsConservative) {
  Result<Expression> truePredicate = Expression::Parse("True");
  Result<Expression> falsePredicate = Expression::Parse("False");
  Result<Expression> unknownPredicate = Expression::Parse("ExpectLt(s0, s1)");
  ASSERT_TRUE(truePredicate && falsePredicate && unknownPredicate);
  EXPECT_EQ(detail::ExpressionAccess::Prove(truePredicate.value()).value(), detail::TruthValue::True);
  EXPECT_EQ(detail::ExpressionAccess::Prove(falsePredicate.value()).value(), detail::TruthValue::False);
  EXPECT_EQ(detail::ExpressionAccess::Prove(unknownPredicate.value()).value(), detail::TruthValue::Unknown);
}

}  // namespace
}  // namespace mlir::afir
