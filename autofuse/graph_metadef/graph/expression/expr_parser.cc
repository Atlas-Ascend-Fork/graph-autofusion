/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "expr_parser.h"
#include <unordered_map>
#include "common/checker.h"
#include "symengine/real_double.h"
namespace af {
ExpressionImplPtr ExprParser::ParserExpression() {
  auto ret = ParserAddSubtract();
  GE_ASSERT_NOTNULL(ret);
  return ret;
}

graphStatus ExprParser::Init() {
  GE_ASSERT_SUCCESS(scanner_.GetNextToken(currentToken_));
  return ge::GRAPH_SUCCESS;
}
graphStatus ExprParser::Eat(TokenType type) {
  GE_ASSERT(currentToken_.type == type);
  GE_ASSERT_SUCCESS(scanner_.GetNextToken(currentToken_));
  return ge::GRAPH_SUCCESS;
}

ExpressionImplPtr ExprParser::ParserFactor() {
  // 单参函数（Log/Ceil/Floor/Abs/Exp/Sqrt）均为 Func(expr) 结构，走分发表统一派发，
  // 新增单参函数只需在表中注册，避免本函数分支随函数 token 增长
  static const std::unordered_map<TokenType, ExpressionImplPtr (ExprParser::*)()> kUnaryFuncParsers = {
      {TokenType::kLog, &ExprParser::ParserLogFunction},     {TokenType::kCeil, &ExprParser::ParserCeilFunction},
      {TokenType::kFloor, &ExprParser::ParserFloorFunction}, {TokenType::kAbs, &ExprParser::ParserAbsFunction},
      {TokenType::kExp, &ExprParser::ParserExpFunction},     {TokenType::kSqrt, &ExprParser::ParserSqrtFunction}};
  switch (currentToken_.type) {
    case TokenType::kIdentifier:
      return ParserIdentifier();
    case TokenType::kLparen:
      return ParserLParen();
    case TokenType::kMax:
      return ParserMaxFunction();
    case TokenType::kMin:
      return ParserMinFunction();
    case TokenType::kPow:
      return ParserPowFunction();
    case TokenType::kMod:
      return ParserModFunction();
    case TokenType::kRational:
      return ParserRationalFunction();
    case TokenType::kNumber:
      return ParserNumber();
    case TokenType::kMinus:
      return ParserMinus();
    case TokenType::kEq:
      return ParserEqual();
    case TokenType::kNe:
      return ParserUnequal();
    case TokenType::kLe:
      return ParserLessEqual();
    case TokenType::kLt:
      return ParserLessThan();
    case TokenType::kTrue:
    case TokenType::kFalse:
      return ParseConstBoolen();
    case TokenType::kLogicalAnd:
      return ParserLogicalAnd();
    case TokenType::kLogicalOr:
      return ParserLogicalOr();
    default:
      break;
  }
  const auto it = kUnaryFuncParsers.find(currentToken_.type);
  if (it != kUnaryFuncParsers.end()) {
    return (this->*(it->second))();
  }
  GELOGE(ge::PARAM_INVALID, "Unsupported operator %d when Parser factor.", currentToken_.type);
  return nullptr;
}

ExpressionImplPtr ExprParser::ParserAddSubtract() {
  auto node = ParserMulDivide();
  GE_ASSERT_NOTNULL(node);
  while (currentToken_.type == TokenType::kPlus || currentToken_.type == TokenType::kMinus) {
    TokenType op = currentToken_.type;
    GE_ASSERT_SUCCESS(Eat(op));
    auto right = ParserMulDivide();
    GE_ASSERT_NOTNULL(right);
    switch (op) {
      case TokenType::kPlus:
        node = Add(node, right);
        break;
      case TokenType::kMinus:
        node = Sub(node, right);
        break;
      default:
        GELOGE(ge::PARAM_INVALID, "unsupported operator %d when parsing add and sub.", currentToken_.type);
        return nullptr;
    }
  }
  return node;
}

ExpressionImplPtr ExprParser::ParserMulDivide() {
  auto node = ParserFactor();
  GE_ASSERT_NOTNULL(node);

  while (currentToken_.type == TokenType::kMultiply || currentToken_.type == TokenType::kDivide) {
    TokenType op = currentToken_.type;
    GE_ASSERT_SUCCESS(Eat(op));
    auto right = ParserFactor();
    GE_ASSERT_NOTNULL(right);
    switch (op) {
      case TokenType::kMultiply:
        node = Mul(node, right);
        GE_ASSERT_NOTNULL(node);
        break;
      case TokenType::kDivide:
        node = Div(node, right);
        GE_ASSERT_NOTNULL(node);
        break;
      default:
        GELOGE(ge::PARAM_INVALID, "unsupported operator %d when parsing mul and divide.", currentToken_.type);
        return nullptr;
    }
  }
  return node;
}

ExpressionImplPtr ExprParser::ParserMaxFunction() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kMax));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto node = ParserExpression();
  while (currentToken_.type == TokenType::kComma) {
    GE_ASSERT_SUCCESS(Eat(TokenType::kComma));
    node = Max(node, ParserExpression());
  }
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return node;
}

ExpressionImplPtr ExprParser::ParserMinFunction() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kMin));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto node = ParserExpression();
  while (currentToken_.type == TokenType::kComma) {
    GE_ASSERT_SUCCESS(Eat(TokenType::kComma));
    node = Min(node, ParserExpression());
  }
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return node;
}

ExpressionImplPtr ExprParser::ParserEqual() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kEq));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto arg1 = ParserExpression();
  GE_ASSERT_SUCCESS(Eat(TokenType::kComma));
  auto arg2 = ParserExpression();
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return Eq(arg1, arg2);
}

ExpressionImplPtr ExprParser::ParserUnequal() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kNe));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto arg1 = ParserExpression();
  GE_ASSERT_SUCCESS(Eat(TokenType::kComma));
  auto arg2 = ParserExpression();
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return Ne(arg1, arg2);
}

ExpressionImplPtr ExprParser::ParserLessEqual() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kLe));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto arg1 = ParserExpression();
  GE_ASSERT_SUCCESS(Eat(TokenType::kComma));
  auto arg2 = ParserExpression();
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return Le(arg1, arg2);
}

ExpressionImplPtr ExprParser::ParserLessThan() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kLt));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto arg1 = ParserExpression();
  GE_ASSERT_SUCCESS(Eat(TokenType::kComma));
  auto arg2 = ParserExpression();
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return Lt(arg1, arg2);
}

ExpressionImplPtr ExprParser::ParserLogicalAnd() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kLogicalAnd));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  std::vector<ExpressionImplPtr> args;
  args.push_back(ParserExpression());
  while (currentToken_.type == TokenType::kComma) {
    GE_ASSERT_SUCCESS(Eat(TokenType::kComma));
    args.push_back(ParserExpression());
  }
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return LogicalAnd(args);
}

ExpressionImplPtr ExprParser::ParserLogicalOr() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kLogicalOr));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  std::vector<ExpressionImplPtr> args;
  args.push_back(ParserExpression());
  while (currentToken_.type == TokenType::kComma) {
    GE_ASSERT_SUCCESS(Eat(TokenType::kComma));
    args.push_back(ParserExpression());
  }
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return LogicalOr(args);
}

ExpressionImplPtr ExprParser::ParserPowFunction() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kPow));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto arg1 = ParserExpression();
  GE_ASSERT_SUCCESS(Eat(TokenType::kComma));
  auto arg2 = ParserExpression();
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return Pow(arg1, arg2);
}

ExpressionImplPtr ExprParser::ParserModFunction() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kMod));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto arg1 = ParserAddSubtract();
  GE_ASSERT_SUCCESS(Eat(TokenType::kComma));
  auto arg2 = ParserAddSubtract();
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return Mod(arg1, arg2);
}

ExpressionImplPtr ExprParser::ParserLogFunction() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kLog));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto arg1 = ParserExpression();
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return Log(arg1);
}

ExpressionImplPtr ExprParser::ParserCeilFunction() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kCeil));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto arg1 = ParserExpression();
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return Ceiling(arg1);
}

ExpressionImplPtr ExprParser::ParserFloorFunction() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kFloor));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto arg1 = ParserExpression();
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return Floor(arg1);
}

ExpressionImplPtr ExprParser::ParserAbsFunction() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kAbs));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto arg1 = ParserExpression();
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return Abs(arg1);
}

ExpressionImplPtr ExprParser::ParserRationalFunction() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kRational));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto arg1 = ParserExpression();
  GE_ASSERT_SUCCESS(Eat(TokenType::kComma));
  auto arg2 = ParserExpression();
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return Rational(arg1, arg2);
}

// Exp(x)/Sqrt(x) 为打印侧 DefaultPowPrinter 对 Pow(E, x)/Pow(x, 1/2) 的输出形态，
// 解析时还原为等价的 Pow 表达式，保证序列化-反序列化往返闭环
ExpressionImplPtr ExprParser::ParserExpFunction() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kExp));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto arg = ParserExpression();
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return Exp(arg);
}

ExpressionImplPtr ExprParser::ParserSqrtFunction() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kSqrt));
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto arg = ParserExpression();
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return Pow(arg, Rational(ExpressionImpl::CreateExpressionImpl(static_cast<int64_t>(1)),
                           ExpressionImpl::CreateExpressionImpl(static_cast<int64_t>(2))));
}

ExpressionImplPtr ExprParser::ParserNumber() {
  const std::string &numberStr = currentToken_.value;
  try {
    if (numberStr.find('.') != std::string::npos) {
      double value = std::stod(numberStr);
      GE_ASSERT_SUCCESS(Eat(TokenType::kNumber));
      return ExpressionImpl::CreateExpressionImpl(value);  // 返回浮点数节点
    } else {
      int64_t value = std::stoll(numberStr);
      GE_ASSERT_SUCCESS(Eat(TokenType::kNumber));
      return ExpressionImpl::CreateExpressionImpl(value);  // 返回整数节点
    }
  } catch (std::invalid_argument &) {
    GELOGW("number str:%s is invalid", numberStr.c_str());
    return nullptr;
  } catch (std::out_of_range &) {
    GELOGW("number str:%s is out_of_range", numberStr.c_str());
    return nullptr;
  }
}

ExpressionImplPtr ExprParser::ParserIdentifier() {
  const std::string name{currentToken_.value};
  GE_ASSERT_SUCCESS(Eat(TokenType::kIdentifier));
  return ExpressionImpl::CreateExpressionImpl(name);
}

ExpressionImplPtr ExprParser::ParseConstBoolen() {
  bool sym_value = currentToken_.value == "True" ? true : false;
  GE_ASSERT_SUCCESS(Eat(currentToken_.type));
  return ExpressionImpl::CreateExpressionImpl(sym_value);
}

ExpressionImplPtr ExprParser::ParserMinus() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kMinus));
  ExpressionImplPtr node;
  if (currentToken_.type == TokenType::kLparen) {
    node = ParserLParen();
  } else {
    node = ParserFactor();
  }
  GE_ASSERT_NOTNULL(node);
  return Neg(node);
}

ExpressionImplPtr ExprParser::ParserLParen() {
  GE_ASSERT_SUCCESS(Eat(TokenType::kLparen));
  auto node = ParserExpression();
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_SUCCESS(Eat(TokenType::kRparen));
  return node;
}
}  // namespace af
