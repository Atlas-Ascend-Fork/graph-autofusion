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

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "symengine/add.h"
#include "symengine/basic.h"
#include "symengine/constants.h"
#include "symengine/eval_double.h"
#include "symengine/functions.h"
#include "symengine/integer.h"
#include "symengine/logic.h"
#include "symengine/mul.h"
#include "symengine/pow.h"
#include "symengine/rational.h"
#include "symengine/real_double.h"
#include "symengine/simplify.h"
#include "symengine/symengine_casts.h"
#include "symengine/visitor.h"

namespace mlir::afir {
namespace {

using BasicPtr = SymEngine::RCP<const SymEngine::Basic>;

constexpr size_t kMaxSerializedBytes = 64U * 1024U;
constexpr size_t kMaxAstDepth = 128U;
constexpr size_t kMaxNodeCount = 4096U;
constexpr size_t kMaxFreeSymbolCount = 256U;
constexpr uint32_t kMaxExpandedPower = 256U;

class ParseFailure : public std::runtime_error {
 public:
  ParseFailure(ErrorCode code, std::string message) : std::runtime_error(std::move(message)), code_(code) {}

  ErrorCode code() const {
    return code_;
  }

 private:
  ErrorCode code_;
};

enum class TokenKind {
  Identifier,
  Number,
  Plus,
  Minus,
  Multiply,
  Divide,
  Comma,
  LParen,
  RParen,
  End,
};

struct Token {
  TokenKind kind;
  std::string text;
  size_t offset;
};

enum class FunctionKind {
  Max,
  Min,
  Pow,
  Mod,
  Log,
  Ceiling,
  Floor,
  Abs,
  Rational,
  ExpectEq,
  ExpectNe,
  ExpectLe,
  ExpectLt,
  LogicAnd,
  LogicOr,
};

struct FunctionSpec {
  std::string_view name;
  FunctionKind kind;
  size_t arity;
};

constexpr std::array<FunctionSpec, 15> kFunctionSpecs = {{
    {"Max", FunctionKind::Max, 2},
    {"Min", FunctionKind::Min, 2},
    {"Pow", FunctionKind::Pow, 2},
    {"Mod", FunctionKind::Mod, 2},
    {"Log", FunctionKind::Log, 1},
    {"Ceiling", FunctionKind::Ceiling, 1},
    {"Floor", FunctionKind::Floor, 1},
    {"Abs", FunctionKind::Abs, 1},
    {"Rational", FunctionKind::Rational, 2},
    {"ExpectEq", FunctionKind::ExpectEq, 2},
    {"ExpectNe", FunctionKind::ExpectNe, 2},
    {"ExpectLe", FunctionKind::ExpectLe, 2},
    {"ExpectLt", FunctionKind::ExpectLt, 2},
    {"LogicAnd", FunctionKind::LogicAnd, 2},
    {"LogicOr", FunctionKind::LogicOr, 2},
}};

const FunctionSpec *FindFunctionSpec(std::string_view name) {
  for (const FunctionSpec &spec : kFunctionSpecs) {
    if (spec.name == name) {
      return &spec;
    }
  }
  return nullptr;
}

class Scanner {
 public:
  explicit Scanner(std::string_view input) : input_(input) {}

  Token Next() {
    SkipWhitespace();
    const size_t offset = position_;
    if (position_ == input_.size()) {
      return {TokenKind::End, {}, offset};
    }

    const unsigned char current = static_cast<unsigned char>(input_[position_]);
    if (std::isalpha(current) || current == '_') {
      return ReadIdentifier(offset);
    }
    if (std::isdigit(current)) {
      return ReadNumber(offset);
    }

    ++position_;
    switch (current) {
      case '+':
        return {TokenKind::Plus, "+", offset};
      case '-':
        return {TokenKind::Minus, "-", offset};
      case '*':
        return {TokenKind::Multiply, "*", offset};
      case '/':
        return {TokenKind::Divide, "/", offset};
      case ',':
        return {TokenKind::Comma, ",", offset};
      case '(':
        return {TokenKind::LParen, "(", offset};
      case ')':
        return {TokenKind::RParen, ")", offset};
      default:
        throw ParseFailure(ErrorCode::InvalidArgument, "unsupported character at byte " + std::to_string(offset));
    }
  }

 private:
  void SkipWhitespace() {
    while (position_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[position_]))) {
      ++position_;
    }
  }

  Token ReadIdentifier(size_t offset) {
    while (position_ < input_.size()) {
      const unsigned char current = static_cast<unsigned char>(input_[position_]);
      if (!std::isalnum(current) && current != '_' && current != ':') {
        break;
      }
      ++position_;
    }
    return {TokenKind::Identifier, std::string(input_.substr(offset, position_ - offset)), offset};
  }

  Token ReadNumber(size_t offset) {
    bool hasDot = false;
    while (position_ < input_.size()) {
      const char current = input_[position_];
      if (current == '.') {
        if (hasDot) {
          throw ParseFailure(ErrorCode::InvalidArgument, "invalid number at byte " + std::to_string(offset));
        }
        hasDot = true;
        ++position_;
        continue;
      }
      if (!std::isdigit(static_cast<unsigned char>(current))) {
        break;
      }
      ++position_;
    }
    if (input_[position_ - 1] == '.') {
      throw ParseFailure(ErrorCode::InvalidArgument, "invalid number at byte " + std::to_string(offset));
    }
    return {TokenKind::Number, std::string(input_.substr(offset, position_ - offset)), offset};
  }

  std::string_view input_;
  size_t position_ = 0;
};

class Parser {
 public:
  explicit Parser(std::string_view input) : scanner_(input), current_(scanner_.Next()) {}

  BasicPtr Parse() {
    BasicPtr expression = ParseAddSubtract();
    if (current_.kind != TokenKind::End) {
      Fail("unexpected trailing token");
    }
    if (freeSymbols_.size() > kMaxFreeSymbolCount) {
      throw ParseFailure(ErrorCode::ResourceExhausted, "expression has too many free symbols");
    }
    return expression;
  }

 private:
  class DepthGuard {
   public:
    explicit DepthGuard(Parser &parser) : parser_(parser) {
      if (++parser_.depth_ > kMaxAstDepth) {
        throw ParseFailure(ErrorCode::ResourceExhausted, "expression exceeds maximum AST depth");
      }
    }
    ~DepthGuard() {
      --parser_.depth_;
    }

   private:
    Parser &parser_;
  };

  void CountNode() {
    if (++nodeCount_ > kMaxNodeCount) {
      throw ParseFailure(ErrorCode::ResourceExhausted, "expression exceeds maximum node count");
    }
  }

  [[noreturn]] void Fail(const std::string &message) const {
    throw ParseFailure(ErrorCode::InvalidArgument, message + " at byte " + std::to_string(current_.offset));
  }

  void Consume(TokenKind expected) {
    if (current_.kind != expected) {
      Fail("unexpected token");
    }
    current_ = scanner_.Next();
  }

  BasicPtr ParseAddSubtract() {
    DepthGuard guard(*this);
    BasicPtr result = ParseMulDivide();
    while (current_.kind == TokenKind::Plus || current_.kind == TokenKind::Minus) {
      const TokenKind operation = current_.kind;
      Consume(operation);
      BasicPtr right = ParseMulDivide();
      CountNode();
      result = operation == TokenKind::Plus ? SymEngine::add(result, right) : SymEngine::sub(result, right);
    }
    return result;
  }

  BasicPtr ParseMulDivide() {
    DepthGuard guard(*this);
    BasicPtr result = ParseFactor();
    while (current_.kind == TokenKind::Multiply || current_.kind == TokenKind::Divide) {
      const TokenKind operation = current_.kind;
      Consume(operation);
      BasicPtr right = ParseFactor();
      CountNode();
      result = operation == TokenKind::Multiply ? SymEngine::mul(result, right) : SymEngine::div(result, right);
    }
    return result;
  }

  BasicPtr ParseFactor() {
    DepthGuard guard(*this);
    if (current_.kind == TokenKind::Minus) {
      Consume(TokenKind::Minus);
      CountNode();
      return SymEngine::neg(ParseFactor());
    }
    if (current_.kind == TokenKind::LParen) {
      Consume(TokenKind::LParen);
      BasicPtr result = ParseAddSubtract();
      Consume(TokenKind::RParen);
      return result;
    }
    if (current_.kind == TokenKind::Number) {
      return ParseNumber();
    }
    if (current_.kind != TokenKind::Identifier) {
      Fail("expected expression");
    }

    const std::string name = current_.text;
    Consume(TokenKind::Identifier);
    if (current_.kind == TokenKind::LParen) {
      return ParseFunction(name);
    }
    CountNode();
    if (name == detail::kBooleanTrueLiteral) {
      return SymEngine::boolean(true);
    }
    if (name == detail::kBooleanFalseLiteral) {
      return SymEngine::boolean(false);
    }
    if (FindFunctionSpec(name) != nullptr) {
      throw ParseFailure(ErrorCode::InvalidArgument, "function '" + name + "' requires an argument list");
    }
    freeSymbols_.insert(name);
    return SymEngine::symbol(name);
  }

  BasicPtr ParseNumber() {
    const std::string text = current_.text;
    Consume(TokenKind::Number);
    CountNode();
    size_t parsed = 0;
    try {
      if (text.find('.') != std::string::npos) {
        const double value = std::stod(text, &parsed);
        if (parsed != text.size()) {
          Fail("invalid floating-point literal");
        }
        return SymEngine::real_double(value);
      }
      const int64_t value = std::stoll(text, &parsed);
      if (parsed != text.size()) {
        Fail("invalid integer literal");
      }
      return SymEngine::integer(value);
    } catch (const std::invalid_argument &) {
      Fail("invalid numeric literal");
    } catch (const std::out_of_range &) {
      Fail("numeric literal is out of range");
    }
  }

  std::vector<BasicPtr> ParseArguments() {
    Consume(TokenKind::LParen);
    std::vector<BasicPtr> arguments;
    arguments.push_back(ParseAddSubtract());
    while (current_.kind == TokenKind::Comma) {
      Consume(TokenKind::Comma);
      arguments.push_back(ParseAddSubtract());
    }
    Consume(TokenKind::RParen);
    return arguments;
  }

  void RequireArity(const std::string &name, const std::vector<BasicPtr> &arguments, size_t expected) const {
    if (arguments.size() != expected) {
      throw ParseFailure(ErrorCode::InvalidArgument, name + " expects " + std::to_string(expected) + " arguments");
    }
  }

  static BasicPtr MakeRational(const std::vector<BasicPtr> &arguments) {
    if (!SymEngine::is_a<SymEngine::Integer>(*arguments[0]) || !SymEngine::is_a<SymEngine::Integer>(*arguments[1])) {
      throw ParseFailure(ErrorCode::InvalidArgument, "Rational arguments must be integers");
    }
    const auto &numerator = SymEngine::down_cast<const SymEngine::Integer &>(*arguments[0]);
    const auto &denominator = SymEngine::down_cast<const SymEngine::Integer &>(*arguments[1]);
    return SymEngine::Rational::from_two_ints(numerator, denominator);
  }

  static BasicPtr MakeLogical(FunctionKind kind, const std::string &name, const std::vector<BasicPtr> &arguments) {
    SymEngine::set_boolean values;
    for (const BasicPtr &argument : arguments) {
      if (!SymEngine::is_a_Boolean(*argument)) {
        throw ParseFailure(ErrorCode::InvalidArgument, name + " arguments must be boolean expressions");
      }
      values.insert(SymEngine::rcp_static_cast<const SymEngine::Boolean>(argument));
    }
    return kind == FunctionKind::LogicAnd ? SymEngine::logical_and(values) : SymEngine::logical_or(values);
  }

  static BasicPtr ApplyFunctionSpec(const FunctionSpec &spec, const std::string &name,
                                    const std::vector<BasicPtr> &arguments) {
    switch (spec.kind) {
      case FunctionKind::Max:
        return SymEngine::max({arguments[0], arguments[1]});
      case FunctionKind::Min:
        return SymEngine::min({arguments[0], arguments[1]});
      case FunctionKind::Pow:
        return SymEngine::pow(arguments[0], arguments[1]);
      case FunctionKind::Mod:
        return SymEngine::mod(arguments[0], arguments[1]);
      case FunctionKind::Log:
        return SymEngine::log(arguments[0]);
      case FunctionKind::Ceiling:
        return SymEngine::ceiling(arguments[0]);
      case FunctionKind::Floor:
        return SymEngine::floor(arguments[0]);
      case FunctionKind::Abs:
        return SymEngine::abs(arguments[0]);
      case FunctionKind::Rational:
        return MakeRational(arguments);
      case FunctionKind::ExpectEq:
        return SymEngine::Eq(arguments[0], arguments[1]);
      case FunctionKind::ExpectNe:
        return SymEngine::Ne(arguments[0], arguments[1]);
      case FunctionKind::ExpectLe:
        return SymEngine::Le(arguments[0], arguments[1]);
      case FunctionKind::ExpectLt:
        return SymEngine::Lt(arguments[0], arguments[1]);
      case FunctionKind::LogicAnd:
      case FunctionKind::LogicOr:
        return MakeLogical(spec.kind, name, arguments);
    }
    throw ParseFailure(ErrorCode::Internal, "missing symbolic function implementation '" + name + "'");
  }

  BasicPtr ParseFunction(const std::string &name) {
    std::vector<BasicPtr> arguments = ParseArguments();
    CountNode();
    const FunctionSpec *spec = FindFunctionSpec(name);
    if (spec == nullptr) {
      throw ParseFailure(ErrorCode::Unsupported, "unsupported function '" + name + "'");
    }
    RequireArity(name, arguments, spec->arity);
    return ApplyFunctionSpec(*spec, name, arguments);
  }

  Scanner scanner_;
  Token current_;
  size_t depth_ = 0;
  size_t nodeCount_ = 0;
  std::set<std::string> freeSymbols_;
};

std::string Print(const BasicPtr &expression);

detail::ExpressionNode MakeExpressionNode(const BasicPtr &expression);

std::string PrintArguments(const std::vector<BasicPtr> &arguments, const std::string &delimiter) {
  std::vector<std::string> values;
  values.reserve(arguments.size());
  for (const BasicPtr &argument : arguments) {
    values.push_back(Print(argument));
  }
  std::sort(values.begin(), values.end());

  std::string result;
  for (const std::string &value : values) {
    if (!result.empty()) {
      result += delimiter;
    }
    result += value;
  }
  return result;
}

std::vector<BasicPtr> GetArguments(const BasicPtr &expression) {
  const auto &arguments = expression->get_args();
  return {arguments.begin(), arguments.end()};
}

std::string PrintAdd(const BasicPtr &expression) {
  std::vector<BasicPtr> positive;
  std::vector<BasicPtr> negative;
  for (const BasicPtr &argument : expression->get_args()) {
    if (SymEngine::is_a<SymEngine::Mul>(*argument) &&
        SymEngine::down_cast<const SymEngine::Mul &>(*argument).get_coef()->is_negative()) {
      negative.push_back(SymEngine::mul(argument, SymEngine::minus_one));
    } else {
      positive.push_back(argument);
    }
  }
  std::string result = "(";
  if (!positive.empty()) {
    result += PrintArguments(positive, " + ");
  }
  if (!negative.empty()) {
    result += " - " + PrintArguments(negative, " - ");
  }
  return result + ")";
}

std::string PrintMul(const BasicPtr &expression) {
  std::vector<BasicPtr> numerator;
  std::vector<BasicPtr> denominator;
  for (const BasicPtr &argument : expression->get_args()) {
    if (SymEngine::is_a<SymEngine::Pow>(*argument)) {
      const BasicPtr exponent = SymEngine::down_cast<const SymEngine::Pow &>(*argument).get_exp();
      if (SymEngine::is_a_Number(*exponent) &&
          SymEngine::down_cast<const SymEngine::Number &>(*exponent).is_negative()) {
        denominator.push_back(SymEngine::div(SymEngine::one, argument));
        continue;
      }
    }
    numerator.push_back(argument);
  }
  std::string result = "(" + (numerator.empty() ? std::string("1") : PrintArguments(numerator, " * "));
  if (!denominator.empty()) {
    result += " / (" + PrintArguments(denominator, " * ") + ")";
  }
  return result + ")";
}

std::string PrintBinary(const char *name, const BasicPtr &expression, bool sortArguments) {
  std::vector<BasicPtr> arguments = GetArguments(expression);
  if (arguments.size() != 2) {
    throw ParseFailure(ErrorCode::Unsupported, std::string(name) + " must contain two arguments");
  }
  if (sortArguments) {
    return std::string(name) + "(" + PrintArguments(arguments, ", ") + ")";
  }
  return std::string(name) + "(" + Print(arguments[0]) + ", " + Print(arguments[1]) + ")";
}

std::string PrintPow(const BasicPtr &expression) {
  const auto &power = SymEngine::down_cast<const SymEngine::Pow &>(*expression);
  const BasicPtr base = power.get_base();
  const BasicPtr exponent = power.get_exp();
  if (SymEngine::eq(*base, *SymEngine::E)) {
    throw ParseFailure(ErrorCode::Unsupported, "Exp is not part of the canonical input grammar");
  }
  if (SymEngine::eq(*exponent, *SymEngine::rational(1, 2))) {
    throw ParseFailure(ErrorCode::Unsupported, "Sqrt is not part of the canonical input grammar");
  }
  if (SymEngine::eq(*exponent, *SymEngine::one)) {
    return "(" + Print(base) + ")";
  }
  if (SymEngine::is_a<SymEngine::Integer>(*exponent)) {
    const auto &integer = SymEngine::down_cast<const SymEngine::Integer &>(*exponent);
    if (integer.is_positive()) {
      const uint32_t count = integer.as_uint();
      if (count > kMaxExpandedPower) {
        throw ParseFailure(ErrorCode::ResourceExhausted, "positive power exceeds serialization limit");
      }
      std::vector<BasicPtr> factors(count, base);
      return "(" + PrintArguments(factors, " * ") + ")";
    }
  }
  return "Pow(" + Print(base) + ", " + Print(exponent) + ")";
}

std::string PrintMinMax(const char *name, const BasicPtr &expression) {
  const std::vector<BasicPtr> arguments = GetArguments(expression);
  if (arguments.size() < 2) {
    throw ParseFailure(ErrorCode::Unsupported, std::string(name) + " must contain at least two arguments");
  }
  std::string result = std::string(name) + "(" + Print(arguments[0]) + ", " + Print(arguments[1]) + ")";
  for (size_t index = 2; index < arguments.size(); ++index) {
    result = std::string(name) + "(" + result + ", " + Print(arguments[index]) + ")";
  }
  return result;
}

std::string Print(const BasicPtr &expression) {
  if (SymEngine::is_a<SymEngine::Rational>(*expression)) {
    const auto &value = SymEngine::down_cast<const SymEngine::Rational &>(*expression);
    return "Rational(" + value.get_num()->__str__() + " , " + value.get_den()->__str__() + ")";
  }
  if (SymEngine::is_a_Number(*expression) || SymEngine::is_a<SymEngine::BooleanAtom>(*expression) ||
      SymEngine::is_a<SymEngine::Symbol>(*expression)) {
    return expression->__str__();
  }
  if (SymEngine::is_a<SymEngine::Add>(*expression)) return PrintAdd(expression);
  if (SymEngine::is_a<SymEngine::Mul>(*expression)) return PrintMul(expression);
  if (SymEngine::is_a<SymEngine::Pow>(*expression)) return PrintPow(expression);
  if (SymEngine::is_a<SymEngine::Max>(*expression)) return PrintMinMax("Max", expression);
  if (SymEngine::is_a<SymEngine::Min>(*expression)) return PrintMinMax("Min", expression);
  if (SymEngine::is_a<SymEngine::Mod>(*expression)) return PrintBinary("Mod", expression, false);
  if (SymEngine::is_a<SymEngine::Log>(*expression)) {
    return "Log(" + PrintArguments(GetArguments(expression), ", ") + ")";
  }
  if (SymEngine::is_a<SymEngine::Ceiling>(*expression)) {
    return "Ceiling(" + PrintArguments(GetArguments(expression), ", ") + ")";
  }
  if (SymEngine::is_a<SymEngine::Floor>(*expression)) {
    return "Floor(" + PrintArguments(GetArguments(expression), ", ") + ")";
  }
  if (SymEngine::is_a<SymEngine::Abs>(*expression)) {
    return "Abs(" + PrintArguments(GetArguments(expression), ", ") + ")";
  }
  if (SymEngine::is_a<SymEngine::Equality>(*expression)) return PrintBinary("ExpectEq", expression, true);
  if (SymEngine::is_a<SymEngine::Unequality>(*expression)) return PrintBinary("ExpectNe", expression, true);
  if (SymEngine::is_a<SymEngine::LessThan>(*expression)) return PrintBinary("ExpectLe", expression, false);
  if (SymEngine::is_a<SymEngine::StrictLessThan>(*expression)) return PrintBinary("ExpectLt", expression, false);
  if (SymEngine::is_a<SymEngine::And>(*expression)) {
    return "LogicAnd(" + PrintArguments(GetArguments(expression), ", ") + ")";
  }
  if (SymEngine::is_a<SymEngine::Or>(*expression)) {
    return "LogicOr(" + PrintArguments(GetArguments(expression), ", ") + ")";
  }
  throw ParseFailure(ErrorCode::Unsupported,
                     "unsupported SymEngine node '" + SymEngine::type_code_name(expression->get_type_code()) + "'");
}

detail::ExpressionNode MakeExpressionNode(const BasicPtr &expression) {
  using detail::ExpressionNode;
  using detail::NodeKind;
  auto makeNode = [&](NodeKind kind) {
    std::vector<ExpressionNode> arguments;
    arguments.reserve(expression->get_args().size());
    for (const BasicPtr &argument : expression->get_args()) {
      arguments.push_back(MakeExpressionNode(argument));
    }
    return ExpressionNode{kind, {}, std::move(arguments)};
  };

  if (SymEngine::is_a<SymEngine::Integer>(*expression)) {
    return {NodeKind::Integer, expression->__str__(), {}};
  }
  if (SymEngine::is_a<SymEngine::RealDouble>(*expression)) {
    return {NodeKind::Real, expression->__str__(), {}};
  }
  if (SymEngine::is_a<SymEngine::BooleanAtom>(*expression)) {
    return {NodeKind::Boolean, expression->__str__(), {}};
  }
  if (SymEngine::is_a<SymEngine::Symbol>(*expression)) {
    return {NodeKind::Symbol, expression->__str__(), {}};
  }
  if (SymEngine::is_a<SymEngine::Rational>(*expression)) {
    const auto &value = SymEngine::down_cast<const SymEngine::Rational &>(*expression);
    return {NodeKind::Rational,
            {},
            {{NodeKind::Integer, value.get_num()->__str__(), {}}, {NodeKind::Integer, value.get_den()->__str__(), {}}}};
  }
  if (SymEngine::is_a<SymEngine::Add>(*expression)) return makeNode(NodeKind::Add);
  if (SymEngine::is_a<SymEngine::Mul>(*expression)) return makeNode(NodeKind::Mul);
  if (SymEngine::is_a<SymEngine::Pow>(*expression)) return makeNode(NodeKind::Pow);
  if (SymEngine::is_a<SymEngine::Max>(*expression)) return makeNode(NodeKind::Max);
  if (SymEngine::is_a<SymEngine::Min>(*expression)) return makeNode(NodeKind::Min);
  if (SymEngine::is_a<SymEngine::Mod>(*expression)) return makeNode(NodeKind::Mod);
  if (SymEngine::is_a<SymEngine::Log>(*expression)) return makeNode(NodeKind::Log);
  if (SymEngine::is_a<SymEngine::Ceiling>(*expression)) return makeNode(NodeKind::Ceiling);
  if (SymEngine::is_a<SymEngine::Floor>(*expression)) return makeNode(NodeKind::Floor);
  if (SymEngine::is_a<SymEngine::Abs>(*expression)) return makeNode(NodeKind::Abs);
  if (SymEngine::is_a<SymEngine::Equality>(*expression)) return makeNode(NodeKind::Equal);
  if (SymEngine::is_a<SymEngine::Unequality>(*expression)) return makeNode(NodeKind::NotEqual);
  if (SymEngine::is_a<SymEngine::LessThan>(*expression)) return makeNode(NodeKind::LessEqual);
  if (SymEngine::is_a<SymEngine::StrictLessThan>(*expression)) return makeNode(NodeKind::LessThan);
  if (SymEngine::is_a<SymEngine::And>(*expression)) return makeNode(NodeKind::LogicAnd);
  if (SymEngine::is_a<SymEngine::Or>(*expression)) return makeNode(NodeKind::LogicOr);
  throw ParseFailure(ErrorCode::Unsupported,
                     "unsupported SymEngine node '" + SymEngine::type_code_name(expression->get_type_code()) + "'");
}

}  // namespace

class Expression::Impl {
 public:
  explicit Impl(BasicPtr expression) : expression_(std::move(expression)) {}

  const BasicPtr &expression() const {
    return expression_;
  }

 private:
  BasicPtr expression_;
};

Result<Expression> Expression::ParseImpl(std::string_view text, bool requireCanonical) {
  if (text.empty()) {
    return Error{ErrorCode::InvalidArgument, "expression is empty"};
  }
  if (text.size() > kMaxSerializedBytes) {
    return Error{ErrorCode::ResourceExhausted, "expression exceeds maximum serialized size"};
  }
  try {
    BasicPtr expression = Parser(text).Parse();
    if (requireCanonical) {
      const std::string serialized = Print(expression);
      if (serialized != text) {
        return Error{ErrorCode::InvalidArgument, "expression is not canonical; canonical form is '" + serialized + "'"};
      }
    }
    return Expression(std::make_shared<const Impl>(std::move(expression)));
  } catch (const ParseFailure &failure) {
    return Error{failure.code(), failure.what()};
  } catch (const std::exception &exception) {
    return Error{ErrorCode::InvalidArgument, exception.what()};
  } catch (...) {
    return Error{ErrorCode::Internal, "unknown SymEngine parser failure"};
  }
}

Result<Expression> Expression::Parse(std::string_view canonical) {
  return ParseImpl(canonical, /*requireCanonical=*/true);
}

Result<Expression> Expression::ParseRelaxed(std::string_view text) {
  return ParseImpl(text, /*requireCanonical=*/false);
}

Result<Expression> Expression::Simplify() const {
  try {
    BasicPtr expanded = SymEngine::expand(impl_->expression());
    BasicPtr simplified = SymEngine::simplify(expanded);
    (void)Print(simplified);
    return Expression(std::make_shared<const Impl>(std::move(simplified)));
  } catch (const ParseFailure &failure) {
    return Error{failure.code(), failure.what()};
  } catch (const std::exception &exception) {
    return Error{ErrorCode::Internal, exception.what()};
  } catch (...) {
    return Error{ErrorCode::Internal, "unknown SymEngine simplify failure"};
  }
}

std::string Expression::Serialize() const {
  return Print(impl_->expression());
}

Result<detail::ExpressionNode> detail::ExpressionAccess::GetTree(const Expression &expression) {
  try {
    return MakeExpressionNode(expression.impl_->expression());
  } catch (const ParseFailure &failure) {
    return Error{failure.code(), failure.what()};
  } catch (const std::exception &exception) {
    return Error{ErrorCode::Internal, exception.what()};
  } catch (...) {
    return Error{ErrorCode::Internal, "unknown SymEngine tree traversal failure"};
  }
}

Result<std::vector<std::string>> detail::ExpressionAccess::GetFreeSymbols(const Expression &expression) {
  try {
    std::vector<std::string> symbols;
    for (const BasicPtr &symbol : SymEngine::free_symbols(*expression.impl_->expression())) {
      symbols.push_back(symbol->__str__());
    }
    std::sort(symbols.begin(), symbols.end());
    return symbols;
  } catch (const std::exception &exception) {
    return Error{ErrorCode::Internal, exception.what()};
  } catch (...) {
    return Error{ErrorCode::Internal, "unknown SymEngine free-symbol traversal failure"};
  }
}

Result<detail::NormalizedExpression> detail::ExpressionAccess::NormalizeFreeSymbols(const Expression &expression) {
  Result<std::vector<std::string>> symbols = GetFreeSymbols(expression);
  if (!symbols) {
    return symbols.error();
  }
  SymEngine::map_basic_basic replacements;
  for (size_t index = 0; index < symbols.value().size(); ++index) {
    if (symbols.value()[index].rfind(detail::kOperandPlaceholderPrefix, 0) == 0) {
      return Error{ErrorCode::InvalidArgument,
                   "source expression uses reserved placeholder symbol '" + symbols.value()[index] + "'"};
    }
    replacements[SymEngine::symbol(symbols.value()[index])] =
        SymEngine::symbol(std::string(detail::kOperandPlaceholderPrefix) + std::to_string(index));
  }
  try {
    BasicPtr normalized = expression.impl_->expression()->xreplace(replacements);
    (void)Print(normalized);
    return NormalizedExpression{Expression(std::make_shared<const Expression::Impl>(std::move(normalized))),
                                std::move(symbols.value())};
  } catch (const ParseFailure &failure) {
    return Error{failure.code(), failure.what()};
  } catch (const std::exception &exception) {
    return Error{ErrorCode::Internal, exception.what()};
  } catch (...) {
    return Error{ErrorCode::Internal, "unknown SymEngine normalization failure"};
  }
}

Result<Expression> detail::ExpressionAccess::Substitute(const Expression &expression,
                                                        const std::map<std::string, std::string> &replacements) {
  try {
    SymEngine::map_basic_basic substitutions;
    for (const auto &[symbol, replacement] : replacements) {
      Result<Expression> parsedReplacement = Expression::Parse(replacement);
      if (!parsedReplacement) {
        return Error{parsedReplacement.error().code,
                     "invalid replacement for '" + symbol + "': " + parsedReplacement.error().message};
      }
      substitutions[SymEngine::symbol(symbol)] = parsedReplacement.value().impl_->expression();
    }
    BasicPtr substituted = expression.impl_->expression()->xreplace(substitutions);
    (void)Print(substituted);
    return Expression(std::make_shared<const Expression::Impl>(std::move(substituted)));
  } catch (const ParseFailure &failure) {
    return Error{failure.code(), failure.what()};
  } catch (const std::exception &exception) {
    return Error{ErrorCode::Internal, exception.what()};
  } catch (...) {
    return Error{ErrorCode::Internal, "unknown SymEngine substitution failure"};
  }
}

Result<double> detail::ExpressionAccess::Evaluate(const Expression &expression,
                                                  const std::map<std::string, double> &bindings) {
  try {
    SymEngine::map_basic_basic substitutions;
    for (const auto &[symbol, value] : bindings) {
      substitutions[SymEngine::symbol(symbol)] = SymEngine::real_double(value);
    }
    BasicPtr substituted = expression.impl_->expression()->xreplace(substitutions);
    if (!SymEngine::free_symbols(*substituted).empty()) {
      return Error{ErrorCode::InvalidArgument, "cannot evaluate expression with unbound symbols"};
    }
    return SymEngine::eval_double(*substituted);
  } catch (const std::exception &exception) {
    return Error{ErrorCode::Internal, exception.what()};
  } catch (...) {
    return Error{ErrorCode::Internal, "unknown SymEngine evaluation failure"};
  }
}

Result<detail::TruthValue> detail::ExpressionAccess::Prove(const Expression &predicate) {
  try {
    BasicPtr simplified = SymEngine::simplify(predicate.impl_->expression());
    if (SymEngine::is_a<SymEngine::BooleanAtom>(*simplified)) {
      return SymEngine::down_cast<const SymEngine::BooleanAtom &>(*simplified).get_val() ? TruthValue::True
                                                                                         : TruthValue::False;
    }
    if (!SymEngine::is_a_Boolean(*simplified)) {
      return Error{ErrorCode::InvalidArgument, "proof input is not a boolean expression"};
    }
    return TruthValue::Unknown;
  } catch (const std::exception &exception) {
    return Error{ErrorCode::Internal, exception.what()};
  } catch (...) {
    return Error{ErrorCode::Internal, "unknown SymEngine proof failure"};
  }
}

Result<std::string> MultiplyExpressions(std::string_view lhs, std::string_view rhs) {
  Result<Expression> product = Expression::Parse("(__arg0 * __arg1)");
  if (!product) {
    return product.error();
  }
  Result<Expression> substituted = detail::ExpressionAccess::Substitute(
      product.value(), {{"__arg0", std::string(lhs)}, {"__arg1", std::string(rhs)}});
  if (!substituted) {
    return substituted.error();
  }
  Result<Expression> simplified = substituted.value().Simplify();
  if (!simplified) {
    return simplified.error();
  }
  return simplified.value().Serialize();
}

namespace {
// Size symbols are integers. Ceiling/Floor return integers even for rational
// arguments; a general Pow may represent division and is deliberately excluded.
bool isIntegerValued(const afir::detail::ExpressionNode &node) {
  using afir::detail::NodeKind;
  switch (node.kind) {
    case NodeKind::Ceiling:
    case NodeKind::Floor:
      return true;
    case NodeKind::Integer:
    case NodeKind::Symbol:
      return true;
    case NodeKind::Add:
    case NodeKind::Mul:
    case NodeKind::Max:
    case NodeKind::Min:
    case NodeKind::Mod:
    case NodeKind::Abs:
      return std::all_of(node.arguments.begin(), node.arguments.end(), isIntegerValued);
    default:
      return false;
  }
}

// 字面量正整数节点的值。不是那个形态返回 nullopt。
std::optional<int64_t> positiveIntegerValue(const afir::detail::ExpressionNode &node) {
  if (node.kind != afir::detail::NodeKind::Integer) {
    return std::nullopt;
  }
  try {
    size_t parsed = 0;
    const int64_t value = std::stoll(node.value, &parsed);
    if (parsed == node.value.size() && value > 0) {
      return value;
    }
  } catch (const std::exception &) {
    // An integer outside int64_t cannot establish this bounded factor check.
  }
  return std::nullopt;
}

}  // namespace

bool ProveExpressionMultipleOf(std::string_view expression, int64_t factor) {
  if (expression.empty() || factor <= 0) {
    return false;
  }
  Result<Expression> parsed = Expression::Parse(expression);
  if (!parsed) {
    return false;
  }
  Result<Expression> simplified = parsed.value().Simplify();
  if (!simplified) {
    return false;
  }
  Result<afir::detail::ExpressionNode> tree = afir::detail::ExpressionAccess::GetTree(simplified.value());
  if (!tree) {
    return false;
  }
  const afir::detail::ExpressionNode &root = tree.value();
  if (std::optional<int64_t> literal = positiveIntegerValue(root)) {
    return *literal % factor == 0;
  }
  if (root.kind != afir::detail::NodeKind::Mul) {
    return false;
  }
  for (size_t index = 0; index < root.arguments.size(); ++index) {
    std::optional<int64_t> literal = positiveIntegerValue(root.arguments[index]);
    if (!literal || *literal % factor != 0) {
      continue;
    }
    bool othersAreIntegers = true;
    for (size_t other = 0; other < root.arguments.size() && othersAreIntegers; ++other) {
      othersAreIntegers = other == index || isIntegerValued(root.arguments[other]);
    }
    if (othersAreIntegers) {
      return true;
    }
  }
  return false;
}

bool ProveExpressionsEqual(std::string_view lhs, std::string_view rhs) {
  Result<Expression> left = Expression::Parse(lhs);
  Result<Expression> right = Expression::Parse(rhs);
  if (!left || !right) {
    return false;
  }
  Result<Expression> leftSimplified = left.value().Simplify();
  Result<Expression> rightSimplified = right.value().Simplify();
  if (!leftSimplified || !rightSimplified) {
    return false;
  }
  // 先比规范串。两个可证相等的表达式化简后序列化成同一个串，这一步就够了，而且不依赖
  // 谓词文本能不能被 `Parse` 接受。
  const std::string leftText = leftSimplified.value().Serialize();
  const std::string rightText = rightSimplified.value().Serialize();
  if (leftText == rightText) {
    return true;
  }
  // 化简后仍不同名的，再交给证明器。注意 `Parse` 要求规范形式：当两侧恰好相等时
  // `ExpectEq(a, a)` 的规范形式是 `True`，拼出来的串会被 Parse 拒掉——所以上面那一步
  // 不是优化，是这条路的必要补充。
  Result<Expression> predicate = Expression::Parse("ExpectEq(" + leftText + ", " + rightText + ")");
  if (!predicate) {
    return false;
  }
  Result<detail::TruthValue> proof = detail::ExpressionAccess::Prove(predicate.value());
  return proof && proof.value() == detail::TruthValue::True;
}

}  // namespace mlir::afir
