/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "scanner.h"
#include <unordered_map>
#include <set>
#include "common/checker.h"

namespace af {
namespace {
std::unordered_map<std::string, TokenType> kTokenMap = {{"Max", TokenType::kMax},
                                                        {"Min", TokenType::kMin},
                                                        {"Pow", TokenType::kPow},
                                                        {"Log", TokenType::kLog},
                                                        {"Ceiling", TokenType::kCeil},
                                                        {"Rational", TokenType::kRational},
                                                        {"+", TokenType::kPlus},
                                                        {"-", TokenType::kMinus},
                                                        {"*", TokenType::kMultiply},
                                                        {"/", TokenType::kDivide},
                                                        {"(", TokenType::kLparen},
                                                        {")", TokenType::kRparen},
                                                        {",", TokenType::kComma},
                                                        {"Abs", TokenType::kAbs},
                                                        {"ExpectEq", TokenType::kEq},
                                                        {"ExpectNe", TokenType::kNe},
                                                        {"ExpectLe", TokenType::kLe},
                                                        {"ExpectLt", TokenType::kLt},
                                                        {"True", TokenType::kTrue},
                                                        {"False", TokenType::kFalse},
                                                        {"Floor", TokenType::kFloor},
                                                        {"Mod", TokenType::kMod},
                                                        {"Exp", TokenType::kExp},
                                                        {"Sqrt", TokenType::kSqrt},
                                                        {"LogicAnd", TokenType::kLogicalAnd},
                                                        {"LogicOr", TokenType::kLogicalOr}};
}
Scanner::Scanner(const std::string &input) : input_(input), pos_(0) {
  Advance();
}

graphStatus Scanner::GetNextToken(Token &token) {
  SkipWhitespace();

  if (currentChar_ == '\0') {
    token = {TokenType::kEnd, ""};
    return GRAPH_SUCCESS;
  }

  // 识别数字
  if (std::isdigit(currentChar_)) {
    token = ReadNumber();
    return GRAPH_SUCCESS;
  }
  // 识别函数
  if (std::isalpha(currentChar_)) {
    std::string identifier = ReadIdentifier();
    if (kTokenMap.find(identifier) != kTokenMap.end()) {
      token = {kTokenMap[identifier], identifier};
    } else {
      token = {TokenType::kIdentifier, identifier};
    }
    return GRAPH_SUCCESS;
  }
  // 识别操作符
  const auto token_key = std::string(1, currentChar_);
  if (kTokenMap.find(token_key) != kTokenMap.end()) {
    Advance();
    token = {kTokenMap[token_key], token_key};
    return GRAPH_SUCCESS;
  }
  GELOGE(ge::PARAM_INVALID, "Unsupported operator: %s", token_key.c_str());
  token = {TokenType::kEnd, ""};
  return GRAPH_FAILED;
}

void Scanner::Advance(size_t steps) {
  while (steps-- > 0) {
    if (pos_ < input_.length()) {
      currentChar_ = input_[pos_++];
    } else {
      currentChar_ = '\0';
      break;
    }
  }
}

void Scanner::SkipWhitespace() {
  while (std::isspace(currentChar_)) {
    Advance();
  }
}

std::string Scanner::ReadIdentifier() {
  std::string result;
  while (std::isalnum(currentChar_) || currentChar_ == ':' || currentChar_ == '_') {
    result += currentChar_;
    Advance();
  }
  return result;
}

Token Scanner::ReadNumber() {
  std::string result;
  while (std::isdigit(currentChar_) || currentChar_ == '.') {
    result += currentChar_;
    Advance();
  }
  // 兼容 RealDouble 的科学计数法打印形式(如 9.99999993922529e-09)；
  // 仅当 e/E 后为 [可选符号]+数字 时才并入数字，避免误吞后续标识符
  if (currentChar_ == 'e' || currentChar_ == 'E') {
    size_t next = pos_;
    if (next < input_.length() && (input_[next] == '+' || input_[next] == '-')) {
      ++next;
    }
    if (next < input_.length() && std::isdigit(input_[next])) {
      result += currentChar_;
      Advance();
      if (currentChar_ == '+' || currentChar_ == '-') {
        result += currentChar_;
        Advance();
      }
      while (std::isdigit(currentChar_)) {
        result += currentChar_;
        Advance();
      }
    }
  }
  return {TokenType::kNumber, result};
}
}  // namespace af
