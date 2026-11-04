/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_DIALECT_AFIR_SUPPORT_SYMBOLICEXPRESSION_H
#define AUTOFUSE_MLIR_DIALECT_AFIR_SUPPORT_SYMBOLICEXPRESSION_H

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace mlir::afir {

namespace detail {
class ExpressionAccess;
}

enum class ErrorCode {
  InvalidArgument,
  ResourceExhausted,
  Unsupported,
  Internal,
};

struct Error {
  ErrorCode code;
  std::string message;
};

template <typename T>
class Result {
 public:
  Result(T value) : storage_(std::move(value)) {}
  Result(Error error) : storage_(std::move(error)) {}

  explicit operator bool() const {
    return std::holds_alternative<T>(storage_);
  }
  const T &value() const {
    return std::get<T>(storage_);
  }
  T &value() {
    return std::get<T>(storage_);
  }
  const Error &error() const {
    return std::get<Error>(storage_);
  }

 private:
  std::variant<T, Error> storage_;
};

class Expression {
 public:
  /// Parses text that is already in canonical form, rejecting anything else.
  /// This is the form stored in `#afir.expr`, so attribute parsing uses it.
  static Result<Expression> Parse(std::string_view canonical);

  /// Parses valid non-canonical text, including generated expressions and predicates.
  /// Unlike Parse, does not require the input to match its canonical serialization.
  static Result<Expression> ParseRelaxed(std::string_view text);

  Result<Expression> Simplify() const;
  std::string Serialize() const;

 private:
  friend class detail::ExpressionAccess;
  class Impl;

  static Result<Expression> ParseImpl(std::string_view text, bool requireCanonical);

  explicit Expression(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}

  std::shared_ptr<const Impl> impl_;
};

// `lhs * rhs` 的规范形式。
//
// `Expression::Parse` 只接受 canonical 形式，调用方不能直接拼 `"(lhs * rhs)"`。
// 这里用规范模板 `(__arg0 * __arg1)` 做替换，再交给 SymEngine 化简；模板只在这一处出现。
Result<std::string> MultiplyExpressions(std::string_view lhs, std::string_view rhs);

// 两个规范表达式是否可证相等。
//
// 判据是 `ExpectEq(Simplify(lhs), Simplify(rhs))` 交给 `Prove`，只有答 `True` 才算相等。
// **`Unknown` 与解析失败都返回 false**：调用方拿到的是「证不了」，不是「不等」。所有调用方
// 都必须能承受这个保守方向——合轴用它时，证不了就封段，后果是多一层循环而不是算错地址。
//
// 放在这里而不是各调用点各写一份：`bridge/AfirBuilder/AfirBuilder.cpp` 里
// `proveExpressionsEqual` 是同一件事的一份文件内静态实现，那一份应当切到这里来，
// 但切它不属于本需求，先记下不动。
bool ProveExpressionsEqual(std::string_view lhs, std::string_view rhs);

}  // namespace mlir::afir

#endif  // AUTOFUSE_MLIR_DIALECT_AFIR_SUPPORT_SYMBOLICEXPRESSION_H
