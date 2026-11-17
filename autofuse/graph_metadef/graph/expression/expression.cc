/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <vector>
#include <map>
#include <symengine/rational.h>
#include "graph/symbolizer/symbolic.h"
#include "attribute_group/attr_group_shape_env.h"
#include "expression_impl.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/math_util.h"
#include "common/checker.h"
#include "const_values.h"

namespace af {
Expression::~Expression() {}

Expression::Expression(Expression &&other) noexcept {
  impl_ = std::move(other.impl_);
}

Expression &Expression::operator=(const Expression &other) {
  if (&other != this) {
    impl_ = ComGraphMakeUnique<ExpressionImpl>();
    if ((other.impl_ != nullptr) && (impl_ != nullptr)) {
      *impl_ = *other.impl_;
    }
  }
  return *this;
}

Expression::Expression(const Expression &other) {
  // Copy
  impl_ = ComGraphMakeUnique<ExpressionImpl>();
  if ((other.impl_ != nullptr) && (impl_ != nullptr)) {
    *impl_ = *other.impl_;
  }
}

Expression &Expression::operator=(Expression &&other) noexcept {
  if (&other != this) {
    impl_ = std::move(other.impl_);
  }
  return *this;
}

std::unique_ptr<char_t[]> Expression::Str(const StrType type) const {
  if (impl_ != nullptr) {
    auto str = impl_->Str(type);
    if (str.empty()) {
      return nullptr;
    }
    auto uni_ptr = ComGraphMakeUnique<char_t[]>(str.size() + 1);
    IF_NULL_RETURN_NULL(uni_ptr);
    // 当src size < dst size时，strncpy_s会在末尾str.size()位置添加'\0'
    GE_ASSERT_EOK(strncpy_s(uni_ptr.get(), str.size() + 1, str.c_str(), str.size()));
    return uni_ptr;
  }
  return nullptr;
}

Expression Expression::Parse(const char_t *str) {
  if (str == nullptr) {
    GELOGE(FAILED, "Parse expression str is nullptr");
    return Expression(nullptr);
  }
  return Expression(ExpressionImpl::Parse(str));
}

std::unique_ptr<char_t[]> Expression::Serialize() const {
  return Str(StrType::kStrCpp);
}

Expression Expression::Deserialize(const char_t *str) {
  return Expression(ExpressionImpl::Deserialize(str));
}

std::vector<Expression> Expression::GetArgs() {
  std::vector<Expression> args;
  if (impl_ == nullptr) {
    return args;
  }

  for (ExpressionImplPtr &arg : impl_->GetArgs()) {
    args.emplace_back(Expression(std::move(arg)));
  }
  return args;
}

ExprType Expression::GetExprType() const {
  if (impl_ != nullptr) {
    return impl_->GetExprType();
  }
  return ExprType::kExprNone;
}

bool Expression::IsConstExpr() const {
  if (impl_ != nullptr) {
    return impl_->IsConstExpr();
  }
  return false;
}

bool Expression::IsVariableExpr() const {
  if (impl_ != nullptr) {
    return impl_->IsVariableExpr();
  }
  return false;
}

bool Expression::IsBooleanExpr() const {
  if (impl_ != nullptr) {
    return impl_->IsBooleanExpr();
  }
  return false;
}

Expression Expression::Replace(const std::vector<std::pair<Expression, Expression>> &replace_vars) const {
  if (impl_ != nullptr) {
    std::map<ExpressionImpl *, ExpressionImpl *> impl_map;
    for (auto &item : replace_vars) {
      impl_map[item.first.impl_.get()] = item.second.impl_.get();
    }
    return Expression(impl_->Replace(impl_map));
  }
  return Expression(nullptr);
}

Expression Expression::Subs(const std::vector<std::pair<Expression, Expression>> &subs_vars) const {
  if (impl_ != nullptr) {
    std::map<ExpressionImpl *, ExpressionImpl *> impl_map;
    for (auto &item : subs_vars) {
      impl_map[item.first.impl_.get()] = item.second.impl_.get();
    }
    return Expression(impl_->Subs(impl_map));
  }
  return Expression(nullptr);
}

std::vector<Expression> Expression::FreeSymbols() const {
  if (impl_ != nullptr) {
    std::vector<Expression> ret;
    for (auto &free_symbol : impl_->FreeSymbols()) {
      ret.emplace_back(Expression(std::move(free_symbol)));
    }
    return ret;
  }
  return {};
}

graphStatus Expression::GetResult(const std::vector<std::pair<Expression, Expression>> &vars_value,
                                  double &result) const {
  Expression replace_expr = Replace(vars_value);
  if ((replace_expr.impl_ != nullptr) && (replace_expr.impl_->GetResult(result))) {
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

bool Expression::IsValid() const {
  return impl_ != nullptr;
}

uint64_t Expression::Hash() const {
  if (impl_ != nullptr) {
    return impl_->Hash();
  }
  return std::numeric_limits<uint64_t>::max();
}

int64_t Expression::Compare(const Expression &e) const {
  if (impl_ != nullptr && e.impl_ != nullptr) {
    return impl_->Compare(*e.impl_);
  }
  return std::numeric_limits<int64_t>::max();
}

// 模板函数的定义
template <typename T>
typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, bool>::type
Expression::GetConstValue(T &value) const {
  if (!IsConstExpr() || impl_ == nullptr) {
    return false;
  }
  return impl_->GetConstValue(value);
}

// 显式实例化
template bool Expression::GetConstValue<int32_t>(int32_t &) const;    // 实例化 int32 类型
template bool Expression::GetConstValue<uint32_t>(uint32_t &) const;  // 实例化 uint32 类型
template bool Expression::GetConstValue<int64_t>(int64_t &) const;    // 实例化 int64 类型
template bool Expression::GetConstValue<uint64_t>(uint64_t &) const;  // 实例化 uint64 类型
template bool Expression::GetConstValue<double>(double &) const;      // 实例化 double 类型
template bool Expression::GetConstValue<float>(float &) const;        // 实例化 float 类型
template bool Expression::GetConstValue<bool>(bool &) const;          // 实例化 bool 类型

Expression Expression::operator+(const Expression &other) const {
  return sym::Add(*this, other);
}

Expression Expression::operator-(const Expression &other) const {
  return sym::Sub(*this, other);
}

Expression Expression::operator*(const Expression &other) const {
  return sym::Mul(*this, other);
}

Expression Expression::operator/(const Expression &other) const {
  return sym::Div(*this, other);
}

Expression Expression::Simplify() const {
  if (GetCurShapeEnvContext() != nullptr) {
    return GetCurShapeEnvContext()->Simplify(*this);
  }
  if (impl_ != nullptr) {
    return Expression(impl_->Simplify());
  }
  return Expression(nullptr);
}

bool Expression::ContainVar(const Expression &e) const {
  if (impl_ != nullptr) {
    return impl_->ContainVar(e.impl_.get());
  }
  return false;
}

void Expression::AsNumerDenom(Expression &numer, Expression &denom) const {
  if (impl_ != nullptr) {
    ExpressionImplPtr impl_numer;
    ExpressionImplPtr impl_denom;
    impl_->AsNumerDenom(impl_numer, impl_denom);
    numer = Expression(std::move(impl_numer));
    denom = Expression(std::move(impl_denom));
  } else {
    numer = Expression(nullptr);
    denom = Expression(nullptr);
  }
}

bool Expression::operator==(const Expression &e) const {
  if (impl_ != nullptr && e.impl_ != nullptr) {
    return (*impl_ == *e.impl_);
  }
  return false;
}

bool Expression::operator!=(const Expression &e) const {
  return !(*this == e);
}

std::ostream &operator<<(std::ostream &os, const Expression &e) {
  if (e.impl_ != nullptr) {
    os << *e.impl_;
  }
  return os;
}

Expression::Expression(ExpressionImplPtr &&e) : impl_(std::move(e)) {}

Expression::Expression() {
  impl_ = ComGraphMakeUnique<ExpressionImpl>("");
}

Expression Expression::CanonicalizeBoolExpr() const {
  if (impl_ != nullptr) {
    return Expression(impl_->CanonicalizeBoolExpr());
  }
  return Expression(nullptr);
}

Symbol::Symbol(ExpressionImplPtr &&e) : Expression(std::move(e)) {}

Symbol::Symbol(int32_t value, const char_t *name) {
  impl_ = ComGraphMakeUnique<ExpressionImpl>(value, name);
}

Symbol::Symbol(int64_t value, const char_t *name) {
  impl_ = ComGraphMakeUnique<ExpressionImpl>(value, name);
}
Symbol::Symbol(uint32_t value, const char_t *name) {
  impl_ = ComGraphMakeUnique<ExpressionImpl>(value, name);
}
Symbol::Symbol(uint64_t value, const char_t *name) {
  impl_ = ComGraphMakeUnique<ExpressionImpl>(value, name);
}
Symbol::Symbol(double value, const char_t *name) {
  impl_ = ComGraphMakeUnique<ExpressionImpl>(value, name);
}

Symbol::Symbol(const char_t *name) {
  impl_ = ComGraphMakeUnique<ExpressionImpl>(name);
}

std::unique_ptr<char_t[]> Symbol::GetName() const {
  if (impl_ != nullptr) {
    auto str = impl_->GetName();
    if (str.empty()) {
      return nullptr;
    }
    auto uni_ptr = ComGraphMakeUnique<char_t[]>(str.size() + 1U);
    IF_NULL_RETURN_NULL(uni_ptr);
    // 当src size < dst size时，strncpy_s会在末尾str.size()位置添加'\0'
    GE_ASSERT_EOK(strncpy_s(uni_ptr.get(), str.size() + 1, str.c_str(), str.size()));
    return uni_ptr;
  }
  return nullptr;
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value, bool>::type
Expression::ComputeHint(T &hint) const {
  if (IsConstExpr()) {
    return GetConstValue(hint);
  }
  if (GetCurShapeEnvContext() == nullptr) {
    GELOGW("Shape env is nullptr, cannot compute hint, expr: %s", Str().get());
    return false;
  }
  return GetCurShapeEnvContext()->EvaluateExpr(*this).GetConstValue(hint);
}

template bool Expression::ComputeHint<int32_t>(int32_t &) const;    // 实例化 int32 类型
template bool Expression::ComputeHint<uint32_t>(uint32_t &) const;  // 实例化 uint32 类型
template bool Expression::ComputeHint<int64_t>(int64_t &) const;    // 实例化 int64 类型
template bool Expression::ComputeHint<uint64_t>(uint64_t &) const;  // 实例化 uint64 类型
template bool Expression::ComputeHint<double>(double &) const;      // 实例化 double 类型
template bool Expression::ComputeHint<float>(float &) const;        // 实例化 float 类型
template bool Expression::ComputeHint<bool>(bool &) const;          // 实例化 bool 类型

namespace sym {
Expression operator+(const Expression &e1, const Expression &e2) {
  return Add(e1, e2);
}

Expression operator-(const Expression &e1, const Expression &e2) {
  return Sub(e1, e2);
}

Expression operator*(const Expression &e1, const Expression &e2) {
  return Mul(e1, e2);
}

Expression operator/(const Expression &e1, const Expression &e2) {
  return Div(e1, e2);
}
}  // namespace sym
}  // namespace af
