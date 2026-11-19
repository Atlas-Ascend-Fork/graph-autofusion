/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/symbolizer/symbolic.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "common/checker.h"
#include "attribute_group/attr_group_shape_env.h"

namespace af {
const char_t *kInvalidExpr = "invalid expression";

std::string SymbolicUtils::ToString(const Expression &e) {
  auto ret = e.Str(StrType::kStrCpp);
  return (ret != nullptr) ? ret.get() : kInvalidExpr;
}

TriBool SymbolicUtils::StaticCheckEq(const Expression &e1, const Expression &e2) {
  return StaticCheckBool(sym::Eq(e1.Simplify(), e2.Simplify()));
}

TriBool SymbolicUtils::StaticCheckNe(const Expression &e1, const Expression &e2) {
  return StaticCheckBool(sym::Ne(e1.Simplify(), e2.Simplify()));
}

TriBool SymbolicUtils::StaticCheckLt(const Expression &e1, const Expression &e2) {
  return StaticCheckBool(sym::Lt(e1.Simplify(), e2.Simplify()));
}

TriBool SymbolicUtils::StaticCheckLe(const Expression &e1, const Expression &e2) {
  return StaticCheckBool(sym::Le(e1.Simplify(), e2.Simplify()));
}

TriBool SymbolicUtils::StaticCheckGt(const Expression &e1, const Expression &e2) {
  return StaticCheckBool(sym::Gt(e1.Simplify(), e2.Simplify()));
}

TriBool SymbolicUtils::StaticCheckGe(const Expression &e1, const Expression &e2) {
  return StaticCheckBool(sym::Ge(e1.Simplify(), e2.Simplify()));
}

std::string SymbolicUtils::AsNumerDenomToString(const Expression &x) {
  Expression numer;
  Expression denom;
  x.AsNumerDenom(numer, denom);
  auto numer_str = ToString(numer);
  auto denom_str = ToString(denom);
  if (numer_str == kInvalidExpr || denom_str == kInvalidExpr) {
    return kInvalidExpr;
  }
  return "(" + numer_str + ")" + "/" + "(" + denom_str + ")";
}

TriBool SymbolicUtils::StaticCheckBool(const Expression &expr) {
  GE_ASSERT_TRUE(expr.IsBooleanExpr(), "Only boolean expr can do static check, expr: %s", expr.Str().get());
  bool value = false;
  if (expr.IsConstExpr()) {
    GE_ASSERT_TRUE(expr.GetConstValue(value));
    return value ? TriBool::kTrue : TriBool::kFalse;
  }
  if (GetCurShapeEnvContext() == nullptr) {
    GELOGW("Shape env is nullptr, cannot do static check, expr: %s", expr.Str().get());
    return TriBool::kUnknown;
  }
  if (GetCurShapeEnvContext()->HasSymbolInfo(expr) == TriBool::kTrue) {
    GELOGI("Find check info of expr: %s, no need simplify guard", SymbolicUtils::ToString(expr).c_str());
    return TriBool::kTrue;
  }
  const auto simplify_expr = expr.Simplify();
  value = false;
  // 化简后判断是否是常量
  if (simplify_expr.IsConstExpr()) {
    GE_ASSERT_TRUE(simplify_expr.GetConstValue(value));
    return value ? TriBool::kTrue : TriBool::kFalse;
  }
  return GetCurShapeEnvContext()->HasSymbolInfo(simplify_expr);
}
}  // namespace af
