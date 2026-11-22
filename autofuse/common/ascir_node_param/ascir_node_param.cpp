/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir_node_param/ascir_node_param.h"

#include <algorithm>

#include "common/checker.h"
#include "symbolizer/symbolic_utils.h"

namespace ascir_param {
namespace {
constexpr const char *kAscirNodeParams = "AscirNodeParams";

bool HasOnlySemanticRole(const ParamExprProduct &expr) {
  return std::all_of(expr.factors.begin(), expr.factors.end(),
                     [](const ParamExprLeaf &factor) { return factor.role == ParamExprRole::kSemantic; });
}

bool StaticCheckExprNe(const ge::Expression &lhs, const ge::Expression &rhs) {
  return af::SymbolicUtils::StaticCheckEq(lhs, rhs) == af::TriBool::kFalse;
}

af::Status ValidateParamExprProduct(const ParamExprProduct &expr, const ge::Expression &canonical_expr,
                                    const char *param_name) {
  GE_ASSERT_TRUE(expr.valid, "Reduce %s expr is invalid.", param_name);
  GE_ASSERT_TRUE(!expr.factors.empty(), "Reduce %s expr is empty.", param_name);
  for (const auto &factor : expr.factors) {
    GE_ASSERT_TRUE(factor.expr.IsValid(), "Reduce %s expr factor is invalid.", param_name);
  }
  if (!HasOnlySemanticRole(expr)) {
    return af::SUCCESS;
  }
  const auto resolved_expr = ResolveForAtt(expr);
  GE_ASSERT_TRUE(!StaticCheckExprNe(resolved_expr, canonical_expr),
                 "Reduce %s expr does not match canonical params, expr[%s], canonical[%s].", param_name,
                 resolved_expr.Str().get(), canonical_expr.Str().get());
  return af::SUCCESS;
}
}  // namespace

AscirNodeParamsPtr GetAscirNodeParams(af::AscNodePtr node) {
  if (node == nullptr) {
    return nullptr;
  }
  auto op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    return nullptr;
  }
  return op_desc->TryGetExtAttr(kAscirNodeParams, AscirNodeParamsPtr{});
}

AscirNodeParamsPtr GetOrCreateAscirNodeParams(af::AscNodePtr node) {
  GE_ASSERT_NOTNULL(node);
  auto params = GetAscirNodeParams(node);
  if (params != nullptr) {
    return params;
  }

  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  params = std::make_shared<AscirNodeParams>();
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kAscirNodeParams, params), "Node:%s SetExtAttr failed", node->GetNamePtr());
  return params;
}

const codegen::ReduceSpecificParams &GetCanonicalReduceParams(const ReduceNodeParams &params) {
  return params.canonical_params;
}

void RecordCacheExecution(AscirNodeParams &params, const CacheExecutionParams &execution) {
  params.cache_executions.push_back(execution);
}

ge::Expression ResolveForAtt(const ParamExprProduct &expr) {
  ge::Expression value = ge::Symbol(1U);
  if (!expr.valid) {
    return value;
  }
  for (const auto &factor : expr.factors) {
    value = value * factor.expr;
  }
  return value;
}

af::Status ValidateReduceNodeParams(const ReduceNodeParams &params) {
  GE_ASSERT_TRUE(params.canonical_params.valid, "Reduce canonical params are invalid.");
  GE_ASSERT_SUCCESS(
      ValidateParamExprProduct(params.exprs.merge_size, params.canonical_params.merge_size, "merge size"));
  GE_ASSERT_SUCCESS(
      ValidateParamExprProduct(params.exprs.merge_times, params.canonical_params.merge_times, "merge times"));
  return af::SUCCESS;
}
}  // namespace ascir_param
