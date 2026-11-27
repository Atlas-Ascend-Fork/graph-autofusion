/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_TESTS_V35_ATT_GEN_MODEL_INFO_API_PERF_REGISTER_PERF_SCENARIO_UTILS_H_
#define AUTOFUSE_TESTS_V35_ATT_GEN_MODEL_INFO_API_PERF_REGISTER_PERF_SCENARIO_UTILS_H_

#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "api_perf_register/api_perf_factory.h"
#include "base/att_const_values.h"
#include "v35/att/api_perf_register/ascendc_regbase_perf.h"

namespace att {
namespace perf_scenario {
using namespace af::sym;

// LoadV2 引入单核分支后，pipe_res[AIV_MTE2]为外层load_core_case_node三元变量（block_dim==1选单核参数）。
// 用例通过本函数取指定分支的表达式后再做公式断言；取分支失败时记录失败并返回哑值，避免解引用UB。
inline Expr SelectLoadCoreBranch(const std::map<Expr, TernaryOp, ExprCmp> &ternary_ops, bool single_core) {
  const Expr var = CreateExpr("load_core_case_node");
  const auto iter = ternary_ops.find(var);
  if (iter == ternary_ops.end()) {
    ADD_FAILURE() << "load_core_case_node not found in ternary_ops";
    return CreateExpr(0);
  }
  const auto if_case = iter->second.DeepCopyIfCase();
  if (if_case == nullptr) {
    ADD_FAILURE() << "DeepCopyIfCase of load_core_case_node returned nullptr";
    return CreateExpr(0);
  }
  const auto branch = single_core ? if_case->GetChoiceA() : if_case->GetChoiceB();
  if (branch == nullptr) {
    ADD_FAILURE() << "branch of load_core_case_node is nullptr, single_core=" << single_core;
    return CreateExpr(0);
  }
  return branch->GetExpr();
}

struct TransposePerfResult {
  std::string pipe_expr;
  std::string ternary_expr;
  PerfOutputInfo perf;
};

inline TransposePerfResult GetTransposePerfExpr(uint32_t total_dim, uint32_t inner_dim,
                                                const std::vector<Expr> &output_dims,
                                                const std::vector<Expr> &outer_loop_axes,
                                                const std::vector<Expr> &input_strides, bool valid = true,
                                                const std::string &dtype = kFloat16,
                                                const std::vector<Expr> &output_strides = {}) {
  auto api = ApiPerfFactory::Instance().Create("TransposeV2");
  EXPECT_NE(api, nullptr);
  NodeDetail node_info;
  node_info.input_dtype = {dtype};
  node_info.input_dims = output_dims;
  node_info.transpose_node_params.valid = valid;
  node_info.transpose_node_params.total_dim = total_dim;
  node_info.transpose_node_params.inner_dim = inner_dim;
  node_info.transpose_node_params.output_dims = output_dims;
  node_info.transpose_node_params.outer_loop_axes = outer_loop_axes;
  node_info.transpose_node_params.input_strides = input_strides;
  node_info.transpose_node_params.output_strides = output_strides;
  PerfOutputInfo perf_res;
  EXPECT_EQ(ascendcperf_v2::TransposePerf(node_info, perf_res), af::SUCCESS);

  std::string ternary_expr;
  if (!perf_res.ternary_ops.empty()) {
    const auto &ternary = perf_res.ternary_ops.begin()->second;
    ternary_expr = ternary.GetTernaryOpStr();
  }
  return {Str(perf_res.pipe_res[PipeType::AIV_VEC]), ternary_expr, perf_res};
}

inline bool GetNumericConstant(Expr value, double &result) {
  value.Simplify();
  if (value.GetExprType() == af::ExprType::kExprConstantInteger) {
    int64_t integer = 0;
    if (!value.GetConstValue(integer)) {
      return false;
    }
    result = static_cast<double>(integer);
    return true;
  }
  return value.IsConstExpr() && value.GetConstValue(result);
}

inline double EvaluatePipeCycles(const PerfOutputInfo &perf, PipeType pipe,
                                 std::vector<std::pair<Expr, Expr>> bindings) {
  const Expr &cycles = perf.pipe_res.at(pipe);
  for (size_t pass = 0U; pass <= perf.ternary_ops.size(); ++pass) {
    for (const auto &item : perf.ternary_ops) {
      auto choice = item.second.DeepCopyIfCase();
      choice->Replace(bindings);
      bool resolved = true;
      while (!choice->IsLeaf()) {
        const auto &left = choice->GetCondLeft();
        const auto &right = choice->GetCondRight();
        auto taken = af::SymbolicUtils::StaticCheckEq(left, right);
        if (choice->GetCondType() == CondType::K_GE) {
          taken = af::SymbolicUtils::StaticCheckLe(right, left);
        } else if (choice->GetCondType() == CondType::K_LT) {
          taken = af::SymbolicUtils::StaticCheckLt(left, right);
        }
        if (taken == af::TriBool::kUnknown) {
          resolved = false;
          break;
        }
        choice = taken == af::TriBool::kTrue ? choice->GetChoiceA() : choice->GetChoiceB();
      }
      double value = 0.0;
      if (resolved && GetNumericConstant(choice->GetExpr(), value)) {
        bindings.emplace_back(item.first, CreateExpr(value));
      }
    }
    Expr current = cycles.Replace(bindings);
    current.Simplify();
    double value = 0.0;
    if (GetNumericConstant(current, value)) {
      return value;
    }
  }
  ADD_FAILURE() << "Unresolved pipeline cost: " << Str(cycles);
  return 0.0;
}

inline double EvaluateTransposeCycles(const TransposePerfResult &result,
                                      std::vector<std::pair<Expr, Expr>> bindings = {}) {
  return EvaluatePipeCycles(result.perf, PipeType::AIV_VEC, std::move(bindings));
}

}  // namespace perf_scenario
}  // namespace att

#endif  // AUTOFUSE_TESTS_V35_ATT_GEN_MODEL_INFO_API_PERF_REGISTER_PERF_SCENARIO_UTILS_H_
