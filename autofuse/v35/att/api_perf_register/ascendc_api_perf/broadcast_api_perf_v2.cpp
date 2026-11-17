/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "broadcast_api_perf_v2.h"

#include "base/att_const_values.h"
#include "common/checker.h"
#include "broadcast_perf_utils_v2.h"
#include "broadcast_last_axis_perf_v2.h"
#include "broadcast_nlast_axis_perf_v2.h"
#include "api_perf_register/ascendc_api_perf.h"

namespace att {
namespace ascendcperf_v2 {
namespace {
using ascendcapi_v2::BroadcastTilingInfo;
using ascendcapi_v2::ParamExprInputs;
using ascendcapi_v2::VfCostAccumulator;

ParamExprInputs MakeParamInputs(const std::vector<Expr> &dims, const std::vector<Expr> &actual_dims) {
  ParamExprInputs inputs;
  inputs.semantic = dims;
  inputs.size = dims;
  inputs.actual_size = actual_dims.size() == dims.size() ? actual_dims : dims;
  return inputs;
}

af::Status SetBroadcastCost(const VfCostAccumulator &acc, PerfOutputInfo &perf) {
  perf.pipe_res[PipeType::AIV_VEC] = ascendcapi_v2::GetVfCost(acc);
  return af::SUCCESS;
}

af::Status BuildScalarPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GE_ASSERT_TRUE(!node_info.input_dtype.empty(), "Broadcast scalar input dtype is missing.");
  std::string helper_dtype;
  GE_ASSERT_SUCCESS(ascendcapi_v2::GetEffectiveHelperDtype(node_info.input_dtype[0], helper_dtype));
  Expr vl;
  Expr half_vl;
  Expr block_elements;
  GE_ASSERT_SUCCESS(ascendcapi_v2::GetBroadcastVectorElements(node_info.input_dtype[0], vl, half_vl, block_elements));
  (void)half_vl;
  (void)block_elements;
  Expr repeat_time;
  GE_ASSERT_SUCCESS(ascendcapi_v2::CeilDiv(node_info.broadcast_node_params.duplicate_count, vl, repeat_time));
  VfCostAccumulator acc;
  GE_ASSERT_SUCCESS(ascendcapi_v2::AddVfInstructPerf(kDuplicate, helper_dtype, repeat_time, 1U, acc));
  return SetBroadcastCost(acc, perf);
}

af::Status BuildEqualSizePerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GE_ASSERT_TRUE(!node_info.input_dtype.empty(), "Broadcast equal-size input dtype is missing.");
  VfCostAccumulator acc;
  GE_ASSERT_SUCCESS(ascendcapi_v2::AddBroadcastDataCopyPerf(node_info.input_dtype[0], CreateExpr(1), acc));
  return SetBroadcastCost(acc, perf);
}

af::Status BuildSingleElementPerf(const NodeDetail &node_info, const Expr &dst_size, PerfOutputInfo &perf) {
  GE_ASSERT_TRUE(!node_info.input_dtype.empty(), "Broadcast duplicate input dtype is missing.");
  std::string helper_dtype;
  GE_ASSERT_SUCCESS(ascendcapi_v2::GetEffectiveHelperDtype(node_info.input_dtype[0], helper_dtype));
  Expr vl;
  Expr half_vl;
  Expr block_elements;
  GE_ASSERT_SUCCESS(ascendcapi_v2::GetBroadcastVectorElements(node_info.input_dtype[0], vl, half_vl, block_elements));
  (void)half_vl;
  (void)block_elements;
  Expr repeat_time;
  GE_ASSERT_SUCCESS(ascendcapi_v2::CeilDiv(dst_size, vl, repeat_time));
  VfCostAccumulator acc;
  GE_ASSERT_SUCCESS(ascendcapi_v2::AddVfInstructPerf(kLoad, node_info.input_dtype[0], CreateExpr(1), 1U, acc));
  GE_ASSERT_SUCCESS(ascendcapi_v2::AddVfInstructPerf(kUpdateMask, helper_dtype, repeat_time, 1U, acc));
  GE_ASSERT_SUCCESS(ascendcapi_v2::AddVfInstructPerf(kStore, node_info.output_dtype[0], repeat_time, 1U, acc));
  return SetBroadcastCost(acc, perf);
}

af::Status ValidateBroadcastNodeDetail(const NodeDetail &node_info) {
  const auto &params = node_info.broadcast_node_params;
  GE_ASSERT_TRUE(!node_info.input_dtype.empty() && !node_info.output_dtype.empty(),
                 "Broadcast dtype information is missing.");
  if (params.is_scalar) {
    return af::SUCCESS;
  }
  GE_ASSERT_TRUE(!node_info.input_dims.empty() && node_info.input_dims.size() == node_info.output_dims.size(),
                 "Broadcast shape ranks are invalid.");
  GE_ASSERT_TRUE(
      params.src_shape.size() == params.dst_shape.size() && params.src_shape.size() == node_info.input_dims.size(),
      "Broadcast parameter shape rank is invalid.");
  GE_ASSERT_TRUE(node_info.repeats.size() == node_info.input_dims.size(), "Broadcast input repeats length is invalid.");
  return af::SUCCESS;
}

Expr ResolveBroadcastSize(const std::vector<ascir_param::ParamExprLeaf> &shape, const ParamExprInputs &inputs) {
  std::vector<Expr> resolved;
  for (size_t i = 0U; i < shape.size(); ++i) {
    resolved.push_back(
        ascendcapi_v2::ResolveParamExprLeaf(shape[i], inputs.semantic[i], inputs.size[i], inputs.actual_size[i]));
  }
  return ascendcapi_v2::ShapeProduct(resolved);
}
}  // namespace

bool IsBroadcastFallback(const NodeDetail &node_info) {
  if (node_info.input_dtype.empty() || node_info.input_dims.empty() || node_info.output_dims.empty()) {
    return false;
  }
  const auto &params = node_info.broadcast_node_params;
  const auto src_inputs = MakeParamInputs(node_info.input_dims, node_info.repeats);
  const auto dst_inputs = MakeParamInputs(node_info.output_dims, node_info.output_dims);
  BroadcastTilingInfo tiling;
  if (BuildBroadcastTiling(params.src_shape, params.dst_shape, node_info.input_dtype[0], src_inputs, dst_inputs,
                           tiling) != af::SUCCESS) {
    return false;
  }
  const Expr src_size = ResolveBroadcastSize(params.src_shape, src_inputs);
  const Expr dst_size = ResolveBroadcastSize(params.dst_shape, dst_inputs);
  if (src_size == dst_size || src_size == CreateExpr(1)) {
    return false;
  }
  if (IsLastAxisBroadcast(tiling)) {
    return GetLastAxisBranch(tiling, node_info.input_dtype[0], params.const_rank) ==
           ascendcapi_v2::LastAxisBranch::kFallback;
  }
  if (params.const_rank == -1 && ascendcperf_v2::HasUnknownNlastCondition(node_info, tiling)) {
    return false;
  }
  if (tiling.original_rank > 4U) {
    return false;
  }
  return GetNlastAxisBranch(tiling, node_info.input_dtype[0], params.const_rank) ==
         ascendcapi_v2::NlastAxisBranch::kFallback;
}

af::Status BroadcastPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GE_ASSERT_TRUE(node_info.broadcast_node_params.valid, "Broadcast parameters are invalid.");
  GE_ASSERT_SUCCESS(ValidateBroadcastNodeDetail(node_info));
  const auto &params = node_info.broadcast_node_params;
  if (params.is_scalar) {
    // Scalar shape metadata uses a footprint instead of per-axis expressions.
    // Rank one records the BroadcastExtend API, whose source is read by a vector load.
    if (params.const_rank == 1) {
      return BuildSingleElementPerf(node_info, params.duplicate_count, perf);
    }
    return BuildScalarPerf(node_info, perf);
  }
  GE_ASSERT_TRUE(!node_info.input_dtype.empty() && !node_info.output_dtype.empty(),
                 "Broadcast dtype information is missing.");
  const auto src_inputs = MakeParamInputs(node_info.input_dims, node_info.repeats);
  const auto dst_inputs = MakeParamInputs(node_info.output_dims, node_info.output_dims);
  const Expr raw_src_size = ResolveBroadcastSize(params.src_shape, src_inputs);
  const Expr raw_dst_size = ResolveBroadcastSize(params.dst_shape, dst_inputs);
  BroadcastTilingInfo tiling;
  GE_ASSERT_SUCCESS(ascendcapi_v2::BuildBroadcastTiling(params.src_shape, params.dst_shape, node_info.input_dtype[0],
                                                        src_inputs, dst_inputs, tiling));
  if (raw_src_size == raw_dst_size) {
    return BuildEqualSizePerf(node_info, perf);
  }
  if (raw_src_size == CreateExpr(1)) {
    return BuildSingleElementPerf(node_info, raw_dst_size, perf);
  }
  if (ascendcapi_v2::IsLastAxisBroadcast(tiling)) {
    const auto branch =
        ascendcapi_v2::GetLastAxisBranch(tiling, node_info.input_dtype[0], node_info.broadcast_node_params.const_rank);
    if (branch == ascendcapi_v2::LastAxisBranch::kFallback && params.const_rank != -1) {
      return af::FAILED;
    }
    return BuildLastAxisPerf(node_info, tiling, perf);
  }
  const auto branch =
      ascendcapi_v2::GetNlastAxisBranch(tiling, node_info.input_dtype[0], node_info.broadcast_node_params.const_rank);
  if (branch == ascendcapi_v2::NlastAxisBranch::kFallback && params.const_rank != -1 && tiling.original_rank <= 4U) {
    return af::FAILED;
  }
  return BuildNlastAxisPerf(node_info, tiling, perf);
}

}  // namespace ascendcperf_v2
}  // namespace att
