/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "specific_params_builder.h"

#include <array>

#include "ascir_node_param/ascir_node_param.h"
#include "ascir_node_param/ascir_param_builder.h"
#include "base/att_const_values.h"
#include "common/checker.h"
#include "arg_reduce_utils.h"

namespace att {
namespace {
af::Status FillReduceParams(const ascir_param::AscirNodeParams &params, NodeInfo &node_info) {
  const auto *reduce = ascir_param::GetSpecificParams<ascir_param::ReduceNodeParams>(params);
  GE_ASSERT_NOTNULL(reduce, "Reduce specific params are null, node[%s].", node_info.name.c_str());
  node_info.reduce_specific_params = *reduce;
  return af::SUCCESS;
}

af::Status FillVectorFuncParams(const ascir_param::AscirNodeParams &params, NodeInfo &node_info) {
  const auto *vector_func_params = ascir_param::GetSpecificParams<ascir_param::VectorFuncNodeParams>(params);
  GE_ASSERT_NOTNULL(vector_func_params, "VectorFunc specific params are null, node[%s].", node_info.name.c_str());
  node_info.vector_func_params = *vector_func_params;
  return af::SUCCESS;
}

af::Status FillCastParams(const ascir_param::AscirNodeParams &params, NodeInfo &node_info) {
  const auto *cast_params = ascir_param::GetSpecificParams<ascir_param::CastNodeParams>(params);
  GE_ASSERT_NOTNULL(cast_params, "Cast specific params are null, node[%s].", node_info.name.c_str());
  node_info.cast_node_params = *cast_params;
  return af::SUCCESS;
}

af::Status FillBroadcastParams(const ascir_param::AscirNodeParams &params, NodeInfo &node_info) {
  const auto *broadcast_params = ascir_param::GetSpecificParams<ascir_param::BroadcastNodeParams>(params);
  GE_ASSERT_NOTNULL(broadcast_params, "Broadcast specific params is null, node[%s].", node_info.name.c_str());
  node_info.broadcast_node_params = *broadcast_params;
  return af::SUCCESS;
}

af::Status FillCompareParams(const ascir_param::AscirNodeParams &params, NodeInfo &node_info) {
  const auto *compare_params = ascir_param::GetSpecificParams<ascir_param::CompareNodeParams>(params);
  GE_ASSERT_NOTNULL(compare_params, "Compare specific params are null, node[%s].", node_info.name.c_str());
  node_info.compare_node_params = *compare_params;
  return af::SUCCESS;
}

af::Status FillWhereParams(const ascir_param::AscirNodeParams &params, NodeInfo &node_info) {
  const auto *where_params = ascir_param::GetSpecificParams<ascir_param::WhereNodeParams>(params);
  GE_ASSERT_NOTNULL(where_params, "Where specific params are null, node[%s].", node_info.name.c_str());
  node_info.where_node_params = *where_params;
  return af::SUCCESS;
}

af::Status FillUnaryBitWidthChangeParams(const ascir_param::AscirNodeParams &params, NodeInfo &node_info) {
  const auto *unary_params = ascir_param::GetSpecificParams<ascir_param::UnaryBitWidthChangeNodeParams>(params);
  GE_ASSERT_NOTNULL(unary_params, "Unary bitwidth change specific params are null, node[%s].", node_info.name.c_str());
  node_info.unary_bitwidth_change_node_params = *unary_params;
  return af::SUCCESS;
}

af::Status FillTransposeParams(const ascir_param::AscirNodeParams &params, NodeInfo &node_info) {
  const auto *transpose_params = ascir_param::GetSpecificParams<ascir_param::TransposeNodeParams>(params);
  GE_ASSERT_NOTNULL(transpose_params, "Transpose specific params are null, node[%s].", node_info.name.c_str());
  node_info.transpose_node_params = *transpose_params;
  return af::SUCCESS;
}

af::Status FillArgReduceParams(const ascir_param::AscirNodeParams &params, NodeInfo &node_info) {
  const auto *argmax = ascir_param::GetSpecificParams<ascir_param::ArgReduceNodeParams>(params);
  GE_ASSERT_NOTNULL(argmax, "Arg-reduce specific params are null, node[%s].", node_info.name.c_str());
  node_info.argmax_node_params = *argmax;
  return af::SUCCESS;
}

bool IsVectorFuncNodeType(const std::string &node_type) {
  return node_type == kVectorFunc;
}

bool IsCastNodeType(const std::string &node_type) {
  return node_type == kCast || node_type == kCeil2Int;
}

bool IsBroadcastNodeType(const std::string &node_type) {
  return node_type == kBroadcast;
}

bool IsCompareNodeType(const std::string &node_type) {
  return node_type == kGe || node_type == kEq || node_type == kNe || node_type == kGt || node_type == kLe ||
         node_type == kLt;
}

bool IsWhereNodeType(const std::string &node_type) {
  return node_type == kWhere || node_type == kSelect;
}

bool IsUnaryBitWidthChangeNodeType(const std::string &node_type) {
  return node_type == kIsnan || node_type == kIsFinite || node_type == kIsInf;
}

bool IsTransposeNodeType(const std::string &node_type) {
  return node_type == kTranspose;
}

using FillParamsFunc = af::Status (*)(const ascir_param::AscirNodeParams &params, NodeInfo &node_info);

struct SpecificParamsFiller {
  bool (*matches)(const std::string &node_type);
  FillParamsFunc fill;
};

constexpr std::array<SpecificParamsFiller, 9U> kSpecificParamsFillers = {{
    {ascgen_utils::IsArgReduceType, FillArgReduceParams},
    {ascir_param::IsReduceParamSupported, FillReduceParams},
    {IsVectorFuncNodeType, FillVectorFuncParams},
    {IsCastNodeType, FillCastParams},
    {IsBroadcastNodeType, FillBroadcastParams},
    {IsCompareNodeType, FillCompareParams},
    {IsWhereNodeType, FillWhereParams},
    {IsUnaryBitWidthChangeNodeType, FillUnaryBitWidthChangeParams},
    {IsTransposeNodeType, FillTransposeParams},
}};

}  // namespace

af::Status FillSpecificParams(const af::AscNodePtr &ge_node, NodeInfo &node_info) {
  const auto params = ascir_param::GetAscirNodeParams(ge_node);
  if (params == nullptr) {
    return af::SUCCESS;
  }
  node_info.cache_executions = params->cache_executions;
  if (params->status == ascir_param::ParamBuildStatus::kSkipped) {
    return af::SUCCESS;
  }
  GE_ASSERT_TRUE(params->status == ascir_param::ParamBuildStatus::kBuilt, "Build ascir node params failed, node[%s].",
                 node_info.name.c_str());
  for (const auto &filler : kSpecificParamsFillers) {
    if (filler.matches(node_info.node_type)) {
      return filler.fill(*params, node_info);
    }
  }
  return af::SUCCESS;
}
}  // namespace att
