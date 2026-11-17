/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "v35/att/api_perf_register/ascir_reduce_api_perf_v2.h"

#include <memory>
#include "ascir_node_param/ascir_node_param.h"
#include "codegen_api_param/codegen_api_param.h"
#include "v35/att/api_perf_register/ascendc_api_perf/reduce_api_perf_v2.h"
#include "v35/att/api_perf_register/ascendc_regbase_perf.h"

namespace att {
namespace ascir_reduce_v2 {
namespace {
void SetReduceDims(const Expr &first, const Expr &last, ascendcapi_v2::ReducePattern pattern, NodeDetail &node_detail) {
  node_detail.input_dims = {first, last};
  node_detail.output_dims = {pattern == ascendcapi_v2::ReducePattern::kRA ? last : first};
}

void MergeReduceDimsByLastAxis(ascendcapi_v2::ReducePattern pattern, NodeDetail &node_detail) {
  auto &in_dims = node_detail.input_dims;
  if (in_dims.size() <= 2U) {
    return;
  }
  Expr first = CreateExpr(1);
  for (size_t i = 0U; i < in_dims.size() - 1U; ++i) {
    first = first * in_dims[i];
  }
  GELOGD("[ATT Reduce] fallback merged dims by last axis: node[%s], pattern[%s], first[%s], last[%s].",
         node_detail.name.c_str(), pattern == ascendcapi_v2::ReducePattern::kAR ? "AR" : "RA", first.Str().get(),
         in_dims.back().Str().get());
  SetReduceDims(first, in_dims.back(), pattern, node_detail);
}

Expr ProductExpressions(const std::vector<Expr> &values) {
  Expr product = CreateExpr(1);
  for (const auto &value : values) {
    product = product * value;
  }
  return product;
}

Expr BuildSemanticMergeSizeForAtt(const TensorShapeInfo &output_shape) {
  return ProductExpressions(output_shape.repeats.empty() ? output_shape.dims : output_shape.repeats);
}

af::Status TryGetReuseSource(const NodeInfo &node, bool &is_reuse_source) {
  const auto &params = ascir_param::GetCanonicalReduceParams(node.reduce_specific_params);
  if (params.reuse.valid) {
    is_reuse_source = params.reuse.is_reuse_source;
    return af::SUCCESS;
  }
  GELOGW("[ATT Reduce] reuse-source fallback by input fanout, node[%s].", node.name.c_str());
  GE_ASSERT_NOTNULL(node.node_ptr, "Reduce reuse-source: node_ptr is null, node[%s].", node.name.c_str());
  auto node_in_anchor = node.node_ptr->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(node_in_anchor, "Reduce reuse-source: in_anchor is null, node[%s].", node.name.c_str());
  auto peer_out_anchor = node_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(peer_out_anchor, "Reduce reuse-source: peer_out_anchor is null, node[%s].", node.name.c_str());
  const auto &in_node = std::dynamic_pointer_cast<ge::AscNode>(peer_out_anchor->GetOwnerNode());
  GE_ASSERT_NOTNULL(in_node, "Reduce reuse-source: in_node is null, node[%s].", node.name.c_str());
  is_reuse_source = in_node->GetOutAllNodes().size() == 1UL;
  GELOGD("[ATT Reduce] reuse-source: is_reuse[%d], node[%s].", static_cast<int32_t>(is_reuse_source),
         node.name.c_str());
  return af::SUCCESS;
}

af::Status ConvertReducePattern(codegen::ReducePattern pattern, ascendcapi_v2::ReducePattern &perf_pattern) {
  switch (pattern) {
    case codegen::ReducePattern::kAR:
      perf_pattern = ascendcapi_v2::ReducePattern::kAR;
      return af::SUCCESS;
    case codegen::ReducePattern::kRA:
      perf_pattern = ascendcapi_v2::ReducePattern::kRA;
      return af::SUCCESS;
    default:
      GE_ASSERT_TRUE(false, "Reduce pattern is unknown.");
  }
}

af::Status ConvertReduceMergeMode(codegen::ReduceMergeMode mode, ascendcapi_v2::ReduceMergeMode &perf_mode) {
  switch (mode) {
    case codegen::ReduceMergeMode::kNone:
      perf_mode = ascendcapi_v2::ReduceMergeMode::kNone;
      return af::SUCCESS;
    case codegen::ReduceMergeMode::kCopy:
      perf_mode = ascendcapi_v2::ReduceMergeMode::kCopy;
      return af::SUCCESS;
    case codegen::ReduceMergeMode::kMergeByElementwise:
      perf_mode = ascendcapi_v2::ReduceMergeMode::kMergeByElementwise;
      return af::SUCCESS;
    default:
      GE_ASSERT_TRUE(false, "Reduce merge mode is unknown.");
  }
}

bool IsMultiReduceMode(ascendcapi_v2::ReduceMergeMode mode) {
  return mode != ascendcapi_v2::ReduceMergeMode::kNone;
}

af::Status BuildCurrentShapeParams(const std::vector<TensorShapeInfo> &input_shapes,
                                   const std::vector<TensorShapeInfo> &output_shapes,
                                   const codegen::ReduceSpecificParams &base_params,
                                   ascendcapi_v2::ReduceMergeMode merge_mode,
                                   codegen::ReduceSpecificParams &shape_params) {
  codegen::ReduceSpecificParamBuildInput input;
  input.reduce_type = base_params.reduce_type;
  input.input_repeats = input_shapes[0].repeats;
  input.input_strides = input_shapes[0].strides;
  input.output_dims = output_shapes[0].dims;
  input.output_strides = output_shapes[0].strides;
  input.dtype_size = input_shapes[0].data_type_size;
  input.pattern = base_params.pattern;
  input.need_multi_reduce = IsMultiReduceMode(merge_mode);
  input.merge_times = base_params.merge_times.IsValid() ? base_params.merge_times : CreateExpr(1);
  input.reuse = base_params.reuse;
  return codegen::BuildReduceSpecificParams(input, shape_params);
}

af::Status BuildReduceContext(const std::vector<TensorShapeInfo> &input_shapes,
                              const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                              ascendcapi_v2::ReduceApiPerfContext &context) {
  const auto &params = node.reduce_specific_params;
  GE_ASSERT_SUCCESS(ascir_param::ValidateReduceNodeParams(params), "Reduce specific params are invalid, node[%s].",
                    node.name.c_str());
  const auto &codegen_params = ascir_param::GetCanonicalReduceParams(params);
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, context.node_detail));
  GE_ASSERT_TRUE(!context.node_detail.input_dims.empty(), "Reduce input dims are empty, node[%s].", node.name.c_str());
  GE_ASSERT_SUCCESS(ConvertReducePattern(codegen_params.pattern, context.pattern));
  GE_ASSERT_SUCCESS(ConvertReduceMergeMode(codegen_params.merge_mode, context.merge_mode));
  codegen::ReduceSpecificParams current_shape_params;
  GE_ASSERT_SUCCESS(
      BuildCurrentShapeParams(input_shapes, output_shapes, codegen_params, context.merge_mode, current_shape_params));
  if (current_shape_params.merged_dims.valid) {
    SetReduceDims(current_shape_params.merged_dims.first, current_shape_params.merged_dims.last, context.pattern,
                  context.node_detail);
  } else {
    MergeReduceDimsByLastAxis(context.pattern, context.node_detail);
  }
  GE_ASSERT_TRUE(context.node_detail.input_dims.size() == 2U, "Reduce dims must be {first,last}, node[%s].",
                 node.name.c_str());
  GE_ASSERT_SUCCESS(TryGetReuseSource(node, context.is_reuse_source),
                    "Reduce reuse-source branch is unknown, node[%s].", node.name.c_str());
  context.merge_size =
      params.exprs.merge_size.valid ? BuildSemanticMergeSizeForAtt(output_shapes[0]) : current_shape_params.merge_size;
  context.merge_times = ascir_param::ResolveForAtt(params.exprs.merge_times);
  GELOGD(
      "[ATT Reduce] BuildReduceContext: node[%s], pattern[%s], merge_mode[%d], reuse[%d], "
      "merge_size[%s], merge_times[%s].",
      node.name.c_str(), context.pattern == ascendcapi_v2::ReducePattern::kAR ? "AR" : "RA",
      static_cast<int32_t>(context.merge_mode), static_cast<int32_t>(context.is_reuse_source),
      context.merge_size.Str().get(), context.merge_times.Str().get());
  return af::SUCCESS;
}

bool HasReduceSpecificParams(const NodeInfo &node) {
  return node.reduce_specific_params.canonical_params.valid;
}

af::Status SkipUnsupportedReducePerf(const NodeInfo &node, const ascendcapi_v2::ReduceApiPerfContext &context,
                                     const std::string &op_name, af::Status status) {
  GELOGW("[ATT Reduce] Skip unsupported reduce perf, op[%s], dtype[%s], status[%u], node[%s].", op_name.c_str(),
         context.node_detail.input_dtype[0].c_str(), status, node.name.c_str());
  return af::SUCCESS;
}

af::Status CallReducePerf(const NodeInfo &node, const ascendcapi_v2::ReduceApiPerfContext &context,
                          const std::string &op_name,
                          af::Status (*perf_func)(const ascendcapi_v2::ReduceApiPerfContext &, PerfOutputInfo &),
                          PerfOutputInfo &perf_res) {
  const af::Status status = perf_func(context, perf_res);
  if (status != af::SUCCESS) {
    return SkipUnsupportedReducePerf(node, context, op_name, status);
  }
  return af::SUCCESS;
}
}  // namespace

af::Status ElementwiseMaxApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                             [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                             [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::MaxPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status ElementwiseMinApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                             [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                             [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::MinPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status ElementwiseSumApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                             [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                             [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::SumPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status ElementwiseMeanApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                              [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                              [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::MeanPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status ElementwiseProdApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                              [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                              [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::MulPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status ReduceMaxApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  ascendcapi_v2::ReduceApiPerfContext context;
  GE_ASSERT_SUCCESS(BuildReduceContext(input_shapes, output_shapes, node, context));
  GE_ASSERT_SUCCESS(ascendcapi_v2::ReduceMaxPerf(context, perf_res));
  return af::SUCCESS;
}

af::Status ReduceMinApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  ascendcapi_v2::ReduceApiPerfContext context;
  GE_ASSERT_SUCCESS(BuildReduceContext(input_shapes, output_shapes, node, context));
  GE_ASSERT_SUCCESS(ascendcapi_v2::ReduceMinPerf(context, perf_res));
  return af::SUCCESS;
}

af::Status ReduceAnyApi(const std::vector<TensorShapeInfo> &input_shapes,
                        const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                        PerfOutputInfo &perf_res) {
  ascendcapi_v2::ReduceApiPerfContext context;
  GE_ASSERT_SUCCESS(BuildReduceContext(input_shapes, output_shapes, node, context));
  GE_ASSERT_SUCCESS(ascendcapi_v2::ReduceAnyPerf(context, perf_res));
  return af::SUCCESS;
}

af::Status ReduceAllApi(const std::vector<TensorShapeInfo> &input_shapes,
                        const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                        PerfOutputInfo &perf_res) {
  ascendcapi_v2::ReduceApiPerfContext context;
  GE_ASSERT_SUCCESS(BuildReduceContext(input_shapes, output_shapes, node, context));
  GE_ASSERT_SUCCESS(ascendcapi_v2::ReduceAllPerf(context, perf_res));
  return af::SUCCESS;
}

af::Status ReduceSumApi(const std::vector<TensorShapeInfo> &input_shapes,
                        const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                        PerfOutputInfo &perf_res) {
  ascendcapi_v2::ReduceApiPerfContext context;
  GE_ASSERT_SUCCESS(BuildReduceContext(input_shapes, output_shapes, node, context));
  return CallReducePerf(node, context, kReduceSum, ascendcapi_v2::ReduceSumPerf, perf_res);
}

af::Status ReduceMeanApi(const std::vector<TensorShapeInfo> &input_shapes,
                         const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                         PerfOutputInfo &perf_res) {
  ascendcapi_v2::ReduceApiPerfContext context;
  GE_ASSERT_SUCCESS(BuildReduceContext(input_shapes, output_shapes, node, context));
  return CallReducePerf(node, context, kReduceMean, ascendcapi_v2::ReduceMeanPerf, perf_res);
}

af::Status ReduceProdApi(const std::vector<TensorShapeInfo> &input_shapes,
                         const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                         PerfOutputInfo &perf_res) {
  ascendcapi_v2::ReduceApiPerfContext context;
  GE_ASSERT_SUCCESS(BuildReduceContext(input_shapes, output_shapes, node, context));
  return CallReducePerf(node, context, kReduceProd, ascendcapi_v2::ReduceProdPerf, perf_res);
}

af::Status MaxApi(const std::vector<TensorShapeInfo> &input_shapes, const std::vector<TensorShapeInfo> &output_shapes,
                  const NodeInfo &node, PerfOutputInfo &perf_res) {
  return HasReduceSpecificParams(node) ? ReduceMaxApi(input_shapes, output_shapes, node, perf_res)
                                       : ElementwiseMaxApi(input_shapes, output_shapes, node, perf_res);
}

af::Status MinApi(const std::vector<TensorShapeInfo> &input_shapes, const std::vector<TensorShapeInfo> &output_shapes,
                  const NodeInfo &node, PerfOutputInfo &perf_res) {
  return HasReduceSpecificParams(node) ? ReduceMinApi(input_shapes, output_shapes, node, perf_res)
                                       : ElementwiseMinApi(input_shapes, output_shapes, node, perf_res);
}

af::Status AnyApi(const std::vector<TensorShapeInfo> &input_shapes, const std::vector<TensorShapeInfo> &output_shapes,
                  const NodeInfo &node, PerfOutputInfo &perf_res) {
  return HasReduceSpecificParams(node) ? ReduceAnyApi(input_shapes, output_shapes, node, perf_res)
                                       : ElementwiseMaxApi(input_shapes, output_shapes, node, perf_res);
}

af::Status AllApi(const std::vector<TensorShapeInfo> &input_shapes, const std::vector<TensorShapeInfo> &output_shapes,
                  const NodeInfo &node, PerfOutputInfo &perf_res) {
  return HasReduceSpecificParams(node) ? ReduceAllApi(input_shapes, output_shapes, node, perf_res)
                                       : ElementwiseMinApi(input_shapes, output_shapes, node, perf_res);
}

af::Status SumApi(const std::vector<TensorShapeInfo> &input_shapes, const std::vector<TensorShapeInfo> &output_shapes,
                  const NodeInfo &node, PerfOutputInfo &perf_res) {
  return HasReduceSpecificParams(node) ? ReduceSumApi(input_shapes, output_shapes, node, perf_res)
                                       : ElementwiseSumApi(input_shapes, output_shapes, node, perf_res);
}

af::Status MeanApi(const std::vector<TensorShapeInfo> &input_shapes, const std::vector<TensorShapeInfo> &output_shapes,
                   const NodeInfo &node, PerfOutputInfo &perf_res) {
  return HasReduceSpecificParams(node) ? ReduceMeanApi(input_shapes, output_shapes, node, perf_res)
                                       : ElementwiseMeanApi(input_shapes, output_shapes, node, perf_res);
}

af::Status ProdApi(const std::vector<TensorShapeInfo> &input_shapes, const std::vector<TensorShapeInfo> &output_shapes,
                   const NodeInfo &node, PerfOutputInfo &perf_res) {
  return HasReduceSpecificParams(node) ? ReduceProdApi(input_shapes, output_shapes, node, perf_res)
                                       : ElementwiseProdApi(input_shapes, output_shapes, node, perf_res);
}

/*
Argmax 系算子（Argmax / ArgMaxMultiRPhase1 / ArgMaxMultiRPhase2 / ArgmaxWithValue /
ArgMaxMultiRPhase2WithValue）的性能公式：
  Argmax 输出索引而非归约值，成本近似 ReduceMax（值+索引双路），复用 v2 ReduceMaxPerf；
  Phase1 在 Argmax 基础上多一路 value 输出（额外 store 开销，按输入规模线性放大）；
  Phase2 只沿核维度 N 归约（成本 ∝ A×N，显著小于单算子 A×R）；
  ArgmaxWithValue 与 Argmax 同构（额外多一路 value 输出），与 Phase1 成本相当；
  ArgMaxMultiRPhase2WithValue 为 ArgmaxWithValue 分核的 Phase2（一次归约双输出，成本 ∝ A×N），
  与 ArgMaxMultiRPhase2 成本同构；
  因此各算子独立注册、分别建模，避免 ATT 打分直通单算子成本造成错估。
*/
af::Status ArgmaxPerf([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                      [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                      [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::ReduceMaxPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status ArgmaxApi(const std::vector<TensorShapeInfo> &input_shapes,
                     const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                     PerfOutputInfo &perf_res) {
  return ArgmaxPerf(input_shapes, output_shapes, node, perf_res);
}

af::Status ArgMaxMultiRPhase1Api(const std::vector<TensorShapeInfo> &input_shapes,
                                 const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                                 PerfOutputInfo &perf_res) {
  return ArgmaxPerf(input_shapes, output_shapes, node, perf_res);
}

af::Status ArgMaxMultiRPhase2Api(const std::vector<TensorShapeInfo> &input_shapes,
                                 const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                                 PerfOutputInfo &perf_res) {
  return ArgmaxPerf(input_shapes, output_shapes, node, perf_res);
}

af::Status ArgmaxWithValueApi(const std::vector<TensorShapeInfo> &input_shapes,
                              const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                              PerfOutputInfo &perf_res) {
  return ArgmaxPerf(input_shapes, output_shapes, node, perf_res);
}

af::Status ArgMaxMultiRPhase2WithValueApi(const std::vector<TensorShapeInfo> &input_shapes,
                                          const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                                          PerfOutputInfo &perf_res) {
  return ArgmaxPerf(input_shapes, output_shapes, node, perf_res);
}
}  // namespace ascir_reduce_v2
}  // namespace att
