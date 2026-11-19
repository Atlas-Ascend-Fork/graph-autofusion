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

#include <algorithm>
#include <memory>
#include "ascir_node_param/ascir_node_param.h"
#include "codegen_api_param/codegen_api_param.h"
#include "v35/att/api_perf_register/ascendc_api_perf/reduce_api_perf_v2.h"
#include "v35/att/api_perf_register/ascendc_regbase_perf.h"
#include "api_perf_register/utils/vf_perf_utils.h"

namespace att {
namespace ascir_reduce_v2 {
namespace {
constexpr int64_t kPerfFactorTwo = 2;
constexpr int64_t kPerfFactorThree = 3;
constexpr int64_t kBlockBytes = 32;
constexpr size_t kReduceDimPairCount = 2U;
constexpr size_t kPhase2ShapeRank = 3U;
constexpr size_t kPhase2ColumnsDimIndex = 2U;

void SetReduceDims(const Expr &first, const Expr &last, ascendcapi_v2::ReducePattern pattern, NodeDetail &node_detail) {
  node_detail.input_dims = {first, last};
  node_detail.output_dims = {pattern == ascendcapi_v2::ReducePattern::kRA ? last : first};
}

void MergeReduceDimsByLastAxis(ascendcapi_v2::ReducePattern pattern, NodeDetail &node_detail) {
  auto &in_dims = node_detail.input_dims;
  if (in_dims.size() <= kReduceDimPairCount) {
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
  GE_ASSERT_TRUE(context.node_detail.input_dims.size() == kReduceDimPairCount,
                 "Reduce dims must be {first,last}, node[%s].", node.name.c_str());
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

namespace {
af::Status GetArgReduceRepeatReciprocal(const std::string &dtype, Expr &inv_vl) {
  int32_t repeat_elements = 0;
  GE_ASSERT_TRUE(kRptEleMap.at(dtype).GetConstValue(repeat_elements), "Arg-reduce repeat elements must be constant.");
  // Preserve Max(repeat_elements, 1), using an exact rational rather than a symbolic divisor.
  inv_vl = af::sym::Rational(1, std::max(repeat_elements, 1));
  return af::SUCCESS;
}

Expr SelectArgReduceCost(const std::string &name, const Expr &lhs, const Expr &rhs, const Expr &yes, const Expr &no,
                         PerfOutputInfo &perf, CondType condition = CondType::K_EQ) {
  const auto check = condition == CondType::K_LE ? af::SymbolicUtils::StaticCheckLe(lhs, rhs)
                                                 : af::SymbolicUtils::StaticCheckEq(lhs, rhs);
  if (check != af::TriBool::kUnknown) {
    return check == af::TriBool::kTrue ? yes : no;
  }
  Expr result;
  GetPerfVar(name, result, perf.ternary_ops);
  TernaryOp choice(condition, lhs, rhs, yes, no);
  choice.SetVariable(result);
  perf.ternary_ops[result] = choice;
  return result;
}

af::Status AddArgReduceRowFloatCost(const std::string &dtype, const Expr &a, const Expr &updates, Expr &latency,
                                    Expr &cost) {
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, kInt32, latency, cost, CreateExpr(kPerfFactorTwo)));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, kInt32, latency, cost, a));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kNe, dtype, latency, cost, updates + CreateExpr(kPerfFactorThree) * a));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kNe, kInt32, latency, cost, a));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kReduceMax, kInt32, latency, cost, a));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskOr, dtype, latency, cost, updates + a));
  return af::SUCCESS;
}

af::Status AddArgReduceRowValueCost(const ascir_param::ArgmaxNodeParams &params, const std::string &dtype,
                                    const std::string &index_dtype, const Expr &a, Expr &latency, Expr &cost) {
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAdds, index_dtype, latency, cost,
                                                   params.is_ar ? a : CreateExpr(kPerfFactorTwo) * a));
  if (!params.is_ar) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMuls, index_dtype, latency, cost, a));
  }
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, dtype, latency, cost, a));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, dtype, latency, cost, a));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, dtype, latency, cost, a));
  // Reg::LocalMemBar<VEC_STORE, VEC_STORE>; no exact performance-table entry.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, dtype, latency, cost, a));
  return af::SUCCESS;
}

af::Status AddArgReduceRowFinalCost(const std::string &dtype, const std::string &index_dtype, const Expr &a,
                                    const Expr &updates, Expr &latency, Expr &cost, bool is_min) {
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, index_dtype, latency, cost, updates + a));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(is_min ? kReduceMin : kReduceMax, dtype, latency, cost, a));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kEq, dtype, latency, cost, a));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAdd, index_dtype, latency, cost, a));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kReduceMin, index_dtype, latency, cost, a));
  // Final Eq is masked by validLaneMask even for integer values.
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kMaskAnd, dtype, latency, cost, a + (dtype == kFloat32 ? a : CreateExpr(0))));
  return af::SUCCESS;
}

af::Status ArgReduceRowCost(const ascir_param::ArgReduceNodeParams &params, const std::string &dtype, const Expr &a,
                            const Expr &valid_r, bool aligned, PerfOutputInfo &perf, Expr &result, bool is_min) {
  const Expr zero = CreateExpr(0);
  const Expr one = CreateExpr(1);
  const Expr two = CreateExpr(kPerfFactorTwo);
  Expr inv_vl;
  GE_ASSERT_SUCCESS(GetArgReduceRepeatReciprocal(dtype, inv_vl));
  const std::string index_dtype = dtype == kInt64 ? kInt64 : kInt32;
  const bool fp32 = dtype == kFloat32;
  const Expr blocks = af::sym::Ceiling(valid_r * inv_vl);
  const Expr updates = a * af::sym::Max(blocks - one, zero);
  const Expr first_partial =
      SelectArgReduceCost("arg_reduce_row_first_partial", af::sym::Floor(valid_r * inv_vl), zero, a, zero, perf);
  const Expr partial_loads = a * (blocks - af::sym::Floor(valid_r * inv_vl));
  const Expr tail = partial_loads - first_partial;
  Expr latency = zero;
  Expr cost = zero;
  // CreateMask<T, ALL> once per VF; no exact performance-table entry.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, dtype, latency, cost, one));
  // low, per-row rowMaxBcast, and float nanValue once per VF.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, dtype, latency, cost, one + a + (fp32 ? one : zero)));
  // bigReg, per-row bestBlock, and each update's currentIndex.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, index_dtype, latency, cost, one + a + updates));
  // One source pass. ArgmaxLoadBlock's UB Gather is a load, not register Gather2.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, dtype, latency, cost, a * blocks));
  if (params.is_ar && !aligned) {
    // Reg::LoadUnAlignPre, separate from LoadUnAlign (kLoad) and CreateMask.
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kPlaceholder, dtype, latency, cost, a * af::sym::Floor(valid_r * inv_vl)));
  }
  // validLaneMask once per VF, then only a subsequent partial block needs UpdateMask.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, dtype, latency, cost, one + tail));
  // laneIndex once per VF; AR gathers partial blocks, RA gathers every source block.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kArange, index_dtype, latency, cost,
                                                   one + (params.is_ar ? partial_loads : a * blocks)));
  if (!params.is_ar) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMuls, index_dtype, latency, cost, a * blocks));
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAdds, index_dtype, latency, cost, a * blocks));
  }
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(is_min ? kLe : kGe, dtype, latency, cost, updates));
  // First-block/tail selects, update selects, and float finiteValue/rowMax selects.
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kSelect, dtype, latency, cost, a + tail + updates + (fp32 ? two * a : zero)));
  // Update's bestBlock and the final candidate; float nanFlag is counted separately below.
  GE_ASSERT_SUCCESS(AddArgReduceRowFinalCost(dtype, index_dtype, a, updates, latency, cost, is_min));
  if (fp32) {
    GE_ASSERT_SUCCESS(AddArgReduceRowFloatCost(dtype, a, updates, latency, cost));
  }
  if (params.with_value) {
    GE_ASSERT_SUCCESS(AddArgReduceRowValueCost(params, dtype, index_dtype, a, latency, cost));
  }
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, kInt32, latency, cost, a));
  if (dtype == kInt64) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDeInterleave, kInt32, latency, cost, a));
  }
  result = VfPerfUtils::GetVFHeadCost() + latency + cost;
  result.Simplify();
  return af::SUCCESS;
}

// argmax.h: ArgmaxRaFullVectorImpl, ArgmaxRaVectorImpl or ArgMaxMultiRPhase2VectorImpl.
af::Status AddArgReducePhase2ColumnCost(const std::string &dtype, const std::string &index_dtype, const Expr &a,
                                        const Expr &inv_vl, const Expr &columns, const Expr &merges, const Expr &r,
                                        Expr &latency, Expr &cost, bool is_min) {
  const Expr one = CreateExpr(1);
  const Expr two = CreateExpr(kPerfFactorTwo);
  const Expr three = CreateExpr(kPerfFactorThree);
  const Expr partial_value = (columns - af::sym::Floor(a * inv_vl)) * r;
  // b64 values use 32 lanes; every int32 index load is partial (32 < 64 lanes).
  const Expr partial_index = dtype == kInt64 ? columns * r : partial_value;
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, kInt32, latency, cost, columns * r));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, dtype, latency, cost, partial_value));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, kInt32, latency, cost, partial_index));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kArange, index_dtype, latency, cost, partial_value));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kArange, kInt32, latency, cost, partial_index));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLt, index_dtype, latency, cost, merges));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kEq, dtype, latency, cost, merges));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(is_min ? kLt : kGt, dtype, latency, cost, merges));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kMaskAnd, dtype, latency, cost, merges * (dtype == kFloat32 ? three : one)));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kMaskOr, dtype, latency, cost, merges * (dtype == kFloat32 ? three : one)));
  if (dtype == kFloat32) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kNe, dtype, latency, cost, two * merges));
    // Reg::Not(MaskReg); kNot describes vector data, not mask registers.
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, dtype, latency, cost, merges));
  }
  if (dtype == kInt64) {
    // indexAllMask creation, then one zero32 initialization per VF.
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kInt32, latency, cost, one));
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, kInt32, latency, cost, one));
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kInterleave, kInt32, latency, cost, columns * r));
  }
  return af::SUCCESS;
}

af::Status AddArgReduceRaColumnCost(const ascir_param::ArgmaxNodeParams &params, const std::string &dtype,
                                    const std::string &index_dtype, const Expr &a, const Expr &inv_vl,
                                    const Expr &columns, const Expr &merges, const Expr &history, const Expr &r,
                                    bool full_columns, bool aligned, Expr &latency, Expr &cost, bool is_min) {
  const Expr one = CreateExpr(1);
  const Expr two = CreateExpr(kPerfFactorTwo);
  const Expr three = CreateExpr(kPerfFactorThree);
  if (!full_columns) {
    // ArgmaxLoadRow: a partial column block creates its own mask and offsets on every R load.
    const Expr partial = (columns - af::sym::Floor(a * inv_vl)) * r;
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, dtype, latency, cost, partial));
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kArange, index_dtype, latency, cost, partial));
    if (!aligned) {
      // First R row uses LoadAlign; subsequent full column blocks use LoadUnAlignPre + LoadUnAlign.
      GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(
          kPlaceholder, dtype, latency, cost, af::sym::Floor(a * inv_vl) * af::sym::Max(r - one, CreateExpr(0))));
    }
  }
  // A-vector groups have independent destinations and visit R in order.
  // Only historical accumulation requires MergeMasked.
  const Expr masked_merges = history;
  const Expr updates = merges;
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, index_dtype, latency, cost, columns + updates));
  if (params.has_accumulate && full_columns) {
    // Same VF: two history loads and one MergeMasked, no temporary stores,
    // Copy/MergeRound VF head, or reload of the current tile result.
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, dtype, latency, cost, history));
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, kInt32, latency, cost, history));
    if (dtype == kInt64) {
      // ArgmaxLoadIndexBlock widens a masked 32-lane int32 load.
      GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, kInt32, latency, cost, history));
      GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kArange, kInt32, latency, cost, history));
      GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, kInt32, latency, cost, history));
      GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kInterleave, kInt32, latency, cost, history));
    }
  }
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(is_min ? kLe : kGe, dtype, latency, cost, updates));
  if (full_columns) {
    // Full-vector indices start local; partial-column indices already include indexBase in Duplicate.
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAdds, index_dtype, latency, cost, columns));
  }
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLt, index_dtype, latency, cost, masked_merges));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kEq, dtype, latency, cost, masked_merges));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(is_min ? kLt : kGt, dtype, latency, cost, masked_merges));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskAnd, dtype, latency, cost,
                                                   masked_merges * (dtype == kFloat32 ? three : one)));
  if (dtype == kFloat32) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kNe, dtype, latency, cost, updates + two * masked_merges));
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskOr, dtype, latency, cost, updates + three * masked_merges));
    // Reg::Not(MaskReg) in ArgmaxMergeMasked; no mask-register Not entry.
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, dtype, latency, cost, masked_merges));
  } else {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskOr, dtype, latency, cost, masked_merges));
  }
  return af::SUCCESS;
}

af::Status ArgReduceColumnCost(const ascir_param::ArgReduceNodeParams &params, const std::string &dtype, const Expr &a,
                               const Expr &r, bool full_columns, bool aligned, Expr &result, bool is_min) {
  const Expr one = CreateExpr(1);
  Expr inv_vl;
  GE_ASSERT_SUCCESS(GetArgReduceRepeatReciprocal(dtype, inv_vl));
  const Expr columns = af::sym::Ceiling(a * inv_vl);
  const Expr merges = columns * af::sym::Max(r - one, CreateExpr(0));
  // firstTile is a runtime loop condition. Charge the steady-state (non-first)
  // path for every nominal invocation, conservatively overcounting the first.
  const Expr history = params.has_accumulate && !params.is_phase2 && full_columns ? columns : CreateExpr(0);
  const std::string index_dtype = dtype == kInt64 ? kInt64 : kInt32;
  Expr latency = CreateExpr(0);
  Expr cost = CreateExpr(0);
  if (params.is_phase2 || !full_columns) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, dtype, latency, cost, columns));
  } else {
    // Full-VL RA: aGroups=4 when A%(4*VL)==0, otherwise 1. Both issue C=A/VL groups of instructions.
    // Grouping changes head parallelism; no empirical throughput multiplier is applied.
    // Reg::CreateMask<T, ALL> has no exact performance-table entry.
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, dtype, latency, cost, one));
  }
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, dtype, latency, cost, columns * r));
  if (params.is_phase2) {
    GE_ASSERT_SUCCESS(
        AddArgReducePhase2ColumnCost(dtype, index_dtype, a, inv_vl, columns, merges, r, latency, cost, is_min));
  } else {
    GE_ASSERT_SUCCESS(AddArgReduceRaColumnCost(params, dtype, index_dtype, a, inv_vl, columns, merges, history, r,
                                               full_columns, aligned, latency, cost, is_min));
  }
  // Each ordinary update and historical MergeMasked selects value/index once: merges + history.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, dtype, latency, cost, merges + history));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, index_dtype, latency, cost, merges + history));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, kInt32, latency, cost, columns));
  if (dtype == kInt64) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDeInterleave, kInt32, latency, cost, columns));
  }
  if (params.with_value) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, dtype, latency, cost, columns));
    // Reg::LocalMemBar<VEC_STORE, VEC_STORE>; no exact performance-table entry.
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, dtype, latency, cost, columns));
  }
  result = VfPerfUtils::GetVFHeadCost() + latency + cost;
  result.Simplify();
  return af::SUCCESS;
}
af::Status ArgReduceRaCost(const ascir_param::ArgReduceNodeParams &params, const std::string &dtype, const Expr &a,
                           const Expr &valid, const Expr &physical_a, PerfOutputInfo &perf_res, Expr &result,
                           bool is_min) {
  const Expr zero = CreateExpr(0);
  const Expr vl = kRptEleMap.at(dtype);
  GE_ASSERT_TRUE(vl != zero, "Argmax repeat elements must be nonzero.");
  Expr vector_cost;
  Expr partial_aligned_cost;
  Expr partial_unaligned_cost;
  Expr gather_cost;
  GE_ASSERT_SUCCESS(ArgReduceColumnCost(params, dtype, a, valid, true, true, vector_cost, is_min));
  GE_ASSERT_SUCCESS(ArgReduceColumnCost(params, dtype, a, valid, false, true, partial_aligned_cost, is_min));
  GE_ASSERT_SUCCESS(ArgReduceColumnCost(params, dtype, a, valid, false, false, partial_unaligned_cost, is_min));
  GE_ASSERT_SUCCESS(ArgReduceRowCost(params, dtype, a, valid, false, perf_res, gather_cost, is_min));
  const Expr partial_cost = SelectArgReduceCost("arg_reduce_ra_aligned",
                                                af::sym::Mod(a, CreateExpr(kBlockBytes) / kDataTypeSizeMap.at(dtype)),
                                                zero, partial_aligned_cost, partial_unaligned_cost, perf_res);
  // Non-full RA: A>=VL || (validR<=VL && validR<=2*A), exactly as both API entry points.
  const Expr short_wide = SelectArgReduceCost("arg_reduce_ra_short_wide", valid, CreateExpr(kPerfFactorTwo) * a,
                                              partial_cost, gather_cost, perf_res, CondType::K_LE);
  const Expr short_r =
      SelectArgReduceCost("arg_reduce_ra_short_r", valid, vl, short_wide, gather_cost, perf_res, CondType::K_LE);
  const Expr nonfull =
      SelectArgReduceCost("arg_reduce_ra_partial_columns", vl, a, partial_cost, short_r, perf_res, CondType::K_LE);
  result = SelectArgReduceCost("arg_reduce_ra_full_columns", af::sym::Mod(a, vl), zero, vector_cost, nonfull, perf_res);
  if (params.valid_a != zero) {
    // Gather strides use physical_a; output and instruction counts use logical a.
    result = SelectArgReduceCost("arg_reduce_ra_valid_a", a, physical_a, result, gather_cost, perf_res);
  }
  return af::SUCCESS;
}
}  // namespace

af::Status ArgReducePerf(const std::vector<TensorShapeInfo> &input_shapes,
                         [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                         PerfOutputInfo &perf_res, bool is_min) {
  const auto &params = node.argmax_node_params;
  GE_ASSERT_TRUE(params.valid && params.src_shape.size() == (params.is_phase2 ? kPhase2ShapeRank : kReduceDimPairCount),
                 "Missing Argmax codegen params, node[%s].", node.name.c_str());
  GE_ASSERT_TRUE(!input_shapes.empty());
  const auto &dtype = input_shapes[0].data_type;
  GE_ASSERT_TRUE(dtype == kFloat32 || dtype == kInt32 || dtype == kInt64);
  const Expr zero = CreateExpr(0);
  const Expr bound = params.is_phase2 ? af::sym::Min(params.src_shape[1], params.src_shape[kPhase2ColumnsDimIndex])
                                      : params.src_shape[params.is_ar ? 1U : 0U];
  const Expr valid = SelectArgReduceCost("arg_reduce_valid_extent", params.valid_extent, zero, bound,
                                         af::sym::Min(params.valid_extent, bound), perf_res);
  const Expr physical_a = params.src_shape[params.is_ar ? 0U : 1U];
  Expr a = params.is_phase2 ? valid : physical_a;
  if (!params.is_phase2 && params.valid_a != zero) {
    a = SelectArgReduceCost("arg_reduce_valid_a", params.valid_a, zero, physical_a,
                            af::sym::Min(params.valid_a, physical_a), perf_res);
  }
  const Expr r = params.is_phase2 ? params.src_shape[0] : valid;
  Expr result;
  if (params.is_phase2) {
    GE_ASSERT_SUCCESS(ArgReduceColumnCost(params, dtype, a, r, false, true, result, is_min));
  } else if (params.is_ar) {
    Expr aligned_cost;
    Expr unaligned_cost;
    GE_ASSERT_SUCCESS(ArgReduceRowCost(params, dtype, a, valid, true, perf_res, aligned_cost, is_min));
    GE_ASSERT_SUCCESS(ArgReduceRowCost(params, dtype, a, valid, false, perf_res, unaligned_cost, is_min));
    result = SelectArgReduceCost("arg_reduce_ar_aligned",
                                 af::sym::Mod(bound, CreateExpr(kBlockBytes) / kDataTypeSizeMap.at(dtype)), zero,
                                 aligned_cost, unaligned_cost, perf_res);
  } else {
    GE_ASSERT_SUCCESS(ArgReduceRaCost(params, dtype, a, valid, physical_a, perf_res, result, is_min));
  }
  // One core VF call. Outer invocation counts belong to ATT's caller.
  // Full-VL RA Accumulate includes history loads/merge in this VF. Layout fallback
  // Copy/MergeRound and frontend index CastExtend are still NOT included.
  // Nominal tile extents approximate the last outer-loop tail; no additional execution multiplier is applied here.
  // kLoad/placeholder table entries are currently zero; this is instruction accounting, not calibrated memory timing.
  result.Simplify();
  perf_res.pipe_res[PipeType::AIV_VEC] = result;
  return af::SUCCESS;
}

af::Status ArgmaxApi(const std::vector<TensorShapeInfo> &input_shapes,
                     const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                     PerfOutputInfo &perf_res) {
  return ArgReducePerf(input_shapes, output_shapes, node, perf_res, false);
}

af::Status ArgMaxMultiRPhase1Api(const std::vector<TensorShapeInfo> &input_shapes,
                                 const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                                 PerfOutputInfo &perf_res) {
  return ArgReducePerf(input_shapes, output_shapes, node, perf_res, false);
}

af::Status ArgMaxMultiRPhase2Api(const std::vector<TensorShapeInfo> &input_shapes,
                                 const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                                 PerfOutputInfo &perf_res) {
  return ArgReducePerf(input_shapes, output_shapes, node, perf_res, false);
}

af::Status ArgmaxWithValueApi(const std::vector<TensorShapeInfo> &input_shapes,
                              const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                              PerfOutputInfo &perf_res) {
  return ArgReducePerf(input_shapes, output_shapes, node, perf_res, false);
}

af::Status ArgMaxMultiRPhase2WithValueApi(const std::vector<TensorShapeInfo> &input_shapes,
                                          const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                                          PerfOutputInfo &perf_res) {
  return ArgReducePerf(input_shapes, output_shapes, node, perf_res, false);
}

af::Status ArgminApi(const std::vector<TensorShapeInfo> &input_shapes,
                     const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                     PerfOutputInfo &perf_res) {
  return ArgReducePerf(input_shapes, output_shapes, node, perf_res, true);
}
}  // namespace ascir_reduce_v2
}  // namespace att
