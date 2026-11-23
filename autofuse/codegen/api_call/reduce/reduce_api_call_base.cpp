/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_api_call_base.h"

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "ascir_node_param/ascir_node_param.h"
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common/checker.h"
#include "common/ge_common/debug/log.h"
#include "common_utils.h"
#include "arg_reduce_utils.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "api_call/utils/api_call_factory.h"
#include "api_call/utils/api_call_utils.h"

namespace reduce_base {
using namespace codegen;
using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;

namespace {
struct ReduceShadowCheckResult {
  std::vector<std::string> mismatch_fields;
  std::vector<std::string> mismatch_details;
};

struct ReduceLegacyBehavior {
  codegen::ReducePattern pattern{codegen::ReducePattern::kUnknown};
  bool need_multi_reduce{false};
  bool has_reuse{false};
  bool is_reuse_source{false};
};

struct ReduceShadowBuildContext {
  af::AscNodePtr node;
  std::string api_name;
  const TPipe *tpipe{nullptr};
  const Tensor *input{nullptr};
  const Tensor *output{nullptr};
  ascir::AxisId axis_id{-1};
};

struct ReduceShadowCheckContext {
  std::string api_name;
  ReduceShadowBuildContext build;
  ReduceLegacyBehavior legacy;
};

void LogMissingAscirNodeParams(const ReduceShadowCheckContext &ctx) {
  const auto &node = ctx.build.node;
  if (node == nullptr) {
    GELOGW("[ASCIR_PARAM_TRACE] Codegen missing api[%s], node is null.", ctx.api_name.c_str());
    return;
  }
  const auto op_desc = node->GetOpDesc();
  const auto owner_graph = node->GetOwnerComputeGraph();
  const auto node_name = node->GetName();
  const auto node_type = node->GetType();
  const auto graph_name = owner_graph == nullptr ? std::string("<null>") : owner_graph->GetName();
  GELOGW(
      "[ASCIR_PARAM_TRACE] Codegen missing api[%s], node[%s], type[%s], graph[%s], node_ptr[%p], "
      "op_desc_ptr[%p], owner_graph_ptr[%p].",
      ctx.api_name.c_str(), node_name.c_str(), node_type.c_str(), graph_name.c_str(), static_cast<void *>(node.get()),
      static_cast<void *>(op_desc.get()), static_cast<void *>(owner_graph.get()));
}

std::string BoolToString(bool value) {
  return value ? "true" : "false";
}

std::string ExprToString(const ge::Expression &expr) {
  return expr.IsValid() ? std::string(expr.Str().get()) : std::string("<invalid>");
}

std::string ReducePatternToString(codegen::ReducePattern pattern) {
  switch (pattern) {
    case codegen::ReducePattern::kUnknown:
      return "Unknown";
    case codegen::ReducePattern::kAR:
      return "AR";
    case codegen::ReducePattern::kRA:
      return "RA";
    default:
      return "Invalid(" + std::to_string(static_cast<int32_t>(pattern)) + ")";
  }
}

std::string ReduceMergeModeToString(codegen::ReduceMergeMode mode) {
  switch (mode) {
    case codegen::ReduceMergeMode::kUnknown:
      return "Unknown";
    case codegen::ReduceMergeMode::kNone:
      return "None";
    case codegen::ReduceMergeMode::kCopy:
      return "Copy";
    case codegen::ReduceMergeMode::kMergeByElementwise:
      return "MergeByElementwise";
    default:
      return "Invalid(" + std::to_string(static_cast<int32_t>(mode)) + ")";
  }
}

void AddMismatchDetail(const std::string &field, const std::string &lhs_name, const std::string &lhs_value,
                       const std::string &rhs_name, const std::string &rhs_value, ReduceShadowCheckResult &result) {
  std::stringstream ss;
  ss << field << "{" << lhs_name << "=" << lhs_value << "," << rhs_name << "=" << rhs_value << "}";
  result.mismatch_details.emplace_back(ss.str());
}

void AddMismatch(bool matched, const std::string &field, const std::string &parser_value,
                 const std::string &shadow_value, ReduceShadowCheckResult &result) {
  if (matched) {
    return;
  }
  result.mismatch_fields.emplace_back(field);
  AddMismatchDetail(field, "parser", parser_value, "shadow", shadow_value, result);
}

bool FindAxisIndex(const std::vector<ascir::AxisId> &axis_ids, ascir::AxisId axis_id, size_t &axis_index) {
  const auto iter = std::find(axis_ids.begin(), axis_ids.end(), axis_id);
  if (iter == axis_ids.end()) {
    return false;
  }
  axis_index = static_cast<size_t>(std::distance(axis_ids.begin(), iter));
  return true;
}

std::vector<ge::Expression> GetVectorizedRepeats(const Tensor &tensor) {
  std::vector<ge::Expression> repeats;
  repeats.reserve(tensor.vectorized_axis.size());
  for (const auto axis_id : tensor.vectorized_axis) {
    size_t axis_index = 0U;
    const auto repeat = FindAxisIndex(tensor.axis, axis_id, axis_index) ? tensor.axis_size[axis_index] : ge::Symbol(1U);
    repeats.emplace_back(repeat);
  }
  return repeats;
}

std::vector<ge::Expression> GetOutputDims(const Tensor &tensor) {
  return GetVectorizedRepeats(tensor);
}

codegen::ReducePattern GetCodegenReducePattern(const Tensor &output) {
  if (output.vectorized_strides.empty()) {
    return codegen::ReducePattern::kUnknown;
  }
  return output.vectorized_strides.back() == af::sym::kSymbolZero ? codegen::ReducePattern::kAR
                                                                  : codegen::ReducePattern::kRA;
}

af::Status FillReduceReuseSource(const af::AscNodePtr &node, codegen::ReduceReuseInfo &reuse) {
  GE_ASSERT_NOTNULL(node);
  auto node_in_anchor = node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(node_in_anchor);
  auto peer_out_anchor = node_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(peer_out_anchor);
  const auto &in_node = std::dynamic_pointer_cast<af::AscNode>(peer_out_anchor->GetOwnerNode());
  GE_ASSERT_NOTNULL(in_node);
  reuse.valid = true;
  reuse.is_reuse_source = in_node->GetOutAllNodes().size() == 1UL;
  return af::SUCCESS;
}

af::Status BuildReduceInput(const ReduceShadowBuildContext &ctx, codegen::ReduceSpecificParamBuildInput &input) {
  GE_ASSERT_NOTNULL(ctx.tpipe);
  GE_ASSERT_NOTNULL(ctx.input);
  GE_ASSERT_NOTNULL(ctx.output);
  input.node_name = ctx.node == nullptr ? ctx.api_name : ctx.node->GetName();
  input.reduce_type = ctx.api_name;
  input.input_repeats = GetVectorizedRepeats(*ctx.input);
  input.input_strides = ctx.input->vectorized_strides;
  input.output_dims = GetOutputDims(*ctx.output);
  input.output_strides = ctx.output->vectorized_strides;
  const auto dtype_size = ge::GetSizeByDataType(ctx.input->dtype);
  GE_ASSERT_TRUE(dtype_size > 0, "Invalid reduce input dtype size, dtype[%d].", static_cast<int32_t>(ctx.input->dtype));
  input.dtype_size = static_cast<uint32_t>(dtype_size);
  input.pattern = GetCodegenReducePattern(*ctx.output);
  input.need_multi_reduce = IsNeedMultiReduce(ctx.tpipe->tiler, *ctx.input, *ctx.output, ctx.axis_id);
  input.merge_times = input.need_multi_reduce ? ctx.tpipe->tiler.GetAxis(ctx.axis_id).size_expr : ge::Symbol(1U);
  if (ctx.node != nullptr) {
    GE_ASSERT_SUCCESS(FillReduceReuseSource(ctx.node, input.reuse));
  }
  return af::SUCCESS;
}

std::string JoinMismatchFields(const std::vector<std::string> &fields) {
  std::stringstream ss;
  for (size_t i = 0U; i < fields.size(); ++i) {
    if (i != 0U) {
      ss << ",";
    }
    ss << fields[i];
  }
  return ss.str();
}

std::string JoinMismatchDetails(const std::vector<std::string> &details) {
  std::stringstream ss;
  for (size_t i = 0U; i < details.size(); ++i) {
    if (i != 0U) {
      ss << ";";
    }
    ss << details[i];
  }
  return ss.str();
}

af::Status BuildCodegenShadowReduceSpecificParams(const ReduceShadowBuildContext &ctx,
                                                  codegen::ReduceSpecificParams &params) {
  codegen::ReduceSpecificParamBuildInput input;
  GE_ASSERT_SUCCESS(BuildReduceInput(ctx, input));
  GE_ASSERT_SUCCESS(codegen::BuildReduceSpecificParams(input, params));
  return af::SUCCESS;
}

ReduceShadowCheckContext BuildReduceShadowCheckContext(const ReduceCodegenShadowCheckInput &input) {
  ReduceShadowCheckContext ctx;
  ctx.api_name = input.api_name;
  ctx.build.node = input.node;
  ctx.build.api_name = input.api_name;
  ctx.build.tpipe = input.tpipe;
  ctx.build.input = input.input;
  ctx.build.output = input.output;
  ctx.build.axis_id = input.axis_id;
  if (input.output != nullptr) {
    ctx.legacy.pattern = GetCodegenReducePattern(*input.output);
  }
  if (input.tpipe != nullptr && input.input != nullptr && input.output != nullptr) {
    ctx.legacy.need_multi_reduce = IsNeedMultiReduce(input.tpipe->tiler, *input.input, *input.output, input.axis_id);
  }
  ctx.legacy.has_reuse = input.has_reuse;
  ctx.legacy.is_reuse_source = input.is_reuse_source;
  return ctx;
}

void CompareReduceSpecificParams(const codegen::ReduceSpecificParams &parser_params,
                                 const codegen::ReduceSpecificParams &shadow_params, ReduceShadowCheckResult &result) {
  AddMismatch(parser_params.valid == shadow_params.valid, "valid", BoolToString(parser_params.valid),
              BoolToString(shadow_params.valid), result);
  AddMismatch(parser_params.reduce_type == shadow_params.reduce_type, "reduce_type", parser_params.reduce_type,
              shadow_params.reduce_type, result);
  AddMismatch(parser_params.pattern == shadow_params.pattern, "pattern", ReducePatternToString(parser_params.pattern),
              ReducePatternToString(shadow_params.pattern), result);
  AddMismatch(parser_params.merge_mode == shadow_params.merge_mode, "merge_mode",
              ReduceMergeModeToString(parser_params.merge_mode), ReduceMergeModeToString(shadow_params.merge_mode),
              result);
  AddMismatch(ExpressEq(parser_params.merge_size, shadow_params.merge_size), "merge_size",
              ExprToString(parser_params.merge_size), ExprToString(shadow_params.merge_size), result);
  AddMismatch(ExpressEq(parser_params.merge_times, shadow_params.merge_times), "merge_times",
              ExprToString(parser_params.merge_times), ExprToString(shadow_params.merge_times), result);
  AddMismatch(parser_params.reuse.valid == shadow_params.reuse.valid, "reuse.valid",
              BoolToString(parser_params.reuse.valid), BoolToString(shadow_params.reuse.valid), result);
  AddMismatch(parser_params.reuse.is_reuse_source == shadow_params.reuse.is_reuse_source, "reuse.source",
              BoolToString(parser_params.reuse.is_reuse_source), BoolToString(shadow_params.reuse.is_reuse_source),
              result);
  AddMismatch(parser_params.merged_dims.valid == shadow_params.merged_dims.valid, "merged_dims.valid",
              BoolToString(parser_params.merged_dims.valid), BoolToString(shadow_params.merged_dims.valid), result);
  if (parser_params.merged_dims.valid && shadow_params.merged_dims.valid) {
    AddMismatch(ExpressEq(parser_params.merged_dims.first, shadow_params.merged_dims.first), "merged_dims.first",
                ExprToString(parser_params.merged_dims.first), ExprToString(shadow_params.merged_dims.first), result);
    AddMismatch(ExpressEq(parser_params.merged_dims.last, shadow_params.merged_dims.last), "merged_dims.last",
                ExprToString(parser_params.merged_dims.last), ExprToString(shadow_params.merged_dims.last), result);
  }
}

void CompareReduceLegacyBehavior(const codegen::ReduceSpecificParams &parser_params, const ReduceLegacyBehavior &legacy,
                                 ReduceShadowCheckResult &result) {
  if (parser_params.pattern != legacy.pattern) {
    result.mismatch_fields.emplace_back("legacy.pattern");
    AddMismatchDetail("legacy.pattern", "parser", ReducePatternToString(parser_params.pattern), "legacy",
                      ReducePatternToString(legacy.pattern), result);
  }
  const bool need_multi_reduce = parser_params.merge_mode != codegen::ReduceMergeMode::kNone;
  if (need_multi_reduce != legacy.need_multi_reduce) {
    result.mismatch_fields.emplace_back("legacy.merge_mode");
    AddMismatchDetail(
        "legacy.merge_mode", "parser",
        BoolToString(need_multi_reduce) + ",parser.merge_mode=" + ReduceMergeModeToString(parser_params.merge_mode),
        "legacy", BoolToString(legacy.need_multi_reduce), result);
  }
  if (legacy.has_reuse) {
    if (!parser_params.reuse.valid) {
      result.mismatch_fields.emplace_back("legacy.reuse.valid");
      AddMismatchDetail("legacy.reuse.valid", "parser", BoolToString(parser_params.reuse.valid), "legacy", "true",
                        result);
    }
    if (parser_params.reuse.is_reuse_source != legacy.is_reuse_source) {
      result.mismatch_fields.emplace_back("legacy.reuse.source");
      AddMismatchDetail("legacy.reuse.source", "parser", BoolToString(parser_params.reuse.is_reuse_source), "legacy",
                        BoolToString(legacy.is_reuse_source), result);
    }
  }
}

}  // namespace

// 用于将代码中的"first"和"last"相互替换
static void ReplaceSS(std::string &str, const std::string &oldSubStr, const std::string &newSubStr) {
  size_t pos = 0;
  while ((pos = str.find(oldSubStr, pos)) != std::string::npos) {
    str.replace(pos, oldSubStr.length(), newSubStr);
    pos += newSubStr.length();
  }
  return;
}

static void ReplaceSSWithSwappingFirstAndLast(const std::string &first, const std::string &first_actual,
                                              const std::string &last, const std::string &last_actual,
                                              const bool &isAllAxisReduce, std::vector<std::string> &lines) {
  std::string f = first;
  std::string fa = first_actual;
  std::string l = last;
  std::string la = last_actual;
  if (isAllAxisReduce) {
    ReplaceSS(f, "first", "last");
    ReplaceSS(fa, "first", "last");
    ReplaceSS(l, "last", "first");
    ReplaceSS(la, "last", "first");
  }
  // 添加四条独立语句，每条语句末尾加分号和换行符
  lines.emplace_back(f + ";\n");
  lines.emplace_back(fa + ";\n");
  lines.emplace_back(l + ";\n");
  lines.emplace_back(la + ";\n");
  return;
}

/*
  生成最内层向量化轴（尾轴）的 merged size 表达式：
  src/dst 尾轴 stride 均为 0 表示尾轴长度为 1，按 SizeAlign(1, 32/sizeof(dtype)) 对齐；
  尾轴为首个有效轴时按 SizeAlign(axis_size, 32/sizeof(dtype)) 生成（全轴 reduce 时写 first，末态统一替换为 last）；
  其余情况按紧邻的非零 stride 生成。
*/
static void GenLastAxisMergedSize(const TPipe &tpipe, const Tensor &src, const Tensor &dst, const size_t axis_index,
                                  const std::string &dtype_name, const bool is_first, const bool is_all_axis_reduce,
                                  const ascir::SizeExpr &last_non_zero_stride, std::stringstream &first,
                                  std::stringstream &first_actual, std::stringstream &last,
                                  std::stringstream &last_actual) {
  const auto axis = tpipe.tiler.GetAxis(src.vectorized_axis[axis_index]);
  const auto axis_size = tpipe.tiler.AxisSize(src.vectorized_axis[axis_index]);
  if (src.vectorized_strides[axis_index] == 0 && dst.vectorized_strides[axis_index] == 0) {
    last << " * " << KernelUtils::SizeAlign() << "(1, 32/sizeof(" << dtype_name << "))";
    last_actual << " * " << KernelUtils::SizeAlign() << "(1, 32/sizeof(" << dtype_name << "))";
    return;
  }
  if (is_first) {
    std::stringstream &size_target = is_all_axis_reduce ? first : last;
    std::stringstream &actual_size_target = is_all_axis_reduce ? first_actual : last_actual;
    size_target << " * " << KernelUtils::SizeAlign() << "(" << axis_size << ", 32/sizeof(" << dtype_name << "))";
    actual_size_target << " * " << KernelUtils::SizeAlign() << "(" << axis.actual_size << ", 32/sizeof(" << dtype_name
                       << "))";
    return;
  }
  last << " * " << tpipe.tiler.Size(last_non_zero_stride);
  last_actual << " * " << tpipe.tiler.Size(last_non_zero_stride);
}

/*
  返回AR或者RA的fist_size和last_size；
  为了避免由于存在src[i].axis_size=1(此时strides=0)导致误判为R轴，所以在遍历过程中过滤掉了src[i].axis_size为1的情况；
  用last_not_1_axis_size_index来记录上一次的axis_size != 1的位置。
  当src与dst的尾轴stride均为0时表示尾轴为1，此时last按SizeAlign(1, 32/sizeof(dtype))生成（对齐粒度随dtype）。
*/
void ReduceMergedSizeCodeGen(const TPipe &tpipe, std::vector<std::string> &lines, const Tensor &src, const Tensor &dst,
                             bool is_tail) {
  std::stringstream first;
  std::stringstream first_actual;
  std::stringstream last;
  std::stringstream last_actual;
  first << "uint32_t " << (is_tail ? "first_tail" : "first") << " = 1";
  first_actual << "uint32_t " << (is_tail ? "first_tail_actual" : "first_actual") << " = 1";
  last << "uint32_t " << (is_tail ? "last_tail" : "last") << " = 1";
  last_actual << "uint32_t " << (is_tail ? "last_tail_actual" : "last_actual") << " = 1";
  std::string dtype_name;
  Tensor::DtypeName(src.dtype, dtype_name);
  bool is_first = true;
  const size_t num_axes = src.vectorized_axis.size();
  ascir::SizeExpr lastNonZeroStride = Zero;
  size_t last_not_1_axis_size_index = 0xFFFFFFFF;
  bool isAllAxisReduce = true;
  for (size_t i = 0; i < num_axes; ++i) {
    isAllAxisReduce =
        isAllAxisReduce && (src.vectorized_strides[i] != 0 || i != num_axes - 1) && (dst.vectorized_strides[i] == 0);
    if (i == num_axes - 1U) {
      GenLastAxisMergedSize(tpipe, src, dst, i, dtype_name, is_first, isAllAxisReduce, lastNonZeroStride, first,
                            first_actual, last, last_actual);
      break;
    }
    const auto axis = tpipe.tiler.GetAxis(src.vectorized_axis[i]);
    const auto axis_size = tpipe.tiler.AxisSize(src.vectorized_axis[i]);
    if (src.vectorized_strides[i] == 0 && dst.vectorized_strides[i] == 0) {
      continue;
    }
    if (is_first && last_not_1_axis_size_index != 0xFFFFFFFF) {
      is_first = !((dst.vectorized_strides[i] == 0 && dst.vectorized_strides[last_not_1_axis_size_index] != 0) ||
                   (dst.vectorized_strides[i] != 0 && dst.vectorized_strides[last_not_1_axis_size_index] == 0));
    }
    if (!is_first) {
      if (src.vectorized_strides[i] != Zero) {
        lastNonZeroStride = src.vectorized_strides[i];
      }
      last << " * " << axis_size;
      last_actual << " * " << axis.actual_size;
    } else {
      first << " * " << axis_size;
      first_actual << " * " << axis.actual_size;
      last_not_1_axis_size_index = i;
    }
  }
  // 先添加代码块开始符号
  lines.emplace_back("{\n");
  ReplaceSSWithSwappingFirstAndLast(first.str(), first_actual.str(), last.str(), last_actual.str(), isAllAxisReduce,
                                    lines);
}

bool IsNeedMultiReduce(const Tiler &tiler, const Tensor &input, const Tensor &output, ascir::AxisId axis_id) {
  int64_t total_count = 0;
  int64_t valid_count = 0;
  std::function<void(ascir::AxisId)> recursive_functor = [&tiler, &input, &output, &total_count, &valid_count,
                                                          &recursive_functor](ascir::AxisId id) {
    Axis axis = tiler.GetAxis(id);
    auto pos = std::find(output.axis.begin(), output.axis.end(), id);
    if (pos != output.axis.end()) {
      size_t diff = pos - output.axis.begin();
      total_count++;
      valid_count =
          output.axis_strides[diff] == Zero && (input.axis_strides[diff] != Zero || input.axis_size[diff] != One)
              ? valid_count + 1
              : valid_count;
      return;
    }
    for (size_t i = 0; i < axis.from.size(); i++) {
      auto from_axis = tiler.GetAxis(axis.from[i]);
      bool need_recursive = from_axis.type != Axis::Type::kAxisTypeOriginal;
      auto pos = std::find(output.axis.begin(), output.axis.end(), axis.from[i]);
      if (pos != output.axis.end()) {
        size_t diff = pos - output.axis.begin();
        total_count++;
        valid_count =
            output.axis_strides[diff] == Zero && (input.axis_strides[diff] != Zero || input.axis_size[diff] != One)
                ? valid_count + 1
                : valid_count;
        return;
      }
      if (need_recursive) {
        for (size_t j = 0; j < from_axis.from.size(); j++) {
          recursive_functor(from_axis.from[j]);
        }
      }
    }
  };
  recursive_functor(axis_id);
  return total_count == valid_count;
}

void ReduceMeanCodeGen(std::string &dtype_name, const TPipe &tpipe, const Tensor &src, const Tensor &dst,
                       std::vector<std::string> &lines) {
  std::set<ascir::AxisId> r_from_axis;
  for (size_t i = 0; i < dst.axis_strides.size(); i++) {
    if ((src.axis_strides[i] != 0 || src.axis_size[i] != 1) && dst.axis_strides[i] == 0) {  // 如果目标张量的轴步长为0
      auto axis_id = dst.axis[i];                                                           // 获取当前轴ID
      // 定义递归函数用于收集原始轴
      std::function<void(int)> collect_original_axes = [&tpipe, &r_from_axis,
                                                        &collect_original_axes](int current_axis_id) {
        auto axis = tpipe.tiler.GetAxis(current_axis_id);  // 获取当前轴对象
        if (axis.type == ascir::Axis::Type::kAxisTypeOriginal) {
          r_from_axis.insert(current_axis_id);  // 如果是原始轴则加入集合
          return;
        }
        // 否则递归处理所有来源轴
        for (auto from_axis_id : axis.from) {
          collect_original_axes(from_axis_id);
        }
      };
      collect_original_axes(axis_id);  // 从当前轴开始递归收集
    }
  }
  std::string dimr_recip_line = "dimr_recip = 1.0f / (";
  uint32_t count = 0;
  for (auto axis_id : r_from_axis) {
    if (count++ == 0) {
      dimr_recip_line += tpipe.tiler.AxisSize(axis_id);
    } else {
      dimr_recip_line += " * " + tpipe.tiler.AxisSize(axis_id);
    }
  }
  dimr_recip_line += ");\n";
  lines.emplace_back(dimr_recip_line);
  lines.emplace_back("Muls(" + dst.Str() + ", " + dst.Str() + ", dimr_recip, " + KernelUtils::SizeAlign() +
                     "(reduce_dim_a, 32 / sizeof(" + dtype_name + ")));\n");
  return;
}

void GetIsArAndPattern(const Tensor &x, const Tensor &y, bool &isAr, std::string &reduce_pattern) {
  isAr = (x.vectorized_strides.back() != 0) && (y.vectorized_strides.back() == 0);
  std::unordered_map<bool, std::string> reduce_pattern_map = {{true, "AscendC::Pattern::Reduce::AR"},
                                                              {false, "AscendC::Pattern::Reduce::RA"}};
  reduce_pattern = reduce_pattern_map[isAr];
  return;
}

void CheckReduceSpecificParamsForCodegen(const ReduceCodegenShadowCheckInput &input) {
  const auto ctx = BuildReduceShadowCheckContext(input);
  const auto params = ascir_param::GetAscirNodeParams(ctx.build.node);
  if (params == nullptr) {
    LogMissingAscirNodeParams(ctx);
    return;
  }
  if (params->status != ascir_param::ParamBuildStatus::kBuilt) {
    GELOGW("[ASCIR_PARAM] AscirNodeParams status is not built, api[%s].", ctx.api_name.c_str());
    return;
  }
  const auto *parser_params = ascir_param::GetSpecificParams<ascir_param::ReduceNodeParams>(*params);
  if (parser_params == nullptr) {
    GELOGW("[ASCIR_PARAM] Reduce parser params missing, api[%s].", ctx.api_name.c_str());
    return;
  }
  codegen::ReduceSpecificParams shadow_params;
  const auto shadow_status = BuildCodegenShadowReduceSpecificParams(ctx.build, shadow_params);
  if (shadow_status != af::SUCCESS) {
    GELOGW("[ASCIR_PARAM] Build codegen reduce shadow params failed, api[%s].", ctx.api_name.c_str());
    return;
  }
  ReduceShadowCheckResult result;
  AddMismatch(params->api_name == ctx.api_name, "api_name", params->api_name, ctx.api_name, result);
  CompareReduceSpecificParams(parser_params->canonical_params, shadow_params, result);
  CompareReduceLegacyBehavior(parser_params->canonical_params, ctx.legacy, result);
  if (!result.mismatch_fields.empty()) {
    GELOGW("[ASCIR_PARAM] Reduce param shadow check mismatch, api[%s], fields[%s], details[%s].", ctx.api_name.c_str(),
           JoinMismatchFields(result.mismatch_fields).c_str(), JoinMismatchDetails(result.mismatch_details).c_str());
  }
}

bool IsTilerLastReduceAxis(const Tensor &tensor) {
  int count = 0;
  for (auto stride : tensor.vectorized_strides) {
    if (stride == 0) {
      count++;
    }
  }
  return count == 1;
}

void ReduceInitCodeGen(const Tensor &x, const Tensor &y, const int &type_value, std::vector<std::string> &lines,
                       const TPipe &tpipe, const std::string &dtype_name, const std::string &x_name_after_reinterpret) {
  if (x.isAr) {
    std::string is_last_axis_str = IsTilerLastReduceAxis(y) ? "true" : "false";
    std::stringstream ss;
    ss << "ReduceInit<" << dtype_name << ", " << type_value << ", " << is_last_axis_str << ">("
       << (x_name_after_reinterpret.empty() ? x.Str() : x_name_after_reinterpret) << ", "
       << "first_actual" << ", last" << ", last_actual, " << tpipe.tiler.GetAxis(x.vectorized_axis.back()).actual_size
       << ");" << std::endl;
    lines.emplace_back(ss.str());
    lines.emplace_back("AscendC::PipeBarrier<PIPE_V>();\n");
  }
  return;
}

void ReduceDimACodeGen(const Tensor &x, const std::string &apiName, std::vector<std::string> &lines) {
  if (apiName == "ReduceMean") {
    if (x.isAr) {
      lines.emplace_back("reduce_dim_a = first_actual;\n");
    } else {
      lines.emplace_back("reduce_dim_a = last_actual;\n");
    }
  }
  return;
}

void GenLastTwoRAxisSizeProductCode(const Tensor &x, const Tensor &y, const TPipe &tpipe,
                                    std::vector<std::string> &lines) {
  // 收集所有R轴
  std::vector<std::pair<ascir::AxisId, size_t>> r_axes;

  for (size_t i = 0; i < x.axis.size(); ++i) {
    bool is_r_axis = (y.axis_strides[i] == Zero && x.axis_strides[i] != Zero);
    if (is_r_axis) {
      r_axes.push_back({x.axis[i], i});
    }
  }

  // 根据R轴数量生成不同的代码
  if (r_axes.size() >= 2) {  // 如果有2个以上的R轴，则R轴块大小为最后2个R轴的乘积
    // 有至少两个R轴，使用最后两个R轴
    ascir::AxisId last_r_axis = r_axes[r_axes.size() - 1].first;
    ascir::AxisId second_last_r_axis = r_axes[r_axes.size() - 2].first;

    lines.emplace_back("// 最后两个R轴大小的乘积，作为每个核处理的R轴块大小\n");
    std::stringstream ss;
    ss << "int64_t r_axis_block_size = " << tpipe.tiler.AxisSize(last_r_axis) << " * "
       << tpipe.tiler.AxisSize(second_last_r_axis) << ";" << std::endl;
    lines.emplace_back(ss.str());
  } else if (r_axes.size() == 1) {
    // 只有一个R轴
    lines.emplace_back("// 只有一个R轴，使用其大小作为块大小\n");
    std::stringstream ss;
    ss << "int64_t r_axis_block_size = " << tpipe.tiler.AxisSize(r_axes[0].first) << ";" << std::endl;
    lines.emplace_back(ss.str());
  } else {
    // 没有R轴（特殊情况）
    lines.emplace_back("// 没有R轴，使用默认值\n");
    lines.emplace_back("int64_t r_axis_block_size = 1;\n");
  }
}

Status GetDtypeNameForReduce(const std::string &api_name, const Tensor &x, const Tensor &y, std::string &dtype_name) {
  // Argmax系列算子（Argmax、ArgMaxMultiRPhase1、ArgMaxMultiRPhase2、ArgmaxWithValue、
  // ArgMaxMultiRPhase2WithValue）需要使用value的类型作为模板参数
  // 而不是index的类型，因此统一使用x（inputs[0]）的dtype
  if (ascgen_utils::IsArgReduceType(api_name)) {
    GE_CHK_STATUS_RET(Tensor::DtypeName(x.dtype, dtype_name), "Codegen get data type:%d failed",
                      static_cast<int32_t>(x.dtype));
  } else {
    GE_CHK_STATUS_RET(Tensor::DtypeName(y.dtype, dtype_name), "Codegen get data type:%d failed",
                      static_cast<int32_t>(y.dtype));
  }
  return af::SUCCESS;
}

void GenAccumulatedOffsetDeclForArgmax(const std::string &api_name, const Tensor &x, const Tensor &y,
                                       const TPipe &tpipe, std::vector<std::string> &lines) {
  // Argmax 和 ArgMaxMultiRPhase1 需要在循环外声明累加的 offset 变量（使用 static 保存状态）
  if (api_name == "Argmax") {
    lines.emplace_back("static int64_t accumulated_offset = 0;\n");
  } else if (api_name == "ArgMaxMultiRPhase1") {
    // ArgMaxMultiRPhase1的初始offset = block_dim * 最后两个R轴大小的乘积
    // 使用辅助函数生成计算最后两个R轴大小乘积的代码
    GenLastTwoRAxisSizeProductCode(x, y, tpipe, lines);
    lines.emplace_back("static int64_t accumulated_offset = 0;\n");
  }
}

}  // namespace reduce_base
