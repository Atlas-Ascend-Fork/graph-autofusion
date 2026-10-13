/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_tiling.h"

#include <algorithm>
#include <limits>
#include <set>
#include <unordered_map>

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"

namespace codegen {
using namespace af::ascir_op;
using namespace af::ops;
using namespace ascgen_utils;
using namespace ascir;

std::string TilingLib::CalculateTensorMemorySizeStr(const ascir::TensorAttr &tensor) const {
  return CalculateTensorMemorySizeStr(tensor, af::ops::Zero);
}

namespace {
bool GetTensorTypeSize(ge::DataType dtype, uint64_t &type_size) {
  static const std::unordered_map<ge::DataType, uint64_t> type_size_map = {
      {ge::DT_FLOAT, 4U},  {ge::DT_FLOAT16, 2U}, {ge::DT_INT8, 1U},   {ge::DT_INT16, 2U},  {ge::DT_INT32, 4U},
      {ge::DT_INT64, 8U},  {ge::DT_UINT8, 1U},   {ge::DT_UINT16, 2U}, {ge::DT_UINT32, 4U}, {ge::DT_UINT64, 8U},
      {ge::DT_DOUBLE, 8U}, {ge::DT_BF16, 2U},    {ge::DT_BOOL, 1U}};
  const auto iter = type_size_map.find(dtype);
  if (iter == type_size_map.end()) {
    return false;
  }
  type_size = iter->second;
  return true;
}

bool CheckedPgoSizeAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  constexpr uint64_t kMaxPgoIoSize = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  if (lhs > kMaxPgoIoSize - rhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

bool CheckedPgoSizeMul(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  constexpr uint64_t kMaxPgoIoSize = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  if (lhs != 0U && rhs > kMaxPgoIoSize / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

enum class ConstMemorySizeStatus { kSuccess, kSymbolic, kInvalid };

ConstMemorySizeStatus CalculateConstMemorySize(const ascir::TensorAttr &tensor, const af::Expression &element_offset,
                                               uint64_t type_size, uint64_t &memory_size) {
  bool all_const = element_offset.IsConstExpr();
  int64_t offset = 0;
  if ((all_const && !element_offset.GetConstValue(offset)) || offset < 0) {
    return ConstMemorySizeStatus::kInvalid;
  }
  std::vector<std::pair<int64_t, int64_t>> dims;
  bool has_zero_repeat = false;
  for (size_t i = 0UL; i < tensor.attr.repeats.size(); ++i) {
    int64_t repeat = 0;
    int64_t stride = 0;
    const bool dim_const = tensor.attr.repeats[i].IsConstExpr() && tensor.attr.strides[i].IsConstExpr();
    if (dim_const && (!tensor.attr.repeats[i].GetConstValue(repeat) || !tensor.attr.strides[i].GetConstValue(stride))) {
      return ConstMemorySizeStatus::kInvalid;
    }
    if (dim_const && (repeat < 0 || stride < 0)) {
      return ConstMemorySizeStatus::kInvalid;
    }
    has_zero_repeat = has_zero_repeat || (dim_const && repeat == 0);
    all_const = all_const && dim_const;
    dims.emplace_back(repeat, stride);
  }
  if (has_zero_repeat) {
    memory_size = 0U;
    return ConstMemorySizeStatus::kSuccess;
  }
  if (!all_const) {
    return ConstMemorySizeStatus::kSymbolic;
  }
  uint64_t span = static_cast<uint64_t>(offset) + 1U;
  for (const auto &[repeat, stride] : dims) {
    uint64_t contribution = 0U;
    if (!CheckedPgoSizeMul(static_cast<uint64_t>(repeat - 1), static_cast<uint64_t>(stride), contribution) ||
        !CheckedPgoSizeAdd(span, contribution, span)) {
      return ConstMemorySizeStatus::kInvalid;
    }
  }
  return CheckedPgoSizeMul(span, type_size, memory_size) ? ConstMemorySizeStatus::kSuccess
                                                         : ConstMemorySizeStatus::kInvalid;
}
}  // namespace

std::string TilingLib::CalculateTensorMemorySizeStr(const ascir::TensorAttr &tensor,
                                                    const af::Expression &element_offset) const {
  const auto dtype = tensor.attr.dtype.operator ge::DataType();
  uint64_t type_size = 0U;
  if (!GetTensorTypeSize(dtype, type_size)) {
    GELOGE(ge::GRAPH_FAILED, "Unsupported data type: %d", static_cast<int32_t>(dtype));
    return "0";
  }
  if (tensor.attr.repeats.empty() || tensor.attr.repeats.size() != tensor.attr.strides.size()) {
    GELOGE(ge::GRAPH_FAILED, "Invalid repeats or strides when calculating tensor memory size");
    return "0";
  }
  uint64_t const_memory_size = 0U;
  const auto const_status = CalculateConstMemorySize(tensor, element_offset, type_size, const_memory_size);
  if (const_status == ConstMemorySizeStatus::kSuccess) {
    return std::to_string(const_memory_size);
  }
  if (const_status == ConstMemorySizeStatus::kInvalid) {
    GELOGE(ge::GRAPH_FAILED, "Tensor memory size has invalid layout or overflow");
    return "0";
  }
  af::Expression element_span = af::sym::Add(element_offset, af::ops::One);
  for (size_t i = 0UL; i < tensor.attr.repeats.size(); ++i) {
    const auto repeat_span = af::sym::Sub(tensor.attr.repeats[i], af::ops::One);
    element_span = af::sym::Add(element_span, af::sym::Mul(repeat_span, tensor.attr.strides[i]));
  }
  const auto type_size_expr = af::Expression::Parse(std::to_string(type_size).c_str());
  af::Expression need_malloc_size = af::sym::Mul(element_span.Simplify(), type_size_expr).Simplify();
  GELOGD("Tensor element span: %s, need malloc size: %s", element_span.Str().get(), need_malloc_size.Str().get());
  return std::string(need_malloc_size.Str().get());
}

namespace {
enum class PgoIoKind { kInput, kOutput };
using PgoVarReplacements = std::vector<std::pair<af::Expression, af::Expression>>;

PgoVarReplacements BuildPgoVarReplacements(const ascir::ScheduledResult &result, size_t group_id) {
  PgoVarReplacements replacements;
  const auto dst_group_iter = result.var_relations.find(group_id);
  if (dst_group_iter == result.var_relations.end()) {
    return replacements;
  }
  for (const auto &relations_by_src : dst_group_iter->second) {
    for (const auto &[dst_var_name, src_expr] : relations_by_src.second) {
      replacements.emplace_back(af::Expression::Parse(dst_var_name.c_str()), src_expr);
    }
  }
  return replacements;
}

af::Expression ApplyPgoVarReplacements(const af::Expression &expr, const PgoVarReplacements &replacements) {
  return replacements.empty() ? expr : expr.Replace(replacements).Simplify();
}

ascir::TensorAttr ApplyPgoVarReplacements(const ascir::TensorAttr &tensor, const PgoVarReplacements &replacements) {
  ascir::TensorAttr replaced_tensor = tensor;
  for (auto &repeat : replaced_tensor.attr.repeats) {
    repeat = ApplyPgoVarReplacements(repeat, replacements);
  }
  for (auto &stride : replaced_tensor.attr.strides) {
    stride = ApplyPgoVarReplacements(stride, replacements);
  }
  return replaced_tensor;
}

int64_t GetPgoIoIndex(const af::AscNodePtr &node, int64_t fallback_index) {
  int64_t index = fallback_index;
  if (node == nullptr || node->attr.ir_attr == nullptr) {
    return index;
  }
  (void)node->attr.ir_attr->GetAttrValue("index", index);
  return index;
}

bool IsPgoIoNode(const af::AscNodePtr &node, int64_t io_index, PgoIoKind kind) {
  const bool type_matches =
      kind == PgoIoKind::kInput ? IsOps<Data>(node) || IsOps<ScalarData>(node) : IsOps<Output>(node);
  return type_matches && GetPgoIoIndex(node, -1) == io_index;
}

template <typename TensorVisitor>
void VisitPgoNodeTensors(const af::AscNodePtr &node, PgoIoKind kind, TensorVisitor &visitor) {
  if (kind == PgoIoKind::kOutput) {
    const auto &input_nodes = node->GetInDataNodes();
    const auto *access_node = input_nodes.empty() ? nullptr : static_cast<const af::AscNode *>(input_nodes.at(0).get());
    visitor(node->inputs[0], access_node);
    return;
  }
  for (auto *out_node : node->GetOutNodesPtr()) {
    auto *asc_out_node = static_cast<af::AscNode *>(out_node);
    if (asc_out_node != nullptr) {
      visitor(asc_out_node->outputs[0], asc_out_node);
    }
  }
}

template <typename TensorVisitor>
void VisitPgoGraphTensors(const af::AscGraph &graph, int64_t io_index, PgoIoKind kind,
                          const PgoVarReplacements &replacements, TensorVisitor &visitor) {
  for (const auto &node : graph.GetAllNodes()) {
    if (IsPgoIoNode(node, io_index, kind)) {
      auto replace_and_visit = [&replacements, &visitor](const ascir::TensorAttr &tensor,
                                                         const af::AscNode *access_node) {
        visitor(ApplyPgoVarReplacements(tensor, replacements), access_node, replacements);
      };
      VisitPgoNodeTensors(node, kind, replace_and_visit);
    }
  }
}

template <typename TensorVisitor>
void VisitPgoCandidateTensors(const ascir::FusedScheduledResult &fused_schedule_result, int64_t io_index,
                              PgoIoKind kind, TensorVisitor &visitor) {
  for (const auto &scheduled_results : fused_schedule_result.node_idx_to_scheduled_results) {
    for (const auto &result : scheduled_results) {
      for (size_t group_id = 0UL; group_id < result.schedule_groups.size(); ++group_id) {
        const auto replacements = BuildPgoVarReplacements(result, group_id);
        const auto &schedule_group = result.schedule_groups[group_id];
        for (const auto &impl_graph : schedule_group.impl_graphs) {
          VisitPgoGraphTensors(impl_graph, io_index, kind, replacements, visitor);
        }
      }
    }
  }
}

void AppendPgoTensorMalloc(std::stringstream &ss, const std::string &tensor_name,
                           const std::vector<std::string> &size_expressions) {
  if (size_expressions.empty()) {
    return;
  }
  if (std::find(size_expressions.begin(), size_expressions.end(), "") != size_expressions.end()) {
    ss << "  DLOGE(\"Invalid or symbolic PGO " << tensor_name << " memory size\");" << std::endl;
    ss << "  return FAILED;" << std::endl;
    return;
  }
  const std::string size_name = tensor_name + "_size";
  ss << "  size_t " << size_name << " = " << size_expressions[0] << ";" << std::endl;
  for (size_t i = 1UL; i < size_expressions.size(); ++i) {
    ss << "  " << size_name << " = std::max(" << size_name << ", static_cast<size_t>(" << size_expressions[i] << "));"
       << std::endl;
  }
  ss << "  ret = aclrtMalloc(&" << tensor_name << ", " << size_name << ", ACL_MEM_MALLOC_HUGE_FIRST);" << std::endl;
  ss << "  if (ret != ACL_SUCCESS) {" << std::endl;
  ss << "    DLOGE(\"aclrtMalloc " << tensor_name << " failed. ERROR: %d\", ret);" << std::endl;
  ss << "    return FAILED;" << std::endl;
  ss << "  }" << std::endl;
}
}  // namespace

std::vector<std::string> TilingLib::CalculatePgoIoMemorySizeStrs(
    const ascir::FusedScheduledResult &fused_schedule_result, int64_t io_index, bool is_input,
    const ascir::TensorAttr &fallback_tensor) const {
  std::vector<std::string> size_expressions;
  std::set<std::string> seen_expressions;
  auto append_size = [this, &size_expressions, &seen_expressions](const ascir::TensorAttr &tensor,
                                                                  const af::AscNode *access_node,
                                                                  const PgoVarReplacements &replacements) {
    af::Expression element_offset = af::ops::Zero;
    if (access_node != nullptr && access_node->attr.ir_attr != nullptr) {
      (void)access_node->attr.ir_attr->GetAttrValue("offset", element_offset);
    }
    element_offset = ApplyPgoVarReplacements(element_offset, replacements);
    const std::string size_expression = CalculateTensorMemorySizeStr(tensor, element_offset);
    if (size_expression == "0" || !af::Expression::Parse(size_expression.c_str()).IsConstExpr()) {
      GELOGD("Reject invalid or symbolic PGO candidate memory size: %s", size_expression.c_str());
      if (seen_expressions.emplace("").second) {
        size_expressions.emplace_back("");
      }
      return;
    }
    if (seen_expressions.emplace(size_expression).second) {
      size_expressions.emplace_back(size_expression);
    }
  };
  append_size(fallback_tensor, nullptr, {});
  const PgoIoKind kind = is_input ? PgoIoKind::kInput : PgoIoKind::kOutput;
  VisitPgoCandidateTensors(fused_schedule_result, io_index, kind, append_size);
  return size_expressions;
}

std::string TilingLib::PGOSearchTensorMallocDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  int index = 0;
  for (auto &input : fused_schedule_result.input_nodes) {
    if (input->GetOutNodesPtr().empty()) {
      continue;
    }
    af::Node *out_node = input->GetOutNodesPtr()[0];
    af::AscNode *asc_out_node = static_cast<af::AscNode *>(out_node);
    const auto size_expressions = CalculatePgoIoMemorySizeStrs(fused_schedule_result, GetPgoIoIndex(input, index), true,
                                                               asc_out_node->outputs[0]);
    AppendPgoTensorMalloc(ss, "input" + std::to_string(index), size_expressions);
    index++;
  }
  index = 0;
  for (auto &output : fused_schedule_result.output_nodes) {
    if (af::ops::IsOps<af::ascir_op::Output>(output)) {
      const auto size_expressions =
          CalculatePgoIoMemorySizeStrs(fused_schedule_result, GetPgoIoIndex(output, index), false, output->inputs[0]);
      AppendPgoTensorMalloc(ss, "output" + std::to_string(index), size_expressions);
      index++;
    }
  }
  return ss.str();
}

std::string TilingLib::PGOSearchTensorFreeDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  int index = 0;
  for ([[maybe_unused]] auto &input : fused_schedule_result.input_nodes) {
    ss << "  if (input" << index << " != nullptr) {" << std::endl;
    ss << "    ret = aclrtFree(input" << index << ");" << std::endl;
    ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
    ss << "      DLOGW(\"aclrtFree input" << index << " failed. ERROR: %d\", ret);" << std::endl;
    ss << "    }" << std::endl;
    ss << "    input" << index << " = nullptr;" << std::endl;
    ss << "  }" << std::endl;
    index++;
  }
  index = 0;
  for (auto &output : fused_schedule_result.output_nodes) {
    if (af::ops::IsOps<af::ascir_op::Output>(output)) {
      ss << "  if (output" << index << " != nullptr) {" << std::endl;
      ss << "    ret = aclrtFree(output" << index << ");" << std::endl;
      ss << "    if (ret != ACL_SUCCESS) {" << std::endl;
      ss << "      DLOGW(\"aclrtFree output" << index << " failed. ERROR: %d\", ret);" << std::endl;
      ss << "    }" << std::endl;
      ss << "    output" << index << " = nullptr;" << std::endl;
      ss << "  }" << std::endl;
      index++;
    }
  }
  return ss.str();
}

std::string TilingLib::ExternFunctionDeclare(const ascir::FusedScheduledResult &fused_schedule_result,
                                             const std::string tiling) const {
  (void)tiling;
  std::stringstream ss;

  // 生成判断是否为静态shape的接口
  bool is_static = IsFrontendStaticSchedResult(fused_schedule_result);
  ss << GenCheckStaticShapeFunc(is_static);
  return ss.str();
}

std::string TilingLib::PGOTensorArgsDef() const {
  std::stringstream ss;
  ss << "#ifndef AUTOFUSE_PGO_TENSOR_ARGS_DEFINED" << std::endl;
  ss << "#define AUTOFUSE_PGO_TENSOR_ARGS_DEFINED" << std::endl;
  ss << "struct PgoTensorArgs {" << std::endl;
  ss << "  void **inputs = nullptr;" << std::endl;
  ss << "  uint32_t input_num = 0;" << std::endl;
  ss << "  void **outputs = nullptr;" << std::endl;
  ss << "  uint32_t output_num = 0;" << std::endl;
  ss << "};" << std::endl;
  ss << "#endif" << std::endl;
  return ss.str();
}

void TilingLib::AppendPgoConfigDef(std::stringstream &ss) const {
  ss << "class PgoConfig {" << std::endl;
  ss << "public:" << std::endl;
  ss << "  static PgoConfig& Instance() {" << std::endl;
  ss << "    static PgoConfig instance;" << std::endl;
  ss << "    return instance;" << std::endl;
  ss << "  }" << std::endl;
  ss << "  void ResetRuntimeOverrides() {" << std::endl;
  ss << "    need_change_solver_run = false;" << std::endl;
  ss << "    pgo_threshold_index = 0;" << std::endl;
  ss << "    pgo_ub_threshold_list = {0.2, 0.1, 0, 0.05, 0.1};" << std::endl;
  ss << "    pgo_corenum_threshold_list = {0.4, 0.4, 1, 1, 0.8};" << std::endl;
  ss << "  }" << std::endl;
  ss << "  ProfilingCallback single_callback = nullptr;" << std::endl;
  ss << "  ProfilingBatchCallback batch_callback = nullptr;" << std::endl;
  ss << "  PgoTensorArgs *tensor_args = nullptr;" << std::endl;
  ss << "  std::vector<AutofuseTilingDataPerf> *measured_candidates = nullptr;" << std::endl;
  ss << "  void *stream = nullptr;" << std::endl;
  ss << "  int32_t pgo_algorithm = 1; // 0 for pruning, 1 for core num" << std::endl;
  ss << "  bool need_change_solver_run = false;" << std::endl;
  ss << "  size_t pgo_threshold_index = 0;" << std::endl;
  ss << "  constexpr static size_t pgo_threshold_list_size = 5;" << std::endl;
  ss << "  std::array<double, pgo_threshold_list_size> pgo_ub_threshold_list{0.2, 0.1, 0, 0.05, 0.1};" << std::endl;
  ss << "  std::array<double, pgo_threshold_list_size> pgo_corenum_threshold_list{0.4, 0.4, 1, 1, 0.8};" << std::endl;
  ss << "private:" << std::endl;
  ss << "  PgoConfig() = default;" << std::endl;
  ss << "  ~PgoConfig() = default;" << std::endl;
  ss << "  PgoConfig(const PgoConfig &) = delete;" << std::endl;
  ss << "  PgoConfig &operator=(const PgoConfig &) = delete;" << std::endl;
  ss << "};" << std::endl;
  ss << "class PgoConfigRuntimeGuard {" << std::endl;
  ss << "public:" << std::endl;
  ss << "  PgoConfigRuntimeGuard() { PgoConfig::Instance().ResetRuntimeOverrides(); }" << std::endl;
  ss << "  ~PgoConfigRuntimeGuard() { PgoConfig::Instance().ResetRuntimeOverrides(); }" << std::endl;
  ss << "};" << std::endl;
}

std::string TilingLib::PGOProfilingCallbackDef(const ascir::FusedScheduledResult &fused_schedule_result,
                                               const std::string tiling, bool include_headers) const {
  std::stringstream ss;

  if (include_headers) {
    ss << "#include <cfloat>" << std::endl;
    ss << "#include <cstdint>" << std::endl;
    ss << "#include <vector>" << std::endl;
    ss << "#include <unordered_set>" << std::endl;
    ss << "#include <array>" << std::endl;
    ss << std::endl;
  }
  ss << PGOTensorArgsDef();
  ss << "typedef long int (*ProfilingCallback)(";
  ss << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result);
  ss << "void *stream, uint64_t workspaceSize, " << tiling << " *tiling_data, double *cost_time);" << std::endl;
  ss << "typedef long int (*ProfilingBatchCallback)(";
  ss << PGOSearchFuncInputOutputCallBackDef(fused_schedule_result);
  ss << "void *stream, uint64_t workspaceSize, std::vector<AutofuseTilingDataPerf> *profiles);" << std::endl;
  AppendPgoConfigDef(ss);
  ss << std::endl;

  return ss.str();
}

std::string TilingLib::PGOSearchFuncInputOutputCallBackDef(
    const ascir::FusedScheduledResult &fused_schedule_result) const {
  (void)fused_schedule_result;
  return "PgoTensorArgs *tensor_args, ";
}

std::string TilingLib::PGOSearchFuncInputOutputDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  (void)fused_schedule_result;
  return "PgoTensorArgs *tensor_args = nullptr, ";
}

std::string TilingLib::PGOSearchFuncInputOutputCall(const ascir::FusedScheduledResult &fused_schedule_result) const {
  (void)fused_schedule_result;
  return "tensor_args, ";
}

std::string TilingLib::PGOSearchStructInputOutputDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  int index = 0;
  for ([[maybe_unused]] auto &input : fused_schedule_result.input_nodes) {
    ss << "  uint64_t input" << index++ << ";" << std::endl;
  }
  index = 0;
  for (auto &node : fused_schedule_result.output_nodes) {
    if (af::ops::IsOps<af::ascir_op::Output>(node)) {
      ss << "  uint64_t output" << index++ << ";" << std::endl;
    }
  }

  return ss.str();
}

std::string TilingLib::PGOSearchTensorInputOutputDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  int index = 0;
  for ([[maybe_unused]] auto &input : fused_schedule_result.input_nodes) {
    ss << "void* input" << index++ << ";" << std::endl;
  }
  index = 0;
  for (auto &node : fused_schedule_result.output_nodes) {
    if (af::ops::IsOps<af::ascir_op::Output>(node)) {
      ss << "void* output" << index++ << ";" << std::endl;
    }
  }
  ss << "uint64_t ffts;" << std::endl;
  const size_t input_array_size = std::max<size_t>(fused_schedule_result.input_nodes.size(), 1UL);
  const uint32_t output_count = PGOSearchFuncGetOutputCount(fused_schedule_result);
  const uint32_t output_array_size = std::max<uint32_t>(output_count, 1U);
  ss << "void *g_pgo_inputs[" << input_array_size << "] = {nullptr};" << std::endl;
  ss << "void *g_pgo_outputs[" << output_array_size << "] = {nullptr};" << std::endl;
  ss << "PgoTensorArgs g_pgo_tensor_args = {g_pgo_inputs, " << fused_schedule_result.input_nodes.size()
     << "U, g_pgo_outputs, " << output_count << "U};" << std::endl;

  return ss.str();
}

std::string TilingLib::PGOSearchTensorArgsUpdateDef(const ascir::FusedScheduledResult &fused_schedule_result) const {
  std::stringstream ss;
  int index = 0;
  for ([[maybe_unused]] auto &input : fused_schedule_result.input_nodes) {
    ss << "  g_pgo_inputs[" << index << "] = input" << index << ";" << std::endl;
    index++;
  }
  index = 0;
  for (auto &node : fused_schedule_result.output_nodes) {
    if (af::ops::IsOps<af::ascir_op::Output>(node)) {
      ss << "  g_pgo_outputs[" << index << "] = output" << index << ";" << std::endl;
      index++;
    }
  }
  ss << "  g_pgo_tensor_args.inputs = g_pgo_inputs;" << std::endl;
  ss << "  g_pgo_tensor_args.input_num = " << fused_schedule_result.input_nodes.size() << "U;" << std::endl;
  ss << "  g_pgo_tensor_args.outputs = g_pgo_outputs;" << std::endl;
  ss << "  g_pgo_tensor_args.output_num = " << PGOSearchFuncGetOutputCount(fused_schedule_result) << "U;" << std::endl;
  return ss.str();
}

std::string TilingLib::PGOSearchFuncInputOutputStructAssignDef(const ascir::FusedScheduledResult &fused_schedule_result,
                                                               const std::string &struct_var_name) const {
  std::stringstream ss;
  int index = 0;
  for ([[maybe_unused]] auto &input : fused_schedule_result.input_nodes) {
    ss << struct_var_name << ".input" << index << " = reinterpret_cast<uint64_t>(tensor_args->inputs[" << index << "]);"
       << std::endl;
    index++;
  }
  index = 0;
  for (auto &node : fused_schedule_result.output_nodes) {
    if (af::ops::IsOps<af::ascir_op::Output>(node)) {
      ss << struct_var_name << ".output" << index << " = reinterpret_cast<uint64_t>(tensor_args->outputs[" << index
         << "]);" << std::endl;
      index++;
    }
  }
  return ss.str();
}

uint32_t TilingLib::PGOSearchFuncGetInputOutputCount(const ascir::FusedScheduledResult &fused_schedule_result) const {
  return fused_schedule_result.input_nodes.size() + PGOSearchFuncGetOutputCount(fused_schedule_result);
}

uint32_t TilingLib::PGOSearchFuncGetOutputCount(const ascir::FusedScheduledResult &fused_schedule_result) const {
  uint32_t count = 0;
  for (auto &node : fused_schedule_result.output_nodes) {
    if (af::ops::IsOps<af::ascir_op::Output>(node)) {
      count++;
    }
  }
  return count;
}

}  // namespace codegen
