/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "l2_cache_hint_manager.h"

#include "attr_utils.h"
#include "ascir_ops.h"
#include "ascgen_log.h"
#include "ascir_ops_utils.h"
#include "common_utils.h"
#include "common/platform_context.h"
#include "fusion/autofuse_attrs.h"

using namespace ascir;
using namespace optimize;
using namespace af::ascir_op;
using namespace af::ops;

namespace {
constexpr char const kAscBackendType[] = "AscBackend";
constexpr char const kSkipL2CacheHintAttr[] = "_skip_l2_cache_hint";
// 单广播轴扩成2 / 广播前的轴尺寸为1 / L2减半校验的除数
constexpr int64_t kBroadcastFactor = 2;
constexpr int64_t kPreBroadcastDimSize = 1;
constexpr int64_t kHalfL2Divider = 2;

std::string ExprToStr(const af::Expression &expr) {
  if (!expr.IsValid()) {
    return "<invalid>";
  }
  const auto expr_str_ptr = expr.Str();
  return (expr_str_ptr == nullptr) ? std::string("<null>") : std::string(expr_str_ptr.get());
}

bool IsSmallTensor(const af::AscNodePtr &node) {
  if (node == nullptr || node->outputs().empty()) {
    return false;
  }
  af::Expression output_size = af::sym::kSymbolOne;
  for (const auto &repeat : node->outputs[0].attr.repeats) {
    output_size = output_size * repeat;
  }
  output_size = output_size * af::Symbol(static_cast<int64_t>(ge::GetSizeByDataType(node->outputs[0].attr.dtype)));
  int64_t output_size_value = 0;
  bool is_small = output_size.IsConstExpr() && output_size.GetConstValue(output_size_value) &&
                  (output_size_value <= 2L * 1024L * 1024L);
  if (is_small) {
    GELOGD("IsSmallTensor node[%s] output_size[%lld bytes] <= 2MB.", node->GetNamePtr(), output_size_value);
  }
  return is_small;
}

// 计算tensor attr的物理字节数(repeats连乘×dtype)，可静态求值时返回true
bool CalcAttrSizeBytes(const af::AscTensorAttr &attr, int64_t &size_bytes) {
  af::Expression size_expr = af::sym::kSymbolOne;
  for (const auto &repeat : attr.repeats) {
    size_expr = size_expr * repeat;
  }
  size_expr = size_expr * af::Symbol(static_cast<int64_t>(ge::GetSizeByDataType(attr.dtype)));
  return size_expr.IsConstExpr() && size_expr.GetConstValue(size_bytes);
}

bool NeedSkipL2CacheHint(const af::AscNodePtr &node) {
  const auto &out_data_nodes = node->GetOutDataNodes();
  if (out_data_nodes.empty()) {
    return false;
  }
  const auto out_node = std::dynamic_pointer_cast<af::AscNode>(out_data_nodes.at(0));
  GE_ASSERT_NOTNULL(out_node);
  GE_CHK_BOOL_RET_SPECIAL_STATUS(out_data_nodes.size() > 1UL, true,
                                 "NeedSkipL2CacheHint node[%s] has multiple out data nodes.", node->GetNamePtr());
  GE_CHK_BOOL_RET_SPECIAL_STATUS(ascgen_utils::IsNodeCacheable(out_data_nodes.at(0)), true,
                                 "NeedSkipL2CacheHint node[%s] out node is cacheable.", node->GetNamePtr());
  GE_CHK_BOOL_RET_SPECIAL_STATUS(IsSmallTensor(out_node), true,
                                 "NeedSkipL2CacheHint node[%s] out node is small tensor.", node->GetNamePtr());
  GE_CHK_BOOL_RET_SPECIAL_STATUS(!af::ops::IsOps<af::ascir_op::Load>(out_node), true,
                                 "NeedSkipL2CacheHint node[%s] out node is not Load.", node->GetNamePtr());
  return false;
}

// 按index属性建立的IO节点映射: index -> 节点
struct IoNodeIndexMap {
  std::map<int64_t, af::NodePtr> inputs;
  std::map<int64_t, af::NodePtr> outputs;
};

// 优先从AscNode的ir_attr读取index; 否则回退到op_desc的AscNodeAttr(参考FusedGraphModifier::ProcessDataNodes)
bool GetNodeIndex(const af::NodePtr &node, int64_t &index) {
  const auto asc_node = std::dynamic_pointer_cast<af::AscNode>(node);
  if (asc_node != nullptr && asc_node->attr.ir_attr != nullptr) {
    return asc_node->attr.ir_attr->GetAttrValue("index", index) == af::GRAPH_SUCCESS;
  }
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto node_attr = op_desc->GetAttrsGroup<af::AscNodeAttr>();
  GE_WARN_ASSERT(node_attr != nullptr);
  GE_WARN_ASSERT(node_attr->ir_attr != nullptr);
  return node_attr->ir_attr->GetAttrValue("index", index) == af::GRAPH_SUCCESS;
}

// 遍历ComputeGraph，按index属性建立input(Data)/output(Output)节点映射
IoNodeIndexMap CollectIoNodesByIndex(const af::ComputeGraph &graph) {
  IoNodeIndexMap index_map;
  for (const auto &node : graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    const bool is_input = af::ops::IsOps<af::ascir_op::Data>(node);
    const bool is_output = af::ops::IsOps<af::ascir_op::Output>(node);
    if (!is_input && !is_output) {
      continue;
    }
    int64_t index = -1;
    if (!GetNodeIndex(node, index)) {
      GELOGW("L2Ctrl skip node[%s] without index attr.", node->GetNamePtr());
      continue;
    }
    if (is_input) {
      index_map.inputs[index] = node;
    } else {
      index_map.outputs[index] = node;
    }
  }
  return index_map;
}

// 计算节点输出tensor大小(字节, 符号化):
// total_size为outputs[0].attr.repeats中所有轴(含非常量符号)维度相乘再乘以dtype size的Expression;
// min_size为仅常量轴相乘(跳过非常量符号)再乘以dtype size的int64估算
af::Status CalcOutputTensorSizeExpr(const af::AscNodePtr &node, af::Expression &total_size, int64_t &min_size) {
  total_size = af::sym::kSymbolOne;
  min_size = 1;
  GE_ASSERT_NOTNULL(node, "CalcOutputTensorSizeExpr node is nullptr.");
  GE_ASSERT_TRUE(!node->outputs().empty(), "CalcOutputTensorSizeExpr node[%s] has no output.", node->GetNamePtr());
  const auto &repeats = node->outputs[0].attr.repeats;
  for (const auto &repeat : repeats) {
    total_size = total_size * repeat;
    if (repeat.IsConstExpr()) {
      int64_t dim = 0;
      GE_ASSERT_TRUE(repeat.GetConstValue(dim), "CalcOutputTensorSizeExpr node[%s] get const value failed.",
                     node->GetNamePtr());
      min_size *= dim;
    }
  }
  const auto dtype_size = ge::GetSizeByDataType(node->outputs[0].attr.dtype);
  total_size = total_size * af::Symbol(static_cast<int64_t>(dtype_size));
  min_size *= static_cast<int64_t>(dtype_size);
  GELOGD("CalcOutputTensorSizeExpr node:%s, total_size:%s bytes, min_size:%lld bytes.", node->GetNamePtr(),
         ExprToStr(total_size).c_str(), min_size);
  return af::SUCCESS;
}

// 从AscBackend后继的子图中, 查找index匹配的Data节点, 累加其后继的output size
af::Status CalcInputL2SizeFromAscBackend(const af::NodePtr &backend_node, int64_t input_index,
                                         af::Expression &input_size, int64_t &min_size) {
  const auto op_desc = backend_node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc, "AscBackend node[%s] has no OpDesc.", backend_node->GetNamePtr());
  const auto fuse_attr = op_desc->GetAttrsGroup<af::AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(fuse_attr, "AscBackend node[%s] has no AutoFuseAttrs.", backend_node->GetNamePtr());
  const auto &sub_asc_graph = fuse_attr->GetAscGraph();
  GE_ASSERT_NOTNULL(sub_asc_graph, "AscBackend node[%s] has no AscGraph.", backend_node->GetNamePtr());
  bool found = false;
  for (const auto &sub_node : sub_asc_graph->GetAllNodes()) {
    GE_ASSERT_NOTNULL(sub_node);
    if (!af::ops::IsOps<af::ascir_op::Data>(sub_node)) {
      continue;
    }
    int64_t sub_index = -1;
    if (!GetNodeIndex(sub_node, sub_index) || sub_index != input_index) {
      continue;
    }
    found = true;
    for (const auto &sub_successor : sub_node->GetOutDataNodes()) {
      const auto sub_successor_asc = std::dynamic_pointer_cast<af::AscNode>(sub_successor);
      if (sub_successor_asc == nullptr) {
        continue;
      }
      af::Expression sub_size = af::sym::kSymbolOne;
      int64_t sub_min = 0;
      GE_ASSERT_SUCCESS(CalcOutputTensorSizeExpr(sub_successor_asc, sub_size, sub_min));
      input_size = input_size + sub_size;
      min_size += sub_min;
    }
    GELOGD("L2Ctrl input[%lld] match AscBackend[%s] sub Data[%s].", input_index, backend_node->GetNamePtr(),
           sub_node->GetNamePtr());
    break;
  }
  if (!found) {
    GELOGW("L2Ctrl input[%lld] not found in AscBackend[%s] AscGraph.", input_index, backend_node->GetNamePtr());
  }
  return af::SUCCESS;
}

// 计算单个input节点的GM大小(符号化): 遍历其后继节点累加size;
// 若后继为AscBackend节点, 则进入其关联AscGraph中找index相同的Data节点, 改用该Data的后继累加
af::Status CalcInputL2SizeExpr(const af::NodePtr &input_node, int64_t input_index, af::Expression &input_size,
                               int64_t &min_size) {
  input_size = af::sym::kSymbolZero;
  min_size = 0;
  for (const auto &successor : input_node->GetOutDataNodes()) {
    if (successor == nullptr) {
      continue;
    }
    if (successor->GetType() == kAscBackendType) {
      GE_ASSERT_SUCCESS(CalcInputL2SizeFromAscBackend(successor, input_index, input_size, min_size));
    } else {
      const auto successor_asc = std::dynamic_pointer_cast<af::AscNode>(successor);
      if (successor_asc != nullptr) {
        af::Expression successor_size = af::sym::kSymbolOne;
        int64_t successor_min = 0;
        GE_ASSERT_SUCCESS(CalcOutputTensorSizeExpr(successor_asc, successor_size, successor_min));
        input_size = input_size + successor_size;
        min_size += successor_min;
      }
    }
  }
  return af::SUCCESS;
}

// 计算单个output节点的GM大小(符号化): 取其前驱计算size;
// 若前驱为AscBackend节点, 则进入其关联AscGraph中找index相同的Output节点, 改用该Output的前驱计算
af::Status CalcOutputL2SizeExpr(const af::NodePtr &output_node, int64_t output_index, af::Expression &output_size,
                                int64_t &min_size) {
  output_size = af::sym::kSymbolZero;
  min_size = 0;
  const auto &predecessors = output_node->GetInDataNodes();
  if (predecessors.empty()) {
    return af::SUCCESS;
  }
  const auto &predecessor = predecessors.at(0UL);
  GE_ASSERT_NOTNULL(predecessor, "CalcOutputL2SizeExpr predecessor is nullptr.");
  if (predecessor->GetType() != kAscBackendType) {
    const auto predecessor_asc = std::dynamic_pointer_cast<af::AscNode>(predecessor);
    if (predecessor_asc != nullptr) {
      GE_ASSERT_SUCCESS(CalcOutputTensorSizeExpr(predecessor_asc, output_size, min_size));
    }
    return af::SUCCESS;
  }
  const auto op_desc = predecessor->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc, "AscBackend node[%s] has no OpDesc.", predecessor->GetNamePtr());
  const auto fuse_attr = op_desc->GetAttrsGroup<af::AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(fuse_attr, "AscBackend node[%s] has no AutoFuseAttrs.", predecessor->GetNamePtr());
  const auto &sub_asc_graph = fuse_attr->GetAscGraph();
  GE_ASSERT_NOTNULL(sub_asc_graph, "AscBackend node[%s] has no AscGraph.", predecessor->GetNamePtr());
  for (const auto &sub_node : sub_asc_graph->GetAllNodes()) {
    GE_ASSERT_NOTNULL(sub_node);
    if (!af::ops::IsOps<af::ascir_op::Output>(sub_node)) {
      continue;
    }
    int64_t sub_index = -1;
    if (!GetNodeIndex(sub_node, sub_index) || sub_index != output_index) {
      continue;
    }
    const auto &sub_predecessors = sub_node->GetInDataNodes();
    if (!sub_predecessors.empty()) {
      const auto sub_predecessor_asc = std::dynamic_pointer_cast<af::AscNode>(sub_predecessors.at(0UL));
      if (sub_predecessor_asc != nullptr) {
        GE_ASSERT_SUCCESS(CalcOutputTensorSizeExpr(sub_predecessor_asc, output_size, min_size));
      }
    }
    GELOGD("L2Ctrl output[%lld] match AscBackend[%s] sub Output[%s].", output_index, predecessor->GetNamePtr(),
           sub_node->GetNamePtr());
    break;
  }
  return af::SUCCESS;
}

bool ExprInGraph(const af::Expression &expr, const std::set<std::string> &graph_size_var_names) {
  if (expr.IsConstExpr()) {
    return true;
  }
  for (const auto &sym : expr.FreeSymbols()) {
    auto str_ptr = sym.Str();
    GE_WARN_ASSERT(str_ptr != nullptr);
    std::string sym_name(str_ptr.get());
    if (graph_size_var_names.find(sym_name) == graph_size_var_names.end()) {
      GELOGI("ExprInGraph: symbol[%s] not found in current graph.", sym_name.c_str());
      return false;
    }
  }
  return true;
}
}  // namespace

namespace optimize {
af::Status L2CacheHintManager::GetL2Size(int64_t &l2_size) {
  ge::PlatformInfo platform_info;
  GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetPlatformInfo(platform_info));
  l2_size = platform_info.l2_size;
  return af::SUCCESS;
}

std::set<size_t> L2CacheHintManager::CollectSkipL2CacheHintIndices(const ascir::ImplGraph &graph) {
  std::set<size_t> skip_indices;
  for (const auto &node : graph.GetAllNodes()) {
    if (!af::ops::IsOps<af::ascir_op::Data>(node)) {
      continue;
    }
    if (!af::AttrUtils::HasAttr(node->GetOpDesc(), kSkipL2CacheHintAttr)) {
      continue;
    }
    int64_t index = -1;
    (void)node->attr.ir_attr->GetAttrValue("index", index);
    GELOGD("skip input index: %ld", index);
    skip_indices.insert(static_cast<size_t>(index));
  }
  return skip_indices;
}

bool L2CacheHintManager::AllExprSymbolsInGraph(const ascir::GmTensorSizes &sizes, const ascir::ImplGraph &graph) {
  std::set<std::string> graph_size_var_names;
  for (const auto &size_var : graph.GetAllSizeVar()) {
    graph_size_var_names.insert(std::string(size_var->expr.Str().get()));
  }
  if (!ExprInGraph(sizes.total_size, graph_size_var_names)) {
    return false;
  }
  for (const auto &output_size : sizes.output_sizes) {
    if (!ExprInGraph(output_size, graph_size_var_names)) {
      return false;
    }
  }
  return true;
}

af::Status L2CacheHintManager::ParseGraph(const af::ComputeGraph &graph,
                                          ::ascir::FusedScheduledResult &fused_scheduled_result) {
  // 暂不支持多个AscBackend的场景
  GE_CHK_BOOL_RET_SPECIAL_STATUS(fused_scheduled_result.node_idx_to_scheduled_results.size() > 1, af::SUCCESS,
                                 "ParseGraph skip: multiple AscBackend nodes");
  GE_ASSERT_SUCCESS(CalcTensorSizes(graph, fused_scheduled_result, fused_scheduled_result.gm_tensor_sizes));
  const auto &sizes = fused_scheduled_result.gm_tensor_sizes;
  int64_t total_size_value = 0;
  if (sizes.total_size.IsConstExpr() && sizes.total_size.GetConstValue(total_size_value)) {
    int64_t l2_size = -1;
    GE_ASSERT_SUCCESS(GetL2Size(l2_size));
    if (total_size_value <= l2_size) {
      GELOGD("ParseGraph skip: total_size[%lld] <= l2_size[%lld].", total_size_value, l2_size);
      return af::SUCCESS;
    }
  }
  GE_ASSERT_SUCCESS(MarkSingleBroadcastInlineSkip(graph, fused_scheduled_result));
  GE_ASSERT_SUCCESS(MarkInputsNeedSkipL2CacheHint(fused_scheduled_result));
  return af::SUCCESS;
}

af::Status L2CacheHintManager::CalcTensorSizes(const af::ComputeGraph &graph, const ::ascir::FusedScheduledResult &fsr,
                                               GmTensorSizes &global_tensor_sizes) {
  GELOGI("CalcTensorSizes start, graph:%s, input_nodes_num:%zu, output_nodes_num:%zu.", graph.GetName().c_str(),
         fsr.input_nodes.size(), fsr.output_nodes.size());
  global_tensor_sizes.total_size = af::sym::kSymbolZero;
  global_tensor_sizes.min_total_size = 0;
  global_tensor_sizes.input_sizes.clear();
  global_tensor_sizes.output_sizes.clear();
  const auto index_map = CollectIoNodesByIndex(graph);

  global_tensor_sizes.input_sizes.reserve(fsr.input_nodes.size());
  for (size_t i = 0; i < fsr.input_nodes.size(); ++i) {
    af::Expression input_size = af::sym::kSymbolZero;
    int64_t min_size = 0;
    const auto it = index_map.inputs.find(static_cast<int64_t>(i));
    if (it != index_map.inputs.end()) {
      const auto &input_node = it->second;
      GE_ASSERT_SUCCESS(CalcInputL2SizeExpr(input_node, static_cast<int64_t>(i), input_size, min_size));
      GELOGD("CalcTensorSizes input[%zu] node:%s, total_size:%s, min_size:%lld.", i, input_node->GetNamePtr(),
             ExprToStr(input_size).c_str(), min_size);
    } else {
      GELOGW("CalcTensorSizes input[%zu] not found in graph, tensor_size set to 0.", i);
    }
    global_tensor_sizes.input_sizes.push_back(input_size);
    global_tensor_sizes.total_size = global_tensor_sizes.total_size + input_size;
    global_tensor_sizes.min_total_size += min_size;
  }

  global_tensor_sizes.output_sizes.reserve(fsr.output_nodes.size());
  for (size_t i = 0UL; i < fsr.output_nodes.size(); ++i) {
    af::Expression output_size = af::sym::kSymbolZero;
    int64_t min_size = 0;
    const auto it = index_map.outputs.find(static_cast<int64_t>(i));
    if (it != index_map.outputs.end()) {
      const auto &output_node = it->second;
      GE_ASSERT_SUCCESS(CalcOutputL2SizeExpr(output_node, static_cast<int64_t>(i), output_size, min_size));
      GELOGD("CalcTensorSizes output[%zu] node:%s, total_size:%s, min_size:%lld.", i, output_node->GetNamePtr(),
             ExprToStr(output_size).c_str(), min_size);
    } else {
      GELOGW("CalcTensorSizes output[%zu] not found in graph, tensor_size set to 0.", i);
    }
    global_tensor_sizes.output_sizes.push_back(output_size);
    global_tensor_sizes.total_size = global_tensor_sizes.total_size + output_size;
    global_tensor_sizes.min_total_size += min_size;
  }

  GELOGI("CalcTensorSizes end, total_size:%s, min_total_size:%lld.", ExprToStr(global_tensor_sizes.total_size).c_str(),
         global_tensor_sizes.min_total_size);
  return af::SUCCESS;
}

af::Status L2CacheHintManager::MarkInternal(af::AscGraph &impl_graph) {
  GELOGD("MarkInternal start, graph:%s.", impl_graph.GetName().c_str());
  std::map<int64_t, std::vector<af::AscNodePtr>> index_to_data_nodes;
  for (const auto &node : impl_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node, "MarkInternal node is nullptr.");
    if (!af::ops::IsOps<af::ascir_op::Data>(node)) {
      continue;
    }
    int64_t index = -1;
    GE_ASSERT_TRUE(GetNodeIndex(node, index), "MarkInternal get index attr failed, node[%s].", node->GetNamePtr());
    index_to_data_nodes[index].emplace_back(std::dynamic_pointer_cast<af::AscNode>(node));
  }

  // 再次遍历所有Data节点: 若该节点的index对应多个Data节点, 或其输出节点满足跳过条件(见NeedSkipL2CacheHint), 则标记
  for (const auto &node : impl_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node, "MarkInternal node is nullptr.");
    if (!af::ops::IsOps<af::ascir_op::Data>(node)) {
      continue;
    }
    int64_t index = -1;
    GE_ASSERT_TRUE(GetNodeIndex(node, index), "MarkInternal get index attr failed, node[%s].", node->GetNamePtr());
    const auto &out_data_nodes = node->GetOutDataNodes();
    if (out_data_nodes.empty()) {
      continue;
    }
    const bool need_skip = (index_to_data_nodes[index].size() > 1UL) || NeedSkipL2CacheHint(node);
    if (!need_skip) {
      continue;
    }
    GE_ASSERT_TRUE(af::AttrUtils::SetBool(node->GetOpDesc(), kSkipL2CacheHintAttr, true),
                   "MarkInternal set _skip_l2_cache_hint failed, node[%s].", node->GetNamePtr());
    GELOGD("MarkInternal set _skip_l2_cache_hint on Data node[%s], index:%lld.", node->GetNamePtr(), index);
  }
  GELOGD("MarkInternal end, graph:%s.", impl_graph.GetName().c_str());
  return af::SUCCESS;
}

af::Status L2CacheHintManager::MarkInputsNeedSkipL2CacheHint(::ascir::FusedScheduledResult &fused_scheduled_result) {
  for (auto &scheduled_results : fused_scheduled_result.node_idx_to_scheduled_results) {
    for (auto &scheduled_result : scheduled_results) {
      for (auto &schedule_group : scheduled_result.schedule_groups) {
        for (auto &impl_graph : schedule_group.impl_graphs) {
          GE_ASSERT_SUCCESS(MarkInternal(impl_graph));
        }
      }
    }
  }
  return af::SUCCESS;
}

// 同维场景：恰一位 1->2，其余轴全等
bool IsSameDimSingleAxisTimesTwo(const std::vector<af::Expression> &input_repeats,
                                 const std::vector<af::Expression> &output_repeats) {
  size_t diff_count = 0UL;
  for (size_t i = 0UL; i < output_repeats.size(); ++i) {
    if (ascgen_utils::ExpressEq(input_repeats[i], output_repeats[i])) {
      continue;
    }
    ++diff_count;
    int64_t input_dim = 0;
    int64_t output_dim = 0;
    if (diff_count > 1UL || !input_repeats[i].IsConstExpr() || !input_repeats[i].GetConstValue(input_dim) ||
        !output_repeats[i].IsConstExpr() || !output_repeats[i].GetConstValue(output_dim) ||
        input_dim != kPreBroadcastDimSize || output_dim != kBroadcastFactor) {
      return false;
    }
  }
  return diff_count == 1UL;
}

// 插轴场景：删除输出某根常量2轴后与输入序列一致
bool IsInsertedAxisTimesTwo(const std::vector<af::Expression> &input_repeats,
                            const std::vector<af::Expression> &output_repeats) {
  for (size_t erased = 0UL; erased < output_repeats.size(); ++erased) {
    int64_t erased_dim = 0;
    if (!output_repeats[erased].IsConstExpr() || !output_repeats[erased].GetConstValue(erased_dim) ||
        erased_dim != kBroadcastFactor) {
      continue;
    }
    bool matched = true;
    size_t input_idx = 0UL;
    for (size_t output_idx = 0UL; output_idx < output_repeats.size(); ++output_idx) {
      if (output_idx == erased) {
        continue;
      }
      if (input_idx >= input_repeats.size() ||
          !ascgen_utils::ExpressEq(input_repeats[input_idx], output_repeats[output_idx])) {
        matched = false;
        break;
      }
      ++input_idx;
    }
    if (matched && input_idx == input_repeats.size()) {
      return true;
    }
  }
  return false;
}

// 判定Broadcast节点是否只含一根广播轴且广播成2：
// 同维场景恰一位 1->2；插轴(unsqueeze+expand)场景删除输出的某根常量2轴后与输入序列一致。
// 任一环节缺失按"不满足"处理，不升级为错误。
bool L2CacheHintManager::IsSingleBroadcastAxisTimesTwo(const af::AscNodePtr &broadcast_node) {
  if (broadcast_node == nullptr || broadcast_node->outputs().empty()) {
    return false;
  }
  const auto in_anchor = broadcast_node->GetInDataAnchor(0);
  if (in_anchor == nullptr || in_anchor->GetPeerOutAnchor() == nullptr) {
    return false;
  }
  const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  const auto src_node = std::dynamic_pointer_cast<af::AscNode>(peer_out_anchor->GetOwnerNode());
  if (src_node == nullptr || src_node->outputs().empty() ||
      static_cast<size_t>(peer_out_anchor->GetIdx()) >= src_node->outputs().size()) {
    return false;
  }
  const auto &input_repeats = src_node->outputs[peer_out_anchor->GetIdx()].attr.repeats;
  const auto &output_repeats = broadcast_node->outputs[0].attr.repeats;
  if (input_repeats.empty() || output_repeats.empty()) {
    return false;
  }
  if (input_repeats.size() == output_repeats.size()) {
    return IsSameDimSingleAxisTimesTwo(input_repeats, output_repeats);
  }
  if (output_repeats.size() == input_repeats.size() + 1UL) {
    return IsInsertedAxisTimesTwo(input_repeats, output_repeats);
  }
  return false;
}

// 单broadcast inline融合场景豁免：单输入单输出、图中恰一个Broadcast节点、其输入物理大小恰为
// L2的一半且输出恰为L2大小、只有一根广播轴并广播成2。此时输入整体驻留L2(半容量)，广播复读
// 可被L2捕获，禁用输入L2反而损失复读命中；给输入Data节点打_skip_l2_cache_hint跳过
// SetL2CacheHint(DISABLE)代码生成，保留L2。
af::Status L2CacheHintManager::MarkSingleBroadcastInlineSkip(const af::ComputeGraph &graph,
                                                             ::ascir::FusedScheduledResult &fused_scheduled_result) {
  const auto &sizes = fused_scheduled_result.gm_tensor_sizes;
  if (sizes.input_sizes.size() != 1UL || sizes.output_sizes.size() != 1UL) {
    GELOGD("L2Ctrl single broadcast skip rejected: input_sizes[%zu] or output_sizes[%zu] is not 1.",
           sizes.input_sizes.size(), sizes.output_sizes.size());
    return af::SUCCESS;
  }
  std::vector<af::NodePtr> analysis_nodes;
  GE_ASSERT_SUCCESS(CollectAnalysisNodes(graph, analysis_nodes));
  af::AscNodePtr broadcast_node = nullptr;
  GE_ASSERT_SUCCESS(FindSingleBroadcastNode(analysis_nodes, broadcast_node));
  if (broadcast_node == nullptr) {
    return af::SUCCESS;
  }
  int64_t input_bytes = 0;
  int64_t output_bytes = 0;
  GE_ASSERT_SUCCESS(ValidateBroadcastInputSize(broadcast_node, input_bytes, output_bytes));
  if (input_bytes <= 0) {
    return af::SUCCESS;
  }
  int64_t l2_size = 0;
  GE_ASSERT_SUCCESS(GetL2Size(l2_size));
  MarkSkipHintOnDataNodes(fused_scheduled_result, input_bytes, output_bytes, l2_size);
  return af::SUCCESS;
}

// 识别分析图层级并收集分析节点：inductor路径穿透AscBackend取融合子图，GE在线直接取本图
af::Status L2CacheHintManager::CollectAnalysisNodes(const af::ComputeGraph &graph,
                                                    std::vector<af::NodePtr> &analysis_nodes) {
  bool has_outer_broadcast = false;
  af::NodePtr asc_backend_node = nullptr;
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetType() == kAscBackendType) {
      if (asc_backend_node != nullptr) {
        GELOGD("L2Ctrl single broadcast skip rejected: multiple AscBackend nodes.");
        return af::SUCCESS;
      }
      asc_backend_node = node;
      continue;
    }
    if (af::ops::IsOps<af::ascir_op::Broadcast>(node)) {
      has_outer_broadcast = true;
      continue;
    }
    if (!af::ops::IsOps<af::ascir_op::Data>(node) && !af::ops::IsOps<af::ascir_op::Load>(node) &&
        !af::ops::IsOps<af::ascir_op::Store>(node) && !af::ops::IsOps<af::ascir_op::Output>(node)) {
      GELOGD("L2Ctrl single broadcast skip rejected: non-pure node[%s, %s].", node->GetNamePtr(), node->GetTypePtr());
      return af::SUCCESS;
    }
  }
  if (asc_backend_node != nullptr) {
    if (has_outer_broadcast) {
      GELOGD("L2Ctrl single broadcast skip rejected: mixed outer Broadcast with AscBackend.");
      return af::SUCCESS;
    }
    const auto op_desc = asc_backend_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const auto fuse_attr = op_desc->GetAttrsGroup<af::AutoFuseAttrs>();
    if (fuse_attr == nullptr || fuse_attr->GetAscGraph() == nullptr) {
      GELOGD("L2Ctrl single broadcast skip rejected: AscBackend node[%s] has no AscGraph.",
             asc_backend_node->GetNamePtr());
      return af::SUCCESS;
    }
    for (const auto &inner_node : fuse_attr->GetAscGraph()->GetAllNodes()) {
      analysis_nodes.push_back(inner_node);
    }
    return af::SUCCESS;
  }
  for (const auto &outer_node : graph.GetAllNodes()) {
    analysis_nodes.push_back(outer_node);
  }
  return af::SUCCESS;
}

// 节点数快速否决+白名单校验+定位唯一Broadcast节点
af::Status L2CacheHintManager::FindSingleBroadcastNode(const std::vector<af::NodePtr> &analysis_nodes,
                                                       af::AscNodePtr &broadcast_node) {
  constexpr size_t kMaxPureBroadcastGraphNodes = 5UL;
  if (analysis_nodes.size() > kMaxPureBroadcastGraphNodes) {
    GELOGD("L2Ctrl single broadcast skip rejected: node count[%zu] exceeds %zu.", analysis_nodes.size(),
           kMaxPureBroadcastGraphNodes);
    return af::SUCCESS;
  }
  broadcast_node = nullptr;
  for (const auto &node : analysis_nodes) {
    if (af::ops::IsOps<af::ascir_op::Broadcast>(node)) {
      if (broadcast_node != nullptr) {
        GELOGD("L2Ctrl single broadcast skip rejected: multiple Broadcast nodes.");
        return af::SUCCESS;
      }
      broadcast_node = std::dynamic_pointer_cast<af::AscNode>(node);
      continue;
    }
    if (!af::ops::IsOps<af::ascir_op::Data>(node) && !af::ops::IsOps<af::ascir_op::Load>(node) &&
        !af::ops::IsOps<af::ascir_op::Store>(node) && !af::ops::IsOps<af::ascir_op::Output>(node)) {
      GELOGD("L2Ctrl single broadcast skip rejected: non-pure node[%s, %s].", node->GetNamePtr(), node->GetTypePtr());
      return af::SUCCESS;
    }
  }
  if (broadcast_node == nullptr) {
    GELOGD("L2Ctrl single broadcast skip rejected: no Broadcast node.");
    return af::SUCCESS;
  }
  if (!IsSingleBroadcastAxisTimesTwo(broadcast_node)) {
    GELOGD("L2Ctrl single broadcast skip rejected: broadcast node[%s] is not single axis times two.",
           broadcast_node->GetNamePtr());
    broadcast_node = nullptr;
    return af::SUCCESS;
  }
  return af::SUCCESS;
}

// 解析Broadcast输入的锚点链并溯源到Data节点；不可溯源返回nullptr
af::AscNodePtr L2CacheHintManager::ResolveBroadcastInputSource(const af::AscNodePtr &broadcast_node) {
  if (broadcast_node == nullptr || broadcast_node->outputs().empty()) {
    GELOGD("L2Ctrl single broadcast skip rejected: invalid broadcast node.");
    return nullptr;
  }
  const auto in_anchor = broadcast_node->GetInDataAnchor(0);
  if (in_anchor == nullptr || in_anchor->GetPeerOutAnchor() == nullptr) {
    GELOGD("L2Ctrl single broadcast skip rejected: broadcast node[%s] has null anchor.", broadcast_node->GetNamePtr());
    return nullptr;
  }
  const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  const auto src_node = std::dynamic_pointer_cast<af::AscNode>(peer_out_anchor->GetOwnerNode());
  if (src_node == nullptr || src_node->outputs().empty() ||
      static_cast<size_t>(peer_out_anchor->GetIdx()) >= src_node->outputs().size()) {
    GELOGD("L2Ctrl single broadcast skip rejected: invalid source node of broadcast node[%s].",
           broadcast_node->GetNamePtr());
    return nullptr;
  }
  // 溯源校验：Broadcast输入沿单输入链上溯到达Data节点（图输入）
  af::AscNodePtr cursor = src_node;
  while ((cursor != nullptr) && !af::ops::IsOps<af::ascir_op::Data>(cursor)) {
    const auto &predecessors = cursor->GetInDataNodes();
    if (predecessors.size() != 1UL) {
      cursor = nullptr;
      break;
    }
    cursor = std::dynamic_pointer_cast<af::AscNode>(predecessors.at(0UL));
  }
  if (cursor == nullptr) {
    GELOGD("L2Ctrl single broadcast skip rejected: broadcast input of node[%s] does not trace to Data.",
           broadcast_node->GetNamePtr());
    return nullptr;
  }
  return src_node;
}

// 校验广播输入/输出物理大小：输入==L2/2、输出==L2
af::Status L2CacheHintManager::ValidateBroadcastInputSize(const af::AscNodePtr &broadcast_node, int64_t &input_bytes,
                                                          int64_t &output_bytes) {
  input_bytes = 0;
  output_bytes = 0;
  const auto src_node = ResolveBroadcastInputSource(broadcast_node);
  if (src_node == nullptr) {
    return af::SUCCESS;
  }
  const auto in_anchor = broadcast_node->GetInDataAnchor(0);
  const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  int64_t l2_size = 0;
  GE_ASSERT_SUCCESS(GetL2Size(l2_size));
  if (l2_size <= 0 || l2_size % kHalfL2Divider != 0) {
    GELOGD("L2Ctrl single broadcast skip rejected: invalid l2_size[%lld].", l2_size);
    return af::SUCCESS;
  }
  if (!CalcAttrSizeBytes(src_node->outputs[peer_out_anchor->GetIdx()].attr, input_bytes) ||
      !CalcAttrSizeBytes(broadcast_node->outputs[0].attr, output_bytes) || input_bytes * kBroadcastFactor != l2_size ||
      output_bytes != l2_size) {
    GELOGD(
        "L2Ctrl single broadcast skip rejected: size mismatch, input_bytes[%lld], output_bytes[%lld], "
        "l2_size[%lld].",
        input_bytes, output_bytes, l2_size);
    input_bytes = 0;
    output_bytes = 0;
    return af::SUCCESS;
  }
  return af::SUCCESS;
}

// 给单个节点打_skip_l2_cache_hint标记（仅Data类型）
static void MarkDataNodeIfData(const af::NodePtr &node, int64_t input_bytes, int64_t output_bytes, int64_t l2_size) {
  if (!af::ops::IsOps<af::ascir_op::Data>(node)) {
    return;
  }
  (void)af::AttrUtils::SetBool(node->GetOpDesc(), kSkipL2CacheHintAttr, true);
  GELOGI(
      "L2Ctrl single broadcast inline scene, keep L2 cache for input[%s], input_bytes[%lld], "
      "output_bytes[%lld], l2_size[%lld].",
      node->GetNamePtr(), input_bytes, output_bytes, l2_size);
}

// 给单个impl graph内所有Data节点打标记
static af::Status MarkDataNodesInGraph(const af::AscGraph &impl_graph, int64_t input_bytes, int64_t output_bytes,
                                       int64_t l2_size) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    MarkDataNodeIfData(node, input_bytes, output_bytes, l2_size);
  }
  return af::SUCCESS;
}

// 给所有impl graph的Data节点打_skip_l2_cache_hint标记
af::Status L2CacheHintManager::MarkSkipHintOnDataNodes(::ascir::FusedScheduledResult &fused_scheduled_result,
                                                       int64_t input_bytes, int64_t output_bytes, int64_t l2_size) {
  for (const auto &scheduled_results : fused_scheduled_result.node_idx_to_scheduled_results) {
    for (const auto &scheduled_result : scheduled_results) {
      for (const auto &schedule_group : scheduled_result.schedule_groups) {
        for (const auto &impl_graph : schedule_group.impl_graphs) {
          (void)MarkDataNodesInGraph(impl_graph, input_bytes, output_bytes, l2_size);
        }
      }
    }
  }
  return af::SUCCESS;
}
}  // namespace optimize
