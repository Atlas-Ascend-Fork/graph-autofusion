/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dtype_consistency.h"
#include "v35/ascir/ascir_codegen_v2.h"
#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "common_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph_utils.h"
#include "node_utils.h"
#include "schedule_utils.h"
#include "graph_pass/pass_utils.h"

using namespace af::ascir_op;

namespace {
bool IsSignedIntegerType(af::DataType dtype) {
  return (dtype == ge::DT_INT8) || (dtype == ge::DT_INT16) || (dtype == ge::DT_INT32) || (dtype == ge::DT_INT64) ||
         (dtype == ge::DT_INT4) || (dtype == ge::DT_INT2);
}

bool IsUnsignedIntegerType(af::DataType dtype) {
  return (dtype == ge::DT_UINT8) || (dtype == ge::DT_UINT16) || (dtype == ge::DT_UINT32) || (dtype == ge::DT_UINT64) ||
         (dtype == ge::DT_UINT1) || (dtype == ge::DT_UINT2);
}

bool IsFloatingType(af::DataType dtype) {
  return (dtype == ge::DT_FLOAT16) || (dtype == ge::DT_FLOAT) || (dtype == ge::DT_DOUBLE) || (dtype == ge::DT_BF16);
}

int32_t GetDTypeSize(af::DataType dtype) {
  return ge::GetSizeByDataType(dtype);
}

// 判断 from_type -> to_type 的 Cast 是否保持所有值不变
// 规则1：同一类型类别内位宽递增（原规则）
// 规则2：位宽递增（用于 TryMergeWithUpstreamCast 中 A->B 和 A->C 都递增的场景）
bool IsCastPreservesValues(af::DataType from_type, af::DataType to_type) {
  if (from_type == to_type) {
    return true;
  }

  // 浮点类型扩展：fp16 -> fp32 -> fp64 保持值
  if (IsFloatingType(from_type) && IsFloatingType(to_type)) {
    return GetDTypeSize(from_type) <= GetDTypeSize(to_type);
  }

  // 有符号整数扩展：int8 -> int16 -> int32 -> int64 保持值
  if (IsSignedIntegerType(from_type) && IsSignedIntegerType(to_type)) {
    return GetDTypeSize(from_type) <= GetDTypeSize(to_type);
  }

  // 无符号整数扩展：uint8 -> uint16 -> uint32 -> uint64 保持值
  if (IsUnsignedIntegerType(from_type) && IsUnsignedIntegerType(to_type)) {
    return GetDTypeSize(from_type) <= GetDTypeSize(to_type);
  }
  return false;
}

bool IsDTypeSizeChainIncreasing(af::DataType dtype_a, af::DataType dtype_b, af::DataType dtype_c) {
  auto width_a = ge::GetSizeByDataType(dtype_a);
  auto width_b = ge::GetSizeByDataType(dtype_b);
  auto width_c = ge::GetSizeByDataType(dtype_c);
  return width_a < width_b && width_b < width_c;
}

Status RemoveDuplicateCast(const af::AscNodePtr &keep_cast, const af::AscNodePtr &remove_cast,
                           const af::OutDataAnchorPtr &src_out_anchor) {
  auto keep_out_anchor = keep_cast->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(keep_out_anchor);

  auto remove_out_anchor = remove_cast->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(remove_out_anchor);

  // Reconnect remove_cast's downstream to keep_cast
  auto dst_anchors = remove_out_anchor->GetPeerInDataAnchors();
  for (auto &dst_anchor : dst_anchors) {
    GE_ASSERT_SUCCESS(af::GraphUtils::RemoveEdge(remove_out_anchor, dst_anchor));
    GE_ASSERT_SUCCESS(af::GraphUtils::AddEdge(keep_out_anchor, dst_anchor));
  }

  // Remove remove_cast
  auto remove_in_anchor = remove_cast->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(remove_in_anchor);
  GE_ASSERT_SUCCESS(af::GraphUtils::RemoveEdge(src_out_anchor, remove_in_anchor));
  auto owner_graph = remove_cast->GetOwnerComputeGraph();
  GE_ASSERT_SUCCESS(owner_graph->RemoveNode(remove_cast));

  return af::SUCCESS;
}
}  // namespace

namespace optimize {
Status DtypeConsistency::EnsureDtypeConsistency(af::AscGraph &graph) {
  std::vector<NodeDtypeRequirement> requirements;
  GE_ASSERT_SUCCESS(CollectDtypeRequirements(graph, requirements), "Failed to collect dtype requirements");

  GE_ASSERT_SUCCESS(ApplyDtypeConversions(graph, requirements), "Failed to apply dtype conversions");

  GE_ASSERT_SUCCESS(CancelRedundantCast(graph), "Failed to cancel redundant cast");
  GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  return af::SUCCESS;
}

Status DtypeConsistency::CollectDtypeRequirements(af::AscGraph &graph,
                                                  std::vector<NodeDtypeRequirement> &requirements) {
  auto all_nodes = graph.GetAllNodes();
  for (const auto &node : all_nodes) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    GE_ASSERT_NOTNULL(node);
    auto codegen_impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
    GE_ASSERT_NOTNULL(codegen_impl, "Cannot find impl for ir type:[%s].", node->GetTypePtr());
    auto [input_dtypes, output_dtypes] = codegen_impl->GetConversionDtype(*node);
    auto *codegen_impl_v2 = dynamic_cast<af::ascir::AscIrCodegenV2 *>(codegen_impl.get());
    std::vector<std::vector<af::DataType>> output_cast_paths;
    if (codegen_impl_v2 != nullptr) {
      output_cast_paths = codegen_impl_v2->GetOutputConversionPath(*node);
    }
    requirements.push_back({node, input_dtypes, output_dtypes, output_cast_paths});
  }
  return af::SUCCESS;
}

Status DtypeConsistency::ApplyDtypeConversions(af::AscGraph &graph,
                                               const std::vector<NodeDtypeRequirement> &requirements) {
  for (const auto &req : requirements) {
    GE_CHK_STATUS_RET(ProcessOutputDtype(req), "Failed to process output dtype");
    GE_CHK_STATUS_RET(ProcessOutputCastPath(graph, req), "Failed to process output cast path");
    GE_CHK_STATUS_RET(ProcessInputDtype(graph, req), "Failed to process input dtype for node %s.",
                      req.node->GetNamePtr());
  }
  return af::SUCCESS;
}

Status DtypeConsistency::ProcessOutputDtype(const NodeDtypeRequirement &req) {
  size_t output_nums = req.node->outputs().size();
  for (size_t i = 0UL; i < output_nums; ++i) {
    GE_ASSERT_TRUE(i < req.output_dtypes.size());
    if (req.node->outputs[i].attr.dtype != req.output_dtypes[i]) {
      GELOGD("Node [%s]'s output[%zu] needs to change dtype from [%s] to [%s].", req.node->GetNamePtr(), i,
             ge::TypeUtils::DataTypeToSerialString(req.node->outputs[i].attr.dtype).c_str(),
             ge::TypeUtils::DataTypeToSerialString(req.output_dtypes[i]).c_str());
      req.node->outputs[i].attr.dtype = req.output_dtypes[i];
    }
  }
  return af::SUCCESS;
}

// Insert each ASCIR codegen output cast path as consecutive Cast nodes when direct conversion cannot preserve
// semantics.
Status DtypeConsistency::ProcessOutputCastPath(af::AscGraph &graph, const NodeDtypeRequirement &req) {
  GE_ASSERT_TRUE(req.output_cast_paths.size() <= req.node->outputs().size());
  for (size_t output_idx = 0UL; output_idx < req.output_cast_paths.size(); ++output_idx) {
    const auto &cast_path = req.output_cast_paths[output_idx];
    if (cast_path.empty()) {
      continue;
    }

    auto src_out_anchor = req.node->GetOutDataAnchor(static_cast<int32_t>(output_idx));
    GE_ASSERT_NOTNULL(src_out_anchor);
    auto src_node = std::dynamic_pointer_cast<af::AscNode>(src_out_anchor->GetOwnerNode());
    GE_ASSERT_NOTNULL(src_node);

    for (size_t cast_idx = 0UL; cast_idx < cast_path.size(); ++cast_idx) {
      const size_t src_output_idx = static_cast<size_t>(src_out_anchor->GetIdx());
      auto src_dtype = src_node->outputs[src_output_idx].attr.dtype;
      auto dst_dtype = cast_path[cast_idx];
      std::vector<af::DataType> cast_input_dtypes = {src_dtype};
      std::vector<af::DataType> cast_output_dtypes = {dst_dtype};
      GE_ASSERT_TRUE(ScheduleUtils::CallAscirInferDataType<af::ascir_op::Cast>(cast_input_dtypes, cast_output_dtypes) ==
                         af::SUCCESS,
                     "Node [%s] output [%zu] unsupported cast path from [%s] to [%s].", req.node->GetNamePtr(),
                     output_idx, ge::TypeUtils::DataTypeToSerialString(src_dtype).c_str(),
                     ge::TypeUtils::DataTypeToSerialString(dst_dtype).c_str());

      const auto peer_in_anchor_range = src_out_anchor->GetPeerInDataAnchors();
      const std::vector<af::InDataAnchorPtr> peer_in_anchors(peer_in_anchor_range.begin(), peer_in_anchor_range.end());
      if (peer_in_anchors.empty()) {
        GELOGD("Node [%s] output [%zu] has no consumer, skip output cast path.", req.node->GetNamePtr(), output_idx);
        break;
      }

      std::string cast_name = std::string(src_node->GetName()) + "_cast_to_" +
                              ge::TypeUtils::DataTypeToSerialString(dst_dtype) + "_" + std::to_string(cast_idx);
      Cast cast_node(cast_name.c_str());
      auto cast_node_ptr = graph.AddNode(cast_node);
      GE_ASSERT_NOTNULL(cast_node_ptr);
      cast_node_ptr->outputs[0].attr = src_node->outputs[src_output_idx].attr;
      cast_node_ptr->outputs[0].attr.dtype = dst_dtype;
      GE_ASSERT_SUCCESS(af::AscGraphUtils::InsertNodeAfter(src_out_anchor, peer_in_anchors, cast_node_ptr));

      src_node = cast_node_ptr;
      src_out_anchor = cast_node_ptr->GetOutDataAnchor(0);
      GE_ASSERT_NOTNULL(src_out_anchor);
    }
  }
  return af::SUCCESS;
}

Status DtypeConsistency::ProcessInputDtype(af::AscGraph &graph, const NodeDtypeRequirement &req) {
  size_t input_nums = req.node->inputs().size();
  for (size_t i = 0UL; i < input_nums; ++i) {
    GE_ASSERT_TRUE(i < req.input_dtypes.size());
    if (req.input_dtypes[i] == req.node->inputs[i].attr.dtype) {
      continue;
    }

    auto in_anchor = req.node->GetInDataAnchor(static_cast<int32_t>(i));
    GE_ASSERT_NOTNULL(in_anchor);
    auto src_out_anchor = in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(src_out_anchor);

    auto src_node = std::dynamic_pointer_cast<af::AscNode>(src_out_anchor->GetOwnerNode());
    GE_ASSERT_NOTNULL(src_node);

    auto src_dtype = src_node->outputs[static_cast<int32_t>(src_out_anchor->GetIdx())].attr.dtype;
    auto dst_dtype = req.input_dtypes[i];

    // Check if cast conversion is supported
    GE_CHK_STATUS_RET(CheckCastSupported(src_dtype, dst_dtype, req.node, i), "Node:%s check cast supported failed.",
                      req.node->GetNamePtr());

    // If upstream is already a cast, try to merge two casts
    if (af::ops::IsOps<Cast>(src_node)) {
      if (TryMergeWithUpstreamCast(graph, src_node, req.node, i, dst_dtype)) {
        continue;
      }
    }

    // Insert a new cast node
    GE_CHK_STATUS_RET(InsertCastNode(graph, src_node, req.node, i, dst_dtype), "Insert cast node failed");
  }
  return af::SUCCESS;
}

Status DtypeConsistency::CheckCastSupported(af::DataType src_dtype, af::DataType dst_dtype, const af::AscNodePtr &node,
                                            size_t input_idx) {
  std::vector<af::DataType> cast_input_dtypes = {src_dtype};
  std::vector<af::DataType> cast_output_dtypes = {dst_dtype};
  auto infer_ret = ScheduleUtils::CallAscirInferDataType<af::ascir_op::Cast>(cast_input_dtypes, cast_output_dtypes);
  if (infer_ret != af::SUCCESS) {
    GELOGE(af::FAILED, "Failed to insert cast for node [%s] input [%zu]: cast from [%s] to [%s] is not supported.",
           node->GetNamePtr(), input_idx, ge::TypeUtils::DataTypeToSerialString(src_dtype).c_str(),
           ge::TypeUtils::DataTypeToSerialString(dst_dtype).c_str());
    return af::FAILED;
  }
  return af::SUCCESS;
}

bool DtypeConsistency::TryMergeWithUpstreamCast(af::AscGraph &graph, const af::AscNodePtr &upstream_cast,
                                                const af::AscNodePtr &downstream_node, size_t input_idx,
                                                af::DataType target_dtype) {
  auto orig_src_dtype = upstream_cast->inputs[0].attr.dtype;
  auto intermediate_dtype = upstream_cast->outputs[0].attr.dtype;
  // 合并条件（满足其一即可）：
  // 1. 同一类型类别内位宽递增（A->B 和 A->C 都满足 IsCastPreservesValues）
  // 2. 位宽始终递增
  bool same_category_widening =
      IsCastPreservesValues(orig_src_dtype, intermediate_dtype) && IsCastPreservesValues(orig_src_dtype, target_dtype);
  bool bitwidth_chain_increasing = IsDTypeSizeChainIncreasing(orig_src_dtype, intermediate_dtype, target_dtype);
  if (!same_category_widening && !bitwidth_chain_increasing) {
    return false;
  }

  // 额外检查：合并后的 A->C 必须是合法的 Cast
  std::vector<af::DataType> merge_input_dtypes = {orig_src_dtype};
  std::vector<af::DataType> merge_output_dtypes = {target_dtype};
  if ((orig_src_dtype != target_dtype) && ScheduleUtils::CallAscirInferDataType<af::ascir_op::Cast>(
                                              merge_input_dtypes, merge_output_dtypes) != af::SUCCESS) {
    return false;
  }

  auto orig_cast_out_anchor = upstream_cast->GetOutDataAnchor(0);
  bool has_multiple_consumers = orig_cast_out_anchor->GetPeerInDataAnchorsPtr().size() > 1U;
  auto in_anchor = downstream_node->GetInDataAnchor(static_cast<int32_t>(input_idx));
  GE_ASSERT_NOTNULL(in_anchor);

  if (!has_multiple_consumers) {
    return MergeCastWithSingleConsumer(upstream_cast, downstream_node, input_idx, target_dtype);
  }
  return MergeCastWithMultipleConsumers(graph, upstream_cast, downstream_node, input_idx, target_dtype);
}

bool DtypeConsistency::MergeCastWithSingleConsumer(const af::AscNodePtr &upstream_cast,
                                                   const af::AscNodePtr &downstream_node, size_t input_idx,
                                                   af::DataType target_dtype) {
  auto orig_src_dtype = upstream_cast->inputs[0].attr.dtype;
  GELOGD("Merge cast (single consumer) for node [%s] input [%zu]: [%s] -> [%s]", downstream_node->GetNamePtr(),
         input_idx, ge::TypeUtils::DataTypeToSerialString(orig_src_dtype).c_str(),
         ge::TypeUtils::DataTypeToSerialString(target_dtype).c_str());
  upstream_cast->outputs[0].attr.dtype = target_dtype;
  downstream_node->inputs[input_idx].attr.dtype = target_dtype;
  return true;
}

bool DtypeConsistency::MergeCastWithMultipleConsumers(af::AscGraph &graph, const af::AscNodePtr &upstream_cast,
                                                      const af::AscNodePtr &downstream_node, size_t input_idx,
                                                      af::DataType target_dtype) {
  auto orig_src_dtype = upstream_cast->inputs[0].attr.dtype;
  GELOGD("Merge cast (multiple consumers) for node [%s] input [%zu]: [%s] -> [%s]", downstream_node->GetNamePtr(),
         input_idx, ge::TypeUtils::DataTypeToSerialString(orig_src_dtype).c_str(),
         ge::TypeUtils::DataTypeToSerialString(target_dtype).c_str());

  auto orig_cast_in_anchor = upstream_cast->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(orig_cast_in_anchor);
  auto orig_src_out_anchor = orig_cast_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(orig_src_out_anchor);

  auto orig_cast_out_anchor = upstream_cast->GetOutDataAnchor(0);
  auto in_anchor = downstream_node->GetInDataAnchor(static_cast<int32_t>(input_idx));
  GE_ASSERT_NOTNULL(in_anchor);

  std::string merged_cast_name = std::string(upstream_cast->GetName()) + "_merged_to_" + downstream_node->GetName();
  Cast merged_cast_node(merged_cast_name.c_str());
  auto merged_cast_ptr = graph.AddNode(merged_cast_node);
  GE_ASSERT_NOTNULL(merged_cast_ptr);
  merged_cast_ptr->attr.sched = downstream_node->attr.sched;
  merged_cast_ptr->outputs[0].attr = downstream_node->inputs[input_idx].attr;
  merged_cast_ptr->outputs[0].attr.dtype = target_dtype;

  GE_ASSERT_SUCCESS(af::GraphUtils::RemoveEdge(orig_cast_out_anchor, in_anchor));
  GE_ASSERT_SUCCESS(af::GraphUtils::AddEdge(orig_src_out_anchor, merged_cast_ptr->GetInDataAnchor(0)));
  GE_ASSERT_SUCCESS(af::GraphUtils::AddEdge(merged_cast_ptr->GetOutDataAnchor(0), in_anchor));
  return true;
}

Status DtypeConsistency::InsertCastNode(af::AscGraph &graph, const af::AscNodePtr &src_node,
                                        const af::AscNodePtr &dst_node, size_t input_idx, af::DataType target_dtype) {
  auto in_anchor = dst_node->GetInDataAnchor(static_cast<int32_t>(input_idx));
  GE_ASSERT_NOTNULL(in_anchor);
  auto src_out_anchor = in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(src_out_anchor);

  // Create a new cast node
  std::string cast_name = std::string(src_node->GetName()) + "_cast_to_" + dst_node->GetName();
  Cast cast_node(cast_name.c_str());
  auto cast_node_ptr = graph.AddNode(cast_node);
  GE_ASSERT_NOTNULL(cast_node_ptr);
  cast_node_ptr->outputs[0].attr = dst_node->inputs[input_idx].attr;
  cast_node_ptr->outputs[0].attr.dtype = target_dtype;

  GE_ASSERT_SUCCESS(af::AscGraphUtils::InsertNodeBefore(in_anchor, cast_node_ptr));

  return af::SUCCESS;
}

Status DtypeConsistency::CancelRedundantCast(af::AscGraph &graph) {
  // 1. Cast CSE: Merge multiple identical dtype Cast nodes from the same upstream into one
  GE_ASSERT_SUCCESS(DoCastCSE(graph), "Failed to do cast CSE");
  // 2. Remove redundant Cast(A->A)
  GE_ASSERT_SUCCESS(CancelIdentityCast(graph), "Failed to cancel identity cast");
  return af::SUCCESS;
}

Status DtypeConsistency::DoCastCSE(af::AscGraph &graph) {
  std::map<af::OutDataAnchorPtr, std::map<af::DataType, std::vector<af::AscNodePtr>>> src_casts_map;
  auto all_nodes = graph.GetAllNodes();
  for (const auto &node : all_nodes) {
    if (!af::ops::IsOps<Cast>(node)) {
      continue;
    }

    auto in_anchor = node->GetInDataAnchor(0);
    GE_ASSERT_NOTNULL(in_anchor);
    auto src_out_anchor = in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(src_out_anchor);

    auto out_dtype = node->outputs[0].attr.dtype;
    src_casts_map[src_out_anchor][out_dtype].push_back(node);
  }

  // Merge identical casts for each source and dtype
  for (auto &src_casts_iter : src_casts_map) {
    auto &src_out_anchor = src_casts_iter.first;
    auto &dtype_casts = src_casts_iter.second;
    for (auto &dtype_casts_iter : dtype_casts) {
      auto &cast_nodes = dtype_casts_iter.second;
      if (cast_nodes.size() <= 1UL) {
        continue;
      }

      // Keep the first Cast, remove the rest
      auto &keep_cast = cast_nodes[0];
      for (size_t i = 1UL; i < cast_nodes.size(); ++i) {
        auto &remove_cast = cast_nodes[i];
        GE_ASSERT_SUCCESS(RemoveDuplicateCast(keep_cast, remove_cast, src_out_anchor));
      }
    }
  }

  return af::SUCCESS;
}

Status DtypeConsistency::CancelIdentityCast(af::AscGraph &graph) {
  auto all_nodes = graph.GetAllNodes();
  for (const auto &node : all_nodes) {
    if (!af::ops::IsOps<Cast>(node)) {
      continue;
    }

    auto in_dtype = node->inputs[0].attr.dtype;
    auto out_dtype = node->outputs[0].attr.dtype;
    if (in_dtype != out_dtype) {
      continue;
    }

    // Remove redundant Cast: src -> cast -> dst  =>  src -> dst
    auto in_anchor = node->GetInDataAnchor(0);
    GE_ASSERT_NOTNULL(in_anchor);
    auto new_src = in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(new_src);
    auto cast_out_anchor = node->GetOutDataAnchor(0);
    GE_ASSERT_NOTNULL(cast_out_anchor);
    GE_ASSERT_SUCCESS(PassUtils::RelinkAllOutNodeToSrc(cast_out_anchor, new_src));
    auto owner_graph = node->GetOwnerComputeGraph();
    GE_ASSERT_SUCCESS(owner_graph->RemoveNode(node));
  }
  return af::SUCCESS;
}
}  // namespace optimize
