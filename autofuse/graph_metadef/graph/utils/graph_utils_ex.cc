/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/graph_utils_ex.h"

#include "graph_metadef/common/ge_common/util.h"
#include "common/util/trace_manager/trace_manager.h"
#include "graph/refiner/format_refiner.h"
#include "graph/normal_graph/operator_impl.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/transformer_utils.h"
#include "graph/utils/node_utils_ex.h"
#include "common/util/mem_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/operator_factory_af.h"

namespace af {
graphStatus GraphUtilsEx::InferOriginFormat(const ComputeGraphPtr &graph) {
  return FormatRefiner::InferOrigineFormat(graph);
}

graphStatus GraphUtilsEx::InferShapeInNeed(const ComputeGraphPtr &graph) {
  const auto topological_ret = graph->TopologicalSorting();
  GE_LOGW_IF(topological_ret != GRAPH_SUCCESS, "[Check][Graph] TopologicalSorting failed, graph name:%s, ret:%d.",
             graph->GetName().c_str(), static_cast<int32_t>(topological_ret));
  for (const auto &node_ptr : graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node_ptr);
    const auto op_desc = node_ptr->GetOpDesc();
    bool is_need_infer = false;
    (void)AttrUtils::GetBool(op_desc, NEED_INFER, is_need_infer);
    if (is_need_infer) {
      if (NodeUtilsEx::Verify(node_ptr) != GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E18888", "Verifying %s failed.", node_ptr->GetName().c_str());
        GELOGE(FAILED, "[Call][Verify] Verifying %s failed.", node_ptr->GetName().c_str());
        return GRAPH_FAILED;
      }

      const graphStatus status = NodeUtilsEx::InferShapeAndType(node_ptr);
      if ((!OpTypeUtils::IsDataNode(node_ptr->GetType())) && (status == GRAPH_PARAM_INVALID)) {
        GELOGI(
            "Op %s does not have the IMPLEMT_INFERFUNC definition, "
            "and subsequent operators no longer perform shape inference.",
            node_ptr->GetName().c_str());
        break;
      }
      if (status != GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E18888", "Inferring %s failed.", node_ptr->GetName().c_str());
        GELOGE(FAILED, "[Call][InferShapeAndType] Inferring %s failed.", node_ptr->GetName().c_str());
        return GRAPH_FAILED;
      }

      for (const auto &out_anchor : node_ptr->GetAllOutDataAnchors()) {
        GE_CHECK_NOTNULL(out_anchor->GetOwnerNodeBarePtr()->GetOpDesc());
        auto output_tensor = out_anchor->GetOwnerNodeBarePtr()->GetOpDesc()->MutableOutputDesc(
            static_cast<uint32_t>(out_anchor->GetIdx()));
        GE_CHECK_NOTNULL(output_tensor);
        TensorUtils::SetRealDimCnt(*(output_tensor.get()),
                                   static_cast<uint32_t>(output_tensor->GetShape().GetDims().size()));

        for (const auto &peer_anchor : out_anchor->GetPeerInDataAnchors()) {
          const auto peer_in_tensor_desc = peer_anchor->GetOwnerNodeBarePtr()->GetOpDesc()->MutableInputDesc(
              static_cast<uint32_t>(peer_anchor->GetIdx()));
          GE_CHECK_NOTNULL(peer_in_tensor_desc);
          OpDescUtilsEx::UpdateShapeAndDType(output_tensor, peer_in_tensor_desc);
        }
      }
    }
  }
  return GRAPH_SUCCESS;
}

std::vector<NodePtr> GraphUtilsEx::GetUserInputDataNodes(const ComputeGraphPtr &compute_graph) {
  std::vector<NodePtr> user_input_nodes;
  for (const auto &node : compute_graph->GetInputNodes()) {
    if (!AttrUtils::HasAttr(node->GetOpDesc(), "_is_multi_batch_shape_data")) {
      user_input_nodes.emplace_back(node);
    }
  }
  return user_input_nodes;
}

graphStatus GraphUtilsEx::CopyGraph(const Graph &src_graph, Graph &dst_graph) {
  std::string graph_name;
  AscendString ascend_name;
  if (dst_graph.GetName(ascend_name) == GRAPH_SUCCESS) {
    graph_name = std::string((ascend_name.GetString() != nullptr) ? ascend_name.GetString() : "");
  }
  if (graph_name.empty() && (src_graph.GetName(ascend_name) == GRAPH_SUCCESS)) {
    graph_name = std::string((ascend_name.GetString() != nullptr) ? ascend_name.GetString() : "");
  }

  ComputeGraphPtr new_compute_graph = MakeShared<ComputeGraph>(graph_name);
  GE_CHECK_NOTNULL(new_compute_graph);
  const ComputeGraphPtr src_compute_graph = GraphUtilsEx::GetComputeGraph(src_graph);
  GE_CHECK_NOTNULL(src_compute_graph);
  if (src_compute_graph->GetParentGraph() != nullptr) {
    GELOGE(GRAPH_FAILED,
           "[Check][RootGraph] Only support copy root graph, current graph name:%s, "
           "parent graph name:%s.",
           src_compute_graph->GetName().c_str(), src_compute_graph->GetParentGraph()->GetName().c_str());
    return GRAPH_FAILED;
  }
  const int32_t depth = 0;
  std::map<ConstNodePtr, NodePtr> node_old_2_new;
  std::map<ConstOpDescPtr, OpDescPtr> op_desc_old_2_new;
  graphStatus ret =
      GraphUtils::CopyComputeGraph(src_compute_graph, new_compute_graph, node_old_2_new, op_desc_old_2_new, depth);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Copy][Graph] failed, ret:%d.", ret);
    return GRAPH_FAILED;
  }
  Graph tmp_graph = GraphUtilsEx::CreateGraphFromComputeGraph(new_compute_graph);
  ret = GraphUtilsEx::CopyGraphImpl(src_graph, tmp_graph, node_old_2_new, op_desc_old_2_new);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Copy][GraphImpl] failed, ret:%d.", ret);
    return GRAPH_FAILED;
  }
  std::swap(dst_graph, tmp_graph);
  return GRAPH_SUCCESS;
}

Graph GraphUtilsEx::CreateGraph() {
  return Graph("");
}

Operator GraphUtilsEx::CreateOperator(const char_t *const operator_name, const char_t *const operator_type) {
  return af::OperatorFactory::CreateOperator(operator_name, operator_type);
}
}  // namespace af

af::Graph GeApiWrapper_CreateGraphFromComputeGraph(const af::ComputeGraphPtr &compute_graph) {
  return af::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
}

size_t GeApiWrapper_GetComputeGraphInputSize(const af::Graph &graph) {
  const auto compute_graph = af::GraphUtilsEx::GetComputeGraph(graph);
  if (compute_graph == nullptr) {
    return 0;
  }
  return compute_graph->GetInputSize();
}

size_t GeApiWrapper_GetComputeGraphOutputSize(const af::Graph &graph) {
  const auto compute_graph = af::GraphUtilsEx::GetComputeGraph(graph);
  if (compute_graph == nullptr) {
    return 0;
  }
  return compute_graph->GetOutputSize();
}
