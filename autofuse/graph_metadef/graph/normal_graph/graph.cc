/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/graph.h"
#include <cstring>
#include "graph/graph_buffer_af.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph/model.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_adapter.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "common/checker.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/type_utils.h"
#include "utils/tensor_adapter.h"
#include "proto/onnx/af_onnx.pb.h"
#include "utils/ge_ir_utils.h"
#include "graph/utils/readable_dump.h"
#include "base/err_msg.h"

namespace af {
class GraphImpl {
 public:
  friend class GraphUtils;
  friend class GraphUtilsEx;
  GraphImpl(const GraphImpl &) = delete;
  GraphImpl &operator=(const GraphImpl &) = delete;

  explicit GraphImpl(const std::string &name) : name_(name) {}

  ~GraphImpl() {
    if (IsValid()) {
      if (compute_graph_ != nullptr) {
        GraphUtilsEx::BreakConnect(compute_graph_->GetAllNodesInfo());
      }
    }
    for (const auto &it : op_list_) {
      const Operator op = it.second;
      op.BreakConnect();
    }
  }

  graphStatus SetInputs(const std::vector<Operator> &inputs) {
    GE_ASSERT(!IsValid(),
              "Inner graph has been inited, maybe you call `SetInputs` again or call `AddNodeByOp or SetValid` before "
              "`SetInputs`");
    compute_graph_ = GraphUtilsEx::CreateGraphFromOperator(name_, inputs);
    GE_CHK_BOOL_RET_STATUS(compute_graph_ != nullptr, GRAPH_FAILED, "[Build][Graph] failed.");
    GE_CHK_BOOL_RET_STATUS(inputs.size() != 0U, GRAPH_FAILED, "[Check][Param] set input NULL.");
    compute_graph_->SetInputSize(static_cast<uint32_t>(inputs.size()));
    return GRAPH_SUCCESS;
  }

  graphStatus SetOutputs(const std::vector<Operator> &outputs) {
    if (compute_graph_ == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "compute graph is nullptr, check invalid.");
      GELOGE(GRAPH_FAILED, "[Check][Param] set ComputeGraph failed.");
      return GRAPH_FAILED;
    }
    if (outputs.empty()) {
      GELOGI("Set outputs size is 0.");
      return GRAPH_SUCCESS;
    }

    // Construct special output node
    std::vector<std::pair<Operator, std::vector<size_t>>> output_indexes;
    for (size_t i = 0U; i < outputs.size(); ++i) {
      output_indexes.emplace_back(outputs[i], std::vector<size_t>{});
    }

    const graphStatus ret = SetOutputs(output_indexes);
    return ret;
  }

  graphStatus SetOutputs(const std::vector<std::pair<Operator, std::vector<size_t>>> &output_indexes) {
    if (compute_graph_ == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "compute graph is nullptr, check invalid.");
      GELOGE(GRAPH_FAILED, "[Check][Param] set ComputeGraph failed.");
      return GRAPH_FAILED;
    }
    if (output_indexes.empty()) {
      GELOGW("[SetOutputs][CheckParam] Set outputs size is 0.");
      return GRAPH_SUCCESS;
    }

    // Construct special output node
    std::vector<std::pair<NodePtr, int32_t>> output_nodes;
    for (const auto &item : output_indexes) {
      const Operator &output = item.first;
      const std::vector<size_t> &indexes = item.second;
      AscendString out_name;
      (void)output.GetName(out_name);
      NodePtr node = compute_graph_->FindNode(out_name.GetString());
      if (node == nullptr) {
        GELOGW("[SetOutputs][Check] User designated out_node %s does not exist in graph, skip it",
               out_name.GetString());
        continue;
      }

      const OpDescPtr tmp_op_ptr = node->GetOpDesc();
      if (tmp_op_ptr == nullptr) {
        GELOGE(GRAPH_FAILED, "op_desc in node must not be null.");
        continue;
      }
      const size_t out_size = tmp_op_ptr->GetOutputsSize();
      if (indexes.empty()) {
        for (size_t i = 0U; i < out_size; ++i) {
          output_name_ += std::string(out_name.GetString()) + ":" + std::to_string(i) + ";";
          output_nodes.emplace_back(node, i);
        }
      } else {
        for (size_t i = 0U; i < indexes.size(); ++i) {
          if (indexes[i] >= out_size) {
            GELOGW("[SetOutputs][Check] User designated out_node %s has no output %zu, output_size=%zu, skip it",
                   out_name.GetString(), indexes[i], out_size);
          } else {
            output_name_ += std::string(out_name.GetString()) + ":" + std::to_string(i) + ";";
            output_nodes.emplace_back(node, indexes[i]);
          }
        }
      }
    }

    // Del last ";"
    if (!output_name_.empty()) {
      output_name_ = output_name_.substr(0U, output_name_.length() - 1U);
    }
    compute_graph_->SetUserDefOutput(output_name_);
    compute_graph_->SetOutputSize(static_cast<uint32_t>(output_indexes.size()));
    GE_ASSERT_SUCCESS(compute_graph_->SetGraphOutNodesInfo(output_nodes));
    return GRAPH_SUCCESS;
  }

  graphStatus SetOutputs(const std::vector<std::pair<Operator, std::string>> &outputs) {
    GE_CHK_BOOL_RET_STATUS(compute_graph_ != nullptr, GRAPH_FAILED, "[Check][Param] set ComputeGraph failed.");
    if (outputs.empty()) {
      GELOGI("set outputs size is 0.");
      return GRAPH_SUCCESS;
    }

    // Construct specified output
    std::vector<std::pair<NodePtr, int32_t>> output_nodes;
    for (const auto &item : outputs) {
      AscendString out_name;
      (void)item.first.GetName(out_name);
      NodePtr node = compute_graph_->FindNode(out_name.GetString());
      if (node == nullptr) {
        REPORT_INNER_ERR_MSG("E18888", "designated out_node (%s) does not exist in graph:%s, this out_node ignored!",
                             out_name.GetString(), compute_graph_->GetName().c_str());
        GELOGE(GRAPH_FAILED,
               "[Check][Param] Warning, user designated out_node (%s) does not exist in graph:%s, "
               "this out_node ignored!",
               out_name.GetString(), compute_graph_->GetName().c_str());
        return GRAPH_FAILED;
      }
      const OpDescPtr tmp_op_ptr = node->GetOpDesc();
      if (tmp_op_ptr == nullptr) {
        GELOGE(GRAPH_FAILED, "op_desc_ptr in node must not be null.");
        continue;
      }
      const size_t out_size = tmp_op_ptr->GetOutputsSize();

      if (item.second.empty()) {
        for (size_t i = 0U; i < out_size; ++i) {
          output_name_ += std::string(out_name.GetString()) + ":" + std::to_string(i) + ";";
          output_nodes.emplace_back(node, i);
        }
      } else {
        int32_t index = tmp_op_ptr->GetOutputIndexByName(item.second);
        if (index < 0) {
          REPORT_INNER_ERR_MSG("E18888",
                               "user designated out_node (%s):(%s) does not exist in graph:%s, "
                               "this out_node ignored!",
                               out_name.GetString(), item.second.c_str(), compute_graph_->GetName().c_str());
          GELOGE(GRAPH_FAILED,
                 "[Check][Param] Warning, user designated out_node (%s):(%s) does not exist in graph:%s, "
                 "this out_node ignored!",
                 out_name.GetString(), item.second.c_str(), compute_graph_->GetName().c_str());
          return GRAPH_FAILED;
        }
        output_name_ += std::string(out_name.GetString()) + ":" + std::to_string(index) + ";";
        output_nodes.emplace_back(node, index);
      }
    }
    // Del last ";"
    if (!output_name_.empty()) {
      output_name_ = output_name_.substr(0U, output_name_.length() - 1U);
    }
    compute_graph_->SetOutputSize(static_cast<uint32_t>(outputs.size()));
    GE_ASSERT_SUCCESS(compute_graph_->SetGraphOutNodesInfo(output_nodes));
    GELOGI("********************SetOutputs Success***********************");
    GE_IF_BOOL_EXEC(!output_name_.empty(), GELOGI(" NetOutputs: (%s)", output_name_.c_str()));

    return GRAPH_SUCCESS;
  }

  graphStatus SetOutputs(const std::vector<std::pair<GNode, int32_t>> &outputs) {
    GE_ASSERT_NOTNULL(compute_graph_);
    std::vector<std::pair<NodePtr, int32_t>> output_node_and_index;
    for (const auto &item : outputs) {
      output_node_and_index.emplace_back(NodeAdapter::GNode2Node(item.first), item.second);
    }
    GE_ASSERT_SUCCESS(compute_graph_->SetGraphOutNodesInfo(output_node_and_index));
    return GRAPH_SUCCESS;
  }

  graphStatus SetTargets(const std::vector<Operator> &targets) {
    GE_CHK_BOOL_RET_STATUS(compute_graph_ != nullptr, GRAPH_FAILED, "[Check][Param] set ComputeGraph failed.");
    if (targets.empty()) {
      GELOGI("set targets size is 0.");
      return GRAPH_SUCCESS;
    }

    std::vector<NodePtr> target_nodes;
    for (const auto &item : targets) {
      AscendString name;
      (void)item.GetName(name);
      const NodePtr node = compute_graph_->FindNode(name.GetString());
      if (node == nullptr) {
        GELOGW("[SetTargets][Check] User designated target_node %s does not exist in graph, skip it", name.GetString());
        continue;
      }
      target_nodes.push_back(node);
    }
    GE_ASSERT_SUCCESS(compute_graph_->SetGraphTargetNodesInfo(target_nodes));
    return GRAPH_SUCCESS;
  }
  bool IsValid() const {
    return (compute_graph_ != nullptr);
  }

  graphStatus AddOp(const Operator &op) {
    AscendString name;
    (void)op.GetName(name);
    const auto ret = op_list_.emplace(std::pair<std::string, Operator>(name.GetString(), op));
    GE_CHK_BOOL_RET_STATUS(ret.second, GRAPH_FAILED, "[Check][Param] the op has already been added, op name:%s.",
                           name.GetString());
    return GRAPH_SUCCESS;
  }

  graphStatus GetAllOpName(std::vector<std::string> &op_name) const {
    for (const auto &it : op_list_) {
      AscendString name;
      it.second.GetName(name);
      op_name.emplace_back(name.GetString());
    }
    return GRAPH_SUCCESS;
  }

  graphStatus FindOpByName(const std::string &name, Operator &op) const {
    const auto it = op_list_.find(name);
    GE_CHK_BOOL_EXEC(it != op_list_.end(), REPORT_INNER_ERR_MSG("E18888", "there is no op: %s.", name.c_str());
                     return GRAPH_FAILED, "[Find][Op] there is no op: %s.", name.c_str());
    op = it->second;
    return GRAPH_SUCCESS;
  }

  graphStatus FindOpByType(const std::string &type, std::vector<Operator> &ops) const {
    for (auto &op : op_list_) {
      AscendString op_type;
      (void)op.second.GetOpType(op_type);
      if (op_type.GetString() == type) {
        ops.push_back(op.second);
        continue;
      }
      if (op_type == FRAMEWORKOP) {
        (void)op.second.GetAttr(ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE.c_str(), op_type);
        if (op_type.GetString() == type) {
          ops.push_back(op.second);
        }
      }
    }
    return GRAPH_SUCCESS;
  }

  void SetNeedIteration(bool need_iteration) {
    if (compute_graph_ == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "Set need iteration failed, as compute graph is null.");
      GELOGE(GRAPH_FAILED, "[Check][Param] Set need iteration failed, as compute graph is null.");
      return;
    }
    compute_graph_->SetNeedIteration(need_iteration);
  }

  const std::string &GetName() const {
    return name_;
  }

  ComputeGraphPtr GetComputeGraph() const {
    return compute_graph_;
  }

  graphStatus InitComputeGraph() {
    GE_WARN_ASSERT(compute_graph_ == nullptr, "No need to init again");
    compute_graph_ = ComGraphMakeSharedAndThrow<ComputeGraph>(name_);
    return GRAPH_SUCCESS;
  }

  graphStatus RemoveEdge(const NodePtr &src_node_ptr, const int32_t src_port_index, const NodePtr &dst_node_ptr,
                         const int32_t dst_port_index) {
    GE_CHECK_NOTNULL(src_node_ptr);
    GE_CHECK_NOTNULL(dst_node_ptr);

    graphStatus res = GRAPH_FAILED;
    if ((src_port_index == -1) && (dst_port_index == -1)) {
      if (src_node_ptr->GetOutControlAnchor() == nullptr) {
        REPORT_INNER_ERR_MSG("E18888", "src node:%s out control anchor is null.", src_node_ptr->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Get][Anchor] src node:%s out control anchor is null.", src_node_ptr->GetName().c_str());
        return GRAPH_FAILED;
      }
      res = GraphUtils::RemoveEdge(src_node_ptr->GetOutControlAnchor(), dst_node_ptr->GetInControlAnchor());
      if (res != GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E18888", "remove control edge between [%s] and [%s] failed.",
                             src_node_ptr->GetName().c_str(), dst_node_ptr->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Remove][ControlEdge] between [%s] and [%s] failed.", src_node_ptr->GetName().c_str(),
               dst_node_ptr->GetName().c_str());
        return GRAPH_FAILED;
      }
      return GRAPH_SUCCESS;
    }

    if (src_node_ptr->GetOutDataAnchor(src_port_index) == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "src node[%s] out data anchor[%d] is null.", src_node_ptr->GetName().c_str(),
                           src_port_index);
      GELOGE(GRAPH_FAILED, "[Get][Anchor] src node[%s] out data anchor[%d] is null.", src_node_ptr->GetName().c_str(),
             src_port_index);
      return GRAPH_FAILED;
    }

    if ((src_port_index != -1) && (dst_port_index == -1)) {
      res = GraphUtils::RemoveEdge(src_node_ptr->GetOutDataAnchor(src_port_index), dst_node_ptr->GetInControlAnchor());
      if (res != GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E18888", "remove data-control edge between [%s] and [%s] failed.",
                             src_node_ptr->GetName().c_str(), dst_node_ptr->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Remove][Edge] between [%s] and [%s] failed.", src_node_ptr->GetName().c_str(),
               dst_node_ptr->GetName().c_str());
        return GRAPH_FAILED;
      }
      return GRAPH_SUCCESS;
    }

    res = GraphUtils::RemoveEdge(src_node_ptr->GetOutDataAnchor(src_port_index),
                                 dst_node_ptr->GetInDataAnchor(dst_port_index));
    if (res != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "remove data edge between [%s] and [%s] failed.", src_node_ptr->GetName().c_str(),
                           dst_node_ptr->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Remove][Edge] between [%s] and [%s] failed.", src_node_ptr->GetName().c_str(),
             dst_node_ptr->GetName().c_str());
      return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
  }

 private:
  std::string name_;
  std::string output_name_;
  std::map<std::string, Operator> op_list_;
  ComputeGraphPtr compute_graph_{nullptr};
};

Graph::Graph(const std::string &name) {
  impl_ = ComGraphMakeShared<GraphImpl>(name);
  if (impl_ == nullptr) {
    GELOGW("[Check][Impl] Make graph impl failed");
  }
}

Graph::Graph(const char_t *name) {
  if (name != nullptr) {
    std::string graph_name = name;
    impl_ = ComGraphMakeShared<GraphImpl>(graph_name);
    if (impl_ == nullptr) {
      GELOGW("[Check][Impl] Make graph impl failed");
    }
  } else {
    GELOGW("[Check][Param] Input graph name is nullptr.");
  }
}

graphStatus Graph::AddOp(const Operator &op) {
  GE_CHK_BOOL_EXEC(impl_ != nullptr, REPORT_INNER_ERR_MSG("E18888", "graph can not be used, impl is nullptr.");
                   return GRAPH_FAILED, "[Check][Param] AddOp failed: graph can not be used, impl is nullptr.");
  return impl_->AddOp(op);
}

graphStatus Graph::GetAllOpName(std::vector<std::string> &op_name) const {
  GE_CHK_BOOL_EXEC(impl_ != nullptr, REPORT_INNER_ERR_MSG("E18888", "graph can not be used, impl is nullptr.");
                   return GRAPH_FAILED, "[Check][Param] GetAllOpName failed: graph can not be used, impl is nullptr.");
  return impl_->GetAllOpName(op_name);
}

graphStatus Graph::GetAllOpName(std::vector<AscendString> &names) const {
  GE_CHK_BOOL_EXEC(impl_ != nullptr, REPORT_INNER_ERR_MSG("E18888", "graph can not be used, impl is nullptr.");
                   return GRAPH_FAILED, "[Check][Param] GetAllOpName failed: graph can not be used, impl is nullptr.");
  std::vector<std::string> op_names;
  if (impl_->GetAllOpName(op_names) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Get][AllOpName] failed.");
    return GRAPH_FAILED;
  }

  for (auto &op_name : op_names) {
    names.emplace_back(op_name.c_str());
  }

  return GRAPH_SUCCESS;
}

graphStatus Graph::FindOpByName(const std::string &name, Operator &op) const {
  const Operator op_find_op_def("NULL");
  op = op_find_op_def;
  GE_CHK_BOOL_EXEC(impl_ != nullptr, REPORT_INNER_ERR_MSG("E18888", "graph can not be used, impl is nullptr.");
                   return GRAPH_FAILED, "[Check][Param] FindOpByName failed: graph can not be used, impl is nullptr.");
  return impl_->FindOpByName(name, op);
}

graphStatus Graph::FindOpByName(const char_t *name, Operator &op) const {
  if (name == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "param name is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] FindOpByName: name is nullptr.");
    return GRAPH_FAILED;
  }
  const Operator op_find_op_def("NULL");
  op = op_find_op_def;
  GE_CHK_BOOL_EXEC(impl_ != nullptr, REPORT_INNER_ERR_MSG("E18888", "graph can not be used, impl is nullptr.");
                   return GRAPH_FAILED, "[Check][Param] FindOpByName failed: graph can not be used, impl is nullptr.");
  const std::string op_name = name;
  return impl_->FindOpByName(op_name, op);
}

graphStatus Graph::FindOpByType(const std::string &type, std::vector<Operator> &ops) const {
  GE_CHECK_NOTNULL(impl_);
  return impl_->FindOpByType(type, ops);
}

graphStatus Graph::FindOpByType(const char_t *type, std::vector<Operator> &ops) const {
  if (type == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "param type is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] FindOpByType: type is nullptr.");
    return GRAPH_FAILED;
  }
  GE_CHECK_NOTNULL(impl_);
  const std::string op_type = type;
  return impl_->FindOpByType(op_type, ops);
}

Graph &Graph::SetInputs(const std::vector<Operator> &inputs) {
  GE_CHK_BOOL_EXEC(impl_ != nullptr, REPORT_INNER_ERR_MSG("E18888", "graph can not be used, impl is nullptr.");
                   return *this, "[Check][Param] SetInputs failed: graph can not be used, impl is nullptr.");
  GE_CHK_BOOL_EXEC(!inputs.empty(), REPORT_INNER_ERR_MSG("E18888", "input operator size can not be 0");
                   return *this, "[Check][Param] SetInputs failed: input operator size can not be 0, graph: %s",
                          impl_->GetName().c_str());
  (void)impl_->SetInputs(inputs);
  return *this;
}

Graph &Graph::SetOutputs(const std::vector<Operator> &outputs) {
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "graph can not be used, impl is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] SetOutputs failed: graph can not be used, impl is nullptr.");
    return *this;
  }
  (void)impl_->SetOutputs(outputs);
  return *this;
}

graphStatus Graph::SetOutputs(const std::vector<std::pair<GNode, int32_t>> &outputs) {
  GE_ASSERT(IsValid(), "Graph is not valid, call SetValid() first.");
  return impl_->SetOutputs(outputs);
}

Graph &Graph::SetOutputs(const std::vector<std::pair<Operator, std::vector<size_t>>> &output_indexes) {
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "graph can not be used, impl is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] SetOutputs failed: graph can not be used, impl is nullptr.");
    return *this;
  }
  (void)impl_->SetOutputs(output_indexes);
  return *this;
}

Graph &Graph::SetOutputs(const std::vector<std::pair<Operator, std::string>> &outputs) {
  GE_CHK_BOOL_EXEC(impl_ != nullptr, REPORT_INNER_ERR_MSG("E18888", "graph can not be used, impl is nullptr.");
                   return *this, "[Check][Param] SetOutputs failed: graph can not be used, impl is nullptr.");
  (void)impl_->SetOutputs(outputs);
  return *this;
}

Graph &Graph::SetOutputs(const std::vector<std::pair<Operator, AscendString>> &outputs) {
  GE_CHK_BOOL_EXEC(impl_ != nullptr, REPORT_INNER_ERR_MSG("E18888", "graph can not be used, impl is nullptr.");
                   return *this, "[Check][Param] SetOutputs failed: graph can not be used, impl is nullptr.");
  std::vector<std::pair<Operator, std::string>> graph_outputs;
  for (auto &item : outputs) {
    const char_t *const name = item.second.GetString();
    if (name != nullptr) {
      graph_outputs.emplace_back((std::pair<Operator, std::string>(item.first, name)));
    } else {
      GELOGW("[SetOutputs][CheckParam] Input output_op_name is nullptr.");
    }
  }

  (void)impl_->SetOutputs(graph_outputs);
  return *this;
}

Graph &Graph::SetTargets(const std::vector<Operator> &targets) {
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "graph can not be used, impl is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] SetTargets failed: graph can not be used, impl is nullptr.");
    return *this;
  }
  (void)impl_->SetTargets(targets);
  return *this;
}

bool Graph::IsValid() const {
  if (impl_ == nullptr) {
    return false;
  }
  return impl_->IsValid();
}

void Graph::SetNeedIteration(bool need_iteration) {
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "graph can not be used, impl is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Set need iteration failed, as impl is null.");
    return;
  }
  impl_->SetNeedIteration(need_iteration);
}

std::vector<GNode> Graph::GetAllNodes() const {
  std::vector<GNode> graph_nodes;
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "graph can not be used, impl is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetAllNodes: graph can not be used, impl is nullptr.");
    return graph_nodes;
  }

  const ComputeGraphPtr compute_graph_ptr = impl_->GetComputeGraph();
  if (compute_graph_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "impl compute graph is nullptr.");
    GELOGE(GRAPH_FAILED, "[Get][Graph] GetAllNodes: compute graph ptr is nullptr, graph %s", impl_->GetName().c_str());
    return graph_nodes;
  }

  for (auto &node : compute_graph_ptr->GetAllNodes()) {
    GNode gnode = NodeAdapter::Node2GNode(node);
    graph_nodes.emplace_back(gnode);
  }

  return graph_nodes;
}

std::vector<GraphPtr> Graph::GetAllSubgraphs() const {
  GE_ASSERT(IsValid(), "[Check][Param] current graph is invalid.");
  std::vector<GraphPtr> subgraphs;
  auto sub_compute_graphs = impl_->GetComputeGraph()->GetAllSubgraphs();
  subgraphs.reserve(sub_compute_graphs.size());
  for (const auto &sub_compute_graph : sub_compute_graphs) {
    auto subgraph = GraphUtilsEx::CreateGraphPtrFromComputeGraph(sub_compute_graph);
    subgraphs.emplace_back(subgraph);
  }
  return subgraphs;
}

GraphPtr Graph::GetSubGraph(const char *name) const {
  GE_ASSERT(IsValid(), "[Check][Param] current graph is invalid.");
  auto sub_compute_graph = impl_->GetComputeGraph()->GetSubgraph(name);
  if (sub_compute_graph != nullptr) {
    return GraphUtilsEx::CreateGraphPtrFromComputeGraph(sub_compute_graph);
  }
  return nullptr;
}

graphStatus Graph::AddSubGraph(const Graph &subgraph) {
  GE_ASSERT(IsValid(), "[Check][Param] current graph is invalid.");
  GE_ASSERT(subgraph.IsValid(), "[Check][Param] subgraph is invalid.");
  auto compute_graph = impl_->GetComputeGraph();
  auto sub_compute_graph = GraphUtilsEx::GetComputeGraph(subgraph);
  if (compute_graph->AddSubgraph(sub_compute_graph) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "graph:%s add subgraph:%s failed", compute_graph->GetName().c_str(),
                         sub_compute_graph->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Add][Subgraph] %s to graph:%s failed.", sub_compute_graph->GetName().c_str(),
           compute_graph->GetName().c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus Graph::RemoveSubgraph(const char *name) {
  GE_ASSERT(IsValid(), "[Check][Param] current graph is invalid.");
  impl_->GetComputeGraph()->RemoveSubgraph(name);
  return GRAPH_SUCCESS;
}

std::vector<GNode> Graph::GetDirectNode() const {
  std::vector<GNode> graph_nodes;
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "graph can not be used, impl is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] GetDirectNode: graph can not be used, impl is nullptr.");
    return graph_nodes;
  }
  const ComputeGraphPtr compute_graph_ptr = impl_->GetComputeGraph();
  if (compute_graph_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "impl compute graph is nullptr.");
    GELOGE(GRAPH_FAILED, "[Get][Graph] GetDirectNode: compute graph ptr is nullptr, graph %s",
           impl_->GetName().c_str());
    return graph_nodes;
  }

  for (auto &node : compute_graph_ptr->GetDirectNode()) {
    GNode gnode = NodeAdapter::Node2GNode(node);
    graph_nodes.emplace_back(gnode);
  }

  return graph_nodes;
}

graphStatus Graph::RemoveNode(GNode &node) {
  return RemoveNode(node, false);
}

graphStatus Graph::RemoveNode(GNode &node, bool contain_subgraph) {
  GE_CHECK_NOTNULL(impl_);

  const NodePtr node_ptr = NodeAdapter::GNode2Node(node);
  GE_CHECK_NOTNULL(node_ptr);

  const ComputeGraphPtr owner_compute_graph = node_ptr->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(owner_compute_graph);

  ComputeGraphPtr compute_graph_ptr = impl_->GetComputeGraph();
  GE_CHECK_NOTNULL(compute_graph_ptr);

  if (contain_subgraph) {
    if (!GraphUtils::IsNodeInGraphRecursively(compute_graph_ptr, *node_ptr)) {
      REPORT_INNER_ERR_MSG("E18888", "node[%s] is not in the graph[%s] or not in subgraph.",
                           node_ptr->GetName().c_str(), compute_graph_ptr->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] node[%s] is not in the graph[%s].", node_ptr->GetName().c_str(),
             compute_graph_ptr->GetName().c_str());
      return GRAPH_FAILED;
    }
    compute_graph_ptr = owner_compute_graph;
  } else {
    if (compute_graph_ptr != owner_compute_graph) {
      REPORT_INNER_ERR_MSG("E18888", "node[%s] is not in the graph[%s].", node_ptr->GetName().c_str(),
                           compute_graph_ptr->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] node[%s] is not in the graph[%s].", node_ptr->GetName().c_str(),
             compute_graph_ptr->GetName().c_str());
      return GRAPH_FAILED;
    }
  }

  NodeUtils::UnlinkAll(*node_ptr);
  if (GraphUtils::RemoveNodeWithoutRelink(compute_graph_ptr, node_ptr) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "graph:%s remove node:%s failed", compute_graph_ptr->GetName().c_str(),
                         node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Remove][Node] %s from graph:%s failed.", node_ptr->GetName().c_str(),
           compute_graph_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }
  (void)node_ptr->ClearOwnerGraph(nullptr);
  return GRAPH_SUCCESS;
}

graphStatus Graph::RemoveEdge(GNode &src_node, const int32_t src_port_index, GNode &dst_node,
                              const int32_t dst_port_index) {
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "graph can not be used, impl is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] graph can not be used, impl is nullptr.");
    return GRAPH_FAILED;
  }

  if ((src_port_index == -1) && (dst_port_index != -1)) {
    REPORT_INNER_ERR_MSG("E18888", "src_port_index == -1 and dst_port_index != -1, check invalid .");
    GELOGE(GRAPH_FAILED, "[Check][Param] src control anchor link to dst data anchor does not exist.");
    return GRAPH_FAILED;
  }

  const NodePtr src_node_ptr = NodeAdapter::GNode2Node(src_node);
  if (src_node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "src gnode to node failed.");
    GELOGE(GRAPH_FAILED, "[Get][Node] src gnode to node failed.");
    return GRAPH_FAILED;
  }

  const NodePtr dst_node_ptr = NodeAdapter::GNode2Node(dst_node);
  if (dst_node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "dst gnode to node failed.");
    GELOGE(GRAPH_FAILED, "[Get][Node] dst gnode to node failed.");
    return GRAPH_FAILED;
  }

  if (src_node_ptr->GetOwnerComputeGraph() == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "src node:%s compute graph is nullptr.", src_node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][Graph] src node:%s compute graph is nullptr.", src_node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  if (dst_node_ptr->GetOwnerComputeGraph() == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "dst node:%s compute graph is nullptr", dst_node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][Graph] dst node:%s compute graph is nullptr.", dst_node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  if (impl_->RemoveEdge(src_node_ptr, src_port_index, dst_node_ptr, dst_port_index) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "remove edge between %s(%d) and %s(%d) failed.", src_node_ptr->GetName().c_str(),
                         src_port_index, dst_node_ptr->GetName().c_str(), dst_port_index);
    GELOGE(GRAPH_FAILED, "[Remove][Edge] between %s(%d) and %s(%d) failed.", src_node_ptr->GetName().c_str(),
           src_port_index, dst_node_ptr->GetName().c_str(), dst_port_index);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

GNode Graph::AddNodeByOp(const Operator &op) {
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "graph can not be used, impl is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] graph can not be used, impl is nullptr.");
    return GNode();
  }

  const OpDescPtr op_desc = OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc == nullptr) {
    AscendString name;
    (void)op.GetName(name);
    REPORT_INNER_ERR_MSG("E18888", "get op desc from op:%s failed", name.GetString());
    GELOGE(GRAPH_FAILED, "[Get][OpDesc] from op[%s] failed.", name.GetString());
    return GNode();
  }

  GE_ASSERT_GRAPH_SUCCESS(SetValid());
  const auto compute_graph_ptr = impl_->GetComputeGraph();
  GE_ASSERT_NOTNULL(compute_graph_ptr);
  const NodePtr node_ptr = compute_graph_ptr->AddNode(op_desc);
  const GNode gnode = NodeAdapter::Node2GNode(node_ptr);

  return gnode;
}

graphStatus Graph::AddDataEdge(GNode &src_node, const int32_t src_port_index, GNode &dst_node,
                               const int32_t dst_port_index) {
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "graph can not be used, impl is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] graph can not be used, impl is nullptr.");
    return GRAPH_FAILED;
  }

  const NodePtr src_node_ptr = NodeAdapter::GNode2Node(src_node);
  if (src_node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "src gnode to node failed.");
    GELOGE(GRAPH_FAILED, "[Get][Node] src gnode to node failed.");
    return GRAPH_FAILED;
  }

  const NodePtr dst_node_ptr = NodeAdapter::GNode2Node(dst_node);
  if (dst_node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "dst gnode to node failed.");
    GELOGE(GRAPH_FAILED, "[Get][Node] dst gnode to node failed.");
    return GRAPH_FAILED;
  }

  if (src_node_ptr->GetOwnerComputeGraph() == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "src node[%s] owner compute graph is nullptr.", src_node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][Graph] src node[%s] owner compute graph is nullptr.", src_node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  if (dst_node_ptr->GetOwnerComputeGraph() == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "dst node[%s] owner compute graph is nullptr.", dst_node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][Graph] dst node[%s] owner compute graph is nullptr.", dst_node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  const graphStatus res = GraphUtils::AddEdge(src_node_ptr->GetOutDataAnchor(src_port_index),
                                              dst_node_ptr->GetInDataAnchor(dst_port_index));
  if (res != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "add data edge from %s(%d) to %s(%d) failed.", src_node_ptr->GetName().c_str(),
                         src_port_index, dst_node_ptr->GetName().c_str(), dst_port_index);
    GELOGE(GRAPH_FAILED, "[Add][DataEdge] from %s(%d) to %s(%d) failed.", src_node_ptr->GetName().c_str(),
           src_port_index, dst_node_ptr->GetName().c_str(), dst_port_index);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

graphStatus Graph::AddControlEdge(GNode &src_node, GNode &dst_node) {
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "graph can not be used, impl is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] graph can not be used, impl is nullptr.");
    return GRAPH_FAILED;
  }

  const NodePtr src_node_ptr = NodeAdapter::GNode2Node(src_node);
  if (src_node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "src gnode to node failed.");
    GELOGE(GRAPH_FAILED, "[Get][Node] src gnode to node failed.");
    return GRAPH_FAILED;
  }

  const NodePtr dst_node_ptr = NodeAdapter::GNode2Node(dst_node);
  if (dst_node_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "dst gnode to node failed.");
    GELOGE(GRAPH_FAILED, "[Get][Node] dst gnode to node failed.");
    return GRAPH_FAILED;
  }

  if (src_node_ptr->GetOwnerComputeGraph() == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "src node[%s] owner compute graph is nullptr.", src_node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][Graph] src node[%s] owner compute graph is nullptr.", src_node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  if (dst_node_ptr->GetOwnerComputeGraph() == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "dst node[%s] owner compute graph is nullptr.", dst_node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Get][Graph] dst node[%s] owner compute graph is nullptr.", dst_node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  const graphStatus res = GraphUtils::AddEdge(src_node_ptr->GetOutControlAnchor(), dst_node_ptr->GetInControlAnchor());
  if (res != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "add control edge from %s to %s failed.", src_node_ptr->GetName().c_str(),
                         dst_node_ptr->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Add][ControlEdge] from %s to %s failed.", src_node_ptr->GetName().c_str(),
           dst_node_ptr->GetName().c_str());
    return GRAPH_FAILED;
  }

  return SUCCESS;
}

GraphPtr Graph::ConstructFromInputs(const std::vector<Operator> &inputs, const AscendString &name) {
  const char_t *const ascend_name = name.GetString();
  if (ascend_name == nullptr) {
    REPORT_INNER_ERR_MSG("E18888",
                         "ConstructFromInputs failed: input graph name is invalid, expect a non-empty AscendString.");
    GELOGE(GRAPH_PARAM_INVALID,
           "[Check][Param] ConstructFromInputs failed: input graph name is invalid, expect a non-empty AscendString.");
    return nullptr;
  }

  if (inputs.empty()) {
    REPORT_INNER_ERR_MSG("E18888", "inputs size can not be 0.");
    GELOGE(GRAPH_FAILED, "[Check][Param] inputs size can not be 0, graph: %s", ascend_name);
    return nullptr;
  }

  const std::string graph_name = ascend_name;
  const ComputeGraphPtr compute_graph = GraphUtilsEx::CreateGraphFromOperator(graph_name, inputs);
  if (compute_graph == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "create compute graph from op failed, name:%s", graph_name.c_str());
    GELOGE(GRAPH_FAILED, "[Create][ComputeGraph] failed, name:%s.", graph_name.c_str());
    return nullptr;
  }

  compute_graph->SetInputSize(static_cast<uint32_t>(inputs.size()));
  const GraphPtr graph_ptr = GraphUtilsEx::CreateGraphPtrFromComputeGraph(compute_graph);
  if (graph_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "create graph from compute graph:%s failed.", compute_graph->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Create][Graph] from compute graph:%s failed.", compute_graph->GetName().c_str());
    return nullptr;
  }

  return graph_ptr;
}

graphStatus Graph::SaveToFile(const std::string &file_name) const {
  Model model = Model();
  model.SetGraph(GraphUtilsEx::GetComputeGraph(*this));
  return model.SaveToFile(file_name);
}

graphStatus Graph::SaveToFile(const char_t *file_name) const {
  if (file_name == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "file name is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] file name is nullptr.");
    return GRAPH_FAILED;
  }

  Model model = Model();
  model.SetGraph(GraphUtilsEx::GetComputeGraph(*this));
  const std::string name = file_name;
  return model.SaveToFile(name);
}

graphStatus Graph::LoadFromFile(const std::string &file_name) {
  Model model = Model();
  const graphStatus ret = model.LoadFromFile(file_name);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  *this = GraphUtilsEx::CreateGraphFromComputeGraph(model.GetGraph());
  return GRAPH_SUCCESS;
}

graphStatus Graph::LoadFromFile(const char_t *file_name) {
  if (file_name == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "param file name is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] file name is nullptr.");
    return GRAPH_FAILED;
  }

  Model model = Model();
  const std::string file = file_name;
  const graphStatus ret = model.LoadFromFile(file);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  *this = GraphUtilsEx::CreateGraphFromComputeGraph(model.GetGraph());
  return GRAPH_SUCCESS;
}

graphStatus Graph::LoadFromSerializedModelArray(const void *serialized_model, size_t size) {
  GE_ASSERT_NOTNULL(serialized_model, "param serialized_model is nullptr");
  GE_ASSERT(size != 0U, "param size is 0");
  Model model;
  GE_ASSERT_GRAPH_SUCCESS(Model::Load(static_cast<const uint8_t *>(serialized_model), size, model),
                          "Failed to load model from serialized model def.");
  GE_ASSERT_NOTNULL(model.GetGraph(), "Failed to get root graph from model.");
  *this = GraphUtilsEx::CreateGraphFromComputeGraph(model.GetGraph());
  return GRAPH_SUCCESS;
}

graphStatus Graph::SaveToMem(GraphBuffer &graph_buffer) const {
  Model model = Model();
  model.SetGraph(GraphUtilsEx::GetComputeGraph(*this));
  GE_ASSERT_GRAPH_SUCCESS(model.Save(*(graph_buffer.buffer_)), "Failed to save graph to memory.");
  return GRAPH_SUCCESS;
}

graphStatus Graph::LoadFromMem(const GraphBuffer &graph_buffer) {
  Model model = Model();
  GE_ASSERT_GRAPH_SUCCESS(Model::Load(graph_buffer.GetData(), graph_buffer.GetSize(), model),
                          "Failed to load graph from memory.");

  *this = GraphUtilsEx::CreateGraphFromComputeGraph(model.GetGraph());
  return GRAPH_SUCCESS;
}

graphStatus Graph::LoadFromMem(const uint8_t *data, const size_t len) {
  Model model = Model();
  GE_ASSERT_GRAPH_SUCCESS(Model::Load(data, len, model), "Failed to load graph from memory.");

  *this = GraphUtilsEx::CreateGraphFromComputeGraph(model.GetGraph());
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::string &Graph::GetName() const {
  return impl_->GetName();
}

graphStatus Graph::GetName(AscendString &name) const {
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] impl is nullptr.");
    return GRAPH_FAILED;
  }
  const std::string graph_name = impl_->GetName();
  name = AscendString(graph_name.c_str());
  return GRAPH_SUCCESS;
}

graphStatus Graph::CopyFrom(const Graph &src_graph) {
  const auto res = GraphUtilsEx::CopyGraph(src_graph, *this);
  if (res != GRAPH_SUCCESS) {
    AscendString name;
    (void)src_graph.GetName(name);
    REPORT_INNER_ERR_MSG("E18888", "copy graph from %s failed.", name.GetString());
    GELOGE(GRAPH_FAILED, "[Copy][Graph] from %s failed.", name.GetString());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// 添加AttrValue版本的SetAttr和GetAttr方法
graphStatus Graph::SetAttr(const AscendString &name, const AttrValue &attr_value) {
  GE_ASSERT_NOTNULL(impl_);
  const auto compute_graph = impl_->GetComputeGraph();
  GE_ASSERT_NOTNULL(compute_graph, "Inner source is not ready, call `SetInputs` or `SetValid` at first");
  return compute_graph->SetAttr(name.GetString(), attr_value.impl->MutableAnyValue());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus Graph::GetAttr(const AscendString &name,
                                                                          AttrValue &attr_value) const {
  GE_ASSERT_NOTNULL(impl_);
  const auto compute_graph = impl_->GetComputeGraph();
  GE_ASSERT_NOTNULL(compute_graph, "Inner source is not ready, call `SetInputs` or `SetValid` at first");
  return compute_graph->GetAttr(name.GetString(), attr_value.impl->MutableAnyValue());
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus Graph::SetValid() {
  GE_ASSERT_NOTNULL(impl_);
  if (IsValid()) {
    return GRAPH_SUCCESS;
  }
  GELOGD("Inner graph is invalid, start to init at first");
  return impl_->InitComputeGraph();
}

namespace {
graphStatus ConvertComputeGraphToProto(Graph::DumpFormat format, const ComputeGraphPtr &compute_graph,
                                       google::protobuf::Message *const proto) {
  Model model("", "");
  model.SetGraph(compute_graph);
  switch (format) {
    case Graph::DumpFormat::kOnnx: {
      model.SetName(compute_graph->GetName());
      auto *model_proto = reinterpret_cast<onnx::ModelProto *>(proto);
      GE_ASSERT_NOTNULL(model_proto);
      GE_ASSERT_TRUE(OnnxUtils::ConvertGeModelToModelProto(model, *model_proto, DumpLevel::DUMP_WITH_OUT_DESC),
                     "[Convert][GeModel] DumpGEGraphToOnnx failed.");
      break;
    }
    case Graph::DumpFormat::kTxt: {
      auto *model_proto = reinterpret_cast<proto::ModelDef *>(proto);
      GE_ASSERT_NOTNULL(model_proto);
      GE_ASSERT_GRAPH_SUCCESS(model.Save(*model_proto, true));
      break;
    }
    case Graph::DumpFormat::kReadable: {
      break;
    }
    default: {
      break;
    }
  }
  return SUCCESS;
}

graphStatus ConvertComputeGraphToReadableStr(std::stringstream &readable_ss, const ComputeGraphPtr &compute_graph) {
  return ReadableDump::GenReadableDump(readable_ss, compute_graph) == SUCCESS ? GRAPH_SUCCESS : GRAPH_FAILED;
}

}  // namespace

graphStatus Graph::Dump(Graph::DumpFormat format, std::ostream &o_stream) const {
  GE_ASSERT_TRUE(IsValid());
  const auto compute_graph = impl_->GetComputeGraph();
  GE_ASSERT_NOTNULL(compute_graph);

  onnx::ModelProto onnx_model_proto;
  proto::ModelDef txt_model_proto;
  google::protobuf::Message *proto = nullptr;
  switch (format) {
    case DumpFormat::kOnnx: {
      proto = &onnx_model_proto;
      GE_ASSERT_NOTNULL(proto);
      GE_ASSERT_GRAPH_SUCCESS(ConvertComputeGraphToProto(format, compute_graph, proto));
      GraphUtils::WriteProtoToOStream(*proto, o_stream);
      break;
    }
    case DumpFormat::kTxt: {
      proto = &txt_model_proto;
      GE_ASSERT_NOTNULL(proto);
      GE_ASSERT_GRAPH_SUCCESS(ConvertComputeGraphToProto(format, compute_graph, proto));
      GraphUtils::WriteProtoToOStream(*proto, o_stream);
      break;
    }
    case DumpFormat::kReadable: {
      std::stringstream readable_ss;
      GE_ASSERT_SUCCESS(ConvertComputeGraphToReadableStr(readable_ss, compute_graph));
      GraphUtils::WriteReadableDumpToOStream(readable_ss, o_stream);
    }
    default: {
      break;
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus Graph::DumpToFile(Graph::DumpFormat format, const AscendString &suffix) const {
#ifdef FMK_SUPPORT_DUMP
  GE_ASSERT_TRUE(IsValid());
  const auto compute_graph = impl_->GetComputeGraph();
  GE_ASSERT_NOTNULL(compute_graph);

  onnx::ModelProto onnx_model_proto;
  proto::ModelDef txt_model_proto;
  google::protobuf::Message *proto = nullptr;
  std::string file_absolut_name;
  switch (format) {
    case DumpFormat::kOnnx: {
      proto = &onnx_model_proto;
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::GenDumpOnnxFileName(compute_graph, suffix.GetString(), file_absolut_name));
      GE_ASSERT_NOTNULL(proto);
      GE_ASSERT_GRAPH_SUCCESS(ConvertComputeGraphToProto(format, compute_graph, proto));
      GraphUtils::WriteProtoToTextFile(*proto, file_absolut_name.c_str());
      break;
    }
    case DumpFormat::kTxt: {
      proto = &txt_model_proto;
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::GenDumpTxtFileName(compute_graph, suffix.GetString(), "", file_absolut_name));
      GE_ASSERT_NOTNULL(proto);
      GE_ASSERT_GRAPH_SUCCESS(ConvertComputeGraphToProto(format, compute_graph, proto));
      GraphUtils::WriteProtoToTextFile(*proto, file_absolut_name.c_str());
      break;
    }
    case DumpFormat::kReadable: {
      std::stringstream readable_ss;
      GE_ASSERT_GRAPH_SUCCESS(
          GraphUtils::GenDumpReadableTxtFileName(compute_graph, suffix.GetString(), "", file_absolut_name));
      GE_ASSERT_SUCCESS(ConvertComputeGraphToReadableStr(readable_ss, compute_graph));
      GraphUtils::WriteReadableDumpToTextFile(readable_ss, file_absolut_name.c_str());
      break;
    }
    default: {
      break;
    }
  }
  return SUCCESS;
#else
  (void)format;
  (void)suffix;
  GELOGW("[Graph][DumpToFile] Need to define FMK_SUPPORT_DUMP for dump graph.");
  return FAILED;
#endif
}

GNodePtr Graph::FindNodeByName(const AscendString &node_name) const {
  if (!IsValid()) {
    REPORT_INNER_ERR_MSG("E18888", "current graph is invalid.");
    GELOGE(GRAPH_FAILED, "[Find][Node] impl is invalid.");
    return nullptr;
  }
  auto node = impl_->GetComputeGraph()->FindNode(node_name.GetString());
  GE_ASSERT_NOTNULL(node, "Node name: %s was not found in the current graph: %s.", node_name.GetString(),
                    impl_->GetName().c_str());
  return NodeAdapter::Node2GNodePtr(node);
}

ConstGraphPtr Graph::GetParentGraph() const {
  if (!IsValid()) {
    REPORT_INNER_ERR_MSG("E18888", "current graph is invalid.");
    GELOGE(GRAPH_FAILED, "[Get][ParentGraph] current impl is invalid.");
    return nullptr;
  }
  auto parent_compute_graph_ptr = impl_->GetComputeGraph()->GetParentGraph();
  GE_ASSERT_NOTNULL(parent_compute_graph_ptr);
  return GraphUtilsEx::CreateGraphPtrFromComputeGraph(parent_compute_graph_ptr);
}

GNodePtr Graph::GetParentNode() const {
  if (!IsValid()) {
    REPORT_INNER_ERR_MSG("E18888", "current graph is invalid.");
    GELOGE(GRAPH_FAILED, "[Get][ParentNode] current impl is invalid.");
    return nullptr;
  }
  auto parent_graph_node_ptr = impl_->GetComputeGraph()->GetParentNode();
  GE_ASSERT_NOTNULL(parent_graph_node_ptr);
  return NodeAdapter::Node2GNodePtr(parent_graph_node_ptr);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::CopyGraphImpl(
    const Graph &src_graph, Graph &dst_graph, const std::map<ConstNodePtr, NodePtr> &node_old_2_new,
    const std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new) {
  GE_CHECK_NOTNULL(dst_graph.impl_);
  GE_CHECK_NOTNULL(src_graph.impl_);

  std::map<std::string, Operator> &dst_op_list = dst_graph.impl_->op_list_;
  const std::map<std::string, Operator> &src_op_list = src_graph.impl_->op_list_;
  auto &dst_compute_graph = dst_graph.impl_->compute_graph_;

  dst_graph.impl_->output_name_ = src_graph.impl_->output_name_;

  auto ret = OpDescUtils::CopyOperators(dst_compute_graph, node_old_2_new, op_desc_old_2_new, src_op_list, dst_op_list);
  if (ret != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "copy operators to graph:%s failed.", dst_compute_graph->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Copy][Operators] to graph:%s failed.", dst_compute_graph->GetName().c_str());
    return GRAPH_FAILED;
  }

  ret = OpDescUtils::CopyOperatorLinks(src_op_list, dst_op_list);
  if (ret != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "copy operator links failed, ret:%u.", ret);
    GELOGE(GRAPH_FAILED, "[Copy][OperatorLinks] failed, ret:%u.", ret);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ComputeGraphPtr GraphUtilsEx::GetComputeGraph(const Graph &graph) {
  if (!graph.IsValid()) {
    return nullptr;
  }
  return graph.impl_->compute_graph_;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtilsEx::CreateGraphFromOperatorWithStableTopo(Graph &graph, const std::vector<Operator> &ops) {
  AscendString graph_name;
  GE_ASSERT_SUCCESS(graph.GetName(graph_name));
  GE_ASSERT_TRUE(graph.impl_->compute_graph_ == nullptr, "Compute graph of graph: %s has been created",
                 graph_name.GetString());
  graph.impl_->compute_graph_ = GraphUtilsEx::CreateComputeGraphFromOperatorWithStableTopo(graph_name.GetString(), ops);
  GE_ASSERT_NOTNULL(graph.impl_->compute_graph_);
  return SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Graph
GraphUtilsEx::CreateGraphFromComputeGraph(const ComputeGraphPtr compute_graph) {
  if (compute_graph == nullptr) {
    return Graph("");
  }

  const auto name = compute_graph->GetName();
  const auto graph = Graph(name.c_str());
  if (graph.impl_ == nullptr) {
    return graph;
  }
  graph.impl_->compute_graph_ = compute_graph;
  return graph;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::unique_ptr<Graph>
GraphUtilsEx::CreateGraphUniquePtrFromComputeGraph(const ComputeGraphPtr &compute_graph) {
  GE_ASSERT_NOTNULL(compute_graph);
  auto name = compute_graph->GetName();
  auto graph = ComGraphMakeUnique<Graph>(name.c_str());
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_NOTNULL(graph->impl_);
  graph->impl_->compute_graph_ = compute_graph;
  return graph;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GraphPtr
GraphUtilsEx::CreateGraphPtrFromComputeGraph(const ComputeGraphPtr compute_graph) {
  return CreateGraphUniquePtrFromComputeGraph(compute_graph);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtilsEx::RecoverGraphOperators(const Graph &graph) {
  GE_CHECK_NOTNULL(graph.impl_);
  GE_CHECK_NOTNULL(graph.impl_->compute_graph_);

  graph.impl_->op_list_.clear();
  for (const auto &node : graph.impl_->compute_graph_->GetDirectNode()) {
    graph.impl_->op_list_[node->GetName()] = OpDescUtils::CreateOperatorFromNode(node);
  }
  return SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtilsEx::CopyGraphImpl(
    const Graph &src_graph, Graph &dst_graph, const std::map<ConstNodePtr, NodePtr> &node_old_2_new,
    const std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new) {
  GE_CHECK_NOTNULL(dst_graph.impl_);
  GE_CHECK_NOTNULL(src_graph.impl_);

  std::map<std::string, Operator> &dst_op_list = dst_graph.impl_->op_list_;
  const std::map<std::string, Operator> &src_op_list = src_graph.impl_->op_list_;
  auto &dst_compute_graph = dst_graph.impl_->compute_graph_;

  dst_graph.impl_->output_name_ = src_graph.impl_->output_name_;

  auto ret = OpDescUtils::CopyOperators(dst_compute_graph, node_old_2_new, op_desc_old_2_new, src_op_list, dst_op_list);
  if (ret != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "copy operators to graph:%s failed.", dst_compute_graph->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Copy][Operators] to graph:%s failed.", dst_compute_graph->GetName().c_str());
    return GRAPH_FAILED;
  }

  ret = OpDescUtils::CopyOperatorLinks(src_op_list, dst_op_list);
  if (ret != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "copy operator links failed, ret:%u.", ret);
    GELOGE(GRAPH_FAILED, "[Copy][OperatorLinks] failed, ret:%u.", ret);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
}  // namespace af

#ifdef __cplusplus
extern "C" {
#endif

void GeApiWrapper_RenameAllNodes(void *graph_ptr, const char *prefix) {
  if (graph_ptr == nullptr || prefix == nullptr) {
    return;
  }
  std::string prefix_str(prefix);
  af::Graph *graph = static_cast<af::Graph *>(graph_ptr);
  for (auto &sub_node : graph->GetAllNodes()) {
    auto snode = af::NodeAdapter::GNode2Node(sub_node);
    auto orig_name = snode->GetNamePtr() == nullptr ? std::string("") : std::string(snode->GetNamePtr());
    auto modi_name = prefix_str + '_' + orig_name;
    snode->GetOpDesc()->SetNamePtr(modi_name.c_str());
  }
}

#ifdef __cplusplus
}
#endif
