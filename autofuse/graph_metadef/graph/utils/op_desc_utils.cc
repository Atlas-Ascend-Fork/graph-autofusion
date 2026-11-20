/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/op_desc_utils.h"

#include "common/util/mem_utils.h"
#include "common/checker.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/ge_context.h"
#include "graph/op_desc.h"
#include "normal_graph/op_desc_impl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/type/sym_dtype.h"
#include "mmpa/mmpa_api.h"

/*lint -e512 -e737 -e752*/
namespace af {
namespace {
const uint32_t CONST_OP_NORMAL_WEIGHT_SIZE = 1U;
const char *const kMultiThreadCompile = "MULTI_THREAD_COMPILE";
const char *const kDisEnableFlag = "0";
void GetConstantOpName(std::string &op_name) {
  thread_local int64_t const_count = 0;
  std::string compile_thread;
  if ((GetContext().GetOption(kMultiThreadCompile, compile_thread) == GRAPH_SUCCESS) &&
      (compile_thread.compare(kDisEnableFlag) == 0)) {
    op_name = "dynamic_const_" + std::to_string(const_count);
  } else {
    op_name = "dynamic_const_" + std::to_string(GeLog::GetTid()) + "_" + std::to_string(const_count);
  }
  ++const_count;
}

}  // namespace

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDescUtils::ClearInputDesc(const OpDescPtr op_desc,
                                                                                const uint32_t index) {
  return NodeUtils::ClearInputDesc(op_desc, index);
}

GeTensorPtr OpDescUtils::MutableWeights(OpDesc &op_desc) {
  GeTensorPtr weight = nullptr;
  (void)AttrUtils::MutableTensor(&op_desc, ATTR_NAME_WEIGHTS, weight);
  return weight;
}

GE_FUNC_HOST_VISIBILITY GeTensorPtr OpDescUtils::MutableWeights(const OpDescPtr op_desc) {
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "op_desc is null, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] op_desc is null");
    return nullptr;
  }
  return MutableWeights(*op_desc);
}

graphStatus OpDescUtils::SetWeights(OpDesc &op_desc, const GeTensorPtr weight) {
  if (weight == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "weight is null, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] weight is null");
    return GRAPH_FAILED;
  }
  return AttrUtils::SetTensor(&op_desc, ATTR_NAME_WEIGHTS, weight) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus OpDescUtils::SetWeights(OpDescPtr op_desc, const GeTensorPtr weight) {
  GE_CHECK_NOTNULL(op_desc);
  GE_CHECK_NOTNULL(weight);
  return SetWeights(*op_desc, weight);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<NodePtr> OpDescUtils::GetConstInputs(
    const ConstNodePtr &node) {
  if (node == nullptr) {
    return std::vector<NodePtr>();
  }
  return GetConstInputs(*node);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<NodePtr> OpDescUtils::GetConstInputs(const Node &node,
                                                                                                const uint32_t depth) {
  std::vector<NodePtr> ret;
  if (depth == 0U) {
    return ret;
  }

  const auto in_anchors = node.GetAllInDataAnchorsPtr();
  for (const auto &in_anchor : in_anchors) {
    const auto out_anchor = in_anchor->GetPeerOutAnchor();
    if (out_anchor == nullptr) {
      continue;
    }

    const auto in_node = out_anchor->GetOwnerNode();
    if (in_node->GetType() == CONSTANT) {
      ret.push_back(in_node);
    } else if ((in_node->GetType() == SWITCH) && (node.GetType() == MATMUL)) {
      // const --> switch --> matmul
      auto switch_input = GetConstInputs(*in_node, depth - 1U);
      if (switch_input.size() > 0U) {
        (void)ret.insert(ret.end(), switch_input.begin(), switch_input.end());
      }
    } else if (in_node->GetType() == DATA) {
      const auto parent = NodeUtils::GetParentInput(in_node);
      if ((parent != nullptr) && (parent->GetType() == CONSTANT)) {
        ret.push_back(parent);
      }
    } else {
      // do nothing
    }
  }
  return ret;
}

graphStatus OpDescUtils::SetNoneConstNodeWeights(Node &node, const std::vector<GeTensorPtr> &weights) {
  const auto input_nodes = GetConstInputs(node);
  if (weights.size() < input_nodes.size()) {
    REPORT_INNER_ERR_MSG("E18888", "weights count:%zu can't be less than const input count:%zu, node:%s(%s)",
                         weights.size(), input_nodes.size(), node.GetName().c_str(), node.GetType().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] weights count:%zu can't be less than const input count:%zu", weights.size(),
           input_nodes.size());
    return GRAPH_PARAM_INVALID;
  }

  NamedAttrs named_attrs;
  (void)AttrUtils::SetListTensor(named_attrs, "key", weights);
  std::vector<GeTensorPtr> copy_weights;
  (void)AttrUtils::MutableListTensor(named_attrs, "key", copy_weights);

  for (size_t i = 0UL; i < input_nodes.size(); ++i) {
    if (input_nodes[i]->GetOpDesc() != nullptr) {
      if (SetWeights(input_nodes[i]->GetOpDesc(), copy_weights[i]) != GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E18888", "set weights failed, node:%s(%s)", input_nodes[i]->GetName().c_str(),
                             input_nodes[i]->GetType().c_str());
        GELOGE(GRAPH_FAILED, "[Set][Weights] failed, node:%s(%s)", input_nodes[i]->GetName().c_str(),
               input_nodes[i]->GetType().c_str());
        return GRAPH_FAILED;
      }
    }
  }

  // If set more weights than constop, need to add constop
  for (size_t i = input_nodes.size(); i < copy_weights.size(); ++i) {
    // Use org weight before SetWeights Overwrite
    const auto const_opdesc = CreateConstOpZeroCopy(copy_weights[i]);
    GE_CHECK_NOTNULL(const_opdesc);

    const auto owner_graph = node.GetOwnerComputeGraph();
    if (owner_graph == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "node's graph is empty, node name: %s", node.GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Get][Graph] node's graph is empty, name: %s", node.GetName().c_str());
      return GRAPH_PARAM_INVALID;
    }
    const auto const_node = owner_graph->AddNodeFront(const_opdesc);
    GE_CHK_BOOL_EXEC(
        node.AddLinkFrom(const_node) == GRAPH_SUCCESS,
        REPORT_INNER_ERR_MSG("E18888", "node:%s add link failed.", node.GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Invoke][AddLinkFrom] graph add link failed! node:%s", node.GetName().c_str());
        return GRAPH_FAILED);
    const std::vector<NodePtr> original_nodes;
    GraphUtils::RecordOriginalNames(original_nodes, const_node);
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<GeTensorPtr> OpDescUtils::MutableWeights(const Node &node) {
  std::vector<GeTensorPtr> ret;
  auto op_desc = node.GetOpDesc();
  GE_CHK_BOOL_EXEC(op_desc != nullptr, REPORT_INNER_ERR_MSG("E18888", "param node's op_desc is nullptr.");
                   return ret, "[Check][Param] op_desc is nullptr!");
  // Const operator, take the weight directly
  if ((op_desc->GetType() == CONSTANT) || (op_desc->GetType() == CONSTANTOP)) {
    const auto weight = MutableWeights(op_desc);
    if (weight == nullptr) {
      GELOGD("op type %s has no weight, op name:%s", node.GetType().c_str(), node.GetName().c_str());
      return ret;
    }
    ret.push_back(weight);
    return ret;
  }
  // Place holder operator, try to get the weight from parent node
  // when parent node is const operator
  if (node.GetType() == PLACEHOLDER) {
    ConstGeTensorPtr ge_tensor = nullptr;
    if (NodeUtils::TryGetWeightByPlaceHolderNode(std::const_pointer_cast<Node>(node.shared_from_this()), ge_tensor) ==
            GRAPH_SUCCESS &&
        ge_tensor != nullptr) {
      ret.push_back(std::const_pointer_cast<GeTensor>(ge_tensor));
    }
    return ret;
  }

  if (node.GetType() == DATA) {
    ConstGeTensorPtr ge_tensor = nullptr;
    if (NodeUtils::TryGetWeightByDataNode(std::const_pointer_cast<Node>(node.shared_from_this()), ge_tensor) ==
            GRAPH_SUCCESS &&
        ge_tensor != nullptr) {
      ret.push_back(std::const_pointer_cast<GeTensor>(ge_tensor));
    }
    return ret;
  }

  // Other operators, get weights from connected constop
  const auto input_nodes = GetConstInputs(node);
  for (const auto &input_node : input_nodes) {
    const auto temp_weight = MutableWeights(input_node->GetOpDesc());
    if (temp_weight == nullptr) {
      REPORT_INNER_ERR_MSG("E18888", "const op's weight is null, name: %s", input_node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Invoke][MutableWeights] const op's weight is null, name: %s",
             input_node->GetName().c_str());
      return std::vector<GeTensorPtr>();
    }
    ret.push_back(temp_weight);
  }

  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::vector<GeTensorPtr> OpDescUtils::MutableWeights(
    const NodePtr node) {
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "node is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Node is nullptr");
    return std::vector<GeTensorPtr>();
  }
  return MutableWeights(*node);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDescUtils::SetWeights(Node &node, const std::vector<GeTensorPtr> &weights) {
  GE_CHK_BOOL_EXEC(node.GetOpDesc() != nullptr, REPORT_INNER_ERR_MSG("E18888", "opdesc of node is nullptr.");
                   return GRAPH_PARAM_INVALID, "[Check][Param] node.GetOpDesc is nullptr!");
  if (node.GetOpDesc()->GetType() == CONSTANT) {
    if (weights.size() == CONST_OP_NORMAL_WEIGHT_SIZE) {
      return SetWeights(node.GetOpDesc(), weights[0UL]);
    }
    GELOGI("const op weight size %zu should be 1", weights.size());
    return GRAPH_PARAM_INVALID;
  }

  return SetNoneConstNodeWeights(node, weights);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr OpDescUtils::CopyOpDesc(const ConstOpDescPtr &org_op_desc) {
  return GraphUtils::CopyOpDesc(org_op_desc);
}

OpDescPtr OpDescUtils::CreateConstOp(const GeTensorPtr &tensor_ptr) {
  return CreateConstOp(tensor_ptr, true);
}

OpDescPtr OpDescUtils::CreateConstOpZeroCopy(const GeTensorPtr &tensor_ptr) {
  return CreateConstOp(tensor_ptr, false);
}

OpDescPtr OpDescUtils::CreateConstOp(const GeTensorPtr &tensor_ptr, const bool copy) {
  GE_ASSERT_NOTNULL(tensor_ptr);
  const shared_ptr<OpDesc> const_opdesc = ComGraphMakeShared<OpDesc>();
  GE_ASSERT_NOTNULL(const_opdesc, "[Create][OpDesc] failed.");
  if (copy) {
    GE_ASSERT_GRAPH_SUCCESS(SetWeights(const_opdesc, tensor_ptr), "[Set][Weights] failed, op[%s]",
                            const_opdesc->GetNamePtr());
  } else {
    GE_ASSERT_TRUE(AttrUtils::SetShareTensor(const_opdesc, ATTR_NAME_WEIGHTS, *tensor_ptr),
                   "[Set][ShardTensor] failed for %s.", const_opdesc->GetNamePtr());
  }
  const_opdesc->SetType(CONSTANT);
  std::string op_name;
  GetConstantOpName(op_name);
  const_opdesc->SetName(op_name);
  GELOGI("add const op: %s", const_opdesc->GetNamePtr());
  (void)const_opdesc->AddOutputDesc("y", tensor_ptr->GetTensorDesc());
  GELOGI("after add const op: %s", const_opdesc->GetName().c_str());
  return const_opdesc;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDescUtils::SetWeights(NodePtr node, const std::vector<GeTensorPtr> &weights) {
  GE_CHECK_NOTNULL(node);
  return SetWeights(*node, weights);
}

std::map<size_t, std::pair<size_t, size_t>> OpDescUtils::GetInputIrIndexes2InstanceIndexesPairMap(
    const OpDescPtr &op_desc) {
  std::map<size_t, std::pair<size_t, size_t>> ir_index_to_instance_index_pair_map;

  if (GetIrInputInstanceDescRange(op_desc, ir_index_to_instance_index_pair_map) == GRAPH_SUCCESS) {
    return ir_index_to_instance_index_pair_map;
  }

  return {};
}

std::map<size_t, std::pair<size_t, size_t>> OpDescUtils::GetOutputIrIndexes2InstanceIndexesPairMap(
    const OpDescPtr &op_desc) {
  std::map<size_t, std::pair<size_t, size_t>> ir_index_to_instance_index_pair_map;

  if (GetIrOutputDescRange(op_desc, ir_index_to_instance_index_pair_map) == GRAPH_SUCCESS) {
    return ir_index_to_instance_index_pair_map;
  }

  return {};
}

graphStatus OpDescUtils::GetIrInputInstanceDescRange(const OpDescPtr &op,
                                                     std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range) {
  return af::GetIrInputInstanceDescRange(op, ir_input_2_range);
}

graphStatus OpDescUtils::GetIrInputRawDescRange(const OpDescPtr &op,
                                                std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range) {
  return af::GetIrInputRawDescRange(op, ir_input_2_range);
}

graphStatus OpDescUtils::GetIrOutputDescRange(const OpDescPtr &op,
                                              std::map<size_t, std::pair<size_t, size_t>> &ir_output_2_range) {
  return af::GetIrOutputDescRange(op, ir_output_2_range);
}

}  // namespace af
/*lint +e512 +e737 +e752*/
