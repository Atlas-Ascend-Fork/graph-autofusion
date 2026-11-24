/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/tuning_utils.h"

#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/debug/ge_op_types.h"
#include "graph/normal_graph/node_impl.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "graph/utils/recover_ir_utils.h"
#include "common/checker.h"
#include "mmpa/mmpa_api.h"

namespace af {
namespace {
const int64_t kControlIndex = -1;
const char_t *const peer_node_name_attr = "_peerNodeName";
const char_t *const parent_node_name_attr = "_parentNodeName";
const char_t *const alias_name_attr = "_aliasName";
const char_t *const alias_indexes_attr = "_aliasIndexes";
const char_t *const parent_node_anchor_index_attr = "_parentNodeAnchorIndex";
const char_t *const tuning_subgraph_prefix = "/aicore_subgraph_";
const char_t *const non_tuning_subgraph_prefix = "/subgraph_";
const char_t *const kTmpWeightDir = "tmp_weight_";
const char_t *const kOriginName4Recover = "_origin_name_4_recover";
const char_t *const kOriginType4Recover = "_origin_type_4_recover";
const char_t *const kLocation4Recover = "_location_4_recover";
const char_t *const kLength4Recover = "_length_4_recover";
const std::set<std::string> kPartitionOpTypes = {PLACEHOLDER, END};
const std::set<std::string> kExeTypes = {DATA, CONSTANT, FILECONSTANT, NETOUTPUT};
const size_t kConstOpNormalWeightSize = 1U;
const size_t kMaxDataLen = 1048576U;  // 1M
}  // namespace
const std::set<std::string> ir_builder_supported_options_for_lx_fusion = {BUILD_MODE, BUILD_STEP, TUNING_PATH};

const std::set<std::string> build_mode_options = {BUILD_MODE_NORMAL, BUILD_MODE_TUNING, BUILD_MODE_BASELINE,
                                                  BUILD_MODE_OPAT_RESULT};

const std::set<std::string> build_step_options = {
    BUILD_STEP_BEFORE_UB_MATCH, BUILD_STEP_AFTER_UB_MATCH, BUILD_STEP_AFTER_BUILDER, BUILD_STEP_AFTER_BUILDER_SUB,
    BUILD_STEP_BEFORE_BUILD,    BUILD_STEP_AFTER_BUILD,    BUILD_STEP_AFTER_MERGE};

NodeNametoNodeNameMap TuningUtils::data_2_end_;
NodetoNodeNameMap TuningUtils::data_node_2_end_node_;
NodetoNodeMap TuningUtils::data_node_2_netoutput_node_;
NodeVec TuningUtils::netoutput_nodes_;
NodeVec TuningUtils::merged_graph_nodes_;
SubgraphCreateOutNode TuningUtils::create_output_;
std::mutex TuningUtils::mutex_;
std::set<std::string> TuningUtils::reusable_weight_files_;
std::map<std::string, int64_t> TuningUtils::name_to_index_;
std::map<size_t, std::vector<std::string>> TuningUtils::hash_to_files_;

std::string TuningUtils::PrintCheckLog() {
  std::stringstream ss;
  ss << "d2e:{";
  for (const auto &pair : data_2_end_) {
    ss << "data:" << pair.first << "-" << "end:" << pair.second;
    ss << " | ";
  }
  ss << "}";
  ss << "netoutputs:{";
  for (const auto &node : netoutput_nodes_) {
    ss << "netoutput:" << node->GetName();
    ss << " | ";
  }
  ss << "}";
  return ss.str();
}

std::string TuningUtils::GetNodeNameByAnchor(const Anchor *const anchor) {
  if (anchor == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "Anchor is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Anchor is nullptr");
    return "Null";
  }
  const auto node = anchor->GetOwnerNodeBarePtr();
  return (node == nullptr) ? "Null" : node->GetName();
}

// part 1
graphStatus TuningUtils::ConvertGraphToFile(std::vector<ComputeGraphPtr> tuning_subgraphs,
                                            std::vector<ComputeGraphPtr> non_tuning_subgraphs, const bool exe_flag,
                                            const std::string &path, const std::string &user_path) {
  int64_t i = 0;
  int64_t j = 0;
  const std::lock_guard<std::mutex> lock(mutex_);
  reusable_weight_files_.clear();
  name_to_index_.clear();
  hash_to_files_.clear();
  GELOGI("Total tuning graph num: %zu, non tuning graph: %zu.", tuning_subgraphs.size(), non_tuning_subgraphs.size());
  for (auto &subgraph : tuning_subgraphs) {
    (void)create_output_.emplace(subgraph, nullptr);
    auto help_info = HelpInfo{i, exe_flag, true, path, user_path};
    help_info.need_preprocess_ = true;
    if (MakeExeGraph(subgraph, help_info) != SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Invoke][MakeExeGraph] subgraph %zu generate exe graph failed", i);
      return GRAPH_FAILED;
    }
    i++;
  }

  for (auto &subgraph : non_tuning_subgraphs) {
    (void)create_output_.emplace(subgraph, nullptr);
    const auto help_info = HelpInfo{j, true, false, path, user_path};
    if (MakeExeGraph(subgraph, help_info) != SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Invoke][MakeExeGraph] non tuning_subgraph %zu generate exe graph failed", j);
      return GRAPH_FAILED;
    }
    j++;
  }
  create_output_.clear();
  return SUCCESS;
}

graphStatus TuningUtils::ConvertConstToWeightAttr(const ComputeGraphPtr &exe_graph) {
  GELOGI("Start to convert const to weight attr of graph %s.", exe_graph->GetName().c_str());
  for (const auto &node : exe_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (node->GetType() != PLACEHOLDER) {
      continue;
    }
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    std::vector<af::GeTensorPtr> weight;
    TryGetWeight(node, weight);
    if (weight.empty()) {
      continue;
    }
    if (!af::AttrUtils::SetTensor(op_desc, ATTR_NAME_WEIGHTS, weight[0U])) {
      REPORT_INNER_ERR_MSG("E18888", "Set tensor to node[%s] failed", op_desc->GetName().c_str());
      GELOGE(FAILED, "[Set][Tensor] to node[%s] failed", op_desc->GetName().c_str());
      return FAILED;
    }
    GELOGD("Set tensor to node[%s].", op_desc->GetName().c_str());
  }
  return SUCCESS;
}

// +---------------+
// | pld     pld   |
// |  \      /     |
// | relu relu     |
// |   \   /       |
// |   add         |
// |    |          |
// |   end         |
// +---------------+
//        |
//        |
//        V
// +---------------+
// | data   data   |
// |  \      /     |
// | relu relu     |
// |   \   /       |
// |   add         |
// |    |          |
// |  netoutput    |
// +---------------+
graphStatus TuningUtils::MakeExeGraph(ComputeGraphPtr &exe_graph, const HelpInfo &help_info) {
  GE_CHECK_NOTNULL(exe_graph);
  graphStatus ret = exe_graph->TopologicalSortingGraph(true);
  if (ret != SUCCESS) {
    GraphUtils::DumpGEGraphToOnnx(*exe_graph, "black_box");
    REPORT_INNER_ERR_MSG("E18888", "TopologicalSortingGraph [%s] failed, saved to file black_box ret:%u.",
                         exe_graph->GetName().c_str(), ret);
    GELOGE(ret, "[Sort][Graph] Graph[%s] topological sort failed, saved to file black_box ret:%u.",
           exe_graph->GetName().c_str(), ret);
    return ret;
  }
  // clear graph id
  GE_ASSERT_TRUE(AttrUtils::SetStr(*exe_graph, ATTR_NAME_SESSION_GRAPH_ID, ""));
  GELOGI("clear [%s] session_graph_id success", exe_graph->GetName().c_str());
  // if not make exe, just dump and return
  if (!help_info.exe_flag_) {
    if (ConvertConstToWeightAttr(exe_graph) != SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "Convert const to weight attr of graph %s failed", exe_graph->GetName().c_str());
      GELOGE(FAILED, "[Convert][Const] to weight attr of graph %s failed", exe_graph->GetName().c_str());
      return FAILED;
    }
    DumpGraphToPath(exe_graph, help_info.index_, help_info.is_tuning_graph_, help_info.path_);
    GELOGI("just return, dump original sub_graph[%s]index[%" PRId64 "]", exe_graph->GetName().c_str(),
           help_info.index_);
    return SUCCESS;
  }
  // modify sub graph
  for (NodePtr &node : exe_graph->GetDirectNode()) {
    // 1.handle pld
    if (node->GetType() == PLACEHOLDER) {
      GE_ASSERT_GRAPH_SUCCESS(HandlePld(node, help_info.path_));
    }
    // 2.handle end
    if (node->GetType() == END) {
      GE_ASSERT_GRAPH_SUCCESS(HandleEnd(node));
    }
    GE_ASSERT_GRAPH_SUCCESS(HandleConst(node, help_info.path_));
    if (help_info.need_preprocess_) {
      GE_ASSERT_GRAPH_SUCCESS(PreProcessNode(node));
    }
  }
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveNodesByTypeWithoutRelink(exe_graph, std::string(PLACEHOLDER)));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveNodesByTypeWithoutRelink(exe_graph, std::string(END)));
  GE_ASSERT_GRAPH_SUCCESS(exe_graph->TopologicalSortingGraph(true));
  // dump subgraphs which modified by us
  if (help_info.user_path_.empty()) {
    DumpGraphToPath(exe_graph, help_info.index_, help_info.is_tuning_graph_, help_info.path_);
  } else {
    GraphUtils::DumpGEGraph(exe_graph, "", true, help_info.user_path_);
  }
  return SUCCESS;
}

void TuningUtils::DumpGraphToPath(const ComputeGraphPtr &exe_graph, const int64_t index, const bool is_tuning_graph,
                                  std::string path) {
  if (!path.empty()) {
    if (is_tuning_graph) {
      GraphUtils::DumpGEGraph(exe_graph, "", true, path + tuning_subgraph_prefix + std::to_string(index) + ".txt");
    } else {
      GraphUtils::DumpGEGraph(exe_graph, "", true, path + non_tuning_subgraph_prefix + std::to_string(index) + ".txt");
    }
  } else {
    path = "./";
    if (is_tuning_graph) {
      GraphUtils::DumpGEGraph(exe_graph, "", true, path + tuning_subgraph_prefix + std::to_string(index) + ".txt");
    } else {
      GraphUtils::DumpGEGraph(exe_graph, "", true, path + non_tuning_subgraph_prefix + std::to_string(index) + ".txt");
    }
  }
}

void TuningUtils::TryGetWeight(const NodePtr &node, std::vector<af::GeTensorPtr> &weight) {
  // The caller guarantees that the node is not null
  ConstGeTensorPtr ge_tensor = nullptr;
  (void)NodeUtils::TryGetWeightByPlaceHolderNode(node, ge_tensor);
  if (ge_tensor != nullptr) {
    weight.emplace_back(std::const_pointer_cast<GeTensor>(ge_tensor));
  }
}

graphStatus TuningUtils::HandleConst(NodePtr &node, const std::string &aoe_path) {
  if (kConstOpTypes.count(node->GetType()) == 0U) {
    return SUCCESS;
  }
  const auto &weights = OpDescUtils::MutableWeights(node);
  GE_ASSERT_TRUE(weights.size() == kConstOpNormalWeightSize);
  GE_CHECK_NOTNULL(weights[0]);

  const size_t data_length = weights[0]->GetData().GetSize();
  // empty tensor
  if (data_length == 0U) {
    return SUCCESS;
  }

  const auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  GE_ASSERT_TRUE(AttrUtils::SetStr(op_desc, kOriginName4Recover, node->GetName()));
  GE_ASSERT_TRUE(AttrUtils::SetStr(op_desc, kOriginType4Recover, node->GetType()));
  op_desc->SetType(FILECONSTANT);
  op_desc->SetName(op_desc->GetName() + "_" + FILECONSTANT);

  GE_ASSERT_SUCCESS(SetFileConstInfo(node, weights[0U], aoe_path, op_desc));
  weights[0U]->ClearData();
  return SUCCESS;
}

std::string TuningUtils::GenerateFileConstPath(const std::string &aoe_path, const OpDescPtr &op_desc) {
  std::string file_path;
  const std::string *file_path_str = AttrUtils::GetStr(op_desc, parent_node_name_attr);
  if ((file_path_str == nullptr) || (file_path_str->empty())) {
    file_path = op_desc->GetName();
  } else {
    file_path = *file_path_str;
  }
  static std::atomic<int64_t> node_count{0};
  const auto iter = name_to_index_.find(file_path);
  if (iter == name_to_index_.end()) {
    name_to_index_[file_path] = node_count;
    file_path = kTmpWeightDir + std::to_string(mmGetPid()) + "/" + std::to_string(node_count);
    ++node_count;
  } else {
    file_path = kTmpWeightDir + std::to_string(mmGetPid()) + "/" + std::to_string(iter->second);
  }

  if (aoe_path.empty()) {
    return "./" + file_path;
  }
  return aoe_path + "/" + file_path;
}

Status TuningUtils::CheckFilesSame(const std::string &file_name, const char_t *const data, const size_t data_length,
                                   bool &is_content_same) {
  const auto file_buff = ComGraphMakeUnique<char_t[]>(data_length);
  GE_CHECK_NOTNULL(file_buff);
  const auto &real_path = RealPath(file_name.c_str());
  GE_ASSERT_TRUE(!real_path.empty());
  std::ifstream ifs(real_path, std::ifstream::binary);
  GE_ASSERT_TRUE(ifs.is_open());
  (void)ifs.seekg(0, std::ifstream::end);
  const size_t file_length = static_cast<size_t>(ifs.tellg());
  if (data_length != file_length) {
    ifs.close();
    return SUCCESS;
  }
  (void)ifs.seekg(0, std::ifstream::beg);
  (void)ifs.read(static_cast<char_t *>(file_buff.get()), static_cast<std::streamsize>(file_length));
  GE_ASSERT_TRUE(ifs.good());
  ifs.close();
  if ((memcmp(data, file_buff.get(), data_length) == 0)) {
    is_content_same = true;
    GELOGD("Check files with same content success");
  }
  return SUCCESS;
}

Status TuningUtils::GetOrSaveReusableFileConst(const GeTensorPtr &tensor, std::string &file_path) {
  if (reusable_weight_files_.count(file_path) != 0U) {
    GELOGD("File: %s is reusable.", file_path.c_str());
    return SUCCESS;
  }

  const char_t *data = PtrToPtr<uint8_t, char_t>(tensor->GetData().GetData());
  const size_t data_length = tensor->GetData().GetSize();
  GE_ASSERT_TRUE(data_length > 0U);
  const size_t file_buff_len = std::min(data_length, kMaxDataLen);
  const std::string file_buff_str(data, data + file_buff_len);
  const size_t hash_value = std::hash<std::string>{}(file_buff_str);
  GELOGD("Get hash of file[%s] success, value[%zu]", file_path.c_str(), hash_value);
  if (hash_to_files_.find(hash_value) == hash_to_files_.end()) {
    GE_ASSERT_SUCCESS(SaveBinToFile(data, data_length, file_path));
    reusable_weight_files_.emplace(file_path);
    hash_to_files_[hash_value].emplace_back(file_path);
    GELOGD("Save reusable weight file: %s, hash_value: %zu.", file_path.c_str(), hash_value);
    return SUCCESS;
  }

  for (const auto &file : hash_to_files_[hash_value]) {
    bool has_same_content = false;
    GE_ASSERT_SUCCESS(CheckFilesSame(file, data, data_length, has_same_content));
    if (has_same_content) {
      GELOGD("External weight file[%s] can be reused, skip generate file:%s", file.c_str(), file_path.c_str());
      file_path = file;
      return SUCCESS;
    }
  }

  GE_ASSERT_SUCCESS(SaveBinToFile(data, data_length, file_path));
  reusable_weight_files_.emplace(file_path);
  hash_to_files_[hash_value].emplace_back(file_path);
  GELOGD("Save reusable weight file: %s, hash_value: %zu.", file_path.c_str(), hash_value);

  return SUCCESS;
}

graphStatus TuningUtils::SetFileConstInfo(const NodePtr &node, const GeTensorPtr &tensor, const std::string &aoe_path,
                                          const OpDescPtr &op_desc) {
  GE_CHECK_NOTNULL(node->GetOpDesc());
  std::string file_path = GenerateFileConstPath(aoe_path, node->GetOpDesc());
  GELOGD("Generate tmp weight file path: %s of %s.", file_path.c_str(), node->GetName().c_str());
  GE_ASSERT_SUCCESS(GetOrSaveReusableFileConst(tensor, file_path));
  GE_ASSERT_TRUE(AttrUtils::SetStr(op_desc, kLocation4Recover, file_path));

  const int64_t length = static_cast<int64_t>(tensor->GetData().GetSize());
  GE_ASSERT_TRUE(AttrUtils::SetInt(op_desc, kLength4Recover, length));
  const auto tensor_desc = tensor->GetTensorDesc();
  GE_ASSERT_TRUE(AttrUtils::SetDataType(op_desc, VAR_ATTR_DTYPE, tensor_desc.GetDataType()));
  GE_ASSERT_TRUE(AttrUtils::SetListInt(op_desc, VAR_ATTR_SHAPE, tensor_desc.GetShape().GetDims()));

  GELOGD("Convert node: %s to file constant: %s success, file path: %s, length: %ld.", node->GetName().c_str(),
         op_desc->GetName().c_str(), file_path.c_str(), length);

  return SUCCESS;
}

graphStatus TuningUtils::CreateDataNode(NodePtr &node, const std::string &aoe_path, NodePtr &data_node) {
  const auto graph = node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(graph);
  OpDescPtr data_op_desc;
  std::vector<af::GeTensorPtr> weight;
  TryGetWeight(node, weight);
  GeTensorDesc output_desc;
  if (!weight.empty()) {
    GE_ASSERT_TRUE(weight.size() == kConstOpNormalWeightSize);
    GE_CHECK_NOTNULL(weight[0U]);
    const size_t data_length = weight[0U]->GetData().GetSize();
    // empty tensor
    if (data_length == 0U) {
      data_op_desc = ComGraphMakeShared<OpDesc>(node->GetName(), CONSTANT);
    } else {
      const std::string file_const_name = node->GetName() + "_" + FILECONSTANT;
      data_op_desc = ComGraphMakeShared<OpDesc>(file_const_name, FILECONSTANT);
      GE_CHECK_NOTNULL(data_op_desc);
      GE_ASSERT_SUCCESS(SetFileConstInfo(node, weight[0U], aoe_path, data_op_desc));
    }
    output_desc = weight[0U]->GetTensorDesc();
    const std::string *parent_node_name = AttrUtils::GetStr(node->GetOpDesc(), parent_node_name_attr);
    if (parent_node_name != nullptr && (!parent_node_name->empty())) {
      (void)AttrUtils::SetStr(data_op_desc, ATTR_NAME_SRC_CONST_NAME, *parent_node_name);
    }
    GELOGD("Create const node for %s, output_desc shape is:%s", node->GetName().c_str(),
           output_desc.GetShape().ToString().c_str());
  } else {
    data_op_desc = ComGraphMakeShared<OpDesc>(node->GetName(), DATA);
    const auto pld_op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(pld_op_desc);
    output_desc = pld_op_desc->GetOutputDesc(0U);  // only one output for pld and data
    GELOGD("Create data node for %s, output_desc shape is:%s", node->GetName().c_str(),
           output_desc.GetShape().ToString().c_str());
  }
  GE_CHECK_NOTNULL(data_op_desc);
  // data inputdesc & outputdesc set as same
  GE_ASSERT_GRAPH_SUCCESS(data_op_desc->AddInputDesc(output_desc));
  GE_ASSERT_GRAPH_SUCCESS(data_op_desc->AddOutputDesc(output_desc));
  data_node = graph->AddNode(data_op_desc);
  GE_CHECK_NOTNULL(data_node);
  if (data_node->GetType() == CONSTANT) {
    if (OpDescUtils::SetWeights(data_node, weight) != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "const node %s add weight failed", data_op_desc->GetName().c_str());
      GELOGE(FAILED, "[Set][Weights] const node %s add weight failed", data_op_desc->GetName().c_str());
      return FAILED;
    }
  }
  GE_ASSERT_GRAPH_SUCCESS(data_node->SetOwnerComputeGraph(graph));
  return SUCCESS;
}

graphStatus TuningUtils::AddAttrToDataNodeForMergeGraph(const NodePtr &pld, const NodePtr &data_node) {
  const auto op_desc = data_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  const auto pld_desc = pld->GetOpDesc();
  GE_CHECK_NOTNULL(pld_desc);
  // inherit
  // a.  set `end's input node type` as attr
  const std::string *parent_op_type = AttrUtils::GetStr(pld_desc, "parentOpType");
  if (parent_op_type == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "pld %s get parentOpType failed", pld_desc->GetName().c_str());
    GELOGE(FAILED, "[Invoke][GetStr] pld %s get parentOpType failed", pld_desc->GetName().c_str());
    return FAILED;
  }
  (void)AttrUtils::SetStr(op_desc, "parentOpType", *parent_op_type);
  // b. set `end's input node name` as attr
  const std::string *parent_op_name = AttrUtils::GetStr(pld_desc, parent_node_name_attr);
  if (parent_op_name == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "pld %s get _parentNodeName failed", pld_desc->GetName().c_str());
    GELOGE(FAILED, "[Invoke][GetStr] pld %s get _parentNodeName failed", pld_desc->GetName().c_str());
    return FAILED;
  }
  (void)AttrUtils::SetStr(op_desc, parent_node_name_attr, *parent_op_name);
  // c. set `end's input node's out anchor index` as attr
  int32_t parent_node_anchor_index;
  if (!AttrUtils::GetInt(pld_desc, "anchorIndex", parent_node_anchor_index)) {
    REPORT_INNER_ERR_MSG("E18888", "pld %s get anchorIndex failed", pld_desc->GetName().c_str());
    GELOGE(FAILED, "[Invoke][GetStr] pld %s get anchorIndex failed", pld_desc->GetName().c_str());
    return FAILED;
  }
  (void)AttrUtils::SetInt(op_desc, parent_node_anchor_index_attr, parent_node_anchor_index);
  GELOGD("from node %s(%s) to add attr to node %s(%s) success", pld->GetName().c_str(), pld->GetType().c_str(),
         data_node->GetName().c_str(), data_node->GetType().c_str());
  // d. set `end node name` as attr
  const std::string *peer_end_name = AttrUtils::GetStr(pld_desc, peer_node_name_attr);
  if (peer_end_name == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "pld %s get _peerNodeName failed", pld_desc->GetName().c_str());
    GELOGE(FAILED, "[Invoke][GetStr] pld %s get _peerNodeName failed", pld_desc->GetName().c_str());
    return FAILED;
  }
  (void)AttrUtils::SetStr(op_desc, peer_node_name_attr, *peer_end_name);
  GELOGD("from node %s(%s) to add attr to node %s(%s) success", pld->GetName().c_str(), pld->GetType().c_str(),
         data_node->GetName().c_str(), data_node->GetType().c_str());
  return SUCCESS;
}

graphStatus TuningUtils::ChangePld2Data(const NodePtr &node, const NodePtr &data_node) {
  const auto type_pld = node->GetType();
  const auto type_data = data_node->GetType();
  if ((type_pld != PLACEHOLDER) || (kExeTypes.count(type_data) == 0U)) {
    REPORT_INNER_ERR_MSG("E18888", "Failed to change node %s from type %s to type %s", node->GetName().c_str(),
                         type_pld.c_str(), type_data.c_str());
    GELOGE(FAILED, "[Check][Param] Failed to change node %s from type %s to type %s", node->GetName().c_str(),
           type_pld.c_str(), type_data.c_str());
    return FAILED;
  }
  const auto graph = node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(graph);
  std::vector<int32_t> output_map(static_cast<size_t>(node->GetAllOutDataAnchorsSize()));
  for (size_t i = 0UL; i < node->GetAllOutDataAnchorsSize(); ++i) {
    output_map[i] = static_cast<int32_t>(i);
  }

  const auto ret = GraphUtils::ReplaceNodeAnchors(data_node, node, {}, output_map);
  if (ret != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "Failed to replace node %s by node %s, ret:%u", node->GetName().c_str(),
                         data_node->GetName().c_str(), ret);
    GELOGE(FAILED, "[Replace][Node] %s by node %s failed, ret:%u", node->GetName().c_str(),
           data_node->GetName().c_str(), ret);
    return FAILED;
  }

  NodeUtils::UnlinkAll(*node);

  GELOGD("Remove node %s(%s) by the ChangePld2Data process, replace it with node %s(%s)", node->GetName().c_str(),
         node->GetType().c_str(), data_node->GetName().c_str(), data_node->GetType().c_str());
  return ret;
}

graphStatus TuningUtils::HandlePld(NodePtr &node, const std::string &aoe_path) {
  GE_CHECK_NOTNULL(node);
  const auto graph = node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(graph);

  NodePtr data_node = nullptr;
  // 1. create data node
  if (CreateDataNode(node, aoe_path, data_node) != SUCCESS) {
    GELOGE(FAILED, "[Create][DataNode] Failed to handle node %s from graph %s", node->GetName().c_str(),
           graph->GetName().c_str());
    return FAILED;
  }
  // 2. add necessary info to data_node for recovery whole graph
  if (AddAttrToDataNodeForMergeGraph(node, data_node) != SUCCESS) {
    GELOGE(FAILED, "[Add][Attr] Failed to handle node %s from graph %s", node->GetName().c_str(),
           graph->GetName().c_str());
    return FAILED;
  }
  // 3. replace pld node by data node created before
  if (ChangePld2Data(node, data_node) != SUCCESS) {
    GELOGE(FAILED, "[Change][Pld2Data] Failed to handle node %s from graph %s", node->GetName().c_str(),
           graph->GetName().c_str());
    return FAILED;
  }
  GELOGD("pld[%s] handle success", node->GetName().c_str());
  return SUCCESS;
}

graphStatus TuningUtils::CreateNetOutput(const NodePtr &node, NodePtr &out_node) {
  GE_CHECK_NOTNULL(node);
  const auto graph = node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(graph);
  const auto search = create_output_.find(graph);
  if (search == create_output_.end()) {
    REPORT_INNER_ERR_MSG("E18888", "node %s's owner sub graph %s does not exist in create_output map",
                         node->GetName().c_str(), graph->GetName().c_str());
    GELOGE(FAILED, "[Check][Param] node %s's owner sub graph %s does not exist in create_output map",
           node->GetName().c_str(), graph->GetName().c_str());
    return FAILED;
  }
  if (search->second != nullptr) {
    out_node = search->second;
    GELOGD("sub graph %s has created output node, just return", graph->GetName().c_str());
    return SUCCESS;
  }
  const auto out_op_desc = ComGraphMakeShared<OpDesc>(node->GetName(), NETOUTPUT);
  GE_CHECK_NOTNULL(out_op_desc);
  out_node = graph->AddNode(out_op_desc);
  GE_CHECK_NOTNULL(out_node);
  if (out_node->SetOwnerComputeGraph(graph) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "SetOwnerComputeGraph failed, graph:%s", graph->GetName().c_str());
    GELOGE(FAILED, "[Set][Graph] SetOwnerComputeGraph failed, graph:%s", graph->GetName().c_str());
    return FAILED;
  }
  create_output_[graph] = out_node;
  return SUCCESS;
}

graphStatus TuningUtils::AddAttrToNetOutputForMergeGraph(const NodePtr &end, const NodePtr &out_node,
                                                         const int64_t index) {
  GE_CHECK_NOTNULL(end);
  GE_CHECK_NOTNULL(out_node);
  const auto op_desc = out_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  std::vector<std::string> alias_names = {};
  (void)AttrUtils::GetListStr(op_desc, alias_name_attr, alias_names);
  alias_names.push_back(end->GetName());
  (void)AttrUtils::SetListStr(op_desc, alias_name_attr, alias_names);

  std::vector<std::int64_t> indexes = {};
  (void)AttrUtils::GetListInt(op_desc, alias_indexes_attr, indexes);
  indexes.push_back(index);
  (void)AttrUtils::SetListInt(op_desc, alias_indexes_attr, indexes);

  return SUCCESS;
}

graphStatus TuningUtils::LinkEnd2NetOutput(NodePtr &end_node, NodePtr &out_node) {
  GE_CHECK_NOTNULL(end_node);
  GE_CHECK_NOTNULL(out_node);
  GE_CHECK_NOTNULL(end_node->GetInDataAnchor(0));
  // get end in node is control node or normal node
  const AnchorPtr end_in_anchor = (end_node->GetInDataAnchor(0)->GetFirstPeerAnchor() == nullptr)
                                      ? Anchor::DynamicAnchorCast<Anchor>(end_node->GetInControlAnchor())
                                      : Anchor::DynamicAnchorCast<Anchor>(end_node->GetInDataAnchor(0));
  GE_CHECK_NOTNULL(end_in_anchor);
  const auto src_anchor = end_in_anchor->GetFirstPeerAnchor();  // src_anchor should be only 1
  GE_CHECK_NOTNULL(src_anchor);
  if (GraphUtils::RemoveEdge(src_anchor, end_in_anchor) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888",
                         "remove end input edge from %s(%d) to %s(%d) failed. "
                         "node_name:%s, graph_name:%s",
                         GetNodeNameByAnchor(src_anchor.get()).c_str(), src_anchor->GetIdx(),
                         GetNodeNameByAnchor(end_in_anchor.get()).c_str(), end_in_anchor->GetIdx(),
                         end_node->GetName().c_str(), end_node->GetOwnerComputeGraph()->GetName().c_str());
    GELOGE(FAILED,
           "[Remove][Edge] remove end input edge from %s(%d) to %s(%d) failed. "
           "node_name:%s, graph_name:%s",
           GetNodeNameByAnchor(src_anchor.get()).c_str(), src_anchor->GetIdx(),
           GetNodeNameByAnchor(end_in_anchor.get()).c_str(), end_in_anchor->GetIdx(), end_node->GetName().c_str(),
           end_node->GetOwnerComputeGraph()->GetName().c_str());
    return FAILED;
  }
  // add edge between `end in node` and `out_node`
  if (src_anchor->IsTypeIdOf<OutDataAnchor>()) {
    const std::shared_ptr<InDataAnchor> anchor =
        ComGraphMakeShared<InDataAnchor>(out_node, out_node->GetAllInDataAnchors().size());
    GE_CHECK_NOTNULL(anchor);
    GE_CHECK_NOTNULL(out_node->impl_);
    out_node->impl_->in_data_anchors_.push_back(anchor);
    if (GraphUtils::AddEdge(src_anchor, anchor) != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "add edge from %s(%d) to %s(%d) failed. node_name:%s, graph_name:%s",
                           GetNodeNameByAnchor(src_anchor.get()).c_str(), src_anchor->GetIdx(),
                           GetNodeNameByAnchor(anchor.get()).c_str(), anchor->GetIdx(), end_node->GetName().c_str(),
                           end_node->GetOwnerComputeGraph()->GetName().c_str());
      GELOGE(FAILED, "[Add][Edge] from %s(%d) to %s(%d) failed. node_name:%s, graph_name:%s",
             GetNodeNameByAnchor(src_anchor.get()).c_str(), src_anchor->GetIdx(),
             GetNodeNameByAnchor(anchor.get()).c_str(), anchor->GetIdx(), end_node->GetName().c_str(),
             end_node->GetOwnerComputeGraph()->GetName().c_str());
      return FAILED;
    }
    const auto end_op_desc = end_node->GetOpDesc();
    GE_CHECK_NOTNULL(end_op_desc);
    const auto out_node_op_desc = out_node->GetOpDesc();
    GE_CHECK_NOTNULL(out_node_op_desc);
    // end node always has one input
    if (out_node_op_desc->AddInputDesc(end_op_desc->GetInputDesc(0U)) != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "node %s add input desc failed.", out_node_op_desc->GetName().c_str());
      GELOGE(FAILED, "[Add][InputDesc] failed, node %s .", out_node_op_desc->GetName().c_str());
      return FAILED;
    }
    // add necessary info to out_node for recovery whole graph
    if (AddAttrToNetOutputForMergeGraph(end_node, out_node, static_cast<int64_t>(anchor->GetIdx())) != SUCCESS) {
      GELOGE(FAILED, "[Add][Attr] Failed to handle node %s from graph %s", end_node->GetName().c_str(),
             end_node->GetOwnerComputeGraph()->GetName().c_str());
      return FAILED;
    }
  } else if (src_anchor->IsTypeIdOf<OutControlAnchor>()) {
    OpDescPtr noop = nullptr;
    noop = ComGraphMakeShared<OpDesc>(end_node->GetName() + NOOP, NOOP);
    GE_CHECK_NOTNULL(noop);
    const auto noop_node = end_node->GetOwnerComputeGraph()->AddNode(noop);
    GE_CHECK_NOTNULL(noop_node);
    const auto out_in_anchor = out_node->GetInControlAnchor();
    if ((GraphUtils::AddEdge(src_anchor, noop_node->GetInControlAnchor()) != GRAPH_SUCCESS) ||
        (GraphUtils::AddEdge(noop_node->GetOutControlAnchor(), out_in_anchor) != GRAPH_SUCCESS)) {
      REPORT_INNER_ERR_MSG("E18888", "add edge from %s(%d) to %s(%d) failed. node_name:%s, graph_name:%s",
                           GetNodeNameByAnchor(src_anchor.get()).c_str(), src_anchor->GetIdx(),
                           GetNodeNameByAnchor(noop_node->GetInControlAnchor().get()).c_str(),
                           noop_node->GetInControlAnchor()->GetIdx(), end_node->GetName().c_str(),
                           end_node->GetOwnerComputeGraph()->GetName().c_str());
      GELOGE(FAILED, "[Add][Edge] from %s(%d) to %s(%d) failed. node_name:%s, graph_name:%s",
             GetNodeNameByAnchor(src_anchor.get()).c_str(), src_anchor->GetIdx(),
             GetNodeNameByAnchor(noop_node->GetInControlAnchor().get()).c_str(),
             noop_node->GetInControlAnchor()->GetIdx(), end_node->GetName().c_str(),
             end_node->GetOwnerComputeGraph()->GetName().c_str());
      return FAILED;
    }
    // add necessary info to out_node for recovery whole graph
    if (AddAttrToNetOutputForMergeGraph(end_node, out_node, kControlIndex) != SUCCESS) {
      GELOGE(FAILED, "[Add][Attr] Failed to handle node %s from graph %s", end_node->GetName().c_str(),
             end_node->GetOwnerComputeGraph()->GetName().c_str());
      return FAILED;
    }
  } else {
    REPORT_INNER_ERR_MSG("E18888", "failed to handle node_name:%s, graph_name:%s", end_node->GetName().c_str(),
                         end_node->GetOwnerComputeGraph()->GetName().c_str());
    GELOGE(FAILED, "[Handle][Node] failed to handle node_name:%s, graph_name:%s", end_node->GetName().c_str(),
           end_node->GetOwnerComputeGraph()->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

graphStatus TuningUtils::ChangeEnd2NetOutput(NodePtr &end_node, NodePtr &out_node) {
  GE_CHECK_NOTNULL(end_node);
  GE_CHECK_NOTNULL(out_node);
  const auto type_end = end_node->GetType();
  const auto type_out = out_node->GetType();
  if ((type_end != END) || (type_out != NETOUTPUT)) {
    REPORT_INNER_ERR_MSG("E18888", "Failed to change end_node %s from type %s to type %s", end_node->GetName().c_str(),
                         type_end.c_str(), type_out.c_str());
    GELOGE(FAILED, "[Check][Param] Failed to change end_node %s from type %s to type %s", end_node->GetName().c_str(),
           type_end.c_str(), type_out.c_str());
    return FAILED;
  }
  // link all `end nodes's in node` to this out_node
  if (LinkEnd2NetOutput(end_node, out_node) != SUCCESS) {
    GELOGE(FAILED, "[Invoke][LinkEnd2NetOutput] failed, end_node [%s].", end_node->GetName().c_str());
    return FAILED;
  }
  // remove `end node`
  NodeUtils::UnlinkAll(*end_node);
  return SUCCESS;
}

graphStatus TuningUtils::HandleEnd(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  const auto graph = node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(graph);
  NodePtr out_node = nullptr;

  // 1. create net_output node , add only one NetOutput node to one subgraph
  if (CreateNetOutput(node, out_node) != SUCCESS) {
    GELOGE(FAILED, "[Create][NetOutput] Failed to handle node %s from graph %s", node->GetName().c_str(),
           graph->GetName().c_str());
    return FAILED;
  }
  // 2. replace all end nodes by one output node created before
  if (ChangeEnd2NetOutput(node, out_node) != SUCCESS) {
    GELOGE(FAILED, "[Invoke][ChangeEnd2NetOutput] Failed to handle node %s from graph %s", node->GetName().c_str(),
           graph->GetName().c_str());
    return FAILED;
  }
  GELOGD("end[%s] handle success", node->GetName().c_str());
  return SUCCESS;
}

// part 2
graphStatus TuningUtils::ConvertFileToGraph(const std::map<int64_t, std::string> &options, af::Graph &graph) {
  // 1. get all subgraph object
  std::vector<ComputeGraphPtr> root_graphs;
  std::map<std::string, std::vector<ComputeGraphPtr>> name_to_subgraphs;
  if (LoadGraphFromFile(options, root_graphs, name_to_subgraphs) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "Load graph from file according to options failed, options size:%zu", options.size());
    return GRAPH_FAILED;
  }

  // 2. merge root graph
  ComputeGraphPtr merged_root_graph = ComGraphMakeShared<ComputeGraph>("whole_graph_after_tune");
  GE_CHECK_NOTNULL(merged_root_graph);
  if (MergeGraph(root_graphs, merged_root_graph) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "merge root graph failed");
    return GRAPH_FAILED;
  }

  // 3. merge subgraphs
  std::map<std::string, ComputeGraphPtr> name_to_merged_subgraph;
  for (const auto &pair : name_to_subgraphs) {
    ComputeGraphPtr merged_subgraph = ComGraphMakeShared<ComputeGraph>(pair.first);
    GE_CHECK_NOTNULL(merged_subgraph);
    if (MergeGraph(pair.second, merged_subgraph) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "merge root graph failed");
      return GRAPH_FAILED;
    }
    name_to_merged_subgraph[pair.first] = merged_subgraph;
  }

  // 4. construct relation of root graph and subgraphs
  const auto ret_link_subgraph = LinkSubgraph(merged_root_graph, merged_root_graph, name_to_merged_subgraph);
  if (ret_link_subgraph != GRAPH_SUCCESS) {
    return ret_link_subgraph;
  }

  // 5. construct relation of root graph and subgraph of subgrah
  for (const auto &subgraph_iter : name_to_merged_subgraph) {
    const auto ret = LinkSubgraph(merged_root_graph, subgraph_iter.second, name_to_merged_subgraph);
    if (ret != GRAPH_SUCCESS) {
      return ret;
    }
  }

  graph = GraphUtilsEx::CreateGraphFromComputeGraph(merged_root_graph);
  return SUCCESS;
}

graphStatus TuningUtils::LinkSubgraph(ComputeGraphPtr &root_graph, const ComputeGraphPtr &graph,
                                      const std::map<std::string, ComputeGraphPtr> &name_to_merged_subgraph) {
  for (const auto &node : graph->GetDirectNode()) {
    const auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    for (const auto &subgraph_name : op_desc->GetSubgraphInstanceNames()) {
      const auto iter = name_to_merged_subgraph.find(subgraph_name);
      if (iter == name_to_merged_subgraph.end()) {
        REPORT_INNER_ERR_MSG("E18888", "can not find subgraph with name:%s for op:%s.", subgraph_name.c_str(),
                             op_desc->GetName().c_str());
        GELOGE(GRAPH_FAILED, "can not find subgraph with name:%s for op:%s", subgraph_name.c_str(),
               op_desc->GetName().c_str());
        return GRAPH_FAILED;
      }

      iter->second->SetParentNode(node);
      iter->second->SetParentGraph(graph);
      (void)root_graph->AddSubGraph(iter->second);
      GELOGI("add subgraph:%s for node:%s success", subgraph_name.c_str(), op_desc->GetName().c_str());
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus TuningUtils::MergeGraph(const std::vector<ComputeGraphPtr> &subgraphs,
                                    ComputeGraphPtr &output_merged_compute_graph) {
  GE_CHECK_NOTNULL(output_merged_compute_graph);
  const std::function<void()> callback = [&]() {
    data_2_end_.clear();
    data_node_2_end_node_.clear();
    data_node_2_netoutput_node_.clear();
    netoutput_nodes_.clear();
    merged_graph_nodes_.clear();
  };
  GE_MAKE_GUARD(release, callback);

  // merge graph
  if (MergeAllSubGraph(subgraphs, output_merged_compute_graph) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Merge][Graph] failed");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus TuningUtils::LoadGraphFromFile(const std::map<int64_t, std::string> &options,
                                           std::vector<ComputeGraphPtr> &root_graphs,
                                           std::map<std::string, std::vector<ComputeGraphPtr>> &name_to_subgraphs) {
  // options format like {index:"subgraph_path"}
  for (const auto &pair : options) {
    auto compute_graph = ComGraphMakeShared<ComputeGraph>(std::to_string(pair.first));
    if (!af::GraphUtils::LoadGEGraph(pair.second.c_str(), compute_graph)) {
      REPORT_INNER_ERR_MSG("E18888", "LoadGEGraph from file:%s failed", pair.second.c_str());
      GELOGE(FAILED, "[Load][Graph] from file:%s failed", pair.second.c_str());
    }
    bool is_root_graph = false;
    if (af::AttrUtils::GetBool(compute_graph, ATTR_NAME_IS_ROOT_GRAPH, is_root_graph) && is_root_graph) {
      root_graphs.emplace_back(compute_graph);
    } else {
      const std::string *parent_graph_name = af::AttrUtils::GetStr(compute_graph, ATTR_NAME_PARENT_GRAPH_NAME);
      if (parent_graph_name == nullptr) {
        REPORT_INNER_ERR_MSG("E18888", "get attr ATTR_NAME_PARENT_GRAPH_NAME failed for subgraph.");
        GELOGE(GRAPH_FAILED, "get attr ATTR_NAME_PARENT_GRAPH_NAME failed for subgraph:%s",
               compute_graph->GetName().c_str());
        return GRAPH_FAILED;
      }
      name_to_subgraphs[*parent_graph_name].emplace_back(compute_graph);
    }
  }

  if (root_graphs.empty()) {
    REPORT_INNER_ERR_MSG("E18888", "root graph has no subgraphs, can not merge.");
    GELOGE(GRAPH_FAILED, "root graph has no subgraphs, can not merge");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

// +----------------------------------+
// | const const                      |
// |  \     /                         |
// | netoutput(end,end)               |
// +----------------------------------+
//         +
// +----------------------------------+
// | data(pld)   data(pld)            |
// |  \         /                     |
// | relu     relu                    |
// |   \      /                       |
// |    \   /                         |
// |    add                           |
// |     |                            |
// |  netoutput(end)                  |
// +----------------------------------+
//         +
// +----------------------------------+
// |  data(pld)                       |
// |      /                           |
// |  netoutput                       |
// +----------------------------------+
//        |
//        |
//        V
// +----------------------------------+
// | const     const                  |
// |  \         /                     |
// | relu     relu                    |
// |   \      /                       |
// |    \   /                         |
// |    add                           |
// |     |                            |
// |  netoutput                       |
// +----------------------------------+
graphStatus TuningUtils::MergeAllSubGraph(const std::vector<ComputeGraphPtr> &subgraphs,
                                          ComputeGraphPtr &output_merged_compute_graph) {
  GE_CHECK_NOTNULL(output_merged_compute_graph);
  // 1. handle all subgraphs
  for (auto &subgraph : subgraphs) {
    const Status ret_status = MergeSubGraph(subgraph);
    if (ret_status != SUCCESS) {
      GELOGE(ret_status, "[Invoke][MergeSubGraph] subgraph %s merge failed", subgraph->GetName().c_str());
      return ret_status;
    }
  }

  for (const auto &node : merged_graph_nodes_) {
    (void)output_merged_compute_graph->AddNode(node);
    // set owner graph
    GE_CHK_STATUS_RET(node->SetOwnerComputeGraph(output_merged_compute_graph),
                      "[Set][Graph] node %s set owner graph failed", node->GetName().c_str());
    GELOGD("graph %s add node %s success", output_merged_compute_graph->GetName().c_str(), node->GetName().c_str());
  }

  // 2. remove data and output node added by us
  if (RemoveDataNetoutputEdge(output_merged_compute_graph) != SUCCESS) {
    GELOGE(FAILED, "[Remove][Edge] Failed to merge graph %s", output_merged_compute_graph->GetName().c_str());
    return FAILED;
  }
  const graphStatus ret = output_merged_compute_graph->TopologicalSorting();
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "Graph[%s] topological sort failed, ret:%u.",
                         output_merged_compute_graph->GetName().c_str(), ret);
    GELOGE(ret, "[Sort][Graph] Graph[%s] topological sort failed, ret:%u.",
           output_merged_compute_graph->GetName().c_str(), ret);
    return ret;
  }
  GELOGD("Print-%s", PrintCheckLog().c_str());
  GELOGI("output_merged_compute_graph %s success", output_merged_compute_graph->GetName().c_str());
  return SUCCESS;
}

graphStatus TuningUtils::MergeSubGraph(const ComputeGraphPtr &subgraph) {
  for (auto &node : subgraph->GetDirectNode()) {
    if (kPartitionOpTypes.count(node->GetType()) > 0UL) {
      REPORT_INNER_ERR_MSG("E18888", "subgraph passed in should not contain nodes of end or pld type");
      GELOGE(FAILED, "[Check][Param] subgraph passed in should not contain nodes of end or pld type");
      return FAILED;
    }
    // handle data converted from pld node
    if ((node->GetType() == DATA) || (node->GetType() == CONSTANT)) {
      const auto op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      const std::string *peer_out_name = AttrUtils::GetStr(op_desc, peer_node_name_attr);
      const bool has_valid_str = (peer_out_name != nullptr) && (!peer_out_name->empty());
      if (has_valid_str) {
        const std::lock_guard<std::mutex> lock(mutex_);
        (void)data_2_end_.emplace(op_desc->GetName(), *peer_out_name);
        (void)data_node_2_end_node_.emplace(node, *peer_out_name);
        continue;
      }
    }
    // handle netoutput converted from end node
    if (node->GetType() == NETOUTPUT) {
      const auto op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      std::vector<std::string> out_alias_name;
      const bool has_valid_str =
          (AttrUtils::GetListStr(op_desc, alias_name_attr, out_alias_name)) && (!out_alias_name.empty());
      if (has_valid_str) {
        const std::lock_guard<std::mutex> lock(mutex_);
        netoutput_nodes_.emplace_back(node);
      }
    }
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      merged_graph_nodes_.emplace_back(node);
    }
    GELOGD("subgraph %s add node %s success", subgraph->GetName().c_str(), node->GetName().c_str());
  }
  GELOGI("merge subgraph %s success", subgraph->GetName().c_str());
  return SUCCESS;
}

NodePtr TuningUtils::FindNode(const std::string &name, int64_t &in_index) {
  for (const auto &node : netoutput_nodes_) {
    if (node == nullptr) {
      continue;
    }
    std::vector<std::string> out_alias_name;
    std::vector<int64_t> alias_indexes;
    if (AttrUtils::GetListStr(node->GetOpDesc(), alias_name_attr, out_alias_name) &&
        AttrUtils::GetListInt(node->GetOpDesc(), alias_indexes_attr, alias_indexes) &&
        (out_alias_name.size() == alias_indexes.size())) {
      for (size_t i = 0UL; i < out_alias_name.size(); i++) {
        if (out_alias_name[i] == name) {
          in_index = alias_indexes[i];
          return node;
        }
      }
    }
  }
  return nullptr;
}

graphStatus TuningUtils::RemoveDataNetoutputEdge(ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  // 1. traverse
  for (auto &pair : data_node_2_end_node_) {
    auto data_node = pair.first;
    GE_CHECK_NOTNULL(data_node);
    const auto end_name = pair.second;
    int64_t index = 0;
    auto netoutput_node = FindNode(end_name, index);
    GELOGD("start to find info[%s][%s][%" PRId64 "] ", data_node->GetName().c_str(), end_name.c_str(), index);
    GE_CHECK_NOTNULL(netoutput_node);
    (void)data_node_2_netoutput_node_.emplace(data_node, netoutput_node);
    // 2. get `data out anchor` and `net output in anchor` and `net output in node's out anchor`
    GE_CHECK_NOTNULL(data_node->GetOutDataAnchor(0));
    const AnchorPtr data_out_anchor = (data_node->GetOutDataAnchor(0)->GetFirstPeerAnchor() == nullptr)
                                          ? Anchor::DynamicAnchorCast<Anchor>(data_node->GetOutControlAnchor())
                                          : Anchor::DynamicAnchorCast<Anchor>(data_node->GetOutDataAnchor(0));
    AnchorPtr net_output_in_anchor = nullptr;
    AnchorPtr src_out_anchor = nullptr;
    if (index != kControlIndex) {
      net_output_in_anchor = netoutput_node->GetInDataAnchor(static_cast<int32_t>(index));
      GE_CHECK_NOTNULL(net_output_in_anchor);
      src_out_anchor = net_output_in_anchor->GetFirstPeerAnchor();
    } else {
      net_output_in_anchor = netoutput_node->GetInControlAnchor();
      for (const auto &out_ctrl : net_output_in_anchor->GetPeerAnchorsPtr()) {
        const auto noop_node = out_ctrl->GetOwnerNode();
        GE_CHECK_NOTNULL(noop_node);
        if ((noop_node->GetType() == NOOP) && (noop_node->GetName() == (end_name + NOOP))) {
          src_out_anchor = noop_node->GetInControlAnchor()->GetFirstPeerAnchor();
          // remove noop node
          NodeUtils::UnlinkAll(*noop_node);
          if (GraphUtils::RemoveJustNode(graph, noop_node) != SUCCESS) {
            REPORT_INNER_ERR_MSG("E18888", "noop node [%s] RemoveNodeWithoutRelink failed.",
                                 noop_node->GetName().c_str());
            GELOGE(FAILED, "[Remove][Node]noop node [%s] RemoveNodeWithoutRelink failed.",
                   noop_node->GetName().c_str());
            return FAILED;
          }
          break;
        }
      }
    }
    GE_CHECK_NOTNULL(src_out_anchor);
    GELOGD("get out node:%s 's in anchor(%d) peer_src_node:%s 's out anchor(%d) match info[%s][%s][%" PRId64 "]",
           netoutput_node->GetName().c_str(), net_output_in_anchor->GetIdx(),
           src_out_anchor->GetOwnerNode()->GetName().c_str(), src_out_anchor->GetIdx(), data_node->GetName().c_str(),
           end_name.c_str(), index);

    // 3. relink
    // unlink netoutput_node with it's input in stage 4
    GE_CHECK_NOTNULL(data_out_anchor);
    for (const auto &peer_in_anchor : data_out_anchor->GetPeerAnchors()) {
      if (GraphUtils::RemoveEdge(data_out_anchor, peer_in_anchor) != GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E18888",
                             "[Remove][Edge] from %s(%d) to %s(%d) failed. "
                             "node_name:(data:%s;netoutput:%s), graph_name:%s",
                             GetNodeNameByAnchor(data_out_anchor.get()).c_str(), data_out_anchor->GetIdx(),
                             GetNodeNameByAnchor(peer_in_anchor.get()).c_str(), peer_in_anchor->GetIdx(),
                             data_node->GetName().c_str(), netoutput_node->GetName().c_str(), graph->GetName().c_str());
        GELOGE(FAILED, "[Remove][Edge] from %s(%d) to %s(%d) failed. node_name:(data:%s;netoutput:%s), graph_name:%s",
               GetNodeNameByAnchor(data_out_anchor.get()).c_str(), data_out_anchor->GetIdx(),
               GetNodeNameByAnchor(peer_in_anchor.get()).c_str(), peer_in_anchor->GetIdx(),
               data_node->GetName().c_str(), netoutput_node->GetName().c_str(), graph->GetName().c_str());
        return FAILED;
      }
      if (GraphUtils::AddEdge(src_out_anchor, peer_in_anchor) != GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E18888",
                             "add edge from %s(%d) to %s(%d) failed. "
                             "node_name:(data:%s;netoutput:%s), graph_name:%s",
                             GetNodeNameByAnchor(src_out_anchor.get()).c_str(), src_out_anchor->GetIdx(),
                             GetNodeNameByAnchor(peer_in_anchor.get()).c_str(), peer_in_anchor->GetIdx(),
                             data_node->GetName().c_str(), netoutput_node->GetName().c_str(), graph->GetName().c_str());
        GELOGE(FAILED, "[Add][Edge] from %s(%d) to %s(%d) failed. node_name:(data:%s;netoutput:%s), graph_name:%s",
               GetNodeNameByAnchor(src_out_anchor.get()).c_str(), src_out_anchor->GetIdx(),
               GetNodeNameByAnchor(peer_in_anchor.get()).c_str(), peer_in_anchor->GetIdx(),
               data_node->GetName().c_str(), netoutput_node->GetName().c_str(), graph->GetName().c_str());
        return FAILED;
      }
    }
  }
  // 4. remove out nodes added by us
  for (auto &node : netoutput_nodes_) {
    NodeUtils::UnlinkAll(*node);
    if (GraphUtils::RemoveNodeWithoutRelink(graph, node) != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E18888", "Failed to remove node %s from graph", node->GetName().c_str());
      GELOGE(FAILED, "[Remove][Node] %s from graph failed.", node->GetName().c_str());
      return FAILED;
    }
    GELOGD("Remove node %s by the RemoveDataNetoutputEdge process success", node->GetName().c_str());
  }
  return SUCCESS;
}

graphStatus TuningUtils::PreProcessNode(const NodePtr &node) {
  const auto &op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  if (op_desc->GetType() == PLACEHOLDER || op_desc->GetType() == END) {
    return GRAPH_SUCCESS;
  }
  // strep 0: recovery ir
  if (op_desc->GetIrInputs().empty() && op_desc->GetIrOutputs().empty() && (op_desc->GetAllOutputsDescSize() != 0U)) {
    GE_ASSERT_GRAPH_SUCCESS(RecoverIrUtils::RecoverOpDescIrDefinition(op_desc), "Failed to recover ir def for %s %s",
                            op_desc->GetNamePtr(), op_desc->GetTypePtr());
    GELOGI("Node %s %s recover ir def successfully", node->GetNamePtr(), node->GetTypePtr());
  }
  GELOGI("Node %s %s pre-process successfully", node->GetNamePtr(), node->GetTypePtr());
  return GRAPH_SUCCESS;
}
}  // namespace af
