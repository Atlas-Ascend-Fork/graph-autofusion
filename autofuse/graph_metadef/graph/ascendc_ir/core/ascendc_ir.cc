/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "ascendc_ir_impl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils_ex.h"
#include "graph/utils/cg_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_op_types.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "expression/const_values.h"
#include "graph/attribute_group/attr_group_serializer_registry.h"
#include "graph/attribute_group/attr_group_shape_env.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "graph/attribute_group/attr_group_symbolic_desc.h"

namespace af {
namespace {
constexpr int32_t kDefaultAlignVal = 1;
constexpr uint32_t kMinMergeAxisFromSize = 2U;
const char *const kAscData = DATA;
const char *const kAscOutput = "Output";
}  // namespace

// TODO ascend attr will be split into asc_attr_group
std::unique_ptr<AfAttrGroupsBase> AscGraphAttr::CloneAf() {
  auto ptr = ComGraphMakeUnique<AscGraphAttr>(*this);
  GE_ASSERT_NOTNULL(ptr);
  return ptr;
}

std::unique_ptr<AfAttrGroupsBase> AscNodeAttr::CloneAf() {
  auto ptr = ComGraphMakeUnique<AscNodeAttr>(*this);
  GE_ASSERT_NOTNULL(ptr);
  return ptr;
}

AscNodeAttr *AscNodeAttr::CreateImpl(Operator &op) {
  auto opdesc = OpDescUtils::GetOpDescFromOperator(op).get();
  GE_ASSERT_NOTNULL(opdesc);
  GE_ASSERT_TRUE(opdesc->GetAttrsGroup<AscNodeAttr>() == nullptr);
  auto attr_group = opdesc->GetOrCreateAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(attr_group);
  return attr_group;
}

AscNodeAttr &AscNodeAttr::operator=(const AscNodeAttr &other) {
  if (this == &other) {
    return *this;
  }
  name = other.name;
  type = other.type;
  sched = other.sched;
  api = other.api;
  tmp_buffers = other.tmp_buffers;

  if (other.ir_attr) {
    ir_attr = other.ir_attr->Clone();
  } else {
    ir_attr.reset();
  }
  return *this;
}

AscNodeAttr *AscNodeAttr::Create(Operator &op) {
  return CreateImpl(op);
}

AscTensorAttr &AscTensorAttr::GetTensorAttr(Operator *op, const uint32_t index) {
  try {
    auto attr_group = GetTensorAttrPtr(op, index);
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(attr_group);
    return *attr_group;
  } catch (const AscIRException &exception) {
    GELOGE(FAILED, "Create failed, reason is %s", exception.GetInfo().error_msg.c_str());
    static AscTensorAttr asc_tensor_attr;
    return asc_tensor_attr;
  }
}

AscTensorAttr *AscTensorAttr::GetTensorAttrPtr(Operator *op, const uint32_t index) {
  GE_ASSERT_NOTNULL(op);
  const auto desc = OpDescUtils::GetOpDescFromOperator(*op);
  GE_ASSERT_NOTNULL(desc);
  auto tensor = desc->MutableOutputDesc(index);
  if (tensor == nullptr) {
    return nullptr;
  }
  const auto attr_group = tensor->GetOrCreateAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(attr_group);
  attr_group->dtype.tensor_desc_ = tensor.get();
  return attr_group;
}

AscTensorAttr &AscTensorAttr::GetTensorAttr(const OutDataAnchor &output) {
  try {
    auto attr_group = GetTensorAttrPtr(output);
    CHECK_NOTNULL_WITH_THROW_EXCEPTION(attr_group);
    return *attr_group;
  } catch (const AscIRException &exception) {
    GELOGE(FAILED, "Create failed, reason is %s", exception.GetInfo().error_msg.c_str());
    static AscTensorAttr asc_tensor_attr;
    return asc_tensor_attr;
  }
}

AscTensorAttr *AscTensorAttr::GetTensorAttrPtr(const OutDataAnchor &output) {
  const auto node = output.GetOwnerNodeBarePtr();
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  const auto tensor_desc = op_desc->MutableOutputDesc(output.GetIdx());
  GE_ASSERT_NOTNULL(tensor_desc);
  const auto attr_group = tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(attr_group);
  attr_group->dtype.tensor_desc_ = tensor_desc.get();
  return attr_group;
}

AscTensorAttr *AscTensorAttr::GetOrCreateFromOpDescRaw(void *op_desc_raw, uint32_t index) {
  GE_ASSERT_NOTNULL(op_desc_raw);
  auto *op_desc = static_cast<OpDesc *>(op_desc_raw);
  auto tensor = op_desc->MutableOutputDesc(index);
  if (tensor == nullptr) {
    return nullptr;
  }
  const auto attr_group = tensor->GetOrCreateAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(attr_group);
  attr_group->dtype.tensor_desc_ = tensor.get();
  return attr_group;
}

std::unique_ptr<AfAttrGroupsBase> AscTensorAttr::CloneAf() {
  auto ptr = ComGraphMakeUnique<AscTensorAttr>(*this);
  GE_ASSERT_NOTNULL(ptr);
  return ptr;
}

void AscNodeOutputs::Init() {
  // node_和out_data_anchor代码逻辑可以保证非空
  for (const auto &output : node_->GetAllOutDataAnchorsPtr()) {
    tensors_.emplace_back(AscTensor(*output));
  }
}

AscTensor &AscNodeOutputs::operator[](uint32_t index) {
  if (tensors_.empty()) {
    Init();
  }
  CHECK_BOOL_WITH_THROW_EXCEPTION(index < tensors_.size(), "index = %u but tensors_.size() = %zu", index,
                                  tensors_.size());
  return tensors_[index];
}

std::vector<AscTensor *> AscNodeOutputs::operator()() {
  if (tensors_.empty()) {
    Init();
  }
  if (tensors_.empty()) {
    return {};
  }
  std::vector<AscTensor *> tensors;
  for (auto &tensor : tensors_) {
    tensors.push_back(&tensor);
  }
  return tensors;
}

void AscNodeInputs::Init() {
  std::vector<AscTensor> tmp_tensors;
  // node_和in_data_anchor代码逻辑可以保证非空
  for (const auto &in_anchor : node_->GetAllInDataAnchorsPtr()) {
    const auto &peer_out_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      GELOGD("node[%s, %s] link [%d] is not ready", node_->GetNamePtr(), node_->GetTypePtr(), in_anchor->GetIdx());
      continue;
    }
    tmp_tensors.emplace_back(AscTensor(*peer_out_anchor));
  }
  tensors_ = std::move(tmp_tensors);
}

// make sure ascend graph is fixed, if index 1 is first linked, tensors_ 0 means index 1, that may cause bug
AscTensor &AscNodeInputs::operator[](uint32_t index) {
  // as not all input is ready at the same time, must call Init on every function call
  Init();
  CHECK_BOOL_WITH_THROW_EXCEPTION(index < tensors_.size());
  return tensors_[index];
}

std::vector<AscTensor *> AscNodeInputs::operator()() {
  // as not all input is ready at the same time, must call Init on every function call
  Init();
  if (tensors_.empty()) {
    return {};
  }
  const auto node = ascir::AscTensorUtils::GetOwner(tensors_[0]);
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  std::vector<AscTensor *> tensors;
  for (auto &tensor : tensors_) {
    tensors.emplace_back(&tensor);
  }
  return tensors;
}

uint32_t AscNodeInputs::Size() {
  // as not all input is ready at the same time, must call Init on every function call
  Init();
  return tensors_.size();
}

// 此处op_desc和GetOrCreateAttrsGroup的返回值未判空,内部构造AscNode前已判空
// 资料需注明不允许外部用户构造AscNode
AscNode::AscNode(const OpDescPtr &op_desc, const ComputeGraphPtr &compute_graph)
    : Node(op_desc, compute_graph),
      inputs(this),
      outputs(this),
      attr(*(op_desc->GetOrCreateAttrsGroup<AscNodeAttr>())) {
  if (op_desc != nullptr) {
    attr.name = op_desc->GetName();
    attr.type = op_desc->GetType();
  }
}

AscNodeIter::AscNodeIter(ComputeGraph::Vistor<NodePtr>::Iterator &&iter) : impl_(iter) {}

AscNodeIter &AscNodeIter::operator++() {
  impl_++;
  return *this;
}

AscNodePtr AscNodeIter::operator*() {
  auto ptr = *impl_;
  return std::dynamic_pointer_cast<AscNode>(ptr);
}

bool AscNodeIter::operator!=(const AscNodeIter &other) const {
  return impl_ != other.impl_;
}

AscNodeVisitor::AscNodeVisitor(ComputeGraph::Vistor<NodePtr> &&visitor) : impl_(visitor) {}

AscNodeIter AscNodeVisitor::begin() {
  return AscNodeIter(impl_.begin());
}

AscNodeIter AscNodeVisitor::end() {
  return AscNodeIter(impl_.end());
}

AscGraphImpl::AscGraphImpl(const char *name) : compute_graph_(ComGraphMakeSharedAndThrow<ComputeGraph>(name)) {}

std::string AscGraphImpl::GetName() const {
  return compute_graph_->GetName();
}

void AscGraphImpl::SetTilingKey(const uint32_t tiling_key) {
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  GE_CHECK_NOTNULL_JUST_RETURN(graph_attr_group_ptr);
  graph_attr_group_ptr->tiling_key = static_cast<int64_t>(tiling_key);
}

int64_t AscGraphImpl::GetTilingKey() const {
  const auto graph_attr_group_ptr = GetGraphAttrsGroup();
  GE_WARN_ASSERT(graph_attr_group_ptr);
  return graph_attr_group_ptr->tiling_key;
}

void AscGraphImpl::SetGraphType(const AscGraphType type) {
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  GE_CHECK_NOTNULL_JUST_RETURN(graph_attr_group_ptr);
  graph_attr_group_ptr->type = type;
}

AscGraphType AscGraphImpl::GetGraphType() const {
  const auto graph_attr_group_ptr = GetGraphAttrsGroup();
  GE_WARN_ASSERT(graph_attr_group_ptr);
  return graph_attr_group_ptr->type;
}

AscNodePtr AscGraphImpl::AddNode(Operator &op) {
  const auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  GE_ASSERT_NOTNULL(op_desc);
  AscNodePtr asc_node = std::make_shared<AscNode>(op_desc, compute_graph_);
  GE_ASSERT_NOTNULL(asc_node);
  GE_ASSERT_GRAPH_SUCCESS(asc_node->Init());
  ConstNodePtr const_node = asc_node;
  GE_ASSERT_GRAPH_SUCCESS(NodeUtilsEx::SetNodeToOperator(op, const_node));
  auto node = compute_graph_->AddNode(asc_node);
  auto new_node = std::dynamic_pointer_cast<AscNode>(node);
  // update
  (void)new_node->inputs();
  (void)new_node->outputs();
  return new_node;
}

AscNodePtr AscGraphImpl::FindNode(const char *name) const {
  auto node = compute_graph_->FindNode(name);
  auto dst_node = std::dynamic_pointer_cast<AscNode>(node);
  return dst_node;
}

AscNodeVisitor AscGraphImpl::GetAllNodes() const {
  return AscNodeVisitor(compute_graph_->GetAllNodes());
}

AscNodeVisitor AscGraphImpl::GetInputNodes() const {
  return AscNodeVisitor(compute_graph_->GetInputNodes());
}

Expression AscGraphImpl::CreateSizeVar(const int64_t value) {
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  GE_ASSERT_NOTNULL(graph_attr_group_ptr);
  const auto expr = Symbol(value);
  const auto size_var = ComGraphMakeShared<SizeVar>(expr);
  GE_ASSERT_NOTNULL(size_var);
  graph_attr_group_ptr->size_vars.push_back(size_var);
  return graph_attr_group_ptr->size_vars.back()->expr;
}

Expression AscGraphImpl::CreateSizeVar(const std::string &name) {
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  GE_ASSERT_NOTNULL(graph_attr_group_ptr);
  const auto expr = Symbol(name.c_str());
  const auto size_var = ComGraphMakeShared<SizeVar>(expr);
  GE_ASSERT_NOTNULL(size_var);
  graph_attr_group_ptr->size_vars.push_back(size_var);
  return graph_attr_group_ptr->size_vars.back()->expr;
}

AxisPtr AscGraphImpl::CreateAxis(const std::string &name, Axis::Type type, const Expression &size,
                                 const std::vector<int64_t> &from, const int64_t split_peer) {
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  GE_ASSERT_NOTNULL(graph_attr_group_ptr);
  auto axis = ComGraphMakeShared<Axis>();
  GE_ASSERT_NOTNULL(axis);
  axis->type = type;
  axis->name = name;
  axis->size = size;
  axis->from = from;
  axis->align = Symbol(kDefaultAlignVal);
  axis->split_pair_other_id = split_peer;
  axis->allow_oversize_axis = false;
  axis->allow_unaligned_tail = true;
  axis->id = static_cast<int64_t>(graph_attr_group_ptr->axis.size());

  graph_attr_group_ptr->axis.push_back(std::move(axis));

  return graph_attr_group_ptr->axis.back();
}

std::vector<AxisPtr> AscGraphImpl::GetAllAxis() const {
  const auto graph_attr_group_ptr = GetGraphAttrsGroup();
  GE_WARN_ASSERT(graph_attr_group_ptr);
  return graph_attr_group_ptr->axis;
}

std::vector<SizeVarPtr> AscGraphImpl::GetAllSizeVar() const {
  const auto graph_attr_group_ptr = GetGraphAttrsGroup();
  GE_WARN_ASSERT(graph_attr_group_ptr);
  return graph_attr_group_ptr->size_vars;
}

TransInfoRoadOfGraph AscGraphImpl::GetAllAxisTransInfo() const {
  const auto graph_attr_group_ptr = GetGraphAttrsGroup();
  GE_WARN_ASSERT(graph_attr_group_ptr);
  return graph_attr_group_ptr->trans_info_road;
}

Axis *AscGraphImpl::FindAxis(const int64_t axis_id) const {
  const auto graph_attr_group_ptr = GetGraphAttrsGroup();
  GE_WARN_ASSERT(graph_attr_group_ptr);
  if (axis_id < 0 || axis_id > static_cast<int64_t>(graph_attr_group_ptr->axis.size())) {
    return nullptr;
  }
  return graph_attr_group_ptr->axis[axis_id].get();
}

std::pair<AxisPtr, AxisPtr> AscGraphImpl::DoSplit(const int64_t axis_id, const std::string &outer_axis_name,
                                                  const std::string &inner_axis_name, const bool is_tile_split) {
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  GE_ASSERT_NOTNULL(graph_attr_group_ptr);
  const auto &axis = graph_attr_group_ptr->axis;
  GE_ASSERT_TRUE((axis_id >= 0) && (static_cast<size_t>(axis_id) < axis.size()));

  const auto &single_axis = *axis[axis_id];
  const std::string inner_suffix = is_tile_split ? "t" : "b";
  const std::string outer_suffix = is_tile_split ? "T" : "B";
  std::string actual_inner_axis_name = inner_axis_name;
  if (actual_inner_axis_name.empty()) {
    actual_inner_axis_name = single_axis.name + inner_suffix;
  }
  std::string actual_outer_axis_name = outer_axis_name;
  if (actual_outer_axis_name.empty()) {
    actual_outer_axis_name = single_axis.name + outer_suffix;
  }
  Expression inner_size;
  Expression outer_size;
  if (single_axis.size == sym::kSymbolOne) {
    inner_size = sym::kSymbolOne;
    outer_size = sym::kSymbolOne;
  } else {
    inner_size = CreateSizeVar(actual_inner_axis_name + "_size");
    outer_size = sym::Ceiling(single_axis.size / inner_size);
  }

  Axis::Type inner_type = is_tile_split ? Axis::kAxisTypeTileInner : Axis::kAxisTypeBlockInner;
  Axis::Type outer_type = is_tile_split ? Axis::kAxisTypeTileOuter : Axis::kAxisTypeBlockOuter;
  auto outter_id = static_cast<int64_t>(graph_attr_group_ptr->axis.size());
  int64_t inner_id = outter_id + 1;
  AxisPtr outer = CreateAxis(actual_outer_axis_name, outer_type, outer_size, {axis_id}, inner_id);
  AxisPtr inner = CreateAxis(actual_inner_axis_name, inner_type, inner_size, {axis_id}, outter_id);
  graph_attr_group_ptr->trans_info_road.push_back({TransType::kSplit, {axis[axis_id]}, {outer, inner}});
  return {outer, inner};
}

std::pair<AxisPtr, AxisPtr> AscGraphImpl::BlockSplit(const int64_t axis_id, const std::string &outer_axis_name,
                                                     const std::string &inner_axis_name) {
  return DoSplit(axis_id, outer_axis_name, inner_axis_name, false);
}

std::pair<AxisPtr, AxisPtr> AscGraphImpl::TileSplit(const int64_t axis_id, const std::string &outer_axis_name,
                                                    const std::string &inner_axis_name) {
  return DoSplit(axis_id, outer_axis_name, inner_axis_name, true);
}

AxisPtr AscGraphImpl::MergeAxis(const std::vector<int64_t> &axis_ids, const std::string &merge_axis_name) {
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  const auto &axis = graph_attr_group_ptr->axis;
  std::string name;
  Expression size = sym::kSymbolOne;
  std::vector<int64_t> from_axis_ids;
  std::vector<AxisPtr> from_axis;
  for (const auto &axis_id : axis_ids) {
    GE_ASSERT_TRUE((axis_id >= 0) && (static_cast<size_t>(axis_id) < axis.size()));
    from_axis.push_back(axis[axis_id]);
    name += axis[axis_id]->name;
    size = size * axis[axis_id]->size;
    from_axis_ids.push_back(axis_id);
  }
  name = merge_axis_name.empty() ? name : merge_axis_name;
  AxisPtr merge_axis = CreateAxis(name, Axis::kAxisTypeMerged, size, from_axis_ids);
  graph_attr_group_ptr->trans_info_road.push_back({TransType::kMerge, from_axis, {merge_axis}});
  return merge_axis;
}

bool AscGraphImpl::BindBlock(const int64_t outter_id, const int64_t inner_id) {
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  GE_ASSERT_NOTNULL(graph_attr_group_ptr);
  const auto &axis = graph_attr_group_ptr->axis;
  GE_ASSERT_TRUE((outter_id >= 0) && (static_cast<size_t>(outter_id) < axis.size()));
  GE_ASSERT_TRUE((inner_id >= 0) && (static_cast<size_t>(inner_id) < axis.size()));

  auto outter_axis = axis[outter_id];
  GE_ASSERT_NOTNULL(outter_axis);
  outter_axis->type = Axis::kAxisTypeBlockOuter;
  outter_axis->name.append("B");

  auto inner_axis = axis[inner_id];
  GE_ASSERT_NOTNULL(inner_axis);
  inner_axis->type = Axis::kAxisTypeBlockInner;
  inner_axis->name.append("b");
  return true;
}

bool AscGraphImpl::DoApplySplit(const AscNodePtr &node, const int64_t outter_id, const int64_t inner_id,
                                const int64_t original_id) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_TRUE(DoApplySchedAxisSplit(node, outter_id, inner_id, original_id));
  GE_ASSERT_TRUE(DoApplyTensorAxisSplit(node, outter_id, inner_id, original_id));
  return true;
}

bool AscGraphImpl::DoApplyTensorAxisSplit(const AscNodePtr &node, const int64_t outter_id, const int64_t inner_id,
                                          const int64_t original_id) {
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  GE_ASSERT_NOTNULL(graph_attr_group_ptr);
  const auto &all_axis = graph_attr_group_ptr->axis;
  // check inner_axis before
  const Expression &split_size = all_axis[inner_id]->size;
  for (uint32_t i = 0; i < node->GetAllOutDataAnchorsSize(); i++) {
    const auto &result =
        AxisUtils::SplitView({node->outputs[i].attr.axis, node->outputs[i].attr.repeats, node->outputs[i].attr.strides},
                             split_size, outter_id, inner_id, original_id);
    GE_ASSERT_TRUE(!result.axis_ids.empty(), "Split out view failed for node %s %s, index %u", node->GetNamePtr(),
                   node->GetTypePtr(), i);
    node->outputs[i].attr.axis = result.axis_ids;
    node->outputs[i].attr.repeats = result.repeats;
    node->outputs[i].attr.strides = result.strides;
  }
  return true;
}

bool AscGraphImpl::DoApplySchedAxisSplit(const AscNodePtr &node, const int64_t outter_id, const int64_t inner_id,
                                         const int64_t original_id) {
  std::vector<int64_t> new_node_attr_axis;
  const auto &node_axis = node->attr.sched.axis;
  for (auto &node_axis_id : node_axis) {
    if (node_axis_id == original_id) {
      new_node_attr_axis.push_back(outter_id);
      new_node_attr_axis.push_back(inner_id);
    } else {
      new_node_attr_axis.push_back(node_axis_id);
    }
  }
  node->attr.sched.axis = new_node_attr_axis;
  return true;
}

bool AscGraphImpl::ApplySplit(const AscNodePtr &node, const int64_t outter_id, const int64_t inner_id) {
  GE_ASSERT_NOTNULL(node);
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  GE_ASSERT_NOTNULL(graph_attr_group_ptr);
  const auto &all_axis = graph_attr_group_ptr->axis;
  GE_ASSERT_TRUE((outter_id >= 0) && (outter_id < static_cast<int64_t>(all_axis.size())) && (inner_id >= 0) &&
                 (inner_id < static_cast<int64_t>(all_axis.size())));
  const auto &out_axis = *all_axis[outter_id];
  const auto &in_axis = *all_axis[inner_id];
  GE_ASSERT_TRUE((out_axis.type == Axis::kAxisTypeBlockOuter && in_axis.type == Axis::kAxisTypeBlockInner) ||
                 (out_axis.type == Axis::kAxisTypeTileOuter && in_axis.type == Axis::kAxisTypeTileInner));
  GE_ASSERT_TRUE((out_axis.from.size() == 1U) && (in_axis.from.size() == 1U) && (out_axis.from[0] == in_axis.from[0]));
  return DoApplySplit(node, outter_id, inner_id, out_axis.from[0]);
}

bool AscGraphImpl::DoApplyMerge(const AscNodePtr &node, const int64_t merged_axis_id,
                                const std::vector<int64_t> &original) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_TRUE(DoApplySchedAxisMerge(node, merged_axis_id, original));
  GE_ASSERT_TRUE(DoApplyTensorAxisMerge(node, merged_axis_id, original));
  return true;
}

bool AscGraphImpl::DoApplySchedAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id,
                                         const std::vector<int64_t> &original) {
  std::vector<int64_t> new_node_attr_axis;
  std::set<int64_t> original_set(original.begin(), original.end());
  GE_ASSERT_TRUE(original_set.size() == original.size(), "merge axis redundant");
  std::set<int64_t> merge_axis_set;
  size_t first_merge_axis_index = SIZE_MAX;
  for (size_t axis_index = 0; axis_index < node->attr.sched.axis.size(); ++axis_index) {
    if (original_set.find(node->attr.sched.axis[axis_index]) != original_set.end()) {
      if (first_merge_axis_index == SIZE_MAX) {
        first_merge_axis_index = axis_index;  // 记录首个待合并轴的位置
      }
      merge_axis_set.emplace(node->attr.sched.axis[axis_index]);
      if (merge_axis_set.size() == original.size()) {
        new_node_attr_axis.insert(new_node_attr_axis.begin() + first_merge_axis_index,
                                  merged_axis_id);  // 合并轴放入首个待合并轴位置
      }
    } else {
      new_node_attr_axis.push_back(node->attr.sched.axis[axis_index]);
    }
  }
  GE_ASSERT_TRUE(merge_axis_set.size() == original.size() || merge_axis_set.empty(),
                 "node {%s} has sched.axis %s but origin is %s", node->GetNamePtr(),
                 ViewMemberToString(node->attr.sched.axis).c_str(), ViewMemberToString(original).c_str());
  node->attr.sched.axis = new_node_attr_axis;
  return true;
}

bool AscGraphImpl::DoApplySchedAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis) {
  const auto &node_axis = node->attr.sched.axis;
  for (const auto axis_id : reordered_axis) {
    const auto it = std::find(node_axis.begin(), node_axis.end(), axis_id);
    GE_ASSERT_TRUE(it != node_axis.end(), "can not find axis_id[%ld] of reordered_axis, node[%s,%s]", axis_id,
                   node->GetNamePtr(), node->GetTypePtr());
  }
  node->attr.sched.axis = reordered_axis;
  return true;
}

bool AscGraphImpl::DoApplyTensorAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis) {
  const auto &node_axis = node->attr.sched.axis;
  for (const auto axis_id : reordered_axis) {
    const auto it = std::find(node_axis.begin(), node_axis.end(), axis_id);
    GE_ASSERT_TRUE(it != node_axis.end(), "can not find axis_id[%ld] of reordered_axis, node[%s,%s]", axis_id,
                   node->GetNamePtr(), node->GetTypePtr());
  }
  for (const auto output_ptr : node->outputs()) {
    auto &output = *output_ptr;
    std::vector<int64_t> new_axis;
    std::vector<Expression> new_repeat;
    std::vector<Expression> new_strides;
    auto output_axis = output.attr.axis;
    for (const auto axis_id : reordered_axis) {
      const auto it = std::find(output_axis.begin(), output_axis.end(), axis_id);
      if (it == output_axis.end()) {
        continue;
      }
      const auto pos = std::distance(output_axis.begin(), it);
      new_axis.push_back(output_axis[pos]);
      new_repeat.push_back(output.attr.repeats[pos]);
      new_strides.push_back(output.attr.strides[pos]);
    }
    output.attr.axis = new_axis;
    output.attr.repeats = new_repeat;
    output.attr.strides = new_strides;
  }
  return true;
}

bool AscGraphImpl::DoCopyAscGraphAttr(const AscGraph &src_asc_graph, AscGraph &dst_asc_graph) {
  return DoCopyAscGraphAttrImpl(AscGraphUtils::GetComputeGraph(src_asc_graph),
                                AscGraphUtils::GetComputeGraph(dst_asc_graph));
}

bool AscGraphImpl::DoCopyAscGraphAttrImpl(const ComputeGraphPtr &src_compute_graph,
                                          const ComputeGraphPtr &dst_compute_graph) {
  GE_ASSERT_NOTNULL(src_compute_graph);
  GE_ASSERT_NOTNULL(dst_compute_graph);
  const auto dst_graph_attr = dst_compute_graph->GetOrCreateAttrsGroup<AscGraphAttr>();
  const auto src_graph_attr = src_compute_graph->GetOrCreateAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(dst_graph_attr);
  GE_ASSERT_NOTNULL(src_graph_attr);

  ascendc_ir::proto::AscGraphAttrGroupsDef asc_graph_group;
  GE_ASSERT_GRAPH_SUCCESS(src_graph_attr->SerializeAttr(asc_graph_group));
  GE_ASSERT_GRAPH_SUCCESS(dst_graph_attr->DeserializeAttr(asc_graph_group));

  return true;
}

bool AscGraphImpl::DoCopyAscNodeAndRelink(const AscGraph &src_asc_graph, AscGraph &dst_asc_graph) {
  const auto src_compute_graph = AscGraphUtils::GetComputeGraph(src_asc_graph);
  auto dst_compute_graph = AscGraphUtils::GetComputeGraph(dst_asc_graph);
  GE_ASSERT_NOTNULL(src_compute_graph);
  GE_ASSERT_NOTNULL(dst_compute_graph);
  NodeCloneMap node_clone_map;
  for (const auto &src_node : src_asc_graph.GetAllNodes()) {
    const auto &op_desc = GraphUtils::CopyOpDesc(src_node->GetOpDesc(), nullptr);
    GE_ASSERT_NOTNULL(op_desc);
    op_desc->SetName(src_node->GetName());
    Operator op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
    auto dst_new_node = dst_asc_graph.AddNode(op);
    node_clone_map[src_node.get()] = std::dynamic_pointer_cast<Node>(dst_new_node);
    DoCopyAscNodeTensorAttr(src_node, dst_new_node);
  }

  for (const auto &src_node : src_compute_graph->GetAllNodes()) {
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RelinkGraphEdges(src_node, node_clone_map));
  }
  return true;
}

bool AscGraphImpl::DoCopyAscNodeTensorAttr(const AscNodePtr &src_node, AscNodePtr &dst_node) {
  // op_desc保证非空
  auto op_desc = dst_node->GetOpDesc();
  auto dst_asc_node_attr = op_desc->GetOrCreateAttrsGroup<AscNodeAttr>();
  auto src_asc_node_attr = src_node->GetOpDesc()->GetOrCreateAttrsGroup<AscNodeAttr>();
  if (src_asc_node_attr != nullptr && dst_asc_node_attr != nullptr) {
    *dst_asc_node_attr = *src_asc_node_attr;
  }
  for (uint32_t i = 0; i < src_node->outputs().size(); i++) {
    GE_ASSERT_NOTNULL(op_desc->MutableOutputDesc(i));
    auto tensor_attr_group = op_desc->MutableOutputDesc(i)->GetAttrsGroup<AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_attr_group);
    *tensor_attr_group = src_node->outputs[i].attr;
  }
  return true;
}

// original中的轴不连续时没法做合轴
// 判断轴是否连续 stride_i == repeat_{i+1} * stride_{i+1}
bool AscGraphImpl::CheckContinuous(const AscNodePtr &node, const uint32_t tensor_index,
                                   const std::vector<int64_t> &original) {
  std::vector<Expression> repeats;
  std::vector<Expression> strides;
  std::set<int64_t> original_set(original.begin(), original.end());
  std::set<int64_t> merge_axis_set;
  auto axis = node->outputs[tensor_index].attr.axis;
  for (uint32_t axis_index = 0U; axis_index < axis.size(); axis_index++) {
    if (original_set.find(axis[axis_index]) != original_set.end()) {
      repeats.emplace_back(node->outputs[tensor_index].attr.repeats[axis_index]);
      strides.emplace_back(node->outputs[tensor_index].attr.strides[axis_index]);
      merge_axis_set.emplace(axis[axis_index]);
    }
  }
  GE_ASSERT_TRUE(merge_axis_set.size() == original_set.size() || merge_axis_set.empty(),
                 "node {%s}'s output[%u] has axis %s but origin is %s", node->GetNamePtr(), tensor_index,
                 ViewMemberToString(axis).c_str(), ViewMemberToString(original).c_str());
  if (repeats.size() <= 1U) {
    return true;
  }
  for (uint32_t i = 0U; i < repeats.size() - 1; i++) {
    auto post_stride = repeats[i + 1] * strides[i + 1];
    if (SymbolicUtils::StaticCheckEq(strides[i], post_stride) != TriBool::kTrue) {
      GELOGD("strides of %u is %s but {repeats * strides} of %u is %s", i, strides[i].Str().get(), i + 1,
             post_stride.Str().get());
      return false;
    }
  }
  return true;
}

bool AscGraphImpl::DoApplyTensorAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id,
                                          const std::vector<int64_t> &original) {
  for (uint32_t i = 0; i < node->GetAllOutDataAnchorsSize(); i++) {
    if (!CheckContinuous(node, i, original)) {
      GELOGW("%s's [%u]th output's view is not continuous.", node->GetNamePtr(), i);
      continue;
    }
    const auto &view =
        AxisUtils::MergeView({node->outputs[i].attr.axis, node->outputs[i].attr.repeats, node->outputs[i].attr.strides},
                             merged_axis_id, original);
    node->outputs[i].attr.axis = view.axis_ids;
    node->outputs[i].attr.repeats = view.repeats;
    node->outputs[i].attr.strides = view.strides;
  }
  return true;
}

bool AscGraphImpl::ApplyMerge(const AscNodePtr &node, const int64_t merged_axis_id) {
  GE_ASSERT_NOTNULL(node);
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  GE_ASSERT_NOTNULL(graph_attr_group_ptr);
  const auto &all_axis = graph_attr_group_ptr->axis;
  GE_ASSERT_TRUE((merged_axis_id >= 0) && (merged_axis_id < static_cast<int64_t>(all_axis.size())));
  const auto &axis = *all_axis[merged_axis_id];
  GE_ASSERT_TRUE((axis.type == Axis::kAxisTypeMerged) && axis.from.size() >= kMinMergeAxisFromSize);
  return DoApplyMerge(node, merged_axis_id, axis.from);
}

bool AscGraphImpl::ApplyTensorAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id) {
  GE_ASSERT_NOTNULL(node);
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  GE_ASSERT_NOTNULL(graph_attr_group_ptr);
  const auto &all_axis = graph_attr_group_ptr->axis;
  GE_ASSERT_TRUE((merged_axis_id >= 0) && (merged_axis_id < static_cast<int64_t>(all_axis.size())));
  const auto &axis = *all_axis[merged_axis_id];
  GE_ASSERT_TRUE((axis.type == Axis::kAxisTypeMerged) && axis.from.size() >= kMinMergeAxisFromSize);
  return DoApplyTensorAxisMerge(node, merged_axis_id, axis.from);
}

bool AscGraphImpl::ApplySchedAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id) {
  GE_ASSERT_NOTNULL(node);
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  GE_ASSERT_NOTNULL(graph_attr_group_ptr);
  const auto &all_axis = graph_attr_group_ptr->axis;
  GE_ASSERT_TRUE((merged_axis_id >= 0) && (merged_axis_id < static_cast<int64_t>(all_axis.size())));
  const auto &axis = *all_axis[merged_axis_id];
  GE_ASSERT_TRUE((axis.type == Axis::kAxisTypeMerged) && axis.from.size() >= kMinMergeAxisFromSize);
  return DoApplySchedAxisMerge(node, merged_axis_id, axis.from);
}

bool AscGraphImpl::ApplyTensorAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id,
                                        const std::vector<int64_t> &original) {
  GE_ASSERT_NOTNULL(node);
  return DoApplyTensorAxisMerge(node, merged_axis_id, original);
}

bool AscGraphImpl::ApplySchedAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id,
                                       const std::vector<int64_t> &original) {
  GE_ASSERT_NOTNULL(node);
  return DoApplySchedAxisMerge(node, merged_axis_id, original);
}

bool AscGraphImpl::ApplyReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_TRUE(DoApplySchedAxisReorder(node, reordered_axis));
  return DoApplyTensorAxisReorder(node, reordered_axis);
}

bool AscGraphImpl::ApplySchedAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis) {
  GE_ASSERT_NOTNULL(node);
  const auto &node_axis = node->attr.sched.axis;
  GE_ASSERT_EQ(node_axis.size(), reordered_axis.size());
  return DoApplySchedAxisReorder(node, reordered_axis);
}

bool AscGraphImpl::ApplyTensorAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis) {
  GE_ASSERT_NOTNULL(node);
  const auto &node_axis = node->attr.sched.axis;
  GE_ASSERT_EQ(node_axis.size(), reordered_axis.size());
  return DoApplyTensorAxisReorder(node, reordered_axis);
}

bool AscGraphImpl::TryApplyAxisReplace(const AscNodePtr &node, const Axis &src, const Axis &dst) {
  GE_ASSERT_NOTNULL(node);
  std::vector<AxisId> new_axes = node->attr.sched.axis;
  bool found{false};
  for (int64_t &id : new_axes) {
    if (id == src.id) {
      id = dst.id;
      found = true;
    }
  }
  node->attr.sched.axis = new_axes;
  for (auto outputs : node->outputs()) {
    auto new_output_axes = outputs->attr.axis;
    for (auto &id : new_output_axes) {
      if (id == src.id) {
        id = dst.id;
        found = true;
      }
    }
    outputs->attr.axis = new_output_axes;
  }
  return found;
}

AscGraphAttr *AscGraphImpl::GetOrCreateGraphAttrsGroup() {
  return compute_graph_->GetOrCreateAttrsGroup<AscGraphAttr>();
}

AscGraphAttr *AscGraphImpl::GetGraphAttrsGroup() const {
  return compute_graph_->GetAttrsGroup<AscGraphAttr>();
}

AscOpOutput AscGraphImpl::CreateContiguousData(const char *name, const ge::DataType &dt, const vector<Axis> &axes,
                                               const size_t index, const Format &format) {
  auto data_op_desc = OpDescBuilder(name, kAscData).AddOutput("y").Build();
  GE_ASSERT_NOTNULL(data_op_desc);
  // Add output and attr
  data_op_desc->AppendIrAttrName("index");
  data_op_desc->AppendIrOutput("y", kIrOutputRequired);
  auto data_op = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(data_op_desc));
  GE_ASSERT_NOTNULL(data_op);
  auto data_attr = data_op_desc->GetOrCreateAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(data_attr);
  AddNode(*data_op);
  data_op_desc->SetExtAttr(ascir::cg::RELATED_OP, data_op);
  auto data_ir_attr = ComGraphMakeUnique<AscDataIrAttrDef>();
  GE_ASSERT_NOTNULL(data_ir_attr);
  GE_ASSERT_GRAPH_SUCCESS(data_ir_attr->SetIndex(index));
  data_attr->ir_attr = std::move(data_ir_attr);

  AscOpOutput asc_op_output(data_op.get(), 0U);  // data只有一个输出
  asc_op_output.dtype = dt;
  asc_op_output.format = format;  // tensor上的format
  GE_ASSERT_TRUE(asc_op_output.SetContiguousView(axes));
  *asc_op_output.vectorized_axis = AxisUtils::GetDefaultVectorizedAxis(*asc_op_output.axis, -1);
  return asc_op_output;
}

AscOpOutput AscGraphImpl::CreateContiguousOut(const char *name, const DataType &dt, const vector<Axis> &axes,
                                              const Format &format) {
  auto out_op_desc = OpDescBuilder(name, kAscOutput).AddInput("x").AddOutput("y").Build();
  GE_ASSERT_NOTNULL(out_op_desc);
  auto out_op = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(out_op_desc));
  GE_ASSERT_NOTNULL(out_op);
  AddNode(*out_op);
  out_op_desc->SetExtAttr(ascir::cg::RELATED_OP, out_op);
  AscOpOutput asc_op_output(out_op.get(), 0U);  // output只有一个输出
  asc_op_output.dtype = dt;
  asc_op_output.format = format;
  GE_ASSERT_TRUE(asc_op_output.SetContiguousView(axes));
  *asc_op_output.vectorized_axis = AxisUtils::GetDefaultVectorizedAxis(*asc_op_output.axis, -1);
  return asc_op_output;
}

const ComputeGraphPtr AscGraphImpl::GetComputeGraph() const {
  return compute_graph_;
}

bool AscGraphImpl::CopyFrom(const AscGraph &src_graph, AscGraph &dst_graph) {
  GE_ASSERT_TRUE(DoCopyAscGraphAttr(src_graph, dst_graph));
  GE_ASSERT_TRUE(DoCopyAscNodeAndRelink(src_graph, dst_graph));
  return true;
}

graphStatus AscGraphImpl::CreateSizeVar(const Expression &expression) {
  const auto graph_attr_group_ptr = GetOrCreateGraphAttrsGroup();
  GE_ASSERT_NOTNULL(graph_attr_group_ptr);
  const auto size_var = ComGraphMakeShared<SizeVar>(expression);
  GE_ASSERT_NOTNULL(size_var);
  graph_attr_group_ptr->size_vars.push_back(size_var);
  return GRAPH_SUCCESS;
}

Status AscGraphImpl::AddSubGraph(const ComputeGraphPtr &sub_graph) const {
  GE_ASSERT_NOTNULL(sub_graph);
  auto root_graph = GraphUtils::FindRootGraph(compute_graph_);
  GE_ASSERT_NOTNULL(root_graph);
  root_graph->AddSubGraph(sub_graph);
  return af::SUCCESS;
}

Status AscGraphImpl::FindSubGraph(const std::string &name, std::shared_ptr<AscGraphImpl> &graph_impl) const {
  auto root_graph = GraphUtils::FindRootGraph(compute_graph_);
  GE_ASSERT_NOTNULL(root_graph);
  auto sub_graph = root_graph->GetSubgraph(name);
  GE_ASSERT_NOTNULL(sub_graph, "Failed to get subgraph named [%s] from [%s].", name.c_str(),
                    compute_graph_->GetName().c_str());

  graph_impl = ComGraphMakeShared<AscGraphImpl>(name.c_str());
  GE_ASSERT_NOTNULL(graph_impl);
  graph_impl->compute_graph_ = sub_graph;
  return af::SUCCESS;
}

AscGraph::AscGraph(const char *name) : impl_(ComGraphMakeSharedAndThrow<AscGraphImpl>(name)) {}

std::string AscGraph::GetName() const {
  return impl_->GetName();
}

bool AscGraph::CopyFrom(const AscGraph &graph) {
  GE_ASSERT_TRUE(impl_->CopyFrom(graph, *this));
  std::vector<AscGraph> sub_graphs;
  GE_ASSERT_SUCCESS(graph.GetAllSubGraphs(sub_graphs));
  for (const auto &sub_graph : sub_graphs) {
    AscGraph new_sub(sub_graph.GetName().c_str());
    GE_ASSERT_TRUE(new_sub.impl_->CopyFrom(sub_graph, new_sub));
    GE_ASSERT_SUCCESS(AddSubGraph(new_sub));
  }
  return true;
}

bool AscGraph::CopyAttrFrom(const AscGraph &src_graph) {
  GE_ASSERT_TRUE(AscGraphImpl::DoCopyAscGraphAttr(src_graph, *this));
  return true;
}

bool AscGraph::CopyAscNodeTensorAttr(const AscNodePtr &src_node, AscNodePtr &dst_node) {
  GE_ASSERT_TRUE(AscGraphImpl::DoCopyAscNodeTensorAttr(src_node, dst_node));
  return true;
}

void AscGraph::SetTilingKey(const uint32_t tiling_key) {
  impl_->SetTilingKey(tiling_key);
}

void AscGraph::SetGraphType(const AscGraphType type) {
  impl_->SetGraphType(type);
}

Status AscGraph::AddSubGraph(const AscGraph &graph) const {
  return impl_->AddSubGraph(graph.impl_->compute_graph_);
}

Status AscGraph::GetAllSubGraphs(std::vector<AscGraph> &graphs) const {
  auto root_graph = GraphUtils::FindRootGraph(impl_->compute_graph_);
  GE_ASSERT_NOTNULL(root_graph);
  auto subgraphs = root_graph->GetAllSubgraphs();
  graphs.reserve(subgraphs.size());
  for (const auto &iter : subgraphs) {
    AscGraph graph(iter->GetName().c_str());
    graph.impl_->compute_graph_ = iter;
    graphs.emplace_back(std::move(graph));
  }
  return af::SUCCESS;
}

Status AscGraph::FindSubGraph(const std::string &name, AscGraph &graph) const {
  return impl_->FindSubGraph(name, graph.impl_);
}

int64_t AscGraph::GetTilingKey() const {
  return impl_->GetTilingKey();
}

AscGraphType AscGraph::GetGraphType() const {
  return impl_->GetGraphType();
}

Expression AscGraph::CreateSizeVar(const int64_t value) {
  return impl_->CreateSizeVar(value);
}

Expression AscGraph::CreateSizeVar(const std::string &name) {
  return impl_->CreateSizeVar(name);
}

graphStatus AscGraph::CreateSizeVar(const Expression &expression) {
  return impl_->CreateSizeVar(expression);
}

Axis &AscGraph::CreateAxis(const std::string &name, const Expression &size) {
  return *(impl_->CreateAxis(name, Axis::kAxisTypeOriginal, size, {}));
}

Axis &AscGraph::CreateAxis(const std::string &name, Axis::Type type, const Expression &size,
                           const std::vector<AxisId> &from, AxisId split_peer) {
  return *(impl_->CreateAxis(name, type, size, from, split_peer));
}

Axis *AscGraph::FindAxis(const int64_t axis_id) {
  return impl_->FindAxis(axis_id);
}

AscNodePtr AscGraph::AddNode(Operator &op) {
  return impl_->AddNode(op);
}

AscNodePtr AscGraph::FindNode(const char *name) const {
  return impl_->FindNode(name);
}

AscNodeVisitor AscGraph::GetAllNodes() const {
  return impl_->GetAllNodes();
}

AscNodeVisitor AscGraph::GetInputNodes() const {
  return impl_->GetInputNodes();
}

std::pair<AxisPtr, AxisPtr> AscGraph::BlockSplit(const int64_t axis_id, const std::string &outer_axis_name,
                                                 const std::string &inner_axis_name) {
  GE_ASSERT_TRUE(IsVarNameValidAllowEmpty(inner_axis_name));
  GE_ASSERT_TRUE(IsVarNameValidAllowEmpty(outer_axis_name));
  return impl_->BlockSplit(axis_id, outer_axis_name, inner_axis_name);
}

std::pair<AxisPtr, AxisPtr> AscGraph::TileSplit(const int64_t axis_id, const std::string &outer_axis_name,
                                                const std::string &inner_axis_name) {
  return impl_->TileSplit(axis_id, outer_axis_name, inner_axis_name);
}

AxisPtr AscGraph::MergeAxis(const std::vector<int64_t> &axis_ids, const std::string &merge_axis_name) {
  return impl_->MergeAxis(axis_ids, merge_axis_name);
}

bool AscGraph::BindBlock(const int64_t outter_id, const int64_t inner_id) {
  return impl_->BindBlock(outter_id, inner_id);
}

bool AscGraph::ApplySplit(const AscNodePtr &node, const int64_t outter_id, const int64_t inner_id) {
  return impl_->ApplySplit(node, outter_id, inner_id);
}

bool AscGraph::ApplyMerge(const AscNodePtr &node, const int64_t merged_axis_id) {
  return impl_->ApplyMerge(node, merged_axis_id);
}

bool AscGraph::ApplySchedAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id) {
  return impl_->ApplySchedAxisMerge(node, merged_axis_id);
}

bool AscGraph::ApplyTensorAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id) {
  return impl_->ApplyTensorAxisMerge(node, merged_axis_id);
}

bool AscGraph::ApplySchedAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id,
                                   const std::vector<int64_t> &original) {
  return impl_->ApplySchedAxisMerge(node, merged_axis_id, original);
}

bool AscGraph::ApplyTensorAxisMerge(const AscNodePtr &node, const int64_t merged_axis_id,
                                    const std::vector<int64_t> &original) {
  return impl_->ApplyTensorAxisMerge(node, merged_axis_id, original);
}

bool AscGraph::ApplyReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis) {
  return impl_->ApplyReorder(node, reordered_axis);
}

bool AscGraph::ApplySchedAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis) {
  return impl_->ApplySchedAxisReorder(node, reordered_axis);
}

bool AscGraph::ApplyTensorAxisReorder(const AscNodePtr &node, const std::vector<int64_t> &reordered_axis) {
  return impl_->ApplyTensorAxisReorder(node, reordered_axis);
}

bool AscGraph::TryApplyAxisReplace(const AscNodePtr &node, const Axis &src, const Axis &dst) {
  return impl_->TryApplyAxisReplace(node, src, dst);
}

std::vector<AxisPtr> AscGraph::GetAllAxis() const {
  return impl_->GetAllAxis();
}

std::vector<SizeVarPtr> AscGraph::GetAllSizeVar() const {
  return impl_->GetAllSizeVar();
}

AscGraph::~AscGraph() {
  for (const auto &node : impl_->GetAllNodes()) {
    if (node == nullptr) {
      continue;
    }
    const auto &op_desc = node->GetOpDesc();
    if (op_desc != nullptr) {
      // 打破shared ptr的循环引用
      op_desc->DelExtAttr(ascir::cg::RELATED_OP);
    }
  }
}

bool AscGraph::CheckExprValid() const {
  int32_t node_index = -1;
  for (const auto &node : GetAllNodes()) {
    node_index++;
    GE_ASSERT_NOTNULL(node, "Node ptr is null, index[%d].", node_index);
    int32_t output_index = -1;
    for (const auto &tensor : node->outputs()) {
      output_index++;
      GE_ASSERT_NOTNULL(tensor, "Tensor ptr is null, index[%d], node name[%s].", output_index, node->GetName().c_str());
    }
  }
  return true;
}

bool AscGraph::CheckAxisValid() const {
  int64_t id_index = 0;
  const auto axes = GetAllAxis();
  for (const auto &axis : axes) {
    GE_ASSERT_NOTNULL(axis, "Axis ptr is null, index[%ld].", id_index);
    GE_ASSERT_TRUE(axis->id == id_index, "Axis index[%ld] is not equal to id[%ld].", id_index, axis->id);
    id_index++;
  }
  int32_t node_index = -1;
  for (const auto &node : GetAllNodes()) {
    node_index++;
    GE_ASSERT_NOTNULL(node, "Node ptr is null, index[%d].", node_index);
    std::set<int64_t> sched_axis_set;
    int32_t sched_axis_index = -1;
    for (const auto &sched_axis : node->attr.sched.axis) {
      sched_axis_index++;
      GE_ASSERT_TRUE(sched_axis >= 0L, "Invalid sched axis[%ld], node_name[%s], index[%d].", sched_axis,
                     node->GetName().c_str(), sched_axis_index);
      GE_ASSERT_TRUE(sched_axis < static_cast<int64_t>(axes.size()),
                     "Invalid sched axis[%ld], node_name[%s], index[%d].", sched_axis, node->GetName().c_str(),
                     sched_axis_index);
      const auto iter = sched_axis_set.find(sched_axis);
      GE_ASSERT_TRUE(iter == sched_axis_set.cend(), "Redundant sched axis[%ld], node_name[%s].", sched_axis,
                     node->GetName().c_str());
      sched_axis_set.insert(sched_axis);
    }
    int32_t output_index = -1;
    for (const auto &tensor : node->outputs()) {
      output_index++;
      GE_ASSERT_TRUE(tensor != nullptr, "Tensor ptr is null, index[%d], node name[%s].", output_index,
                     node->GetName().c_str());
      GE_ASSERT_TRUE(tensor->attr.axis.size() == tensor->attr.repeats.size(),
                     "Tensor axis size[%zu] is not equal to repeat size[%zu], index[%d], node name[%s].",
                     tensor->attr.axis.size(), tensor->attr.repeats.size(), output_index, node->GetName().c_str());
      GE_ASSERT_TRUE(tensor->attr.axis.size() == tensor->attr.strides.size(),
                     "Tensor axis size[%zu] is not equal to stride size[%zu], index[%d], node name[%s].",
                     tensor->attr.axis.size(), tensor->attr.strides.size(), output_index, node->GetName().c_str());
      for (const auto &axis : tensor->attr.axis) {
        GE_ASSERT_TRUE(axis >= 0, "Invalid tensor axis[%ld].", axis);
        GE_ASSERT_TRUE(axis < static_cast<int64_t>(axes.size()), "Invalid tensor axis[%ld].", axis);
      }
      for (const auto &vectorized_axis : tensor->attr.vectorized_axis) {
        GE_ASSERT_TRUE(vectorized_axis >= 0, "Invalid tensor vectorized_axis[%ld].", vectorized_axis);
        GE_ASSERT_TRUE(vectorized_axis < static_cast<int64_t>(axes.size()), "Invalid tensor vectorized_axis[%ld].",
                       vectorized_axis);
      }
    }
  }
  return true;
}

bool AscGraph::CheckTensorValid() const {
  for (const auto &node : GetAllNodes()) {
    int32_t output_index = -1;
    for (const auto &tensor : node->outputs()) {
      output_index++;
      if (tensor->attr.mem.alloc_type == AllocType::kAllocTypeGlobal) {
        continue;
      }
      if ((tensor->attr.buf.id != kIdNone) && (tensor->attr.que.id == kIdNone)) {
        continue;
      }
      if ((tensor->attr.buf.id == kIdNone) && (tensor->attr.que.id != kIdNone)) {
        GE_ASSERT_TRUE(tensor->attr.que.depth > 0, "Invalid que depth[%ld], tensor index[%d], node[%s].",
                       tensor->attr.que.depth, output_index, node->GetName().c_str());
        GE_ASSERT_TRUE(tensor->attr.que.buf_num > 0, "Invalid que buf_num[%ld], tensor index[%d], node[%s].",
                       tensor->attr.que.buf_num, output_index, node->GetName().c_str());
        continue;
      }
      GE_LOGE("Invalid mem, alloc type[%d], que id[%ld], buf id[%ld], tensor index[%d], node[%s].",
              static_cast<int32_t>(tensor->attr.mem.alloc_type), tensor->attr.que.id, tensor->attr.buf.id, output_index,
              node->GetName().c_str());
      return false;
    }
  }
  return true;
}

bool AscGraph::CheckNodeConnectionValid() const {
  for (const auto &node : GetAllNodes()) {
    for (uint32_t index = 0U; index < node->inputs.Size(); index++) {
      GE_ASSERT_TRUE(node->GetInDataAnchor(index) != nullptr, "Input is not connected, index[%u], node[%s].", index,
                     node->GetName().c_str());
      GE_ASSERT_TRUE(node->GetInDataAnchor(index)->GetPeerOutAnchor() != nullptr,
                     "Input is not connected, index[%u], node[%s].", index, node->GetName().c_str());
    }
  }
  return true;
}

bool AscGraph::CheckValid() const {
  if (!CheckExprValid()) {
    return false;
  }
  if (!CheckAxisValid()) {
    return false;
  }
  if (!CheckTensorValid()) {
    return false;
  }
  if (!CheckNodeConnectionValid()) {
    return false;
  }
  return true;
}

TransInfoRoadOfGraph AscGraph::GetAllAxisTransInfo() const {
  return impl_->GetAllAxisTransInfo();
}

AscOpOutput AscGraph::CreateContiguousData(const char *name, const ge::DataType &dt, const std::vector<Axis> &axes,
                                           const size_t index, const ge::Format &format) {
  return impl_->CreateContiguousData(name, dt, axes, index, format);
}

AscOpOutput AscGraph::CreateContiguousOut(const char *name, const ge::DataType &dt, const std::vector<Axis> &axes,
                                          const ge::Format &format) {
  return impl_->CreateContiguousOut(name, dt, axes, format);
}

graphStatus AddEdgeForNode(const Operator &src_op, int32_t src_index, Operator &dst_op, int32_t dst_index) {
  auto src_node = NodeUtilsEx::GetNodeFromOperator(src_op);
  auto dst_node = NodeUtilsEx::GetNodeFromOperator(dst_op);
  GE_ASSERT_NOTNULL(src_node);
  if (dst_node == nullptr) {
    auto com_graph = src_node->GetOwnerComputeGraph();
    GE_ASSERT_NOTNULL(com_graph);
    auto dst_op_desc = OpDescUtils::GetOpDescFromOperator(dst_op);
    auto dst_asc_node = std::make_shared<AscNode>(dst_op_desc, com_graph);
    GE_ASSERT_NOTNULL(dst_asc_node);
    (void)dst_asc_node->Init();
    ConstNodePtr const_dst_node = dst_asc_node;
    GE_ASSERT_GRAPH_SUCCESS(NodeUtilsEx::SetNodeToOperator(dst_op, const_dst_node));
    dst_node = com_graph->AddNode(dst_asc_node);
    GE_ASSERT_NOTNULL(dst_node);
    GE_ASSERT_GRAPH_SUCCESS(
        GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_index), dst_node->GetInDataAnchor(dst_index)));
    // update tensors
    (void)dst_asc_node->inputs();
    (void)dst_asc_node->outputs();
  } else {
    GE_ASSERT_GRAPH_SUCCESS(
        GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_index), dst_node->GetInDataAnchor(dst_index)));
  }
  return GRAPH_SUCCESS;
}

int64_t AscOpOutput::GenContainerId() {
  GE_ASSERT_NOTNULL(op_);
  return ascir::cg::CodeGenUtils::GenNextContainerId(*op_);
}

int64_t AscOpOutput::GenNextReuseId() {
  GE_ASSERT_NOTNULL(op_);
  return ascir::cg::CodeGenUtils::GenNextReuseId(*op_);
}

bool AscOpOutput::UseTQue(const Position pos, const int64_t depth, const int64_t buf_num, const int64_t id) {
  GE_ASSERT_TRUE(!HasBindToContainer(), " this tensor has been bound to a que, can not use any other que.");
  GE_ASSERT_TRUE(buf_num > 0, "input buf_num should be greater than 0.");
  GE_ASSERT_TRUE(buf_num < static_cast<int64_t>(INT32_MAX), "input buf_num should be less than INT32_MAX.");
  GE_ASSERT_TRUE(depth > 0, "input depth should be greater than 0.");
  GE_ASSERT_TRUE(depth < static_cast<int64_t>(INT32_MAX), "input depth should be less than INT32_MAX.");
  mem->position = pos;
  mem->alloc_type = AllocType::kAllocTypeQueue;
  buf->id = kIdNone;
  que->depth = depth;
  que->buf_num = buf_num;
  if (id == kIdNone) {
    que->id = GenContainerId();
  } else {
    que->id = id;
  }
  return true;
}

bool AscOpOutput::UseTBuf(const Position pos, const int64_t id) {
  GE_ASSERT_TRUE(!HasBindToContainer(), " this tensor has been bound to a buf, can not use any other buf.");
  mem->position = pos;
  mem->alloc_type = AllocType::kAllocTypeBuffer;
  que->id = kIdNone;
  if (id == kIdNone) {
    buf->id = GenContainerId();
  } else {
    buf->id = id;
  }
  return true;
}

bool AscOpOutput::HasBindToContainer() const {
  bool has_bind_que = (que->id != kIdNone);
  bool has_bind_buf = (buf->id != kIdNone);
  // 1.if alloc type has set to que or buffer means has binding to a container
  // 2.if que/buf is valid, means also means has binding to a container
  return ((mem->alloc_type == AllocType::kAllocTypeQueue) || (mem->alloc_type == AllocType::kAllocTypeBuffer)) &&
         (has_bind_que || has_bind_buf);
}

// 既有动态输出，也有普通的输出，返回错误
static Status GetAndCheckDynamicOutput(const std::vector<std::pair<std::string, IrOutputType>> &ir_outputs,
                                       bool &only_has_one_dynamic_output) {
  bool has_dynamic_output = false;
  bool has_com_output = false;
  for (const auto &ir_output : ir_outputs) {
    if (ir_output.second == IrOutputType::kIrOutputDynamic) {
      has_dynamic_output = true;
    } else {
      has_com_output = true;
    }
  }
  only_has_one_dynamic_output = has_dynamic_output && (ir_outputs.size() == 1U);

  return (has_dynamic_output && has_com_output) ? af::FAILED : af::SUCCESS;
}

graphStatus LinkByIrIndex(const Operator &src_op, uint32_t src_ir_index, Operator &dst_op, uint32_t dst_ir_index,
                          uint32_t dynamic_index) {
  auto dst_op_desc = OpDescUtils::GetOpDescFromOperator(dst_op);
  auto src_op_desc = OpDescUtils::GetOpDescFromOperator(src_op);
  GE_ASSERT_NOTNULL(src_op_desc);
  GE_ASSERT_NOTNULL(dst_op_desc);
  const std::vector<std::pair<std::string, IrInputType>> &ir_inputs = dst_op_desc->GetIrInputs();
  const std::vector<std::pair<std::string, IrOutputType>> &ir_outputs = src_op_desc->GetIrOutputs();
  bool only_has_one_dynamic_output = false;
  GE_ASSERT_SUCCESS(GetAndCheckDynamicOutput(ir_outputs, only_has_one_dynamic_output),
                    "Not supporting both dynamic and non dynamic outputs");

  GE_ASSERT_TRUE(dst_ir_index < ir_inputs.size(), "dst_ir_index = %u, ir_inputs size = %zu", dst_ir_index,
                 ir_inputs.size());
  auto &name_to_input_idx = dst_op_desc->MutableAllInputName();
  auto &name_to_output_idx = src_op_desc->MutableAllOutputName();
  uint32_t src_index;
  uint32_t dst_index;
  if (ir_inputs[dst_ir_index].second == IrInputType::kIrInputDynamic) {
    std::map<size_t, std::pair<size_t, size_t>> ir_input_2_range;
    (void)OpDescUtils::GetIrInputInstanceDescRange(dst_op_desc, ir_input_2_range);
    dst_index = ir_input_2_range[dst_ir_index].first + dynamic_index;
  } else {
    dst_index = name_to_input_idx[ir_inputs[dst_ir_index].first];
  }

  if (only_has_one_dynamic_output) {
    src_index = src_ir_index;
  } else {
    src_index = name_to_output_idx[ir_outputs[src_ir_index].first];
  }

  dst_op.SetInput(dst_index, src_op, src_index);
  AddEdgeForNode(src_op, static_cast<int32_t>(src_index), dst_op, static_cast<int32_t>(dst_index));

  return GRAPH_SUCCESS;
}

graphStatus SetDynamicInputNumByIrIndex(Operator &op, uint32_t ir_index, uint32_t dynamic_num) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  const std::vector<std::pair<std::string, IrInputType>> &ir_inputs = op_desc->GetIrInputs();
  GE_ASSERT_TRUE(ir_index < ir_inputs.size());
  GE_ASSERT_TRUE(ir_inputs[ir_index].second == IrInputType::kIrInputDynamic);
  std::map<size_t, std::pair<size_t, size_t>> ir_input_2_range;
  (void)OpDescUtils::GetIrInputInstanceDescRange(op_desc, ir_input_2_range);

  GE_ASSERT_TRUE(ir_input_2_range[ir_index].second < dynamic_num, "Dynamic index [%u] is invalid.", dynamic_num);
  op_desc->AddDynamicInputDescByIndex(ir_inputs[ir_index].first, dynamic_num, ir_input_2_range[ir_index].first);
  GELOGD("Add DynamicInputDescByIndex for op_desc[%s], ir_index[%u], dynamic_num[%u]", op_desc->GetNamePtr(), ir_index,
         dynamic_num);
  return GRAPH_SUCCESS;
}

graphStatus AscGraphAttr::SerializeAttr(ascendc_ir::proto::AscGraphAttrGroupsDef &asc_graph_group) {
  asc_graph_group.set_tiling_key(tiling_key);
  // axis serialize
  auto axis_defs = asc_graph_group.axis();
  for (const auto &ax : axis) {
    auto ax_def = asc_graph_group.add_axis();
    ax_def->set_id(ax->id);
    ax_def->set_name(ax->name);
    ax_def->set_axis_type(ax->type);
    ax_def->set_bind_block(ax->bind_block);
    ax_def->set_size(SymbolicUtils::ToString(ax->size));
    ax_def->set_align(SymbolicUtils::ToString(ax->align));
    for (const auto fm : ax->from) {
      ax_def->add_from(fm);
    }
    ax_def->set_split_pair_other_id(ax->split_pair_other_id);
    ax_def->set_allow_oversize_axis(ax->allow_oversize_axis);
    ax_def->set_allow_unaligned_tail(ax->allow_unaligned_tail);
  }
  for (const auto &var : size_vars) {
    asc_graph_group.add_size_var(SymbolicUtils::ToString(var->expr));
  }
  if (!sched.axis.empty() || sched.loop_axis != kIdNone) {
    auto sched_def = asc_graph_group.mutable_sched();
    for (const auto axis_id : sched.axis) {
      sched_def->add_axis(axis_id);
    }
    sched_def->set_loop_axis(sched.loop_axis);
  }
  asc_graph_group.set_type(static_cast<int64_t>(type));
  GELOGD("Graph serialization successful, tiling_key[%ld] type[%ld]", tiling_key, static_cast<int64_t>(type));
  return GRAPH_SUCCESS;
}
graphStatus AscGraphAttr::Serialize(proto::AttrGroupDef &attr_group_def) {
  auto asc_graph_attr_group = attr_group_def.mutable_asc_graph_attr_group();
  GE_ASSERT_NOTNULL(asc_graph_attr_group);
  return SerializeAttr(*asc_graph_attr_group);
}

graphStatus AscGraphAttr::DeserializeAttr(const ascendc_ir::proto::AscGraphAttrGroupsDef &asc_graph_group) {
  tiling_key = asc_graph_group.tiling_key();
  type = static_cast<AscGraphType>(asc_graph_group.type());
  for (const auto &ax : asc_graph_group.axis()) {
    auto new_axis = std::make_shared<Axis>();
    GE_ASSERT_NOTNULL(new_axis);
    new_axis->id = ax.id();
    new_axis->name = ax.name();
    new_axis->type = static_cast<Axis::Type>(ax.axis_type());
    new_axis->bind_block = ax.bind_block();
    new_axis->size = Expression::Deserialize(ax.size().c_str());
    new_axis->align = Expression::Deserialize(ax.align().c_str());
    for (const auto &fm : ax.from()) {
      new_axis->from.emplace_back(fm);
    }
    new_axis->split_pair_other_id = ax.split_pair_other_id();
    new_axis->allow_oversize_axis = ax.allow_oversize_axis();
    new_axis->allow_unaligned_tail = ax.allow_unaligned_tail();
    axis.emplace_back(new_axis);
  }
  for (const auto &var : asc_graph_group.size_var()) {
    auto new_size_var = std::make_shared<SizeVar>(Expression::Deserialize(var.c_str()));
    size_vars.emplace_back(new_size_var);
  }
  sched.axis.clear();
  sched.loop_axis = kIdNone;
  if (asc_graph_group.has_sched()) {
    for (const auto axis_id : asc_graph_group.sched().axis()) {
      sched.axis.emplace_back(axis_id);
    }
    sched.loop_axis = asc_graph_group.sched().loop_axis();
  }
  type = static_cast<AscGraphType>(asc_graph_group.type());
  GELOGD("Graph deserialization successful, tiling_key[%ld], type[%ld]", tiling_key, asc_graph_group.type());
  return GRAPH_SUCCESS;
}

graphStatus AscGraphAttr::Deserialize(const proto::AttrGroupDef &attr_group_def, AttrHolder *attr_holder) {
  (void)attr_holder;
  const auto &asc_graph_attr_group_def = attr_group_def.asc_graph_attr_group();
  return DeserializeAttr(asc_graph_attr_group_def);
}

graphStatus AscNodeAttr::SerializeAttr(ascendc_ir::proto::AscNodeAttrGroupsDef &asc_node_group) const {
  asc_node_group.set_name(name);
  asc_node_group.set_type(type);
  auto sched_def = asc_node_group.mutable_sched();
  for (const int64_t axis_id : sched.axis) {
    sched_def->add_axis(axis_id);
  }
  sched_def->set_loop_axis(sched.loop_axis);
  sched_def->set_exec_condition(static_cast<int32_t>(sched.exec_condition));
  auto api_def = asc_node_group.mutable_api();
  api_def->set_type(static_cast<int32_t>(api.type));
  api_def->set_compute_type(static_cast<int32_t>(api.compute_type));
  api_def->set_unit(static_cast<int32_t>(api.unit));
  if (ir_attr != nullptr) {
    ir_attr->Serialize(*(asc_node_group.mutable_ir_attr_def()));
  }
  for (const auto &tmp_buffer : tmp_buffers) {
    auto tmp_buffer_def = asc_node_group.add_tmp_buffers();
    tmp_buffer_def->set_id(tmp_buffer.id);
    auto buf_desc_def = tmp_buffer_def->mutable_buf_desc();
    buf_desc_def->set_size(SymbolicUtils::ToString(tmp_buffer.buf_desc.size));
    buf_desc_def->set_life_time_axis_id(tmp_buffer.buf_desc.life_time_axis_id);
    auto mem_def = tmp_buffer_def->mutable_mem();
    mem_def->set_tensor_id(tmp_buffer.mem.tensor_id);
    mem_def->set_alloc_type(static_cast<int64_t>(tmp_buffer.mem.alloc_type));
    mem_def->set_position(static_cast<int64_t>(tmp_buffer.mem.position));
    mem_def->set_hardware(static_cast<int64_t>(tmp_buffer.mem.hardware));
    mem_def->set_reuse_id(static_cast<int64_t>(tmp_buffer.mem.reuse_id));
    for (const int64_t buf_id : tmp_buffer.mem.buf_ids) {
      mem_def->add_buf_ids(buf_id);
    }
    mem_def->set_name(tmp_buffer.mem.name);
  }
  GELOGD("Serialize node[%s:%s] success.", name.c_str(), type.c_str());
  return GRAPH_SUCCESS;
}

graphStatus AscNodeAttr::DeserializeAttr(const ascendc_ir::proto::AscNodeAttrGroupsDef &asc_node_group) {
  name = asc_node_group.name();
  type = asc_node_group.type();
  const auto &sched_def = asc_node_group.sched();
  for (const auto &ax : sched_def.axis()) {
    sched.axis.emplace_back(ax);
  }
  sched.loop_axis = sched_def.loop_axis();
  sched.exec_condition = static_cast<ExecuteCondition>(sched_def.exec_condition());
  const auto &api_def = asc_node_group.api();
  api.type = static_cast<ApiType>((api_def.type()));
  api.compute_type = static_cast<ComputeType>(api_def.compute_type());
  api.unit = static_cast<ComputeUnit>(api_def.unit());
  if (asc_node_group.has_ir_attr_def()) {
    if (ir_attr == nullptr) {
      ir_attr = ComGraphMakeUnique<AscIrAttrDefBase>();
    }
    GE_ASSERT_NOTNULL(ir_attr);
    ir_attr->Deserialize(asc_node_group.ir_attr_def());
  }
  for (const auto &tmp_buffer_def : asc_node_group.tmp_buffers()) {
    TmpBufDesc new_tmp_buffer_desc;
    new_tmp_buffer_desc.size = Expression::Deserialize(tmp_buffer_def.buf_desc().size().c_str());
    new_tmp_buffer_desc.life_time_axis_id = tmp_buffer_def.buf_desc().life_time_axis_id();
    MemAttr new_mem_attr;
    new_mem_attr.name = tmp_buffer_def.mem().name();
    new_mem_attr.tensor_id = tmp_buffer_def.mem().tensor_id();
    new_mem_attr.alloc_type = static_cast<AllocType>(tmp_buffer_def.mem().alloc_type());
    new_mem_attr.position = static_cast<Position>(tmp_buffer_def.mem().position());
    new_mem_attr.hardware = static_cast<MemHardware>(tmp_buffer_def.mem().hardware());
    new_mem_attr.reuse_id = tmp_buffer_def.mem().reuse_id();
    for (const int64_t buf_id : tmp_buffer_def.mem().buf_ids()) {
      new_mem_attr.buf_ids.emplace_back(buf_id);
    }
    TmpBuffer new_tmp_buffer;
    new_tmp_buffer.buf_desc = new_tmp_buffer_desc;
    new_tmp_buffer.mem = new_mem_attr;
    new_tmp_buffer.id = tmp_buffer_def.id();
    tmp_buffers.emplace_back(new_tmp_buffer);
  }
  return GRAPH_SUCCESS;
}

graphStatus AscNodeAttr::Serialize(proto::AttrGroupDef &attr_group_def) {
  auto asc_node_attr = attr_group_def.mutable_asc_node_attr_group();
  GE_ASSERT_NOTNULL(asc_node_attr);
  return SerializeAttr(*asc_node_attr);
}

graphStatus AscNodeAttr::Deserialize(const proto::AttrGroupDef &attr_group_def, AttrHolder *attr_holder) {
  (void)attr_holder;
  const auto &asc_node_attr_def = attr_group_def.asc_node_attr_group();
  return DeserializeAttr(asc_node_attr_def);
}

graphStatus AscTensorAttr::SerializeAttr(ascendc_ir::proto::AscTensorAttrGroupsDef &asc_tensor_group) {
  if (dtype.tensor_desc_ != nullptr) {
    asc_tensor_group.set_dtype(static_cast<int64_t>(dtype));
  }
  for (const int64_t axis_id : axis) {
    asc_tensor_group.add_axis_ids(axis_id);
  }
  for (const auto &repeat : repeats) {
    asc_tensor_group.add_repeats(SymbolicUtils::ToString(repeat));
  }
  for (const auto &stride : strides) {
    asc_tensor_group.add_strides(SymbolicUtils::ToString(stride));
  }
  for (const auto &vectorized_axis_id : vectorized_axis) {
    asc_tensor_group.add_vectorized_axis(vectorized_axis_id);
  }
  for (const auto &vectorized_stride : vectorized_strides) {
    asc_tensor_group.add_vectorized_strides(SymbolicUtils::ToString(vectorized_stride));
  }
  auto mem_def = asc_tensor_group.mutable_mem();
  mem_def->set_tensor_id(mem.tensor_id);
  mem_def->set_alloc_type(static_cast<int64_t>(mem.alloc_type));
  mem_def->set_position(static_cast<int64_t>(mem.position));
  mem_def->set_hardware(static_cast<int64_t>(mem.hardware));
  for (const int64_t buf_id : mem.buf_ids) {
    mem_def->add_buf_ids(buf_id);
  }
  mem_def->set_name(mem.name);
  auto que_def = asc_tensor_group.mutable_que();
  que_def->set_id(que.id);
  que_def->set_depth(que.depth);
  que_def->set_buf_num(que.buf_num);
  que_def->set_name(que.name);
  auto buf_def = asc_tensor_group.mutable_buf();
  buf_def->set_id(buf.id);
  buf_def->set_name(buf.name);
  auto opt_def = asc_tensor_group.mutable_opt();
  opt_def->set_reuse_id(opt.reuse_id);
  opt_def->set_ref_tensor(opt.ref_tensor);
  opt_def->set_merge_scope(opt.merge_scope);
  return GRAPH_SUCCESS;
}

graphStatus AscTensorAttr::DeserializeAttr(const ascendc_ir::proto::AscTensorAttrGroupsDef &asc_tensor_group,
                                           GeTensorDesc *tensor_desc) {
  if ((tensor_desc != nullptr) && (dtype.tensor_desc_ == nullptr)) {
    dtype.tensor_desc_ = tensor_desc;
  }
  dtype.tensor_desc_->SetDataType(static_cast<ge::DataType>(asc_tensor_group.dtype()));
  for (const auto &axis_id : asc_tensor_group.axis_ids()) {
    axis.emplace_back(axis_id);
  }
  const auto &repeat_defs = asc_tensor_group.repeats();
  for (const auto &repeat : repeat_defs) {
    repeats.emplace_back(Expression::Deserialize(repeat.c_str()));
  }
  const auto &strides_defs = asc_tensor_group.strides();
  for (const auto &stride : strides_defs) {
    strides.emplace_back(Expression::Deserialize(stride.c_str()));
  }
  const auto &vectorized_axis_ids = asc_tensor_group.vectorized_axis();
  for (const auto &vectorized_axis_id : vectorized_axis_ids) {
    vectorized_axis.emplace_back(vectorized_axis_id);
  }
  const auto &vectorized_strides_def = asc_tensor_group.vectorized_strides();
  for (const auto &vectorized_stride : vectorized_strides_def) {
    vectorized_strides.emplace_back(Expression::Deserialize(vectorized_stride.c_str()));
  }
  const auto &mem_def = asc_tensor_group.mem();
  mem.name = mem_def.name();
  mem.tensor_id = mem_def.tensor_id();
  mem.alloc_type = static_cast<AllocType>(mem_def.alloc_type());
  mem.position = static_cast<Position>(mem_def.position());
  mem.hardware = static_cast<MemHardware>(mem_def.hardware());
  for (const int64_t buf_id : mem_def.buf_ids()) {
    mem.buf_ids.emplace_back(buf_id);
  }
  mem.name = mem_def.name();
  const auto &que_def = asc_tensor_group.que();
  que.id = que_def.id();
  que.name = que_def.name();
  que.depth = que_def.depth();
  que.buf_num = que_def.buf_num();
  const auto &buf_def = asc_tensor_group.buf();
  buf.id = buf_def.id();
  buf.name = buf_def.name();
  const auto &opt_def = asc_tensor_group.opt();
  opt.merge_scope = opt_def.merge_scope();
  opt.ref_tensor = opt_def.ref_tensor();
  opt.reuse_id = opt_def.reuse_id();
  return GRAPH_SUCCESS;
}

graphStatus AscTensorAttr::Serialize(proto::AttrGroupDef &attr_group_def) {
  auto asc_tensor_attr_group = attr_group_def.mutable_asc_tensor_attr_group();
  GE_ASSERT_NOTNULL(asc_tensor_attr_group);
  return SerializeAttr(*asc_tensor_attr_group);
}

graphStatus AscTensorAttr::Deserialize(const proto::AttrGroupDef &attr_group_def, AttrHolder *attr_holder) {
  const auto &asc_tensor_attr_group_def = attr_group_def.asc_tensor_attr_group();
  return DeserializeAttr(asc_tensor_attr_group_def, dynamic_cast<GeTensorDesc *>(attr_holder));
}

graphStatus AscIrAttrDefBase::Serialize(ascendc_ir::proto::AscIrAttrDef &asc_ir_attr_def) {
  std::map<std::string, AnyValue> names_to_attr;
  attr_store_.GetAllAttrs(names_to_attr);
  auto &attr_map = *asc_ir_attr_def.mutable_attr();
  for (const auto &pair : names_to_attr) {
    const auto serializer = AttrSerializerRegistry::GetInstance().GetSerializer(pair.second.GetValueTypeId());
    GE_ASSERT_NOTNULL(serializer);
    proto::AttrDef attr_def;
    GE_ASSERT_GRAPH_SUCCESS(serializer->Serialize(pair.second, attr_def));
    attr_map[pair.first] = attr_def;
  }
  return GRAPH_SUCCESS;
}

graphStatus AscIrAttrDefBase::Deserialize(const ascendc_ir::proto::AscIrAttrDef &asc_ir_attr_def) {
  const auto &attr_map = asc_ir_attr_def.attr();
  for (const auto &pair : attr_map) {
    const auto deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(pair.second.value_case());
    GE_ASSERT_NOTNULL(deserializer);
    auto attr_value = attr_store_.GetOrCreateAnyValue(pair.first);
    GE_ASSERT_NOTNULL(attr_value);
    GE_ASSERT_GRAPH_SUCCESS(deserializer->Deserialize(pair.second, *attr_value));
  }
  return GRAPH_SUCCESS;
}

std::unique_ptr<AscIrAttrDefBase> AscIrAttrDefBase::Clone() {
  auto ptr = ComGraphMakeUnique<AscIrAttrDefBase>();
  GE_ASSERT_NOTNULL(ptr);
  ptr->attr_store_ = this->attr_store_;
  return ptr;
}

graphStatus AscDataIrAttrDef::GetIndex(int64_t &index) const {
  auto value = attr_store_.GetAnyValue(kDataIndex);
  GE_WARN_ASSERT(value != nullptr);
  return value->GetValue(index);
}

graphStatus AscDataIrAttrDef::SetIndex(int64_t index) {
  auto value = attr_store_.GetOrCreateAnyValue(kDataIndex);
  GE_ASSERT_NOTNULL(value);
  return value->SetValue(index);
}
REG_ATTR_GROUP_SERIALIZER(AscNodeAttr, AscNodeAttr, GetTypeId<AscNodeAttr>(), proto::AttrGroupDef::kAscNodeAttrGroup);
REG_ATTR_GROUP_SERIALIZER(AscGraphAttr, AscGraphAttr, GetTypeId<AscGraphAttr>(),
                          proto::AttrGroupDef::kAscGraphAttrGroup);
REG_ATTR_GROUP_SERIALIZER(AscTensorAttr, AscTensorAttr, GetTypeId<AscTensorAttr>(),
                          proto::AttrGroupDef::kAscTensorAttrGroup);
REG_ATTR_GROUP_SERIALIZER(ShapeEnvAttr, ShapeEnvAttr, GetTypeId<ShapeEnvAttr>(),
                          proto::AttrGroupDef::kShapeEnvAttrGroup);
REG_ATTR_GROUP_SERIALIZER(SymbolicDescAttr, SymbolicDescAttr, GetTypeId<SymbolicDescAttr>(),
                          proto::AttrGroupDef::kTensorAttrGroup);
}  // namespace af
