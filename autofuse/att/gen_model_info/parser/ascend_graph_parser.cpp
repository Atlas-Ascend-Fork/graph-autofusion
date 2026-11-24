/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascend_graph_parser.h"
#include <fstream>
#include <algorithm>
#include <stack>
#include <queue>
#include <set>
#include <climits>
#include "common/checker.h"
#include "common/util/mem_utils.h"
#include "base/att_const_values.h"
#include "util/thread_local_context.h"
#include "att_utils.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "ascir_node_param/ascir_node_param.h"
#include "base_types_printer.h"
#include "common_utils.h"
#include "indirect_load_utils.h"
#include "schedule_result.h"
#include "specific_params_builder.h"
#include "vector_function_graph_parser.h"

namespace att {
namespace {
bool HasComputeType(const ascir::ImplGraph &impl_graph, const af::ComputeType compute_type) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (node->attr.api.compute_type == compute_type) {
      return true;
    }
  }
  return false;
}

bool ShouldPreserveCacheCondition(const af::AscNodePtr &node, const bool graph_has_reduce) {
  if (!graph_has_reduce) {
    return true;
  }
  if (node == nullptr) {
    return false;
  }
  // Codegen evaluates cache eligibility per node in a Reduce graph.  Keep the
  // condition for data movement/broadcast nodes so ATT models the same Nddma
  // (Load converted to Nddma) execution count; reduction nodes retain the
  // historical conservative behavior.
  const auto &type = node->GetType();
  return type == kLoad || type == "Nddma" || type == kBroadcast;
}

// 通过输出获取当前节点单次执行涉及到的轴的范围
std::vector<int64_t> GetNodeVectorizedAxis(const af::AscNodePtr &ge_node, int64_t loop_axis_id) {
  std::vector<int64_t> total_vectorized_axis;
  for (size_t out_id = 0U; out_id < ge_node->outputs().size(); out_id++) {
    auto &output_tensor = ge_node->outputs[out_id].attr;
    if (output_tensor.vectorized_axis.empty()) {
      continue;
    }
    bool inside_loop_flag{false};
    std::vector<int64_t> vectorized_axis_ids;
    for (auto &axis_id : output_tensor.vectorized_axis) {
      if (inside_loop_flag) {
        vectorized_axis_ids.emplace_back(axis_id);
      }
      if (axis_id == loop_axis_id) {
        inside_loop_flag = true;
      }
    }
    // 在vectorized_axis中找不到loop_axis，默认vecotrized_axis中所有轴都是单次执行涉及到的轴
    if (!inside_loop_flag) {
      vectorized_axis_ids = output_tensor.vectorized_axis;
    }
    // 取所有输出节点vectorized_axis并集
    for (auto &axis_id : vectorized_axis_ids) {
      if (std::find(total_vectorized_axis.begin(), total_vectorized_axis.end(), axis_id) ==
          total_vectorized_axis.end()) {
        total_vectorized_axis.emplace_back(axis_id);
      }
    }
  }
  return total_vectorized_axis;
}

void UpdateConcatVectorizedAxes(const af::AscNodePtr &ge_node, int64_t loop_axis,
                                std::map<size_t, SubAxisPtr> &sub_axes_info) {
  if (ge_node->GetType() != "Concat" || ge_node->outputs().empty()) {
    return;
  }
  const auto dtype = ge_node->outputs()[0]->attr.dtype;
  const auto data_type_size = ge::GetSizeByDataType(dtype);
  for (int64_t axis_id : GetNodeVectorizedAxis(ge_node, loop_axis)) {
    const auto sub_axis_iter = sub_axes_info.find(axis_id);
    if (sub_axis_iter == sub_axes_info.end()) {
      continue;
    }
    sub_axis_iter->second->data_type_size = data_type_size;
    sub_axis_iter->second->is_concat_vec_axis = true;
  }
}
}  // namespace
const std::string kInputNamePrefix = "_input_";
const std::string kOutputNamePrefix = "_output_";
const Expr kUBAlign = CreateExpr(32);

AxisPosition ConvertAxisType(const af::Axis::Type &type) {
  static const std::map<af::Axis::Type, AxisPosition> kAxisTypeMap = {
      {af::Axis::kAxisTypeOriginal, AxisPosition::ORIGIN},  {af::Axis::kAxisTypeBlockOuter, AxisPosition::OUTER},
      {af::Axis::kAxisTypeBlockInner, AxisPosition::INNER}, {af::Axis::kAxisTypeTileOuter, AxisPosition::OUTER},
      {af::Axis::kAxisTypeTileInner, AxisPosition::INNER},  {af::Axis::kAxisTypeMerged, AxisPosition::MERGED},
  };
  if (kAxisTypeMap.find(type) == kAxisTypeMap.end()) {
    GELOGE(af::FAILED, "Convert ascir axis type[%d] failed.", type);
    return AxisPosition::POSERR;
  }
  return kAxisTypeMap.at(type);
}

af::Status GetPhyType(const af::Position pos, af::MemHardware &phy_type) {
  static const std::map<af::Position, af::MemHardware> kDefPosToPhy = {
      {af::Position::kPositionGM, af::MemHardware::kMemHardwareGM},
      {af::Position::kPositionVecIn, af::MemHardware::kMemHardwareUB},
      {af::Position::kPositionVecOut, af::MemHardware::kMemHardwareUB},
      {af::Position::kPositionVecCalc, af::MemHardware::kMemHardwareUB},
  };
  const auto iter = kDefPosToPhy.find(pos);
  GE_ASSERT_TRUE(iter != kDefPosToPhy.cend(), "[Get][PhyType] failed, pos=%d", static_cast<int32_t>(pos));
  phy_type = iter->second;
  return af::SUCCESS;
}

af::Status SetAscGraphPhyType(const af::AscGraph &graph) {
  for (auto node : graph.GetAllNodes()) {
    for (size_t i = 0; i < node->outputs().size(); i++) {
      GE_ASSERT_SUCCESS(GetPhyType(node->outputs[i].attr.mem.position, node->outputs[i].attr.mem.hardware));
    }
  }
  return af::SUCCESS;
}

// 获取所有轴对应的原始轴id
af::Status AscendGraphParser::ParserOriginAxis(const af::AscGraph &graph) {
  std::queue<int64_t> axis_ids;
  int64_t cur_axis_id;

  for (auto &axis_info : graph.GetAllAxis()) {
    // 稀疏场景记录原始值和向量轴对应关系初始化
    GE_ASSERT_NOTNULL(axis_info, "Get sub axes failed.");
    if (axis_info->type == af::Axis::kAxisTypeOriginal) {
      orig_to_first_vec_id_.emplace(axis_info->id, axis_info->id);
    }

    axis_ids.push(axis_info->id);
    while (axis_ids.size() > 0U) {
      cur_axis_id = axis_ids.front();
      axis_ids.pop();
      if (orig_axes_info_.find(cur_axis_id) != orig_axes_info_.end()) {  // 已经搜索过的直接拿过来用
        orig_axes_info_[axis_info->id].insert(orig_axes_info_[axis_info->id].end(),
                                              orig_axes_info_[cur_axis_id].begin(), orig_axes_info_[cur_axis_id].end());
        continue;
      }
      GE_ASSERT_SUCCESS(CheckAxisIdValid(cur_axis_id), "Invalid axis id [%ld]", cur_axis_id);
      if (axes_info_[cur_axis_id]->type == af::Axis::kAxisTypeOriginal) {
        orig_axes_info_[axis_info->id].emplace_back(cur_axis_id);
      } else {
        for (uint32_t i = 0U; i < axes_info_[cur_axis_id]->from.size(); i++) {
          axis_ids.push(axes_info_[cur_axis_id]->from[i]);
        }
      }
    }
  }
  return af::SUCCESS;
}

af::Status AscendGraphParser::CheckAxisIdValid(const int64_t axis_id) {
  GE_ASSERT_TRUE(axes_info_.find(axis_id) != axes_info_.end(), "Invalid axis id [%ld].", axis_id);
  return af::SUCCESS;
}

af::Status AscendGraphParser::CheckAxisIdValid(std::vector<int64_t> &axis_ids) {
  for (auto &axis_id : axis_ids) {
    GE_ASSERT_TRUE(axes_info_.find(axis_id) != axes_info_.end(), "Invalid axis id [%ld].", axis_id);
  }
  return af::SUCCESS;
}

void AscendGraphParser::SaveTmpBufferInfos(const std::string &node_name, std::map<int64_t, Expr> &max_tmp_buffers_map,
                                           std::vector<af::TmpBuffer> &tmp_buffers) const {
  std::map<int64_t, Expr> node_tmp_buffers_map;
  for (const auto &buffer : tmp_buffers) {
    GELOGD("Save tmp buffer [%ld, %s] for node %s.", buffer.id, buffer.buf_desc.size.Str().get(), node_name.c_str());
    const auto &iter = node_tmp_buffers_map.find(buffer.id);
    if (iter == node_tmp_buffers_map.cend()) {
      node_tmp_buffers_map[buffer.id] = buffer.buf_desc.size;
    } else {
      iter->second = iter->second + buffer.buf_desc.size;
    }
  }
  for (const auto &pair : node_tmp_buffers_map) {
    const Expr &buffer = pair.second;
    const auto &iter = max_tmp_buffers_map.find(pair.first);
    if (iter == max_tmp_buffers_map.cend()) {
      max_tmp_buffers_map[pair.first] = buffer;
    } else {
      max_tmp_buffers_map[pair.first] = af::sym::Max(max_tmp_buffers_map[pair.first], buffer);
    }
  }
  for (const auto &tmp_buffer_map : max_tmp_buffers_map) {
    GELOGD("Save tmp buffer [%ld, %s] for node %s.", tmp_buffer_map.first, tmp_buffer_map.second.Str().get(),
           node_name.c_str());
  }
}

af::Status AscendGraphParser::ParserSchedInfo(const af::AscGraph &graph) {
  for (auto &ax : graph.GetAllAxis()) {
    GE_ASSERT_NOTNULL(ax, "Get sub axes failed.");
    axes_info_[ax->id] = ax;  // 记录ascend ir轴信息便于后续搜索
  }

  uint32_t topo_idx = 0U;
  // 获取node调度信息
  std::map<int64_t, Expr> max_tmp_buffers_map;
  for (const auto &node : graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node, "Get graph node failed.");
    std::vector<af::AxisPtr> axes;   // 记录调度轴
    std::vector<int64_t> block_dim;  // block outer的轴id
    bool loop_inside_flag = false;
    for (auto &axis_id : node->attr.sched.axis) {
      GE_ASSERT_SUCCESS(CheckAxisIdValid(axis_id), "Invalid axis id [%ld]", axis_id);
      if (axes_info_[axis_id]->type == af::Axis::kAxisTypeBlockOuter) {
        block_dim.emplace_back(axis_id);
        loop_inside_flag = (axis_id == node->attr.sched.loop_axis) || loop_inside_flag;
        continue;
      }
      if (!loop_inside_flag) {
        axes.emplace_back(axes_info_[axis_id]);
      }
      // 遍历到loop_axis停止
      if (axis_id == node->attr.sched.loop_axis) {
        loop_inside_flag = true;
      }
    }
    topo_order_node_[topo_idx] = node;
    ScheduleAttr sched_info = {axes, block_dim, topo_idx, node->attr.sched.loop_axis, node->attr.sched.exec_condition};
    graph_sched_info_.emplace(node, std::move(sched_info));
    topo_idx++;
    SaveTmpBufferInfos(node->GetName(), max_tmp_buffers_map, node->attr.tmp_buffers);
  }
  tuning_space_->tmp_buffer = max_tmp_buffers_map;
  tuning_space_->builtin_tmp_buffer = ascgen_utils::CalcExtraTmpBufForAscGraph(graph);
  return af::SUCCESS;
}

af::Status AscendGraphParser::AddSubAxisInfo(af::AxisPtr &axis_info) {
  std::vector<std::string> orig_name;
  SubAxisPtr sub_axis_ptr = std::make_unique<SubAxis>();
  GE_ASSERT_NOTNULL(sub_axis_ptr, "Create sub axes failed.");
  ParserSubAxis(axis_info, sub_axis_ptr);
  auto iter = orig_axes_info_.find(axis_info->id);
  if (iter != orig_axes_info_.end()) {
    for (const auto &axis_id : iter->second) {
      auto iter2 = axes_info_.find(axis_id);
      if (iter2 != axes_info_.end()) {
        orig_name.emplace_back(iter2->second->name);
      }
    }
  }
  sub_axis_ptr->orig_axis_name = orig_name;
  sub_axes_info_[axis_info->id] = std::move(sub_axis_ptr);
  return af::SUCCESS;
}

af::Status AscendGraphParser::CreateSubAxisInfo(const af::AscGraph &graph) {
  for (auto &axis_info : graph.GetAllAxis()) {
    GE_ASSERT_NOTNULL(axis_info, "Get sub axes failed.");
    if (sub_axes_info_.find(axis_info->id) == sub_axes_info_.end()) {
      GE_ASSERT_SUCCESS(AddSubAxisInfo(axis_info));
    }
  }
  for (auto &ax : graph.GetAllAxis()) {
    GE_ASSERT_NOTNULL(ax, "Get sub axes failed.");
    for (size_t i = 0U; i < ax->from.size(); i++) {
      parent_axes_info_[ax->id].push_back(ax->from[i]);
    }
  }
  //  BindMulticore属性传递到原始轴
  for (auto &axis_info : graph.GetAllAxis()) {
    GE_ASSERT_NOTNULL(axis_info, "Get sub axes failed.");
    if (sub_axes_info_[axis_info->id]->is_bind_multi_core) {
      for (auto &origin_axis : orig_axes_info_[axis_info->id]) {
        sub_axes_info_[origin_axis]->is_bind_multi_core = true;
      }
    }
  }
  MakeSubAxisRelation();
  return af::SUCCESS;
}

void AscendGraphParser::ParserSubAxis(const af::AxisPtr &axis, SubAxisPtr &sub_axis_ptr) const {
  auto name = axis->name;
  // 轴没有设置名字用轴id表示，轴id唯一
  if (name.empty()) {
    name += std::to_string(axis->id);
  }
  sub_axis_ptr->name = name;
  sub_axis_ptr->id = axis->id;

  if ((axis->type == af::Axis::kAxisTypeBlockInner) || (axis->type == af::Axis::kAxisTypeBlockOuter)) {
    sub_axis_ptr->is_bind_multi_core = true;
  } else {
    sub_axis_ptr->is_bind_multi_core = false;
  }
  sub_axis_ptr->enable_pad = axis->allow_oversize_axis;
  sub_axis_ptr->enable_tail = axis->allow_unaligned_tail;
  sub_axis_ptr->axis_type = ConvertAxisType(axis->type);
  sub_axis_ptr->is_split = (sub_axis_ptr->axis_type == AxisPosition::OUTER);
  sub_axis_ptr->align = axis->align;
  sub_axis_ptr->repeat = axis->size;
}

void AscendGraphParser::MakeSubAxisRelation(void) {
  for (auto &sub : sub_axes_info_) {
    if (orig_axes_info_.find(sub.first) != orig_axes_info_.end()) {
      for (auto &axis_id : orig_axes_info_[sub.first]) {
        if (sub_axes_info_.find(axis_id) != sub_axes_info_.end()) {
          sub.second->orig_axis.emplace_back(sub_axes_info_[axis_id].get());
        }
      }
      for (auto &axis_id : parent_axes_info_[sub.first]) {
        if (sub_axes_info_.find(axis_id) != sub_axes_info_.end()) {
          sub.second->parent_axis.emplace_back(sub_axes_info_[axis_id].get());
        }
      }
    }
  }
}

af::Status AscendGraphParser::ConstructQueueContainer(const af::AscTensorAttr &ascir_tensor_info) {
  if (queue_containers_.find(ascir_tensor_info.que.id) == queue_containers_.end()) {
    const std::string user_set_name = ascir_tensor_info.que.name;
    std::string name = user_set_name.empty() ? "q" + std::to_string(ascir_tensor_info.que.id) + "_size" : user_set_name;
    auto queue_ptr = af::MakeShared<Queue>(name);
    GE_ASSERT_NOTNULL(queue_ptr, "Create queue failed.");
    queue_ptr->reuse_id = ascir_tensor_info.que.id;
    queue_ptr->buffer_num = ascir_tensor_info.que.buf_num;

    if (ascir_tensor_info.mem.hardware == af::MemHardware::kMemHardwareGM) {
      queue_ptr->buf_location.emplace_back(HardwareDef::GM);
    } else {
      queue_ptr->buf_location.emplace_back(HardwareDef::UB);
      queue_ptr->align = kUBAlign;
    }
    queue_containers_[ascir_tensor_info.que.id] = queue_ptr;
  }
  return af::SUCCESS;
}

af::Status AscendGraphParser::ConstructBufferContainer(const af::AscTensorAttr &ascir_tensor_info) {
  if (buf_containers_.find(ascir_tensor_info.buf.id) == buf_containers_.end()) {
    const std::string user_set_name = ascir_tensor_info.buf.name;
    std::string name = user_set_name.empty() ? "b" + std::to_string(ascir_tensor_info.buf.id) + "_size" : user_set_name;
    auto buf_ptr = af::MakeShared<Buf>(name);
    GE_ASSERT_NOTNULL(buf_ptr, "Create buffer failed.");
    buf_ptr->reuse_id = ascir_tensor_info.buf.id;
    if (ascir_tensor_info.mem.hardware == af::MemHardware::kMemHardwareGM) {
      buf_ptr->buf_location.emplace_back(HardwareDef::GM);
    } else {
      buf_ptr->buf_location.emplace_back(HardwareDef::UB);
      buf_ptr->align = kUBAlign;
    }
    buf_containers_[ascir_tensor_info.buf.id] = buf_ptr;
  }
  return af::SUCCESS;
}

af::Status AscendGraphParser::ConstructGlobalContainer(const af::AscTensorAttr &ascir_tensor_info) {
  GE_ASSERT_TRUE(ascir_tensor_info.mem.hardware == af::MemHardware::kMemHardwareGM,
                 "Hardware for workspace should be MEM_HARDWARE_GM.");
  auto location = HardwareDef::GM;
  if (global_containers_.find(location) == global_containers_.end()) {
    std::string name = "GlobalContainer-GM";
    auto container_ptr = af::MakeShared<GlobalCache>(name);
    GE_ASSERT_NOTNULL(container_ptr, "Create global cache failed.");
    container_ptr->reuse_id = ascir_tensor_info.mem.reuse_id;
    container_ptr->buf_location.emplace_back(location);
    global_containers_[location] = container_ptr;
  }
  return af::SUCCESS;
}

af::Status AscendGraphParser::ParseTensorMemInfo(const af::AscTensorAttr &ascir_tensor_info, std::string &node_type,
                                                 const TensorPtr &tensor) {
  ContainerPtr container;
  if (ascir_tensor_info.mem.alloc_type == af::AllocType::kAllocTypeQueue) {
    GE_ASSERT_SUCCESS(ConstructQueueContainer(ascir_tensor_info), "Construct queue failed.");
    container = queue_containers_[ascir_tensor_info.que.id];
    container->alloc_type = af::AllocType::kAllocTypeQueue;
    container->allocated_tensors.emplace_back(tensor);
    auto tqueue = static_cast<Queue *>(container.get());
    tqueue->buffer_num =
        (tqueue->buffer_num > ascir_tensor_info.que.buf_num) ? tqueue->buffer_num : ascir_tensor_info.que.buf_num;
  } else if (ascir_tensor_info.mem.alloc_type == af::AllocType::kAllocTypeBuffer) {
    GE_ASSERT_SUCCESS(ConstructBufferContainer(ascir_tensor_info), "Construct buffer failed.");
    container = buf_containers_[ascir_tensor_info.buf.id];
    container->alloc_type = af::AllocType::kAllocTypeBuffer;
    container->allocated_tensors.emplace_back(tensor);
    container->container_id = ascir_tensor_info.buf.id;
  } else if (ascir_tensor_info.mem.alloc_type == af::AllocType::kAllocTypeGlobal) {
    if (node_type == kWorkspace) {
      GE_ASSERT_TRUE(ascir_tensor_info.mem.hardware == af::MemHardware::kMemHardwareGM,
                     "Hardware for workspace should be MEM_HARDWARE_GM.");
      GE_ASSERT_SUCCESS(ConstructGlobalContainer(ascir_tensor_info), "Construct global container failed.");
      container = global_containers_[HardwareDef::GM];
      container->allocated_tensors.emplace_back(tensor);
    }
  } else {
    GELOGW("Mem alloc type[%d].", static_cast<int32_t>(ascir_tensor_info.mem.alloc_type));
  }
  if (container != nullptr) {
    for (const auto &axis_id : ascir_tensor_info.vectorized_axis) {
      for (const auto &buf_scope : container->buf_location) {
        tuning_space_->related_scopes[sub_axes_info_[axis_id].get()].insert(buf_scope);
      }
    }
  }

  GELOGD("Try to add combined tensors of type %s, container id = %s, reuse_id = %ld", node_type.c_str(),
         container == nullptr ? "nil" : container->name.c_str(), ascir_tensor_info.mem.reuse_id);
  if ((container != nullptr) && (ascir_tensor_info.mem.reuse_id != af::kIdNone)) {
    combined_tensors_[container][ascir_tensor_info.mem.reuse_id].emplace_back(tensor);
  }
  if (container == nullptr) {
    GELOGW("Tensor [%s] container was not found.", tensor->name.c_str());
  } else {
    GELOGD("Get tensor [%s] container [%s][%d] success.", tensor->name.c_str(), container->name.c_str(),
           container->container_id);
  }
  return af::SUCCESS;
}

void AscendGraphParser::ParseTensorOrigIdx(TensorPtr &tensor) const {
  for (int32_t i = static_cast<int32_t>(tensor->orig_idx.size()) - 1; i >= 0; i--) {
    if ((i > 0) && (tensor->orig_idx[i] <= tensor->orig_idx[i - 1])) {
      int64_t idx = tensor->orig_idx[i];
      for (int32_t j = 0; j < i; j++) {
        if (tensor->orig_idx[j] == idx) {
          tensor->orig_idx[j] = i - tensor->orig_idx.size();
        }
      }
    }
    if (tensor->orig_idx[i] > 0) {
      tensor->orig_idx[i] = i - tensor->orig_idx.size();
    }
  }
}

void AscendGraphParser::SetContinuesStrides(TensorPtr &tensor, af::AscTensorAttr &tensor_attr) const {
  GELOGD("tensor [%s] repeat size : %zu", tensor->name.c_str(), tensor->repeat.size());
  if (!tensor_attr.vectorized_strides.empty()) {
    tensor->stride.clear();
    for (auto &stride : tensor_attr.vectorized_strides) {
      tensor->stride.emplace_back(stride);
    }
  } else {  // 当tensor没有设置vectorized_strides时，根据repeat计算stride
    std::vector<Expr> new_stride;
    if (tensor->repeat.size() < 2U) {
      return;
    }
    new_stride.resize(tensor->stride.size());
    bool set_innerest_strde{false};
    int32_t pre_no_zero_index;  // 记录上一个非0 stride的子轴
    for (int32_t i = (static_cast<int32_t>(tensor->repeat.size()) - 1); i >= 0; --i) {
      // stride为0的维度， stride不修改
      if (tensor->stride[i] == 0) {
        new_stride[i] = tensor->stride[i];
        continue;
      }
      // 从内到外遇到第一个stride非0的子轴，其stride设置为1
      if (!set_innerest_strde) {
        Expr last_dim_size = af::sym::kSymbolOne;
        new_stride[i] = last_dim_size;
        set_innerest_strde = true;
        pre_no_zero_index = i;
        continue;
      }
      Expr stride_size = af::sym::Mul(tensor->repeat[pre_no_zero_index], tensor->stride[pre_no_zero_index]);
      new_stride[i] = stride_size;
      pre_no_zero_index = i;
    }
    tensor->stride = new_stride;
  }
}

af::Status AscendGraphParser::ParseTensorDims(TensorPtr &tensor, af::AscTensorAttr &tensor_attr) {
  std::unordered_map<int64_t, int64_t> axis_id_to_index;
  for (size_t i = 0u; i < tensor_attr.axis.size(); i++) {
    axis_id_to_index[tensor_attr.axis[i]] = i;
    GELOGD("Got tensor axis id = %ld, axis index = %ld", tensor_attr.axis[i], i);
  }
  size_t vec_cur_idx = 0;
  for (auto &axis_id : tensor_attr.vectorized_axis) {
    // vectorized轴必须要在tensor轴中，不在需要报错。
    auto axis_index_iter = axis_id_to_index.find(axis_id);
    GE_ASSERT_TRUE(vec_cur_idx < tensor_attr.vectorized_strides.size(), "%s(%s) vec_cur_idx %d is over limit size %zu",
                   tensor->name.c_str(), tensor->node_type.c_str(), vec_cur_idx, tensor_attr.vectorized_strides.size());
    GE_ASSERT_TRUE(axis_index_iter != axis_id_to_index.end(),
                   "Vectorized axis[%ld] not in tensor axis, tensor info is %s.", axis_id, tensor->ToString().c_str());
    auto axis_index = axis_index_iter->second;
    auto axis_repeat = tensor_attr.repeats[axis_index];
    auto axis_stride = tensor_attr.strides[axis_index];
    auto vectorized_stride = tensor_attr.vectorized_strides[vec_cur_idx];
    tensor->repeat.emplace_back(axis_repeat);
    tensor->stride.emplace_back(vectorized_stride);
    if (AttUtils::IsLoadStoreNode(tensor->owner_node)) {
      tensor->gm_stride.emplace_back(axis_stride);
    }
    tensor->dim_info.emplace_back(sub_axes_info_[axis_id].get());
    if (tensor->orig_idx.size() < tensor->repeat.size()) {
      tensor->orig_idx.emplace_back(axis_index);
    }
    vec_cur_idx += 1;
  }
  // 处理tensor的strides
  SetContinuesStrides(tensor, tensor_attr);
  GE_ASSERT_TRUE(tensor->stride.size() == tensor->dim_info.size(),
                 "Tensor [%s] stride num[%lu] not equal to dim info num[%lu].", tensor->name.c_str(),
                 tensor->stride.size(), tensor->dim_info.size());
  GELOGD("[DFX]parse tensor %s(%s): repeats [%s], gm_stride [%s], stride [%s]", tensor->name.c_str(),
         tensor->node_type.c_str(), GetVecString(tensor->repeat).c_str(), GetVecString(tensor->gm_stride).c_str(),
         GetVecString(tensor->stride).c_str());
  return af::SUCCESS;
}

af::Status AscendGraphParser::GetTensorAxes(TensorPtr &tensor, af::AscTensorAttr &tensor_attr) {
  // 获取vectorized轴之外包含stride==0的repeat信息
  if (tensor_attr.vectorized_axis.empty()) {
    GELOGW("Get tensor [%s] vectorized axis num==0.", tensor->name.c_str());
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(ParseTensorDims(tensor, tensor_attr), "Parse tensor dims failed.");
  ParseTensorOrigIdx(tensor);
  return af::SUCCESS;
}

af::Status AscendGraphParser::GetTensorAttrs(const af::AscNodePtr &node, const TensorPtr &tensor, size_t id,
                                             bool input) {
  std::string node_type = node->GetType();
  if (input) {
    GE_ASSERT_TRUE(id < node->inputs.Size(), "Get tensor [%zu] info failed.", id);
    auto &ascir_tensor_info = node->inputs[id].attr;
    GE_ASSERT_SUCCESS(ParseTensorMemInfo(ascir_tensor_info, node_type, tensor), "Parse tensor info failed.");
  } else {
    GE_ASSERT_TRUE(id < node->outputs().size(), "Get tensor [%zu] info failed.", id);
    auto &ascir_tensor_info = node->outputs[id].attr;
    GE_ASSERT_SUCCESS(ParseTensorMemInfo(ascir_tensor_info, node_type, tensor), "Parse tensor info failed.");
  }
  return af::SUCCESS;
}

af::Status AscendGraphParser::SetAxisPriority(const af::AscGraph &graph) {
  std::vector<SubAxis *> dims;
  for (auto &node : tuning_space_->node_infos) {
    auto dim_size = 0U;
    for (auto &input : node.inputs) {
      GE_ASSERT_NOTNULL(input, "Get input failed, node name is %s.", node.name.c_str());
      if (input->dim_info.size() > dim_size) {
        dims = input->dim_info;
      }
    }
    for (auto &output : node.outputs) {
      GE_ASSERT_NOTNULL(output, "Get output failed, node name is %s.", node.name.c_str());
      if (output->dim_info.size() > dim_size) {
        dims = output->dim_info;
      }
    }
    // 对多个Node均会设置最内轴，可能会存在多个最内轴(比如Transpose的场景，Load和Store有不同的最内轴)
    if (!dims.empty()) {
      // 最后一个向量轴为最内轴
      auto &innerest_dim = dims.back();
      innerest_dim->is_node_innerest_dim = true;
      GELOGD("Update innerest_dim[%s] for node %s.", innerest_dim->name.c_str(), node.name.c_str());
    }
  }

  std::unordered_map<int64_t, int64_t> innest_axis;
  for (const auto &node : graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node, "Get graph node failed.");
    innest_axis.clear();
    for (auto &axis_id : node->attr.sched.axis) {
      if (orig_axes_info_.find(axis_id) != orig_axes_info_.end()) {
        for (auto &orig_axis : orig_axes_info_[axis_id]) {
          innest_axis[orig_axis] = axis_id;
        }
      }
    }
    // 原始轴的最内轴，搜索轴初始值设置为最大值（原始轴的大小）
    for (auto &axis_id : innest_axis) {
      if (sub_axes_info_.find(axis_id.second) != sub_axes_info_.end()) {
        sub_axes_info_[axis_id.second]->is_last = true;
      }
    }
  }
  return af::SUCCESS;
}

af::Status AscendGraphParser::ParseWorkspaceNode(const af::AscNodePtr &ge_node) {
  if (ge_node->GetType() == kWorkspace) {
    auto ws_size = ascgen_utils::CalculateOneWorkspaceSize(ge_node);
    int64_t tensor_id = ge_node->outputs[0u].attr.mem.tensor_id;
    auto &max_workspace_sizes = tuning_space_->workspace_size_map;
    if (max_workspace_sizes.find(tensor_id) == max_workspace_sizes.end()) {
      max_workspace_sizes[tensor_id] = ws_size;
    } else {
      max_workspace_sizes[tensor_id] = af::sym::Max(max_workspace_sizes[tensor_id], ws_size);
    }
  }
  return af::SUCCESS;
}

af::Status AscendGraphParser::ParserNodeOutputInfos(const af::AscNodePtr &ge_node, const af::AscGraph &graph,
                                                    NodeInfo &node_info) {
  uint32_t depth = 0U;
  for (size_t out_id = 0U; out_id < ge_node->outputs().size(); out_id++) {
    // 取vectorized axis作为tensor dim, 认为vectorized axis的size才会占用local buffer
    auto &output_tensor = ge_node->outputs[out_id].attr;
    GE_ASSERT_SUCCESS(CheckAxisIdValid(output_tensor.vectorized_axis), "Invalid axis id.");
    GE_ASSERT_SUCCESS(CheckAxisIdValid(output_tensor.axis), "Invalid axis id.");
    // 向量轴为空不需要创建tensor
    if (output_tensor.vectorized_axis.empty()) {
      GELOGD("Node [%s] output [%zu] output vectorized size empty.", node_info.name.c_str(), out_id);
      continue;
    }
    TensorPtr tensor = af::MakeShared<Tensor>();
    GE_ASSERT_NOTNULL(tensor, "Create tensor failed.");
    tensor->name = ge_node->GetName() + kOutputNamePrefix + std::to_string(out_id);
    tensor->node_type = ge_node->GetType();
    auto data_type = ge_node->outputs[out_id].attr.dtype;
    GELOGD("Get node [%s] output[%zu] output datatype [%s] name[%s]", node_info.name.c_str(), out_id,
           BaseTypeUtils::DtypeToStr(data_type).c_str(), tensor->name.c_str());

    tensor->owner_node = ge_node.get();
    tensor->data_type = BaseTypeUtils::DtypeToStr(data_type);
    tensor->data_type_size = ge::GetSizeByDataType(data_type);
    if (output_tensor.mem.hardware == af::MemHardware::kMemHardwareGM) {
      tensor->loc = HardwareDef::GM;
    } else if (output_tensor.mem.hardware == af::MemHardware::kMemHardwareUB) {
      tensor->loc = HardwareDef::UB;
    }

    GE_ASSERT_SUCCESS(GetTensorAxes(tensor, output_tensor), "Get tensor size[%s] failed, graph name[%s].",
                      tensor->name.c_str(), graph.GetName().c_str());
    GE_ASSERT_SUCCESS(GetTensorAttrs(ge_node, tensor, out_id, false), "Get node[%s] output [%zu] attrs failed.",
                      node_info.name.c_str(), out_id);

    if (output_tensor.mem.alloc_type == af::AllocType::kAllocTypeQueue) {
      depth = depth > output_tensor.que.depth ? depth : output_tensor.que.depth;
    }
    node_info.depth = depth;
    node_info.outputs.emplace_back(tensor);
    tensor_info_.emplace(tensor->name, tensor);
  }
  return af::SUCCESS;
}

void AscendGraphParser::UpdateTensorLocType(const af::AscNodePtr &ge_node, size_t &in_id, TensorPtr &tensor) const {
  auto in_anchor = ge_node->GetInDataAnchor(in_id);
  if (in_anchor != nullptr) {
    auto peer_out = in_anchor->GetPeerOutAnchor();
    auto &input_tensor = ge_node->inputs[in_id].attr;
    if (input_tensor.mem.hardware == af::MemHardware::kMemHardwareGM) {
      tensor->loc = HardwareDef::GM;
    } else if (input_tensor.mem.hardware == af::MemHardware::kMemHardwareUB) {
      tensor->loc = HardwareDef::UB;
    }
  }
}

af::Status AscendGraphParser::ParseInputTensor(const af::AscNodePtr &ge_node, const NodeInfo &node_info, size_t in_id,
                                               TensorPtr &tensor) {
  // 只有data类型的输入 tensor才有mem信息
  auto in_anchor = ge_node->GetInDataAnchor(in_id);
  if (in_anchor != nullptr) {
    auto peer_out = in_anchor->GetPeerOutAnchor();
    if (peer_out != nullptr) {
      if (peer_out->GetOwnerNodeBarePtr()->GetType() == kData) {
        GE_ASSERT_SUCCESS(GetTensorAttrs(ge_node, tensor, in_id, true), "Get input [%zu] attrs failed.",
                          node_info.name.c_str(), in_id);
      }
      tensor->node_type = peer_out->GetOwnerNodeBarePtr()->GetType();
    }
  }
  return af::SUCCESS;
}

af::Status AscendGraphParser::ParserNodeInputInfos(const af::AscNodePtr &ge_node, const af::AscGraph &graph,
                                                   NodeInfo &node_info) {
  uint32_t depth = 0U;

  for (size_t in_id = 0U; in_id < ge_node->inputs.Size(); in_id++) {
    auto &input_tensor = ge_node->inputs[in_id].attr;
    GE_ASSERT_SUCCESS(CheckAxisIdValid(input_tensor.vectorized_axis), "Invalid axis id.");
    GE_ASSERT_SUCCESS(CheckAxisIdValid(input_tensor.axis), "Invalid axis id.");
    TensorPtr tensor = af::MakeShared<Tensor>();
    GE_ASSERT_NOTNULL(tensor, "Create tensor failed.");
    tensor->name = ge_node->GetName() + kInputNamePrefix + std::to_string(in_id);
    auto data_type = ge_node->inputs[in_id].attr.dtype;
    tensor->owner_node = ge_node.get();
    tensor->data_type = BaseTypeUtils::DtypeToStr(data_type);
    tensor->data_type_size = ge::GetSizeByDataType(data_type);
    GELOGD("Get node [%s] input[%zu] datatype [%s] name[%s]", node_info.name.c_str(), in_id,
           BaseTypeUtils::DtypeToStr(data_type).c_str(), tensor->name.c_str());
    // 取vectorized axis作为tensor dim, 认为vectorized axis的size才会占用local buffer
    GE_ASSERT_SUCCESS(GetTensorAxes(tensor, ge_node->inputs[in_id].attr), "Get tensor size[%s] failed, graph name[%s].",
                      tensor->name.c_str(), graph.GetName().c_str());
    GE_ASSERT_SUCCESS(ParseInputTensor(ge_node, node_info, in_id, tensor), "Parse input tensor failed, graph name[%s].",
                      graph.GetName().c_str());

    UpdateTensorLocType(ge_node, in_id, tensor);

    if (input_tensor.mem.alloc_type == af::AllocType::kAllocTypeQueue) {
      depth = depth > input_tensor.que.depth ? depth : input_tensor.que.depth;
    }
    node_info.depth = depth;
    node_info.inputs.emplace_back(tensor);
  }
  return af::SUCCESS;
}

void AscendGraphParser::UpdateContainer(ContainerPtr &container, const int32_t new_id) {
  auto &ori_id = container->reuse_id;
  if (combined_tensors_.find(container) != combined_tensors_.end()) {
    for (auto &tensors : combined_tensors_[container]) {
      std::vector<TensorPtr> tensor_list;
      for (const auto &tensor : tensors.second) {
        if (std::find(tensor_list.begin(), tensor_list.end(), tensor) == tensor_list.end()) {
          tensor_list.emplace_back(tensor);
        }
      }
      // 这里tensor_list只有1个时，在FA场景需要考虑ping/pang，但是在自动融合场景通过buffer_num表达，当前暂去掉该隐式double
      // buffer
      container->coexist_tensors.emplace_back(std::move(tensor_list));
    }
  }
  ori_id = new_id;
  for (auto &tensor : container->allocated_tensors) {
    tensor->resource_id = ori_id;
  }
}

af::Status AscendGraphParser::GetNodeFromData(const af::AscNodePtr &ge_node, NodeInfo &node_info) {
  if (node_info.node_type == kData) {
    node_info.from_data.insert(node_info.name);
    GELOGD("[%s] is Data node.", node_info.name.c_str());
    return af::SUCCESS;
  }
  for (const auto &input_node : ge_node->GetInNodes()) {
    for (const auto &node : tuning_space_->node_infos) {
      if (node.name != input_node->GetName()) {
        continue;
      }
      for (const auto &node_name : node.from_data) {
        node_info.from_data.insert(node_name);
      }
    }
  }
  std::string log;
  for (const auto &node_name : node_info.from_data) {
    log += node_name + ", ";
  }
  GELOGD("[%s] from Data node {%s}.", node_info.name.c_str(), log.c_str());
  return af::SUCCESS;
}

af::Status AscendGraphParser::ConvertNodeInfos(const af::AscNodePtr &ge_node, const ScheduleAttr &attrs,
                                               const af::AscGraph &graph, const bool use_cache_flag) {
  if (ascgen_utils::indirect_load::GetTemplateBehavior(ge_node).uses_direct_gm_pipeline) {
    const auto post_reduce_consumer = ascgen_utils::indirect_load::GetOnlyOutputConsumer(ge_node);
    if (post_reduce_consumer != nullptr &&
        post_reduce_consumer->attr.api.compute_type == af::ComputeType::kComputeReduce) {
      NodeInfo reduce_input_tensor_info;
      reduce_input_tensor_info.name = ge_node->GetName();
      GE_ASSERT_SUCCESS(ParserNodeOutputInfos(ge_node, graph, reduce_input_tensor_info));
    }
    return af::SUCCESS;
  }
  static const std::map<af::ComputeUnit, std::string> kUnitMap = {
      {af::ComputeUnit::kUnitNone, "UnitNone"},     {af::ComputeUnit::kUnitMTE1, "UnitMTE1"},
      {af::ComputeUnit::kUnitMTE2, "UnitMTE2"},     {af::ComputeUnit::kUnitMTE3, "UnitMTE3"},
      {af::ComputeUnit::kUnitScalar, "UnitScalar"}, {af::ComputeUnit::kUnitVector, "UnitVector"},
      {af::ComputeUnit::kUnitCube, "UnitCube"},     {af::ComputeUnit::kUnitInvalid, "UnitInvalid"},
  };
  NodeInfo node_info;
  node_info.name = ge_node->GetName();
  node_info.node_type = ge_node->GetType();
  const bool preserve_cache_condition =
      use_cache_flag || ShouldPreserveCacheCondition(ge_node, /*graph_has_reduce=*/!use_cache_flag);
  node_info.exec_condition = preserve_cache_condition ? attrs.exec_condition : af::ExecuteCondition::kNoCache;
  if (kUnitMap.find(ge_node->attr.api.unit) == kUnitMap.end()) {
    node_info.node_unit = "UnitInvalid";
  } else {
    node_info.node_unit = kUnitMap.at(ge_node->attr.api.unit);
  }
  node_info.trans_config = "";
  int64_t loop_axis = attrs.loop_axis_id;
  GE_ASSERT_SUCCESS(ParseWorkspaceNode(ge_node), "Parser workspace node info failed.");
  GE_ASSERT_SUCCESS(ParserNodeOutputInfos(ge_node, graph, node_info), "Parser node output info failed.");
  GE_ASSERT_SUCCESS(ParserNodeInputInfos(ge_node, graph, node_info), "Parser node input info failed.");
  for (const auto &axis_info : attrs.sched_axis_info) {
    if (sub_axes_info_.find(axis_info->id) != sub_axes_info_.end()) {
      node_info.loop_axes.emplace_back(sub_axes_info_[axis_info->id].get());
    }
  }
  node_info.node_ptr = ge_node;
  GE_ASSERT_SUCCESS(FillSpecificParams(ge_node, node_info));
  UpdateConcatVectorizedAxes(ge_node, loop_axis, sub_axes_info_);
  GE_ASSERT_SUCCESS(GetNodeFromData(ge_node, node_info));
  VectorFunctionGraphParser vector_function_graph_parser(ge_node, graph);
  GE_ASSERT_SUCCESS(vector_function_graph_parser.Parse(), "Parse node infos failed, graph = %s, node = %s %s.",
                    graph.GetName().c_str(), ge_node->GetNamePtr(), ge_node->GetTypePtr());
  node_info.sub_nodes_infos = vector_function_graph_parser.GetNodesInfos();
  // 解析ge_node, 设置到node_info中的sub_nodes_infos
  tuning_space_->node_infos.push_back(node_info);
  GELOGD("[DFX]Parse %s, use_cache_flag %d", node_info.DebugString().c_str(), use_cache_flag);
  return af::SUCCESS;
}

void AscendGraphParser::AssembleTensorInfos() {
  GELOGD("Assemble tensor info start.");
  uint32_t container_id = 0U;
  for (auto &container : queue_containers_) {
    UpdateContainer(container.second, container_id);
    tuning_space_->containers.emplace_back(container.second);
    container_id++;
  }
  for (auto &container : buf_containers_) {
    UpdateContainer(container.second, container_id);
    tuning_space_->containers.emplace_back(container.second);
    container_id++;
  }
  for (auto &container : global_containers_) {
    UpdateContainer(container.second, container_id);
    tuning_space_->global_containers.emplace_back(container.second);
    container_id++;
  }
  GELOGD("Assemble tensor info success.");
}

af::Status AscendGraphParser::ParserBlockDimInfo() {
  std::vector<std::vector<int64_t>> visited;
  std::vector<int64_t> block_info;

  for (auto &node_order : topo_order_node_) {
    auto &node = node_order.second;
    if (graph_sched_info_.find(node) == graph_sched_info_.end()) {
      continue;
    }
    block_info = graph_sched_info_[node].block_out_dim_info;
    if (block_info.empty()) {
      continue;
    }
    std::sort(block_info.begin(), block_info.end());
    if (find(visited.begin(), visited.end(), block_info) != visited.end()) {
      continue;
    }
    visited.emplace_back(block_info);
  }
  for (auto &block_ids : visited) {
    std::vector<SubAxis *> sub_axes;
    for (auto &block_id : block_ids) {
      GE_ASSERT_NOTNULL(sub_axes_info_[block_id], "Get sub axis info failed.");
      sub_axes.emplace_back(sub_axes_info_[block_id].get());
    }
    tuning_space_->block_dims.emplace_back(sub_axes);
  }
  return af::SUCCESS;
}

void AscendGraphParser::ParserOptionalInfos(const af::AscGraph &graph) const {
  (void)graph;
}

af::Status AscendGraphParser::CalculateReservedUbSize(const af::AscGraph &graph) {
  tuning_space_->reserve_ub["ascendc"] = ascgen_utils::CalcReservedTmpBufSizeForAscGraph(graph);
  for (const auto &node : graph.GetAllNodes()) {
    const int64_t simt_dcache_size = ::ascir::GetDcacheSize(*node);
    if (simt_dcache_size > 0) {
      tuning_space_->reserve_ub["simt_dcache"] = static_cast<uint32_t>(simt_dcache_size);
      break;
    }
  }
  for (const auto &reserve_ub : tuning_space_->reserve_ub) {
    GELOGD("Got calculated reserved UB size [%s:%u].", reserve_ub.first.c_str(), reserve_ub.second);
  }
  return af::SUCCESS;
}

af::Status AscendGraphParser::ConvertToTuningSpace(const af::AscGraph &graph) {
  bool use_cache_flag = !HasComputeType(graph, af::ComputeType::kComputeReduce);
  for (auto &node_order : topo_order_node_) {
    auto &node = node_order.second;
    auto node_iter = graph_sched_info_.find(node);
    if (node_iter == graph_sched_info_.end()) {
      continue;
    }
    const auto &sched_attrs = node_iter->second;
    GE_ASSERT_SUCCESS(ConvertNodeInfos(node, sched_attrs, graph, use_cache_flag), "Parse node info failed.");
  }
  GE_ASSERT_SUCCESS(CalculateReservedUbSize(graph), "Calculate reserved ub size failed, graph:%s",
                    graph.GetName().c_str());
  GE_ASSERT_SUCCESS(ParserBlockDimInfo(), "Parse block dim info failed.");
  AssembleTensorInfos();
  SetAxisPriority(graph);
  ParserOptionalInfos(graph);
  for (auto &pair : sub_axes_info_) {
    GELOGD("[DFX] Axis id[%lu], info [%s], graph[%s]", pair.first, pair.second->ToString().c_str(),
           graph.GetName().c_str());
    tuning_space_->sub_axes.emplace_back(std::move(pair.second));
  }
  tuning_space_->asc_graph = &graph;
  return af::SUCCESS;
}

// 检测Reduce/Broadcast分核Store冲突场景
// 使用 AttUtils 的公共函数检测 Reduce/Broadcast 轴
af::Status AscendGraphParser::CheckReduceBroadcastSplitStoreConflict() {
  // 首先收集所有Reduce轴和Brodacast轴的原始名称（从所有节点）
  std::set<std::string> reduce_axis_orig_names;
  std::set<std::string> broadcast_axis_orig_names;

  for (const auto &node : tuning_space_->node_infos) {
    // 从所有节点收集Reduce/Broadcast轴（使用新签名）
    AttUtils::CollectReduceAxisNames(node, reduce_axis_orig_names);
    AttUtils::CollectBroadcastAxisNames(node, broadcast_axis_orig_names);
  }

  // 然后遍历所有节点的 loop_axes 进行标记（不限制 Store 节点）
  for (const auto &node : tuning_space_->node_infos) {
    GELOGD("[DFX] Check node [%s] type[%s] for Reduce/Broadcast split axis.", node.name.c_str(),
           node.node_type.c_str());

    // 遍历loop_axes，标记Reduce/Broadcast分核轴
    for (const auto &axis : node.loop_axes) {
      // 跳过非分核轴或已被标记为Reduce分核轴的轴
      if (!axis->is_bind_multi_core || axis->is_reduce_split_axis) {
        continue;
      }

      // 检查并标记该轴是否为Reduce分核轴
      if (CheckAndMarkReduceSplitAxis(axis, reduce_axis_orig_names)) {
        continue;  // 已标记为Reduce分核轴，跳过Broadcast检查
      }

      // 检查并标记该轴是否为Broadcast分核轴
      CheckAndMarkBroadcastSplitAxis(axis, broadcast_axis_orig_names);
    }
  }

  return af::SUCCESS;
}

// 检查并标记轴是否为 Reduce 分核轴
bool AscendGraphParser::CheckAndMarkReduceSplitAxis(SubAxis *axis,
                                                    const std::set<std::string> &reduce_axis_orig_names) const {
  for (const auto &orig_name : axis->orig_axis_name) {
    if (reduce_axis_orig_names.find(orig_name) != reduce_axis_orig_names.end()) {
      axis->is_reduce_split_axis = true;
      GELOGD("[DFX] Marked axis [%s] as reduce split axis", axis->name.c_str());
      return true;
    }
  }
  return false;
}

// 检查并标记轴是否为 Broadcast 分核轴
bool AscendGraphParser::CheckAndMarkBroadcastSplitAxis(SubAxis *axis,
                                                       const std::set<std::string> &broadcast_axis_orig_names) const {
  for (const auto &orig_name : axis->orig_axis_name) {
    if (broadcast_axis_orig_names.find(orig_name) != broadcast_axis_orig_names.end()) {
      axis->is_broadcast_split_axis = true;
      GELOGD("[DFX] Marked axis [%s] as broadcast split axis", axis->name.c_str());
      return true;
    }
  }
  return false;
}

af::Status AscendGraphParser::GraphParser(const af::AscGraph &graph) {
  GE_ASSERT_SUCCESS(SetAscGraphPhyType(graph));
  GE_ASSERT_SUCCESS(ParserSchedInfo(graph), "Parser sched info failed.");
  GE_ASSERT_SUCCESS(ParserOriginAxis(graph), "Parser origin axis info failed.");
  GE_ASSERT_SUCCESS(CreateSubAxisInfo(graph), "Create sub axis info failed.");
  GE_ASSERT_SUCCESS(ConvertToTuningSpace(graph), "Construct tuning space from infos failed");

  // 新增：检测Reduce/Broadcast分核Store冲突
  GE_ASSERT_SUCCESS(CheckReduceBroadcastSplitStoreConflict());

  std::string dump_debug_info;
  if (GetThreadLocalContext().GetOption(kDumpDebugInfo, dump_debug_info) != af::SUCCESS) {
    return af::SUCCESS;
  }
  ge::char_t realpath_file[PATH_MAX] = {0x00};
  std::string json_path;
  if (dump_debug_info.back() == '/') {
    json_path = dump_debug_info + "tuning_space.json";
  } else {
    json_path = dump_debug_info + "/" + "tuning_space.json";
  }
  auto ret = realpath(json_path.c_str(), realpath_file);
  if (ret == nullptr) {
    GELOGD("Json path [%s] not found.", json_path.c_str());
    return af::SUCCESS;
  }
  std::ofstream tuning_space_file(realpath_file, std::ios::trunc);
  if (tuning_space_file.is_open()) {
    tuning_space_file << TuningSpacePrint();
    tuning_space_file.close();
  }
  return af::SUCCESS;
}

std::string AscendGraphParser::TuningSpacePrint(const NodeInfo &node_info) const {
  std::ostringstream oss;
  Expr size = CreateExpr(1U);
  for (auto &loop_axis : node_info.loop_axes) {
    size = af::sym::Mul(size, loop_axis->repeat);
  }
  std::string loop_times = (IsValid(size)) ? Str(size) : "";
  oss << "NodeInfo{\n";
  oss << "name: \"" << node_info.name << "\", ";
  oss << "node_type: \"" << node_info.node_type << "\", ";
  oss << "trans_config: \"" << node_info.trans_config << "\", ";
  oss << "loop_times: " << loop_times << ", ";
  oss << "depth: " << node_info.depth << ", ";
  if (!node_info.inputs.empty()) {
    oss << "inputs: {";
    for (const auto &tensor : node_info.inputs) {
      oss << tensor->name << ", ";
    }
    oss << "}, ";
  }
  if (!node_info.outputs.empty()) {
    oss << "outputs: {";
    for (const auto &tensor : node_info.outputs) {
      oss << tensor->name << ", ";
    }
    oss << "}, ";
  }
  return oss.str();
}

std::string AscendGraphParser::TuningSpacePrint(const Tensor &tensor) const {
  std::ostringstream oss;
  oss << "Tensor{\n";
  oss << "name: \"" << tensor.name << "\", ";
  oss << "data_type_size: " << tensor.data_type_size << ", ";
  oss << "resource_id: " << tensor.resource_id << ", ";
  oss << "node_type: " << tensor.node_type << ", ";
  oss << "data_type: " << tensor.data_type << ", ";
  oss << "loc: " << HardwareType2Str.at(tensor.loc) << ", ";
  if (!tensor.dim_info.empty()) {
    oss << "dim_info: {";
    for (const auto &sub_axis : tensor.dim_info) {
      oss << sub_axis->name << ", ";
    }
    oss << "}, ";
  }
  oss << "ori_dim_info: {";
  for (const auto &sub_axis : tensor.ori_dim_info) {
    oss << sub_axis->name << ", ";
  }
  oss << "}, ";
  oss << tensor.GetStride();
  oss << tensor.GetRepeat();
  oss << "}";
  return oss.str();
}

std::string AscendGraphParser::TuningSpacePrint(const SubAxis &sub_axis) const {
  std::ostringstream oss;
  oss << "SubAxis{\n";
  oss << "name: \"" << sub_axis.name << "\", ";
  oss << "axis_type: " << AxisType2Str.at(sub_axis.axis_type) << ", ";
  oss << "is_bind_multi_core: " << std::boolalpha << sub_axis.is_bind_multi_core << ", ";
  oss << "enable_tail: " << std::boolalpha << sub_axis.enable_tail << ", ";
  oss << "enable_pad: " << std::boolalpha << sub_axis.enable_pad << ", ";
  oss << "is_split: " << std::boolalpha << sub_axis.is_split << ", ";
  oss << "is_node_innerest_dim: " << std::boolalpha << sub_axis.is_node_innerest_dim << ", ";
  oss << "is_last: " << std::boolalpha << sub_axis.is_last << ", ";
  auto align = IsValid(sub_axis.align) ? Str(sub_axis.align) : "";
  oss << "align: " << align << ", ";
  auto repeat = IsValid(sub_axis.repeat) ? Str(sub_axis.repeat) : "";
  oss << "repeat: " << repeat << ",";
  oss << ", orig_axis_name: ";
  for (auto &name : sub_axis.orig_axis_name) {
    oss << name << ",";
  }
  oss << ", parent_axis_name: ";
  for (auto &axis : sub_axis.parent_axis) {
    oss << axis->name << ",";
  }
  oss << "}";
  return oss.str();
}

std::string AscendGraphParser::TuningSpacePrint(const Container &container) const {
  std::ostringstream oss;
  oss << "Container{\n";
  oss << "name: \"" << container.name << "\", ";
  oss << "buffer_num: \"" << container.GetBufferNum() << "\", ";
  oss << "reuse_id: " << container.reuse_id << ", ";
  oss << "container_id: " << container.container_id << ", ";
  if (!container.allocated_tensors.empty()) {
    oss << "allocated_tensors: {";
    for (const auto &tensor : container.allocated_tensors) {
      oss << TuningSpacePrint(*tensor) << ", ";
    }
    oss << "}, ";
  }
  if (!container.buf_location.empty()) {
    oss << "buf_location: {";
    for (const auto &buf_scope : container.buf_location) {
      oss << "{" << HardwareType2Str.at(buf_scope) << "}, ";
    }
    oss << "}, ";
  }
  if (!container.coexist_tensors.empty()) {
    oss << "coexist_tensors: {";
    for (const auto &tensors : container.coexist_tensors) {
      oss << "{";
      for (const auto &tensor : tensors) {
        oss << TuningSpacePrint(*tensor) << ", ";
      }
      oss << "}, ";
    }
    oss << "}, ";
  }
  oss << "}";
  return oss.str();
}

std::string AscendGraphParser::TuningSpacePrint() const {
  std::ostringstream oss;
  oss << "TuningSpace{ \n";
  oss << "sub_axes: {\n";
  for (const auto &sub_axis : tuning_space_->sub_axes) {
    oss << TuningSpacePrint(*sub_axis) << ", \n";
  }
  oss << "}, \n\n";

  oss << "containers: {\n";
  for (const auto &container : tuning_space_->containers) {
    oss << TuningSpacePrint(*container) << "\n";
  }
  oss << "}, \n\n";

  oss << "node_infos: {\n";
  for (const auto &node_info : tuning_space_->node_infos) {
    oss << TuningSpacePrint(node_info) << "\n";
  }
  oss << "}, \n\n";

  oss << "tensors: {\n";
  for (const auto &node_info : tuning_space_->node_infos) {
    for (auto &ascir_tensor_info : node_info.inputs) {
      oss << TuningSpacePrint(*ascir_tensor_info) << "\n";
    }
    for (auto &ascir_tensor_info : node_info.outputs) {
      oss << TuningSpacePrint(*ascir_tensor_info) << "\n";
    }
  }
  oss << "}, \n\n";

  oss << "block_dim: {\n";
  for (const auto &dim_group : tuning_space_->block_dims) {
    oss << "group: ";
    for (const auto &dim : dim_group) {
      oss << dim->name << ",";
    }
    oss << ";\n";
  }
  oss << "}, \n\n";

  oss << "related_scopes: {\n";
  for (const auto &iter : tuning_space_->related_scopes) {
    oss << iter.first->name << ":";
    for (auto &type : iter.second) {
      oss << HardwareType2Str.at(type) << ",";
    }
    oss << "\n";
  }
  oss << "}, \n";

  oss << "}";
  return oss.str();
}
}  // namespace att
