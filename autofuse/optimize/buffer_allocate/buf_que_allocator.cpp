/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "buf_que_allocator.h"
#include <queue>
#include "ascir_ops.h"
#include "ascgen_log.h"
#include "ascir_ops_utils.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "schedule_utils.h"
#include "graph_utils.h"
#include "common_utils.h"
#include "indirect_load_utils.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "platform/platform_factory.h"
#include "mem_reuse_manager.h"
#include "v35/optimize/partition/vector_func_partitioner.h"
#include "ascir_utils.h"

using namespace af::ascir_op;
using namespace af::ops;

namespace {
bool IsSupportInplace(const af::AscNodePtr &node) {
  // 1. ascir注册信息表示该节点不支持inplace
  if (!ascgen_utils::IsNodeSupportsInplace(node)) {
    GELOGD("Node %s[%s] does not support inplace.", node->GetTypePtr(), node->GetNamePtr());
    return false;
  }
  // 2. 当前节点如果是多输出，不支持复用（白名单可保证没有多输出节点，但是还是加一个校验）
  GE_WARN_ASSERT(node->GetAllOutDataAnchorsSize() == 1U, "%s[%s] does not support output anchor size=%u.",  // 单输出
                 node->GetTypePtr(), node->GetNamePtr(), node->GetAllOutDataAnchorsSize());
  GE_WARN_ASSERT(node->GetInDataNodesSize() > 0UL,  // 多输入
                 "%s[%s] does not support input size=0.", node->GetTypePtr(), node->GetNamePtr());
  GE_WARN_ASSERT(node->GetOutDataNodesSize() > 0UL,  // 单输出，多引用
                 "%s[%s] does not support output size=0.", node->GetTypePtr(), node->GetNamePtr());

  // 3. 若节点的任一输入是个单输出多引用，则不支持复用（存在优化空间，当前先不细化
  for (const auto &input_node : node->GetInDataNodes()) {
    if (input_node->GetOutDataNodesSize() > 1U) {
      GELOGD("Node %s[%s] has %u outputs, does not support inplace.", input_node->GetTypePtr(),
             input_node->GetNamePtr());
      return false;
    }
  }
  // 4. 输入输出节点数据类型不同，不支持复用
  auto tmp_dtype = ge::DT_MAX;
  for (const auto &input : node->inputs()) {
    if (tmp_dtype == ge::DT_MAX) {
      tmp_dtype = input->attr.dtype;
    } else if (tmp_dtype != input->attr.dtype) {
      const auto &dtype_str1 = af::TypeUtils::DataTypeToSerialString(tmp_dtype).c_str();
      const auto &dtype_str2 = af::TypeUtils::DataTypeToSerialString(input->attr.dtype).c_str();
      GELOGD("Node %s[%s] input data type not equal (%s, %s).", node->GetTypePtr(), node->GetNamePtr(), dtype_str1,
             dtype_str2);
      return false;
    }
  }
  for (const auto output : node->outputs()) {
    if (tmp_dtype != output->attr.dtype) {
      const auto &dtype_str1 = af::TypeUtils::DataTypeToSerialString(tmp_dtype).c_str();
      const auto &dtype_str2 = af::TypeUtils::DataTypeToSerialString(output->attr.dtype).c_str();
      GELOGD("Node %s[%s] output data type not equal (%s, %s).", node->GetTypePtr(), node->GetNamePtr(), dtype_str1,
             dtype_str2);
      return false;
    }
  }
  return true;
}

// 从节点调度轴中取最外层BlockOuter轴id：外提Load标记为该轴（语义：block层级一次执行/每核一次），
// ATT的ParserSchedInfo据此将BlockOuter轴归入block_dim、其后轴不计入loop_axes，exe_time为空积=1。
// 无BlockOuter轴（理论上被图级门控与调度不变量排除）时返回kIdNone，保守不外提。
int64_t FindBlockOuterAxis(af::AscGraph &graph, const af::AscNodePtr &node) {
  for (const auto &axis_id : node->attr.sched.axis) {
    const auto &axis = graph.FindAxis(axis_id);
    if (axis != nullptr && axis->type == af::Axis::kAxisTypeBlockOuter) {
      return axis_id;
    }
  }
  return af::kIdNone;
}
}  // namespace

namespace optimize {
struct NodeLifecycle {
  af::AscNodePtr node;
  int64_t start;
  int64_t end;
  mutable uint32_t seen_nums;
};

struct LifecycleComparator {
  bool operator()(const NodeLifecycle &lhs, const NodeLifecycle &rhs) const {
    return (lhs.seen_nums > rhs.seen_nums) || ((lhs.end - lhs.start) > (rhs.end - rhs.start)) ||
           (lhs.node->GetOpDesc()->GetId() < rhs.node->GetOpDesc()->GetId());
  }
};
using LifecycleSet = std::set<NodeLifecycle, LifecycleComparator>;

static bool GetOverlapWithSetFlag(NodeLifecycle &node, LifecycleSet &set) {
  bool overlap = false;
  for (auto &it : set) {
    if (!(node.end < it.start || it.end < node.start)) {
      overlap = true;
      node.seen_nums++;
      it.seen_nums++;
    }
  }
  return overlap;
}

static std::list<LifecycleSet> FindOverlappingNodeSets(const std::vector<NodeLifecycle> &lifecycles,
                                                       size_t max_que_num) {
  std::list<LifecycleSet> overlapping_sets;
  const size_t total_cycle_size = lifecycles.size();
  std::vector<bool> used(total_cycle_size, false);
  for (size_t i = 0UL; i < total_cycle_size; ++i) {
    if (used[i]) {
      continue;
    }
    LifecycleSet cur_set = {lifecycles[i]};
    used[i] = true;
    for (size_t j = i + 1UL; j < total_cycle_size; ++j) {
      NodeLifecycle cur = lifecycles[j];
      if (!used[j] && GetOverlapWithSetFlag(cur, cur_set)) {
        cur_set.emplace(cur);
        used[j] = true;
      }
    }
    std::set<int64_t> used_que_ids;
    for (const auto &iter : cur_set) {
      used_que_ids.emplace(iter.node->outputs[0].attr.que.id);
    }
    if (used_que_ids.size() > max_que_num) {
      overlapping_sets.push_back(cur_set);
    }
  }
  return overlapping_sets;
}

Status BufQueAllocator::AllocBufQueForSingleImplGraph(af::AscGraph &impl_graph, size_t max_que_num,
                                                      bool is_reduce_mem_reuse) const {
  size_t total_vecin_nums{0UL};
  size_t total_vecout_nums{0UL};
  AllocateWithinGroup(impl_graph, total_vecin_nums, total_vecout_nums, is_reduce_mem_reuse);

  if (total_vecin_nums > max_que_num) {
    GELOGD("Graph [%s] occupies [%zu] vecin queues, exceeding limit [%zu]. Attempting to shorten lifetime.",
           impl_graph.GetName().c_str(), total_vecin_nums, max_que_num);
    GE_CHK_STATUS_RET(ShortenVecinLifetime(impl_graph, max_que_num), "Failed to shorten vecin lifetime for graph [%s].",
                      impl_graph.GetName().c_str());
    AllocateWithinGroup(impl_graph, total_vecin_nums, total_vecout_nums, is_reduce_mem_reuse);
  }

  if (total_vecout_nums > max_que_num) {
    GELOGD("Graph [%s] occupies [%zu] vecout queues, exceeding limit [%zu]. Attempting to shorten lifetime.",
           impl_graph.GetName().c_str(), total_vecout_nums, max_que_num);
    GE_CHK_STATUS_RET(ShortenVecoutLifetime(impl_graph, max_que_num),
                      "Failed to shorten vecout lifetime for graph [%s].", impl_graph.GetName().c_str());
    AllocateWithinGroup(impl_graph, total_vecin_nums, total_vecout_nums, is_reduce_mem_reuse);
  }

  // 超过限制存在kernel卡死风险
  GE_CHK_BOOL_ONLY_LOG(
      total_vecin_nums <= max_que_num && total_vecout_nums <= max_que_num,
      "Graph [%s] still exceeds queue limits after lifetime adjustment: vecin=%zu, vecout=%zu, may be error.",
      impl_graph.GetName().c_str(), total_vecin_nums, total_vecout_nums);
  return af::SUCCESS;
}

Status BufQueAllocator::AllocBufQue(::ascir::FusedScheduledResult &fused_scheduled_result) {
  GE_CHK_STATUS_RET(PrepareImplGraphMemoryPlan(fused_scheduled_result), "PrepareImplGraphMemoryPlan failed");
  GE_CHK_STATUS_RET(CollectFusedIoNodes(fused_scheduled_result), "CollectFusedIoNodes failed");
  return ge::GRAPH_SUCCESS;
}

Status BufQueAllocator::PrepareImplGraphMemoryPlan(::ascir::FusedScheduledResult &fused_scheduled_result) {
  const auto &platform = PlatformFactory::GetInstance().GetPlatform();
  GE_CHECK_NOTNULL(platform, "Platform is not found.");

  const PlatformConfig config = platform->GetPlatformConfig();
  GE_CHK_STATUS_RET(AllocateForIoNodes(fused_scheduled_result), "AllocateForIoNodes failed");
  for (auto &scheduled_results : fused_scheduled_result.node_idx_to_scheduled_results) {
    for (auto &scheduled_result : scheduled_results) {
      cube_type = scheduled_result.cube_type;
      for (auto &schedule_group : scheduled_result.schedule_groups) {
        for (auto &impl_graph : schedule_group.impl_graphs) {
          GE_CHK_STATUS_RET(
              ProcessSingleImplGraph(impl_graph, *platform, config.max_que_num, scheduled_result.is_reduce_mem_reuse));
          GE_CHK_STATUS_RET(HoistInvariantLoads(impl_graph), "HoistInvariantLoads failed");
        }
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

// brc缓存Load可外提的条件判定（HoistInvariantLoads调用，外提资格与TBuf转换等价）：
// 1. 普通Load且为brc缓存节点；
// 2. 输出的直接消费者中无Store（保护load->store TQueBind直连MTE2->MTE3路径）；
// 3. 输出队列为单缓冲（无双缓冲流水可破坏）；
// 4. Load的GM地址在整个循环嵌套内恒定（外提前导仅能表达静态offset与BlockOuter贡献）；
// 5. Load使用的que不存在复用或共用（que上无其它tensor，转换后share/reuse链无残留交互）；
// 6. 输出tensor元素个数大于1（各repeats因子均静态≤1即为单元素/标量，拷贝代价极低且走
//    ub_scalar/Duplicate特殊路径，外提无收益；动态shape无法判定时放行）。
bool BufQueAllocator::CanHoist(const af::AscGraph &graph, const af::AscNodePtr &node) {
  if (!IsOps<Load>(node)) {
    return false;
  }
  if (!ascgen_utils::IsNodeCacheable(node)) {
    return false;
  }
  if (node->outputs().empty()) {
    return false;
  }
  // 元素个数>1判定：所有repeats因子均可静态判定为≤1（含空repeats，即单元素/标量tensor）时拒绝——
  // 拷贝代价极低且走ub_scalar/Duplicate特殊路径，外提无收益；存在>1或无法判定的因子则放行
  bool all_elem_le_one = true;
  for (const auto &repeat : node->outputs[0].attr.repeats) {
    if (af::SymbolicUtils::StaticCheckLe(repeat, af::sym::kSymbolOne) != af::TriBool::kTrue) {
      all_elem_le_one = false;
      break;
    }
  }
  if (all_elem_le_one) {
    return false;
  }
  for (const auto &peer_anchor : node->outputs[0].anchor.GetPeerInDataAnchorsPtr()) {
    GE_ASSERT_NOTNULL(peer_anchor);
    const auto peer_node = std::dynamic_pointer_cast<af::AscNode>(peer_anchor->GetOwnerNode());
    GE_ASSERT_NOTNULL(peer_node);
    if (IsOps<Store>(peer_node)) {
      return false;
    }
  }
  if (node->outputs[0].attr.que.buf_num != 1) {
    return false;
  }
  if (!IsQueExclusive(graph, &node->outputs[0])) {
    return false;
  }
  return IsLoadAddressLoopInvariant(&node->outputs[0]);
}

// que独占判定：同que_id且Queue类型的tensor仅output自身才允许转换。
// que_id+reuse_id表达que的复用/共用语义（同que同reuse_id为共用、同que异reuse_id为复用），
// 统一以"que上存在其它tensor"排除，避免转换后与share/reuse链（share_offset、IsLastShare、
// reuse_from/next等）交互。
bool BufQueAllocator::IsQueExclusive(const af::AscGraph &graph, const af::AscTensor *output) {
  for (const auto &node : graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    for (const auto &out_tensor : node->outputs()) {
      if (out_tensor->attr.mem.alloc_type != af::AllocType::kAllocTypeQueue ||
          out_tensor->attr.que.id != output->attr.que.id ||
          out_tensor->attr.mem.tensor_id == output->attr.mem.tensor_id) {
        continue;
      }
      GELOGD("Que[%ld] is shared or reused by tensor[%ld], cannot convert tensor[%ld] to TBuf.", output->attr.que.id,
             out_tensor->attr.mem.tensor_id, output->attr.mem.tensor_id);
      return false;
    }
  }
  return true;
}

// 地址恒定判定
bool BufQueAllocator::IsLoadAddressLoopInvariant(const af::AscTensor *output) {
  const auto &axis_list = output->attr.axis;
  const auto &strides = output->attr.strides;
  if (axis_list.size() != strides.size()) {
    return false;
  }
  const auto &vectorized_axis = output->attr.vectorized_axis;
  for (size_t i = 0UL; i < axis_list.size(); ++i) {
    bool is_vectorized = false;
    for (const auto vectorized_id : vectorized_axis) {
      if (vectorized_id == axis_list[i]) {
        is_vectorized = true;
        break;
      }
    }
    if (is_vectorized) {
      continue;  // 已折叠进连续搬运段（block_len），不参与GM offset表达式
    }
    if (af::SymbolicUtils::StaticCheckEq(strides[i], af::sym::kSymbolZero) == af::TriBool::kTrue) {
      continue;  // stride为0，不参与GM offset表达式
    }
    return false;
  }
  return true;
}

// 当前图的TBuf id最大值：tensor的buf.id与tmp_buffers的id共用MemReuseManager的buf_id_计数器，
// 新TBuf取max+1避免与ques/bufs冲突（各impl graph独立成kernel，id按图独立）。
int64_t BufQueAllocator::GetMaxBufId(const af::AscGraph &graph) {
  int64_t max_buf_id = -1;
  for (const auto &node : graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    for (const auto &output : node->outputs()) {
      max_buf_id = std::max(max_buf_id, output->attr.buf.id);
    }
    for (const auto &tmp_buffer : node->attr.tmp_buffers) {
      max_buf_id = std::max(max_buf_id, tmp_buffer.id);
    }
  }
  return max_buf_id;
}

Status BufQueAllocator::HoistInvariantLoads(af::AscGraph &graph) {
  std::vector<std::pair<af::AscNodePtr, int64_t>> hoistable_nodes;
  for (const auto &node : graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    // 图中包含 IndirectLoad（GM地址数据相关，天然循环变化）或 kComputeCube
    // （CV/Cube 管线）节点时，不做外提，直接返回
    if (IsOps<IndirectLoad>(node) || node->attr.api.compute_type == af::ComputeType::kComputeCube) {
      GELOGI("Graph[%s] contains IndirectLoad/Cube node[%s], skip hoist.", graph.GetName().c_str(), node->GetNamePtr());
      return af::SUCCESS;
    }
    if (!CanHoist(graph, node)) {
      continue;
    }
    const auto block_outer_axis = FindBlockOuterAxis(graph, node);
    if (block_outer_axis != af::kIdNone) {
      hoistable_nodes.emplace_back(node, block_outer_axis);
    }
  }
  if (hoistable_nodes.empty()) {
    return af::SUCCESS;
  }
  // 改图前落盘现场图，便于比对外提前后的Load/mem属性变化（AUTOFUSE_DFX_FLAGS控制落盘）
  ascir::utils::DumpGraph(graph, "BeforeHoistInvariantLoads");
  int64_t next_buf_id = GetMaxBufId(graph) + 1;
  for (const auto &hoistable_node : hoistable_nodes) {
    const auto &node = hoistable_node.first;
    node->attr.sched.loop_axis = hoistable_node.second;
    node->attr.sched.exec_condition = af::ExecuteCondition::kNoCache;
    auto &output_attr = node->outputs[0].attr;
    output_attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
    output_attr.mem.position = af::Position::kPositionVecCalc;
    output_attr.mem.reuse_id = af::kIdNone;
    output_attr.buf.id = next_buf_id++;
    output_attr.que.id = af::kIdNone;
    output_attr.que.depth = af::kIdNone;
    output_attr.que.buf_num = af::kIdNone;
    GELOGI(
        "Hoist invariant Load[%s]: mark loop_axis BlockOuter[%ld] and convert output tensor[%ld] to TBuf, "
        "buf_id = %ld.",
        node->GetNamePtr(), hoistable_node.second, output_attr.mem.tensor_id, output_attr.buf.id);
  }
  // 外提Load前置排序（见ScheduleUtils::SortHoistedLoadsFirst）：保证codegen循环树构建时
  // 外提call先于兄弟深层Loop加入body
  ScheduleUtils::SortHoistedLoadsFirst(graph);
  return af::SUCCESS;
}

Status BufQueAllocator::CollectFusedIoNodes(::ascir::FusedScheduledResult &fused_scheduled_result) {
  fused_scheduled_result.input_nodes.clear();
  fused_scheduled_result.output_nodes.clear();
  fused_scheduled_result.workspace_nodes.clear();
  node_type_to_index_to_node_.clear();
  for (const auto &scheduled_results : fused_scheduled_result.node_idx_to_scheduled_results) {
    for (const auto &result : scheduled_results) {
      for (const auto &schedule_group : result.schedule_groups) {
        for (const auto &impl_graph : schedule_group.impl_graphs) {
          GE_CHK_STATUS_RET(CollectIoNodes(impl_graph), "CollectIoNodes failed, graph = %s",
                            impl_graph.GetName().c_str());
        }
      }
    }
  }
  AppendCollectedIoNodes(fused_scheduled_result);
  return ge::GRAPH_SUCCESS;
}

Status BufQueAllocator::AllocateForIoNodes(const af::AscGraph &impl_graph) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (ScheduleUtils::IsDataInput(node) || IsOps<Output>(node)) {
      int64_t index = -1;
      GE_CHK_STATUS_RET(node->attr.ir_attr->GetAttrValue("index", index), "Get attr index failed, node = %s[%s]",
                        node->GetNamePtr(), node->GetTypePtr());
      auto &index_to_tensor_id = node_type_to_index_to_tensor_id_[node->GetType()];
      const auto it = index_to_tensor_id.find(index);
      int64_t tensor_id;
      if (it != index_to_tensor_id.cend()) {
        tensor_id = it->second;
        GELOGI("same index, cur_node: %s", node->GetName().c_str());
        auto &index_to_node = node_type_to_index_to_node_[node->GetType()];
        auto node_it = index_to_node.find(index);
        if (node_it == index_to_node.end() || ShouldReplaceRepresentative(node_it->second, node)) {
          index_to_node[index] = node;
        }
      } else {
        tensor_id = prev_tensor_id_++;
        index_to_tensor_id[index] = tensor_id;
        node_type_to_index_to_node_[node->GetType()][index] = node;
      }
      if (ScheduleUtils::IsDataInput(node)) {
        SetGlobalMemInfo(node->outputs[0], tensor_id);
      } else {
        if (node->GetInDataNodesSize() != 0UL) {
          SetGlobalMemInfo(node->inputs[0], tensor_id);
        }
        SetGlobalMemInfo(node->outputs[0], tensor_id);
      }
      GELOGI("node: %s[%s] set tensor_id = %ld", node->GetName().c_str(), node->GetType().c_str(), tensor_id);
      continue;
    }
    // workspace 根据name来确定是否是同一块内存提前分配好id
    if (IsOps<Workspace>(node)) {
      int64_t tensor_id;
      const auto it = workspace_name_to_tensor_id_.find(node->GetName());
      if (it != workspace_name_to_tensor_id_.end()) {
        tensor_id = it->second;
      } else {
        tensor_id = prev_tensor_id_++;
        node_type_to_index_to_node_[node->GetType()][static_cast<int64_t>(workspace_name_to_tensor_id_.size())] = node;
        workspace_name_to_tensor_id_[node->GetName()] = tensor_id;
      }
      if (node->GetInDataNodesSize() != 0UL) {
        SetGlobalMemInfo(node->inputs[0], tensor_id);
      }
      SetGlobalMemInfo(node->outputs[0], tensor_id);
      GELOGI("node: %s[%s] set tensor_id = %ld", node->GetName().c_str(), node->GetType().c_str(), tensor_id);
    }
  }
  return ge::GRAPH_SUCCESS;
}

Status BufQueAllocator::AllocateForIoNodes(::ascir::FusedScheduledResult &fused_scheduled_result) {
  for (auto &scheduled_results : fused_scheduled_result.node_idx_to_scheduled_results) {
    for (auto &result : scheduled_results) {
      for (auto &schedule_group : result.schedule_groups) {
        for (auto &impl_graph : schedule_group.impl_graphs) {
          GE_CHK_STATUS_RET(AllocateForIoNodes(impl_graph), "AllocateForIoNodes failed, graph = %s",
                            impl_graph.GetName().c_str());
        }
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

bool BufQueAllocator::ShouldReplaceRepresentative(const ascir::NodeView &current, const ascir::NodeView &candidate) {
  if (current == nullptr) {
    return true;
  }
  if (candidate == nullptr) {
    return false;
  }
  return candidate->GetName().size() < current->GetName().size();
}

Status BufQueAllocator::CollectIoNodes(const af::AscGraph &impl_graph) {
  for (const auto &node : impl_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (ScheduleUtils::IsDataInput(node) || IsOps<Output>(node)) {
      int64_t index = -1;
      GE_CHK_STATUS_RET(node->attr.ir_attr->GetAttrValue("index", index), "Get attr index failed, node = %s[%s]",
                        node->GetNamePtr(), node->GetTypePtr());
      auto &index_to_node = node_type_to_index_to_node_[node->GetType()];
      const auto iter = index_to_node.find(index);
      if (iter == index_to_node.end() || ShouldReplaceRepresentative(iter->second, node)) {
        index_to_node[index] = node;
      }
      continue;
    }
    if (IsOps<Workspace>(node)) {
      const auto iter = workspace_name_to_tensor_id_.find(node->GetName());
      GE_ASSERT_TRUE(iter != workspace_name_to_tensor_id_.end(), "Workspace tensor id is not allocated, node = %s",
                     node->GetNamePtr());
      node_type_to_index_to_node_[node->GetType()][iter->second] = node;
    }
  }
  return ge::GRAPH_SUCCESS;
}

void BufQueAllocator::AppendCollectedIoNodes(::ascir::FusedScheduledResult &fused_scheduled_result) const {
  std::multimap<int64_t, ::ascir::NodeView> index_to_input_node;
  for (const auto &type : {Data::Type, af::ascir_op::ScalarData::Type}) {
    const auto iter = node_type_to_index_to_node_.find(type);
    if (iter == node_type_to_index_to_node_.cend()) {
      continue;
    }
    for (const auto &index_and_node : iter->second) {
      index_to_input_node.emplace(index_and_node.first, index_and_node.second);
    }
  }
  for (const auto &index_and_node : index_to_input_node) {
    fused_scheduled_result.input_nodes.emplace_back(index_and_node.second);
  }
  const auto output_iter = node_type_to_index_to_node_.find(Output::Type);
  if (output_iter != node_type_to_index_to_node_.cend()) {
    for (const auto &index_and_node : output_iter->second) {
      fused_scheduled_result.output_nodes.emplace_back(index_and_node.second);
    }
  }
  const auto workspace_iter = node_type_to_index_to_node_.find(Workspace::Type);
  if (workspace_iter != node_type_to_index_to_node_.cend()) {
    for (const auto &index_and_node : workspace_iter->second) {
      fused_scheduled_result.workspace_nodes.emplace_back(index_and_node.second);
    }
  }
}

Status BufQueAllocator::SetOutputTensorAttr(const af::AscGraph &impl_graph) const {
  auto tensor_id = prev_tensor_id_;
  for (const auto &node : impl_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    // 前面在IO分配时已经分配过了
    static std::set<std::string> allocated_types = {Data::Type, ScalarData::Type, Workspace::Type, Store::Type,
                                                    Output::Type};
    if (allocated_types.count(node->GetType()) > 0UL) {
      continue;
    }

    if (IsOps<Scalar>(node) || IsOps<IndexExpr>(node)) {
      node->outputs[0].attr.mem.tensor_id = tensor_id++;
      continue;
    }
    GE_CHK_STATUS_RET(GetAndSetNodeTempBuffer(node), "Get and set node temp buffers failed.");

    for (auto output : node->outputs()) {
      output->attr.mem.tensor_id = tensor_id++;
      output->attr.mem.hardware = af::MemHardware::kMemHardwareUB;
      // Currently, we don't support inplace/merge optimization
      output->attr.opt.ref_tensor = af::kIdNone;
      output->attr.opt.merge_scope = af::kIdNone;

      // if tensor is used by other unit, use tque
      const bool output_use_by_other_unit = IsTensorUsedByOtherUnit(node, output);
      if (output_use_by_other_unit) {
        output->attr.mem.alloc_type = af::AllocType::kAllocTypeQueue;
        output->attr.buf.id = af::kIdNone;
        if (node->attr.api.unit == af::ComputeUnit::kUnitMTE2) {
          output->attr.mem.position = af::Position::kPositionVecIn;
        } else {
          output->attr.mem.position = af::Position::kPositionVecOut;
        }
      } else {
        output->attr.mem.alloc_type = af::AllocType::kAllocTypeBuffer;
        output->attr.que = {.id = af::kIdNone, .depth = 1, .buf_num = 1};
        if (node->attr.api.unit == af::ComputeUnit::kUnitVector) {
          output->attr.mem.position = af::Position::kPositionVecCalc;
        }
      }
    }
  }

  // VF子图中的常量Scalar副本在分区时拷贝了未分配的tensor_id, 此处按命名映射回填根图分配结果。
  constexpr std::string_view kSubgraphScalarPrefix = "Scalar_";
  std::vector<af::AscGraph> sub_graphs;
  GE_ASSERT_GRAPH_SUCCESS(impl_graph.GetAllSubGraphs(sub_graphs));
  for (auto &sub_graph : sub_graphs) {
    for (const auto &node : sub_graph.GetAllNodes()) {
      if (!IsOps<Scalar>(node) || node->GetName().size() <= kSubgraphScalarPrefix.size()) {
        continue;
      }
      const auto root_node = impl_graph.FindNode(node->GetName().substr(kSubgraphScalarPrefix.size()).c_str());
      if ((root_node != nullptr) && (IsOps<Scalar>(root_node) || IsOps<IndexExpr>(root_node))) {
        node->outputs[0].attr.mem.tensor_id = root_node->outputs[0].attr.mem.tensor_id;
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

Status BufQueAllocator::GetAndSetNodeTempBuffer(const af::AscNodePtr &node) {
  if (ascgen_utils::indirect_load::GetTemplateBehavior(node).skips_api_emit) {
    node->attr.tmp_buffers.clear();
    return af::SUCCESS;
  }
  auto impl = ascgen_utils::GetAscIrCodegenImpl(node->GetType());
  GE_ASSERT_NOTNULL(impl, "GetAscIrCodegenImpl of node %s[%s] is null", node->GetTypePtr(), node->GetNamePtr());
  std::vector<std::unique_ptr<af::TmpBufDesc>> buffers =
      impl->CalcTmpBufSize(*node);  // 新注册方式获取路径，在impl里注册
  GE_LOGI_IF(buffers.empty(), "Node(%s/%s) temporary buffers are empty.", node->GetTypePtr(), node->GetNamePtr());
  node->attr.tmp_buffers.clear();
  for (auto &buf_desc : buffers) {
    if (buf_desc != nullptr) {
      GELOGD("Node(%s/%s) temp buffer size=%s, axis=%ld", node->GetTypePtr(), node->GetNamePtr(),
             buf_desc->size.Str().get(), buf_desc->life_time_axis_id);
      af::TmpBuffer temp_buffer;
      temp_buffer.buf_desc = std::move(*buf_desc);
      node->attr.tmp_buffers.emplace_back(std::move(temp_buffer));
    }
  }
  return af::SUCCESS;
}

bool BufQueAllocator::IsTensorUsedByOtherUnit(const af::AscNodePtr &node, const af::AscTensor *output) {
  if (IsOps<Gather>(node)) {
    return true;
  }
  if (ScheduleUtils::IsLoad(node) && !IsOps<IndirectLoad>(node)) {
    return true;
  }
  for (const auto &input : output->anchor.GetPeerInDataAnchorsPtr()) {
    GE_ASSERT_NOTNULL(input);
    auto peer_node = std::dynamic_pointer_cast<af::AscNode>(input->GetOwnerNode());
    GE_ASSERT_NOTNULL(peer_node);
    if (node->attr.api.unit != peer_node->attr.api.unit) {
      return true;
    }
  }
  return false;
}

void BufQueAllocator::SetGlobalMemInfo(const af::AscTensor &tensor, int64_t tensor_id) {
  tensor.attr.mem.tensor_id = tensor_id;
  tensor.attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
  tensor.attr.mem.hardware = af::MemHardware::kMemHardwareGM;
  tensor.attr.mem.position = af::Position::kPositionGM;
  tensor.attr.buf.id = af::kIdNone;
  tensor.attr.que.id = af::kIdNone;
}

void BufQueAllocator::InitTensorReuseInfoAndLifeTime(const ascir::NodeView &node, const af::AscTensor *output,
                                                     TensorInfo &tensor_info, bool is_reduce_mem_reuse,
                                                     bool is_cube_none_db) const {
  bool is_node_cached = ascgen_utils::IsNodeCacheable(node);
  InitTensorReuseInfo(node, output, tensor_info, is_reduce_mem_reuse, is_node_cached);
  InitTensorLifeTime(node, output, tensor_info, is_node_cached, is_cube_none_db);
}

void BufQueAllocator::InitTensorReuseInfo(const ascir::NodeView &node, const af::AscTensor *output,
                                          TensorInfo &tensor_info, bool is_reduce_mem_reuse,
                                          bool is_node_cached) const {
  if (output->attr.mem.position == af::Position::kPositionVecCalc &&
      ascgen_utils::IsScalarInput(output->attr.repeats)) {
    tensor_info.is_reusable = false;
  }
  if (node->GetName().find("Cube_Load_") != string::npos && cube_type == ascir::CubeTemplateType::kUBFuse) {
    tensor_info.is_reusable = false;
    tensor_info.is_can_reuse_others = false;
  }
  // Reduce节点是VecOut的时候，其不可复用其他空间
  if (ScheduleUtils::IsReduce(node) && output->attr.mem.position == af::Position::kPositionVecOut) {
    tensor_info.is_can_reuse_others = false;
  }
  if (!is_reduce_mem_reuse) {
    tensor_info.is_reusable = false;
  }
  std::vector<int64_t> no_reuse_output_indices;
  (void)af::AttrUtils::GetListInt(node->GetOpDesc(), kAttrNameNoReuseOutputIndices, no_reuse_output_indices);
  if (std::find(no_reuse_output_indices.cbegin(), no_reuse_output_indices.cend(), output->anchor.GetIdx()) !=
      no_reuse_output_indices.cend()) {
    tensor_info.is_reusable = false;
    tensor_info.is_can_reuse_others = false;
  }
  if (is_node_cached) {
    const auto &next_in_anchors = output->anchor.GetPeerInDataAnchorsPtr();
    for (auto &next_in_anchor : next_in_anchors) {
      if (next_in_anchor->GetOwnerNode() != nullptr && !ascgen_utils::IsNodeCacheable(next_in_anchor->GetOwnerNode())) {
        tensor_info.is_reusable = false;
        tensor_info.is_can_reuse_others = false;
      }
    }
  }
}

void BufQueAllocator::InitTensorLifeTime(const ascir::NodeView &node, const af::AscTensor *output,
                                         TensorInfo &tensor_info, bool is_node_cached, bool is_cube_none_db) {
  tensor_info.life_start = node->GetOpDescBarePtr()->GetId();
  tensor_info.life_end = node->GetOpDescBarePtr()->GetId();
  if (tensor_info.is_reusable) {
    const auto &next_in_anchors = output->anchor.GetPeerInDataAnchorsPtr();
    for (auto &next_in_anchor : next_in_anchors) {
      auto out_node = next_in_anchor->GetOwnerNodeBarePtr();
      if (out_node != nullptr) {
        tensor_info.life_end = std::max(tensor_info.life_end, out_node->GetOpDescBarePtr()->GetId());
      }
    }
  } else {
    tensor_info.life_end = std::numeric_limits<int64_t>::max();
  }
  // 刷新db信息
  if (is_cube_none_db) {
    tensor_info.buf_num = 1;
    return;
  }
  if (output->attr.mem.position == af::Position::kPositionVecIn) {
    tensor_info.buf_num = is_node_cached ? 1 : kDbBufNum;
  } else if (output->attr.mem.position == af::Position::kPositionVecOut) {
    tensor_info.buf_num = (is_node_cached && !tensor_info.is_reusable) ? 1 : kDbBufNum;
  } else {
    tensor_info.buf_num = 1;
  }
}

Status BufQueAllocator::InitTensorMemInfo(af::AscGraph &graph, const af::AscTensor *output, TensorInfo &tensor_info) {
  tensor_info.mem_position = output->attr.mem.position;
  auto &repeats = output->attr.repeats;
  auto &axis = output->attr.axis;
  GE_ASSERT_EQ(repeats.size(), axis.size());

  auto &vectorized_axis = output->attr.vectorized_axis;
  bool is_scalar{false};
  for (auto axis_id : vectorized_axis) {
    auto graph_axis = graph.FindAxis(axis_id);
    GE_ASSERT_NOTNULL(graph_axis);

    auto axis_tensor_iter = std::find(axis.begin(), axis.end(), axis_id);
    GE_ASSERT_TRUE(axis_tensor_iter != axis.end(), "Cannot find vectorized axis [%ld]", axis_id);

    const int64_t axis_index = std::distance(axis.begin(), axis_tensor_iter);
    const auto &repeat = repeats[axis_index];
    if (af::SymbolicUtils::StaticCheckEq(repeat, graph_axis->size) == af::TriBool::kTrue) {
      continue;
    }
    if (af::SymbolicUtils::StaticCheckEq(repeat, af::sym::kSymbolOne) != af::TriBool::kTrue) {
      tensor_info.size_level = MemorySizeLevel::kMedium;
      return af::SUCCESS;
    }
    is_scalar = true;
  }
  tensor_info.size_level = is_scalar ? MemorySizeLevel::kScalar : MemorySizeLevel::kLargest;

  return af::SUCCESS;
}

Status BufQueAllocator::InitTensorInfo(af::AscGraph &graph, TensorInfoMap &tensor_attr_to_tensor_info,
                                       bool is_reduce_mem_reuse) const {
  bool is_reduce_after = false;
  bool is_cube_none_db = false;
  if (graph.GetName().find("non_db") != std::string::npos) {
    is_cube_none_db = true;
  }
  for (const auto &node : graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node) || ScheduleUtils::IsStore(node)) {
      continue;
    }
    if (ScheduleUtils::IsReduce(node)) {
      is_reduce_after = true;
    }
    for (const auto &output : node->outputs()) {
      auto &tensor_info = tensor_attr_to_tensor_info[&output->attr];
      tensor_info.output_tensor_attr = &output->attr;
      tensor_info.loop_axes.emplace(node->attr.sched.loop_axis);
      for (auto &peer_in_anchor : output->anchor.GetPeerInDataAnchors()) {
        if (peer_in_anchor == nullptr) {
          continue;
        }
        auto out_asc_node = dynamic_cast<af::AscNode *>(peer_in_anchor->GetOwnerNodeBarePtr());
        GE_ASSERT_NOTNULL(out_asc_node);
        tensor_info.loop_axes.emplace(out_asc_node->attr.sched.loop_axis);
      }
      InitTensorReuseInfoAndLifeTime(node, output, tensor_info, !is_reduce_after || is_reduce_mem_reuse,
                                     is_cube_none_db);
      GE_ASSERT_SUCCESS(InitTensorMemInfo(graph, output, tensor_info), "Failed to init tensor info for graph [%s].",
                        graph.GetName().c_str());
      GELOGD("[MemReuse] Init node [%s]'s output tensor[%d] [%s].", node->GetNamePtr(), output->anchor.GetIdx(),
             tensor_info.ToString().c_str());
    }
  }
  return af::SUCCESS;
}

Status BufQueAllocator::InitNodeTmpBuffInfo(af::AscGraph &graph, TmpBuffInfoMap &node_attr_to_tensor_info) {
  int64_t allocation_order = 0;
  for (const auto &node : graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    for (auto &tmp_buff : node->attr.tmp_buffers) {
      auto &tmp_buff_info = node_attr_to_tensor_info[&tmp_buff];
      tmp_buff_info.allocation_order = allocation_order++;
      tmp_buff_info.mem_position = af::Position::kPositionVecCalc;
      tmp_buff_info.life_start = 0L;
      tmp_buff_info.life_end = std::numeric_limits<int64_t>::max();
      tmp_buff_info.group_id = tmp_buff.buf_desc.life_time_axis_id;
      if (tmp_buff_info.group_id == -1) {
        tmp_buff_info.life_start = node->GetOpDescBarePtr()->GetId();
        tmp_buff_info.life_end = node->GetOpDescBarePtr()->GetId();
      }
    }
  }
  return af::SUCCESS;
}

// reuse id 和que id 是独立编码，分reuse id时只区分复用和共用, 不考虑是否在一个que内。reuse id相同表示共用，否则表示复用
void BufQueAllocator::AllocateReuseId(const af::AscGraph &graph, TensorInfoMap &tensor_attr_to_tensor_info) {
  int64_t reuse_id = 0;
  std::map<int64_t, int64_t> out_id_to_reuse_id;
  for (const auto &node : graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    for (const auto &output : node->outputs()) {
      if (ScheduleUtils::IsStore(node)) {
        continue;
      }
      auto &tensor_info = tensor_attr_to_tensor_info[&output->attr];
      if (output->attr.mem.position == af::Position::kPositionVecIn) {
        auto output_nodes = node->GetOutNodes();
        if ((output_nodes.size() != 1UL) || !tensor_info.is_reusable) {
          output->attr.mem.reuse_id = reuse_id++;
          continue;
        }
        auto iter = out_id_to_reuse_id.find(output_nodes.at(0)->GetOpDescBarePtr()->GetId());
        if (iter != out_id_to_reuse_id.end()) {
          output->attr.mem.reuse_id = iter->second;
        } else {
          out_id_to_reuse_id[output_nodes.at(0)->GetOpDescBarePtr()->GetId()] = reuse_id;
          output->attr.mem.reuse_id = reuse_id++;
        }
      } else {
        output->attr.mem.reuse_id = reuse_id++;
      }
    }
  }
}

TensorInfo *BufQueAllocator::FindBestInplaceSource(const af::AscNodePtr &node, const TensorInfo &output_info,
                                                   TensorInfoMap &tensor_attr_to_tensor_info) {
  TensorInfo *best_source = nullptr;
  int32_t min_distance = std::numeric_limits<int32_t>::max();
  MemorySizeLevel output_size = output_info.size_level;
  for (const auto &in_tensor : node->inputs()) {
    auto iter = tensor_attr_to_tensor_info.find(&in_tensor->attr);
    if (iter == tensor_attr_to_tensor_info.end()) {
      continue;
    }
    auto &input_info = iter->second;
    if (!input_info.is_reusable || input_info.mem_position == af::Position::kPositionVecIn) {
      continue;
    }
    int32_t distance = std::abs(static_cast<int32_t>(input_info.size_level) - static_cast<int32_t>(output_size));
    if (best_source == nullptr) {
      best_source = &input_info;
      min_distance = distance;
    } else {
      if (distance < min_distance) {
        best_source = &input_info;
        min_distance = distance;
      } else if (distance == min_distance) {
        if (input_info.size_level > best_source->size_level) {
          best_source = &input_info;
        }
      }
    }
  }
  return best_source;
}

// 默认用reuse_id表示group_id，inplace复用场景刷成输入tensor的group_id
void BufQueAllocator::InitGroupId(const af::AscGraph &graph, TensorInfoMap &tensor_attr_to_tensor_info) {
  std::map<int64_t, int64_t> out_id_to_reuse_id;
  for (const auto &node : graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node) || ScheduleUtils::IsStore(node)) {
      continue;
    }
    bool is_node_support_inplace = IsSupportInplace(node);
    for (const auto &output : node->outputs()) {
      auto iter = tensor_attr_to_tensor_info.find(&output->attr);
      if (iter == tensor_attr_to_tensor_info.end()) {
        GELOGW("[MemReuse] node[%s]'s output tensor[%d] may not have been properly initialized",
               node->GetName().c_str(), output->anchor.GetIdx());
        continue;
      }
      auto &tensor_info = iter->second;
      tensor_info.group_id = tensor_info.output_tensor_attr->mem.reuse_id;
      if (is_node_support_inplace && tensor_info.is_can_reuse_others) {
        const TensorInfo *best_source = FindBestInplaceSource(node, tensor_info, tensor_attr_to_tensor_info);
        if (best_source != nullptr) {
          tensor_info.group_id = best_source->group_id;
        }
      }
      GELOGD("[MemReuse] Set group id [%ld] for node [%s]'s output tensor[%d].", tensor_info.group_id,
             node->GetName().c_str(), output->anchor.GetIdx());
    }
  }
}

Status BufQueAllocator::AllocateWithinGroup(af::AscGraph &graph, size_t &total_vecin_nums, size_t &total_vecout_nums,
                                            bool is_reduce_mem_reuse) const {
  GE_ASSERT_SUCCESS(SetOutputTensorAttr(graph));
  TmpBuffInfoMap tmp_buff_attr_to_tensor_info;
  TensorInfoMap tensor_attr_to_tensor_info;
  GE_ASSERT_SUCCESS(InitNodeTmpBuffInfo(graph, tmp_buff_attr_to_tensor_info));
  GE_ASSERT_SUCCESS(MarkUnreusableTensors(graph));
  GE_ASSERT_SUCCESS(InitTensorInfo(graph, tensor_attr_to_tensor_info, is_reduce_mem_reuse));
  AllocateReuseId(graph, tensor_attr_to_tensor_info);
  InitGroupId(graph, tensor_attr_to_tensor_info);

  MemReuseManager manager = MemReuseManager(tensor_attr_to_tensor_info, tmp_buff_attr_to_tensor_info);
  manager.AllocMemBlocks();
  manager.GetCopyInCopyOutQueNums(total_vecin_nums, total_vecout_nums);
  GELOGD("[MemReuse] graph[%s] has [%zu] copy in queues and [%zu] copy out queues after mem reuse.",
         graph.GetName().c_str(), total_vecin_nums, total_vecout_nums);
  return af::SUCCESS;
}

Status BufQueAllocator::ShortenVecinLifetime(af::AscGraph &graph, size_t max_que_num) {
  // step1 init lifecycles
  std::vector<NodeLifecycle> lifecycles;
  for (const auto &node : graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (!IsOps<Load>(node)) {
      continue;
    }
    NodeLifecycle lifecycle{node, node->GetOpDescBarePtr()->GetId(), node->GetOpDescBarePtr()->GetId(), 1U};
    for (const auto &out_node : node->GetOutDataNodesPtr()) {
      lifecycle.end = std::max(lifecycle.end, out_node->GetOpDescBarePtr()->GetId());
    }
    GELOGD("Load [%s]'s lifecycle is in [%ld, %ld].", node->GetNamePtr(), lifecycle.start, lifecycle.end);
    lifecycles.emplace_back(lifecycle);
  }

  // // step 2 get overlapping-lifecycle loads with more than 4 que ids.
  std::list<LifecycleSet> all_sets = FindOverlappingNodeSets(lifecycles, max_que_num);

  // step3 insert Ub2Ub after the load that has seen the most nodes.
  while (!all_sets.empty()) {
    LifecycleSet load_set = std::move(all_sets.front());
    all_sets.erase(all_sets.begin());

    GE_ASSERT_TRUE(!load_set.empty());

    auto top_cycle = load_set.begin();
    const std::string ub_name = "ub_cpy_" + top_cycle->node->GetName();
    Ub2ub ub2ub(ub_name.c_str());
    af::AscNodePtr ub2ub_node = graph.AddNode(ub2ub);
    GE_ASSERT_NOTNULL(ub2ub_node);

    auto load_out_anchor = top_cycle->node->GetOutDataAnchor(0);
    GE_ASSERT_NOTNULL(load_out_anchor);
    GE_ASSERT_SUCCESS(af::AscGraphUtils::InsertNodeAfter(load_out_anchor, ub2ub_node));
    ub2ub_node->attr.api.compute_type = af::ComputeType::kComputeElewise;
    ub2ub_node->attr.api.type = af::ApiType::kAPITypeCompute;
    ub2ub_node->attr.api.unit = af::ComputeUnit::kUnitVector;
    ub2ub_node->outputs[0].attr = top_cycle->node->outputs[0].attr;
    ub2ub_node->outputs[0].attr.buf = {};
    ub2ub_node->outputs[0].attr.que = {};
    load_set.erase(top_cycle);

    auto split_lists = FindOverlappingNodeSets({load_set.begin(), load_set.end()}, max_que_num);
    if (!split_lists.empty()) {
      all_sets.insert(all_sets.end(), split_lists.begin(), split_lists.end());
    }
  }
  GE_ASSERT_GRAPH_SUCCESS(TopoSortByLoadPriority(graph), "Failed to do topologic for graph:[%s].",
                          graph.GetName().c_str());

  return af::SUCCESS;
}

Status BufQueAllocator::ShortenVecoutLifetime(af::AscGraph &graph, size_t max_que_num) {
  // step1 init lifecycles
  std::vector<NodeLifecycle> lifecycles;
  for (const auto &node : graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    NodeLifecycle lifecycle{node, std::numeric_limits<int64_t>::max(), node->GetOpDescBarePtr()->GetId(), 1U};
    bool has_vecout = false;
    for (const auto &in_anchor : node->GetAllInDataAnchorsPtr()) {
      GE_ASSERT_NOTNULL(in_anchor);
      auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
      if (peer_out_anchor == nullptr) {
        continue;
      }
      auto peer_node = peer_out_anchor->GetOwnerNodeBarePtr();
      GE_ASSERT_NOTNULL(peer_node);
      for (const auto &out_node : node->GetOutDataNodesPtr()) {
        GE_ASSERT_NOTNULL(out_node);
        if (out_node->GetType() == Store::Type) {
          lifecycle.start = std::min(lifecycle.start, peer_node->GetOpDescBarePtr()->GetId());
        }
      }
    }

    for (const auto &out_node : node->GetOutDataNodesPtr()) {
      GE_ASSERT_NOTNULL(out_node);
      if (out_node->GetType() == Store::Type) {
        has_vecout = true;
      }
      lifecycle.start = std::min(lifecycle.start, out_node->GetOpDescBarePtr()->GetId());
      lifecycle.end = std::max(lifecycle.end, out_node->GetOpDescBarePtr()->GetId());
    }
    if (has_vecout) {
      GELOGD("Vecout [%s]'s lifecycle is in [%ld, %ld].", node->GetNamePtr(), lifecycle.start, lifecycle.end);
      lifecycles.emplace_back(lifecycle);
    }
  }

  // // step 2 get overlapping-lifecycle vecout with more than 4 que ids.
  std::list<LifecycleSet> all_sets = FindOverlappingNodeSets(lifecycles, max_que_num);

  // step3 insert Ub2Ub after the vecout that has seen the most nodes.
  while (!all_sets.empty()) {
    LifecycleSet vecout_set = std::move(all_sets.front());
    all_sets.erase(all_sets.begin());
    GE_ASSERT_TRUE(!vecout_set.empty());

    auto top_cycle = vecout_set.begin();
    size_t idx = 0UL;
    for (const auto &out_data_anchor : top_cycle->node->GetAllOutDataAnchors()) {
      GE_ASSERT_NOTNULL(out_data_anchor);
      for (const auto &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
        GE_ASSERT_NOTNULL(peer_in_anchor);
        auto peer_in_node = peer_in_anchor->GetOwnerNodeBarePtr();
        GE_ASSERT_NOTNULL(peer_in_node);
        if (peer_in_node->GetType() != Store::Type) {
          continue;
        }
        // add ub_2ub
        const std::string ub_name = "ub_cpy_" + top_cycle->node->GetName() + "_" + std::to_string(idx);
        Ub2ub ub2ub(ub_name.c_str());
        af::AscNodePtr ub2ub_node = graph.AddNode(ub2ub);
        GE_ASSERT_NOTNULL(ub2ub_node);

        GE_ASSERT_SUCCESS(af::AscGraphUtils::InsertNodeAfter(out_data_anchor, {peer_in_anchor}, ub2ub_node));
        ub2ub_node->attr.api.compute_type = af::ComputeType::kComputeElewise;
        ub2ub_node->attr.api.type = af::ApiType::kAPITypeCompute;
        ub2ub_node->attr.api.unit = af::ComputeUnit::kUnitVector;
        ub2ub_node->outputs[0].attr = top_cycle->node->outputs[0].attr;
        ub2ub_node->outputs[0].attr.buf = {};
        ub2ub_node->outputs[0].attr.que = {};
        idx++;
      }
    }

    vecout_set.erase(top_cycle);
    auto split_lists = FindOverlappingNodeSets({vecout_set.begin(), vecout_set.end()}, max_que_num);
    if (!split_lists.empty()) {
      all_sets.insert(all_sets.end(), split_lists.begin(), split_lists.end());
    }
  }
  GE_ASSERT_GRAPH_SUCCESS(TopoSortByLoadPriority(graph), "Failed to do topologic for graph:[%s].",
                          graph.GetName().c_str());

  return af::SUCCESS;
}

Status BufQueAllocator::TopoSortByLoadPriority(af::AscGraph &graph) {
  GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(graph, true));
  std::unordered_set<af::Node *> priority_sequences;
  for (const auto &node : graph.GetAllNodes()) {
    if (!ScheduleUtils::IsLoad(node) || node->GetOutDataNodesSize() > 1UL) {
      continue;
    }
    auto load_after = node->GetOutDataNodesPtr()[0];
    GE_ASSERT_NOTNULL(load_after);
    if (load_after->GetInDataNodesSize() == 1UL) {
      priority_sequences.insert(node->inputs[0].anchor.GetOwnerNodeBarePtr());
      priority_sequences.insert(node.get());
      priority_sequences.insert(node->GetOutDataNodesPtr()[0UL]);
    }
  }

  const auto func = [&priority_sequences](const af::NodePtr &node1, const af::NodePtr &node2) -> bool {
    bool is_node1_in_priority_seq = priority_sequences.find(node1.get()) != priority_sequences.end();
    bool is_node2_in_priority_seq = priority_sequences.find(node2.get()) != priority_sequences.end();
    if (is_node1_in_priority_seq && !is_node2_in_priority_seq) {
      return true;
    } else {
      return node1->GetOpDescBarePtr()->GetId() < node2->GetOpDescBarePtr()->GetId();
    }
  };

  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  GE_ASSERT_NOTNULL(compute_graph);
  compute_graph->TopologicalSorting(func);

  return af::SUCCESS;
}

Status BufQueAllocator::ProcessSingleImplGraph(af::AscGraph &impl_graph, BasePlatform &platform, size_t max_que_num,
                                               bool is_reduce_mem_reuse) {
  // UBFuse 上下文通过内部 graph attr 传递给平台分区逻辑, 不扩展 BasePlatform 公共接口。
  GE_ASSERT_TRUE(af::AttrUtils::SetBool(af::AscGraphUtils::GetComputeGraph(impl_graph), kGraphAttrIsUBFuseContext,
                                        cube_type == ascir::CubeTemplateType::kUBFuse),
                 "Failed to set UBFuse context attr for graph %s.", impl_graph.GetName().c_str());
  GE_ASSERT_SUCCESS(platform.PartitionSubFunctions(impl_graph), "Failed to partition vf func for graph %s.",
                    impl_graph.GetName().c_str());
  if (cube_type == ascir::CubeTemplateType::kUBFuse) {
    GE_ASSERT_SUCCESS(TopoSortByCubeLoadPriority(impl_graph), "Failed to topo sort by cube load priority for graph %s.",
                      impl_graph.GetName().c_str());
  } else {
    GE_ASSERT_SUCCESS(ScheduleUtils::TopologicalSorting(impl_graph, true), "Failed to topo sort for graph %s.",
                      impl_graph.GetName().c_str());
  }
  return AllocBufQueForSingleImplGraph(impl_graph, max_que_num, is_reduce_mem_reuse);
}

Status BufQueAllocator::TopoSortByCubeLoadPriority(af::AscGraph &graph) {
  GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  std::unordered_set<af::Node *> priority_sequences;
  // 收集 Cube_Load 节点及其直连输出消费者
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetName().find("Cube_Load_") == std::string::npos) {
      continue;
    }
    for (const auto &out_node : node->GetOutNodes()) {
      GE_ASSERT_NOTNULL(out_node);
      if (IsOps<Store>(out_node)) {
        continue;
      }
      priority_sequences.insert(out_node.get());
    }
  }
  if (priority_sequences.empty()) {
    return af::SUCCESS;
  }
  // 向上 BFS 回溯所有输入依赖
  std::unordered_set<af::Node *> visited;
  std::queue<af::Node *> bfs_queue;
  for (auto *n : priority_sequences) {
    bfs_queue.push(n);
  }
  while (!bfs_queue.empty()) {
    auto *current = bfs_queue.front();
    bfs_queue.pop();
    for (const auto &in_node : current->GetInDataNodes()) {
      GE_ASSERT_NOTNULL(in_node);
      if (visited.insert(in_node.get()).second) {
        priority_sequences.insert(in_node.get());
        bfs_queue.push(in_node.get());
      }
    }
  }

  const auto func = [&priority_sequences](const af::NodePtr &node1, const af::NodePtr &node2) -> bool {
    bool is_node1_in = priority_sequences.count(node1.get()) > 0;
    bool is_node2_in = priority_sequences.count(node2.get()) > 0;
    if (is_node1_in && !is_node2_in) {
      return true;
    }
    return node1->GetOpDescBarePtr()->GetId() < node2->GetOpDescBarePtr()->GetId();
  };

  auto compute_graph = af::AscGraphUtils::GetComputeGraph(graph);
  GE_ASSERT_NOTNULL(compute_graph);
  compute_graph->TopologicalSorting(func);

  return af::SUCCESS;
}

Status BufQueAllocator::MarkUnreusableTensors(const ge::AscGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    bool input_is_unreusable = false;
    (void)af::AttrUtils::GetBool(node->GetOpDesc(), kAttrNameNoReuseInputs, input_is_unreusable);
    if (!input_is_unreusable) {
      continue;
    }
    GELOGD("node: %s, input is unreusable", node->GetName().c_str());
    std::map<af::NodePtr, std::vector<int64_t>> node_to_indices;
    for (const auto &[in_node, out_anchor] : node->GetInDataNodesAndAnchors()) {
      node_to_indices[in_node].push_back(out_anchor->GetIdx());
    }
    for (const auto &[in_node, indices] : node_to_indices) {
      (void)af::AttrUtils::SetListInt(in_node->GetOpDesc(), kAttrNameNoReuseOutputIndices, indices);
      GELOGD("mark unreusable output indices, node = %s, indices = %s", in_node->GetName().c_str(),
             af::ToString(indices).c_str());
    }
  }
  return af::SUCCESS;
}
}  // namespace optimize
