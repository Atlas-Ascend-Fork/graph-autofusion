/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_kernel_loop.h"
#include "codegen_kernel.h"
#include "codegen_api_param/codegen_api_param.h"
#include "codegen/expression_convert_struct.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "ascir_utils.h"
#include "optimize/platform/platform_factory.h"
#include "api_call/utils/api_call_factory.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "indirect_load_utils.h"

using namespace std;
using namespace af::ops;
using namespace codegen;
using namespace af::ascir_op;
using namespace ascgen_utils;

namespace {
constexpr size_t kDoubleAxisSize = 2U;
constexpr size_t kDoubleTileAxisSize = 2;
const std::string kEnCacheOriginBroadcastAxis = "enable_cache_origin_brc_axis";
const std::string kEnCacheFusedBroadcastAxis = "enable_cache_fused_brc_axis";
const std::string kEnCacheA = "dis_enable_cache_a";
const std::string kEnCacheR = "dis_enable_cache_r";

struct ConstructState {
  std::map<ascir::TensorId, ApiCall *> tensor_calls;
  std::map<ascir::BufId, ApiTensor *> buf_last_use;
  std::map<ascir::QueId, ApiTensor *> que_last_use;
  std::map<ascir::QueId, std::map<ascir::ReuseId, ApiTensor *>> que_last_share;
};

ApiCall *FindIndirectLoadOpCall(Loop *loop, bool require_direct_gm) {
  if (loop == nullptr) {
    return nullptr;
  }
  for (const auto &body : loop->bodys) {
    if (body.type == LoopType::CALL) {
      if (body.call == nullptr || !af::ops::IsOps<af::ascir_op::IndirectLoad>(body.call->node)) {
        continue;
      }
      const auto behavior =
          ascgen_utils::indirect_load::GetTemplateBehavior(std::dynamic_pointer_cast<af::AscNode>(body.call->node));
      if (!require_direct_gm || (behavior.uses_direct_gm_pipeline && behavior.skips_ub_lifecycle)) {
        return body.call;
      }
    } else if (ApiCall *call = FindIndirectLoadOpCall(body.loop, require_direct_gm); call != nullptr) {
      return call;
    }
  }
  return nullptr;
}

bool HasRedirectedIndirectLoadOutput(const ApiCall *call) {
  if (call == nullptr || !IsOps<IndirectLoad>(call->node) || call->outputs.size() != 1UL ||
      call->node->outputs().size() != 1UL) {
    return false;
  }
  return call->outputs[0].id != call->node->outputs()[0]->attr.mem.tensor_id;
}

Status ResolveSkippedOutputCall(const ascir::NodeView &node, const std::map<ascir::TensorId, ApiCall *> &tensor_calls,
                                ApiCall *call, ApiCall *&output_call) {
  output_call = call;
  if (IsOps<Store>(node) || node->outputs().size() != 1UL) {
    return af::SUCCESS;
  }
  ApiCall *common_input_call = nullptr;
  for (const auto input : node->inputs()) {
    if (input == nullptr) {
      return af::SUCCESS;
    }
    const auto input_call = tensor_calls.find(input->attr.mem.tensor_id);
    GE_CHK_BOOL_RET_STATUS(input_call != tensor_calls.end(), af::FAILED,
                           "Codegen node[%s] no API call found for skipped input tensor id[%ld]", node->GetNamePtr(),
                           input->attr.mem.tensor_id);
    if (common_input_call == nullptr) {
      common_input_call = input_call->second;
    } else if (common_input_call != input_call->second) {
      return af::SUCCESS;
    }
  }
  if (common_input_call == nullptr) {
    return af::SUCCESS;
  }
  if (common_input_call->node != nullptr && IsOps<IndirectLoad>(common_input_call->node)) {
    const auto result_tensor =
        ascgen_utils::indirect_load::FindSkippedChainResultTensor(std::dynamic_pointer_cast<af::AscNode>(node));
    if (result_tensor != af::kIdNone && !call->outputs.empty()) {
      call->outputs[0].id = result_tensor;
    }
  } else if (common_input_call->outputs.size() == 1UL) {
    output_call = common_input_call;
  }
  return af::SUCCESS;
}

Status AddSkippedApiEmitProcessCall(const ascir::NodeView &node, Loop *current_loop,
                                    const std::vector<ascir::AxisId> &current_axis,
                                    std::map<ascir::TensorId, ApiCall *> &tensor_calls) {
  auto call = CreateApiCallObject(node);
  GE_ASSERT_NOTNULL(call, "Create api call object failed, ascir type:%s", node->GetTypePtr());
  current_loop->AddCall(call);
  GE_CHK_STATUS_RET(call->Init(node), "ApiCall Init failed, ascir type:%s", node->GetTypePtr());
  call->skip_api_emit = true;
  call->exec_condition = node->attr.sched.exec_condition;
  call->axis = current_loop->axis_id;
  call->depth = current_axis.size();
  ApiCall *output_call = nullptr;
  GE_ASSERT_SUCCESS(ResolveSkippedOutputCall(node, tensor_calls, call, output_call));
  for (auto out : node->outputs()) {
    tensor_calls.insert({out->attr.mem.tensor_id, output_call});
  }
  return af::SUCCESS;
}
std::string GetQueueSliceByteSize(const TPipe &tpipe, const Tensor &tensor) {
  auto size = af::GetSizeByDataType(tensor.dtype);
  std::string byte_size = tensor.size.name + " * " + std::to_string(size);
  if (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
    byte_size = "KernelUtils::BlkAlign<uint8_t>(" + byte_size + ")";
  }
  return byte_size;
}

Status MoveToNodeLoop(const ascir::NodeView &node, std::vector<ascir::AxisId> &current_axis, Loop *&current_loop) {
  if (node->attr.api.unit == af::ComputeUnit::kUnitNone) {
    return af::SUCCESS;
  }
  const auto &node_axis = node->attr.sched.axis;
  const auto &node_loop_axis = node->attr.sched.loop_axis;
  int32_t loop_distance = 0;
  GE_CHK_STATUS_RET(LoopAxisDistance(current_axis, node_axis, node_loop_axis, loop_distance),
                    "Codegen get loop axis distance failed");
  while (loop_distance != 0) {
    if (loop_distance > 0) {
      const auto axis = node_axis[current_axis.size()];
      current_axis.push_back(axis);
      current_loop->AddLoop(new Loop(axis));
      current_loop = current_loop->bodys.back().loop;
    } else {
      current_axis.pop_back();
      current_loop = current_loop->parent;
    }
    GE_CHK_STATUS_RET(LoopAxisDistance(current_axis, node_axis, node_loop_axis, loop_distance),
                      "Codegen get loop axis distance failed");
  }
  return af::SUCCESS;
}

bool IsShareInputs(const af::AscNodePtr &node) {
  std::set<int64_t> queue_ids;
  for (uint32_t i = 0; i < node->inputs.Size(); ++i) {
    queue_ids.emplace(node->inputs[i].attr.que.id);
  }
  return queue_ids.size() == 1UL;
}

Status RequireContiguousInputBufs(const af::AscNodePtr &node, ApiCall &api_call, TPipe &tpipe) {
  GE_CHK_BOOL_RET_SPECIAL_STATUS((!ascir::utils::AreAllInputDistinct(node)), af::SUCCESS,
                                 "%s cannot require contiguous inputs, contain multi-ref input", node->GetNamePtr());
  if (ascir::utils::AreAllInputsFromPosition(node, af::Position::kPositionVecIn)) {
    GE_CHK_BOOL_RET_SPECIAL_STATUS((!IsShareInputs(node)), af::SUCCESS,
                                   "%s(%s) cannot require contiguous inputs, not sharing single input TQue",
                                   node->GetNamePtr(), api_call.api_name_.c_str());
    for (size_t i = 0; i < api_call.inputs.size(); ++i) {
      auto &in_tensor = api_call.inputs[i];
      in_tensor->share_order = static_cast<int32_t>(i);
    }
    api_call.is_input_tbuf_contiguous = true;
    return af::SUCCESS;
  }

  GE_CHK_BOOL_RET_SPECIAL_STATUS((!ascir::utils::AreAllInputsFromPosition(node, af::Position::kPositionVecCalc)),
                                 af::SUCCESS,
                                 "%s(%s) cannot require contiguous inputs, inputs come from multiple position",
                                 node->GetNamePtr(), api_call.api_name_.c_str());
  std::vector<ascir::BufId> input_buf_ids;
  for (const auto &input : api_call.inputs) {
    const auto input_tensor = tpipe.GetTensor(input->id);
    GE_ASSERT_NOTNULL(input_tensor);
    GE_CHK_BOOL_RET_SPECIAL_STATUS(input_tensor->alloc_type != af::AllocType::kAllocTypeBuffer, af::SUCCESS,
                                   "%s(%s) cannot require contiguous TBufs, input contains non-TBuf",
                                   node->GetNamePtr(), api_call.api_name_.c_str());
    const auto &buf = tpipe.GetBuf(input_tensor->buf_id);
    const auto ref_count = buf.merge_scopes.size() + buf.not_merge_tensors.size() + buf.tmp_buf_size_list.size();
    GE_CHK_BOOL_RET_SPECIAL_STATUS(
        (ref_count > 1UL), af::SUCCESS,
        "%s(%s) cannot require contiguous TBufs, input buf is reused with other tensors, buf_id = %ld",
        node->GetNamePtr(), api_call.api_name_.c_str(), input_tensor->buf_id);
    input_buf_ids.emplace_back(input_tensor->buf_id);
  }
  GE_CHK_BOOL_RET_SPECIAL_STATUS((!tpipe.contiguous_buf_ids.empty()), af::SUCCESS,
                                 "%s(%s) cannot require contiguous tbufs, already required by other ApiCall",
                                 node->GetNamePtr(), api_call.api_name_.c_str());
  tpipe.contiguous_buf_ids = std::move(input_buf_ids);
  api_call.is_input_tbuf_contiguous = true;
  GELOGD("%s(%s) can require contiguous input TBuf, buf list = %s", node->GetNamePtr(), api_call.api_name_.c_str(),
         af::ToString(tpipe.contiguous_buf_ids).c_str());
  return af::SUCCESS;
}

Status ConnectApiCallInputs(const ascir::NodeView &node, const std::map<ascir::TensorId, ApiCall *> &tensor_calls,
                            ApiCall &call, TPipe &tpipe) {
  const bool contiguous_bufs_preferred = call.AreContiguousBufsPreferred();
  for (const auto input : node->inputs()) {
    if (input == nullptr) {
      call.inputs.emplace_back(nullptr);
      continue;
    }
    const auto input_call = tensor_calls.find(input->attr.mem.tensor_id);
    GE_CHK_BOOL_RET_STATUS(input_call != tensor_calls.end(), af::FAILED,
                           "Codegen node[%s] no API call found for input tensor id[%ld]", node->GetNamePtr(),
                           input->attr.mem.tensor_id);
    const auto input_index = af::ascir::AscTensorUtils::Index(*input);
    ApiTensor *input_tensor = &input_call->second->outputs[input_index];
    input_tensor->reads.push_back(&call);
    call.inputs.emplace_back(input_tensor);
    GELOGI("node[%s] input tensor id[%ld] from call type[%s] outputs[%d], read by call type[%s]", node->GetNamePtr(),
           input->attr.mem.tensor_id, input_call->second->type.c_str(), input_index, call.type.c_str());
  }
  if (contiguous_bufs_preferred) {
    GE_ASSERT_SUCCESS(RequireContiguousInputBufs(node, call, tpipe));
  }
  return af::SUCCESS;
}

Status UpdateQueueOutputReuse(const af::AscTensor *output, size_t output_index, ApiCall &call, TPipe &tpipe,
                              ConstructState &state) {
  const auto &attr = output->attr;
  GELOGI("Que[%ld] update last use call type[%s] output[%zu]", attr.que.id, call.type.c_str(), output_index);
  GE_CHK_BOOL_RET_STATUS(attr.que.id != af::kIdNone && attr.mem.reuse_id != af::kIdNone, af::FAILED,
                         "ConstructFromNodes tensor[%ld] que id[%ld] or reuse id[%ld] invalid",
                         call.outputs[output_index].id, attr.que.id, attr.mem.reuse_id);
  auto &last_share = state.que_last_share[attr.que.id];
  const auto share_tensor = last_share.find(attr.mem.reuse_id);
  if (share_tensor != last_share.end()) {
    Tensor *tensor = tpipe.GetTensor(attr.mem.tensor_id);
    Tensor *previous_tensor = tpipe.GetTensor(share_tensor->second->id);
    GE_ASSERT_NOTNULL(tensor, "Queue output tensor[%ld] is missing.", attr.mem.tensor_id);
    GE_ASSERT_NOTNULL(previous_tensor, "Queue shared tensor[%ld] is missing.", share_tensor->second->id);
    tensor->share_pre_size = previous_tensor->size.name;
    share_tensor->second->share_next = &call.outputs[output_index];
    call.outputs[output_index].share_prev = share_tensor->second;
    GELOGI("Que[%ld] reuse id[%ld] tensor id[%ld] share with id[%ld]", attr.que.id, attr.mem.reuse_id,
           call.outputs[output_index].id, share_tensor->second->id);
  }
  last_share[attr.mem.reuse_id] = &call.outputs[output_index];
  const auto reused_tensor = state.que_last_use.find(attr.que.id);
  if (reused_tensor != state.que_last_use.end() &&
      reused_tensor->second->reuse_id != call.outputs[output_index].reuse_id) {
    reused_tensor->second->reuse_next = &call.outputs[output_index];
    call.outputs[output_index].reuse_from = reused_tensor->second;
    GELOGI("Que[%ld] reuse id[%ld] tensor id[%ld] reuse from tensor id[%ld] reuse id[%ld]", attr.que.id,
           attr.mem.reuse_id, call.outputs[output_index].id, reused_tensor->second->id,
           reused_tensor->second->reuse_id);
  } else if (reused_tensor != state.que_last_use.end()) {
    GELOGI("Que[%ld] reuse id[%ld] tensor id[%ld] share with last same que tensor", attr.que.id, attr.mem.reuse_id,
           call.outputs[output_index].id, reused_tensor->second->id);
  }
  state.que_last_use[attr.que.id] = &call.outputs[output_index];
  return af::SUCCESS;
}

void UpdateBufferOutputReuse(const af::AscTensor *output, size_t output_index, ApiCall &call, ConstructState &state) {
  const auto buffer_id = output->attr.buf.id;
  GELOGI("Buf[%ld] update last use call type[%s] output[%zu]", buffer_id, call.type.c_str(), output_index);
  const auto reused_tensor = state.buf_last_use.find(buffer_id);
  if (reused_tensor != state.buf_last_use.end()) {
    reused_tensor->second->reuse_next = &call.outputs[output_index];
    call.outputs[output_index].reuse_from = reused_tensor->second;
    GELOGI("Buf[%ld] tensor id[%ld] reuse from tensor id[%ld]", buffer_id, call.outputs[output_index].id,
           reused_tensor->second->id);
  }
  state.buf_last_use[buffer_id] = &call.outputs[output_index];
}

Status RegisterApiCallOutputs(const ascir::NodeView &node, ApiCall &call, TPipe &tpipe, ConstructState &state) {
  for (const auto output : node->outputs()) {
    state.tensor_calls.insert({output->attr.mem.tensor_id, &call});
    const size_t output_index = af::ascir::AscTensorUtils::Index(*output);
    if (IsOps<IndirectLoad>(node) && HasRedirectedIndirectLoadOutput(&call)) {
      Tensor *output_tensor = tpipe.GetTensor(call.outputs[output_index].id);
      GE_ASSERT_NOTNULL(output_tensor, "Codegen output tensor[%ld] is missing.", call.outputs[output_index].id);
      call.outputs[output_index].reuse_id = output_tensor->reuse_id;
      state.tensor_calls.insert({output_tensor->id, &call});
      continue;
    }
    if (output->attr.mem.alloc_type == af::AllocType::kAllocTypeQueue) {
      GE_ASSERT_SUCCESS(UpdateQueueOutputReuse(output, output_index, call, tpipe, state));
    } else if (output->attr.mem.alloc_type == af::AllocType::kAllocTypeBuffer) {
      UpdateBufferOutputReuse(output, output_index, call, state);
    }
  }
  return af::SUCCESS;
}
}  // namespace

Loop::Loop(const ascir::AxisId axis) : axis_id(axis), parent(nullptr) {}

void Loop::AddLoop(Loop *loop) {
  LoopBody tmp;
  tmp.type = LoopType::LOOP;
  tmp.loop = loop;
  tmp.loop->is_graph_has_reduce_node = this->is_graph_has_reduce_node;
  tmp.loop->is_ar = this->is_ar;
  this->bodys.emplace_back(tmp);
  loop->parent = this;
}

void Loop::AddCall(ApiCall *call) {
  LoopBody tmp;
  tmp.type = LoopType::CALL;
  tmp.call = call;
  this->bodys.emplace_back(tmp);
}

static bool IsReduceOp(const ascir::NodeView &node) {
  return IsOps<Max>(node) || IsOps<Min>(node) || IsOps<Sum>(node) || IsOps<Mean>(node) || IsOps<Prod>(node) ||
         IsOps<All>(node) || IsOps<Any>(node);
}

static void TraverseGraphForReduceNodes(ascir::NodeViewVisitorConst nodes, bool &is_graph_has_reduce_node,
                                        bool &is_ar) {
  for (auto node : nodes) {
    if (IsReduceOp(node)) {
      is_graph_has_reduce_node = true;
      af::AscNodeOutputs node_outputs = node->outputs;
      if (!node_outputs().empty() && !node_outputs[0].attr.vectorized_strides.empty()) {
        is_ar = node_outputs[0].attr.vectorized_strides.back() == 0;
        return;
      }
    }
  }
  is_graph_has_reduce_node = false;
  return;
}

static bool IsRemovePadLinkBroadcast(const ascir::NodeView &node) {
  return IsOps<RemovePad>(node) && node->GetInDataNodesSize() == 1UL && IsOps<Broadcast>(node->GetInDataNodes().at(0));
}

static bool IsLoadNodeSplitB(const ascir::NodeView &node, const Tiler &tiler, std::string &enable_cache_with_condition,
                             bool is_ar, bool is_link_to_brc) {
  auto out = node->outputs[0];
  bool split_b = false;
  std::ostringstream ss;
  int32_t matching_counts = 0;
  int32_t matching_success_counts = 0;
  int32_t matching_success_current = 0;
  for (auto axis : out.attr.vectorized_axis) {
    if (axis == af::kIdNone || tiler.GetAxis(axis).type != ascir::Axis::Type::kAxisTypeTileInner) {
      continue;
    }
    auto axis_iter = std::find(out.attr.axis.begin(), out.attr.axis.end(), axis);
    if (axis_iter != out.attr.axis.end()) {
      matching_counts++;
      bool res = (out.attr.strides.at(axis_iter - out.attr.axis.begin()) == 0);
      if (res) {
        matching_success_current = matching_counts;
        matching_success_counts++;
      }
      split_b = split_b || res;
    }
  }

  if (matching_success_counts > 1) {
    ss << kEnCacheR;
  } else if (matching_success_counts == 1) {
    const bool enableCacheA = (matching_success_current == 1 && is_ar) || (matching_success_current != 1 && !is_ar);

    ss << (enableCacheA ? kEnCacheA : kEnCacheR);
    if (enableCacheA) {
      ss << " || control_dis_enable_cache_a";
    }
  }

  enable_cache_with_condition = ss.str();

  if (is_link_to_brc) {
    return split_b;
  } else {
    const auto platform = optimize::PlatformFactory::GetInstance().GetPlatform();
    GE_ASSERT_NOTNULL(platform);
    return split_b && IsLinkToBrdcst(node, platform->BroadcastTypes());
  }
}

// 多输入节点的缓存刷新条件按 “||” 语义取并集：任一输入到达自身刷新点就必须重新执行，复用仅在
// 所有输入都保持不变的窗口内有效。按 “ || ” 拆分子项去重，避免嵌套多输入场景重复累加同一条件。
// 示例：Add 的两个输入链分别缓存于 A 轴和 R 轴，输入0的条件为
// "dis_enable_cache_a || control_dis_enable_cache_a"、输入1的条件为 "dis_enable_cache_r"，
// 合并结果为 "dis_enable_cache_a || control_dis_enable_cache_a || dis_enable_cache_r"；
// 若两个输入的条件相同（如均为 "dis_enable_cache_r"），去重后合并结果仍为 "dis_enable_cache_r"。
static void MergeCacheACacheRCondition(const std::string &cur_input_cache_condition, std::string &merged) {
  static const std::string kCondSep = " || ";
  if (cur_input_cache_condition.empty()) {
    return;
  }
  std::vector<std::string> merged_terms;
  size_t pos = 0;
  while (pos < merged.size()) {
    size_t end = merged.find(kCondSep, pos);
    merged_terms.emplace_back(merged.substr(pos, end == std::string::npos ? end : end - pos));
    pos = (end == std::string::npos) ? end : end + kCondSep.size();
  }
  pos = 0;
  while (pos < cur_input_cache_condition.size()) {
    size_t end = cur_input_cache_condition.find(kCondSep, pos);
    std::string term = cur_input_cache_condition.substr(pos, end == std::string::npos ? end : end - pos);
    if (std::find(merged_terms.begin(), merged_terms.end(), term) == merged_terms.end()) {
      if (!merged.empty()) {
        merged += kCondSep;
      }
      merged += term;
      merged_terms.emplace_back(term);
    }
    pos = (end == std::string::npos) ? end : end + kCondSep.size();
  }
}

static bool IsNodeSplitB(const ascir::NodeView &node, const Tiler &tiler, std::string &enable_cache_with_condition,
                         bool is_ar, bool is_link_to_brc = false) {
  if (IsOps<Data>(node) || node->GetInDataNodesSize() == 0U) {
    return false;
  }

  if (node->attr.api.compute_type == af::ComputeType::kComputeLoad) {
    return IsLoadNodeSplitB(node, tiler, enable_cache_with_condition, is_ar, is_link_to_brc);
  }

  const auto platform = optimize::PlatformFactory::GetInstance().GetPlatform();
  GE_ASSERT_NOTNULL(platform);
  bool node_link_to_brc = IsLinkToBrdcst(std::dynamic_pointer_cast<af::AscNode>(node), platform->BroadcastTypes());
  bool remove_pad_link_brc = IsRemovePadLinkBroadcast(std::dynamic_pointer_cast<af::AscNode>(node));
  if (!node_link_to_brc && !remove_pad_link_brc) {
    return false;
  }
  // 各输入链的缓存刷新条件相互独立，必须取并集：任一输入数据变化都会使本节点的缓存失效。
  std::string all_inputs_merged_cache_condition;
  for (const auto &in_node : node->GetInDataNodes()) {
    GE_ASSERT_NOTNULL(in_node, "Input of node %s[%s] is null", node->GetTypePtr(), node->GetNamePtr());
    GE_ASSERT_NOTNULL(std::dynamic_pointer_cast<af::AscNode>(in_node));
    const auto &prev_node = std::dynamic_pointer_cast<af::AscNode>(in_node);
    std::string cur_input_cache_condition;
    if (!IsOps<Scalar>(prev_node) && !IsNodeSplitB(prev_node, tiler, cur_input_cache_condition, is_ar, true)) {
      return false;
    }
    MergeCacheACacheRCondition(cur_input_cache_condition, all_inputs_merged_cache_condition);
  }
  enable_cache_with_condition = all_inputs_merged_cache_condition;
  return !all_inputs_merged_cache_condition.empty();
}

static bool IsValidCacheCondition(const af::ExecuteCondition &exec_condition) {
  return exec_condition == af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis ||
         exec_condition == af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis;
}

static int64_t GetLifecycleEdge(ascir::NodeViewVisitorConst nodes, const TPipe &tpipe) {
  int64_t lifecycle_edge = 0;
  if (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
    for (const auto &node : nodes) {
      if (IsOps<Load>(node) && (node->outputs[0].attr.mem.tensor_id == tpipe.cube_output_tensor_id)) {
        for (const auto &out_node : node->GetOutDataNodesPtr()) {
          lifecycle_edge = std::max(lifecycle_edge, out_node->GetOpDescBarePtr()->GetId());
        }
      }
    }
  }
  return lifecycle_edge;
}

static void InitApiCallContext(const ascir::NodeView &node, const TPipe &tpipe, ApiCall *call, int64_t lifecycle_edge) {
  if (tpipe.cv_fusion_type != ascir::CubeTemplateType::kUBFuse) {
    return;
  }
  int64_t node_topo_id = node->GetOpDescBarePtr()->GetId();
  if (IsOps<Load>(node) && (node->outputs[0].attr.mem.tensor_id == tpipe.cube_output_tensor_id)) {
    call->api_call_context.scene = ApiScene::kCVFuseUBLoad;
  }
  if (node_topo_id <= lifecycle_edge) {
    call->api_call_context.stage = ComputeStage::kCVFuseStage1;
  } else {
    call->api_call_context.stage = ComputeStage::kCVFuseStage2;
  }
  return;
}

Status Loop::ConstructFromNodes(ascir::NodeViewVisitorConst nodes, const Tiler &tiler, TPipe &tpipe) {
  auto current_loop = this;
  std::vector<ascir::AxisId> current_axis;
  ConstructState state;
  TraverseGraphForReduceNodes(nodes, current_loop->is_graph_has_reduce_node, current_loop->is_ar);
  const int64_t lifecycle_edge = GetLifecycleEdge(nodes, tpipe);
  for (const auto &node : nodes) {
    if (ascgen_utils::indirect_load::GetTemplateBehavior(std::dynamic_pointer_cast<af::AscNode>(node)).skips_api_emit) {
      GE_CHK_STATUS_RET(AddSkippedApiEmitProcessCall(node, current_loop, current_axis, state.tensor_calls));
      continue;
    }

    GELOGI("node:%s, ComputeUnit:%u\r\n", node->GetNamePtr(), static_cast<uint32_t>(node->attr.api.unit));
    GE_ASSERT_SUCCESS(MoveToNodeLoop(node, current_axis, current_loop));

    auto call = CreateApiCallObject(node);
    GE_ASSERT_NOTNULL(call, "Create api call object failed, ascir type:%s", node->GetTypePtr());
    current_loop->AddCall(call);
    GE_CHK_STATUS_RET(call->Init(node), "ApiCall Init failed, ascir type:%s", node->GetTypePtr());
    call->exec_condition = node->attr.sched.exec_condition;
    // Reduce 图必须通过整条 Broadcast 输入链的 split-B 检查，非 Reduce 图使用 AutoSchedule 缓存标记。
    bool is_fixed_indirect_load_parameter = false;
    if (this->is_graph_has_reduce_node && node->attr.api.compute_type == af::ComputeType::kComputeLoad &&
        node->attr.sched.exec_condition == af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis &&
        !node->outputs().empty()) {
      const auto &load_output = node->outputs()[0]->attr;
      is_fixed_indirect_load_parameter =
          std::any_of(load_output.strides.begin(), load_output.strides.end(), [](const auto &stride) {
            return af::SymbolicUtils::StaticCheckEq(stride, af::sym::kSymbolZero) == af::TriBool::kTrue;
          });
    }
    call->enable_cache = this->is_graph_has_reduce_node
                             ? (is_fixed_indirect_load_parameter ||
                                IsNodeSplitB(node, tiler, call->enable_cache_with_condition, current_loop->is_ar))
                             : IsValidCacheCondition(call->exec_condition);
    GELOGI(
        "Node[%s][%s] cache eligibility: has_reduce[%d], enable_cache[%d], fixed_indirect_load_param[%d], "
        "exec_condition[%u], reduce_cache_condition[%s]",
        node->GetNamePtr(), node->GetTypePtr(), static_cast<int32_t>(this->is_graph_has_reduce_node),
        static_cast<int32_t>(call->enable_cache), static_cast<int32_t>(is_fixed_indirect_load_parameter),
        static_cast<uint32_t>(call->exec_condition), call->enable_cache_with_condition.c_str());
    call->axis = current_loop->axis_id;
    call->depth = current_axis.size();
    InitApiCallContext(node, tpipe, call, lifecycle_edge);
    GE_ASSERT_SUCCESS(ConnectApiCallInputs(node, state.tensor_calls, *call, tpipe));
    if (IsOps<Output>(node)) {
      continue;
    }
    GE_ASSERT_SUCCESS(RegisterApiCallOutputs(node, *call, tpipe, state));
  }
  return af::SUCCESS;
}

void Loop::Destruct() {
  for (auto body : this->bodys) {
    if (body.type == LoopType::LOOP) {
      body.loop->Destruct();
      delete body.loop;
    } else if (body.type == LoopType::CALL) {
      delete body.call;
    }
  }
}

void Loop::CollectTensorCrossLoop(std::map<ascir::AxisId, std::map<Loop *, std::vector<ApiCall *>>> &api_calls) {
  if (this->bodys.size() <= 1) {
    return;
  }
  for (auto body : this->bodys) {
    if (body.type == LoopType::LOOP) {
      for (auto inner_body : body.loop->bodys) {
        if (inner_body.type != LoopType::CALL) {
          inner_body.loop->CollectTensorCrossLoop(api_calls);
          continue;
        }
        ascir::AxisId target_axis;
        bool flag = inner_body.call->IsReadOutersideWrite(target_axis);
        if (flag) {
          api_calls[target_axis][body.loop].emplace_back(inner_body.call);
        }
      }
    }
  }
  return;
}

bool Loop::IsFindInUsedCalls(const ApiCall *call) const {
  if (parent == nullptr) {
    return false;
  }
  if (parent->used_calls.count(call) > 0) {
    return true;
  }
  return parent->IsFindInUsedCalls(call);
}

bool Loop::IsBodyContainLoop() const {
  size_t loop_count = 0;
  for (auto &body : bodys) {
    if (body.type == LoopType::LOOP) {
      loop_count++;
    }
  }
  return loop_count != 0;
}

static bool IsReduceDoubleTile(const Tiler &tiler, const TPipe &tpipe, bool has_reduce_node) {
  (void)tiler;
  for (const auto &tensor : tpipe.tensors) {
    size_t tile_inner_size = 0;
    for (auto axis_id : tensor.second.vectorized_axis) {
      auto &axis = tpipe.tiler.GetAxis(axis_id);
      if (axis.type == ascir::Axis::Type::kAxisTypeTileInner) {
        tile_inner_size += 1;
      }
    }
    if (tile_inner_size < kDoubleAxisSize) {
      continue;
    }
    return has_reduce_node;
  }
  return false;
}

static bool IsLoadSourceInvariantOnLoopAxis(const af::AscNodePtr &node, const Tiler &tiler,
                                            ascir::AxisId loop_axis_id) {
  if (node == nullptr) {
    return false;
  }
  // Load 的 GM 地址由 Tensor 轴和 stride 共同决定。当前循环轴如果能传递到某个
  // Tensor 轴，且该轴 stride 非零，则循环迭代会改变 GM offset，不能复用上一轮缓存。
  for (const auto &output : node->outputs()) {
    const auto &attr = output->attr;
    if (attr.axis.size() != attr.strides.size()) {
      return false;
    }
    for (size_t i = 0U; i < attr.axis.size(); ++i) {
      if (attr.axis[i] == af::kIdNone || !tiler.IsFrom(loop_axis_id, attr.axis[i])) {
        continue;
      }
      if (!ascgen_utils::ExpressEq(attr.strides[i], af::ops::Zero)) {
        return false;
      }
    }
  }
  return true;
}

static bool IsCacheInputInvariantOnLoopAxis(const af::AscNodePtr &node, const Tiler &tiler,
                                            ascir::AxisId loop_axis_id) {
  if (node == nullptr) {
    return false;
  }
  if (node->attr.api.compute_type == af::ComputeType::kComputeLoad) {
    return IsLoadSourceInvariantOnLoopAxis(node, tiler, loop_axis_id);
  }
  // Broadcast 前可能存在多个计算节点；只要任一路输入地址会变化，整个缓存链就不能复用。
  for (const auto &input_node : node->GetInDataNodes()) {
    const auto input_asc_node = std::dynamic_pointer_cast<af::AscNode>(input_node);
    if (!IsCacheInputInvariantOnLoopAxis(input_asc_node, tiler, loop_axis_id)) {
      return false;
    }
  }
  return true;
}

static std::string GetCacheGuardCondition(const ApiCall &call, bool is_enable_cache, bool is_double_tile,
                                          bool has_reduce_node, bool is_cache_input_invariant) {
  if (!is_enable_cache) {
    return "";
  }
  // 双 Tile Reduce 使用专用的 A/R 缓存条件。
  if (is_double_tile) {
    return call.enable_cache_with_condition;
  }
  // 普通 Reduce 的缓存刷新周期不能覆盖会改变 GM 地址的循环轴；无法证明地址不变时，
  // 返回空条件，调用方会在每个 Tile 正常执行 Load/Broadcast。
  if (has_reduce_node && !is_cache_input_invariant) {
    return "";
  }
  // 外层有效轴均为广播轴，缓存值在当前分块循环内保持不变，只需在第一次迭代生成。
  if (call.exec_condition == af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis) {
    return kEnCacheOriginBroadcastAxis;
  }
  // 广播轴与非广播轴融合，缓存值只在一个广播复用周期内不变，需要在每个周期起点重新生成。
  if (call.exec_condition == af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis) {
    return kEnCacheFusedBroadcastAxis;
  }
  return "";
}

Status Loop::GenerateBody(const Tiler &tiler, const TPipe &tpipe, std::vector<ascir::AxisId> &current_axis,
                          std::stringstream &ss) {
  std::map<ascir::AxisId, std::map<Loop *, std::vector<ApiCall *>>> api_calls_cross_loop;
  if (this->bodys.size() > 1) {
    CollectTensorCrossLoop(api_calls_cross_loop);
  }
  auto &cross_loop_map = api_calls_cross_loop[this->axis_id];
  static const std::vector<ApiCall *> kEmptyCrossLoopCalls;

  for (const auto &body : this->bodys) {
    if ((body.type == LoopType::CALL) && (body.call->api_call_context.scene == ApiScene::kCVFuseUBLoad ||
                                          body.call->api_call_context.stage != this->compute_stage)) {
      continue;
    }
    if (body.type == LoopType::LOOP) {
      auto it = cross_loop_map.find(body.loop);
      const auto &cross_loop_calls = it == cross_loop_map.end() ? kEmptyCrossLoopCalls : it->second;
      for (auto call : cross_loop_calls) {
        GE_CHK_STATUS_RET(call->AllocOutputs(tpipe, ss), "Codegen alloc outputs failed");
        used_calls.insert(call);
      }
      body.loop->compute_stage = this->compute_stage;
      GE_CHK_STATUS_RET(body.loop->GenerateLoop(tiler, tpipe, current_axis, ss), "Generate loop for body failed");
      for (auto call : cross_loop_calls) {
        GE_CHK_BOOL_RET_STATUS(call->SyncOutputs(tpipe, ss), af::FAILED, "Func SyncOutputs return false");
      }
      used_calls.clear();
    } else {
      if (body.call->unit == af::ComputeUnit::kUnitNone || body.call->skip_api_emit) {
        continue;
      }
      const auto behavior =
          ascgen_utils::indirect_load::GetTemplateBehavior(std::dynamic_pointer_cast<af::AscNode>(body.call->node));
      // SIMT IndirectLoad 的 input0/input1 不进入通用 Wait/Free；输出生命周期仍由 skips_ub_lifecycle 单独控制。
      if (!behavior.skips_input_lifecycle) {
        GE_CHK_BOOL_RET_STATUS(body.call->WaitInputs(tpipe, ss), af::FAILED, "Func WaitInputs return false");
      }
      if (!IsFindInUsedCalls(body.call) && !behavior.skips_ub_lifecycle) {
        GE_CHK_STATUS_RET(body.call->AllocOutputs(tpipe, ss), "Codegen alloc outputs failed");
      }
      std::string call;
      std::string cache_guard;

      if (this->axis_id != af::kIdNone && this->compute_stage == ComputeStage::kDefault) {
        auto axis = tiler.GetAxis(this->axis_id);
        const bool is_enable_cache = axis.is_split_b && body.call->enable_cache;
        // 双 Tile Reduce 的 A/R 缓存有独立的刷新协议；普通 Reduce 才需要额外检查
        // 缓存链的 GM 地址是否随当前 loop axis 变化。
        const bool is_double_tile = IsReduceDoubleTile(tiler, tpipe, this->is_graph_has_reduce_node) &&
                                    current_axis.size() > kDoubleTileAxisSize;
        const bool is_cache_input_invariant =
            !this->is_graph_has_reduce_node || IsCacheInputInvariantOnLoopAxis(body.call->node, tiler, this->axis_id);
        cache_guard = GetCacheGuardCondition(*body.call, is_enable_cache, is_double_tile,
                                             this->is_graph_has_reduce_node, is_cache_input_invariant);
        if (is_enable_cache && this->is_graph_has_reduce_node && !is_double_tile && !is_cache_input_invariant) {
          GELOGD("Node[%s][%s] cache guard skipped because GM offset depends on loop axis[%ld]",
                 body.call->node_name.c_str(), body.call->type.c_str(), this->axis_id);
        }
        if (!cache_guard.empty()) {
          ss << "if (" << cache_guard << ") {" << std::endl;
        }
      }
      GE_CHK_STATUS_RET(body.call->Generate(tpipe, current_axis, call), "Codegen generate call failed");
      ss << call;

      if (!cache_guard.empty()) {
        ss << "}" << std::endl;
      }

      // post-reduce SIMT 需要保留输出 Alloc/Sync/Free，因此不能与输入生命周期共用一个开关。
      if (!behavior.skips_ub_lifecycle) {
        if (!IsFindInUsedCalls(body.call)) {
          GE_CHK_BOOL_RET_STATUS(body.call->SyncOutputs(tpipe, ss), af::FAILED, "Func SyncOutputs return false");
        }
        if (!behavior.skips_input_lifecycle) {
          GE_CHK_BOOL_RET_STATUS(body.call->FreeInputs(tpipe, ss), af::FAILED, "Func FreeInputs return false");
        }
        GE_CHK_BOOL_RET_STATUS(body.call->FreeUnusedOutputs(tpipe, ss), af::FAILED, "Func FreeUnusedOutputs fails");
      }
      ss << std::endl;
    }
  }

  return af::SUCCESS;
}

std::string Loop::GetReduceType() const {
  std::vector<std::string> reduce_map = {Max::Type, Sum::Type, Min::Type, Mean::Type, Prod::Type};
  for (auto body : this->bodys) {
    if (body.type == LoopType::CALL) {
      for (size_t i = 0; i < reduce_map.size(); ++i) {
        if (reduce_map[i] == body.call->type) {
          return reduce_map[i];
        }
      }
    }
  }
  GELOGI("No Reduce type found");
  return "";
}

bool Loop::IsHaveReduceType(const std::string &type) const {
  for (auto body : this->bodys) {
    if (body.type == LoopType::CALL && body.call->type == type) {
      return true;
    }
  }
  return false;
}

/* 获取reduce api的输出tensor */
const Tensor *Loop::GetReduceOutputTensor(const TPipe &tpipe) const {
  for (auto it = this->bodys.rbegin(); it != this->bodys.rend(); ++it) {
    if (it->type == LoopType::CALL) {
      auto out_tensor_ptr = tpipe.GetTensor(it->call->outputs[0].id);
      return out_tensor_ptr;
    }
  }
  GELOGE(af::FAILED, "No valid reduce output tensor found.");
  return nullptr;
}

/* 获取reduce api的输入tensor */
const Tensor *Loop::GetReduceInputTensor(const TPipe &tpipe) const {
  for (auto it = this->bodys.rbegin(); it != this->bodys.rend(); ++it) {
    if (it->type == LoopType::CALL) {
      auto in_tensor_ptr = tpipe.GetTensor(it->call->inputs[0]->id);
      return in_tensor_ptr;
    }
  }
  GELOGE(af::FAILED, "No valid reduce input tensor found.");
  return nullptr;
}

static void CreateInnerLoopSizeAndActualSize(const TPipe &tpipe, const Tiler &tiler, const Axis &axis,
                                             std::stringstream &ss) {
  if (axis.from.size() == 1) {
    ss << tiler.GenInnerLoopSizeAndActualSize(axis.split_pair_other_id, axis.id);
    return;
  }

  ascir::AxisId inner_id = -1;
  ascir::AxisId outer_id = -1;
  for (size_t i = 0; i < axis.from.size(); i++) {
    auto current_axis = tiler.GetAxis(axis.from[i]);
    if (current_axis.IsOuter() &&
        tiler.GetAxis(current_axis.split_pair_other_id).type == Axis::Type::kAxisTypeBlockInner) {
      outer_id = current_axis.id;
      inner_id = current_axis.split_pair_other_id;
      break;
    }
  }
  ss << tiler.GenInnerLoopSizeAndActualSize(inner_id, outer_id);
  std::set<ascir::AxisId> vectorized_axis;
  for (const auto &tensor : tpipe.tensors) {
    for (auto axis_id : tensor.second.vectorized_axis) {
      int32_t count = 0;
      for (auto &from : axis.from) {
        if (tiler.HasSameOriginAxis(axis_id, from)) {
          count++;
        }
      }
      if (vectorized_axis.find(axis_id) == vectorized_axis.end() &&
          (count != 0 && !tiler.HasSameOriginAxis(axis_id, inner_id))) {
        ss << tpipe.tiler.GenInnerLoopSizeAndActualSize(axis_id, axis.id);
        vectorized_axis.insert(axis_id);
      }
    }
  }
}

Status Loop::GenerateLoop(const Tiler &tiler, const TPipe &tpipe, std::vector<ascir::AxisId> &current_axis,
                          std::stringstream &ss) {
  if (this->axis_id == af::kIdNone) {
    GE_CHK_STATUS_RET(this->GenerateBody(tiler, tpipe, current_axis, ss),
                      "Codegen generate body failed when axis id is none");
    return af::SUCCESS;
  }

  const auto &axis = tiler.GetAxis(this->axis_id);
  current_axis.push_back(this->axis_id);
  if (axis.type == Axis::Type::kAxisTypeBlockOuter) {
    CreateInnerLoopSizeAndActualSize(tpipe, tiler, axis, ss);
    GE_CHK_STATUS_RET(this->GenerateBody(tiler, tpipe, current_axis, ss),
                      "Codegen generate body failed when axis type is block outer");
  } else {
    if (axis.type == Axis::Type::kAxisTypeTileInner && FindIndirectLoadOpCall(this, false) != nullptr) {
      ss << tiler.GenInnerLoopSizeAndActualSize(axis.id, axis.split_pair_other_id);
    }
    std::string reduce_dim_a = "reduce_dim_a";
    if (IsHaveReduceType("Mean")) {
      ss << "uint32_t " << reduce_dim_a << ";" << std::endl;
    }
    if (axis.type != Axis::Type::kAxisTypeBlockInner && this->is_graph_has_reduce_node) {
      ss << "control_dis_enable_cache_a = true;" << std::endl;
      ss << "if ( " << axis.loop_size.Str() << " == 1) {" << std::endl;
      ss << "control_dis_enable_cache_a = false;" << std::endl;
      ss << "}" << std::endl;
    }
    if (axis.type == Axis::Type::kAxisTypeBlockInner) {
      auto peer = tiler.GetAxis(axis.split_pair_other_id);
      ss << "int32_t block_dim_offset = " << peer.Str() << " * " << tiler.Size(axis.size) << ";" << std::endl;
      ApiCall *call = FindIndirectLoadOpCall(this, true);
      if (call != nullptr) {
        const auto behavior =
            ascgen_utils::indirect_load::GetTemplateBehavior(std::dynamic_pointer_cast<af::AscNode>(call->node));
        const auto indirect_load = std::dynamic_pointer_cast<af::AscNode>(call->node);
        const bool has_post_reduce = ascgen_utils::indirect_load::GetPostReduceConsumer(indirect_load) != nullptr;
        if (!behavior.skips_ub_lifecycle || has_post_reduce) {
          GE_CHK_STATUS_RET(call->AllocOutputs(tpipe, ss), "Codegen alloc hoisted IndirectLoad outputs failed");
        }
        std::string call_code;
        GE_CHK_STATUS_RET(call->Generate(tpipe, current_axis, call_code), "Codegen generate hoisted call failed");
        ss << call_code << std::endl;
        call->skip_api_emit = true;
        current_axis.pop_back();
        return af::SUCCESS;
      }
    }
    if (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse && axis.type == Axis::Type::kAxisTypeTileOuter) {
      ss << axis.loop_size.AsArg() << " = 1;" << std::endl;
    }
    ss << "for (" << axis.AsArg() << " = 0; " << axis << " < " << axis.loop_size.Str() << "; " << axis << "++) "
       << "{" << std::endl;
    if (tpipe.cv_fusion_type != ascir::CubeTemplateType::kUBFuse) {
      ss << tiler.CalcFromAxis(axis.id);
    }
    if (this->compute_stage == ComputeStage::kDefault) {
      GenerateEnCacheCondition(tiler, tpipe, axis, ss);
    }
    if (tpipe.cv_fusion_type != ascir::CubeTemplateType::kUBFuse) {
      std::set<ascir::AxisId> vectorized_axis;
      for (const auto &tensor : tpipe.tensors) {
        for (auto axis_id : tensor.second.vectorized_axis) {
          if (vectorized_axis.find(axis_id) == vectorized_axis.end()) {
            ss << tpipe.tiler.GenInnerLoopSizeAndActualSize(axis_id, this->axis_id);
            vectorized_axis.insert(axis_id);
          }
        }
      }
    } else if (axis.type == Axis::Type::kAxisTypeTileOuter) {
      const auto &tile_inner = tiler.GetAxis(axis.split_pair_other_id);
      af::Expression actual_size = af::Symbol(tile_inner.actual_size.name.c_str());
      tpipe.tiler.actual_sizes.emplace_back(std::make_pair(tile_inner.size_expr, actual_size));
      ss << tile_inner.actual_size.AsArg() << " = stageSize;"
         << std::endl;  // 多轮循环不能使用curAivM * curAivN，否则奇数尾块计算有精度问题
      auto ub_tensor = tpipe.GetTensor(tpipe.cube_output_tensor_id);
      GE_CHK_BOOL_RET_STATUS(ub_tensor != nullptr, af::FAILED,
                             "Codegen CV Fusion MatmulOutput UB tensor id[%ld] "
                             "not found",
                             tpipe.cube_output_tensor_id);
      ss << ub_tensor->Str() << "_actual_size = " << tile_inner.actual_size.Str() << ";" << std::endl;
    }
    GE_CHK_STATUS_RET(this->GenerateBody(tiler, tpipe, current_axis, ss),
                      "Codegen generate body failed for normal loop");
    ss << "}" << std::endl;
    if (IsReduceDoubleTile(tiler, tpipe, this->is_graph_has_reduce_node) && GetReduceType() == "Mean") {
      auto reduce_src_tensor = GetReduceInputTensor(tpipe);
      auto reduce_dst_tensor = GetReduceOutputTensor(tpipe);
      std::string dtype_name;
      Tensor::DtypeName(reduce_dst_tensor->dtype, dtype_name);
      std::set<ascir::AxisId> r_from_axis;
      for (size_t i = 0; i < reduce_dst_tensor->axis_strides.size(); i++) {
        if ((reduce_src_tensor->axis_strides[i] != 0 || reduce_src_tensor->axis_size[i] != 1) &&
            reduce_dst_tensor->axis_strides[i] == 0) {  // 如果目标张量的轴步长为0
          auto axis_id = reduce_dst_tensor->axis[i];    // 获取当前轴ID
          // 定义递归函数用于收集原始轴
          std::function<void(int32_t)> collect_original_axes = [&tiler, &r_from_axis,
                                                                &collect_original_axes](int32_t current_axis_id) {
            auto axis = tiler.GetAxis(current_axis_id);  // 获取当前轴对象
            if (axis.type == ascir::Axis::Type::kAxisTypeOriginal) {
              r_from_axis.insert(current_axis_id);  // 如果是原始轴则加入集合
            } else {
              // 否则递归处理所有来源轴
              for (auto from_axis_id : axis.from) {
                collect_original_axes(from_axis_id);
              }
            }
          };
          collect_original_axes(axis_id);  // 从当前轴开始递归收集
        }
      }
      ss << "const float dimr_recip = 1.0f / (";
      uint32_t count = 0;
      for (auto axis_id : r_from_axis) {
        if (count == 0) {
          ss << tiler.AxisSize(axis_id);
          count++;
        } else {
          ss << " * " << tiler.AxisSize(axis_id);
        }
      }
      ss << ");" << std::endl;
      ss << "Muls(" << reduce_dst_tensor->Str() << ", " << reduce_dst_tensor->Str() << ", "
         << "dimr_recip, " << KernelUtils::SizeAlign() << "(" << reduce_dim_a << ", 32 / sizeof(" << dtype_name
         << ")));" << std::endl;
    }
  }
  current_axis.pop_back();
  return af::SUCCESS;
}

static const Axis &GetTileOutAxisAnother(const Tiler &tiler, const Axis &axis) {
  int32_t count = 0;
  for (auto &[id, cur_axis] : tiler.axis_map) {
    (void)id;
    if (cur_axis.type == ascir::Axis::Type::kAxisTypeTileOuter) {
      count++;
    }

    if (count > 1) {
      return cur_axis;
    }
  }
  return axis;
}

static const Axis &GetTileOutAxis(const Tiler &tiler, const Axis &axis) {
  for (auto &[id, cur_axis] : tiler.axis_map) {
    (void)id;
    if (cur_axis.type == ascir::Axis::Type::kAxisTypeTileOuter) {
      return cur_axis;
    }
  }
  return axis;
}

void Loop::GenerateEnCacheCondition(const Tiler &tiler, const TPipe &tpipe, const Axis &axis,
                                    std::stringstream &ss) const {
  (void)tpipe;
  if (!axis.is_split_b) {
    return;
  }
  af::Expression block_in_size = Zero;
  for (auto &[id, cur_axis] : tiler.axis_map) {
    (void)id;
    if (cur_axis.type == ascir::Axis::Type::kAxisTypeBlockInner) {
      block_in_size = af::Symbol(cur_axis.axis_size.name.c_str());
      break;
    }
  }
  Axis tile_out_axis = GetTileOutAxis(tiler, axis);
  bool is_double_tile = IsReduceDoubleTile(tiler, tpipe, this->is_graph_has_reduce_node);
  bool is_cache_valid = this->IsBodyContainLoop() || axis.type == ascir::Axis::Type::kAxisTypeTileOuter ||
                        axis.type == ascir::Axis::Type::kAxisTypeMerged;
  if (is_double_tile && is_cache_valid) {
    Axis thile_out_axis_another = GetTileOutAxisAnother(tiler, axis);
    for (auto body : this->bodys) {
      if (body.type == LoopType::LOOP) {
        ss << "bool " << kEnCacheA << " = (" << axis << " < 1) || ((" << tiler.block_dim << " * "
           << block_in_size.Str().get() << " + " << axis << ") % " << tile_out_axis.loop_size << " < 1);" << std::endl;
        break;
      }

      if (body.call->unit == af::ComputeUnit::kUnitNone || !body.call->enable_cache) {
        continue;
      }

      ss << "bool " << kEnCacheR << " = (" << axis << " < 1) || ((" << axis << ") % "
         << thile_out_axis_another.loop_size << " < 1);" << std::endl;
      break;
    }
  } else {
    bool need_create_fused_cond = true;
    bool need_create_origin_cond = true;
    for (auto body : this->bodys) {
      if (body.type != LoopType::CALL || body.call->unit == af::ComputeUnit::kUnitNone || !body.call->enable_cache) {
        continue;
      }
      if (body.call->exec_condition == af::ExecuteCondition::kCacheBlockSplitFusedBroadcastAxis &&
          need_create_fused_cond) {
        ss << "bool " << kEnCacheFusedBroadcastAxis << " = (" << axis << " < 1) || ((" << tiler.block_dim << " * "
           << block_in_size.Str().get() << " + " << axis << ") % " << tile_out_axis.loop_size << " < 1);" << std::endl;
        need_create_fused_cond = false;
        continue;
      }
      if (body.call->exec_condition == af::ExecuteCondition::kCacheBlockSplitOriginBroadcastAxis &&
          need_create_origin_cond) {
        ss << "bool " << kEnCacheOriginBroadcastAxis << " = (" << axis << " < 1);" << std::endl;
        need_create_origin_cond = false;
        continue;
      }
    }
  }
}

Status Loop::Generate(const Tiler &tiler, const TPipe &tpipe, std::string &result, ComputeStage stage) {
  std::vector<ascir::AxisId> current_axis;
  this->compute_stage = stage;
  stringstream ss;
  if (this->is_graph_has_reduce_node) {
    ss << "bool control_dis_enable_cache_a = true;" << std::endl;
  }
  GE_CHK_STATUS_RET(this->GenerateLoop(tiler, tpipe, current_axis, ss), "Generate loop failed");
  result = ss.str();
  return af::SUCCESS;
}

Status Loop::ActualSizeDefine(const Tiler &tiler, const TPipe &tpipe, std::string dtype_name, std::string &result) {
  std::stringstream ss;
  (void)dtype_name;
  if (this->axis_id == af::kIdNone) {
    for (const auto &body : this->bodys) {
      if (body.type == LoopType::LOOP) {
        GE_CHK_STATUS_RET(body.loop->ActualSizeDefine(tiler, tpipe, dtype_name, result), "Get axis id failed.");
      }
    }
    return af::SUCCESS;
  }
  const auto &axis = tiler.GetAxis(this->axis_id);
  if (axis.type == Axis::Type::kAxisTypeTileOuter) {
    const auto &tile_inner = tiler.GetAxis(axis.split_pair_other_id);
    af::Expression actual_size = af::Symbol(tile_inner.actual_size.name.c_str());
    tpipe.tiler.actual_sizes.emplace_back(std::make_pair(tile_inner.size_expr, actual_size));
    ss << tile_inner.actual_size.AsArg() << " = stage_size;" << std::endl;
  }
  result = ss.str();
  return af::SUCCESS;
}

Status LoopAxisDistance(const std::vector<ascir::AxisId> &current_loop,
                        const std::vector<ascir::AxisId> &node_sched_axis, const ascir::AxisId node_loop_axis,
                        int32_t &distance) {
  if (node_sched_axis.size() == 0 || node_loop_axis == af::kIdNone) {
    distance = -1 * current_loop.size();
    return af::SUCCESS;
  }

  int32_t same_axis_num = 0;
  for (size_t i = 0; i < node_sched_axis.size() && i < current_loop.size(); ++i) {
    if (node_sched_axis[i] == current_loop[i]) {
      same_axis_num++;
    } else if (node_sched_axis[i] == node_loop_axis) {
      break;
    }
  }

  int32_t loop_axis_pos = -1;
  for (size_t i = 0; i < node_sched_axis.size(); ++i) {
    if (node_loop_axis == node_sched_axis[i]) {
      loop_axis_pos = i;
      break;
    }
  }

  GE_ASSERT_TRUE(loop_axis_pos >= 0, "Codegen node loop axis not found in node_sched_axis");

  if (static_cast<size_t>(same_axis_num) < current_loop.size()) {
    if (loop_axis_pos < same_axis_num) {
      distance = -(current_loop.size() - loop_axis_pos);
      return af::SUCCESS;
    } else {
      distance = -(current_loop.size() - same_axis_num);
      return af::SUCCESS;
    }
  } else {
    distance = (loop_axis_pos + 1) - current_loop.size();
    return af::SUCCESS;
  }

  return af::SUCCESS;
}

ApiTensor::ApiTensor()
    : id(af::kIdNone),
      reuse_id(af::kIdNone),
      reuse_from(nullptr),
      reuse_next(nullptr),
      share_prev(nullptr),
      share_next(nullptr),
      share_order(-1),
      write(nullptr) {}

Status ApiCall::Init(const ascir::NodeView &node) {
  this->unit = node->attr.api.unit;
  this->type = node->GetType();
  this->compute_type = node->attr.api.compute_type;
  for (auto tmp_buffer : node->attr.tmp_buffers) {
    if (tmp_buffer.id == -1L) {
      continue;
    }
    this->tmp_buf_id[tmp_buffer.buf_desc.life_time_axis_id] = tmp_buffer.id;
  }
  if (!IsOps<Output>(node)) {
    for (auto o : node->outputs()) {
      auto &t = this->outputs.emplace_back();
      t.id = o->attr.mem.tensor_id;
      t.reuse_id = o->attr.mem.reuse_id;
      t.write = this;
    }
  }
  GE_CHK_STATUS_RET(ParseAttr(node));
  this->node = node;
  this->graph_name = node->GetOwnerComputeGraph()->GetName();
  this->node_name = node->GetNamePtr();
  return af::SUCCESS;
}

Status ApiCall::PreProcess(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                           const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                           std::string &result) const {
  stringstream ss;
  bool is_all_outputs_ub_scalar =
      std::all_of(outputs.begin(), outputs.end(),
                  [](const std::reference_wrapper<const Tensor> &t) { return t.get().is_ub_scalar; });
  bool is_any_output_need_two_loop =
      std::any_of(outputs.begin(), outputs.end(), [](const std::reference_wrapper<const Tensor> &t) {
        return t.get().alloc_type == af::AllocType::kAllocTypeQueue && t.get().que_buf_num_value == 2;
      });
  if (is_all_outputs_ub_scalar && !current_axis.empty()) {
    const auto loop_axis = tpipe.tiler.GetAxis(current_axis.back());
    // 如果当前节点输出tensor是ub_scalar，且ub的queue buffer num是2
    // 则代表存在一个输出节点的输出tensor不是ub_scalar，此时下一个节点的计算不在if (loop_axis < 1)的逻辑包含中
    // 而下一个节点依赖当前节点的输出tensor，因此需要改成if (loop_axis < 2)，保证DOUBLE_BUFFER流程中两个buffer都被计算
    if (is_any_output_need_two_loop) {
      ss << "if (" << loop_axis << " < 2) {" << std::endl;
    } else {
      ss << "if (" << loop_axis << " < 1) {" << std::endl;
    }
  }

  result = ss.str();
  return af::SUCCESS;
}

Status ApiCall::PostProcess(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                            const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                            std::string &result) const {
  (void)tpipe;
  stringstream ss;
  bool is_all_outputs_ub_scalar =
      std::all_of(outputs.begin(), outputs.end(),
                  [](const std::reference_wrapper<const Tensor> &t) { return t.get().is_ub_scalar; });
  bool first_gen_get_value = true;
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto &ub = outputs[i].get();
    if (ub.is_ub_scalar && !current_axis.empty()) {
      GELOGD("t_name:%s, need_gen_get_value_of_ub_scalar:%d", ub.Str().c_str(),
             static_cast<int32_t>(ub.need_gen_get_value_of_ub_scalar));
      // 生成ub_scalar的变量初始化定义
      if (ub.need_gen_get_value_of_ub_scalar) {
        if (first_gen_get_value) {
          std::string sync_type = (this->type == Load::Type) ? "MTE2_S" : "V_S";
          ss << "event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::";
          ss << sync_type;
          ss << "));" << std::endl;

          ss << "SetFlag<HardEvent::";
          ss << sync_type;
          ss << ">(eventID);" << std::endl;

          ss << "WaitFlag<HardEvent::";
          ss << sync_type;
          ss << ">(eventID);" << std::endl;

          first_gen_get_value = false;
        }
        std::string tmp;
        GE_CHK_STATUS_RET(ub.InitUbScalar(tmp));
        ss << tmp;
        if (ub.need_duplicate_value_of_ub_scalar) {
          GE_CHK_STATUS_RET(ub.GenDuplicateValueOfUbScalar(tmp));
          ss << tmp;
        }
      }
    }
  }

  if (is_all_outputs_ub_scalar && !current_axis.empty()) {
    ss << "}" << std::endl;
  }

  result = ss.str();
  return af::SUCCESS;
}

Status ApiCall::GenerateFuncDefinition(const TPipe &tpipe, const Tiler &tiler, stringstream &ss) const {
  (void)tpipe;
  (void)tiler;
  (void)ss;
  return af::SUCCESS;
}

Status ApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                         std::string &result) const {
  std::vector<reference_wrapper<const Tensor>> input_tensors;
  for (const auto &in : this->inputs) {
    auto tensor_ptr = tpipe.GetTensor(in->id);
    GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, af::FAILED, "Check[Param] tensor_ptr is nullptr");
    input_tensors.emplace_back(*tensor_ptr);
  }

  std::vector<reference_wrapper<const Tensor>> output_tensors;
  for (const auto &out : this->outputs) {
    auto tensor_ptr = tpipe.GetTensor(out.id);
    GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, af::FAILED, "Check[Param] tensor_ptr is nullptr");
    output_tensors.emplace_back(*tensor_ptr);
  }

  stringstream ss;
  // apicall pre process
  std::string pre_result;
  GE_CHK_STATUS_RET(PreProcess(tpipe, current_axis, output_tensors, pre_result),
                    "Codegen generate API call PreProcess failed");
  ss << pre_result;
  std::string local_result;
  GE_CHK_STATUS_RET(Generate(tpipe, current_axis, input_tensors, output_tensors, local_result),
                    "Codegen generate API call failed, api_name: %s", api_name_.c_str());
  ss << local_result;

  // apicall post process
  std::string post_result;
  GE_CHK_STATUS_RET(PostProcess(tpipe, current_axis, output_tensors, post_result),
                    "Codegen generate API call PostProcess failed");
  ss << post_result;

  result = ss.str();
  return af::SUCCESS;
}

static bool IsFirstRead(const ApiCall &call, const ApiTensor &tensor) {
  return !tensor.reads.empty() && tensor.reads[0] == &call;
}

static bool IsInShareQue(const ApiTensor &tensor) {
  return (tensor.share_prev != nullptr) || (tensor.share_next != nullptr);
}

static bool IsFirstShare(const ApiTensor &tensor) {
  return (tensor.share_prev == nullptr);
}

static bool IsLastShare(const ApiTensor &tensor) {
  return (tensor.share_next == nullptr);
}

static bool IsReuseWithDefine(const ApiTensor &tensor) {
  if (tensor.reuse_from == nullptr) {
    return true;
  }
  if (tensor.reuse_from->write == nullptr) {
    return false;
  }
  ascir::AxisId cur_axis = tensor.write->axis;
  ascir::AxisId reuse_axis = tensor.reuse_from->write->axis;
  bool with_define = cur_axis == reuse_axis ? false : true;
  return with_define;
}

/*
 * que共用只存在于 多个load的输出连接同一vec运算算子的输入场景
 *  load0 load1 load2
 *     \   |     /
 *        vec
 *
 * load0 load1 load2 的输出共用一个que
 */
BoolType ApiCall::WaitShareInputs(const TPipe &tpipe, const ApiTensor *in, const Tensor t,
                                  std::stringstream &ss) const {
  if (IsInShareQue(*in)) {
    if (in->share_prev == nullptr && IsFirstRead(*this, *in)) {
      auto t_que = tpipe.GetQue(t.que_id);
      GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, BoolType::FAILED, "Codegen que[%ld] not found", t.que_id);
      ss << t_que->DequeBuf(false);
    }
    return BoolType::TRUE;
  }
  return BoolType::FALSE;
}

af::Status DefineShareOffsets(const TPipe &tpipe, const ApiTensor &out, const Tensor &t, std::stringstream &ss) {
  std::map<int32_t, const ApiTensor *> order_to_tensor;
  for (auto it = out.share_prev; it != nullptr; it = it->share_prev) {
    order_to_tensor.emplace(it->share_order, it);
  }
  for (auto it = &out; it != nullptr; it = it->share_next) {
    order_to_tensor.emplace(it->share_order, it);
  }
  const auto cur_order = out.share_order;
  int32_t prev_max = -1;
  for (auto it = out.share_prev; it != nullptr; it = it->share_prev) {
    if (it->share_order > prev_max) {
      prev_max = it->share_order;
    }
  }
  for (int32_t i = prev_max + 1; i <= cur_order; ++i) {
    if (i == 0) {
      continue;
    }
    auto prev_var_name = t.que_share_offset.name;
    if (i - 1 > 0) {
      prev_var_name += ("_part_" + std::to_string(i - 1));
    }
    auto prev_tensor = tpipe.GetTensor(order_to_tensor[i - 1]->id);
    GE_ASSERT_NOTNULL(prev_tensor, "Check[Param] tensor_ptr is nullptr");
    auto var_size = prev_var_name + " + " + GetQueueSliceByteSize(tpipe, *prev_tensor);
    const auto &cur_var_name = t.que_share_offset.name + "_part_" + std::to_string(i);
    decltype(t.que_share_offset) offset_var(cur_var_name);
    ss << offset_var.DefineConst(std::move(var_size)) << std::endl;
  }
  return af::SUCCESS;
}

BoolType ApiCall::AllocShareOutputs(const TPipe &tpipe, const ApiTensor &out, const Tensor t,
                                    std::stringstream &ss) const {
  std::string relative_offset;
  if (IsFirstShare(out)) {
    // 第一个share tensor 或 仅复用的tenosr 初始化为 uint32_t q<id>_reuse<id>_offset = 0;
    ss << t.que_share_offset.Define("0") << std::endl;
  }
  if (IsInShareQue(out)) {
    if (this->unit == af::ComputeUnit::kUnitMTE2 && t.position == af::Position::kPositionVecIn) {
      if (out.reuse_from != nullptr && out.share_prev == nullptr) {
        bool with_define = IsReuseWithDefine(out);
        auto t_que = tpipe.GetQue(t.que_id);
        GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, BoolType::FAILED, "Codegen que[%ld] not found", t.que_id);
        ss << t_que->AllocBuf(with_define);
      } else if (out.reuse_from == nullptr && out.share_prev == nullptr) {
        auto t_que = tpipe.GetQue(t.que_id);
        GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, BoolType::FAILED, "Codegen que[%ld] not found", t.que_id);
        ss << t_que->AllocBuf();
      }
    }
    if (out.share_order == -1) {
      if (!IsFirstShare(out)) {
        auto tensor_ptr = tpipe.GetTensor(out.share_prev->id);
        GE_CHK_BOOL_RET_SPECIAL_STATUS(tensor_ptr == nullptr, BoolType::FAILED, "Check[Param] tensor_ptr is nullptr");
        auto prev_tensor = *tensor_ptr;
        relative_offset = t.que_share_offset.name + " + " + GetQueueSliceByteSize(tpipe, prev_tensor);
        ss << t.que_share_offset.Assign(relative_offset);
        ss << std::endl;
      }
    } else {
      // 需要按给定顺序计算offset
      GE_CHK_BOOL_RET_SPECIAL_STATUS(DefineShareOffsets(tpipe, out, t, ss), BoolType::FAILED);
      auto cur_var_name = t.que_share_offset.name;
      std::string offset = "0";
      if (out.share_order > 0) {
        offset = t.que_share_offset.name + "_part_" + std::to_string(out.share_order);
      }
      ss << t.que_share_offset.Assign(offset);
      ss << std::endl;
    }
    return BoolType::TRUE;
  }
  return BoolType::FALSE;
}

bool IsUnitFirstRead(const ApiCall &call, const ApiTensor &tensor) {
  for (auto r : tensor.reads) {
    if (r->unit != call.unit) {
      continue;
    }

    if (r == &call) {
      return true;
    } else {
      return false;
    }
  }

  return false;
}

bool ApiCall::WaitInputVector(const TPipe &tpipe, const ApiTensor *in, const Tensor &t, std::stringstream &ss) const {
  if (t.position == af::Position::kPositionVecIn && IsFirstRead(*this, *in)) {
    auto t_que = tpipe.GetQue(t.que_id);
    GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, false, "Codegen que[%ld] not found", t.que_id);
    if (t.que_id != tpipe.cube_output_que_id) {
      ss << t_que->DequeBuf(false);
    }
  } else if (t.position == af::Position::kPositionVecCalc && IsFirstRead(*this, *in)) {
    ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;
  } else if (t.position == af::Position::kPositionVecOut && IsUnitFirstRead(*this, *in)) {
    ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;
  }
  return true;
}

bool ApiCall::WaitInputMte(const TPipe &tpipe, const ApiTensor *in, const Tensor &t, std::stringstream &ss) const {
  // 1. load->store 2. load->store store 3. load->vec store store
  if (this->type == Store::Type &&
      ((in->write->compute_type == ascir::ComputeType::kComputeLoad) && (in->write->type != Gather::Type) &&
       (in->write->type != IndirectLoad::Type)) &&
      IsUnitFirstRead(*this, *in)) {
    // TQueBind直连MTE2到MTE3, 无需同步
    const auto t_que = tpipe.GetQue(t.que_id);
    if (t_que == nullptr || !t_que->is_bind) {
      ss << tpipe.SyncMte2ToMte3(t) << std::endl;
    }
  }
  if ((t.position == af::Position::kPositionVecOut) && IsUnitFirstRead(*this, *in)) {
    // 1. vec->store 2. vec->vec store store 3. vec->store vec store
    auto t_que = tpipe.GetQue(t.que_id);
    GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, false, "Codegen que[%ld] not found", t.que_id);
    ss << t_que->DequeBuf(false);
  } else if ((t.position == af::Position::kPositionVecIn) && IsFirstRead(*this, *in)) {
    // 1. load->store
    auto t_que = tpipe.GetQue(t.que_id);
    GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, false, "Codegen que[%ld] not found", t.que_id);
    ss << t_que->DequeBuf(false);
  } else if (t.position == af::Position::kPositionGM && IsUnitFirstRead(*this, *in) && in->write->type == Store::Type) {
    // store->workspace->load
    ss << tpipe.SyncMte3ToMte2(t) << std::endl;
  }
  return true;
}

bool ApiCall::WaitInputs(const TPipe &tpipe, std::stringstream &ss) const {
  std::vector<int64_t> handled_tensors;
  for (auto in : this->inputs) {
    auto it = std::find(handled_tensors.begin(), handled_tensors.end(), in->id);
    if (it != handled_tensors.end()) {
      GELOGI("WaitInputs tensor id[%ld] from same tensor", in->id);
      continue;
    }
    handled_tensors.push_back(in->id);
    auto tensor_ptr = tpipe.GetTensor(in->id);
    GE_CHK_BOOL_EXEC(tensor_ptr != nullptr, return false, "tensor_ptr nullptr");
    auto t = *tensor_ptr;
    if (this->unit == af::ComputeUnit::kUnitVector) {
      BoolType ret = this->WaitShareInputs(tpipe, in, t, ss);
      GE_CHK_BOOL_RET_SPECIAL_STATUS(ret == BoolType::FAILED, false, "Func WaitShareInputs return FAILED");
      if (ret == BoolType::TRUE) {
        continue;
      }
      GE_CHK_BOOL_RET_SPECIAL_STATUS(!this->WaitInputVector(tpipe, in, t, ss), false,
                                     "Func WaitInputVector return false");
    } else if (this->unit == af::ComputeUnit::kUnitMTE2 && (t.que_id != tpipe.cube_output_que_id)) {
      GE_CHK_BOOL_RET_SPECIAL_STATUS(!this->WaitInputMte(tpipe, in, t, ss), false, "Func WaitInputMte return false");
    }
  }
  return true;
}

static bool IsInplaceOutput(const ApiCall &call, const ApiTensor &tensor) {
  for (auto in : call.inputs) {
    if (in->reuse_next == &tensor) {
      return true;
    }
  }
  return false;
}

// 被复用的 buffer 是否还有 Store(MTE3) 读者。向量算子覆盖写该 buffer 前需等待其读完。
static bool HasMte3StoreReader(const ApiTensor &reuse_from) {
  return std::any_of(reuse_from.reads.begin(), reuse_from.reads.end(),
                     [](const ApiCall *read) { return read != nullptr && read->type == Store::Type; });
}

// 复用若通过 FreeTensor->AllocTensor 的 que 途径完成（复用来源为 VecOut），que 的 free 事件
// 已保证 MTE3 读者完成；仅原地覆盖（TBuf 复用、复用来源非 VecOut 的 que 复用）需要显式同步。
static bool IsQueProtectedReuse(const TPipe &tpipe, const ApiTensor &out, const Tensor &t) {
  if (t.alloc_type != af::AllocType::kAllocTypeQueue || out.reuse_from == nullptr) {
    return false;
  }
  auto reused = tpipe.GetTensor(out.reuse_from->id);
  return reused != nullptr && reused->position == af::Position::kPositionVecOut;
}

Status ApiCall::HandleVecOutAlloc(const TPipe &tpipe, const ApiTensor &out, const Tensor &t, std::stringstream &ss,
                                  bool with_define) const {
  if (IsInplaceOutput(*this, out)) {
    return af::SUCCESS;
  }
  if (out.reuse_from == nullptr) {
    auto t_que = tpipe.GetQue(t.que_id);
    GE_CHK_BOOL_RET_STATUS(t_que != nullptr, af::FAILED, "Codegen que[%ld] not found", t.que_id);
    ss << t_que->AllocBuf();
    return af::SUCCESS;
  }
  if (out.reuse_from->write->unit == af::ComputeUnit::kUnitVector) {
    auto tensor_ptr = tpipe.GetTensor(out.reuse_from->id);
    GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, af::FAILED, "Check[Param] tensor_ptr is nullptr");
    if (tensor_ptr->position == af::Position::kPositionVecOut) {
      auto t_que = tpipe.GetQue(t.que_id);
      GE_CHK_BOOL_RET_STATUS(t_que != nullptr, af::FAILED, "Codegen que[%ld] not found", t.que_id);
      ss << t_que->AllocBuf(with_define);
    } else {
      ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;
    }
    return af::SUCCESS;
  }
  if (out.reuse_from->write->unit == af::ComputeUnit::kUnitMTE2) {
    auto t_que = tpipe.GetQue(t.que_id);
    GE_CHK_BOOL_RET_STATUS(t_que != nullptr, af::FAILED, "Codegen que[%ld] not found", t.que_id);
    ss << t_que->AllocBuf(with_define);
  }
  return af::SUCCESS;
}

Status ApiCall::AllocOutputs(const TPipe &tpipe, std::stringstream &ss) const {
  for (auto &out : this->outputs) {
    auto tensor_ptr = tpipe.GetTensor(out.id);
    GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, af::FAILED, "Check[Param] tensor_ptr is nullptr");
    auto t = *tensor_ptr;

    // 向量算子以原地覆盖方式复用某 buffer 时，若该 buffer 仍有 Store(MTE3) 读者未完成，
    // 覆盖写之前需要插入 MTE3->V 同步，避免 Store 读取到被改写的 buffer。
    if (this->unit == af::ComputeUnit::kUnitVector && out.reuse_from != nullptr &&
        HasMte3StoreReader(*out.reuse_from) && !IsQueProtectedReuse(tpipe, out, t)) {
      ss << tpipe.SyncMte3ToVec(t) << std::endl;
    }

    if (t.alloc_type == af::AllocType::kAllocTypeBuffer) {
      if (out.reuse_from != nullptr) {
        ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;
      }
      ss << tpipe.TensorActualSizeCalc(t.id);
      std::string tmp;
      if (!t.no_need_realloc) {
        GE_CHK_STATUS_RET(tpipe.TensorAlloc(t, tmp), "Codegen alloc tensor failed");
      }
      ss << tmp;
      continue;
    }
    if (t.alloc_type != af::AllocType::kAllocTypeQueue) {
      continue;
    }
    bool with_define = IsReuseWithDefine(out);
    BoolType ret = this->AllocShareOutputs(tpipe, out, t, ss);
    GE_CHK_BOOL_RET_STATUS(ret != BoolType::FAILED, af::FAILED, "AllocShareOutputs return BoolType::FAILED");
    if (ret == BoolType::TRUE) {
      GELOGI("tensor id %ld alloc with share", out.id);
    } else if (this->unit == af::ComputeUnit::kUnitMTE2 && t.position == af::Position::kPositionVecIn) {
      if (out.reuse_from != nullptr) {
        auto t_que = tpipe.GetQue(t.que_id);
        GE_CHK_BOOL_RET_STATUS(t_que != nullptr, af::FAILED, "Codegen que[%ld] not found", t.que_id);
        ss << t_que->AllocBuf(with_define);
      } else {
        auto t_que = tpipe.GetQue(t.que_id);
        GE_CHK_BOOL_RET_STATUS(t_que != nullptr, af::FAILED, "Codegen que[%ld] not found", t.que_id);
        ss << t_que->AllocBuf();
      }
    } else if (this->unit == af::ComputeUnit::kUnitVector && t.position == af::Position::kPositionVecOut) {
      GE_CHK_BOOL_RET_STATUS(HandleVecOutAlloc(tpipe, out, t, ss, with_define) == af::SUCCESS, af::FAILED,
                             "HandleVecOutAlloc failed");
    } else if (this->unit == af::ComputeUnit::kUnitVector && t.position == af::Position::kPositionVecCalc) {
      if (out.reuse_from == nullptr) {
        // vec ..> store
        // vec ..> vec ..> store
        auto t_que = tpipe.GetQue(t.que_id);
        GE_CHK_BOOL_RET_STATUS(t_que != nullptr, af::FAILED, "Codegen que[%ld] not found", t.que_id);
        ss << t_que->AllocBuf();
      } else {
        auto tensor_ptr = tpipe.GetTensor(out.reuse_from->id);
        GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, af::FAILED, "Check[Param] tensor_ptr is nullptr");
        if (tensor_ptr->position == af::Position::kPositionVecOut) {
          // vec -> store -> free ..> alloc -> vec
          auto t_que = tpipe.GetQue(t.que_id);
          GE_CHK_BOOL_RET_STATUS(t_que != nullptr, af::FAILED, "Codegen que[%ld] not found", t.que_id);
          ss << t_que->AllocBuf(with_define);
        } else {
          // load ..> vec
          // vec ..> vec
          ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;
        }
      }
    } else {
      std::runtime_error("Unsupported case.");
    }
    ss << tpipe.TensorActualSizeCalc(t.id);
    std::string tmp;
    if (!(t.alloc_type == af::AllocType::kAllocTypeBuffer && t.no_need_realloc)) {
      GE_CHK_STATUS_RET(tpipe.TensorAlloc(t, tmp), "Codegen alloc tensor failed");
    }
    ss << tmp;
  }  // end for
  return af::SUCCESS;
}

bool ApiCall::SyncOutputs(const TPipe &tpipe, std::stringstream &ss) const {
  for (auto out : this->outputs) {
    auto tensor_ptr = tpipe.GetTensor(out.id);
    GE_CHK_BOOL_EXEC(tensor_ptr != nullptr, return false, "tensor_ptr nullptr");
    auto t = *tensor_ptr;
    if (t.alloc_type == af::AllocType::kAllocTypeQueue) {
      if (IsLastShare(out)) {  // 非共用 或者 最后一个共用
        if (t.position == af::Position::kPositionVecIn || t.position == af::Position::kPositionVecOut) {
          auto t_que = tpipe.GetQue(t.que_id);
          GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, false, "Codegen que[%ld] not found", t.que_id);
          ss << t_que->EnqueBuf();
        }
      }
    }
  }
  return true;
}

bool ApiCall::IsReadOutersideWrite(ascir::AxisId &target_id) const {
  uint64_t count = 0;
  uint64_t total_count = 0;
  int64_t prev_depth = INT64_MAX;
  for (auto output : outputs) {
    for (auto read : output.reads) {
      total_count++;
      if (read->depth < depth) {
        count++;
        target_id = read->depth < prev_depth ? read->axis : target_id;
        prev_depth = read->depth < prev_depth ? read->depth : prev_depth;
      }
    }
  }
  return count == total_count;
}

static bool IsLastRead(const ApiCall &call, const ApiTensor &tensor) {
  return !tensor.reads.empty() && tensor.reads.back() == &call;
}

bool ApiCall::FreeInputs(const TPipe &tpipe, std::stringstream &ss) const {
  std::vector<ascir::QueId> freed_que;
  for (auto in : this->inputs) {
    if (!IsLastRead(*this, *in)) {
      continue;
    }
    auto tensor_ptr = tpipe.GetTensor(in->id);
    GE_CHK_BOOL_EXEC(tensor_ptr != nullptr, return false, "tensor_ptr nullptr");
    auto t = *tensor_ptr;
    auto reuse_next = in->reuse_next == nullptr ? nullptr : tpipe.GetTensor(in->reuse_next->id);
    auto it = find(freed_que.begin(), freed_que.end(), t.que_id);
    if (it != freed_que.end()) {
      continue;
    }
    if (!IsLastShare(*in)) {
      continue;
    }
    if (t.alloc_type == af::AllocType::kAllocTypeQueue) {
      if (t.que_id == tpipe.cube_output_que_id) {
        continue;
      }
      auto t_que = tpipe.GetQue(t.que_id);
      if (reuse_next == nullptr) {
        // 1 alloc -> load ..> vec ..> vec ..> vec -> free
        // 2 alloc -> vec ..> vec ..> store -> free
        // 3 alloc -> vec ..> store -> free -> alloc ..> vec -> free
        GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, false, "Codegen que[%ld] not found", t.que_id);
        ss << t_que->FreeBuf();
        freed_que.push_back(t.que_id);
      } else if (t.position == af::Position::kPositionVecOut) {
        // vec ..> store -> free -> alloc -> vec
        GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, false, "Codegen que[%ld] not found", t.que_id);
        ss << t_que->FreeBuf();
        freed_que.push_back(t.que_id);
      } else if (reuse_next != nullptr && reuse_next->position == af::Position::kPositionVecIn) {
        // alloc -> load -> vec -> free ..> alloc -> load
        GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, false, "Codegen que[%ld] not found", t.que_id);
        ss << t_que->FreeBuf();
        freed_que.push_back(t.que_id);
      }
    }
  }
  return true;
}

bool ApiCall::FreeUnusedOutputs(const TPipe &tpipe, std::stringstream &ss) const {
  std::vector<ascir::QueId> freed_que;
  for (auto out : this->outputs) {
    if (out.reads.size() != 0) {
      continue;
    }
    auto tensor_ptr = tpipe.GetTensor(out.id);
    GE_CHK_BOOL_EXEC(tensor_ptr != nullptr, return false, "tensor_ptr nullptr");
    auto t = *tensor_ptr;
    auto it = find(freed_que.begin(), freed_que.end(), t.que_id);
    if (it != freed_que.end()) {
      continue;
    }
    if (t.alloc_type == af::AllocType::kAllocTypeQueue) {
      if (out.reuse_next == nullptr && IsLastShare(out)) {
        auto t_que = tpipe.GetQue(t.que_id);
        GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, false, "Codegen que[%ld] not found", t.que_id);
        ss << t_que->FreeBuf();
        freed_que.push_back(t.que_id);
      } else if (t.position == af::Position::kPositionVecOut) {
        auto t_que = tpipe.GetQue(t.que_id);
        GE_CHK_BOOL_RET_SPECIAL_STATUS(t_que == nullptr, false, "Codegen que[%ld] not found", t.que_id);
        ss << t_que->FreeBuf();
        freed_que.push_back(t.que_id);
      }
    }
  }
  return true;
}
Status ApiCall::BuildApiParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                              const std::vector<std::reference_wrapper<const Tensor>> &input,
                              const std::vector<std::reference_wrapper<const Tensor>> &output) const {
  (void)tpipe;
  (void)current_axis;
  (void)input;
  (void)output;
  CodegenApiParamPtr api_param = af::ComGraphMakeShared<CodegenApiParam>();
  GE_CHK_STATUS_RET(CodegenApiParam::Register(this->node, api_param));
  return af::SUCCESS;
}

Status ApiCall::Generate(const TPipe &tpipe, const vector<ascir::AxisId> &current_axis,
                         const vector<std::reference_wrapper<const Tensor>> &input,
                         const vector<std::reference_wrapper<const Tensor>> &output, string &result) const {
  GE_CHK_STATUS_RET(BuildApiParam(tpipe, current_axis, input, output),
                    "BuildApiParam failed, graph name: %s, node name: %s", graph_name.c_str(), node_name.c_str());
  GE_CHK_STATUS_RET(GenerateApiCallString(tpipe, result), "GenerateApiCallString failed, graph name: %s, node name: %s",
                    graph_name.c_str(), node_name.c_str());
  return af::SUCCESS;
}

bool ApiCall::IsUnitLastRead(const ApiTensor &tensor) const {
  for (int32_t i = tensor.reads.size() - 1; i >= 0; i--) {
    if (tensor.reads[i]->unit == this->unit) {
      return (tensor.reads[i] == this);
    }
  }
  return false;
}

af::Status ApiCall::RegisterBasicDumpParam(const std::string &api_name,
                                           const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                           const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                           const CombinedExpression &cal_count, const std::string &tmp_buf_name) const {
  auto api_param = std::make_shared<CodegenApiParam>();
  api_param->api_name = api_name;

  for (const auto &ref : outputs) {
    const auto &t = ref.get();
    api_param->output_params.emplace_back(t.name, true, CombinedExprFactory::Constant(0));
  }
  for (const auto &ref : inputs) {
    const auto &t = ref.get();
    api_param->input_params.emplace_back(t.name, !t.is_constant, CombinedExprFactory::Constant(0));
  }
  if (!tmp_buf_name.empty()) {
    api_param->tmp_buf_name = tmp_buf_name;
  }
  if (!cal_count.IsEmpty()) {
    api_param->cal_count = cal_count;
  }

  GE_CHK_STATUS_RET(CodegenApiParam::Register(this->node, api_param));
  return af::SUCCESS;
}

af::Status ApiCall::SetSemanticFeature(const std::string &semantic_feature) const {
  GE_ASSERT_NOTNULL(this->node);
  auto api_param = CodegenApiParam::TryGetNodeApiParam(this->node);
  if (api_param == nullptr) {
    return af::SUCCESS;
  }
  api_param->semantic_feature = semantic_feature;
  return af::SUCCESS;
}

namespace {
static void GenTemplateParams(const CodegenApiParam &api_param, stringstream &ss) {
  if (api_param.template_params.empty()) {
    return;
  }
  ss << "<";
  bool first = true;
  for (const auto &template_param : api_param.template_params) {
    if (first) {
      first = false;
    } else {
      ss << ", ";
    }
    ss << template_param;
  }
  ss << ">";
}

static void GenOuterLoopAxesPreProcess(const CodegenApiParam &api_param, const Tiler &tiler, stringstream &ss) {
  if (api_param.outer_loop_axes.empty()) {
    return;
  }
  for (size_t i = 0; i < api_param.outer_loop_axes.size(); i++) {
    std::string loop_iter = "outer_for_" + std::to_string(i);
    ss << "for (int " << loop_iter << " = 0; " << loop_iter << " < " << api_param.outer_loop_axes[i].ToStr(tiler)
       << "; " << loop_iter << "++) {" << std::endl;
  }
}

static void GenOuterLoopAxesPostProcess(const CodegenApiParam &api_param, stringstream &ss) {
  if (api_param.outer_loop_axes.empty()) {
    return;
  }
  for (size_t i = 0; i < api_param.outer_loop_axes.size(); i++) {
    ss << "}" << std::endl;
  }
}

static void GenApiCallCommon(const CodegenApiParam &api_param, const Tiler &tiler, stringstream &ss) {
  ss << api_param.api_name;
  GenTemplateParams(api_param, ss);
  ss << "(";
  for (const auto &output : api_param.output_params) {
    ss << output.name << (output.is_tensor ? "[" + output.offset.ToStr(tiler) + "]" : "") << ", ";
  }
  for (const auto &input : api_param.input_params) {
    ss << input.name << (input.is_tensor ? "[" + input.offset.ToStr(tiler) + "]" : "") << ", ";
  }
  if (!api_param.tmp_buf_name.empty()) {
    ss << api_param.tmp_buf_name << ", ";
  }
}

static void GenApiCallPreProcess(const CodegenApiParam &api_param, stringstream &ss) {
  if (api_param.api_pre_process.empty()) {
    return;
  }
  for (const auto &pre_process : api_param.api_pre_process) {
    ss << pre_process;
  }
}

static void GenApiCallPostProcess(const CodegenApiParam &api_param, stringstream &ss) {
  if (api_param.api_post_process.empty()) {
    return;
  }
  for (const auto &post_process : api_param.api_post_process) {
    ss << post_process;
  }
}
}  // namespace

Status ApiCall::GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler, stringstream &ss) const {
  ss << api_param.cal_count.ToStr(tiler) << ");" << std::endl;
  return af::SUCCESS;
}

Status ApiCall::GenerateApiCallString(const TPipe &tpipe, std::string &result) const {
  auto api_param = CodegenApiParam::GetNodeApiParam(this->node);
  GE_ASSERT_NOTNULL(api_param, "ApiParam of graph %s node %s is null", graph_name.c_str(), node_name.c_str());
  stringstream ss;
  GenOuterLoopAxesPreProcess(*api_param, tpipe.tiler, ss);
  GenApiCallPreProcess(*api_param, ss);
  GenApiCallCommon(*api_param, tpipe.tiler, ss);
  GE_CHK_STATUS_RET(GenDimensionParam(*api_param, tpipe.tiler, ss),
                    "GenDimensionParam failed, graph name: %s, node name: %s", graph_name.c_str(), node_name.c_str());
  GenApiCallPostProcess(*api_param, ss);
  GenOuterLoopAxesPostProcess(*api_param, ss);
  result = ss.str();
  return af::SUCCESS;
}
