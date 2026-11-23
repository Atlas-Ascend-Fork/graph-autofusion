/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "codegen_kernel_utils.h"
#include <algorithm>
#include <utility>
#include "common/checker.h"
#include "common/ub_expr/asc_graph_ub_expr_builder.h"

using namespace std;
using namespace codegen;

// ===== CommonQueueSplit 候选规划 =====

bool IsUnconditionalCall(const ApiCall &call) {
  return call.node != nullptr && !call.enable_cache && !call.skip_api_emit &&
         call.api_call_context.stage == ComputeStage::kDefault && call.node->GetInControlNodesSize() == 0U &&
         call.node->GetOutControlNodesSize() == 0U;
}

bool IsSingleQueueTensor(const TQue &queue, const Tensor &tensor) {
  const auto group = queue.share_group.find(tensor.reuse_id);
  return group != queue.share_group.end() && group->second.size() == 1U && group->second.front() == tensor.id;
}

// GetCommonQueueSplitAnchor 定义保留在 codegen_kernel_loop.cpp(圈复杂度超标项, 原样未拆),
// 此处经 codegen_kernel_utils.h 声明调用

bool HasIndependentQueueLifetime(const Loop &loop, size_t begin, size_t end, const ApiCall &load, const TPipe &tpipe) {
  const auto *tensor = tpipe.GetTensor(load.outputs[0].id);
  for (size_t i = begin; i < end; ++i) {
    const auto &body = loop.bodys[i];
    if (body.type != LoopType::CALL || !IsUnconditionalCall(*body.call) ||
        body.call->unit == af::ComputeUnit::kUnitMTE3) {
      return false;
    }
    for (const auto &output : body.call->outputs) {
      const auto *other = tpipe.GetTensor(output.id);
      if (other == nullptr ||
          (other->alloc_type == af::AllocType::kAllocTypeQueue && other->que_id == tensor->que_id)) {
        return false;
      }
    }
  }
  return true;
}

std::map<const ApiCall *, const ApiCall *> GetCommonQueueSplitCandidates(const Loop &loop, const TPipe &tpipe) {
  std::map<const ApiCall *, const ApiCall *> plan;
  if (tpipe.cv_fusion_type != ascir::CubeTemplateType::kCommon || tpipe.is_conv2d ||
      tpipe.cube_output_tensor_id == af::kIdNone || loop.is_graph_has_reduce_node) {
    return plan;
  }
  std::set<ascir::QueId> used_queues;
  for (size_t i = 0; i < loop.bodys.size(); ++i) {
    const auto &body = loop.bodys[i];
    if (body.type != LoopType::CALL || loop.used_calls.count(body.call) != 0U) {
      continue;
    }
    const auto *anchor = GetCommonQueueSplitAnchor(*body.call, tpipe);
    if (anchor == nullptr) {
      continue;
    }
    if (std::any_of(body.call->outputs[0].reads.begin(), body.call->outputs[0].reads.end(),
                    [&loop](const ApiCall *reader) {
                      return std::none_of(loop.bodys.begin(), loop.bodys.end(), [reader](const LoopBody &entry) {
                        return entry.type == LoopType::CALL && entry.call == reader;
                      });
                    })) {
      continue;
    }
    const auto *tensor = tpipe.GetTensor(body.call->outputs[0].id);
    if (used_queues.count(tensor->que_id) != 0U) {
      continue;
    }
    const auto *source = body.call->outputs[0].reuse_from->write;
    bool source_seen = false;
    for (size_t j = 0; j < i; ++j) {
      source_seen = source_seen || (loop.bodys[j].type == LoopType::CALL && loop.bodys[j].call == source);
      if (source_seen && loop.bodys[j].type == LoopType::CALL && loop.bodys[j].call == anchor &&
          HasIndependentQueueLifetime(loop, j, i, *body.call, tpipe)) {
        plan.emplace(anchor, body.call);
        used_queues.insert(tensor->que_id);
        break;
      }
    }
  }
  return plan;
}

void CollectCommonQueueSplitCandidates(const Loop &loop, const TPipe &tpipe, std::vector<const ApiCall *> &loads) {
  const auto candidates = GetCommonQueueSplitCandidates(loop, tpipe);
  for (const auto &candidate : candidates) {
    loads.push_back(candidate.second);
  }
  for (const auto &body : loop.bodys) {
    if (body.type == LoopType::LOOP) {
      CollectCommonQueueSplitCandidates(*body.loop, tpipe, loads);
    }
  }
}

bool BuildCommonQueueUbGuard(const TPipe &tpipe, const af::AscGraph &graph, CommonQueueSplitPlan &plan) {
  ascir::UbExprContext context;
  if (ascir::AscGraphUbExprBuilder().Build(graph, context) != af::SUCCESS || !context.ub_expr.IsValid()) {
    return false;
  }
  std::set<std::string> known_containers;
  for (const auto &entry : tpipe.ques) {
    known_containers.insert("q" + std::to_string(entry.first) + "_size");
  }
  for (const auto &entry : tpipe.bufs) {
    known_containers.insert("b" + std::to_string(entry.first) + "_size");
  }
  std::set<std::string> actual_containers;
  for (const auto &entry : context.container_names) {
    if (known_containers.count(entry.second) == 0U) {
      return false;
    }
    actual_containers.insert(entry.second);
  }
  if (actual_containers != known_containers) {
    return false;
  }
  std::vector<std::pair<af::Expression, af::Expression>> replacements;
  for (const auto &symbol : context.ub_expr.FreeSymbols()) {
    const auto name = context.container_names.find(symbol);
    if (name == context.container_names.end()) {
      return false;
    }
    replacements.emplace_back(symbol, af::Symbol((plan.QueueAliasName() + "_" + name->second).c_str()));
  }
  plan.ub_containers.assign(actual_containers.begin(), actual_containers.end());
  plan.original_ub_expr = context.ub_expr.Subs(replacements).Simplify().Str().get();
  return true;
}

std::unique_ptr<TPipe> MakeCommonMmQueuePipe(const TPipe &tpipe) {
  if (tpipe.common_queue_split_plan == nullptr) {
    return nullptr;
  }
  const auto &plan = *tpipe.common_queue_split_plan;
  const auto *queue = tpipe.GetQue(plan.queue_id);
  if (queue == nullptr) {
    return nullptr;
  }
  auto view = std::make_unique<TPipe>(tpipe);
  TQue alias(*queue, plan.QueueAliasName(), plan.BufferAliasName());
  view->ques.erase(queue->id);
  view->ques.emplace(queue->id, alias);
  return view;
}

// ===== TQue 规模表达式与定义发射 =====

// 单个 TQue 的 size/buf_num 常量定义与 TQue 对象声明发射: UBFuse 只声明常量(buf句柄由
// CV 路径管理); 非 ATT 侧计算 qbt size 时声明常量+TQue 对象; ATT 侧计算时 size 常量降级
// 为注释, buf_num 与 TQue 对象照常声明
void GenLocalTQueDefine(std::stringstream &ss, const TQue &que, std::string tensor_size_max,
                        std::string tensor_bufnum_max, bool is_ub_fuse, bool using_att_calc_qbt_size) {
  if (is_ub_fuse) {
    ss << que.size.DefineConst(std::move(tensor_size_max)) << std::endl;
    ss << que.buf_num.DefineConst(std::move(tensor_bufnum_max)) << std::endl;
  } else if (!using_att_calc_qbt_size) {
    ss << que.size.DefineConst(std::move(tensor_size_max)) << std::endl;
    ss << que.buf_num.DefineConst(std::move(tensor_bufnum_max)) << std::endl;
    ss << que.Define() << std::endl;
  } else {
    ss << "// " << que.size.DefineConst(std::move(tensor_size_max)) << std::endl;
    ss << que.buf_num.DefineConst(std::move(tensor_bufnum_max)) << std::endl;
    ss << que.Define() << std::endl;
  }
}

// 单个TQue的Max规模表达式中合并域项: 各merge scope的size与buf_num并列参与Max,
// 并与后续非合并张量项共用首项分隔状态is_first
Status AppendMergeScopeMaxTerms(const TPipe &tpipe, const TQue &que, std::stringstream &tensor_size_max,
                                std::stringstream &tensor_bufnum_max, bool &is_first) {
  for (auto mid : que.merge_scopes) {
    auto merge_scope = tpipe.merge_scopes.find(mid);
    GE_ASSERT_TRUE(merge_scope != tpipe.merge_scopes.end(), "Codegen merge scope not found:%ld", mid);
    if (is_first) {
      is_first = false;
    } else {
      tensor_size_max << ", ";
      tensor_bufnum_max << ", ";
    }
    tensor_size_max << merge_scope->second.size;
    tensor_bufnum_max << merge_scope->second.buf_num;
  }
  return af::SUCCESS;
}

// 单个TQue的Max规模表达式中非合并张量项: size项为 张量大小 * sizeof(dtype) 并列参与Max;
// buf_num不并列各项, 仅追加一个各项que_buf_num_value的最大值项
Status AppendNotMergeTensorMaxTerms(const TPipe &tpipe, const TQue &que, std::stringstream &tensor_size_max,
                                    std::stringstream &tensor_bufnum_max, bool &is_first) {
  uint32_t tensor_buf_num_max_val = 0;
  for (auto tid : que.not_merge_tensors) {
    auto tensor = tpipe.tensors.find(tid);
    GE_ASSERT_TRUE(tensor != tpipe.tensors.end(), "Codegen tensor not found:%ld", tid);
    if (is_first) {
      is_first = false;
    } else {
      tensor_size_max << ", ";
    }
    std::string dtype_name;
    GE_CHK_STATUS_RET(Tensor::DtypeName(tensor->second.dtype, dtype_name), "Codegen get data type:%d failed",
                      static_cast<int32_t>(tensor->second.dtype));
    tensor_size_max << tensor->second.size << " * sizeof(" << dtype_name << ")";
    tensor_buf_num_max_val = std::max(tensor_buf_num_max_val, tensor->second.que_buf_num_value);
  }
  tensor_bufnum_max << (que.merge_scopes.empty() ? "" : ", ") << tensor_buf_num_max_val;
  return af::SUCCESS;
}

// 单个TQue的Max规模表达式中共享组项: 同que内reuse id相同的多个张量, size求和后作为
// 一个并列项参与Max; UBFuse场景求和项再按BlkAlign对齐。仅有1个tensor的组无需合并计算
Status AppendShareGroupMaxTerms(const TPipe &tpipe, const TQue &que, std::stringstream &tensor_size_max) {
  for (auto share_tensors : que.share_group) {
    if (share_tensors.second.size() <= 1) {  // 仅有1个tensor 不需要合并计算
      continue;
    }
    // 此处肯定不是第一个size
    tensor_size_max << ", ";
    // 多个tensor在一个que中且reuse id相同，需要计算reuse id相同这一组所有tensor size的综合
    bool is_first_share = true;
    for (auto tid : share_tensors.second) {
      auto tensor = tpipe.tensors.find(tid);
      GE_ASSERT_TRUE(tensor != tpipe.tensors.end(), "Codegen share tensor not found:%ld", tid);
      if (is_first_share) {
        is_first_share = false;
      } else {
        tensor_size_max << " + ";
      }
      std::string dtype_name;
      GE_CHK_STATUS_RET(Tensor::DtypeName(tensor->second.dtype, dtype_name), "Codegen get data type:%d failed",
                        static_cast<int32_t>(tensor->second.dtype));
      std::string tensor_byte_size = tensor->second.size.Str() + " * sizeof(" + dtype_name + ")";
      if (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
        tensor_byte_size = "KernelUtils::BlkAlign<uint8_t>(" + tensor_byte_size + ")";
      }
      tensor_size_max << tensor_byte_size;
    }
  }
  return af::SUCCESS;
}

// ===== ApiCall 输出分配子路径 =====

// AllocOutputs子路径: TBuf类输出原地分配——复用TBuf时先加PIPE_V屏障; no_need_realloc张量跳过TensorAlloc
Status AllocBufferOutput(const TPipe &tpipe, const ApiTensor &out, const Tensor &t, std::stringstream &ss) {
  if (out.reuse_from != nullptr) {
    ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;
  }
  ss << tpipe.TensorActualSizeCalc(t.id);
  std::string tmp;
  if (!t.no_need_realloc) {
    GE_CHK_STATUS_RET(tpipe.TensorAlloc(t, tmp), "Codegen alloc tensor failed");
  }
  ss << tmp;
  return af::SUCCESS;
}

// AllocOutputs子路径: Vector算子VecCalc位置输出, 按复用来源分场景——无复用与复用自VecOut
// 均常规AllocBuf; 复用自其它位置为同pipe原地覆盖, 仅加PIPE_V屏障
Status AllocVecCalcOutput(const TPipe &tpipe, const ApiTensor &out, const Tensor &t, std::stringstream &ss) {
  if (out.reuse_from == nullptr) {
    // vec ..> store
    // vec ..> vec ..> store
    auto t_que = tpipe.GetQue(t.que_id);
    GE_CHK_BOOL_RET_STATUS(t_que != nullptr, af::FAILED, "Codegen que[%ld] not found", t.que_id);
    ss << t_que->AllocBuf();
    return af::SUCCESS;
  }
  auto tensor_ptr = tpipe.GetTensor(out.reuse_from->id);
  GE_CHK_BOOL_RET_STATUS(tensor_ptr != nullptr, af::FAILED, "Check[Param] tensor_ptr is nullptr");
  if (tensor_ptr->position == af::Position::kPositionVecOut) {
    // vec -> store -> free ..> alloc -> vec
    auto t_que = tpipe.GetQue(t.que_id);
    GE_CHK_BOOL_RET_STATUS(t_que != nullptr, af::FAILED, "Codegen que[%ld] not found", t.que_id);
    ss << t_que->AllocBuf();
  } else {
    // load ..> vec
    // vec ..> vec
    ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;
  }
  return af::SUCCESS;
}
