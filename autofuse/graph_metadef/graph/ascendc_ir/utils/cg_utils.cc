/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/cg_utils.h"
#include "graph/utils/node_utils_ex.h"
#include "graph/utils/graph_utils_ex.h"
#include "ascendc_ir/core/ascendc_ir_impl.h"
#include "framework/common/debug/ge_log.h"
namespace af {
namespace ascir {
namespace cg {
namespace {
thread_local std::weak_ptr<CgContext> t_context;
int64_t GenNextId(const ComputeGraphPtr &graph, const std::string &key) {
  if (graph == nullptr) {
    throw std::invalid_argument("Invalid graph");
  }
  auto id = graph->GetExtAttr<int64_t>(key);
  if (id == nullptr) {
    graph->SetExtAttr(key, static_cast<int64_t>(1));
    return 0;
  }
  return (*id)++;
}

int64_t GenNextId(const Operator &op, const std::string &key) {
  auto node = NodeUtilsEx::GetNodeFromOperator(op);
  GE_ASSERT_NOTNULL(node);
  return GenNextId(node->GetOwnerComputeGraph(), key);
}
}  // namespace

CgContext *CgContext::GetThreadLocalContext() {
  return GetSharedThreadLocalContext().get();
}
std::shared_ptr<CgContext> CgContext::GetSharedThreadLocalContext() {
  return t_context.lock();
}
void CgContext::SetThreadLocalContext(const std::shared_ptr<CgContext> &context) {
  t_context = context;
}
const std::vector<Axis> &CgContext::GetLoopAxes() const {
  return loop_axes_;
}
void CgContext::SetLoopAxes(std::vector<Axis> axes) {
  loop_axes_ = std::move(axes);
  loop_axis_ids_cache_.clear();
  loop_axis_ids_cache_.reserve(loop_axes_.size());
  for (const auto &axis : loop_axes_) {
    loop_axis_ids_cache_.emplace_back(axis.id);
  }
}
void CgContext::PushLoopAxis(const Axis &axis) {
  loop_axes_.emplace_back(axis);
  loop_axis_ids_cache_.emplace_back(axis.id);
}
void CgContext::PopBackLoopAxis(const Axis &axis) {
  if (loop_axis_ids_cache_.empty()) {
    GELOGE(FAILED, "Axes stack is empty");
    return;
  }
  auto last_id = *(loop_axis_ids_cache_.rbegin());
  if (last_id != axis.id) {
    GELOGE(FAILED, "Pop axis order mismatch");
    return;
  }
  loop_axis_ids_cache_.pop_back();
  loop_axes_.pop_back();
}
const std::vector<AxisId> &CgContext::GetLoopAxisIds() const {
  return loop_axis_ids_cache_;
}
void CgContext::SetBlockLoopEnd(AxisId id) {
  block_loop_end_ = id;
}
AxisId CgContext::GetBlockLoopEnd() const {
  return block_loop_end_;
}
void CgContext::SetVectorizedLoopEnd(AxisId id) {
  vectorized_loop_end_ = id;
}
AxisId CgContext::GetVectorizedLoopEnd() const {
  return vectorized_loop_end_;
}
void CgContext::SetLoopEnd(AxisId id) {
  loop_end_ = id;
}
AxisId CgContext::GetLoopEnd() const {
  return loop_end_;
}
void CgContext::SetOption(const LoopOption &option) {
  option_ = option;
}
const LoopOption &CgContext::GetOption() const {
  return option_;
}
LoopGuard::~LoopGuard() {
  context_->PopBackLoopAxis(axis_);
}
LoopGuard::LoopGuard(const Axis &axis) {
  context_ = CgContext::GetSharedThreadLocalContext();
  if (context_ == nullptr) {
    context_ = std::make_shared<CgContext>();
    CgContext::SetThreadLocalContext(context_);
  }

  axis_ = axis;
  context_->PushLoopAxis(axis_);
}
std::unique_ptr<LoopGuard> LoopGuard::Create(const Axis &axis, const LoopOption &option) {
  auto loop_guard = ComGraphMakeUnique<LoopGuard>(axis);
  loop_guard->context_->SetOption(option);
  return loop_guard;
}

int64_t CodeGenUtils::GenNextExecId(const Operator &op) {
  static const std::string kExecIdKey = "cg.ExecId";
  return GenNextId(op, kExecIdKey);
}

int64_t CodeGenUtils::GenNextContainerId(const Operator &op) {
  static const std::string kContainerIdKey = "cg.ContainerId";
  return GenNextId(op, kContainerIdKey);
}

int64_t CodeGenUtils::GenNextReuseId(const Operator &op) {
  static const std::string kReuseIdKey = "cg.ReuseId";
  return GenNextId(op, kReuseIdKey);
}

int64_t CodeGenUtils::GenNextTensorId(const Operator &op) {
  static const std::string kTensorIdKey = "cg.TensorId";
  return GenNextId(op, kTensorIdKey);
}

int64_t CodeGenUtils::GenNextExecId(const AscGraph &graph) {
  // impl_ is always valid
  return GenNextExecId(graph.impl_->compute_graph_);
}

int64_t CodeGenUtils::GenNextExecId(const ComputeGraphPtr &graph) {
  static const std::string kExecIdKey = "cg.ExecId";
  return GenNextId(graph, kExecIdKey);
}
}  // namespace cg
}  // namespace ascir
}  // namespace af
