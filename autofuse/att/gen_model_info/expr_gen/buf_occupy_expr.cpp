/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "buf_occupy_expr.h"
#include "arg_list_manager.h"
#include "api_perf_register/ascendc_api_perf.h"

namespace att {
const std::unordered_map<HardwareDef, std::string> kScope2Str = {
    {HardwareDef::L1, "L1"},   {HardwareDef::UB, "UB"},
    {HardwareDef::L0A, "L0A"}, {HardwareDef::L0B, "L0B"},
    {HardwareDef::L0C, "L0C"}, {HardwareDef::BTBUF, "BTBUF"},
    {HardwareDef::GM, "GM"},   {HardwareDef::HARDWAREERR, "INVALID"}};

void BufOccupyExpr::SummaryBufferOccup(std::unordered_map<HardwareDef, Expr> &current_occup, const HardwareDef scope,
                                       Expr &new_occup) const {
  if (current_occup.find(scope) == current_occup.end()) {
    current_occup[scope] = new_occup;
  } else {
    auto &a = current_occup[scope];
    current_occup[scope] = af::sym::Add(a, new_occup);
  }
}

af::Status BufOccupyExpr::GetCoTensorSizeExpr(const std::vector<std::vector<TensorPtr>> &co_tensors, Expr &expr,
                                              const Expr &align) const {
  for (const auto &tensors : co_tensors) {
    Expr total_size;
    for (const auto &tensor : tensors) {
      Expr tensor_size_expr = ArgListManager::GetInstance().GetArgExpr(tensor->name);
      if (IsValid(align) && !(align == 1)) {
        tensor_size_expr = af::sym::Mul(af::sym::Ceiling(af::sym::Div(tensor_size_expr, align)), align);
      }
      GELOGD("Get tensor [%s] size : [%s]", tensor->name.c_str(), tensor_size_expr.Str().get());
      GE_ASSERT_TRUE(IsValid(tensor_size_expr), "Tensor [%s] has no expr.", tensor->name.c_str());
      if (IsValid(total_size)) {
        total_size = af::sym::Add(total_size, tensor_size_expr);
      } else {
        total_size = tensor_size_expr;
      }
    }
    if (!IsValid(expr)) {
      expr = total_size;
    } else {
      expr = af::sym::Max(expr, total_size);
    }
  }
  return af::SUCCESS;
}

af::Status BufOccupyExpr::GetOccupInContainer(ContainerPtr &container, Expr &occup_per_tensor,
                                              Expr &occup_total) const {
  std::set<TensorPtr> co_tensors;  // 收集所用有同存节点的tensor
  for (const auto &tensors : container->GetCoTensors()) {
    for (const auto &tensor : tensors) {
      co_tensors.insert(tensor);
    }
  }
  // 获取共存tensor size total
  GE_ASSERT_SUCCESS(GetCoTensorSizeExpr(container->GetCoTensors(), occup_per_tensor, container->align),
                    "Get tensor size failed.");
  for (const auto &tensor : container->allocated_tensors) {
    if (co_tensors.find(tensor) != co_tensors.end()) {
      continue;
    }
    // 对于单个container内的占用，取max
    Expr tensor_size_expr = ArgListManager::GetInstance().GetArgExpr(tensor->name);
    if (IsValid(container->align) && !(container->align == 1)) {
      tensor_size_expr =
          af::sym::Mul(af::sym::Ceiling(af::sym::Div(tensor_size_expr, container->align)), container->align);
    }
    GELOGD("Get tensor [%s] size : [%s]", tensor->name.c_str(), tensor_size_expr.Str().get());
    GE_ASSERT_TRUE(IsValid(tensor_size_expr), "Tensor [%s] has no expr.", tensor->name.c_str());
    if (IsValid(occup_per_tensor)) {
      occup_per_tensor = af::sym::Max(occup_per_tensor, tensor_size_expr);
    } else {
      occup_per_tensor = tensor_size_expr;
    }
  }
  if (container->alloc_type == af::AllocType::kAllocTypeBuffer) {
    GELOGD("trying to reuse temp buffer for tbuf %s, id %lld", container->name.c_str(), container->container_id);
    auto temp_buffer = tuning_space_->tmp_buffer.find(container->container_id);
    if (temp_buffer != tuning_space_->tmp_buffer.end()) {
      occup_per_tensor = af::sym::Max(occup_per_tensor, temp_buffer->second);
      GELOGD("reuse temp buffer for tbuf, buf id %lld, result buffer %s", container->container_id,
             Str(occup_per_tensor).c_str());
      tuning_space_->tmp_buffer.erase(container->container_id);
    }
  }
  // 最大tensor_size * buffer_num
  Expr buffer_num_expr = ArgListManager::GetInstance().GetArgExpr(container->name);
  occup_total = occup_per_tensor;
  if (IsValid(buffer_num_expr)) {
    occup_total = af::sym::Mul(occup_per_tensor, buffer_num_expr);
  }
  GELOGD("[DFX]Get container [%s] occupy : occup_per_tensor[%s], occup_total[%s]", container->name.c_str(),
         occup_per_tensor.Str().get(), occup_total.Str().get());
  return af::SUCCESS;
}

af::Status BufOccupyExpr::GetBufferOccupInContainer(std::unordered_map<HardwareDef, Expr> &buffer_occup,
                                                    std::map<std::string, Expr> &container_exprs) {
  for (auto &container : tuning_space_->containers) {
    Expr container_occup_expr;
    Expr occup_total;
    GE_ASSERT_SUCCESS(GetOccupInContainer(container, container_occup_expr, occup_total),
                      "Get container occupy failed.");
    container_exprs[container->name] = container_occup_expr;
    for (const auto &scope : container->buf_location) {
      SummaryBufferOccup(buffer_occup, scope, occup_total);
      GELOGD("Get scope [%d] name: [%s] occupy : [%s]", static_cast<int32_t>(scope), container->name.c_str(),
             buffer_occup[scope].Str().get());
    }
  }
  for (auto &pair : tuning_space_->tmp_buffer) {
    string arg_name = "b" + std::to_string(pair.first) + "_size";
    container_exprs[arg_name] = pair.second;
    SummaryBufferOccup(buffer_occup, HardwareDef::UB, container_exprs[arg_name]);
    GELOGD("Add temp buffer %s [%s] occupy for UB", arg_name.c_str(), pair.second.Str().get());
  }
  auto builtin_tmp_buffer = ArgListManager::GetInstance().GetArgExpr(kArgsNameBuiltInTmpBuffer);
  SummaryBufferOccup(buffer_occup, HardwareDef::UB, builtin_tmp_buffer);
  Expr kernel_init_buf_size = CreateExpr(0);
  for (const auto &reserved_ub : tuning_space_->reserve_ub) {
    kernel_init_buf_size = kernel_init_buf_size + CreateExpr(reserved_ub.second);
  }
  SummaryBufferOccup(buffer_occup, HardwareDef::UB, kernel_init_buf_size);
  GELOGD("Add temp buffer %s [%s] and init buf %s occupy for UB", kArgsNameBuiltInTmpBuffer,
         builtin_tmp_buffer.Str().get(), kernel_init_buf_size.Str().get());
  return af::SUCCESS;
}

af::Status BufOccupyExpr::GetTotalGlobalOccup(Expr &global_occup_expr) {
  Expr container_occup_expr;
  Expr occup_per_tensor;
  for (auto &container : tuning_space_->global_containers) {
    GE_ASSERT_SUCCESS(GetOccupInContainer(container, occup_per_tensor, container_occup_expr),
                      "Get container occupy failed.");
    GELOGD("Get container [%s] occupy : [%s]", container->name.c_str(), container_occup_expr.Str().get());
    if (IsValid(global_occup_expr)) {
      global_occup_expr = af::sym::Add(global_occup_expr, container_occup_expr);
    } else {
      global_occup_expr = container_occup_expr;
    }
  }
  return af::SUCCESS;
}

af::Status BufOccupyExpr::GetTotalBufferOccup(std::unordered_map<HardwareDef, Expr> &buffer_occup,
                                              std::map<std::string, Expr> &container_exprs) {
  // 获取queue的buffer占用
  GetBufferOccupInContainer(buffer_occup, container_exprs);
  for (auto &buffer_occup_item : buffer_occup) {
    auto scope_iter = kScope2Str.find(buffer_occup_item.first);
    if (scope_iter == kScope2Str.end()) {
      continue;
    }
    ArgListManager::GetInstance().SetArgExpr(scope_iter->second, buffer_occup_item.second);
  }
  return af::SUCCESS;
}

}  // namespace att
