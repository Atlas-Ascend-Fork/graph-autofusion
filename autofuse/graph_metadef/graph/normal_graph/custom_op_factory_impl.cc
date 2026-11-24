/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/custom_op_factory_impl.h"

#include "debug/ge_log.h"
#include "common/util/mem_utils.h"

namespace af {
Status CustomOpFactoryImpl::RegisterCustomOpCreator(const AscendString &op_type, const BaseOpCreator &op_creator) {
  const auto it = custom_op_creators_.find(op_type);
  if (it != custom_op_creators_.cend()) {
    GELOGW("[CUSTOM OP] custom op creator for %s already exists.", op_type.GetString());
    return GRAPH_FAILED;
  }
  (void)custom_op_creators_.emplace(op_type, op_creator);
  GELOGI("[CUSTOM OP] register custom operator creator for %s.", op_type.GetString());
  return GRAPH_SUCCESS;
}

std::unique_ptr<BaseCustomOp> CustomOpFactoryImpl::CreateCustomOp(const AscendString &op_type) {
  const auto it = custom_op_creators_.find(op_type);
  if (it != custom_op_creators_.cend()) {
    return it->second();
  }
  GELOGW("[CUSTOM OP] get custom operator creator failed for %s.", op_type.GetString());
  return nullptr;
}
Status CustomOpFactoryImpl::GetAllRegisteredOps(std::vector<AscendString> &all_registered_ops) {
  for (const auto &op_creator : custom_op_creators_) {
    all_registered_ops.push_back(op_creator.first);
  }
  return GRAPH_SUCCESS;
}

bool CustomOpFactoryImpl::IsExistOp(const AscendString &op_type) {
  return custom_op_creators_.find(op_type) != custom_op_creators_.end();
}

CustomOpFactoryImpl::CustomOpFactoryImpl() = default;
}  // namespace af
