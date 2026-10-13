/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_TESTS_COMMON_CODEGEN_WORKSPACE_TEST_UTILS_H_
#define AUTOFUSE_TESTS_COMMON_CODEGEN_WORKSPACE_TEST_UTILS_H_

#include <cstddef>
#include <string>

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "common/schedule_result.h"

namespace autofuse::tests {

inline ascir::FusedScheduledResult MakeWorkspaceFusedScheduleResult(size_t workspace_count) {
  af::AscGraph graph("workspace_graph");
  ascir::FusedScheduledResult fused_schedule_result;
  for (size_t i = 0UL; i < workspace_count; ++i) {
    const std::string workspace_name = "workspace" + std::to_string(i + 1UL);
    af::ascir_op::Workspace workspace_op(workspace_name.c_str());
    graph.AddNode(workspace_op);
    auto workspace = graph.FindNode(workspace_name.c_str());
    workspace->outputs[0].attr.mem.tensor_id = static_cast<ascir::TensorId>(i + 1UL);
    workspace->outputs[0].attr.mem.alloc_type = af::AllocType::kAllocTypeGlobal;
    fused_schedule_result.workspace_nodes.push_back(workspace);
  }
  return fused_schedule_result;
}

}  // namespace autofuse::tests

#endif  // AUTOFUSE_TESTS_COMMON_CODEGEN_WORKSPACE_TEST_UTILS_H_
