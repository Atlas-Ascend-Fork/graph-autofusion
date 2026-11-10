/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_MLIR_COMMON_TILING_SCHEMA_SCHEDULE_GROUP_NAMES_H
#define AUTOFUSE_MLIR_COMMON_TILING_SCHEMA_SCHEDULE_GROUP_NAMES_H

#include <cstdint>
#include <string>

// ATT ScheduleGroupIdent 的命名规则，供 MLIR 各层共享；比对测试防止两侧漂移。
// 身份为 (子图, plan, stage)，其中 plan 对应 ATT 的 impl_graph_id，不是 template ID。
// 仅依赖标准库，避免 conversion/artifact 引入 ATT 依赖。
namespace mlir::common {

// group 在顶层 TilingData 中的成员名，对应 ATT GetItemPrefix() + "_tiling_data"。
inline std::string MakeScheduleGroupTilingFieldName(int64_t subgraphId, int64_t planId, int64_t stageId) {
  return "graph" + std::to_string(subgraphId) + "_result" + std::to_string(planId) + "_g" + std::to_string(stageId) +
         "_tiling_data";
}

// group 结构体的类型名，对应 ATT GetGroupPrefix() + "TilingData"。
inline std::string MakeScheduleGroupTilingTypeName(int64_t subgraphId, int64_t planId, int64_t stageId) {
  return "AscGraph" + std::to_string(subgraphId) + "ScheduleResult" + std::to_string(planId) + "G" +
         std::to_string(stageId) + "TilingData";
}

}  // namespace mlir::common

#endif  // AUTOFUSE_MLIR_COMMON_TILING_SCHEMA_SCHEDULE_GROUP_NAMES_H
