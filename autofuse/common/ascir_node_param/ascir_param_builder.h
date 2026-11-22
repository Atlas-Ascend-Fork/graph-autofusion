/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_NODE_PARAM_ASCIR_PARAM_BUILDER_H_
#define ASCIR_NODE_PARAM_ASCIR_PARAM_BUILDER_H_

#include "ascir_node_param/ascir_node_param.h"

namespace ascir_param {
// Codegen 提供公共轴及最终条件；公共层只组装记录，不重新判断缓存资格。
// axes 及其指向的轴在本次生成期间保持有效，嵌套作用域共用只读轴并复制自己的 bindings。
struct CacheExecutionContext {
  using AxisMap = std::map<ascir::AxisId, const ascir::Axis *>;
  const AxisMap &axes;
  uint32_t stage;
  std::map<ReduceCacheCondition, CacheGuardInfo> bindings;
  std::map<CacheExecutionMode, CacheGuardInfo> broadcast_bindings;
};

void ResetCacheExecutions(const af::AscNodePtr &node, uint32_t stage);
void BindCacheCondition(CacheExecutionContext &context, ReduceCacheCondition condition, const ascir::Axis &axis,
                        const ascir::Axis &period, const ge::Expression &block_inner_extent);
void BindCacheCondition(CacheExecutionContext &context, CacheExecutionMode mode, const ascir::Axis &axis,
                        const ascir::Axis &period, const ge::Expression &block_inner_extent);
af::Status CollectCacheExecution(const CacheExecutionContext &context, const af::AscNodePtr &node,
                                 const std::vector<ascir::AxisId> &loop_axes, ascir::AxisId call_axis,
                                 CacheExecutionMode mode, const std::vector<ReduceCacheCondition> &conditions);

bool IsReduceParamSupported(const std::string &api_name);
af::Status EnrichAscirGraphNodeParams(const af::AscGraph &graph);
}  // namespace ascir_param

#endif  // ASCIR_NODE_PARAM_ASCIR_PARAM_BUILDER_H_
