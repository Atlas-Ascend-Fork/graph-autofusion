/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_L2_CACHE_HINT_MANAGER_H_
#define COMMON_L2_CACHE_HINT_MANAGER_H_

#include "schedule_result.h"
#include "ascir/meta/ascir.h"

namespace optimize {
class L2CacheHintManager {
 public:
  static af::Status ParseGraph(const af::ComputeGraph &graph, ::ascir::FusedScheduledResult &fused_scheduled_result);
  static bool AllExprSymbolsInGraph(const ascir::GmTensorSizes &sizes, const ascir::ImplGraph &graph);
  static std::set<size_t> CollectSkipL2CacheHintIndices(const ascir::ImplGraph &graph);
  static af::Status GetL2Size(int64_t &l2_size);

 private:
  static af::Status CalcTensorSizes(const af::ComputeGraph &graph, const ::ascir::FusedScheduledResult &fsr,
                                    ascir::GmTensorSizes &global_tensor_sizes);
  static af::Status MarkInputsNeedSkipL2CacheHint(::ascir::FusedScheduledResult &fused_scheduled_result);
  static af::Status MarkInternal(af::AscGraph &impl_graph);
  static af::Status MarkSingleBroadcastInlineSkip(const af::ComputeGraph &graph,
                                                  ::ascir::FusedScheduledResult &fused_scheduled_result);
  static bool IsSingleBroadcastAxisTimesTwo(const af::AscNodePtr &broadcast_node);
  static af::Status CollectAnalysisNodes(const af::ComputeGraph &graph, std::vector<af::NodePtr> &analysis_nodes);
  static af::Status FindSingleBroadcastNode(const std::vector<af::NodePtr> &analysis_nodes,
                                            af::AscNodePtr &broadcast_node);
  static af::Status ValidateBroadcastInputSize(const af::AscNodePtr &broadcast_node, int64_t &input_bytes,
                                               int64_t &output_bytes);
  static af::AscNodePtr ResolveBroadcastInputSource(const af::AscNodePtr &broadcast_node);
  static af::Status MarkSkipHintOnDataNodes(::ascir::FusedScheduledResult &fused_scheduled_result, int64_t input_bytes,
                                            int64_t output_bytes, int64_t l2_size);
};
}  // namespace optimize

#endif  // COMMON_L2_CACHE_HINT_MANAGER_H_
