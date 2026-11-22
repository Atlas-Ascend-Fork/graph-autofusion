/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_TYPES_H
#define AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_TYPES_H

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "common_utils.h"
#include "graph_utils.h"
#include "indirect_load_utils.h"
#include "graph/symbolizer/symbolic.h"
#include "norm_utils.h"
#include "optimize/graph_pass/pass_utils.h"
#include "optimize/graph_pass/softmax_pattern_fusion_utils.h"
#include "schedule_utils.h"

#include <algorithm>
#include <limits>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace optimize {
namespace indirect_load_detail {
// IndirectLoad 调度候选生成器共享常量与数据结构（自 indirect_load_schedule_case_generator.cpp 纯搬迁）。
inline constexpr int64_t kIndirectLoadSimtDcacheSize = 32 * 1024;
inline constexpr int64_t kEmbeddingSimdPayloadBytesThreshold = 2048;
inline constexpr int64_t kEmbeddingSimdLookupCountThreshold = 32;
inline constexpr int32_t kEmbeddingFastPathScore = 2;
inline constexpr int32_t kEmbeddingAlternateFastPathScore = 1;
// SIMD 候选退化到 strided facade（逐元素标量组装 offsets 并逐元素重排）时，SIMT 把同样的
// 逐元素地址计算分摊到 lane 上，评分上偏好 SIMT。
inline constexpr int32_t kSimtStridedFallbackScore = 1;
inline constexpr size_t kIndirectLoadInputCount = 2UL;
inline constexpr size_t kIndirectLoadOutputCount = 1UL;
inline constexpr size_t kIndirectLoadInvalidAxisIndex = std::numeric_limits<size_t>::max();
inline constexpr int64_t kInvalidBroadcastIndex = -1L;
inline constexpr char kInputInnerAxisName[] = "indirect_load_input_inner";
inline constexpr char kSingleOuterAxisName[] = "indirect_load_single_outer";
inline constexpr char kOuterAxisName[] = "indirect_load_outer";
inline constexpr char kInnerAxisName[] = "indirect_load_inner";
using NodePath = std::vector<af::AscNodePtr>;
using NodeSet = std::unordered_set<const af::AscNode *>;

struct TemplateCase {
  ascir::TemplateId template_id;
  ascgen_utils::indirect_load::Implementation implementation;
};

struct RewrittenGraphAnalysis {
  NodePath input_region;  // SIMD: input 侧链；SIMT 不使用
  NodePath index_region;  // SIMD: index 侧链；SIMT: 融合集（index 链 + 输出链）
  NodePath input_path;
  NodePath index_path;
  af::AscNodePtr input_root;
  af::AscNodePtr input_boundary;
  af::AscNodePtr index_root;
  af::AscNodePtr output_store;
  af::AscNodePtr post_reduce;
  // 多 Reduce 复合区域：全部已配对同轴 Broadcast 的统计 Reduce（≥2 时非空）。
  std::vector<af::AscNodePtr> composite_reduces;
  bool align_input_path = false;
  bool align_index_path = false;
  // 输入/索引路径含 Load 后接 Transpose：staging 经逐行 TransposeExtend，无法批量化
  // 为多行窗口（且其布局被强制标记 kStrided 以表达合法置换视图，绕过单调性分类）。
  bool input_load_transposed = false;
  bool index_load_transposed = false;
  bool simd_index_uses_output_inner_axis = false;
};

struct InputViewPlan {
  NodePath path;
  af::AscNodePtr load_transpose;
  ascgen_utils::indirect_load::IndirectLoadTensorLayout layout;
  int64_t path_broadcast_index = kInvalidBroadcastIndex;
  bool simd_index_uses_output_inner_axis = false;
};

struct PhysicalViewPreparation {
  InputViewPlan input;
  InputViewPlan index;
  ascgen_utils::indirect_load::TemplateLogicalView logical_view;
};

enum class ReduceAxisKind : uint8_t { kRetained, kReduced, kIgnored };

struct PostReduceLayout {
  std::vector<af::AxisId> axes;
  std::vector<ReduceAxisKind> kinds;
  size_t first_reduce;
};

}  // namespace indirect_load_detail
}  // namespace optimize

#endif  // AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_TYPES_H
