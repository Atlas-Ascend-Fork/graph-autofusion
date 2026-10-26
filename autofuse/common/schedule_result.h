/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_BASE_COMMON_SCHEDULE_RESULT_H_
#define ASCGEN_DEV_BASE_COMMON_SCHEDULE_RESULT_H_

#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "common/checker.h"

namespace {
constexpr char kTemplateIdAttr[] = "af.internal.template.id";
constexpr char kTemplateRoleAttr[] = "af.internal.indirect_load.role";
// [行级广播 GM Load] 视图尾轴零贡献（stride==0 且 size==1）、其余轴稠密的广播
// side-input Load：语义为『读 [行,列] 的值沿尾轴广播』（如 gather+norm 图 load3
// [8,2048,1]/[2048,1,0]）。在视图被调度期 split 改写前（generator 的视图补全阶段）
// 判定记录，供 codegen 坐标重建兜底使用——改写后视图 rank/尺寸失配无法再判定。
constexpr char kRowBroadcastLoadAttr[] = "af.internal.indirect_load.row_broadcast_load";
// [SK分段SIMT] SK 候选在窗口超限形态下将 IL 段重定向为 SIMT 执行（仅保留 output
// workspace 边界，Reduce/Norm 段独立调度承载 R 轴分 tile）。该标记使分区后的
// 模板轴恢复走 SIMT 归一（与原 SK 的 RestoreSkTemplateAxes 对应）。
constexpr char kSkSegmentedSimtAttr[] = "af.internal.indirect_load.sk_segmented_simt";
constexpr char kDcacheSizeAttr[] = "af.internal.template.dcache_size";
}  // namespace

namespace ascir {
struct ScheduleGroup {
  std::vector<af::AscGraph> impl_graphs;
  std::map<std::string, std::string> graph_name_to_score_funcs;
  bool double_buffer{false};
};

enum class CubeTemplateType : int32_t {
  kDefault = -1,  // no cube
  kFixpip,        // fixpip模板
  kCommon,        // 兜底模板
  kUBFuse,        // ub复用模板
  kL2Fuse,        // L2复用模板
};

struct ScheduledResult {
  std::vector<ScheduleGroup> schedule_groups;
  // dst -> src, <dst_groupid, <src_groupid, <dst_var_name, src_var>>>;
  std::map<size_t, std::map<size_t, std::map<std::string, af::Expression>>> var_relations;
  ge::AscendString score_func;
  bool is_reduce_mem_reuse{false};
  bool enable_group_parallel{false};
  CubeTemplateType cube_type{CubeTemplateType::kDefault};
  std::string semantic_name{};
};

struct GmTensorSizes {
  af::Expression total_size;
  int64_t min_total_size;
  std::vector<af::Expression> input_sizes;
  std::vector<af::Expression> output_sizes;
};

struct FusedScheduledResult {
  ge::AscendString fused_graph_name;
  std::vector<af::AscNodePtr> input_nodes;
  std::vector<af::AscNodePtr> output_nodes;
  std::vector<af::AscNodePtr> workspace_nodes;
  // Symbols exposed by the frontend AutofuseTiling ABI.  This list is captured
  // from the original ASC graph before graph optimization and must not be
  // rebuilt from impl graphs, since an impl graph may legitimately not use all
  // frontend shape symbols.
  std::vector<af::Expression> frontend_shape_vars;
  // Distinguish a captured empty frontend symbol list (static frontend graph)
  // from legacy results that predate frontend_shape_vars.
  bool frontend_shape_vars_collected{false};
  std::vector<af::Expression> origin_vars;
  std::vector<std::vector<ScheduledResult>> node_idx_to_scheduled_results;
  GmTensorSizes gm_tensor_sizes;
};

enum class TemplateId : int64_t {
  kDefault = -1,
  kIndirectLoadSimd = 0,
  kIndirectLoadSimt = 1,
  kIndirectLoadSK = 2,
};

inline af::Status SetTemplateId(const af::AscNodePtr &node, TemplateId template_id) {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kTemplateIdAttr, static_cast<int64_t>(template_id)),
                 "Set internal template id failed, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

inline af::Status SetSkSegmentedSimt(const af::AscNodePtr &node, bool enabled) {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kSkSegmentedSimtAttr, static_cast<int64_t>(enabled ? 1 : 0)),
                 "Set sk segmented simt flag failed, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

inline bool IsSkSegmentedSimt(const af::AscNode &node) {
  if (node.GetOpDesc() == nullptr) {
    return false;
  }
  return node.GetOpDesc()->TryGetExtAttr(kSkSegmentedSimtAttr, static_cast<int64_t>(0)) != 0;
}

inline af::Status SetRowBroadcastLoad(const af::AscNodePtr &node, bool enabled) {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kRowBroadcastLoadAttr, static_cast<int64_t>(enabled ? 1 : 0)),
                 "Set row broadcast load flag failed, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

inline bool IsRowBroadcastLoad(const af::AscNode &node) {
  if (node.GetOpDesc() == nullptr) {
    return false;
  }
  return node.GetOpDesc()->TryGetExtAttr(kRowBroadcastLoadAttr, static_cast<int64_t>(0)) != 0;
}

inline af::Status SetTemplateRole(const af::AscNodePtr &node, int64_t role) {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kTemplateRoleAttr, role), "Set internal template role failed, node = %s",
                 node->GetNamePtr());
  return af::SUCCESS;
}

inline int64_t GetTemplateRoleOrDefault(const af::AscNode &node, int64_t default_role = -1) {
  if (node.GetOpDesc() == nullptr) {
    return default_role;
  }
  return node.GetOpDesc()->TryGetExtAttr(kTemplateRoleAttr, default_role);
}

inline TemplateId GetTemplateIdOrDefault(const af::AscNode &node, TemplateId default_id = TemplateId::kDefault) {
  if (node.GetOpDesc() == nullptr) {
    return default_id;
  }
  return static_cast<TemplateId>(node.GetOpDesc()->TryGetExtAttr(kTemplateIdAttr, static_cast<int64_t>(default_id)));
}

inline af::Status SetDcacheSize(const af::AscNodePtr &node, int64_t dcache_size) {
  GE_ASSERT_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kDcacheSizeAttr, dcache_size),
                 "Set internal template dcache size failed, node = %s", node->GetNamePtr());
  return af::SUCCESS;
}

inline int64_t GetDcacheSize(const af::AscNode &node) {
  if (node.GetOpDesc() == nullptr) {
    return 0;
  }
  return node.GetOpDesc()->TryGetExtAttr(kDcacheSizeAttr, int64_t{0});
}
}  // namespace ascir

#endif  // ASCGEN_DEV_BASE_COMMON_SCHEDULE_RESULT_H_
