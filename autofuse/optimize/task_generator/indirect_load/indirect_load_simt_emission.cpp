/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_generator/indirect_load/indirect_load_simt_emission.h"
#include "task_generator/indirect_load/indirect_load_graph_utils.h"
#include "task_generator/indirect_load/indirect_load_view_layout.h"
#include "task_generator/indirect_load/indirect_load_input_path.h"
#include "task_generator/indirect_load/indirect_load_output_analysis.h"
#include "task_generator/indirect_load/indirect_load_template_plan.h"
#include "task_generator/indirect_load/indirect_load_template_annotation.h"
#include "task_generator/indirect_load/indirect_load_axis_normalize.h"
#include "task_generator/indirect_load/indirect_load_score_generator.h"
#include "task_generator/indirect_load/indirect_load_ub_feasibility.h"
#include "task_generator/indirect_load/indirect_load_sk_partition.h"
#include "task_generator/indirect_load/indirect_load_group_candidate.h"

#include "v35/ascir/ascir_codegen_v2.h"
#include "ascgraph_info_complete.h"
#include "schedule_result.h"

namespace optimize {
namespace indirect_load_detail {

// 多 gather 组员（SIMT）扇出标注：组员计划不含主输出链（selected 仅含全组 IL 节点作
// [仅 mixed 路径引用·mixed 已下线] 本函数与 AnnotateSimtSideInputClosure 在组 SIMT
// 路径已无调用（组链无角色，VF/normal 由分区器裁决）；当前唯一调用方是
// ApplyGraphPassMixed（kEnableMixedGroupCandidate=false 编译期关闭）。保留供 mixed
// 重启用（需先落地 mixed 链 MVP——见含 Reduce 文档 11 节），单 gather 扇出路径使用
// 独立的 AnnotateSimtFanoutBranches，不受影响。
// 排除集），IL 输出的全部消费者（归并 ele/输出链）均为 normal schedule 扇出分支——
// 标注 kSimtFanoutBranch 使调度侧按既有语义处理 SIMT 输出到普通向量代码的边界；
// side input 闭包随分支标注，全组 IL 排除避免改写他组成员的 kSimtOp 角色。
af::Status AnnotateSimtGroupFanoutBranches(const std::vector<af::AscNodePtr> &members) {
  NodeSet member_set;
  for (const af::AscNodePtr &member : members) {
    member_set.emplace(member.get());
  }
  NodeSet normal_schedule_nodes;
  NodeSet side_visited;
  const auto visit = [&member_set, &normal_schedule_nodes, &side_visited](
                         const af::AscNodePtr &member, const af::AscNodePtr &node, bool &stop) -> af::Status {
    // store/reduce 为 normal schedule 终端，不标注扇出角色（单 gather 版本中它们位于
    // selected 主输出链内、同样不标注；组员无主输出链，需显式排除）。
    const bool is_fanout_branch = member_set.count(node.get()) == 0UL && !IsInputDataSource(node) &&
                                  !ScheduleUtils::IsReduce(node) && !ScheduleUtils::IsStore(node);
    if (is_fanout_branch) {
      GELOGD("[IndirectLoad] SIMT group fanout branch node[%s] annotated.", node->GetNamePtr());
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
          node, ascgen_utils::indirect_load::TemplateRole::kSimtFanoutBranch));
      GE_ASSERT_SUCCESS(AnnotateSimtSideInputClosure(node, member, member_set, normal_schedule_nodes, side_visited));
    }
    if (ScheduleUtils::IsStore(node) || ScheduleUtils::IsReduce(node)) {
      stop = true;
    }
    return af::SUCCESS;
  };
  for (const af::AscNodePtr &member : members) {
    NodeSet visited{member.get()};
    NodePath roots;
    for (const auto &out_node : member->GetOutDataNodes()) {
      const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out_node);
      GE_ASSERT_NOTNULL(consumer, "IndirectLoad output successor is invalid.");
      roots.emplace_back(consumer);
    }
    for (size_t cursor = 0UL; cursor < roots.size(); ++cursor) {
      const af::AscNodePtr node = roots[cursor];
      if (node == nullptr || !visited.emplace(node.get()).second) {
        continue;
      }
      bool stop = false;
      GE_ASSERT_SUCCESS(visit(member, node, stop));
      if (stop) {
        continue;
      }
      for (const auto &out_node : node->GetOutDataNodes()) {
        const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out_node);
        GE_ASSERT_NOTNULL(consumer, "IndirectLoad output successor is invalid.");
        roots.emplace_back(consumer);
      }
    }
  }
  return af::SUCCESS;
}

af::Status PlanSimtFusedEmission(const af::AscGraph &graph, const std::vector<af::AscNodePtr> &members,
                                 ascgen_utils::indirect_load::SimtFusedMergeInfo &merge_info) {
  merge_info = ascgen_utils::indirect_load::SimtFusedMergeInfo{};
  // 归并树收集：全部成员输出 Consumer 的前向闭包（成员为边界；Store/Reduce 为终端）。
  NodeSet member_set;
  for (const af::AscNodePtr &member : members) {
    member_set.emplace(member.get());
  }
  NodeSet forward;
  std::vector<af::AscNodePtr> pending;
  for (const af::AscNodePtr &member : members) {
    for (const auto &out_node : member->GetOutDataNodes()) {
      const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out_node);
      GE_ASSERT_NOTNULL(consumer, "IndirectLoad output successor is invalid.");
      pending.emplace_back(consumer);
    }
  }
  bool has_reduce_terminal = false;
  std::vector<af::AscNodePtr> store_terminals;
  std::vector<af::AscNodePtr> forward_nodes;
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const af::AscNodePtr node = pending[cursor];
    if (node == nullptr || member_set.count(node.get()) != 0UL) {
      continue;
    }
    if (ScheduleUtils::IsStore(node)) {
      store_terminals.emplace_back(node);
      continue;
    }
    if (ScheduleUtils::IsReduce(node)) {
      // 共享 post-Reduce 终端：merge 结果需 UB 局部目标 + 向量 Reduce 消费（v1 边界，降 L3）。
      has_reduce_terminal = true;
      continue;
    }
    if (!forward.emplace(node.get()).second) {
      continue;
    }
    forward_nodes.emplace_back(node);
    for (const auto &out_node : node->GetOutDataNodes()) {
      const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out_node);
      GE_ASSERT_NOTNULL(consumer, "IndirectLoad output successor is invalid.");
      pending.emplace_back(consumer);
    }
  }
  // side input 发现与分类（②③）：
  // - Scalar/ScalarData：直接放行（context 常量/值字段）
  // - GM load（Load/direct-gm Nddma）：side load，BuildSimtMergeSideLoadMetadata 分类
  //   （kZeroOffset/kOutputOffset + use_logical_offset 坐标折叠；视图匹配成员 index load 的
  //   ② 形态由折叠按自身物理视图从 output_index 正确寻址，无需特判）
  // - 其余（③ 区域外 elewise 链）：后向闭包收集链节点（逐节点可发射性校验），链上 Load
  //   收为 side load，Scalar/ScalarData/Arange 为终端放行；不可发射节点 → L3。
  bool has_unsupported_side_input = false;
  std::vector<af::AscNodePtr> side_loads;
  std::vector<af::AscNodePtr> side_chain_nodes;
  NodeSet side_chain_set;
  std::vector<af::AscNodePtr> backward_pending;
  for (const af::AscNodePtr &node : forward_nodes) {
    for (const auto &in_node : node->GetInDataNodes()) {
      const auto producer = std::dynamic_pointer_cast<af::AscNode>(in_node);
      if (producer == nullptr || member_set.count(producer.get()) != 0UL || forward.count(producer.get()) != 0UL ||
          side_chain_set.count(producer.get()) != 0UL) {
        continue;
      }
      if (af::ops::IsOps<af::ascir_op::Scalar>(producer) || af::ops::IsOps<af::ascir_op::ScalarData>(producer)) {
        continue;
      }
      if (ascgen_utils::indirect_load::IsSimtDirectGmLoadNode(producer)) {
        if (std::find(side_loads.begin(), side_loads.end(), producer) == side_loads.end()) {
          ascgen_utils::indirect_load::SimtLoadMetadata side_meta;
          GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::BuildSimtMergeSideLoadMetadata(producer, side_meta));
          side_loads.emplace_back(producer);
        }
        continue;
      }
      backward_pending.emplace_back(producer);
    }
  }
  for (size_t cursor = 0UL; cursor < backward_pending.size(); ++cursor) {
    const af::AscNodePtr node = backward_pending[cursor];
    if (node == nullptr || member_set.count(node.get()) != 0UL || forward.count(node.get()) != 0UL ||
        side_chain_set.count(node.get()) != 0UL) {
      continue;
    }
    if (ascgen_utils::indirect_load::IsSimtDirectGmLoadNode(node)) {
      if (std::find(side_loads.begin(), side_loads.end(), node) == side_loads.end()) {
        ascgen_utils::indirect_load::SimtLoadMetadata side_meta;
        GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::BuildSimtMergeSideLoadMetadata(node, side_meta));
        side_loads.emplace_back(node);
      }
      continue;
    }
    // 常量/运行时输入引用不做标量计算，豁免后作为链终点结束回溯；Arange 是真正的
    // 计算节点（生成 gather 坐标），与 L2 判定的"链上节点全部可标量发射"口径一致，
    // 必须走 CanEmitSimtScalar 真实查询（不可发射则 has_unsupported_side_input 回退
    // L3），避免调度期放行、发射期 EmitSimtArangeExpr 断言硬挂（与 index 链/输出链
    // 两条判定路径同一事实源）。
    if (af::ops::IsOps<af::ascir_op::Scalar>(node) || af::ops::IsOps<af::ascir_op::ScalarData>(node)) {
      continue;
    }
    if (!af::ops::IsOps<af::ascir_op::Transpose>(node) && !CanEmitSimtScalar(node)) {
      has_unsupported_side_input = true;
      break;
    }
    side_chain_set.emplace(node.get());
    side_chain_nodes.emplace_back(node);
    for (const auto &in_node : node->GetInDataNodes()) {
      const auto producer = std::dynamic_pointer_cast<af::AscNode>(in_node);
      if (producer != nullptr) {
        backward_pending.emplace_back(producer);
      }
    }
  }
  // 可发射性：闭包消费节点逐个查 V2 标量注册表（Transpose 为表达式直通节点，豁免）。
  bool all_emittable = true;
  for (const af::AscNodePtr &node : forward_nodes) {
    if (af::ops::IsOps<af::ascir_op::Transpose>(node) || CanEmitSimtScalar(node)) {
      continue;
    }
    all_emittable = false;
    break;
  }
  // [发射计划组件收集] 压力无关的结构组件与计数：形态/压力/线程档由 PlanSimtEmission
  // 在 lowering 元数据之后落定（单一真值源）——本函数只收集不决策。链发射者不再由
  // 本处判定：L3/L4 链无角色，VF/normal 由分区器按簇形态自然裁决。
  merge_info.member_count = static_cast<uint32_t>(members.size());
  merge_info.side_load_count = static_cast<uint32_t>(side_loads.size());
  merge_info.chain_node_count = static_cast<uint32_t>(side_chain_nodes.size());
  merge_info.forward_node_count = static_cast<uint32_t>(forward_nodes.size());
  merge_info.l2_structural = all_emittable && !has_reduce_terminal && !has_unsupported_side_input &&
                             !store_terminals.empty() && !forward_nodes.empty();
  merge_info.forward_chain_width =
      static_cast<uint32_t>(ascgen_utils::indirect_load::ComputeMergeChainWidth(graph, forward_nodes, members));
  if (!merge_info.l2_structural) {
    GELOGI(
        "[IndirectLoad] Fused merge falls back to UB merge (reduce_terminal[%d], unsupported_side_input[%d], "
        "all_emittable[%d], stores[%zu], side_loads[%zu]).",
        has_reduce_terminal ? 1 : 0, has_unsupported_side_input ? 1 : 0, all_emittable ? 1 : 0, store_terminals.size(),
        side_loads.size());
  }
  // 归并树闭包节点名（GetAllNodes 拓扑序，含 ③ side 链节点）：L2 由 evaluator 按拓扑序
  // 发射（kSimtInlineTransform 注解在 PlanSimtEmission 之后按计划写入）；L3/L4 不消费。
  for (const af::AscNodePtr &graph_node : graph.GetAllNodes()) {
    if (graph_node != nullptr &&
        (forward.count(graph_node.get()) != 0UL || side_chain_set.count(graph_node.get()) != 0UL)) {
      merge_info.merge_node_names.emplace_back(graph_node->GetName());
    }
  }
  // Store 终端名：仅 L2 消费（evaluator 直写 GM，角色注解在计划之后按 kEvaluator 写入）；
  // L3/L4 由向量侧正常发射读取，不做标注。
  for (const af::AscNodePtr &store : store_terminals) {
    merge_info.merge_store_names.emplace_back(store->GetName());
  }
  // side GM load 节点名：仅 L2 消费（evaluator 直读 GM，kSimtDirectGmBoundary 注解在
  // 计划之后按 kEvaluator 写入）；L3/L4 由向量侧正常发射读取。
  for (const af::AscNodePtr &side_load : side_loads) {
    merge_info.side_load_names.emplace_back(side_load->GetName());
  }
  GELOGI(
      "[IndirectLoad] Fused merge components: %zu merge nodes, %zu stores, %zu side loads, %zu side chain nodes, "
      "width[%u] (N=%zu, l2_structural[%d]).",
      merge_info.merge_node_names.size(), merge_info.merge_store_names.size(), side_loads.size(),
      side_chain_nodes.size(), static_cast<unsigned int>(merge_info.forward_chain_width), members.size(),
      merge_info.l2_structural ? 1 : 0);
  return af::SUCCESS;
}

af::Status PlanSimtEmission(const ascgen_utils::indirect_load::SimtFusedMergeInfo &merge_info,
                            const bool all_offset_uint32, SimtEmissionPlan &plan) {
  uint64_t demand = 0U;
  if (ascgen_utils::indirect_load::kUsePeakModel) {
    // 峰值模型：L2 结构合法且峰值可行 → 寄存器归并；否则 L3 峰值可行 → UB 归并
    // （L2→L3 压力降级，保住 1 次 invocation + 1 次 barrier）；否则 L4（压力兜底，
    // 真实触发仅 base+3N 超 quota 的超大组）。可行性统一走 SimtFusedQuotaFeasible
    // （唯一判据，与生成期预检 IsGroupSimtL4UbDoomed 共用）。
    const uint64_t l2_demand = ascgen_utils::indirect_load::SimtPeakRegisterDemand(
        all_offset_uint32, merge_info.member_count, merge_info.side_load_count, merge_info.chain_node_count,
        merge_info.forward_chain_width);
    if (merge_info.l2_structural && ascgen_utils::indirect_load::SimtFusedQuotaFeasible(l2_demand)) {
      demand = l2_demand;
      plan.use_fused = true;
      plan.register_merge = true;
    } else {
      demand = ascgen_utils::indirect_load::SimtPeakRegisterDemandL3(all_offset_uint32, merge_info.member_count);
      plan.use_fused = ascgen_utils::indirect_load::SimtFusedQuotaFeasible(demand);
    }
  } else {
    // sum 回退：旧压力公式与旧降级行为（单 demand，超限直接 L4，无 L2→L3 压力降级）；
    // 角色一致性修复（降级重注解）对两模型同时生效。
    demand = ascgen_utils::indirect_load::SimtFusedRegisterDemand(
        all_offset_uint32, merge_info.member_count, merge_info.side_load_count, merge_info.chain_node_count);
    plan.use_fused = demand <= ascgen_utils::indirect_load::kSimtFusedMinTierQuota;
    plan.register_merge = plan.use_fused && merge_info.l2_structural;
  }
  plan.demand = demand;
  plan.max_thread_tier = ascgen_utils::indirect_load::SimtFusedThreadTier(demand);
  return af::SUCCESS;
}
}  // namespace indirect_load_detail
}  // namespace optimize
