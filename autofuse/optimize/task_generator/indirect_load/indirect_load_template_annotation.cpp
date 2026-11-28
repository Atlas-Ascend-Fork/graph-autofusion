/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_generator/indirect_load/indirect_load_template_annotation.h"
#include "task_generator/indirect_load/indirect_load_graph_utils.h"
#include "task_generator/indirect_load/indirect_load_view_layout.h"
#include "task_generator/indirect_load/indirect_load_input_path.h"
#include "task_generator/indirect_load/indirect_load_output_analysis.h"
#include "task_generator/indirect_load/indirect_load_template_plan.h"
#include "task_generator/indirect_load/indirect_load_axis_normalize.h"
#include "task_generator/indirect_load/indirect_load_score_generator.h"
#include "task_generator/indirect_load/indirect_load_ub_feasibility.h"
#include "task_generator/indirect_load/indirect_load_sk_partition.h"
#include "task_generator/indirect_load/indirect_load_simt_emission.h"
#include "task_generator/indirect_load/indirect_load_group_candidate.h"

#include "v35/ascir/ascir_codegen_v2.h"

namespace optimize {
namespace indirect_load_detail {

// SIMT direct-GM nodes deliberately preserve their vectorized view through the
// scheduler.  User graphs, however, do not necessarily initialize that view;
// fill it from the node's physical strides before VF partitioning consumes it.
// 跳过全部 IndirectLoad 节点而非仅当前成员：IL 输出 vectorized 视图按模板约定保持空，
// 由调度器的 tensor merge/split 主动改写。组路径按成员循环调用本函数，若仅跳过当前
// 成员，primary 的调用会把其他成员（kSimtOp，preserves_vectorized_axis=true）当普通
// 保留节点播种——SIMT 组 post_reduce 置空使 outer=全部输出轴合并，调度期 tensor merge
// 将播种轴重映射为 merged outer，SelectLoopAxis 据此把 outer 放进 not_loop 集并经
// from 递归过滤掉成员全部调度轴（tile 轴 from={outer}），触发 Cannot find loop axis
// 断言。
af::Status CompletePreservedVectorizedViews(const af::AscGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    if (node == nullptr || af::ops::IsOps<af::ascir_op::IndirectLoad>(node) ||
        af::ops::IsOps<af::ascir_op::Output>(node) ||
        !ascgen_utils::indirect_load::GetTemplateBehavior(node).preserves_vectorized_axis) {
      continue;
    }
    // [行级广播 GM Load 标记] 此处视图尚未被调度期 SIMT 边界的 split 改写——按
    // 原始结构判定（尾轴零贡献 stride==0 且 size==1、其余轴稠密递推），语义为
    // 『读 [行,列] 值沿尾轴广播』。记录到节点 attr，供 codegen 坐标重建兜底在
    // 改写后视图 rank 失配时按原始语义寻址（改写后无法再判定）。
    if (af::ops::IsOps<af::ascir_op::Load>(node) && !node->outputs().empty() && node->outputs()[0] != nullptr) {
      const auto &mark_attr = node->outputs()[0]->attr;
      const auto &mark_sizes = mark_attr.repeats;
      const auto &mark_strides = mark_attr.strides;
      // 行级广播：尾轴 stride==0（值沿尾轴广播；size 即广播因子，无约束——原始
      // 形态 size==1，折叠广播后可为任意值）。其余轴构成 row-major 稠密物理布局：
      // 递推【从后往前】（stride[last_nonzero]==1，stride[i]==stride[i+1]*size[i+1]，
      // 中间零贡献轴不占物理空间直接跳过）——如 [8,2048,*]/[2048,1,0]：
      // a1.stride==1，a0.stride==1*2048，判定成立。
      // （递推方向不可倒置：从前往后时 expected 初值 1 会使 a0.stride=2048≠1 误判
      // 非稠密，标记从未写入，兜底恒不命中，生产 load3 反复报 rank mismatch。）
      const bool tail_zero =
          !mark_sizes.empty() && mark_sizes.size() == mark_strides.size() &&
          af::SymbolicUtils::StaticCheckEq(mark_strides.back(), af::sym::kSymbolZero) == af::TriBool::kTrue;
      bool dense_rows = true;
      if (tail_zero) {
        af::Expression expected = af::sym::kSymbolOne;
        for (size_t index = mark_sizes.size(); index > 0UL; --index) {
          const size_t dim = index - 1UL;
          if (af::SymbolicUtils::StaticCheckEq(mark_strides[dim], af::sym::kSymbolZero) == af::TriBool::kTrue) {
            continue;
          }
          if (af::SymbolicUtils::StaticCheckEq(mark_strides[dim], expected) != af::TriBool::kTrue) {
            dense_rows = false;
            break;
          }
          expected = af::sym::Mul(expected, mark_sizes[dim]);
        }
      } else {
        dense_rows = false;
      }
      GE_ASSERT_SUCCESS(ascir::SetRowBroadcastLoad(node, tail_zero && dense_rows));
    }
    for (const auto &output : node->outputs()) {
      if (output == nullptr || !output->attr.vectorized_axis.empty()) {
        continue;
      }
      const auto &axes = output->attr.axis;
      const auto &strides = output->attr.strides;
      GE_ASSERT_TRUE(!axes.empty() && axes.size() == strides.size(),
                     "IndirectLoad preserved node[%s] has invalid tensor view.", node->GetNamePtr());
      size_t vectorized_index = axes.size() - 1UL;
      for (size_t index = axes.size(); index > 0UL; --index) {
        const auto stride = strides[index - 1UL];
        if (af::SymbolicUtils::StaticCheckEq(stride, af::sym::kSymbolOne) == af::TriBool::kTrue) {
          vectorized_index = index - 1UL;
          break;
        }
        if (af::SymbolicUtils::StaticCheckEq(stride, af::sym::kSymbolZero) != af::TriBool::kTrue) {
          vectorized_index = index - 1UL;
        }
      }
      output->attr.vectorized_axis = {axes[vectorized_index]};
      output->attr.vectorized_strides = {strides[vectorized_index]};
      GELOGD("[IndirectLoad] Seed preserved vectorized view for node[%s], axis[%ld].", node->GetNamePtr(),
             axes[vectorized_index]);
    }
  }
  return af::SUCCESS;
}

af::Status AnnotateSimdTemplateRoles(const RewrittenGraphAnalysis &analysis) {
  for (const af::AscNodePtr &node : analysis.input_region) {
    if (IsInputDataSource(node)) {
      continue;
    }
    const auto role = ascgen_utils::indirect_load::GetTemplateRole(node);
    const auto simd_role = role == ascgen_utils::indirect_load::TemplateRole::kStridedUbPath
                               ? ascgen_utils::indirect_load::TemplateRole::kSimdInputPreStridedUbPath
                               : ascgen_utils::indirect_load::TemplateRole::kSimdInputPre;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(node, simd_role));
  }
  if (analysis.simd_index_uses_output_inner_axis) {
    for (const af::AscNodePtr &node : analysis.index_region) {
      if (IsInputDataSource(node)) {
        continue;
      }
      GE_ASSERT_SUCCESS(
          ascgen_utils::indirect_load::SetTemplateRole(node, ascgen_utils::indirect_load::TemplateRole::kSimdIndexPre));
    }
  }
  return af::SUCCESS;
}

af::Status ValidateSimtTemplateRegion(const RewrittenGraphAnalysis &analysis, bool &is_candidate_legal) {
  is_candidate_legal = false;
  if (!af::ops::IsOps<af::ascir_op::Load>(analysis.input_boundary)) {
    GELOGI("[IndirectLoad] Reject SIMT: input boundary[%s(%s)] is not a Load node",
           analysis.input_boundary == nullptr ? "null" : analysis.input_boundary->GetNamePtr(),
           analysis.input_boundary == nullptr ? "-" : analysis.input_boundary->GetTypePtr());
    return af::SUCCESS;
  }
  if (analysis.index_region.empty()) {
    GELOGI("[IndirectLoad] Reject SIMT: index region is empty (size=%zu)", analysis.index_region.size());
    return af::SUCCESS;
  }
  for (const af::AscNodePtr &node : analysis.index_region) {
    if (GetLoadTransposeSource(node) != nullptr) {
      continue;
    }
    // Compile-time Scalar values are emitted as local constants by the SIMT evaluator.
    // ScalarData is runtime input and still requires an explicit context/GM binding.
    if (af::ops::IsOps<af::ascir_op::ScalarData>(node) || HasControlEdge(node)) {
      GELOGI("[IndirectLoad] Reject SIMT: node[%s(%s)] has ScalarData or control edge", node->GetNamePtr(),
             node->GetTypePtr());
      return af::SUCCESS;
    }
    const bool is_gm_boundary = af::ops::IsOps<af::ascir_op::Load>(node) || af::ops::IsOps<af::ascir_op::Store>(node);
    // Arange 不再无条件豁免：其 SIMT 专用发射路径 EmitSimtArangeExpr 有前置条件
    // （ArangeAscIrCodegenImplV2::IsSimtScalarSupported：零输入、int32/int64、单一
    // varying 轴），与发射期同一 V2 能力查询，不可发射时在此拒绝候选（index 链无法
    // 退出求值器——index 计算必须折叠），而不是放行到 codegen 硬失败。
    if (!is_gm_boundary && !CanEmitSimtScalar(node)) {
      GELOGD("[IndirectLoad] SIMT scalar codegen does not support node[%s, %s].", node->GetNamePtr(),
             node->GetTypePtr());
      return af::SUCCESS;
    }
  }
  is_candidate_legal = true;
  return af::SUCCESS;
}

af::Status CollectSimtNormalScheduleNodes(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis,
                                          NodeSet &normal_schedule_nodes) {
  normal_schedule_nodes.clear();
  if (analysis.post_reduce == nullptr) {
    return af::SUCCESS;
  }

  NodePath normal_schedule_path;
  const auto collect = [&normal_schedule_nodes, &normal_schedule_path](const af::AscNodePtr &node,
                                                                       bool &stop) -> af::Status {
    normal_schedule_nodes.emplace(node.get());
    normal_schedule_path.emplace_back(node);
    if (ScheduleUtils::IsStore(node)) {
      stop = true;
    }
    return af::SUCCESS;
  };
  const auto boundary = ascgen_utils::indirect_load::GetSimtPostReduceLocalProducer(indirect_load);
  GE_ASSERT_NOTNULL(boundary);
  NodePath roots;
  if (boundary == indirect_load) {
    roots.emplace_back(analysis.post_reduce);
  } else {
    for (const auto &out : boundary->GetOutDataNodes()) {
      roots.emplace_back(std::dynamic_pointer_cast<af::AscNode>(out));
    }
  }
  GE_ASSERT_SUCCESS(TraverseOutputConsumers(roots, normal_schedule_nodes, collect));

  NodeSet normal_schedule_dependencies;
  const auto collect_dependency = [indirect_load, boundary](const af::AscNodePtr &node) {
    if (node == indirect_load || node == boundary) {
      return false;
    }
    return true;
  };
  TraverseInputProducers(normal_schedule_path, normal_schedule_dependencies, collect_dependency);
  normal_schedule_dependencies.erase(indirect_load.get());
  normal_schedule_dependencies.erase(boundary.get());
  normal_schedule_nodes.insert(normal_schedule_dependencies.begin(), normal_schedule_dependencies.end());
  return af::SUCCESS;
}

af::Status AnnotateSimtMainOutputPath(const af::AscNodePtr &indirect_load, const af::AscNodePtr &selected_root,
                                      const NodeSet &selected, const NodeSet &normal_schedule_nodes) {
  // Nodes on the selected output path are emitted by the SIMT scalar evaluator.
  // With a user fan-out this path can contain ordinary elementwise transforms
  // that are not part of the index region handled by AnnotateSimtTemplateRoles.
  NodePath pending = {selected_root};
  NodeSet visited;
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const auto &node = pending[cursor];
    if (node == nullptr || node == indirect_load || !visited.emplace(node.get()).second) {
      continue;
    }
    if (normal_schedule_nodes.count(node.get()) == 0UL && !IsInputDataSource(node) && !ScheduleUtils::IsReduce(node) &&
        !ScheduleUtils::IsStore(node)) {
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
          node, ascgen_utils::indirect_load::TemplateRole::kSimtInlineTransform));
    }
    for (const auto &producer_node : node->GetInDataNodes()) {
      const auto producer = std::dynamic_pointer_cast<af::AscNode>(producer_node);
      if (producer != nullptr && selected.count(producer.get()) != 0UL) {
        pending.emplace_back(producer);
      }
    }
  }
  return af::SUCCESS;
}

af::Status AnnotateSimtSideInputClosure(const af::AscNodePtr &branch_node, const af::AscNodePtr &indirect_load,
                                        const NodeSet &selected, const NodeSet &normal_schedule_nodes,
                                        NodeSet &visited) {
  NodePath pending;
  for (const auto &producer_node : branch_node->GetInDataNodes()) {
    const auto producer = std::dynamic_pointer_cast<af::AscNode>(producer_node);
    if (producer != nullptr) {
      pending.emplace_back(producer);
    }
  }
  for (size_t cursor = 0UL; cursor < pending.size(); ++cursor) {
    const auto &producer = pending[cursor];
    if (producer == nullptr || producer == indirect_load || selected.count(producer.get()) != 0UL ||
        normal_schedule_nodes.count(producer.get()) != 0UL || ScheduleUtils::IsReduce(producer) ||
        !visited.emplace(producer.get()).second) {
      continue;
    }
    if (IsInputDataSource(producer) && !af::ops::IsOps<af::ascir_op::Load>(producer)) {
      continue;
    }
    const auto role = ascgen_utils::indirect_load::GetTemplateRole(producer);
    const auto side_role = af::ops::IsOps<af::ascir_op::Load>(producer)
                               ? ascgen_utils::indirect_load::TemplateRole::kSimtDirectGmBoundary
                               : ascgen_utils::indirect_load::TemplateRole::kSimtFanoutBranch;
    if (role == ascgen_utils::indirect_load::TemplateRole::kNone ||
        role == ascgen_utils::indirect_load::TemplateRole::kSimtFanoutBranch) {
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(producer, side_role));
    }
    for (const auto &upstream_node : producer->GetInDataNodes()) {
      const auto upstream = std::dynamic_pointer_cast<af::AscNode>(upstream_node);
      if (upstream != nullptr) {
        pending.emplace_back(upstream);
      }
    }
  }
  return af::SUCCESS;
}

af::Status AnnotateSimtFanoutBranches(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis,
                                      NodeSet &normal_schedule_nodes) {
  // [边界化] 输出链已退出标量求值器（gather-only SIMT）：主输出链与扇出分支均不再
  // 标注 SIMT 求值器角色，整链（含 side-input 闭包）交普通 VF 调度消费——此处若照常
  // 标注 kSimtInlineTransform/kSimtFanoutBranch 会把降级形态重新拉回标量求值器，
  // 发射期必然断言失败。
  if (ascgen_utils::indirect_load::IsSimtBoundaryFusion(indirect_load)) {
    return af::SUCCESS;
  }
  const af::AscNodePtr selected_root = analysis.post_reduce == nullptr
                                           ? analysis.output_store
                                           : ascgen_utils::indirect_load::GetSimtPostReduceLocalProducer(indirect_load);
  if (selected_root == nullptr) {
    return af::SUCCESS;
  }
  NodeSet selected;
  if (!CollectSimtBackwardRegion({selected_root}, indirect_load, selected)) {
    GELOGW("[IndirectLoad] Cannot identify SIMT main output chain from root[%s]; keep existing roles.",
           selected_root->GetNamePtr());
    return af::SUCCESS;
  }

  // 复合 Norm 场景：首个 Reduce 之后的后置链路（含汇入该链路的分支）由普通调度
  // 执行，与 lowering 侧 keep_chain 的取舍保持一致；该区域不能标注 SIMT 角色，
  // 否则会跳过主 tiling 而保持原始 rank，与参与 tiling 的 Reduce 输出 view 不一致。
  // 普通调度区域及其外部输入必须统一排除 SIMT 角色；否则父图不发射 Load，
  // VectorFunc 仍把该 Tensor 作为跨子图输入，最终出现 no API call found。
  GE_ASSERT_SUCCESS(CollectSimtNormalScheduleNodes(indirect_load, analysis, normal_schedule_nodes));
  GE_ASSERT_SUCCESS(AnnotateSimtMainOutputPath(indirect_load, selected_root, selected, normal_schedule_nodes));
  if (analysis.post_reduce != nullptr) {
    GELOGD("[IndirectLoad] SIMT post Reduce[%s] keeps %zu nodes on the normal schedule chain.",
           analysis.post_reduce->GetNamePtr(), normal_schedule_nodes.size());
  }
  GELOGD("[IndirectLoad] Normal schedule dependencies keep %zu nodes out of SIMT fanout annotation.",
         normal_schedule_nodes.size());

  NodeSet visited{indirect_load.get()};
  NodeSet side_visited;
  NodePath roots;
  for (const auto &out_node : indirect_load->GetOutDataNodes()) {
    const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out_node);
    GE_ASSERT_NOTNULL(consumer, "IndirectLoad output successor is invalid.");
    roots.emplace_back(consumer);
  }
  const auto visit = [&](const af::AscNodePtr &node, bool &stop) -> af::Status {
    const bool is_fanout_branch = selected.count(node.get()) == 0UL && !IsInputDataSource(node) &&
                                  !ScheduleUtils::IsReduce(node) && normal_schedule_nodes.count(node.get()) == 0UL;
    if (is_fanout_branch) {
      GELOGD("[IndirectLoad] SIMT fanout branch node[%s] annotated role[%ld].", node->GetNamePtr(),
             static_cast<int64_t>(ascgen_utils::indirect_load::TemplateRole::kSimtFanoutBranch));
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
          node, ascgen_utils::indirect_load::TemplateRole::kSimtFanoutBranch));
      GE_ASSERT_SUCCESS(
          AnnotateSimtSideInputClosure(node, indirect_load, selected, normal_schedule_nodes, side_visited));
    }
    if (ScheduleUtils::IsStore(node) || ScheduleUtils::IsReduce(node)) {
      stop = true;
      return af::SUCCESS;
    }
    return af::SUCCESS;
  };
  return TraverseOutputConsumers(roots, visited, visit);
}

// 为区域内全部统计 Reduce 种输入向量化视图：SIMD 多阶段链中每个 Reduce
// 都按模板向量化轴消费数据，缺失种子会导致 Codegen 无法组织 A/R 参数。
af::Status SeedCompositeReduceInputViews(const af::AscNodePtr &indirect_load,
                                         const std::vector<af::AscNodePtr> &composite_reduces) {
  for (const auto &reduce : composite_reduces) {
    GE_ASSERT_SUCCESS(SeedPostReduceInputVectorizedView(indirect_load, reduce));
  }
  return af::SUCCESS;
}

af::Status AnnotateSimtTemplateRoles(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis) {
  GE_ASSERT_NOTNULL(analysis.input_boundary, "IndirectLoad SIMT input boundary is missing.");
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
      analysis.input_boundary, ascgen_utils::indirect_load::TemplateRole::kSimtInputBoundary));
  if (analysis.input_boundary != analysis.input_root) {
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
        analysis.input_root, ascgen_utils::indirect_load::TemplateRole::kSimtInlineTransform));
  }
  GE_ASSERT_SUCCESS(
      ascgen_utils::indirect_load::SetTemplateRole(indirect_load, ascgen_utils::indirect_load::TemplateRole::kSimtOp));
  size_t annotated_count = 0UL;
  for (const af::AscNodePtr &node : analysis.index_region) {
    // 注意：index_region 即 lowering 侧 BuildSimtLoweringMetadata 会校验的集合
    // （index_root 与 post Reduce 输入链 backward）。其中同时落在 normal schedule
    // 依赖闭包内的节点（如与 IndirectLoad 输出直连、又汇入 post Reduce 输入的
    // Broadcast），lowering 侧仍按 SIMT 区域消费并要求 scalar evaluator 角色；
    // 不能被 normal_schedule 排除，否则出现 no scalar evaluator role 断言。
    // normal schedule 排除仅作用于 fanout 标注（AnnotateSimtFanoutBranches）。
    // Every node in the fused index/output backward region is emitted by the
    // SIMT scalar evaluator, including nodes that fan out to multiple
    // consumers inside that same region.  A fan-out role is reserved for
    // branches outside the selected scalar path and is assigned by
    // AnnotateSimtFanoutBranches below; marking an internal fan-out here would
    // make ValidateSimtRegionNode reject it as not inline-transformable.
    const auto role = (af::ops::IsOps<af::ascir_op::Load>(node) || af::ops::IsOps<af::ascir_op::Store>(node))
                          ? ascgen_utils::indirect_load::TemplateRole::kSimtDirectGmBoundary
                          : ascgen_utils::indirect_load::TemplateRole::kSimtInlineTransform;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(node, role));
    GELOGD("[IndirectLoad] SIMT region node[%s] annotated role[%ld].", node->GetNamePtr(), static_cast<int64_t>(role));
    ++annotated_count;
  }
  GELOGD("[IndirectLoad] SIMT index_region annotated %zu nodes (region size %zu), post reduce[%s].", annotated_count,
         analysis.index_region.size(), analysis.post_reduce == nullptr ? "none" : analysis.post_reduce->GetNamePtr());
  return af::SUCCESS;
}
}  // namespace indirect_load_detail
}  // namespace optimize
