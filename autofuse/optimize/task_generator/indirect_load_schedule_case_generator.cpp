/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_generator/indirect_load_schedule_case_generator.h"

#include "task_generator/indirect_load/indirect_load_graph_utils.h"
#include "task_generator/indirect_load/indirect_load_axis_normalize.h"
#include "task_generator/indirect_load/indirect_load_group_candidate.h"
#include "task_generator/indirect_load/indirect_load_simt_emission.h"
#include "task_generator/indirect_load/indirect_load_sk_partition.h"
#include "task_generator/indirect_load/indirect_load_view_layout.h"
#include "task_generator/indirect_load/indirect_load_input_path.h"
#include "task_generator/indirect_load/indirect_load_template_plan.h"
#include "task_generator/indirect_load/indirect_load_score_generator.h"
#include "task_generator/indirect_load/indirect_load_ub_feasibility.h"
#include "task_generator/indirect_load/indirect_load_output_analysis.h"
#include "task_generator/indirect_load/indirect_load_types.h"

#include "ascir_ops.h"
#include "ascir_ops_utils.h"
#include "v35/ascir/ascir_codegen_v2.h"
#include "ascgraph_info_complete.h"
#include "common_utils.h"
#include "graph_utils.h"
#include "graph/symbolizer/symbolic.h"
#include "indirect_load_utils.h"
#include "norm_utils.h"
#include "optimize/graph_pass/pass_utils.h"
#include "optimize/graph_pass/softmax_pattern_fusion_utils.h"
#include "schedule_utils.h"
#include "schedule_result.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace optimize {
namespace {

// —— 自 indirect_load_detail（拆分自本文件）引入的共享符号 ——

using indirect_load_detail::BuildPostReduceLayout;
using indirect_load_detail::CollectGatherNormInfo;
using indirect_load_detail::CollectRewrittenBoundaries;
// —— S3b 编排函数迁入 input_path ——
using indirect_load_detail::AnalyzeRewrittenGraph;
using indirect_load_detail::ApplyTemplatePathLayouts;
using indirect_load_detail::CollectRewrittenRegion;
using indirect_load_detail::CollectSimtBackwardRegion;

// —— S3 自 indirect_load_detail 引入 ——
using indirect_load_detail::AnnotateSimtGroupFanoutBranches;
using indirect_load_detail::ApplyGraphPassGroup;
using indirect_load_detail::ApplyGraphPassMixed;
using indirect_load_detail::ApplySkGraphPass;
using indirect_load_detail::BuildSkPartitionOrder;
using indirect_load_detail::CollectIndirectLoadEntry;
using indirect_load_detail::CopyBoundaryTensorAttr;
using indirect_load_detail::CopyWorkspaceTensorAttr;
using indirect_load_detail::FindUnemittableSimtEvaluatorNode;
using indirect_load_detail::GenerateGroupCandidates;
using indirect_load_detail::IndirectLoadGroup;
using indirect_load_detail::InsertWorkspaceBoundary;
using indirect_load_detail::IsGroupMemberLegal;
using indirect_load_detail::IsNormRegionDag;
using indirect_load_detail::PartitionNormReduceBoundaries;
using indirect_load_detail::PartitionSkGraph;
using indirect_load_detail::PlanSimtEmission;
using indirect_load_detail::PlanSimtFusedEmission;
using indirect_load_detail::SeedGroupMemberValueVectorizedViews;
using indirect_load_detail::SimtEmissionPlan;
using indirect_load_detail::SplitBoundaryFusionColumnTile;

// —— S1 自 indirect_load_detail 引入 ——
using indirect_load_detail::CollectGroupAccessStats;
using indirect_load_detail::CountRDomainChainTensors;
using indirect_load_detail::GenerateGroupScoreFunc;
using indirect_load_detail::GenerateScoreFunc;
using indirect_load_detail::GroupAccessStats;
using indirect_load_detail::GroupCandidateForm;
using indirect_load_detail::IsEmbeddingFastPathCapable;
using indirect_load_detail::IsGroupSimdUbDoomed;
using indirect_load_detail::IsGroupSimtL4UbDoomed;
using indirect_load_detail::IsGroupSimtNValueTileUbDoomed;
using indirect_load_detail::IsGroupUbAbsoluteDoomed;
using indirect_load_detail::IsSimtUbInfeasible;
using indirect_load_detail::IsSkRedirectRequiredForDynamicTail;
using indirect_load_detail::IsSkWindowUbInfeasible;
using indirect_load_detail::kIndirectLoadUbReservedBytes;

// —— S2 自 indirect_load_detail 引入 ——
using indirect_load_detail::AnalyzeInputPath;
using indirect_load_detail::AnnotateSimdTemplateRoles;
using indirect_load_detail::AnnotateSimtFanoutBranches;
using indirect_load_detail::AnnotateSimtMainOutputPath;
using indirect_load_detail::AnnotateSimtSideInputClosure;
using indirect_load_detail::AnnotateStridedUbPath;
using indirect_load_detail::AnnotateTemplate;
using indirect_load_detail::ApplyIndirectLoadPathLayout;
using indirect_load_detail::ApplyPhysicalView;
using indirect_load_detail::BuildAxisViewByBoundary;
using indirect_load_detail::BuildBroadcastLogicalView;
using indirect_load_detail::BuildGroupIndexExemptSet;
using indirect_load_detail::BuildLoadTransposeSourceView;
using indirect_load_detail::BuildSimdInnerAxis;
using indirect_load_detail::BuildSkInputInnerAxis;
using indirect_load_detail::CanEmitSimtScalar;
using indirect_load_detail::CanMoveInputPreNode;
using indirect_load_detail::ClearNormalScheduleSimtRoles;
using indirect_load_detail::ClearSimtTemplateRoles;
using indirect_load_detail::CollectInputPaths;
using indirect_load_detail::CollectInputRegionMembers;
using indirect_load_detail::CollectReachableLoads;
using indirect_load_detail::CollectSimtBackwardRegion;
using indirect_load_detail::CollectSimtFusedRegionMembers;
using indirect_load_detail::CollectSimtGroupIndexRegion;
using indirect_load_detail::CollectSimtNormalScheduleNodes;
using indirect_load_detail::CollectSimtOutputConsumerClosure;
using indirect_load_detail::CompleteInputDataTensorAttrs;
using indirect_load_detail::CompletePreservedVectorizedViews;
using indirect_load_detail::CreateFixedTileSplit;
using indirect_load_detail::FillSkWindowReuseInfo;
using indirect_load_detail::FinalizeTemplate;
using indirect_load_detail::FoldBroadcastPath;
using indirect_load_detail::FoldSimtIndexChainExternalStores;
using indirect_load_detail::GetBroadcastPhysicalAttr;
using indirect_load_detail::GetSimtGroupSize;
using indirect_load_detail::HasBroadcastInsideWindow;
using indirect_load_detail::HasBroadcastMultiInputNode;
using indirect_load_detail::InlineBroadcastPath;
using indirect_load_detail::IsInputDataSource;
using indirect_load_detail::IsInputRegionBoundary;
using indirect_load_detail::IsSimdInputShapeEnlarged;
using indirect_load_detail::IsSimtTemplateRole;
using indirect_load_detail::IsSkTemplateCandidateLegal;
using indirect_load_detail::IsSupportedBroadcastPath;
using indirect_load_detail::IsSupportedIndirectLoadTopologyNode;
using indirect_load_detail::LowersToStridedSimdFacade;
using indirect_load_detail::MemberScoreLeaves;
using indirect_load_detail::MergeAxesForTemplate;
using indirect_load_detail::MoveInputPreNode;
using indirect_load_detail::NeedsAlignedUbWindow;
using indirect_load_detail::NormalizeAxesForTemplate;
using indirect_load_detail::NormalizeSimdAxesForGroupTemplate;
using indirect_load_detail::NormalizeSimdAxesForMember;
using indirect_load_detail::NormalizeSimdAxesForTemplate;
using indirect_load_detail::NormalizeSimtAxesForGroupTemplate;
using indirect_load_detail::NormalizeSimtAxesForTemplate;
using indirect_load_detail::NormalizeTemplateAxes;
using indirect_load_detail::PreparePhysicalViews;
using indirect_load_detail::ReplaceAxisPrefix;
using indirect_load_detail::ReplaceRegionAxisPrefix;
using indirect_load_detail::ResolveInputPhysicalView;
using indirect_load_detail::RestoreSkTemplateAxes;
using indirect_load_detail::RewriteBroadcastPaths;
using indirect_load_detail::RewriteInputPreNodes;
using indirect_load_detail::RewriteSimdInputLayouts;
using indirect_load_detail::RewriteSimtInputBroadcast;
using indirect_load_detail::RewriteSkInputPaths;
using indirect_load_detail::SeedCompositeReduceInputViews;
using indirect_load_detail::SeedGroupMemberValueVectorizedViews;
using indirect_load_detail::SeedPostReduceInputVectorizedView;
using indirect_load_detail::TryGetIndirectLoadPlatformUbSize;
using indirect_load_detail::TryGetTensorLogicalBytes;
using indirect_load_detail::ValidateIndirectLoadNode;
using indirect_load_detail::ValidateIndirectLoadOutputVectorizedAxes;
using indirect_load_detail::ValidateSimdCompositeLayouts;
using indirect_load_detail::ValidateSimtTemplateRegion;
using indirect_load_detail::ValidateTemplate;

using indirect_load_detail::GetFoldableLoadTransposeSource;
using indirect_load_detail::GetIndirectLoadAxisIndex;
using indirect_load_detail::GetLoadTransposeSource;
using indirect_load_detail::HasControlEdge;
using indirect_load_detail::HasReduceBeforeGatherAxis;
using indirect_load_detail::InputViewPlan;
using indirect_load_detail::IsBroadcastNode;
using indirect_load_detail::IsShapePreservingTailAxisNorm;
using indirect_load_detail::IsSimtArPostReduce;
using indirect_load_detail::IsSingleConsumerWithoutControlEdge;
using indirect_load_detail::kEmbeddingAlternateFastPathScore;
using indirect_load_detail::kEmbeddingFastPathScore;
using indirect_load_detail::kEmbeddingSimdLookupCountThreshold;
using indirect_load_detail::kEmbeddingSimdPayloadBytesThreshold;
using indirect_load_detail::kIndirectLoadInputCount;
using indirect_load_detail::kIndirectLoadInvalidAxisIndex;
using indirect_load_detail::kIndirectLoadOutputCount;
using indirect_load_detail::kIndirectLoadSimtDcacheSize;
using indirect_load_detail::kInnerAxisName;
using indirect_load_detail::kInputInnerAxisName;
using indirect_load_detail::kInvalidBroadcastIndex;
using indirect_load_detail::kOuterAxisName;
using indirect_load_detail::kSimtStridedFallbackScore;
using indirect_load_detail::kSingleOuterAxisName;
using indirect_load_detail::NodePath;
using indirect_load_detail::NodeSet;
using indirect_load_detail::PhysicalViewPreparation;
using indirect_load_detail::PostReduceLayout;
using indirect_load_detail::ReduceAxisKind;
using indirect_load_detail::RewrittenGraphAnalysis;
using indirect_load_detail::TemplateCase;
using indirect_load_detail::TraverseInputProducers;
using indirect_load_detail::TraverseOutputConsumers;
using indirect_load_detail::ValidateOutputStoreTargets;
using indirect_load_detail::ValidateSimdPostReduceLayout;
using indirect_load_detail::ValidateSimtPostReduceLayout;

// 记录候选被拒绝：动态尾轴重定向触发的拒绝回传信号（供 Generate 回退常规 SK），
// 其余拒绝仅打日志。
void MarkCandidateRejected(ascir::TemplateId template_id, const af::AscNodePtr &indirect_load,
                           bool sk_dynamic_redirect_fired, bool *sk_dynamic_redirect_rejected) {
  GELOGI("[IndirectLoad] Reject template[%d] for node[%s]: post-reduce or region validation failed.",
         static_cast<int32_t>(template_id), indirect_load->GetNamePtr());
  if (sk_dynamic_redirect_fired && sk_dynamic_redirect_rejected != nullptr) {
    *sk_dynamic_redirect_rejected = true;
  }
}

// 值宽度 × 模板可行性过滤：64 位值（int64/uint64）由 SIMD 8 字节 ValuePolicy
// （RegTraitNumTwo）与 SIMT 支持；1 字节值（int8/uint8/bool）当前仅 SIMT 标量路径
// 支持（SIMD b8 gather 语义待定）；SK 的 AscendC::Gather 对 8 字节与 1 字节均不
// 支持，需跳过。不可行时调用方置 is_candidate_legal=false 结束本候选。
bool IsValueDtypeSupportedByTemplate(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id) {
  const auto value_dtype = indirect_load->inputs()[ascgen_utils::indirect_load::kInputTensorIndex]->attr.dtype;
  const bool is_64bit_value = value_dtype == af::DT_INT64 || value_dtype == af::DT_UINT64;
  const bool is_1byte_value = value_dtype == af::DT_INT8 || value_dtype == af::DT_UINT8 || value_dtype == af::DT_BOOL;
  if ((is_64bit_value || is_1byte_value) && template_id == ascir::TemplateId::kIndirectLoadSK) {
    GELOGI("[IndirectLoad] value dtype[%d] is unsupported by SK, skip template[%d] for node[%s].",
           static_cast<int32_t>(value_dtype), static_cast<int32_t>(template_id), indirect_load->GetNamePtr());
    return false;
  }
  if (is_1byte_value && template_id != ascir::TemplateId::kIndirectLoadSimt) {
    GELOGI("[IndirectLoad] 1-byte value dtype[%d] only supports SIMT, skip template[%d] for node[%s].",
           static_cast<int32_t>(value_dtype), static_cast<int32_t>(template_id), indirect_load->GetNamePtr());
    return false;
  }
  return true;
}

// [SK段SIMT化] index 源头不收敛且全量窗口超出单核 UB 的形态（如
// gather(axis=0) 逐元素 index + 尾轴大 R）：IndirectLoadSk 行窗口模型退化为
// 全量不可行。保留 SK 的 output workspace 边界（消费子图/Reduce 段独立调度，
// 承载 R 轴分 tile 归约），IL 段整体重定向为 SIMT 模板走全套流程（input/
// index GM 直读，输出 tile 写 output workspace）——SK 分段设计对 R 非全载
// 场景的本意。softmax 消费链在此形态下不规范化（见 Generate），以原始
// Pattern 链作为 Norm Region DAG 进入分段；复合 Norm 区域（多 Reduce）同理。
// 门禁放宽：SIMT 候选可能因布局校验被拒（如 mean(dim=-2) 中间轴归约，
// IsSimtUbInfeasible 只看尾轴 UB 下界无法覆盖）——前序 SIMD/SIMT 候选均不可行时
// SK 是最后活路，此时即使 SIMT 的 UB 下界可行也必须重定向。
// 动态尾轴（非收敛 index + 尾轴运行期符号）也需重定向；分段候选为任意
// 运行期 R 提供可解的 UB 兜底，小 R 仍保留可行的 inline 候选。
// suppress_sk_redirect 置位时跳过重定向走常规 SK 路径，供 Generate 在分段
// 候选被校验拒绝后回退重试。返回 true 表示已重定向为 SIMT 模板（template_id
// 由调用方更新）；dynamic_tail_fired 出参标记本次拒绝源是否为动态尾轴重定向。
bool TryRedirectSkCandidateToSimt(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                  ascir::TemplateId template_id, bool has_live_candidate, bool suppress_sk_redirect,
                                  bool &dynamic_tail_fired) {
  if (template_id != ascir::TemplateId::kIndirectLoadSK) {
    return false;
  }
  const bool window_redirect = !suppress_sk_redirect && indirect_load_detail::IsSkWindowUbInfeasible(indirect_load);
  const bool dynamic_tail_redirect =
      !suppress_sk_redirect && indirect_load_detail::IsSkRedirectRequiredForDynamicTail(graph, indirect_load);
  // [SK-segmented SIMT for R-before-gather-axis] Layouts whose post-Reduce axis precedes the gather
  // axis cannot fuse the reduce into a non-SK row window, so whole-graph SIMD/SIMT candidates stay
  // gated and the SK partition remains the semantic vehicle. This trigger turns such graphs into
  // the segmented form (SK partition kept + SIMT-executed gather segment with its native SIMT
  // tiling) so it can compete with the plain SK candidate instead of degrading to single-core SK
  // windows (measured 2113us vs no_af 34us, ratio 0.016 on the SliceEmbeddingReduce workload).
  const bool rbg_redirect = !suppress_sk_redirect && HasReduceBeforeGatherAxis(indirect_load);
  if (!(window_redirect || dynamic_tail_redirect || rbg_redirect) || !IsNormRegionDag(indirect_load)) {
    return false;
  }
  // The UB fallback preference only applies to the legacy window/dynamic-tail triggers; the
  // R-before-gather-axis segmented form is a target candidate, not a fallback, and registers
  // unconditionally (it competes with the separately registered plain SK candidate).
  if (!rbg_redirect && !indirect_load_detail::IsSimtUbInfeasible(graph, indirect_load) && has_live_candidate &&
      !dynamic_tail_redirect) {
    return false;
  }
  GELOGI("[IndirectLoad] SK candidate redirects IL to SIMT (window exceeds UB or R-before-gather-axis), node[%s].",
         indirect_load->GetNamePtr());
  dynamic_tail_fired = dynamic_tail_redirect;
  // 先 Norm 边界化再 output 边界：闭包按 IL 下游数据流计算，output 边界插入后
  // IL 直接消费者变为边界 store，Norm 链被隔断在闭包之外（生产 9.28 复验：
  // 边界 0 个、分区 0 子图、has_none_graph）。Norm 边界先行，IL 直连 Norm 链首，
  // 闭包完整；随后 output 边界插在 IL 与 Norm 链首之间。
  GE_ASSERT_SUCCESS(indirect_load_detail::PartitionNormReduceBoundaries(graph, indirect_load));
  GE_ASSERT_SUCCESS(indirect_load_detail::PartitionSkGraph(graph, indirect_load, false, false,
                                                           /*skip_input_boundaries=*/true));
  GE_ASSERT_SUCCESS(ascir::SetSkSegmentedSimt(indirect_load, true));
  return true;
}

// SIMT 候选的 normal-schedule 角色标注：豁免集即 lowering 侧
// BuildSimtLoweringMetadata 会校验的 index_region，其中与 normal_schedule
// 闭包重叠的共享节点不能被兜底清理掉角色。
af::Status AnnotateSimtScheduleRoles(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                     const RewrittenGraphAnalysis &analysis) {
  NodeSet normal_schedule_nodes;
  GE_ASSERT_SUCCESS(AnnotateSimtFanoutBranches(indirect_load, analysis, normal_schedule_nodes));
  NodeSet lowering_exempt;
  for (const auto &region_node : analysis.index_region) {
    lowering_exempt.emplace(region_node.get());
  }
  return indirect_load_detail::ClearNormalScheduleSimtRoles(graph, normal_schedule_nodes, lowering_exempt);
}

af::Status ApplyGraphPass(af::AscGraph &graph, const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                          bool &is_candidate_legal, bool has_live_candidate = true,
                          bool *sk_dynamic_redirect_rejected = nullptr, bool suppress_sk_redirect = false) {
  GELOGD("[IndirectLoad] Apply graph pass for node[%s], template_id[%d].", indirect_load->GetNamePtr(),
         static_cast<int32_t>(template_id));
  is_candidate_legal = true;
  // [动态重定向回退信号] 分段重定向候选被校验拒绝时由调用方（Generate）回退常规
  // SK 候选；sk_dynamic_redirect_fired 仅在拒绝源自动态尾轴重定向时置位（静态窗口
  // 超限的 SK 全载候选必然被 StaticUbTemplateFilter 淘汰，回退无价值）。
  bool sk_dynamic_redirect_fired = false;
  if (!IsValueDtypeSupportedByTemplate(indirect_load, template_id)) {
    is_candidate_legal = false;
    return af::SUCCESS;
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSK) {
    if (TryRedirectSkCandidateToSimt(graph, indirect_load, template_id, has_live_candidate, suppress_sk_redirect,
                                     sk_dynamic_redirect_fired)) {
      template_id = ascir::TemplateId::kIndirectLoadSimt;
    } else if (!sk_dynamic_redirect_fired) {
      return indirect_load_detail::ApplySkGraphPass(graph, indirect_load, is_candidate_legal);
    } else {
      return af::SUCCESS;
    }
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    GE_ASSERT_SUCCESS(indirect_load_detail::ClearSimtTemplateRoles(graph));
  }
  RewrittenGraphAnalysis analysis;
  GE_ASSERT_SUCCESS(AnalyzeRewrittenGraph(graph, indirect_load, template_id, is_candidate_legal, analysis));
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(CompleteInputDataTensorAttrs(analysis));

  // [边界化两阶段判定] SIMT 求值器把 IL 前向链整体折叠为标量表达式，链上出现无 V2
  // 标量实现的算子（pow/fmod/remainder 等，按 V2 能力查询判定，不设算子白名单）时
  // 发射期必然断言失败——降级为边界化形态：SIMT 只做 gather，value 逐 tile 写 UB
  // 局部目标（组员空链同构），输出链交普通 VF 代码在同一循环迭代内消费。
  // 判定对象为“求值器消费链”：无 post-Reduce 时即整条输出链；有 post-Reduce 时为其
  // Reduce 之前的 producer 链（Reduce 及其后置链本就由普通调度执行，不进入求值器）。
  // 判定先于区域校验与角色注解执行：融合区域（index_region）同步收缩为 index 链，
  // ValidateSimtTemplateRegion 的“求值器消费集全部可发射”校验收缩为 index 链校验
  // （index 链无法退出求值器——index 计算必须折叠，不可发射时直接拒绝候选），
  // AnnotateSimtTemplateRoles/AnnotateSimtFanoutBranches 在收缩后的区域上工作（输出链
  // 不再获得求值器角色）。形状保持 Norm（Softmax）的输入视图播种与常规 Reduce 不同，
  // 维持既有路径不参与本判定；SK 分段重定向形态（output workspace 边界）同理。
  const bool is_shape_preserving_norm =
      analysis.post_reduce != nullptr && af::ops::IsOps<af::ascir_op::Softmax>(analysis.post_reduce);
  bool boundary_fusion = false;
  if (template_id == ascir::TemplateId::kIndirectLoadSimt && !is_shape_preserving_norm &&
      !ascir::IsSkSegmentedSimt(*indirect_load) && analysis.output_store != nullptr) {
    // output_chain：从 Store 回溯的整条输出链，用于把输出侧闭包移出 SIMT 求值器；
    // evaluator_chain：当前形态下真正进入标量求值器的节点集，用于可发射性判定。
    NodeSet output_chain;
    NodeSet evaluator_chain;
    bool chain_ready = CollectSimtBackwardRegion({analysis.output_store}, indirect_load, output_chain);
    if (chain_ready) {
      if (analysis.post_reduce == nullptr) {
        evaluator_chain = output_chain;
      } else {
        const af::AscNodePtr local_producer =
            ascgen_utils::indirect_load::GetSimtPostReduceLocalProducer(indirect_load);
        chain_ready =
            local_producer != nullptr && CollectSimtBackwardRegion({local_producer}, indirect_load, evaluator_chain);
      }
    }
    if (chain_ready) {
      const auto unemittable = FindUnemittableSimtEvaluatorNode(graph, evaluator_chain);
      if (unemittable != nullptr) {
        const size_t output_rank = indirect_load->outputs()[0]->attr.axis.size();
        // 无 post-Reduce 时边界化轴布局要求 outer/inner 分离（boundary=rank-1）：rank-1
        // 输出无行结构可分离，逐 tile 局部目标形态退化，拒绝候选回退其他模板。有
        // post-Reduce 时 boundary 取 first_reduce，由 Reduce 布局提供行/归约段分离。
        if (analysis.post_reduce == nullptr && output_rank <= 1UL) {
          is_candidate_legal = false;
          GELOGI(
              "[IndirectLoad] Reject SIMT boundary fusion candidate: rank-%zu output has no separable row axis, "
              "node[%s].",
              output_rank, indirect_load->GetNamePtr());
          return af::SUCCESS;
        }
        boundary_fusion = true;
        GELOGI(
            "[IndirectLoad] SIMT evaluator chain node[%s] has no scalar emission; fall back to boundary fusion "
            "(gather-only SIMT), post_reduce[%s], node[%s].",
            unemittable->GetNamePtr(), analysis.post_reduce == nullptr ? "none" : analysis.post_reduce->GetNamePtr(),
            indirect_load->GetNamePtr());
        // 融合区域收缩：输出链闭包（含汇入链中的 side-input）退出 SIMT 求值器，
        // index_region 收缩为 index 链（与多 gather 组员计划同构）。有 post-Reduce 时
        // 一并退出的是整条输出链（含 Reduce 及其后置链，均转普通调度执行）。
        analysis.index_region.erase(std::remove_if(analysis.index_region.begin(), analysis.index_region.end(),
                                                   [&output_chain](const af::AscNodePtr &node) {
                                                     return output_chain.count(node.get()) != 0UL;
                                                   }),
                                    analysis.index_region.end());
      }
    }
  }
  if (boundary_fusion && analysis.output_store != nullptr && analysis.post_reduce == nullptr) {
    // [边界化列分块] 无 post-Reduce 时下游 VF 链 tile 块 = 行 × 全列会超平台 UB，
    // 拆分列轴收敛块大小。有 post-Reduce 时 tile 已由归约段（inner=归约段×后缀）
    // 界定，拆列会改变 Reduce 的输入视图，保持原布局。
    GE_ASSERT_SUCCESS(
        SplitBoundaryFusionColumnTile(graph, indirect_load, analysis.output_store, analysis.index_region));
  }
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetSimtBoundaryFusion(indirect_load, boundary_fusion));

  // The SIMT template overwrites this boundary from its post-reduce layout; without a post Reduce the
  // full output rank is the boundary. The SIMD template does not use this value.
  size_t boundary = indirect_load->outputs()[0]->attr.axis.size();
  GE_ASSERT_SUCCESS(ValidateTemplate(indirect_load, template_id, analysis, boundary, is_candidate_legal));
  if (!is_candidate_legal) {
    MarkCandidateRejected(template_id, indirect_load, sk_dynamic_redirect_fired, sk_dynamic_redirect_rejected);
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(AnnotateTemplate(indirect_load, template_id, analysis));
  if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    GE_ASSERT_SUCCESS(AnnotateSimtScheduleRoles(graph, indirect_load, analysis));
  }
  GE_ASSERT_SUCCESS(NormalizeTemplateAxes(graph, indirect_load, template_id, analysis, boundary));
  // [边界化视图播种] value tensor 的向量化视图按模板 vectorized 轴播种（与组员同构）：
  // 行为复位使 preserves_vectorized_axis 失效，调度期通用 DeriveNodeVectorizedAxes 会
  // 重算写入，播种提供调度前的中间态防护（actual_size/offset 表达式的量纲来源，
  // CalcVectorizedElementCount 读该视图）。
  if (ascgen_utils::indirect_load::IsSimtBoundaryFusion(indirect_load)) {
    GE_ASSERT_SUCCESS(SeedGroupMemberValueVectorizedViews({indirect_load}, is_candidate_legal));
    if (!is_candidate_legal) {
      return af::SUCCESS;
    }
  }
  GE_ASSERT_SUCCESS(CompletePreservedVectorizedViews(graph));
  GE_ASSERT_SUCCESS(FinalizeTemplate(indirect_load, template_id));
  bool metadata_supported = false;
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::FinalizeLoweringMetadata(indirect_load, metadata_supported));
  if (!metadata_supported) {
    is_candidate_legal = false;
    MarkCandidateRejected(template_id, indirect_load, sk_dynamic_redirect_fired, sk_dynamic_redirect_rejected);
  }
  return af::SUCCESS;
}

// 六级降级阶梯 L2/L3/L4 判定（全部 schedule 侧编译期判定，算法与常量见设计文档
// 「六级降级阶梯（六级终态设计）」章节）：
// - L4（V2 多 evaluator）：资源压力模型超最低档配额（模型规格与常量依据见设计文档
//   「寄存器压力模型」章节）。
// - L2（寄存器归并）：归并树（成员输出前向闭包，成员为边界）全部节点可标量发射、无共享
//   Reduce 终端、side input 可整体标量发射——Scalar/ScalarData/GM load 直接支持；③ 区域外
//   elewise 链经后向闭包拉入发射（链上 Load 收为 side load）；② 视图匹配成员 index load 的
//   GM load 无需特判（use_logical_offset 坐标折叠按自身物理视图寻址）。
// - L3（UB 归并）：其余情形——各路 value 写 UB 局部目标，归并链向量代码执行。
// 防漂移：此处仅做门控与角色标注；side load 寻址分类与输出链 tensor id 由 codegen 在
// codegen 期图上用同一分类函数重建（节点名跨图副本稳定，视图/张量 id 会漂移）。
// 寄存器压力模型常数与档位映射已上移至 indirect_load_utils.h（kSimtFusedRegister* /
// SimtFusedRegisterDemand / SimtFusedThreadTier，位宽感知——调度期门控与 codegen 漂移
// 防护共用；取值依据与推导见设计文档「寄存器压力模型」）。

// ==================== 调度期组候选 UB 可行性预估（硬下界，保守放行） ====================
// 与单 gather 的 IsSkWindowUbInfeasible/IsSimtUbInfeasible 同源思路：只估"再怎么缩
// tile 也省不掉的硬下界"；平台信息不可得或任一成员 shape 动态时保守放行（并存竞标下
// 漏放行的后果是 tiling 期资源否决后由 SK 兜底候选接管，误放行的后果是白白损失融合
// 机会）。定位为编译期优化（跳过资源下界超限候选的生成与求解），非正确性必需。

af::Status PartitionTaskGraph(const ascir::ImplGraph &graph, const af::AscNodePtr &indirect_load, bool is_sk_template,
                              std::vector<ascir::ImplGraph> &grouped_graphs) {
  if (!is_sk_template) {
    return ScheduleGroupGraphPartitioner::PartitionByConnectivity(graph, grouped_graphs);
  }
  std::vector<af::AscNodePtr> node_order;
  GE_ASSERT_SUCCESS(BuildSkPartitionOrder(graph, indirect_load, node_order));
  GE_ASSERT_SUCCESS(ScheduleGroupGraphPartitioner::PartitionByConnectivity(graph, grouped_graphs, node_order));
  return RestoreSkTemplateAxes(grouped_graphs);
}

af::Status RefreshTaskAxisSizes(std::vector<ascir::ImplGraph> &grouped_graphs, bool need_update_axis,
                                bool is_sk_template) {
  if ((!need_update_axis && !is_sk_template) || grouped_graphs.size() <= 1UL) {
    return af::SUCCESS;
  }
  for (auto &subgraph : grouped_graphs) {
    if (ScheduleUtils::FindFirstNodeOfType<af::ascir_op::Concat>(subgraph) != nullptr) {
      continue;
    }
    af::AscNodePtr subgraph_indirect_load;
    if (is_sk_template) {
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::ValidateSingleIndirectLoadNode(subgraph, subgraph_indirect_load));
    }
    if (subgraph_indirect_load == nullptr) {
      GE_ASSERT_SUCCESS(ScheduleGroupGraphPartitioner::RefreshAxisSize(subgraph));
    }
  }
  return af::SUCCESS;
}

af::Status ReduceTaskGraphCount(std::vector<ascir::ImplGraph> &grouped_graphs, const OptimizerOptions &options) {
  if (options.graph_type != GraphType::kFusedAscBackend) {
    return af::SUCCESS;
  }
  const auto backend_spec = BackendSpec::GetInstance();
  GE_ASSERT_NOTNULL(backend_spec);
  return ScheduleGroupGraphPartitioner::ReduceGraphCount(grouped_graphs, backend_spec->max_group_num_per_compile_unit);
}

af::Status FinalizeGroupedGraphLoweringMetadata(std::vector<ascir::ImplGraph> &grouped_graphs) {
  for (auto &graph : grouped_graphs) {
    // 多 gather 水平融合：同一 grouped graph 可包含组内全部 IndirectLoad，逐个校验。
    for (const af::AscNodePtr &indirect_load : ascgen_utils::indirect_load::GetAllIndirectLoadNodes(graph)) {
      const auto template_id = ascir::GetTemplateIdOrDefault(*indirect_load);
      // FinalizeLoweringMetadata 仅支持 SIMD/SIMT：SK 的分区图不含该 lowering
      // 元数据，不进入本循环的元数据检查。
      if (template_id != ascir::TemplateId::kIndirectLoadSimd && template_id != ascir::TemplateId::kIndirectLoadSimt) {
        continue;
      }
      // Norm 区域完整性：区域节点必须与 IndirectLoad 同组，否则统计分支与
      // 原始输入回流被拆到不同 Kernel。SK 除外：SK 的 workspace 分区本身就是
      // 把 Gather 与后续计算合法拆到不同 grouped graph，区域节点位于其他分区
      // 属预期形态，仅记录日志不阻断。
      ascgen_utils::norm::NormInfo norm_info;
      GE_ASSERT_SUCCESS(ascgen_utils::norm::TryGetNormInfo(indirect_load, norm_info));
      if (norm_info.kind != ascgen_utils::norm::NormInfo::Kind::kNone && !norm_info.region_node_names.empty()) {
        const bool is_sk_split = template_id == ascir::TemplateId::kIndirectLoadSK;
        for (const auto &node_name : norm_info.region_node_names) {
          if (graph.FindNode(node_name.c_str()) == nullptr) {
            GE_ASSERT_TRUE(!is_sk_split, "Norm region node[%s] is split out of grouped graph[%s], IndirectLoad[%s].",
                           node_name.c_str(), graph.GetName().c_str(), indirect_load->GetNamePtr());
            GELOGI("[IndirectLoad] Norm region node[%s] resides in another SK workspace partition of graph[%s].",
                   node_name.c_str(), graph.GetName().c_str());
          }
        }
      }
      bool metadata_supported = false;
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::FinalizeLoweringMetadata(indirect_load, metadata_supported));
      GE_ASSERT_TRUE(metadata_supported,
                     "IndirectLoad lowering metadata is unsupported in grouped graph[%s], node[%s].",
                     graph.GetName().c_str(), indirect_load->GetNamePtr());
    }
  }
  return af::SUCCESS;
}
}  // namespace

Status IndirectLoadScheduleCaseGenerator::Generate(ascir::HintGraph &graph, std::vector<ascir::ImplGraph> &graphs,
                                                   std::vector<std::string> &score_functions) {
  af::AscNodePtr indirect_load;
  IndirectLoadGroup group;
  bool is_group = false;
  GE_ASSERT_SUCCESS(CollectIndirectLoadEntry(graph, indirect_load, group, is_group));
  if (indirect_load == nullptr && !is_group) {
    return af::SUCCESS;
  }
  if (is_group) {
    return GenerateGroupCandidates(graph, group, graphs, score_functions);
  }
  GE_ASSERT_SUCCESS(ValidateIndirectLoadNode(indirect_load));
  GELOGI("[IndirectLoad] Generate schedule candidates for graph[%s], node[%s].", graph.GetName().c_str(),
         indirect_load->GetNamePtr());
  const std::string indirect_load_name = indirect_load->GetName();
  const TemplateCase cases[] = {
      {ascir::TemplateId::kIndirectLoadSimd, ascgen_utils::indirect_load::Implementation::kDefault},
      {ascir::TemplateId::kIndirectLoadSimd, ascgen_utils::indirect_load::Implementation::kGatherApi},
      {ascir::TemplateId::kIndirectLoadSimt, ascgen_utils::indirect_load::Implementation::kDefault},
      {ascir::TemplateId::kIndirectLoadSK, ascgen_utils::indirect_load::Implementation::kDefault}};
  // 首个 SIMD 候选（kDefault）的 lowering facade 判定结果：退化到 strided 时给后续 SIMT
  // 候选加分。facade 判定与实现无关（kGatherApi 同样先受 strided 判定约束），故取首个 SIMD 候选。
  bool simd_lowers_to_strided = false;
  for (const TemplateCase &template_case : cases) {
    const ascir::TemplateId template_id = template_case.template_id;
    if (HasReduceBeforeGatherAxis(indirect_load) && template_id != ascir::TemplateId::kIndirectLoadSK) {
      GELOGI("[IndirectLoad] Skip non-SK candidate[%d] for R-before-gather-axis post-Reduce layout, node[%s].",
             static_cast<int32_t>(template_id), indirect_load->GetNamePtr());
      continue;
    }
    // [SK-segmented SIMT for R-before-gather-axis] For these layouts the SK partition stays the
    // semantic vehicle (the reduce is bridged through workspace segments), and the gather segment
    // may execute with the SIMT template under its native tiling. Register the segmented variant
    // first; the plain SK candidate below (registered with the redirect suppressed) remains as the
    // scoring fallback. Both share the same partition semantics.
    if (template_id == ascir::TemplateId::kIndirectLoadSK && HasReduceBeforeGatherAxis(indirect_load)) {
      ascir::ImplGraph segmented_graph(graph.GetName().c_str());
      GE_ASSERT_TRUE(segmented_graph.CopyFrom(graph), "Failed to copy graph [%s] for segmented candidate.",
                     graph.GetName().c_str());
      const af::AscNodePtr segmented_il = segmented_graph.FindNode(indirect_load_name.c_str());
      GE_ASSERT_NOTNULL(segmented_il, "Failed to find copied IndirectLoad node[%s] for segmented candidate.",
                        indirect_load_name.c_str());
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetImplementation(segmented_il, template_case.implementation));
      // Keep the original Softmax pattern for the redirected form: segmented tiling cuts R and a
      // normalized SoftmaxAR full-load form cannot be statically verified (same policy as the
      // window/dynamic-tail redirect keep-pattern guard), so no normalization is applied here.
      bool segmented_legal = false;
      GE_ASSERT_SUCCESS(ApplyGraphPass(segmented_graph, segmented_il, template_id, segmented_legal, !graphs.empty(),
                                       nullptr, /*suppress_sk_redirect=*/false));
      // [SK段SIMT化·注册守卫] ApplyGraphPass 在重定向未触发时（如 !IsNormRegionDag）
      // 会静默回落 ApplySkGraphPass，副本实为普通 SK 图而 segmented_legal 仍为 true。
      // 注册前必须确认重定向真实发生（模板已切为 SIMT 且带 SK 分段标记），否则该图
      // 会以 SIMT 计分身份与下方普通 SK 候选重复注册，且其未规范化的 Softmax 副本
      // 可能因平票按发射序胜出，绕过普通 SK 依赖的规范化产物。
      const bool segmented_redirect_fired =
          segmented_legal && ascir::GetTemplateIdOrDefault(*segmented_il) == ascir::TemplateId::kIndirectLoadSimt &&
          ascir::IsSkSegmentedSimt(*segmented_il);
      if (segmented_redirect_fired) {
        ascgen_utils::indirect_load::IndirectLoadAccessInfo segmented_access_info;
        GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetIndirectLoadAccessInfo(segmented_il, segmented_access_info));
        graphs.emplace_back(std::move(segmented_graph));
        score_functions.emplace_back(GenerateScoreFunc(
            TemplateCase{ascir::TemplateId::kIndirectLoadSimt, ascgen_utils::indirect_load::Implementation::kDefault},
            segmented_access_info, segmented_il, graphs.back(), /*simd_lowers_to_strided=*/false));
        GELOGI("[IndirectLoad] Add SK-segmented SIMT schedule candidate for node[%s] (R-before-gather-axis).",
               indirect_load->GetNamePtr());
      } else if (segmented_legal) {
        GELOGW(
            "[IndirectLoad] SK->SIMT redirect did not fire for node[%s] (not a NormRegionDag?); skip the"
            " segmented candidate, plain SK fallback remains.",
            indirect_load->GetNamePtr());
      } else {
        GELOGW("[IndirectLoad] SK-segmented SIMT candidate rejected for node[%s]; plain SK fallback remains.",
               indirect_load->GetNamePtr());
      }
    }
    ascir::ImplGraph candidate_graph(graph.GetName().c_str());
    GE_ASSERT_TRUE(candidate_graph.CopyFrom(graph), "Failed to copy graph [%s].", graph.GetName().c_str());
    af::AscNodePtr candidate_indirect_load = candidate_graph.FindNode(indirect_load_name.c_str());
    GE_ASSERT_NOTNULL(candidate_indirect_load, "Failed to find copied IndirectLoad node[%s].",
                      indirect_load_name.c_str());
    GE_ASSERT_SUCCESS(
        ascgen_utils::indirect_load::SetImplementation(candidate_indirect_load, template_case.implementation));
    // 候选图副本内规范化直接后置稳定 Softmax Pattern：主图因 Reduce Generator
    // 遇 IndirectLoad 提前返回而不会执行 Softmax 替换，必须在副本内完成，
    // 使 CollectOutputBoundaries 看到的是专用 Softmax 节点而非 Max/Sum 双 Reduce。
    // 仅替换原始输入直接来自本 IndirectLoad 输出的 Pattern；Pattern 未命中或
    // 中间有 Elementwise 时保持原图形态，由通用复合区域路径处理。例外：SK 窗口
    // 不可行的候选不做规范化，保留原始 Pattern 链——专用 Softmax 节点按整行一次
    // 归一化，被分段 tiling 切 R 且无跨 tile 合并协议会产生精度错误，而原始链是
    // Norm Region DAG，可按 SK 分段形态正确归约（重定向见 ApplyGraphPass）；窗口
    // 可行时维持规范化，SIMD/SIMT 候选不受影响仍走 SoftmaxAR 内联路径。
    bool softmax_normalized = false;
    // [SK段SIMT化·动态shape] 动态尾轴的 SK 候选同样不规范化：规范化 Softmax 的
    // SoftmaxAR 全载形态对符号 R 不可静态验证（真机大 R 下 tiling 必败），重定向后的
    // 分段形态才能兜底；见 IsSkRedirectRequiredForDynamicTail。
    // 保留 Pattern 的前提是"本候选确实会走重定向"：R>G 布局的普通 SK 候选固定抑制
    // 重定向（段化候选已单独注册），按普通 SK 执行需要规范化产物（与动态尾轴回退
    // 分支的显式规范化同语义），不得沿用保留策略，否则 rbg ∩ 动态尾轴时该候选既不
    // 重定向也不规范化。
    const bool suppress_sk_redirect_for_plain_candidate =
        template_id == ascir::TemplateId::kIndirectLoadSK && HasReduceBeforeGatherAxis(indirect_load);
    const bool keep_softmax_pattern_for_sk_redirect =
        template_id == ascir::TemplateId::kIndirectLoadSK && !suppress_sk_redirect_for_plain_candidate &&
        (indirect_load_detail::IsSkWindowUbInfeasible(candidate_indirect_load) ||
         indirect_load_detail::IsSkRedirectRequiredForDynamicTail(candidate_graph, candidate_indirect_load));
    if (!keep_softmax_pattern_for_sk_redirect) {
      GE_ASSERT_SUCCESS(
          softmax_pattern::NormalizeDirectPostSoftmax(candidate_graph, candidate_indirect_load, softmax_normalized));
    }
    if (softmax_normalized) {
      // 替换后候选图中节点名不变（Softmax 以原 TrueDiv 名+"_softmax"重建），
      // 后续 ApplyGraphPass 按模板重新执行改图与校验。
      GELOGI("[IndirectLoad] Candidate[%d] normalized direct post Softmax for node[%s].",
             static_cast<int32_t>(template_id), candidate_indirect_load->GetNamePtr());
    }
    bool is_candidate_legal = false;
    bool sk_dynamic_redirect_rejected = false;
    GE_ASSERT_SUCCESS(ApplyGraphPass(candidate_graph, candidate_indirect_load, template_id, is_candidate_legal,
                                     !graphs.empty(), &sk_dynamic_redirect_rejected,
                                     suppress_sk_redirect_for_plain_candidate));
    if (!is_candidate_legal && sk_dynamic_redirect_rejected) {
      // [动态重定向回退] 分段候选被布局/metadata 校验拒绝时，基于原图重新生成常规
      // SK 全载候选：保持改动前的编译下限（小 R launch 期 tiling 可行，大 R 运行期
      // 行为与改动前一致），避免重定向使零候选升级为整图编译失败。回退图先规范化
      // softmax（重定向路径为保留原始 Pattern 链跳过了规范化，常规 SK 依赖规范化产物）。
      GELOGW("[IndirectLoad] Dynamic-tail segmented candidate rejected for node[%s], retry plain SK candidate.",
             candidate_indirect_load->GetNamePtr());
      ascir::ImplGraph fallback_graph(candidate_graph.GetName().c_str());
      GE_ASSERT_TRUE(fallback_graph.CopyFrom(candidate_graph), "Failed to copy graph [%s] for SK fallback.",
                     candidate_graph.GetName().c_str());
      const af::AscNodePtr fallback_il = fallback_graph.FindNode(indirect_load_name.c_str());
      GE_ASSERT_NOTNULL(fallback_il, "Failed to find copied IndirectLoad node[%s] in SK fallback graph.",
                        indirect_load_name.c_str());
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetImplementation(
          fallback_il, ascgen_utils::indirect_load::Implementation::kDefault));
      bool fallback_softmax_normalized = false;
      GE_ASSERT_SUCCESS(
          softmax_pattern::NormalizeDirectPostSoftmax(fallback_graph, fallback_il, fallback_softmax_normalized));
      bool fallback_legal = false;
      GE_ASSERT_SUCCESS(ApplyGraphPass(fallback_graph, fallback_il, template_id, fallback_legal, !graphs.empty(),
                                       nullptr, /*suppress_sk_redirect=*/true));
      if (fallback_legal) {
        candidate_graph = std::move(fallback_graph);
        candidate_indirect_load = fallback_il;
        is_candidate_legal = true;
      }
    }
    if (!is_candidate_legal) {
      GELOGW("[IndirectLoad] Skip illegal template candidate[%d] for node[%s].", static_cast<int32_t>(template_id),
             candidate_indirect_load->GetNamePtr());
      continue;
    }
    ascgen_utils::indirect_load::IndirectLoadAccessInfo access_info;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetIndirectLoadAccessInfo(candidate_indirect_load, access_info));
    if (template_id == ascir::TemplateId::kIndirectLoadSimd &&
        template_case.implementation == ascgen_utils::indirect_load::Implementation::kDefault) {
      simd_lowers_to_strided = LowersToStridedSimdFacade(candidate_indirect_load);
    }
    graphs.emplace_back(std::move(candidate_graph));
    score_functions.emplace_back(
        GenerateScoreFunc(template_case, access_info, candidate_indirect_load, graphs.back(), simd_lowers_to_strided));
    GELOGI("[IndirectLoad] Add schedule candidate[%d, %d] for node[%s].", static_cast<int32_t>(template_id),
           static_cast<int32_t>(template_case.implementation), candidate_indirect_load->GetNamePtr());
  }
  return af::SUCCESS;
}

Status IndirectLoadScheduleCaseGenerator::GeneratorTask(ascir::HintGraph &optimize_graph,
                                                        std::vector<ScheduleTask> &tasks,
                                                        const OptimizerOptions &options) {
  bool need_update_axis = false;
  GE_ASSERT_SUCCESS(ScheduleGroupGraphPartitioner::NeedRefreshAxisSize(optimize_graph, need_update_axis));
  std::vector<ascir::ImplGraph> optimize_graphs;
  std::vector<std::string> score_funcs;
  GE_CHK_STATUS_RET(Generate(optimize_graph, optimize_graphs, score_funcs), "GenerateScheduleCases failed");
  if (optimize_graphs.empty() && !ascgen_utils::indirect_load::GetAllIndirectLoadNodes(optimize_graph).empty()) {
    // 本生成器独占含 IndirectLoad 的图（其余生成器按约定跳过）。模板候选全部被拒时
    // 不存在通用调度兜底——未标注模板的 IndirectLoad 没有通用调度与发射路径，落入
    // Recomputation 兜底会在调度期触发内部断言。此处显式失败，让上层按不支持处理。
    GELOGE(af::FAILED,
           "[IndirectLoad] Graph[%s] contains IndirectLoad but no template candidate was generated; the graph is "
           "not supported by autofuse.",
           optimize_graph.GetName().c_str());
    return af::FAILED;
  }
  for (size_t i = 0UL; i < optimize_graphs.size(); ++i) {
    const auto &graph = optimize_graphs[i];
    // [护栏] 候选产出后校验 IL 输出 vectorized 轴有效（空，或均为视图内轴），违约在
    // 生成侧报清晰错误，避免调度期以无关断言隐蔽爆炸。
    GE_ASSERT_SUCCESS(ValidateIndirectLoadOutputVectorizedAxes(graph));
    ScheduleTask task{graph, {}, score_funcs[i]};
    task.has_load_store_conversion = HasLoadStoreConversion();
    // 多 gather 候选图包含组内全部 IndirectLoad（组 SIMD/SIMT 与混模板候选经归并树连通
    // 落在同一 grouped graph；SK 桥接候选经 workspace 边界分段、各段恒单 IL）；取首个
    // 节点判定模板类型即可。
    const auto indirect_loads = ascgen_utils::indirect_load::GetAllIndirectLoadNodes(graph);
    const af::AscNodePtr indirect_load = indirect_loads.empty() ? nullptr : indirect_loads.front();
    // [SK分段SIMT] 重定向形态（template 已是 SIMT 但图含 output workspace 边界）与
    // SK 走同一分区/恢复路径（分区后子图需重新执行模板归一）。
    const bool is_sk_template =
        indirect_load != nullptr &&
        (ascir::GetTemplateIdOrDefault(*indirect_load) == ascir::TemplateId::kIndirectLoadSK ||
         (ascir::GetTemplateIdOrDefault(*indirect_load) == ascir::TemplateId::kIndirectLoadSimt &&
          ascir::IsSkSegmentedSimt(*indirect_load)));
    GE_CHK_STATUS_RET(PartitionTaskGraph(graph, indirect_load, is_sk_template, task.grouped_graphs),
                      "Failed to partition graph");
    GE_ASSERT_SUCCESS(RefreshTaskAxisSizes(task.grouped_graphs, need_update_axis, is_sk_template));
    GE_CHK_STATUS_RET(ReduceTaskGraphCount(task.grouped_graphs, options), "Failed to reduce graph count");
    GE_ASSERT_SUCCESS(FinalizeGroupedGraphLoweringMetadata(task.grouped_graphs));
    tasks.emplace_back(std::move(task));
  }
  return af::SUCCESS;
}

}  // namespace optimize
