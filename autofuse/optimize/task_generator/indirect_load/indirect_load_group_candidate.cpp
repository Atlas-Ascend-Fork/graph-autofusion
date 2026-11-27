/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_generator/indirect_load/indirect_load_group_candidate.h"
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
#include "task_generator/indirect_load/indirect_load_simt_emission.h"

#include "v35/ascir/ascir_codegen_v2.h"
#include "ascgraph_info_complete.h"
#include "schedule_result.h"

namespace optimize {
namespace indirect_load_detail {

bool IsGroupMemberLegal(const af::AscNodePtr &primary, const af::AscNodePtr &member) {
  const auto report_illegal = [&primary, &member](const char *const reason) {
    GELOGE(af::FAILED, "[IndirectLoad] Gather group member[%s] with primary[%s] is illegal: %s.", member->GetNamePtr(),
           primary->GetNamePtr(), reason);
    return false;
  };
  const size_t primary_axis = GetIndirectLoadAxisIndex(primary);
  const size_t member_axis = GetIndirectLoadAxisIndex(member);
  if (primary_axis == kIndirectLoadInvalidAxisIndex || member_axis == kIndirectLoadInvalidAxisIndex ||
      primary_axis != member_axis) {
    return report_illegal("gather axis mismatch");
  }
  const auto primary_outputs = primary->outputs();
  const auto member_outputs = member->outputs();
  if (primary_outputs.size() != kIndirectLoadOutputCount || member_outputs.size() != kIndirectLoadOutputCount) {
    return report_illegal("output count mismatch");
  }
  if (primary_outputs[0] == nullptr || member_outputs[0] == nullptr ||
      primary_outputs[0]->attr.axis.size() != member_outputs[0]->attr.axis.size()) {
    // [组共享轴等价化] 放宽 axis id 逐元素相等为 rank 相等：同索引空间、仅 id 不同
    // 的成员（Broadcast 副本轴/split 兄弟轴）在 ApplyGraphPassGroup 的
    // NormalizeMemberAxes 中归一到组共享轴后天然对齐；size 不等价由该步拒绝。
    return report_illegal("output axis space mismatch");
  }
  const auto primary_inputs = primary->inputs();
  const auto member_inputs = member->inputs();
  if (primary_inputs.size() != kIndirectLoadInputCount || member_inputs.size() != kIndirectLoadInputCount ||
      primary_inputs[ascgen_utils::indirect_load::kIndexTensorIndex] == nullptr ||
      member_inputs[ascgen_utils::indirect_load::kIndexTensorIndex] == nullptr) {
    return report_illegal("input count mismatch");
  }
  const auto &primary_index = primary_inputs[ascgen_utils::indirect_load::kIndexTensorIndex]->attr;
  const auto &member_index = member_inputs[ascgen_utils::indirect_load::kIndexTensorIndex]->attr;
  if (primary_index.repeats.size() != member_index.repeats.size()) {
    return report_illegal("index rank mismatch");
  }
  for (size_t dim = 0UL; dim < primary_index.repeats.size(); ++dim) {
    if (af::SymbolicUtils::StaticCheckEq(primary_index.repeats[dim], member_index.repeats[dim]) != af::TriBool::kTrue) {
      return report_illegal("index shape mismatch");
    }
  }
  // 输出链汇聚：成员输出与 primary 的前向消费者闭包存在公共节点（归并 ele / 归并树公共节点）。
  const auto collect_closure = [](NodeSet &closure) {
    return [&closure](const af::AscNodePtr &node, bool &stop) -> af::Status {
      (void)stop;
      closure.emplace(node.get());
      return af::SUCCESS;
    };
  };
  const auto collect_roots = [](const af::AscNodePtr &node, NodePath &roots) {
    for (const auto &out_node : node->GetOutDataNodes()) {
      const auto consumer = std::dynamic_pointer_cast<af::AscNode>(out_node);
      if (consumer != nullptr) {
        roots.emplace_back(consumer);
      }
    }
  };
  NodePath primary_roots;
  collect_roots(primary, primary_roots);
  NodeSet primary_closure;
  NodeSet primary_visited{primary.get()};
  GE_ASSERT_SUCCESS(TraverseOutputConsumers(primary_roots, primary_visited, collect_closure(primary_closure)));
  NodePath member_roots;
  collect_roots(member, member_roots);
  NodeSet member_closure;
  NodeSet member_visited{member.get()};
  GE_ASSERT_SUCCESS(TraverseOutputConsumers(member_roots, member_visited, collect_closure(member_closure)));
  const bool has_common_consumer =
      std::any_of(member_closure.begin(), member_closure.end(),
                  [&primary_closure](const af::AscNode *const node) { return primary_closure.count(node) != 0UL; });
  if (!has_common_consumer) {
    return report_illegal("output chains do not converge");
  }
  return true;
}

af::Status CollectIndirectLoadEntry(const ascir::HintGraph &graph, af::AscNodePtr &indirect_load,
                                    IndirectLoadGroup &group, bool &is_group) {
  is_group = false;
  group = {};
  indirect_load = nullptr;
  const auto nodes = ascgen_utils::indirect_load::GetAllIndirectLoadNodes(graph);
  if (nodes.empty()) {
    return af::SUCCESS;
  }
  if (nodes.size() == 1UL) {
    indirect_load = nodes.front();
  } else {
    // 数量不限制：N（>= 2）路全部进入组配对，逐路与 primary 判定。
    group.primary = nodes.front();
    group.others.assign(nodes.begin() + 1, nodes.end());
    for (const af::AscNodePtr &member : group.others) {
      if (!IsGroupMemberLegal(group.primary, member)) {
        return af::FAILED;
      }
    }
    is_group = true;
  }
  const af::AscNodePtr &anchor = is_group ? group.primary : indirect_load;
  af::AscNodePtr unsupported_node;
  for (const af::AscNodePtr &node : graph.GetAllNodes()) {
    if (!IsSupportedIndirectLoadTopologyNode(node)) {
      unsupported_node = node;
      break;
    }
  }
  if (unsupported_node != nullptr) {
    GELOGE(af::FAILED,
           "[IndirectLoad] Graph[%s] node[%s] contains unsupported topology operator[%s, compute_type=%u] "
           "around IndirectLoad[%s].",
           graph.GetName().c_str(), unsupported_node->GetNamePtr(), unsupported_node->GetTypePtr(),
           static_cast<uint32_t>(unsupported_node->attr.api.compute_type), anchor->GetNamePtr());
    return af::FAILED;
  }
  return af::SUCCESS;
}

namespace {
// [L2 降级预判] 在候选图副本上走完整链（SetTemplateId → 归一化 →
// FinalizeLoweringMetadata → PlanSimtEmission），预测结构性 L2 候选是否降级。
// 契约：pre_judged 仅在 PlanSimtEmission 真正执行后置 true——任何早退分支
// （归一化失败/元数据失败）保持 false + GELOGW，调用方据此回退正常计算，
// 绝不将零值 plan 当真 plan 复用（SimtEmissionPlan{} 的 register_merge=false
// 恰好是合法 L4 档位，静默复用会把 L2 主路径打到 L4）。
af::Status PreJudgeL2Degradation(const af::AscGraph &graph, const std::vector<af::AscNodePtr> &members,
                                 const ascgen_utils::indirect_load::SimtFusedMergeInfo &merge_info,
                                 ascir::TemplateId template_id, SimtEmissionPlan &early_plan, bool &will_degrade,
                                 bool &pre_judged) {
  will_degrade = false;
  pre_judged = false;
  ascir::ImplGraph pre_graph(graph.GetName().c_str());
  GE_ASSERT_TRUE(pre_graph.CopyFrom(graph), "Failed to copy graph for L2 degradation pre-judgment.");
  std::vector<af::AscNodePtr> pre_members;
  for (const af::AscNodePtr &member : members) {
    const af::AscNodePtr pre_member = pre_graph.FindNode(member->GetName().c_str());
    GE_ASSERT_NOTNULL(pre_member, "Pre-judgment member[%s] not found in copy.", member->GetNamePtr());
    GE_ASSERT_SUCCESS(::ascir::SetTemplateId(pre_member, template_id));
    pre_members.emplace_back(pre_member);
  }
  bool pre_legal = true;
  GE_ASSERT_SUCCESS(NormalizeSimtAxesForGroupTemplate(pre_graph, pre_members, nullptr, pre_legal, 0UL));
  if (!pre_legal) {
    GELOGW("[IndirectLoad] Pre-judge bailed: normalization failed, falling back to normal plan.");
    return af::SUCCESS;
  }
  bool pre_all_u32 = true;
  for (const af::AscNodePtr &pre_member : pre_members) {
    bool metadata_ok = false;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::FinalizeLoweringMetadata(pre_member, metadata_ok));
    if (!metadata_ok) {
      GELOGW("[IndirectLoad] Pre-judge bailed: metadata failed for node[%s], falling back to normal plan.",
             pre_member->GetNamePtr());
      return af::SUCCESS;
    }
    ascgen_utils::indirect_load::IndirectLoadLoweringMetadata metadata;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetLoweringMetadata(pre_member, metadata));
    if (metadata.simt.policy.offset_width != ascgen_utils::indirect_load::SimtOffsetWidth::kUint32) {
      pre_all_u32 = false;
      break;
    }
  }
  GE_ASSERT_SUCCESS(PlanSimtEmission(merge_info, pre_all_u32, early_plan));
  will_degrade = !early_plan.register_merge;
  pre_judged = true;
  GELOGI("[IndirectLoad] L2 degradation pre-judgment: will_degrade=%s (N=%zu, demand=%llu).",
         will_degrade ? "true" : "false", static_cast<size_t>(merge_info.member_count),
         static_cast<unsigned long long>(early_plan.demand));
  return af::SUCCESS;
}
}  // namespace

// 多 gather 水平融合候选流水：逐 IL 复用单 gather 的分析/校验/标注/元数据机制，仅轴归一化
// 改为共享模板轴（NormalizeSimdAxesForGroupTemplate / NormalizeSimtAxesForGroupTemplate）。
// G≥R、复合 Norm 等输出侧校验对组内各路判定必然一致（axis 相同、输出边界相同），逐 IL 执行
// 以完整复用既有校验实现。SIMT 档位采用组员形态：各路 evaluator 仅含本路 index 链与 input
// 读取，value 写 UB 局部目标；共享归并/输出链作为普通向量代码在全部 invocation 之后执行。
af::Status ApplyGraphPassGroup(af::AscGraph &graph, const IndirectLoadGroup &group, ascir::TemplateId template_id,
                               bool &is_candidate_legal) {
  is_candidate_legal = true;
  std::vector<af::AscNodePtr> members{group.primary};
  members.insert(members.end(), group.others.begin(), group.others.end());
  GELOGD("[IndirectLoad] Apply gather group graph pass for primary[%s] with %zu members, template_id[%d].",
         group.primary->GetNamePtr(), members.size(), static_cast<int32_t>(template_id));
  const bool is_simt = template_id == ascir::TemplateId::kIndirectLoadSimt;
  // L6 SK 分段桥接：逐成员复用单 gather SK 流水（ApplySkGraphPass 自包含——仅触碰本路的
  // 边/区域/角色，无全局清理，顺序调用天然可组合）。workspace 边界把各路 IL 隔入独立段
  // （段内恒单 IL，下游 RestoreSkTemplateAxes 等单 IL 假设在段内成立），归并/输出链作为
  // 消费段独立调度（读各路 workspace）。
  if (template_id == ascir::TemplateId::kIndirectLoadSK) {
    // 与单 gather 路径的 dtype 门控（见 ApplyGraphPass）保持一致：SK 的 AscendC::Gather
    // 不支持 8 字节与 1 字节值，任一成员命中即整组 SK 候选拒绝（该图对本生成器不可融合，
    // 由前端 BackendSpec 预防兜底承接）。
    for (const af::AscNodePtr &member : members) {
      const auto value_dtype = member->inputs()[ascgen_utils::indirect_load::kInputTensorIndex]->attr.dtype;
      const bool is_64bit_value = value_dtype == af::DT_INT64 || value_dtype == af::DT_UINT64;
      const bool is_1byte_value =
          value_dtype == af::DT_INT8 || value_dtype == af::DT_UINT8 || value_dtype == af::DT_BOOL;
      if (is_64bit_value || is_1byte_value) {
        is_candidate_legal = false;
        GELOGI("[IndirectLoad] value dtype[%d] is unsupported by SK, skip gather group SK candidate for node[%s].",
               static_cast<int32_t>(value_dtype), member->GetNamePtr());
        return af::SUCCESS;
      }
    }
    for (const af::AscNodePtr &member : members) {
      bool softmax_normalized = false;
      GE_ASSERT_SUCCESS(softmax_pattern::NormalizeDirectPostSoftmax(graph, member, softmax_normalized));
      bool member_legal = false;
      GE_ASSERT_SUCCESS(ApplySkGraphPass(graph, member, member_legal));
      if (!member_legal) {
        is_candidate_legal = false;
        // 候选阶梯的正常出局：SK 非法后由 SIMD/SIMT 候选承接，健康路径不应产生
        // ERROR 告警——与单 gather ApplySkGraphPass 的拒绝日志（GELOGI）及上方
        // SK dtype 门控（GELOGI）约定一致。
        GELOGI("[IndirectLoad] Reject gather group SK candidate: member[%s] is SK-illegal.", member->GetNamePtr());
        return af::SUCCESS;
      }
    }
    return af::SUCCESS;
  }
  if (template_id != ascir::TemplateId::kIndirectLoadSimd && !is_simt) {
    is_candidate_legal = false;
    // 组候选拒绝原因需在默认日志级别可观测，与成员非法判定（report_illegal）的 GELOGE 约定一致。
    GELOGE(af::FAILED, "[IndirectLoad] Reject gather group template[%d]: only SIMD/SIMT/SK are supported.",
           static_cast<int32_t>(template_id));
    return af::SUCCESS;
  }
  // 与单 gather 路径的 dtype 门控（见 ApplyGraphPass）保持一致：1 字节值（int8/uint8/bool）
  // 仅 SIMT 标量路径支持——SIMD 候选直接拒绝（SIMT 候选继续，即模板阶梯的 1 字节档）；
  // 64 位值由 SIMD 8 字节 ValuePolicy（RegTraitNumTwo）与 SIMT 共同支持。
  if (!is_simt) {
    for (const af::AscNodePtr &member : members) {
      const auto value_dtype = member->inputs()[ascgen_utils::indirect_load::kInputTensorIndex]->attr.dtype;
      const bool is_1byte_value =
          value_dtype == af::DT_INT8 || value_dtype == af::DT_UINT8 || value_dtype == af::DT_BOOL;
      if (is_1byte_value) {
        is_candidate_legal = false;
        GELOGI("[IndirectLoad] 1-byte value dtype[%d] only supports SIMT, skip gather group template[%d] for node[%s].",
               static_cast<int32_t>(value_dtype), static_cast<int32_t>(template_id), member->GetNamePtr());
        return af::SUCCESS;
      }
    }
  }
  // 与单 gather SIMT 入口一致：先清理整图残留的 SIMT 角色（组员输出链保持无 SIMT 计划角色，
  // 由组员扇出标注统一标为 normal schedule 扇出分支）。
  if (is_simt) {
    GE_ASSERT_SUCCESS(ClearSimtTemplateRoles(graph));
  }
  std::vector<RewrittenGraphAnalysis> analyses(members.size());
  // [共享生产者视图隔离] AnalyzeRewrittenGraph 的 ApplyTemplatePathLayouts 会把成员 input/index
  // 的直接生产者物理视图就地写回（ApplyPhysicalView 改 sched.axis 与 output 的 axis/repeats/
  // strides）。组成员共享同一生产者（如同一个 index Broadcast）时，后分析的成员会读到被前
  // 一个成员改写过的视图，导致各成员 TemplateLogicalView/policy 不一致（如 index sizes
  // [56240,8] 与 [1,8] 分裂，第二路寻址错误）。逐成员分析前把直接生产者恢复到分析前快照，
  // 保证所有成员从同一视图出发；最终视图由最后一个成员的分析写回（组内布局一致）。
  struct DirectProducerViewSnapshot {
    std::string name;
    std::vector<ascir::AxisId> sched_axis;
    std::vector<ascir::AxisId> axis;
    std::vector<af::Expression> repeats;
    std::vector<af::Expression> strides;
  };
  std::vector<DirectProducerViewSnapshot> producer_snapshots;
  std::set<std::string> recorded_producers;
  for (const af::AscNodePtr &member : members) {
    for (size_t input_idx = 0UL; input_idx < static_cast<size_t>(kIndirectLoadInputCount); ++input_idx) {
      const af::AscNodePtr producer = ascgen_utils::indirect_load::GetInputProducer(member, input_idx);
      if (producer == nullptr || producer->outputs().empty() || producer->outputs()[0] == nullptr ||
          !recorded_producers.insert(producer->GetName()).second) {
        continue;
      }
      const auto &output_attr = producer->outputs()[0]->attr;
      producer_snapshots.push_back(
          {producer->GetName(), producer->attr.sched.axis, output_attr.axis, output_attr.repeats, output_attr.strides});
    }
  }
  const auto restore_direct_producers = [&producer_snapshots](const af::AscNodePtr &member) {
    for (size_t input_idx = 0UL; input_idx < static_cast<size_t>(kIndirectLoadInputCount); ++input_idx) {
      const af::AscNodePtr producer = ascgen_utils::indirect_load::GetInputProducer(member, input_idx);
      if (producer == nullptr || producer->outputs().empty() || producer->outputs()[0] == nullptr) {
        continue;
      }
      for (const auto &snapshot : producer_snapshots) {
        if (snapshot.name != producer->GetName()) {
          continue;
        }
        producer->attr.sched.axis = snapshot.sched_axis;
        auto &output_attr = producer->outputs()[0]->attr;
        output_attr.axis = snapshot.axis;
        output_attr.repeats = snapshot.repeats;
        output_attr.strides = snapshot.strides;
        break;
      }
    }
  };
  for (size_t index = 0UL; index < members.size(); ++index) {
    restore_direct_producers(members[index]);
    GE_ASSERT_SUCCESS(AnalyzeRewrittenGraph(graph, members[index], template_id, is_candidate_legal, analyses[index]));
    if (!is_candidate_legal) {
      return af::SUCCESS;
    }
  }
  // 组配对约束已保证各路输出前向闭包与 primary 存在公共节点；此处进一步要求输出边界（store
  // 与首个 post Reduce）全部与 primary 为同一节点，即全组归并到同一条输出链，避免边界分析
  // 各自收敛到不同链路。
  for (size_t index = 1UL; index < members.size(); ++index) {
    if (analyses.front().output_store != analyses[index].output_store ||
        analyses.front().post_reduce != analyses[index].post_reduce) {
      is_candidate_legal = false;
      GELOGE(af::FAILED,
             "[IndirectLoad] Reject gather group candidate: output chains of node[%s] and node[%s] diverge.",
             members.front()->GetNamePtr(), members[index]->GetNamePtr());
      return af::SUCCESS;
    }
  }
  // [组路径 post-Reduce 轴结构] 共享 Reduce 终端的组（发散检查已保证各路 post_reduce 为同一
  // 节点）：boundary 镜像单 gather post-Reduce 语义覆写为 first_reduce——outer=行/批（G）、
  // inner=归约段（R），BlockSplit 只作用于 G 的 tile-outer，行不被劈到多核（共享 Reduce 的
  // 扇出向量代码按核执行，行劈开只得部分和）。布局非法（非支持的归约后缀，如中维归约且
  // 其后仍有保留轴）时与单 gather 同口径拒绝本候选，由其余候选/回退承接。post_reduce 在
  // 下方组员计划收窄处置空，故先保存。
  af::AscNodePtr group_post_reduce = nullptr;
  bool l2_will_degrade = false;
  SimtEmissionPlan early_emission_plan = {};
  bool l2_pre_judged = false;
  if (is_simt) {
    group_post_reduce = analyses.front().post_reduce;
    if (group_post_reduce != nullptr) {
      // 形状保持的专用 Norm 节点（Softmax）：归约发生在算子内部、输出沿归约轴仍为满
      // stride，BuildPostReduceLayout 识别不出归约轴（first_reduce == rank），须与
      // 单 gather 的 ValidateTemplate Softmax 分流同口径——特判放行，不经过
      // ValidateSimtPostReduceLayout 的 first_reduce 校验。boundary 覆写取 softmax
      // 输入 rank-1，与单 gather 的 boundary 覆写语义一致（边界在
      // NormalizeSimtAxesForGroupTemplate 内由 ValidateSimtPostReduceLayout 走不到时
      // 保持 rank，此处不额外覆写——Softmax 的 solve_tile_size 在
      // NormalizeSimtAxesForGroupTemplate 中由 IsShapePreservingTailAxisNorm 开启）。
      if (!IsShapePreservingTailAxisNorm(group_post_reduce)) {
        size_t group_reduce_boundary = 0UL;
        bool group_reduce_legal = true;
        GE_ASSERT_SUCCESS(ValidateSimtPostReduceLayout(group_post_reduce, group_reduce_boundary, group_reduce_legal));
        if (!group_reduce_legal) {
          is_candidate_legal = false;
          GELOGI("[IndirectLoad] Reject gather group SIMT candidate: shared post Reduce layout is unsupported.");
          return af::SUCCESS;
        }
      }
    }
  }
  if (is_simt) {
    // 组员 SIMT 计划收窄：SIMT 区域重收集为仅本路 index 链；共享归并/post-Reduce 链不进入
    // 任何组员的 SIMT 计划（普通向量代码），post_reduce 置空使 ValidateTemplate 的 boundary
    // 保持输出 rank（固定单行 tile 语义，与单 gather 无 post-Reduce SIMT 一致）。
    // [跨成员 index 链共享节点豁免] 先建全组并集（BuildGroupIndexExemptSet），再做逐成员
    // 的外部消费者检查——预收集的 backward region 复用传入 CollectSimtGroupIndexRegion，
    // 消除冗余遍历。
    std::vector<NodeSet> back_regions;
    const NodeSet group_exempt = BuildGroupIndexExemptSet(members, &back_regions);
    for (size_t index = 0UL; index < members.size(); ++index) {
      if (!CollectSimtGroupIndexRegion(graph, members[index], analyses[index].index_region, group_exempt,
                                       &back_regions[index])) {
        is_candidate_legal = false;
        GELOGE(af::FAILED,
               "[IndirectLoad] Reject gather group SIMT candidate: index region of node[%s] is empty or unsupported.",
               members[index]->GetNamePtr());
        return af::SUCCESS;
      }
      analyses[index].post_reduce = nullptr;
    }
  }
  for (const RewrittenGraphAnalysis &analysis : analyses) {
    GE_ASSERT_SUCCESS(CompleteInputDataTensorAttrs(analysis));
  }
  for (size_t index = 0UL; index < members.size(); ++index) {
    // 与单 gather 路径一致：ValidateTemplate 的 boundary 为出/入参（SIMT 模板会改写），
    // 组 SIMT 的 post_reduce 已置空，boundary 保持输出 rank。
    size_t boundary = members[index]->outputs()[0]->attr.axis.size();
    GE_ASSERT_SUCCESS(ValidateTemplate(members[index], template_id, analyses[index], boundary, is_candidate_legal));
    if (!is_candidate_legal) {
      GELOGE(af::FAILED,
             "[IndirectLoad] Reject gather group template[%d] for node[%s]: post-reduce or region validation failed.",
             static_cast<int32_t>(template_id), members[index]->GetNamePtr());
      return af::SUCCESS;
    }
  }
  for (size_t index = 0UL; index < members.size(); ++index) {
    GE_ASSERT_SUCCESS(AnnotateTemplate(members[index], template_id, analyses[index]));
  }
  // 六级降级阶梯的 L2/L3/L4 判定结果（L4=V2 时不设融合发射标记，成员各自走标准单 gather
  // invocation——codegen 的成员发射路径完整复用单 gather 机制）。
  ascgen_utils::indirect_load::SimtFusedMergeInfo merge_info;
  bool use_fused_emission = false;
  if (is_simt) {
    // 组员标记先于全部角色标注写入（顺序修复）：扇出标注的排除集与 BuildSimtLoweringMetadata
    // 的组员分支均依赖该标记。
    for (const af::AscNodePtr &member : members) {
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetSimtGroupMember(member));
    }
    GE_ASSERT_SUCCESS(PlanSimtFusedEmission(graph, members, merge_info));
    // [链发射者（结构预期）+ 按结构注解] 结构性 L2 候选按 L2 注解（evaluator 折叠
    // 整链 + side load/Store 直读直写边界），压力判定落定后若降级（L3/L4）复位为
    // 无角色。非结构性候选（L3/L4）的归并链/侧链/side load/Store **不注解任何
    // IndirectLoad 角色**：无 skip → 进 tiling group、有 tiling/block split、UB 生命
    // 周期与主循环发射全套恢复——发射者由分区器决定（VF 能收就收，收不了 normal
    // 兜底），彻底消除"节点数预测 VF 成图"的错配窗口（铁律：不依赖预测；改
    // kSimtFanoutBranch 行为会波及单 gather 扇出路径，故组链一律无角色）。链的
    // [k,R] 视图由 TileSplit 的标准推导路径（非 preserves 分支的
    // DeriveNodeVectorizedAxes——链节点入组后 ApplyTiling 已切出 TileInner，求交
    // 自然保 k）获得，无需成员专用的显式写入扩展。
    GELOGI("[IndirectLoad] Gather group chain plan: nodes[%u] width[%u] (emitter decided by partitioner).",
           static_cast<unsigned int>(merge_info.forward_node_count),
           static_cast<unsigned int>(merge_info.forward_chain_width));
    if (merge_info.l2_structural) {
      // L2 寄存器归并（结构判定）：归并树整链标 kSimtInlineTransform——在 fused evaluator
      // 的 FusedEval 内按拓扑序发射（N 种子 + 表达式折叠），向量 codegen 跳过其发射。
      for (const std::string &merge_node_name : merge_info.merge_node_names) {
        const af::AscNodePtr merge_node = graph.FindNode(merge_node_name.c_str());
        GE_ASSERT_NOTNULL(merge_node, "Fused merge node[%s] is missing.", merge_node_name.c_str());
        GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
            merge_node, ascgen_utils::indirect_load::TemplateRole::kSimtInlineTransform));
        GELOGD("[IndirectLoad] Register merge: node[%s] marked kSimtInlineTransform (in-evaluator merge).",
               merge_node->GetNamePtr());
      }
      // side load：kSimtDirectGmBoundary——evaluator 直读 GM，向量 codegen 跳过其独立
      // 发射（与单 gather index 区域 load 的角色语义一致；其向量化视图由
      // CompletePreservedVectorizedViews 的全图遍历统一播种，行广播标记同理）。
      for (const std::string &side_load_name : merge_info.side_load_names) {
        const af::AscNodePtr side_load = graph.FindNode(side_load_name.c_str());
        GE_ASSERT_NOTNULL(side_load, "Fused merge side load[%s] is missing.", side_load_name.c_str());
        GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
            side_load, ascgen_utils::indirect_load::TemplateRole::kSimtDirectGmBoundary));
      }
      // Store 终端：evaluator 的归并结果经 OutputTargets 直写 Store 的 GM 目标（codegen
      // 按 merge_store_names 在当前图重解析输出链 tensor id），Store 自身跳过独立发射，
      // 否则普通发射会与直写冲突——与单 gather output_store 的 kSimtDirectGmBoundary
      // 语义一致。仅 L2 进入本分支；L3/L4 归并链由 VF/normal 产出后 Store 正常发射读取。
      for (const std::string &store_name : merge_info.merge_store_names) {
        const af::AscNodePtr store = graph.FindNode(store_name.c_str());
        GE_ASSERT_NOTNULL(store, "Fused merge store terminal[%s] is missing.", store_name.c_str());
        GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateRole(
            store, ascgen_utils::indirect_load::TemplateRole::kSimtDirectGmBoundary));
      }
    }
    // 非 L2（L3/L4）：链/侧链/side load/Store 无角色（见上方注释）——普通调度节点，
    // SIMT 写成员 value UB 与其读取之间由标准 loop 同步衔接。
    // [组共享轴等价化 + 传递归一化] 在模板轴归一化之前：各成员输出视图中"同索引空间、
    // 仅 id 不同"的轴归一到新建组共享轴（非等价时拒绝当前候选，不影响其它候选与原图
    // 回退）；替换映射沿成员前向闭包传递到链/Store/Reduce（真新维度拒绝、广播轴豁免、
    // 原子 + 幂等）——链节点进 tiling group 前必须完成，否则 GetLoopAxis 把旧 id 并进
    // y_group（GenTilingCase 枚举变大、MergeAxesGroup 可能失败、[k,R] 窗口两套）。
    {
      bool axes_normalized = true;
      std::unordered_map<ascir::AxisId, ascir::AxisId> old_to_group;
      GE_ASSERT_SUCCESS(
          ascgen_utils::indirect_load::NormalizeMemberAxes(graph, members, axes_normalized, &old_to_group));
      if (!axes_normalized) {
        is_candidate_legal = false;
        return af::SUCCESS;
      }
      bool axes_propagated = true;
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::PropagateGroupAxes(graph, members, old_to_group, axes_propagated));
      if (!axes_propagated) {
        is_candidate_legal = false;
        return af::SUCCESS;
      }
    }
    // [结构型 UB 归并的轴布局] 结构判定已确定该组不可能走寄存器归并（链不可标量折叠 /
    // 无 Store / 空链）且无共享 Reduce 时，输出链只能由普通向量代码在 tile 循环内消费：
    // 取 boundary=rank-1（outer=行/批、inner=最内轴），与边界化单 gather 同形态，避免整空间
    // 布局导致调用被提升、输出链脱离循环。其它形态（L2 结构候选、有 Reduce 的组）不变。
    const size_t group_output_rank = members.front()->outputs()[0]->attr.axis.size();
    if (merge_info.l2_structural && group_post_reduce == nullptr) {
      GE_ASSERT_SUCCESS(PreJudgeL2Degradation(graph, members, merge_info, template_id, early_emission_plan,
                                              l2_will_degrade, l2_pre_judged));
    }
    const bool use_l3_layout =
        (group_post_reduce == nullptr) && (group_output_rank > 1UL) && (!merge_info.l2_structural || l2_will_degrade);
    GE_ASSERT_SUCCESS(NormalizeSimtAxesForGroupTemplate(graph, members, group_post_reduce, is_candidate_legal,
                                                        use_l3_layout ? (group_output_rank - 1UL) : 0UL));
    if (!is_candidate_legal) {
      return af::SUCCESS;
    }
    // [组路径 post-Reduce 视图播种] 与单 gather（AnnotateTemplate 尾部）及 SIMD 组分支同构：
    // 共享 Reduce 的输入视图按模板 vectorized 轴播种——SIMT 组路径此前缺失该播种，扇出链
    // 下游视图与模板轴系失配（Reduce 对齐推断读到不在本张量轴列表里的轴 id）。成员 value
    // tensor 的视图播种（actual_size 的直接来源）同位补齐——单 gather 靠输出重定向的 id
    // 绑定隐式完成，组路径必须显式播种。无共享 Reduce 的组保持原行为。
    if (group_post_reduce != nullptr || use_l3_layout) {
      if (group_post_reduce != nullptr) {
        GE_ASSERT_SUCCESS(SeedPostReduceInputVectorizedView(members.front(), group_post_reduce));
      }
      GE_ASSERT_SUCCESS(SeedGroupMemberValueVectorizedViews(members, is_candidate_legal));
      if (!is_candidate_legal) {
        return af::SUCCESS;
      }
    }
  } else {
    // [组共享轴等价化] 与 SIMT 路径同构：模板轴归一化之前先做成员轴等价化。
    {
      bool axes_normalized = true;
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::NormalizeMemberAxes(graph, members, axes_normalized));
      if (!axes_normalized) {
        is_candidate_legal = false;
        return af::SUCCESS;
      }
    }
    GE_ASSERT_SUCCESS(NormalizeSimdAxesForGroupTemplate(graph, members, analyses));
    // 与单 gather NormalizeTemplateAxes 的 SIMD 分支保持一致：post Reduce（含复合区域）输入
    // 的向量化视图按模板轴播种；组内 vectorized_axes 相同，以 primary 播种一次即对全组生效。
    const bool is_softmax_post =
        analyses.front().post_reduce != nullptr && af::ops::IsOps<af::ascir_op::Softmax>(analyses.front().post_reduce);
    if (!is_softmax_post) {
      if (!analyses.front().composite_reduces.empty()) {
        GE_ASSERT_SUCCESS(SeedCompositeReduceInputViews(members.front(), analyses.front().composite_reduces));
      } else {
        GE_ASSERT_SUCCESS(SeedPostReduceInputVectorizedView(members.front(), analyses.front().post_reduce));
      }
    }
    // [per-IL index 内轴 hint] 已下沉至 NormalizeSimdAxesForTemplate（primary）与
    // NormalizeSimdAxesForMember（非 primary 成员），组路径与混模板路径共用同一设置点。
  }
  if (!is_simt) {
    for (const af::AscNodePtr &member : members) {
      GE_ASSERT_SUCCESS(CompletePreservedVectorizedViews(graph));
      GE_ASSERT_SUCCESS(FinalizeTemplate(member, template_id));
    }
  } else {
    // [播种后置于角色终态] SIMT 路径把 CompletePreservedVectorizedViews 移到发射计划
    // 落定与降级重注解之后：结构性 L2 候选降级时，side load/Store 先复位为 kNone、链
    // 重注解为扇出/无角色，再播种——重注解后的 preserves_vectorized_axis 语义与原生
    // L3 一致（kNone 不播种、扇出播种），两条路径产物同源；播种先行会残留 L2 期结果，
    // 与原生 L3 分叉（视图/后续 DeriveNodeVectorizedAxes 求交可能不同）。FinalizeTemplate
    // 仅写模板 id/dcache，不读写向量化视图，拆出环路不改变内容。
    for (const af::AscNodePtr &member : members) {
      GE_ASSERT_SUCCESS(FinalizeTemplate(member, template_id));
    }
  }
  // [位宽感知门控后置] lowering 元数据先于融合发射标记构建（FinalizeLoweringMetadata
  // 读模板轴与归一化产物，不可前移到角色注解之前）；门控从成员元数据读 offset_width
  // （与 codegen 选 fused OffsetType 的判定同源）。SIMD/SIMT 同环路构建（既有行为）。
  for (const af::AscNodePtr &indirect_load : members) {
    bool metadata_supported = false;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::FinalizeLoweringMetadata(indirect_load, metadata_supported));
    if (!metadata_supported) {
      is_candidate_legal = false;
      GELOGW("[IndirectLoad] Reject gather group candidate: lowering metadata of node[%s] is unsupported.",
             indirect_load->GetNamePtr());
      return af::SUCCESS;
    }
  }
  if (is_simt) {
    bool all_offset_uint32 = true;
    for (const af::AscNodePtr &member : members) {
      ascgen_utils::indirect_load::IndirectLoadLoweringMetadata metadata;
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetLoweringMetadata(member, metadata));
      if (metadata.simt.policy.offset_width != ascgen_utils::indirect_load::SimtOffsetWidth::kUint32) {
        all_offset_uint32 = false;
        break;
      }
    }
    // [发射计划落定] 形态/压力/线程档一次性判定（单一真值源，见 PlanSimtEmission）。
    SimtEmissionPlan emission_plan;
    if (l2_pre_judged) {
      emission_plan = early_emission_plan;
    } else {
      GE_ASSERT_SUCCESS(PlanSimtEmission(merge_info, all_offset_uint32, emission_plan));
    }
    use_fused_emission = emission_plan.use_fused;
    merge_info.register_merge = emission_plan.register_merge;
    merge_info.max_thread_tier = emission_plan.max_thread_tier;
    merge_info.demand = emission_plan.demand;
    merge_info.peak_model = ascgen_utils::indirect_load::kUsePeakModel;
    // [降级复位] 结构性 L2 候选压力超限降级（L3/L4）时，链/side load/Store 的角色必须
    // 与最终形态一致：早期按 L2 注解的 InlineTransform/DirectGmBoundary 复位（kNone =
    // 恢复未注解状态，L3/L4 链无角色——见 is_simt 分支注释），并清空 merge_info 的
    // L2 专用残留（归并树/side load/Store 终端名列表，codegen 仅在 register_merge=true
    // 时消费）——杜绝"L2 注解 + L3/L4 发射"的形态-角色错配（错配形态下 tensor 无
    // 写者无注册）。复位幂等：kNone 复位与列表清空重复执行无副作用。
    if (merge_info.l2_structural && !emission_plan.register_merge) {
      const auto reset_role = [&graph](const std::vector<std::string> &names) -> af::Status {
        for (const std::string &name : names) {
          const af::AscNodePtr node = graph.FindNode(name.c_str());
          GE_ASSERT_NOTNULL(node, "Fused merge role reset target[%s] is missing.", name.c_str());
          GE_ASSERT_SUCCESS(
              ascgen_utils::indirect_load::SetTemplateRole(node, ascgen_utils::indirect_load::TemplateRole::kNone));
        }
        return af::SUCCESS;
      };
      GE_ASSERT_SUCCESS(reset_role(merge_info.merge_node_names));
      GE_ASSERT_SUCCESS(reset_role(merge_info.side_load_names));
      GE_ASSERT_SUCCESS(reset_role(merge_info.merge_store_names));
      merge_info.merge_node_names.clear();
      merge_info.side_load_names.clear();
      merge_info.merge_store_names.clear();
      GELOGI(
          "[IndirectLoad] Fused register merge degrades to UB merge on register demand[%llu] (model[%s], "
          "quota[%llu], N=%zu, S=%zu, E=%zu, W=%u).",
          static_cast<unsigned long long>(merge_info.demand), merge_info.peak_model ? "peak" : "sum",
          static_cast<unsigned long long>(ascgen_utils::indirect_load::kSimtFusedMinTierQuota),
          static_cast<size_t>(merge_info.member_count), static_cast<size_t>(merge_info.side_load_count),
          static_cast<size_t>(merge_info.chain_node_count), static_cast<unsigned int>(merge_info.forward_chain_width));
    }
    // [非 L2 落地 UB 硬检查] 形态 2（共享 Reduce 终端）的 N 块 value tile 下界已由生成
    // 期 IsGroupSimtL4UbDoomed 预检（该检查仅在生成期知道形态 2 才生效）；L2 压力降级
    // 到 L3、以及 L4（成员仍为组员形态——输出链为空、local_target=true、value 写 N 块
    // UB）同样需要 N 块成员 value tile 驻留——此处按最终形态统一补检（唯一豁免是
    // register_merge=true 的 L2：value 不落 UB）。静态尾轴可算时判定，超限即拒绝候选
    // 由 SK/回退承接，避免产出必然 tiling 失败的候选。
    if (!emission_plan.register_merge && IsGroupSimtNValueTileUbDoomed(members)) {
      is_candidate_legal = false;
      GELOGW("[IndirectLoad] Reject gather group candidate: N value tiles exceed platform UB (N=%zu).", members.size());
      return af::SUCCESS;
    }
    if (!use_fused_emission) {
      // L4（V2 压力兜底）：形态与角色已按计划注解/重注解，仅跳过融合发射标记——各成员
      // 走标准单 gather invocation。播种保持既有 L4 时序（原 FinalizeTemplate 环路内）。
      GE_ASSERT_SUCCESS(CompletePreservedVectorizedViews(graph));
      GELOGI(
          "[IndirectLoad] Fused register demand[%llu] (model[%s]) exceeds min tier quota[%llu], fall back to V2 "
          "(N=%zu, S=%zu, E=%zu, W=%u).",
          static_cast<unsigned long long>(merge_info.demand), merge_info.peak_model ? "peak" : "sum",
          static_cast<unsigned long long>(ascgen_utils::indirect_load::kSimtFusedMinTierQuota),
          static_cast<size_t>(merge_info.member_count), static_cast<size_t>(merge_info.side_load_count),
          static_cast<size_t>(merge_info.chain_node_count), static_cast<unsigned int>(merge_info.forward_chain_width));
      return af::SUCCESS;
    }
    // 角色终态后统一播种（L2 的 InlineTransform/side load/Store 与 L3 的扇出链均
    // preserves_vectorized_axis=true → 播种；kNormal 链无角色 → 不播种，由调度器
    // 标准机制接管——与两条原生路径产物一致）。
    GE_ASSERT_SUCCESS(CompletePreservedVectorizedViews(graph));
  }
  if (is_simt && use_fused_emission) {
    // seq-V1 融合发射标记（L2/L3）：primary 携带全组成员名与归并信息，codegen 生成单个
    // fused invocation（IndirectLoadSimtFused，寄存器归并或 UB 归并）；非 primary 成员跳过
    // 发射（primary 统一处理）。L4（V2）不设该标记——各成员走标准单 gather invocation。
    std::vector<std::string> member_names;
    member_names.reserve(members.size());
    for (const af::AscNodePtr &member : members) {
      member_names.emplace_back(member->GetName());
    }
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetSimtFusedPrimary(members.front(), member_names));
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetSimtFusedMergeInfo(members.front(), merge_info));
    for (size_t index = 1UL; index < members.size(); ++index) {
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetSimtFusedMemberEmitted(members[index]));
    }
  }
  return af::SUCCESS;
}

// L5 混模板独立兜底（六级降级阶梯第 5 级）：全部共模板候选（SIMD×2 + SIMT）被否决时的
// 混模板形态承接——逐成员探测可行模板（SIMD 优先含 1 字节 dtype 门控，SIMT 兜底），
// 异构选择时各成员以各自模板标注于同一 kernel：SIMD 成员子集共享 SIMD 模板轴（组机制，
// per-IL input 内轴提示），SIMT 成员走组员形态（组员标记 + value 写 UB 局部目标）并复用
// SIMD 子集的 TemplateAxes（SIMD 轴含 outer/inner/tile/vectorized 全集，SIMT 仅消费 outer）；
// 全组 post_reduce 置空——归并/输出链（含 Reduce）统一作为 normal schedule 扇出分支的
// 向量代码执行（混模板下归并链不做 SIMD 的 G≥R 融合，保证语义一致）。发射形态：SIMD 成员
// 正常向量发射，SIMT 成员各走标准单 gather invocation（不设融合发射标记）。同构选择
// （全 SIMD 或全 SIMT）时跳过（与共模板候选重复）。
af::Status ApplyGraphPassMixed(af::AscGraph &graph, const std::vector<af::AscNodePtr> &members,
                               bool &is_candidate_legal, size_t &simd_selected_count) {
  is_candidate_legal = false;
  simd_selected_count = 0UL;
  std::vector<RewrittenGraphAnalysis> analyses(members.size());
  std::vector<ascir::TemplateId> selection(members.size(), ascir::TemplateId::kIndirectLoadSimt);
  // [混模板不做跨成员豁免] 共享 index 链场景下，某节点同时落在 SIMT 成员的 index
  // region 与 SIMD 成员的 index region，角色赋值冲突（kSimtOp 的 skips_api_emit
  // 覆盖 kSimdIndexPre 的 staging 发射），执行模型不支持双重消费。此处不建 group_exempt：
  // 探测自然失败（共享 Load 的跨成员消费者被判外部）→ 该成员结构强制回 SIMD → 全组同构
  // → 混模板候选不产出。豁免仅保留在"全组 SIMT"的组路径。逐成员探测：性能序择优（两模板
  // 均合法时按单 gather 同款判据 + 同款能力门控选更优的模板——embedding 形态且切片 ≥ 2KB、
  // 查表 ≥ 32 次选 SIMD，否则选 SIMT）；单模板合法时结构强制；dense（无 embedding 能力）
  // 无偏好依据，默认选 SIMT（择优处注释）。探测在混模板候选自身的图副本上执行，
  // AnalyzeRewrittenGraph 的改图（前置 ele 后移）与共模板路径一致，两模板探测间的重复调用
  // 幂等（改写模式不匹配即无操作）。
  for (size_t index = 0UL; index < members.size(); ++index) {
    const af::AscNodePtr &member = members[index];
    const auto value_dtype = member->inputs()[ascgen_utils::indirect_load::kInputTensorIndex]->attr.dtype;
    const bool is_1byte_value = value_dtype == af::DT_INT8 || value_dtype == af::DT_UINT8 || value_dtype == af::DT_BOOL;
    bool simd_legal = false;
    if (!is_1byte_value) {
      GE_ASSERT_SUCCESS(
          AnalyzeRewrittenGraph(graph, member, ascir::TemplateId::kIndirectLoadSimd, simd_legal, analyses[index]));
      if (simd_legal) {
        // 与单 gather canonical 顺序一致（Analyze → CompleteInputDataTensorAttrs → Validate）：
        // Data 节点轴属性补齐先于校验，保证选档结论与正式校验同输入状态。
        GE_ASSERT_SUCCESS(CompleteInputDataTensorAttrs(analyses[index]));
        size_t boundary = member->outputs()[0]->attr.axis.size();
        GE_ASSERT_SUCCESS(
            ValidateTemplate(member, ascir::TemplateId::kIndirectLoadSimd, analyses[index], boundary, simd_legal));
      }
    }
    bool simt_legal = false;
    if (simd_legal) {
      // SIMD 合法时仍需探测 SIMT：两模板均合法时按性能择优（大表路选 SIMD、小表路选
      // SIMT → 异构 → MIXED 产出与共模板候选同场竞标——消除"两路都合法时 MIXED
      // 必跳过、性能分裂无入口"的缺口）。SIMT 探测使用独立分析（不覆盖 SIMD 分析，
      // 后续按 selection 取对应的 analyses）。
      RewrittenGraphAnalysis simt_analysis;
      GE_ASSERT_SUCCESS(
          AnalyzeRewrittenGraph(graph, member, ascir::TemplateId::kIndirectLoadSimt, simt_legal, simt_analysis));
      if (simt_legal) {
        if (!CollectSimtGroupIndexRegion(graph, member, simt_analysis.index_region, NodeSet{})) {
          simt_legal = false;  // 探测失败（含共享 index 链被判外部）：结构强制回 SIMD
        } else {
          simt_analysis.post_reduce = nullptr;
          GE_ASSERT_SUCCESS(CompleteInputDataTensorAttrs(simt_analysis));
          size_t boundary = member->outputs()[0]->attr.axis.size();
          GE_ASSERT_SUCCESS(
              ValidateTemplate(member, ascir::TemplateId::kIndirectLoadSimt, simt_analysis, boundary, simt_legal));
        }
      }
      if (simd_legal && simt_legal) {
        // 性能择优（仅 embedding 形态，与单 gather 的 IsEmbeddingFastPathCapable 同款
        // 能力门控）：切片 ≥ 2KB 且查表 ≥ 32 → SIMD，动态 shape 保留候选序默认（SIMD
        // 先）。dense / SIMD fast-path 视图不具备的成员无偏好依据（判据来自 embedding
        // 行拷贝模型，对 dense"SIMD 窗口 = 整表驻留 UB"的语义不成立），默认选 SIMT——
        // MIXED 构建的生存导向（无 UB 窗口约束一侧），使 dense 组仅经结构强制产出
        // MIXED；共模板路径的平局语义不受影响（bid 层候选序仍 SIMD 先，与单 gather
        // "无偏好 → 平局按候选序"一致）。
        ascgen_utils::indirect_load::IndirectLoadAccessInfo access_info;
        GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetIndirectLoadAccessInfo(member, access_info));
        int64_t slice_bytes = 0L;
        int64_t lookup_count = 0L;
        const bool static_cost = access_info.input_slice_bytes.GetConstValue(slice_bytes) &&
                                 access_info.index_varying_extent.GetConstValue(lookup_count);
        const bool prefers_simd = access_info.can_use_simd_embedding &&
                                  (!static_cost || (slice_bytes >= kEmbeddingSimdPayloadBytesThreshold &&
                                                    lookup_count >= kEmbeddingSimdLookupCountThreshold));
        if (prefers_simd) {
          selection[index] = ascir::TemplateId::kIndirectLoadSimd;
          continue;  // 保留 SIMD 分析（analyses[index] 不变）
        }
        // 选 SIMT：用 SIMT 分析替换（后续 ValidateTemplate/AnnotateTemplate 消费）。
        analyses[index] = simt_analysis;
        selection[index] = ascir::TemplateId::kIndirectLoadSimt;
        continue;
      }
      // 仅 SIMD 合法（SIMT 探测失败或校验不通过）：结构强制选 SIMD。
      selection[index] = ascir::TemplateId::kIndirectLoadSimd;
      continue;
    }
    // SIMD 不合法（int8 / 布局 / G≥R）：SIMT 兜底。
    GE_ASSERT_SUCCESS(
        AnalyzeRewrittenGraph(graph, member, ascir::TemplateId::kIndirectLoadSimt, simt_legal, analyses[index]));
    if (!simt_legal) {
      GELOGW("[IndirectLoad] Mixed mode: node[%s] is illegal on both templates.", member->GetNamePtr());
      return af::SUCCESS;
    }
    if (!CollectSimtGroupIndexRegion(graph, member, analyses[index].index_region, NodeSet{})) {
      GELOGW("[IndirectLoad] Mixed mode: index region of node[%s] is empty or unsupported.", member->GetNamePtr());
      return af::SUCCESS;
    }
    analyses[index].post_reduce = nullptr;
    GE_ASSERT_SUCCESS(CompleteInputDataTensorAttrs(analyses[index]));
    size_t boundary = member->outputs()[0]->attr.axis.size();
    GE_ASSERT_SUCCESS(
        ValidateTemplate(member, ascir::TemplateId::kIndirectLoadSimt, analyses[index], boundary, simt_legal));
    if (!simt_legal) {
      GELOGW("[IndirectLoad] Mixed mode: node[%s] is illegal on both templates.", member->GetNamePtr());
      return af::SUCCESS;
    }
    selection[index] = ascir::TemplateId::kIndirectLoadSimt;
  }
  // 成员分类（探测完成后）：异构性判定 + 后续子集处理。
  std::vector<af::AscNodePtr> simd_members;
  std::vector<af::AscNodePtr> simt_members;
  for (size_t index = 0UL; index < members.size(); ++index) {
    if (selection[index] == ascir::TemplateId::kIndirectLoadSimd) {
      simd_members.emplace_back(members[index]);
    } else {
      simt_members.emplace_back(members[index]);
    }
  }
  simd_selected_count = simd_members.size();
  if (simd_members.empty() || simt_members.empty()) {
    GELOGI("[IndirectLoad] Mixed mode skipped: selection is homogeneous (simd=%zu, simt=%zu).", simd_members.size(),
           simt_members.size());
    return af::SUCCESS;
  }
  // [已知限制] 共享 index 链场景（某节点同时落在 SIMT 与 SIMD 成员的 index region）
  // 存在角色赋值冲突风险（kSimtOp 的 skips_api_emit 覆盖 kSimdIndexPre 的 staging
  // 发射），当前执行模型不支持双重消费。group_exempt 豁免只防探测期误拒，不解决
  // 执行期的角色冲突——共享 index 链的混模板场景需后续专项设计（如按消费侧拆分
  // 节点角色，或共享 Load 双发射）。
  // 输出链一致性（与共模板组一致）：全组输出边界为同一节点。
  for (size_t index = 1UL; index < members.size(); ++index) {
    if (analyses.front().output_store != analyses[index].output_store) {
      GELOGE(af::FAILED, "[IndirectLoad] Reject mixed mode candidate: output chains of node[%s] and node[%s] diverge.",
             members.front()->GetNamePtr(), members[index]->GetNamePtr());
      return af::SUCCESS;
    }
  }
  // 全组 post_reduce 置空：归并链（含 Reduce）统一作为扇出向量代码（混模板下不做 SIMD 的
  // G≥R 融合，避免同一 Reduce 同时处于 SIMD 融合与 SIMT 扇出两种语义）。
  for (size_t index = 0UL; index < members.size(); ++index) {
    analyses[index].post_reduce = nullptr;
  }
  // SIMD 子集分析在 post_reduce 置空后构建，与 ValidateTemplate/AnnotateTemplate/
  // NormalizeSimdAxesForGroupTemplate 消费的分析保持一致。
  std::vector<RewrittenGraphAnalysis> simd_analyses;
  for (size_t index = 0UL; index < members.size(); ++index) {
    if (selection[index] == ascir::TemplateId::kIndirectLoadSimd) {
      simd_analyses.emplace_back(analyses[index]);
    }
  }
  for (size_t index = 0UL; index < members.size(); ++index) {
    size_t boundary = members[index]->outputs()[0]->attr.axis.size();
    bool member_legal = false;
    GE_ASSERT_SUCCESS(ValidateTemplate(members[index], selection[index], analyses[index], boundary, member_legal));
    if (!member_legal) {
      GELOGW("[IndirectLoad] Mixed mode: post-probe validation of node[%s] failed.", members[index]->GetNamePtr());
      return af::SUCCESS;
    }
  }
  for (size_t index = 0UL; index < members.size(); ++index) {
    GE_ASSERT_SUCCESS(AnnotateTemplate(members[index], selection[index], analyses[index]));
  }
  // 归并/输出链 = normal schedule 扇出分支（SIMT 成员 invocation 之后执行；SIMD 成员输出
  // 为 UB 张量，向量归并直接消费）。
  GE_ASSERT_SUCCESS(AnnotateSimtGroupFanoutBranches(members));
  // 轴：SIMD 子集共享轴归一化（首个 SIMD 成员建轴 + per-IL input 内轴提示），SIMT 成员
  // 复用同一 TemplateAxes。
  GE_ASSERT_SUCCESS(NormalizeSimdAxesForGroupTemplate(graph, simd_members, simd_analyses));
  ascgen_utils::indirect_load::TemplateAxes shared_axes;
  GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetTemplateAxes(simd_members.front(), shared_axes));
  for (const af::AscNodePtr &simt_member : simt_members) {
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetTemplateAxes(simt_member, shared_axes));
  }
  for (size_t index = 0UL; index < members.size(); ++index) {
    GE_ASSERT_SUCCESS(CompletePreservedVectorizedViews(graph));
    GE_ASSERT_SUCCESS(FinalizeTemplate(members[index], selection[index]));
  }
  // SIMT 成员组员标记（member-form 元数据：index 链 + value 写 UB 局部目标）；不设融合
  // 发射标记（各成员走标准单 gather invocation）。
  for (const af::AscNodePtr &simt_member : simt_members) {
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetSimtGroupMember(simt_member));
  }
  for (const af::AscNodePtr &indirect_load : members) {
    bool metadata_supported = false;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::FinalizeLoweringMetadata(indirect_load, metadata_supported));
    if (!metadata_supported) {
      is_candidate_legal = false;
      GELOGW("[IndirectLoad] Reject mixed mode candidate: lowering metadata of node[%s] is unsupported.",
             indirect_load->GetNamePtr());
      return af::SUCCESS;
    }
  }
  is_candidate_legal = true;
  GELOGI("[IndirectLoad] Add mixed template candidate: %zu SIMD + %zu SIMT members (N=%zu).", simd_members.size(),
         simt_members.size(), members.size());
  return af::SUCCESS;
}

// 多 gather 水平融合候选生成（并存竞标制）：全部形态在调度期并存生成、进同一竞争
// 池交 CalcScore 竞标——L1/L4 共模板候选（SIMD×2 + SIMT×1，SIMT 内部按寄存器压力
// 模型定型 L2/L3/L4）+ L5 混模板候选（异构选择时产出，同构跳过）+ L6 SK 分段桥接
// 候选（打分为 0，可参与 ATT 性能比较）。统一打分为成员偏好
// 满足模型（GenerateGroupScoreFunc），同一套组聚合输入对全部形态可比；tiling 期
// 资源失败的兜底由 SK 在场结构性成立（单 gather 四候选并存机制的组路径推广）。
af::Status GenerateGroupCandidates(ascir::HintGraph &graph, const IndirectLoadGroup &group,
                                   std::vector<ascir::ImplGraph> &graphs, std::vector<std::string> &score_functions) {
  std::vector<af::AscNodePtr> members{group.primary};
  members.insert(members.end(), group.others.begin(), group.others.end());
  for (const af::AscNodePtr &member : members) {
    GE_ASSERT_SUCCESS(ValidateIndirectLoadNode(member));
  }
  GELOGI("[IndirectLoad] Generate gather group schedule candidates for graph[%s], primary[%s] with %zu members.",
         graph.GetName().c_str(), group.primary->GetNamePtr(), members.size());
  // [调度期 UB 预估]（编译期优化，非正确性依赖）：绝对下限硬门（与 shape 无关——
  // N×32B 最小行 + 32KB 预留超平台 UB 即任何形态都放不下）+ 形态化下界（SIMD 共循环
  // 流水要求全组切片同时驻留，硬下界 = Σ 切片 + 预留；SIMT 形态 2 需 N 份 value tile
  // 驻留），下界超平台 UB 即跳过该候选的生成与求解，由其余候选（含 SK 兜底）接管——
  // 未跳过的候选仍由 tiling 期资源校验把关。
  const bool ub_absolute_doomed = IsGroupUbAbsoluteDoomed(members);
  if (ub_absolute_doomed) {
    GELOGW(
        "[IndirectLoad] Skip gather group SIMD/SIMT candidates for primary[%s]: absolute UB lower bound "
        "(reserve + N*32B) exceeds platform capacity.",
        group.primary->GetNamePtr());
  }
  const bool simd_ub_doomed = ub_absolute_doomed || IsGroupSimdUbDoomed(members);
  const bool simt_ub_doomed = ub_absolute_doomed || IsGroupSimtL4UbDoomed(members);
  // [SK 前置检查] 图含 Transpose 时 SK 必非法，组级一次判定（消除逐成员重复扫描）。
  const bool sk_blocked = HasTransposeNode(graph);
  // [R-before-gather-axis 门控] 与单 gather 路径（Generate 的候选循环）同口径：归约轴位于
  // gather 轴之前时，SIMD/SIMT 的 post-Reduce 行/窗口语义不可用（归约段横跨 gather 轴外的
  // 行维，actual_size=k×R 与边界覆盖假定失效），仅 SK 分段桥接可行——组路径同样跳过非 SK
  // 共模板候选。布局以 primary 的共享 Reduce 终端判定（组配对已保证输出链一致）；无共享
  // Reduce（返回 false）与 Softmax（内部尾轴归约，BuildPostReduceLayout 识别不出归约轴）
  // 均不受影响。混模板候选无需显式门控：该布局下 SIMD 逐成员探测必拒、全组退化为同构
  // SIMT，与共模板候选重复而自然跳过。
  const bool reduce_before_gather_axis = HasReduceBeforeGatherAxis(group.primary);
  const TemplateCase cases[] = {
      {ascir::TemplateId::kIndirectLoadSimd, ascgen_utils::indirect_load::Implementation::kDefault},
      {ascir::TemplateId::kIndirectLoadSimd, ascgen_utils::indirect_load::Implementation::kGatherApi},
      {ascir::TemplateId::kIndirectLoadSimt, ascgen_utils::indirect_load::Implementation::kDefault}};
  // 首个 SIMD 候选（kDefault）的 lowering facade 判定：退化到 strided 时给组内 SIMT 候选
  // 加分（与单 gather 的 Generate 同源同判，组内以 primary 的布局为准）。
  bool simd_lowers_to_strided = false;
  for (const TemplateCase &template_case : cases) {
    const ascir::TemplateId template_id = template_case.template_id;
    if (reduce_before_gather_axis && template_id != ascir::TemplateId::kIndirectLoadSK) {
      GELOGI(
          "[IndirectLoad] Skip non-SK gather group candidate[%d] for R-before-gather-axis post-Reduce layout, "
          "primary[%s].",
          static_cast<int32_t>(template_id), group.primary->GetNamePtr());
      continue;
    }
    if (template_id == ascir::TemplateId::kIndirectLoadSimd && simd_ub_doomed) {
      GELOGI(
          "[IndirectLoad] Skip gather group SIMD candidate for primary[%s]: UB hard lower bound exceeds platform "
          "capacity (schedule-time estimation).",
          group.primary->GetNamePtr());
      continue;
    }
    if (template_id == ascir::TemplateId::kIndirectLoadSimt && simt_ub_doomed) {
      GELOGI(
          "[IndirectLoad] Skip gather group SIMT candidate for primary[%s]: L4 value tiles exceed platform UB "
          "(schedule-time estimation).",
          group.primary->GetNamePtr());
      continue;
    }
    ascir::ImplGraph candidate_graph(graph.GetName().c_str());
    GE_ASSERT_TRUE(candidate_graph.CopyFrom(graph), "Failed to copy graph [%s].", graph.GetName().c_str());
    IndirectLoadGroup candidate_group;
    candidate_group.primary = candidate_graph.FindNode(group.primary->GetName().c_str());
    GE_ASSERT_NOTNULL(candidate_group.primary, "Failed to find copied IndirectLoad node[%s].",
                      group.primary->GetName().c_str());
    for (const af::AscNodePtr &member : group.others) {
      const af::AscNodePtr candidate_member = candidate_graph.FindNode(member->GetName().c_str());
      GE_ASSERT_NOTNULL(candidate_member, "Failed to find copied IndirectLoad node[%s].", member->GetName().c_str());
      candidate_group.others.emplace_back(candidate_member);
    }
    for (const af::AscNodePtr &member : candidate_group.others) {
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetImplementation(member, template_case.implementation));
    }
    GE_ASSERT_SUCCESS(
        ascgen_utils::indirect_load::SetImplementation(candidate_group.primary, template_case.implementation));
    // 组内经 ele 归并后 Softmax Pattern 的原始输入不直接来自任一 IndirectLoad 输出，
    // 规范化必然不命中；保持与单 gather 一致的调用形态以覆盖归并节点即 Pattern 头的形态。
    bool softmax_normalized = false;
    GE_ASSERT_SUCCESS(
        softmax_pattern::NormalizeDirectPostSoftmax(candidate_graph, candidate_group.primary, softmax_normalized));
    if (softmax_normalized) {
      GELOGI("[IndirectLoad] Gather group candidate[%d] normalized direct post Softmax for node[%s].",
             static_cast<int32_t>(template_id), candidate_group.primary->GetNamePtr());
    }
    bool is_candidate_legal = false;
    GE_ASSERT_SUCCESS(ApplyGraphPassGroup(candidate_graph, candidate_group, template_id, is_candidate_legal));
    if (!is_candidate_legal) {
      GELOGW("[IndirectLoad] Skip illegal gather group template candidate[%d] for primary[%s] with %zu members.",
             static_cast<int32_t>(template_id), candidate_group.primary->GetNamePtr(), members.size());
      continue;
    }
    // 组聚合统计 + 统一打分（成员偏好满足模型，同一套输入对全部形态可比）。
    std::vector<af::AscNodePtr> candidate_members{candidate_group.primary};
    candidate_members.insert(candidate_members.end(), candidate_group.others.begin(), candidate_group.others.end());
    const GroupAccessStats stats = CollectGroupAccessStats(candidate_members);
    const bool simt_unfused = template_id == ascir::TemplateId::kIndirectLoadSimt &&
                              !ascgen_utils::indirect_load::IsSimtFusedPrimary(candidate_group.primary);
    if (template_id == ascir::TemplateId::kIndirectLoadSimd &&
        template_case.implementation == ascgen_utils::indirect_load::Implementation::kDefault) {
      // 任一成员退化到逐元素 strided facade 即视为该组 SIMD 候选退化，供后续 SIMT
      // 候选打分施加让位偏好（单 gather 的 LowersToStridedSimdFacade 同源语义，
      // 组版为全成员 ANY 判定）。
      for (const af::AscNodePtr &member : candidate_members) {
        if (LowersToStridedSimdFacade(member)) {
          simd_lowers_to_strided = true;
          break;
        }
      }
    }
    const GroupCandidateForm form =
        template_id == ascir::TemplateId::kIndirectLoadSimd ? GroupCandidateForm::kSimd : GroupCandidateForm::kSimt;
    graphs.emplace_back(std::move(candidate_graph));
    score_functions.emplace_back(GenerateGroupScoreFunc(form, template_case, stats, candidate_group.primary,
                                                        graphs.back(), simt_unfused, 0UL, simd_lowers_to_strided));
    GELOGI("[IndirectLoad] Add gather group schedule candidate[%d, %d] for primary[%s] with %zu members.",
           static_cast<int32_t>(template_id), static_cast<int32_t>(template_case.implementation),
           candidate_group.primary->GetNamePtr(), members.size());
  }
  // [L5 混模板候选·本期下线]（开关控制，不删代码）：mixed 的成员输出轴空间不做归一
  // （轴一致性仅隐式假设，无法保证链的轴空间一致 → 链无角色进组存在切错轴风险），
  // 完整 MVP/归一化留作后续独立需求。下线后异构组回落到共模板候选（全 SIMD/全
  // SIMT）或逐算子调度——编译失败风险优先于融合机会。
  constexpr bool kEnableMixedGroupCandidate = false;  // 本期下线（重启用需先落地 mixed 链 MVP）
  if (!kEnableMixedGroupCandidate) {
    GELOGI("[IndirectLoad] mixed disabled: group[%s] members[%zu] dropped (no homogeneous template covers all).",
           graph.GetName().c_str(), members.size());
  } else {
    ascir::ImplGraph candidate_graph(graph.GetName().c_str());
    GE_ASSERT_TRUE(candidate_graph.CopyFrom(graph), "Failed to copy graph [%s].", graph.GetName().c_str());
    IndirectLoadGroup candidate_group;
    candidate_group.primary = candidate_graph.FindNode(group.primary->GetName().c_str());
    GE_ASSERT_NOTNULL(candidate_group.primary, "Failed to find copied IndirectLoad node[%s].",
                      group.primary->GetNamePtr());
    for (const af::AscNodePtr &member : group.others) {
      const af::AscNodePtr candidate_member = candidate_graph.FindNode(member->GetName().c_str());
      GE_ASSERT_NOTNULL(candidate_member, "Failed to find copied IndirectLoad node[%s].", member->GetName().c_str());
      candidate_group.others.emplace_back(candidate_member);
    }
    std::vector<af::AscNodePtr> candidate_members{candidate_group.primary};
    candidate_members.insert(candidate_members.end(), candidate_group.others.begin(), candidate_group.others.end());
    for (const af::AscNodePtr &member : candidate_members) {
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetImplementation(
          member, ascgen_utils::indirect_load::Implementation::kDefault));
    }
    bool softmax_normalized = false;
    GE_ASSERT_SUCCESS(
        softmax_pattern::NormalizeDirectPostSoftmax(candidate_graph, candidate_group.primary, softmax_normalized));
    bool is_candidate_legal = false;
    size_t simd_selected_count = 0UL;
    GE_ASSERT_SUCCESS(ApplyGraphPassMixed(candidate_graph, candidate_members, is_candidate_legal, simd_selected_count));
    if (is_candidate_legal) {
      const GroupAccessStats stats = CollectGroupAccessStats(candidate_members);
      graphs.emplace_back(std::move(candidate_graph));
      score_functions.emplace_back(GenerateGroupScoreFunc(
          GroupCandidateForm::kMixed,
          TemplateCase{ascir::TemplateId::kIndirectLoadSimd, ascgen_utils::indirect_load::Implementation::kDefault},
          stats, candidate_group.primary, graphs.back(), false, simd_selected_count, false));
      GELOGI("[IndirectLoad] Add mixed template candidate for graph[%s] (simd=%zu, simt=%zu).", graph.GetName().c_str(),
             simd_selected_count, candidate_members.size() - simd_selected_count);
    }
  }
  // [L6 SK 分段桥接候选]（并存兜底，打分恒 0）：逐成员复用单 gather SK 流水（workspace
  // 边界把各路 IL 隔入独立段、归并/输出链作为消费段独立调度——各段保留各自调度）。
  // ATT 按性能估值比较可用的兜底候选；SK 自身
  // 不可行（Transpose / 复合 Norm / SK 校验否决 / dtype 门控）时无此候选。
  if (!sk_blocked) {
    ascir::ImplGraph candidate_graph(graph.GetName().c_str());
    GE_ASSERT_TRUE(candidate_graph.CopyFrom(graph), "Failed to copy graph [%s].", graph.GetName().c_str());
    IndirectLoadGroup candidate_group;
    candidate_group.primary = candidate_graph.FindNode(group.primary->GetName().c_str());
    GE_ASSERT_NOTNULL(candidate_group.primary, "Failed to find copied IndirectLoad node[%s].",
                      group.primary->GetNamePtr());
    for (const af::AscNodePtr &member : group.others) {
      const af::AscNodePtr candidate_member = candidate_graph.FindNode(member->GetName().c_str());
      GE_ASSERT_NOTNULL(candidate_member, "Failed to find copied IndirectLoad node[%s].", member->GetName().c_str());
      candidate_group.others.emplace_back(candidate_member);
    }
    for (const af::AscNodePtr &member : candidate_group.others) {
      GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetImplementation(
          member, ascgen_utils::indirect_load::Implementation::kDefault));
    }
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::SetImplementation(
        candidate_group.primary, ascgen_utils::indirect_load::Implementation::kDefault));
    bool is_candidate_legal = false;
    GE_ASSERT_SUCCESS(
        ApplyGraphPassGroup(candidate_graph, candidate_group, ascir::TemplateId::kIndirectLoadSK, is_candidate_legal));
    if (is_candidate_legal) {
      graphs.emplace_back(std::move(candidate_graph));
      score_functions.emplace_back(GenerateGroupScoreFunc(
          GroupCandidateForm::kSk,
          TemplateCase{ascir::TemplateId::kIndirectLoadSK, ascgen_utils::indirect_load::Implementation::kDefault},
          GroupAccessStats{}, candidate_group.primary, graphs.back(), false, 0UL, false));
      GELOGI("[IndirectLoad] Add gather group SK bridge candidate for graph[%s] (coexisting fallback).",
             graph.GetName().c_str());
    }
  }
  if (graphs.empty()) {
    GELOGW(
        "[IndirectLoad] All gather group candidates (common/mixed/SK bridge) rejected for graph[%s] with %zu "
        "members; no candidates emitted (frontend BackendSpec prevention applies).",
        graph.GetName().c_str(), members.size());
  }
  return af::SUCCESS;
}
// SK 前置检查：图含 Transpose 时 SK 必非法（ApplySkGraphPass 的逐成员全图扫描结论
// 提前到组级一次），同时避免逐成员重复扫描的冗余。
bool HasTransposeNode(const ascir::HintGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    if (af::ops::IsOps<af::ascir_op::Transpose>(node)) {
      return true;
    }
  }
  return false;
}

}  // namespace indirect_load_detail
}  // namespace optimize
