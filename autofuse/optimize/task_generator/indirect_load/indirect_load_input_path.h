/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_INPUT_PATH_H
#define AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_INPUT_PATH_H

#include "task_generator/indirect_load/indirect_load_types.h"

namespace optimize {
namespace indirect_load_detail {
// 输入侧分析：拓扑校验/区域收集/物理视图/input-pre 改写/输入路径重写（自 indirect_load_schedule_case_generator.cpp
// 纯搬迁，语义不变）。

bool IsSupportedIndirectLoadTopologyNode(const af::AscNodePtr &node);

void CollectInputRegionMembers(const af::AscNodePtr &indirect_load, size_t input_index, NodeSet &region);

bool CollectSimtBackwardRegion(const NodePath &roots, const af::AscNodePtr &indirect_load, NodeSet &region);

void CollectSimtOutputConsumerClosure(const af::AscNodePtr &indirect_load, NodeSet &closure);

// [index 链外部 store] index 生产链尾部张量的区域外直连 store 会被 lowering 收为
// 额外输出链（链值 = Index() 求值器结果），此处将其纳入 SIMT 融合区域（获得
// kSimtDirectGmBoundary 角色，跳过普通调度发射）；index 链其余无法供值的外部消费者
// 静默丢弃会导致目标 buffer 永不写入，必须整候选淘汰。合法消费者集 = IndirectLoad
// 自身 ∪ 融合区域（index 链与输出链的 backward 闭包——含经多输入算子合流进输出链
// 的 side-input，如 embedding+reduce 图中消费 index load 的 Ge/Select）∪ IL 输出的
// 前向闭包（post-Reduce 普通调度链）；守卫只扫 index 链子集，不扫全融合区域。
bool FoldSimtIndexChainExternalStores(const af::AscNodePtr &indirect_load, const af::AscNodePtr &index_root,
                                      NodeSet &region, const NodeSet &group_exempt,
                                      const NodeSet *pre_index_chain = nullptr);

void CollectSimtFusedRegionMembers(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis,
                                   NodeSet &region);

// The IndirectLoad logical view is expressed in the Transpose output order,
// while the direct-GM boundary still reads the original Load buffer. Map the
// source Load strides into that output order for the strided address policy.
af::Status BuildLoadTransposeSourceView(const af::AscNodePtr &transpose, const af::AscTensorAttr &logical_attr,
                                        ascgen_utils::indirect_load::LogicalTensorView &view);

bool CanMoveInputPreNode(const af::AscNodePtr &node, const af::AscNodePtr &consumer);

bool IsSimdInputShapeEnlarged(const af::AscNodePtr &indirect_load);

af::Status MoveInputPreNode(const af::AscNodePtr &node, const af::AscNodePtr &indirect_load);

af::Status RewriteInputPreNodes(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                ascir::TemplateId template_id);

af::Status RewriteSimtInputBroadcast(const af::AscNodePtr &indirect_load, InputViewPlan &input_plan,
                                     bool &is_candidate_legal);

// 不区分一元/二元路径，沿所有上游数据边收集 Load；Load 节点是输入边界，停止继续回溯。
void CollectReachableLoads(const af::AscNodePtr &root, NodeSet &visited, NodePath &loads);

af::Status CompleteInputDataTensorAttrs(const RewrittenGraphAnalysis &analysis);

// 收集 IndirectLoad 两个输入的路径，并按模板规则选出待处理的 Broadcast：SIMD 只看直接生产者，SIMT/SK
// 沿单输入链回溯并查找最近 Broadcast。
void CollectInputPaths(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                       PhysicalViewPreparation &preparation);

af::Status ResolveInputPhysicalView(const af::AscTensorAttr &logical_attr, ascir::TemplateId template_id,
                                    InputViewPlan &plan, af::AscTensorAttr &physical_attr, bool &is_supported);

af::Status AnalyzeInputPath(const af::AscNodePtr &indirect_load, size_t input_idx, ascir::TemplateId template_id,
                            InputViewPlan &plan, bool &is_path_supported);

af::Status RewriteSkInputPaths(PhysicalViewPreparation &preparation, RewrittenGraphAnalysis &analysis,
                               bool &is_candidate_legal);

af::Status RewriteSimdInputLayouts(PhysicalViewPreparation &preparation);

af::Status RewriteBroadcastPaths(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                                 PhysicalViewPreparation &preparation, RewrittenGraphAnalysis &analysis,
                                 bool &is_candidate_legal);

af::Status PreparePhysicalViews(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                                PhysicalViewPreparation &preparation, bool &is_candidate_legal);

af::Status CollectRewrittenRegion(const af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                  ascir::TemplateId template_id, RewrittenGraphAnalysis &analysis);

af::Status ApplyTemplatePathLayouts(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                                    const PhysicalViewPreparation &preparation, const RewrittenGraphAnalysis &analysis);

af::Status AnalyzeRewrittenGraph(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                 ascir::TemplateId template_id, bool &is_candidate_legal,
                                 RewrittenGraphAnalysis &analysis);

}  // namespace indirect_load_detail
}  // namespace optimize

#endif  // {guard}
