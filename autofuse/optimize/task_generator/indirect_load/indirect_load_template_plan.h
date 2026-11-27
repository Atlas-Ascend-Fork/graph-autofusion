/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_TEMPLATE_PLAN_H
#define AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_TEMPLATE_PLAN_H

#include "task_generator/indirect_load/indirect_load_types.h"

namespace optimize {
namespace indirect_load_detail {

// 全图 vectorized 视图补全（含行广播标记播种）。
af::Status CompletePreservedVectorizedViews(const af::AscGraph &graph);

// SIMT 模板角色标注（SIMD 版本见 axis 归一化模块）。

// 模板计划：候选合法性校验 + 角色标注 + Validate/Annotate/Finalize（自 indirect_load_schedule_case_generator.cpp
// 纯搬迁，语义不变）。

bool HasBroadcastMultiInputNode(const af::AscNodePtr &node, NodeSet &visited);

bool IsSkTemplateCandidateLegal(const af::AscNodePtr &indirect_load);

af::Status ValidateIndirectLoadNode(const af::AscNodePtr &indirect_load);

// SIMT direct-GM nodes deliberately preserve their vectorized view through the
// scheduler.  User graphs, however, do not necessarily initialize that view;
// fill it from the node's physical strides before VF partitioning consumes it.
// 跳过全部 IndirectLoad 节点而非仅当前成员：IL 输出 vectorized 视图按模板约定保持空，
// 由调度器的 tensor merge/split 主动改写。组路径按成员循环调用本函数，若仅跳过当前
// 成员，primary 的调用会把其他成员（kSimtOp，preserves_vectorized_axis=true）当普通
// 保留节点播种——SIMT 组 post_reduce 置空使 outer=全部输出轴合并，调度期 tensor merge
// 将播种轴重映射为 merged outer，SelectLoopAxis 据此把 outer 放进 not_loop 集并经
// from 递归过滤掉成员全部调度轴（tile 轴 from=;

// [护栏] IL 输出 vectorized 轴有效性前置校验：为空，或全部是其当前视图 attr.axis 的
// 子序列。合法形态（如带 post-Reduce 的单 gather SIMD/SIMT）IL 输出 vectorized 可
// 非空（inner 轴族）；但轴必须仍存在于视图——生成期任何路径把已失效（被 tensor
// merge 折叠掉）或外来轴写进 vectorized，调度期会在 GetVectorRepeats/
// InitTensorMemInfo 或 SelectLoopAxis 以无关断言隐蔽失败。候选产出后即刻校验，
// 把违约暴露在生成侧。
af::Status ValidateIndirectLoadOutputVectorizedAxes(const ascir::ImplGraph &graph);

af::Status AnnotateSimdTemplateRoles(const RewrittenGraphAnalysis &analysis);

af::Status ValidateSimtTemplateRegion(const RewrittenGraphAnalysis &analysis, bool &is_candidate_legal);

af::Status CollectSimtNormalScheduleNodes(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis,
                                          NodeSet &normal_schedule_nodes);

af::Status AnnotateSimtMainOutputPath(const af::AscNodePtr &indirect_load, const af::AscNodePtr &selected_root,
                                      const NodeSet &selected, const NodeSet &normal_schedule_nodes);

// [fanout side-input 闭包] fanout 分支节点生产链的角色标注；upgrade_inline_transform
// （无 post-Reduce 形态传 true）允许把区域标注的 kSimtInlineTransform 升级为 fanout
// 分支角色——多分支区域扩容会把分支侧 side-input 生产者先标为 InlineTransform，不升级
// 则 UB 生命周期被无谓保留且与分支其余节点角色割裂；post-Reduce 形态禁止升级（0159
// 语义：区域内成员角色是 DirectPostReduceKeepsParallelSimtOutputRoles 的断言对象）。
// 缺省 false：组员 mixed 路径（已下线）保持既有行为。
af::Status AnnotateSimtSideInputClosure(const af::AscNodePtr &branch_node, const af::AscNodePtr &indirect_load,
                                        const NodeSet &selected, const NodeSet &normal_schedule_nodes, NodeSet &visited,
                                        bool upgrade_inline_transform = false);

af::Status AnnotateSimtFanoutBranches(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis,
                                      NodeSet &normal_schedule_nodes);

// SIMD 多阶段布局校验：区域内每个统计 Reduce 独立通过
// ValidateSimdPostReduceLayout（G≥R + 后缀连续），首个失败即淘汰候选。
af::Status ValidateSimdCompositeLayouts(const af::AscNodePtr &indirect_load,
                                        const std::vector<af::AscNodePtr> &composite_reduces, bool &is_candidate_legal);

af::Status ValidateTemplate(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                            const RewrittenGraphAnalysis &analysis, size_t &boundary, bool &is_candidate_legal);

af::Status AnnotateTemplate(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                            const RewrittenGraphAnalysis &analysis);

// 为区域内全部统计 Reduce 种输入向量化视图：SIMD 多阶段链中每个 Reduce
// 都按模板向量化轴消费数据，缺失种子会导致 Codegen 无法组织 A/R 参数。
af::Status SeedCompositeReduceInputViews(const af::AscNodePtr &indirect_load,
                                         const std::vector<af::AscNodePtr> &composite_reduces);

af::Status FinalizeTemplate(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id);

}  // namespace indirect_load_detail
}  // namespace optimize

#endif  // {guard}
