/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_TEMPLATE_ANNOTATION_H
#define AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_TEMPLATE_ANNOTATION_H

#include "task_generator/indirect_load/indirect_load_types.h"

namespace optimize {
namespace indirect_load_detail {
// 模板角色标注：SIMD/SIMT 角色标注、SIMT 融合集与 side input 闭包、扇出分支标注、
// vectorized 视图补全（S4 自 template_plan 细分，纯搬迁）。

// SIMT direct-GM nodes deliberately preserve their vectorized view through the
// scheduler.  User graphs, however, do not necessarily initialize that view;
// fill it from the node's physical strides before VF partitioning consumes it.
// 跳过全部 IndirectLoad 节点而非仅当前成员：IL 输出 vectorized 视图按模板约定保持空，
// 由调度器的 tensor merge/split 主动改写。组路径按成员循环调用本函数，若仅跳过当前
// 成员，primary 的调用会把其他成员（kSimtOp，preserves_vectorized_axis=true）当普通
// 保留节点播种——SIMT 组 post_reduce 置空使 outer=全部输出轴合并，调度期 tensor merge
// 将播种轴重映射为 merged outer，SelectLoopAxis 据此把 outer 放进 not_loop 集并经
// from 递归过滤掉成员全部调度轴（tile 轴 from=;

af::Status AnnotateSimdTemplateRoles(const RewrittenGraphAnalysis &analysis);

af::Status ValidateSimtTemplateRegion(const RewrittenGraphAnalysis &analysis, bool &is_candidate_legal);

af::Status CollectSimtNormalScheduleNodes(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis,
                                          NodeSet &normal_schedule_nodes);

af::Status AnnotateSimtMainOutputPath(const af::AscNodePtr &indirect_load, const af::AscNodePtr &selected_root,
                                      const NodeSet &selected, const NodeSet &normal_schedule_nodes);

// upgrade_inline_transform 语义与边界说明（含缺省值）见 indirect_load_template_plan.h
// （同一函数的两处声明保持同步，缺省参数只在 template_plan.h 给出）。
af::Status AnnotateSimtSideInputClosure(const af::AscNodePtr &branch_node, const af::AscNodePtr &indirect_load,
                                        const NodeSet &selected, const NodeSet &normal_schedule_nodes, NodeSet &visited,
                                        bool upgrade_inline_transform);

af::Status AnnotateSimtFanoutBranches(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis,
                                      NodeSet &normal_schedule_nodes);

// 为区域内全部统计 Reduce 种输入向量化视图：SIMD 多阶段链中每个 Reduce
// 都按模板向量化轴消费数据，缺失种子会导致 Codegen 无法组织 A/R 参数。
af::Status SeedCompositeReduceInputViews(const af::AscNodePtr &indirect_load,
                                         const std::vector<af::AscNodePtr> &composite_reduces);

af::Status AnnotateSimtTemplateRoles(const af::AscNodePtr &indirect_load, const RewrittenGraphAnalysis &analysis);

}  // namespace indirect_load_detail
}  // namespace optimize

#endif  // AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_TEMPLATE_ANNOTATION_H
