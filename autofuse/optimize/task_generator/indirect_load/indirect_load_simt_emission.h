/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_SIMT_EMISSION_H
#define AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_SIMT_EMISSION_H

#include "task_generator/indirect_load/indirect_load_types.h"

namespace optimize {
namespace indirect_load_detail {
// SIMT 融合发射计划：组扇出分支标注与发射计划编排（自 indirect_load_schedule_case_generator.cpp 纯搬迁，语义不变）。

// [SIMT 融合发射计划] 形态（L2/L3/L4）+ 寄存器压力 + 线程档的单一真值源。
// 时序：本函数在 lowering 元数据（位宽源，依赖模板轴归一化）之后落定最终形态；结构性
// L2 候选先按 L2 预注解，若因压力降级（L3/L4），调用方按本计划复位角色（角色与形态
// 必须一致）。模型由 kUsePeakModel 决定（peak 默认；sum 仅现场应急回退与 CI 对拍，
// 回退沿用旧公式与旧降级行为——单 demand、超限直接 L4）。可行性判定带 5% 安全边际
// （防低估 spill），线程档用未上浮的 demand（防无谓降档）。
// 链发射者不由本计划决定：L3/L4 的链不注解任何角色，VF/normal 由分区器按簇形态
// 自然裁决（多节点簇→VF；单节点簇不建子图→normal 主循环兜底）——不依赖任何预测。
struct SimtEmissionPlan {
  bool use_fused = false;       // L2/L3（seq-V1 融合发射）；false=L4（V2 逐成员 invocation）
  bool register_merge = false;  // true=L2（evaluator 内折叠整链）；仅 use_fused 有效
  uint64_t demand = 0U;
  uint32_t max_thread_tier = 2048U;
};

// 多 gather 组员（SIMT）扇出标注：组员计划不含主输出链（selected 仅含全组 IL 节点作
// [仅 mixed 路径引用·mixed 已下线] 本函数与 AnnotateSimtSideInputClosure 在组 SIMT
// 路径已无调用（组链无角色，VF/normal 由分区器裁决）；当前唯一调用方是
// ApplyGraphPassMixed（kEnableMixedGroupCandidate=false 编译期关闭）。保留供 mixed
// 重启用（需先落地 mixed 链 MVP——见含 Reduce 文档 11 节），单 gather 扇出路径使用
// 独立的 AnnotateSimtFanoutBranches，不受影响。
// 排除集），IL 输出的全部消费者（归并 ele/输出链）均为 normal schedule 扇出分支——
// 标注 kSimtFanoutBranch 使调度侧按既有语义处理 SIMT 输出到普通向量代码的边界；
// side input 闭包随分支标注，全组 IL 排除避免改写他组成员的 kSimtOp 角色。
af::Status AnnotateSimtGroupFanoutBranches(const std::vector<af::AscNodePtr> &members);

af::Status PlanSimtFusedEmission(const af::AscGraph &graph, const std::vector<af::AscNodePtr> &members,
                                 ascgen_utils::indirect_load::SimtFusedMergeInfo &merge_info);

af::Status PlanSimtEmission(const ascgen_utils::indirect_load::SimtFusedMergeInfo &merge_info,
                            const bool all_offset_uint32, SimtEmissionPlan &plan);

}  // namespace indirect_load_detail
}  // namespace optimize

#endif  // {guard}
