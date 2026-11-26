/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_GROUP_CANDIDATE_H
#define AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_GROUP_CANDIDATE_H

#include "task_generator/indirect_load/indirect_load_types.h"
#include "task_generator/indirect_load/indirect_load_score_generator.h"

namespace optimize {
namespace indirect_load_detail {
// 多 gather 组候选：组配对合法性、共模板/混模板改写与组候选生成（自 indirect_load_schedule_case_generator.cpp
// 纯搬迁，语义不变）。

// 多 gather 水平融合组配对（约束见《双gather水平融合设计文档》）：
// 1) 组内各路 G 轴（负轴归一化后）与 primary 相等；
// 2) 输出 tensor 轴空间一致（axis id 逐元素相等，归并 ele 的图合法性天然保证）；
// 3) index 逻辑 shape 符号级完全相等（逐路与 primary 比较）；
// 4) 组内各路输出的前向消费者闭包与 primary 存在公共节点（经 ele 归并树汇入同一条输出链）。
// 数量不设上限（N >= 2 均进入组配对）。全部为符号级静态判定，判定不确定（kUnknown）一律视为不满足。
struct IndirectLoadGroup {
  af::AscNodePtr primary;              // GetAllNodes 顺序中的第一个 IndirectLoad
  std::vector<af::AscNodePtr> others;  // 其余 IndirectLoad（GetAllNodes 顺序）
};

bool IsGroupMemberLegal(const af::AscNodePtr &primary, const af::AscNodePtr &member);

af::Status CollectIndirectLoadEntry(const ascir::HintGraph &graph, af::AscNodePtr &indirect_load,
                                    IndirectLoadGroup &group, bool &is_group);

// 多 gather 水平融合候选流水：逐 IL 复用单 gather 的分析/校验/标注/元数据机制，仅轴归一化
// 改为共享模板轴（NormalizeSimdAxesForGroupTemplate / NormalizeSimtAxesForGroupTemplate）。
// G≥R、复合 Norm 等输出侧校验对组内各路判定必然一致（axis 相同、输出边界相同），逐 IL 执行
// 以完整复用既有校验实现。SIMT 档位采用组员形态：各路 evaluator 仅含本路 index 链与 input
// 读取，value 写 UB 局部目标；共享归并/输出链作为普通向量代码在全部 invocation 之后执行。
af::Status ApplyGraphPassGroup(af::AscGraph &graph, const IndirectLoadGroup &group, ascir::TemplateId template_id,
                               bool &is_candidate_legal);

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
                               bool &is_candidate_legal, size_t &simd_selected_count);

// 多 gather 水平融合候选生成（并存竞标制）：全部形态在调度期并存生成、进同一竞争
// 池交 CalcScore 竞标——L1/L4 共模板候选（SIMD×2 + SIMT×1，SIMT 内部按寄存器压力
// 模型定型 L2/L3/L4）+ L5 混模板候选（异构选择时产出，同构跳过）+ L6 SK 分段桥接
// 候选（打分为 0，可参与 ATT 性能比较）。统一打分为成员偏好
// 满足模型（GenerateGroupScoreFunc），同一套组聚合输入对全部形态可比；tiling 期
// 资源失败的兜底由 SK 在场结构性成立（单 gather 四候选并存机制的组路径推广）。
af::Status GenerateGroupCandidates(ascir::HintGraph &graph, const IndirectLoadGroup &group,
                                   std::vector<ascir::ImplGraph> &graphs, std::vector<std::string> &score_functions);

// 图内是否含 Transpose（SK 前置检查）。
bool HasTransposeNode(const ascir::HintGraph &graph);

}  // namespace indirect_load_detail
}  // namespace optimize

#endif  // {guard}
