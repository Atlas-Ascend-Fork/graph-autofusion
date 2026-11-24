/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_AXIS_NORMALIZE_H
#define AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_AXIS_NORMALIZE_H

#include "task_generator/indirect_load/indirect_load_types.h"

namespace optimize {
namespace indirect_load_detail {
// 轴归一化：SIMD/SIMT/SK 轴归一化与模板轴恢复（自 indirect_load_schedule_case_generator.cpp 纯搬迁，语义不变）。

af::Status BuildSimdInnerAxis(af::AscGraph &graph, const af::AscNodePtr &input_producer, size_t axis_index,
                              const char *name, ascir::AxisId &input_inner_axis);

af::Status BuildSkInputInnerAxis(af::AscGraph &graph, const af::AscNodePtr &indirect_load, size_t axis_index,
                                 ascir::AxisId &input_inner_axis);

// SK 模板 IL 节点的 tmp 由串行窗口循环复用：单次 IndirectLoadSk 调用仅驻留固定 tile 行数
// （tile_inner 尺寸）的输入与 offsets；tile_outer 轴上 block 切分出的每核串行循环次数
// （<tile_outer 名>b_size）不进入驻留量，否则 UB 上限会与多核切分下限互相矛盾致 tiling
// 无解。填充条件：静态 shape 下全量驻留估算超出平台 UB（SK 会被 StaticUbTemplateFilter
// 误淘汰的形态）；动态 shape 下全量驻留量不可静态判定，同样填充。平台信息不可得 / outer
// 退化（size=1）时不填充，保持全量估算语义。
af::Status FillSkWindowReuseInfo(af::AscGraph &graph, const af::AscNodePtr &indirect_load, size_t boundary,
                                 af::AxisId outer_axis, af::AxisId tile_outer_axis);

// Only a standalone direct-GM SIMT region can change the scheduling unit
// without changing a normal scheduled consumer's UB tile or reduction window.
// Bound the grouped extent to the existing signed block-offset domain.
int64_t GetSimtGroupSize(const af::AscGraph &graph, const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                         const af::Axis &outer);

af::Status NormalizeAxesForTemplate(af::AscGraph &graph, const af::AscNodePtr &indirect_load, size_t boundary,
                                    ascir::TemplateId template_id, ascir::AxisId input_inner_axis,
                                    ascir::AxisId index_inner_axis, bool simd_index_uses_output_inner_axis = false,
                                    bool solve_tile_size = false);

af::Status SeedPostReduceInputVectorizedView(const af::AscNodePtr &indirect_load, const af::AscNodePtr &reduce);

// [组成员 value tensor 视图播种] 单 gather post-Reduce 的 IL 输出经输出重定向与 reduce 输入
// 为同一 tensor，SeedPostReduceInputVectorizedView 一步完成向量化视图播种；组路径无此 id
// 绑定——成员各自持有 value tensor，且 kSimtOp（preserves_vectorized_axis）使调度期
// TileSplit 的全局向量化写入跳过它们（视图保持任务生成期状态）。actual_size 经
// CalcVectorizedElementCount 读该视图，为空则恒为 1（即使逐行 tile，每次调用也
// 只写 1 个元素而非整行）。此处按共享模板的 vectorized 轴（boundary=first_reduce 时为
// inner 归约段的成员轴，仍在组员原轴空间内——组配对约束保证全组输出轴空间一致）为每个
// 成员的 value tensor 播种向量化视图，使 actual_size = 每行元素数；调度期 tensor merge
// 对播种轴的重映射与单 gather 同构（inner 成员轴重映射为 merged inner，不触碰 outer 的
// SelectLoopAxis 判定）。视图缺轴/视图非法是候选形态问题而非程序错误——NormalizeMemberAxes
// 跳过复合轴（from 非空）与分核/分 tile 轴的维（不归一到组共享轴）时，模板 vectorized 轴
// （merged inner 的 from 含成员轴）可能不在某成员视图内，is_candidate_legal=false 拒绝当前
// 候选（候选图为独立副本，丢弃无害），由其余候选/回退承接。
af::Status SeedGroupMemberValueVectorizedViews(const std::vector<af::AscNodePtr> &members, bool &is_candidate_legal);

af::Status NormalizeSimtAxesForTemplate(af::AscGraph &graph, const af::AscNodePtr &indirect_load, size_t boundary,
                                        bool solve_tile_size);
af::Status RestoreSkTemplateAxes(std::vector<ascir::ImplGraph> &grouped_graphs);

af::Status ReplaceAxisPrefix(std::vector<af::AxisId> &target_axes, const std::vector<af::AxisId> &output_axes,
                             size_t axis_index);

af::Status ReplaceRegionAxisPrefix(const NodePath &region, const std::vector<af::AxisId> &output_axes,
                                   size_t axis_index, bool is_input_region);

af::Status NormalizeSimdAxesForTemplate(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                        const RewrittenGraphAnalysis &analysis, bool solve_tile_size);

// 多 gather 共享轴归一化：primary 完整建轴（outer/inner/tile/vectorized 与单 gather 完全一致），
// 其余各成员复用共享轴、仅按需构建自己的 input 内轴（各路 input 的 dim 轴尺寸可以不同）；
// 各成员区域节点写入其 per-IL 内轴提示。组配对约束保证组内各路输出 axis id 完全一致，
// 因此 primary 的切分规则与循环组织形式天然对其余各路生效。
af::Status NormalizeSimdAxesForMember(af::AscGraph &graph, const af::AscNodePtr &primary,
                                      const RewrittenGraphAnalysis &primary_analysis, const af::AscNodePtr &member,
                                      const RewrittenGraphAnalysis &member_analysis);

// members 与 analyses 按序对齐：members[0] 为 primary（analyses[0] 对应），
// 其余为组内成员。primary 完整建轴一次，各成员逐个复用共享轴。
af::Status NormalizeSimdAxesForGroupTemplate(af::AscGraph &graph, const std::vector<af::AscNodePtr> &members,
                                             const std::vector<RewrittenGraphAnalysis> &analyses);

af::Status NormalizeSimtAxesForTemplate(af::AscGraph &graph, const af::AscNodePtr &indirect_load, size_t boundary,
                                        bool solve_tile_size);

// [跨成员 index 链共享节点豁免集] 收集全组成员的 backward region 并集 + 全部 IL 节点。
// 共享 index Load 的跨成员消费者不视为"外部"——由另一成员的 evaluator 逐元素消费
// （寄存器级），不要求向量侧物化 tensor。单成员/单 gather 语义下并集 = 自身 region，
// 行为与空集等价。
NodeSet BuildGroupIndexExemptSet(const std::vector<af::AscNodePtr> &members,
                                 std::vector<NodeSet> *back_regions = nullptr);

// 多 gather 组员（SIMT）：仅收集本路 index 生产链 backward 区域（含 index 链外部 store 折叠）。
// 共享归并/输出链不进入任何组员的 SIMT 区域——它们作为普通向量代码在全部组员 invocation
// 之后执行，避免输出链角色冲突与 evaluator 跨依赖。
bool CollectSimtGroupIndexRegion(const af::AscGraph &graph, const af::AscNodePtr &indirect_load, NodePath &index_region,
                                 const NodeSet &group_exempt, const NodeSet *pre_back_region = nullptr);

// 多 gather 组员（SIMT）共享轴归一化：primary 完整归一一次，其余成员复用 primary 的
// TemplateAxes——输出轴空间与 boundary 全组相同，轴结构一致；SIMT 归一不触碰区域节点轴，
// 拷贝语义等价，且避免 BuildAxisViewByBoundary/CreateFixedTileSplit 按成员重复建轴。
// boundary 语义（与单 gather 同构，按归约形态分档）：
// - 无共享 Reduce 的组（L2 寄存器归并 / 非归约触发的 L3）：boundary=输出 rank——整空间合并
//   轴，固定 tile inner=1 + codegen 整核调用形态；
// - 有共享 Reduce 的组（归约触发的 L3）：boundary=first_reduce——outer=行/批（G）、
//   inner=归约段（R），行不被 BlockSplit 劈开。solve_tile 与单 gather 同款判定：AR 形态
//   （归约段后无 Retained 轴，含尾轴整体归约的 Sum）与形状保持 Norm 开多行求解（k 行
//   tile，一次调用覆盖 k 行）；非 AR 形态（归约段后有 Retained 后缀）固定单行——多行窗口
//   会把多组 RA 段并进同一 Reduce 窗口，归约语义错误，k=1 是正确性约束。
// boundary_override：无共享 Reduce 的组可由调用方指定 template boundary（0 = 沿用输出 rank）。
af::Status NormalizeSimtAxesForGroupTemplate(af::AscGraph &graph, const std::vector<af::AscNodePtr> &members,
                                             const af::AscNodePtr &group_post_reduce, bool &is_candidate_legal,
                                             size_t boundary_override = 0UL);

af::Status NormalizeTemplateAxes(af::AscGraph &graph, const af::AscNodePtr &indirect_load,
                                 ascir::TemplateId template_id, const RewrittenGraphAnalysis &analysis,
                                 size_t boundary);

}  // namespace indirect_load_detail
}  // namespace optimize

#endif  // {guard}
