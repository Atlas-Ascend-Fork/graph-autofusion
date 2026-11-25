/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_INDIRECT_LOAD_SCORE_GENERATOR_H
#define AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_INDIRECT_LOAD_SCORE_GENERATOR_H

#include "task_generator/indirect_load/indirect_load_types.h"

namespace optimize {
namespace indirect_load_detail {

// 算法只有一份：vote(m) = slice_m ≥ 2048B ∧ lookup_m ≥ 32 → votes → satisfied(form)
// → 比例三档（satisfied×3/N ≥2 → 2、≥1 → 1、否则 0）→ 结构代价项（SIMD 组切片总量
// 过半 UB / SIMT L4 形态 / mixed 少数派 < 1/3）→ Softmax 专用 Norm 加分 → 负值截断
// → 基分 ×4 + 平局加分（平均 payload ≥ 2048B → SIMD，否则 SIMT——两侧同式可算，
// 静态/动态结论一致）。静态与动态的差别只在表达式叶子能否在生成期折叠：
// - 全部叶子常量 → 生成期折叠，发射 `return <字面量>;`（纯静态图行为与历史一致）
// - 任一叶子动态 → 运行期发射：sizevar → t.<name>（Subs 符号替换 + 逐原子 int64
//   提升防 uint32 乘法回绕），组绑定用占位符 kGroupTilingDataPlaceholder——ATT 发射
//   CalcScore 时按本模型 schedule_group_ident 替换为真实组子结构路径，与候选/组序
//   位置无关（序漂移不再错读他组形状；未替换的异常通路是未声明标识符，tiling 代码
//   编译期即失败，显性暴露）。
// 逐成员判定（无全组代表值）：历史静态路径取末成员 lookup 作代表并全组门控、动态
// 路径取首成员——成员 lookup 不同时两侧结论分歧；统一算法逐成员投票后两端一致。
// 不可物化成员（叶子含本图原始 sizevar 之外的符号）从分子分母同时剔除（N = 可评估
// 成员数，不引入方向性偏差）；N==0 时双方 0 分，由 ATT 比较性能估值。叶子符号判定按词法
// token：表达式 Str() 的全部标识符 token ∈ 原始 sizevar 名集合（ContainVar 为子串
// 语义，ks 会假命中 ks0，不作判据；Replace 为表达式整体匹配，改用 Subs 符号替换）。
enum class GroupCandidateForm {
  kSimd,   // 组 SIMD 共模板候选（L1）
  kSimt,   // 组 SIMT 共模板候选（内部形态 L2/L3/L4 由压力模型定型）
  kMixed,  // L5 混模板候选（异构选择，各成员以各自模板标注于同一 kernel）
  kSk      // L6 SK 分段桥接候选（终态兜底，打分恒 0）
};

// 统一算法的逐成员叶子（同一表达式树：静态图叶子为常量、动态图为符号——公式只有一份）。
struct MemberScoreLeaves {
  af::Expression slice_bytes;    // info.input_slice_bytes（payload 后缀字节）
  af::Expression lookup_extent;  // info.index_varying_extent（gather 侧 lookup 次数）
  // 尾轴窗口口径（info.has_tail_window 时有效）：结构回退平局在该形态下以整行窗口为
  // staging 单位替代退化为 dtype 字节的 input_slice_bytes；has_tail_window 为"已填充"
  // 哨兵，false 时两表达式不消费。capable（embedding）路径恒为非尾轴成员，不消费。
  bool has_tail_window = false;
  af::Expression window_bytes;         // info.window_bytes（单窗口 staging 字节）
  af::Expression window_lookup_count;  // info.window_lookup_count（单窗口 lookup 次数）
};

struct GroupAccessStats {
  size_t member_count = 0UL;     // 成员总数 N
  bool all_static = true;        // 全成员 slice/lookup 常量可得（仅结构回退路径的平局规则消费）
  int64_t slice_bytes_sum = 0L;  // Σ 常量切片字节（仅结构回退路径的静态平均 payload 消费）
  int64_t ub_size = 0L;          // 平台 UB 容量（0 = 未知，不施加 UB 代价项）
  bool simd_embedding_capable = false;     // AND(can_use_simd_embedding)
  bool simt_structured_capable = false;    // AND(can_use_simt_structured)
  std::vector<MemberScoreLeaves> members;  // 统一算法逐成员叶子（capable 路径消费）
};

// IndirectLoad 候选打分

// 组访问统计（G 轴形态与负载叶子），供组候选生成消费。
GroupAccessStats CollectGroupAccessStats(const std::vector<af::AscNodePtr> &members);

// 打分叶子的单一构造点：保证窗口口径在单 gather 与组路径间一致填充。
MemberScoreLeaves MakeMemberScoreLeaves(const ascgen_utils::indirect_load::IndirectLoadAccessInfo &info);
// IndirectLoad 候选打分：embedding fast-path 能力门控 / 统一打分与结构回退 ScoreFunc 代码生成（自
// indirect_load_schedule_case_generator.cpp 纯搬迁，语义不变）。
bool IsEmbeddingFastPathCapable(const TemplateCase &template_case,
                                const ascgen_utils::indirect_load::IndirectLoadAccessInfo &access_info);

bool LowersToStridedSimdFacade(const af::AscNodePtr &candidate_node);

std::string GenerateGroupScoreFunc(const GroupCandidateForm form, const TemplateCase &template_case,
                                   const GroupAccessStats &stats, const af::AscNodePtr &candidate_node,
                                   const af::AscGraph &candidate_graph, const bool simt_unfused,
                                   const size_t mixed_simd_count, const bool simd_lowers_to_strided);

std::string GenerateScoreFunc(const TemplateCase &template_case,
                              const ascgen_utils::indirect_load::IndirectLoadAccessInfo &access_info,
                              const af::AscNodePtr &candidate_node, const af::AscGraph &candidate_graph,
                              const bool simd_lowers_to_strided);

}  // namespace indirect_load_detail
}  // namespace optimize

#endif  // AUTOFUSE_OPTIMIZE_TASK_GENERATOR_INDIRECT_LOAD_{h_name.upper()}_H
