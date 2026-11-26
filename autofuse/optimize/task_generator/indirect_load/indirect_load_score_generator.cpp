/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_generator/indirect_load/indirect_load_score_generator.h"
#include "task_generator/indirect_load/indirect_load_view_layout.h"
#include "task_generator/indirect_load/indirect_load_input_path.h"
#include "task_generator/indirect_load/indirect_load_output_analysis.h"
#include "task_generator/indirect_load/indirect_load_template_plan.h"
#include "task_generator/indirect_load/indirect_load_template_annotation.h"
#include "task_generator/indirect_load/indirect_load_axis_normalize.h"
#include "task_generator/indirect_load/indirect_load_sk_partition.h"
#include "task_generator/indirect_load/indirect_load_simt_emission.h"
#include "task_generator/indirect_load/indirect_load_group_candidate.h"

#include "task_generator/indirect_load/indirect_load_graph_utils.h"
#include "task_generator/indirect_load/indirect_load_ub_feasibility.h"

#include "v35/ascir/ascir_codegen_v2.h"
#include "ascgraph_info_complete.h"
#include "schedule_result.h"

namespace optimize {
namespace indirect_load_detail {
// 尾轴稠密 gather 窗口口径的触及粒度（字节，cache line 级）：单窗口 lookup 次数按该粒度
// 折算触及字节，折算值不低于窗口字节视为高密度（窗口驻留 staging 的读取放大有界）；
// 低于该密度时逐元素 SIMT 直读更优，维持 input_slice_bytes 口径不变。
constexpr int64_t kDenseWindowLineBytes = 64;
bool IsEmbeddingFastPathCapable(const TemplateCase &template_case,
                                const ascgen_utils::indirect_load::IndirectLoadAccessInfo &access_info) {
  if (template_case.implementation != ascgen_utils::indirect_load::Implementation::kDefault) {
    return false;
  }
  if (template_case.template_id == ascir::TemplateId::kIndirectLoadSimt) {
    return access_info.can_use_simt_structured;
  }
  if (template_case.template_id == ascir::TemplateId::kIndirectLoadSimd) {
    return access_info.can_use_simd_embedding;
  }
  return false;
}

// SIMD 候选是否退化到 strided facade：索引/输入布局非 dense 或输出存在物理 padding，
// 或调度期预测通用对齐将引入 padded 视图时，SIMD 只能逐元素标量组装 offsets 并逐元素
// 重排（与最终 lowering 的 facade 判定同源）。
bool LowersToStridedSimdFacade(const af::AscNodePtr &candidate_node) {
  bool is_supported = false;
  if (ascgen_utils::indirect_load::FinalizeLoweringMetadata(candidate_node, is_supported) != af::SUCCESS ||
      !is_supported) {
    return false;
  }
  ascgen_utils::indirect_load::IndirectLoadLoweringMetadata metadata;
  if (ascgen_utils::indirect_load::GetLoweringMetadata(candidate_node, metadata) != af::SUCCESS) {
    return false;
  }
  if (metadata.simd.fallback == ascgen_utils::indirect_load::SimdFallback::kStrided) {
    return true;
  }
  // [调度期 padding 预测] 通用对齐晚于调度：lowering metadata 序列化时视图仍为原生
  // 稠密（pre-check out_padded[0] 漏判），codegen 时期尾轴 pad 后 dense 必降级 strided。
  // 按尾轴对齐性静态预测，让 GenerateScoreFunc 的 SIMT 让位评分提前生效。
  return ascgen_utils::indirect_load::PredictSimdAlignmentPadding(candidate_node);
}

// ==================== 统一打分（静态/动态同算法，组/单 gather 同源） ====================

// 在候选图副本上收集（access_info ext attr 由成员各自的分析流水写入，与既有
// primary 单点读取同一时机）。
// 打分叶子的单一构造点：保证窗口口径在单 gather 与组路径间一致填充。
MemberScoreLeaves MakeMemberScoreLeaves(const ascgen_utils::indirect_load::IndirectLoadAccessInfo &info) {
  MemberScoreLeaves leaves;
  leaves.slice_bytes = info.input_slice_bytes;
  leaves.lookup_extent = info.index_varying_extent;
  leaves.has_tail_window = info.has_tail_window;
  leaves.window_bytes = info.window_bytes;
  leaves.window_lookup_count = info.window_lookup_count;
  return leaves;
}

GroupAccessStats CollectGroupAccessStats(const std::vector<af::AscNodePtr> &members) {
  GroupAccessStats stats;
  stats.member_count = members.size();
  stats.all_static = true;
  stats.simd_embedding_capable = true;
  stats.simt_structured_capable = true;
  (void)TryGetIndirectLoadPlatformUbSize(stats.ub_size);
  for (const af::AscNodePtr &member : members) {
    ascgen_utils::indirect_load::IndirectLoadAccessInfo info;
    GE_ASSERT_SUCCESS(ascgen_utils::indirect_load::GetIndirectLoadAccessInfo(member, info));
    stats.simd_embedding_capable = stats.simd_embedding_capable && info.can_use_simd_embedding;
    stats.simt_structured_capable = stats.simt_structured_capable && info.can_use_simt_structured;
    stats.members.emplace_back(MakeMemberScoreLeaves(info));
    int64_t slice_bytes = 0L;
    if (!info.input_slice_bytes.GetConstValue(slice_bytes)) {
      stats.all_static = false;
      continue;
    }
    int64_t member_lookup = 0L;
    if (!info.index_varying_extent.GetConstValue(member_lookup)) {
      stats.all_static = false;
      continue;
    }
    stats.slice_bytes_sum += slice_bytes;
  }
  return stats;
}
// —— 叶子物化辅助 ——
// 表达式 Str() 的标识符 token（[A-Za-z_][A-Za-z0-9_]*；数字开头视为字面量跳过）。
std::set<std::string> ExprIdentifierTokens(const af::Expression &expr) {
  const std::string text = std::string(expr.Str().get());
  std::set<std::string> tokens;
  std::string current;
  const auto flush = [&tokens, &current]() {
    if (!current.empty() && std::isdigit(static_cast<unsigned char>(current.front())) == 0) {
      tokens.insert(current);
    }
    current.clear();
  };
  for (const char ch : text) {
    if (std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '_') {
      current += ch;
      continue;
    }
    flush();
  }
  flush();
  return tokens;
}

// 叶子可物化：全部标识符 token ∈ 原始 sizevar 名集合 ∪ 已知符号函数白名单
// （Ceiling/Floor/Min/Max 等在 tiling 代码上下文可直接发射；常量叶子无标识符
// token，恒可物化）。白名单外的函数 token（未知对齐/取整类）→ 该成员剔除并记
// 日志（否则线上只能看到分数莫名平票，无法定位）。
bool ExprMaterializable(const af::Expression &expr, const std::set<std::string> &var_names,
                        std::string &rejected_token) {
  // [函数白名单] 表达式打印器词表（expr_print_manager.cc kPrint* 系）与 tiling 代码
  // solver 头 helper 的交集——既能出现在 Str() 又有同名可编译定义。helper 由
  // GenExpressionMacro()（tiling_code_gen_impl.cpp:2011，源 embed 自
  // att/core/scalar/expression_helpers.inc）和 GenToolFuncs()（:2070，源 embed 自
  // att/core/scalar/scalar_helpers.inc）无条件发射到 CalcScore 所在翻译单元：
  //   Max/Min/Abs/Log/Pow/Rational ← GenExpressionMacro（inline 模板 + 宏）；
  //   Ceiling/Floor/Mod ← GenToolFuncs（Mod 有整型 % 与浮点 fmod 双重载）。
  // 交集外不收：Exp/Sqrt 能打印但无 helper（放行会编译失败）；IsEqual/TernaryOp
  // 有 helper 但不在打印器词表（不会作为 token 出现）；ExpectEq/Ne/Le/Lt 比较结果
  // 当字节数无语义（出现时保守剔除 + 日志）。Pow/Log/Rational 返回 double——入表
  // 只为不误剔除（防御性条目），不代表出现在尺寸表达式中有物理意义。白名单只按
  // token 名放行，不校验 arity/类型——靠 C++ 编译报错兜底（显性失败）。
  static const std::set<std::string> kKnownFunctionTokens = {"Max", "Min", "Abs", "Ceiling", "Floor",
                                                             "Mod", "Pow", "Log", "Rational"};
  const std::set<std::string> tokens = ExprIdentifierTokens(expr);
  for (const std::string &token : tokens) {
    if (var_names.count(token) != 0UL || kKnownFunctionTokens.count(token) != 0UL) {
      continue;
    }
    rejected_token = token;
    return false;
  }
  return true;
}

// 运行期物化：Subs 符号替换（叶级符号替换，规避 Replace 的表达式整体匹配限制）。
std::string RuntimeExprToCpp(const af::Expression &expr,
                             const std::vector<std::pair<af::Expression, af::Expression>> &subs) {
  return std::string(expr.Subs(subs).Str().get());
}

// —— 折叠物化（全部叶子常量；与运行期发射同公式） ——
std::string FoldScoreFuncText(const GroupCandidateForm form, const std::vector<MemberScoreLeaves> &members,
                              const bool softmax_bonus, const bool simd_ub_penalty, const int64_t ub_size,
                              const bool simt_l4_penalty, const bool mixed_minority_penalty) {
  size_t votes = 0UL;
  int64_t slice_sum = 0L;
  if (form != GroupCandidateForm::kMixed) {
    for (const MemberScoreLeaves &member : members) {
      int64_t slice_bytes = 0L;
      int64_t lookup_extent = 0L;
      GE_ASSERT_TRUE(member.slice_bytes.GetConstValue(slice_bytes) && member.lookup_extent.GetConstValue(lookup_extent),
                     "IndirectLoad unified score fold requires constant member leaves.");
      if (slice_bytes >= kEmbeddingSimdPayloadBytesThreshold && lookup_extent >= kEmbeddingSimdLookupCountThreshold) {
        ++votes;
      }
      slice_sum += slice_bytes;
    }
  }
  const size_t member_count = members.size();
  const size_t satisfied = form == GroupCandidateForm::kSimd
                               ? votes
                               : (form == GroupCandidateForm::kSimt ? member_count - votes : member_count);
  const int64_t ratio_x3 =
      member_count == 0UL ? 0L : static_cast<int64_t>(satisfied * 3UL) / static_cast<int64_t>(member_count);
  int32_t score = ratio_x3 >= 2L ? kEmbeddingFastPathScore : (ratio_x3 >= 1L ? kEmbeddingAlternateFastPathScore : 0);
  if (simd_ub_penalty && member_count > 1UL && ub_size > 0L && slice_sum * 2L > ub_size) {
    --score;  // 组切片总量过半 UB：tile 缩水风险进分数
  }
  if (simt_l4_penalty) {
    --score;  // L4 形态（非融合发射）：相位 N+1、barrier ×N
  }
  if (mixed_minority_penalty) {
    --score;  // 少数派 < 1/3：混搭整合开销不划算
  }
  if (softmax_bonus) {
    score += 1;
  }
  if (score < 0) {
    score = 0;
  }
  int32_t tie_bonus = 0;
  if (form == GroupCandidateForm::kSimd || form == GroupCandidateForm::kSimt) {
    const int64_t avg_payload = member_count == 0UL ? 0L : slice_sum / static_cast<int64_t>(member_count);
    tie_bonus = form == GroupCandidateForm::kSimd ? (avg_payload >= kEmbeddingSimdPayloadBytesThreshold ? 1 : 0)
                                                  : (avg_payload < kEmbeddingSimdPayloadBytesThreshold ? 1 : 0);
  }
  const int32_t final_score = score * 4 + tie_bonus;
  return "int32_t CalcScore(const AutofuseTilingData &tiling_data) {\n"
         "  (void)tiling_data;\n"
         "  return " +
         std::to_string(final_score) +
         ";\n"
         "}\n";
}

// —— 运行期物化（任一叶子动态；与折叠同公式，叶子为 C++ 表达式文本） ——
std::string EmitRuntimeScoreFuncText(const GroupCandidateForm form, const std::vector<MemberScoreLeaves> &members,
                                     const std::vector<std::pair<af::Expression, af::Expression>> &subs,
                                     const bool softmax_bonus, const bool simd_ub_penalty, const int64_t ub_size,
                                     const bool simt_l4_penalty, const bool mixed_minority_penalty) {
  const size_t member_count = members.size();
  const auto leaf_text = [&subs](const af::Expression &leaf) { return RuntimeExprToCpp(leaf, subs); };
  std::ostringstream oss;
  oss << "int32_t CalcScore(const AutofuseTilingData &tiling_data) {\n"
      << "  const auto &t = tiling_data." << ascgen_utils::kGroupTilingDataPlaceholder << ";\n"
      << "  size_t votes = 0UL;\n";
  for (const MemberScoreLeaves &member : members) {
    oss << "  if (" << leaf_text(member.slice_bytes) << " >= " << kEmbeddingSimdPayloadBytesThreshold << "LL && "
        << leaf_text(member.lookup_extent) << " >= " << kEmbeddingSimdLookupCountThreshold << "LL) {\n"
        << "    ++votes;\n"
        << "  }\n";
  }
  oss << "  const size_t member_count = " << member_count << "UL;\n"
      << "  const size_t satisfied = "
      << (form == GroupCandidateForm::kSimd
              ? "votes"
              : (form == GroupCandidateForm::kSimt ? "member_count - votes" : "member_count"))
      << ";\n"
      << "  int32_t score = 0;\n"
      << "  if (satisfied * 3UL >= member_count * 2UL) {\n"
      << "    score = " << kEmbeddingFastPathScore << ";\n"
      << "  } else if (satisfied * 3UL >= member_count) {\n"
      << "    score = " << kEmbeddingAlternateFastPathScore << ";\n"
      << "  }\n";
  if (form == GroupCandidateForm::kSimd || form == GroupCandidateForm::kSimt) {
    oss << "  const int64_t slice_sum = " << leaf_text(members.front().slice_bytes);
    for (size_t index = 1UL; index < member_count; ++index) {
      oss << " + " << leaf_text(members[index].slice_bytes);
    }
    oss << ";\n"
        << "  const int64_t avg_payload = slice_sum / " << static_cast<int64_t>(member_count) << "LL;\n";
    if (form == GroupCandidateForm::kSimd && simd_ub_penalty && member_count > 1UL && ub_size > 0L) {
      oss << "  if (slice_sum * 2LL > " << ub_size << "LL) {\n"
          << "    --score;\n"
          << "  }\n";
    }
  }
  if (simt_l4_penalty) {
    oss << "  --score;\n";
  }
  if (mixed_minority_penalty) {
    oss << "  --score;\n";
  }
  if (softmax_bonus) {
    oss << "  score += 1;\n";
  }
  oss << "  if (score < 0) {\n"
      << "    score = 0;\n"
      << "  }\n";
  if (form == GroupCandidateForm::kSimd || form == GroupCandidateForm::kSimt) {
    oss << "  const int32_t tie_bonus = " << (form == GroupCandidateForm::kSimd ? "avg_payload >= " : "avg_payload < ")
        << kEmbeddingSimdPayloadBytesThreshold << "LL ? 1 : 0;\n"
        << "  OP_LOGI(OP_NAME, \"[ScoreFunc] unified form=" << static_cast<int32_t>(form)
        << " votes=%zu/%zu score=%d tie=%d avg_payload=%lld\", votes, member_count, score, tie_bonus, "
           "static_cast<long long>(avg_payload));\n"
        << "  return score * 4 + tie_bonus;\n";
  } else {
    oss << "  return score * 4;\n";
  }
  oss << "}\n";
  return oss.str();
}

// —— 统一发射入口（capable 形态唯一打分路径；组/单 gather 共用） ——
// 返回值域：final_score = base(0..2) ×4 + tie_bonus(0..1) → {0,1,4,5,8,9}
// （capable 路径）；结构回退路径 base(0..2) ×4 + tie(0..1) 同域。历史值域 {0,1,2}
// 的外部绝对值断言需同步；平局由 tie_bonus 显式裁决，不再依赖候选发射序。
std::string EmitUnifiedScoreFunc(const GroupCandidateForm form, const std::vector<MemberScoreLeaves> &members,
                                 const af::AscGraph &candidate_graph, const bool softmax_bonus,
                                 const bool simd_ub_penalty, const int64_t ub_size, const bool simt_l4_penalty,
                                 const bool mixed_minority_penalty) {
  // 本图原始 sizevar：名集合（叶子可物化判据）+ Subs 替换表（动态 → t.<name>，常量 →
  // 字面量——常量叶子的取值在发射期折入文本，不依赖其 tiling data 字段存在性）。
  SizeVarSet original_var_set;
  AscGraphInfoComplete::AppendOriginalSizeVar(candidate_graph, original_var_set);
  std::set<std::string> var_names;
  std::vector<std::pair<af::Expression, af::Expression>> subs;
  for (const auto &size_var : original_var_set) {
    var_names.insert(std::string(size_var.Str().get()));
    int64_t const_value = 0L;
    if (size_var.GetConstValue(const_value)) {
      subs.emplace_back(size_var, af::Symbol(std::to_string(const_value).c_str()));
    } else {
      subs.emplace_back(size_var,
                        af::Symbol((std::string("static_cast<int64_t>(t.") + size_var.Str().get() + ")").c_str()));
    }
  }
  // 不可物化成员从分子分母同时剔除（N = 可评估成员数，非整候选回退——避免
  // 方向性偏差）；全部不可评估（N==0）时统一给 0 分，由 ATT 比较性能估值。注意：结构代价项
  // （SIMT L4 相位数 / mixed 少数派）按**全组物理成员数** stats.member_count 判定，
  // 不随剔除缩减——L4 的 N+1 相位/barrier ×N 是物理代价（与打分可评估性无关），
  // mixed 少数派是生成期模板选择分裂（同样是物理事实）；打分侧（SIMD UB）的
  // slice_sum 仅累计可评估成员（不可评估成员无运行期表达式可读）。
  std::vector<MemberScoreLeaves> materialized;
  bool all_const = true;
  for (size_t member_index = 0UL; member_index < members.size(); ++member_index) {
    const MemberScoreLeaves &member = members[member_index];
    std::string rejected_token;
    if (!ExprMaterializable(member.slice_bytes, var_names, rejected_token) ||
        !ExprMaterializable(member.lookup_extent, var_names, rejected_token)) {
      GELOGI(
          "[IndirectLoad] Unified score: member[%zu] excluded (token[%s] not in sizevar set or function "
          "allowlist); evaluable N shrinks to %zu.",
          member_index, rejected_token.c_str(), materialized.size());
      continue;
    }
    materialized.emplace_back(member);
    int64_t slice_bytes = 0L;
    int64_t lookup_extent = 0L;
    all_const =
        all_const && member.slice_bytes.GetConstValue(slice_bytes) && member.lookup_extent.GetConstValue(lookup_extent);
  }
  if (materialized.empty()) {
    // N==0：双方 0 分，由 ATT 比较性能估值。
    GELOGI("[ScoreFunc] unified graph[%s] members=%zu evaluable=0 runtime=0", candidate_graph.GetName().c_str(),
           members.size());
    return "int32_t CalcScore(const AutofuseTilingData &tiling_data) {\n"
           "  (void)tiling_data;\n"
           "  return 0;\n"
           "}\n";
  }
  // mixed 的分数不依赖叶子（满足率恒 N、无 UB/平局项）——恒折叠；重启用
  // kEnableMixedGroupCandidate 后经同一入口获得一致打分。
  if (all_const || form == GroupCandidateForm::kMixed) {
    GELOGI("[ScoreFunc] unified graph[%s] members=%zu evaluable=%zu runtime=0 form=%d",
           candidate_graph.GetName().c_str(), members.size(), materialized.size(), static_cast<int32_t>(form));
    return FoldScoreFuncText(form, materialized, softmax_bonus, simd_ub_penalty, ub_size, simt_l4_penalty,
                             mixed_minority_penalty);
  }
  GELOGI("[ScoreFunc] unified graph[%s] members=%zu evaluable=%zu runtime=1 form=%d", candidate_graph.GetName().c_str(),
         members.size(), materialized.size(), static_cast<int32_t>(form));
  return EmitRuntimeScoreFuncText(form, materialized, subs, softmax_bonus, simd_ub_penalty, ub_size, simt_l4_penalty,
                                  mixed_minority_penalty);
}

// —— 结构回退（能力门控不满足——图结构属性，与 shape 无关） ——
// strided 让位 / 0 分 + Softmax 加分 + ×4 与平局加分（静态按平均 payload、动态
// SIMD +1——历史非 capable 行为；组/单 gather 同口径）。
// 结构回退（能力门控不满足——图结构属性，与 shape 无关）。
// [动静态对齐] 原实现：静态按 avg_payload 给平局分、动态固定 SIMD +1，同一张非 embedding 图
// 会得到相反倾向。现在两条分支统一为「运行期算 avg_payload」，与静态口径完全一致；
// 另外 strided 让位（simd_lowers_to_strided）时 SIMD 不再拿平局加分，确保 SIMT 严格胜出。
// 结构回退平局的有效 payload（静态可折算路径）：尾轴窗口形态且窗口可整驻 UB（含
// 预留）、lookup 密度按 cache line 折算覆盖窗口字节时，取整行窗口字节（SIMD dense
// facade 的真实 staging 单位）；否则取 input_slice_bytes（含尾轴退化的 dtype 字节，
// 保持既有行为）。动态窗口叶交由运行期发射的同式条件表达式判定。
bool TryGetMemberTiePayload(const MemberScoreLeaves &member, const bool use_window_payload, const int64_t ub_size,
                            int64_t &tie_payload) {
  int64_t slice_bytes = 0L;
  if (!member.slice_bytes.GetConstValue(slice_bytes)) {
    return false;
  }
  tie_payload = slice_bytes;
  if (!use_window_payload || !member.has_tail_window) {
    return true;
  }
  int64_t window_bytes = 0L;
  int64_t lookup_count = 0L;
  if (member.window_bytes.GetConstValue(window_bytes) && window_bytes > 0L &&
      member.window_lookup_count.GetConstValue(lookup_count) && lookup_count > 0L &&
      window_bytes + kIndirectLoadUbReservedBytes <= ub_size && lookup_count * kDenseWindowLineBytes >= window_bytes) {
    tie_payload = window_bytes;
  }
  return true;
}

// —— 结构回退（能力门控不满足——图结构属性，与 shape 无关） ——
// strided 让位 / 0 分 + Softmax 加分 + ×4 与平局加分（静态按平均 payload、动态
// SIMD +1——历史非 capable 行为；组/单 gather 同口径）。
// 结构回退（能力门控不满足——图结构属性，与 shape 无关）。
// [动静态对齐] 原实现：静态按 avg_payload 给平局分、动态固定 SIMD +1，同一张非 embedding 图
// 会得到相反倾向。现在两条分支统一为「运行期算 avg_payload」，与静态口径完全一致；
// 另外 strided 让位（simd_lowers_to_strided）时 SIMD 不再拿平局加分，确保 SIMT 严格胜出。
// [尾轴窗口口径] 尾轴 gather 的 input_slice_bytes 恒为 dtype 字节（无 payload 后缀），
// 平局的「staging 单位 ≥ 2KB 偏 SIMD」判据在该形态退化为恒偏 SIMT。窗口口径把该形态
// 的有效 payload 修正为整行窗口字节（SIMD dense facade 的真实 staging 单位），启用条件：
// SIMD 未退化 strided（strided 逐元素组装无窗口经济性，且让位契约要求 SIMT 严格胜出时
// 两侧口径维持 input_slice_bytes）+ 平台 UB 可得 + 窗口可整驻 UB + lookup 密度折算
// 覆盖窗口字节。静态/动态同式（动态叶运行期判定），非尾轴成员行为不变。
std::string EmitStructuralFallbackScoreFunc(const GroupCandidateForm form, const bool simd_lowers_to_strided,
                                            const bool softmax_bonus, const std::vector<MemberScoreLeaves> &members,
                                            const af::AscGraph &candidate_graph, const int64_t ub_size) {
  const bool is_simd = form == GroupCandidateForm::kSimd;
  const bool is_simt = form == GroupCandidateForm::kSimt;
  int32_t base_score = (simd_lowers_to_strided && is_simt) ? kSimtStridedFallbackScore : 0;
  if (softmax_bonus) {
    base_score += 1;
  }
  const auto emit_constant = [base_score](int32_t tie_bonus) {
    return "int32_t CalcScore(const AutofuseTilingData &tiling_data) {\n"
           "  (void)tiling_data;\n"
           "  return " +
           std::to_string(base_score * 4 + tie_bonus) +
           ";\n"
           "}\n";
  };
  if (!is_simd && !is_simt) {
    return emit_constant(0);  // SK：无平局项
  }
  // SIMD 让位：strided 降级形态下 SIMD 不参与平局加分（与静态同口径）
  const bool simd_yields = simd_lowers_to_strided && is_simd;
  // 尾轴窗口口径启用条件（见函数头注释）：让位或平台 UB 缺失时整体停用，两侧平局
  // 口径与历史行为完全一致。
  const bool use_window_payload = !simd_lowers_to_strided && ub_size > 0L &&
                                  std::any_of(members.begin(), members.end(),
                                              [](const MemberScoreLeaves &member) { return member.has_tail_window; });

  // 结构性常量判定：只要叶子里的自由符号全是常量，就按常量折叠。
  // 不用"整棵表达式能否 GetConstValue"——const sizevar 参与的乘法不一定折叠，
  // 那会把静态图误判成动态，发射出"没有任何 t. 读取"的伪运行期函数。
  // 窗口口径启用时，尾轴成员的窗口两叶同样参与动态性判定（窗口字节来自
  // sizes[axis]*strides[axis]，动态 shape 下为符号叶）。
  int64_t slice_bytes_sum = 0L;
  bool has_dynamic_leaf = false;
  bool sum_known = true;
  for (const MemberScoreLeaves &member : members) {
    std::vector<af::Expression> tie_leaves{member.slice_bytes, member.lookup_extent};
    if (use_window_payload && member.has_tail_window) {
      tie_leaves.emplace_back(member.window_bytes);
      tie_leaves.emplace_back(member.window_lookup_count);
    }
    for (const af::Expression &expr : tie_leaves) {
      for (const auto &symbol : expr.FreeSymbols()) {
        int64_t symbol_value = 0L;
        if (!symbol.GetConstValue(symbol_value)) {
          has_dynamic_leaf = true;
        }
      }
    }
    int64_t tie_payload = 0L;
    if (!TryGetMemberTiePayload(member, use_window_payload, ub_size, tie_payload)) {
      sum_known = false;
      continue;
    }
    slice_bytes_sum += tie_payload;
  }
  const bool all_const = !has_dynamic_leaf;

  // 运行期形态：全部成员可物化且存在动态叶子（与 capable 路径同一套替换规则）
  std::vector<MemberScoreLeaves> materialized;
  std::vector<std::pair<af::Expression, af::Expression>> subs;
  if (!all_const && !members.empty()) {
    SizeVarSet original_var_set;
    AscGraphInfoComplete::AppendOriginalSizeVar(candidate_graph, original_var_set);
    std::set<std::string> var_names;
    for (const auto &size_var : original_var_set) {
      var_names.insert(std::string(size_var.Str().get()));
      int64_t const_value = 0L;
      if (size_var.GetConstValue(const_value)) {
        subs.emplace_back(size_var, af::Symbol(std::to_string(const_value).c_str()));
      } else {
        subs.emplace_back(size_var,
                          af::Symbol((std::string("static_cast<int64_t>(t.") + size_var.Str().get() + ")").c_str()));
      }
    }
    for (const MemberScoreLeaves &member : members) {
      std::string rejected_token;
      if (!ExprMaterializable(member.slice_bytes, var_names, rejected_token) ||
          (use_window_payload && member.has_tail_window &&
           (!ExprMaterializable(member.window_bytes, var_names, rejected_token) ||
            !ExprMaterializable(member.window_lookup_count, var_names, rejected_token)))) {
        GELOGI("[ScoreFunc] fallback member excluded (token[%s]); use constant tie rule.", rejected_token.c_str());
        materialized.clear();
        break;
      }
      materialized.emplace_back(member);
    }
    if (materialized.size() != members.size()) {
      materialized.clear();
    }
  }

  if (materialized.empty()) {
    // 常量折叠（静态图 / 含不可物化成员）。
    // 注意：动态图若物化失败，slice_bytes_sum 是"部分累加"，不可用来算 avg_payload——
    // 否则 avg 恒 0，会把不可物化的动态图静默推向 SIMT。此类情形退回历史规则（SIMD +1）。
    int32_t tie_bonus = 0;
    if (!simd_yields) {
      if (all_const && sum_known) {
        const int64_t avg_payload = members.empty() ? 0L : slice_bytes_sum / static_cast<int64_t>(members.size());
        tie_bonus = is_simd ? (avg_payload >= kEmbeddingSimdPayloadBytesThreshold ? 1 : 0)
                            : (avg_payload < kEmbeddingSimdPayloadBytesThreshold ? 1 : 0);
      } else {
        tie_bonus = is_simd ? 1 : 0;
      }
    }
    return emit_constant(tie_bonus);
  }

  std::ostringstream oss;
  oss << "int32_t CalcScore(const AutofuseTilingData &tiling_data) {\n"
      << "  const auto &t = tiling_data." << ascgen_utils::kGroupTilingDataPlaceholder << ";\n";
  if (use_window_payload) {
    // 尾轴窗口成员的有效 payload：与静态折叠（TryGetMemberTiePayload）同式的运行期
    // 条件——窗口整驻 UB（含预留）且 lookup 密度折算覆盖窗口字节时取窗口字节，
    // 否则回退 input_slice_bytes；非尾轴成员直接取 input_slice_bytes。
    for (size_t index = 0UL; index < materialized.size(); ++index) {
      const MemberScoreLeaves &member = materialized[index];
      if (member.has_tail_window) {
        const std::string window_text = RuntimeExprToCpp(member.window_bytes, subs);
        oss << "  const int64_t tie_payload_" << index << " = (" << window_text << " + " << kIndirectLoadUbReservedBytes
            << "LL <= " << ub_size << "LL && " << RuntimeExprToCpp(member.window_lookup_count, subs) << " * "
            << kDenseWindowLineBytes << "LL >= " << window_text << ") ? " << window_text << " : "
            << RuntimeExprToCpp(member.slice_bytes, subs) << ";\n";
      } else {
        oss << "  const int64_t tie_payload_" << index << " = " << RuntimeExprToCpp(member.slice_bytes, subs) << ";\n";
      }
    }
    oss << "  const int64_t slice_sum = tie_payload_0";
    for (size_t index = 1UL; index < materialized.size(); ++index) {
      oss << " + tie_payload_" << index;
    }
  } else {
    oss << "  const int64_t slice_sum = " << RuntimeExprToCpp(materialized[0].slice_bytes, subs);
    for (size_t index = 1UL; index < materialized.size(); ++index) {
      oss << " + " << RuntimeExprToCpp(materialized[index].slice_bytes, subs);
    }
  }
  oss << ";\n"
      << "  const int64_t avg_payload = slice_sum / " << static_cast<int64_t>(materialized.size()) << "LL;\n"
      << "  const int32_t base_score = " << base_score << ";\n";
  if (simd_yields) {
    oss << "  const int32_t tie_bonus = 0;\n";
  } else if (is_simd) {
    oss << "  const int32_t tie_bonus = avg_payload >= " << kEmbeddingSimdPayloadBytesThreshold << "LL ? 1 : 0;\n";
  } else {
    oss << "  const int32_t tie_bonus = avg_payload < " << kEmbeddingSimdPayloadBytesThreshold << "LL ? 1 : 0;\n";
  }
  oss << "  OP_LOGI(OP_NAME, \"[ScoreFunc] fallback form=" << static_cast<int32_t>(form)
      << " slice_sum=%lld avg_payload=%lld base=%d tie=%d\", static_cast<long long>(slice_sum), "
         "static_cast<long long>(avg_payload), base_score, tie_bonus);\n"
      << "  return base_score * 4 + tie_bonus;\n"
      << "}\n";
  GELOGI("[ScoreFunc] fallback graph[%s] runtime=1 form=%d members=%zu base=%d simd_yields=%d use_window=%d",
         candidate_graph.GetName().c_str(), static_cast<int32_t>(form), materialized.size(), base_score,
         static_cast<int32_t>(simd_yields), static_cast<int32_t>(use_window_payload));
  return oss.str();
}

// 组候选（SIMD/SIMT/混模板/SK）统一打分入口。能力门控（访问形态维度，结构属性、与
// shape 无关）：SIMD → AND(can_use_simd_embedding)；SIMT/MIXED → AND(embedding_like)。
// can_use_simd_embedding ⟹ embedding_like，探测保证 SIMD 侧偏好选择仅发生于
// can_use_simd_embedding 成员——满足率 N/N 的 2 分仅授予 embedding 性能分裂；dense 组
// 无偏好依据，0 分参与 ATT 性能比较。结构合法性由探测逐成员校验，此处不二次门控。
std::string GenerateGroupScoreFunc(const GroupCandidateForm form, const TemplateCase &template_case,
                                   const GroupAccessStats &stats, const af::AscNodePtr &candidate_node,
                                   const af::AscGraph &candidate_graph, const bool simt_unfused,
                                   const size_t mixed_simd_count, const bool simd_lowers_to_strided) {
  const bool fast_path_eligible = template_case.implementation == ascgen_utils::indirect_load::Implementation::kDefault;
  const bool capable = fast_path_eligible && (form == GroupCandidateForm::kSimd    ? stats.simd_embedding_capable
                                              : form == GroupCandidateForm::kSimt  ? stats.simt_structured_capable
                                              : form == GroupCandidateForm::kMixed ? stats.simt_structured_capable
                                                                                   : false);
  // [ScoreFunc 诊断] capable=0 → 结构回退（常量打分），与 shape 无关；all_static=0 但 capable=1
  // 才是真正的运行期打分形态。
  GELOGI("[ScoreFunc] group primary[%s] form=%d capable=%d all_static=%d members=%zu",
         candidate_node != nullptr ? candidate_node->GetNamePtr() : "unknown", static_cast<int32_t>(form),
         static_cast<int32_t>(capable), static_cast<int32_t>(stats.all_static), stats.member_count);
  bool softmax_bonus = false;
  if (candidate_node != nullptr) {
    ascgen_utils::norm::NormInfo norm_info;
    GE_ASSERT_SUCCESS(ascgen_utils::norm::TryGetNormInfo(candidate_node, norm_info));
    softmax_bonus = norm_info.kind == ascgen_utils::norm::NormInfo::Kind::kSoftmaxDedicated;
  }
  if (capable) {
    // [统一打分] capable 形态唯一路径：静态折叠 / 动态运行期同算法（见区块头注释）。
    // mixed 预埋：kEnableMixedGroupCandidate 关闭期不生成候选，重启用后经同一入口
    // 获得一致打分（满足率恒 N、无叶子依赖，恒折叠为常量）。
    const bool mixed_minority_penalty =
        form == GroupCandidateForm::kMixed && stats.member_count > 0UL &&
        std::min(mixed_simd_count, stats.member_count - mixed_simd_count) * 3UL < stats.member_count;
    return EmitUnifiedScoreFunc(form, stats.members, candidate_graph, softmax_bonus, true, stats.ub_size,
                                form == GroupCandidateForm::kSimt && simt_unfused && stats.member_count > 2UL,
                                mixed_minority_penalty);
  }
  // 能力门控不满足：strided 让位 / 0 分的常量回退（组 SIMD 候选退化到逐元素 strided
  // facade 时偏好 SIMT——strided 的逐元素标量组装 offsets + 逐元素重排劣于 SIMT 标量
  // 直读；kSimtStridedFallbackScore 让位语义）。
  return EmitStructuralFallbackScoreFunc(form, simd_lowers_to_strided, softmax_bonus, stats.members, candidate_graph,
                                         stats.ub_size);
}

// [单 gather 统一打分] 与组路径同一算法（N=1：vote → SIMD 满足 = votes / SIMT 满足
// = 1−votes → 三档 → ×4+平局）；静态折叠 / 动态运行期同形，组绑定同占位符。能力
// 门控（IsEmbeddingFastPathCapable）与结构回退（strided 让位）保持既有语义；回退
// 分统一到 ×4+平局尺度（单 gather 候选仅在同任务内竞标，尺度内部一致）。
std::string GenerateScoreFunc(const TemplateCase &template_case,
                              const ascgen_utils::indirect_load::IndirectLoadAccessInfo &access_info,
                              const af::AscNodePtr &candidate_node, const af::AscGraph &candidate_graph,
                              const bool simd_lowers_to_strided) {
  const GroupCandidateForm form =
      template_case.template_id == ascir::TemplateId::kIndirectLoadSimd   ? GroupCandidateForm::kSimd
      : template_case.template_id == ascir::TemplateId::kIndirectLoadSimt ? GroupCandidateForm::kSimt
                                                                          : GroupCandidateForm::kSk;
  const bool capable = IsEmbeddingFastPathCapable(template_case, access_info);
  // [ScoreFunc 诊断] capable=0 → 走结构回退（常量），动态 shape 下分数与 shape 无关；
  // 排查"恒 0"时必须先看这个标志，而不是看分数。
  GELOGI("[ScoreFunc] single gather node[%s] template=%d capable=%d",
         candidate_node != nullptr ? candidate_node->GetNamePtr() : "unknown",
         static_cast<int32_t>(template_case.template_id), static_cast<int32_t>(capable));
  bool softmax_bonus = false;
  if (candidate_node != nullptr) {
    ascgen_utils::norm::NormInfo norm_info;
    GE_ASSERT_SUCCESS(ascgen_utils::norm::TryGetNormInfo(candidate_node, norm_info));
    softmax_bonus = norm_info.kind == ascgen_utils::norm::NormInfo::Kind::kSoftmaxDedicated;
  }
  if (capable) {
    const std::vector<MemberScoreLeaves> members{MakeMemberScoreLeaves(access_info)};
    return EmitUnifiedScoreFunc(form, members, candidate_graph, softmax_bonus, false, 0L, false, false);
  }
  int64_t ub_size = 0L;
  (void)TryGetIndirectLoadPlatformUbSize(ub_size);
  const std::vector<MemberScoreLeaves> members{MakeMemberScoreLeaves(access_info)};
  return EmitStructuralFallbackScoreFunc(form, simd_lowers_to_strided, softmax_bonus, members, candidate_graph,
                                         ub_size);
}

// ==================== 调度期组候选 UB 可行性预估（硬下界，保守放行） ====================
// 与单 gather 的 IsSkWindowUbInfeasible/IsSimtUbInfeasible 同源思路：只估"再怎么缩
// tile 也省不掉的硬下界"；平台信息不可得或任一成员 shape 动态时保守放行（并存竞标下
// 漏放行的后果是 tiling 期资源否决后由 SK 兜底候选接管，误放行的后果是白白损失融合
// 机会）。定位为编译期优化（跳过资源下界超限候选的生成与求解），非正确性必需。

}  // namespace indirect_load_detail
}  // namespace optimize
