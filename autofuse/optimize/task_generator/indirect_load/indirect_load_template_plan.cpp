/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_generator/indirect_load/indirect_load_template_plan.h"
#include "task_generator/indirect_load/indirect_load_graph_utils.h"
#include "task_generator/indirect_load/indirect_load_view_layout.h"
#include "task_generator/indirect_load/indirect_load_input_path.h"
#include "task_generator/indirect_load/indirect_load_output_analysis.h"
#include "task_generator/indirect_load/indirect_load_axis_normalize.h"
#include "task_generator/indirect_load/indirect_load_score_generator.h"
#include "task_generator/indirect_load/indirect_load_ub_feasibility.h"
#include "task_generator/indirect_load/indirect_load_sk_partition.h"
#include "task_generator/indirect_load/indirect_load_simt_emission.h"
#include "task_generator/indirect_load/indirect_load_group_candidate.h"
#include "task_generator/indirect_load/indirect_load_template_annotation.h"

#include "v35/ascir/ascir_codegen_v2.h"
#include "ascgraph_info_complete.h"
#include "schedule_result.h"

namespace optimize {
namespace indirect_load_detail {

bool HasBroadcastMultiInputNode(const af::AscNodePtr &node, NodeSet &visited) {
  if (node == nullptr || !visited.emplace(node.get()).second) {
    return false;
  }
  if (node->inputs.Size() > 1UL) {
    for (size_t input_idx = 0UL; input_idx < node->inputs.Size(); ++input_idx) {
      const auto producer = ascgen_utils::indirect_load::GetInputProducer(node, input_idx);
      if (producer != nullptr && af::ops::IsOps<af::ascir_op::Broadcast>(producer)) {
        return true;
      }
    }
  }
  for (size_t input_idx = 0UL; input_idx < node->inputs.Size(); ++input_idx) {
    if (HasBroadcastMultiInputNode(ascgen_utils::indirect_load::GetInputProducer(node, input_idx), visited)) {
      return true;
    }
  }
  return false;
}

bool IsSkTemplateCandidateLegal(const af::AscNodePtr &indirect_load) {
  // SK 仅支持单输出消费者：多消费者分区（为多 Reduce 复合区域引入）
  // 已随 LayerNorm SK 支持一同回退，待后续补齐后再放开。
  if (indirect_load == nullptr || indirect_load->GetOutDataNodesSize() != 1UL) {
    GELOGI("[IndirectLoad] Reject SK: IndirectLoad output consumer count is not 1.");
    return false;
  }
  for (size_t input_idx = 0UL; input_idx < kIndirectLoadInputCount; ++input_idx) {
    const auto input_anchor = indirect_load->GetInDataAnchor(input_idx);
    if (input_anchor == nullptr || input_anchor->GetPeerOutAnchor() == nullptr) {
      GELOGI("[IndirectLoad] Reject SK: input[%zu] has no connected producer.", input_idx);
      return false;
    }
  }
  NodeSet visited;
  for (size_t input_idx = 0UL; input_idx < kIndirectLoadInputCount; ++input_idx) {
    if (HasBroadcastMultiInputNode(ascgen_utils::indirect_load::GetInputProducer(indirect_load, input_idx), visited)) {
      GELOGI("[IndirectLoad] Reject SK: input[%zu] contains multi-input Broadcast dependency.", input_idx);
      return false;
    }
  }
  const auto output_consumer = ascgen_utils::indirect_load::GetOnlyOutputConsumer(indirect_load);
  if (HasBroadcastMultiInputNode(output_consumer, visited)) {
    return false;
  }
  return true;
}

af::Status ValidateIndirectLoadNode(const af::AscNodePtr &indirect_load) {
  const auto outputs = indirect_load->outputs();
  GE_ASSERT_TRUE(outputs.size() == kIndirectLoadOutputCount, "Invalid IndirectLoad output number:%zu, node[%s].",
                 outputs.size(), indirect_load->GetNamePtr());
  const auto output = outputs[0];
  GE_ASSERT_NOTNULL(output, "IndirectLoad output tensor is null.");
  const size_t output_rank = output->attr.axis.size();
  // 注意：AscNodeInputs/AscNodeOutputs 的 operator() 每次调用都会重建内部快照，先前调用
  // 返回的 tensor 指针会悬垂。因此这里只调用一次 outputs()，并提前拷贝后续要用的数据。
  const auto output_repeats = output->attr.repeats;
  const size_t axis_index = GetIndirectLoadAxisIndex(indirect_load);
  GE_ASSERT_TRUE(axis_index != kIndirectLoadInvalidAxisIndex, "IndirectLoad axis index of node[%s] is invalid.",
                 indirect_load->GetNamePtr());
  const auto inputs = indirect_load->inputs();
  GE_ASSERT_TRUE(inputs.size() == kIndirectLoadInputCount, "Invalid IndirectLoad input number:%zu, node[%s].",
                 inputs.size(), indirect_load->GetNamePtr());
  const auto input = inputs[ascgen_utils::indirect_load::kInputTensorIndex];
  const auto index = inputs[ascgen_utils::indirect_load::kIndexTensorIndex];
  GE_ASSERT_NOTNULL(input, "IndirectLoad input tensor is null.");
  GE_ASSERT_NOTNULL(index, "IndirectLoad index tensor is null.");
  GE_ASSERT_TRUE(input->attr.repeats.size() == output_rank && index->attr.repeats.size() == output_rank &&
                     output_repeats.size() == output_rank,
                 "IndirectLoad logical shape rank is invalid.");
  for (size_t i = 0UL; i < output_rank; ++i) {
    GE_ASSERT_TRUE(af::SymbolicUtils::StaticCheckEq(index->attr.repeats[i], output_repeats[i]) == af::TriBool::kTrue,
                   "IndirectLoad index and output logical shape must match.");
    if (i != axis_index) {
      GE_ASSERT_TRUE(af::SymbolicUtils::StaticCheckLt(input->attr.repeats[i], output_repeats[i]) != af::TriBool::kTrue,
                     "IndirectLoad input dimension %zu must not be smaller than index/output outside axis %zu.", i,
                     axis_index);
    }
  }
  return af::SUCCESS;
}

// [护栏] IL 输出 vectorized 轴有效性前置校验：为空，或全部是其当前视图 attr.axis 的
// 子序列。合法形态（如带 post-Reduce 的单 gather SIMD/SIMT）IL 输出 vectorized 可
// 非空（inner 轴族）；但轴必须仍存在于视图——生成期任何路径把已失效（被 tensor
// merge 折叠掉）或外来轴写进 vectorized，调度期会在 GetVectorRepeats/
// InitTensorMemInfo 或 SelectLoopAxis 以无关断言隐蔽失败。候选产出后即刻校验，
// 把违约暴露在生成侧。
af::Status ValidateIndirectLoadOutputVectorizedAxes(const ascir::ImplGraph &graph) {
  for (const af::AscNodePtr &node : ascgen_utils::indirect_load::GetAllIndirectLoadNodes(graph)) {
    for (const auto &output : node->outputs()) {
      if (output == nullptr || output->attr.vectorized_axis.empty()) {
        continue;
      }
      for (const auto vec_axis : output->attr.vectorized_axis) {
        GE_ASSERT_TRUE(
            std::find(output->attr.axis.begin(), output->attr.axis.end(), vec_axis) != output->attr.axis.end(),
            "IndirectLoad node[%s] output vectorized axis[%ld] is absent from its view.", node->GetNamePtr(), vec_axis);
      }
    }
  }
  return af::SUCCESS;
}

// SIMD 多阶段布局校验：区域内每个统计 Reduce 独立通过
// ValidateSimdPostReduceLayout（G≥R + 后缀连续），首个失败即淘汰候选。
af::Status ValidateSimdCompositeLayouts(const af::AscNodePtr &indirect_load,
                                        const std::vector<af::AscNodePtr> &composite_reduces,
                                        bool &is_candidate_legal) {
  is_candidate_legal = true;
  for (const auto &reduce : composite_reduces) {
    bool reduce_legal = true;
    GE_ASSERT_SUCCESS(ValidateSimdPostReduceLayout(indirect_load, reduce, reduce_legal));
    if (!reduce_legal) {
      is_candidate_legal = false;
      GELOGI("[IndirectLoad] Reject SIMD composite candidate: Reduce[%s] layout is invalid.", reduce->GetNamePtr());
      return af::SUCCESS;
    }
  }
  return af::SUCCESS;
}

af::Status ValidateTemplate(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                            const RewrittenGraphAnalysis &analysis, size_t &boundary, bool &is_candidate_legal) {
  // Softmax 专用节点保持完整输入输出形状，普通 Reduce 的 stride 推导（BuildPostReduceLayout）
  // 无法识别其内部归约轴，必须先分流再做布局校验：
  // - 尾轴约束满足 → 走专用 API，仅校验 View 与 IndirectLoad 输出一致；
  // - 尾轴约束不满足 → Pattern 已在主图替换、无法回退原始结构，淘汰候选。
  const bool is_softmax_post =
      analysis.post_reduce != nullptr && af::ops::IsOps<af::ascir_op::Softmax>(analysis.post_reduce);
  if (is_softmax_post) {
    GE_ASSERT_SUCCESS(CollectGatherNormInfo(indirect_load, analysis, is_candidate_legal, template_id));
    if (is_candidate_legal) {
      // Softmax 专用路径：R 轴是 Softmax 输入的尾轴，必须完整保留在 tile 内。
      // 将 boundary 覆写为尾轴前一位（与 composite 路径的 first_reduce 语义对齐）：
      // outer=[保留轴]（可切、tile 行数可求解），inner=[尾轴]（SoftmaxAR 的 R 参数，
      // 由 vectorized 视图最后一个轴推导）。否则 boundary 保持输出 rank，outer 会
      // merge 全部轴，vectorized 退化为 size=1 的 tile 轴，SoftmaxAR 的 R 恒为 1。
      const auto softmax_inputs = analysis.post_reduce->inputs();
      GE_ASSERT_TRUE(!softmax_inputs.empty() && softmax_inputs[0] != nullptr,
                     "IndirectLoad Softmax node[%s] has no input.", analysis.post_reduce->GetNamePtr());
      const size_t softmax_input_rank = softmax_inputs[0]->attr.axis.size();
      GE_ASSERT_TRUE(softmax_input_rank > 0UL, "IndirectLoad Softmax input rank is invalid, node[%s].",
                     analysis.post_reduce->GetNamePtr());
      boundary = softmax_input_rank - 1UL;
    }
    return af::SUCCESS;
  }
  // SIMD 多阶段：区域内每个 Reduce 独立布局校验后，整链保留在候选图中，
  // 由连通性分区保持同组并复用既有 Reduce/Broadcast/Elementwise Codegen。
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    if (!analysis.composite_reduces.empty()) {
      GE_ASSERT_SUCCESS(ValidateSimdCompositeLayouts(indirect_load, analysis.composite_reduces, is_candidate_legal));
    } else {
      GE_ASSERT_SUCCESS(ValidateSimdPostReduceLayout(indirect_load, analysis.post_reduce, is_candidate_legal));
    }
  } else {
    // FindPostReduceChain 保留首个 Reduce 作为 SIMT local target 锚点，并继续
    // 遍历其后的 Broadcast/Elementwise/Reduce；后续统计阶段由普通调度链执行，
    // 因而串行和共享输入两类复合区域均可复用现有 Codegen。
    GE_ASSERT_SUCCESS(ValidateSimtPostReduceLayout(analysis.post_reduce, boundary, is_candidate_legal));
    if (is_candidate_legal) {
      GE_ASSERT_SUCCESS(ValidateSimtTemplateRegion(analysis, is_candidate_legal));
    }
  }
  if (!is_candidate_legal) {
    return af::SUCCESS;
  }
  // 通用路径：保存 NormInfo 供后续多阶段扩展使用。
  GE_ASSERT_SUCCESS(CollectGatherNormInfo(indirect_load, analysis, is_candidate_legal, template_id));
  return af::SUCCESS;
}

af::Status AnnotateTemplate(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id,
                            const RewrittenGraphAnalysis &analysis) {
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    return AnnotateSimdTemplateRoles(analysis);
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    return AnnotateSimtTemplateRoles(indirect_load, analysis);
  }
  GE_ASSERT_TRUE(false, "IndirectLoad template id %d is invalid.", static_cast<int32_t>(template_id));
}

af::Status FinalizeTemplate(const af::AscNodePtr &indirect_load, ascir::TemplateId template_id) {
  if (template_id == ascir::TemplateId::kIndirectLoadSimd) {
    return ::ascir::SetTemplateId(indirect_load, template_id);
  }
  if (template_id == ascir::TemplateId::kIndirectLoadSimt) {
    GE_ASSERT_SUCCESS(::ascir::SetDcacheSize(indirect_load, kIndirectLoadSimtDcacheSize));
    return ::ascir::SetTemplateId(indirect_load, template_id);
  }
  GE_ASSERT_TRUE(false, "IndirectLoad template id %d is invalid.", static_cast<int32_t>(template_id));
}

}  // namespace indirect_load_detail
}  // namespace optimize
