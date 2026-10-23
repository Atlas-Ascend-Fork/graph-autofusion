/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "brc_inline_template.h"
#include "common_utils.h"
#include <sstream>
#include <string>
#include <queue>

namespace optimize {
namespace semantic = ascgen_utils::template_semantic;

std::string BrcInlineTemplate::GenName(const std::string &general_case_name) {
  return general_case_name + "_inline";
}

bool BrcInlineTemplate::IsNodeSupportBrcInline(const af::NodePtr &node) {
  for (const auto &brc_out_node : node->GetOutDataNodes()) {
    const auto &out_node = std::dynamic_pointer_cast<af::AscNode>(brc_out_node);
    GE_ASSERT_NOTNULL(out_node);
    if (!ascgen_utils::IsNodeSupportsBrcInline(out_node)) {
      GELOGD("[%s][%s] is not in the supported list.", brc_out_node->GetTypePtr(), brc_out_node->GetNamePtr());
      return false;
    }
    std::unique_ptr<af::AscTensor> input0;
    std::unique_ptr<af::AscTensor> input1;
    GE_WARN_ASSERT(ScheduleUtils::GetNonBrcInputTensor(out_node, 0UL, input0) == af::SUCCESS);
    GE_WARN_ASSERT(ScheduleUtils::GetNonBrcInputTensor(out_node, 1UL, input1) == af::SUCCESS);
    std::vector<uint8_t> input_idx_2_brc_inline;
    if (!ascgen_utils::IsGeneralizeBrcInlineScene(out_node, *input0, *input1, input_idx_2_brc_inline)) {
      GELOGD("[%s][%s] does not support brc inline.", out_node->GetTypePtr(), out_node->GetNamePtr());
      return false;
    }
    for (size_t i = 0UL; i < input_idx_2_brc_inline.size(); ++i) {
      if (input_idx_2_brc_inline.at(i) == 1 && brc_out_node->GetInDataNodes().at(i) != node) {
        GELOGD("[%s][%s] supports brc inline, but input[%zu] is another brc node.", out_node->GetTypePtr(),
               out_node->GetNamePtr(), i);
        return false;
      }
    }
  }
  return true;
}

/**
 * 在optimized graph基础上，查找满足brc inline的节点并优化，具体逻辑如下：
 * 1. 遍历所有节点，如果遇到brc节点，则逐个检查brc与其输出节点是否满足brc inline
 * 2. 若满足brc inline，则将本brc节点删掉，计数+1。继续查找其他brc节点
 * 3. 最后，若计数>0，返回成功；否则返回失败。
 */
af::Status BrcInlineTemplate::Generate(const af::AscGraph &origin_graph, const af::AscGraph &based_case,
                                       af::AscGraph &new_case) {
  (void)origin_graph;
  (void)based_case;
  int32_t brc_inlined_count = 0;
  for (const auto &node : new_case.GetAllNodes()) {
    GE_WARN_ASSERT(!ScheduleUtils::IsReduce(node), "Brc inline does not support Reduce(%s) now.", node->GetNamePtr());
    if (!af::ops::IsOps<af::ascir_op::Broadcast>(node)) {
      continue;
    }
    GE_ASSERT_TRUE(node->GetOutDataNodesSize() > 0U);
    if (IsNodeSupportBrcInline(node)) {
      GELOGD("Graph[%s] find brc inline node: %s", new_case.GetName().c_str(), node->GetNamePtr());
      for (const auto &out_node : node->GetOutDataNodes()) {
        const auto consumer_node = std::dynamic_pointer_cast<af::AscNode>(out_node);
        if (consumer_node != nullptr) {
          GE_ASSERT_SUCCESS(ascgen_utils::SetTemplateSemanticRole(consumer_node, semantic::kInline));
        }
      }
      brc_inlined_count++;
      const auto in_data_anchor = node->GetInDataAnchor(0);
      GE_CHECK_NOTNULL(in_data_anchor);
      GE_ASSERT_SUCCESS(ScheduleUtils::RemoveNode(new_case, node, in_data_anchor->GetPeerOutAnchor()));
    }
  }
  return brc_inlined_count == 0 ? af::FAILED : af::SUCCESS;
}

bool BrcInlineTemplate::NeedDropBasedCase(const af::AscGraph &origin_graph, const af::AscGraph &based_case,
                                          const af::AscGraph &new_case) {
  (void)based_case;
  (void)new_case;
  return ScheduleUtils::IsStaticGraph(origin_graph);
}

}  // namespace optimize
