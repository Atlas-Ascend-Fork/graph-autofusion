/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_PLATFORM_V2_UN_ALIGNMENT_STRATEGY_H
#define OPTIMIZE_PLATFORM_V2_UN_ALIGNMENT_STRATEGY_H

#include "platform/common/base_alignment_strategy.h"
namespace optimize {
class UnAlignmentStrategy : public BaseAlignmentStrategy {
 public:
  ~UnAlignmentStrategy() override = default;
  UnAlignmentStrategy() = default;

  af::Status BackPropagateAlignment(const af::AscNodePtr &node, AlignmentType aligned_type) override;
  af::Status ModifyVectorizedStrides(ascir::ImplGraph &impl_graph) override;
  static af::Status ModifyTransposeFusionVectorizedStrides(af::AscGraph &graph, uint32_t align_width);

 protected:
  af::Status LoadAlignmentInferFunc(const af::AscNodePtr &node) override;
  af::Status StoreAlignmentInferFunc(const af::AscNodePtr &node) override;
  af::Status ConcatAlignmentInferFunc(const af::AscNodePtr &node) override;

 private:
  AlignmentType GetDefaultAlignmentType() override;
  af::Status SetAlignInfoForTailBrcNodes(AlignmentType aligned_type, af::AscNode *node,
                                         std::set<af::Node *> &visited_nodes, std::queue<af::Node *> &node_queue);
  static af::Status GetCurrentNodeContinuousTailAxisNum(const af::AscNodePtr &node, uint32_t &continuous_tail_axis_num,
                                                        uint32_t &discontinuous_axis_num);
  static af::Status GetNodeContinuousTailAxisNumByStore(const af::AscNodePtr &node, uint32_t &continuous_tail_axis_num,
                                                        uint32_t &discontinuous_axis_num);
  static af::Status GetNodeContinuousTailAxisNumByLoad(const af::AscNodePtr &node, uint32_t &continuous_tail_axis_num,
                                                       uint32_t &discontinuous_axis_num);
  static af::Status CollectTransposePreNodes(const af::AscGraph &graph, std::set<af::AscNodePtr> &transpose_pre_nodes);
  static af::Status UpdateOutputVectorizedStrides(const af::AscNodePtr &node, uint32_t continuous_tail_axis_num,
                                                  uint32_t align_width);
  static af::Status ModifyIndirectLoadVectorizedStrides(ascir::ImplGraph &impl_graph);
};

Status GenLoadToGenNddmaNode(const af::AscNodePtr &node_load);
Status PadNddmaOutputVectorizedStrides(const af::AscGraph &graph);
}  // namespace optimize
#endif  // OPTIMIZE_PLATFORM_V2_UN_ALIGNMENT_STRATEGY_H
