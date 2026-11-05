/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_PLATFORM_V2_TEMPLATE_NDDMA_TEMPLATE_H
#define OPTIMIZE_PLATFORM_V2_TEMPLATE_NDDMA_TEMPLATE_H

#include "platform/common/base_template.h"

namespace optimize {

class NddmaTemplate : public BaseTemplate {
 public:
  ~NddmaTemplate() override = default;
  explicit NddmaTemplate() = default;

  std::string GenName(const std::string &general_case_name) override;
  static Status GenNddmaNode(const af::AscNodePtr &node_pre, const af::AscNodePtr &node_brc, af::AscGraph &new_case,
                             const bool is_need_realignment = false);
  static Status AddTransposeNodeAfter(af::AscGraph &graph, const af::AscNodePtr &node,
                                      af::AscNodePtr &new_transpose_node, const af::AscNodePtr &old_transpose_node);
  static Status MergeLoadAndTranspose(const af::AscNodePtr &load_node, af::AscGraph &new_case);
  static Status TransposeToNddmaNode(const af::AscNodePtr &transpose_node, af::AscGraph &new_case);
  af::Status Generate(const af::AscGraph &origin_graph, const af::AscGraph &based_case,
                      af::AscGraph &new_case) override;
  static Status SwapCastBrcAndGenNddma(const af::AscNodePtr &node_cast, const af::AscNodePtr &node_load,
                                       af::AscGraph &new_case);
  bool NeedDropBasedCase(const af::AscGraph &origin_graph, const af::AscGraph &based_case,
                         const af::AscGraph &new_case) override;
  static Status ReAlignVectorizedStrides(const af::AscNodePtr &node);
  static bool IsSecondaryTailAxisAligned(const af::AscNodePtr &node);
  std::string GetScoreFunc(const af::AscGraph &origin_graph, const af::AscGraph &nddma_graph) override;
  static af::Status ReorderRepeats(const af::AscNodePtr &node_src, const af::AscNodePtr &node_dst);
  static af::Status ProcessSliceToNddma(const af::AscNodePtr &node_load, bool &is_nddma_generated_cur,
                                        af::AscGraph &new_case);
  static Status ProcessTransposeNodes(af::AscGraph &new_case, bool &is_nddma_generated);
  static Status ProcessSliceNodes(af::AscGraph &new_case, bool &is_nddma_generated);
  static af::Status IsNeedDropTransposeBrcFuse(af::AscGraph &graph, bool &is_need_drop);
  static af::Status BroadcastInputNodeIsScalar(const af::AscNodePtr &node, bool &is_scalar);
  static af::Status IsVectorizedAxisDoubleCut(af::AscGraph &graph, const af::AscNodePtr &node,
                                              bool &is_vectorized_axis_double_cut);
  static af::Status IsDataTypeValidInBasedCase(const af::AscGraph &graph, bool &is_dtype_valid);
  static af::Status IsTransposeContinuousVectorizedAxisNumValid(af::AscGraph &graph,
                                                                bool &is_vectorized_continuous_axis_num_valid);
  NddmaTemplate(const NddmaTemplate &) = delete;
  NddmaTemplate &operator=(const NddmaTemplate &) = delete;
  NddmaTemplate(NddmaTemplate &&) = delete;
  NddmaTemplate &operator=(NddmaTemplate &&) = delete;
};
}  // namespace optimize

#endif  // OPTIMIZE_PLATFORM_V2_TEMPLATE_NDDMA_TEMPLATE_H
