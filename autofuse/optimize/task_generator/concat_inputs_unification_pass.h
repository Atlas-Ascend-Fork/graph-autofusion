/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_TASK_GENERATOR_CONCAT_INPUTS_UNIFICATION_PASS_H
#define OPTIMIZE_TASK_GENERATOR_CONCAT_INPUTS_UNIFICATION_PASS_H

#include "ascir_ops.h"
#include "ascir/meta/ascir.h"
#include "common/ascgen_log.h"

namespace optimize {

class ConcatInputUnificationPass {
 public:
  static Status Run(std::vector<ascir::ImplGraph> &graphs);
  static bool CanOptimize(const af::AscNodePtr &concat_node, size_t concat_dim);

 private:
  static Status RunOneGraph(ascir::ImplGraph &graph);
  static bool NeedOptimize(const af::AscNodePtr &concat_node, std::set<int32_t> &input_indices_need_copy);
  static bool CanOptimize(const af::AscNodePtr &concat_node, size_t concat_dim,
                          std::set<int32_t> &input_indices_need_copy);
  static Status DoOptimize(ascir::ImplGraph &graph, const af::AscNodePtr &concat_node,
                           const std::set<int32_t> &input_indices_need_copy);
  static af::Expression GetColSize(const af::AscTensor &tensor, size_t concat_dim);
  static af::Status GetQueInputIndices(const af::AscNodePtr &concat_node, std::set<int32_t> &input_indices_need_copy);
  static bool IsSrcColSizeAlignedToB4(const af::AscNodePtr &concat_node, size_t concat_dim, int32_t dtype_size);
  static bool IsSrcColSizeOverLimit(const af::AscNodePtr &concat_node, size_t concat_dim, int32_t dtype_size);
  static void LogSrcColSizeOverLimit(const af::AscNodePtr &concat_node, size_t concat_dim, int32_t dtype_size);
  static af::Status CollectSharedInputs(const af::AscNodePtr &concat_node, std::set<int32_t> &input_indices_need_copy);
};

}  // namespace optimize

#endif  // OPTIMIZE_TASK_GENERATOR_CONCAT_INPUTS_UNIFICATION_PASS_H
