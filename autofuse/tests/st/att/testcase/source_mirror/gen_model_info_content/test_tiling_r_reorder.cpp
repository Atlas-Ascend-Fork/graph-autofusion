/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "gen_model_info.h"
#include "graph_construct_utils.h"
#include "generator/preprocess/args_manager.h"

namespace att {
namespace {
AttAxisPtr FindAxis(const ModelInfo &model_info, const std::string &name) {
  for (const auto &axis : model_info.arg_list) {
    if (axis->name == name) {
      return axis;
    }
  }
  return nullptr;
}
}  // namespace

TEST(TilingRReorderST, DuplicateReduceInnermostOrderReachesSolverPriorities) {
  af::AscGraph graph("DuplicateReduceInnermost");
  ASSERT_EQ(GraphConstructUtils::BuildConcatGroupAscendGraphS0S1ReduceMultiTiling(graph), af::SUCCESS);
  GraphConstructUtils::UpdateGraphVectorizedStride(graph);
  graph.SetTilingKey(0U);
  std::vector<ModelInfo> model_infos;
  ASSERT_EQ(GenerateModelInfo({graph}, model_infos, {}), af::SUCCESS);
  ASSERT_EQ(model_infos.size(), 2U);
  const auto &tiling_r = model_infos[0];
  const auto &canonical = model_infos[1];
  EXPECT_EQ(tiling_r.sub_case_tag, "R");
  EXPECT_TRUE(canonical.sub_case_tag.empty());
  EXPECT_EQ(tiling_r.tiling_case_id, canonical.tiling_case_id);
  EXPECT_TRUE(tiling_r.runtime_reorder_rules.empty());
  const auto reduce_axis = FindAxis(canonical, "s0t");
  const auto non_reduce_axis = FindAxis(canonical, "s1t");
  ASSERT_NE(reduce_axis, nullptr);
  ASSERT_NE(non_reduce_axis, nullptr);
  const auto reduce_size = reduce_axis->size->symbol_expr;
  const auto non_reduce_size = non_reduce_axis->size->symbol_expr;
  const auto canonical_priority = ArgsManager(canonical).GetAxesPriority();
  const auto tiling_r_priority = ArgsManager(tiling_r).GetAxesPriority();
  EXPECT_LT(canonical_priority.at(reduce_size), canonical_priority.at(non_reduce_size));
  EXPECT_LT(tiling_r_priority.at(non_reduce_size), tiling_r_priority.at(reduce_size));
  EXPECT_EQ(FindAxis(tiling_r, "s0t"), reduce_axis);
  EXPECT_EQ(FindAxis(tiling_r, "s1t"), non_reduce_axis);
}
}  // namespace att
