/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> abs -> neg -> store CPU case (reg-base, int8) -*- C++ -*-===//
//
// int8 的 VF 边界，取自一张真实调度出来的图。bridge 收集子图时有一道 dtype 白名单
// （`ScheduleToAfir.cpp` 的 `AppendVectorFuncBody`），int8 这一位此前不在名单上：带 int8
// 边界的 VF 在那里报能力不足、整张回落。名单一放开，就得有人证明它真的降得下去——
// 那道白名单上方的注释写的就是这件事：放开一位 dtype，拦住它的就从「干净回落」变成
// 「ODS 与 conversion 的硬失败」，而硬失败只有真编一遍 kernel 才看得见。
//
// 图是 `load_abs_neg_reg_base_store` 把 dtype 换成 int8。图上写的是两个 int8 的算子，
// 但 legacy 把 int8 的 `neg` 升到 int16 算（`NegAscIrCodegenImplV2` 的 dtype 映射，两头各
// 插一个 Cast），所以调度之后落进 VF 的是后半段：
//
//     abs(int8) [tile 级] -> VF[ cast(i8 -> i16) -> neg(i16) ] -> cast(i16 -> i8) [tile 级]
//
// VF 的源边界因此是 int8：取进来时 1 字节被 `DIST_UNPACK_B8` 撑进 2 字节的 lane 槽位，
// lane 按最宽的 i16 算；落点是 i16，与 lane 等宽、不带 dist。分区器只把两个及以上的节点
// 聚成 `VectorFunc`（`vector_func_partitioner.cpp` 的 `kMinVfNodesNum`），所以 `abs` 与降
// 回去的那一个 Cast 都留在根图里。
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_abs_neg_i8_reg_base_store/LoadAbsNegI8RegBaseStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_abs_neg_i8_reg_base_store;

  const auto output_shape = config::GetShapeConfig(config::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 0).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_INT8)
      .Load("load0", "input0", input0_shape, {});

  auto graph = builder.Abs("abs", "load0")
                   .Neg("neg", "abs")
                   .Store("store", "neg")
                   .Output("output", "store", 0, af::DT_INT8)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_abs_neg_i8_reg_base_store;
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 产物里有一段寄存器形态：一个向量作用域、一层 lane 循环。
  EXPECT_NE(kernel.find("__VEC_SCOPE__"), std::string::npos);
  // lane 按最宽的那个 dtype 算：VF 里是 i16。
  EXPECT_NE(kernel.find("AscendC::Reg::UpdateMask<int16_t>("), std::string::npos);
  // 源边界是 int8，取进来带 `DIST_UNPACK_B8`；落点与 lane 等宽、不带 dist。计数式而不是
  // 「存在一条」：少生成一条 Load 也能编过、也能算对。
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::LoadAlign<int8_t, AscendC::Reg::LoadDist::DIST_UNPACK_B8>"), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::StoreAlign<int16_t>("), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "LoadAlign"), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "StoreAlign"), 1U);
  // VF 里只有升精度与取负两步。
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::Cast<int16_t, int8_t, cast_trait_bf16_2_float>("), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::Neg("), 1U);
  // `abs` 与降回 int8 的那个 Cast 留在根图里，走 tile 级调用。
  EXPECT_NE(kernel.find("AscendC::Abs("), std::string::npos);
  EXPECT_NE(kernel.find("CastExtend("), std::string::npos);
}
}  // namespace mlir::afir
