/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> eq(bool) -> where -> store CPU case (reg-base) ===//
//
// `load_eq_where_reg_base_store` 的 bool 版：那一张图把 `eq` 的输出手写成 `DT_UINT8`，
// 这里改成 `DT_BOOL`。比较在寄存器形态下产出的是一条谓词（`MaskReg`），一次都不落 UB，
// 但 bridge 仍然先要把子图里的这个节点建成 `ui8` 的 `afir.eq`——bool 不在 AFIR 的供给表上，
// 不归一化的话这张图在 bridge 的子图收集那一步就整张回落。
//
// 子图用的是另一份上下文（`AppendVectorFuncBody` 里的 `bodyCtx`），所以「比较输出是 bool
// 就按 ui8 建」在那一份上同样要成立。生成的 kernel 与 ui8 版逐字同形，`CheckGeneratedKernel`
// 的判据照搬那一份。
//===----------------------------------------------------------------------===//

#include <cstddef>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_eq_where_bool_reg_base_store/LoadEqWhereBoolRegBaseStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
namespace {
// kernel 形参的序号，`kInputShapes` 按同一个顺序排：比较的两路在前，`where` 的两路候选在后。
constexpr size_t kCompareLhsArg = 0;
constexpr size_t kCompareRhsArg = 1;
constexpr size_t kWhereThenArg = 2;
constexpr size_t kWhereElseArg = 3;
}  // namespace

af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_eq_where_bool_reg_base_store;

  const auto output_shape = config::GetShapeConfig(config::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, kCompareLhsArg).shape);
  const auto input1_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, kCompareRhsArg).shape);
  const auto input2_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, kWhereThenArg).shape);
  const auto input3_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, kWhereElseArg).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", kCompareLhsArg, input0_shape, {}, af::DT_FLOAT)
      .Data("input1", kCompareRhsArg, input1_shape, {}, af::DT_FLOAT)
      .Data("input2", kWhereThenArg, input2_shape, {}, af::DT_FLOAT)
      .Data("input3", kWhereElseArg, input3_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {})
      .Load("load2", "input2", input2_shape, {})
      .Load("load3", "input3", input3_shape, {});

  // `builder.Eq(...)` 把输出写死成 `DT_UINT8`；这一格要的是 bool。
  builder.OpWithDtype<af::ascir_op::Eq>("eq", {"load0", "load1"}, 0U, af::DT_BOOL);
  auto graph = builder.Where("where", "eq", "load2", "load3")
                   .Store("store", "where")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_eq_where_bool_reg_base_store;
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 产物必须是寄存器形态：一个向量作用域、一层 lane 循环，中间结果不落 UB。
  EXPECT_NE(kernel.find("__VEC_SCOPE__"), std::string::npos);
  // 四路各取一次、结果存一次。计数式而不是「存在一条」。
  EXPECT_EQ(CountOccurrences(kernel, "LoadAlign"), 4U);
  EXPECT_EQ(CountOccurrences(kernel, "StoreAlign"), 1U);
  // 比较与选择各一条。
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::Compare<"), 1U);
  EXPECT_EQ(CountOccurrences(kernel, "AscendC::Reg::Select("), 1U);
  // 谓词是一条 `MaskReg`，没有被物化回值。
  EXPECT_EQ(kernel.find("AscendC::Reg::Duplicate"), std::string::npos);
  // 这两条是「真的走了 VF 这条路」的判据：tile 级的包装一条都不该剩。
  EXPECT_EQ(kernel.find("CompareExtend"), std::string::npos);
  EXPECT_EQ(kernel.find("WhereExtend"), std::string::npos);
}
}  // namespace mlir::afir
