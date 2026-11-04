/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x2 -> select(mask, scalar, tile) -> store (reg-base) ===//
//
// reg-base 一侧的三元标量形态：`y = isnan(p) ? 3.0 : v`，标量在 then 位。与
// `load_select_scalar_store`（mem 侧）同一张图、同一份数值断言，差别只有 soc。
//
// 两侧差三件，都只有真编一遍才看得到：
//
//   - 包装不同名：mem 侧 `Where`、reg 侧 `WhereExtend`；
//   - **tmp 的去向不同**。reg 侧的签名里没有 tmp 形参，而调度照样给这个节点分了一块
//     ——生成的 kernel 里因此有一块取出来、`ReinterpretCast` 过、**没有任何调用读它**的
//     `TBuf`。这是 `TmpBufferUse::Unused` 那一档，与 `load_eq_reg_base_store` 同一个形状；
//     它不该被优化掉，ATT 的 UB 预算是按分了这块算的；
//   - mem 侧还要内联 `duplicate.h`（标量先 `Duplicate` 进一个 block），reg 侧不用。
//
// 掩码由图里的 `Isnan` 现算。reg 侧的 `isnan` 发的是标准调用 `AscendC::IsNan`、不收 tmp，
// 与 mem 侧的包装不是一份东西——这条路也顺带在这里第一次走通。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_select_scalar_reg_base_store/LoadSelectScalarRegBaseStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_select_scalar_reg_base_store;

  const auto output_shape = cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape;
  const auto probe_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  const auto value_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 1).shape);
  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, probe_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, value_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", probe_shape, {})
      .Load("load1", "input1", value_shape, {});

  // 输出属性跟**第三个**输入（follow_index = 2）：第一个是 ui8 掩码，第二个是标量，
  // 两个都说不出输出的 dtype 与形状。
  auto graph = builder.Isnan("mask", "load0")
                   .Scalar("three", "3.0", af::DT_FLOAT)
                   .Op<af::ascir_op::Select>("select", {"mask", "three", "load1"}, 2U)
                   .Store("store", "select")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace cfg = load_select_scalar_reg_base_store;
  EXPECT_NE(kernel.find(std::string(cfg::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // **字面量写全**：emitter 打的是定长科学计数。
  EXPECT_NE(kernel.find("constexpr float c3_f32 = (float)3.000000000e+00;"), std::string::npos);
  // **标量在第三个实参位**（dst, mask, src0, src1, count），也就是 AFIR 上的原次序。
  // 从最后一处 `WhereExtend(` 起算：头整段内联进 kernel，前面那些是各重载的定义。
  const size_t call = kernel.rfind("WhereExtend(");
  ASSERT_NE(call, std::string::npos);
  size_t comma = kernel.find(',', call);
  ASSERT_NE(comma, std::string::npos);
  comma = kernel.find(',', comma + 1);
  ASSERT_NE(comma, std::string::npos);
  EXPECT_EQ(kernel.compare(comma + 1, std::string(" c3_f32,").size(), " c3_f32,"), 0)
      << kernel.substr(call, kernel.find(';', call) - call);
  // 内联进来的是 reg 那一份；mem 侧那两个头一个都不该在。
  EXPECT_NE(kernel.find("__ASCENDC_API_REGBASE_WHERE_H__"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_WHERE_H__"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_DUPLICATE_H__"), std::string::npos);
  // 掩码这一路走的是标准调用，不是 mem 侧那个包装。
  EXPECT_NE(kernel.find("AscendC::IsNan("), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_ISNAN_H__"), std::string::npos);
  // `TmpBufferUse::Unused`：分了一块，调用收不下。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
