/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x2 -> where(mask, tile, scalar) -> store --*- C++ -*-===//
//
// **`afir.where` 的第一条 E2E**，同时是选择族标量形态的另一位：
// `y = isnan(p) ? v : 7.0`，标量在 **else 位**。`load_select_scalar_store` 把标量放在
// then 位，两条走的是 `where.h` 的两个不同重载（场景 3 与场景 2）：一个把
// `src1BlkStride` / `src1RepStride` 置 0，另一个置的是 `src0` 那两位，串了就是拿另一路
// 去广播。
//
// **`where` 与 `select` 在这条路上发的是同一个调用**（`Where(dst, mask, src0, src1,
// count, tmp_buf)`），差别只在 AFIR 的 op 与选中的 implementation id
// （`where.tensor_scalar.extend` / `select.tensor_scalar.extend`），生成的 kernel 文本
// 看不出来。所以两个 op 各留一条用例：谁那一侧的注册写错了，都只红自己这一份。理由与
// `load_div_store` / `load_sub_store` 分两份一样。
//
// legacy 给 `Where` 多收了 ui8 / ui16 而 `Select` 没有（见 Condition.td 上那段），这也是
// 两个 op 没有合成一个的原因。这条用例取 f32，共有的那一格。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_where_scalar_store/LoadWhereScalarStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_where_scalar_store;

  const auto output_shape = cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape;
  const auto probe_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  const auto value_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 1).shape);
  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, probe_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, value_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", probe_shape, {})
      .Load("load1", "input1", value_shape, {});

  // 输出属性跟**第二个**输入（follow_index = 1）：第一个是 ui8 掩码，第三个是标量。
  auto graph = builder.Isnan("mask", "load0")
                   .Scalar("seven", "7.0", af::DT_FLOAT)
                   .Op<af::ascir_op::Where>("where", {"mask", "load1", "seven"}, 1U)
                   .Store("store", "where")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace cfg = load_where_scalar_store;
  EXPECT_NE(kernel.find(std::string(cfg::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // **字面量写全**：emitter 打的是定长科学计数。
  EXPECT_NE(kernel.find("constexpr float c7_f32 = (float)7.000000000e+00;"), std::string::npos);
  // **标量在第四个实参位**（dst, mask, src0, src1, count, tmp），也就是 AFIR 上的原次序。
  // 从最后一处 `Where(` 起算：`where.h` 整段内联进 kernel，前面那些是各重载的定义。
  const size_t call = kernel.rfind("Where(");
  ASSERT_NE(call, std::string::npos);
  size_t comma = kernel.find(',', call);
  ASSERT_NE(comma, std::string::npos);
  for (int skipped = 0; skipped < 2; ++skipped) {
    comma = kernel.find(',', comma + 1);
    ASSERT_NE(comma, std::string::npos);
  }
  EXPECT_EQ(kernel.compare(comma + 1, std::string(" c7_f32,").size(), " c7_f32,"), 0)
      << kernel.substr(call, kernel.find(';', call) - call);
  // 包装是全局函数，没有 `AscendC::` 限定，也没有显式模板实参。
  EXPECT_EQ(kernel.find("AscendC::Where"), std::string::npos);
  EXPECT_EQ(kernel.find("Where<"), std::string::npos);
  // 两个扩展头都内联进来了，前导 `utils.h` 也在。
  EXPECT_NE(kernel.find("__ASCENDC_API_WHERE_H__"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_DUPLICATE_H__"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
  // 三个输入里只有掩码源与值那一路走 Load；标量不是 GM 输入。
  EXPECT_NE(kernel.find(std::string("__global__ __aicore__ void ") + cfg::kKernelName +
                        "(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ uint8_t* v4, "
                        "AutofuseTilingData v5)"),
            std::string::npos);
  // mem 一侧要 tmp buffer：标量那一路先 `Duplicate` 进去一个 block。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
