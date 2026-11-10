/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// where 和 select 的标量形态生成同一 Where 包装调用，但 AFIR op 和 implementation id 不同。
// 因此各自保留用例，分别验证 where.tensor_scalar.extend 与 select.tensor_scalar.extend 的注册。

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

  // 输出属性跟第二个输入（follow_index = 1）：第一个是 ui8 掩码，第三个是标量。
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
  // 字面量写全：emitter 打的是定长科学计数。
  EXPECT_NE(kernel.find("constexpr float c7_f32 = (float)7.000000000e+00;"), std::string::npos);
  // 标量在第四个实参位（dst, mask, src0, src1, count, tmp），也就是 AFIR 上的原次序。
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
                        "AutofuseTilingData t)"),
            std::string::npos);
  // mem 一侧要 tmp buffer：标量那一路先 `Duplicate` 进去一个 block。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
