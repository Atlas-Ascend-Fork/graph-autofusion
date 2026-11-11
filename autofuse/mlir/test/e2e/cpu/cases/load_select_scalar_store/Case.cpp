/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x2 -> select(mask, scalar, tile) -> store -*- C++ -*-===//
//
// 选择族标量形态的第一条 E2E：`y = isnan(p) ? 3.0 : v`，标量在 then 位。
// `load_where_scalar_store` 是同类的另一位（标量在 else），两条合起来把
// `where.h` 的场景 2 与场景 3 各走一遍——那是两个不同的重载，各自把
// `src0BlkStride` / `src1BlkStride` 里的一侧置 0，写串了就是拿另一路去广播。
//
// 与 `load_clip_by_value_scalar_store` 的差别不在「都是三元带标量」，而在
// 第一个输入的元素类型：这一类的第一位是 ui8 掩码、与另外两位不同，clip 族三位同
// dtype，两类在 ODS 与 capability 上就是按这一条分开的。标量形态这一档现在两类各有一条。
//
// 掩码由图里的 `Isnan` 现算，与 `load_select_store` 同一种形状；值那一路单独从 GM 搬，
// 所以 GM 输入是两个、kernel 形参比 tensor 形态少一个。
//
// 位置是这条用例钉的东西。 这一类没有承载方向的模板实参，标量在哪一位由 C++ 重载
// 决议挑，所以把 tile 挪到第一个数据实参位（规则一对可交换的八个是对的）在这里生成的
// 是 `isnan(p) ? v : 3.0`——合法、编得过、算反。取值让两者处处不同。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_select_scalar_store/LoadSelectScalarStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_select_scalar_store;

  const auto output_shape = cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape;
  const auto probe_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  const auto value_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 1).shape);
  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, probe_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, value_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", probe_shape, {})
      .Load("load1", "input1", value_shape, {});

  // 输出属性跟第三个输入（follow_index = 2）：第一个是 ui8 掩码，第二个是标量，
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
  namespace cfg = load_select_scalar_store;
  EXPECT_NE(kernel.find(std::string(cfg::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 字面量写全：emitter 打的是定长科学计数。
  EXPECT_NE(kernel.find("constexpr float c3_f32 = (float)3.000000000e+00;"), std::string::npos);
  // 标量在第三个实参位（dst, mask, src0, src1, count, tmp），也就是 AFIR 上的原次序。
  // 从最后一处 `Where(` 起算：`where.h` 整段内联进 kernel，前面那些是各重载的定义，
  // 调用点在 kernel 体里、排在它们之后。
  const size_t call = kernel.rfind("Where(");
  ASSERT_NE(call, std::string::npos);
  size_t comma = kernel.find(',', call);
  ASSERT_NE(comma, std::string::npos);
  comma = kernel.find(',', comma + 1);
  ASSERT_NE(comma, std::string::npos);
  EXPECT_EQ(kernel.compare(comma + 1, std::string(" c3_f32,").size(), " c3_f32,"), 0)
      << kernel.substr(call, kernel.find(';', call) - call);
  // 包装是全局函数，没有 `AscendC::` 限定，也没有显式模板实参。
  EXPECT_EQ(kernel.find("AscendC::Where"), std::string::npos);
  EXPECT_EQ(kernel.find("Where<"), std::string::npos);
  // 两个扩展头都内联进来了，前导 `utils.h` 也在——`where.h` / `duplicate.h` 用的
  // `KernelUtils::` 与 `MAX_REPEAT_TIME` 只有它声明。
  EXPECT_NE(kernel.find("__ASCENDC_API_WHERE_H__"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_DUPLICATE_H__"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
  // 三个输入里只有掩码源与值那一路走 Load；标量不是 GM 输入，所以形参比 tensor 形态少一个。
  EXPECT_NE(kernel.find(std::string("__global__ __aicore__ void ") + cfg::kKernelName +
                        "(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ uint8_t* v4, "
                        "AutofuseTilingData v5)"),
            std::string::npos);
  // mem 一侧要 tmp buffer：标量那一路先 `Duplicate` 进去一个 block。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
