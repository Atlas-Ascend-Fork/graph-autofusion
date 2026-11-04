/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x2 -> axpy(f16) -> store CPU case --------*- C++ -*-===//
//
// **这份用例存在的唯一理由：把 `AxpyExtend` 的 f16 分支真的跑一遍。**
//
// `load_axpy_store` 走 f32，而 `axpy.h` 里 f32 落在 `else` 那一支——它**一次都没用那块
// tmp buffer**，形参在签名里只是占个位置。真正用它的是 f16 分支：把 tmp 切成两段，两个
// 输入各 `Cast` 成 float，算完再 `CAST_RINT` 回 half。所以 f32 那份用例证明的是「tmp
// buffer 被正确传进去了」，证不了「那块 tmp 够大、切得对、算得对」——包括
// `CalcAxpyTmpSize` 的 f16 那支算出来的尺寸。这一条是进度文档「已知覆盖空洞」第 3 条
// （清单里的 L-10）。
//
// **也是整个 CPU E2E 套件里第一个 f16 用例**。在它之前所有用例都是
// f32 / i32 / i16 / ui8，半精度这条路（ATT tiling 的 dtype、GM 上的 2 字节步长、
// `Cast` 往返）在这一层一次都没走过。
//
// 数值选得让期望值在 half 里精确可表示，见 KernelTest.cpp 那一段——不精确的话这份用例
// 就只能比容差，而容差一放宽，tmp 切错段那种「结果偏一点」的错就漏过去了。
//
// 其余（alpha 的三种形态、tier D1 只有一个 implementation）与 `load_axpy_store` 相同，
// 不在这里重复断言。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_axpy_half_store/LoadAxpyHalfStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_axpy_half_store::GetShapeConfig(load_axpy_half_store::kOutputShapes, 0).shape;
  const auto input0_shape =
      ToExpressions(load_axpy_half_store::GetShapeConfig(load_axpy_half_store::kInputShapes, 0).shape);
  const auto input1_shape =
      ToExpressions(load_axpy_half_store::GetShapeConfig(load_axpy_half_store::kInputShapes, 1).shape);
  AscGraphBuilder builder(load_axpy_half_store::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_FLOAT16)
      .Data("input1", 1, input1_shape, {}, af::DT_FLOAT16)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  auto graph = builder.Axpy("axpy", "load0", "load1", load_axpy_half_store::kAlpha)
                   .Store("store", "axpy")
                   .Output("output", "store", 0, af::DT_FLOAT16)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_axpy_half_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 裸名，不带 `AscendC::` 限定——`axpy.h` 里的包装是全局函数。
  EXPECT_NE(kernel.find("AxpyExtend("), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_AXPY_H__"), std::string::npos);
  // **alpha 仍然是 f32**：包装的形参是 `const float alpha`，不跟 tile 的元素类型走。
  // 这一条只有 f16 的用例拦得住——f32 那份里两者恰好一致，写反了看不出来。
  EXPECT_NE(kernel.find("constexpr float"), std::string::npos);
  EXPECT_NE(kernel.find("(float)5.000000000e-01"), std::string::npos);
  EXPECT_EQ(kernel.find("constexpr half"), std::string::npos);
  // 包装要一块 tmp buffer，f16 这一支是真用它。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
  // 共用的是 `api/` 那一份，不存在 reg-base 的对应物。
  EXPECT_EQ(kernel.find("axpy_reg_base.h"), std::string::npos);
}
}  // namespace mlir::afir
