/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> clip_by_value(tile, s, s) -> store -----*- C++ -*-===//
//
// **三元族标量形态的第一条 E2E**：`clip(x, -1, 1)`，上下界都是编译期常量。这是 clip 在真实
// 图里最常见的形态。
//
// 与 M1 那一族（`Subs` 的 bool 模板实参）不同，三元这一族的标量形态**没有新的调用形状**：
// `clipbyvalue.h` 的场景 2 与 tensor 形态同名同位，只有那两个位置上的实参从 tile 变成标量。
// 所以这条用例真正钉的是**位置**：
//
//   - 上下界不对称（-1 / +1），而且输入两侧都超界。min 与 max 互换的话逐元素断言当场红
//     ——`clip(x, 1, -1)` 在这份取值上处处不同。
//   - 标量没有块内偏移。`materializeOperandTiles` 原先对三个输入一律加块内偏移，标量那两位
//     现在按「是不是 tile」分别判；判错的话给一个 `f32` 建 `local_tensor.sub_index`，
//     在 MLIR 这一层就编不过（类型对不上），到不了这里。
//
// 取值都是 0.25 的倍数、在 f32 里精确可表示，边界值也精确，所以逐元素比相等。
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_clip_by_value_scalar_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_clip_by_value_scalar_store;

  const auto output_shape = cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape;
  const auto x_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, x_shape, {}, af::DT_FLOAT).Load("load0", "input0", x_shape, {});

  auto graph = builder.Scalar("lower", "-1.0", af::DT_FLOAT)
                   .Scalar("upper", "1.0", af::DT_FLOAT)
                   .Op<af::ascir_op::ClipByValue>("clip", {"load0", "lower", "upper"})
                   .Store("store", "clip")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace cfg = load_clip_by_value_scalar_store;
  EXPECT_NE(kernel.find(std::string(cfg::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 两个常量各自按元素类型建出来，**字面量写全**：emitter 打的是定长科学计数。
  EXPECT_NE(kernel.find("constexpr float cm1_f32 = (float)-1.000000000e+00;"), std::string::npos);
  EXPECT_NE(kernel.find("constexpr float c1_f32 = (float)1.000000000e+00;"), std::string::npos);
  // 与 tensor 形态同名的裸调用；两个标量落在 min / max 的位置上，次序就是 (x, min, max)。
  EXPECT_NE(kernel.find("ClipByValue("), std::string::npos);
  EXPECT_NE(kernel.find(", cm1_f32, c1_f32, "), std::string::npos);
  // 三个输入里只有 x 走 Load；两个标量不是 GM 输入，所以 kernel 形参比 tensor 形态少两个。
  EXPECT_NE(kernel.find(std::string("__global__ __aicore__ void ") + cfg::kKernelName +
                        "(__gm__ float* v1, __gm__ float* v2, __gm__ uint8_t* v3, AutofuseTilingData v4)"),
            std::string::npos);
  // mem 侧这个包装要 tmp buffer。TBuf 只会因为 tmp buffer 出现。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
