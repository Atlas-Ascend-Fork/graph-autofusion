/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> clip_by_value(tile, s, s) -> store (reg-base) ------===//
//
// reg-base 一侧的三元标量形态（非掩码那一支）：`clip(x, -1, 1)`。与
// `load_clip_by_value_scalar_store`（mem 侧）同一张图、同一份数值断言，差别只有 soc。
//
// 两份 `clipbyvalue.h` 的 include guard **撞名**（都是 `__ASCENDC_API_CLIPBYVALUE_H__`），
// 与 `pow` 一样，所以钉的是各自独有的名字：reg 那份用 `Clamp`，mem 那份用
// `AscendC::ClampMax` / `ClampMin` / `Maxs` / `Mins`。
//
// 实参序也不同：mem 侧是 `(dst, x, min, max, calCount, tmp_buf)`，reg 侧**没有 tmp**，
// count 就是最后一个。上一轮给 mem 侧补三个转发重载的那件事（六份 lit 全绿而 device
// 编译器报 `no matching function`）说明这一族的重载表就是要靠 E2E 兜——reg 侧此前
// 一条都没有。
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_clip_by_value_scalar_reg_base_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_clip_by_value_scalar_reg_base_store;

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
  namespace cfg = load_clip_by_value_scalar_reg_base_store;
  EXPECT_NE(kernel.find(std::string(cfg::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 两个常量各自按元素类型建出来，**字面量写全**。
  EXPECT_NE(kernel.find("constexpr float cm1_f32 = (float)-1.000000000e+00;"), std::string::npos);
  EXPECT_NE(kernel.find("constexpr float c1_f32 = (float)1.000000000e+00;"), std::string::npos);
  // 与 tensor 形态同名的裸调用；两个标量落在 min / max 的位置上，**count 是最后一个
  // 实参**——reg 侧这一族不收 tmp。
  EXPECT_NE(kernel.find(", cm1_f32, c1_f32, static_cast<uint32_t>("), std::string::npos);
  // 内联进来的是 reg 那一份。guard 撞名，所以钉两侧各自独有的名字。
  EXPECT_NE(kernel.find("__ASCENDC_API_CLIPBYVALUE_H__"), std::string::npos);
  EXPECT_NE(kernel.find("Clamp("), std::string::npos);
  EXPECT_EQ(kernel.find("AscendC::ClampMax("), std::string::npos);
  // reg 侧不要 tmp buffer——**与 mem 侧相反**。TBuf 只会因为 tmp buffer 出现。
  EXPECT_EQ(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
