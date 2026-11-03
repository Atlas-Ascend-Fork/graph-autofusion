/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> truediv(tile, scalar) -> store CPU case -*- C++ -*-===//
//
// 方向要发出去的那一族（规则二）的第三个 op，也是这一族**标量在右**的第一条 E2E。
// `load_scalar_sub_store` 与 `load_scalar_div_store` 都把标量放在左边，走的是
// `IS_SCALAR_LATTER == false`；标量在右那一侧此前只有 lit。它正好是**缺省值**那一侧
// （`DefaultValuedAttr<BoolAttr, "true">`），也就是这一位整个丢掉时会落到的那一侧——
// 两侧各有一条数值用例，这一位才是「写错就红」而不是「写错一半会红」。
//
// 输入 i32、输出 f32，与 `load_truediv_store` 同一档，理由也一样：`true_div.h` 的
// `else` 分支就是 `Divs` 的逐字转发，拿 f32 写这份用例的话它就是 `load_scalar_div_store`
// 的别名。i32 那一支多做两件事，**两件都只有这里走得到**：
//
//   - 先把 tile `Cast` 成 f32（`CAST_RINT`）再除，而标量是**按 `U` 也就是 f32 强转**后
//     传给 `Divs<U, ...>` 的，不是按 `T`；
//   - `Divs` 用的 tmp 是 `tmp_buf[cast_buf_size * sizeof(U)]` 之后那一段，**cast 结果与
//     `Duplicate` 出来的那个 block 共用同一块 tmp、分两段**。切错段两边互相踩，这是
//     `load_axpy_half_store` 之后第二处「同一块 tmp 切两段」。
//
// 还有一件只有编译期拦得住：`TrueDivExtends` 是这一族唯一有**两个** dtype 模板实参的
// （`<T, U, LATTER>`），方向那一位跳不过前面两个。少写一个 `U`，`LATTER` 就落到类型的
// 位置上，bool 当类型用编不过。
//
// 除数取 4：商是精确的四分之一倍数，`Cast` 与除法都不引入误差，所以逐元素比相等。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_truediv_scalar_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_truediv_scalar_store;

  const auto output_shape = cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_INT32).Load("load0", "input0", input_shape, {});

  // 标量与 tile 同 dtype（i32）；输出是 f32，由 `TrueDiv` 那条对表改的。
  auto graph = builder.Scalar("four", "4", af::DT_INT32)
                   .TrueDiv("truediv", "load0", "four")
                   .Store("store", "truediv")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace cfg = load_truediv_scalar_store;
  EXPECT_NE(kernel.find(std::string(cfg::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 两个 dtype 模板实参都写出来了，**最后一位是方向**。`true` 是缺省值，但缺省值不等于
  // 「可以不写」：少写一位的话 `LATTER` 落到 `U` 的位置上，bool 当类型用编不过。
  EXPECT_NE(kernel.find("TrueDivExtends<int32_t, float, true>("), std::string::npos);
  EXPECT_EQ(kernel.find("TrueDivExtends<int32_t, float, false>("), std::string::npos);
  // 标量是第三个实参，按 i32 建出来。
  EXPECT_NE(kernel.find("c4_i32"), std::string::npos);
  // **不能查「`TrueDivExtend(` 一个都没有」**：tensor-tensor 那一份的**定义**就在
  // `true_div.h` 里，而这个头是整段内联进 kernel 的。「选中的是标量形态」由上面两条
  // 带模板实参的正 / 负断言管。
  // 三个头都内联进来了：包装自己、它转发去的 `Divs`，以及两者共同的前导。
  EXPECT_NE(kernel.find("__ASCENDC_API_TRUE_DIV_H__"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_SCALAR_DIV_H__"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
  // 一块 tmp buffer，i32 这一支真的用它：前一段装 cast 结果，后一段给 `Divs` 的
  // `Duplicate`。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
  // mem 那一侧：reg 侧的 truediv 发的是 `DivExtends`（`div_reg_base.h`）且不收 i32。
  EXPECT_EQ(kernel.find("__ASCENDC_REGBASE_API_DIV_H__"), std::string::npos);
}
}  // namespace mlir::afir
