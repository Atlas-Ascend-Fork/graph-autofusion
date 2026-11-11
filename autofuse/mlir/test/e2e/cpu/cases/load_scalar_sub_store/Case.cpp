/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> sub(scalar, tile) -> store CPU case ----*- C++ -*-===//
//
// 方向要生成的那一类的第一条 E2E：`y = 2.5 - x`，标量在左。
//
// 与 `load_scalar_add_store` 的差别就是这个 op 不可交换。`add` 那一条交换完生成的是逐字
// 相同的 `Adds`，所以它钉的只是「生成顺序换过来了」；`sub` 交换之后生成的是
// `Subs<float, false>`——**方向落在一个 bool 模板实参上**，而那一位写错的后果是算出
// `x - 2.5` 而不是 `2.5 - x`。两者在数值上处处不同（除了 x = 1.25 那一个点），所以逐元素
// 断言就是这一位的埋点。
//
// 这也是第一条走 `Subs` 的用例，顺带把两件事钉住：
//
//   - mem 侧的标量形态要 tmp buffer，而同一个 op 的 tensor-tensor 形态不要
//     （`AscendC::Sub` 是四参调用）。同一个 op 两档 tmp 取值不同，这是第一次。
//   - `scalar_sub.h` 的 `utils.h` 前导。少了它 lit 全绿、device 编译器报的是内联进来那段
//     头文本的行号——与 `floor_div.h` / `sign.h` 同一条。
//
// 取值都在 f32 里精确可表示（0.5 的倍数），减完也是，所以逐元素比相等而不是比容差：
// 容差一放宽，「方向反了」这种错在某些取值上会正好落进容差里。
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_scalar_sub_store/LoadScalarSubStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_scalar_sub_store;

  const auto output_shape = cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  // 标量在前，tile 在后；输出属性跟第二个输入（follow_index = 1）。
  auto graph = builder.Scalar("two", "2.5", af::DT_FLOAT)
                   .Op<af::ascir_op::Sub>("sub", {"two", "load0"}, 1U)
                   .Store("store", "sub")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_scalar_sub_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 字面量写全：emitter 打的是定长科学计数。
  EXPECT_NE(kernel.find("constexpr float c2_5_f32 = (float)2.500000000e+00;"), std::string::npos);
  // `false` 是这条用例的全部理由：标量在左。写成 `true`（或者少写这一位、落到缺省值）
  // 逐元素断言当场红。
  EXPECT_NE(kernel.find("Subs<float, false>("), std::string::npos);
  EXPECT_EQ(kernel.find("Subs<float, true>("), std::string::npos);
  // 这里不能查「`AscendC::Sub<` 一个都没有」：`scalar_sub.h` 的函数体自己就调
  // `AscendC::Sub`，而这个头是整段内联进 kernel 的，所以那个名字必然出现。第一版照
  // `load_scalar_add_store` 抄了这一条，当场红——`Adds` 是标准 API、没有头被内联进来，
  // 那边查得通。「选中的是标量形态」由上面两条正/负断言管，这一条本来就是多余的。
  // 包装的头与它的前导都内联进来了。
  EXPECT_NE(kernel.find("__ASCENDC_API_SCALAR_SUB_H__"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
  // 标量形态要 tmp buffer，tensor-tensor 那一支不要。TBuf 只会因为 tmp buffer 出现。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
