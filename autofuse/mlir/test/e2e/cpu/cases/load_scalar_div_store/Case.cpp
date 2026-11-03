/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> div(scalar, tile) -> store CPU case ----*- C++ -*-===//
//
// 方向要发出去的那一族（规则二）的第二个 op：`y = 4.0 / x`，标量在**左**。
// `load_scalar_sub_store` 是这一族的第一条，两者钉的不是同一件事：
//
//   - 那一条钉的是**这条发射规则本身**（`printApiExtensionScalarBinary` 会把方向写进模板
//     实参，不是把 tile 挪到第一位了事）；
//   - 这一条钉的是 `div` 自己那一份供给：`Divs<T, false>` 与 `Subs<T, false>` 不共用任何
//     代码，方向在 `scalar_div.h` 里落成的是 `BinaryRepeatParams` 的四个 stride
//     （`src0BlkStride` / `src0RepStride` 置 0，让被 `Duplicate` 出来的那一个 block 在
//     src0 位上原地重复），以及两处 `AscendC::Div` 的实参次序。**那一整段只有
//     `IS_SCALAR_LATTER == false` 时才走到**，发射规则再对也走不到它。
//
// 取值全是 2 的整数次幂并带符号，`4.0 / x` 的商也是，所以逐元素比相等而不是比容差
// （`load_div_store` 那一条比的是 1e-5 绝对量，因为它的输入是任意值）。方向写反算的是
// `x / 4.0`，它与 `4.0 / x` 相等要 |x| = 4，而取值全在 [0.25, 2] 里，一个都不是。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_scalar_div_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_scalar_div_store;

  const auto output_shape = cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  // 标量在前，tile 在后；输出属性跟第二个输入（follow_index = 1）。
  auto graph = builder.Scalar("four", "4.0", af::DT_FLOAT)
                   .Op<af::ascir_op::Div>("div", {"four", "load0"}, 1U)
                   .Store("store", "div")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace cfg = load_scalar_div_store;
  EXPECT_NE(kernel.find(std::string(cfg::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // **字面量写全**：emitter 打的是定长科学计数。
  EXPECT_NE(kernel.find("constexpr float c4_f32 = (float)4.000000000e+00;"), std::string::npos);
  // **`false` 是这条用例的全部理由**：标量在左。写成 `true`（或者少写这一位、落到缺省值）
  // 逐元素断言当场红。
  EXPECT_NE(kernel.find("Divs<float, false>("), std::string::npos);
  EXPECT_EQ(kernel.find("Divs<float, true>("), std::string::npos);
  // **不能查「`AscendC::Div<` 一个都没有」**：`scalar_div.h` 的函数体自己就发
  // `AscendC::Div<T, false>`，而这个头是整段内联进 kernel 的。「选中的是标量形态」
  // 由上面两条正 / 负断言管。
  //
  // 包装的头与它的前导都内联进来了：`scalar_div.h` 用的 `MAX_REPEAT_TIME` 与
  // `KernelUtils::BlkSize` / `RptSize` 只有 `utils.h` 声明，少了它 lit 全绿、
  // device 编译器报的是内联进来那段头文本的行号。
  EXPECT_NE(kernel.find("__ASCENDC_API_SCALAR_DIV_H__"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
  // 标量形态要 tmp buffer（`Duplicate` 出来的那一个 block 放在里面），而 `afir.div` 的
  // tensor-tensor 形态发的是四参 `AscendC::Div`、不收 tmp。TBuf 只会因为 tmp buffer 出现。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
  // mem 那一侧：reg 侧发的是 `DivExtends`（`div_reg_base.h`）。
  EXPECT_EQ(kernel.find("__ASCENDC_REGBASE_API_DIV_H__"), std::string::npos);
}
}  // namespace mlir::afir
