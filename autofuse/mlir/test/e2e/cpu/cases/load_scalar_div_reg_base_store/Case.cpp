/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> div(scalar, tile) -> store (reg-base) --*- C++ -*-===//
//
// **reg-base 一侧标量形态的第一条 E2E**，与 `load_scalar_div_store`（mem 侧）同一张图、
// 同一份数值断言，差别只有 soc。
//
// 在这一批之前，12 个 `*.tensor_scalar.*_reg_base` implementation **一个都没有被任何
// 编译器看过**——四条 reg-base 用例（cast / sign / eq / neg）全是 tensor 形态。这正是
// L-13 的形状：lit 比的是发出来的文本，而包装头是整段内联进 kernel 的，少一个前导、
// 少一个重载都要到 device 编译器那一步才炸。
//
// 这一条同时钉住**两侧 tmp 契约相反**这件事：mem 侧的 `Divs` 要一块 tmp（把标量
// `Duplicate` 进去一个 block），reg 侧的 `DivExtends` 签名里根本没有那个形参。
// 调度那一侧照哪一档算 UB 是按 NpuArch 分流的，两边对不上时 `checkTmpBuffer` 当场判
// unsupported——所以这条用例也是「换 soc 的三处都改对了」的判据之一。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_scalar_div_reg_base_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_scalar_div_reg_base_store;

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
  namespace cfg = load_scalar_div_reg_base_store;
  EXPECT_NE(kernel.find(std::string(cfg::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // **字面量写全**：emitter 打的是定长科学计数。
  EXPECT_NE(kernel.find("constexpr float c4_f32 = (float)4.000000000e+00;"), std::string::npos);
  // 整条调用照实际输出写死。**`false` = 标量在左**，与 mem 侧同一位；但这边的包装叫
  // `DivExtends`、**没有 tmp 形参**，count 直接跟在标量后面。
  EXPECT_NE(kernel.find("DivExtends<float, false>("), std::string::npos);
  EXPECT_EQ(kernel.find("DivExtends<float, true>("), std::string::npos);
  // 内联进来的是 reg 那一份。mem 侧那个头一个字都不该在——它在的话说明 soc 换了而
  // NpuArch 没换，或者反过来。
  EXPECT_NE(kernel.find("__ASCENDC_REGBASE_API_DIV_H__"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_SCALAR_DIV_H__"), std::string::npos);
  // reg 侧这份包装不要 tmp buffer，也不声明前导头——**两条都与 mem 侧相反**。
  EXPECT_EQ(kernel.find("AscendC::TBuf<"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
}
}  // namespace mlir::afir
