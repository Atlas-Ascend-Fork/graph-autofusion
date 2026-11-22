/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - scalar + load -> store CPU case ---------------*- C++ -*-===//
//
// 标量在第一个操作数（L-4 的 `swap(2nd)` 那半张格子）。与
// `load_add_scalar_store` 成对：两份图只差操作数次序，生成的 kernel 必须逐字相同。
//
// legacy 那边这半张格子是靠改图支持的——`ScalarBroadcastOptimizationPass` 的
// step7 `SwapInputIndex` 把两个输入换过来，`binary_api_call.cpp` 再按 `switch_scalar`
// 换一次。但那个 pass 只对 `Broadcast` 节点触发，直接建一个 `Scalar` 节点的图（就是
// 这一份）它不管，于是图原样到 MLIR。L-4 之前 ODS 当场拒收
// （`operand #0 must be Tensor or MemRef of ... but got 'f32'`），整条 MLIR codegen bail
// 掉、回落 legacy。
//
// 判据是数值加「与不交换那份同一个调用」：`ScalarOperand::Any` 只把有形状的操作数挪到
// 第一个实参上，不换调用形状。生成顺序要是没换过来，`Adds` 的第二个实参会是一个标量
// ——那时 device 编译器直接报错，红得比数值断言还早。
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_scalar_add_store/LoadScalarAddStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_scalar_add_store;

  const auto output_shape = cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  // 标量在前，tile 在后；输出属性跟第二个输入（follow_index = 1）。
  auto graph = builder.Scalar("two", "2.5", af::DT_FLOAT)
                   .Op<af::ascir_op::Add>("add", {"two", "load0"}, 1U)
                   .Store("store", "add")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_scalar_add_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 与 `load_add_scalar_store` 逐字相同的一行：tile 在前、标量在后。
  EXPECT_NE(kernel.find("constexpr float c2_5_f32 = (float)2.500000000e+00;"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::Adds<float, 1>("), std::string::npos);
  EXPECT_NE(kernel.find("c2_5_f32, "), std::string::npos);
  // 不是 tensor-tensor 那一支——标量没有被当成一块 tile。
  EXPECT_EQ(kernel.find("AscendC::Add<"), std::string::npos);
}
}  // namespace mlir::afir
