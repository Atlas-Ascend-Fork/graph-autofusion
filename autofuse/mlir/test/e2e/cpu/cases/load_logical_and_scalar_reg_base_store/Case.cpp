/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> logical_and(tile, scalar) -> store (reg-base) ------===//
//
// 逻辑族标量形态在 reg 侧的唯一一条路：`y = and(x, 1)`，输入 f32、输出 ui8。
//
// `logical_or` 在 reg 侧没有标量形态的 implementation（`logical_and` 有，见
// `BinaryElementwise.cpp` 的注册段），所以这一类在这边就这一个。两侧差的不只是头：
// mem 侧的 `LogicalAndScalarExtend` 要一块 tmp（把两个输入 `Cast` 成 half 落在里面），
// reg 侧的 `LogicalAndExtends` 不要。
//
// **这条用例钉的是「reg 侧这个重载编得过、跑得动，而且标量真的以 1 参与了运算」，
// 不是「与和或分得开」**：包装假定输入是 0 / 1 真值（喂 4 进去出来是 4），标量那一侧
// 因此只有 0 和 1 两个取值可选，表达不出那个区分。and / or 的区分由
// `load_logical_and_store` / `load_logical_or_store` 两条 tensor 形态的用例管。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_logical_and_scalar_reg_base_store/LoadLogicalAndScalarRegBaseStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_logical_and_scalar_reg_base_store;

  const auto output_shape = cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  // 输入 f32、标量 f32，输出恒为 ui8。
  auto graph = builder.Scalar("one", "1.0", af::DT_FLOAT)
                   .LogicalAnd("logical_and", "load0", "one")
                   .Store("store", "logical_and")
                   .Output("output", "store", 0, af::DT_UINT8)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace cfg = load_logical_and_scalar_reg_base_store;
  EXPECT_NE(kernel.find(std::string(cfg::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 字面量写全：emitter 打的是定长科学计数。标量按 tile 的元素类型建（f32），
  // 不是按结果的 ui8。
  EXPECT_NE(kernel.find("constexpr float c1_f32 = (float)1.000000000e+00;"), std::string::npos);
  // 包装是全局函数、没有模板实参；标量是第三个实参，count 跟在后面、没有 tmp。
  EXPECT_NE(kernel.find("LogicalAndExtends("), std::string::npos);
  EXPECT_NE(kernel.find(", c1_f32, static_cast<uint32_t>("), std::string::npos);
  // 内联进来的是 reg 那一份；mem 侧那个头不该在。
  EXPECT_NE(kernel.find("__ASCENDC_API_REGBASE_LOGICAL_H__"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_LOGICAL_H__"), std::string::npos);
  // reg 侧这份包装不要 tmp buffer，也不声明前导头——两条都与 mem 侧相反
  // （mem 侧的 `logical.h` 用了 `MAX_REPEAT_TIME`，非要 `utils.h` 不可）。
  EXPECT_EQ(kernel.find("AscendC::TBuf<"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
}
}  // namespace mlir::afir
