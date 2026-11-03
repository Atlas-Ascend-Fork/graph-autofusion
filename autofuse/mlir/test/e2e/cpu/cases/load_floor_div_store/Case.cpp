/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x2 -> floor_div -> store CPU case --------*- C++ -*-===//
//
// **第一个把扩展头的前导也一起编出来的二元用例。** `floor_div.h` 用了 `MAX_REPEAT_TIME`
// ——那是 fork 自己的常量，只有 `utils.h` 声明。前导声明漏了的话 lit 不会红（它比对调用
// 文本，不编译内联进来的头），要靠这条 E2E 才编得炸。`where.h` 当初就是这么漏过去的。
//
// harness 把 soc 钉在 Ascend910B1（cpu/CMakeLists.txt），走 V1 也就是 mem-based 一侧：
// `FloorDivExtend(dst, x1, x2, tmp_buf, size)`，扩展头 `floor_div.h`，要 tmp buffer。
// reg 侧那份同名但不收 tmp，在这里到不了，由
// afir-floor-div-reg-base-extension-api.mlir 钉。
//
// 取 f32：mem 侧 legacy 的 `REG_ASC_IR(FloorDiv)` 只注册了 f16 / f32，整数那一半只有
// reg 侧收，这条路上走不到。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_floor_div_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_floor_div_store::GetShapeConfig(load_floor_div_store::kOutputShapes, 0).shape;
  const auto input0_shape =
      ToExpressions(load_floor_div_store::GetShapeConfig(load_floor_div_store::kInputShapes, 0).shape);
  const auto input1_shape =
      ToExpressions(load_floor_div_store::GetShapeConfig(load_floor_div_store::kInputShapes, 1).shape);
  AscGraphBuilder builder(load_floor_div_store::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, input1_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  auto graph = builder.FloorDiv("floor_div", "load0", "load1")
                   .Store("store", "floor_div")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_floor_div_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 包装，不是标准调用：全局函数、没有 `AscendC::` 限定。
  EXPECT_NE(kernel.find("FloorDivExtend("), std::string::npos);
  // 扩展头与它的前导都被内联进来了。少了 `utils.h` 这份 kernel 编不过（MAX_REPEAT_TIME）。
  EXPECT_NE(kernel.find("__ASCENDC_API_FLOOR_DIV_H__"), std::string::npos);
  EXPECT_NE(kernel.find("MAX_REPEAT_TIME"), std::string::npos);
  EXPECT_NE(kernel.find("using namespace AscendC;"), std::string::npos);
}
}  // namespace mlir::afir
