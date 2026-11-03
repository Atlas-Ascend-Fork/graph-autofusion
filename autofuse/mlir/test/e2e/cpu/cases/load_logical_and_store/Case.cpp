/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x2 -> logical_and -> store CPU case ------*- C++ -*-===//
//
// **两件只有这里拦得住的事。**
//
// 一、`utils.h` 前导。`logical.h` 的三条 `*LogicalCommon` 都用了 `MAX_REPEAT_TIME`，
// 那是 fork 自己的常量，只有 `utils.h` 声明。legacy 不写这条是因为它走
// `IncludeAndDefines` 无条件内联；MLIR 这边得在 td 上自己说出来。**少了它 lit 全绿**
// ——IR 建得出来、`ascir-translate` 也照打那行调用，只有真去编 kernel 才炸。
// `where.h` 与 `sign.h` 当初都栽在这条上。
//
// 二、输入与输出不同 dtype 真的落地了。这一对是二元一族里第一个结果 dtype 与输入无关
// 的 op：输入 f32，输出 ui8。图上两头的 dtype 不同，调度、bridge、capability 三处
// 任何一处把它当成同 dtype，这份用例在跑之前就建不出图。
//
// harness 把 soc 钉在 Ascend910B1，走 mem 侧：`LogicalAnd(dst, x1, x2, tmp, size)`，
// 扩展头 `logical.h`，要 tmp buffer 而且**真的用**（把两个输入 `Cast` 成 half 落在
// 里面）。reg 侧的 `LogicalAndExtend` 在这里到不了，由
// afir-logical-reg-base-extension-api.mlir 钉。
//
// 取 f32：mem 侧的 `T1` 收 {ui8, i16, i32, f16, f32}，挑一个非 ui8 的才看得出结果
// dtype 没有跟着输入走。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_logical_and_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_logical_and_store::GetShapeConfig(load_logical_and_store::kOutputShapes, 0).shape;
  const auto input0_shape =
      ToExpressions(load_logical_and_store::GetShapeConfig(load_logical_and_store::kInputShapes, 0).shape);
  const auto input1_shape =
      ToExpressions(load_logical_and_store::GetShapeConfig(load_logical_and_store::kInputShapes, 1).shape);
  AscGraphBuilder builder(load_logical_and_store::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, input1_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  // 输出是 `DT_UINT8`，两个输入是 `DT_FLOAT`——这份用例的一半意义就在这一行不一致上。
  auto graph = builder.LogicalAnd("logical_and", "load0", "load1")
                   .Store("store", "logical_and")
                   .Output("output", "store", 0, af::DT_UINT8)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_logical_and_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 裸名，不带 `AscendC::` 限定——`logical.h` 里的包装是全局函数。**名字上没有 `Extend`
  // 后缀**，这是与 reg 侧那份分得开的地方。
  EXPECT_NE(kernel.find("LogicalAnd("), std::string::npos);
  // 业务头被内联进来了。
  EXPECT_NE(kernel.find("__ASCENDC_API_LOGICAL_H__"), std::string::npos);
  // **前导头也被内联进来了。** 这一条就是这份用例的第一个理由：把 td 上那个
  // `["utils.h"]` 删掉，lit 一条都不红，这里先编不过、再红在这一行。
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
  // 包装要 tmp buffer，而且真的用它。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
  // 进来的是 mem 那一份，不是 reg 那一份。
  EXPECT_EQ(kernel.find("__ASCENDC_API_REGBASE_LOGICAL_H__"), std::string::npos);
  EXPECT_EQ(kernel.find("LogicalAndExtend("), std::string::npos);
}
}  // namespace mlir::afir
