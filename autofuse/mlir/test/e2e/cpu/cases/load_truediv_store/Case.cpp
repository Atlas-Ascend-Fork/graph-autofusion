/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x2 -> truediv -> store CPU case ----------*- C++ -*-===//
//
// **输入 i32、输出 f32。取这一档不是为了换个 dtype 试试——那是 `afir.truediv` 唯一
// 不同于 `afir.div` 的地方。** `true_div.h` 的 `TrueDivExtend` 里，`else` 分支字面上
// 就是 `AscendC::Div(dst, src1, src2, size)`，与 `Div` 的 mem 侧发的调用等价；只有
// `if constexpr (IsSameType<T, int32_t>)` 那一支不同：先把两个 src `Cast` 成 f32
// （`CAST_RINT`）再除。拿 f32 写这份用例的话，`truediv` 全套就是 `div` 的逐字别名，
// 用例全绿而覆盖面一个没涨。
//
// **三件只有这里拦得住的事。**
//
// 一、`utils.h` 前导。`scalar_div.h` 用了 `MAX_REPEAT_TIME` 与 `KernelUtils::BlkSize` /
// `RptSize`，只有 `utils.h` 声明。`TrueDivExtends` 这条路今天发不出来，但头是当作文本
// 内联进来的、模板里那些名字又不依赖模板形参，少了前导当场编不过。**lit 一条都测不
// 出来**——IR 建得出来、`ascir-translate` 也照打那行调用。`where.h` / `sign.h` /
// `logical.h` / `floor_div.h` 都栽过。
//
// 二、**两个扩展头一起进来，而且次序对**。`true_div.h` 的标量重载调 `Divs`，那是
// `scalar_div.h` 定义的；kernel source 层按字典序发 include，恰好是这个次序。
// 前置 A 的形状在二元这边到这里才第一次兑现。
//
// 三、i32→f32 那一行真的落地了。图上两头 dtype 不同，调度、bridge、capability 三处
// 任何一处把它当成同 dtype，这份用例在跑之前就建不出图。
//
// harness 把 soc 钉在 Ascend910B1，走 mem 侧。reg 侧发的是 `DivExtend`
// （`div_reg_base.h`，与 `afir.div` 的 reg 侧同一个调用）且不收 i32，在这里到不了，
// 由 afir-truediv-reg-base-extension-api.mlir 钉。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_truediv_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_truediv_store::GetShapeConfig(load_truediv_store::kOutputShapes, 0).shape;
  const auto input0_shape =
      ToExpressions(load_truediv_store::GetShapeConfig(load_truediv_store::kInputShapes, 0).shape);
  const auto input1_shape =
      ToExpressions(load_truediv_store::GetShapeConfig(load_truediv_store::kInputShapes, 1).shape);
  AscGraphBuilder builder(load_truediv_store::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_INT32)
      .Data("input1", 1, input1_shape, {}, af::DT_INT32)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  // 输出是 `DT_FLOAT`，两个输入是 `DT_INT32`——`kTrueDivTypePairs` 的 i32 那一行。
  // builder 里的 `TrueDiv` 就是按这条对表把继承来的 dtype 改掉的。
  auto graph = builder.TrueDiv("truediv", "load0", "load1")
                   .Store("store", "truediv")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_truediv_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 裸名，不带 `AscendC::` 限定——`true_div.h` 里的包装是全局函数。
  EXPECT_NE(kernel.find("TrueDivExtend("), std::string::npos);
  // 两个业务头都被内联进来了。
  EXPECT_NE(kernel.find("__ASCENDC_API_TRUE_DIV_H__"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_SCALAR_DIV_H__"), std::string::npos);
  // **前导头也被内联进来了。** 把 td 上那个 `["utils.h"]` 删掉，lit 一条都不红，
  // 这里先编不过、再红在这一行。
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
  // 包装要一块 tmp buffer，i32 这一支真的用它（切两段装两份 cast 结果）。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
  // 进来的是 mem 那一份：reg 侧发的是 `DivExtend`（`div_reg_base.h`），与 `afir.div`
  // 的 reg 侧同一个调用。**不能按 `DivExtend(` 去找**——`TrueDivExtend(` 含着它。
  EXPECT_EQ(kernel.find("__ASCENDC_REGBASE_API_DIV_H__"), std::string::npos);
}
}  // namespace mlir::afir
