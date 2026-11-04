/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x2 -> eq -> store CPU case ---------------*- C++ -*-===//
//
// 比较族六个 op 的 e2e，只做 `eq` 一个。**这不是抽样图省事**：六个 op 在 MLIR 这一侧
// 逐字共用同一条路——同一个 `ascendc.compare_extend_contiguous`、同一个 emitter、同一
// 张元素类型集合，唯一的差别是 `CMPMODE` 那个属性，而它由「是哪个 AFIR op」定死。
// 六个属性值各自到不到得了发射端，由 afir-compare-extension-api.mlir 与它的 reg 侧
// 那一份按 `cmpMode = N` 钉。这里要的是另外两件只有真编一遍 kernel 才拦得住的事：
//
// 一、**`utils.h` 前导**。`compare.h` 用了 `MAX_REPEAT_TIME(S)` 与 `KernelUtils::` 的七
// 个成员，两者都只有 `utils.h` 声明。把 td 上那个 `["utils.h"]` 删掉，lit 一条都不红。
//
// 二、**tmp buffer 排在 count 之后**。`CompareExtend(dst, s0, s1, mode, size, tmp)` 是
// 这条路上唯一一个 tmp 不在 count 前面的包装。两个位置上都是 `LocalTensor`，写反了
// IR 与 C++ 都成立，只有算出来的数不对——下面那条逐元素断言是它唯一的检查点。
//
// harness 把 soc 钉在 Ascend910B1，走 mem 侧。reg 侧的 `CompareExtend<T, 1, CMPMODE::EQ>`
// 在这里到不了，由 afir-compare-reg-base-extension-api.mlir 钉。
//
// 取 f32 而不是 i32：i32 的 `EQ` 在包装里走的就是通用那一支（`CompareExtendInt32` 只
// 接 GT/GE/LE/LT/NE），f32 与它同一条路而且不必绕开那条 `ASSERT`。输入 f32、输出 ui8,
// 两头 dtype 不同这件事与 `load_logical_and_store` 同形，那边已经钉过。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_eq_store/LoadEqStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_eq_store::GetShapeConfig(load_eq_store::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(load_eq_store::GetShapeConfig(load_eq_store::kInputShapes, 0).shape);
  const auto input1_shape = ToExpressions(load_eq_store::GetShapeConfig(load_eq_store::kInputShapes, 1).shape);
  AscGraphBuilder builder(load_eq_store::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, input1_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  // 输出是 `DT_UINT8`，两个输入是 `DT_FLOAT`。
  auto graph =
      builder.Eq("eq", "load0", "load1").Store("store", "eq").Output("output", "store", 0, af::DT_UINT8).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_eq_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // **不能拿 `CompareExtend(` 当正面证据**：`compare.h` 的定义本身就含着它，而整段头
  // 文本是内联进 kernel 的。钉的是 emitter 自己打出来的那一段——第四个实参位上的
  // `CMPMODE::EQ` 加后面那个 `static_cast<uint32_t>`，头里一处都没有。
  EXPECT_NE(kernel.find("CMPMODE::EQ, static_cast<uint32_t>("), std::string::npos);
  // 两个业务头都被内联进来了。连续形态只调得到 `compare.h` 里那个重载，但供给表上这
  // 一族的记录是两个头，legacy 也是两个都发。
  EXPECT_NE(kernel.find("__ASCENDC_API_COMPARE_H__"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_COMPARE_V2_H__"), std::string::npos);
  // **前导头也被内联进来了。** 这一条就是这份用例的第一个理由。
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
  // 包装要 tmp buffer，而且真的用它。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
  // 进来的是 mem 那一份，不是 reg 那一份。
  EXPECT_EQ(kernel.find("__ASCENDC_API_REGBASE_COMPARE_H__"), std::string::npos);
}
}  // namespace mlir::afir
