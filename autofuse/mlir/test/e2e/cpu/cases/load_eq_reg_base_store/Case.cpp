/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x2 -> eq -> store CPU case (reg-base) ----*- C++ -*-===//
//
// reg-base 一侧，钉两件 mem 侧那份钉不到的事。
//
// 一、**micro-API 那一族真的编得过**。`api_regbase/compare.h` 里 `CreateMask` /
// `UpdateMask` / `RegTensor` / `Std::` / `MicroAPI::` 全是不带限定的名字，靠
// `AddKernelIncludes` 在有扩展头时发的那条 `using namespace AscendC;` 兜着。这条链此前
// 没有任何用例走过——它是 L-13 里最不放心的一处。
//
// 二、**`TmpBufferUse::Unused` 那一档第一次看得见**。`GetCompareSizeV2` 照样给这个节点
// 分了一块 32 字节，而 reg 侧的 `CompareExtend` 签名里没有那个形参。生成的 kernel 里
// 因此有一个没人读的 `TBuf`——它不该被优化掉，ATT 的 UB 预算是按分了这块算的。
//
// 调用形状与 mem 侧**同名而完全不同**：比较模式在这边是模板实参、在那边是运行期实参，
// 三个 brace-init 数组收 `const uint16_t (&)[dim]`。写成 `{1U}` 在窄化初始化里编不过,
// 这种错只有真编一遍才拦得住。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_eq_reg_base_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_eq_reg_base_store::GetShapeConfig(load_eq_reg_base_store::kOutputShapes, 0).shape;
  const auto input0_shape =
      ToExpressions(load_eq_reg_base_store::GetShapeConfig(load_eq_reg_base_store::kInputShapes, 0).shape);
  const auto input1_shape =
      ToExpressions(load_eq_reg_base_store::GetShapeConfig(load_eq_reg_base_store::kInputShapes, 1).shape);
  AscGraphBuilder builder(load_eq_reg_base_store::kKernelName);
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
  EXPECT_NE(kernel.find(std::string(load_eq_reg_base_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 整条调用照实际输出写死。reg 侧与 mem 侧**同名而形状完全不同**：比较模式在这边是
  // 模板实参、在那边是运行期实参；三个 brace-init 数组是 `const uint16_t (&)[dim]`，
  // `dim == 1` 就是「连续形态」的全部含义。写成 `{1U}` 在窄化初始化里编不过——这一条
  // 只有真编一遍才拦得住。
  EXPECT_NE(kernel.find("CompareExtend<float, 1, CMPMODE::EQ>("), std::string::npos);
  EXPECT_NE(kernel.find("}, {static_cast<uint16_t>(1)}, {static_cast<uint16_t>(1)});"), std::string::npos);
  // 内联进来的是 reg 那一份，mem 那两个头一个都不该在。
  EXPECT_NE(kernel.find("__ASCENDC_API_REGBASE_COMPARE_H__"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_COMPARE_H__"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_COMPARE_V2_H__"), std::string::npos);
  // reg 侧这份包装不声明前导头：它用的 `CreateMask` / `RegTensor` / `Std::` /
  // `MicroAPI::` 都是不带限定的 AscendC 名字，靠 `AddKernelIncludes` 发的那条
  // `using namespace AscendC;` 兜着。**这条链此前没有任何用例走过。**
  EXPECT_EQ(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
  EXPECT_NE(kernel.find("using namespace AscendC;"), std::string::npos);
  // **`TmpBufferUse::Unused` 那一档在这里第一次看得见**：`GetCompareSizeV2` 照样分了
  // 一块，调用收不下，于是 kernel 里有一个没人读的 `TBuf`。它不该被优化掉——ATT 的 UB
  // 预算是按分了这块算的。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
