/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> eq(tile, scalar) -> store (reg-base) ---*- C++ -*-===//
//
// 比较族标量形态在 reg 侧的第一条 E2E：`y = (x == 2.0)`，输出 ui8。
//
// 与 `load_eq_reg_base_store` 成对：那一条走的是 tensor-tensor 的
// `CompareExtendContiguousRegBaseOp`，这一条走的是**另一个 def**
// `CompareScalarExtendContiguousRegBaseOp`。同一个头里的两个函数，`CompareScalarExtend`
// 此前一个用例都没有——上一轮 `clipbyvalue.h` 缺一个实参序的重载时，六份 lit 全绿、
// 只有 device 编译器报 `no matching function`，这一类的重载表就是这么兜住的。
//
// 六个比较 op 共用这一个 def，`CMPMODE` 由节点类型定死，所以一条 `eq` 就够；模式与 op
// 的对应关系由两侧的 lit 各钉三个。
//
// 这一条也是 `ScalarOperand::Any` 在 reg 侧第一次跑到：标量放在右边，生成的
// 实参序就是 (dst, tile, 标量, ...)。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_eq_scalar_reg_base_store/LoadEqScalarRegBaseStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_eq_scalar_reg_base_store;

  const auto output_shape = cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  // 输入 f32、标量 f32，输出恒为 ui8。
  auto graph = builder.Scalar("two", "2.0", af::DT_FLOAT)
                   .Eq("eq", "load0", "two")
                   .Store("store", "eq")
                   .Output("output", "store", 0, af::DT_UINT8)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace cfg = load_eq_scalar_reg_base_store;
  EXPECT_NE(kernel.find(std::string(cfg::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 字面量写全：emitter 打的是定长科学计数。
  EXPECT_NE(kernel.find("constexpr float c2_f32 = (float)2.000000000e+00;"), std::string::npos);
  // 整条调用照实际输出写死。与 tensor 形态的 `CompareExtend` 不是同一个函数：多了
  // 一个 `Scalar`，第三个实参是标量而不是 tile；三个 brace-init 数组仍然是
  // `const uint16_t (&)[dim]`，`dim == 1` 就是连续形态。
  EXPECT_NE(kernel.find("CompareScalarExtend<float, 1, CMPMODE::EQ>("), std::string::npos);
  EXPECT_NE(kernel.find(", c2_f32, {static_cast<uint16_t>("), std::string::npos);
  EXPECT_NE(kernel.find("}, {static_cast<uint16_t>(1)}, {static_cast<uint16_t>(1)});"), std::string::npos);
  // 内联进来的是 reg 那一份，mem 那两个头一个都不该在。
  EXPECT_NE(kernel.find("__ASCENDC_API_REGBASE_COMPARE_H__"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_COMPARE_H__"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_COMPARE_V2_H__"), std::string::npos);
  // micro-API 那一类靠 `AddKernelIncludes` 生成的这条 using 兜着不带限定的名字。
  EXPECT_NE(kernel.find("using namespace AscendC;"), std::string::npos);
  // `TmpBufferUse::Unused`：`GetCompareSizeV2` 照样分了一块，调用收不下。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
