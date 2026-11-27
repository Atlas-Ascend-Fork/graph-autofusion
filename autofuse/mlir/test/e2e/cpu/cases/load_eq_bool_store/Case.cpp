/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x2 -> eq(bool) -> store CPU case --------*- C++ -*-===//
//
// 比较结果是 `DT_BOOL` 的那一种图。`load_eq_store` 钉的是 mem 侧比较调用的形状，但它的图
// 把比较输出手写成 `DT_UINT8`；真实前端给的比较结果是 bool，而 bool 不在 AFIR 的供给表上
// ——bridge 得在比较、Store、Output 三处把它按 ui8 建，否则这张图整张回落 legacy。
//
// 归一化本身由 `ScheduleToAfirTest` 那几条 UT 钉（哪几位建成了 ui8、范围外的 bool 仍
// 回落）。E2E 在这里补的是 UT 钉不到的两件事：
//
// 一、整条流水线走得通。UT 止于 AFIR 模块；这里还要过实现选择、内存分配、下降与 kernel
//    编译——归一化要是漏了哪一处（比如 kernel 签名仍按 bool 生成），到这里才红。
// 二、数值。bool 的比较结果经 ui8 存回 GM 是 0 / 1，逐元素比对。
//
// harness 把 soc 钉在 Ascend910B1，走 mem 侧。
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_eq_bool_store/LoadEqBoolStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace config = load_eq_bool_store;

  const auto output_shape = config::GetShapeConfig(config::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 0).shape);
  const auto input1_shape = ToExpressions(config::GetShapeConfig(config::kInputShapes, 1).shape);
  AscGraphBuilder builder(config::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, input1_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  // 比较结果、Store 的输出与图输出三处都是 `DT_BOOL`。`builder.Eq(...)` 把输出写死成
  // `DT_UINT8`，说不出这一种图，所以走带 dtype 的那个通道。
  builder.OpWithDtype<af::ascir_op::Eq>("eq", {"load0", "load1"}, 0U, af::DT_BOOL);
  auto graph = builder.Store("store", "eq").Output("output", "store", 0, af::DT_BOOL).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_eq_bool_store;
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // kernel 输出按一个字节的 ui8 生成，不是 bool。签名整段写死：workspace 那一位也是
  // `__gm__ uint8_t*`，只找一个 `__gm__ uint8_t*` 分不清输出到底是不是它。
  EXPECT_NE(kernel.find("(__gm__ float* v1, __gm__ float* v2, __gm__ uint8_t* v3, __gm__ uint8_t* v4, "
                        "AutofuseTilingData t)"),
            std::string::npos);
  EXPECT_EQ(kernel.find("__gm__ bool*"), std::string::npos);
  // 走的仍是 mem 侧那一条比较调用（同 `load_eq_store`）。
  EXPECT_NE(kernel.find("CMPMODE::EQ, static_cast<uint32_t>("), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_REGBASE_COMPARE_H__"), std::string::npos);
}
}  // namespace mlir::afir
