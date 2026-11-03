/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> logical_not -> store CPU case ---------*- C++ -*-===//
//
// harness 把 soc 钉在 Ascend910B1（cpu/CMakeLists.txt），走 V1 也就是 mem-based 一侧：
// `LogicalNot(dst, src, blk_tensor_with_value_1, tmp_buf, size)`，扩展头 `logical_not.h`。
//
// **两侧在这个 op 上差的不止扩展头，还有结果 dtype**：这一侧两端同 dtype（f32 进 f32
// 出），reg 侧的包装叫 `LogicalNotExtend`、结果恒为 ui8。reg 侧在这里到不了，由
// afir-logical-not-reg-base-extension-api.mlir 钉。
//
// 与 load_rsqrt_store 一起拦那块预先填好 1 的常量 block，但拦的是**另一半**：两个 op
// 要的常量 dtype 不同（`RsqrtExtend` 收 `LocalTensor<float>`，`LogicalNot` 收
// `LocalTensor<half>`），而这一块的 dtype 由包装的形参钉死、与正在算的 dtype 无关。
// 按 dst / src 的 dtype 去造那一块，这条用例是 f32 进 f32 出，会造出 float 那一块，
// device 编译器当场报重载对不上。
//
// 包装算的是 `|1 - min(1, |x|)|`，所以那块常量既是 `Min` 的左操作数也是 `Sub` 的——
// 没被 `Duplicate` 填过的话结果不是「全 0」而是一堆随机值，数值比对拦得住。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_logical_not_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_logical_not_store::GetShapeConfig(load_logical_not_store::kOutputShapes, 0).shape;
  const auto input_shape =
      ToExpressions(load_logical_not_store::GetShapeConfig(load_logical_not_store::kInputShapes, 0).shape);
  AscGraphBuilder builder(load_logical_not_store::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph = builder.LogicalNot("logical_not", "load0")
                   .Store("store", "logical_not")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_logical_not_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // **裸名**，不带 `Extend` 后缀，也不带 `AscendC::` 限定——包装是全局函数，与
  // `AscendC::LogicalNot` 同名而不同实参个数。legacy 的 `GetApiName` 写的也是这个裸名。
  EXPECT_NE(kernel.find("LogicalNot<float>("), std::string::npos);
  // 内联进来的是 mem 侧那份头，不是 reg 侧那份（`logical_not_reg_base.h` 的 guard 是
  // `__ASCENDC_API_REGBASE_LOGICAL_NOT_H`，两个 guard 不撞名）。
  EXPECT_NE(kernel.find("__ASCENDC_API_LOGICAL_NOT_H__"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_REGBASE_LOGICAL_NOT_H"), std::string::npos);
  // 序言里那块常量是 `half` 的，不是 `float` 的。
  EXPECT_NE(kernel.find("AscendC::LocalTensor<half>"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::Duplicate("), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::PipeBarrier<PIPE_V>();"), std::string::npos);
  // 前导头：`logical_not.h` 用的 `MAX_REPEAT_TIME` 只有它声明。
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
}
}  // namespace mlir::afir
