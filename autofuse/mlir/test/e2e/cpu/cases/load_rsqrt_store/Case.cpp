/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> rsqrt -> store CPU case ---------------*- C++ -*-===//
//
// harness 把 soc 钉在 Ascend910B1（cpu/CMakeLists.txt），走 V1 也就是 mem-based 一侧：
// `RsqrtExtend(dst, src, blk_tensor_with_value_1, tmp_buf, size)`，扩展头 `rsqrt.h`。
// reg 侧是标准调用 `AscendC::Rsqrt`，在这里到不了，由 afir-rsqrt-stock-api.mlir 钉。
//
// 这个用例真正拦的是那块预先填好 1 的常量 block。 它在 AFIR 上不存在——不是操作数、
// 不是 tmp buffer、调度也不知道有它——整块都是 lowering 在 kernel 序言里现造的
// （`Implementations/ConstBlockTensor.cpp`）。三种错法只有真去编 kernel、真去算数才拦得住：
//
//   - 那一块漏了：包装少一个实参，device 编译器报重载对不上；
//   - 那一块造了但没 `Duplicate`：编得过，算出来的是 `未初始化 / sqrt(x)`，数值全错；
//   - `Duplicate` 与包装之间少一条 `PipeBarrier<PIPE_V>`：两条都在 V 上，是一次 RAW
//     竞争，CPU 模拟器上未必每次都露出来，但它是真的。
//
// 前导头 `utils.h` 也在这条路上：`rsqrt.h` 用了只有它声明的 `MAX_REPEAT_TIME`。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_rsqrt_store/LoadRsqrtStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_rsqrt_store::GetShapeConfig(load_rsqrt_store::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(load_rsqrt_store::GetShapeConfig(load_rsqrt_store::kInputShapes, 0).shape);
  AscGraphBuilder builder(load_rsqrt_store::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph =
      builder.Rsqrt("rsqrt", "load0").Store("store", "rsqrt").Output("output", "store", 0, af::DT_FLOAT).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_rsqrt_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 包装，不是标准调用：全局函数、没有 `AscendC::` 限定。
  EXPECT_NE(kernel.find("RsqrtExtend<float>("), std::string::npos);
  // 序言里那块常量：一条 `Duplicate`，紧跟一条 `PipeBarrier<PIPE_V>`。
  EXPECT_NE(kernel.find("AscendC::Duplicate("), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::PipeBarrier<PIPE_V>();"), std::string::npos);
  // 内联进来的是 mem 侧那份头。
  EXPECT_NE(kernel.find("__ASCENDC_API_RSQRT_H__"), std::string::npos);
  // 前导头：`rsqrt.h` 用的 `MAX_REPEAT_TIME` 只有它声明。
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
}
}  // namespace mlir::afir
