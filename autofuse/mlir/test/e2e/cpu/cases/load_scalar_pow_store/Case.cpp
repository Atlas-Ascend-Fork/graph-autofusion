/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> pow(scalar, tile) -> store CPU case ----*- C++ -*-===//
//
// 位置原样保留那一类（规则三）里二元的那一个：`y = 2.0 ^ x`，标量在左。
// 三元那一侧是 `load_clip_by_value_scalar_store`。
//
// 这一类没有承载方向的模板实参：`pow.h` 有四个同名重载，(t,t) / (t,s) / (s,t) /
// (s,s)，标量在哪一位由 C++ 重载决议挑。所以这条用例钉的是实参位置，而位置错了
// 没有任何编译期后果——`Pow(dst, tile, 标量, tmp, cnt)` 与 `Pow(dst, 标量, tile, tmp, cnt)`
// 都是合法调用，都编得过，只是算的不是同一个函数。
//
// 数值断言是这条用例的全部理由：标量在左生成标量在右，算的就是 `x ^ 2` 而不是
// `2 ^ x`。规则一（可交换的八个）那一类交换完生成的调用逐字相同，所以那边没有这个风险；
// `pow` 不可交换，而它又不像 `sub` / `div` 那样有一位 bool 可以钉——生成的文本在
// lit 那一层看起来完全正常。
//
// 顺带走到 `pow.h` 的第三个重载：它转发的是 `Power(dst, T src1, LocalTensor src2, ...)`，
// 与 `load_pow_store` 走的第一个重载不是同一个 `Power` 形态。
//
// 底数固定 2.0、指数取 [-3, 3] 的整数：`2 ^ x` 与 `x ^ 2` 在这七个点上只有 x = 2 相同。
// 设备上算的是 `exp(x * ln 2)`，所以比的是相对量，与 `load_pow_store` 同一种比较方式。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_scalar_pow_store/LoadScalarPowStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_scalar_pow_store;

  const auto output_shape = cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  // 标量是底数、在前；tile 是指数、在后。输出属性跟第二个输入（follow_index = 1）。
  auto graph = builder.Scalar("two", "2.0", af::DT_FLOAT)
                   .Op<af::ascir_op::Pow>("pow", {"two", "load0"}, 1U)
                   .Store("store", "pow")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace cfg = load_scalar_pow_store;
  EXPECT_NE(kernel.find(std::string(cfg::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 字面量写全：emitter 打的是定长科学计数。
  EXPECT_NE(kernel.find("constexpr float c2_f32 = (float)2.000000000e+00;"), std::string::npos);
  // 标量在第二个实参位，也就是 AFIR 上的原次序。`Pow(` 后面第一个逗号之后必须就是
  // 它——这是「没有把 tile 挪到第一位」唯一看得见的地方。
  //
  // 从最后一处 `Pow(` 起算：`pow.h` 整段内联进 kernel，前面四处是那四个重载的定义
  // （`Pow(const AscendC::LocalTensor<T> &dst, ...`），调用点在 kernel 体里、排在它们
  // 之后。内部转发的是 `Power(`，不含 `Pow(`，所以数不进来。
  const size_t call = kernel.rfind("Pow(");
  ASSERT_NE(call, std::string::npos);
  const size_t firstComma = kernel.find(',', call);
  ASSERT_NE(firstComma, std::string::npos);
  EXPECT_EQ(kernel.compare(firstComma + 1, std::string(" c2_f32,").size(), " c2_f32,"), 0)
      << kernel.substr(call, kernel.find(';', call) - call);
  // 内联进来的是 mem 侧那份头，不是 reg 侧那份。两份的 include guard 撞名，所以 reg
  // 那一侧改钉它独有的 `pow_config`。前导 `utils.h` 也在——`pow.h` 用的 `KernelUtils::`
  // 只有它声明。
  EXPECT_NE(kernel.find("__ASCENDC_API_POW_H__"), std::string::npos);
  EXPECT_EQ(kernel.find("PowerAlgo::DOUBLE_FLOAT_TECH"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
  // 两侧的 `pow` 都要 tmp buffer。TBuf 只会因为 tmp buffer 出现。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
