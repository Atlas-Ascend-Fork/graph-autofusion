/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load x2 -> axpy -> store CPU case -------------*- C++ -*-===//
//
// **两件只有这里拦得住的事。**
//
// 一、`alpha` 那个数原样到得了 kernel，而且落在**第四个实参位**上。它在 AscGraph 上是
// 节点的 `ir_attr`、在 AFIR 上是 op 属性、在生成的调用里是一个 `const float` 实参——
// 三种形态，bridge 两段各有一处按具名属性表做的转换。把表里 `Axpy` 那一行去掉，
// `AfirBuilder` 就不挂这个属性，ODS 的 `F32Attr` 缺了当场报错；位置写错（`axpy.h` 的
// 签名是 `dst, src0, src1, alpha, tmp_buf, count`，tmp 在 count 前面、alpha 在 tmp
// 前面）则编得过而算错——`load_axpy_store` 的期望值立刻对不上。
//
// 二、tier D1 真的落地了：两套 API 家族共用同一个包装 `AxpyExtend`（`axpy.h`），
// `api_regbase/` 下根本没有 axpy。harness 把 soc 钉在 Ascend910B1，但这条路上
// **没有第二个 implementation 可选**——`ApiFamily::Shared` 那一档说的就是这件事。
// 两个 target 各自选中同一个 id 由 afir-axpy-extension-api.mlir 钉。
//
// 取 f32：`REG_ASC_IR(Axpy)` 的 `T` 是 {f16, f32}。**f16 那一支在这里走不到**——
// 包装里只有 f16 分支真的用那块 tmp buffer（切两段做 `Cast` / `CAST_RINT`），f32 走的
// `else` 分支一次没用它，形参在签名里只是少一个实参就调不上。所以这份用例证明的是
// 「tmp buffer 被正确传进去了」，不是「那块 tmp 被正确用了」；后者要一份 f16 的用例，
// 见进度文档 Axpy 那一节。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_axpy_store/LoadAxpyStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_axpy_store::GetShapeConfig(load_axpy_store::kOutputShapes, 0).shape;
  const auto input0_shape = ToExpressions(load_axpy_store::GetShapeConfig(load_axpy_store::kInputShapes, 0).shape);
  const auto input1_shape = ToExpressions(load_axpy_store::GetShapeConfig(load_axpy_store::kInputShapes, 1).shape);
  AscGraphBuilder builder(load_axpy_store::kKernelName);
  builder.Loops(output_shape)
      .Data("input0", 0, input0_shape, {}, af::DT_FLOAT)
      .Data("input1", 1, input1_shape, {}, af::DT_FLOAT)
      .Load("load0", "input0", input0_shape, {})
      .Load("load1", "input1", input1_shape, {});

  auto graph = builder.Axpy("axpy", "load0", "load1", load_axpy_store::kAlpha)
                   .Store("store", "axpy")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_axpy_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 裸名，不带 `AscendC::` 限定——`axpy.h` 里的包装是全局函数。
  EXPECT_NE(kernel.find("AxpyExtend("), std::string::npos);
  // 业务头被内联进来了。
  EXPECT_NE(kernel.find("__ASCENDC_API_AXPY_H__"), std::string::npos);
  // alpha 作为一个 **f32** 常量出现在 kernel 里，不跟 tile 的元素类型走——包装的形参是
  // `const float alpha`。`5.000000000e-01` 就是 0.5：emitter 打的是定长的科学计数形式，
  // legacy 那边打的是 `(float)0.500000`，两者只是写法不同。
  EXPECT_NE(kernel.find("constexpr float"), std::string::npos);
  EXPECT_NE(kernel.find("(float)5.000000000e-01"), std::string::npos);
  // 包装要一块 tmp buffer（f32 这一支其实不用它，但形参在签名里）。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
  // 共用的是 `api/` 那一份，不存在 reg-base 的对应物。
  EXPECT_EQ(kernel.find("axpy_reg_base.h"), std::string::npos);
  // `axpy.h` 不用 `MAX_REPEAT_TIME` / `Rational`，所以**不**声明 `utils.h` 前导。
  // 多声明一个前导头不会报错、只会白白内联一份文本，这一条挡的是那个。
  EXPECT_EQ(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
}
}  // namespace mlir::afir
