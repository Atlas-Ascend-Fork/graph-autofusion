/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> add(IndexExpr) -> store CPU case ------*- C++ -*-===//
//
// **`IndexExpr` 的第一个用例。** 与同目录的 `load_add_scalar_store` /
// `load_add_scalar_data_store` 是同一族的第三种标量源：
//
//   | 节点 | 值从哪来 | 落到 kernel 上 |
//   |---|---|---|
//   | `Scalar` | 编译期字面量 | `constexpr float c2_5_f32` |
//   | `ScalarData` | 主机运行期传入 | 一个按值传的 kernel 形参 |
//   | `IndexExpr` | 由 size var 算出来的表达式 | **在 kernel 顶部算出来的常量** |
//
// 本用例钉的是第三行：`IndexExpr` **不产生 kernel 形参**，签名与不含它的图一模一样。
//
// **这条路上没有归纳变量。** legacy 把三者一起收进 `kernel.constant_tensors`，由
// `AppendConstTensorInit` 在 kernel 顶部物化，而那一处跑在 `root_loop.Generate`
// 之前；渲染只把 size var 换成 TilingData 字段，轴不在替换表里。
//
// **覆盖到哪为止**：这条用例的表达式是常量——`AscGraphBuilder::Loops` 收字面量时建的是
// `Symbol(n)`，图上没有 size var。带自由符号的那一档由同族的
// `load_add_index_expr_dynamic_store` 覆盖，它每根轴各登记一个符号。见 L-9 的
// `IndexExpr` 一节。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_add_index_expr_store/Case.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_add_index_expr_store;

  const auto output_shape = cfg::GetShapeConfig(cfg::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(cfg::GetShapeConfig(cfg::kInputShapes, 0).shape);
  AscGraphBuilder builder(cfg::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph = builder.IndexExpr("k", af::Expression::Parse("2"), af::DT_FLOAT)
                   .Add("add", "load0", "k")
                   .Store("store", "add")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace cfg = load_add_index_expr_store;
  EXPECT_NE(kernel.find(std::string(cfg::kKernelName) + "_subgraph0_plan0"), std::string::npos);

  // **主判据：它不占 kernel 形参。** 与 `load_add_scalar_data_store` 那条恰好相反——
  // 那边标量夹在两个指针中间，这边签名里压根没有它。发成形参的话这一行就对不上。
  EXPECT_NE(kernel.find(std::string("__global__ __aicore__ void ") + cfg::kKernelName +
                        "(__gm__ float* v1, __gm__ float* v2, __gm__ uint8_t* v3, AutofuseTilingData v4)"),
            std::string::npos);
  EXPECT_NE(kernel.find("extern \"C\" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, void* input0, "
                        "void* output0, void* workspace, AutofuseTilingData* tiling_data)"),
            std::string::npos);

  // 值折成了一个有类型的常量。**字面量写全**——emitter 打的是定长科学计数，写成
  // `(float)2.0` 会匹配上前缀，值改成 2.0e+01 也照样绿。
  EXPECT_NE(kernel.find("constexpr float c2_f32 = (float)2.000000000e+00;"), std::string::npos);

  // 走的是 tensor_scalar 形态，与另外两种标量源发的是同一个调用。
  EXPECT_NE(kernel.find("AscendC::Adds<float, 1>("), std::string::npos);
  EXPECT_EQ(kernel.find("AscendC::Add<"), std::string::npos);
  EXPECT_NE(kernel.find("c2_f32, "), std::string::npos);
}
}  // namespace mlir::afir
