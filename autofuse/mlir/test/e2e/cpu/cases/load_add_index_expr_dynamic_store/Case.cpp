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
// **覆盖到哪为止**：这条用例是套件里唯一有 size var 的一条（其余用例的 `Loops` 收的是
// 字面量，`AscGraphBuilder` 把它们建成 `Symbol(n)`，图上没有符号），所以带自由符号的
// 表达式只在这里走得到 `materializeRuntimeSize`。见 L-9 的 `IndexExpr` 一节。
//
//===----------------------------------------------------------------------===//

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_add_index_expr_dynamic_store/LoadAddIndexExprDynamicStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;
  namespace cfg = load_add_index_expr_dynamic_store;

  const auto input_shape = cfg::GetShapeConfig(cfg::kInputShapes, 0).shape;
  AscGraphBuilder builder(cfg::kKernelName);
  // 与 load_add_index_expr_store 的唯一差别：每一维都是**运行期尺寸**，不是字面量。
  // 于是 `s0` 起的每个符号各成为 TilingData 的一个字段，表达式 `2 * s0` 里的那个自由
  // 符号有了真来源。
  //
  // **每根轴各登记一个符号，不是把元素总数塞进一个符号。** 后者会把这条用例退化成一轴，
  // 而这一族真正要钉的是「轴不参与替换」——只有一根轴时它与「没有轴」看不出区别。
  std::vector<af::Expression> extents;
  std::vector<uint32_t> runtimeExtents;
  extents.reserve(input_shape.size());
  runtimeExtents.reserve(input_shape.size());
  for (size_t axis = 0; axis < input_shape.size(); ++axis) {
    extents.push_back(builder.SizeVar("s" + std::to_string(axis)));
    runtimeExtents.push_back(static_cast<uint32_t>(input_shape.begin()[axis]));
  }
  // 运行期尺寸的值：生成的 `AutofuseTiling` 会按登记顺序多出同样多个 `uint32_t` 形参，
  // 由它们传进去。
  g_e2eRuntimeShapeArgs = runtimeExtents;
  builder.Loops(extents).Data("input0", 0, extents, {}, af::DT_FLOAT).Load("load0", "input0", extents, {});

  auto graph = builder.IndexExpr("k", af::Expression::Parse("2 * s0"), af::DT_FLOAT)
                   .Add("add", "load0", "k")
                   .Store("store", "add")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  namespace cfg = load_add_index_expr_dynamic_store;
  EXPECT_NE(kernel.find(std::string(cfg::kKernelName) + "_subgraph0_plan0"), std::string::npos);

  // **主判据：表达式在运行期从 TilingData 读出来算，不是折成常量。** 折掉的话下面这条
  // 读字段的语句就没有了，而结果照样"对"——因为这条用例的 s0 恰好只取一个值。所以
  // 两条判据一起上：既要有运行期读取，又不能出现把它烧死的常量。
  //
  // 烧死的那个值按形状算，不写字面量：形状一改字面量就悄悄失配，这条负例会从「没找到」
  // 变成「永远没找到」，也就永远绿。
  const auto shape = cfg::GetShapeConfig(cfg::kInputShapes, 0).shape;
  const std::string foldedConstant = "constexpr float c" + std::to_string(2 * shape.begin()[0]);
  EXPECT_NE(kernel.find(".s0;"), std::string::npos);
  EXPECT_EQ(kernel.find(foldedConstant), std::string::npos);

  // index → i64 → float 两次转换。少一次或写反都还编得过，只是算出另一个数。
  EXPECT_NE(kernel.find("static_cast<float>("), std::string::npos);

  // 它**不占 kernel 形参**——与 ScalarData 的分界。
  EXPECT_NE(kernel.find(std::string("__global__ __aicore__ void ") + cfg::kKernelName +
                        "(__gm__ float* v1, __gm__ float* v2, __gm__ uint8_t* v3, AutofuseTilingData v4)"),
            std::string::npos);

  EXPECT_NE(kernel.find("AscendC::Adds<float, 1>("), std::string::npos);
  EXPECT_EQ(kernel.find("AscendC::Add<"), std::string::npos);
}
}  // namespace mlir::afir
