/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 验证第三种标量来源 IndexExpr，与 Scalar 字面量和 ScalarData 按值参数对照。
// IndexExpr 根据 size var 在 kernel 顶部计算，不增加 kernel 参数，也不引用归纳变量。
// legacy 将它加入 constant_tensors，并在 root_loop.Generate 前物化；替换表仅包含
// TilingData 的 size var，不包含轴变量。
// 符号尺寸版本覆盖带自由符号表达式的 materializeRuntimeSize，固定尺寸版本用于对照。
// 语义依据见 L-9 的 IndexExpr 说明。

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_add_index_expr_store/LoadAddIndexExprStoreCase.h"
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

  // 主判据：它不占 kernel 形参。 与 `load_add_scalar_data_store` 那条恰好相反——
  // 那边标量夹在两个指针中间，这边签名里压根没有它。生成形参的话这一行就对不上。
  EXPECT_NE(kernel.find(std::string("__global__ __aicore__ void ") + cfg::kKernelName +
                        "(__gm__ float* v1, __gm__ float* v2, __gm__ uint8_t* v3, AutofuseTilingData t)"),
            std::string::npos);
  EXPECT_NE(kernel.find("extern \"C\" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, void* input0, "
                        "void* output0, void* workspace, AutofuseTilingData* tiling_data)"),
            std::string::npos);

  // 值折成了一个有类型的常量。字面量写全——emitter 打的是定长科学计数，写成
  // `(float)2.0` 会匹配上前缀，值改成 2.0e+01 也照样绿。
  EXPECT_NE(kernel.find("constexpr float c2_f32 = (float)2.000000000e+00;"), std::string::npos);

  // 走的是 tensor_scalar 形态，与另外两种标量源生成的是同一个调用。
  EXPECT_NE(kernel.find("AscendC::Adds<float, 1>("), std::string::npos);
  EXPECT_EQ(kernel.find("AscendC::Add<"), std::string::npos);
  EXPECT_NE(kernel.find("c2_f32, "), std::string::npos);
}
}  // namespace mlir::afir
