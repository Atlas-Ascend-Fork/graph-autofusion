/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> reciprocal -> store CPU case ----------*- C++ -*-===//
//
// 这个用例存在的理由只有一个：证明内联进 kernel 的扩展头文本编译得过。
//
// LIT 只看 emitter 打印出来的调用文本，UT 只看头文件名在支持表里取得到——两层都不会
// 编译内联进来的那段头。`reciprocal.h` 是第一个不自足的扩展头：它用不带 `AscendC::`
// 限定的 `LocalTensor` / `BinaryRepeatParams` / `ONE_REPEAT_BYTE_SIZE`，还用了只在
// `utils.h` 里的 `KernelUtils` / `MAX_REPEAT_TIME`。在 op 声明前导头之前，这些名字一个
// 都找不到，而失败发生在 device 编译器上，离写错的那处隔了几层。
//
// 先接通的 `abs.h` / `neg.h` 恰好把名字都写全了，所以它们挡不住这类回归。
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_reciprocal_store/LoadReciprocalStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_reciprocal_store::GetShapeConfig(load_reciprocal_store::kOutputShapes, 0).shape;
  const auto input_shape =
      ToExpressions(load_reciprocal_store::GetShapeConfig(load_reciprocal_store::kInputShapes, 0).shape);
  AscGraphBuilder builder(load_reciprocal_store::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph = builder.Reciprocal("reciprocal", "load0")
                   .Store("store", "reciprocal")
                   .Output("output", "store", 0, af::DT_FLOAT)
                   .Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_reciprocal_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  EXPECT_NE(kernel.find("ReciprocalExtend<float>("), std::string::npos);
  // 包装的头与它声明的前导头都被整段内联进来了，而且前导在前——`utils.h` 自己也用了
  // 不带限定的 `LocalTensor`，排在 using-directive 之后才成立。
  const size_t using_directive = kernel.find("using namespace AscendC;");
  const size_t prelude = kernel.find("__ASCENDC_API_UTILS_H__");
  const size_t wrapper = kernel.find("__ASCENDC_API_RECIPROCAL_H__");
  EXPECT_NE(using_directive, std::string::npos);
  EXPECT_NE(prelude, std::string::npos);
  EXPECT_NE(wrapper, std::string::npos);
  EXPECT_LT(using_directive, prelude);
  EXPECT_LT(prelude, wrapper);
  // 内联的是文本，不是 include——kernel 是个没有 include 路径的独立编译单元。
  EXPECT_EQ(kernel.find("#include \"utils.h\""), std::string::npos);
  EXPECT_EQ(kernel.find("#include \"reciprocal.h\""), std::string::npos);
  // 包装要 tmp buffer，所以这里恰恰应该有 TBuf——和 load_neg_store 互为反面。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
