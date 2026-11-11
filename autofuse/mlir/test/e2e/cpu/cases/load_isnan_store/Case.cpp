/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> isnan -> store CPU case ---------------*- C++ -*-===//
//
// 第一个结果 dtype 与输入不同的端到端用例：输入 f32，输出 ui8。一路上每一层都要
// 不认为「结果跟输入同类型」——ODS 的 `AllElementTypesMatch`、capability 的元素类型
// 判据、emitter 的显式模板实参（`IsnanExtend<T>` 的 T 是源的类型，目的地恒为
// uint8）。其中最后一条只有真的把 kernel 编出来才挡得住。
//
// `isnan.h` 同时是又一个不自足的扩展头（用了 `MAX_REPEAT_TIME`），所以这个用例也顺带
// 复验前导那条通道。
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_isnan_store/LoadIsnanStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_isnan_store::GetShapeConfig(load_isnan_store::kOutputShapes, 0).shape;
  const auto input_shape = ToExpressions(load_isnan_store::GetShapeConfig(load_isnan_store::kInputShapes, 0).shape);
  AscGraphBuilder builder(load_isnan_store::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph =
      builder.Isnan("isnan", "load0").Store("store", "isnan").Output("output", "store", 0, af::DT_UINT8).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_isnan_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 模板实参是源的 float，不是目的地的 uint8。
  EXPECT_NE(kernel.find("IsnanExtend<float>("), std::string::npos);
  EXPECT_EQ(kernel.find("IsnanExtend<uint8_t>("), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_ISNAN_H__"), std::string::npos);
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
  // ReinterpretCast<bool> 只属于标准调用那一侧；包装收的就是 ui8。
  EXPECT_EQ(kernel.find("ReinterpretCast<bool>"), std::string::npos);
  // 包装要 tmp buffer。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
