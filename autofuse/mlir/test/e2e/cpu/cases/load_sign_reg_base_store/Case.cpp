/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//===- Case.cpp - load -> sign -> store CPU case (reg-base) -----*- C++ -*-===//
//
// reg-base 一侧，钉的是前导头也可以是中立的这条判断。
//
// `sign_reg_base.h` 用了 `MAX_REPEAT_TIME`，而那是 **mem 侧 `utils.h`** 里的东西
// ——`utils_reg_base.h` 是 micro-API 的 `CastTrait` 常量表，一个相关符号都没有。所以
// 这个 reg 侧的 op 声明了一个 mem 侧的前导头。`headersMatchFamily` 当初为此把「每个头
// 都属于某一类」这条前提整个推翻了（见 `ElementwiseApiFamily.h`），但那之后，这条判断
// 一直只是读出来的：lit 只比文本，前导头声明删掉照样绿。这份用例是第一次让编译器
// 表态。
//
// 它同时也是「一个 op 两个头」在 reg 侧的第一个编译型用例：`cast.h` 是中立头，两侧
// 点的是同一个文件。
//
//===----------------------------------------------------------------------===//

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_sign_reg_base_store/LoadSignRegBaseStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_sign_reg_base_store::GetShapeConfig(load_sign_reg_base_store::kOutputShapes, 0).shape;
  const auto input_shape =
      ToExpressions(load_sign_reg_base_store::GetShapeConfig(load_sign_reg_base_store::kInputShapes, 0).shape);
  AscGraphBuilder builder(load_sign_reg_base_store::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph = builder.Sign("sign", "load0").Store("store", "sign").Output("output", "store", 0, af::DT_FLOAT).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_sign_reg_base_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 两侧的包装同名同签名，生成的调用逐字相同——所以这一条两侧都成立，分不开它们。
  EXPECT_NE(kernel.find("SignExtend<float>("), std::string::npos);
  // 分得开的是内联进来的头文本。两份 `sign.h` 的 include guard 是同一个
  // （`__ASCENDC_API_SIGN_H__`），所以这里不能比 guard，只能挑一处只有 reg 那份才有的
  // 文本：它的直通分支多认一个 int64。`load_sign_store`（mem 侧）钉的是这一条的反面。
  EXPECT_NE(kernel.find("std::is_same_v<T, int64_t>"), std::string::npos);
  // `cast.h` 是**中立的头**：reg 侧的 sign 点的就是 mem 这张表里的那一份，两侧同一个
  // 文件。家族判据放行它，正是因为这种中立头存在。
  EXPECT_NE(kernel.find("__ASCENDC_API_CAST_H__"), std::string::npos);
  // 前导头 `utils.h`——这份用例的要害。 `sign_reg_base.h` 用了 `MAX_REPEAT_TIME`，
  // 而那是 mem 侧 `utils.h` 里的东西（`utils_reg_base.h` 是 micro-API 的常量表，
  // 一个都没有）。所以「前导头也可以是中立的」这条判断，是在这里第一次被编译器验证。
  EXPECT_NE(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
  // 包装要 tmp buffer，两侧都要。
  EXPECT_NE(kernel.find("AscendC::TBuf<"), std::string::npos);
}
}  // namespace mlir::afir
