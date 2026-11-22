/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 验证 reg-base 的单节点 NegExtend 路径：optimizer / ATT 选择对应实现，MLIR
// 使用 reg 侧 implementation，neg_reg_base.h 被内联并通过编译。
// 该包装无 tmp buffer 和前导头，适合检查基础通路；其他用例负责 micro-API 和数组参数。
// 与 load_neg_store 使用相同图和数值断言、不同目标，补充 API-family LIT 的文本检查。

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_neg_reg_base_store/LoadNegRegBaseStoreCase.h"
#include "test/e2e/cpu/support/E2eCase.h"
#include "asc_graph_builder.h"

namespace mlir::afir {
af::AscGraph BuildGraph() {
  using af::testing::AscGraphBuilder;

  const auto output_shape = load_neg_reg_base_store::GetShapeConfig(load_neg_reg_base_store::kOutputShapes, 0).shape;
  const auto input_shape =
      ToExpressions(load_neg_reg_base_store::GetShapeConfig(load_neg_reg_base_store::kInputShapes, 0).shape);
  AscGraphBuilder builder(load_neg_reg_base_store::kKernelName);
  builder.Loops(output_shape).Data("input0", 0, input_shape, {}, af::DT_FLOAT).Load("load0", "input0", input_shape, {});

  auto graph = builder.Neg("neg", "load0").Store("store", "neg").Output("output", "store", 0, af::DT_FLOAT).Build();
  graph.SetTilingKey(0);
  return graph;
}

void CheckGeneratedKernel(const std::string &kernel) {
  EXPECT_NE(kernel.find(std::string(load_neg_reg_base_store::kKernelName) + "_subgraph0_plan0"), std::string::npos);
  // 包装是全局函数，元素类型是显式模板实参。两张表的 `NegExtend` 同名同签名，所以这一条
  // 两侧都成立——分得开它们的只有 include guard，也就是下面那两条。
  EXPECT_NE(kernel.find("NegExtend<float>("), std::string::npos);
  // 内联进来的必须是 reg-base 那一份。这两条反过来就是 `load_neg_store` 里写的那两条；
  // 换 soc 只改了第 1 处（`CodegenOptions::soc_version`）而漏了 NpuArch 的话，MLIR 这边
  // 照样选 reg 侧的 implementation，这一对仍然会过——真正会先炸的是调度那一侧算出来的
  // tmp buffer 与所选实现对不上。
  EXPECT_NE(kernel.find("__ASCENDC_API_REGBASE_NEG_H__"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_NEG_H__"), std::string::npos);
  // reg 侧这份包装同样不要 tmp buffer，也不声明前导头。
  EXPECT_EQ(kernel.find("AscendC::TBuf<"), std::string::npos);
  EXPECT_EQ(kernel.find("__ASCENDC_API_UTILS_H__"), std::string::npos);
}
}  // namespace mlir::afir
