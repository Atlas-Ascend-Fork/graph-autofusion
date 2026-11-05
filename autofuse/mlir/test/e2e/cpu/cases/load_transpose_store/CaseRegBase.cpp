/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>

#include <gtest/gtest.h>

#include "test/e2e/cpu/cases/load_transpose_store/Case.h"
namespace mlir::afir {
// reg 侧（`PLATFORM reg_base`）那一份的 kernel 文本断言；图在 `Graph.cpp`。
void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_transpose_store;
  // MLIR 路而非回落 legacy（口径同 mem 侧：`GenerateForInductor` 在 MLIR 报不支持时会接着
  // 跑 legacy 并**返回成功**，这条不是形式检查）。
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);

  // 调度同时保留显式转置与折入 NDDMA 的候选；数值测试执行 ATT 选出的候选。
  EXPECT_NE(kernel.find("TransposeExtend<3, 3>("), std::string::npos);
  EXPECT_NE(kernel.find("TransposeExtend<2, 2>("), std::string::npos);
  EXPECT_NE(kernel.find("static_cast<int32_t>("), std::string::npos);
  EXPECT_NE(kernel.find("DataCopyNddma<float, 3>("), std::string::npos);
  EXPECT_NE(kernel.find("DataCopyNddma<float, 2>("), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::PaddingMode::Compact>"), std::string::npos);

  // 不发 mem 侧那条：同一张图两个家族各走各的，两条路不该同时命中。
  EXPECT_EQ(kernel.find("codegen::ConfusionTranspose<"), std::string::npos);
}
}  // namespace mlir::afir
