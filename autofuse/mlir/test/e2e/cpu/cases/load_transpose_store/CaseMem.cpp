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
// mem 侧（默认平台）那一份的 kernel 文本断言；图在 `Graph.cpp`。
void CheckGeneratedKernel(const std::string &kernel) {
  namespace config = load_transpose_store;
  // 走的是 MLIR 那条路而不是回落 legacy：`GenerateForInductor` 在 MLIR 报不支持时会接着跑
  // legacy 并**返回成功**，所以这一条不是形式检查。
  EXPECT_NE(kernel.find(std::string(config::kKernelName) + "_subgraph0_plan0"), std::string::npos);

  // mem 侧的 `codegen::ConfusionTranspose` 包装与折好的枚举。枚举值由 perm 推出
  // （`foldMemPermutation`），21 项表里 3 维 `[0,2,1]` 那一行。
  EXPECT_NE(kernel.find("codegen::ConfusionTranspose<"), std::string::npos);
  EXPECT_NE(kernel.find("AutoFuseTransposeType::TRANSPOSE_ND2ND_021"), std::string::npos);

  // 扩展头按注册表键整份内联（`transpose_base_type.h` 的枚举 + `transpose.h` 的包装），
  // 不是留一条 `#include`——键写错成源文件名就会剩下解析不了的头。
  EXPECT_NE(kernel.find("enum class AutoFuseTransposeType"), std::string::npos);
  EXPECT_NE(kernel.find("void ConfusionTranspose(const LocalTensor<T>"), std::string::npos);
  EXPECT_EQ(kernel.find("#include \"transpose.h\""), std::string::npos);

  // host tiling 字段按名字取（`tiling.transpose_tilingData_0`，bridge 与 ATT 的
  // `api_tiling_gen` 同一条命名规则），调用后紧跟 `PipeBarrier<PIPE_ALL>`（emitter 内建，
  // 与 legacy 的 `transpose_api_call.cpp` 同一形态）。
  EXPECT_NE(kernel.find("transpose_tilingData_0"), std::string::npos);
  EXPECT_NE(kernel.find("AscendC::PipeBarrier<PIPE_ALL>()"), std::string::npos);

  // The scheduled transpose graph also exercises RemovePad -> Pad. The 23-wide
  // tail requires padding, while the aligned registration checks the copy path.
  EXPECT_NE(kernel.find("AscendC::Pad("), std::string::npos);
  EXPECT_NE(kernel.find("KernelUtils::BlkAlign<float>"), std::string::npos);
  EXPECT_EQ(kernel.find("#include \"\""), std::string::npos);

  // 不发 reg 侧那条：同一张图两个家族各走各的，两条路不该同时命中。
  EXPECT_EQ(kernel.find("TransposeExtend<"), std::string::npos);
}
}  // namespace mlir::afir
