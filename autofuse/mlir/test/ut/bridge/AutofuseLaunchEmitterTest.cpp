/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "artifact/KernelSource/Emit/AutofuseLaunchEmitter.h"

#include <gtest/gtest.h>

using mlir::ascend::kernel::AggregateArgumentLayout;
using mlir::ascend::kernel::KernelArgumentIdentity;
using mlir::ascend::kernel::KernelArgumentKind;

TEST(AutofuseLaunchEmitterTest, EmitsSignatureAndCallInCanonicalOrder) {
  mlir::MLIRContext context;
  AggregateArgumentLayout layout;
  layout.arguments = {
      KernelArgumentIdentity{KernelArgumentKind::Input, 0},
      KernelArgumentIdentity{KernelArgumentKind::Input, 1},
      KernelArgumentIdentity{KernelArgumentKind::Output, 0},
      KernelArgumentIdentity{KernelArgumentKind::Workspace, -1},
  };
  layout.tiling_arg_index = layout.arguments.size();
  auto gm = mlir::ascend::kernel::getAscendCGlobalMemorySpaceAttr(&context);
  llvm::SmallVector<mlir::Type> argumentTypes = {
      mlir::MemRefType::get({}, mlir::Float32Type::get(&context), mlir::MemRefLayoutAttrInterface{}, gm),
      mlir::MemRefType::get({}, mlir::Float16Type::get(&context), mlir::MemRefLayoutAttrInterface{}, gm),
      mlir::MemRefType::get({}, mlir::IntegerType::get(&context, 32), mlir::MemRefLayoutAttrInterface{}, gm),
      mlir::ascend::kernel::getAscendCWorkspaceType(&context),
  };

  auto text = mlir::ascend::kernel::emitAutofuseLaunch(layout, argumentTypes, "FusedGraph", "AutofuseTilingData");
  ASSERT_TRUE(succeeded(text));

  const std::string expected =
      "extern \"C\" int64_t AutofuseLaunch(uint32_t blockDim, void* stream, void* input0, void* input1, "
      "void* output0, void* workspace, AutofuseTilingData* tiling_data)\n"
      "{\n"
      "  FusedGraph<<<blockDim, nullptr, stream>>>((__gm__ float*)input0, (__gm__ half*)input1, "
      "(__gm__ int32_t*)output0, (__gm__ uint8_t*)workspace, *tiling_data);\n"
      "  return 0;\n"
      "}\n";
  EXPECT_EQ(*text, expected);
}

TEST(AutofuseLaunchEmitterTest, RejectsMissingWorkspaceArgument) {
  AggregateArgumentLayout layout;
  layout.arguments = {
      KernelArgumentIdentity{KernelArgumentKind::Input, 0},
      KernelArgumentIdentity{KernelArgumentKind::Output, 0},
  };
  layout.tiling_arg_index = layout.arguments.size();

  EXPECT_TRUE(failed(mlir::ascend::kernel::emitAutofuseLaunch(layout, {}, "FusedGraph", "AutofuseTilingData")));
}

TEST(AutofuseLaunchEmitterTest, RejectsDuplicateWorkspaceArguments) {
  AggregateArgumentLayout layout;
  layout.arguments = {
      KernelArgumentIdentity{KernelArgumentKind::Input, 0},
      KernelArgumentIdentity{KernelArgumentKind::Workspace, -1},
      KernelArgumentIdentity{KernelArgumentKind::Workspace, -1},
  };
  layout.tiling_arg_index = layout.arguments.size();

  EXPECT_TRUE(failed(mlir::ascend::kernel::emitAutofuseLaunch(layout, {}, "FusedGraph", "AutofuseTilingData")));
}

TEST(AutofuseLaunchEmitterTest, RejectsTilingArgIndexNotAtEnd) {
  AggregateArgumentLayout layout;
  layout.arguments = {
      KernelArgumentIdentity{KernelArgumentKind::Input, 0},
      KernelArgumentIdentity{KernelArgumentKind::Workspace, -1},
  };
  layout.tiling_arg_index = layout.arguments.size() + 1;

  EXPECT_TRUE(failed(mlir::ascend::kernel::emitAutofuseLaunch(layout, {}, "FusedGraph", "AutofuseTilingData")));
}
