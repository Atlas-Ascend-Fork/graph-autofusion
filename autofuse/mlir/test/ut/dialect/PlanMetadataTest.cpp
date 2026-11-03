/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dialect/AFIR/IR/PlanMetadata.h"
#include "dialect/AFIR/IR/AFIRAttributes.h"

#include <gtest/gtest.h>

#include "mlir/IR/Builders.h"

using namespace mlir;

namespace {

// 候选身份（subgraph / plan）已随设计文档 5.1 步 7 迁到 `afplan`，其读取与校验在
// AFPlanAttrsTest。这里只剩 module kind——单候选时它仍决定生成哪种 module，按 2.3 的
// 判别留在 `afir`。
TEST(PlanMetadataTest, RecognizesPlanModule) {
  MLIRContext context;
  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc(), "plan_module");
  module->setAttr(afir::kModuleKindAttrName, builder.getStringAttr(afir::kPlanModuleKind));
  EXPECT_TRUE(afir::isPlanModule(module));
  EXPECT_FALSE(afir::isEntryModule(module));
}

TEST(PlanMetadataTest, RecognizesEntryModule) {
  MLIRContext context;
  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc());
  module->setAttr(afir::kModuleKindAttrName, builder.getStringAttr(afir::kEntryModuleKind));
  EXPECT_TRUE(afir::isEntryModule(module));
  EXPECT_FALSE(afir::isPlanModule(module));
}

TEST(PlanMetadataTest, RejectsModuleWithoutKind) {
  MLIRContext context;
  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc());
  EXPECT_FALSE(afir::isPlanModule(module));
  EXPECT_FALSE(afir::isEntryModule(module));
}

}  // namespace
