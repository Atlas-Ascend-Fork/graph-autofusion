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

// 候选身份 subgraph / plan 由 afplan 管理，读取和校验见 AFPlanAttrsTest。
// 本文件只测试单候选时仍影响模块形态的 afir module kind（设计文档 2.3）。
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
