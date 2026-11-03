/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dialect/AFPlan/IR/AFPlan.h"

#include <gtest/gtest.h>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"

using namespace mlir;

namespace {

// `afplan` 承载候选身份（设计文档 2.2 / 2.3、5.1 步 7）。用例覆盖两件事：
//   1) 访问器——取到即代表已过 verifier，调用方不必重复非负校验；
//   2) verifier 的每条分支——其中「空分派表」这条在文本 IR 里够不到（ODS 生成的解析器
//      要求列表非空），只有程序化构造这条路径能触发，因此必须在 UT 里覆盖。

class AFPlanAttrsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context.getOrLoadDialect<afplan::AFPlanDialect>();
  }

  // getChecked 把 verifier 的失败变成空属性而不是断言，用来探测各条分支。
  Location loc() {
    return UnknownLoc::get(&context);
  }

  MLIRContext context;
};

TEST_F(AFPlanAttrsTest, ReadsCandidateFromModule) {
  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc(), "plan");
  module->setAttr(afplan::kCandidateAttrName, afplan::CandidateAttr::get(&context, 2, 3));

  afplan::CandidateAttr candidate = afplan::getCandidate(module);
  ASSERT_TRUE(candidate);
  EXPECT_EQ(candidate.getSubgraph(), 2);
  EXPECT_EQ(candidate.getPlan(), 3);
}

// 候选身份是 uniqued 的：查找候选可以直接比属性，不必逐字段取值再比。
TEST_F(AFPlanAttrsTest, CandidateIsUniqued) {
  EXPECT_EQ(afplan::CandidateAttr::get(&context, 1, 4), afplan::CandidateAttr::get(&context, 1, 4));
  EXPECT_NE(afplan::CandidateAttr::get(&context, 1, 4), afplan::CandidateAttr::get(&context, 4, 1));
}

// 聚合入口的 plan 表把候选身份放在字典项里，和挂在 module 上的是同一个属性。
TEST_F(AFPlanAttrsTest, ReadsCandidateFromAggregateEntry) {
  OpBuilder builder(&context);
  auto entry = builder.getDictionaryAttr(
      {builder.getNamedAttr(afplan::kCandidateAttrName, afplan::CandidateAttr::get(&context, 1, 4))});

  afplan::CandidateAttr candidate = afplan::getCandidate(entry);
  ASSERT_TRUE(candidate);
  EXPECT_EQ(candidate.getSubgraph(), 1);
  EXPECT_EQ(candidate.getPlan(), 4);
}

TEST_F(AFPlanAttrsTest, ReadsVariantAndStages) {
  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc(), "plan");
  module->setAttr(afplan::kStagesAttrName, afplan::StagesAttr::get(&context, {2, 1}));

  afplan::StagesAttr stages = afplan::getStages(module);
  ASSERT_TRUE(stages);
  EXPECT_EQ(stages.getTemplateCounts().size(), 2u);
  EXPECT_EQ(stages.getTemplateCounts()[0], 2);

  module->setAttr(afplan::kVariantAttrName, afplan::VariantAttr::get(&context, 1, 0));
  afplan::VariantAttr variant = afplan::getVariant(module);
  ASSERT_TRUE(variant);
  EXPECT_EQ(variant.getStage(), 1);
  EXPECT_EQ(variant.getTemplateId(), 0);
}

// 缺失与类型不符对调用方是同一件事：这个容器没有可用的候选身份。
TEST_F(AFPlanAttrsTest, AccessorsReturnNullWhenAbsentOrMistyped) {
  OpBuilder builder(&context);
  auto module = ModuleOp::create(builder.getUnknownLoc());
  EXPECT_FALSE(afplan::getCandidate(module));
  EXPECT_FALSE(afplan::getVariant(module));
  EXPECT_FALSE(afplan::getStages(module));
  EXPECT_FALSE(afplan::getCandidate(DictionaryAttr()));

  module->setAttr(afplan::kCandidateAttrName, builder.getI64IntegerAttr(0));
  EXPECT_FALSE(afplan::getCandidate(module));
}

TEST_F(AFPlanAttrsTest, RejectsNegativeCandidateIds) {
  ScopedDiagnosticHandler handler(&context, [](Diagnostic &) { return success(); });
  EXPECT_FALSE(afplan::CandidateAttr::getChecked([&] { return emitError(loc()); }, &context, int64_t{-1}, int64_t{0}));
  EXPECT_FALSE(afplan::CandidateAttr::getChecked([&] { return emitError(loc()); }, &context, int64_t{0}, int64_t{-1}));
  EXPECT_TRUE(afplan::CandidateAttr::getChecked([&] { return emitError(loc()); }, &context, int64_t{0}, int64_t{0}));
}

TEST_F(AFPlanAttrsTest, RejectsNegativeVariantIds) {
  ScopedDiagnosticHandler handler(&context, [](Diagnostic &) { return success(); });
  EXPECT_FALSE(afplan::VariantAttr::getChecked([&] { return emitError(loc()); }, &context, int64_t{-1}, int64_t{0}));
  EXPECT_FALSE(afplan::VariantAttr::getChecked([&] { return emitError(loc()); }, &context, int64_t{0}, int64_t{-1}));
  EXPECT_TRUE(afplan::VariantAttr::getChecked([&] { return emitError(loc()); }, &context, int64_t{0}, int64_t{0}));
}

// 空分派表没有任何 stage 可分派，没有 template 的 stage 同样无从分派。前者在文本 IR
// 里够不到（解析器先一步拒掉），只有这条程序化路径能覆盖到 verifier 的这条分支。
TEST_F(AFPlanAttrsTest, RejectsEmptyOrZeroCountStages) {
  ScopedDiagnosticHandler handler(&context, [](Diagnostic &) { return success(); });
  EXPECT_FALSE(afplan::StagesAttr::getChecked([&] { return emitError(loc()); }, &context, ArrayRef<int64_t>{}));
  EXPECT_FALSE(afplan::StagesAttr::getChecked([&] { return emitError(loc()); }, &context, ArrayRef<int64_t>{2, 0}));
  EXPECT_FALSE(afplan::StagesAttr::getChecked([&] { return emitError(loc()); }, &context, ArrayRef<int64_t>{-1}));
  EXPECT_TRUE(afplan::StagesAttr::getChecked([&] { return emitError(loc()); }, &context, ArrayRef<int64_t>{2, 1}));
}

}  // namespace
