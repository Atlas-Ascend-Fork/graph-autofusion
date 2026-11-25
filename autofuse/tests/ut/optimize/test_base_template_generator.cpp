/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "ascendc_ir.h"
#include "ascir_ops.h"
#include "ascir_utils.h"

#include "platform/common/base_template_generator.h"

namespace optimize {
namespace {
// 覆盖 GetBasedCasesByGenMode 的三种返回路径：
//   kBaseCase   -> 原样返回 tiling_cases
//   kAppendCase -> tiling_cases 在前、generated_cases 在后拼接
//   其余取值    -> 走 "Unknown generation mode" 告警分支并返回空集合
class BaseTemplateGeneratorGenModeTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};
}  // namespace

TEST_F(BaseTemplateGeneratorGenModeTest, GetBasedCasesByGenMode_ShouldReturnTilingCases_WhenModeIsBaseCase) {
  BaseTemplateGenerator generator;
  std::vector<autoschedule::AutoScheduleOutput> tiling_cases(2);
  std::vector<autoschedule::AutoScheduleOutput> generated_cases(3);

  auto result = generator.GetBasedCasesByGenMode(GenerationMode::kBaseCase, tiling_cases, generated_cases);
  ASSERT_EQ(result.size(), tiling_cases.size());
}

TEST_F(BaseTemplateGeneratorGenModeTest, GetBasedCasesByGenMode_ShouldMergeCases_WhenModeIsAppendCase) {
  BaseTemplateGenerator generator;
  std::vector<autoschedule::AutoScheduleOutput> tiling_cases(2);
  std::vector<autoschedule::AutoScheduleOutput> generated_cases(3);

  auto result = generator.GetBasedCasesByGenMode(GenerationMode::kAppendCase, tiling_cases, generated_cases);
  ASSERT_EQ(result.size(), tiling_cases.size() + generated_cases.size());
}

TEST_F(BaseTemplateGeneratorGenModeTest, GetBasedCasesByGenMode_ShouldReturnEmpty_WhenModeIsUnknown) {
  BaseTemplateGenerator generator;
  std::vector<autoschedule::AutoScheduleOutput> tiling_cases(2);
  std::vector<autoschedule::AutoScheduleOutput> generated_cases(3);
  // 越界枚举值模拟后续新增但此处未处理的分支，触发告警并返回空集合（fail-closed，不静默当 kBaseCase）
  const auto unknown_mode = static_cast<GenerationMode>(static_cast<std::uint32_t>(GenerationMode::kAppendCase) + 1U);

  auto result = generator.GetBasedCasesByGenMode(unknown_mode, tiling_cases, generated_cases);
  ASSERT_TRUE(result.empty());
}
}  // namespace optimize
