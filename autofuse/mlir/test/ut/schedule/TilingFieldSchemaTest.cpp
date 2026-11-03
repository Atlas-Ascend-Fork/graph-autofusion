/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/tiling_schema/tiling_field_schema.h"

#include <gtest/gtest.h>

#include <set>
#include <string>

#include "ScheduleTestUtils.h"

namespace {

TEST(TilingSchemaTest, RejectsEmptyFusion) {
  ascir::FusedScheduledResult result;
  mlir::common::TilingFieldSchema schema;
  std::string diagnostic;
  EXPECT_FALSE(mlir::common::BuildTilingFieldSchema(result, schema, &diagnostic));
  EXPECT_TRUE(schema.fields.empty());
  EXPECT_EQ(diagnostic, "fused schedule contains no subgraph");
}

TEST(TilingSchemaTest, CollectsCommonAndGraphFields) {
  const auto result = mlir::test::MakeFusedResult(1U, 2U, false, true);
  mlir::common::TilingFieldSchema schema;
  ASSERT_TRUE(mlir::common::BuildTilingFieldSchema(result, schema));

  EXPECT_TRUE(mlir::common::LookupFieldName(schema, "block_dim").has_value());
  EXPECT_TRUE(mlir::common::LookupFieldName(schema, "tiling_key").has_value());
  EXPECT_TRUE(mlir::common::LookupFieldName(schema, "s0").has_value());
  EXPECT_FALSE(mlir::common::LookupFieldName(schema, "missing").has_value());

  const std::string emitted = mlir::common::EmitTilingDataStructCpp(schema, "TestTilingData");
  EXPECT_NE(emitted.find("BEGIN_TILING_DATA_DEF_T(TestTilingData)"), std::string::npos);
  EXPECT_NE(emitted.find("TILING_DATA_FIELD_DEF_T(uint32_t, s0);"), std::string::npos);
  EXPECT_NE(emitted.find("END_TILING_DATA_DEF_T;"), std::string::npos);
}

TEST(TilingSchemaTest, RejectsEmptyStage) {
  const auto result = mlir::test::MakeFusedResult(1U, 1U, true);
  mlir::common::TilingFieldSchema schema;
  std::string diagnostic;
  EXPECT_FALSE(mlir::common::BuildTilingFieldSchema(result, schema, &diagnostic));
  EXPECT_EQ(diagnostic, "schedule stage contains no implementation graph");
}

TEST(TilingSchemaTest, ScheduledResultOverloadUsesSameCommonFields) {
  const auto scheduled = mlir::test::MakeScheduledResult(1U, false, true);
  mlir::common::TilingFieldSchema schema;
  ASSERT_TRUE(mlir::common::BuildTilingFieldSchema(scheduled, 0U, 0U, schema));
  EXPECT_TRUE(mlir::common::LookupFieldName(schema, "block_dim").has_value());
  EXPECT_TRUE(mlir::common::LookupFieldName(schema, "s0").has_value());
}

// Regression for the dedup-scope bug: legacy ProcessMultiGroup isolates each
// (plan, stage) into its own nested struct with its own dedup set, so the
// same size-var name "s0" reused across stages must produce TWO distinct
// schema fields (one per stage's scope), not collapse into one. A schema
// builder that deduped by bare field name would silently drop stage 1's
// tiling data.
TEST(TilingSchemaTest, MultiStageSameSizeVarNameStaysIsolatedPerStage) {
  const auto scheduled = mlir::test::MakeMultiStageScheduledResult(/*stage_count=*/2U);
  mlir::common::TilingFieldSchema schema;
  ASSERT_TRUE(mlir::common::BuildTilingFieldSchema(scheduled, 0U, 0U, schema));

  size_t s0_count = 0U;
  std::set<std::string> s0_paths;
  for (const auto &field : schema.fields) {
    if (field.name == "s0") {
      ++s0_count;
      s0_paths.insert(field.path);
    }
  }
  EXPECT_EQ(s0_count, 2U);
  EXPECT_EQ(s0_paths.size(), 2U) << "each stage's s0 must live in its own scope, not be merged into one field";

  // The nested (non-flat) emission must declare a distinct struct per stage
  // scope and reference both from the outer struct, mirroring
  // ProcessMultiGroup's per-stage nested struct layout.
  const std::string emitted = mlir::common::EmitTilingDataStructCpp(schema, "TestTilingData");
  size_t s0_occurrences = 0U;
  for (size_t pos = emitted.find("s0);"); pos != std::string::npos; pos = emitted.find("s0);", pos + 1U)) {
    ++s0_occurrences;
  }
  EXPECT_EQ(s0_occurrences, 2U);
}

// Same isolation requirement across plans: two plans of the same subgraph
// must not let a shared field name in one plan's single stage collide with
// the other's.
TEST(TilingSchemaTest, MultiPlanFieldsStayIsolatedPerPlan) {
  const auto fused = mlir::test::MakeMultiStageFusedResult(/*plan_count=*/2U, /*stage_count=*/1U);
  mlir::common::TilingFieldSchema schema;
  ASSERT_TRUE(mlir::common::BuildTilingFieldSchema(fused, schema));

  size_t s0_count = 0U;
  std::set<std::string> s0_paths;
  for (const auto &field : schema.fields) {
    if (field.name == "s0") {
      ++s0_count;
      s0_paths.insert(field.path);
    }
  }
  EXPECT_EQ(s0_count, 2U);
  EXPECT_EQ(s0_paths.size(), 2U);

  // 两个 plan，于是不是单组那一支。真源是 `codegen_tiling_data.cpp` 的
  // `GetCommonTilingField`，它发两处键，这里都要对上：
  //
  //   外层（"fusion" 作用域）  `graph<i>_tiling_key`，**每个子图一个**——不带 `result<M>`，
  //                            也不是每个 plan 一个；
  //   组内（嵌套 stage 作用域）每个 stage 结构体里一个裸 `tiling_key`。
  //
  // 断言按作用域数个数，不按名字 `LookupFieldName`：后者名字匹配上就返回，分不开这两处，
  // 而这两处恰好一个叫 `graph0_tiling_key`、一个叫 `tiling_key`。
  size_t outer_keys = 0U;
  size_t nested_keys = 0U;
  std::set<std::string> outer_names;
  for (const auto &field : schema.fields) {
    if (field.path == "fusion" && field.name.find("tiling_key") != std::string::npos) {
      ++outer_keys;
      outer_names.insert(field.name);
    }
    if (field.path != "fusion" && field.name == "tiling_key") {
      ++nested_keys;
    }
  }
  EXPECT_EQ(outer_keys, 1U);
  EXPECT_EQ(outer_names, (std::set<std::string>{"graph0_tiling_key"}));
  // 两个 plan 各一个 stage，两个嵌套作用域，于是两个组局部键。
  EXPECT_EQ(nested_keys, 2U);
}

}  // namespace
