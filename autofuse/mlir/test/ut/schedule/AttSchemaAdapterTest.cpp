/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bridge/AttSchemaAdapter/AttSchemaAdapter.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <string>

#include "ScheduleTestUtils.h"

namespace {

mlir::bridge::PlanSelection ValidSelection() {
  mlir::bridge::PlanSelection selection;
  selection.template_ids_per_stage = {{0U}};
  return selection;
}

TEST(AttSchemaAdapterTest, RejectsEmptySelectionBeforeCallingAtt) {
  mlir::common::TilingFieldSchema schema;
  schema.fields.push_back({"block_dim", "fusion", mlir::common::TilingFieldType::kUInt32});
  mlir::bridge::TilingCodeBundle bundle;
  std::string diagnostic;
  EXPECT_EQ(mlir::bridge::RunAttForCandidates({"test", mlir::test::MakeFusedResult(), {}, schema, {}, false}, bundle,
                                              &diagnostic),
            af::FAILED);
  EXPECT_EQ(diagnostic, "ATT adapter received no candidate selection");
}

TEST(AttSchemaAdapterTest, RejectsInvalidSelectionBeforeCallingAtt) {
  mlir::common::TilingFieldSchema schema;
  schema.fields.push_back({"block_dim", "fusion", mlir::common::TilingFieldType::kUInt32});
  auto selection = ValidSelection();
  selection.template_ids_per_stage = {{1U}};
  mlir::bridge::TilingCodeBundle bundle;
  std::string diagnostic;
  EXPECT_EQ(mlir::bridge::RunAttForCandidates({"test", mlir::test::MakeFusedResult(), {selection}, schema, {}, false},
                                              bundle, &diagnostic),
            af::FAILED);
  EXPECT_EQ(diagnostic, "template_id is out of range for stage 0");
}

TEST(AttSchemaAdapterTest, RejectsMalformedSchemaBeforeCallingAtt) {
  mlir::common::TilingFieldSchema schema;
  schema.fields.push_back({"bad-name", "fusion", mlir::common::TilingFieldType::kUInt32});
  mlir::bridge::TilingCodeBundle bundle;
  std::string diagnostic;
  EXPECT_EQ(mlir::bridge::RunAttForCandidates(
                {"test", mlir::test::MakeFusedResult(), {ValidSelection()}, schema, {}, false}, bundle, &diagnostic),
            af::FAILED);
  EXPECT_EQ(diagnostic, "ATT adapter received an invalid tiling schema");
}

TEST(AttSchemaAdapterDetailTest, ExtractsGetSetAndAddrAccessors) {
  const std::string source =
      "  uint32_t x = tiling_data.get_ub_size();\n"
      "  tiling_data.set_tiling_key(3);\n"
      "  auto *p = tiling_data.get_addr_block_dim();\n";
  const auto names = mlir::bridge::detail::ExtractReferencedFieldNames(source);
  EXPECT_NE(std::find(names.begin(), names.end(), "ub_size"), names.end());
  EXPECT_NE(std::find(names.begin(), names.end(), "tiling_key"), names.end());
  EXPECT_NE(std::find(names.begin(), names.end(), "block_dim"), names.end());
  EXPECT_EQ(names.size(), 3U);
}

TEST(AttSchemaAdapterDetailTest, IgnoresIdentifiersThatAreNotRealAccessorCalls) {
  // "offset_get_value" and "budget_setter" are not TILING_DATA_FIELD_DEF_T
  // accessors: "get_"/"set_" must start the identifier, and the identifier
  // must be followed by a call '('.
  const std::string source = "offset_get_value(1); budget_setter = 2; auto get_ready = true;";
  EXPECT_TRUE(mlir::bridge::detail::ExtractReferencedFieldNames(source).empty());
}

TEST(AttSchemaAdapterDetailTest, ValidateGeneratedFieldReferencesAcceptsKnownFields) {
  mlir::common::TilingFieldSchema schema;
  schema.fields.push_back({"ub_size", "fusion", mlir::common::TilingFieldType::kUInt32});
  schema.fields.push_back({"tiling_key", "fusion", mlir::common::TilingFieldType::kUInt32});
  std::map<std::string, std::string> generated{{"GetTiling", "tiling_data.set_tiling_key(tiling_data.get_ub_size());"}};
  std::string diagnostic;
  EXPECT_TRUE(mlir::bridge::detail::ValidateGeneratedFieldReferences(generated, schema, &diagnostic));
  EXPECT_TRUE(diagnostic.empty());
}

TEST(AttSchemaAdapterDetailTest, ValidateGeneratedFieldReferencesRejectsUnknownField) {
  mlir::common::TilingFieldSchema schema;
  schema.fields.push_back({"ub_size", "fusion", mlir::common::TilingFieldType::kUInt32});
  std::map<std::string, std::string> generated{{"GetTiling", "tiling_data.set_block_dim(1);"}};
  std::string diagnostic;
  EXPECT_FALSE(mlir::bridge::detail::ValidateGeneratedFieldReferences(generated, schema, &diagnostic));
  EXPECT_NE(diagnostic.find("block_dim"), std::string::npos);
}

// 这一组守的是「结构体成员」那一半对账。此前只扫标量访问器名，多 group 时 ATT 按成员名
// 直接访问各 schedule group 的结构体（取引用、整体赋值），命名不一致一路漏到真机 host 编译期
// 才以 `member access into incomplete type` 的形式暴露。
TEST(AttSchemaAdapterDetailTest, ExtractsScheduleGroupMemberAccesses) {
  const std::string source =
      "  auto &g0 = tiling_data.graph0_result0_g0_tiling_data;\n"
      "  autofuse_tiling_data->graph0_result0_g1_tiling_data = tmp;\n";
  const auto names = mlir::bridge::detail::ExtractReferencedGroupMemberNames(source);
  EXPECT_EQ(names.size(), 2U);
  EXPECT_NE(names.find("graph0_result0_g0_tiling_data"), names.end());
  EXPECT_NE(names.find("graph0_result0_g1_tiling_data"), names.end());
}

TEST(AttSchemaAdapterDetailTest, IgnoresGroupNamesThatAreNotMemberAccesses) {
  // 类型名与局部变量声明都不算：成员访问前面必须紧跟 `.` 或 `->`。
  const std::string source =
      "AscGraph0ScheduleResult0G0TilingData graph0_result0_g0_tiling_data;\n"
      "using T = graph0_result0_g1_tiling_data_t;\n";
  EXPECT_TRUE(mlir::bridge::detail::ExtractReferencedGroupMemberNames(source).empty());
}

TEST(AttSchemaAdapterDetailTest, ValidateGeneratedFieldReferencesAcceptsKnownGroupMembers) {
  // 嵌套作用域的 `path` 就是这个 group 在最外层 TilingData 里的成员名，两半对同一张 schema。
  mlir::common::TilingFieldSchema schema;
  schema.fields.push_back({"tiling_key", "graph0_result0_g0_tiling_data", mlir::common::TilingFieldType::kUInt32});
  std::map<std::string, std::string> generated{{"GetTiling", "auto &g0 = tiling_data.graph0_result0_g0_tiling_data;"}};
  std::string diagnostic;
  EXPECT_TRUE(mlir::bridge::detail::ValidateGeneratedFieldReferences(generated, schema, &diagnostic));
  EXPECT_TRUE(diagnostic.empty());
}

TEST(AttSchemaAdapterDetailTest, ValidateGeneratedFieldReferencesRejectsUnknownGroupMember) {
  // 这正是本次对齐之前的实际形态：schema 声明的是多出一层 plan 的路径，ATT 访问的是
  // 平铺在根上的成员名。
  mlir::common::TilingFieldSchema schema;
  schema.fields.push_back(
      {"tiling_key", "graph0_result0_tiling_data.g0_tiling_data", mlir::common::TilingFieldType::kUInt32});
  std::map<std::string, std::string> generated{{"GetTiling", "auto &g0 = tiling_data.graph0_result0_g0_tiling_data;"}};
  std::string diagnostic;
  EXPECT_FALSE(mlir::bridge::detail::ValidateGeneratedFieldReferences(generated, schema, &diagnostic));
  EXPECT_NE(diagnostic.find("graph0_result0_g0_tiling_data"), std::string::npos);
}

}  // namespace
