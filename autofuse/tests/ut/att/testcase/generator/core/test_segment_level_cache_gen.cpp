/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "base/arch_param.h"
#include "base/model_info.h"
#include "cache/segment_level_cache_gen.h"
#include "gen_model_info/api_perf_register/v1/perf_param_v1.h"

namespace att {
namespace {
class SegmentCacheGeneratorUT : public testing::Test {};

// 通过覆写既有的vector_len_size/cache_line_size接口控制分段缓存公式阈值
class MockSegmentCacheConfigTable : public TilingScheduleConfigTableV1 {
 public:
  MockSegmentCacheConfigTable(const uint32_t vector_len_size, const uint32_t cache_line_size)
      : vector_len_size_(vector_len_size), cache_line_size_(cache_line_size) {}
  [[nodiscard]] uint32_t GetVectorLenSize() const override {
    return vector_len_size_;
  }
  [[nodiscard]] uint32_t GetCacheLineSize() const override {
    return cache_line_size_;
  }

 private:
  uint32_t vector_len_size_;
  uint32_t cache_line_size_;
};

AttAxisPtr MakeOriginAxis(const std::string &name, const std::string &var_name, const uint32_t dtype_size,
                          const bool is_reduce_related, const bool is_load_store_innermost) {
  auto axis = std::make_shared<AttAxis>();
  axis->name = name;
  axis->axis_pos = AxisPosition::ORIGIN;
  axis->is_reduce_related = is_reduce_related;
  axis->is_load_store_innermost = is_load_store_innermost;
  auto size = std::make_shared<SymVarInfo>(CreateExpr(var_name.c_str()));
  size->data_type_size = dtype_size;
  axis->size = size;
  return axis;
}

AttAxisPtr MakeMulticoreSplitAxis(const std::string &name, const std::string &var_name, AttAxis *orig_axis) {
  auto axis = std::make_shared<AttAxis>();
  axis->name = name;
  axis->axis_pos = AxisPosition::INNER;
  axis->bind_multicore = true;
  auto size = std::make_shared<SymVarInfo>(CreateExpr(var_name.c_str()));
  size->data_type_size = 4U;
  axis->size = size;
  axis->orig_axis.push_back(orig_axis);
  return axis;
}

ModelInfo MakeSegmentModelInfo(const std::vector<AttAxisPtr> &arg_list) {
  ModelInfo model_info;
  model_info.graph_name = "segment_cache_ut_graph";
  model_info.tiling_case_id = 0U;
  model_info.arg_list = arg_list;
  model_info.schedule_group_ident.asc_graph_id = 0UL;
  model_info.schedule_group_ident.impl_graph_id = 0UL;
  model_info.schedule_group_ident.group_id = 0UL;
  return model_info;
}
}  // namespace

TEST_F(SegmentCacheGeneratorUT, TypesIncludeFixedArrayAndKeySpace) {
  ge::CodePrinter code_printer;
  ASSERT_EQ(cache::SegmentLevelCacheGen::GenSegmentCacheTypes(code_printer, 2UL, 5UL), af::SUCCESS);
  const std::string generated = code_printer.GetOutputStr();
  EXPECT_NE(generated.find("constexpr uint32_t kSegmentCacheKeySpace = 8U;"), std::string::npos);
  EXPECT_NE(generated.find("struct SegmentGroupCase"), std::string::npos);
  EXPECT_NE(generated.find("uint32_t result_ids[2]{};"), std::string::npos);
  EXPECT_NE(generated.find("SegmentGroupCase group_cases[5]{};"), std::string::npos);
  EXPECT_NE(generated.find("struct SegmentCacheContext"), std::string::npos);
  EXPECT_NE(generated.find("bool selected_is_sub_case[5]{};"), std::string::npos);
}

TEST_F(SegmentCacheGeneratorUT, StorageDefinesThreadLocalCacheAndHelpers) {
  ge::CodePrinter code_printer;
  ASSERT_EQ(cache::SegmentLevelCacheGen::GenSegmentCacheStorage(code_printer, 256U, 128U), af::SUCCESS);
  const std::string generated = code_printer.GetOutputStr();
  EXPECT_NE(generated.find("constexpr uint32_t kSegmentCacheRAxisBytesThreshold = 256U;"), std::string::npos);
  EXPECT_NE(generated.find("constexpr uint32_t kSegmentCacheInnermostBytesThreshold = 128U;"), std::string::npos);
  EXPECT_NE(generated.find("thread_local std::unique_ptr<SegmentCacheEntry[]> g_segment_cache;"), std::string::npos);
  EXPECT_NE(generated.find("g_segment_cache = std::make_unique<SegmentCacheEntry[]>(kSegmentCacheKeySpace);"),
            std::string::npos);
  EXPECT_NE(generated.find("const SegmentCacheEntry *FindSegmentCache(const uint32_t segment_key)"), std::string::npos);
  EXPECT_NE(generated.find("void InvalidateSegmentCache(const uint32_t segment_key)"), std::string::npos);
  EXPECT_NE(generated.find("void SaveSegmentCache(const uint32_t segment_key, const SegmentCacheEntry &entry)"),
            std::string::npos);
}

TEST_F(SegmentCacheGeneratorUT, KeyComputeFuncEmitsThreeFormulas) {
  auto r_axis = MakeOriginAxis("r", "r_size", 2U, true, false);
  auto innermost_axis = MakeOriginAxis("in", "in_size", 4U, false, true);
  auto m_axis = MakeOriginAxis("m", "m_size", 4U, false, false);
  auto mc_axis = MakeMulticoreSplitAxis("mc", "mc_size", m_axis.get());
  TilingModelInfo model_infos{MakeSegmentModelInfo({r_axis, innermost_axis, m_axis, mc_axis})};

  TilingCodeGenConfig config;
  config.tiling_data_type_name = "SegmentTestTilingData";
  ge::CodePrinter code_printer;
  ASSERT_EQ(cache::SegmentLevelCacheGen::GenKeyComputeFunc(code_printer, model_infos, config), af::SUCCESS);
  const std::string generated = code_printer.GetOutputStr();

  EXPECT_NE(generated.find("uint32_t ComputeSegmentKey(const SegmentTestTilingData &tiling_data)"), std::string::npos);
  // 公式①：R轴 原始轴×dtype 短路链
  EXPECT_NE(generated.find("segment_r_product = segment_r_product * static_cast<uint64_t>(tiling_data.get_r_size())"
                           " * 2U) < kSegmentCacheRAxisBytesThreshold"),
            std::string::npos);
  // 公式②：最内轴 全满足
  EXPECT_NE(generated.find("segment_innermost_small = segment_innermost_small && ((static_cast<uint64_t>"
                           "(tiling_data.get_in_size()) * 4U) < kSegmentCacheInnermostBytesThreshold)"),
            std::string::npos);
  // 公式③：分核原始轴乘积 < 可用核数
  EXPECT_NE(generated.find("segment_mc_product = segment_mc_product * static_cast<uint64_t>(tiling_data.get_m_size())"
                           ") < static_cast<uint64_t>(tiling_data.get_block_dim())"),
            std::string::npos);
  // key 组合与日志
  EXPECT_NE(generated.find("(segment_r_small ? 4U : 0U)"), std::string::npos);
  EXPECT_NE(generated.find("ATT_SEGMENT_CACHE_LOGD(OP_NAME, \"[Segment Cache] key=%u"), std::string::npos);
  // 非R/非最内轴不参与公式
  EXPECT_EQ(generated.find("get_mc_size()"), std::string::npos);
}

TEST_F(SegmentCacheGeneratorUT, KeyComputeFuncWithoutRAxisKeepsFormulaTrue) {
  auto m_axis = MakeOriginAxis("m", "m_size", 4U, false, false);
  auto mc_axis = MakeMulticoreSplitAxis("mc", "mc_size", m_axis.get());
  TilingModelInfo model_infos{MakeSegmentModelInfo({m_axis, mc_axis})};

  TilingCodeGenConfig config;
  config.tiling_data_type_name = "SegmentTestTilingData";
  ge::CodePrinter code_printer;
  ASSERT_EQ(cache::SegmentLevelCacheGen::GenKeyComputeFunc(code_printer, model_infos, config), af::SUCCESS);
  const std::string generated = code_printer.GetOutputStr();

  EXPECT_NE(generated.find("bool segment_r_small = true;"), std::string::npos);
  EXPECT_NE(generated.find("bool segment_innermost_small = true;"), std::string::npos);
  // 无R轴/最内轴时不出累乘行
  EXPECT_EQ(generated.find("segment_r_product = segment_r_product *"), std::string::npos);
  EXPECT_EQ(generated.find("segment_innermost_small = segment_innermost_small &&"), std::string::npos);
}

TEST_F(SegmentCacheGeneratorUT, KeyComputeFuncUsesConfigTableThresholds) {
  auto r_axis = MakeOriginAxis("r", "r_size", 2U, true, false);
  MockSegmentCacheConfigTable config_table(512U, 64U);
  TilingModelInfo model_infos{MakeSegmentModelInfo({r_axis})};
  model_infos[0].tiling_schedule_config_table = &config_table;

  TilingCodeGenConfig config;
  config.tiling_data_type_name = "SegmentTestTilingData";
  ge::CodePrinter code_printer;
  ASSERT_EQ(cache::SegmentLevelCacheGen::GenKeyComputeFunc(code_printer, model_infos, config), af::SUCCESS);
  const std::string generated = code_printer.GetOutputStr();
  EXPECT_NE(generated.find("512"), std::string::npos);
  EXPECT_NE(generated.find("64"), std::string::npos);
}

TEST_F(SegmentCacheGeneratorUT, ThresholdsFallbackToDefaultWithoutTable) {
  TilingModelInfo model_infos{MakeSegmentModelInfo({})};
  uint32_t r_axis_threshold = 0U;
  uint32_t innermost_threshold = 0U;
  cache::SegmentLevelCacheGen::GetThresholds(model_infos, r_axis_threshold, innermost_threshold);
  // 表缺失时回退既有arch默认值：R轴阈值=vector_len_size，最内轴阈值=cache_line_size
  EXPECT_EQ(r_axis_threshold, arch_param::kDefaultVectorLenSize);
  EXPECT_EQ(innermost_threshold, arch_param::kDefaultCacheLineSize);

  MockSegmentCacheConfigTable override_table(1024U, 32U);
  model_infos[0].tiling_schedule_config_table = &override_table;
  cache::SegmentLevelCacheGen::GetThresholds(model_infos, r_axis_threshold, innermost_threshold);
  EXPECT_EQ(r_axis_threshold, 1024U);
  EXPECT_EQ(innermost_threshold, 32U);
}

TEST_F(SegmentCacheGeneratorUT, GroupSlotMapAssignsAscendingSlots) {
  auto axis0 = MakeOriginAxis("a", "a_size", 4U, false, false);
  auto axis1 = MakeOriginAxis("b", "b_size", 4U, false, false);
  TilingModelInfo model_infos;
  for (const auto &axis : {axis0, axis1}) {
    ModelInfo model_info = MakeSegmentModelInfo({axis});
    model_infos.emplace_back(model_info);
  }
  model_infos[0].schedule_group_ident.asc_graph_id = 1UL;
  model_infos[0].schedule_group_ident.impl_graph_id = 0UL;
  model_infos[0].schedule_group_ident.group_id = 0UL;
  model_infos[1].schedule_group_ident.asc_graph_id = 0UL;
  model_infos[1].schedule_group_ident.impl_graph_id = 1UL;
  model_infos[1].schedule_group_ident.group_id = 2UL;

  const auto slot_map = cache::SegmentLevelCacheGen::BuildGroupSlotMap(model_infos);
  ASSERT_EQ(slot_map.size(), 2UL);
  EXPECT_EQ(slot_map.at({0UL, 1UL, 2UL}), 0UL);
  EXPECT_EQ(slot_map.at({1UL, 0UL, 0UL}), 1UL);
}
}  // namespace att
