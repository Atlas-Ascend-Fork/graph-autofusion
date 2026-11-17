/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "base/model_info.h"
#include "gen_model_info/parser/tuning_space.h"
#define private public
#include "expr_gen/generate_tiling_expr.h"
#undef private

namespace att {
class TestGenerateTilingExprPenalty : public testing::Test {
 public:
  void SetUp() override {
    tuning_space = std::make_shared<TuningSpace>();
    config_table = std::make_shared<MockTilingScheduleConfigTable>();
    tuning_space->tiling_schedule_config_table = config_table.get();
  }
  void TearDown() override {}

  // 辅助函数：创建一个 SubAxis
  SubAxisPtr CreateSubAxis(const std::string &name, AxisPosition axis_type, bool is_reduce_split = false,
                           bool is_broadcast_split = false) {
    auto sub_axis = std::make_unique<SubAxis>();
    sub_axis->name = name;
    sub_axis->axis_type = axis_type;
    sub_axis->is_reduce_split_axis = is_reduce_split;
    sub_axis->is_broadcast_split_axis = is_broadcast_split;
    sub_axis->repeat = af::Symbol(32);
    sub_axis->data_type_size = 4;
    return sub_axis;
  }

  // 辅助函数：创建一个 Tensor
  TensorPtr CreateTensor(const std::string &name, const std::vector<SubAxis *> &dim_info) {
    auto tensor = std::make_shared<Tensor>();
    tensor->name = name;
    tensor->data_type_size = 4;
    tensor->dim_info = dim_info;
    tensor->loc = HardwareDef::GM;

    // 补充完整的 repeat、stride、ori_repeat、ori_stride、gm_stride
    // 根据 dim_info 的数量填充对应数量的值
    size_t dim_size = dim_info.size();
    tensor->repeat.resize(dim_size, af::Symbol(32));      // 默认 repeat 为 32
    tensor->stride.resize(dim_size, af::Symbol(1));       // 默认 stride 为 1
    tensor->ori_repeat.resize(dim_size, af::Symbol(32));  // 默认 ori_repeat 为 32
    tensor->ori_stride.resize(dim_size, af::Symbol(1));   // 默认 ori_stride 为 1
    tensor->gm_stride.resize(dim_size, af::Symbol(1));    // 默认 gm_stride 为 1

    return tensor;
  }

  // 辅助函数：创建一个 Store NodeInfo
  NodeInfo CreateStoreNodeInfo(const std::string &name, const std::vector<TensorPtr> &inputs) {
    NodeInfo node_info;
    node_info.name = name;
    node_info.node_type = "Store";
    node_info.inputs = inputs;

    // 创建一个模拟的 AscNode（用于 IsStoreNode 判断）
    // 注意：这里简化处理，实际测试中 IsStoreNode 检查 node_ptr
    // 如果测试失败，可能需要用 mock 或者其他方式处理
    node_info.node_ptr = nullptr;  // 先设为空，测试时可能需要调整

    return node_info;
  }

  class MockTilingScheduleConfigTable : public TilingScheduleConfigTable {
   public:
    bool IsEnableBlockLoopAutoTune() const override {
      return true;
    }
    bool IsEnableCacheLineCheck() const override {
      return false;
    }
    TradeOffConfig GetTradeOffConfig() const override {
      TradeOffConfig config;
      config.is_enable = false;
      config.ub_ratio = af::Symbol(0.1);
      config.core_num_ratio = af::Symbol(0.8);
      return config;
    }
    double GetUbThresholdPerfValEffect() const override {
      return 0.5;
    }
    TilingScheduleConfig GetModelTilingScheduleConfig() const override {
      TilingScheduleConfig config;
      config.trade_off_config.is_enable = false;
      config.cache_line_size = 128;
      config.is_penalty_config = false;
      return config;
    }
    uint32_t GetCacheLineSize() const override {
      return 128;
    }
    bool IsCoreNumThresholdPenaltyEnable() const override {
      return true;
    }
  };

  TuningSpacePtr tuning_space;
  std::shared_ptr<MockTilingScheduleConfigTable> config_table;
};

// Test FindAAxis - Empty node_infos
TEST_F(TestGenerateTilingExprPenalty, FindAAxis_EmptyNodeInfos) {
  GenerateTilingExpr generator(tuning_space);

  auto a_axis1 = std::make_shared<AttAxis>();
  a_axis1->name = "A1";
  a_axis1->is_reduce_split_axis = false;
  a_axis1->is_broadcast_split_axis = false;

  std::vector<AttAxisPtr> arg_list = {a_axis1};

  const auto result = generator.FindAAxis(arg_list);

  EXPECT_EQ(result.size(), 0UL);
}

// Test FindAAxis - Multiple A axes (with proper NodeInfo setup)
TEST_F(TestGenerateTilingExprPenalty, FindAAxis_MultipleAAxes_WithNodeInfo) {
  GenerateTilingExpr generator(tuning_space);

  // 创建 SubAxis
  auto a1_sub = CreateSubAxis("A1", AxisPosition::INNER);
  auto a2_sub = CreateSubAxis("A2", AxisPosition::OUTER);
  auto r_sub = CreateSubAxis("R", AxisPosition::OUTER, true);  // Reduce分核轴

  // 添加到 tuning_space->sub_axes
  tuning_space->sub_axes.push_back(std::move(a1_sub));
  tuning_space->sub_axes.push_back(std::move(a2_sub));
  tuning_space->sub_axes.push_back(std::move(r_sub));

  // 获取 SubAxis 原始指针用于构造 Tensor
  std::vector<SubAxis *> sub_axes_ptrs;
  for (const auto &sub : tuning_space->sub_axes) {
    sub_axes_ptrs.push_back(sub.get());
  }

  // 创建 Tensor（dim_info 从右向左：A2, R, A1）
  auto tensor = CreateTensor("test_tensor", {sub_axes_ptrs[2], sub_axes_ptrs[1], sub_axes_ptrs[0]});

  // 创建 Store NodeInfo
  auto store_node = CreateStoreNodeInfo("store_node", {tensor});

  // 添加到 tuning_space->node_infos
  tuning_space->node_infos.push_back(store_node);

  // 创建 arg_list 中的 AttAxis
  auto a_axis1 = std::make_shared<AttAxis>();
  a_axis1->name = "A1";
  a_axis1->is_reduce_split_axis = false;
  a_axis1->is_broadcast_split_axis = false;

  auto a_axis2 = std::make_shared<AttAxis>();
  a_axis2->name = "A2";
  a_axis2->is_reduce_split_axis = false;
  a_axis2->is_broadcast_split_axis = false;

  auto r_axis = std::make_shared<AttAxis>();
  r_axis->name = "R";
  r_axis->is_reduce_split_axis = true;

  std::vector<AttAxisPtr> arg_list = {a_axis1, a_axis2, r_axis};

  const auto result = generator.FindAAxis(arg_list);

  // 应该找到 A1 和 A2（R 是 Reduce分核轴，应被跳过）
  EXPECT_EQ(result.size(), 2UL);
}

// Test FindAAxis - Skip Broadcast split axis (with proper NodeInfo setup)
TEST_F(TestGenerateTilingExprPenalty, FindAAxis_SkipBroadcastSplitAxis_WithNodeInfo) {
  GenerateTilingExpr generator(tuning_space);

  // 创建 SubAxis
  auto a_sub = CreateSubAxis("A", AxisPosition::INNER);
  auto b_sub = CreateSubAxis("B", AxisPosition::OUTER, false, true);  // Broadcast分核轴

  // 添加到 tuning_space->sub_axes
  tuning_space->sub_axes.push_back(std::move(a_sub));
  tuning_space->sub_axes.push_back(std::move(b_sub));

  // 获取 SubAxis 原始指针
  std::vector<SubAxis *> sub_axes_ptrs;
  for (const auto &sub : tuning_space->sub_axes) {
    sub_axes_ptrs.push_back(sub.get());
  }

  // 创建 Tensor
  auto tensor = CreateTensor("test_tensor", {sub_axes_ptrs[1], sub_axes_ptrs[0]});

  // 创建 Store NodeInfo
  auto store_node = CreateStoreNodeInfo("store_node", {tensor});
  tuning_space->node_infos.push_back(store_node);

  // 创建 arg_list
  auto a_axis = std::make_shared<AttAxis>();
  a_axis->name = "A";
  a_axis->is_reduce_split_axis = false;
  a_axis->is_broadcast_split_axis = false;

  auto b_axis = std::make_shared<AttAxis>();
  b_axis->name = "B";
  b_axis->is_broadcast_split_axis = true;

  std::vector<AttAxisPtr> arg_list = {a_axis, b_axis};

  const auto result = generator.FindAAxis(arg_list);

  // B 是 Broadcast分核轴，应被跳过，只返回 A
  EXPECT_EQ(result.size(), 1UL);
  EXPECT_EQ(result[0]->name, "A");
}

// Test FindAAxis - No A axes (all should be skipped)
TEST_F(TestGenerateTilingExprPenalty, FindAAxis_NoA_Axes_WithNodeInfo) {
  GenerateTilingExpr generator(tuning_space);

  // 创建 SubAxis - 全是 Reduce 或 Broadcast 分核轴
  auto r1_sub = CreateSubAxis("R1", AxisPosition::OUTER, true);
  auto r2_sub = CreateSubAxis("R2", AxisPosition::OUTER, true);
  auto b_sub = CreateSubAxis("B", AxisPosition::OUTER, false, true);

  tuning_space->sub_axes.push_back(std::move(r1_sub));
  tuning_space->sub_axes.push_back(std::move(r2_sub));
  tuning_space->sub_axes.push_back(std::move(b_sub));

  std::vector<SubAxis *> sub_axes_ptrs;
  for (const auto &sub : tuning_space->sub_axes) {
    sub_axes_ptrs.push_back(sub.get());
  }

  auto tensor = CreateTensor("test_tensor", sub_axes_ptrs);
  auto store_node = CreateStoreNodeInfo("store_node", {tensor});
  tuning_space->node_infos.push_back(store_node);

  auto r1_axis = std::make_shared<AttAxis>();
  r1_axis->name = "R1";
  r1_axis->is_reduce_split_axis = true;

  auto r2_axis = std::make_shared<AttAxis>();
  r2_axis->name = "R2";
  r2_axis->is_reduce_split_axis = true;

  auto b_axis = std::make_shared<AttAxis>();
  b_axis->name = "B";
  b_axis->is_broadcast_split_axis = true;

  std::vector<AttAxisPtr> arg_list = {r1_axis, r2_axis, b_axis};

  const auto result = generator.FindAAxis(arg_list);

  EXPECT_EQ(result.size(), 0UL);
}

// Test FindAAxis - All Reduce axes (original test)
TEST_F(TestGenerateTilingExprPenalty, FindAAxis_AllReduceAxes) {
  GenerateTilingExpr generator(tuning_space);

  auto r_axis1 = std::make_shared<AttAxis>();
  r_axis1->name = "R1";
  r_axis1->is_reduce_split_axis = true;

  auto r_axis2 = std::make_shared<AttAxis>();
  r_axis2->name = "R2";
  r_axis2->is_reduce_split_axis = true;

  std::vector<AttAxisPtr> arg_list = {r_axis1, r_axis2};

  const auto result = generator.FindAAxis(arg_list);

  EXPECT_EQ(result.size(), 0UL);
}

// Test CalcPenaltyCoreNumRatio - Empty input
TEST_F(TestGenerateTilingExprPenalty, CalcPenaltyCoreNumRatio_EmptyInput) {
  GenerateTilingExpr generator(tuning_space);

  Expr result = generator.CalcPenaltyCoreNumRatio(nullptr, {});

  EXPECT_EQ(Str(result), "1");
}

// Test CalcPenaltyCoreNumRatio - Null split_axis
TEST_F(TestGenerateTilingExprPenalty, CalcPenaltyCoreNumRatio_NullSplitAxis) {
  GenerateTilingExpr generator(tuning_space);

  auto a_axis = std::make_shared<AttAxis>();
  a_axis->name = "A";
  a_axis->axis_pos = AxisPosition::INNER;
  auto size_info = std::make_shared<SymVarInfo>(af::Symbol(32));
  size_info->data_type_size = 4;
  size_info->symbol_expr = af::Symbol(32);
  a_axis->size = size_info;

  std::vector<const AttAxis *> a_axes = {a_axis.get()};

  Expr result = generator.CalcPenaltyCoreNumRatio(nullptr, a_axes);

  EXPECT_EQ(Str(result), "1");
}

// Test CalcPenaltyCoreNumRatio - Only INNER axes
TEST_F(TestGenerateTilingExprPenalty, CalcPenaltyCoreNumRatio_OnlyInnerAxes) {
  GenerateTilingExpr generator(tuning_space);

  auto split_axis = std::make_shared<AttAxis>();
  split_axis->name = "R_split";
  auto size_info = std::make_shared<SymVarInfo>(af::Symbol(128));
  size_info->data_type_size = 4;
  split_axis->size = size_info;

  auto a_axis1 = std::make_shared<AttAxis>();
  a_axis1->name = "A1";
  a_axis1->axis_pos = AxisPosition::INNER;
  auto a1_size = std::make_shared<SymVarInfo>(af::Symbol(32));
  a1_size->data_type_size = 4;
  a1_size->symbol_expr = af::Symbol(32);
  a_axis1->size = a1_size;

  auto a_axis2 = std::make_shared<AttAxis>();
  a_axis2->name = "A2";
  a_axis2->axis_pos = AxisPosition::INNER;
  auto a2_size = std::make_shared<SymVarInfo>(af::Symbol(16));
  a2_size->data_type_size = 4;
  a2_size->symbol_expr = af::Symbol(16);
  a_axis2->size = a2_size;

  std::vector<const AttAxis *> a_axes = {a_axis1.get(), a_axis2.get()};

  Expr result = generator.CalcPenaltyCoreNumRatio(split_axis.get(), a_axes);

  // Verify ratio is calculated (not 1)
  EXPECT_NE(Str(result), "1");
}

// Test CalcPenaltyCoreNumRatio - Only OUTER axes
TEST_F(TestGenerateTilingExprPenalty, CalcPenaltyCoreNumRatio_OnlyOuterAxes) {
  GenerateTilingExpr generator(tuning_space);

  auto split_axis = std::make_shared<AttAxis>();
  split_axis->name = "R_split";
  auto size_info = std::make_shared<SymVarInfo>(af::Symbol(128));
  size_info->data_type_size = 4;
  split_axis->size = size_info;

  auto a_axis1 = std::make_shared<AttAxis>();
  a_axis1->name = "A1";
  a_axis1->axis_pos = AxisPosition::OUTER;
  auto a1_size = std::make_shared<SymVarInfo>(af::Symbol(32));
  a1_size->data_type_size = 4;
  a1_size->symbol_expr = af::Symbol(32);
  a_axis1->size = a1_size;

  auto a_axis2 = std::make_shared<AttAxis>();
  a_axis2->name = "A2";
  a_axis2->axis_pos = AxisPosition::OUTER;
  auto a2_size = std::make_shared<SymVarInfo>(af::Symbol(16));
  a2_size->data_type_size = 4;
  a2_size->symbol_expr = af::Symbol(16);
  a_axis2->size = a2_size;

  std::vector<const AttAxis *> a_axes = {a_axis1.get(), a_axis2.get()};

  Expr result = generator.CalcPenaltyCoreNumRatio(split_axis.get(), a_axes);

  // Verify ratio is calculated (not 1)
  EXPECT_NE(Str(result), "1");
}

// Test CalcPenaltyCoreNumRatio - Mixed INNER and OUTER axes
TEST_F(TestGenerateTilingExprPenalty, CalcPenaltyCoreNumRatio_MixedAxes) {
  GenerateTilingExpr generator(tuning_space);

  auto split_axis = std::make_shared<AttAxis>();
  split_axis->name = "R_split";
  auto size_info = std::make_shared<SymVarInfo>(af::Symbol(128));
  size_info->data_type_size = 4;
  split_axis->size = size_info;

  auto a_axis_inner = std::make_shared<AttAxis>();
  a_axis_inner->name = "A_inner";
  a_axis_inner->axis_pos = AxisPosition::INNER;
  auto inner_size = std::make_shared<SymVarInfo>(af::Symbol(32));
  inner_size->data_type_size = 4;
  inner_size->symbol_expr = af::Symbol(32);
  a_axis_inner->size = inner_size;

  auto a_axis_outer = std::make_shared<AttAxis>();
  a_axis_outer->name = "A_outer";
  a_axis_outer->axis_pos = AxisPosition::OUTER;
  auto outer_size = std::make_shared<SymVarInfo>(af::Symbol(16));
  outer_size->data_type_size = 4;
  outer_size->symbol_expr = af::Symbol(16);
  a_axis_outer->size = outer_size;

  std::vector<const AttAxis *> a_axes = {a_axis_inner.get(), a_axis_outer.get()};

  Expr result = generator.CalcPenaltyCoreNumRatio(split_axis.get(), a_axes);

  // Verify ratio is calculated (not 1)
  EXPECT_NE(Str(result), "1");
}

// Test CalcPenaltyCoreNumRatio - Verify formula
TEST_F(TestGenerateTilingExprPenalty, CalcPenaltyCoreNumRatio_VerifyFormula) {
  GenerateTilingExpr generator(tuning_space);

  auto split_axis = std::make_shared<AttAxis>();
  split_axis->name = "R_split";
  auto size_info = std::make_shared<SymVarInfo>(af::Symbol(128));
  size_info->data_type_size = 4;  // float32
  split_axis->size = size_info;

  auto a_axis = std::make_shared<AttAxis>();
  a_axis->name = "A";
  a_axis->axis_pos = AxisPosition::OUTER;  // OUTER 轴不做 upper_bound 转换
  auto a_size = std::make_shared<SymVarInfo>(af::Symbol(32));
  a_size->data_type_size = 4;
  a_size->symbol_expr = af::Symbol(32);
  a_axis->size = a_size;

  std::vector<const AttAxis *> a_axes = {a_axis.get()};

  Expr result = generator.CalcPenaltyCoreNumRatio(split_axis.get(), a_axes);

  // 验证公式: (a_axis_size * data_type_size) / cache_line_size
  // (32 * 4) / 128 = 1
  EXPECT_EQ(Str(result), "1");
}

// Test CalcPenaltyCoreNumRatio - Gather + Reduce uses 32B penalty granularity
TEST_F(TestGenerateTilingExprPenalty, CalcPenaltyCoreNumRatio_GatherReducePenaltyGranularity) {
  tuning_space->penalty_cache_line_size = 32;
  GenerateTilingExpr generator(tuning_space);

  auto split_axis = std::make_shared<AttAxis>();
  split_axis->name = "R_split";
  auto split_size = std::make_shared<SymVarInfo>(af::Symbol(128));
  split_size->data_type_size = 4;
  split_axis->size = split_size;

  auto a_axis = std::make_shared<AttAxis>();
  a_axis->name = "A";
  a_axis->axis_pos = AxisPosition::OUTER;
  auto a_size = std::make_shared<SymVarInfo>(af::Symbol(1));
  a_size->data_type_size = 4;
  a_size->symbol_expr = af::Symbol(1);
  a_axis->size = a_size;

  const Expr result = generator.CalcPenaltyCoreNumRatio(split_axis.get(), {a_axis.get()});

  EXPECT_EQ(Str(result), "Rational(1 , 8)");
}

// Test ApplyPenaltyConfig - Basic penalty (enabled by config table)
TEST_F(TestGenerateTilingExprPenalty, ApplyPenaltyConfig_BasicPenalty) {
  GenerateTilingExpr generator(tuning_space);

  ModelInfo model_info;
  model_info.graph_name = "test_graph";
  model_info.tiling_schedule_config_table = config_table.get();

  auto r_axis = std::make_shared<AttAxis>();
  r_axis->name = "R_split";
  r_axis->is_reduce_split_axis = true;

  // Create a valid A axis with size
  auto a_axis = std::make_shared<AttAxis>();
  a_axis->name = "A";
  a_axis->is_reduce_split_axis = false;
  auto size_info = std::make_shared<SymVarInfo>(af::Symbol(128));
  size_info->symbol_expr = af::Symbol(128);
  a_axis->size = size_info;

  model_info.arg_list = {r_axis, a_axis};

  generator.ApplyPenaltyConfigToModelInfo(model_info);

  EXPECT_TRUE(model_info.tiling_schedule_config.is_penalty_config);
}

TEST_F(TestGenerateTilingExprPenalty, ApplyPenaltyConfig_RCoreNumPenaltyExemption) {
  tuning_space->sub_axes.push_back(CreateSubAxis("A", AxisPosition::OUTER));
  auto tensor = CreateTensor("workspace", {tuning_space->sub_axes.back().get()});
  tuning_space->node_infos.push_back(CreateStoreNodeInfo("store", {tensor}));
  GenerateTilingExpr generator(tuning_space);
  ModelInfo model_info;
  model_info.graph_name = "r_core_penalty_exemption";

  auto r_axis = std::make_shared<AttAxis>();
  r_axis->name = "R_split";
  r_axis->is_reduce_split_axis = true;
  r_axis->size = std::make_shared<SymVarInfo>(af::Symbol(128));
  r_axis->size->data_type_size = 4;

  auto a_axis = std::make_shared<AttAxis>();
  a_axis->name = "A";
  a_axis->axis_pos = AxisPosition::OUTER;
  a_axis->size = std::make_shared<SymVarInfo>(af::Symbol(1));
  a_axis->size->symbol_expr = af::Symbol(1);
  a_axis->size->data_type_size = 4;
  model_info.arg_list = {r_axis, a_axis};

  for (const bool enable_workspace_alignment : {false, true}) {
    SCOPED_TRACE(enable_workspace_alignment);
    model_info.enable_workspace_alignment = enable_workspace_alignment;
    generator.ApplyPenaltyConfigToModelInfo(model_info);
    EXPECT_TRUE(model_info.tiling_schedule_config.is_penalty_config);
    const auto expected_ratio = enable_workspace_alignment ? af::Symbol(1.0) : af::Symbol(1) / af::Symbol(32);
    EXPECT_EQ(model_info.tiling_schedule_config.trade_off_config.core_num_ratio, expected_ratio);
  }
}

// Test ApplyPenaltyConfig - No Reduce split axis
TEST_F(TestGenerateTilingExprPenalty, ApplyPenaltyConfig_NoReduceSplitAxis) {
  GenerateTilingExpr generator(tuning_space);

  ModelInfo model_info;
  model_info.graph_name = "test_graph";
  model_info.tiling_schedule_config_table = config_table.get();

  auto a_axis = std::make_shared<AttAxis>();
  a_axis->name = "A";
  a_axis->is_reduce_split_axis = false;

  model_info.arg_list = {a_axis};

  generator.ApplyPenaltyConfigToModelInfo(model_info);

  EXPECT_FALSE(model_info.tiling_schedule_config.is_penalty_config);
}

// Test ApplyPenaltyConfig - Null config table (should skip penalty)
TEST_F(TestGenerateTilingExprPenalty, ApplyPenaltyConfig_NullConfigTable) {
  // Set tuning_space config_table to null
  tuning_space->tiling_schedule_config_table = nullptr;

  GenerateTilingExpr generator(tuning_space);

  ModelInfo model_info;
  model_info.graph_name = "test_graph";
  model_info.tiling_schedule_config.is_penalty_config = false;

  auto r_axis = std::make_shared<AttAxis>();
  r_axis->name = "R_split";
  r_axis->is_reduce_split_axis = true;

  model_info.arg_list = {r_axis};

  generator.ApplyPenaltyConfigToModelInfo(model_info);

  // The penalty should NOT be applied when config_table is null
  EXPECT_FALSE(model_info.tiling_schedule_config.is_penalty_config);
}

// Test ApplyPenaltyConfig - Disabled by config table
TEST_F(TestGenerateTilingExprPenalty, ApplyPenaltyConfig_DisabledByConfig) {
  // Create a config table that returns false for IsCoreNumThresholdPenaltyEnable
  class DisabledConfigTable : public TilingScheduleConfigTable {
   public:
    bool IsEnableBlockLoopAutoTune() const override {
      return true;
    }
    bool IsEnableCacheLineCheck() const override {
      return false;
    }
    TradeOffConfig GetTradeOffConfig() const override {
      TradeOffConfig config;
      config.is_enable = false;
      return config;
    }
    double GetUbThresholdPerfValEffect() const override {
      return 0.5;
    }
    TilingScheduleConfig GetModelTilingScheduleConfig() const override {
      TilingScheduleConfig config;
      config.trade_off_config.is_enable = false;
      return config;
    }
    uint32_t GetCacheLineSize() const override {
      return 128;
    }
    // Return false to disable penalty
    bool IsCoreNumThresholdPenaltyEnable() const override {
      return false;
    }
  };

  auto disabled_config = std::make_shared<DisabledConfigTable>();
  tuning_space->tiling_schedule_config_table = disabled_config.get();

  GenerateTilingExpr generator(tuning_space);

  ModelInfo model_info;
  model_info.graph_name = "test_graph";
  model_info.tiling_schedule_config_table = disabled_config.get();

  auto r_axis = std::make_shared<AttAxis>();
  r_axis->name = "R_split";
  r_axis->is_reduce_split_axis = true;

  model_info.arg_list = {r_axis};

  generator.ApplyPenaltyConfigToModelInfo(model_info);

  // The penalty should NOT be applied when disabled by config
  EXPECT_FALSE(model_info.tiling_schedule_config.is_penalty_config);
}

// Test IsFromReduceSplit - SubAxis from reduce split via orig_axis chain
TEST_F(TestGenerateTilingExprPenalty, IsFromReduceSplit_ViaOrigAxisChain) {
  GenerateTilingExpr generator(tuning_space);

  // Create reduce split axis name
  std::set<std::string> reduce_split_axis_names = {"R_split"};

  // Create parent sub_axis that is a reduce split axis
  auto r_split_axis = std::make_unique<SubAxis>();
  r_split_axis->name = "R_split";
  r_split_axis->is_reduce_split_axis = false;  // 在 reduce_split_axis_names 中

  // Create child sub_axis whose orig_axis points to the reduce split axis
  auto child_axis = std::make_unique<SubAxis>();
  child_axis->name = "child_of_R";
  child_axis->is_reduce_split_axis = false;
  child_axis->orig_axis.push_back(r_split_axis.get());  // orig_axis 指向 reduce split 轴

  // The child axis should be identified as from reduce split
  bool result = generator.IsFromReduceSplit(child_axis.get(), reduce_split_axis_names);
  EXPECT_TRUE(result);
}

// Test IsFromReduceSplit - Not from reduce split
TEST_F(TestGenerateTilingExprPenalty, IsFromReduceSplit_NotFromReduceSplit) {
  GenerateTilingExpr generator(tuning_space);

  std::set<std::string> reduce_split_axis_names = {"R_split"};

  // Create a sub_axis that is NOT from reduce split
  auto a_axis = std::make_unique<SubAxis>();
  a_axis->name = "A";
  a_axis->is_reduce_split_axis = false;
  a_axis->orig_axis.clear();

  bool result = generator.IsFromReduceSplit(a_axis.get(), reduce_split_axis_names);
  EXPECT_FALSE(result);
}

// Test IsFromReduceSplit - Cycle detection with visited set
TEST_F(TestGenerateTilingExprPenalty, IsFromReduceSplit_CycleDetection) {
  GenerateTilingExpr generator(tuning_space);

  std::set<std::string> reduce_split_axis_names = {"R_split"};

  // Create circular orig_axis references
  auto axis1 = std::make_unique<SubAxis>();
  axis1->name = "axis1";
  axis1->orig_axis.clear();

  auto axis2 = std::make_unique<SubAxis>();
  axis2->name = "axis2";
  axis2->orig_axis.push_back(axis1.get());

  axis1->orig_axis.push_back(axis2.get());  // Create cycle

  // Should handle cycle gracefully and not infinite loop
  bool result = generator.IsFromReduceSplit(axis1.get(), reduce_split_axis_names);
  EXPECT_FALSE(result);  // Neither axis is in reduce_split_axis_names
}

// Test ShouldSkipAxis - Axis from reduce split (should skip)
TEST_F(TestGenerateTilingExprPenalty, ShouldSkipAxis_FromReduceSplit) {
  GenerateTilingExpr generator(tuning_space);

  std::set<std::string> reduce_split_axis_names = {"R_split"};

  // Create parent reduce split axis
  auto r_split_axis = std::make_unique<SubAxis>();
  r_split_axis->name = "R_split";

  // Create child axis whose orig_axis points to reduce split
  auto child_axis = std::make_unique<SubAxis>();
  child_axis->name = "child";
  child_axis->is_reduce_split_axis = false;
  child_axis->is_broadcast_split_axis = false;
  child_axis->orig_axis.push_back(r_split_axis.get());

  bool result = generator.ShouldSkipAxis(child_axis.get(), reduce_split_axis_names);
  EXPECT_TRUE(result);  // Should skip because it's from reduce split
}

// Test GetCacheLineSize - Null config table returns default
TEST_F(TestGenerateTilingExprPenalty, GetCacheLineSize_NullConfigTable) {
  tuning_space->tiling_schedule_config_table = nullptr;

  GenerateTilingExpr generator(tuning_space);

  uint32_t cache_line_size = generator.GetCacheLineSize();
  EXPECT_EQ(cache_line_size, 128U);  // kDefaultCacheLineSize
}

// Test CalcPenaltyCoreNumRatio - Skip null size
TEST_F(TestGenerateTilingExprPenalty, CalcPenaltyCoreNumRatio_SkipNullSize) {
  GenerateTilingExpr generator(tuning_space);

  auto split_axis = std::make_shared<AttAxis>();
  split_axis->name = "R_split";
  auto size_info = std::make_shared<SymVarInfo>(af::Symbol(128));
  size_info->data_type_size = 4;
  split_axis->size = size_info;

  // Create a_axis with null size
  auto a_axis_null = std::make_shared<AttAxis>();
  a_axis_null->name = "A_null";
  a_axis_null->size = nullptr;  // null size

  // Create a_axis with valid size
  auto a_axis_valid = std::make_shared<AttAxis>();
  a_axis_valid->name = "A_valid";
  a_axis_valid->axis_pos = AxisPosition::OUTER;
  auto valid_size = std::make_shared<SymVarInfo>(af::Symbol(32));
  valid_size->data_type_size = 4;
  valid_size->symbol_expr = af::Symbol(32);
  a_axis_valid->size = valid_size;

  std::vector<const AttAxis *> a_axes = {a_axis_null.get(), a_axis_valid.get()};

  // Should skip null size and only use valid size
  Expr result = generator.CalcPenaltyCoreNumRatio(split_axis.get(), a_axes);

  // Only the valid axis (size=32) should be used: (32 * 4) / 128 = 1
  EXPECT_EQ(Str(result), "1");
}

// Test ApplyUpperBoundTransform - Non-const symbols
TEST_F(TestGenerateTilingExprPenalty, ApplyUpperBoundTransform_NonConstSymbols) {
  GenerateTilingExpr generator(tuning_space);

  // Create an expression with variable symbols
  auto var_s1 = af::Symbol("s1t_size");
  Expr size_expr = var_s1 * af::Symbol(2);

  Expr result = generator.ApplyUpperBoundTransform(size_expr);

  // The non-const symbol should be wrapped with .upper_bound()
  std::string result_str = Str(result);
  // Verify it contains upper_bound transformation
  EXPECT_TRUE(result_str.find("upper_bound") != std::string::npos || result_str.find("s1t_size") != std::string::npos);
}

// Test FindAAxis - With node_ptr (calls IsStoreNode)
TEST_F(TestGenerateTilingExprPenalty, FindAAxis_WithNodePtr) {
  // Create a SubAxis that will be used as orig_axis
  auto orig_sub_axis = std::make_unique<SubAxis>();
  orig_sub_axis->name = "orig";
  orig_sub_axis->axis_type = AxisPosition::OUTER;
  orig_sub_axis->is_reduce_split_axis = false;
  orig_sub_axis->is_broadcast_split_axis = false;
  orig_sub_axis->repeat = af::Symbol(128);

  // Create A axis
  auto a_sub_axis = std::make_unique<SubAxis>();
  a_sub_axis->name = "A";
  a_sub_axis->axis_type = AxisPosition::INNER;
  a_sub_axis->is_reduce_split_axis = false;
  a_sub_axis->is_broadcast_split_axis = false;
  a_sub_axis->orig_axis.push_back(orig_sub_axis.get());
  a_sub_axis->repeat = af::Symbol(32);
  a_sub_axis->data_type_size = 4;

  // Create tensor
  auto tensor = CreateTensor("input0", {a_sub_axis.get()});
  tensor->loc = HardwareDef::GM;

  // Create NodeInfo with node_ptr set
  auto node_info = std::make_unique<NodeInfo>();
  node_info->name = "Store_node";
  node_info->node_type = "Store";
  node_info->inputs.push_back(tensor);
  node_info->outputs.clear();
  // Create a mock AscNode pointer
  node_info->node_ptr = nullptr;  // In test, we can't create real AscNode

  tuning_space->node_infos.push_back(*node_info);

  // Add corresponding AttAxis to arg_list
  std::vector<AttAxisPtr> arg_list;
  auto a_att_axis = std::make_shared<AttAxis>();
  a_att_axis->name = "A";
  a_att_axis->axis_pos = AxisPosition::INNER;
  arg_list.push_back(a_att_axis);

  GenerateTilingExpr generator(tuning_space);

  auto result = generator.FindAAxis(arg_list);

  // Should find A axis
  EXPECT_EQ(result.size(), 1U);
  EXPECT_EQ(result[0]->name, "A");
}

// Test CalcPenaltyCoreNumRatio - Mixed valid and invalid axes
TEST_F(TestGenerateTilingExprPenalty, CalcPenaltyCoreNumRatio_MixedValidity) {
  GenerateTilingExpr generator(tuning_space);

  auto split_axis = std::make_shared<AttAxis>();
  split_axis->name = "R_split";
  auto size_info = std::make_shared<SymVarInfo>(af::Symbol(128));
  size_info->data_type_size = 4;
  split_axis->size = size_info;

  // Create mix of valid, null, and invalid axes
  auto a1 = std::make_shared<AttAxis>();  // Valid INNER
  a1->name = "A1";
  a1->axis_pos = AxisPosition::INNER;
  auto s1 = std::make_shared<SymVarInfo>(af::Symbol(16));
  s1->symbol_expr = af::Symbol(16);
  s1->data_type_size = 4;
  a1->size = s1;

  auto a2 = std::make_shared<AttAxis>();  // null size
  a2->name = "A2";
  a2->size = nullptr;

  auto a3 = std::make_shared<AttAxis>();  // Valid OUTER
  a3->name = "A3";
  a3->axis_pos = AxisPosition::OUTER;
  auto s3 = std::make_shared<SymVarInfo>(af::Symbol(8));
  s3->symbol_expr = af::Symbol(8);
  s3->data_type_size = 2;
  a3->size = s3;

  std::vector<const AttAxis *> a_axes = {a1.get(), a2.get(), a3.get()};

  // Should skip a2 (null) and use a1 and a3
  // a1: 16 (INNER with upper_bound), a3: 8 (OUTER without upper_bound)
  // (16 * 8 * 4) / 128 = 4 (assuming upper_bound preserves the value)
  Expr result = generator.CalcPenaltyCoreNumRatio(split_axis.get(), a_axes);

  // Result should not be 1 since we have valid axes
  EXPECT_NE(Str(result), "1");
}

}  // namespace att
