/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>

#include "perf_scenario_utils.h"
#include "ascir_ops.h"
#include "api_perf_register/utils/api_perf_utils.h"
#include "codegen_api_param/codegen_api_param.h"
#include "common/platform_context.h"
#include "graph_construct_utils.h"
#include "tests/common/att/soc_perf_test_config.h"
#include "tests/ut/att/testcase/gen_model_info/api_perf_register/runtime_stub.h"
#include "v35/att/api_perf_register/ascendc_api_perf/data_move_api_perf_v2.h"
#include "v35/att/api_perf_register/ascir_api_perf_v2.h"

using namespace att;
using namespace af::sym;
using namespace af::ascir;
using namespace att::perf_scenario;

namespace {
class CalibratedAscirPerfV2 : public ::testing::Test {
 protected:
  void SetUp() override {
    ge::RuntimeStub::Install(&runtime_);
    ASSERT_EQ(soc_perf_test::SetSocPerfTestPlatform(soc_perf_test::kV35NpuArch, soc_perf_test::kSocEmpty), af::SUCCESS);
  }

  void TearDown() override {
    ge::RuntimeStub::UnInstall(&runtime_);
    ge::PlatformContext::GetInstance().Reset();
  }

 private:
  ge::RuntimeStubV2 runtime_;
};

TEST_F(CalibratedAscirPerfV2, Fp32TransposeUnitMiddleChargesMeasuredTailRepeats) {
  const auto model = [](int32_t rows, int32_t width, const Expr &middle, const std::string &dtype) {
    return GetTransposePerfExpr(3, 3, {CreateExpr(rows), middle, CreateExpr(width)}, {},
                                {CreateExpr(width), CreateExpr(rows * width), CreateExpr(1)}, true, dtype,
                                {CreateExpr(width) * middle, CreateExpr(width), CreateExpr(1)});
  };
  const auto one = CreateExpr(1);
  EXPECT_NEAR(EvaluateTransposeCycles(model(12, 1007, one, kFloat32)), 4704.268794337531, 1e-6);
  EXPECT_DOUBLE_EQ(EvaluateTransposeCycles(model(12, 900, one, kFloat32)), 3450.0);
  struct Measurement {
    int32_t rows;
    int32_t width;
    double cycles;
  };
  const std::vector<Measurement> holdout = {
      {9, 975, 3493.3}, {11, 1031, 4456.6}, {13, 1003, 5081.0}, {15, 992, 5665.3}, {16, 1089, 6602.1}};
  for (const auto &point : holdout) {
    EXPECT_NEAR(EvaluateTransposeCycles(model(point.rows, point.width, one, kFloat32)), point.cycles,
                point.cycles * 0.02);
  }
  const auto middle = CreateExpr("middle");
  const auto symbolic = model(12, 1007, middle, kFloat32);
  EXPECT_NEAR(EvaluateTransposeCycles(symbolic, {{middle, one}}), 4704.268794337531, 1e-6);
  EXPECT_DOUBLE_EQ(EvaluateTransposeCycles(symbolic, {{middle, CreateExpr(2)}}),
                   EvaluateTransposeCycles(model(12, 1007, CreateExpr(2), kFloat32)));
  EXPECT_EQ(model(12, 1007, middle, kFloat16).perf.ternary_ops.count(CreateExpr("transpose_fp32_unit_middle")), 0U);
}

TEST_F(CalibratedAscirPerfV2, TransposeUnitOuterMatchesIndependentComponentMeasurements) {
  // Separate INDEX/GATHER/FULL microkernels; these points are excluded from coefficient fitting.
  const std::vector<std::pair<std::string, double>> measured = {
      {kFloat16, 9762.27}, {kFloat32, 11200.04}, {kInt64, 20246.41}};
  for (const auto &item : measured) {
    SCOPED_TRACE(item.first);
    const auto result = GetTransposePerfExpr(3, 3, {CreateExpr(1), CreateExpr(64), CreateExpr(128)}, {},
                                             {CreateExpr(8192), CreateExpr(1), CreateExpr(64)}, true, item.first);
    EXPECT_NEAR(EvaluateTransposeCycles(result), item.second, item.second * 0.1);
  }
}

TEST_F(CalibratedAscirPerfV2, TransposeUnitOuterUsesBankGcdForNonPowerOfTwoStride) {
  for (const auto &dtype : {kFloat16, kFloat32, kInt64}) {
    const auto stride_twelve = GetTransposePerfExpr(3, 3, {CreateExpr(1), CreateExpr(12), CreateExpr(64)}, {},
                                                    {CreateExpr(768), CreateExpr(1), CreateExpr(12)}, true, dtype);
    const auto stride_four = GetTransposePerfExpr(3, 3, {CreateExpr(1), CreateExpr(4), CreateExpr(192)}, {},
                                                  {CreateExpr(768), CreateExpr(1), CreateExpr(4)}, true, dtype);
    EXPECT_NEAR(EvaluateTransposeCycles(stride_twelve), EvaluateTransposeCycles(stride_four), 1e-6);
  }
}

TEST_F(CalibratedAscirPerfV2, TransposeUnitOuterSymbolicPredicateAndGcdMatchConstantDescriptors) {
  const Expr outer = CreateExpr("unit_outer"), rows = CreateExpr("transpose_rows");
  const auto symbolic = GetTransposePerfExpr(3, 3, {outer, rows, CreateExpr(64)}, {CreateExpr(2)},
                                             {CreateExpr(768), CreateExpr(1), rows}, true, kFloat32);
  for (const int64_t o : {1L, 2L}) {
    for (const int64_t r : {4L, 12L, 32L, 64L}) {
      const auto constant = GetTransposePerfExpr(3, 3, {CreateExpr(o), CreateExpr(r), CreateExpr(64)}, {CreateExpr(2)},
                                                 {CreateExpr(768), CreateExpr(1), CreateExpr(r)}, true, kFloat32);
      EXPECT_NEAR(EvaluateTransposeCycles(symbolic, {{outer, CreateExpr(o)}, {rows, CreateExpr(r)}}),
                  EvaluateTransposeCycles(constant), 1e-6);
    }
  }
}

TEST_F(CalibratedAscirPerfV2, TransposeTwoInnerIndexStoreUsesInt32ForWideElements) {
  for (const auto &item : std::vector<std::pair<std::string, double>>{{kFloat32, 667.0}, {kInt32, 747.0}}) {
    SCOPED_TRACE(item.first);
    const auto result = GetTransposePerfExpr(2, 2, {CreateExpr(8), CreateExpr(128)}, {}, {CreateExpr(1), CreateExpr(8)},
                                             true, item.first);
    // Sixteen 32-bit index vectors use the existing one-cycle Store entry.
    // Other arithmetic retains its existing dtype-specific table entries.
    EXPECT_DOUBLE_EQ(EvaluateTransposeCycles(result), item.second);
  }
}

TEST_F(CalibratedAscirPerfV2, TransposeTwoInnerIndexStoreSymbolicRepeatsMatchConstantCalls) {
  const Expr width = CreateExpr("index_width");
  const auto symbolic =
      GetTransposePerfExpr(2, 2, {CreateExpr(8), width}, {}, {CreateExpr(1), CreateExpr(8)}, true, kFloat32);
  for (const auto count : {8, 64, 128, 1585}) {
    const auto constant = GetTransposePerfExpr(2, 2, {CreateExpr(8), CreateExpr(count)}, {},
                                               {CreateExpr(1), CreateExpr(8)}, true, kFloat32);
    EXPECT_NEAR(EvaluateTransposeCycles(symbolic, {{width, CreateExpr(count)}}), EvaluateTransposeCycles(constant),
                1e-6);
  }
  const auto pgo =
      GetTransposePerfExpr(2, 2, {CreateExpr(8), CreateExpr(1585)}, {}, {CreateExpr(1), CreateExpr(8)}, true, kFloat32);
  EXPECT_DOUBLE_EQ(EvaluateTransposeCycles(pgo), 6157.0);
}

TEST_F(CalibratedAscirPerfV2, TransposeDim3Inner1ShortAlignedFp32GatherCountsActiveLanes) {
  const Expr lanes = CreateExpr("tail_lanes");
  const Expr stride = CreateExpr("tail_stride");
  const auto make = [](const Expr &tail, const Expr &pitch) {
    return GetTransposePerfExpr(3, 1, {CreateExpr(2), CreateExpr(3), tail}, {}, {CreateExpr(1), CreateExpr(32), pitch},
                                true, kFloat32);
  };
  const auto symbolic = make(lanes, stride);
  for (const auto &sample :
       {std::make_pair(7, 96), std::make_pair(8, 96), std::make_pair(9, 96), std::make_pair(7, 97)}) {
    const auto tail = CreateExpr(sample.first);
    const auto pitch = CreateExpr(sample.second);
    EXPECT_NEAR(EvaluateTransposeCycles(symbolic, {{lanes, tail}, {stride, pitch}}),
                EvaluateTransposeCycles(make(tail, pitch)), 1e-6);
  }
  EXPECT_DOUBLE_EQ(EvaluateTransposeCycles(make(CreateExpr(7), CreateExpr(96))), 140.0);
  EXPECT_DOUBLE_EQ(EvaluateTransposeCycles(make(CreateExpr(8), CreateExpr(96))), 146.0);
  EXPECT_DOUBLE_EQ(EvaluateTransposeCycles(make(CreateExpr(9), CreateExpr(96))), 482.0);
  EXPECT_DOUBLE_EQ(EvaluateTransposeCycles(make(CreateExpr(7), CreateExpr(97))), 110.0);
}

TEST_F(CalibratedAscirPerfV2, TransposeDim2Inner1ShortAlignedFp32GatherCountsActiveLanes) {
  const Expr lanes = CreateExpr("dim2_tail_lanes");
  const Expr stride = CreateExpr("dim2_tail_stride");
  const auto make = [](const Expr &tail, const Expr &pitch) {
    return GetTransposePerfExpr(2, 1, {CreateExpr(30), tail}, {}, {CreateExpr(1), pitch}, true, kFloat32);
  };
  const auto symbolic = make(lanes, stride);
  const auto eval = [&](int32_t tail, int32_t pitch) {
    const std::vector<std::pair<Expr, Expr>> bindings{{lanes, CreateExpr(tail)}, {stride, CreateExpr(pitch)}};
    const double cycles = EvaluateTransposeCycles(symbolic, bindings);
    EXPECT_NEAR(cycles, EvaluateTransposeCycles(make(CreateExpr(tail), CreateExpr(pitch))), 1e-6);
    return cycles;
  };
  EXPECT_DOUBLE_EQ(eval(8, 32) - eval(7, 32), 30.0);
  EXPECT_DOUBLE_EQ(eval(9, 32) - eval(8, 32), 1680.0);
  EXPECT_DOUBLE_EQ(eval(8, 33) - eval(7, 33), 0.0);
}

TEST_F(CalibratedAscirPerfV2, TransposeDim2Fp16GatherMatchesAlignedShortLaneMeasurements) {
  const Expr pitch = CreateExpr("fp16_dim2_pitch");
  const auto make = [](const Expr &stride, bool dense) {
    return GetTransposePerfExpr(2, 1, {CreateExpr(544), CreateExpr(32)}, {}, {CreateExpr(1), stride}, true, kFloat16,
                                {CreateExpr(dense ? 32 : 48), CreateExpr(1)});
  };
  const auto symbolic = make(pitch, true);
  EXPECT_NEAR(EvaluateTransposeCycles(symbolic, {{pitch, CreateExpr(544)}}), 9899.65, 10.0);
  EXPECT_NEAR(EvaluateTransposeCycles(symbolic, {{pitch, CreateExpr(544)}}),
              EvaluateTransposeCycles(make(CreateExpr(544), true)), 1e-6);
  EXPECT_DOUBLE_EQ(EvaluateTransposeCycles(make(CreateExpr(544), false)), 35452.0);
  EXPECT_DOUBLE_EQ(EvaluateTransposeCycles(symbolic, {{pitch, CreateExpr(560)}}),
                   EvaluateTransposeCycles(make(CreateExpr(560), true)));
}

TEST_F(CalibratedAscirPerfV2, TransposeFp32FourDimUnitAxisMatchesMeasuredVectorCost) {
  const auto rows = CreateExpr("transpose_four_dim_rows");
  const auto unit = CreateExpr("transpose_four_dim_unit");
  const auto result = GetTransposePerfExpr(4, 3, {rows, CreateExpr(16), unit, CreateExpr(48)}, {CreateExpr(2)},
                                           {CreateExpr(768), CreateExpr(1), CreateExpr(9216), CreateExpr(16)}, true,
                                           kFloat32, {CreateExpr(768), CreateExpr(48), CreateExpr(48), CreateExpr(1)});
  EXPECT_NEAR(EvaluateTransposeCycles(result, {{rows, CreateExpr(12)}, {unit, CreateExpr(1)}}), 2.0 * 5193.67, 0.02);
  const auto outside =
      GetTransposePerfExpr(4, 3, {CreateExpr(12), CreateExpr(16), CreateExpr(2), CreateExpr(48)}, {CreateExpr(2)},
                           {CreateExpr(768), CreateExpr(1), CreateExpr(9216), CreateExpr(16)}, true, kFloat32,
                           {CreateExpr(768), CreateExpr(48), CreateExpr(48), CreateExpr(1)});
  EXPECT_NEAR(EvaluateTransposeCycles(result, {{rows, CreateExpr(12)}, {unit, CreateExpr(2)}}),
              EvaluateTransposeCycles(outside), 1e-6);
}

TEST_F(CalibratedAscirPerfV2, TransposeDim3Fp16GatherMatchesMaskedTwoOuterLoop) {
  const auto make = [](const Expr &lanes, int32_t dst_row) {
    return GetTransposePerfExpr(3, 1, {CreateExpr(36), CreateExpr(58), lanes}, {},
                                {CreateExpr(1), CreateExpr(48), CreateExpr(2784)}, true, kFloat16,
                                {CreateExpr(dst_row * 58), CreateExpr(dst_row), CreateExpr(1)});
  };
  const Expr lanes = CreateExpr("fp16_dim3_lanes");
  const auto symbolic = make(lanes, 16);
  const auto cycles = EvaluateTransposeCycles(make(CreateExpr(4), 16));
  EXPECT_NEAR(cycles, 10807, 10807 * 0.05);  // Independent production-geometry bench.
  EXPECT_NEAR(cycles, EvaluateTransposeCycles(symbolic, {{lanes, CreateExpr(4)}}), 1e-6);
  EXPECT_GT(EvaluateTransposeCycles(make(CreateExpr(4), 32)), 10 * cycles);
  EXPECT_NEAR(EvaluateTransposeCycles(symbolic, {{lanes, CreateExpr(17)}}),
              EvaluateTransposeCycles(make(CreateExpr(17), 16)), 1e-6);
}

TEST_F(CalibratedAscirPerfV2, TransposeUnitOuterB64SymbolicRowsMatchConstantDescriptors) {
  const Expr rows = CreateExpr("b64_rows"), columns = CreateExpr("b64_columns");
  const auto symbolic = GetTransposePerfExpr(3, 3, {CreateExpr(1), rows, columns}, {},
                                             {rows * columns, CreateExpr(1), rows}, true, kInt64);
  for (const int64_t r : {5L, 12L, 28L, 40L}) {
    for (const int64_t c : {8L, 16L, 20L, 36L, 56L, 104L}) {
      const auto constant = GetTransposePerfExpr(3, 3, {CreateExpr(1), CreateExpr(r), CreateExpr(c)}, {},
                                                 {CreateExpr(r * c), CreateExpr(1), CreateExpr(r)}, true, kInt64);
      EXPECT_NEAR(EvaluateTransposeCycles(symbolic, {{rows, CreateExpr(r)}, {columns, CreateExpr(c)}}),
                  EvaluateTransposeCycles(constant), 1e-6);
    }
  }
}

TEST_F(CalibratedAscirPerfV2, TransposeUnitOuterB32ShortRowsMatchIndependentMeasurements) {
  // Independent short-row holdout; no coefficient was fitted to these points.
  for (const auto &dtype : {kFloat32, kInt32}) {
    const auto short_row = GetTransposePerfExpr(3, 3, {CreateExpr(1), CreateExpr(20), CreateExpr(14)}, {},
                                                {CreateExpr(280), CreateExpr(1), CreateExpr(20)}, true, dtype);
    const auto near_boundary = GetTransposePerfExpr(3, 3, {CreateExpr(1), CreateExpr(80), CreateExpr(26)}, {},
                                                    {CreateExpr(2080), CreateExpr(1), CreateExpr(80)}, true, dtype);
    EXPECT_NEAR(EvaluateTransposeCycles(short_row), 379.6842, 379.6842 * 0.1);
    EXPECT_NEAR(EvaluateTransposeCycles(near_boundary), 1718.0726, 1718.0726 * 0.1);
  }
}

TEST_F(CalibratedAscirPerfV2, TransposeUnitOuterB32SymbolicShortRowBoundaryMatchesConstantDescriptors) {
  const Expr rows = CreateExpr("b32_rows"), columns = CreateExpr("b32_columns");
  const auto symbolic = GetTransposePerfExpr(3, 3, {CreateExpr(1), rows, columns}, {},
                                             {rows * columns, CreateExpr(1), rows}, true, kFloat32);
  for (const int64_t r : {12L, 48L, 80L}) {
    for (const int64_t c : {8L, 24L, 31L, 32L, 33L, 40L}) {
      const auto constant = GetTransposePerfExpr(3, 3, {CreateExpr(1), CreateExpr(r), CreateExpr(c)}, {},
                                                 {CreateExpr(r * c), CreateExpr(1), CreateExpr(r)}, true, kFloat32);
      EXPECT_NEAR(EvaluateTransposeCycles(symbolic, {{rows, CreateExpr(r)}, {columns, CreateExpr(c)}}),
                  EvaluateTransposeCycles(constant), 1e-6);
    }
  }
}

TEST_F(CalibratedAscirPerfV2, TransposeUnitOuterSymbolicPaddedInputKeepsLegacyBranch) {
  const Expr pitch = CreateExpr("transpose_input_pitch");
  const auto symbolic = GetTransposePerfExpr(3, 3, {CreateExpr(1), CreateExpr(64), CreateExpr(128)}, {},
                                             {CreateExpr(8192), CreateExpr(1), pitch}, true, kFloat32);
  const auto compact = GetTransposePerfExpr(3, 3, {CreateExpr(1), CreateExpr(64), CreateExpr(128)}, {},
                                            {CreateExpr(8192), CreateExpr(1), CreateExpr(64)}, true, kFloat32);
  EXPECT_NEAR(EvaluateTransposeCycles(symbolic, {{pitch, CreateExpr(64)}}), EvaluateTransposeCycles(compact), 1e-6);
  EXPECT_NEAR(EvaluateTransposeCycles(symbolic, {{pitch, CreateExpr(128)}}), 6680.0, 1e-6);
}

TEST_F(CalibratedAscirPerfV2, TransposeUnitOuterPreservesPaddedInputLegacyCost) {
  const auto padded = GetTransposePerfExpr(3, 3, {CreateExpr(1), CreateExpr(64), CreateExpr(128)}, {},
                                           {CreateExpr(8192), CreateExpr(1), CreateExpr(128)}, true, kFloat32);
  EXPECT_NEAR(EvaluateTransposeCycles(padded), 6680.0, 1e-6);
}

void ExpectCompactLoopStoreCost(const PerfOutputInfo &perf, const Expr &rows, const Expr &cores,
                                const Expr &short_width) {
  const auto one = CreateExpr(1);
  const auto guard = perf.ternary_ops.at(CreateExpr("store_compact_loop_short_rows")).DeepCopyIfCase();
  EXPECT_EQ(af::SymbolicUtils::StaticCheckEq(
                guard->GetCondLeft().Replace({{rows, CreateExpr(4)}, {cores, CreateExpr(3)}, {short_width, one}}),
                CreateExpr(0)),
            af::TriBool::kTrue);
  EXPECT_EQ(af::SymbolicUtils::StaticCheckEq(
                guard->GetCondLeft().Replace({{rows, CreateExpr(4)}, {cores, CreateExpr(13)}, {short_width, one}}),
                CreateExpr(0)),
            af::TriBool::kFalse);
  const auto fitted = perf.ternary_ops.at(CreateExpr("store_compact_loop_per_call")).DeepCopyIfCase();
  double cycles = 0.0;
  ASSERT_TRUE(fitted->GetChoiceA()->GetExpr().GetConstValue(cycles));
  EXPECT_NEAR(cycles, 689.7, 0.001);
  ASSERT_TRUE(fitted->GetChoiceB()->GetExpr().Replace({{rows, CreateExpr(4)}}).GetConstValue(cycles));
  EXPECT_NEAR(cycles, 1694.484, 0.001);
}

TEST_F(CalibratedAscirPerfV2, TestCompactLoopStoreMatchesFoldedDescriptor) {
  const auto rows = CreateExpr("store_rows");
  const auto cores = CreateExpr("block_dim");
  const auto one = CreateExpr(1);
  TensorShapeInfo shape;
  shape.data_type = "float16";
  shape.data_type_size = 2;
  shape.repeats = {CreateExpr(5), rows, CreateExpr(2)};
  shape.origin_repeats = shape.repeats;
  shape.strides = {CreateExpr(16), CreateExpr(2), one};
  shape.gm_strides = {CreateExpr(24), CreateExpr(6), one};
  auto folded = shape;
  const auto short_width = CreateExpr("store_short_width");
  folded.repeats = {CreateExpr(5), rows, short_width, CreateExpr(2)};
  folded.origin_repeats = folded.repeats;
  folded.strides = {CreateExpr(16), short_width * CreateExpr(2), CreateExpr(2), one};
  folded.gm_strides = {CreateExpr(24), CreateExpr(6), CreateExpr(2), one};
  SubAxis axis;
  axis.axis_type = AxisPosition::ORIGIN;
  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Store", 1, 1);
  node.outputs.push_back(std::make_shared<Tensor>());
  node.outputs.front()->dim_info = {&axis, &axis, &axis, &axis};
  auto store = ApiPerfFactory::Instance().Create("StoreV2");
  ASSERT_NE(store, nullptr);
  const auto perf_func = store->GetPerfFunc();
  const auto selected = CreateExpr("store_compact_loop_short_rows");
  for (const auto &layout : {shape, folded}) {
    PerfOutputInfo perf;
    ASSERT_EQ(perf_func({layout}, {layout}, node, perf), af::SUCCESS);
    ASSERT_EQ(perf.pipe_res.at(PipeType::AIV_MTE3), selected);
    ExpectCompactLoopStoreCost(perf, rows, cores, short_width);
  }
  axis.axis_type = AxisPosition::INNER;
  PerfOutputInfo incompatible;
  ASSERT_EQ(perf_func({folded}, {folded}, node, incompatible), af::SUCCESS);
  EXPECT_EQ(incompatible.ternary_ops.count(selected), 0U);
}

TEST_F(CalibratedAscirPerfV2, TestStoreStrideUsesFoldedInnermostAxis) {
  const auto tile = CreateExpr("store_tile");
  const auto inner = CreateExpr(3);
  const auto one = CreateExpr(1);
  const auto rows = CreateExpr("store_rows");
  TensorShapeInfo raw;
  raw.data_type = "float16";
  raw.data_type_size = 2;
  raw.repeats = {CreateExpr(331), rows, tile, inner};
  raw.origin_repeats = raw.repeats;
  raw.gm_strides = {CreateExpr(127104), CreateExpr(384), inner, one};
  raw.strides = {CreateExpr(160), tile * inner, inner, one};
  auto physical = raw;
  physical.repeats = {raw.repeats[0], rows, tile * inner};
  physical.origin_repeats = physical.repeats;
  physical.gm_strides = {raw.gm_strides[0], raw.gm_strides[1], one};
  physical.strides = {raw.strides[0], tile * inner, one};

  SubAxis axis;
  axis.axis_type = AxisPosition::ORIGIN;
  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Store", 1, 1);
  node.outputs.push_back(std::make_shared<Tensor>());
  node.outputs.front()->dim_info = {&axis, &axis, &axis, &axis};
  auto store = ApiPerfFactory::Instance().Create("StoreV2");
  ASSERT_NE(store, nullptr);
  PerfOutputInfo raw_perf;
  PerfOutputInfo physical_perf;
  ASSERT_EQ(store->GetPerfFunc()({raw}, {raw}, node, raw_perf), af::SUCCESS);
  ASSERT_EQ(store->GetPerfFunc()({physical}, {physical}, node, physical_perf), af::SUCCESS);
  EXPECT_EQ(raw_perf.ternary_ops.count(CreateExpr("mte3_block_len")), 1U);
  EXPECT_EQ(
      Str(raw_perf.pipe_res.at(PipeType::AIV_MTE3).Replace(ConcursiveReplaceVars(raw_perf.ternary_ops))),
      Str(physical_perf.pipe_res.at(PipeType::AIV_MTE3).Replace(ConcursiveReplaceVars(physical_perf.ternary_ops))));

  // A shorter loop retains the previous stride path even if the tail is foldable.
  raw.repeats[0] = CreateExpr(32);
  raw.origin_repeats[0] = raw.repeats[0];
  PerfOutputInfo short_loop_perf;
  ASSERT_EQ(store->GetPerfFunc()({raw}, {raw}, node, short_loop_perf), af::SUCCESS);
  EXPECT_EQ(short_loop_perf.ternary_ops.count(CreateExpr("mte3_block_len")), 0U);
}

void ExpectDmaDomain(const Expr &condition, const std::vector<Expr> &symbols,
                     const std::vector<std::vector<int32_t>> &points, af::TriBool expected) {
  for (const auto &values : points) {
    ASSERT_EQ(symbols.size(), values.size());
    std::vector<std::pair<Expr, Expr>> bindings;
    for (size_t i = 0; i < symbols.size(); ++i) {
      bindings.emplace_back(symbols[i], CreateExpr(values[i]));
    }
    EXPECT_EQ(af::SymbolicUtils::StaticCheckEq(condition.Replace(bindings), af::sym::kSymbolZero), expected);
  }
}

bool HasLoadFallbackChain(const std::map<Expr, TernaryOp, ExprCmp> &ternary_ops, Expr current, const Expr &expected) {
  for (size_t depth = 0; depth <= ternary_ops.size(); ++depth) {
    if (current == expected) {
      return true;
    }
    const auto it = ternary_ops.find(current);
    if (it == ternary_ops.end()) {
      return false;
    }
    const auto fallback = it->second.DeepCopyIfCase()->GetChoiceB();
    if (fallback == nullptr) {
      return false;
    }
    current = fallback->GetExpr();
  }
  return false;
}

void CheckFlattenedAlignedSingleRowLoad(NodeDetail node, const Expr &width, const Expr &cores, const Expr &calibrated) {
  node.input_dtype = {"float32"};
  node.output_dtype = {"float32"};
  node.input_dims = {width};
  node.gm_stride = CreateExpr(0);
  PerfOutputInfo flattened;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, flattened), af::SUCCESS);
  ASSERT_EQ(flattened.ternary_ops.count(calibrated), 1U);
  const auto flat_guard = flattened.ternary_ops.at(calibrated).DeepCopyIfCase();
  ASSERT_NE(flat_guard, nullptr);
  const auto in_flattened_range = [&](int32_t elements, int32_t core_count) {
    return af::SymbolicUtils::StaticCheckEq(
        flat_guard->GetCondLeft().Replace({{width, CreateExpr(elements)}, {cores, CreateExpr(core_count)}}),
        af::sym::kSymbolZero);
  };
  EXPECT_EQ(in_flattened_range(2112, 42), af::TriBool::kTrue);
  EXPECT_EQ(in_flattened_range(16384, 56), af::TriBool::kTrue);
  EXPECT_EQ(in_flattened_range(2048, 42), af::TriBool::kFalse);
  EXPECT_EQ(in_flattened_range(16392, 56), af::TriBool::kFalse);
  EXPECT_EQ(in_flattened_range(2400, 41), af::TriBool::kFalse);
}

TEST_F(CalibratedAscirPerfV2, TestAlignedSingleRowLoadEstimate) {
  const Expr rows = CreateExpr("rows");
  const Expr width = CreateExpr("width");
  const Expr cores = CreateExpr("block_dim");
  NodeDetail node;
  node.name = "Load";
  node.optype = kLoad;
  node.input_dtype = {"float32"};
  node.output_dtype = {"float32"};
  node.input_dims = {rows, width};
  node.gm_stride = CreateExpr(32128) - width;
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, perf), af::SUCCESS);
  const auto calibrated = CreateExpr("load_aligned_single_row_node");
  const auto it = perf.ternary_ops.find(calibrated);
  ASSERT_NE(it, perf.ternary_ops.end());
  EXPECT_TRUE(HasLoadFallbackChain(perf.ternary_ops, SelectLoadCoreBranch(perf.ternary_ops, false), calibrated));

  const auto guard = it->second.DeepCopyIfCase();
  ASSERT_NE(guard, nullptr);
  const auto condition = guard->GetCondLeft();
  const auto evaluate = [&](int32_t row_count, int32_t elements, int32_t core_count) {
    return af::SymbolicUtils::StaticCheckEq(
        condition.Replace(
            {{rows, CreateExpr(row_count)}, {width, CreateExpr(elements)}, {cores, CreateExpr(core_count)}}),
        af::sym::kSymbolZero);
  };
  EXPECT_EQ(evaluate(1, 30680, 56), af::TriBool::kTrue);
  EXPECT_EQ(evaluate(1, 512, 24), af::TriBool::kTrue);
  EXPECT_EQ(evaluate(2, 30680, 56), af::TriBool::kFalse);
  EXPECT_EQ(evaluate(1, 30679, 56), af::TriBool::kFalse);
  EXPECT_EQ(evaluate(1, 504, 56), af::TriBool::kFalse);
  EXPECT_EQ(evaluate(1, 30688, 56), af::TriBool::kFalse);
  EXPECT_EQ(evaluate(1, 30680, 23), af::TriBool::kFalse);
  EXPECT_EQ(evaluate(1, 30680, 57), af::TriBool::kFalse);

  double cycles = 0.0;
  ASSERT_TRUE(guard->GetChoiceA()
                  ->GetExpr()
                  .Replace({{width, CreateExpr(30680)}, {cores, CreateExpr(56)}})
                  .GetConstValue(cycles));
  EXPECT_NEAR(cycles, 5498.61, 0.1);

  node.input_dtype = {"float16"};
  node.output_dtype = {"float16"};
  PerfOutputInfo fp16_perf;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, fp16_perf), af::SUCCESS);
  EXPECT_EQ(fp16_perf.ternary_ops.count(calibrated), 0U);

  CheckFlattenedAlignedSingleRowLoad(node, width, cores, calibrated);
}

TEST_F(CalibratedAscirPerfV2, TestAlignedSingleRowLoadAtFewCores) {
  const Expr width = CreateExpr("width");
  const Expr cores = CreateExpr("block_dim");
  NodeDetail node;
  node.name = "Load";
  node.optype = kLoad;
  node.input_dtype = {"float32"};
  node.output_dtype = {"float32"};
  node.input_dims = {width};
  node.gm_stride = af::sym::kSymbolZero;
  node.ub_stride = af::sym::kSymbolZero;
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, perf), af::SUCCESS);
  const auto fitted = CreateExpr("load_aligned_single_row_few_core_node");
  EXPECT_EQ(perf.pipe_res[PipeType::AIV_MTE2], fitted);
  const auto branch = perf.ternary_ops.find(fitted);
  ASSERT_NE(branch, perf.ternary_ops.end());
  const auto guard = branch->second.DeepCopyIfCase();
  ASSERT_NE(guard, nullptr);
  EXPECT_EQ(guard->GetChoiceB()->GetExpr(), CreateExpr("load_core_case_node"));
  ExpectDmaDomain(guard->GetCondLeft(), {width, cores}, {{1024, 1}, {4096, 2}, {8192, 3}}, af::TriBool::kTrue);
  ExpectDmaDomain(guard->GetCondLeft(), {width, cores}, {{4096, 4}, {4097, 1}, {1016, 1}}, af::TriBool::kFalse);
  double cycles = 0.0;
  ASSERT_TRUE(guard->GetChoiceA()->GetExpr().Replace({{width, CreateExpr(4096)}}).GetConstValue(cycles));
  EXPECT_NEAR(cycles, 389.65, 0.1);

  node.gm_stride = CreateExpr(8);
  PerfOutputInfo strided;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, strided), af::SUCCESS);
  const auto strided_branch = strided.ternary_ops.find(fitted);
  ASSERT_NE(strided_branch, strided.ternary_ops.end());
  const auto strided_guard = strided_branch->second.DeepCopyIfCase();
  ASSERT_NE(strided_guard, nullptr);
  ExpectDmaDomain(strided_guard->GetCondLeft(), {width, cores}, {{4096, 1}}, af::TriBool::kFalse);
}

void CheckAlignedLoopLoadDescriptorDomainFallbacks(const TensorShapeInfo &shape, NodeInfo node,
                                                   const Expr &calibrated) {
  auto load = ApiPerfFactory::Instance().Create("LoadV2");
  ASSERT_NE(load, nullptr);
  const auto expect_fallback = [&](const TensorShapeInfo &input, const NodeInfo &info) {
    PerfOutputInfo fallback;
    ASSERT_EQ(load->GetPerfFunc()({input}, {input}, info, fallback), af::SUCCESS);
    EXPECT_EQ(fallback.ternary_ops.count(calibrated), 0U);
  };
  node.is_cv_ub_fusion = true;
  expect_fallback(shape, node);
  node.is_cv_ub_fusion = false;
  auto changed = shape;
  changed.data_type = "float16";
  changed.data_type_size = 2;
  expect_fallback(changed, node);
  changed = shape;
  changed.strides.back() = CreateExpr(2);
  expect_fallback(changed, node);
  changed = shape;
  changed.gm_strides.back() = CreateExpr(2);
  expect_fallback(changed, node);
  changed = shape;
  changed.repeats.push_back(CreateExpr(8));
  changed.strides.push_back(CreateExpr(1));
  changed.gm_strides.push_back(CreateExpr(1));
  expect_fallback(changed, node);
}

void ExpectAlignedLoopLoadDomain(const Expr &condition, const std::vector<Expr> &symbols) {
  ExpectDmaDomain(condition, symbols, {{24, 2, 32, 0, 0, 16}, {56, 6, 64, 1, 0, 56}, {43, 5, 64, 0, 0, 40}},
                  af::TriBool::kTrue);
  ExpectDmaDomain(condition, symbols,
                  {{23, 4, 64, 0, 0, 32},
                   {57, 4, 64, 0, 0, 32},
                   {38, 1, 64, 0, 0, 32},
                   {38, 7, 64, 0, 0, 32},
                   {38, 4, 48, 0, 0, 32},
                   {38, 4, 96, 0, 0, 32},
                   {38, 4, 64, 2, 0, 32},
                   {38, 4, 64, 0, 1, 32},
                   {38, 4, 64, 0, 0, 15},
                   {38, 4, 64, 0, 0, 57}},
                  af::TriBool::kFalse);
}

TEST_F(CalibratedAscirPerfV2, TestAlignedLoopLoadDescriptorDomain) {
  const Expr m = CreateExpr("loop_m");
  const Expr n = CreateExpr("loop_n");
  const Expr width = CreateExpr("loop_width");
  const Expr padding = CreateExpr("loop_padding");
  const Expr gm_offset = CreateExpr("loop_gm_stride_delta");
  const Expr cores = CreateExpr("block_dim");
  TensorShapeInfo shape;
  shape.data_type = "float32";
  shape.data_type_size = 4;
  shape.repeats = {m, n, width};
  shape.gm_strides = {width + gm_offset, m * width, CreateExpr(1)};
  shape.strides = {(n + padding) * width, width, CreateExpr(1)};
  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Load", 1, 1);
  auto load = ApiPerfFactory::Instance().Create("LoadV2");
  ASSERT_NE(load, nullptr);
  PerfOutputInfo perf;
  ASSERT_EQ(load->GetPerfFunc()({shape}, {shape}, node, perf), af::SUCCESS);
  const Expr calibrated = CreateExpr("load_aligned_loop_payload_node");
  const Expr narrow = CreateExpr("load_narrow_loop_payload_node");
  const Expr wide = CreateExpr("load_wide_aligned_loop_payload_node");
  ASSERT_EQ(perf.pipe_res.at(PipeType::AIV_MTE2), wide);
  ASSERT_EQ(perf.ternary_ops.at(wide).DeepCopyIfCase()->GetChoiceB()->GetExpr(), narrow);
  ASSERT_EQ(perf.ternary_ops.at(narrow).DeepCopyIfCase()->GetChoiceB()->GetExpr(), calibrated);
  const auto wide_guard = perf.ternary_ops.at(wide).DeepCopyIfCase();
  ExpectDmaDomain(wide_guard->GetCondLeft(), {m, n, width, padding, gm_offset, cores},
                  {{8, 13, 128, 0, 0, 53}, {8, 16, 96, 0, 0, 32}}, af::TriBool::kTrue);
  ExpectDmaDomain(wide_guard->GetCondLeft(), {m, n, width, padding, gm_offset, cores},
                  {{32, 4, 32, 0, 0, 48}, {8, 13, 128, 1, 0, 53}, {8, 13, 128, 0, 1, 53}}, af::TriBool::kFalse);
  const auto it = perf.ternary_ops.find(calibrated);
  ASSERT_NE(it, perf.ternary_ops.end());
  auto guard = it->second.DeepCopyIfCase();
  ASSERT_NE(guard, nullptr);

  ExpectAlignedLoopLoadDomain(guard->GetCondLeft(), {m, n, width, padding, gm_offset, cores});

  double cycles = 0.0;
  ASSERT_TRUE(guard->GetChoiceA()
                  ->GetExpr()
                  .Replace({{m, CreateExpr(38)}, {n, CreateExpr(4)}, {width, CreateExpr(64)}, {cores, CreateExpr(32)}})
                  .GetConstValue(cycles));
  EXPECT_NEAR(cycles, 294.009388 + 38912.0 / (32.296047 + 252.871196 / 32), 0.01);
  ASSERT_NE(guard->GetChoiceB()->GetExpr(), calibrated);

  // The factory retains the legacy path for unsupported API descriptors.
  CheckAlignedLoopLoadDescriptorDomainFallbacks(shape, node, calibrated);
}

TEST_F(CalibratedAscirPerfV2, TestLongAlignedLoopLoadUsesNormalHardwareLoop) {
  const auto m = CreateExpr("loop_m");
  const auto n = CreateExpr("loop_n");
  const auto pitch = CreateExpr("loop_pitch");
  const auto cores = CreateExpr("block_dim");
  const auto width = CreateExpr(32);
  const auto one = CreateExpr(1);
  TensorShapeInfo shape;
  shape.data_type = "float32";
  shape.data_type_size = sizeof(float);
  shape.repeats = {m, n, width};
  shape.gm_strides = {width, pitch, one};
  shape.strides = {n * width, width, one};
  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Load", 1, 1);
  auto load = ApiPerfFactory::Instance().Create("LoadV2");
  ASSERT_NE(load, nullptr);
  const auto api = std::make_shared<codegen::CodegenApiParam>();
  api->template_params = {"float", "AscendC::PaddingMode::Normal"};
  auto &dma = api->specific_params.emplace<codegen::DmaSpecificParams>();
  dma.data_copy_params.valid = true;
  dma.loop_mode_params.valid = true;
  ASSERT_EQ(codegen::CodegenApiParam::Register(node.node_ptr, api), af::SUCCESS);
  const Expr calibrated = CreateExpr("load_long_aligned_loop_payload_node");
  PerfOutputInfo perf;
  ASSERT_EQ(load->GetPerfFunc()({shape}, {shape}, node, perf), af::SUCCESS);
  const auto it = perf.ternary_ops.find(calibrated);
  ASSERT_NE(it, perf.ternary_ops.end());
  const auto guard = it->second.DeepCopyIfCase();
  ExpectDmaDomain(guard->GetCondLeft(), {m, n, pitch, cores}, {{20, 14, 640, 29}, {22, 14, 704, 25}},
                  af::TriBool::kTrue);
  ExpectDmaDomain(guard->GetCondLeft(), {m, n, pitch, cores}, {{15, 14, 640, 29}, {20, 14, 639, 29}, {20, 14, 640, 41}},
                  af::TriBool::kFalse);
  double cycles = 0;
  ASSERT_TRUE(guard->GetChoiceA()->GetExpr().Replace({{m, CreateExpr(20)}, {n, CreateExpr(14)}}).GetConstValue(cycles));
  EXPECT_NEAR(cycles, 1239.3, 0.1);

  api->template_params.back() = "AscendC::PaddingMode::Compact";
  PerfOutputInfo compact;
  ASSERT_EQ(load->GetPerfFunc()({shape}, {shape}, node, compact), af::SUCCESS);
  EXPECT_EQ(compact.ternary_ops.count(calibrated), 0U);

  api->template_params.back() = "AscendC::PaddingMode::Normal";
  dma.loop_mode_params.valid = false;
  PerfOutputInfo no_loop;
  ASSERT_EQ(load->GetPerfFunc()({shape}, {shape}, node, no_loop), af::SUCCESS);
  EXPECT_EQ(no_loop.ternary_ops.count(calibrated), 0U);
}

TEST_F(CalibratedAscirPerfV2, TestSparsePitchLoadRequiresNormalHardwareLoop) {
  const auto m = CreateExpr("loop_m");
  const auto pitch = CreateExpr("loop_pitch");
  const auto cores = CreateExpr("block_dim");
  const auto one = CreateExpr(1);
  const auto width = CreateExpr(64);
  const auto rows = CreateExpr(12);
  TensorShapeInfo shape;
  shape.data_type = "float32";
  shape.data_type_size = sizeof(float);
  shape.repeats = {m, rows, width};
  shape.gm_strides = {width, pitch, one};
  shape.strides = {rows * width, width, one};
  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Load", 1, 1);
  auto load = ApiPerfFactory::Instance().Create("LoadV2");
  ASSERT_NE(load, nullptr);
  const auto api = std::make_shared<codegen::CodegenApiParam>();
  api->template_params = {"float", "AscendC::PaddingMode::Normal"};
  auto &dma = api->specific_params.emplace<codegen::DmaSpecificParams>();
  dma.data_copy_params.valid = true;
  dma.loop_mode_params.valid = true;
  ASSERT_EQ(codegen::CodegenApiParam::Register(node.node_ptr, api), af::SUCCESS);

  const auto calibrated = CreateExpr("load_sparse_pitch_loop_payload_node");
  PerfOutputInfo perf;
  ASSERT_EQ(load->GetPerfFunc()({shape}, {shape}, node, perf), af::SUCCESS);
  ASSERT_EQ(perf.pipe_res.at(PipeType::AIV_MTE2), calibrated);
  const auto guard = perf.ternary_ops.at(calibrated).DeepCopyIfCase();
  ASSERT_NE(guard, nullptr);
  ExpectDmaDomain(guard->GetCondLeft(), {m, pitch, cores}, {{3, 4096, 17}}, af::TriBool::kTrue);
  ExpectDmaDomain(guard->GetCondLeft(), {m, pitch, cores}, {{1, 4096, 17}, {3, 4097, 17}, {3, 1024, 17}, {3, 4096, 23}},
                  af::TriBool::kFalse);
  double cycles = 0;
  ASSERT_TRUE(
      guard->GetChoiceA()->GetExpr().Replace({{m, CreateExpr(3)}, {cores, CreateExpr(17)}}).GetConstValue(cycles));
  EXPECT_NEAR(cycles, 308.447 + 9216.0 * 17 / 2000.924, 0.01);

  api->template_params.back() = "AscendC::PaddingMode::Compact";
  PerfOutputInfo compact;
  ASSERT_EQ(load->GetPerfFunc()({shape}, {shape}, node, compact), af::SUCCESS);
  EXPECT_EQ(compact.ternary_ops.count(calibrated), 0U);
}

void CheckNarrowLoopLoadDescriptorDomainFallbacks(const TensorShapeInfo &shape, NodeInfo node, const Expr &calibrated) {
  auto load = ApiPerfFactory::Instance().Create("LoadV2");
  ASSERT_NE(load, nullptr);
  const auto expect_fallback = [&](const TensorShapeInfo &input, const NodeInfo &info) {
    PerfOutputInfo fallback;
    ASSERT_EQ(load->GetPerfFunc()({input}, {input}, info, fallback), af::SUCCESS);
    EXPECT_EQ(fallback.ternary_ops.count(calibrated), 0U);
  };
  node.is_cv_ub_fusion = true;
  expect_fallback(shape, node);
  node.is_cv_ub_fusion = false;
  auto changed = shape;
  changed.data_type = "float16";
  changed.data_type_size = 2;
  expect_fallback(changed, node);
  changed = shape;
  changed.gm_strides[0] = CreateExpr(17);
  expect_fallback(changed, node);
  changed = shape;
  changed.strides.back() = CreateExpr(2);
  expect_fallback(changed, node);
}

TEST_F(CalibratedAscirPerfV2, TestNarrowLoopLoadDescriptorDomain) {
  const Expr m = CreateExpr("narrow_m");
  const Expr n = CreateExpr("narrow_n");
  const Expr padding = CreateExpr("narrow_padding");
  const Expr cores = CreateExpr("block_dim");
  TensorShapeInfo shape;
  shape.data_type = "float32";
  shape.data_type_size = 4;
  shape.repeats = {m, n, CreateExpr(16)};
  shape.gm_strides = {CreateExpr(16), m * CreateExpr(16), CreateExpr(1)};
  shape.strides = {(n + padding) * CreateExpr(16), CreateExpr(16), CreateExpr(1)};
  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Load", 1, 1);
  auto load = ApiPerfFactory::Instance().Create("LoadV2");
  ASSERT_NE(load, nullptr);
  PerfOutputInfo perf;
  ASSERT_EQ(load->GetPerfFunc()({shape}, {shape}, node, perf), af::SUCCESS);
  const Expr calibrated = CreateExpr("load_narrow_loop_payload_node");
  const auto it = perf.ternary_ops.find(calibrated);
  ASSERT_NE(it, perf.ternary_ops.end());
  EXPECT_EQ(perf.pipe_res.at(PipeType::AIV_MTE2), calibrated);
  const auto guard = it->second.DeepCopyIfCase();
  ASSERT_NE(guard, nullptr);

  ExpectDmaDomain(guard->GetCondLeft(), {m, n, padding, cores},
                  {{24, 2, 0, 16}, {56, 8, 1, 56}, {38, 6, 0, 22}, {50, 4, 0, 32}}, af::TriBool::kTrue);
  ExpectDmaDomain(
      guard->GetCondLeft(), {m, n, padding, cores},
      {{23, 4, 0, 32}, {57, 4, 0, 32}, {38, 1, 0, 32}, {38, 9, 0, 32}, {38, 2, 4, 22}, {38, 4, 0, 15}, {38, 4, 0, 57}},
      af::TriBool::kFalse);

  double cycles = 0.0;
  ASSERT_TRUE(guard->GetChoiceA()
                  ->GetExpr()
                  .Replace({{m, CreateExpr(38)}, {n, CreateExpr(6)}, {cores, CreateExpr(22)}})
                  .GetConstValue(cycles));
  EXPECT_NEAR(cycles, 1303.6138, 0.01);
  CheckNarrowLoopLoadDescriptorDomainFallbacks(shape, node, calibrated);
}

TEST_F(CalibratedAscirPerfV2, TestAlignedDenseBulkLoadKeepsPhysicalDescriptorDomain) {
  const Expr rows = CreateExpr("bulk_rows");
  const Expr width = CreateExpr("bulk_width");
  const Expr cores = CreateExpr("block_dim");
  const Expr gm_gap = CreateExpr("bulk_gm_gap");
  const Expr ub_gap = CreateExpr("bulk_ub_gap");
  NodeDetail node;
  node.name = "Load";
  node.optype = kLoad;
  node.input_dtype = {"float32"};
  node.output_dtype = {"float32"};
  node.input_dims = {rows, width};
  node.gm_stride = gm_gap;
  node.ub_stride = ub_gap;
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, perf), af::SUCCESS);
  const Expr calibrated = CreateExpr("LoadAlignedDenseBulkPayload_small_dma_node");
  const auto it = perf.ternary_ops.find(calibrated);
  ASSERT_NE(it, perf.ternary_ops.end());
  const auto guard = it->second.DeepCopyIfCase();

  ExpectDmaDomain(guard->GetCondLeft(), {rows, width, cores, gm_gap, ub_gap},
                  {{512, 8, 16, 0, 0}, {2048, 8, 56, 0, 0}, {1024, 16, 48, 0, 0}, {512, 32, 40, 0, 0}},
                  af::TriBool::kTrue);
  ExpectDmaDomain(guard->GetCondLeft(), {rows, width, cores, gm_gap, ub_gap},
                  {{511, 8, 32, 0, 0},
                   {2049, 8, 32, 0, 0},
                   {1024, 7, 32, 0, 0},
                   {512, 40, 32, 0, 0},
                   {2048, 16, 32, 0, 0},
                   {1024, 8, 15, 0, 0},
                   {1024, 8, 57, 0, 0},
                   {1024, 8, 32, 8, 0},
                   {1024, 8, 32, 0, 8}},
                  af::TriBool::kFalse);

  const auto expect_unchanged = [&](const NodeDetail &input) {
    PerfOutputInfo fallback;
    ASSERT_EQ(ascendcapi_v2::LoadPerf(input, fallback), af::SUCCESS);
    EXPECT_EQ(fallback.ternary_ops.count(calibrated), 0U);
  };
  auto control = node;
  control.input_dtype = {"float16"};
  control.output_dtype = {"float16"};
  expect_unchanged(control);
  control = node;
  control.input_dims = {af::sym::kSymbolOne, width};
  expect_unchanged(control);
  control = node;
  control.vectorized_shape = {rows, af::sym::kSymbolOne, width};
  expect_unchanged(control);
}

TEST_F(CalibratedAscirPerfV2, TestAlignedDenseWideLoadKeepsPhysicalDescriptorDomain) {
  const Expr rows = CreateExpr("wide_rows");
  const Expr width = CreateExpr("wide_width");
  const Expr cores = CreateExpr("block_dim");
  const Expr gm_gap = CreateExpr("wide_gm_gap");
  const Expr ub_gap = CreateExpr("wide_ub_gap");
  NodeDetail node;
  node.name = "Load";
  node.optype = kLoad;
  node.input_dtype = {"float32"};
  node.output_dtype = {"float32"};
  node.input_dims = {rows, width};
  node.gm_stride = gm_gap;
  node.ub_stride = ub_gap;
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, perf), af::SUCCESS);
  const Expr calibrated = CreateExpr("LoadAlignedDenseWidePayload_small_dma_node");
  const auto it = perf.ternary_ops.find(calibrated);
  ASSERT_NE(it, perf.ternary_ops.end());
  const auto guard = it->second.DeepCopyIfCase();
  ExpectDmaDomain(guard->GetCondLeft(), {rows, width, cores, gm_gap, ub_gap},
                  {{5, 4096, 6, 0, 0}, {24, 1280, 56, 0, 0}, {12, 1024, 56, 0, 0}}, af::TriBool::kTrue);
  ExpectDmaDomain(guard->GetCondLeft(), {rows, width, cores, gm_gap, ub_gap},
                  {{1, 4096, 24, 0, 0},
                   {12, 1025, 24, 0, 0},
                   {10, 1280, 5, 0, 0},
                   {12, 1024, 24, 8, 0},
                   {12, 1024, 24, 0, 8},
                   {24, 1536, 24, 0, 0}},
                  af::TriBool::kFalse);

  node.input_dtype = {"float16"};
  node.output_dtype = {"float16"};
  PerfOutputInfo fallback;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, fallback), af::SUCCESS);
  EXPECT_EQ(fallback.ternary_ops.count(calibrated), 0U);
}

void CheckDenseLoadDescriptorEquivalence(NodeDetail node, const Expr &calibrated, double packed) {
  const Expr rows = CreateExpr("dense_rows");
  const Expr width = CreateExpr("dense_width");
  const Expr gm_gap = CreateExpr("dense_gm_gap");
  const Expr ub_gap = CreateExpr("dense_ub_gap");
  const Expr cores = CreateExpr("block_dim");
  node.input_dims = {af::sym::kSymbolOne, width};
  PerfOutputInfo single_row;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, single_row), af::SUCCESS);
  const auto single_it = single_row.ternary_ops.find(calibrated);
  ASSERT_NE(single_it, single_row.ternary_ops.end());
  const auto single_guard = single_it->second.DeepCopyIfCase();
  ASSERT_NE(single_guard, nullptr);
  const auto accepts_single = [&](int32_t w, int32_t c) {
    return af::SymbolicUtils::StaticCheckEq(single_guard->GetCondLeft().Replace({{width, CreateExpr(w)},
                                                                                 {gm_gap, af::sym::kSymbolZero},
                                                                                 {ub_gap, af::sym::kSymbolZero},
                                                                                 {cores, CreateExpr(c)}}),
                                            af::sym::kSymbolZero);
  };
  for (const auto w : {256, 264, 272, 280, 512, 992, 1016, 1024}) {
    EXPECT_EQ(accepts_single(w, 16), af::TriBool::kTrue);
    EXPECT_EQ(accepts_single(w, 56), af::TriBool::kTrue);
  }
  for (const auto w : {248, 257, 1032, 1040, 1056, 5344, 5360, 6144, 6152}) {
    EXPECT_EQ(accepts_single(w, 32), af::TriBool::kFalse);
  }
  EXPECT_EQ(accepts_single(512, 15), af::TriBool::kFalse);
  EXPECT_EQ(accepts_single(512, 57), af::TriBool::kFalse);
  double single_estimate = 0.0;
  ASSERT_TRUE(single_guard->GetChoiceA()
                  ->GetExpr()
                  .Replace({{width, CreateExpr(1024)}, {cores, CreateExpr(48)}})
                  .GetConstValue(single_estimate));
  EXPECT_DOUBLE_EQ(single_estimate, packed);
  node.input_dims = {rows, width};
  node.vectorized_shape = {CreateExpr(2), CreateExpr(16), CreateExpr(64)};
  PerfOutputInfo loop;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, loop), af::SUCCESS);
  EXPECT_EQ(loop.ternary_ops.count(calibrated), 0U);
  node.vectorized_shape.clear();
  node.input_dtype = {"float16"};
  node.output_dtype = {"float16"};
  PerfOutputInfo fp16;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, fp16), af::SUCCESS);
  EXPECT_EQ(fp16.ternary_ops.count(calibrated), 0U);
}

TEST_F(CalibratedAscirPerfV2, TestAlignedDenseLoadDomainAndDescriptorEquivalence) {
  const Expr rows = CreateExpr("dense_rows");
  const Expr width = CreateExpr("dense_width");
  const Expr gm_gap = CreateExpr("dense_gm_gap");
  const Expr ub_gap = CreateExpr("dense_ub_gap");
  const Expr cores = CreateExpr("block_dim");
  NodeDetail node;
  node.name = "Load";
  node.optype = kLoad;
  node.input_dtype = {"float32"};
  node.output_dtype = {"float32"};
  node.input_dims = {rows, width};
  node.gm_stride = gm_gap;
  node.ub_stride = ub_gap;
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, perf), af::SUCCESS);
  const Expr calibrated = CreateExpr("LoadAlignedDensePayload_small_dma_node");
  const auto it = perf.ternary_ops.find(calibrated);
  ASSERT_NE(it, perf.ternary_ops.end());
  EXPECT_EQ(SelectLoadCoreBranch(perf.ternary_ops, false), calibrated);
  const auto guard = it->second.DeepCopyIfCase();
  ASSERT_NE(guard, nullptr);

  ExpectDmaDomain(guard->GetCondLeft(), {rows, width, gm_gap, ub_gap, cores},
                  {{16, 64, 0, 0, 16}, {4, 64, 0, 0, 48}, {1, 1024, 0, 0, 48}}, af::TriBool::kTrue);
  ExpectDmaDomain(guard->GetCondLeft(), {rows, width, gm_gap, ub_gap, cores},
                  {{17, 64, 0, 0, 56},
                   {16, 57, 0, 0, 32},
                   {3, 64, 0, 0, 32},
                   {1, 1032, 0, 0, 32},
                   {16, 64, 1, 0, 32},
                   {16, 64, 0, 8, 32},
                   {16, 64, 0, 0, 15},
                   {16, 64, 0, 0, 57}},
                  af::TriBool::kFalse);

  const auto estimate = guard->GetChoiceA()->GetExpr();
  double packed = 0.0;
  double flat = 0.0;
  ASSERT_TRUE(estimate.Replace({{rows, CreateExpr(16)}, {width, CreateExpr(64)}, {cores, CreateExpr(48)}})
                  .GetConstValue(packed));
  ASSERT_TRUE(estimate.Replace({{rows, CreateExpr(1)}, {width, CreateExpr(1024)}, {cores, CreateExpr(48)}})
                  .GetConstValue(flat));
  EXPECT_DOUBLE_EQ(packed, flat);
  EXPECT_NEAR(packed, 457.52, 0.1);
  CheckDenseLoadDescriptorEquivalence(node, calibrated, packed);
}

TEST_F(CalibratedAscirPerfV2, TestWideFp16CompactUsesOnlyPhysicalPayload) {
  const Expr rows = CreateExpr("wide_fp16_rows");
  const Expr width = CreateExpr("wide_fp16_width");
  const Expr pitch = CreateExpr("wide_fp16_pitch");
  const Expr gap = CreateExpr("wide_fp16_gap");
  const Expr cores = CreateExpr("block_dim");
  NodeDetail node;
  node.name = "Load";
  node.optype = kLoad;
  node.input_dtype = {"float16"};
  node.output_dtype = {"float16"};
  node.input_dims = {rows, width};
  node.repeats = {CreateExpr("outer_repeat"), rows, width};
  node.block_count_idx = 1;
  node.gm_stride = pitch - width;
  node.ub_stride = gap;
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, perf), af::SUCCESS);
  const auto fit = perf.ternary_ops.find(CreateExpr("LoadWideFp16Compact_dma_node"));
  ASSERT_NE(fit, perf.ternary_ops.end());
  const auto terms = fit->second.DeepCopyIfCase();
  ASSERT_NE(terms, nullptr);
  ASSERT_NE(terms->GetChoiceA(), nullptr);
  std::vector<std::pair<Expr, Expr>> values{{rows, CreateExpr(101)},
                                            {width, CreateExpr(152)},
                                            {pitch, CreateExpr(8658)},
                                            {gap, CreateExpr(0)},
                                            {cores, CreateExpr(54)}};
  const auto check = [&] {
    return af::SymbolicUtils::StaticCheckEq(terms->GetCondLeft().Replace(values), af::sym::kSymbolZero);
  };
  EXPECT_EQ(check(), af::TriBool::kTrue);
  double estimated = 0;
  ASSERT_TRUE(terms->GetChoiceA()->GetExpr().Replace(values).GetConstValue(estimated));
  EXPECT_NEAR(estimated, 1608.529, 0.01);
  values[3].second = CreateExpr(8);
  EXPECT_EQ(check(), af::TriBool::kFalse);
  values[3].second = CreateExpr(0);
  values[2].second = CreateExpr(9999);
  EXPECT_EQ(check(), af::TriBool::kFalse);

  node.repeats = node.input_dims;
  PerfOutputInfo rank2;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, rank2), af::SUCCESS);
  EXPECT_EQ(rank2.ternary_ops.count(CreateExpr("LoadWideFp16Compact_dma_node")), 0U);
}

TEST_F(CalibratedAscirPerfV2, TestManyShortFp32CompactUsesPhysicalPitchAfterMerge) {
  const Expr outer = CreateExpr("compact_fp32_outer");
  const Expr inner = CreateExpr("compact_fp32_inner");
  const Expr cores = CreateExpr("block_dim");
  TensorShapeInfo shape;
  shape.data_type = "float32";
  shape.data_type_size = 4;
  shape.repeats = {outer, CreateExpr(48), inner, CreateExpr(3)};
  shape.origin_repeats = shape.repeats;
  shape.gm_strides = {CreateExpr(6912), CreateExpr(144), CreateExpr(3), CreateExpr(1)};
  shape.strides = {CreateExpr(144) * inner, CreateExpr(3) * inner, CreateExpr(3), CreateExpr(1)};
  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Load", 1, 1);
  const auto load = ApiPerfFactory::Instance().Create("LoadV2");
  ASSERT_NE(load, nullptr);
  const auto evaluate = load->GetPerfFunc();
  PerfOutputInfo perf;
  ASSERT_EQ(evaluate({shape}, {shape}, node, perf), af::SUCCESS);
  const auto key = CreateExpr("LoadManyShortFp32Compact_dma_node");
  const auto fit = perf.ternary_ops.find(key);
  ASSERT_NE(fit, perf.ternary_ops.end());
  const auto terms = fit->second.DeepCopyIfCase();
  ASSERT_NE(terms, nullptr);
  ASSERT_NE(terms->GetChoiceA(), nullptr);
  const std::vector<std::pair<Expr, Expr>> values{
      {outer, CreateExpr(64)}, {inner, CreateExpr(1)}, {cores, CreateExpr(48)}};
  EXPECT_EQ(af::SymbolicUtils::StaticCheckEq(terms->GetCondLeft().Replace(values), af::sym::kSymbolZero),
            af::TriBool::kTrue);
  double estimate = 0.0;
  ASSERT_TRUE(terms->GetChoiceA()->GetExpr().Replace(values).GetConstValue(estimate));
  EXPECT_NEAR(estimate, 18327.0, 1.0);

  shape.gm_strides[0] = CreateExpr(6916);
  PerfOutputInfo different;
  ASSERT_EQ(evaluate({shape}, {shape}, node, different), af::SUCCESS);
  EXPECT_EQ(different.ternary_ops.count(key), 0U);
}

TEST_F(CalibratedAscirPerfV2, TestFp32CompactHardwareLoopLoadUsesPhysicalRows) {
  TensorShapeInfo shape;
  shape.data_type = "float32";
  shape.data_type_size = sizeof(float);
  shape.repeats = {CreateExpr(33), CreateExpr(6), CreateExpr(39)};
  shape.origin_repeats = shape.repeats;
  shape.gm_strides = {CreateExpr(874458), CreateExpr(8658), CreateExpr(1)};
  shape.strides = {CreateExpr(240), CreateExpr(39), CreateExpr(1)};
  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Load", 1, 1);
  const auto load = ApiPerfFactory::Instance().Create("LoadV2");
  ASSERT_NE(load, nullptr);
  const auto evaluate = load->GetPerfFunc();
  PerfOutputInfo perf;
  ASSERT_EQ(evaluate({shape}, {shape}, node, perf), af::SUCCESS);
  const auto key = CreateExpr("load_fp32_compact_hw_loop");
  const auto fit = perf.ternary_ops.find(key);
  ASSERT_NE(fit, perf.ternary_ops.end());
  const auto terms = fit->second.DeepCopyIfCase();
  ASSERT_NE(terms, nullptr);
  ASSERT_NE(terms->GetChoiceA(), nullptr);
  double cycles = 0.0;
  ASSERT_TRUE(
      terms->GetChoiceA()->GetExpr().Replace({{CreateExpr("block_dim"), CreateExpr(56)}}).GetConstValue(cycles));
  EXPECT_NEAR(cycles, 3923.0, 2.0);

  shape.gm_strides[0] = CreateExpr(874459);
  PerfOutputInfo different;
  ASSERT_EQ(evaluate({shape}, {shape}, node, different), af::SUCCESS);
  EXPECT_EQ(different.ternary_ops.count(key), 0U);
}

codegen::CombinedExpression ConstantDmaExpr(int32_t value) {
  return codegen::CombinedExpression(
      codegen::ExpressionItem(CreateExpr(value), codegen::ExpressionConvertMethod::kActualSize));
}

codegen::CodegenApiParamPtr MakeCompactLoadApi(int32_t rows, int32_t width, int32_t src_stride) {
  const auto api = std::make_shared<codegen::CodegenApiParam>();
  api->template_params = {"float", "AscendC::PaddingMode::Compact"};
  auto &copy = api->specific_params.emplace<codegen::DmaSpecificParams>().data_copy_params;
  copy.valid = true;
  copy.block_count = ConstantDmaExpr(rows);
  copy.block_len = ConstantDmaExpr(width);
  copy.src_stride = ConstantDmaExpr(src_stride);
  copy.dst_stride = ConstantDmaExpr(0);
  return api;
}

void CheckCompactScalarLoad(TensorShapeInfo shape, const NodeInfo &node, codegen::DmaSpecificParams &dma,
                            const ApiPerf &load) {
  shape.repeats = {CreateExpr(64)};
  shape.origin_repeats = shape.repeats;
  shape.gm_strides = {CreateExpr(39)};
  shape.strides = {CreateExpr(1)};
  dma.data_copy_params.block_count = ConstantDmaExpr(64);
  dma.data_copy_params.block_len = ConstantDmaExpr(1);
  dma.data_copy_params.src_stride = ConstantDmaExpr(38);
  const auto scalar_key = CreateExpr("load_compact_scalar_row_node");
  const Expr cores = CreateExpr("block_dim");
  PerfOutputInfo scalar;
  ASSERT_EQ(load.GetPerfFunc()({shape}, {shape}, node, scalar), af::SUCCESS);
  ASSERT_EQ(scalar.ternary_ops.count(scalar_key), 1U);
  EXPECT_NEAR(EvaluatePipeCycles(scalar, PipeType::AIV_MTE2, {{cores, CreateExpr(2)}}), 420.67978, 0.01);
  EXPECT_NE(af::SymbolicUtils::StaticCheckEq(
                scalar.ternary_ops.at(scalar_key).DeepCopyIfCase()->GetCondLeft().Replace({{cores, CreateExpr(7)}}),
                af::sym::kSymbolZero),
            af::TriBool::kTrue);

  dma.data_copy_params.src_stride = ConstantDmaExpr(64);
  PerfOutputInfo unsupported_pitch;
  ASSERT_EQ(load.GetPerfFunc()({shape}, {shape}, node, unsupported_pitch), af::SUCCESS);
  EXPECT_EQ(unsupported_pitch.ternary_ops.count(scalar_key), 0U);
}

TEST_F(CalibratedAscirPerfV2, CompactWidePitchLoadFollowsTheGeneratedDescriptor) {
  TensorShapeInfo shape;
  shape.data_type = "float32";
  shape.data_type_size = sizeof(float);
  shape.repeats = {CreateExpr(96), CreateExpr(32)};
  shape.origin_repeats = shape.repeats;
  shape.gm_strides = {CreateExpr(4608), CreateExpr(1)};
  shape.strides = {CreateExpr(32), CreateExpr(1)};
  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Load", 1, 1);
  ASSERT_NE(node.node_ptr, nullptr);
  const auto load = ApiPerfFactory::Instance().Create("LoadV2");
  ASSERT_NE(load, nullptr);
  const auto key = CreateExpr("load_compact_wide_pitch_node");
  const auto api = MakeCompactLoadApi(96, 32, 4576);
  auto &dma = std::get<codegen::DmaSpecificParams>(api->specific_params);
  ASSERT_EQ(codegen::CodegenApiParam::Register(node.node_ptr, api), af::SUCCESS);
  const auto evaluate = [&](double expected = 0.0) {
    PerfOutputInfo perf;
    ASSERT_EQ(load->GetPerfFunc()({shape}, {shape}, node, perf), af::SUCCESS);
    const bool calibrated = expected > 0.0;
    EXPECT_EQ(perf.ternary_ops.count(key), calibrated ? 1U : 0U);
    if (calibrated) {
      EXPECT_NEAR(EvaluatePipeCycles(perf, PipeType::AIV_MTE2, {{CreateExpr("block_dim"), CreateExpr(48)}}), expected,
                  0.01);
    }
  };
  evaluate(770.51738);
  shape.repeats[0] = CreateExpr(64);
  evaluate(644.58189);
  shape.origin_repeats[0] = CreateExpr(80);
  evaluate();
  shape.repeats[0] = CreateExpr(96);
  shape.origin_repeats = shape.repeats;
  dma.loop_mode_params.valid = true;
  evaluate();
  dma.loop_mode_params.valid = false;
  api->template_params.back() = "AscendC::PaddingMode::Normal";
  evaluate();
  api->template_params.back() = "AscendC::PaddingMode::Compact";
  dma.data_copy_params.src_stride = ConstantDmaExpr(992);
  evaluate();

  CheckCompactScalarLoad(shape, node, dma, *load);
}

TEST_F(CalibratedAscirPerfV2, MergedCompactLoadChargesThePhysicalRowStride) {
  TensorShapeInfo shape;
  shape.data_type = "float32";
  shape.data_type_size = sizeof(float);
  shape.repeats = {CreateExpr(16), CreateExpr(4), CreateExpr(128)};
  shape.origin_repeats = shape.repeats;
  shape.gm_strides = {CreateExpr(2097152), CreateExpr(128), CreateExpr(1)};
  shape.strides = {CreateExpr(512), CreateExpr(128), CreateExpr(1)};
  NodeInfo node;
  node.node_ptr = GraphConstructUtils::ConstructSingleOp("Load", 1, 1);
  ASSERT_NE(node.node_ptr, nullptr);
  const auto load = ApiPerfFactory::Instance().Create("LoadV2");
  ASSERT_NE(load, nullptr);
  const auto evaluate = [&]() {
    PerfOutputInfo perf;
    EXPECT_EQ(load->GetPerfFunc()({shape}, {shape}, node, perf), af::SUCCESS);
    return EvaluatePipeCycles(perf, PipeType::AIV_MTE2, {{CreateExpr("block_dim"), CreateExpr(56)}});
  };
  const double legacy = evaluate();
  const auto api = MakeCompactLoadApi(16, 512, 2096640);
  auto &dma = std::get<codegen::DmaSpecificParams>(api->specific_params);
  ASSERT_EQ(codegen::CodegenApiParam::Register(node.node_ptr, api), af::SUCCESS);
  EXPECT_NEAR(evaluate() - legacy, 0.005 * 16 * 4096, 1.0e-4);
  dma.loop_mode_params.valid = true;
  EXPECT_DOUBLE_EQ(evaluate(), legacy);
  dma.loop_mode_params.valid = false;
  dma.data_copy_params.block_len = ConstantDmaExpr(128);
  EXPECT_DOUBLE_EQ(evaluate(), legacy);

  dma.data_copy_params.block_len = ConstantDmaExpr(512);
  dma.data_copy_params.src_stride = codegen::CombinedExpression(
      codegen::ExpressionItem(CreateExpr("runtime_gap"), codegen::ExpressionConvertMethod::kActualSize));
  EXPECT_DOUBLE_EQ(evaluate(), legacy);

  // The physical descriptor calibration only covers FP32 pitches in the measured range.
  shape.gm_strides.front() = CreateExpr(65536);
  dma.data_copy_params.src_stride = ConstantDmaExpr(65024);
  const double medium_pitch = evaluate();
  dma.data_copy_params.valid = false;
  EXPECT_DOUBLE_EQ(evaluate(), medium_pitch);
  dma.data_copy_params.valid = true;
  shape.gm_strides.front() = CreateExpr(2097152);
  dma.data_copy_params.src_stride = ConstantDmaExpr(2096640);
  shape.data_type = "float16";
  shape.data_type_size = sizeof(uint16_t);
  api->template_params.front() = "half";
  const double half_precision = evaluate();
  dma.data_copy_params.valid = false;
  EXPECT_DOUBLE_EQ(evaluate(), half_precision);
}

codegen::CodegenApiParamPtr MakeCompactCopyApi(const std::string &dtype, int32_t rows, int32_t width) {
  auto api = std::make_shared<codegen::CodegenApiParam>();
  api->template_params = {dtype, "AscendC::PaddingMode::Compact"};
  auto &copy = api->specific_params.emplace<codegen::DmaSpecificParams>().data_copy_params;
  copy.valid = true;
  copy.block_count = ConstantDmaExpr(rows);
  copy.block_len = ConstantDmaExpr(width);
  return api;
}

void MakeAdjacentSliceStores(af::AscGraph &graph, int32_t rows, int32_t width, std::vector<af::AscNodePtr> &stores) {
  constexpr int32_t kSliceCount = 4;
  af::ascir_op::Data data("input", graph);
  for (int32_t i = 0; i < kSliceCount; ++i) {
    af::ascir_op::Store store(("slice_" + std::to_string(i)).c_str());
    const auto node = graph.AddNode(store);
    ASSERT_NE(node, nullptr);
    store.x = data.y;
    store.ir_attr.SetOffset(CreateExpr(i * width));
    node->attr.sched.exec_condition = af::ExecuteCondition::kNoCache;

    const auto api = MakeCompactCopyApi("float", rows, width);
    auto &copy = std::get<codegen::DmaSpecificParams>(api->specific_params).data_copy_params;
    copy.src_stride = ConstantDmaExpr(0);
    copy.dst_stride = ConstantDmaExpr((kSliceCount - 1) * width);
    api->output_params.resize(1U);
    api->output_params[0].name = "joined";
    ASSERT_EQ(codegen::CodegenApiParam::Register(node, api), af::SUCCESS);
    stores.push_back(node);
  }
}

TEST_F(CalibratedAscirPerfV2, AdjacentSliceStoresChargeMeasuredWriteback) {
  constexpr int32_t kRows = 32;
  constexpr int32_t kWidth = 96;
  constexpr int32_t kJoinedWidth = 4 * kWidth;
  af::AscGraph graph("adjacent_slice_store");
  std::vector<af::AscNodePtr> stores;
  ASSERT_NO_FATAL_FAILURE(MakeAdjacentSliceStores(graph, kRows, kWidth, stores));

  TensorShapeInfo shape;
  shape.data_type = "float32";
  shape.data_type_size = sizeof(float);
  shape.dims = {CreateExpr(kRows), CreateExpr(kWidth)};
  shape.repeats = shape.origin_repeats = shape.dims;
  shape.strides = {CreateExpr(kWidth), CreateExpr(1)};
  shape.gm_strides = {CreateExpr(kJoinedWidth), CreateExpr(1)};
  NodeInfo node;
  node.node_ptr = stores.front();
  const auto store = ApiPerfFactory::Instance().Create("StoreV2");
  ASSERT_NE(store, nullptr);
  const auto key = CreateExpr("store_adjacent_slices_payload");
  const auto evaluate = [&](PerfOutputInfo &perf) { return store->GetPerfFunc()({shape}, {shape}, node, perf); };
  PerfOutputInfo adjacent;
  ASSERT_EQ(evaluate(adjacent), af::SUCCESS);
  ASSERT_EQ(adjacent.pipe_res.at(PipeType::AIV_MTE3), key);
  const auto cores = CreateExpr("block_dim");
  const double calibrated = 41.3872239625 + (kRows * kWidth * sizeof(float)) / 59.7052349;
  EXPECT_NEAR(EvaluatePipeCycles(adjacent, PipeType::AIV_MTE3, {{cores, CreateExpr(16)}}), calibrated, 0.01);

  auto *attr = stores.back()->attr.ir_attr->DownCastTo<af::ascir_op::Store::AscStoreIrAttrDef>();
  ASSERT_NE(attr, nullptr);
  attr->SetOffset(CreateExpr(3 * kWidth + 1));
  PerfOutputInfo gapped;
  ASSERT_EQ(evaluate(gapped), af::SUCCESS);
  EXPECT_EQ(gapped.ternary_ops.count(key), 0U);
  const std::vector<std::pair<Expr, Expr>> outside = {{cores, CreateExpr(33)}};
  EXPECT_DOUBLE_EQ(EvaluatePipeCycles(adjacent, PipeType::AIV_MTE3, outside),
                   EvaluatePipeCycles(gapped, PipeType::AIV_MTE3, outside));
}

TEST_F(CalibratedAscirPerfV2, PackedShortRowLoadRequiresCompactNarrowWriteback) {
  af::AscGraph graph("packed_short_row");
  ASSERT_EQ(GraphConstructUtils::CreateSimpleLoadStoreOp(graph), af::SUCCESS);
  NodeInfo node;
  node.node_ptr = graph.FindNode("load");
  const auto store = graph.FindNode("store");
  ASSERT_NE(node.node_ptr, nullptr);
  ASSERT_NE(store, nullptr);
  auto load_api = MakeCompactCopyApi("half", 118, 36);
  auto store_api = MakeCompactCopyApi("uint8_t", 36, 118);
  ASSERT_EQ(codegen::CodegenApiParam::Register(node.node_ptr, load_api), af::SUCCESS);
  ASSERT_EQ(codegen::CodegenApiParam::Register(store, store_api), af::SUCCESS);
  TensorShapeInfo shape;
  shape.data_type = "float16";
  shape.data_type_size = sizeof(uint16_t);
  shape.repeats = {CreateExpr(118), CreateExpr(36)};
  shape.origin_repeats = shape.repeats;
  shape.gm_strides = {CreateExpr(13536), CreateExpr(1)};
  shape.strides = {CreateExpr(36), CreateExpr(1)};
  const auto load = ApiPerfFactory::Instance().Create("LoadV2");
  ASSERT_NE(load, nullptr);
  const auto key = CreateExpr("load_packed_short_row");
  const auto evaluate = [&] {
    PerfOutputInfo result;
    EXPECT_EQ(load->GetPerfFunc()({shape}, {shape}, node, result), af::SUCCESS);
    return result.ternary_ops.count(key);
  };
  EXPECT_EQ(evaluate(), 1U);
  store_api->template_params[0] = "half";
  EXPECT_EQ(evaluate(), 0U);
  store_api->template_params[0] = "uint8_t";
  std::get<codegen::DmaSpecificParams>(load_api->specific_params).loop_mode_params.valid = true;
  EXPECT_EQ(evaluate(), 0U);
  std::get<codegen::DmaSpecificParams>(load_api->specific_params).loop_mode_params.valid = false;
  std::get<codegen::DmaSpecificParams>(load_api->specific_params).data_copy_params.block_count =
      codegen::CombinedExpression(
          codegen::ExpressionItem(CreateExpr(236), codegen::ExpressionConvertMethod::kActualSize));
  EXPECT_EQ(evaluate(), 0U);
}

void BuildSparseNddmaReadGroup(af::AscGraph &graph, const std::vector<Expr> &original, const std::vector<Expr> &gm,
                               const std::vector<Expr> &ub, std::vector<af::AscNodePtr> &reads,
                               std::vector<af::AscNodePtr> &sources) {
  const auto &row_axis = graph.CreateAxis("rows", original[0]);
  const auto &lane_axis = graph.CreateAxis("lanes", original[1]);
  const std::vector<int64_t> axes = {row_axis.id, lane_axis.id};
  for (size_t i = 0; i < 3U; ++i) {
    const auto source_name = "source" + std::to_string(i);
    const auto read_name = "read" + std::to_string(i);
    af::ascir_op::Data source(source_name.c_str(), graph);
    af::ascir_op::Nddma read(read_name.c_str());
    reads.push_back(graph.AddNode(read));
    sources.push_back(graph.FindNode(source_name.c_str()));
    ASSERT_NE(reads.back(), nullptr);
    ASSERT_NE(sources.back(), nullptr);
    read.x = source.y;
    auto &in = sources.back()->outputs[0].attr;
    in.repeats = original;
    auto &out = reads.back()->outputs[0].attr;
    out.axis = axes;
    out.repeats = original;
    out.strides = gm;
    out.vectorized_axis = axes;
    out.vectorized_strides = ub;
    out.dtype = i == 2U ? ge::DT_UINT8 : ge::DT_FLOAT;
    out.mem.alloc_type = af::AllocType::kAllocTypeQueue;
    out.que.depth = 2;
    out.que.buf_num = 2;
  }
}

TEST_F(CalibratedAscirPerfV2, SparseNddmaReadGroupRequiresMatchingStaticReads) {
  af::AscGraph graph("sparse_nddma_reads");
  const std::vector<Expr> original = {CreateExpr(4096), CreateExpr(32)};
  const std::vector<Expr> gm = {CreateExpr(1), CreateExpr(1605632)};
  const std::vector<Expr> ub = {CreateExpr(32), CreateExpr(1)};
  std::vector<af::AscNodePtr> reads;
  std::vector<af::AscNodePtr> sources;
  ASSERT_NO_FATAL_FAILURE(BuildSparseNddmaReadGroup(graph, original, gm, ub, reads, sources));

  const auto api = ApiPerfFactory::Instance().Create("NddmaV2");
  ASSERT_NE(api, nullptr);
  TensorShapeInfo shape;
  shape.data_type = "float32";
  shape.data_type_size = sizeof(float);
  shape.origin_repeats = original;
  shape.gm_strides = gm;
  shape.strides = ub;
  const auto width = CreateExpr("read_lanes");
  shape.dims = shape.repeats = {CreateExpr(256), width};
  const auto bound = CreateExpr("nddma_wide_read_group_bound");
  const auto evaluate = [&] {
    NodeInfo node;
    node.node_ptr = reads.front();
    PerfOutputInfo perf;
    EXPECT_EQ(api->GetPerfFunc()({shape}, {shape}, node, perf), af::SUCCESS);
    return perf;
  };
  const auto scoped = evaluate();
  ASSERT_EQ(scoped.ternary_ops.count(bound), 1U);
  const auto cores = CreateExpr("block_dim");
  EXPECT_NEAR(EvaluatePipeCycles(scoped, PipeType::AIV_MTE2, {{cores, CreateExpr(56)}, {width, CreateExpr(32)}}),
              59215.03125 * 4 / 9, 0.1);

  sources.back()->outputs[0].attr.repeats[0] = CreateExpr("dynamic_rows");
  const auto dynamic = evaluate();
  EXPECT_EQ(dynamic.ternary_ops.count(bound), 0U);
  const std::vector<std::pair<Expr, Expr>> no_cohort = {{cores, CreateExpr(56)}, {width, CreateExpr(16)}};
  EXPECT_DOUBLE_EQ(EvaluatePipeCycles(scoped, PipeType::AIV_MTE2, no_cohort),
                   EvaluatePipeCycles(dynamic, PipeType::AIV_MTE2, no_cohort));
  sources.back()->outputs[0].attr.repeats = original;
  reads.back()->outputs[0].attr.que.depth = 1;
  EXPECT_EQ(evaluate().ternary_ops.count(bound), 0U);
  reads.back()->outputs[0].attr.que.depth = 2;
  reads.back()->outputs[0].attr.dtype = ge::DT_INT8;
  EXPECT_EQ(evaluate().ternary_ops.count(bound), 0U);
  reads.back()->outputs[0].attr.dtype = ge::DT_UINT8;
  reads.back()->outputs[0].attr.strides[1] = CreateExpr(1605640);
  EXPECT_EQ(evaluate().ternary_ops.count(bound), 0U);
  reads.back()->outputs[0].attr.strides = gm;
  shape.gm_strides[1] = CreateExpr(160000);
  EXPECT_EQ(evaluate().ternary_ops.count(bound), 0U);
}

TEST_F(CalibratedAscirPerfV2, PackedFp16Dim2TransposeUsesMeasuredBankRepeat) {
  const auto rows = CreateExpr(36);
  const auto cols = CreateExpr(118);
  const auto one = CreateExpr(1);
  const auto packed = GetTransposePerfExpr(2, 2, {rows, cols}, {}, {one, rows}, true, kFloat16, {cols, one});
  EXPECT_NEAR(EvaluateTransposeCycles(packed), 1094.2, 1.0);
  const auto padded = GetTransposePerfExpr(2, 2, {rows, cols}, {}, {one, rows}, true, kFloat16, {cols + one, one});
  EXPECT_GT(EvaluateTransposeCycles(padded), EvaluateTransposeCycles(packed));
}

TEST_F(CalibratedAscirPerfV2, PackedFp32Dim2TransposeDistinguishesBankAlignedAxes) {
  const auto one = CreateExpr(1);
  const auto cycles = [&](int32_t tile) {
    const auto dim = CreateExpr(tile);
    return EvaluateTransposeCycles(GetTransposePerfExpr(2, 2, {dim, dim}, {}, {one, dim}, true, kFloat32, {dim, one}));
  };
  EXPECT_NEAR(cycles(117), 4769.8263, 0.1);
  EXPECT_GT(cycles(112), cycles(117));
}

TEST_F(CalibratedAscirPerfV2, TestAlignedWideGapLoadEstimate) {
  const Expr rows = CreateExpr("wide_rows");
  const Expr width = CreateExpr("wide_width");
  const Expr pitch = CreateExpr("wide_pitch");
  const Expr ub_gap = CreateExpr("wide_ub_gap");
  const Expr cores = CreateExpr("block_dim");
  NodeDetail node;
  node.name = "Load";
  node.optype = kLoad;
  node.input_dtype = {"float32"};
  node.output_dtype = {"float32"};
  node.input_dims = {rows, width};
  node.gm_stride = pitch - width;
  node.ub_stride = ub_gap;
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, perf), af::SUCCESS);
  const auto calibrated = CreateExpr("LoadAlignedWideGap_small_dma_node");
  const auto it = perf.ternary_ops.find(calibrated);
  ASSERT_NE(it, perf.ternary_ops.end());
  EXPECT_TRUE(HasLoadFallbackChain(perf.ternary_ops, SelectLoadCoreBranch(perf.ternary_ops, false), calibrated));

  const auto guard = it->second.DeepCopyIfCase();
  ASSERT_NE(guard, nullptr);
  const auto evaluate = [&](int32_t row_count, int32_t elements, int32_t gm_pitch, int32_t ub_padding,
                            int32_t core_count) {
    return af::SymbolicUtils::StaticCheckEq(guard->GetCondLeft().Replace({{rows, CreateExpr(row_count)},
                                                                          {width, CreateExpr(elements)},
                                                                          {pitch, CreateExpr(gm_pitch)},
                                                                          {ub_gap, CreateExpr(ub_padding)},
                                                                          {cores, CreateExpr(core_count)}}),
                                            af::sym::kSymbolZero);
  };
  EXPECT_EQ(evaluate(32, 592, 131072, 0, 56), af::TriBool::kTrue);
  EXPECT_EQ(evaluate(16, 384, 131072, 0, 40), af::TriBool::kTrue);
  EXPECT_EQ(evaluate(32, 768, 131072, 0, 56), af::TriBool::kTrue);
  EXPECT_EQ(evaluate(24, 592, 131072, 0, 56), af::TriBool::kFalse);
  EXPECT_EQ(evaluate(32, 240, 131072, 0, 56), af::TriBool::kFalse);
  EXPECT_EQ(evaluate(32, 593, 131072, 0, 56), af::TriBool::kFalse);
  EXPECT_EQ(evaluate(32, 592, 262144, 0, 56), af::TriBool::kFalse);
  EXPECT_EQ(evaluate(32, 592, 131072, 8, 56), af::TriBool::kFalse);
  EXPECT_EQ(evaluate(32, 592, 131072, 0, 57), af::TriBool::kFalse);

  double cycles = 0.0;
  ASSERT_TRUE(guard->GetChoiceA()
                  ->GetExpr()
                  .Replace({{rows, CreateExpr(32)}, {width, CreateExpr(592)}, {cores, CreateExpr(56)}})
                  .GetConstValue(cycles));
  EXPECT_NEAR(cycles, 4295.07, 0.1);
}

void ExpectAlignedLargePitchLoadDomain(const Expr &condition, const std::vector<Expr> &symbols) {
  ExpectDmaDomain(condition, symbols, {{104, 112, 28672, 0, 32}, {184, 168, 36864, 0, 56}, {120, 112, 32128, 0, 41}},
                  af::TriBool::kTrue);
  ExpectDmaDomain(condition, symbols,
                  {{1, 160, 32128, 0, 41},
                   {185, 112, 32128, 0, 41},
                   {120, 111, 32128, 0, 41},
                   {120, 113, 32128, 0, 41},
                   {120, 176, 32128, 0, 41},
                   {120, 112, 16384, 0, 41},
                   {120, 112, 36865, 0, 41},
                   {120, 112, 32128, 8, 41},
                   {120, 112, 32128, 0, 31},
                   {120, 112, 32128, 0, 57}},
                  af::TriBool::kFalse);
}

TEST_F(CalibratedAscirPerfV2, AlignedLargePitchLoadKeepsMeasuredDomainAndSymbolicParity) {
  const Expr rows = CreateExpr("large_pitch_rows");
  const Expr width = CreateExpr("large_pitch_width");
  const Expr pitch = CreateExpr("large_pitch_gm");
  const Expr ub_gap = CreateExpr("large_pitch_ub_gap");
  const Expr cores = CreateExpr("block_dim");
  NodeDetail node;
  node.name = "Load";
  node.optype = kLoad;
  node.input_dtype = {"float32"};
  node.output_dtype = {"float32"};
  node.input_dims = {rows, width};
  node.gm_stride = pitch - width;
  node.ub_stride = ub_gap;
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, perf), af::SUCCESS);
  const auto name = CreateExpr("LoadAlignedLargePitch_small_dma_node");
  const auto it = perf.ternary_ops.find(name);
  ASSERT_NE(it, perf.ternary_ops.end());
  const auto guard = it->second.DeepCopyIfCase();
  ASSERT_NE(guard, nullptr);
  ExpectAlignedLargePitchLoadDomain(guard->GetCondLeft(), {rows, width, pitch, ub_gap, cores});
  double predicted = 0.0;
  ASSERT_TRUE(guard->GetChoiceA()
                  ->GetExpr()
                  .Replace({{rows, CreateExpr(120)}, {width, CreateExpr(112)}, {cores, CreateExpr(41)}})
                  .GetConstValue(predicted));
  EXPECT_NEAR(predicted, 120.0 * 128 * 4 * 41 / 967.9169608668066, 0.01);
  node.input_dims = {CreateExpr(120), CreateExpr(112)};
  node.gm_stride = CreateExpr(32128 - 112);
  node.ub_stride = CreateExpr(0);
  PerfOutputInfo fixed;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, fixed), af::SUCCESS);
  const auto static_guard = fixed.ternary_ops.at(name).DeepCopyIfCase();
  EXPECT_EQ(af::SymbolicUtils::StaticCheckEq(static_guard->GetCondLeft().Replace({{cores, CreateExpr(41)}}),
                                             af::sym::kSymbolZero),
            af::TriBool::kTrue);
  double static_predicted = 0.0;
  ASSERT_TRUE(static_guard->GetChoiceA()->GetExpr().Replace({{cores, CreateExpr(41)}}).GetConstValue(static_predicted));
  EXPECT_NEAR(static_predicted, predicted, 1e-6);
  for (const auto &dtype : {"float16", "uint8"}) {
    node.input_dtype = {dtype};
    node.output_dtype = {dtype};
    PerfOutputInfo outside;
    ASSERT_EQ(ascendcapi_v2::LoadPerf(node, outside), af::SUCCESS);
    EXPECT_EQ(outside.ternary_ops.count(name), 0U);
  }
}

TEST_F(CalibratedAscirPerfV2, TestWidePitch128BLoadEstimate) {
  const Expr rows = CreateExpr("wide_pitch_rows");
  const Expr width = CreateExpr("wide_pitch_width");
  const Expr pitch = CreateExpr("wide_pitch_gm");
  const Expr ub_gap = CreateExpr("wide_pitch_ub_gap");
  NodeDetail node;
  node.name = "Load";
  node.optype = kLoad;
  node.input_dtype = {"float32"};
  node.output_dtype = {"float32"};
  node.input_dims = {rows, width};
  node.gm_stride = pitch - width;
  node.ub_stride = ub_gap;
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, perf), af::SUCCESS);
  const auto it = perf.ternary_ops.find(CreateExpr("LoadWidePitch128B_small_dma_node"));
  ASSERT_NE(it, perf.ternary_ops.end());
  const auto guard = it->second.DeepCopyIfCase();
  ASSERT_NE(guard, nullptr);
  const auto accepts = [&](int32_t row_count, int32_t elements, int32_t gm_pitch, int32_t ub_padding) {
    const auto condition = guard->GetCondLeft().Replace({{rows, CreateExpr(row_count)},
                                                         {width, CreateExpr(elements)},
                                                         {pitch, CreateExpr(gm_pitch)},
                                                         {ub_gap, CreateExpr(ub_padding)}});
    return af::SymbolicUtils::StaticCheckEq(condition, af::sym::kSymbolZero);
  };
  EXPECT_EQ(accepts(512, 32, 8192, 0), af::TriBool::kTrue);
  EXPECT_EQ(accepts(256, 32, 4096, 0), af::TriBool::kTrue);
  EXPECT_EQ(accepts(384, 32, 16384, 0), af::TriBool::kTrue);
  EXPECT_EQ(accepts(512, 32, 8193, 0), af::TriBool::kFalse);
  EXPECT_EQ(accepts(513, 32, 8192, 0), af::TriBool::kFalse);
  EXPECT_EQ(accepts(512, 32, 8192, 8), af::TriBool::kFalse);
  EXPECT_EQ(accepts(512, 128, 8192, 0), af::TriBool::kFalse);
  double cycles = 0.0;
  ASSERT_TRUE(
      guard->GetChoiceA()
          ->GetExpr()
          .Replace({{rows, CreateExpr(512)}, {width, CreateExpr(32)}, {CreateExpr("block_dim"), CreateExpr(52)}})
          .GetConstValue(cycles));
  EXPECT_NEAR(cycles, 3147.44, 0.1);
}

TEST_F(CalibratedAscirPerfV2, TestSmallDmaCalibratedDomains) {
  const Expr rows = CreateExpr("dma_rows");
  const Expr width = CreateExpr("dma_width");
  const Expr gap = CreateExpr("dma_gap");
  const Expr cores = CreateExpr("block_dim");
  NodeDetail node;
  node.name = "Load";
  node.optype = kLoad;
  node.input_dtype = {"float32"};
  node.output_dtype = {"float32"};
  node.input_dims = {rows, width};
  node.gm_stride = gap;
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, perf), af::SUCCESS);
  const auto normal = perf.ternary_ops.find(CreateExpr("LoadSmallPayload_small_dma_node"));
  const auto compact = perf.ternary_ops.find(CreateExpr("LoadSmallPayloadCompact_small_dma_node"));
  ASSERT_NE(normal, perf.ternary_ops.end());
  ASSERT_NE(compact, perf.ternary_ops.end());
  const auto accepts = [&](const TernaryOp &guard, int32_t r, int32_t w, int32_t g, int32_t c) {
    return af::SymbolicUtils::StaticCheckEq(
        guard.DeepCopyIfCase()->GetCondLeft().Replace(
            {{rows, CreateExpr(r)}, {width, CreateExpr(w)}, {gap, CreateExpr(g)}, {cores, CreateExpr(c)}}),
        af::sym::kSymbolZero);
  };
  EXPECT_EQ(accepts(normal->second, 1, 128, 0, 8), af::TriBool::kTrue);
  EXPECT_EQ(accepts(normal->second, 1, 2048, 0, 48), af::TriBool::kTrue);
  EXPECT_EQ(accepts(normal->second, 1, 120, 0, 8), af::TriBool::kFalse);
  EXPECT_EQ(accepts(normal->second, 1, 2056, 0, 48), af::TriBool::kFalse);
  EXPECT_EQ(accepts(normal->second, 4, 128, 64, 32), af::TriBool::kFalse);
  EXPECT_EQ(accepts(normal->second, 4, 128, 0, 7), af::TriBool::kFalse);
  EXPECT_EQ(accepts(normal->second, 4, 128, 0, 49), af::TriBool::kFalse);
  EXPECT_EQ(accepts(compact->second, 2, 64, 64, 8), af::TriBool::kTrue);
  EXPECT_EQ(accepts(compact->second, 18, 64, 4096, 48), af::TriBool::kTrue);
  EXPECT_EQ(accepts(compact->second, 1, 128, 64, 32), af::TriBool::kFalse);
  EXPECT_EQ(accepts(compact->second, 19, 128, 64, 32), af::TriBool::kFalse);
  EXPECT_EQ(accepts(compact->second, 4, 128, 63, 32), af::TriBool::kFalse);
  EXPECT_EQ(accepts(compact->second, 4, 128, 4097, 32), af::TriBool::kFalse);
  EXPECT_EQ(accepts(compact->second, 4, 127, 64, 32), af::TriBool::kFalse);
}

template <typename Accepts>
void CheckAlignedStorePayloadDomains(const PerfOutputInfo &store, const Accepts &accepts) {
  const auto medium = store.ternary_ops.find(CreateExpr("StoreAlignedMediumPayload_small_dma_node"));
  ASSERT_NE(medium, store.ternary_ops.end());
  EXPECT_EQ(accepts(medium->second, 1, 6144, 0, 16), af::TriBool::kTrue);
  EXPECT_EQ(accepts(medium->second, 1, 6140, 0, 16), af::TriBool::kFalse);
  EXPECT_EQ(accepts(medium->second, 2, 3072, 0, 16), af::TriBool::kFalse);
  EXPECT_EQ(accepts(medium->second, 1, 6144, 0, 41), af::TriBool::kFalse);

  const auto concurrent = store.ternary_ops.find(CreateExpr("StoreAlignedConcurrentPayload_small_dma_node"));
  ASSERT_NE(concurrent, store.ternary_ops.end());
  EXPECT_EQ(accepts(concurrent->second, 1, 2112, 0, 42), af::TriBool::kTrue);
  EXPECT_EQ(accepts(concurrent->second, 1, 5120, 0, 56), af::TriBool::kTrue);
  EXPECT_EQ(accepts(concurrent->second, 2, 2560, 0, 56), af::TriBool::kFalse);
  EXPECT_EQ(accepts(concurrent->second, 1, 5112, 0, 56), af::TriBool::kFalse);
  EXPECT_EQ(accepts(concurrent->second, 1, 2080, 0, 42), af::TriBool::kFalse);
  EXPECT_EQ(accepts(concurrent->second, 1, 5152, 0, 56), af::TriBool::kFalse);
  EXPECT_EQ(accepts(concurrent->second, 1, 5120, 0, 41), af::TriBool::kFalse);
}

TEST_F(CalibratedAscirPerfV2, TestSmallStoreCalibratedDomain) {
  const Expr rows = CreateExpr("dma_rows");
  const Expr width = CreateExpr("dma_width");
  const Expr gap = CreateExpr("dma_gap");
  const Expr cores = CreateExpr("block_dim");
  NodeDetail node;
  node.name = "Load";
  node.optype = kLoad;
  node.input_dtype = {"float32"};
  node.output_dtype = {"float32"};
  node.input_dims = {rows, width};
  node.gm_stride = gap;
  PerfOutputInfo perf;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, perf), af::SUCCESS);
  const auto normal = perf.ternary_ops.find(CreateExpr("LoadSmallPayload_small_dma_node"));
  const auto compact = perf.ternary_ops.find(CreateExpr("LoadSmallPayloadCompact_small_dma_node"));
  ASSERT_NE(normal, perf.ternary_ops.end());
  ASSERT_NE(compact, perf.ternary_ops.end());
  const auto accepts = [&](const TernaryOp &guard, int32_t r, int32_t w, int32_t g, int32_t c) {
    return af::SymbolicUtils::StaticCheckEq(
        guard.DeepCopyIfCase()->GetCondLeft().Replace(
            {{rows, CreateExpr(r)}, {width, CreateExpr(w)}, {gap, CreateExpr(g)}, {cores, CreateExpr(c)}}),
        af::sym::kSymbolZero);
  };
  node.optype = kStore;
  node.repeats = node.input_dims;
  node.gm_stride = CreateExpr(0);
  PerfOutputInfo store;
  ASSERT_EQ(ascendcapi_v2::StorePerf(node, store), af::SUCCESS);
  const auto selected = store.ternary_ops.find(CreateExpr("StoreSmallPayload_small_dma_node"));
  ASSERT_NE(selected, store.ternary_ops.end());
  EXPECT_EQ(accepts(selected->second, 1, 256, 0, 8), af::TriBool::kTrue);
  EXPECT_EQ(accepts(selected->second, 1, 2048, 0, 56), af::TriBool::kTrue);
  EXPECT_EQ(accepts(selected->second, 1, 248, 0, 8), af::TriBool::kFalse);
  EXPECT_EQ(accepts(selected->second, 1, 2056, 0, 56), af::TriBool::kFalse);
  double cycles = 0.0;
  ASSERT_TRUE(selected->second.DeepCopyIfCase()
                  ->GetChoiceA()
                  ->GetExpr()
                  .Replace({{rows, CreateExpr(4)}, {width, CreateExpr(128)}, {cores, CreateExpr(32)}})
                  .GetConstValue(cycles));
  EXPECT_NEAR(cycles, 279.5940, 0.01);

  CheckAlignedStorePayloadDomains(store, accepts);
}

TEST_F(CalibratedAscirPerfV2, TestSmallDmaUnsupportedDescriptorsKeepLegacy) {
  NodeDetail node;
  node.name = "Load";
  node.optype = kLoad;
  node.input_dtype = {"float32"};
  node.output_dtype = {"float32"};
  node.input_dims = {CreateExpr(4), CreateExpr(128)};
  node.ub_stride = CreateExpr(8);
  PerfOutputInfo ub_gap;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, ub_gap), af::SUCCESS);
  EXPECT_EQ(ub_gap.ternary_ops.count(CreateExpr("LoadSmallPayload_small_dma_node")), 0U);
  EXPECT_EQ(ub_gap.ternary_ops.count(CreateExpr("LoadSmallPayloadCompact_small_dma_node")), 0U);

  node.ub_stride = CreateExpr(0);
  node.input_dtype = {"float16"};
  node.output_dtype = {"float16"};
  PerfOutputInfo other_dtype;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, other_dtype), af::SUCCESS);
  EXPECT_EQ(other_dtype.ternary_ops.count(CreateExpr("LoadSmallPayload_small_dma_node")), 0U);

  node.input_dtype = {"float32"};
  node.output_dtype = {"float32"};
  node.vectorized_shape = {CreateExpr(2), CreateExpr(2), CreateExpr(128)};
  PerfOutputInfo loop_mode;
  ASSERT_EQ(ascendcapi_v2::LoadPerf(node, loop_mode), af::SUCCESS);
  EXPECT_EQ(loop_mode.ternary_ops.count(CreateExpr("LoadSmallPayload_small_dma_node")), 0U);
}

TEST_F(CalibratedAscirPerfV2, TransposeFp32Dim2Inner1UsesActiveBankRequestsAndMeasuredMovement) {
  const Expr rows = CreateExpr("fp32_dim2_rows");
  const Expr cols = CreateExpr("fp32_dim2_cols");
  const Expr pitch = CreateExpr("fp32_dim2_pitch");
  const auto make = [](const Expr &m, const Expr &n, const Expr &p, const Expr &padding, const std::string &dtype) {
    return GetTransposePerfExpr(2, 1, {m, n}, {CreateExpr(2)}, {CreateExpr(1), p}, true, dtype,
                                {n + padding, CreateExpr(1)});
  };
  const auto symbolic = make(rows, cols, pitch, CreateExpr(0), kFloat32);
  const auto eval = [&](int32_t m, int32_t n, int32_t p) {
    return EvaluateTransposeCycles(symbolic, {{rows, CreateExpr(m)}, {cols, CreateExpr(n)}, {pitch, CreateExpr(p)}});
  };
  EXPECT_NEAR(eval(254, 32, 256), 2.0 * (89.0 + 254.0 * 1.804167591830145 + 8128.0), 1e-5);
  EXPECT_NEAR(eval(88, 88, 88), 2.0 * (89.0 + 176.0 * 1.804167591830145 + 1936.0), 1e-5);
  const auto aligned_stride = af::sym::Ceiling(cols / CreateExpr(8)) * CreateExpr(8);
  const auto symbolic_padded = make(rows, cols, pitch, aligned_stride - cols, kFloat32);
  const auto masked = EvaluateTransposeCycles(
      symbolic_padded, {{rows, CreateExpr(87)}, {cols, CreateExpr(87)}, {pitch, CreateExpr(88)}});
  EXPECT_NEAR(masked, 2.0 * (89.0 + 174.0 * 1.804167591830145 + 1914.0), 1e-5);
  EXPECT_NEAR(masked,
              EvaluateTransposeCycles(make(CreateExpr(87), CreateExpr(87), CreateExpr(88), CreateExpr(1), kFloat32)),
              1e-5);
  for (const auto &point : std::vector<std::vector<int32_t>>{{78, 32, 80},
                                                             {264, 88, 272},
                                                             {99, 88, 104},
                                                             {77, 32, 80},
                                                             {265, 88, 272},
                                                             {88, 24, 88},
                                                             {88, 96, 88},
                                                             {88, 33, 88},
                                                             {88, 32, 89},
                                                             {90, 32, 88},
                                                             {78, 32, 72},
                                                             {264, 32, 280}}) {
    EXPECT_NEAR(eval(point[0], point[1], point[2]),
                EvaluateTransposeCycles(
                    make(CreateExpr(point[0]), CreateExpr(point[1]), CreateExpr(point[2]), CreateExpr(0), kFloat32)),
                1e-5);
  }
  const auto padded = make(CreateExpr(88), CreateExpr(88), CreateExpr(88), CreateExpr(8), kFloat32);
  EXPECT_EQ(padded.perf.ternary_ops.count(CreateExpr("transpose_fp32_dim2_inner1")), 0U);
  EXPECT_GT(EvaluateTransposeCycles(padded), eval(88, 88, 88));
  for (const auto &dtype : {kFloat16, kUInt8}) {
    const auto fallback = make(CreateExpr(88), CreateExpr(88), CreateExpr(88), CreateExpr(0), dtype);
    EXPECT_EQ(fallback.perf.ternary_ops.count(CreateExpr("transpose_fp32_dim2_inner1")), 0U);
  }
}

void ExpectStridedPayloadDomain(const Expr &condition, const std::vector<Expr> &symbols, int32_t padding) {
  ExpectDmaDomain(condition, symbols, {{88, 88, 22144, padding, 54}}, af::TriBool::kTrue);
  ExpectDmaDomain(condition, symbols,
                  {{32, 254, 1597248, 2, 56},
                   {88, 88, 22144, padding + 1, 54},
                   {88, 88, 22145, padding, 54},
                   {71, 88, 22144, padding, 54},
                   {88, 88, 20320, padding, 54},
                   {41, 254, 1597248, 2, 56},
                   {32, 254, 1597248, 2, 57}},
                  af::TriBool::kFalse);
}

TEST_F(CalibratedAscirPerfV2, StridedPayloadLoadUsesDtypeSpecificFitAndPhysicalGuards) {
  const Expr rows = CreateExpr("payload_rows");
  const Expr width = CreateExpr("payload_width");
  const Expr pitch = CreateExpr("payload_pitch");
  const Expr ub = CreateExpr("payload_ub");
  const Expr cores = CreateExpr("block_dim");
  for (const auto &dtype : {kFloat32, kUInt8}) {
    NodeDetail node;
    node.name = "Load";
    node.optype = kLoad;
    node.input_dtype = {dtype};
    node.output_dtype = {dtype};
    node.input_dims = {rows, width};
    node.gm_stride = pitch - width;
    node.ub_stride = ub;
    PerfOutputInfo perf;
    ASSERT_EQ(ascendcapi_v2::LoadPerf(node, perf), af::SUCCESS);
    const auto name = CreateExpr("load_strided_payload");
    ASSERT_NE(perf.ternary_ops.find(name), perf.ternary_ops.end());
    const auto guard = perf.ternary_ops.at(name).DeepCopyIfCase();
    const int32_t padding = dtype == kFloat32 ? 0 : 8;
    ExpectStridedPayloadDomain(guard->GetCondLeft(), {rows, width, pitch, ub, cores}, padding);
    const std::vector<std::pair<Expr, Expr>> fitted_bindings{{rows, CreateExpr(88)},
                                                             {width, CreateExpr(88)},
                                                             {pitch, CreateExpr(22144)},
                                                             {ub, CreateExpr(padding)},
                                                             {cores, CreateExpr(54)}};
    double predicted = 0;
    ASSERT_TRUE(guard->GetChoiceA()->GetExpr().Replace(fitted_bindings).GetConstValue(predicted));
    EXPECT_NEAR(
        predicted,
        dtype == kFloat32 ? 7744.0 * 4 * 54 / 995.867852951217 : 275.87602063506347 + 7744.0 * 54 / 2636.2293141488776,
        1e-4);
    node.input_dims = {CreateExpr(88), CreateExpr(88)};
    node.gm_stride = CreateExpr(22144 - 88);
    node.ub_stride = CreateExpr(padding);
    PerfOutputInfo fixed;
    ASSERT_EQ(ascendcapi_v2::LoadPerf(node, fixed), af::SUCCESS);
    EXPECT_EQ(af::SymbolicUtils::StaticCheckEq(
                  fixed.ternary_ops.at(name).DeepCopyIfCase()->GetCondLeft().Replace({{cores, CreateExpr(54)}}),
                  af::sym::kSymbolZero),
              af::TriBool::kTrue);
    node.input_dtype = {kFloat16};
    node.output_dtype = {kFloat16};
    PerfOutputInfo outside;
    ASSERT_EQ(ascendcapi_v2::LoadPerf(node, outside), af::SUCCESS);
    EXPECT_EQ(outside.ternary_ops.count(name), 0U);
  }
}
}  // namespace
