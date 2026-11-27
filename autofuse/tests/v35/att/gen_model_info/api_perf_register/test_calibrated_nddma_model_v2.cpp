/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <limits>
#include <numeric>
#include "gtest/gtest.h"
#include "common_utils.h"
#include "nddma_test_utils.h"
#include "v35/att/api_perf_register/nddma_model.h"

namespace att {
namespace {
TEST(CalibratedNddmaModelV2, SparseNg2SymbolicDomainMatchesConcreteDescriptors) {
  const Expr tile = CreateExpr("ng2_sparse_tile");
  for (const char *dtype : {"uint8", "float16", "float32", "int64"}) {
    for (const int core : {2, 43}) {
      SCOPED_TRACE(dtype);
      SCOPED_TRACE(core);
      NddmaDescriptorInfo descriptor;
      descriptor.output_dims = {tile, CreateExpr(118)};
      descriptor.input_strides = {CreateExpr(3384), CreateExpr(13536)};
      descriptor.output_strides = {CreateExpr(118), CreateExpr(1)};
      descriptor.vectorized_axis = {1, 0};
      NddmaModelResult symbolic;
      ASSERT_EQ(EvaluateNddmaModel(descriptor, dtype, CreateExpr(core), symbolic), af::SUCCESS);
      ASSERT_TRUE(symbolic.selected);
      ASSERT_EQ(symbolic.ternary_ops.size(), 1U);
      const auto guard = symbolic.ternary_ops.begin()->second.DeepCopyIfCase();
      ASSERT_EQ(guard->GetCondType(), CondType::K_LT);
      for (const int value : {2, 4, 16}) {
        SCOPED_TRACE(value);
        const auto bindings = std::vector<std::pair<Expr, Expr>>{{tile, CreateExpr(value)}};
        EXPECT_EQ(guard->GetCondLeft(), af::sym::kSymbolZero);
        const double left = 0.0;
        const double right = GetNddmaConstCycles(guard->GetCondRight().Replace(bindings));
        const auto choice = left < right ? guard->GetChoiceA() : guard->GetChoiceB();
        const double actual = GetNddmaConstCycles(choice->GetExpr().Replace(bindings));
        auto concrete = descriptor;
        concrete.output_dims[0] = CreateExpr(value);
        NddmaModelResult expected;
        ASSERT_EQ(EvaluateNddmaModel(concrete, dtype, CreateExpr(core), expected), af::SUCCESS);
        ASSERT_TRUE(expected.selected);
        EXPECT_GT(actual, 0.0);
        EXPECT_NEAR(actual, GetNddmaConstCycles(expected.cycles), 1e-6);
        EXPECT_EQ(expected.model_name, left < right ? "NDDMA_ND_MULTICORE_NG2" : "NDDMA_ND_MULTICORE_V1");
      }
    }
  }
}

TEST(CalibratedNddmaModelV2, CalibratedNg2ReplaysMeasuredWholeApiCost) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(512), CreateExpr(8)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(512)};
  descriptor.output_strides = {CreateExpr(8), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(4), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_NG2_V2");
  // Frozen NORMAL/fresh sample: full API cost, with no other-pipe estimate subtraction.
  constexpr double kMeasuredCycles = 941.4653418978562;
  EXPECT_LE(std::abs(GetNddmaConstCycles(result.cycles) - kMeasuredCycles), kMeasuredCycles * 0.5);
}

TEST(CalibratedNddmaModelV2, AppliesWideFp16Ng2OnlyToMeasuredPackedDomain) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(640), CreateExpr(32)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(1500000)};
  descriptor.output_strides = {CreateExpr(32), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(56), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_NG2_WIDE_FP16_V1");
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 8704.454634765, 1e-4);

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(24), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_NG2_V2");
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "bfloat16", CreateExpr(56), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_NG2_V2");

  const auto n = CreateExpr("wide_ng2_n");
  descriptor.output_dims[1] = n;
  descriptor.output_strides[0] = CreateExpr(16) * af::sym::Ceiling(n / CreateExpr(16));
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(56), result), af::SUCCESS);
  auto guarded = false;
  for (const auto &term : result.ternary_ops) {
    if (Str(term.first).find("nddma_ng2_wide_fp16") == std::string::npos) {
      continue;
    }
    guarded = true;
    const auto cond = term.second.DeepCopyIfCase();
    EXPECT_EQ(cond->GetCondType(), CondType::K_EQ);
    int64_t violation = 0;
    ASSERT_TRUE(cond->GetCondLeft().Replace({{n, CreateExpr(32)}}).GetConstValue(violation));
    EXPECT_EQ(violation, 0);
    ASSERT_TRUE(cond->GetCondLeft().Replace({{n, CreateExpr(24)}}).GetConstValue(violation));
    EXPECT_GT(violation, 0);
  }
  EXPECT_TRUE(guarded);
}

TEST(CalibratedNddmaModelV2, FitsFp16SparsePlanesAndUnitAxis) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(18), CreateExpr(4), CreateExpr(118)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(3384), CreateExpr(13536)};
  descriptor.output_strides = {CreateExpr(480), CreateExpr(118), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1, 2};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(56), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_SPARSE_FP16_V2");
  auto cycles = GetNddmaConstCycles(result.cycles);
  EXPECT_NEAR(cycles, 5382, 5382 * 0.25);

  descriptor.output_dims.insert(descriptor.output_dims.begin() + 1, CreateExpr(1));
  descriptor.input_strides.insert(descriptor.input_strides.begin() + 1, CreateExpr(1597248));
  descriptor.output_strides.insert(descriptor.output_strides.begin() + 1, CreateExpr(480));
  descriptor.vectorized_axis = {0, 1, 2, 3};
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(56), result), af::SUCCESS);
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), cycles, 1e-6);

  descriptor.output_dims[1] = CreateExpr(2);
  descriptor.output_strides[0] = CreateExpr(960);
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(56), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_V1");

  descriptor.output_dims = {CreateExpr(36), CreateExpr(4), CreateExpr(118)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(3384), CreateExpr(13536)};
  descriptor.output_strides = {CreateExpr(472), CreateExpr(118), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1, 2};
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(56), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_SPARSE_FP16_V2");
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 7356, 7356 * 0.1);

  descriptor.output_dims[0] = CreateExpr(16);
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(56), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_V1");
}

TEST(CalibratedNddmaModelV2, FitsFp16TwoLevelSparseRank3AndRank4) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(32), CreateExpr(36), CreateExpr(4)};
  descriptor.input_strides = {CreateExpr(1597248), CreateExpr(1), CreateExpr(3384)};
  descriptor.output_strides = {CreateExpr(144), CreateExpr(4), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1, 2};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(56), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_TWO_LEVEL_SPARSE_FP16_V1");
  const auto rank3 = GetNddmaConstCycles(result.cycles);
  EXPECT_NEAR(rank3, 1430.38, 0.01);

  descriptor.output_dims.insert(descriptor.output_dims.begin(), CreateExpr(6));
  descriptor.input_strides.insert(descriptor.input_strides.begin(), CreateExpr(36));
  descriptor.output_strides.insert(descriptor.output_strides.begin(), CreateExpr(4608));
  descriptor.vectorized_axis = {0, 1, 2, 3};
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(56), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_TWO_LEVEL_SPARSE_FP16_V1");
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 6511.98, 0.01);

  descriptor.input_strides[0] = CreateExpr(40);
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(56), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_V1");
}

TEST(CalibratedNddmaModelV2, FitsFp16CommensurateGmRowAndPlaneStrides) {
  NddmaDescriptorInfo desc;
  desc.output_dims = {CreateExpr(118), CreateExpr(36), CreateExpr(4)};
  desc.input_strides = {CreateExpr(13536), CreateExpr(1), CreateExpr(3384)};
  desc.output_strides = {CreateExpr(144), CreateExpr(4), CreateExpr(1)};
  desc.vectorized_axis = {0, 1, 2};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(desc, "float16", CreateExpr(56), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_FP16_COMMENSURATE_STRIDES_V1");
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 4764.11, 0.02);
  desc.input_strides[0] = CreateExpr(13535);
  ASSERT_EQ(EvaluateNddmaModel(desc, "float16", CreateExpr(56), result), af::SUCCESS);
  EXPECT_NE(result.model_name, "NDDMA_ND_MULTICORE_FP16_COMMENSURATE_STRIDES_V1");
}

TEST(CalibratedNddmaModelV2, WidePaddedFp32RowsPreservePayloadAndScopeResidualFit) {
  NddmaDescriptorInfo desc;
  desc.output_dims = {CreateExpr(26), CreateExpr(1007)};
  desc.input_strides = {CreateExpr(1007), CreateExpr(1)};
  desc.output_strides = {CreateExpr(1008), CreateExpr(1)};
  desc.vectorized_axis = {0, 1};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(desc, "float32", CreateExpr(55), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_WIDE_PADDED_ROW_FP32_V1");
  const double fitted = GetNddmaConstCycles(result.cycles);
  desc.output_dims.insert(desc.output_dims.begin() + 1, CreateExpr(1));
  desc.input_strides.insert(desc.input_strides.begin() + 1, CreateExpr(127889));
  desc.output_strides.insert(desc.output_strides.begin() + 1, CreateExpr(1007));
  desc.vectorized_axis = {0, 1, 2};
  ASSERT_EQ(EvaluateNddmaModel(desc, "float32", CreateExpr(55), result), af::SUCCESS);
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), fitted, 1e-6);
  ASSERT_EQ(EvaluateNddmaModel(desc, "float32", CreateExpr(47), result), af::SUCCESS);
  EXPECT_NE(result.model_name, "NDDMA_ND_MULTICORE_WIDE_PADDED_ROW_FP32_V1");
  const double baseline = 26.0 * 1007.0 * 4.0 / 29.4096 + 453.859;
  EXPECT_NEAR(fitted, baseline + 1.9724183277398053 * (GetNddmaConstCycles(result.cycles) - baseline), 1e-6);
  ASSERT_EQ(EvaluateNddmaModel(desc, "int32", CreateExpr(55), result), af::SUCCESS);
  EXPECT_NE(result.model_name, "NDDMA_ND_MULTICORE_WIDE_PADDED_ROW_FP32_V1");
  desc.output_strides[0] = CreateExpr(1007);
  ASSERT_EQ(EvaluateNddmaModel(desc, "float32", CreateExpr(55), result), af::SUCCESS);
  EXPECT_NE(result.model_name, "NDDMA_ND_MULTICORE_WIDE_PADDED_ROW_FP32_V1");
}

TEST(CalibratedNddmaModelV2, WidePaddedFp32RowsGuardRuntimeUnitAxisAndFittedBounds) {
  const auto rows = CreateExpr("wide_padded_rows");
  const auto unit = CreateExpr("wide_padded_unit");
  const auto cores = CreateExpr("wide_padded_cores");
  NddmaDescriptorInfo desc;
  desc.output_dims = {rows, unit, CreateExpr(1007)};
  desc.input_strides = {CreateExpr(1007), CreateExpr(127889), CreateExpr(1)};
  desc.output_strides = {CreateExpr(1008), CreateExpr(1007), CreateExpr(1)};
  desc.vectorized_axis = {0, 1, 2};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(desc, "float32", cores, result), af::SUCCESS);
  for (const auto &entry : result.ternary_ops) {
    if (std::string(entry.first.Str().get()).find("nddma_wide_padded_fp32") == std::string::npos) {
      continue;
    }
    const auto guard = entry.second.DeepCopyIfCase();
    for (const auto r : {17, 18, 26, 30, 31}) {
      for (const auto c : {47, 48, 55, 56, 57}) {
        for (const auto u : {1, 2}) {
          auto violation =
              guard->GetCondLeft().Replace({{rows, CreateExpr(r)}, {cores, CreateExpr(c)}, {unit, CreateExpr(u)}});
          violation.Simplify();
          int64_t value = 0;
          ASSERT_TRUE(violation.GetConstValue(value));
          EXPECT_EQ(value == 0, r >= 18 && r <= 30 && c >= 48 && c <= 56 && u == 1);
        }
      }
    }
    return;
  }
  ADD_FAILURE() << "wide padded row guard was not generated";
}

TEST(CalibratedNddmaModelV2, FitsFp32SparseWidePlane) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(16), CreateExpr(32), CreateExpr(48)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(36864), CreateExpr(16)};
  descriptor.output_strides = {CreateExpr(1536), CreateExpr(48), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1, 2};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(53), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_SPARSE_WIDE_PLANE_FP32_V1");
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 8241.2, 1.0);
  descriptor.input_strides[1] = CreateExpr(36832);
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(53), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_V1");
}

TEST(CalibratedNddmaModelV2, ShortStridedPlanesUsePhysicalFp32Descriptor) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(3), CreateExpr(12), CreateExpr(11)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(704), CreateExpr(64)};
  descriptor.output_strides = {CreateExpr(192), CreateExpr(16), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1, 2};
  NddmaModelResult result;
  const char *model = "NDDMA_ND_MULTICORE_SHORT_STRIDED3D_FP32_V1";
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(20), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, model);
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 700.2627, 0.01);

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(20), result), af::SUCCESS);
  EXPECT_NE(result.model_name, model);
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(4), result), af::SUCCESS);
  EXPECT_NE(result.model_name, model);
  descriptor.input_strides[1] = CreateExpr(720);
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(20), result), af::SUCCESS);
  EXPECT_NE(result.model_name, model);
}

TEST(CalibratedNddmaModelV2, FitsFp32NarrowPackedRows) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(48), CreateExpr(48), CreateExpr(3)};
  descriptor.input_strides = {CreateExpr(3), CreateExpr(144), CreateExpr(1)};
  descriptor.output_strides = {CreateExpr(144), CreateExpr(3), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1, 2};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(48), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_PACKED_SHORT_FP32_V1");
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 2832.41, 0.01);
  descriptor.output_strides[0] = CreateExpr(152);
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(48), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_V1");
}

TEST(CalibratedNddmaModelV2, SymbolicUnitAxisPreservesPackedFp32Transfer) {
  const auto unit = CreateExpr("packed_fp32_unit_extent");
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(24), CreateExpr(16), unit, CreateExpr(48)};
  descriptor.input_strides = {CreateExpr(768), CreateExpr(1), CreateExpr(36864), CreateExpr(16)};
  descriptor.output_strides = {CreateExpr(768) * unit, CreateExpr(48) * unit, CreateExpr(48), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1, 2, 3};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(54), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_RUNTIME_DISPATCH[NDDMA_ND_MULTICORE_PACKED3D_V1|NDDMA_ND_MULTICORE_V1]");
  const auto guard = result.ternary_ops.at(result.cycles).DeepCopyIfCase();
  ASSERT_NE(guard, nullptr);
  EXPECT_EQ(
      af::SymbolicUtils::StaticCheckEq(guard->GetCondLeft().Replace({{unit, CreateExpr(1)}}), af::sym::kSymbolZero),
      af::TriBool::kTrue);
  EXPECT_EQ(
      af::SymbolicUtils::StaticCheckEq(guard->GetCondLeft().Replace({{unit, CreateExpr(2)}}), af::sym::kSymbolZero),
      af::TriBool::kFalse);
  EXPECT_NEAR(GetNddmaConstCycles(guard->GetChoiceA()->GetExpr()), 5669.35, 1.0);
}

TEST(CalibratedNddmaModelV2, CalibratedNg2PreservesLegacyForNonUnitSourceStep) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(128), CreateExpr(32)};
  descriptor.input_strides = {CreateExpr(2), CreateExpr(512)};
  descriptor.output_strides = {CreateExpr(32), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(56), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_NG2");
  EXPECT_TRUE(result.cycles.IsValid());
}

TEST(CalibratedNddmaModelV2, CalibratedNg2KeepsLegacyDomainCoreAndInnerByteConditions) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr("m"), CreateExpr("n")};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(4096)};
  descriptor.output_strides = {CreateExpr("n"), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr("block_dim"), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_EQ(result.model_name,
            "NDDMA_RUNTIME_DISPATCH[NDDMA_RUNTIME_DISPATCH[NDDMA_ND_MULTICORE_NG2|NDDMA_ND_MULTICORE_V1]|"
            "NDDMA_ND_MULTICORE_NG2_V2]");
  EXPECT_EQ(result.ternary_ops.size(), 3U);
}

TEST(CalibratedNddmaModelV2, SymbolicPaddedPitchGuardsPackedCalibrationAndPreservesPaddedCost) {
  const Expr n = CreateExpr("packed_guard_n");
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(960), n};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(122880)};
  descriptor.output_strides = {CreateExpr(8) * af::sym::Ceiling(n / CreateExpr(8)), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1};
  NddmaModelResult dynamic;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(43), dynamic), af::SUCCESS);
  ASSERT_TRUE(dynamic.selected);
  EXPECT_EQ(dynamic.model_name, "NDDMA_RUNTIME_DISPATCH[NDDMA_ND_MULTICORE_NG2_V2|NDDMA_ND_MULTICORE_NG2]");
  ASSERT_EQ(dynamic.ternary_ops.size(), 2U);  // Inner-byte split and packed-layout equality; core count is static.
  bool found = false;
  for (const auto &entry : dynamic.ternary_ops) {
    if (std::string(entry.first.Str().get()).find("nddma_ng2_packed") == std::string::npos) {
      continue;
    }
    found = true;
    auto guard = entry.second.DeepCopyIfCase();
    EXPECT_EQ(guard->GetCondType(), CondType::K_EQ);
    for (const int value : {5, 8}) {
      const auto bindings = std::vector<std::pair<Expr, Expr>>{{n, CreateExpr(value)}};
      int64_t pitch = 0;
      int64_t inner = 0;
      ASSERT_TRUE(guard->GetCondLeft().Replace(bindings).GetConstValue(pitch));
      ASSERT_TRUE(guard->GetCondRight().Replace(bindings).GetConstValue(inner));
      EXPECT_EQ(pitch == inner, value == 8);
      if (value == 5) {
        // The unchanged pre-patch symbolic padded path uses the legacy NG2
        // multiplier. Preserve that expression, rather than selecting a
        // different legacy branch after making the descriptor static.
        constexpr double kOriginalSymbolicPaddedCycles = 30900.75893124455;
        EXPECT_NEAR(GetNddmaConstCycles(guard->GetChoiceB()->GetExpr().Replace(bindings)),
                    kOriginalSymbolicPaddedCycles, 1e-6);
      }
    }
  }
  EXPECT_TRUE(found);
  NddmaModelResult low;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(2), low), af::SUCCESS);
  EXPECT_TRUE(low.selected);
  EXPECT_TRUE(low.ternary_ops.empty());  // The unmeasured low-core path is unchanged.
}

TEST(CalibratedNddmaModelV2, FitsPackedThreeAxisB32AndKeepsOtherPaths) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(10), CreateExpr(64), CreateExpr(39)};
  descriptor.input_strides = {CreateExpr(2496), CreateExpr(1), CreateExpr(64)};
  descriptor.output_strides = {CreateExpr(2496), CreateExpr(39), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1, 2};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(52), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_PACKED3D_V1");
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 7614.416348341308, 1e-6);
  for (const auto cores : {2, 72}) {
    ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(cores), result), af::SUCCESS);
    EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_V1");
  }
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(52), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_V1");
  descriptor.input_strides[0] = CreateExpr(2500);
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(52), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_V1");
}

TEST(CalibratedNddmaModelV2, SparseThreeAxisResidualKeepsBandwidthAndStartup) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(235), CreateExpr(32), CreateExpr(2)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(65536), CreateExpr(2097152)};
  descriptor.output_strides = {CreateExpr(64), CreateExpr(2), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1, 2};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(56), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_SPARSE3D_V2");
  const double baseline = 235.0 * 32 * 2 * 4 / 29.4096 + 453.859;
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), baseline + 0.017212331646550515 * (121055.7889984 - baseline), 1e-4);
  for (const auto cores : {2, 39, 57}) {
    ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(cores), result), af::SUCCESS);
    EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_V1");
  }
  for (const auto width : {191, 257}) {
    descriptor.output_dims[0] = CreateExpr(width);
    ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(56), result), af::SUCCESS);
    EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_V1");
  }
  descriptor.output_dims[0] = CreateExpr(235);
  descriptor.input_strides = {CreateExpr(1), CreateExpr(262144), CreateExpr(8388608)};
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(56), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_V1");
}

TEST(CalibratedNddmaModelV2, SparseThreeAxisSymbolicGuardRetainsOutOfDomainCost) {
  const auto width = CreateExpr("sparse_width");
  const auto cores = CreateExpr("sparse_cores");
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {width, CreateExpr(32), CreateExpr(2)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(65536), CreateExpr(2097152)};
  descriptor.output_strides = {CreateExpr(64), CreateExpr(2), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1, 2};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", cores, result), af::SUCCESS);
  bool found = false;
  for (const auto &entry : result.ternary_ops) {
    if (std::string(entry.first.Str().get()).find("nddma_sparse_nd") == std::string::npos) {
      continue;
    }
    found = true;
    const auto guard = entry.second.DeepCopyIfCase();
    EXPECT_EQ(guard->GetCondType(), CondType::K_EQ);
    for (const auto w : {1, 191, 192, 235, 256, 257}) {
      for (const auto n : {2, 39, 40, 56, 57}) {
        auto condition = guard->GetCondLeft().Replace({{width, CreateExpr(w)}, {cores, CreateExpr(n)}});
        condition.Simplify();
        int64_t outside = 0;
        ASSERT_TRUE(condition.GetConstValue(outside));
        EXPECT_EQ(outside == 0, w >= 192 && w <= 256 && n >= 40 && n <= 56);
      }
    }
  }
  EXPECT_TRUE(found);
}

TEST(CalibratedNddmaModelV2, SparseFourAxisResidualKeepsOtherPitchAndLargerTiles) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(512), CreateExpr(2), CreateExpr(4), CreateExpr(2)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(512), CreateExpr(65536), CreateExpr(2097152)};
  descriptor.output_strides = {CreateExpr(16), CreateExpr(8), CreateExpr(2), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1, 2, 3};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(52), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_SPARSE4D_V2");
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 2796.1584908378627, 1e-6);
  // The four-axis fit is selected by layout, without a product core-count
  // special case; the high-core formula retains symbolic block_dim.
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(64), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_SPARSE4D_V2");
  for (const auto dtype : {"float16", "uint8", "uint64"}) {
    ASSERT_EQ(EvaluateNddmaModel(descriptor, dtype, CreateExpr(52), result), af::SUCCESS);
    EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_V1");
  }
  for (const auto pitch : {81920, 131072, 262144}) {
    descriptor.input_strides[2] = CreateExpr(pitch);
    descriptor.input_strides[3] = CreateExpr(32 * pitch);
    ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(52), result), af::SUCCESS);
    EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_V1");
  }
  descriptor.input_strides = {CreateExpr(1), CreateExpr(512), CreateExpr(65536), CreateExpr(2097152)};
  descriptor.output_dims[2] = CreateExpr(8);
  descriptor.output_strides = {CreateExpr(32), CreateExpr(16), CreateExpr(2), CreateExpr(1)};
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(52), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_V1");
}

TEST(CalibratedNddmaModelV2, SparseFourAxisSymbolicGuardIncludesMeasuredPadding) {
  const auto tile1 = CreateExpr("sparse_tile1");
  const auto tile2 = CreateExpr("sparse_tile2");
  const auto pitch = CreateExpr(8) * af::sym::Ceiling(tile2 / CreateExpr(4));
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(512), tile1, tile2, CreateExpr(2)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(512), CreateExpr(65536), CreateExpr(2097152)};
  descriptor.output_strides = {pitch * tile1, pitch, CreateExpr(2), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1, 2, 3};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(52), result), af::SUCCESS);
  bool found = false;
  for (const auto &entry : result.ternary_ops) {
    if (std::string(entry.first.Str().get()).find("nddma_sparse_nd") == std::string::npos) {
      continue;
    }
    found = true;
    const auto guard = entry.second.DeepCopyIfCase();
    for (const auto a : {1, 2, 3, 4}) {
      for (const auto b : {1, 3, 4, 5, 8}) {
        auto condition = guard->GetCondLeft().Replace({{tile1, CreateExpr(a)}, {tile2, CreateExpr(b)}});
        condition.Simplify();
        int64_t outside = 0;
        ASSERT_TRUE(condition.GetConstValue(outside));
        EXPECT_EQ(outside == 0, a <= 3 && (b == 3 || b == 4 || b == 8) && a * b <= 12);
      }
    }
  }
  EXPECT_TRUE(found);
}

TEST(CalibratedNddmaModelV2, SparseFourAxisPaddingRetainsLongSourceRunsAndOtherUbLayouts) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(512), CreateExpr(3), CreateExpr(3), CreateExpr(2)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(512), CreateExpr(65536), CreateExpr(2097152)};
  descriptor.output_strides = {CreateExpr(24), CreateExpr(8), CreateExpr(2), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1, 2, 3};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(53), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_SPARSE4D_V2");
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 4227.740075369205, 1e-6);
  descriptor.output_dims[0] = CreateExpr(1024);
  descriptor.input_strides[1] = CreateExpr(1024);
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(53), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_V1");
  descriptor.output_dims[0] = CreateExpr(512);
  descriptor.input_strides[1] = CreateExpr(512);
  descriptor.output_strides[1] = CreateExpr(16);
  descriptor.output_strides[0] = CreateExpr(48);
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(53), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_V1");
}

TEST(CalibratedNddmaModelV2, FitsOnlyMeasuredFp32PaddedNg2Domain) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(19), CreateExpr(385)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(19)};
  descriptor.output_strides = {CreateExpr(392), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(54), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_NG2_PADDED_V1");
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 3338.394714124599, 1e-6);

  for (const auto cores : {2, 15, 57}) {
    ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(cores), result), af::SUCCESS);
    EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_V1");
  }
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int32", CreateExpr(54), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_V1");
  descriptor.output_strides[0] = CreateExpr(400);
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(54), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_V1");
}

TEST(CalibratedNddmaModelV2, SeparatesFp32WidePitchTransposeByPhysicalLayout) {
  struct Transfer {
    int32_t rows;
    int32_t width;
    int32_t ub_pitch;
    int32_t cores;
    const char *model;
    double cycles;
  };
  const Transfer transfers[] = {
      {7, 2048, 2048, 49, "NDDMA_ND_MULTICORE_NARROW_SOURCE_WIDE_PITCH_FP32_V1", 8337.8591667},
      {114, 114, 120, 54, "NDDMA_ND_MULTICORE_NG2_SPARSE_PADDED_FP32_V1", 4622.49774115},
  };
  for (const auto &transfer : transfers) {
    SCOPED_TRACE(transfer.model);
    NddmaDescriptorInfo descriptor;
    descriptor.output_dims = {CreateExpr(transfer.rows), CreateExpr(transfer.width)};
    descriptor.input_strides = {CreateExpr(1), CreateExpr(1024)};
    descriptor.output_strides = {CreateExpr(transfer.ub_pitch), CreateExpr(1)};
    descriptor.vectorized_axis = {0, 1};
    NddmaModelResult result;
    ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(transfer.cores), result), af::SUCCESS);
    EXPECT_EQ(result.model_name, transfer.model);
    EXPECT_NEAR(GetNddmaConstCycles(result.cycles), transfer.cycles, 0.1);
    ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(2), result), af::SUCCESS);
    EXPECT_NE(result.model_name, transfer.model);
    ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(transfer.cores), result), af::SUCCESS);
    EXPECT_NE(result.model_name, transfer.model);
  }
}

TEST(CalibratedNddmaModelV2, ShortPaddedFp32TransposeUsesMeasuredDmaCost) {
  constexpr const char *model = "NDDMA_ND_MULTICORE_NG2_SHORT_PADDED_FP32_V1";
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(128), CreateExpr(9)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(4096)};
  descriptor.output_strides = {CreateExpr(16), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(32), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, model);
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 656.67, 0.1);
  descriptor.output_strides[0] = CreateExpr(24);
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(32), result), af::SUCCESS);
  EXPECT_NE(result.model_name, model);
}

TEST(CalibratedNddmaModelV2, SmallDenseTransposeMatchesTwoAndThreeAxisUnitOuter) {
  constexpr const char *model = "NDDMA_ND_MULTICORE_SMALL_DENSE_PADDED_FP32_V1";
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(32), CreateExpr(26)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(32)};
  descriptor.output_strides = {CreateExpr(32), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1};
  NddmaModelResult result;
  const auto check = [&]() {
    ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(8), result), af::SUCCESS);
    EXPECT_EQ(result.model_name, model);
    EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 552.26, 0.1);
  };
  check();
  descriptor.output_dims.insert(descriptor.output_dims.begin(), CreateExpr(1));
  descriptor.input_strides.insert(descriptor.input_strides.begin(), CreateExpr(832));
  descriptor.output_strides.insert(descriptor.output_strides.begin(), CreateExpr(1024));
  descriptor.vectorized_axis = {0, 1, 2};
  check();
  descriptor.output_dims[0] = CreateExpr(2);
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(8), result), af::SUCCESS);
  EXPECT_NE(result.model_name, model);
}

TEST(CalibratedNddmaModelV2, DensePaddedTransposeUsesMeasuredFp32DescriptorDomain) {
  constexpr const char *model = "NDDMA_ND_MULTICORE_DENSE_PADDED_FP32_V1";
  const NddmaDescriptorInfo descriptor{
      {CreateExpr(64), CreateExpr(39)}, {CreateExpr(1), CreateExpr(64)}, {CreateExpr(40), CreateExpr(1)}, {0, 1}};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(52), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, model);
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 916.02061683, 1e-5);

  auto packed = descriptor;
  packed.output_dims[1] = CreateExpr(40);
  auto sparse = descriptor;
  sparse.input_strides[1] = CreateExpr(256);
  for (const auto &control : {packed, sparse}) {
    ASSERT_EQ(EvaluateNddmaModel(control, "float32", CreateExpr(52), result), af::SUCCESS);
    EXPECT_NE(result.model_name, model);
  }
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(18), result), af::SUCCESS);
  EXPECT_NE(result.model_name, model);
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(52), result), af::SUCCESS);
  EXPECT_NE(result.model_name, model);
}

TEST(CalibratedNddmaModelV2, DensePaddedTransposeKeepsSeparateNarrowPackedRowFit) {
  constexpr const char *model = "NDDMA_ND_MULTICORE_NARROW_DENSE_PADDED_FP32_V1";
  NddmaDescriptorInfo descriptor{
      {CreateExpr(96), CreateExpr(27)}, {CreateExpr(1), CreateExpr(96)}, {CreateExpr(32), CreateExpr(1)}, {0, 1}};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(48), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, model);
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 836.2782, 0.01);

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(48), result), af::SUCCESS);
  EXPECT_NE(result.model_name, model);
  descriptor.output_dims[1] = CreateExpr(32);
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(48), result), af::SUCCESS);
  EXPECT_NE(result.model_name, model);
}

TEST(CalibratedNddmaModelV2, PaddedInnerBroadcastUsesPhysicalZeroStride) {
  constexpr const char *model = "NDDMA_ND_PADDED_INNER_BROADCAST_FP32_V1";
  NddmaDescriptorInfo descriptor{
      {CreateExpr(128), CreateExpr(26)}, {CreateExpr(1), CreateExpr(0)}, {CreateExpr(32), CreateExpr(1)}, {0, 1}};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(32), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, model);
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 657.82210, 0.01);

  auto no_broadcast = descriptor;
  no_broadcast.input_strides.back() = CreateExpr(128);
  ASSERT_EQ(EvaluateNddmaModel(no_broadcast, "float32", CreateExpr(32), result), af::SUCCESS);
  EXPECT_NE(result.model_name, model);
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(32), result), af::SUCCESS);
  EXPECT_NE(result.model_name, model);
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(20), result), af::SUCCESS);
  EXPECT_NE(result.model_name, model);

  descriptor.output_dims.insert(descriptor.output_dims.begin(), CreateExpr(1));
  descriptor.input_strides.insert(descriptor.input_strides.begin(), CreateExpr(0));
  descriptor.output_strides.insert(descriptor.output_strides.begin(), CreateExpr(4096));
  descriptor.vectorized_axis = {0, 1, 2};
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(43), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, model);
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 657.82210, 0.01);
}

TEST(CalibratedNddmaModelV2, SymbolicPaddedNg2GuardExcludesPackedAndOutOfDomainTiles) {
  const auto m = CreateExpr("padded_ng2_m");
  const auto n = CreateExpr("padded_ng2_n");
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {m, n};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(19)};
  descriptor.output_strides = {CreateExpr(8) * af::sym::Ceiling(n / CreateExpr(8)), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(54), result), af::SUCCESS);
  EXPECT_EQ(result.model_name,
            "NDDMA_RUNTIME_DISPATCH[NDDMA_ND_MULTICORE_NG2_V2|"
            "NDDMA_RUNTIME_DISPATCH[NDDMA_ND_MULTICORE_NG2_PADDED_V1|"
            "NDDMA_RUNTIME_DISPATCH[NDDMA_ND_MULTICORE_NG2|NDDMA_ND_MULTICORE_V1]]]");
  bool found = false;
  for (const auto &entry : result.ternary_ops) {
    if (std::string(entry.first.Str().get()).find("nddma_ng2_padded") == std::string::npos) {
      continue;
    }
    found = true;
    const auto guard = entry.second.DeepCopyIfCase();
    EXPECT_EQ(guard->GetCondType(), CondType::K_EQ);
    const auto evaluate = [&](const std::vector<std::pair<Expr, Expr>> &bindings) {
      auto condition = guard->GetCondLeft().Replace(bindings);
      condition.Simplify();
      int64_t violation = 0;
      EXPECT_TRUE(condition.GetConstValue(violation));
      return violation;
    };
    for (const auto inner : {192, 384, 385, 521}) {
      const std::vector<std::pair<Expr, Expr>> bindings{{m, CreateExpr(19)}, {n, CreateExpr(inner)}};
      EXPECT_EQ(evaluate(bindings) == 0, inner == 385);
    }
    const std::vector<std::pair<Expr, Expr>> gapped{{m, CreateExpr(18)}, {n, CreateExpr(385)}};
    EXPECT_GT(evaluate(gapped), 0);
  }
  EXPECT_TRUE(found);
}

TEST(CalibratedNddmaModelV2, GuardsPackedThreeAxisCostForSymbolicOuterAndCoreCount) {
  const auto outer = CreateExpr("packed_outer");
  const auto cores = CreateExpr("packed_cores");
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {outer, CreateExpr(64), CreateExpr(39)};
  descriptor.input_strides = {CreateExpr(2496), CreateExpr(1), CreateExpr(64)};
  descriptor.output_strides = {CreateExpr(2496), CreateExpr(39), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1, 2};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", cores, result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_EQ(result.model_name,
            "NDDMA_RUNTIME_DISPATCH[NDDMA_ND_MULTICORE_V1|"
            "NDDMA_RUNTIME_DISPATCH[NDDMA_ND_MULTICORE_PACKED3D_V1|NDDMA_ND_MULTICORE_V1]]");
  ASSERT_EQ(result.ternary_ops.size(), 2U);
  for (const auto &entry : result.ternary_ops) {
    if (std::string(entry.first.Str().get()).find("nddma_packed3d") == std::string::npos) {
      continue;
    }
    const auto guard = entry.second.DeepCopyIfCase();
    for (const auto batch : {1, 10}) {
      for (const auto blocks : {52, 72}) {
        const std::vector<std::pair<Expr, Expr>> bindings{{outer, CreateExpr(batch)}, {cores, CreateExpr(blocks)}};
        int64_t use_fitted = 0;
        auto predicate = guard->GetCondLeft().Replace(bindings);
        predicate.Simplify();
        ASSERT_TRUE(predicate.GetConstValue(use_fitted));
        EXPECT_EQ(use_fitted, batch == 10 && blocks == 52 ? 1 : 0);
      }
    }
    return;
  }
  ADD_FAILURE() << "packed 3D cost guard was not generated";
}

TEST(CalibratedNddmaModelV2, FitsSmallInt64PackedAndPaddedTransposes) {
  for (const auto m : {24, 38, 64}) {
    for (const auto n : {2, 3, 8}) {
      for (const auto padding : {0, 1}) {
        NddmaDescriptorInfo descriptor;
        descriptor.output_dims = {CreateExpr(m), CreateExpr(n)};
        descriptor.input_strides = {CreateExpr(1), CreateExpr(m)};
        descriptor.output_strides = {CreateExpr(n + padding), CreateExpr(1)};
        descriptor.vectorized_axis = {0, 1};
        for (const auto cores : {16, 43, 56}) {
          NddmaModelResult result;
          ASSERT_EQ(EvaluateNddmaModel(descriptor, "int64", CreateExpr(cores), result), af::SUCCESS);
          ASSERT_TRUE(result.selected);
          EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_NG2_SMALL_INT64_V1");
          EXPECT_DOUBLE_EQ(GetNddmaConstCycles(result.cycles), 506.0);
        }
      }
    }
  }
}

TEST(CalibratedNddmaModelV2, KeepsSmallInt64UnmeasuredLayoutsAndDtypes) {
  // Each row crosses one measured-domain boundary while retaining a valid descriptor.
  const int cases[][6] = {{23, 3, 23, 3, 43, 1}, {65, 3, 65, 3, 43, 1}, {38, 1, 38, 1, 43, 1}, {38, 9, 38, 9, 43, 1},
                          {38, 3, 38, 5, 43, 1}, {38, 3, 39, 3, 43, 1}, {38, 3, 76, 3, 43, 2}, {38, 3, 38, 3, 15, 1},
                          {38, 3, 38, 3, 57, 1}, {38, 3, 38, 3, 2, 1}};
  for (const auto &point : cases) {
    NddmaDescriptorInfo descriptor;
    descriptor.output_dims = {CreateExpr(point[0]), CreateExpr(point[1])};
    descriptor.input_strides = {CreateExpr(point[5]), CreateExpr(point[2])};
    descriptor.output_strides = {CreateExpr(point[3]), CreateExpr(1)};
    descriptor.vectorized_axis = {0, 1};
    NddmaModelResult result;
    ASSERT_EQ(EvaluateNddmaModel(descriptor, "int64", CreateExpr(point[4]), result), af::SUCCESS);
    ASSERT_TRUE(result.selected);
    EXPECT_NE(result.model_name, "NDDMA_ND_MULTICORE_NG2_SMALL_INT64_V1");
  }
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(38), CreateExpr(3)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(38)};
  descriptor.output_strides = {CreateExpr(3), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1};
  for (const auto dtype : {"uint8", "float16", "float32", "uint64"}) {
    NddmaModelResult result;
    ASSERT_EQ(EvaluateNddmaModel(descriptor, dtype, CreateExpr(43), result), af::SUCCESS);
    EXPECT_NE(result.model_name, "NDDMA_ND_MULTICORE_NG2_SMALL_INT64_V1");
  }
}

TEST(CalibratedNddmaModelV2, GuardsSmallInt64SymbolicTileAndCoreCount) {
  const auto n = CreateExpr("small_ng2_n");
  const auto cores = CreateExpr("small_ng2_cores");
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(38), n};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(38)};
  descriptor.output_strides = {n + CreateExpr(1), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int64", cores, result), af::SUCCESS);
  EXPECT_EQ(result.model_name,
            "NDDMA_RUNTIME_DISPATCH[NDDMA_ND_MULTICORE_NG2|"
            "NDDMA_RUNTIME_DISPATCH[NDDMA_ND_MULTICORE_NG2_SMALL_INT64_V1|NDDMA_ND_MULTICORE_NG2]]");
  for (const auto &entry : result.ternary_ops) {
    if (std::string(entry.first.Str().get()).find("nddma_ng2_small_int64") == std::string::npos) {
      continue;
    }
    const auto guard = entry.second.DeepCopyIfCase();
    EXPECT_EQ(guard->GetCondType(), CondType::K_EQ);
    EXPECT_DOUBLE_EQ(GetNddmaConstCycles(guard->GetChoiceA()->GetExpr()), 506.0);
    for (const auto tile : {1, 2, 8, 9}) {
      for (const auto core : {15, 16, 56, 57}) {
        auto violation = guard->GetCondLeft().Replace({{n, CreateExpr(tile)}, {cores, CreateExpr(core)}});
        violation.Simplify();
        int64_t value = 0;
        ASSERT_TRUE(violation.GetConstValue(value));
        EXPECT_EQ(value == 0, tile >= 2 && tile <= 8 && core >= 16 && core <= 56);
      }
    }
    return;
  }
  ADD_FAILURE() << "small INT64 NG2 guard was not generated";
}

TEST(CalibratedNddmaModelV2, NarrowSourceUsesIndependentRequestCostAndPhysicalUnitAxis) {
  for (const int m : {1, 3, 8}) {
    NddmaDescriptorInfo descriptor;
    descriptor.output_dims = {CreateExpr(m), CreateExpr(4096)};
    descriptor.input_strides = {CreateExpr(1), CreateExpr(512)};
    descriptor.output_strides = {CreateExpr(4096), CreateExpr(1)};
    descriptor.vectorized_axis = {0, 1};
    NddmaModelResult result;
    ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(56), result), af::SUCCESS);
    ASSERT_TRUE(result.selected);
    EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_NARROW_SOURCE_V1");
    EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 39245.68947748, 0.01);
  }
}

TEST(CalibratedNddmaModelV2, NarrowSourceKeepsUnmeasuredDtypesAndDescriptorDomains) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(3), CreateExpr(4096)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(512)};
  descriptor.output_strides = {CreateExpr(4096), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1};
  for (const auto *dtype : {"float16", "int32", "int64"}) {
    NddmaModelResult result;
    ASSERT_EQ(EvaluateNddmaModel(descriptor, dtype, CreateExpr(56), result), af::SUCCESS);
    EXPECT_EQ(result.model_name.find("NARROW_SOURCE"), std::string::npos);
  }
  for (const auto &dims : {std::vector<int>{9, 4096}, std::vector<int>{3, 511}, std::vector<int>{3, 16385}}) {
    descriptor.output_dims = {CreateExpr(dims[0]), CreateExpr(dims[1])};
    descriptor.output_strides[0] = CreateExpr(dims[1]);
    NddmaModelResult result;
    ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(56), result), af::SUCCESS);
    EXPECT_EQ(result.model_name.find("NARROW_SOURCE"), std::string::npos);
  }
}

TEST(CalibratedNddmaModelV2, NarrowSourceExtendsWithinMeasuredLongAxisAndPitch) {
  NddmaDescriptorInfo desc;
  desc.output_dims = {CreateExpr(9), CreateExpr(2688)};
  desc.input_strides = {CreateExpr(1), CreateExpr(324)};
  desc.output_strides = {CreateExpr(2688), CreateExpr(1)};
  desc.vectorized_axis = {0, 1};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(desc, "float32", CreateExpr(48), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_NARROW_SOURCE_V1");
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 21784.3073, 0.1);

  desc.input_strides[1] = CreateExpr(512);
  ASSERT_EQ(EvaluateNddmaModel(desc, "float32", CreateExpr(48), result), af::SUCCESS);
  EXPECT_EQ(result.model_name.find("NARROW_SOURCE"), std::string::npos);
}

TEST(CalibratedNddmaModelV2, NarrowSourcePublishesSymbolicDomainGuard) {
  const Expr m = CreateExpr("narrow_m");
  const Expr n = CreateExpr("narrow_n");
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {m, n};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(30)};
  descriptor.output_strides = {n, CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(48), result), af::SUCCESS);
  bool found = false;
  for (const auto &entry : result.ternary_ops) {
    if (std::string(entry.first.Str().get()).find("nddma_narrow_source") == std::string::npos) {
      continue;
    }
    found = true;
    auto guard = entry.second.DeepCopyIfCase();
    for (const int rows : {1, 8, 9}) {
      const std::vector<std::pair<Expr, Expr>> bindings = {{m, CreateExpr(rows)}, {n, CreateExpr(2048)}};
      auto violation = guard->GetCondLeft().Replace(bindings);
      violation.Simplify();
      int64_t value = 0;
      ASSERT_TRUE(violation.GetConstValue(value));
      EXPECT_EQ(value == 0, rows <= 8);
      if (rows <= 8) {
        EXPECT_NEAR(GetNddmaConstCycles(guard->GetChoiceA()->GetExpr().Replace(bindings)), 16661.30891215, 0.01);
      }
    }
  }
  EXPECT_TRUE(found);
}

TEST(CalibratedNddmaModelV2, DirectStoreContentionIsSeparateFromIsolatedApiCost) {
  NddmaDescriptorInfo descriptor{
      {CreateExpr(8), CreateExpr(3960)}, {CreateExpr(1), CreateExpr(8)}, {CreateExpr(3960), CreateExpr(1)}, {0, 1}};
  for (const int cores : {8, 56, 64, 72}) {
    NddmaModelResult isolated;
    NddmaModelResult direct;
    ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(cores), isolated), af::SUCCESS);
    ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(cores), direct, true), af::SUCCESS);
    ASSERT_TRUE(direct.selected);
    const double contention = 0.0038940645164791633 * std::max(0.0, 126720.0 * cores - 4456448.0);
    EXPECT_NEAR(GetNddmaConstCycles(direct.cycles) - GetNddmaConstCycles(isolated.cycles), contention, 1e-6);
    EXPECT_EQ(direct.model_name, "NDDMA_ND_MULTICORE_DIRECT_STORE_V1");
    EXPECT_EQ(isolated.model_name, "NDDMA_ND_MULTICORE_NG2_V2");
  }
}

TEST(CalibratedNddmaModelV2, DirectStoreKeepsUncalibratedDtypesAndPhysicalLayouts) {
  const NddmaDescriptorInfo packed{
      {CreateExpr(8), CreateExpr(3960)}, {CreateExpr(1), CreateExpr(8)}, {CreateExpr(3960), CreateExpr(1)}, {0, 1}};
  const auto unchanged = [](const NddmaDescriptorInfo &descriptor, const std::string &dtype, int cores) {
    NddmaModelResult isolated;
    NddmaModelResult direct;
    ASSERT_EQ(EvaluateNddmaModel(descriptor, dtype, CreateExpr(cores), isolated), af::SUCCESS);
    ASSERT_EQ(EvaluateNddmaModel(descriptor, dtype, CreateExpr(cores), direct, true), af::SUCCESS);
    ASSERT_TRUE(isolated.selected);
    ASSERT_TRUE(direct.selected);
    EXPECT_NEAR(GetNddmaConstCycles(direct.cycles), GetNddmaConstCycles(isolated.cycles), 1e-6);
    EXPECT_EQ(direct.model_name, isolated.model_name);
  };
  for (const auto *dtype : {"uint8", "float16", "int32", "int64"}) {
    unchanged(packed, dtype, 56);
  }
  unchanged(packed, "float32", 2);
  auto padded = packed;
  padded.output_strides[0] = CreateExpr(3968);
  unchanged(padded, "float32", 56);
  auto sparse = packed;
  sparse.input_strides[1] = CreateExpr(16);
  unchanged(sparse, "float32", 56);
  for (const int width : {12, 16, 24}) {
    auto wide = packed;
    wide.output_dims[0] = CreateExpr(width);
    wide.input_strides[1] = CreateExpr(width);
    unchanged(wide, "float32", 56);
  }
}

TEST(CalibratedNddmaModelV2, DirectStoreSymbolicGuardChecksRuntimePackingAndPayload) {
  const auto n = CreateExpr("bound_store_n");
  const auto cores = CreateExpr("block_dim");
  NddmaDescriptorInfo descriptor{{CreateExpr(8), n},
                                 {CreateExpr(1), CreateExpr(8)},
                                 {CreateExpr(8) * af::sym::Ceiling(n / CreateExpr(8)), CreateExpr(1)},
                                 {0, 1}};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", cores, result, true), af::SUCCESS);
  bool found = false;
  for (const auto &entry : result.ternary_ops) {
    if (std::string(entry.first.Str().get()).find("nddma_direct_store") == std::string::npos) {
      continue;
    }
    found = true;
    const auto guard = entry.second.DeepCopyIfCase();
    for (const int width : {511, 512, 3960, 3961, 4096, 4104}) {
      const std::vector<std::pair<Expr, Expr>> bindings = {{n, CreateExpr(width)}, {cores, CreateExpr(56)}};
      const bool in_domain = width >= 512 && width <= 4096 && width % 8 == 0;
      EXPECT_EQ(GetNddmaConstCycles(guard->GetCondLeft().Replace(bindings)) == 0, in_domain);
      if (in_domain) {
        const auto increment = guard->GetChoiceA()->GetExpr() - guard->GetChoiceB()->GetExpr();
        EXPECT_NEAR(GetNddmaConstCycles(increment.Replace(bindings)),
                    0.0038940645164791633 * std::max(0.0, 32.0 * width * 56 - 4456448.0), 1e-6);
      }
    }
  }
  EXPECT_TRUE(found);
}

TEST(CalibratedNddmaModelV2, CalibratesFp16PaddedShortRowsWithInnerGmStride) {
  NddmaDescriptorInfo desc;
  desc.output_dims = {CreateExpr(2), CreateExpr(38)};
  desc.input_strides = {CreateExpr(2886), CreateExpr(39)};
  desc.output_strides = {CreateExpr(48), CreateExpr(1)};
  desc.vectorized_axis = {0, 1};
  NddmaModelResult result;
  const char *fitted = "NDDMA_ND_MULTICORE_SHORT_INNER_STRIDE_FP16_V1";
  ASSERT_EQ(EvaluateNddmaModel(desc, "float16", CreateExpr(55), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_EQ(result.model_name, fitted);
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 657.157, 0.01);

  ASSERT_EQ(EvaluateNddmaModel(desc, "float16", CreateExpr(2), result), af::SUCCESS);
  EXPECT_NE(result.model_name, fitted);
  ASSERT_EQ(EvaluateNddmaModel(desc, "float32", CreateExpr(55), result), af::SUCCESS);
  EXPECT_NE(result.model_name, fitted);
  desc.output_strides[0] = CreateExpr(38);
  ASSERT_EQ(EvaluateNddmaModel(desc, "float16", CreateExpr(55), result), af::SUCCESS);
  EXPECT_NE(result.model_name, fitted);
}

TEST(CalibratedNddmaModelV2, CalibratesFp16PackedMediumPitchWithinMeasuredTileRange) {
  NddmaDescriptorInfo desc;
  desc.output_dims = {CreateExpr(176), CreateExpr(118)};
  desc.input_strides = {CreateExpr(1), CreateExpr(13536)};
  desc.output_strides = {CreateExpr(118), CreateExpr(1)};
  desc.vectorized_axis = {0, 1};
  NddmaModelResult result;
  const char *fitted = "NDDMA_ND_MULTICORE_NG2_MEDIUM_PITCH_FP16_V1";
  ASSERT_EQ(EvaluateNddmaModel(desc, "float16", CreateExpr(56), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, fitted);
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 8388.94, 0.1);
  desc.input_strides[1] = CreateExpr(22144);
  ASSERT_EQ(EvaluateNddmaModel(desc, "float16", CreateExpr(56), result), af::SUCCESS);
  EXPECT_NE(result.model_name, fitted);
}
}  // namespace
}  // namespace att
