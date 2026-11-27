/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "tests/depends/slog/src/slog_stub.h"
#include "common_utils.h"
#include "v35/att/api_perf_register/nddma_model.h"
#include "../../../../att/gen_model_info/api_perf_register/nddma_test_utils.h"

namespace att {
namespace {
NddmaDescriptorInfo MakeDescriptor(const Expr &dim, const Expr &input_stride, const Expr &output_stride) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {dim};
  descriptor.input_strides = {input_stride};
  descriptor.output_strides = {output_stride};
  descriptor.vectorized_axis = {0};
  return descriptor;
}

}  // namespace

TEST(NddmaModelV2, EvaluatesStaticB8LowCoreFormula) {
  const auto descriptor = MakeDescriptor(CreateExpr(256), CreateExpr(1), CreateExpr(1));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "uint8", CreateExpr(2), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_EQ(result.model_name, "NDDMA_1D_MULTICORE_V2");
  EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kNone);
  EXPECT_TRUE(result.ternary_ops.empty());
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 400.30727752648556, 1e-6);
}

TEST(NddmaModelV2, UsesMergedLowCoreCoefficientsForStridedOutput) {
  const auto descriptor = MakeDescriptor(CreateExpr("B"), CreateExpr("s"), CreateExpr(2));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "uint8", CreateExpr(2), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_TRUE(result.cycles.IsValid());
}

TEST(NddmaModelV2, AllowsZeroInputStride) {
  const auto descriptor = MakeDescriptor(CreateExpr(256), CreateExpr(0), CreateExpr(1));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "uint8", CreateExpr(2), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 216.18489573733697, 1e-6);
}

TEST(NddmaModelV2, EvaluatesStaticB16HighCoreFormula) {
  const auto descriptor = MakeDescriptor(CreateExpr(256), CreateExpr(4), CreateExpr(2));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(8), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 3604.1465027941949, 1e-6);
}

TEST(NddmaModelV2, EvaluatesStaticB32LowCoreFormula) {
  const auto descriptor = MakeDescriptor(CreateExpr(256), CreateExpr(4), CreateExpr(2));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(2), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 638.66344907846417, 1e-6);
}

TEST(NddmaModelV2, EvaluatesStaticB64HighCoreWithSaturatedInputStride) {
  const auto descriptor = MakeDescriptor(CreateExpr(256), CreateExpr(129), CreateExpr(1));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int64", CreateExpr(8), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 1788.750466100214, 1e-6);
}

TEST(NddmaModelV2, ReplaysAllCoefficientGroupsForFourDtypeSizes) {
  struct DtypeCase {
    const char *dtype;
    double low_core;
    double high_core;
  };
  const DtypeCase cases[] = {{"int8", 2022.2127058258638, 3828.6871159505763},
                             {"float16", 1850.8477176540077, 3604.1465027941949},
                             {"float32", 638.66344907846417, 847.32918281242814},
                             {"int64", 667.21454020411636, 960.23885026652079}};
  const auto descriptor = MakeDescriptor(CreateExpr(256), CreateExpr(4), CreateExpr(2));
  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.dtype);
    NddmaModelResult low_result;
    NddmaModelResult high_result;
    ASSERT_EQ(EvaluateNddmaModel(descriptor, test_case.dtype, CreateExpr(2), low_result), af::SUCCESS);
    ASSERT_EQ(EvaluateNddmaModel(descriptor, test_case.dtype, CreateExpr(8), high_result), af::SUCCESS);
    EXPECT_NEAR(GetNddmaConstCycles(low_result.cycles), test_case.low_core, 1e-5);
    EXPECT_NEAR(GetNddmaConstCycles(high_result.cycles), test_case.high_core, 1e-5);
  }
}

TEST(NddmaModelV2, BuildsDynamicCoreTernary) {
  const auto descriptor = MakeDescriptor(CreateExpr("n"), CreateExpr("input_stride"), CreateExpr("output_stride"));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr("block_dim"), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  ASSERT_EQ(result.ternary_ops.size(), 1U);
  EXPECT_NE(result.ternary_ops.begin()->second.GetTernaryOpStr().find("block_dim"), std::string::npos);
  EXPECT_NE(Str(result.cycles).find("nddma_1d_multicore"), std::string::npos);
}

TEST(NddmaModelV2, RejectsMismatchedDescriptorSchema) {
  auto descriptor = MakeDescriptor(CreateExpr(16), CreateExpr(1), CreateExpr(1));
  descriptor.output_strides.clear();
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int32", CreateExpr(1), result), af::SUCCESS);
  EXPECT_FALSE(result.selected);
  EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kSchemaMismatch);
  EXPECT_STREQ(NddmaFallbackReasonToString(result.fallback_reason), "schema_mismatch");
}

TEST(NddmaModelV2, EvaluatesRegisteredRanks2To5) {
  for (size_t rank = 2U; rank <= 5U; ++rank) {
    SCOPED_TRACE(rank);
    NddmaDescriptorInfo descriptor;
    descriptor.output_dims.assign(rank, CreateExpr(8));
    descriptor.input_strides.assign(rank, CreateExpr(1));
    descriptor.output_strides.assign(rank, CreateExpr(1));
    for (size_t axis = 0U; axis < rank; ++axis) {
      descriptor.vectorized_axis.emplace_back(static_cast<int64_t>(axis));
    }
    NddmaModelResult result;

    ASSERT_EQ(EvaluateNddmaModel(descriptor, "int64", CreateExpr(1), result), af::SUCCESS);
    EXPECT_TRUE(result.selected);
    EXPECT_EQ(result.raw_rank, 0U);
    EXPECT_EQ(result.effective_rank, rank);
    EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kNone);
    EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_V1");
  }
}

TEST(NddmaModelV2, UsesInnerToOuterFormulaOrderForTwoDimensions) {
  NddmaDescriptorInfo descriptor;
  // Descriptor is effective outer -> inner; the formula must consume [inner, outer].
  descriptor.output_dims = {CreateExpr(3), CreateExpr(8)};
  descriptor.input_strides = {CreateExpr(8), CreateExpr(1)};
  descriptor.output_strides = {CreateExpr(8), CreateExpr(1)};
  descriptor.vectorized_axis = {1, 0};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(2), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_EQ(result.effective_rank, 2U);
  EXPECT_EQ(result.model_name, "NDDMA_1D_MULTICORE_V2");
  EXPECT_TRUE(result.cycles.IsValid());
}

TEST(NddmaModelV2, UsesNg2ForAscendingInputStridesAndUbContiguousOutput) {
  NddmaDescriptorInfo descriptor;
  // Descriptor is effective outer -> inner; normalized order is [N, M]/[is1, is2]/[1, N].
  descriptor.output_dims = {CreateExpr(63), CreateExpr(2)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(2)};
  descriptor.output_strides = {CreateExpr(2), CreateExpr(1)};
  descriptor.vectorized_axis = {1, 0};
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int8", CreateExpr(2), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_NG2");
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 227.9001452458901, 1e-6);
}

TEST(NddmaModelV2, ReproducesExpectedTransposeEstimate) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(128), CreateExpr(128)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(1024)};
  descriptor.output_strides = {CreateExpr(128), CreateExpr(1)};
  descriptor.vectorized_axis = {1, 0};
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(64), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_NG2_V2");
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 4705.5751951840657, 1e-3);
}

TEST(NddmaModelV2, SelectsNg2ForSymbolicTilingDimensions) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr("m"), CreateExpr("n")};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(1024)};
  descriptor.output_strides = {CreateExpr("n"), CreateExpr(1)};
  descriptor.vectorized_axis = {1, 0};
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(64), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_NG2_V2");
  EXPECT_TRUE(result.cycles.IsValid());
}

TEST(NddmaModelV2, SelectsNg2ForAlignedSymbolicOutputStride) {
  const Expr m = CreateExpr("m");
  const Expr n = CreateExpr("n");
  const Expr aligned_n = CreateExpr(16) * af::sym::Ceiling(n / CreateExpr(16));
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {m, n};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(1024)};
  descriptor.output_strides = {aligned_n, CreateExpr(1)};
  descriptor.vectorized_axis = {1, 0};
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(64), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_EQ(result.model_name,
            "NDDMA_RUNTIME_DISPATCH[NDDMA_ND_MULTICORE_NG2_V2|"
            "NDDMA_RUNTIME_DISPATCH[NDDMA_ND_MULTICORE_NG2|NDDMA_ND_MULTICORE_V1]]");
  EXPECT_TRUE(result.cycles.IsValid());
}

TEST(NddmaModelV2, SelectsNg2WithUpdatedStaticMultiplier) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(2), CreateExpr(2)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(4)};
  descriptor.output_strides = {CreateExpr(2), CreateExpr(1)};
  descriptor.vectorized_axis = {1, 0};
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int64", CreateExpr(64), result), af::SUCCESS);
  EXPECT_TRUE(result.selected);
  EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kNone);
  EXPECT_NEAR(GetNddmaConstCycles(result.cycles), 569.216677989927, 1e-6);
}

TEST(NddmaModelV2, UsesNg2WhenSingletonAxesArePresent) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(63), CreateExpr(1), CreateExpr(2)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(1), CreateExpr(2)};
  descriptor.output_strides = {CreateExpr(2), CreateExpr(1), CreateExpr(1)};
  descriptor.vectorized_axis = {1, 2, 0};
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int8", CreateExpr(2), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_NG2");
}

TEST(NddmaModelV2, KeepsUnifiedModelWhenInputStridesAreNotAscending) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(63), CreateExpr(2)};
  descriptor.input_strides = {CreateExpr(2), CreateExpr(1)};
  descriptor.output_strides = {CreateExpr(2), CreateExpr(1)};
  descriptor.vectorized_axis = {1, 0};
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int8", CreateExpr(2), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_EQ(result.model_name, "NDDMA_1D_MULTICORE_V2");
}

TEST(NddmaModelV2, RejectsNonPositiveStaticOutputStride) {
  const auto descriptor = MakeDescriptor(CreateExpr(16), CreateExpr(1), CreateExpr(0));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int8", CreateExpr(1), result), af::SUCCESS);
  EXPECT_FALSE(result.selected);
  EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kStrideInvalid);
}

TEST(NddmaModelV2, RejectsRankOutsideSupportedRange) {
  for (const size_t rank : {0U, 6U, 7U}) {
    SCOPED_TRACE(rank);
    NddmaDescriptorInfo descriptor;
    descriptor.output_dims.assign(rank, CreateExpr(8));
    descriptor.input_strides.assign(rank, CreateExpr(1));
    descriptor.output_strides.assign(rank, CreateExpr(1));
    descriptor.vectorized_axis.resize(rank);
    std::iota(descriptor.vectorized_axis.begin(), descriptor.vectorized_axis.end(), 0);
    NddmaModelResult result;

    ASSERT_EQ(EvaluateNddmaModel(descriptor, "int8", CreateExpr(1), result), af::SUCCESS);
    EXPECT_FALSE(result.selected);
    EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kRankUnsupported);
  }
}

TEST(NddmaModelV2, SharedDataCopyAxisRulesMatchCodegenConditions) {
  const std::vector<Expr> ub_strides = {CreateExpr(15), CreateExpr(5), CreateExpr(1)};
  EXPECT_TRUE(ascgen_utils::ShouldIgnoreDataCopyZeroAxis(false, 0U, ub_strides));
  EXPECT_TRUE(ascgen_utils::ShouldIgnoreDataCopyZeroAxis(true, 2U, ub_strides));
  EXPECT_FALSE(ascgen_utils::ShouldIgnoreDataCopyZeroAxis(false, 2U, {CreateExpr(15), CreateExpr(5), CreateExpr(3)}));

  EXPECT_TRUE(ascgen_utils::IsDataCopyAxisContinuous(CreateExpr(5), CreateExpr(1), CreateExpr(5), CreateExpr(1)));
  EXPECT_FALSE(ascgen_utils::IsDataCopyAxisContinuous(CreateExpr(4), CreateExpr(1), CreateExpr(5), CreateExpr(1)));
}

TEST(NddmaModelV2, RejectsUnsupportedDtype) {
  const auto descriptor = MakeDescriptor(CreateExpr(16), CreateExpr(1), CreateExpr(1));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "unsupported_dtype", CreateExpr(1), result), af::SUCCESS);
  EXPECT_FALSE(result.selected);
  EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kDtypeUnsupported);
}

TEST(NddmaModelV2, RejectsZeroDtypeSizeFromUnsupportedMapping) {
  const auto descriptor = MakeDescriptor(CreateExpr(16), CreateExpr(1), CreateExpr(1));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "", CreateExpr(1), result), af::SUCCESS);
  EXPECT_FALSE(result.selected);
  EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kDtypeUnsupported);
}

TEST(NddmaModelV2, RejectsNegativeStaticInputStride) {
  const auto descriptor = MakeDescriptor(CreateExpr(16), CreateExpr(-1), CreateExpr(1));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int8", CreateExpr(1), result), af::SUCCESS);
  EXPECT_FALSE(result.selected);
  EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kStrideInvalid);
}

TEST(NddmaModelV2, RejectsNonPositiveStaticDimension) {
  const auto descriptor = MakeDescriptor(CreateExpr(0), CreateExpr(1), CreateExpr(1));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int8", CreateExpr(1), result), af::SUCCESS);
  EXPECT_FALSE(result.selected);
  EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kStrideInvalid);
}

TEST(NddmaModelV2, RejectsNonPositiveStaticBlockDim) {
  const auto descriptor = MakeDescriptor(CreateExpr(16), CreateExpr(1), CreateExpr(1));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int8", CreateExpr(0), result), af::SUCCESS);
  EXPECT_FALSE(result.selected);
  EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kSchemaMismatch);
}

TEST(NddmaModelV2, RejectsStaticByteCountOverflow) {
  const auto descriptor = MakeDescriptor(CreateExpr(std::numeric_limits<int64_t>::max()), CreateExpr(1), CreateExpr(1));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int64", CreateExpr(1), result), af::SUCCESS);
  EXPECT_FALSE(result.selected);
  EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kSchemaMismatch);
}

TEST(NddmaModelV2, RejectsRepeatedVectorizedAxis) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(4), CreateExpr(8)};
  descriptor.input_strides = {CreateExpr(8), CreateExpr(1)};
  descriptor.output_strides = {CreateExpr(8), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 0};
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int8", CreateExpr(1), result), af::SUCCESS);
  EXPECT_FALSE(result.selected);
  EXPECT_EQ(result.fallback_reason, NddmaFallbackReason::kCodegenMismatch);
}

TEST(NddmaModelV2, BuildsDescriptorFromRawTensorShape) {
  TensorShapeInfo shape_info;
  shape_info.repeats = {CreateExpr(4), CreateExpr(64)};
  shape_info.gm_strides = {CreateExpr(128), CreateExpr(2)};
  shape_info.strides = {CreateExpr(64), CreateExpr(1)};
  NddmaDescriptorInfo descriptor;

  EXPECT_EQ(BuildNddmaDescriptor(shape_info, {3, 1}, descriptor), NddmaFallbackReason::kNone);
  EXPECT_EQ(descriptor.output_dims, (std::vector<Expr>{CreateExpr(256)}));
  EXPECT_EQ(descriptor.input_strides, (std::vector<Expr>{CreateExpr(2)}));
  EXPECT_EQ(descriptor.output_strides, (std::vector<Expr>{CreateExpr(1)}));
  EXPECT_EQ(descriptor.vectorized_axis, (std::vector<int64_t>{1}));
}

TEST(NddmaModelV2, RejectsMissingVectorizedAxisForCodegenParity) {
  TensorShapeInfo shape_info;
  shape_info.repeats = {CreateExpr(64)};
  shape_info.gm_strides = {CreateExpr(2)};
  shape_info.strides = {CreateExpr(1)};
  NddmaDescriptorInfo descriptor;

  EXPECT_EQ(BuildNddmaDescriptor(shape_info, {}, descriptor), NddmaFallbackReason::kCodegenMismatch);
}

TEST(NddmaModelV2, LogsOneStableFallbackSummary) {
  const auto descriptor = MakeDescriptor(CreateExpr(16), CreateExpr(1), CreateExpr(1));
  NddmaModelResult result;
  result.raw_rank = 2U;
  result.effective_rank = 2U;
  result.fallback_reason = NddmaFallbackReason::kNoRegisteredModel;
  auto *slog = ge::SlogStub::GetInstance();
  const int32_t old_level = slog->GetLevel();
  slog->SetLevel(DLOG_WARN);

  testing::internal::CaptureStdout();
  LogNddmaFallback("nddma", "float16", &descriptor, result);
  const std::string output = testing::internal::GetCapturedStdout();
  slog->SetLevel(old_level);

  const auto summary_pos = output.find("[ATT NDDMA] fallback:");
  ASSERT_NE(summary_pos, std::string::npos);
  EXPECT_EQ(output.find("[ATT NDDMA] fallback:", summary_pos + 1U), std::string::npos);
  EXPECT_NE(output.find("fallback_reason=no_registered_model"), std::string::npos);
  EXPECT_EQ(output.find("fallback detail"), std::string::npos);
}

TEST(NddmaModelV2, StaticUnitAxesPreservePhysical1DTransferCost) {
  for (const char *dtype : {"uint8", "float16", "float32", "int64"}) {
    for (const int core : {1, 32, 56}) {
      const auto reference = MakeDescriptor(CreateExpr(256), CreateExpr(1), CreateExpr(1));
      NddmaModelResult expected;
      ASSERT_EQ(EvaluateNddmaModel(reference, dtype, CreateExpr(core), expected), af::SUCCESS);
      ASSERT_TRUE(expected.selected);
      for (const int position : {0, 1, 2}) {
        NddmaDescriptorInfo descriptor;
        descriptor.output_dims.assign(3U, CreateExpr(1));
        descriptor.input_strides.assign(3U, CreateExpr(65536));
        descriptor.output_strides.assign(3U, CreateExpr(4096));
        descriptor.output_dims[position] = CreateExpr(256);
        descriptor.input_strides[position] = CreateExpr(1);
        descriptor.output_strides[position] = CreateExpr(1);
        descriptor.vectorized_axis = {0, 1, 2};
        NddmaModelResult actual;
        ASSERT_EQ(EvaluateNddmaModel(descriptor, dtype, CreateExpr(core), actual), af::SUCCESS);
        ASSERT_TRUE(actual.selected);
        EXPECT_NEAR(GetNddmaConstCycles(actual.cycles), GetNddmaConstCycles(expected.cycles), 1e-6);
      }
    }
  }
}

TEST(NddmaModelV2, RuntimeUnitAxisPreservesGenericCost) {
  const Expr q = CreateExpr("unit_axis_q");
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(16), q, CreateExpr(32)};
  descriptor.input_strides = {CreateExpr(64), CreateExpr(65536), CreateExpr(1)};
  descriptor.output_strides = {CreateExpr(64), CreateExpr(32), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1, 2};
  NddmaModelResult dynamic;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(32), dynamic), af::SUCCESS);
  ASSERT_TRUE(dynamic.selected);
  for (const int value : {1, 2}) {
    auto concrete = descriptor;
    concrete.output_dims[1] = CreateExpr(value);
    if (value == 1) {
      concrete.output_dims = {CreateExpr(16), CreateExpr(32)};
      concrete.input_strides = {CreateExpr(64), CreateExpr(1)};
      concrete.output_strides = {CreateExpr(64), CreateExpr(1)};
      concrete.vectorized_axis = {0, 1};
    }
    NddmaModelResult expected;
    ASSERT_EQ(EvaluateNddmaModel(concrete, "float32", CreateExpr(32), expected), af::SUCCESS);
    ASSERT_TRUE(expected.selected);
    EXPECT_NEAR(GetNddmaConstCycles(dynamic.cycles.Replace({{q, CreateExpr(value)}})),
                GetNddmaConstCycles(expected.cycles), 1e-6);
  }
}

TEST(NddmaModelV2, TinySparseTransfersRemainPositiveWithoutRuntimeClamp) {
  for (const char *dtype : {"uint8", "float16", "float32", "int64"}) {
    for (const int n : {2, 3, 4}) {
      for (const int stride : {1, 16, 128, 1024}) {
        for (const int core : {1, 56}) {
          NddmaModelResult result;
          ASSERT_EQ(EvaluateNddmaModel(MakeDescriptor(CreateExpr(n), CreateExpr(stride), CreateExpr(2)), dtype,
                                       CreateExpr(core), result),
                    af::SUCCESS);
          ASSERT_TRUE(result.selected);
          const double cycles = GetNddmaConstCycles(result.cycles);
          EXPECT_TRUE(std::isfinite(cycles));
          EXPECT_GT(cycles, 0.0);
        }
      }
    }
  }
}

TEST(NddmaModelV2, JointlyContiguousAxesMatchDirect1DCost) {
  for (const char *dtype : {"uint8", "float16", "float32", "int64"}) {
    for (const int core : {2, 56}) {
      NddmaModelResult direct;
      ASSERT_EQ(EvaluateNddmaModel(MakeDescriptor(CreateExpr(1024), CreateExpr(2), CreateExpr(3)), dtype,
                                   CreateExpr(core), direct),
                af::SUCCESS);
      NddmaDescriptorInfo descriptor;
      descriptor.output_dims = {CreateExpr(8), CreateExpr(1), CreateExpr(128)};
      descriptor.input_strides = {CreateExpr(256), CreateExpr(0), CreateExpr(2)};
      descriptor.output_strides = {CreateExpr(384), CreateExpr(65536), CreateExpr(3)};
      descriptor.vectorized_axis = {0, 1, 2};
      NddmaModelResult actual;
      ASSERT_EQ(EvaluateNddmaModel(descriptor, dtype, CreateExpr(core), actual), af::SUCCESS);
      ASSERT_TRUE(direct.selected);
      ASSERT_TRUE(actual.selected);
      EXPECT_NEAR(GetNddmaConstCycles(actual.cycles), GetNddmaConstCycles(direct.cycles), 1e-6);
      EXPECT_EQ(actual.model_name, "NDDMA_1D_MULTICORE_V2");
      EXPECT_EQ(actual.effective_rank, 3U);  // Cost normalization preserves the actual API identity.
    }
  }
}

}  // namespace att
