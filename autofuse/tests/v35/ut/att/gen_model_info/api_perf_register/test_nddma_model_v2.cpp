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

double GetConstCycles(Expr cycles) {
  cycles.Simplify();
  int64_t integer = 0;
  if (cycles.GetConstValue(integer)) {
    return static_cast<double>(integer);
  }
  double value = 0.0;
  EXPECT_TRUE(cycles.GetConstValue(value));
  return value;
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
  EXPECT_NEAR(GetConstCycles(result.cycles), 400.30727752648556, 1e-6);
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
  EXPECT_NEAR(GetConstCycles(result.cycles), 216.18489573733697, 1e-6);
}

TEST(NddmaModelV2, EvaluatesStaticB16HighCoreFormula) {
  const auto descriptor = MakeDescriptor(CreateExpr(256), CreateExpr(4), CreateExpr(2));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float16", CreateExpr(8), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_NEAR(GetConstCycles(result.cycles), 3604.1465027941949, 1e-6);
}

TEST(NddmaModelV2, EvaluatesStaticB32LowCoreFormula) {
  const auto descriptor = MakeDescriptor(CreateExpr(256), CreateExpr(4), CreateExpr(2));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(2), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_NEAR(GetConstCycles(result.cycles), 638.66344907846417, 1e-6);
}

TEST(NddmaModelV2, EvaluatesStaticB64HighCoreWithSaturatedInputStride) {
  const auto descriptor = MakeDescriptor(CreateExpr(256), CreateExpr(129), CreateExpr(1));
  NddmaModelResult result;

  ASSERT_EQ(EvaluateNddmaModel(descriptor, "int64", CreateExpr(8), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_NEAR(GetConstCycles(result.cycles), 1788.750466100214, 1e-6);
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
    EXPECT_NEAR(GetConstCycles(low_result.cycles), test_case.low_core, 1e-5);
    EXPECT_NEAR(GetConstCycles(high_result.cycles), test_case.high_core, 1e-5);
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
  EXPECT_NEAR(GetConstCycles(result.cycles), 227.9001452458901, 1e-6);
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
  EXPECT_NEAR(GetConstCycles(result.cycles), 4705.5751951840657, 1e-3);
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
  EXPECT_EQ(result.model_name, "NDDMA_RUNTIME_DISPATCH[NDDMA_ND_MULTICORE_NG2_V2|NDDMA_ND_MULTICORE_NG2]");
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
  EXPECT_NEAR(GetConstCycles(result.cycles), 569.216677989927, 1e-6);
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
        EXPECT_NEAR(GetConstCycles(actual.cycles), GetConstCycles(expected.cycles), 1e-6);
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
    EXPECT_NEAR(GetConstCycles(dynamic.cycles.Replace({{q, CreateExpr(value)}})), GetConstCycles(expected.cycles),
                1e-6);
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
          const double cycles = GetConstCycles(result.cycles);
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
      EXPECT_NEAR(GetConstCycles(actual.cycles), GetConstCycles(direct.cycles), 1e-6);
      EXPECT_EQ(actual.model_name, "NDDMA_1D_MULTICORE_V2");
      EXPECT_EQ(actual.effective_rank, 3U);  // Cost normalization preserves the actual API identity.
    }
  }
}

TEST(NddmaModelV2, CalibratedNg2ReplaysMeasuredWholeApiCost) {
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
  EXPECT_LE(std::abs(GetConstCycles(result.cycles) - kMeasuredCycles), kMeasuredCycles * 0.5);
}

TEST(NddmaModelV2, CalibratedNg2PreservesLegacyForNonUnitSourceStep) {
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

TEST(NddmaModelV2, CalibratedNg2HasOnlyCoreAndInnerByteConditions) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr("m"), CreateExpr("n")};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(4096)};
  descriptor.output_strides = {CreateExpr("n"), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr("block_dim"), result), af::SUCCESS);
  ASSERT_TRUE(result.selected);
  EXPECT_EQ(result.model_name, "NDDMA_RUNTIME_DISPATCH[NDDMA_ND_MULTICORE_NG2|NDDMA_ND_MULTICORE_NG2_V2]");
  EXPECT_EQ(result.ternary_ops.size(), 2U);
}

TEST(NddmaModelV2, SymbolicPaddedPitchGuardsPackedCalibrationAndPreservesPaddedCost) {
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
        EXPECT_NEAR(GetConstCycles(guard->GetChoiceB()->GetExpr().Replace(bindings)), kOriginalSymbolicPaddedCycles,
                    1e-6);
      }
    }
  }
  EXPECT_TRUE(found);
  NddmaModelResult low;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(2), low), af::SUCCESS);
  EXPECT_TRUE(low.selected);
  EXPECT_TRUE(low.ternary_ops.empty());  // The unmeasured low-core path is unchanged.
}

TEST(NddmaModelV2, FitsPackedThreeAxisB32AndKeepsOtherPaths) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(10), CreateExpr(64), CreateExpr(39)};
  descriptor.input_strides = {CreateExpr(2496), CreateExpr(1), CreateExpr(64)};
  descriptor.output_strides = {CreateExpr(2496), CreateExpr(39), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1, 2};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(52), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_PACKED3D_V1");
  EXPECT_NEAR(GetConstCycles(result.cycles), 7614.416348341308, 1e-6);
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

TEST(NddmaModelV2, SparseThreeAxisResidualKeepsBandwidthAndStartup) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(235), CreateExpr(32), CreateExpr(2)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(65536), CreateExpr(2097152)};
  descriptor.output_strides = {CreateExpr(64), CreateExpr(2), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1, 2};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(56), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_SPARSE3D_V2");
  const double baseline = 235.0 * 32 * 2 * 4 / 29.4096 + 453.859;
  EXPECT_NEAR(GetConstCycles(result.cycles), baseline + 0.017212331646550515 * (121055.7889984 - baseline), 1e-4);
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

TEST(NddmaModelV2, SparseThreeAxisSymbolicGuardRetainsOutOfDomainCost) {
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

TEST(NddmaModelV2, SparseFourAxisResidualKeepsOtherPitchAndLargerTiles) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(512), CreateExpr(2), CreateExpr(4), CreateExpr(2)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(512), CreateExpr(65536), CreateExpr(2097152)};
  descriptor.output_strides = {CreateExpr(16), CreateExpr(8), CreateExpr(2), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1, 2, 3};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(52), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_SPARSE4D_V2");
  EXPECT_NEAR(GetConstCycles(result.cycles), 2796.1584908378627, 1e-6);
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

TEST(NddmaModelV2, SparseFourAxisSymbolicGuardIncludesMeasuredPadding) {
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

TEST(NddmaModelV2, SparseFourAxisPaddingRetainsLongSourceRunsAndOtherUbLayouts) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(512), CreateExpr(3), CreateExpr(3), CreateExpr(2)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(512), CreateExpr(65536), CreateExpr(2097152)};
  descriptor.output_strides = {CreateExpr(24), CreateExpr(8), CreateExpr(2), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1, 2, 3};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(53), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_SPARSE4D_V2");
  EXPECT_NEAR(GetConstCycles(result.cycles), 4227.740075369205, 1e-6);
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

TEST(NddmaModelV2, FitsOnlyMeasuredFp32PaddedNg2Domain) {
  NddmaDescriptorInfo descriptor;
  descriptor.output_dims = {CreateExpr(19), CreateExpr(385)};
  descriptor.input_strides = {CreateExpr(1), CreateExpr(19)};
  descriptor.output_strides = {CreateExpr(392), CreateExpr(1)};
  descriptor.vectorized_axis = {0, 1};
  NddmaModelResult result;
  ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(54), result), af::SUCCESS);
  EXPECT_EQ(result.model_name, "NDDMA_ND_MULTICORE_NG2_PADDED_V1");
  EXPECT_NEAR(GetConstCycles(result.cycles), 3338.394714124599, 1e-6);

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

TEST(NddmaModelV2, SymbolicPaddedNg2GuardExcludesPackedAndOutOfDomainTiles) {
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
            "NDDMA_RUNTIME_DISPATCH[NDDMA_ND_MULTICORE_NG2_PADDED_V1|NDDMA_ND_MULTICORE_NG2]]");
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

TEST(NddmaModelV2, GuardsPackedThreeAxisCostForSymbolicOuterAndCoreCount) {
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

TEST(NddmaModelV2, FitsSmallInt64PackedAndPaddedTransposes) {
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
          EXPECT_DOUBLE_EQ(GetConstCycles(result.cycles), 506.0);
        }
      }
    }
  }
}

TEST(NddmaModelV2, KeepsSmallInt64UnmeasuredLayoutsAndDtypes) {
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

TEST(NddmaModelV2, GuardsSmallInt64SymbolicTileAndCoreCount) {
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
    EXPECT_DOUBLE_EQ(GetConstCycles(guard->GetChoiceA()->GetExpr()), 506.0);
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

TEST(NddmaModelV2, NarrowSourceUsesIndependentRequestCostAndPhysicalUnitAxis) {
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
    EXPECT_NEAR(GetConstCycles(result.cycles), 39245.68947748, 0.01);
  }
}

TEST(NddmaModelV2, NarrowSourceKeepsUnmeasuredDtypesAndDescriptorDomains) {
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

TEST(NddmaModelV2, NarrowSourcePublishesSymbolicDomainGuard) {
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
        EXPECT_NEAR(GetConstCycles(guard->GetChoiceA()->GetExpr().Replace(bindings)), 16661.30891215, 0.01);
      }
    }
  }
  EXPECT_TRUE(found);
}

TEST(NddmaModelV2, DirectStoreContentionIsSeparateFromIsolatedApiCost) {
  NddmaDescriptorInfo descriptor{
      {CreateExpr(8), CreateExpr(3960)}, {CreateExpr(1), CreateExpr(8)}, {CreateExpr(3960), CreateExpr(1)}, {0, 1}};
  for (const int cores : {8, 56, 64, 72}) {
    NddmaModelResult isolated;
    NddmaModelResult direct;
    ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(cores), isolated), af::SUCCESS);
    ASSERT_EQ(EvaluateNddmaModel(descriptor, "float32", CreateExpr(cores), direct, true), af::SUCCESS);
    ASSERT_TRUE(direct.selected);
    const double contention = 0.0038940645164791633 * std::max(0.0, 126720.0 * cores - 4456448.0);
    EXPECT_NEAR(GetConstCycles(direct.cycles) - GetConstCycles(isolated.cycles), contention, 1e-6);
    EXPECT_EQ(direct.model_name, "NDDMA_ND_MULTICORE_DIRECT_STORE_V1");
    EXPECT_EQ(isolated.model_name, "NDDMA_ND_MULTICORE_NG2_V2");
  }
}

TEST(NddmaModelV2, DirectStoreKeepsUncalibratedDtypesAndPhysicalLayouts) {
  const NddmaDescriptorInfo packed{
      {CreateExpr(8), CreateExpr(3960)}, {CreateExpr(1), CreateExpr(8)}, {CreateExpr(3960), CreateExpr(1)}, {0, 1}};
  const auto unchanged = [](const NddmaDescriptorInfo &descriptor, const std::string &dtype, int cores) {
    NddmaModelResult isolated;
    NddmaModelResult direct;
    ASSERT_EQ(EvaluateNddmaModel(descriptor, dtype, CreateExpr(cores), isolated), af::SUCCESS);
    ASSERT_EQ(EvaluateNddmaModel(descriptor, dtype, CreateExpr(cores), direct, true), af::SUCCESS);
    ASSERT_TRUE(isolated.selected);
    ASSERT_TRUE(direct.selected);
    EXPECT_NEAR(GetConstCycles(direct.cycles), GetConstCycles(isolated.cycles), 1e-6);
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

TEST(NddmaModelV2, DirectStoreSymbolicGuardChecksRuntimePackingAndPayload) {
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
      EXPECT_EQ(GetConstCycles(guard->GetCondLeft().Replace(bindings)) == 0, in_domain);
      if (in_domain) {
        const auto increment = guard->GetChoiceA()->GetExpr() - guard->GetChoiceB()->GetExpr();
        EXPECT_NEAR(GetConstCycles(increment.Replace(bindings)),
                    0.0038940645164791633 * std::max(0.0, 32.0 * width * 56 - 4456448.0), 1e-6);
      }
    }
  }
  EXPECT_TRUE(found);
}

}  // namespace att
