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
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "ascir_node_param/ascir_node_param.h"
#include "att/api_perf_register/perf_param_v2.h"
#include "base/att_const_values.h"
#include "v35/att/api_perf_register/ascendc_api_perf/broadcast_api_perf_v2.h"
#include "v35/att/api_perf_register/ascendc_api_perf/broadcast_last_axis_perf_v2.h"

namespace att {
namespace {
using ascendcapi_v2::LastAxisBranch;
using ascendcapi_v2::VfCostAccumulator;

NodeDetail MakeLastNode(const std::string &dtype, const std::vector<int64_t> &src, const std::vector<int64_t> &dst) {
  NodeDetail node;
  node.broadcast_node_params.valid = true;
  node.broadcast_node_params.const_rank = static_cast<int32_t>(src.size());
  for (size_t i = 0U; i < src.size(); ++i) {
    node.broadcast_node_params.src_shape.push_back({CreateExpr(src[i]), ascir_param::ParamExprRole::kSemantic});
    node.broadcast_node_params.dst_shape.push_back({CreateExpr(dst[i]), ascir_param::ParamExprRole::kSemantic});
    node.input_dims.push_back(CreateExpr(src[i]));
    node.output_dims.push_back(CreateExpr(dst[i]));
    node.repeats.push_back(CreateExpr(src[i]));
  }
  node.input_dtype = {dtype};
  node.output_dtype = {dtype};
  return node;
}

void AddExpected(const std::string &op, const std::string &dtype, int64_t count, VfCostAccumulator &acc) {
  ASSERT_EQ(ascendcapi_v2::AddVfInstructPerf(op, dtype, CreateExpr(count), 1U, acc), af::SUCCESS);
}

Expr ExpectedE2B(const std::string &dtype, int64_t count) {
  VfCostAccumulator acc;
  AddExpected(kUpdateMask, dtype, count, acc);
  AddExpected(kLoad, dtype, count, acc);
  AddExpected(kStore, dtype, count, acc);
  return ascendcapi_v2::GetVfCost(acc);
}

Expr ExpectedAligned(const std::string &dtype, int64_t loads, int64_t updates, int64_t stores) {
  VfCostAccumulator acc;
  AddExpected(kUpdateMask, dtype, updates, acc);
  AddExpected(kLoad, dtype, loads, acc);
  AddExpected(kStore, dtype, stores, acc);
  return ascendcapi_v2::GetVfCost(acc);
}

Expr ExpectedRankTwoLargerAligned() {
  return ExpectedAligned(kFloat16, 2, 4, 4);
}

Expr ExpectedRankThreeLargerAligned() {
  return ExpectedAligned(kFloat16, 6, 12, 12);
}

Expr ExpectedRankFourLargerAligned() {
  return ExpectedAligned(kFloat16, 24, 8, 24);
}

Expr ExpectedUnaligned(const std::string &dtype, int64_t loads, int64_t stores) {
  VfCostAccumulator acc;
  AddExpected(kLoad, dtype, loads, acc);
  AddExpected(kStore, dtype, stores, acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, dtype, acc.max_latency, acc.throughput, CreateExpr(1)),
            af::SUCCESS);
  return ascendcapi_v2::GetVfCost(acc);
}

Expr ExpectedDynamicLast(const std::string &dtype, int64_t outer, int64_t stores, int64_t helper_calls) {
  VfCostAccumulator acc;
  AddExpected(kLoad, dtype, outer, acc);
  AddExpected(kStore, dtype, stores, acc);
  std::string helper_dtype;
  EXPECT_EQ(ascendcapi_v2::GetEffectiveHelperDtype(dtype, helper_dtype), af::SUCCESS);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, helper_dtype, acc.max_latency, acc.throughput,
                                           CreateExpr(helper_calls)),
            af::SUCCESS);
  return ascendcapi_v2::GetVfCost(acc);
}

Expr ExpectedRankTwoGather(bool two) {
  VfCostAccumulator acc;
  AddExpected(kDuplicate, kInt16, two ? 2 : 1, acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kInt16, acc.max_latency, acc.throughput, CreateExpr(1)),
            af::SUCCESS);
  AddExpected(kDiv, kInt16, 1, acc);
  AddExpected(kStore, kInt16, 1, acc);
  if (two) {
    AddExpected(kMuls, kUInt16, 1, acc);
    AddExpected(kAdd, kUInt16, 1, acc);
    AddExpected(kAdds, kUInt16, 1, acc);
  } else {
    AddExpected(kUpdateMask, kFloat16, 1, acc);
  }
  AddExpected(kLoad, kUInt16, 1, acc);
  EXPECT_EQ(
      VfPerfUtils::AddVfInstructPerf(kPlaceholder, kUInt16, acc.max_latency, acc.throughput, CreateExpr(two ? 2 : 1)),
      af::SUCCESS);
  AddExpected(kStore, kFloat16, two ? 2 : 1, acc);
  if (two) {
    EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kFloat16, acc.max_latency, acc.throughput, CreateExpr(1)),
              af::SUCCESS);
  }
  return ascendcapi_v2::GetVfCost(acc);
}

Expr ExpectedGatherWrapper(size_t rank, int64_t arithmetic, int64_t gathers) {
  VfCostAccumulator acc;
  AddExpected(kDuplicate, kInt16, static_cast<int64_t>(rank * 2U), acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kInt16, acc.max_latency, acc.throughput, CreateExpr(1)),
            af::SUCCESS);
  AddExpected(kDiv, kInt16, static_cast<int64_t>(rank), acc);
  AddExpected(kMul, kInt16, static_cast<int64_t>(rank + 1U), acc);
  AddExpected(kSub, kInt16, static_cast<int64_t>(rank), acc);
  AddExpected(kMulAddDst, kInt16, static_cast<int64_t>(rank - 1U), acc);
  AddExpected(kStore, kInt16, 1, acc);
  AddExpected(kDuplicate, kInt16, 2, acc);
  AddExpected(kLoad, kInt16, 1, acc);
  AddExpected(kMuls, kUInt16, arithmetic, acc);
  AddExpected(kAdd, kUInt16, arithmetic, acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kUInt16, acc.max_latency, acc.throughput, CreateExpr(gathers)),
            af::SUCCESS);
  AddExpected(kStore, kFloat16, gathers, acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kFloat16, acc.max_latency, acc.throughput, CreateExpr(1)),
            af::SUCCESS);
  return ascendcapi_v2::GetVfCost(acc);
}

Expr ExpectedRankTwoB8GatherOne() {
  VfCostAccumulator acc;
  AddExpected(kDuplicate, kInt16, 2, acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kInt16, acc.max_latency, acc.throughput, CreateExpr(2)),
            af::SUCCESS);
  AddExpected(kDiv, kInt16, 2, acc);
  AddExpected(kStore, kInt16, 2, acc);
  AddExpected(kUpdateMask, kUInt8, 1, acc);
  AddExpected(kLoad, kUInt16, 2, acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kUInt16, acc.max_latency, acc.throughput, CreateExpr(2)),
            af::SUCCESS);
  AddExpected(kDeInterleave, kUInt8, 1, acc);
  AddExpected(kStore, kUInt8, 1, acc);
  return ascendcapi_v2::GetVfCost(acc);
}

Expr ExpectedRankThreeB8GatherWrapper() {
  VfCostAccumulator acc;
  AddExpected(kDuplicate, kInt16, 6, acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kInt16, acc.max_latency, acc.throughput, CreateExpr(2)),
            af::SUCCESS);
  AddExpected(kDiv, kInt16, 6, acc);
  AddExpected(kMul, kInt16, 8, acc);
  AddExpected(kSub, kInt16, 6, acc);
  AddExpected(kMulAddDst, kInt16, 4, acc);
  AddExpected(kStore, kInt16, 2, acc);
  AddExpected(kDuplicate, kInt16, 2, acc);
  AddExpected(kLoad, kInt16, 2, acc);
  AddExpected(kMuls, kUInt16, 2, acc);
  AddExpected(kAdd, kUInt16, 4, acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kUInt16, acc.max_latency, acc.throughput, CreateExpr(2)),
            af::SUCCESS);
  AddExpected(kDeInterleave, kUInt8, 1, acc);
  AddExpected(kStore, kUInt8, 1, acc);
  EXPECT_EQ(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kUInt8, acc.max_latency, acc.throughput, CreateExpr(1)),
            af::SUCCESS);
  return ascendcapi_v2::GetVfCost(acc);
}

ascendcapi_v2::BroadcastTilingInfo BuildTiling(const NodeDetail &node) {
  ascendcapi_v2::ParamExprInputs src_inputs{node.input_dims, node.input_dims, node.repeats};
  ascendcapi_v2::ParamExprInputs dst_inputs{node.output_dims, node.output_dims, node.output_dims};
  ascendcapi_v2::BroadcastTilingInfo tiling;
  EXPECT_EQ(
      ascendcapi_v2::BuildBroadcastTiling(node.broadcast_node_params.src_shape, node.broadcast_node_params.dst_shape,
                                          node.input_dtype[0], src_inputs, dst_inputs, tiling),
      af::SUCCESS);
  return tiling;
}

Expr ActualCost(const NodeDetail &node) {
  PerfOutputInfo perf;
  EXPECT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
  return perf.pipe_res[PipeType::AIV_VEC];
}

Expr ActualLastCost(const NodeDetail &node, const ascendcapi_v2::BroadcastTilingInfo &tiling) {
  PerfOutputInfo perf;
  EXPECT_EQ(ascendcperf_v2::BuildLastAxisPerf(node, tiling, perf), af::SUCCESS);
  return perf.pipe_res[PipeType::AIV_VEC];
}

struct LastBranchCase {
  std::string dtype;
  std::vector<int64_t> src;
  std::vector<int64_t> dst;
  LastAxisBranch branch;
};

struct FixedVectorSpec {
  int64_t vl;
  int64_t half_vl;
  int64_t block;
};

FixedVectorSpec GetFixedVectorSpec(const std::string &dtype) {
  if (dtype == kUInt8 || dtype == kInt8) {
    return {256, 128, 32};
  }
  if (dtype == kFloat32 || dtype == kUInt32 || dtype == kInt32) {
    return {64, 32, 8};
  }
  return {128, 64, 16};
}

int64_t Product(const std::vector<int64_t> &shape) {
  int64_t result = 1;
  for (const auto dim : shape) {
    result *= dim;
  }
  return result;
}

// Independent loop specification copied from the source helper conditions; it is not a production classifier oracle.
LastAxisBranch FixedLastAxisBranch(const std::string &dtype, const std::vector<int64_t> &src,
                                   const std::vector<int64_t> &dst, int32_t const_rank) {
  const auto spec = GetFixedVectorSpec(dtype);
  if (const_rank > 4) {
    return dst.back() <= spec.vl ? LastAxisBranch::kDynamicLessThanVlUnaligned
                                 : LastAxisBranch::kDynamicLargerThanVlUnaligned;
  }
  size_t rank = dst.size();
  size_t first_dim = 0U;
  while (rank > 1U && dst[first_dim] == 1) {
    ++first_dim;
    --rank;
  }
  const int64_t last = dst.back();
  const bool b8 = dtype == kUInt8 || dtype == kInt8;
  const bool aligned = last % spec.block == 0;
  const bool second_axis_contiguous = src.size() > 1U && src[src.size() - 2U] != 1;
  const int64_t dst_size = Product(dst);
  if (rank == 2U && last == spec.block && !b8) {
    return LastAxisBranch::kE2B;
  }
  if (rank == 3U && !b8 && second_axis_contiguous && last == spec.block && dst[dst.size() - 2U] * last > spec.half_vl &&
      dst[dst.size() - 2U] % (spec.vl / spec.block) == 0) {
    return dst[dst.size() - 2U] * last > spec.vl ? LastAxisBranch::kE2BLargerThanVl : LastAxisBranch::kE2BLessThanVl;
  }
  if (rank == 4U && !b8 && src.size() > 2U && src[src.size() - 2U] != 1 && last == spec.block &&
      dst[dst.size() - 2U] % (spec.vl / spec.block) == 0) {
    return LastAxisBranch::kE2B;
  }
  if (last < spec.half_vl && (rank == 2U || !b8)) {
    if (rank == 2U) {
      return dst_size < spec.vl ? LastAxisBranch::kGatherOne : LastAxisBranch::kGatherTwo;
    }
    return rank == 3U ? LastAxisBranch::kGatherWrapper : LastAxisBranch::kGatherWrapperForFourDim;
  }
  if (last <= spec.vl) {
    if (rank == 2U) {
      return LastAxisBranch::kLessThanVlUnaligned;
    }
    if (rank == 3U) {
      return aligned ? LastAxisBranch::kLessThanVlAligned : LastAxisBranch::kLessThanVlUnaligned;
    }
    if (aligned) {
      return LastAxisBranch::kLessThanVlAligned;
    }
    return const_rank == -1 ? LastAxisBranch::kFallback : LastAxisBranch::kLessThanVlUnaligned;
  }
  if (aligned) {
    return LastAxisBranch::kLargerThanVlAligned;
  }
  return const_rank == -1 ? LastAxisBranch::kFallback : LastAxisBranch::kLargerThanVlUnaligned;
}

TEST(BroadcastLastAxisPerfV2, RankTwoCoversImplementationLeavesAndBoundaries) {
  const std::vector<LastBranchCase> cases = {
      {kFloat16, {2, 1}, {2, 16}, LastAxisBranch::kE2B},
      {kFloat16, {2, 1}, {2, 63}, LastAxisBranch::kGatherOne},
      {kFloat16, {15, 1}, {15, 8}, LastAxisBranch::kGatherOne},
      {kFloat16, {16, 1}, {16, 8}, LastAxisBranch::kGatherTwo},
      {kFloat16, {17, 1}, {17, 8}, LastAxisBranch::kGatherTwo},
      {kFloat16, {8, 1}, {8, 16}, LastAxisBranch::kE2B},
      {kFloat16, {9, 1}, {9, 15}, LastAxisBranch::kGatherTwo},
      {kFloat16, {2, 1}, {2, 17}, LastAxisBranch::kGatherOne},
      {kFloat16, {2, 1}, {2, 64}, LastAxisBranch::kLessThanVlUnaligned},
      {kFloat16, {2, 1}, {2, 65}, LastAxisBranch::kLessThanVlUnaligned},
      {kFloat16, {2, 1}, {2, 127}, LastAxisBranch::kLessThanVlUnaligned},
      {kFloat16, {2, 1}, {2, 128}, LastAxisBranch::kLessThanVlUnaligned},
      {kFloat16, {2, 1}, {2, 144}, LastAxisBranch::kLargerThanVlAligned},
      {kFloat16, {2, 1}, {2, 129}, LastAxisBranch::kLargerThanVlUnaligned},
  };
  for (const auto &item : cases) {
    const auto node = MakeLastNode(item.dtype, item.src, item.dst);
    EXPECT_EQ(FixedLastAxisBranch(item.dtype, item.src, item.dst, static_cast<int32_t>(item.dst.size())), item.branch);
    EXPECT_EQ(ascendcperf_v2::GetLastAxisPerfBranch(node), item.branch);
  }
}

TEST(BroadcastLastAxisPerfV2, RankThreeCoversBothE2BPathsAndGatherWrapper) {
  const std::vector<LastBranchCase> cases = {
      {kFloat16, {2, 8, 1}, {2, 8, 16}, LastAxisBranch::kE2BLessThanVl},
      {kFloat16, {2, 16, 1}, {2, 16, 16}, LastAxisBranch::kE2BLargerThanVl},
      {kFloat16, {2, 3, 1}, {2, 3, 15}, LastAxisBranch::kGatherWrapper},
      {kFloat16, {2, 3, 1}, {2, 3, 64}, LastAxisBranch::kLessThanVlAligned},
      {kFloat16, {2, 3, 1}, {2, 3, 65}, LastAxisBranch::kLessThanVlUnaligned},
      {kFloat16, {2, 3, 1}, {2, 3, 144}, LastAxisBranch::kLargerThanVlAligned},
      {kFloat16, {2, 3, 1}, {2, 3, 129}, LastAxisBranch::kLargerThanVlUnaligned},
  };
  for (const auto &item : cases) {
    const auto node = MakeLastNode(item.dtype, item.src, item.dst);
    EXPECT_EQ(FixedLastAxisBranch(item.dtype, item.src, item.dst, static_cast<int32_t>(item.dst.size())), item.branch);
    EXPECT_EQ(ascendcperf_v2::GetLastAxisPerfBranch(node), item.branch);
  }
}

TEST(BroadcastLastAxisPerfV2, RankThreeDynamicRankKeepsAlignedStaticAndFallsBackOnUnaligned) {
  auto unaligned_node = MakeLastNode(kFloat16, {2, 3, 1}, {2, 3, 65});
  unaligned_node.broadcast_node_params.const_rank = -1;
  EXPECT_EQ(ascendcperf_v2::GetLastAxisPerfBranch(unaligned_node), LastAxisBranch::kDynamicLessThanVlUnaligned);

  auto aligned_node = MakeLastNode(kFloat16, {2, 3, 1}, {2, 3, 64});
  aligned_node.broadcast_node_params.const_rank = -1;
  EXPECT_EQ(ascendcperf_v2::GetLastAxisPerfBranch(aligned_node), LastAxisBranch::kLessThanVlAligned);
}

TEST(BroadcastLastAxisPerfV2, CalibratesFp32AlignedLastAxisRowsWithinTheMeasuredShapeRange) {
  const auto cost = [](const std::string &dtype, int64_t outer, int64_t rows, int64_t width) {
    const auto node = MakeLastNode(dtype, {1, rows, 1}, {outer, rows, width});
    PerfOutputInfo perf;
    EXPECT_EQ(ascendcperf_v2::BroadcastPerf(node, perf), af::SUCCESS);
    const auto &vector_cycles = perf.pipe_res.at(PipeType::AIV_VEC);
    int64_t integer_cycles = 0;
    if (vector_cycles.GetConstValue(integer_cycles)) {
      return static_cast<double>(integer_cycles);
    }
    double cycles = 0.0;
    EXPECT_TRUE(vector_cycles.GetConstValue(cycles));
    return cycles;
  };
  EXPECT_NEAR(cost(kFloat32, 1, 128, 32), 785.98778, 0.01);
  EXPECT_NEAR(cost(kFloat32, 1, 96, 40), 594.08583, 0.01);
  EXPECT_GT(std::abs(cost(kFloat32, 1, 128, 24) - 785.98778), 1.0);
  EXPECT_GT(std::abs(cost(kFloat16, 1, 128, 32) - 785.98778), 1.0);
  EXPECT_GT(std::abs(cost(kFloat32, 2, 128, 32) - 785.98778), 1.0);
}

TEST(BroadcastLastAxisPerfV2, RankFourFoldedToThreeDimAlignedUsesAlignedBranch) {
  const auto node = MakeLastNode(kFloat16, {1, 1, 2, 1}, {1, 1, 2, 64});
  EXPECT_EQ(ascendcperf_v2::GetLastAxisPerfBranch(node), LastAxisBranch::kLessThanVlAligned);
}

TEST(BroadcastLastAxisPerfV2, RankFourCoversWrapperLeavesAndReductionToThreeDim) {
  const std::vector<LastBranchCase> cases = {
      {kFloat16, {2, 3, 8, 1}, {2, 3, 8, 16}, LastAxisBranch::kE2B},
      {kFloat16, {2, 3, 2, 1}, {2, 3, 2, 15}, LastAxisBranch::kGatherWrapperForFourDim},
      {kFloat16, {2, 3, 2, 1}, {2, 3, 2, 64}, LastAxisBranch::kLessThanVlAligned},
      {kFloat16, {2, 3, 2, 1}, {2, 3, 2, 65}, LastAxisBranch::kLessThanVlUnaligned},
      {kFloat16, {2, 3, 2, 1}, {2, 3, 2, 144}, LastAxisBranch::kLargerThanVlAligned},
      {kFloat16, {2, 3, 2, 1}, {2, 3, 2, 129}, LastAxisBranch::kLargerThanVlUnaligned},
      {kFloat16, {1, 2, 3, 1}, {1, 2, 3, 65}, LastAxisBranch::kLessThanVlUnaligned},
  };
  for (const auto &item : cases) {
    const auto node = MakeLastNode(item.dtype, item.src, item.dst);
    EXPECT_EQ(FixedLastAxisBranch(item.dtype, item.src, item.dst, static_cast<int32_t>(item.dst.size())), item.branch);
    EXPECT_EQ(ascendcperf_v2::GetLastAxisPerfBranch(node), item.branch);
  }
}

TEST(BroadcastLastAxisPerfV2, LeafHelpersUseExactE2BAndAlignedCounts) {
  EXPECT_EQ(ActualCost(MakeLastNode(kFloat16, {2, 8, 1}, {2, 8, 16})), ExpectedE2B(kFloat16, 2));
  EXPECT_EQ(ActualCost(MakeLastNode(kFloat16, {2, 16, 1}, {2, 16, 16})), ExpectedE2B(kFloat16, 4));
  EXPECT_EQ(ActualCost(MakeLastNode(kFloat16, {2, 3, 8, 1}, {2, 3, 8, 16})), ExpectedE2B(kFloat16, 6));
  EXPECT_EQ(ActualCost(MakeLastNode(kFloat16, {2, 3, 2, 1}, {2, 3, 2, 144})), ExpectedRankFourLargerAligned());
}

TEST(BroadcastLastAxisPerfV2, LeafHelpersUseExactUnalignedCounts) {
  EXPECT_EQ(ActualCost(MakeLastNode(kFloat16, {2, 1}, {2, 65})), ExpectedUnaligned(kFloat16, 2, 2));
  EXPECT_EQ(ActualCost(MakeLastNode(kFloat16, {2, 3, 1}, {2, 3, 65})), ExpectedUnaligned(kFloat16, 6, 6));
  EXPECT_EQ(ActualCost(MakeLastNode(kFloat16, {2, 3, 2, 1}, {2, 3, 2, 65})), ExpectedUnaligned(kFloat16, 12, 12));
  EXPECT_EQ(ActualCost(MakeLastNode(kFloat16, {1, 2, 3, 1}, {1, 2, 3, 65})), ExpectedUnaligned(kFloat16, 6, 6));
  EXPECT_EQ(ActualCost(MakeLastNode(kFloat16, {2, 3, 1}, {2, 3, 129})), ExpectedUnaligned(kFloat16, 6, 12));
  EXPECT_EQ(ActualCost(MakeLastNode(kFloat16, {2, 3, 2, 1}, {2, 3, 2, 129})), ExpectedUnaligned(kFloat16, 24, 24));
}

TEST(BroadcastLastAxisPerfV2, RankTwoGatherAndAlignedHelpersUseExactCounts) {
  EXPECT_EQ(ActualCost(MakeLastNode(kFloat16, {2, 1}, {2, 15})), ExpectedRankTwoGather(false));
  EXPECT_EQ(ActualCost(MakeLastNode(kFloat16, {9, 1}, {9, 15})), ExpectedRankTwoGather(true));
  EXPECT_EQ(ActualCost(MakeLastNode(kFloat16, {2, 1}, {2, 144})), ExpectedRankTwoLargerAligned());
  EXPECT_EQ(ActualCost(MakeLastNode(kUInt8, {2, 1}, {2, 63})), ExpectedRankTwoB8GatherOne());
}

TEST(BroadcastLastAxisPerfV2, RankThreeAndFourHelpersUseExactCounts) {
  EXPECT_EQ(ActualCost(MakeLastNode(kFloat16, {2, 3, 1}, {2, 3, 15})), ExpectedGatherWrapper(3U, 3, 2));
  EXPECT_EQ(ActualCost(MakeLastNode(kFloat16, {2, 3, 2, 1}, {2, 3, 2, 15})), ExpectedGatherWrapper(4U, 5, 3));
  EXPECT_EQ(ActualCost(MakeLastNode(kFloat16, {2, 3, 1}, {2, 3, 64})), ExpectedAligned(kFloat16, 6, 1, 6));
  EXPECT_EQ(ActualCost(MakeLastNode(kFloat16, {2, 3, 1}, {2, 3, 144})), ExpectedRankThreeLargerAligned());
  EXPECT_EQ(ActualCost(MakeLastNode(kFloat16, {2, 3, 2, 1}, {2, 3, 2, 64})), ExpectedAligned(kFloat16, 12, 1, 12));
  EXPECT_EQ(ActualCost(MakeLastNode(kUInt8, {2, 3, 1}, {2, 3, 15})), ExpectedRankThreeB8GatherWrapper());
}

TEST(BroadcastLastAxisPerfV2, RankTwoThreeFourLargerAlignedUseDifferentLoadCounts) {
  EXPECT_EQ(ActualCost(MakeLastNode(kFloat16, {2, 1}, {2, 144})), ExpectedRankTwoLargerAligned());
  EXPECT_EQ(ActualCost(MakeLastNode(kFloat16, {2, 3, 1}, {2, 3, 144})), ExpectedRankThreeLargerAligned());
  EXPECT_EQ(ActualCost(MakeLastNode(kFloat16, {2, 3, 2, 1}, {2, 3, 2, 144})), ExpectedRankFourLargerAligned());
}

TEST(BroadcastLastAxisPerfV2, DynamicRanksFiveThroughNineUseOriginalPrefixForHelperCalls) {
  const std::vector<std::vector<int64_t>> dst_shapes = {
      {2, 3, 2, 3, 65},          {2, 3, 2, 3, 2, 65},          {2, 3, 2, 3, 2, 3, 65},
      {2, 3, 2, 3, 2, 3, 2, 65}, {2, 3, 2, 3, 2, 3, 2, 3, 65},
  };
  for (const auto &dst : dst_shapes) {
    auto src = dst;
    for (size_t i = 0U; i < src.size(); ++i) {
      if (i % 2U == (src.size() - 1U) % 2U) {
        src[i] = 1;
      }
    }
    const auto node = MakeLastNode(kFloat16, src, dst);
    EXPECT_EQ(ascendcperf_v2::GetLastAxisPerfBranch(node), LastAxisBranch::kDynamicLessThanVlUnaligned);
    int64_t outer = 1;
    for (size_t i = 0U; i + 1U < dst.size(); ++i) {
      outer *= dst[i];
    }
    const int64_t helper_calls = [&dst]() {
      int64_t result = 1;
      for (size_t i = 0U; i + 4U < dst.size(); ++i) {
        result *= dst[i];
      }
      return result;
    }();
    EXPECT_EQ(ActualCost(node), ExpectedDynamicLast(kFloat16, outer, outer, helper_calls));
  }
}

TEST(BroadcastLastAxisPerfV2, DynamicOriginalRankFourUsesOneHelperCall) {
  const auto node = MakeLastNode(kUInt64, {2, 3, 1, 1}, {2, 3, 2, 4});
  const auto tiling = BuildTiling(node);
  EXPECT_EQ(ascendcapi_v2::GetLastAxisBranch(tiling, kUInt64, 4), LastAxisBranch::kDynamicLessThanVlUnaligned);
  EXPECT_EQ(ActualLastCost(node, tiling), ExpectedDynamicLast(kUInt64, 48, 48, 1));
}

TEST(BroadcastLastAxisPerfV2, DynamicB8CoversBothVectorSizePaths) {
  auto less = MakeLastNode(kUInt8, {1, 3, 1, 3, 1}, {2, 3, 2, 3, 256});
  auto larger = MakeLastNode(kUInt8, {1, 3, 1, 3, 1}, {2, 3, 2, 3, 257});
  EXPECT_EQ(ascendcperf_v2::GetLastAxisPerfBranch(less), LastAxisBranch::kDynamicLessThanVlUnaligned);
  EXPECT_EQ(ascendcperf_v2::GetLastAxisPerfBranch(larger), LastAxisBranch::kDynamicLargerThanVlUnaligned);
}

TEST(BroadcastLastAxisPerfV2, DynamicLargerHelperUsesExactLoopCounts) {
  const auto node = MakeLastNode(kFloat16, {1, 3, 1, 3, 1}, {2, 3, 2, 3, 129});
  EXPECT_EQ(ActualCost(node), ExpectedDynamicLast(kFloat16, 36, 72, 2));
}

TEST(BroadcastLastAxisPerfV2, B64LastAxisBecomesEffectiveB32AndAppendsDimension) {
  const auto tiling = BuildTiling(MakeLastNode(kUInt64, {2, 1}, {2, 4}));
  std::string helper_dtype;
  ASSERT_EQ(ascendcapi_v2::GetEffectiveHelperDtype(kUInt64, helper_dtype), af::SUCCESS);
  EXPECT_EQ(helper_dtype, kUInt32);
  ASSERT_EQ(tiling.rank, 3U);
  EXPECT_EQ(tiling.dst_shape[2], CreateExpr(2));
  EXPECT_FALSE(ascendcapi_v2::IsLastAxisBroadcast(tiling));
}

TEST(BroadcastLastAxisPerfV2, B64RankNineUsesLoopNumAndStrideNine) {
  const auto zero_stride = BuildTiling(MakeLastNode(kUInt64, {1, 3, 1, 3, 1, 3, 1, 3, 1}, {2, 3, 2, 3, 2, 3, 2, 3, 4}));
  // ApplyB64Tiling uses the original dstShape[0] as loop_num and appends its stride at index 9.
  EXPECT_EQ(zero_stride.loop_num, CreateExpr(2));
  ASSERT_EQ(zero_stride.src_stride.size(), 10U);
  EXPECT_EQ(zero_stride.src_stride[9], CreateExpr(0));
}

}  // namespace
}  // namespace att
