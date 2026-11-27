/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "broadcast_last_axis_perf_v2.h"

#include "base/att_const_values.h"
#include "common/checker.h"

namespace att {
namespace ascendcapi_v2 {
namespace {

bool IsTrue(const Expr &lhs, CondType condition, const Expr &rhs) {
  af::TriBool result = af::TriBool::kUnknown;
  if (condition == CondType::K_EQ) {
    result = af::SymbolicUtils::StaticCheckEq(lhs, rhs);
  }
  if (condition == CondType::K_LT) {
    result = af::SymbolicUtils::StaticCheckLt(lhs, rhs);
  }
  if (condition == CondType::K_LE) {
    result = af::SymbolicUtils::StaticCheckLe(lhs, rhs);
  }
  if (condition == CondType::K_GT) {
    result = af::SymbolicUtils::StaticCheckGt(lhs, rhs);
  }
  return result == af::TriBool::kTrue;
}

bool IsB8(const std::string &dtype) {
  return dtype == kUInt8 || dtype == kInt8;
}

bool IsZero(const Expr &value) {
  return af::SymbolicUtils::StaticCheckEq(value, CreateExpr(0)) == af::TriBool::kTrue;
}

LastAxisBranch ClassifyRankTwo(const BroadcastTilingInfo &tiling, const std::string &dtype, const Expr &vl,
                               const Expr &half_vl, const Expr &block, int32_t const_rank) {
  const Expr last = tiling.dst_shape[1U];
  if (IsTrue(last, CondType::K_EQ, block) && !IsB8(dtype)) {
    return LastAxisBranch::kE2B;
  }
  if (IsTrue(last, CondType::K_LT, half_vl)) {
    return IsTrue(tiling.dst_size, CondType::K_LT, vl) ? LastAxisBranch::kGatherOne : LastAxisBranch::kGatherTwo;
  }
  if (IsTrue(last, CondType::K_LE, vl)) {
    return LastAxisBranch::kLessThanVlUnaligned;
  }
  if (IsTrue(af::sym::Mod(last, block), CondType::K_EQ, CreateExpr(0))) {
    return LastAxisBranch::kLargerThanVlAligned;
  }
  return const_rank == -1 ? LastAxisBranch::kDynamicLargerThanVlUnaligned : LastAxisBranch::kLargerThanVlUnaligned;
}

LastAxisBranch ClassifyRankThree(const BroadcastTilingInfo &tiling, const std::string &dtype, const Expr &vl,
                                 const Expr &half_vl, const Expr &block, int32_t const_rank) {
  const Expr last = tiling.dst_shape[2U];
  if (IsZero(block)) {
    return LastAxisBranch::kFallback;
  }
  const Expr default_block_num = vl / block;
  if (!IsB8(dtype) && tiling.src_stride.size() > 1U && !IsZero(tiling.src_stride[1]) &&
      IsTrue(last, CondType::K_EQ, block) && IsTrue(tiling.dst_shape[1] * last, CondType::K_GT, half_vl) &&
      IsTrue(af::sym::Mod(tiling.dst_shape[1], default_block_num), CondType::K_EQ, CreateExpr(0))) {
    return IsTrue(tiling.dst_shape[1] * last, CondType::K_GT, vl) ? LastAxisBranch::kE2BLargerThanVl
                                                                  : LastAxisBranch::kE2BLessThanVl;
  }
  if (IsTrue(last, CondType::K_LT, half_vl)) {
    return LastAxisBranch::kGatherWrapper;
  }
  if (IsTrue(last, CondType::K_LE, vl)) {
    if (IsTrue(af::sym::Mod(last, block), CondType::K_EQ, CreateExpr(0))) {
      return LastAxisBranch::kLessThanVlAligned;
    }
    return const_rank == -1 ? LastAxisBranch::kDynamicLessThanVlUnaligned : LastAxisBranch::kLessThanVlUnaligned;
  }
  if (IsTrue(af::sym::Mod(last, block), CondType::K_EQ, CreateExpr(0))) {
    return LastAxisBranch::kLargerThanVlAligned;
  }
  return const_rank == -1 ? LastAxisBranch::kDynamicLargerThanVlUnaligned : LastAxisBranch::kLargerThanVlUnaligned;
}

LastAxisBranch ClassifyRankFour(const BroadcastTilingInfo &tiling, const std::string &dtype, const Expr &vl,
                                const Expr &half_vl, const Expr &block, int32_t const_rank) {
  const Expr last = tiling.dst_shape[3U];
  if (IsZero(block)) {
    return LastAxisBranch::kFallback;
  }
  const Expr default_block_num = vl / block;
  if (!IsB8(dtype) && tiling.src_stride.size() > 2U && !IsZero(tiling.src_stride[2]) &&
      IsTrue(last, CondType::K_EQ, block) &&
      IsTrue(af::sym::Mod(tiling.dst_shape[2], default_block_num), CondType::K_EQ, CreateExpr(0))) {
    return LastAxisBranch::kE2B;
  }
  if (IsTrue(last, CondType::K_LT, half_vl) && !IsB8(dtype)) {
    return LastAxisBranch::kGatherWrapperForFourDim;
  }
  if (IsTrue(last, CondType::K_LE, vl)) {
    if (IsTrue(af::sym::Mod(last, block), CondType::K_EQ, CreateExpr(0))) {
      return LastAxisBranch::kLessThanVlAligned;
    }
    return const_rank == -1 ? LastAxisBranch::kDynamicLessThanVlUnaligned : LastAxisBranch::kLessThanVlUnaligned;
  }
  if (IsTrue(af::sym::Mod(last, block), CondType::K_EQ, CreateExpr(0))) {
    return LastAxisBranch::kLargerThanVlAligned;
  }
  return const_rank == -1 ? LastAxisBranch::kDynamicLargerThanVlUnaligned : LastAxisBranch::kLargerThanVlUnaligned;
}

LastAxisBranch ClassifySmallRank(const BroadcastTilingInfo &tiling, const std::string &dtype, int32_t const_rank) {
  Expr vl;
  Expr half_vl;
  Expr block;
  if (GetBroadcastTilingVectorElements(dtype, vl, half_vl, block) != af::SUCCESS) {
    return LastAxisBranch::kFallback;
  }
  if (tiling.rank == 1U) {
    return LastAxisBranch::kFallback;
  }
  if (tiling.rank == 2U) {
    return ClassifyRankTwo(tiling, dtype, vl, half_vl, block, const_rank);
  }
  if (tiling.rank == 3U) {
    return ClassifyRankThree(tiling, dtype, vl, half_vl, block, const_rank);
  }
  return ClassifyRankFour(tiling, dtype, vl, half_vl, block, const_rank);
}

}  // namespace

LastAxisBranch GetLastAxisBranch(const BroadcastTilingInfo &tiling, const std::string &dtype, int32_t const_rank) {
  if (tiling.rank == 0U || tiling.dst_shape.size() < tiling.rank) {
    return LastAxisBranch::kFallback;
  }
  if (tiling.rank > 4U) {
    Expr vl;
    Expr half_vl;
    Expr block;
    if (GetBroadcastTilingVectorElements(dtype, vl, half_vl, block) != af::SUCCESS) {
      return LastAxisBranch::kFallback;
    }
    const Expr last = tiling.dst_shape.back();
    const auto last_check = af::SymbolicUtils::StaticCheckLe(last, vl);
    if (last_check == af::TriBool::kTrue) {
      return LastAxisBranch::kDynamicLessThanVlUnaligned;
    }
    if (last_check == af::TriBool::kFalse) {
      return LastAxisBranch::kDynamicLargerThanVlUnaligned;
    }
    return LastAxisBranch::kFallback;
  }
  return ClassifySmallRank(tiling, dtype, tiling.original_rank > 4U ? -1 : const_rank);
}

}  // namespace ascendcapi_v2

namespace ascendcperf_v2 {
namespace {
using ascendcapi_v2::BroadcastTilingInfo;
using ascendcapi_v2::LastAxisBranch;
using ascendcapi_v2::ParamExprInputs;
using ascendcapi_v2::VfCostAccumulator;

bool IsB8Dtype(const std::string &dtype) {
  return dtype == kUInt8 || dtype == kInt8;
}

std::string HelperDtype(const std::string &dtype) {
  std::string helper_dtype;
  return ascendcapi_v2::GetEffectiveHelperDtype(dtype, helper_dtype) == af::SUCCESS ? helper_dtype : dtype;
}

Expr ProductFrom(const std::vector<Expr> &shape, size_t end) {
  Expr result = CreateExpr(1);
  for (size_t i = 0U; i < end; ++i) {
    result = result * shape[i];
  }
  return result;
}

af::Status Add(const std::string &op, const std::string &dtype, const Expr &count, VfCostAccumulator &acc) {
  return ascendcapi_v2::AddVfInstructPerf(op, dtype, count, 1U, acc);
}

af::Status AddLastGatherIndexPerf(const std::string &dtype, VfCostAccumulator &acc) {
  Expr byte_size;
  GE_ASSERT_SUCCESS(ascendcapi_v2::GetDtypeByteSize(dtype, byte_size));
  const std::string index_dtype =
      af::SymbolicUtils::StaticCheckLe(byte_size, CreateExpr(2)) == af::TriBool::kTrue ? kInt16 : kInt32;
  const Expr calls = IsB8Dtype(dtype) ? CreateExpr(2) : CreateExpr(1);
  GE_ASSERT_SUCCESS(Add(kDuplicate, index_dtype, calls, acc));
  // Placeholder: Reg::Arange in last-axis index generation, index dtype, one index batch.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, index_dtype, acc.max_latency, acc.throughput, calls));
  GE_ASSERT_SUCCESS(Add(kDiv, index_dtype, calls, acc));
  return Add(kStore, index_dtype, calls, acc);
}

bool IsLt(const Expr &lhs, const Expr &rhs) {
  return af::SymbolicUtils::StaticCheckLt(lhs, rhs) == af::TriBool::kTrue;
}

af::Status SafeDiv(const Expr &value, const Expr &divisor, Expr &result) {
  if (af::SymbolicUtils::StaticCheckEq(divisor, CreateExpr(0)) == af::TriBool::kTrue) {
    GELOGW("Broadcast last-axis divisor cannot be zero.");
    return af::SUCCESS;
  }
  result = af::sym::Floor(value / divisor);
  return af::SUCCESS;
}

std::string GatherIndexDtype(const std::string &dtype) {
  std::string helper_dtype;
  if (ascendcapi_v2::GetEffectiveHelperDtype(dtype, helper_dtype) == af::SUCCESS && helper_dtype != dtype) {
    return GatherIndexDtype(helper_dtype);
  }
  Expr bytes;
  if (ascendcapi_v2::GetDtypeByteSize(dtype, bytes) != af::SUCCESS) {
    return dtype;
  }
  return bytes == kSymFour ? kInt32 : kInt16;
}

std::string GatherRegDtype(const std::string &dtype) {
  std::string helper_dtype;
  if (ascendcapi_v2::GetEffectiveHelperDtype(dtype, helper_dtype) == af::SUCCESS && helper_dtype != dtype) {
    return GatherRegDtype(helper_dtype);
  }
  Expr bytes;
  if (ascendcapi_v2::GetDtypeByteSize(dtype, bytes) != af::SUCCESS) {
    return dtype;
  }
  return bytes == kSymFour ? kUInt32 : kUInt16;
}

std::string RankTwoGatherDtype(const std::string &dtype) {
  return GatherRegDtype(dtype);
}

af::Status AddWrapperIndexGeneration(const std::string &dtype, size_t rank, VfCostAccumulator &acc) {
  const std::string index_dtype = GatherIndexDtype(dtype);
  const Expr lanes = IsB8Dtype(dtype) ? CreateExpr(2) : CreateExpr(1);
  GE_ASSERT_SUCCESS(Add(kDuplicate, index_dtype, CreateExpr(static_cast<int64_t>(rank * 2U)), acc));
  // Placeholder: Reg::Arange in VfGenIndex/VfGenIndexB8.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, index_dtype, acc.max_latency, acc.throughput, lanes));
  GE_ASSERT_SUCCESS(Add(kDiv, index_dtype, CreateExpr(static_cast<int64_t>(rank)) * lanes, acc));
  GE_ASSERT_SUCCESS(Add(kMul, index_dtype, CreateExpr(static_cast<int64_t>(rank + 1U)) * lanes, acc));
  GE_ASSERT_SUCCESS(Add(kSub, index_dtype, CreateExpr(static_cast<int64_t>(rank)) * lanes, acc));
  GE_ASSERT_SUCCESS(Add(kMulAddDst, index_dtype, CreateExpr(static_cast<int64_t>(rank - 1U)) * lanes, acc));
  return Add(kStore, index_dtype, lanes, acc);
}

struct GatherLoopCounts {
  Expr first{CreateExpr(1)};
  Expr second{CreateExpr(1)};
  Expr third{CreateExpr(1)};
};

af::Status GetRankThreeGatherLoops(const std::vector<Expr> &shape, const Expr &vl, GatherLoopCounts &counts) {
  const Expr inner = shape[2] * shape[1];
  Expr tile;
  if (IsLt(inner, vl)) {
    GE_ASSERT_SUCCESS(SafeDiv(vl, inner, tile));
    return SafeDiv(shape[0], tile, counts.second);
  }
  GE_ASSERT_SUCCESS(SafeDiv(vl, shape[2], tile));
  counts.first = shape[0];
  return SafeDiv(shape[1], tile, counts.second);
}

af::Status GetRankFourGatherLoops(const std::vector<Expr> &shape, const Expr &vl, GatherLoopCounts &counts) {
  const Expr inner_three = shape[3] * shape[2] * shape[1];
  Expr tile;
  if (IsLt(inner_three, vl)) {
    GE_ASSERT_SUCCESS(SafeDiv(vl, inner_three, tile));
    return SafeDiv(shape[0], tile, counts.third);
  }
  const Expr inner_two = shape[3] * shape[2];
  if (IsLt(inner_two, vl)) {
    GE_ASSERT_SUCCESS(SafeDiv(vl, inner_two, tile));
    counts.second = shape[0];
    return SafeDiv(shape[1], tile, counts.third);
  }
  GE_ASSERT_SUCCESS(SafeDiv(vl, shape[3], tile));
  counts.first = shape[0];
  counts.second = shape[1];
  return SafeDiv(shape[2], tile, counts.third);
}

af::Status AddGatherWrapperArithmetic(const std::string &dtype, size_t rank, const GatherLoopCounts &loops,
                                      VfCostAccumulator &acc) {
  const Expr lanes = IsB8Dtype(dtype) ? CreateExpr(2) : CreateExpr(1);
  const Expr first = loops.first;
  const Expr second = loops.first * (rank == 3U ? loops.second + CreateExpr(1) : loops.second);
  const Expr third = rank == 3U ? CreateExpr(0) : loops.first * loops.second * (loops.third + CreateExpr(1));
  const Expr arithmetic = first + second + third;
  GE_ASSERT_SUCCESS(Add(kMuls, GatherRegDtype(dtype), arithmetic, acc));
  return Add(kAdd, GatherRegDtype(dtype), arithmetic * lanes, acc);
}

af::Status AddGatherWrapperPerf(const NodeDetail &node, const BroadcastTilingInfo &tiling, size_t rank,
                                VfCostAccumulator &acc) {
  Expr vl;
  Expr half_vl;
  Expr block;
  GE_ASSERT_SUCCESS(ascendcapi_v2::GetBroadcastTilingVectorElements(node.input_dtype[0], vl, half_vl, block));
  (void)half_vl;
  (void)block;
  GatherLoopCounts loops;
  const auto &shape = tiling.dst_shape;
  GE_ASSERT_SUCCESS(rank == 3U ? GetRankThreeGatherLoops(shape, vl, loops) : GetRankFourGatherLoops(shape, vl, loops));
  const Expr calls = rank == 3U ? loops.first * (loops.second + CreateExpr(1))
                                : loops.first * loops.second * (loops.third + CreateExpr(1));
  GE_ASSERT_SUCCESS(AddWrapperIndexGeneration(node.input_dtype[0], rank, acc));
  GE_ASSERT_SUCCESS(Add(kDuplicate, GatherIndexDtype(node.input_dtype[0]), kSymTwo, acc));
  const Expr lanes = IsB8Dtype(node.input_dtype[0]) ? CreateExpr(2) : CreateExpr(1);
  GE_ASSERT_SUCCESS(Add(kLoad, GatherIndexDtype(node.input_dtype[0]), lanes, acc));
  GE_ASSERT_SUCCESS(AddGatherWrapperArithmetic(node.input_dtype[0], rank, loops, acc));
  const Expr gathers = IsB8Dtype(node.input_dtype[0]) ? calls * CreateExpr(2) : calls;
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, GatherRegDtype(node.input_dtype[0]), acc.max_latency,
                                                   acc.throughput, gathers));
  if (IsB8Dtype(node.input_dtype[0])) {
    GE_ASSERT_SUCCESS(Add(kDeInterleave, node.input_dtype[0], calls, acc));
  }
  GE_ASSERT_SUCCESS(Add(kStore, node.output_dtype[0], calls, acc));
  return VfPerfUtils::AddVfInstructPerf(kPlaceholder, node.output_dtype[0], acc.max_latency, acc.throughput,
                                        CreateExpr(1));
}

af::Status GetE2BCount(const BroadcastTilingInfo &tiling, const Expr &vl, Expr &count) {
  const size_t last = tiling.rank - 1U;
  const Expr last_size = tiling.dst_shape[last];
  if (af::SymbolicUtils::StaticCheckEq(last_size, CreateExpr(0)) == af::TriBool::kTrue) {
    GELOGW("Broadcast E2B last-axis size cannot be zero.");
    return af::SUCCESS;
  }
  const Expr factor = vl / last_size;
  if (tiling.rank == 2U) {
    return ascendcapi_v2::CeilDiv(tiling.dst_shape[0], factor, count);
  }
  if (tiling.rank == 3U) {
    if (af::SymbolicUtils::StaticCheckGt(tiling.dst_shape[1] * last_size, vl) == af::TriBool::kTrue) {
      Expr inner;
      GE_ASSERT_SUCCESS(ascendcapi_v2::CeilDiv(tiling.dst_shape[1], factor, inner));
      count = tiling.dst_shape[0] * inner;
    } else {
      count = tiling.dst_shape[0];
    }
    return af::SUCCESS;
  }
  Expr inner;
  GE_ASSERT_SUCCESS(ascendcapi_v2::CeilDiv(tiling.dst_shape[2], factor, inner));
  count = tiling.dst_shape[0] * tiling.dst_shape[1] * inner;
  return af::SUCCESS;
}

af::Status AddGatherBranch(const NodeDetail &node, const BroadcastTilingInfo &tiling, const Expr &vl,
                           LastAxisBranch branch, const Expr &outer, VfCostAccumulator &acc) {
  const size_t last = tiling.rank - 1U;
  Expr gather_count = CreateExpr(1);
  if (branch == LastAxisBranch::kGatherTwo) {
    const Expr last_size = tiling.dst_shape[last];
    if (af::SymbolicUtils::StaticCheckEq(last_size, CreateExpr(0)) == af::TriBool::kTrue) {
      GELOGW("Broadcast last-axis size cannot be zero.");
      return af::SUCCESS;
    }
    const Expr factor = vl / last_size;
    GE_ASSERT_SUCCESS(ascendcapi_v2::CeilDiv(outer, factor, gather_count));
  }
  const Expr lanes = IsB8Dtype(node.input_dtype[0]) ? CreateExpr(2) : CreateExpr(1);
  GE_ASSERT_SUCCESS(AddLastGatherIndexPerf(node.input_dtype[0], acc));
  if (branch == LastAxisBranch::kGatherTwo) {
    GE_ASSERT_SUCCESS(Add(kDuplicate, GatherRegDtype(node.input_dtype[0]), CreateExpr(1), acc));
    GE_ASSERT_SUCCESS(Add(kMuls, GatherRegDtype(node.input_dtype[0]), gather_count - CreateExpr(1), acc));
    GE_ASSERT_SUCCESS(Add(kAdd, GatherRegDtype(node.input_dtype[0]), (gather_count - CreateExpr(1)) * lanes, acc));
    GE_ASSERT_SUCCESS(Add(kAdds, GatherRegDtype(node.input_dtype[0]), lanes, acc));
  } else {
    GE_ASSERT_SUCCESS(Add(kUpdateMask, node.input_dtype[0], CreateExpr(1), acc));
  }
  GE_ASSERT_SUCCESS(Add(kLoad, GatherRegDtype(node.input_dtype[0]), lanes, acc));
  // Placeholder: Reg::Gather in last-axis GatherOne/Two, input data dtype, each gather batch.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, RankTwoGatherDtype(node.input_dtype[0]),
                                                   acc.max_latency, acc.throughput, gather_count * lanes));
  if (IsB8Dtype(node.input_dtype[0])) {
    GE_ASSERT_SUCCESS(Add(kDeInterleave, node.input_dtype[0], gather_count, acc));
  }
  GE_ASSERT_SUCCESS(Add(kStore, node.output_dtype[0], gather_count, acc));
  if (branch == LastAxisBranch::kGatherTwo) {
    // Placeholder: Reg::StoreUnAlignPost after BrcLastGatherTwo.
    return VfPerfUtils::AddVfInstructPerf(kPlaceholder, node.output_dtype[0], acc.max_latency, acc.throughput,
                                          CreateExpr(1));
  }
  return af::SUCCESS;
}

af::Status AddTailBranch(const NodeDetail &node, LastAxisBranch branch, const Expr &stores, VfCostAccumulator &acc) {
  if (branch == LastAxisBranch::kLargerThanVlUnaligned || branch == LastAxisBranch::kLessThanVlUnaligned ||
      branch == LastAxisBranch::kDynamicLessThanVlUnaligned ||
      branch == LastAxisBranch::kDynamicLargerThanVlUnaligned) {
    GE_ASSERT_SUCCESS(Add(kStore, node.output_dtype[0], stores, acc));
    // Placeholder: Reg::StoreUnAlignPost after the last-axis unaligned stores.
    return VfPerfUtils::AddVfInstructPerf(kPlaceholder, node.output_dtype[0], acc.max_latency, acc.throughput,
                                          CreateExpr(1));
  }
  return Add(kStore, node.output_dtype[0], stores, acc);
}

af::Status AddDynamicBranch(const NodeDetail &node, const Expr &outer, const Expr &stores, const Expr &helper_calls,
                            VfCostAccumulator &acc) {
  const std::string helper_dtype = HelperDtype(node.input_dtype[0]);
  GE_ASSERT_SUCCESS(Add(kLoad, node.input_dtype[0], outer, acc));
  GE_ASSERT_SUCCESS(Add(kStore, node.output_dtype[0], stores, acc));
  // Placeholder: Reg::StoreUnAlignPost after each dynamic last-axis helper invocation.
  return VfPerfUtils::AddVfInstructPerf(kPlaceholder, helper_dtype, acc.max_latency, acc.throughput, helper_calls);
}

af::Status AddRegularBranch(const NodeDetail &node, const BroadcastTilingInfo &tiling, LastAxisBranch branch,
                            VfCostAccumulator &acc) {
  Expr vl;
  Expr half_vl;
  Expr block;
  GE_ASSERT_SUCCESS(ascendcapi_v2::GetBroadcastTilingVectorElements(node.input_dtype[0], vl, half_vl, block));
  (void)half_vl;
  (void)block;
  const size_t last = tiling.rank - 1U;
  const auto &shape = tiling.dst_shape;
  Expr outer = ProductFrom(shape, shape.size() - 1U);
  if (tiling.loop_num != CreateExpr(0)) {
    outer = outer * tiling.loop_num;
  }
  Expr repeats;
  const Expr last_size = tiling.dst_shape[last];
  GE_ASSERT_SUCCESS(ascendcapi_v2::CeilDiv(last_size, vl, repeats));
  const Expr stores = outer * repeats;
  if (branch == LastAxisBranch::kDynamicLessThanVlUnaligned ||
      branch == LastAxisBranch::kDynamicLargerThanVlUnaligned) {
    Expr helper_calls = CreateExpr(1);
    if (tiling.dst_shape.size() > 4U) {
      helper_calls = ProductFrom(tiling.dst_shape, tiling.dst_shape.size() - 4U);
    }
    return AddDynamicBranch(node, outer, stores, helper_calls, acc);
  }
  if (branch == LastAxisBranch::kE2B || branch == LastAxisBranch::kE2BLessThanVl ||
      branch == LastAxisBranch::kE2BLargerThanVl) {
    Expr e2b_count;
    GE_ASSERT_SUCCESS(GetE2BCount(tiling, vl, e2b_count));
    GE_ASSERT_SUCCESS(Add(kUpdateMask, node.input_dtype[0], e2b_count, acc));
    GE_ASSERT_SUCCESS(Add(kLoad, node.input_dtype[0], e2b_count, acc));
    return Add(kStore, node.output_dtype[0], e2b_count, acc);
  }
  if (branch == LastAxisBranch::kGatherOne || branch == LastAxisBranch::kGatherTwo) {
    return AddGatherBranch(node, tiling, vl, branch, outer, acc);
  }
  if (branch == LastAxisBranch::kGatherWrapper || branch == LastAxisBranch::kGatherWrapperForFourDim) {
    return AddGatherWrapperPerf(node, tiling, branch == LastAxisBranch::kGatherWrapper ? 3U : 4U, acc);
  }
  if (branch == LastAxisBranch::kLessThanVlAligned) {
    GE_ASSERT_SUCCESS(Add(kUpdateMask, node.input_dtype[0], CreateExpr(1), acc));
  }
  if (branch == LastAxisBranch::kLargerThanVlAligned) {
    const Expr updates = tiling.rank == 4U ? tiling.dst_shape[0] * tiling.dst_shape[2] * repeats : stores;
    GE_ASSERT_SUCCESS(Add(kUpdateMask, node.input_dtype[0], updates, acc));
  }
  const Expr loads = tiling.rank == 4U ? stores : outer;
  GE_ASSERT_SUCCESS(Add(kLoad, node.input_dtype[0], loads, acc));
  return AddTailBranch(node, branch, stores, acc);
}

af::TriBool Check(const Expr &lhs, CondType condition, const Expr &rhs) {
  if (condition == CondType::K_EQ) {
    return af::SymbolicUtils::StaticCheckEq(lhs, rhs);
  }
  if (condition == CondType::K_LT) {
    return af::SymbolicUtils::StaticCheckLt(lhs, rhs);
  }
  if (condition == CondType::K_LE) {
    return af::SymbolicUtils::StaticCheckLe(lhs, rhs);
  }
  return af::SymbolicUtils::StaticCheckGt(lhs, rhs);
}

Expr LeafCost(const NodeDetail &node, const BroadcastTilingInfo &tiling, LastAxisBranch branch) {
  const bool dynamic_wrapper = node.broadcast_node_params.const_rank == -1 || tiling.original_rank > 4U;
  if (dynamic_wrapper) {
    if (branch == LastAxisBranch::kLargerThanVlUnaligned) {
      branch = LastAxisBranch::kDynamicLargerThanVlUnaligned;
    }
    if (branch == LastAxisBranch::kLessThanVlUnaligned && tiling.rank >= 3U) {
      branch = LastAxisBranch::kDynamicLessThanVlUnaligned;
    }
  }
  if (branch == LastAxisBranch::kGatherWrapper || branch == LastAxisBranch::kGatherWrapperForFourDim ||
      branch == LastAxisBranch::kGatherOne || branch == LastAxisBranch::kGatherTwo) {
    Expr vl;
    Expr half_vl;
    Expr block;
    if (ascendcapi_v2::GetBroadcastTilingVectorElements(node.input_dtype[0], vl, half_vl, block) != af::SUCCESS) {
      return CreateExpr(0);
    }
    if (af::SymbolicUtils::StaticCheckLt(tiling.dst_shape.back(), half_vl) == af::TriBool::kFalse) {
      return CreateExpr(0);
    }
  }
  VfCostAccumulator acc;
  if (AddRegularBranch(node, tiling, branch, acc) != af::SUCCESS) {
    return CreateExpr(0);
  }
  return ascendcapi_v2::GetVfCost(acc);
}

af::Status BuildDynamicMoreDimLastTree(const NodeDetail &node, const BroadcastTilingInfo &tiling, const Expr &vl,
                                       TernaryOpMap &ternary_ops, Expr &result) {
  const Expr last = tiling.dst_shape.back();
  GE_ASSERT_SUCCESS(ascendcapi_v2::BuildBroadcastTernary(
      "broadcast_dynamic_last_le_vl", CondType::K_LE, last, vl,
      LeafCost(node, tiling, LastAxisBranch::kDynamicLessThanVlUnaligned),
      LeafCost(node, tiling, LastAxisBranch::kDynamicLargerThanVlUnaligned), ternary_ops, result));
  return af::SUCCESS;
}

af::Status Select(const std::string &name, CondType condition, const Expr &lhs, const Expr &rhs, const Expr &true_value,
                  const Expr &false_value, TernaryOpMap &ternary_ops, Expr &result) {
  return ascendcapi_v2::BuildBroadcastTernary(name, condition, lhs, rhs, true_value, false_value, ternary_ops, result);
}

af::Status BuildTailTree(const NodeDetail &node, const BroadcastTilingInfo &tiling, const Expr &vl, const Expr &half_vl,
                         const Expr &block, TernaryOpMap &ternary_ops, Expr &result) {
  const size_t rank = tiling.rank;
  const Expr last = tiling.dst_shape.back();
  const bool b8 = IsB8Dtype(node.input_dtype[0]);
  if (rank == 2U) {
    Expr larger_aligned;
    GE_ASSERT_SUCCESS(Select("broadcast_rank2_block_aligned", CondType::K_EQ, af::sym::Mod(last, block), CreateExpr(0),
                             LeafCost(node, tiling, LastAxisBranch::kLargerThanVlAligned),
                             LeafCost(node, tiling, LastAxisBranch::kLargerThanVlUnaligned), ternary_ops,
                             larger_aligned));
    Expr less_or_larger;
    GE_ASSERT_SUCCESS(Select("broadcast_rank2_last_le_vl", CondType::K_LE, last, vl,
                             LeafCost(node, tiling, LastAxisBranch::kLessThanVlUnaligned), larger_aligned, ternary_ops,
                             less_or_larger));
    Expr gather;
    GE_ASSERT_SUCCESS(Select("broadcast_rank2_total_lt_vl", CondType::K_LT, tiling.dst_size, vl,
                             LeafCost(node, tiling, LastAxisBranch::kGatherOne),
                             LeafCost(node, tiling, LastAxisBranch::kGatherTwo), ternary_ops, gather));
    Expr tail;
    GE_ASSERT_SUCCESS(Select("broadcast_rank2_last_lt_half_vl", CondType::K_LT, last, half_vl, gather, less_or_larger,
                             ternary_ops, tail));
    if (!b8) {
      GE_ASSERT_SUCCESS(Select("broadcast_rank2_e2b", CondType::K_EQ, last, block,
                               LeafCost(node, tiling, LastAxisBranch::kE2B), tail, ternary_ops, result));
    } else {
      result = tail;
    }
    return af::SUCCESS;
  }

  Expr aligned;
  GE_ASSERT_SUCCESS(Select("broadcast_rank_last_le_vl_aligned", CondType::K_EQ, af::sym::Mod(last, block),
                           CreateExpr(0), LeafCost(node, tiling, LastAxisBranch::kLessThanVlAligned),
                           LeafCost(node, tiling, LastAxisBranch::kLessThanVlUnaligned), ternary_ops, aligned));
  Expr tail;
  GE_ASSERT_SUCCESS(Select("broadcast_rank_last_le_vl", CondType::K_LE, last, vl, aligned,
                           LeafCost(node, tiling, LastAxisBranch::kLargerThanVlAligned), ternary_ops, tail));
  Expr larger;
  GE_ASSERT_SUCCESS(Select("broadcast_rank_last_block_aligned", CondType::K_EQ, af::sym::Mod(last, block),
                           CreateExpr(0), LeafCost(node, tiling, LastAxisBranch::kLargerThanVlAligned),
                           LeafCost(node, tiling, LastAxisBranch::kLargerThanVlUnaligned), ternary_ops, larger));
  GE_ASSERT_SUCCESS(
      Select("broadcast_rank_last_le_vl_final", CondType::K_LE, last, vl, tail, larger, ternary_ops, result));
  if (!b8 || rank == 3U) {
    Expr gather;
    const auto gather_branch = rank == 3U ? LastAxisBranch::kGatherWrapper : LastAxisBranch::kGatherWrapperForFourDim;
    GE_ASSERT_SUCCESS(Select("broadcast_rank_last_lt_half_vl", CondType::K_LT, last, half_vl,
                             LeafCost(node, tiling, gather_branch), result, ternary_ops, gather));
    result = gather;
  }
  if (!b8 && rank == 3U && tiling.src_stride.size() > 1U &&
      Check(tiling.src_stride[1], CondType::K_EQ, CreateExpr(0)) == af::TriBool::kFalse) {
    const Expr inner = tiling.dst_shape[1] * last;
    const Expr block_num = vl / block;
    Expr e2b_candidate;
    GE_ASSERT_SUCCESS(Select("broadcast_rank3_e2b_size", CondType::K_GT, inner, vl,
                             LeafCost(node, tiling, LastAxisBranch::kE2BLargerThanVl),
                             LeafCost(node, tiling, LastAxisBranch::kE2BLessThanVl), ternary_ops, e2b_candidate));
    Expr e2b_size;
    GE_ASSERT_SUCCESS(Select("broadcast_rank3_e2b_block_num", CondType::K_EQ,
                             af::sym::Mod(tiling.dst_shape[1], block_num), CreateExpr(0), e2b_candidate, result,
                             ternary_ops, e2b_size));
    Expr e2b_after_half;
    GE_ASSERT_SUCCESS(Select("broadcast_rank3_e2b_half_vl", CondType::K_GT, inner, half_vl, e2b_size, result,
                             ternary_ops, e2b_after_half));
    const Expr base_result = result;
    Expr e2b_result;
    GE_ASSERT_SUCCESS(Select("broadcast_rank3_e2b_last", CondType::K_EQ, last, block, e2b_after_half, base_result,
                             ternary_ops, e2b_result));
    result = e2b_result;
  }
  if (!b8 && rank == 4U && tiling.src_stride.size() > 2U &&
      Check(tiling.src_stride[2], CondType::K_EQ, CreateExpr(0)) == af::TriBool::kFalse) {
    const Expr block_num = vl / block;
    Expr e2b;
    GE_ASSERT_SUCCESS(Select("broadcast_rank4_e2b_block_num", CondType::K_EQ,
                             af::sym::Mod(tiling.dst_shape[2], block_num), CreateExpr(0),
                             LeafCost(node, tiling, LastAxisBranch::kE2B), result, ternary_ops, e2b));
    const Expr base_result = result;
    Expr e2b_result;
    GE_ASSERT_SUCCESS(
        Select("broadcast_rank4_e2b_last", CondType::K_EQ, last, block, e2b, base_result, ternary_ops, e2b_result));
    result = e2b_result;
  }
  return af::SUCCESS;
}

af::Status CalibrateAlignedFp32Rows(const NodeDetail &node, const BroadcastTilingInfo &tiling, Expr &cycles,
                                    TernaryOpMap &ternary_ops) {
  constexpr int32_t kRank = 3;
  if (node.input_dtype[0] != kFloat32 || node.output_dtype[0] != kFloat32 ||
      node.broadcast_node_params.const_rank != kRank || tiling.original_rank != static_cast<size_t>(kRank)) {
    return af::SUCCESS;
  }
  const auto &src = tiling.original_src_shape;
  const auto &dst = tiling.original_dst_shape;
  const auto &one = af::sym::kSymbolOne;
  const auto &zero = af::sym::kSymbolZero;
  if (af::SymbolicUtils::StaticCheckEq(src.front(), one) != af::TriBool::kTrue ||
      af::SymbolicUtils::StaticCheckEq(src.back(), one) != af::TriBool::kTrue) {
    return af::SUCCESS;
  }
  constexpr int32_t kMinRows = 72;
  constexpr int32_t kMaxRows = 168;
  constexpr int32_t kMinWidth = 32;
  constexpr int32_t kMaxWidth = 40;
  constexpr int32_t kAlignElements = 32 / sizeof(float);
  constexpr double kHeadCycles = 18.37997610;
  constexpr double kCyclesPerRow = 5.9969360064;
  const auto outside = [&](const Expr &value, int32_t lower, int32_t upper) {
    return af::sym::Max(zero, af::sym::Max(CreateExpr(lower) - value, value - CreateExpr(upper)));
  };
  const auto &rows = dst[1];
  const auto &width = dst.back();
  const Expr violation = af::sym::Abs(dst.front() - one) + af::sym::Abs(rows - src[1]) +
                         outside(rows, kMinRows, kMaxRows) + outside(width, kMinWidth, kMaxWidth) +
                         af::sym::Abs(af::sym::Mod(width, CreateExpr(kAlignElements)));
  const Expr fitted = CreateExpr(kHeadCycles) + CreateExpr(kCyclesPerRow) * rows;
  const Expr fallback = cycles;
  return ascendcapi_v2::BuildBroadcastTernary("broadcast_fp32_aligned_rows", CondType::K_EQ, violation, zero, fitted,
                                              fallback, ternary_ops, cycles);
}

}  // namespace

LastAxisBranch GetLastAxisPerfBranch(const NodeDetail &node_info) {
  const auto &params = node_info.broadcast_node_params;
  ParamExprInputs src_inputs;
  src_inputs.semantic = node_info.input_dims;
  src_inputs.size = node_info.input_dims;
  src_inputs.actual_size = node_info.repeats;
  ParamExprInputs dst_inputs;
  dst_inputs.semantic = node_info.output_dims;
  dst_inputs.size = node_info.output_dims;
  dst_inputs.actual_size = node_info.output_dims;
  BroadcastTilingInfo tiling;
  if (ascendcapi_v2::BuildBroadcastTiling(params.src_shape, params.dst_shape, node_info.input_dtype[0], src_inputs,
                                          dst_inputs, tiling) != af::SUCCESS) {
    return LastAxisBranch::kFallback;
  }
  return ascendcapi_v2::GetLastAxisBranch(tiling, node_info.input_dtype[0], node_info.broadcast_node_params.const_rank);
}

af::Status BuildLastAxisPerf(const NodeDetail &node_info, const BroadcastTilingInfo &tiling, PerfOutputInfo &perf) {
  const auto branch =
      ascendcapi_v2::GetLastAxisBranch(tiling, node_info.input_dtype[0], node_info.broadcast_node_params.const_rank);
  GE_ASSERT_TRUE(
      branch != LastAxisBranch::kFallback || tiling.rank <= 4U || node_info.broadcast_node_params.const_rank == -1,
      "Unsupported last-axis Broadcast branch.");
  Expr result;
  const bool dynamic_leaf =
      branch == LastAxisBranch::kDynamicLessThanVlUnaligned || branch == LastAxisBranch::kDynamicLargerThanVlUnaligned;
  if (dynamic_leaf && node_info.broadcast_node_params.const_rank != -1) {
    VfCostAccumulator acc;
    GE_ASSERT_SUCCESS(AddRegularBranch(node_info, tiling, branch, acc));
    result = ascendcapi_v2::GetVfCost(acc);
  } else if (tiling.rank >= 2U && tiling.rank <= 4U &&
             !(node_info.broadcast_node_params.const_rank == -1 && tiling.original_rank > 4U)) {
    Expr vl;
    Expr half_vl;
    Expr block;
    GE_ASSERT_SUCCESS(ascendcapi_v2::GetBroadcastTilingVectorElements(node_info.input_dtype[0], vl, half_vl, block));
    GE_ASSERT_SUCCESS(BuildTailTree(node_info, tiling, vl, half_vl, block, perf.ternary_ops, result));
  } else if (branch == LastAxisBranch::kFallback && node_info.broadcast_node_params.const_rank == -1) {
    Expr vl;
    Expr half_vl;
    Expr block;
    GE_ASSERT_SUCCESS(ascendcapi_v2::GetBroadcastTilingVectorElements(node_info.input_dtype[0], vl, half_vl, block));
    GE_ASSERT_SUCCESS(BuildDynamicMoreDimLastTree(node_info, tiling, vl, perf.ternary_ops, result));
  } else {
    VfCostAccumulator acc;
    GE_ASSERT_TRUE(branch != LastAxisBranch::kFallback, "Unsupported last-axis Broadcast branch.");
    GE_ASSERT_SUCCESS(AddRegularBranch(node_info, tiling, branch, acc));
    result = ascendcapi_v2::GetVfCost(acc);
  }
  GE_ASSERT_SUCCESS(CalibrateAlignedFp32Rows(node_info, tiling, result, perf.ternary_ops));
  result.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = result;
  return af::SUCCESS;
}

}  // namespace ascendcperf_v2
}  // namespace att
