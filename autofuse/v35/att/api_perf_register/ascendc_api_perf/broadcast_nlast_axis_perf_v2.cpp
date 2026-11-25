/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "broadcast_nlast_axis_perf_v2.h"

#include "base/att_const_values.h"
#include "common/checker.h"
#include "broadcast_api_perf_v2.h"

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

bool IsUnknown(const Expr &lhs, CondType condition, const Expr &rhs) {
  if (condition == CondType::K_EQ) {
    return af::SymbolicUtils::StaticCheckEq(lhs, rhs) == af::TriBool::kUnknown;
  }
  if (condition == CondType::K_LT) {
    return af::SymbolicUtils::StaticCheckLt(lhs, rhs) == af::TriBool::kUnknown;
  }
  if (condition == CondType::K_LE) {
    return af::SymbolicUtils::StaticCheckLe(lhs, rhs) == af::TriBool::kUnknown;
  }
  if (condition == CondType::K_GT) {
    return af::SymbolicUtils::StaticCheckGt(lhs, rhs) == af::TriBool::kUnknown;
  }
  return false;
}

bool IsB8(const std::string &dtype) {
  return dtype == kUInt8 || dtype == kInt8;
}

NlastAxisBranch ClassifyRankTwo(const BroadcastTilingInfo &tiling, const Expr &vl, const Expr &half_vl,
                                const Expr &block, int32_t const_rank) {
  const Expr last = tiling.dst_shape[1U];
  if (IsTrue(last, CondType::K_LT, half_vl)) {
    const auto gather =
        IsTrue(tiling.dst_size, CondType::K_LT, vl) ? NlastAxisBranch::kGatherOne : NlastAxisBranch::kGatherTwo;
    return IsTrue(af::sym::Mod(last, block), CondType::K_EQ, CreateExpr(0))
               ? (gather == NlastAxisBranch::kGatherOne ? NlastAxisBranch::kGatherBOne : NlastAxisBranch::kGatherBTwo)
               : gather;
  }
  if (IsTrue(last, CondType::K_LE, vl)) {
    return NlastAxisBranch::kLessThanVlUnaligned;
  }
  if (!IsTrue(af::sym::Mod(last, block), CondType::K_EQ, CreateExpr(0))) {
    return const_rank == -1 ? NlastAxisBranch::kDynamicLargerThanVlUnaligned : NlastAxisBranch::kLargerThanVlUnaligned;
  }
  return IsTrue(af::sym::Mod(last, vl), CondType::K_EQ, CreateExpr(0)) &&
                 IsTrue(tiling.dst_shape[0], CondType::K_GT, kSymEight)
             ? NlastAxisBranch::kLargerThanVlAlignedWithVl
             : NlastAxisBranch::kLargerThanVlAlignedWithBlock;
}

NlastAxisBranch ClassifyRankThree(const BroadcastTilingInfo &tiling, const Expr &vl, const Expr &half_vl,
                                  const Expr &block, int32_t const_rank) {
  const Expr last = tiling.dst_shape[2U];
  if (IsTrue(last, CondType::K_LT, half_vl)) {
    return NlastAxisBranch::kGather;
  }
  if (IsTrue(last, CondType::K_LE, vl)) {
    return NlastAxisBranch::kLessThanVlUnaligned;
  }
  if (IsTrue(af::sym::Mod(last, block), CondType::K_EQ, CreateExpr(0))) {
    return IsTrue(af::sym::Mod(last, vl), CondType::K_EQ, CreateExpr(0))
               ? NlastAxisBranch::kLargerThanVlAlignedWithVl
               : NlastAxisBranch::kLargerThanVlAlignedWithBlock;
  }
  return const_rank == -1 ? NlastAxisBranch::kDynamicLargerThanVlUnaligned : NlastAxisBranch::kLargerThanVlUnaligned;
}

NlastAxisBranch ClassifyRankFour(const BroadcastTilingInfo &tiling, const std::string &dtype, const Expr &vl,
                                 const Expr &half_vl, const Expr &block, int32_t const_rank) {
  const Expr last = tiling.dst_shape[3U];
  if (IsTrue(last, CondType::K_LT, half_vl) && !IsB8(dtype)) {
    return NlastAxisBranch::kGather;
  }
  if (IsTrue(last, CondType::K_LE, vl)) {
    if (IsTrue(af::sym::Mod(last, block), CondType::K_EQ, CreateExpr(0))) {
      return NlastAxisBranch::kLessThanVlAligned;
    }
    return const_rank == -1 ? NlastAxisBranch::kDynamicLessThanVlUnaligned : NlastAxisBranch::kLessThanVlUnaligned;
  }
  if (!IsTrue(af::sym::Mod(last, block), CondType::K_EQ, CreateExpr(0))) {
    return const_rank == -1 ? NlastAxisBranch::kDynamicLargerThanVlUnaligned : NlastAxisBranch::kLargerThanVlUnaligned;
  }
  return IsTrue(af::sym::Mod(last, vl), CondType::K_EQ, CreateExpr(0)) ? NlastAxisBranch::kLargerThanVlAlignedWithVl
                                                                       : NlastAxisBranch::kLargerThanVlAlignedWithBlock;
}

NlastAxisBranch ClassifySmallRank(const BroadcastTilingInfo &tiling, const std::string &dtype, int32_t const_rank) {
  Expr vl;
  Expr half_vl;
  Expr block;
  if (GetBroadcastTilingVectorElements(dtype, vl, half_vl, block) != af::SUCCESS) {
    return NlastAxisBranch::kFallback;
  }
  if (tiling.rank == 1U) {
    return NlastAxisBranch::kFallback;
  }
  if (tiling.rank == 2U) {
    return ClassifyRankTwo(tiling, vl, half_vl, block, const_rank);
  }
  if (tiling.rank == 3U) {
    return ClassifyRankThree(tiling, vl, half_vl, block, const_rank);
  }
  return ClassifyRankFour(tiling, dtype, vl, half_vl, block, const_rank);
}

}  // namespace

NlastAxisBranch GetNlastAxisBranch(const BroadcastTilingInfo &tiling, const std::string &dtype, int32_t const_rank) {
  if (tiling.rank == 0U || tiling.dst_shape.size() < tiling.rank) {
    return NlastAxisBranch::kFallback;
  }
  Expr vl;
  Expr half_vl;
  Expr block;
  if (GetBroadcastTilingVectorElements(dtype, vl, half_vl, block) != af::SUCCESS) {
    return NlastAxisBranch::kFallback;
  }
  const Expr last = tiling.dst_shape.back();
  if (const_rank == kCaseFour && (dtype == kUInt64 || dtype == kInt64) &&
      tiling.rank > static_cast<size_t>(kCaseFour)) {
    if (IsUnknown(last, CondType::K_LT, half_vl) || IsUnknown(last, CondType::K_LE, vl) ||
        IsUnknown(af::sym::Mod(last, block), CondType::K_EQ, CreateExpr(0))) {
      return NlastAxisBranch::kFallback;
    }
    return NlastAxisBranch::kB64MoreDimGather;
  }
  if (tiling.rank > 4U) {
    const bool unknown = IsUnknown(last, CondType::K_LT, half_vl) || IsUnknown(last, CondType::K_LE, vl) ||
                         IsUnknown(af::sym::Mod(last, block), CondType::K_EQ, CreateExpr(0));
    if (unknown) {
      return IsB8(dtype) ? NlastAxisBranch::kDynamicLessThanVlUnaligned : NlastAxisBranch::kDynamicGather;
    }
    if (IsTrue(last, CondType::K_LT, half_vl) && !IsB8(dtype)) {
      return NlastAxisBranch::kGatherWrapperForFourDim;
    }
    if (IsTrue(last, CondType::K_LE, vl)) {
      return IsTrue(af::sym::Mod(last, block), CondType::K_EQ, CreateExpr(0)) ? NlastAxisBranch::kLessThanVlAligned
                                                                              : NlastAxisBranch::kLessThanVlUnaligned;
    }
    return IsTrue(af::sym::Mod(last, block), CondType::K_EQ, CreateExpr(0))
               ? NlastAxisBranch::kLargerThanVlAlignedWithBlock
               : NlastAxisBranch::kLargerThanVlUnaligned;
  }
  return ClassifySmallRank(tiling, dtype, tiling.original_rank > 4U ? -1 : const_rank);
}

}  // namespace ascendcapi_v2

namespace ascendcperf_v2 {
namespace {

using ascendcapi_v2::BroadcastTilingInfo;
using ascendcapi_v2::NlastAxisBranch;
using ascendcapi_v2::ParamExprInputs;
using ascendcapi_v2::VfCostAccumulator;

bool IsLt(const Expr &lhs, const Expr &rhs) {
  return af::SymbolicUtils::StaticCheckLt(lhs, rhs) == af::TriBool::kTrue;
}

Expr ProductFrom(const std::vector<Expr> &shape, size_t end) {
  Expr result = CreateExpr(1);
  for (size_t i = 0U; i < end; ++i) {
    result = result * shape[i];
  }
  return result;
}

bool IsB8Dtype(const std::string &dtype) {
  return dtype == kUInt8 || dtype == kInt8;
}

std::string HelperDtype(const std::string &dtype) {
  std::string helper_dtype;
  return ascendcapi_v2::GetEffectiveHelperDtype(dtype, helper_dtype) == af::SUCCESS ? helper_dtype : dtype;
}

af::Status SafeDiv(const Expr &value, const Expr &divisor, Expr &result) {
  if (af::SymbolicUtils::StaticCheckEq(divisor, CreateExpr(0)) == af::TriBool::kTrue) {
    GELOGW("Broadcast non-last divisor cannot be zero.");
    return af::SUCCESS;
  }
  result = af::sym::Floor(value / divisor);
  return af::SUCCESS;
}

af::Status AddReg(const std::string &op, const std::string &dtype, const Expr &count, VfCostAccumulator &acc) {
  return ascendcapi_v2::AddVfInstructPerf(op, dtype, count, 1U, acc);
}

std::string RegDtype(const std::string &dtype) {
  std::string helper_dtype;
  if (ascendcapi_v2::GetEffectiveHelperDtype(dtype, helper_dtype) == af::SUCCESS) {
    if (helper_dtype != dtype) {
      return RegDtype(helper_dtype);
    }
  }
  Expr bytes;
  if (ascendcapi_v2::GetDtypeByteSize(dtype, bytes) != af::SUCCESS) {
    return dtype;
  }
  if (bytes == CreateExpr(1)) {
    return kUInt8;
  }
  if (bytes == kSymTwo) {
    return kUInt16;
  }
  return kUInt32;
}

std::string IndexDtype(const std::string &dtype) {
  std::string helper_dtype;
  if (ascendcapi_v2::GetEffectiveHelperDtype(dtype, helper_dtype) == af::SUCCESS && helper_dtype != dtype) {
    return IndexDtype(helper_dtype);
  }
  Expr bytes;
  if (ascendcapi_v2::GetDtypeByteSize(dtype, bytes) != af::SUCCESS) {
    return dtype;
  }
  if (bytes == CreateExpr(1) || bytes == kSymTwo) {
    return kInt16;
  }
  return kInt32;
}

std::string GatherWrapperIndexDtype(const std::string &dtype) {
  std::string helper_dtype;
  if (ascendcapi_v2::GetEffectiveHelperDtype(dtype, helper_dtype) == af::SUCCESS && helper_dtype != dtype) {
    return GatherWrapperIndexDtype(helper_dtype);
  }
  Expr bytes;
  if (ascendcapi_v2::GetDtypeByteSize(dtype, bytes) != af::SUCCESS) {
    return dtype;
  }
  return bytes == kSymFour ? kInt32 : kInt16;
}

std::string GatherWrapperRegDtype(const std::string &dtype) {
  std::string helper_dtype;
  if (ascendcapi_v2::GetEffectiveHelperDtype(dtype, helper_dtype) == af::SUCCESS && helper_dtype != dtype) {
    return GatherWrapperRegDtype(helper_dtype);
  }
  Expr bytes;
  if (ascendcapi_v2::GetDtypeByteSize(dtype, bytes) != af::SUCCESS) {
    return dtype;
  }
  return bytes == kSymFour ? kUInt32 : kUInt16;
}

af::Status AddIndexGeneration(const std::string &dtype, const std::string &index_dtype, size_t rank, Expr calls,
                              VfCostAccumulator &acc) {
  const Expr sets = IsB8Dtype(dtype) ? CreateExpr(2) : CreateExpr(1);
  const Expr op_calls = calls * sets;
  if (rank == 2U) {
    GE_ASSERT_SUCCESS(AddReg(kDuplicate, index_dtype, op_calls, acc));
    // Placeholder: Reg::Arange for rank-2 index generation, index dtype, each generated index lane.
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, HelperDtype(index_dtype), acc.max_latency,
                                                     acc.throughput, op_calls));
    GE_ASSERT_SUCCESS(AddReg(kDiv, index_dtype, op_calls, acc));
    GE_ASSERT_SUCCESS(AddReg(kMul, index_dtype, op_calls, acc));
    GE_ASSERT_SUCCESS(AddReg(kSub, index_dtype, op_calls, acc));
    return AddReg(kStore, index_dtype, op_calls, acc);
  }
  const Expr duplicate_count = CreateExpr(static_cast<int64_t>(rank * 2U));
  const Expr arithmetic_count = CreateExpr(static_cast<int64_t>(rank));
  GE_ASSERT_SUCCESS(AddReg(kDuplicate, index_dtype, duplicate_count * calls, acc));
  // Placeholder: Reg::Arange for rank-3/4 index generation, index dtype, each generated index lane.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, HelperDtype(index_dtype), acc.max_latency,
                                                   acc.throughput, op_calls));
  GE_ASSERT_SUCCESS(AddReg(kDiv, index_dtype, arithmetic_count * op_calls, acc));
  GE_ASSERT_SUCCESS(AddReg(kMul, index_dtype, (arithmetic_count + CreateExpr(1)) * op_calls, acc));
  GE_ASSERT_SUCCESS(AddReg(kSub, index_dtype, arithmetic_count * op_calls, acc));
  GE_ASSERT_SUCCESS(AddReg(kMulAddDst, index_dtype, CreateExpr(static_cast<int64_t>(rank - 1U)) * op_calls, acc));
  return AddReg(kStore, index_dtype, op_calls, acc);
}

struct GatherLoopCounts {
  Expr first;
  Expr second;
  Expr third;
};

af::Status GetGatherLoopCounts(const std::vector<Expr> &shape, size_t rank, const Expr &vl, GatherLoopCounts &counts) {
  GE_ASSERT_TRUE(rank == 3U || rank == 4U, "GatherWrapper rank must be 3 or 4, current rank:%zu", rank);
  const size_t last = rank - 1U;
  Expr inner_product = shape[last] * shape[last - 1U];
  if (rank == 4U) {
    inner_product = inner_product * shape[last - 2U];
  }
  Expr tile;
  if (IsLt(inner_product, vl)) {
    GE_ASSERT_SUCCESS(SafeDiv(vl, inner_product, tile));
    counts.first = CreateExpr(1);
    if (rank == 3U) {
      GE_ASSERT_SUCCESS(SafeDiv(shape[0], tile, counts.second));
      counts.third = CreateExpr(1);
    } else {
      counts.second = CreateExpr(1);
      GE_ASSERT_SUCCESS(SafeDiv(shape[0], tile, counts.third));
    }
    return af::SUCCESS;
  }
  if (rank == 3U) {
    GE_ASSERT_SUCCESS(SafeDiv(vl, shape[last], tile));
    counts.first = shape[0];
    GE_ASSERT_SUCCESS(SafeDiv(shape[1], tile, counts.second));
    counts.third = CreateExpr(1);
    return af::SUCCESS;
  }
  Expr second_product = shape[last] * shape[last - 1U];
  if (IsLt(second_product, vl)) {
    GE_ASSERT_SUCCESS(SafeDiv(vl, second_product, tile));
    counts.first = CreateExpr(1);
    counts.second = shape[0];
    GE_ASSERT_SUCCESS(SafeDiv(shape[1], tile, counts.third));
    return af::SUCCESS;
  }
  GE_ASSERT_SUCCESS(SafeDiv(vl, shape[last], tile));
  counts.first = shape[0];
  counts.second = shape[1];
  GE_ASSERT_SUCCESS(SafeDiv(shape[2], tile, counts.third));
  return af::SUCCESS;
}

af::Status AddGatherArithmetic(const std::string &reg_dtype, const GatherLoopCounts &counts, size_t rank,
                               Expr multiplier, bool b8, VfCostAccumulator &acc) {
  const Expr lane_multiplier = b8 ? CreateExpr(2) : CreateExpr(1);
  const Expr first = counts.first * multiplier * lane_multiplier;
  const Expr second =
      counts.first * (rank == 3U ? counts.second + CreateExpr(1) : counts.second) * multiplier * lane_multiplier;
  const Expr third = rank == 3U
                         ? CreateExpr(0)
                         : counts.first * counts.second * (counts.third + CreateExpr(1)) * multiplier * lane_multiplier;
  GE_ASSERT_SUCCESS(AddReg(kMuls, reg_dtype, first + second + third, acc));
  return AddReg(kAdd, reg_dtype, first + second + third, acc);
}

af::Status AddGatherWrapper(const std::string &dtype, const std::vector<Expr> &shape, size_t rank, Expr multiplier,
                            VfCostAccumulator &acc) {
  const bool b8 = IsB8Dtype(dtype);
  const std::string data_dtype = dtype;
  const std::string index_dtype = GatherWrapperIndexDtype(dtype);
  const std::string gather_dtype = GatherWrapperRegDtype(dtype);
  Expr vl;
  Expr half_vl;
  Expr block;
  GE_ASSERT_SUCCESS(ascendcapi_v2::GetBroadcastTilingVectorElements(dtype, vl, half_vl, block));
  (void)half_vl;
  (void)block;
  GatherLoopCounts counts;
  GE_ASSERT_SUCCESS(GetGatherLoopCounts(shape, rank, vl, counts));
  const Expr calls = rank == 3U ? counts.first * (counts.second + CreateExpr(1)) * multiplier
                                : counts.first * counts.second * (counts.third + CreateExpr(1)) * multiplier;
  GE_ASSERT_SUCCESS(AddIndexGeneration(dtype, index_dtype, rank, multiplier, acc));
  GE_ASSERT_SUCCESS(AddReg(kDuplicate, index_dtype, kSymTwo * multiplier, acc));
  GE_ASSERT_SUCCESS(AddReg(kLoad, index_dtype, multiplier, acc));
  GE_ASSERT_SUCCESS(AddGatherArithmetic(gather_dtype, counts, rank, multiplier, b8, acc));
  // Placeholder: Reg::Gather in the non-last GatherWrapper, gather register dtype, each wrapper call.
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kPlaceholder, HelperDtype(gather_dtype), acc.max_latency, acc.throughput, calls));
  if (b8) {
    // Placeholder: second Reg::Gather in B8 non-last GatherWrapper, uint16 gather register, each wrapper call.
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, HelperDtype(gather_dtype), acc.max_latency,
                                                     acc.throughput, calls));
    GE_ASSERT_SUCCESS(AddReg(kDeInterleave, data_dtype, calls, acc));
  }
  GE_ASSERT_SUCCESS(AddReg(kStore, data_dtype, calls, acc));
  // Reg::StoreUnAlignPost is separate from the tail store.
  return VfPerfUtils::AddVfInstructPerf(kPlaceholder, HelperDtype(data_dtype), acc.max_latency, acc.throughput,
                                        multiplier);
}

af::Status AddRank2Gather(const std::string &dtype, const Expr &outer, const Expr &last, bool gather_b, bool two,
                          VfCostAccumulator &acc) {
  Expr vl;
  Expr half_vl;
  Expr block;
  GE_ASSERT_SUCCESS(ascendcapi_v2::GetBroadcastTilingVectorElements(dtype, vl, half_vl, block));
  (void)half_vl;
  (void)block;
  Expr factor;
  GE_ASSERT_SUCCESS(SafeDiv(vl, last, factor));
  Expr rounds = CreateExpr(1);
  if (two) {
    GE_ASSERT_SUCCESS(ascendcapi_v2::CeilDiv(outer, factor, rounds));
  }
  const std::string index_dtype = IsB8Dtype(dtype) ? kUInt16 : RegDtype(dtype);
  if (gather_b) {
    const std::string gather_b_index_dtype = kUInt32;
    GE_ASSERT_SUCCESS(AddReg(kLoad, gather_b_index_dtype, CreateExpr(1), acc));
    // Placeholder: Reg::GatherB in rank-2 GatherB mode, effective B32 data register dtype, one index batch.
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, HelperDtype(RegDtype(dtype)), acc.max_latency,
                                                     acc.throughput, CreateExpr(1)));
    GE_ASSERT_SUCCESS(AddReg(kUpdateMask, dtype, two ? kSymTwo : CreateExpr(1), acc));
  } else {
    const Expr index_calls = IsB8Dtype(dtype) ? CreateExpr(2) : CreateExpr(1);
    GE_ASSERT_SUCCESS(AddIndexGeneration(dtype, IndexDtype(dtype), 2U, CreateExpr(1), acc));
    GE_ASSERT_SUCCESS(AddReg(kLoad, index_dtype, index_calls, acc));
    // Placeholder: Reg::Gather in rank-2 Gather mode, effective data register dtype, one index batch.
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, HelperDtype(RegDtype(dtype)), acc.max_latency,
                                                     acc.throughput, index_calls));
    if (IsB8Dtype(dtype)) {
      GE_ASSERT_SUCCESS(AddReg(kDeInterleave, dtype, CreateExpr(1), acc));
    }
    if (!two) {
      GE_ASSERT_SUCCESS(AddReg(kUpdateMask, dtype, CreateExpr(1), acc));
    }
  }
  GE_ASSERT_SUCCESS(AddReg(kStore, dtype, two ? rounds : CreateExpr(1), acc));
  if (two && !gather_b) {
    // BrcNlastGatherTwo finalizes its unaligned tail once.
    return VfPerfUtils::AddVfInstructPerf(kPlaceholder, HelperDtype(dtype), acc.max_latency, acc.throughput,
                                          CreateExpr(1));
  }
  return af::SUCCESS;
}

af::Status AddNlastGatherPerf(const NodeDetail &node_info, const BroadcastTilingInfo &tiling, NlastAxisBranch branch,
                              VfCostAccumulator &acc) {
  const std::string &dtype = node_info.input_dtype[0];
  const size_t rank = tiling.rank;
  const auto &shape = tiling.dst_shape;
  if (branch == NlastAxisBranch::kGather || branch == NlastAxisBranch::kDynamicGather) {
    const size_t gather_rank = branch == NlastAxisBranch::kDynamicGather ? 4U : rank;
    std::vector<Expr> gather_shape(shape.end() - gather_rank, shape.end());
    Expr multiplier = branch == NlastAxisBranch::kDynamicGather ? ProductFrom(shape, shape.size() - 4U) : CreateExpr(1);
    if (tiling.loop_num != CreateExpr(0)) {
      multiplier = multiplier * tiling.loop_num;
    }
    return AddGatherWrapper(dtype, gather_shape, gather_rank, multiplier, acc);
  }
  if (branch == NlastAxisBranch::kGatherWrapperForFourDim) {
    std::vector<Expr> gather_shape(shape.end() - 4U, shape.end());
    Expr multiplier = ProductFrom(shape, shape.size() - 4U);
    if (tiling.loop_num != CreateExpr(0)) {
      multiplier = multiplier * tiling.loop_num;
    }
    return AddGatherWrapper(dtype, gather_shape, 4U, multiplier, acc);
  }
  if (branch == NlastAxisBranch::kB64MoreDimGather) {
    std::vector<Expr> gather_shape(shape.end() - 4U, shape.end());
    Expr multiplier = shape[0];
    if (tiling.loop_num != CreateExpr(0)) {
      multiplier = multiplier * tiling.loop_num;
    }
    return AddGatherWrapper(dtype, gather_shape, 4U, multiplier, acc);
  }
  const bool two = branch == NlastAxisBranch::kGatherTwo || branch == NlastAxisBranch::kGatherBTwo;
  const bool gather_b = branch == NlastAxisBranch::kGatherBOne || branch == NlastAxisBranch::kGatherBTwo;
  return AddRank2Gather(dtype, tiling.dst_shape[0], tiling.dst_shape[1], gather_b, two, acc);
}

af::Status AddLessThanVlUnaligned(const std::string &dtype, size_t rank, const Expr &outer, const Expr &helper_loops,
                                  VfCostAccumulator &acc) {
  const Expr pre_count = rank == 2U ? CreateExpr(1) : outer;
  GE_ASSERT_SUCCESS(AddReg(kUpdateMask, dtype, helper_loops, acc));
  if (rank != 2U) {
    // Placeholder: Reg::LoadUnAlignPre in non-last unaligned mode, data dtype, each outer slice.
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kPlaceholder, HelperDtype(dtype), acc.max_latency, acc.throughput, pre_count));
  }
  // Rank-2 BrcNlastLessThanVLUnaligned performs one LoadAlign before all stores.
  GE_ASSERT_SUCCESS(AddReg(kLoad, dtype, rank == 2U ? CreateExpr(1) : outer, acc));
  GE_ASSERT_SUCCESS(AddReg(kStore, dtype, outer, acc));
  // Placeholder: Reg::StoreUnAlignPost after non-last less-than-VL unaligned mode, data dtype, one finalization.
  return VfPerfUtils::AddVfInstructPerf(kPlaceholder, HelperDtype(dtype), acc.max_latency, acc.throughput,
                                        helper_loops);
}

af::Status AddLargerThanVlUnaligned(const std::string &dtype, size_t rank, const Expr &outer, const Expr &factor,
                                    const Expr &helper_loops, VfCostAccumulator &acc) {
  if (rank == 2U) {
    GE_ASSERT_SUCCESS(AddReg(kLoad, dtype, CreateExpr(1), acc));
  } else {
    // Placeholder: Reg::LoadUnAlignPre before non-last larger-than-VL unaligned loads, data dtype, each outer slice.
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kPlaceholder, HelperDtype(dtype), acc.max_latency, acc.throughput, outer));
  }
  GE_ASSERT_SUCCESS(AddReg(kLoad, dtype, outer * factor, acc));
  GE_ASSERT_SUCCESS(AddReg(kStore, dtype, outer * factor, acc));
  if (rank != 2U) {
    GE_ASSERT_SUCCESS(AddReg(kLoad, dtype, outer, acc));
  }
  GE_ASSERT_SUCCESS(AddReg(kStore, dtype, outer, acc));
  // Placeholder: Reg::StoreUnAlignPost after non-last larger-than-VL unaligned stores, data dtype, one finalization.
  return VfPerfUtils::AddVfInstructPerf(kPlaceholder, HelperDtype(dtype), acc.max_latency, acc.throughput,
                                        helper_loops);
}

af::Status AddNlastDataPerf(const NodeDetail &node_info, const BroadcastTilingInfo &tiling, NlastAxisBranch branch,
                            const Expr &vl, VfCostAccumulator &acc) {
  const std::string data_dtype = node_info.input_dtype[0];
  const bool dynamic = branch == NlastAxisBranch::kDynamicLessThanVlAligned ||
                       branch == NlastAxisBranch::kDynamicLessThanVlUnaligned ||
                       branch == NlastAxisBranch::kDynamicLargerThanVlAlignedWithBlock ||
                       branch == NlastAxisBranch::kDynamicLargerThanVlUnaligned;
  const size_t rank = dynamic ? (tiling.original_rank > 4U ? tiling.rank : 4U) : tiling.rank;
  const auto &shape = dynamic
                          ? (tiling.original_rank > 4U
                                 ? tiling.dst_shape
                                 : (tiling.rank > tiling.original_rank ? tiling.dst_shape : tiling.original_dst_shape))
                          : tiling.dst_shape;
  Expr outer = ProductFrom(shape, shape.size() - 1U);
  const Expr helper_loops = dynamic && shape.size() > 4U ? ProductFrom(shape, shape.size() - 4U) : CreateExpr(1);
  if (tiling.loop_num != CreateExpr(0)) {
    outer = outer * tiling.loop_num;
  }
  const Expr last = shape.back();
  Expr factor;
  if (branch == NlastAxisBranch::kLargerThanVlUnaligned || branch == NlastAxisBranch::kDynamicLargerThanVlUnaligned) {
    GE_ASSERT_SUCCESS(SafeDiv(last, vl, factor));
  } else {
    GE_ASSERT_SUCCESS(ascendcapi_v2::CeilDiv(last, vl, factor));
  }
  if (branch == NlastAxisBranch::kLessThanVlAligned || branch == NlastAxisBranch::kDynamicLessThanVlAligned) {
    GE_ASSERT_SUCCESS(AddReg(kUpdateMask, data_dtype, dynamic ? helper_loops : CreateExpr(1), acc));
    GE_ASSERT_SUCCESS(AddReg(kLoad, data_dtype, outer, acc));
    return AddReg(kStore, data_dtype, outer, acc);
  }
  if (branch == NlastAxisBranch::kLessThanVlUnaligned || branch == NlastAxisBranch::kDynamicLessThanVlUnaligned) {
    return AddLessThanVlUnaligned(data_dtype, rank, outer, helper_loops, acc);
  }
  if (branch == NlastAxisBranch::kLargerThanVlUnaligned || branch == NlastAxisBranch::kDynamicLargerThanVlUnaligned) {
    return AddLargerThanVlUnaligned(data_dtype, rank, outer, factor, helper_loops, acc);
  }
  const bool use_vl = branch == NlastAxisBranch::kLargerThanVlAlignedWithVl;
  const Expr stores = outer * factor;
  const Expr loads = use_vl ? factor * (rank == 2U ? CreateExpr(1) : outer) : stores;
  const Expr updates = rank == 2U ? stores : (rank == 3U ? tiling.dst_shape[0] * factor : tiling.dst_shape[1] * factor);
  if (!use_vl) {
    GE_ASSERT_SUCCESS(AddReg(kUpdateMask, data_dtype, updates, acc));
  }
  GE_ASSERT_SUCCESS(AddReg(kLoad, data_dtype, loads, acc));
  return AddReg(kStore, data_dtype, stores, acc);
}

Expr NlastLeafCost(const NodeDetail &node_info, const BroadcastTilingInfo &tiling, NlastAxisBranch branch) {
  VfCostAccumulator acc;
  Expr vl;
  Expr half_vl;
  Expr block;
  if (ascendcapi_v2::GetBroadcastTilingVectorElements(node_info.input_dtype[0], vl, half_vl, block) != af::SUCCESS) {
    return CreateExpr(0);
  }
  const bool dynamic_wrapper = node_info.broadcast_node_params.const_rank == -1 || tiling.original_rank > 4U;
  if (dynamic_wrapper) {
    if (branch == NlastAxisBranch::kLargerThanVlUnaligned) {
      branch = NlastAxisBranch::kDynamicLargerThanVlUnaligned;
    }
    if (branch == NlastAxisBranch::kLessThanVlUnaligned && tiling.rank == 4U) {
      branch = NlastAxisBranch::kDynamicLessThanVlUnaligned;
    }
  }
  if (branch == NlastAxisBranch::kGather || branch == NlastAxisBranch::kDynamicGather ||
      branch == NlastAxisBranch::kGatherOne || branch == NlastAxisBranch::kGatherTwo ||
      branch == NlastAxisBranch::kGatherBOne || branch == NlastAxisBranch::kGatherBTwo) {
    if (AddNlastGatherPerf(node_info, tiling, branch, acc) != af::SUCCESS) {
      return CreateExpr(0);
    }
  } else if (AddNlastDataPerf(node_info, tiling, branch, vl, acc) != af::SUCCESS) {
    return CreateExpr(0);
  }
  return ascendcapi_v2::GetVfCost(acc);
}

af::Status BuildRankTwoNlastTailTree(const NodeDetail &node_info, const BroadcastTilingInfo &tiling, const Expr &vl,
                                     const Expr &half_vl, const Expr &block, TernaryOpMap &ternary_ops, Expr &result) {
  const Expr last = tiling.dst_shape.back();
  const Expr block_aligned = af::sym::Mod(last, block);
  auto select = [&ternary_ops](const std::string &name, CondType condition, const Expr &lhs, const Expr &rhs,
                               const Expr &true_value, const Expr &false_value, Expr &selected) {
    return ascendcapi_v2::BuildBroadcastTernary(name, condition, lhs, rhs, true_value, false_value, ternary_ops,
                                                selected);
  };
  Expr gather;
  GE_ASSERT_SUCCESS(select("broadcast_nlast_rank2_total_lt_vl", CondType::K_LT, tiling.dst_size, vl,
                           NlastLeafCost(node_info, tiling, NlastAxisBranch::kGatherOne),
                           NlastLeafCost(node_info, tiling, NlastAxisBranch::kGatherTwo), gather));
  Expr gather_b;
  GE_ASSERT_SUCCESS(select("broadcast_nlast_rank2_total_lt_vl_b", CondType::K_LT, tiling.dst_size, vl,
                           NlastLeafCost(node_info, tiling, NlastAxisBranch::kGatherBOne),
                           NlastLeafCost(node_info, tiling, NlastAxisBranch::kGatherBTwo), gather_b));
  Expr gather_choice;
  GE_ASSERT_SUCCESS(select("broadcast_nlast_rank2_block_aligned_gather", CondType::K_EQ, block_aligned, CreateExpr(0),
                           gather_b, gather, gather_choice));
  Expr less_than_vl;
  GE_ASSERT_SUCCESS(select("broadcast_nlast_rank2_last_le_vl", CondType::K_LE, last, vl,
                           NlastLeafCost(node_info, tiling, NlastAxisBranch::kLessThanVlUnaligned),
                           NlastLeafCost(node_info, tiling, NlastAxisBranch::kLargerThanVlUnaligned), less_than_vl));
  Expr vl_aligned;
  GE_ASSERT_SUCCESS(select("broadcast_nlast_rank2_outer_gt_default_blk", CondType::K_GT, tiling.dst_shape[0], kSymEight,
                           NlastLeafCost(node_info, tiling, NlastAxisBranch::kLargerThanVlAlignedWithVl),
                           NlastLeafCost(node_info, tiling, NlastAxisBranch::kLargerThanVlAlignedWithBlock),
                           vl_aligned));
  Expr larger_aligned;
  GE_ASSERT_SUCCESS(select("broadcast_nlast_rank2_last_mod_vl", CondType::K_EQ, af::sym::Mod(last, vl), CreateExpr(0),
                           vl_aligned, NlastLeafCost(node_info, tiling, NlastAxisBranch::kLargerThanVlAlignedWithBlock),
                           larger_aligned));
  Expr larger;
  GE_ASSERT_SUCCESS(select("broadcast_nlast_rank2_block_aligned_larger", CondType::K_EQ, block_aligned, CreateExpr(0),
                           larger_aligned, NlastLeafCost(node_info, tiling, NlastAxisBranch::kLargerThanVlUnaligned),
                           larger));
  Expr normal;
  GE_ASSERT_SUCCESS(
      select("broadcast_nlast_rank2_last_le_vl_ordered", CondType::K_LE, last, vl, less_than_vl, larger, normal));
  return select("broadcast_nlast_rank2_last_lt_half_vl", CondType::K_LT, last, half_vl, gather_choice, normal, result);
}

af::Status BuildNlastTailTree(const NodeDetail &node_info, const BroadcastTilingInfo &tiling, TernaryOpMap &ternary_ops,
                              Expr &result) {
  Expr vl;
  Expr half_vl;
  Expr block;
  GE_ASSERT_SUCCESS(ascendcapi_v2::GetBroadcastTilingVectorElements(node_info.input_dtype[0], vl, half_vl, block));
  const size_t rank = tiling.rank;
  const Expr last = tiling.dst_shape.back();
  const Expr block_aligned = af::sym::Mod(last, block);

  auto select = [&ternary_ops](const std::string &name, CondType condition, const Expr &lhs, const Expr &rhs,
                               const Expr &true_value, const Expr &false_value, Expr &selected) {
    return ascendcapi_v2::BuildBroadcastTernary(name, condition, lhs, rhs, true_value, false_value, ternary_ops,
                                                selected);
  };
  if (tiling.rank == 2U) {
    return BuildRankTwoNlastTailTree(node_info, tiling, vl, half_vl, block, ternary_ops, result);
  }

  Expr less_than_vl;
  if (rank == 3U) {
    less_than_vl = NlastLeafCost(node_info, tiling, NlastAxisBranch::kLessThanVlUnaligned);
  } else {
    GE_ASSERT_SUCCESS(select("broadcast_nlast_rank_last_le_vl_aligned", CondType::K_EQ, block_aligned, CreateExpr(0),
                             NlastLeafCost(node_info, tiling, NlastAxisBranch::kLessThanVlAligned),
                             NlastLeafCost(node_info, tiling, NlastAxisBranch::kLessThanVlUnaligned), less_than_vl));
  }
  Expr larger;
  GE_ASSERT_SUCCESS(select("broadcast_nlast_rank_block_aligned", CondType::K_EQ, block_aligned, CreateExpr(0),
                           NlastLeafCost(node_info, tiling, NlastAxisBranch::kLargerThanVlAlignedWithVl),
                           NlastLeafCost(node_info, tiling, NlastAxisBranch::kLargerThanVlUnaligned), larger));
  Expr normal;
  GE_ASSERT_SUCCESS(select("broadcast_nlast_rank_last_le_vl", CondType::K_LE, last, vl, less_than_vl, larger, normal));
  if (node_info.input_dtype[0] == kUInt8 || node_info.input_dtype[0] == kInt8) {
    result = normal;
  } else {
    Expr gather;
    GE_ASSERT_SUCCESS(select("broadcast_nlast_rank_last_lt_half_vl", CondType::K_LT, last, half_vl,
                             NlastLeafCost(node_info, tiling, NlastAxisBranch::kGather), normal, gather));
    result = gather;
  }
  return af::SUCCESS;
}

af::Status BuildDynamicNlastTailTree(const NodeDetail &node_info, const BroadcastTilingInfo &tiling,
                                     TernaryOpMap &ternary_ops, Expr &result) {
  Expr vl;
  Expr half_vl;
  Expr block;
  GE_ASSERT_SUCCESS(ascendcapi_v2::GetBroadcastTilingVectorElements(node_info.input_dtype[0], vl, half_vl, block));
  const Expr last = tiling.dst_shape.back();
  const Expr block_aligned = af::sym::Mod(last, block);
  const bool b8 = IsB8Dtype(node_info.input_dtype[0]);
  auto select = [&ternary_ops](const std::string &name, CondType condition, const Expr &lhs, const Expr &rhs,
                               const Expr &true_value, const Expr &false_value, Expr &selected) {
    return ascendcapi_v2::BuildBroadcastTernary(name, condition, lhs, rhs, true_value, false_value, ternary_ops,
                                                selected);
  };
  Expr less;
  GE_ASSERT_SUCCESS(select("broadcast_dynamic_nlast_last_le_vl", CondType::K_LE, last, vl,
                           NlastLeafCost(node_info, tiling, NlastAxisBranch::kDynamicLessThanVlAligned),
                           NlastLeafCost(node_info, tiling, NlastAxisBranch::kDynamicLessThanVlUnaligned), less));
  Expr larger;
  GE_ASSERT_SUCCESS(select("broadcast_dynamic_nlast_block_aligned", CondType::K_EQ, block_aligned, CreateExpr(0),
                           NlastLeafCost(node_info, tiling, NlastAxisBranch::kDynamicLargerThanVlAlignedWithBlock),
                           NlastLeafCost(node_info, tiling, NlastAxisBranch::kDynamicLargerThanVlUnaligned), larger));
  Expr normal;
  GE_ASSERT_SUCCESS(
      select("broadcast_dynamic_nlast_last_le_vl_ordered", CondType::K_LE, last, vl, less, larger, normal));
  if (b8) {
    result = normal;
    return af::SUCCESS;
  }
  return select("broadcast_dynamic_nlast_last_lt_half_vl", CondType::K_LT, last, half_vl,
                NlastLeafCost(node_info, tiling, NlastAxisBranch::kDynamicGather), normal, result);
}

}  // namespace

bool HasUnknownNlastCondition(const NodeDetail &node_info, const BroadcastTilingInfo &tiling) {
  if (tiling.rank < 2U || tiling.dst_shape.empty()) {
    return false;
  }
  Expr vl;
  Expr half_vl;
  Expr block;
  if (ascendcapi_v2::GetBroadcastTilingVectorElements(node_info.input_dtype[0], vl, half_vl, block) != af::SUCCESS) {
    return false;
  }
  const Expr last = tiling.dst_shape.back();
  if (af::SymbolicUtils::StaticCheckLt(last, half_vl) == af::TriBool::kUnknown ||
      af::SymbolicUtils::StaticCheckLe(last, vl) == af::TriBool::kUnknown ||
      af::SymbolicUtils::StaticCheckEq(af::sym::Mod(last, block), CreateExpr(0)) == af::TriBool::kUnknown) {
    return true;
  }
  return tiling.rank == 2U && af::SymbolicUtils::StaticCheckLt(tiling.dst_size, vl) == af::TriBool::kUnknown;
}

ascendcapi_v2::NlastAxisBranch GetNlastAxisPerfBranch(const NodeDetail &node_info) {
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
    return ascendcapi_v2::NlastAxisBranch::kFallback;
  }
  return ascendcapi_v2::GetNlastAxisBranch(tiling, node_info.input_dtype[0],
                                           node_info.broadcast_node_params.const_rank);
}

af::Status BuildNlastAxisPerf(const NodeDetail &node_info, const BroadcastTilingInfo &tiling, PerfOutputInfo &perf) {
  const auto branch =
      ascendcapi_v2::GetNlastAxisBranch(tiling, node_info.input_dtype[0], node_info.broadcast_node_params.const_rank);
  if (tiling.original_rank > 4U && tiling.rank > 4U) {
    GE_ASSERT_SUCCESS(BuildDynamicNlastTailTree(node_info, tiling, perf.ternary_ops, perf.pipe_res[PipeType::AIV_VEC]));
    perf.pipe_res[PipeType::AIV_VEC].Simplify();
    return af::SUCCESS;
  }
  if (HasUnknownNlastCondition(node_info, tiling)) {
    GE_ASSERT_SUCCESS(BuildNlastTailTree(node_info, tiling, perf.ternary_ops, perf.pipe_res[PipeType::AIV_VEC]));
    perf.pipe_res[PipeType::AIV_VEC].Simplify();
    return af::SUCCESS;
  }
  GE_ASSERT_TRUE(branch != ascendcapi_v2::NlastAxisBranch::kFallback, "Unsupported non-last Broadcast branch.");
  VfCostAccumulator acc;
  const std::string dtype = node_info.input_dtype[0];
  Expr vl;
  Expr half_vl;
  Expr block;
  GE_ASSERT_SUCCESS(ascendcapi_v2::GetBroadcastTilingVectorElements(dtype, vl, half_vl, block));
  (void)half_vl;
  (void)block;
  if (branch == NlastAxisBranch::kGather || branch == NlastAxisBranch::kDynamicGather ||
      branch == NlastAxisBranch::kGatherWrapperForFourDim || branch == NlastAxisBranch::kGatherOne ||
      branch == NlastAxisBranch::kGatherTwo || branch == NlastAxisBranch::kGatherBOne ||
      branch == NlastAxisBranch::kGatherBTwo || branch == NlastAxisBranch::kB64MoreDimGather) {
    GE_ASSERT_SUCCESS(AddNlastGatherPerf(node_info, tiling, branch, acc));
  } else {
    GE_ASSERT_SUCCESS(AddNlastDataPerf(node_info, tiling, branch, vl, acc));
  }
  perf.pipe_res[PipeType::AIV_VEC] = ascendcapi_v2::GetVfCost(acc);
  perf.pipe_res[PipeType::AIV_VEC].Simplify();
  return af::SUCCESS;
}

}  // namespace ascendcperf_v2
}  // namespace att
