/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascendc_regbase_perf.h"
namespace att {
namespace ascendcperf_v2 {
RepeatParams CalculateRepeatParams(const std::string &input_dtype, const Expr &cal_count) {
  Expr repeat_elm = kRptSizeFloat;
  auto it = kRptEleMap.find(input_dtype);
  if (it != kRptEleMap.end()) {
    repeat_elm = it->second;
  }
  GE_ASSERT_TRUE(repeat_elm != af::sym::kSymbolZero, "repeat_elm is [%s].",
                 af::SymbolicUtils::ToString(repeat_elm).c_str());
  Expr repeat_time = af::sym::Ceiling(cal_count / repeat_elm);
  return {repeat_elm, repeat_time};
}

namespace {
constexpr uint32_t kIsNanMaxLatency = 26U;
constexpr uint32_t kIsFiniteMaxLatency = 16U;

af::Status RegVfPerf(const std::string &vf_instruct_type, const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("[ATT Reduce] %s node info is %s.", vf_instruct_type.c_str(), node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(vf_instruct_type, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

Expr ProductExprs(const std::vector<ge::Expression> &exprs, size_t end) {
  Expr res = CreateExpr(1);
  for (size_t i = 0U; i < end; ++i) {
    res = res * exprs[i];
  }
  return res;
}

Expr GetCastOneRepeatSize(const NodeDetail &node_info) {
  if (node_info.input_dtype[0] == kUInt8 && node_info.output_dtype[0] == kInt4) {
    return kRptEleMap.at(kFloat16);
  }
  Expr one_rep_size = kRptSizeFloat;
  auto it = kRptEleMap.find(node_info.input_dtype[0]);
  if (it != kRptEleMap.end()) {
    one_rep_size = it->second;
  }
  it = kRptEleMap.find(node_info.output_dtype[0]);
  if (it != kRptEleMap.end()) {
    one_rep_size = af::sym::Min(one_rep_size, it->second);
  }
  return one_rep_size;
}

Expr GetCastCallCount(const NodeDetail &node_info) {
  const auto &cast_params = node_info.cast_node_params;
  if (!cast_params.valid) {
    return CreateExpr(1);
  }
  if (cast_params.output_dims.size() > 1U) {
    return cast_params.output_dims.front();
  }
  return CreateExpr(1);
}

Expr GetNonSameBitCastCallCount(const NodeDetail &node_info, PerfOutputInfo &perf) {
  const auto &cast_params = node_info.cast_node_params;
  if (!cast_params.valid || cast_params.output_dims.size() <= 1U || cast_params.output_strides.empty() ||
      cast_params.input_strides.empty()) {
    return CreateExpr(1);
  }

  const Expr count = cast_params.output_dims.back();
  auto contiguous_case = std::make_shared<IfCase>(CreateExpr(1));
  auto inner_strided_case = std::make_shared<IfCase>(GetCastCallCount(node_info));
  auto inner_stride_case = std::make_shared<IfCase>(CondType::K_EQ, cast_params.output_strides.front(), count,
                                                    std::move(contiguous_case), std::move(inner_strided_case));
  auto outer_strided_case = std::make_shared<IfCase>(GetCastCallCount(node_info));
  Expr call_count = CreateExpr("cast_call_count");
  TernaryOp ternary_op(CondType::K_EQ, cast_params.output_strides.front(), cast_params.input_strides.front(),
                       std::move(inner_stride_case), std::move(outer_strided_case));
  ternary_op.SetVariable(call_count);
  perf.ternary_ops[call_count] = std::move(ternary_op);
  return call_count;
}

Expr GetCastRepeatTime(const NodeDetail &node_info, const Expr &one_rep_size, PerfOutputInfo &perf) {
  const auto &cast_params = node_info.cast_node_params;
  GE_ASSERT_TRUE(one_rep_size != af::sym::kSymbolZero, "one_rep_size is [%s].",
                 af::SymbolicUtils::ToString(one_rep_size).c_str());
  if (!cast_params.valid || cast_params.output_dims.empty() || cast_params.output_strides.empty() ||
      cast_params.input_strides.empty()) {
    Expr input_count = ProductExprs(node_info.input_dims, node_info.input_dims.size());
    return af::sym::Ceiling(af::sym::Div(input_count, one_rep_size));
  }

  const Expr count = cast_params.output_dims.back();
  if (cast_params.output_dims.size() == 1U) {
    return af::sym::Ceiling(af::sym::Div(count, one_rep_size));
  }

  const Expr api_outer = cast_params.output_dims.front();
  GE_ASSERT_TRUE(api_outer != af::sym::kSymbolZero, "api_outer is [%s].",
                 af::SymbolicUtils::ToString(api_outer).c_str());
  const Expr contiguous_repeat = af::sym::Ceiling(af::sym::Div(api_outer * count, one_rep_size));
  const Expr strided_repeat = af::sym::Ceiling(af::sym::Div(count, one_rep_size));

  auto contiguous_case = std::make_shared<IfCase>(contiguous_repeat);
  auto inner_strided_case = std::make_shared<IfCase>(strided_repeat);
  auto inner_stride_case = std::make_shared<IfCase>(CondType::K_EQ, cast_params.output_strides.front(), count,
                                                    std::move(contiguous_case), std::move(inner_strided_case));
  auto outer_strided_case = std::make_shared<IfCase>(strided_repeat);
  Expr repeat_time = CreateExpr("cast_repeat_time");
  TernaryOp ternary_op(CondType::K_EQ, cast_params.output_strides.front(), cast_params.input_strides.front(),
                       std::move(inner_stride_case), std::move(outer_strided_case));
  ternary_op.SetVariable(repeat_time);
  perf.ternary_ops[repeat_time] = std::move(ternary_op);
  return repeat_time;
}

Expr GetSameBitCastRepeatTime(const NodeDetail &node_info, const Expr &one_rep_size) {
  const auto &cast_params = node_info.cast_node_params;
  GE_ASSERT_TRUE(one_rep_size != af::sym::kSymbolZero, "one_rep_size is [%s].",
                 af::SymbolicUtils::ToString(one_rep_size).c_str());
  if (!cast_params.valid || cast_params.output_dims.empty()) {
    Expr input_count = ProductExprs(node_info.input_dims, node_info.input_dims.size());
    return af::sym::Ceiling(af::sym::Div(input_count, one_rep_size));
  }

  return af::sym::Ceiling(af::sym::Div(cast_params.output_dims.back(), one_rep_size));
}

bool IsCastPair(const NodeDetail &node_info, const std::string &input_dtype, const std::string &output_dtype) {
  return node_info.input_dtype[0] == input_dtype && node_info.output_dtype[0] == output_dtype;
}

bool IsSameBitIntegerCast(const NodeDetail &node_info) {
  return IsCastPair(node_info, kInt8, kUInt8) || IsCastPair(node_info, kUInt8, kInt8) ||
         IsCastPair(node_info, kInt16, kUInt16) || IsCastPair(node_info, kUInt16, kInt16) ||
         IsCastPair(node_info, kInt32, kUInt32) || IsCastPair(node_info, kUInt32, kInt32) ||
         IsCastPair(node_info, kInt64, kUInt64) || IsCastPair(node_info, kUInt64, kInt64);
}

bool IsB8Cast(const NodeDetail &node_info) {
  return node_info.input_dtype[0] == kUInt8 &&
         (node_info.output_dtype[0] == kFloat32 || node_info.output_dtype[0] == kInt32 ||
          node_info.output_dtype[0] == kInt16 || node_info.output_dtype[0] == kInt4);
}

bool IsB4Cast(const NodeDetail &node_info) {
  return IsCastPair(node_info, kFloat16, kInt4) || IsCastPair(node_info, kInt4, kFloat16);
}

bool IsB64Cast(const NodeDetail &node_info) {
  return IsCastPair(node_info, kInt64, kFloat32) || IsCastPair(node_info, kFloat32, kInt64) ||
         IsCastPair(node_info, kInt64, kInt32) || IsCastPair(node_info, kInt32, kInt64);
}

bool IsB64TransferCast(const NodeDetail &node_info) {
  return IsCastPair(node_info, kInt64, kFloat16) || IsCastPair(node_info, kFloat16, kInt64) ||
         IsCastPair(node_info, kInt64, kBfloat16);
}

Expr GetUnaryBitWidthChangeCallCount(const NodeDetail &node_info, Expr &cal_count) {
  const auto &params = node_info.unary_bitwidth_change_node_params;
  if (!params.valid) {
    GE_ASSERT_TRUE(!node_info.input_dims.empty(), "Unary bitwidth change input dims are empty.");
    cal_count = ProductExprs(node_info.input_dims, node_info.input_dims.size());
    return CreateExpr(1);
  }
  cal_count = params.cal_count;
  return ProductExprs(params.outer_repeats, params.outer_repeats.size());
}

af::Status AddUnaryBitWidthChangeCommonPerf(const NodeDetail &node_info, Expr repeat_count, Expr &max_latency,
                                            Expr &all_vf_instruct_cost) {
  // IsNan: Actual 6 repeat, Duplicate(2)、UpdateMask(1 repeat)、Select(1 repeat) are not recorded.
  // IsFinite: Actual 8 repeat, Duplicate(2)、UpdateMask(1 repeat)、Select(1 repeat)、CompareScalarEQ(2 repeat) are not
  // recorded.
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_count));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kNe, kUInt8, max_latency, all_vf_instruct_cost, repeat_count));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, kFloat32, max_latency, all_vf_instruct_cost, repeat_count));
  return af::SUCCESS;
}

af::Status FinishUnaryBitWidthChangePerf(Expr max_latency, Expr all_vf_instruct_cost, Expr call_count,
                                         PerfOutputInfo &perf) {
  Expr res = (VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost) * call_count;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

}  // namespace

af::Status IsNanPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GE_ASSERT_TRUE(!node_info.input_dtype.empty() && !node_info.output_dtype.empty());
  Expr cal_count;
  const Expr call_count = GetUnaryBitWidthChangeCallCount(node_info, cal_count);
  const RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  const Expr repeat_time = params.repeat_time;
  const bool output_is_bool = node_info.output_dtype[0] == kBool;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);

  GE_ASSERT_SUCCESS(AddUnaryBitWidthChangeCommonPerf(node_info, repeat_time, max_latency, all_vf_instruct_cost));
  if (output_is_bool) {
    if (node_info.input_dtype[0] == kFloat32) {
      GE_ASSERT_SUCCESS(
          VfPerfUtils::AddVfInstructPerf(kMaskPack, kUInt8, max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
      GE_ASSERT_SUCCESS(
          VfPerfUtils::AddVfInstructPerf(kMaskPack, kUInt8, max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
    } else if (node_info.input_dtype[0] == kFloat16) {
      GE_ASSERT_SUCCESS(
          VfPerfUtils::AddVfInstructPerf(kMaskPack, kUInt8, max_latency, all_vf_instruct_cost, repeat_time));
      GE_ASSERT_SUCCESS(
          VfPerfUtils::AddVfInstructPerf(kMaskPack, kUInt8, max_latency, all_vf_instruct_cost, repeat_time));
    }
  }
  return FinishUnaryBitWidthChangePerf(CreateExpr(kIsNanMaxLatency), all_vf_instruct_cost, call_count, perf);
}

af::Status IsFinitePerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GE_ASSERT_TRUE(!node_info.input_dtype.empty() && !node_info.output_dtype.empty());
  Expr cal_count;
  const Expr call_count = GetUnaryBitWidthChangeCallCount(node_info, cal_count);
  const RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  const Expr repeat_time = params.repeat_time;
  const bool output_is_bool = node_info.output_dtype[0] == kBool;
  const std::string scalar_dtype = node_info.input_dtype[0] == kFloat32 ? kUInt32 : kUInt16;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);

  GE_ASSERT_SUCCESS(AddUnaryBitWidthChangeCommonPerf(node_info, repeat_time, max_latency, all_vf_instruct_cost));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskOr, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_time));
  if (output_is_bool) {
    const Expr mask_pack_count = node_info.input_dtype[0] == kFloat32 ? repeat_time * kSymTwo : repeat_time;
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kMaskPack, kUInt8, max_latency, all_vf_instruct_cost, mask_pack_count));
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kMaskPack, kUInt8, max_latency, all_vf_instruct_cost, mask_pack_count));
  }
  return FinishUnaryBitWidthChangePerf(CreateExpr(kIsFiniteMaxLatency), all_vf_instruct_cost, call_count, perf);
}

namespace {
constexpr uint32_t kTransposeDim2Inner1MaxLatency = 69U;
constexpr uint32_t kTransposeDim3Inner1MaxLatency = 69U;
constexpr uint32_t kTransposeDim3Inner2MaxLatency = 167U;
constexpr uint32_t kTransposeDim4Inner1MaxLatency = 69U;
constexpr uint32_t kTransposeDim4Inner2MaxLatency = 167U;
constexpr uint32_t kTransposeDim4Inner3MaxLatency = 386U;

Expr GetTransposeOuterCount(const ascir_param::TransposeNodeParams &params) {
  if (params.outer_loop_axes.empty()) {
    return CreateExpr(1);
  }
  return ProductExprs(params.outer_loop_axes, params.outer_loop_axes.size());
}

Expr GetTransposeInnerCount(const ascir_param::TransposeNodeParams &params) {
  const size_t outer_dim = params.total_dim - params.inner_dim;
  Expr count = CreateExpr(1);
  for (size_t i = outer_dim; i < params.output_dims.size(); ++i) {
    count = count * params.output_dims[i];
  }
  return count;
}

af::Status AddGenOneInnerDimTransposeIndexPerf(const std::string &data_dtype, const Expr &repeat_time,
                                               Expr &max_latency, Expr &all_vf_instruct_cost) {
  // Actual 3 repeat, UpdateMask(1 repeat) is not recorded.
  // GenOneInnerDimTransposeIndex: MicroAPI::Arange is outside the repeat loop.
  // MicroAPI::Arange generates the index sequence; no exact Reg::Arange performance-table entry exists, so use
  // kPlaceholder.
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kPlaceholder, data_dtype, max_latency, all_vf_instruct_cost, kSymOne));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMuls, data_dtype, max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAdds, data_dtype, max_latency, all_vf_instruct_cost, repeat_time));
  // MicroAPI::DataCopy stores generated indices; no exact performance-table entry exists, so use kPlaceholder.
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kPlaceholder, data_dtype, max_latency, all_vf_instruct_cost, repeat_time));
  return af::SUCCESS;
}

af::Status AddGenTwoInnerDimTransposeIndexPerf(const std::string &data_dtype, const Expr &tail_repeat,
                                               Expr &max_latency, Expr &all_vf_instruct_cost) {
  // Actual 17 + 17 repeat, Duplicate(2)、Add(1 repeat) are not recorded.
  // GenTwoInnerDimTransposeIndex: the first vector is calculated before the tail repeat loop.
  // MicroAPI::Arange generates the index sequence; no exact Reg::Arange performance-table entry exists, so use
  // kPlaceholder.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kArange, kInt32, max_latency, all_vf_instruct_cost, kSymOne));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, data_dtype, max_latency, all_vf_instruct_cost, kSymOne));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCopy, data_dtype, max_latency, all_vf_instruct_cost, kSymOne));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDiv, kFloat32, max_latency, all_vf_instruct_cost, kSymOne));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMul, data_dtype, max_latency, all_vf_instruct_cost, kSymOne));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSub, data_dtype, max_latency, all_vf_instruct_cost, kSymOne));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMuls, data_dtype, max_latency, all_vf_instruct_cost, kSymTwo));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAdd, data_dtype, max_latency, all_vf_instruct_cost, kSymOne));
  // MicroAPI::DataCopy stores generated indices; no exact performance-table entry exists, so use kPlaceholder.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, kUInt8, max_latency, all_vf_instruct_cost, kSymOne));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kUpdateMask, data_dtype, max_latency, all_vf_instruct_cost, tail_repeat));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAdds, data_dtype, max_latency, all_vf_instruct_cost, tail_repeat));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kCompareScalarGE, data_dtype, max_latency, all_vf_instruct_cost, tail_repeat));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kSelect, data_dtype, max_latency, all_vf_instruct_cost, tail_repeat));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMuls, data_dtype, max_latency, all_vf_instruct_cost, tail_repeat));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSub, data_dtype, max_latency, all_vf_instruct_cost, tail_repeat));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAdds, data_dtype, max_latency, all_vf_instruct_cost, tail_repeat));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAdd, data_dtype, max_latency, all_vf_instruct_cost, tail_repeat));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kMuls, data_dtype, max_latency, all_vf_instruct_cost, tail_repeat * kSymTwo));
  // MicroAPI::DataCopy stores generated indices; no exact performance-table entry exists, so use kPlaceholder.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, kUInt8, max_latency, all_vf_instruct_cost, tail_repeat));
  return af::SUCCESS;
}

af::Status AddGenThreeInnerDimTransposeIndexPerf(const std::string &data_dtype, const Expr &tail_repeat,
                                                 Expr &max_latency, Expr &all_vf_instruct_cost) {
  // Actual 17 + 17 repeat, Sub(2 + 2repeat)、Duplicate(2)、Adds(2 repeat)、CompareScalarGE(2 repeat) is not recorded.
  // GenThreeInnerDimTransposeIndex: count the initialization sequence before the tail repeat loop.
  // MicroAPI::Arange generates the index sequence; no exact Reg::Arange performance-table entry exists, so use
  // kPlaceholder.
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kPlaceholder, data_dtype, max_latency, all_vf_instruct_cost, kSymOne));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, data_dtype, max_latency, all_vf_instruct_cost, kSymTwo));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCopy, data_dtype, max_latency, all_vf_instruct_cost, kSymTwo));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDiv, kFloat32, max_latency, all_vf_instruct_cost, kSymThree));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMul, data_dtype, max_latency, all_vf_instruct_cost, kSymThree));
  // MicroAPI::DataCopy stores generated indices; no exact performance-table entry exists, so use kPlaceholder.
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kPlaceholder, data_dtype, max_latency, all_vf_instruct_cost, kSymOne));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kUpdateMask, data_dtype, max_latency, all_vf_instruct_cost, tail_repeat));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kSelect, data_dtype, max_latency, all_vf_instruct_cost, tail_repeat * kSymTwo));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kMuls, data_dtype, max_latency, all_vf_instruct_cost, tail_repeat * kSymFive));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kAdd, data_dtype, max_latency, all_vf_instruct_cost, tail_repeat * kSymFour));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kMuls, data_dtype, max_latency, all_vf_instruct_cost, tail_repeat * kSymThree));
  // MicroAPI::DataCopy stores generated indices; no exact performance-table entry exists, so use kPlaceholder.
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kPlaceholder, data_dtype, max_latency, all_vf_instruct_cost, tail_repeat));
  return af::SUCCESS;
}

af::Status AddTransposeExtendPerf(const std::string &data_dtype, const Expr &repeat_time, const Expr &gather_count,
                                  Expr &max_latency, Expr &all_vf_instruct_cost) {
  // Actual 3 repeat, UpdateMask(1 repeat) is not recorded.
  // TransposeExtendImpl: every vector repeat loads one index vector and gathers/stores one data vector.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, data_dtype, max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kGather2, data_dtype, max_latency, all_vf_instruct_cost,
                                                   repeat_time * gather_count));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, kFloat32, max_latency, all_vf_instruct_cost, repeat_time));
  return af::SUCCESS;
}

af::Status AddTransposeOneOuterDimExtendPerf(const std::string &data_dtype, const Expr &outer_dim0,
                                             const Expr &repeat_time, const Expr &gather_count, Expr &max_latency,
                                             Expr &all_vf_instruct_cost) {
  // Actual 3 repeat, UpdateMask(1 repeat) is not recorded.
  // TransposeOneOuterDimExtendImpl: Gather/Store are inside the dst_dim0 loop.
  const Expr movement_count = repeat_time * outer_dim0;
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, data_dtype, max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kGather2, data_dtype, max_latency, all_vf_instruct_cost,
                                                   movement_count * gather_count));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kStore, kFloat32, max_latency, all_vf_instruct_cost, movement_count));
  return af::SUCCESS;
}

af::Status AddTransposeTwoOuterDimExtendPerf(const std::string &data_dtype, const Expr &outer_dim0,
                                             const Expr &outer_dim1, const Expr &repeat_time, const Expr &gather_count,
                                             Expr &max_latency, Expr &all_vf_instruct_cost) {
  // Actual 3 repeat, UpdateMask(1 repeat) is not recorded.
  // TransposeTwoOuterDimExtendImpl: Gather/Store are inside the dst_dim0 * dst_dim1 loops.
  const Expr movement_count = repeat_time * outer_dim0 * outer_dim1;
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, data_dtype, max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kGather2, data_dtype, max_latency, all_vf_instruct_cost,
                                                   movement_count * gather_count));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kStore, kFloat32, max_latency, all_vf_instruct_cost, movement_count));
  return af::SUCCESS;
}

af::Status AddTransposeThreeOuterDimExtendPerf(const std::string &data_dtype, const Expr &outer_dim0,
                                               const Expr &outer_dim1, const Expr &outer_dim2, const Expr &repeat_time,
                                               const Expr &gather_count, Expr &max_latency,
                                               Expr &all_vf_instruct_cost) {
  // Actual 3 repeat, UpdateMask(1 repeat) is not recorded.
  // TransposeThreeOuterDimExtendImpl: Gather/Store are inside the three nested outer loops.
  const Expr movement_count = repeat_time * outer_dim0 * outer_dim1 * outer_dim2;
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, data_dtype, max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kGather2, data_dtype, max_latency, all_vf_instruct_cost,
                                                   movement_count * gather_count));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kStore, kFloat32, max_latency, all_vf_instruct_cost, movement_count));
  return af::SUCCESS;
}

af::Status AddTransposeDim2Inner1Perf(const NodeDetail &node_info, const std::string &data_dtype,
                                      const Expr &repeat_time, const Expr &gather_count, Expr &max_latency,
                                      Expr &all_vf_instruct_cost) {
  const auto &params = node_info.transpose_node_params;
  GE_ASSERT_SUCCESS(AddGenOneInnerDimTransposeIndexPerf(data_dtype, repeat_time, max_latency, all_vf_instruct_cost));
  GE_ASSERT_SUCCESS(AddTransposeOneOuterDimExtendPerf(data_dtype, params.output_dims[0], repeat_time, gather_count,
                                                      max_latency, all_vf_instruct_cost));
  return af::SUCCESS;
}

af::Status AddTransposeDim3Inner1Perf(const NodeDetail &node_info, const std::string &data_dtype,
                                      const Expr &repeat_time, const Expr &gather_count, Expr &max_latency,
                                      Expr &all_vf_instruct_cost) {
  const auto &params = node_info.transpose_node_params;
  GE_ASSERT_SUCCESS(AddGenOneInnerDimTransposeIndexPerf(data_dtype, repeat_time, max_latency, all_vf_instruct_cost));
  GE_ASSERT_SUCCESS(AddTransposeTwoOuterDimExtendPerf(data_dtype, params.output_dims[0], params.output_dims[1],
                                                      repeat_time, gather_count, max_latency, all_vf_instruct_cost));
  return af::SUCCESS;
}

af::Status AddTransposeDim3Inner2Perf(const NodeDetail &node_info, const std::string &data_dtype,
                                      const Expr &repeat_time, const Expr &tail_repeat, const Expr &gather_count,
                                      Expr &max_latency, Expr &all_vf_instruct_cost) {
  const auto &params = node_info.transpose_node_params;
  GE_ASSERT_SUCCESS(AddGenTwoInnerDimTransposeIndexPerf(data_dtype, tail_repeat, max_latency, all_vf_instruct_cost));
  GE_ASSERT_SUCCESS(AddTransposeOneOuterDimExtendPerf(data_dtype, params.output_dims[0], repeat_time, gather_count,
                                                      max_latency, all_vf_instruct_cost));
  return af::SUCCESS;
}

af::Status AddTransposeDim4Inner1Perf(const NodeDetail &node_info, const std::string &data_dtype,
                                      const Expr &repeat_time, const Expr &gather_count, Expr &max_latency,
                                      Expr &all_vf_instruct_cost) {
  const auto &params = node_info.transpose_node_params;
  GE_ASSERT_SUCCESS(AddGenOneInnerDimTransposeIndexPerf(data_dtype, repeat_time, max_latency, all_vf_instruct_cost));
  GE_ASSERT_SUCCESS(AddTransposeThreeOuterDimExtendPerf(data_dtype, params.output_dims[0], params.output_dims[1],
                                                        params.output_dims[2], repeat_time, gather_count, max_latency,
                                                        all_vf_instruct_cost));
  return af::SUCCESS;
}

af::Status AddTransposeDim4Inner2Perf(const NodeDetail &node_info, const std::string &data_dtype,
                                      const Expr &repeat_time, const Expr &tail_repeat, const Expr &gather_count,
                                      Expr &max_latency, Expr &all_vf_instruct_cost) {
  const auto &params = node_info.transpose_node_params;
  GE_ASSERT_SUCCESS(AddGenTwoInnerDimTransposeIndexPerf(data_dtype, tail_repeat, max_latency, all_vf_instruct_cost));
  GE_ASSERT_SUCCESS(AddTransposeTwoOuterDimExtendPerf(data_dtype, params.output_dims[0], params.output_dims[1],
                                                      repeat_time, gather_count, max_latency, all_vf_instruct_cost));
  return af::SUCCESS;
}

af::Status AddTransposeDim4Inner3Perf(const NodeDetail &node_info, const std::string &data_dtype,
                                      const Expr &repeat_time, const Expr &tail_repeat, const Expr &gather_count,
                                      Expr &max_latency, Expr &all_vf_instruct_cost) {
  const auto &params = node_info.transpose_node_params;
  GE_ASSERT_SUCCESS(AddGenThreeInnerDimTransposeIndexPerf(data_dtype, tail_repeat, max_latency, all_vf_instruct_cost));
  GE_ASSERT_SUCCESS(AddTransposeOneOuterDimExtendPerf(data_dtype, params.output_dims[0], repeat_time, gather_count,
                                                      max_latency, all_vf_instruct_cost));
  return af::SUCCESS;
}

}  // namespace

Expr GetTransposeGatherCount(const NodeDetail &node_info, const std::string &data_dtype, PerfOutputInfo &perf) {
  const auto &params = node_info.transpose_node_params;
  GE_ASSERT_TRUE(!params.input_strides.empty());
  const Expr data_size = kDataTypeSizeMap.at(data_dtype);
  const Expr alignment = kRptSizeHalf / data_size;
  const Expr stride_remainder = af::sym::Mod(params.input_strides.back(), alignment);
  Expr stride_count = CreateExpr("transpose_stride_count");
  perf.ternary_ops[stride_count] =
      TernaryOp(CondType::K_EQ, stride_remainder, CreateExpr(0), alignment, stride_remainder);
  return (params.inner_dim == 1U || params.inner_dim == 2U) ? kSymTwo * stride_count : stride_count + CreateExpr(1);
}

af::Status TransposePerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GE_ASSERT_TRUE(!node_info.input_dtype.empty());
  const auto &params = node_info.transpose_node_params;
  if (!params.valid) {
    return RegVfPerf(kUnitVector, node_info, perf);
  }

  const Expr cal_count = GetTransposeInnerCount(params);
  const RepeatParams repeat_params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  const Expr repeat_time = repeat_params.repeat_time;
  const Expr tail_repeat =
      af::sym::Ceiling(af::sym::Max(cal_count - repeat_params.repeat_elm, CreateExpr(0)) / repeat_params.repeat_elm);
  const Expr outer_count = GetTransposeOuterCount(params);
  const std::string &input_dtype = node_info.input_dtype[0];
  const Expr gather_count = GetTransposeGatherCount(node_info, input_dtype, perf);
  uint32_t latency = 0U;
  Expr all_vf_instruct_cost = CreateExpr(0);
  Expr max_latency = CreateExpr(0);

  if (params.total_dim == 2U && params.inner_dim == 1U) {
    GE_ASSERT_SUCCESS(AddTransposeDim2Inner1Perf(node_info, input_dtype, repeat_time, gather_count, max_latency,
                                                 all_vf_instruct_cost));
    latency = kTransposeDim2Inner1MaxLatency;
  } else if (params.total_dim == 3U && params.inner_dim == 1U) {
    GE_ASSERT_SUCCESS(AddTransposeDim3Inner1Perf(node_info, input_dtype, repeat_time, gather_count, max_latency,
                                                 all_vf_instruct_cost));
    latency = kTransposeDim3Inner1MaxLatency;
  } else if (params.total_dim == 3U && params.inner_dim == 2U) {
    GE_ASSERT_SUCCESS(AddTransposeDim3Inner2Perf(node_info, input_dtype, repeat_time, tail_repeat, gather_count,
                                                 max_latency, all_vf_instruct_cost));
    latency = kTransposeDim3Inner2MaxLatency;
  } else if (params.total_dim == 4U && params.inner_dim == 1U) {
    GE_ASSERT_SUCCESS(AddTransposeDim4Inner1Perf(node_info, input_dtype, repeat_time, gather_count, max_latency,
                                                 all_vf_instruct_cost));
    latency = kTransposeDim4Inner1MaxLatency;
  } else if (params.total_dim == 4U && params.inner_dim == 2U) {
    GE_ASSERT_SUCCESS(AddTransposeDim4Inner2Perf(node_info, input_dtype, repeat_time, tail_repeat, gather_count,
                                                 max_latency, all_vf_instruct_cost));
    latency = kTransposeDim4Inner2MaxLatency;
  } else if (params.total_dim == 4U && params.inner_dim == 3U) {
    GE_ASSERT_SUCCESS(AddTransposeDim4Inner3Perf(node_info, input_dtype, repeat_time, tail_repeat, gather_count,
                                                 max_latency, all_vf_instruct_cost));
    latency = kTransposeDim4Inner3MaxLatency;
  } else {
    if (params.inner_dim == 1U) {
      GE_ASSERT_SUCCESS(
          AddGenOneInnerDimTransposeIndexPerf(input_dtype, repeat_time, max_latency, all_vf_instruct_cost));
      latency = kTransposeDim3Inner1MaxLatency;
    } else if (params.inner_dim == 2U) {
      GE_ASSERT_SUCCESS(
          AddGenTwoInnerDimTransposeIndexPerf(input_dtype, tail_repeat, max_latency, all_vf_instruct_cost));
      latency = kTransposeDim3Inner2MaxLatency;
    } else if (params.inner_dim == 3U) {
      GE_ASSERT_SUCCESS(
          AddGenThreeInnerDimTransposeIndexPerf(input_dtype, tail_repeat, max_latency, all_vf_instruct_cost));
      latency = kTransposeDim4Inner3MaxLatency;
    }
    // inner_dim >= 4: index generation not modeled, only extend cost
    GE_ASSERT_SUCCESS(
        AddTransposeExtendPerf(input_dtype, repeat_time, gather_count, max_latency, all_vf_instruct_cost));
  }

  Expr res = (VfPerfUtils::GetVFHeadCost() + CreateExpr(latency) + all_vf_instruct_cost) * outer_count;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;

  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】Compare Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  分支判断（依据 sizeof (T) 的值）：
  2.1 若 sizeof (T) == 8：
  2.1.1 计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T) * 2
  2.1.2 计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  2.1.3 定义 T = 32
  2.1.4 循环（共 repeatTime 次）：
  根据 mode 执行对应操作：
    case Eq:
    调用 vf_ins_vcmp_eq * 2
    执行 MaskAnd
    case Ne:
    调用 vf_ins_vcmp_ne * 2
    执行 MaskOr
    case Gt:
    调用 vf_ins_vcmp_eq
    调用 vf_ins_vcmp_gt * 2
    调用 vf_ins_vsel
    case Ge:
    调用 vf_ins_vcmp_eq
    调用 vf_ins_vcmp_ge * 2
    调用 vf_ins_vsel
    case Lt:
    调用 vf_ins_vcmp_eq
    调用 vf_ins_vcmp_lt * 2
    调用 vf_ins_vsel
    case Le:
    调用 vf_ins_vcmp_eq
    调用 vf_ins_vcmp_le * 2
    调用 vf_ins_vsel
  2.2 若 sizeof (T) != 8：
  2.2.1 计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  2.2.2 计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  2.2.3 循环（共 repeatTime 次）：
    调用 vf_ins_vcmp (mode)
===========================================================================
*/
namespace {
constexpr int32_t kCompare8BitVfHeadCost = 30;
constexpr int32_t kCompare16BitVfHeadCost = 44;
constexpr int32_t kCompare32BitVfHeadCost = 49;
constexpr int32_t kCompare64BitVfHeadCost = 51;

Expr GetCompareVfHeadCost(const std::string &input_dtype) {
  if (input_dtype == kUInt8 || input_dtype == kInt8) {
    return CreateExpr(kCompare8BitVfHeadCost);
  }
  if (input_dtype == kFloat16 || input_dtype == kBfloat16 || input_dtype == kUInt16 || input_dtype == kInt16) {
    return CreateExpr(kCompare16BitVfHeadCost);
  }
  if (input_dtype == kUInt64 || input_dtype == kInt64) {
    return CreateExpr(kCompare64BitVfHeadCost);
  }
  return CreateExpr(kCompare32BitVfHeadCost);
}

Expr GetCompareBlockCount(const NodeDetail &node_info, const Expr &repeat_elm) {
  const auto &params = node_info.compare_node_params;
  std::vector<Expr> output_dims =
      params.valid && !params.output_dims.empty() ? params.output_dims : node_info.input_dims;
  if (output_dims.empty()) {
    return CreateExpr(1);
  }

  Expr vl_size =
      (node_info.input_dtype[0] == kUInt64 || node_info.input_dtype[0] == kInt64) ? repeat_elm * kSymTwo : repeat_elm;
  Expr counter_first = output_dims.size() > 1U ? output_dims.front() : CreateExpr(1);
  Expr inner_count = output_dims.back();
  return counter_first * af::sym::Ceiling(inner_count / vl_size);
}

}  // namespace

af::Status CompareSpecificPerf(const std::string compare_mode, const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Compare mode[%s]: node info is %s.", compare_mode.c_str(), node_info.ToString().c_str());
  // bool 输入底层按 uint8 执行（位宽相同），指令参数表无 bool，统一按 uint8 建模
  const std::string input_dtype = (node_info.input_dtype[0] == kBool) ? kUInt8 : node_info.input_dtype[0];
  Expr cal_count = node_info.input_dims[kNumZero];
  Expr repeat_elm = kRptSizeFloat;
  auto it = kRptEleMap.find(input_dtype);
  if (it != kRptEleMap.end()) {
    repeat_elm = it->second;
  }
  GE_ASSERT_TRUE(repeat_elm != af::sym::kSymbolZero, "repeat_elm is [%s].",
                 af::SymbolicUtils::ToString(repeat_elm).c_str());
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  Expr block_count = GetCompareBlockCount(node_info, repeat_elm);
  Expr compare_vf_head_cost = GetCompareVfHeadCost(input_dtype);
  GELOGD("cal_count is [%s], repeat_elm is [%s], block_count is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(block_count).c_str());
  // MicroAPI::CreateMask<uint8_t>.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kUInt8, max_latency, all_vf_instruct_cost, kSymOne));
  // MicroAPI::Duplicate(oneAllReg, 1) and Duplicate(zeroAllReg, 0).
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, kUInt8, max_latency, all_vf_instruct_cost, kSymTwo));
  if (node_info.compare_node_params.valid && node_info.compare_node_params.is_scalar) {
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kDuplicate, input_dtype, max_latency, all_vf_instruct_cost, kSymOne));
  }
  // MicroAPI::UpdateMask<uint8_t>(mainBlockCount) and UpdateMask<uint8_t>(counterTail).
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, kUInt8, max_latency, all_vf_instruct_cost, kSymTwo));
  // MicroAPI::DataCopy load src0/src1.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, input_dtype, max_latency, all_vf_instruct_cost, block_count));
  if (!node_info.compare_node_params.valid || !node_info.compare_node_params.is_scalar) {
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kLoad, input_dtype, max_latency, all_vf_instruct_cost, block_count));
  }
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(compare_mode, input_dtype, max_latency, all_vf_instruct_cost, block_count));
  if (input_dtype == kFloat16 || input_dtype == kBfloat16 || input_dtype == kUInt16 || input_dtype == kInt16) {
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kMaskPack, kUInt8, max_latency, all_vf_instruct_cost, block_count));
  }
  if (input_dtype == kFloat32 || input_dtype == kUInt32 || input_dtype == kInt32 || input_dtype == kUInt64 ||
      input_dtype == kInt64) {
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kMaskPack, kUInt8, max_latency, all_vf_instruct_cost, block_count * kSymTwo));
  }
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, kUInt8, max_latency, all_vf_instruct_cost, block_count));
  // MicroAPI::DataCopy store dst.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructDtypeMappingPerf(kStore, input_dtype, kUInt8, max_latency,
                                                               all_vf_instruct_cost, block_count));
  Expr res = compare_vf_head_cost + max_latency + all_vf_instruct_cost;
  res = res * node_info.compare_node_params.outer_call_count;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

af::Status CompareGEPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  return CompareSpecificPerf(kGe, node_info, perf);
}

af::Status CompareEQPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  return CompareSpecificPerf(kEq, node_info, perf);
}

af::Status CompareNEPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  return CompareSpecificPerf(kNe, node_info, perf);
}

af::Status CompareGTPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  return CompareSpecificPerf(kGt, node_info, perf);
}

af::Status CompareLEPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  return CompareSpecificPerf(kLe, node_info, perf);
}

af::Status CompareLTPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  return CompareSpecificPerf(kLt, node_info, perf);
}

/*
===========================================================================
【功能描述】Abs Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vabs
===========================================================================
*/
af::Status AbsPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Abs node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kAbs, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】Exp Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vexp
===========================================================================
*/
af::Status ExpPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Exp node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kExp, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】Ln Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vln
===========================================================================
*/
af::Status LnPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Ln node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kLn, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】Sqrt Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vsqrt
===========================================================================
*/
af::Status SqrtPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Sqrt node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kSqrt, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】Rsqrt Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vsqrt
    调用 vf_ins_vdiv
    调用 vf_ins_vsel
===========================================================================
*/
af::Status RsqrtPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Rsqrt node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kSqrt, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kDiv, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】Div Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vdiv
===========================================================================
*/
af::Status DivPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Div node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kDiv, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】Reciprocal Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vdup
    调用 vf_ins_vdiv
===========================================================================
*/
af::Status ReciprocalPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Reciprocal node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kDiv, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】Relu Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vrelu
===========================================================================
*/
af::Status ReluPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Relu node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kRelu, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】Max Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vmax
===========================================================================
*/
af::Status MaxPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Max node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kMax, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】Min Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vmin
===========================================================================
*/
af::Status MinPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Min node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kMin, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

af::Status ReduceMaxPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  return RegVfPerf(kReduceMax, node_info, perf);
}

af::Status ReduceMinPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  return RegVfPerf(kReduceMin, node_info, perf);
}

/*
===========================================================================
【功能描述】Neg Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vmuls
===========================================================================
*/
af::Status NegPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Neg node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kMuls, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】Mean Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vmuls
===========================================================================
*/
af::Status MeanPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Mean node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr all_vf_instruct_cost = CreateExpr(0);
  Expr max_latency = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kMuls, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】Add Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vadd
===========================================================================
*/
af::Status AddPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Add node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kAdd, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】Sub Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vsub
===========================================================================
*/
af::Status SubPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Sub node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kSub, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】Mul Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vmul
===========================================================================
*/
af::Status MulPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Mul node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kMul, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】LeakyRelu Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vlrelu
===========================================================================
*/
af::Status LeakyReluPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("LeakyRelu node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLeakyRelu, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】Cast Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：oneRepSize = VECTOR_REG_WIDTH / (sizeof(src_T) < sizeof(dst_T) ? sizeof(dst_T) : sizeof(src_T))
  计算总迭代次数：repeatTime = Ceil (count / oneRepSize)
  循环（共 repeatTime 次）：
    若 src_type == int32 且 dst_type == half：
      调用 vf_ins_vcvt(float, src_type)
      调用 vf_ins_vcvt(dst_type, float)
    否则：
      调用 vf_ins_vcvt(dst_type, src_type)
===========================================================================
*/
namespace {
af::Status AddSameBitIntegerCastPerf(const NodeDetail &node_info, const Expr &repeat_time, Expr &max_latency,
                                     Expr &all_vf_instruct_cost) {
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, kSymTwo));
  // MicroAPI::DataCopy (normal load).
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  // MicroAPI::DataCopy (normal store).
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructDtypeMappingPerf(
      kStore, node_info.input_dtype[0], node_info.output_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  return af::SUCCESS;
}

af::Status AddCastDataCopyPerf(const NodeDetail &node_info, const Expr &repeat_time, Expr &max_latency,
                               Expr &all_vf_instruct_cost) {
  // MicroAPI::DataCopy (load).
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  // MicroAPI::DataCopy (store).
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructDtypeMappingPerf(
      kStore, node_info.input_dtype[0], node_info.output_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  return af::SUCCESS;
}

af::Status AddFloat32BoolCastPerf(const Expr &repeat_time, Expr &max_latency, Expr &all_vf_instruct_cost) {
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, kUInt32, max_latency, all_vf_instruct_cost, kSymOne));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAnd, kUInt32, max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAdds, kUInt32, max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kOr, kUInt32, max_latency, all_vf_instruct_cost, repeat_time));
  // MicroAPI::Not.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kNot, kUInt32, max_latency, all_vf_instruct_cost, repeat_time));
  // MicroAPI::ShiftRights.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kVshrs, kUInt32, max_latency, all_vf_instruct_cost, repeat_time));
  return af::SUCCESS;
}

af::Status AddB4CastPerf(const NodeDetail &node_info, const Expr &repeat_time, Expr &max_latency,
                         Expr &all_vf_instruct_cost) {
  const Expr input_size = kDataTypeSizeMap.at(node_info.input_dtype[0]);
  const Expr output_size = kDataTypeSizeMap.at(node_info.output_dtype[0]);
  if (input_size.Compare(output_size) > 0) {
    // CastExtendB4 packs the store mask twice for half -> int4x2_t.
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kMaskPack, kFloat16, max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructDtypeMappingPerf(
        kCast, node_info.input_dtype[0], node_info.output_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  } else if (input_size.Compare(output_size) < 0) {
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kMaskPack, kFloat16, max_latency, all_vf_instruct_cost, repeat_time));
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructDtypeMappingPerf(
        kCast, node_info.input_dtype[0], node_info.output_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  }
  return af::SUCCESS;
}

af::Status AddB64TransferCastPerf(const NodeDetail &node_info, const Expr &repeat_time, Expr &max_latency,
                                  Expr &all_vf_instruct_cost) {
  const Expr input_size = kDataTypeSizeMap.at(node_info.input_dtype[0]);
  const Expr float_size = kDataTypeSizeMap.at(kFloat32);
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, kUInt32, max_latency, all_vf_instruct_cost, kSymOne));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskPack, kUInt32, max_latency, all_vf_instruct_cost, repeat_time));
  if (input_size.Compare(float_size) < 0) {
    // MicroAPI::Interleave.
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kInterleave, kUInt32, max_latency, all_vf_instruct_cost, repeat_time));
  } else {
    // MicroAPI::DeInterleave.
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kDeInterleave, kUInt32, max_latency, all_vf_instruct_cost, repeat_time));
  }
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructDtypeMappingPerf(kCast, node_info.input_dtype[0], kFloat32, max_latency,
                                                               all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructDtypeMappingPerf(kCast, kFloat32, node_info.output_dtype[0], max_latency,
                                                               all_vf_instruct_cost, repeat_time));
  return af::SUCCESS;
}

bool IsB8SpecificCast(const NodeDetail &node_info) {
  return IsB8Cast(node_info) || IsCastPair(node_info, kInt8, kFloat32) || IsCastPair(node_info, kFloat32, kInt8) ||
         IsCastPair(node_info, kInt16, kInt8) || IsCastPair(node_info, kInt16, kUInt8);
}

af::Status AddB8SpecificCastPerf(const NodeDetail &node_info, const Expr &repeat_time, Expr &max_latency,
                                 Expr &all_vf_instruct_cost) {
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructDtypeMappingPerf(kCast, node_info.input_dtype[0], kFloat16, max_latency,
                                                               all_vf_instruct_cost, repeat_time));
  if (IsB8Cast(node_info) &&
      kDataTypeSizeMap.at(kFloat16).Compare(kDataTypeSizeMap.at(node_info.output_dtype[0])) < 0) {
    // MicroAPI::UnPack<uint32_t, uint16_t>.
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUnPack, kUInt16, max_latency, all_vf_instruct_cost, repeat_time));
  }
  if (IsCastPair(node_info, kUInt8, kInt4)) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructDtypeMappingPerf(kCast, node_info.input_dtype[0], kInt4, max_latency,
                                                                 all_vf_instruct_cost, repeat_time));
    // MicroAPI::Pack<uint16_t, uint32_t>.
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPack, kUInt32, max_latency, all_vf_instruct_cost, repeat_time));
    // MicroAPI::Pack<uint8_t, uint16_t>.
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPack, kUInt16, max_latency, all_vf_instruct_cost, repeat_time));
  }
  if (!IsCastPair(node_info, kInt16, kUInt8)) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructDtypeMappingPerf(kCast, kFloat16, node_info.output_dtype[0],
                                                                 max_latency, all_vf_instruct_cost, repeat_time));
  }
  return af::SUCCESS;
}

af::Status AddB64CastPerf(const NodeDetail &node_info, const Expr &repeat_time, Expr &max_latency,
                          Expr &all_vf_instruct_cost) {
  const Expr input_size = kDataTypeSizeMap.at(node_info.input_dtype[0]);
  const Expr output_size = kDataTypeSizeMap.at(node_info.output_dtype[0]);
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructDtypeMappingPerf(
      kCast, node_info.input_dtype[0], node_info.output_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  if (input_size.Compare(output_size) > 0) {
    // MicroAPI::Pack<uint32_t, int64_t>. Pack在输入8B，RegTraitNumOne时，用Duplicate + DeInterleave
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kDuplicate, kUInt32, max_latency, all_vf_instruct_cost, repeat_time));
    // DeInterleave
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kDeInterleave, kUInt32, max_latency, all_vf_instruct_cost, repeat_time));
  }
  return af::SUCCESS;
}

af::Status AddInt64ToUInt8CastPerf(const Expr &repeat_time, Expr &max_latency, Expr &all_vf_instruct_cost) {
  // MicroAPI::Pack<uint32_t, int64_t>. Pack在输入8B，RegTraitNumOne时，用Duplicate + DeInterleave
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kDuplicate, kUInt32, max_latency, all_vf_instruct_cost, repeat_time));
  // DeInterleave
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kDeInterleave, kUInt32, max_latency, all_vf_instruct_cost, repeat_time));
  // MicroAPI::Pack<uint16_t, uint32_t>.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPack, kUInt32, max_latency, all_vf_instruct_cost, repeat_time));
  // MicroAPI::Pack<uint8_t, uint16_t>.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPack, kUInt16, max_latency, all_vf_instruct_cost, repeat_time));
  return af::SUCCESS;
}

af::Status AddUInt8ToInt64CastPerf(const Expr &repeat_time, Expr &max_latency, Expr &all_vf_instruct_cost) {
  // MicroAPI::UnPack<uint64_t, uint32_t>. UnPack在输出8B，RegTraitNumOne时用Duplicate + Interleave
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kDuplicate, kUInt32, max_latency, all_vf_instruct_cost, repeat_time));
  // Interleave
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kInterleave, kUInt32, max_latency, all_vf_instruct_cost, repeat_time));
  return af::SUCCESS;
}

af::Status AddSpecificCastComputePerf(const NodeDetail &node_info, const Expr &repeat_time, Expr &max_latency,
                                      Expr &all_vf_instruct_cost) {
  if (IsCastPair(node_info, kFloat32, kBool)) {
    GE_ASSERT_SUCCESS(AddFloat32BoolCastPerf(repeat_time, max_latency, all_vf_instruct_cost));
  } else if (IsB4Cast(node_info)) {
    GE_ASSERT_SUCCESS(AddB4CastPerf(node_info, repeat_time, max_latency, all_vf_instruct_cost));
  } else if (IsB64TransferCast(node_info)) {
    GE_ASSERT_SUCCESS(AddB64TransferCastPerf(node_info, repeat_time, max_latency, all_vf_instruct_cost));
  } else if (IsB8SpecificCast(node_info)) {
    GE_ASSERT_SUCCESS(AddB8SpecificCastPerf(node_info, repeat_time, max_latency, all_vf_instruct_cost));
  } else if (IsB64Cast(node_info)) {
    GE_ASSERT_SUCCESS(AddB64CastPerf(node_info, repeat_time, max_latency, all_vf_instruct_cost));
  } else if (IsCastPair(node_info, kInt64, kUInt8)) {
    GE_ASSERT_SUCCESS(AddInt64ToUInt8CastPerf(repeat_time, max_latency, all_vf_instruct_cost));
  } else if (IsCastPair(node_info, kUInt8, kInt64)) {
    GE_ASSERT_SUCCESS(AddUInt8ToInt64CastPerf(repeat_time, max_latency, all_vf_instruct_cost));
  } else if (node_info.input_dtype[0] == kInt32 && node_info.output_dtype[0] == kFloat16) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructDtypeMappingPerf(kCast, node_info.input_dtype[0], kFloat32, max_latency,
                                                                 all_vf_instruct_cost, repeat_time));
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructDtypeMappingPerf(kCast, kFloat32, node_info.output_dtype[0],
                                                                 max_latency, all_vf_instruct_cost, repeat_time));
  } else {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructDtypeMappingPerf(
        kCast, node_info.input_dtype[0], node_info.output_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  }
  return af::SUCCESS;
}
}  // namespace

af::Status CastPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Cast node info is %s, input dtype is [%s], output dtype is [%s].", node_info.ToString().c_str(),
         node_info.input_dtype[0].c_str(), node_info.output_dtype[0].c_str());
  Expr one_rep_size = GetCastOneRepeatSize(node_info);
  GE_ASSERT_TRUE(one_rep_size != af::sym::kSymbolZero, "one_rep_size is [%s].",
                 af::SymbolicUtils::ToString(one_rep_size).c_str());
  const bool is_same_bit_integer_cast = IsSameBitIntegerCast(node_info);
  Expr repeat_time = is_same_bit_integer_cast ? GetSameBitCastRepeatTime(node_info, one_rep_size)
                                              : GetCastRepeatTime(node_info, one_rep_size, perf);
  Expr call_count =
      is_same_bit_integer_cast ? GetCastCallCount(node_info) : GetNonSameBitCastCallCount(node_info, perf);
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("one_rep_size is [%s], repeat_time is [%s], cast_param_valid[%d].",
         af::SymbolicUtils::ToString(one_rep_size).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str(),
         static_cast<int32_t>(node_info.cast_node_params.valid));

  if (is_same_bit_integer_cast) {
    GE_ASSERT_SUCCESS(AddSameBitIntegerCastPerf(node_info, repeat_time, max_latency, all_vf_instruct_cost));
  } else {
    GE_ASSERT_SUCCESS(AddCastDataCopyPerf(node_info, repeat_time, max_latency, all_vf_instruct_cost));
    GE_ASSERT_SUCCESS(AddSpecificCastComputePerf(node_info, repeat_time, max_latency, all_vf_instruct_cost));
  }
  Expr res = (VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost) * call_count;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】Sum Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：oneRepSize = VECTOR_REG_WIDTH / sizeof (T)
  分支判断（依据 count 与 oneRepSize 的大小关系）：
  1. 若 count <= oneRepSize：
    调用 ReduceSumCount(1)
  2. 若 count > oneRepSize 且 count <= oneRepSize * oneRepSize：
    计算中间次数：count2 = CeilDiv (count, oneRepSize)
    调用 ReduceSumCount(count2)
    调用 ReduceSumCount(1)
  3. 其他情况（count > oneRepSize * oneRepSize）：
    计算中间次数1：count2 = CeilDiv (count, oneRepSize)
    计算中间次数2：count3 = CeilDiv (count2, oneRepSize)
    调用 ReduceSumCount(count2)
    调用 ReduceSumCount(count3)
    调用 ReduceSumCount(1)

  辅助函数 ReduceSumCount(repeat)：
    循环（共 repeat 次）：
      调用 vf_ins_vcadd
===========================================================================
*/
af::Status SumPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Sum node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  Expr oneRepSize = kRptSizeFloat;
  auto it = kRptEleMap.find(node_info.input_dtype[0]);
  if (it != kRptEleMap.end()) {
    oneRepSize = it->second;
  }
  GE_ASSERT_TRUE(oneRepSize != af::sym::kSymbolZero, "oneRepSize is [%s].",
                 af::SymbolicUtils::ToString(oneRepSize).c_str());
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  Expr repeat = CreateExpr("reduce_sum_node");
  Expr small_repeat = CreateExpr(1U);
  GELOGD("small_repeat is [%s].", af::SymbolicUtils::ToString(small_repeat).c_str());
  Expr mid_repeat = af::sym::Ceiling(cal_count / oneRepSize) + CreateExpr(1U);
  GELOGD("mid_repeat is [%s].", af::SymbolicUtils::ToString(mid_repeat).c_str());
  Expr large_repeat = af::sym::Ceiling(af::sym::Ceiling(cal_count / oneRepSize) / oneRepSize) +
                      af::sym::Ceiling(cal_count / oneRepSize) + CreateExpr(1U);
  GELOGD("large_repeat is [%s].", af::SymbolicUtils::ToString(large_repeat).c_str());
  std::shared_ptr<IfCase> branch_small = std::make_shared<IfCase>(small_repeat);
  GE_ASSERT_NOTNULL(branch_small);
  std::shared_ptr<IfCase> branch_mid = std::make_shared<IfCase>(mid_repeat);
  std::shared_ptr<IfCase> branch_large = std::make_shared<IfCase>(large_repeat);
  GE_ASSERT_NOTNULL(branch_mid);
  GE_ASSERT_NOTNULL(branch_large);
  std::shared_ptr<IfCase> branch_not_small = std::make_shared<IfCase>(
      CondType::K_LE, cal_count, oneRepSize * oneRepSize, std::move(branch_mid), std::move(branch_large));
  GE_ASSERT_NOTNULL(branch_not_small);
  TernaryOp ternary_op =
      TernaryOp(CondType::K_LE, cal_count, oneRepSize, std::move(branch_small), std::move(branch_not_small));
  ternary_op.SetVariable(repeat);
  perf.ternary_ops[repeat] = ternary_op;
  GELOGD("cal_count is [%s], oneRepSize is [%s], repeat is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(oneRepSize).c_str(), af::SymbolicUtils::ToString(repeat).c_str());
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kVcadd, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】RemovePad Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  循环（共 repeatTime 次）：
    调用 vf_ins_vsqz
===========================================================================
*/
af::Status RemovePadPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("RemovePad node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kVsqz, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

namespace {
constexpr int kWhereVfHeadCostB8 = 43;
constexpr int kWhereVfHeadCostB16 = 45;
constexpr int kWhereVfHeadCostB32 = 49;
constexpr int kWhereVfHeadCostB64 = 57;
constexpr int kWhereExtend2DRowOverheadB16 = 4;
constexpr int kWhereExtend2DRowOverheadB32 = 3;
constexpr int kWhereSelectOutputFactorB64 = 2;

Expr GetWhereRepeatElm(const std::string &data_dtype) {
  Expr repeat_elm = kRptSizeFloat;
  auto it = kRptEleMap.find(data_dtype);
  if (it != kRptEleMap.end()) {
    repeat_elm = it->second;
  }
  if (data_dtype == kUInt64 || data_dtype == kInt64) {
    repeat_elm = repeat_elm * kSymTwo;
  }
  return repeat_elm;
}

Expr GetWhereRepeatTime(const Expr &input_count, const Expr &one_rep_size) {
  GE_ASSERT_TRUE(one_rep_size != af::sym::kSymbolZero, "Where one_rep_size is [%s].",
                 af::SymbolicUtils::ToString(one_rep_size).c_str());
  return af::sym::Ceiling(af::sym::Div(input_count, one_rep_size));
}

Expr GetWhereMaskUnPackCount(const std::string &data_dtype, const Expr &block_count) {
  if (data_dtype == kUInt16 || data_dtype == kInt16 || data_dtype == kFloat16 || data_dtype == kBfloat16) {
    return block_count;
  }
  if (data_dtype == kUInt32 || data_dtype == kInt32 || data_dtype == kFloat32 || data_dtype == kUInt64 ||
      data_dtype == kInt64) {
    return block_count * kSymTwo;
  }
  return CreateExpr(0);
}

Expr GetWhereSelectOutputFactor(const NodeDetail &node_info) {
  if (node_info.output_dtype.empty()) {
    return CreateExpr(1);
  }
  const std::string &output_dtype = node_info.output_dtype[0];
  if (output_dtype == kUInt64 || output_dtype == kInt64) {
    return CreateExpr(kWhereSelectOutputFactorB64);
  }
  return CreateExpr(1);
}

bool IsWhereB8B16B32Dtype(const std::string &data_dtype) {
  return data_dtype == kUInt8 || data_dtype == kInt8 || data_dtype == kUInt16 || data_dtype == kInt16 ||
         data_dtype == kFloat16 || data_dtype == kBfloat16 || data_dtype == kUInt32 || data_dtype == kInt32 ||
         data_dtype == kFloat32;
}

bool IsWhereB16B32Dtype(const std::string &data_dtype) {
  return data_dtype == kUInt16 || data_dtype == kInt16 || data_dtype == kFloat16 || data_dtype == kBfloat16 ||
         data_dtype == kUInt32 || data_dtype == kInt32 || data_dtype == kFloat32;
}

bool IsWhereB16Dtype(const std::string &data_dtype) {
  return data_dtype == kUInt16 || data_dtype == kInt16 || data_dtype == kFloat16 || data_dtype == kBfloat16;
}

bool IsWhereB32Dtype(const std::string &data_dtype) {
  return data_dtype == kUInt32 || data_dtype == kInt32 || data_dtype == kFloat32;
}

Expr GetWhereExtend2DRowOverhead(const std::string &data_dtype) {
  if (IsWhereB16Dtype(data_dtype)) {
    return CreateExpr(kWhereExtend2DRowOverheadB16);
  }
  if (IsWhereB32Dtype(data_dtype)) {
    return CreateExpr(kWhereExtend2DRowOverheadB32);
  }
  return CreateExpr(0);
}

Expr GetWhereVfHeadCost(const std::string &data_dtype) {
  if (data_dtype == kUInt8 || data_dtype == kInt8) {
    return CreateExpr(kWhereVfHeadCostB8);
  }
  if (data_dtype == kUInt16 || data_dtype == kInt16 || data_dtype == kFloat16 || data_dtype == kBfloat16) {
    return CreateExpr(kWhereVfHeadCostB16);
  }
  if (data_dtype == kUInt32 || data_dtype == kInt32 || data_dtype == kFloat32) {
    return CreateExpr(kWhereVfHeadCostB32);
  }
  if (data_dtype == kUInt64 || data_dtype == kInt64) {
    return CreateExpr(kWhereVfHeadCostB64);
  }
  return VfPerfUtils::GetVFHeadCost();
}

std::string GetWhereDataDtype(const NodeDetail &node_info) {
  if (!node_info.output_dtype.empty()) {
    return node_info.output_dtype[0];
  }
  if (!node_info.input_dtype.empty()) {
    return node_info.input_dtype[0];
  }
  return "";
}

af::Status AddLegacyWherePerf(const NodeDetail &node_info, Expr &max_latency, Expr &all_vf_instruct_cost) {
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_time = params.repeat_time;
  Expr repeat_elm = params.repeat_elm;
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, CreateExpr(node_info.input_dtype.size())));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarNE, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_time));
  return af::SUCCESS;
}

af::Status AddWhereComputePerf(const NodeDetail &node_info, const std::string &data_dtype, const Expr &block_count,
                               Expr &max_latency, Expr &all_vf_instruct_cost) {
  const auto &params = node_info.where_node_params;
  // MicroAPI::CreateMask<uint8_t>.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kUInt8, max_latency, all_vf_instruct_cost, kSymOne));
  if (!IsWhereB16B32Dtype(data_dtype)) {
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kUpdateMask, kInt32, max_latency, all_vf_instruct_cost, block_count));
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kCompareScalarNE, kUInt8, max_latency, all_vf_instruct_cost, block_count));
  }
  // MicroAPI::DataCopy (mask load).
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, kUInt8, max_latency, all_vf_instruct_cost, block_count));
  const Expr mask_unpack_count = GetWhereMaskUnPackCount(data_dtype, block_count);
  if (mask_unpack_count != af::sym::kSymbolZero) {
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kMaskUnPack, kUInt8, max_latency, all_vf_instruct_cost, mask_unpack_count));
  }
  if (!params.is_bcast_src0) {
    // MicroAPI::DataCopy (src0 load).
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kLoad, data_dtype, max_latency, all_vf_instruct_cost, block_count));
  }
  if (!params.is_bcast_src1) {
    // MicroAPI::DataCopy (src1 load).
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kLoad, data_dtype, max_latency, all_vf_instruct_cost, block_count));
  }
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, kInt32, max_latency, all_vf_instruct_cost,
                                                   block_count * GetWhereSelectOutputFactor(node_info)));
  // MicroAPI::DataCopy (dst store).
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, kInt32, max_latency, all_vf_instruct_cost, block_count));

  return af::SUCCESS;
}

af::Status AddWhereImplPerf(const NodeDetail &node_info, Expr &max_latency, Expr &all_vf_instruct_cost) {
  const auto &params = node_info.where_node_params;
  const std::string &data_dtype = node_info.output_dtype[0];
  const Expr count = params.output_dims.front();
  const Expr repeat_elm = GetWhereRepeatElm(data_dtype);
  const Expr repeat_time = GetWhereRepeatTime(count, repeat_elm);
  GELOGD("WhereImpl count is [%s], repeat_elm is [%s], repeat_time is [%s].",
         af::SymbolicUtils::ToString(count).c_str(), af::SymbolicUtils::ToString(repeat_elm).c_str(),
         af::SymbolicUtils::ToString(repeat_time).c_str());

  // Reg::CreateMask<uint8_t>.
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, kUInt8, max_latency, all_vf_instruct_cost, kSymOne));
  if (!IsWhereB8B16B32Dtype(data_dtype)) {
    // Reg::UpdateMask<T, regTrait>(count).
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kUpdateMask, kInt32, max_latency, all_vf_instruct_cost, repeat_time));
    // Reg::CompareScalar<uint8_t, CMPMODE::NE>.
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kCompareScalarNE, kUInt8, max_latency, all_vf_instruct_cost, repeat_time));
  }
  // Reg::LoadAlign(selReg, conditionUb + i * repeatElm).
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, kUInt8, max_latency, all_vf_instruct_cost, repeat_time));
  const Expr mask_unpack_count = GetWhereMaskUnPackCount(data_dtype, repeat_time);
  if (mask_unpack_count != af::sym::kSymbolZero) {
    // Reg::MaskUnPack.
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kMaskUnPack, kUInt8, max_latency, all_vf_instruct_cost, mask_unpack_count));
  }
  if (!params.is_bcast_src0) {
    // Reg::LoadAlign(src0Reg, src0Ub + i * repeatElm).
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kLoad, data_dtype, max_latency, all_vf_instruct_cost, repeat_time));
  }
  if (!params.is_bcast_src1) {
    // Reg::LoadAlign(src1Reg, src1Ub + i * repeatElm).
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(kLoad, data_dtype, max_latency, all_vf_instruct_cost, repeat_time));
  }
  // Reg::Select(dstReg, src0Reg, src1Reg, selMask).
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, kInt32, max_latency, all_vf_instruct_cost,
                                                   repeat_time * GetWhereSelectOutputFactor(node_info)));
  // Reg::StoreAlign(dstUb + i * repeatElm, dstReg, maskReg).
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, kInt32, max_latency, all_vf_instruct_cost, repeat_time));
  return af::SUCCESS;
}

af::Status AddWhereExtend2DPerf(const NodeDetail &node_info, Expr &max_latency, Expr &all_vf_instruct_cost) {
  const auto &params = node_info.where_node_params;
  const std::string &data_dtype = node_info.output_dtype[0];
  const Expr counter_first = params.output_dims.front();
  const Expr counter_last = params.output_dims.back();
  const Expr repeat_elm = GetWhereRepeatElm(data_dtype);
  const Expr repeat_time = GetWhereRepeatTime(counter_last, repeat_elm);
  GELOGD("WhereExtend counter_first is [%s], counter_last is [%s], repeat_elm is [%s], repeat_time is [%s].",
         af::SymbolicUtils::ToString(counter_first).c_str(), af::SymbolicUtils::ToString(counter_last).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str());
  if (IsWhereB16B32Dtype(data_dtype)) {
    Expr row_max_latency = CreateExpr(0);
    Expr row_vf_instruct_cost = CreateExpr(0);
    GE_ASSERT_SUCCESS(AddWhereComputePerf(node_info, data_dtype, repeat_time, row_max_latency, row_vf_instruct_cost));
    all_vf_instruct_cost =
        all_vf_instruct_cost + (row_vf_instruct_cost + GetWhereExtend2DRowOverhead(data_dtype)) * counter_first;
    return af::SUCCESS;
  }
  const Expr block_count = counter_first * repeat_time;
  return AddWhereComputePerf(node_info, data_dtype, block_count, max_latency, all_vf_instruct_cost);
}

af::Status AddWhereExtendPerf(const NodeDetail &node_info, Expr &max_latency, Expr &all_vf_instruct_cost) {
  const auto &params = node_info.where_node_params;
  GE_ASSERT_TRUE(!params.output_dims.empty(), "Where output dims are empty.");
  GE_ASSERT_TRUE(!node_info.output_dtype.empty(), "Where output dtype is empty.");
  if (params.output_dims.size() == 1U) {
    return AddWhereImplPerf(node_info, max_latency, all_vf_instruct_cost);
  }
  return AddWhereExtend2DPerf(node_info, max_latency, all_vf_instruct_cost);
}
}  // namespace

/*
===========================================================================
【功能描述】Where Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
  计算单次处理元素数：repeatElm = VECTOR_REG_WIDTH / sizeof (T)
  计算总迭代次数：repeatTime = Ceil (count / repeatElm)
  按输入数据类型数量执行向量复制：vf_ins_vdup * node_info.input_dtype.size()
  循环（共 repeatTime 次）：
    调用 vf_ins_vcmps_ne（向量不等于比较）
    调用 vf_ins_vsel（向量选择，按比较结果赋值）
===========================================================================
*/
af::Status WherePerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Where node info is %s.", node_info.ToString().c_str());
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  if (node_info.where_node_params.valid) {
    GE_ASSERT_SUCCESS(AddWhereExtendPerf(node_info, max_latency, all_vf_instruct_cost));
  } else {
    GE_ASSERT_SUCCESS(AddLegacyWherePerf(node_info, max_latency, all_vf_instruct_cost));
  }
  Expr res = (node_info.where_node_params.valid ? GetWhereVfHeadCost(GetWhereDataDtype(node_info))
                                                : VfPerfUtils::GetVFHeadCost()) +
             max_latency + all_vf_instruct_cost;
  if (node_info.where_node_params.valid) {
    res = res * node_info.where_node_params.outer_call_count;
  }
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

inline af::Status ProcessFloatPow(const NodeDetail &node_info, Expr &cal_count, Expr &max_latency,
                                  Expr &all_vf_instruct_cost) {
  Expr eleCountPerVL = kRptSizeFloat;
  Expr repeatTimes = af::sym::Ceiling(cal_count / eleCountPerVL);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(eleCountPerVL).c_str(), af::SymbolicUtils::ToString(repeatTimes).c_str());
  if (node_info.input_dtype[0] != kFloat32) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCast, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                     CreateExpr(kNumTwo)));
  }

  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeatTimes * CreateExpr(kNumFour)));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kAbs, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeatTimes));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarLT, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeatTimes * CreateExpr(kNumThree)));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeatTimes * CreateExpr(kNumTwo)));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSub, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeatTimes * CreateExpr(kNumFour)));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kAnd, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeatTimes));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kCast, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeatTimes));
  // Axpy
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeatTimes * CreateExpr(kNumFour)));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAdds, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeatTimes * CreateExpr(kNumTwo)));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kDiv, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeatTimes));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeatTimes * CreateExpr(kNumEight)));

  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMul, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeatTimes * CreateExpr(kNumFive)));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMuls, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeatTimes * CreateExpr(kNumFour)));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAdd, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeatTimes * CreateExpr(kNumSeven)));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kNeg, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeatTimes * CreateExpr(kNumTwo)));
  // FusedMulDstAdd
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kVmAdd, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeatTimes * CreateExpr(kNumFour)));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMulAddDst, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeatTimes * CreateExpr(kNumFour)));

  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kExp, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeatTimes));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarGE, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeatTimes * CreateExpr(kNumTwo)));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeatTimes));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskOr, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeatTimes));

  // ProcessSpecialCase
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarEQ, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeatTimes * CreateExpr(kNumSeven)));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareNE, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeatTimes * CreateExpr(kNumThree)));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kAnd, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeatTimes));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kOr, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeatTimes * CreateExpr(kNumThree)));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kNeg, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeatTimes));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskOr, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeatTimes * CreateExpr(kNumThree)));
  // MaskNot
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeatTimes));
  // MaskXor
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeatTimes * CreateExpr(kNumTwo)));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeatTimes));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kTruncate, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeatTimes));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeatTimes));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarLT, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeatTimes));
  return af::SUCCESS;
}

inline af::Status ProcessIntegerPow(const NodeDetail &node_info, Expr &cal_count, Expr &max_latency,
                                    Expr &all_vf_instruct_cost) {
  Expr dataSize = af::sym::kSymbolZero;
  auto it = kDataTypeSizeMap.find(node_info.input_dtype[0]);
  if (it != kDataTypeSizeMap.end()) {
    dataSize = it->second;
  }
  GE_ASSERT_TRUE(dataSize != af::sym::kSymbolZero, "dataSize is [%s].", af::SymbolicUtils::ToString(dataSize).c_str());

  /**
   * 指数为Int8类型时，位数为8
   * 指数为非Int8类型时，需要ReduceMax取最大值，maxLoop为最大值占用的位数，不含前导0
   * 当前按默认8位计算
   */
  Expr maxLoop = CreateExpr(kNumEight);
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr eleCountPerVL = params.repeat_elm;
  Expr repeat_times = params.repeat_time;
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(eleCountPerVL).c_str(), af::SymbolicUtils::ToString(repeat_times).c_str());

  // CreateMask
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, CreateExpr(kNumOne)));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, CreateExpr(kNumOne)));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_times));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMul, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_times * maxLoop * CreateExpr(kNumTwo)));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_times * maxLoop));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAnd, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_times * maxLoop));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarEQ, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_times * maxLoop));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_times * maxLoop));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kVshrs, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_times * maxLoop));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_times));

  // ProcessSpecialCase
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarEQ, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_times * CreateExpr(kNumTwo)));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskOr, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_times));
  // MaskXor
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kPlaceholder, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_times));
  if (node_info.input_dtype[0] == kInt8 || node_info.input_dtype[0] == kInt16 || node_info.input_dtype[0] == kInt32) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarNE, node_info.input_dtype[0], max_latency,
                                                     all_vf_instruct_cost, repeat_times));
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarLT, node_info.input_dtype[0], max_latency,
                                                     all_vf_instruct_cost, repeat_times));
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskAnd, node_info.input_dtype[0], max_latency,
                                                     all_vf_instruct_cost, repeat_times));
  }
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】Pow Regbase 版本伪代码 (忽略 Reg 与 UB 间的搬运开销)
  分支判断（依据数据类型分类处理）：
  1. 若为浮点类型（float, half, bfloat16）：
      计算单次处理元素数：eleCountPerVL = 256 / sizeof(float)
      计算总迭代次数：repeatTimes = Ceil (count / eleCountPerVL)
      循环（共 repeatTimes 次）：
        若输入数据类型非 float32：
          调用 vf_ins_vcvt * 2
        调用 vf_ins_vcmps_eq * 6
        调用 vf_ins_vcmps_lt * 2
        调用 vf_ins_vdup * 6
        调用 vf_ins_vxor
        调用 vf_ins_vsel * 5
        调用 vf_ins_vand
        调用 vf_ins_vneg
        调用 vf_ins_vcmp_ne

  2. 若为整数类型（uint8, int8, uint16, int16, uint32, int32）：
      计算最大循环次数：maxLoop = sizeof(T) * 8
      定义向量寄存器宽度：VECTOR_REG_WIDTH = 256
      计算单次处理元素数：eleCountPerVL = VECTOR_REG_WIDTH / sizeof(T)
      计算总迭代次数：repeatTime = Ceil (count / eleCountPerVL)
      调用 vf_ins_vdup
      循环（共 repeatTime 次）：
        内层循环（共 maxLoop 次）：
          调用 vf_ins_vmul * 2
          调用 vf_ins_vdup
          调用 vf_ins_vand
          调用 vf_ins_vcmps_eq
          调用 vf_ins_vsel
          调用 vf_ins_vshrs
        调用 vf_ins_vcmps_eq * 2
        调用 vf_ins_vdup
        调用 vf_ins_vsel
        若为有符号整数类型（int8, int16, int32）：
          调用 vf_ins_vcmps_ne
          调用 vf_ins_vcmps_lt
          调用 vf_ins_vdup
          调用 vf_ins_vsel
===========================================================================
*/
af::Status PowPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Pow node info is %s.", node_info.ToString().c_str());
  Expr cal_count = accumulate(node_info.input_dims.begin(), node_info.input_dims.end(), CreateExpr(1),
                              [](const Expr &a, const Expr &b) { return a * b; });
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  if (node_info.input_dtype[0] == kFloat16 || node_info.input_dtype[0] == kFloat32 ||
      node_info.input_dtype[0] == kBfloat16) {
    GE_ASSERT_SUCCESS(ProcessFloatPow(node_info, cal_count, max_latency, all_vf_instruct_cost));
  } else {
    GE_ASSERT_SUCCESS(ProcessIntegerPow(node_info, cal_count, max_latency, all_vf_instruct_cost));
  }
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】Erf Regbase版本逻辑说明
【输入】dst(输出张量), src(输入张量), cal_count(数据量)
【伪代码】
1. 计算单次处理元素数：repeat_elm = 256 / sizeof B32
2. 计算总迭代次数：repeat_time = 向上取整(cal_count / repeat_elm)
3. 主循环（共repeat_time次）：
   vf_ins_plt UpdateMask
   vf_ins_datacopy_ub2reg
   分支：(T = half)
     vf_ins_vcvt(float, half)
   vf_ins_vmins
   vf_ins_vmaxs
   vf_ins_vmul * 11
   vf_ins_vmuls
   vf_ins_vadds * 10
   vf_ins_vdiv
   分支：(T = half)
     vf_ins_vcvt(half, float)
   vf_ins_datacopy_reg2ub
===========================================================================
*/
af::Status ErfPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr cal_count = node_info.input_dims[kNumZero];
  Expr repeat_elm = kRptSizeFloat;
  Expr repeat_time = af::sym::Ceiling(cal_count / repeat_elm);
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("Erf node[%s], cal_count is [%s], repeat_elm is [%s], repeat_time is [%s], max_latency is [%s].",
         node_info.ToString().c_str(), af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str(),
         af::SymbolicUtils::ToString(max_latency).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  if (node_info.input_dtype[0] == kFloat16) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCast, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                     repeat_time));
  }
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kMins, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kMaxs, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMul, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_time * kSymEleven));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kMuls, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAdds, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_time * kSymTen));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kDiv, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  if (node_info.input_dtype[0] == kFloat16) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCast, kFloat32, max_latency, all_vf_instruct_cost, repeat_time));
  }
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kStore, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】Tanh Regbase版本逻辑说明
【输入】dst(输出张量), src(输入张量), cal_count(数据量)
【伪代码】
1. 计算单次处理元素数：repeat_elm = 256 / sizeof B32
2. 计算总迭代次数：repeat_time = 向上取整(cal_count / repeat_elm)
3. 主循环（共repeat_time次）：
   vf_ins_plt UpdateMask
   vf_ins_datacopy_ub2reg
   分支：(T = half)
     vf_ins_vcvt(float, half)
   vf_ins_vmins
   vf_ins_vmaxs
   vf_ins_vmuls
   vf_ins_vexp
   vf_ins_vadds * 2
   vf_ins_vdiv
   分支：(T = half)
     vf_ins_vcvt(half, float)
   vf_ins_datacopy_reg2ub
===========================================================================
*/
af::Status TanhPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Tanh node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  Expr repeat_elm = kRptSizeFloat;
  Expr repeat_time = af::sym::Ceiling(cal_count / repeat_elm);
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("Tanh node[%s], cal_count is [%s], repeat_elm is [%s], repeat_time is [%s], max_latency is [%s].",
         node_info.ToString().c_str(), af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str(),
         af::SymbolicUtils::ToString(max_latency).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  if (node_info.input_dtype[0] == kFloat16) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCast, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                     repeat_time));
  }
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kMins, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kMaxs, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kMuls, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kExp, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAdds, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_time * kSymTwo));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kDiv, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  if (node_info.input_dtype[0] == kFloat16) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCast, kFloat32, max_latency, all_vf_instruct_cost, repeat_time));
  }
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kStore, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】Sigmoid Regbase版本逻辑说明
【输入】dst(输出张量), src(输入张量), cal_count(数据量)
【伪代码】
1. 计算单次处理元素数：repeat_elm = 256 / sizeof B32
2. 计算总迭代次数：repeat_time = 向上取整(cal_count / repeat_elm)
3. 主循环（共repeat_time次）：
   vf_ins_plt UpdateMask
   vf_ins_datacopy_ub2reg
   分支：(T = half)
     vf_ins_vcvt(float, half)
   vf_ins_vmuls
   vf_ins_vexp
   vf_ins_vadds
   vf_ins_vdup
   vf_ins_vdiv
   分支：(T = half)
     vf_ins_vcvt(half, float)
   vf_ins_datacopy_reg2ub
===========================================================================
*/
af::Status SigmoidPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("Sigmoid node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  Expr repeat_elm = kRptSizeFloat;
  Expr repeat_time = af::sym::Ceiling(cal_count / repeat_elm);
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("Sigmoid node[%s], cal_count is [%s], repeat_elm is [%s], repeat_time is [%s], max_latency is [%s].",
         node_info.ToString().c_str(), af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str(),
         af::SymbolicUtils::ToString(max_latency).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  if (node_info.input_dtype[0] == kFloat16) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCast, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                     repeat_time));
  }
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kMuls, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kExp, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kAdds, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kDiv, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  if (node_info.input_dtype[0] == kFloat16) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCast, kFloat32, max_latency, all_vf_instruct_cost, repeat_time));
  }
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kStore, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】Gelu Regbase版本逻辑说明
【输入】dst(输出张量), src(输入张量), cal_count(数据量)
【伪代码】
1. 计算单次处理元素数：repeat_elm = 256 / sizeof T
2. 计算总迭代次数：repeat_time = 向上取整(cal_count / repeat_elm)
3. 主循环（共repeat_time次）：
   vf_ins_plt UpdateMask
   vf_ins_datacopy_ub2reg
   vf_ins_vmul * 3
   vf_ins_vmuls * 3
   vf_ins_vadd
   vf_ins_vmins
   vf_ins_vexp * 2
   vf_ins_vabs
   vf_ins_vadds
   vf_ins_vdiv
   vf_ins_datacopy_reg2ub
===========================================================================
*/
af::Status GeluPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_elm = params.repeat_elm;
  Expr repeat_time = params.repeat_time;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("Gelu node[%s], cal_count is [%s], repeat_elm is [%s], repeat_time is [%s], max_latency is [%s].",
         node_info.ToString().c_str(), af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str(),
         af::SymbolicUtils::ToString(max_latency).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMul, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_time * kSymThree));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMuls, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_time * kSymThree));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kAdd, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kMins, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kExp, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_time * kSymTwo));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kAbs, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kAdds, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kDiv, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kStore, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】Sign Regbase版本逻辑说明
【输入】dst(输出张量), src(输入张量), cal_count(数据量)
【伪代码】
1. 计算单次处理元素数：repeat_elm = 256 / sizeof T
2. 计算总迭代次数：repeat_time = 向上取整(cal_count / repeat_elm)
3. 初始化：vf_ins_vdup * 3
4. 主循环（共repeat_time次）：
   vf_ins_plt UpdateMask
   vf_ins_datacopy_ub2reg
   vf_ins_vcmps_lt
   vf_ins_vcmps_gt
   vf_ins_vsel * 2
   vf_ins_datacopy_reg2ub
===========================================================================
*/
af::Status SignPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_elm = params.repeat_elm;
  Expr repeat_time = params.repeat_time;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("Sign node[%s], cal_count is [%s], repeat_elm is [%s], repeat_time is [%s], max_latency is [%s].",
         node_info.ToString().c_str(), af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str(),
         af::SymbolicUtils::ToString(max_latency).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, kSymThree));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarLT, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarGT, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_time * kSymTwo));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kStore, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】LogicalNot Regbase版本逻辑说明
【输入】dst(输出张量), src(输入张量), cal_count(数据量)
【伪代码】
1. 计算单次处理元素数：repeat_elm = 256 / sizeof T
2. 计算总迭代次数：repeat_time = 向上取整(cal_count / repeat_elm)
3. 初始化：vf_ins_vdup * 2
4. 主循环（共repeat_time次）：
   vf_ins_plt UpdateMask
   vf_ins_datacopy_ub2reg
   vf_ins_vcmps_eq
   分支：(sizeof(T) = 2)
     vf_ins_ppack MaskPack * 2
   分支：(其他)
     vf_ins_ppack MaskPack * 4
   vf_ins_vsel
   vf_ins_datacopy_reg2ub
===========================================================================
*/
af::Status LogicalNotPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("LogicalNot node info is %s.", node_info.ToString().c_str());
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_elm = params.repeat_elm;
  Expr repeat_time = params.repeat_time;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("LogicalNot node[%s], cal_count is [%s], repeat_elm is [%s], repeat_time is [%s], max_latency is [%s].",
         node_info.ToString().c_str(), af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str(),
         af::SymbolicUtils::ToString(max_latency).c_str());
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, kSymTwo));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarEQ, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_time));
  if (node_info.input_dtype[0] == kFloat16 || node_info.input_dtype[0] == kBfloat16 ||
      node_info.input_dtype[0] == kUInt16 || node_info.input_dtype[0] == kInt16) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskPack, node_info.input_dtype[0], max_latency,
                                                     all_vf_instruct_cost, repeat_time * kSymTwo));
  } else {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskPack, node_info.input_dtype[0], max_latency,
                                                     all_vf_instruct_cost, repeat_time * kSymFour));
  }
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kStore, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】LogicalAnd/Or Regbase版本逻辑说明
【输入】dst(输出张量), src(输入张量), cal_count(数据量)
【伪代码】
1. 计算单次处理元素数：repeat_elm = 256 / sizeof T
2. 计算总迭代次数：repeat_time = 向上取整(cal_count / repeat_elm)
3. 初始化：vf_ins_vdup * 2
4. 主循环（共repeat_time次）：
   vf_ins_plt UpdateMask
   vf_ins_datacopy_ub2reg * 2
   vf_ins_vcmps_ne * 2
   MaskAnd/MaskOr
   分支：(sizeof(T) = 2)
     vf_ins_ppack MaskPack * 2
   分支：(其他)
     vf_ins_ppack MaskPack * 4
   vf_ins_vsel
   vf_ins_datacopy_reg2ub
===========================================================================
*/
inline af::Status LogicalAndOrImpl(const std::string &type, const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_elm = params.repeat_elm;
  Expr repeat_time = params.repeat_time;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("LogicalAndOrImpl node[%s], cal_count is [%s], repeat_elm is [%s], repeat_time is [%s], max_latency is [%s].",
         node_info.ToString().c_str(), af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str(),
         af::SymbolicUtils::ToString(max_latency).c_str());
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, kSymTwo));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_time * kSymTwo));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kCompareScalarNE, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_time * kSymTwo));
  if (type == kMaskAnd) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskAnd, node_info.input_dtype[0], max_latency,
                                                     all_vf_instruct_cost, repeat_time));
  } else {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskOr, node_info.input_dtype[0], max_latency,
                                                     all_vf_instruct_cost, repeat_time));
  }
  if (node_info.input_dtype[0] == kFloat16 || node_info.input_dtype[0] == kBfloat16 ||
      node_info.input_dtype[0] == kUInt16 || node_info.input_dtype[0] == kInt16) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskPack, node_info.input_dtype[0], max_latency,
                                                     all_vf_instruct_cost, repeat_time * kSymTwo));
  } else {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMaskPack, node_info.input_dtype[0], max_latency,
                                                     all_vf_instruct_cost, repeat_time * kSymFour));
  }
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kStore, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

af::Status LogicalOrPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  return LogicalAndOrImpl(kMaskOr, node_info, perf);
}

af::Status LogicalAndPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  return LogicalAndOrImpl(kMaskAnd, node_info, perf);
}

/*
===========================================================================
【功能描述】ClipByValue Regbase版本逻辑说明
【输入】dst(输出张量), src(输入张量), cal_count(数据量)
【伪代码】
1. 计算单次处理元素数：repeat_elm = 256 / sizeof T
2. 计算总迭代次数：repeat_time = 向上取整(cal_count / repeat_elm)
3. 主循环（共repeat_time次）：
   vf_ins_plt UpdateMask
   vf_ins_datacopy_ub2reg * 3
   vf_ins_vmax
   vf_ins_vmin
   vf_ins_datacopy_reg2ub
===========================================================================
*/
af::Status ClipByValuePerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], cal_count);
  Expr repeat_elm = params.repeat_elm;
  Expr repeat_time = params.repeat_time;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("ClipByValue node[%s], cal_count is [%s], repeat_elm is [%s], repeat_time is [%s], max_latency is [%s].",
         node_info.ToString().c_str(), af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str(),
         af::SymbolicUtils::ToString(max_latency).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_time * kSymThree));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kMax, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kMin, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kStore, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】BitwiseAnd Regbase版本逻辑说明
【输入】dst(输出张量), src(输入张量), cal_count(数据量)
【伪代码】
1. 计算单次处理元素数：repeat_elm = 256 / sizeof T
2. 计算总迭代次数：repeat_time = 向上取整(cal_count / repeat_elm) / 2
3. 主循环（共repeat_time次）：
   vf_ins_plt UpdateMask * 2
   vf_ins_datacopy_ub2reg * 4
   vf_ins_vand * 2
   vf_ins_datacopy_reg2ub * 2
4. 后续
   vf_ins_plt UpdateMask
   vf_ins_datacopy_ub2reg * 2
   vf_ins_vand
   vf_ins_datacopy_reg2ub
===========================================================================
*/
af::Status BitwiseAndPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("BitwiseAnd node info is %s.", node_info.ToString().c_str());
  // bool 输入在 regbase BitwiseAndExtend 内部按 uint8 视图执行，perf 按 uint8 建模
  const std::string input_dtype = (node_info.input_dtype[0] == kBool) ? kUInt8 : node_info.input_dtype[0];
  Expr cal_count = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(input_dtype, cal_count);
  Expr repeat_elm = params.repeat_elm;
  Expr repeat_time = params.repeat_time / kSymTwo;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("cal_count is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(cal_count).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, input_dtype, max_latency, all_vf_instruct_cost,
                                                   repeat_time * kSymTwo));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kLoad, input_dtype, max_latency, all_vf_instruct_cost, repeat_time * kSymFour));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kAnd, input_dtype, max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kStore, input_dtype, max_latency, all_vf_instruct_cost, repeat_time * kSymTwo));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kUpdateMask, input_dtype, max_latency, all_vf_instruct_cost, kSymOne));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, input_dtype, max_latency, all_vf_instruct_cost, kSymTwo));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kAnd, input_dtype, max_latency, all_vf_instruct_cost, kSymOne));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kStore, input_dtype, max_latency, all_vf_instruct_cost, kSymOne));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

/*
===========================================================================
【功能描述】FloorDiv Regbase版本逻辑说明
【输入】dst(输出张量), src(输入张量), size(数据量)
【伪代码】
1. 计算单次处理元素数：repeat_elm = 256 / sizeof T
2. 计算总迭代次数：repeat_time = 向上取整(size / repeat_elm)
3. 初始化：vf_ins_vdup * 2
4. 主循环（共repeat_time次）：
   vf_ins_plt UpdateMask
   vf_ins_datacopy_ub2reg * 2
   vf_ins_vdiv
   vf_ins_vtrc
   vf_ins_vcmp_eq
   vf_ins_vmul
   vf_ins_vmula
   vf_ins_vcmp_gt
   vf_ins_vadd
   vf_ins_vsel * 2
   vf_ins_datacopy_reg2ub
===========================================================================
*/
af::Status FloorDivPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("FloorDiv node info is %s.", node_info.ToString().c_str());
  Expr size = node_info.input_dims[kNumZero];
  RepeatParams params = CalculateRepeatParams(node_info.input_dtype[0], size);
  Expr repeat_elm = params.repeat_elm;
  Expr repeat_time = params.repeat_time;
  Expr max_latency = CreateExpr(0);
  Expr all_vf_instruct_cost = CreateExpr(0);
  GELOGD("size is [%s], repeat_elm is [%s], repeat_time is [%s].", af::SymbolicUtils::ToString(size).c_str(),
         af::SymbolicUtils::ToString(repeat_elm).c_str(), af::SymbolicUtils::ToString(repeat_time).c_str());
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kUpdateMask, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, kSymTwo));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kDuplicate, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, kSymTwo));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kLoad, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_time * kSymTwo));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kDiv, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kTruncate, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kEq, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kMul, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kMulAddDst, node_info.input_dtype[0], max_latency,
                                                   all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kGt, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kAdd, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(kSelect, node_info.input_dtype[0], max_latency, all_vf_instruct_cost,
                                                   repeat_time * kSymTwo));
  GE_ASSERT_SUCCESS(
      VfPerfUtils::AddVfInstructPerf(kStore, node_info.input_dtype[0], max_latency, all_vf_instruct_cost, repeat_time));
  Expr res = VfPerfUtils::GetVFHeadCost() + max_latency + all_vf_instruct_cost;
  res.Simplify();
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}
}  // namespace ascendcperf_v2
}  // namespace att
