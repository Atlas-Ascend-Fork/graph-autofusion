/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <numeric>
#include "common/checker.h"
#include "v35/att/api_perf_register/perf_param_v2.h"
#include "api_perf_register/ascendc_api_perf.h"
#include "api_perf_register/api_perf_factory.h"
#include "v35/att/api_perf_register/ascendc_api_perf/reduce_api_perf_v2.h"
#include "v35/att/api_perf_register/ascendc_regbase_perf.h"
namespace att {
namespace ascendcapi_v2 {
namespace {
constexpr int64_t kOneStride = 32768;
constexpr int64_t kVectorBlockBytes = 256;
constexpr int64_t kB64RegTraitNumTwoElements = 64;
constexpr int64_t kReduceVfHeadCost = 64;
constexpr uint32_t kReduceMaxMainRPower = 15U;
// ReduceRAImpl uses a 16-bit stride; wider inner axes take the concat path.
constexpr int64_t kMaxNormalRaStride = std::numeric_limits<uint16_t>::max();

// 调用关系：
// ReduceMin/Max/Any/AllPerf -> ReduceApiPerf -> ValidateReduceContext/ValidateReduceDtype
// -> logical / AR-B64 / AR-normal / RA-B64 / RA-normal 分支建模
// -> BuildMergeCost(多Reduce场景) -> perf_breakdowns 输出 body/merge/total。
using ElementwisePerf = af::Status (*)(const NodeDetail &, PerfOutputInfo &);

enum class ReduceOpKind {
  kMin,
  kMax,
  kAny,
  kAll,
  kSum,
  kProd,
  kMean,
};

struct ReduceOpCostModel {
  ElementwisePerf elementwise_perf;
  std::string reduce_op;
  std::string binary_op;
  ReduceOpKind op_kind;
};

struct VfGroupCounts {
  uint32_t load_count;
  uint32_t binary_count;
  uint32_t store_count;
};

struct ReduceVfGroupCounts {
  uint32_t load_count;
  uint32_t reduce_count;
  uint32_t store_count;
};

PerfBreakdownGroup &GetReduceBreakdownGroup(PerfOutputInfo &perf) {
  constexpr char kReduceBreakdownTitle[] = "Reduce perf breakdown";
  if (perf.perf_breakdowns.empty() || perf.perf_breakdowns.back().title != kReduceBreakdownTitle) {
    PerfBreakdownGroup group;
    group.title = kReduceBreakdownTitle;
    perf.perf_breakdowns.emplace_back(std::move(group));
  }
  return perf.perf_breakdowns.back();
}

void AddReduceBreakdown(PerfOutputInfo &perf, const std::string &name, const Expr &expr, const std::string &desc,
                        uint32_t indent = 0U) {
  GetReduceBreakdownGroup(perf).items.emplace_back(PerfBreakdownItem{name, expr, desc, indent});
}

af::Status ValidateReduceContext(const ReduceApiPerfContext &context) {
  const auto &node_detail = context.node_detail;
  GE_ASSERT_TRUE(!node_detail.input_dtype.empty(), "Reduce dtype is empty.");
  GE_ASSERT_TRUE(!node_detail.output_dtype.empty(), "Reduce output dtype is empty.");
  GE_ASSERT_TRUE(node_detail.input_dims.size() == 2U, "Reduce dims must be {first,last}, node[%s].",
                 node_detail.ToString().c_str());
  GE_ASSERT_TRUE(kDataTypeSizeMap.find(node_detail.input_dtype[0]) != kDataTypeSizeMap.end(),
                 "Reduce dtype size missing, node[%s].", node_detail.ToString().c_str());
  GE_ASSERT_TRUE(kBlkEleMap.find(node_detail.input_dtype[0]) != kBlkEleMap.end(), "Reduce block map missing, node[%s].",
                 node_detail.ToString().c_str());
  GE_ASSERT_TRUE(context.merge_times.IsValid(), "Reduce merge times is invalid, node[%s].",
                 node_detail.ToString().c_str());
  return af::SUCCESS;
}

bool IsB64Dtype(const std::string &dtype) {
  return dtype == kInt64 || dtype == kUInt64;
}

bool IsReduceMinMaxDtype(const std::string &dtype) {
  return dtype == kInt8 || dtype == kUInt8 || dtype == kInt16 || dtype == kUInt16 || dtype == kFloat16 ||
         dtype == kBfloat16 || dtype == kInt32 || dtype == kUInt32 || dtype == kFloat32 || dtype == kInt64 ||
         dtype == kUInt64;
}

bool IsReduceSumDtype(const std::string &dtype) {
  return dtype == kUInt32 || dtype == kInt32 || dtype == kFloat32 || dtype == kInt64 || dtype == kUInt64;
}

bool IsReduceProdDtype(const std::string &dtype) {
  return dtype == kInt8 || dtype == kInt16 || dtype == kInt32 || dtype == kInt64 || dtype == kFloat32;
}

bool IsReduceMeanDtype(const std::string &dtype) {
  return dtype == kFloat32;
}

bool IsLogicalReduceDtype(const std::string &dtype) {
  return dtype == kBool || dtype == kUInt8 || dtype == kInt8 || dtype == kInt16 || dtype == kInt32 || dtype == kInt64 ||
         dtype == kFloat32;
}

std::string GetLogicalReduceExecDtype(const std::string &dtype) {
  return dtype == kBool ? kUInt8 : dtype;
}

std::string GetProdReduceExecDtype(const std::string &dtype) {
  return dtype == kInt8 ? kInt16 : dtype;
}

bool IsLogicalReduce(const ReduceOpCostModel &model) {
  return model.op_kind == ReduceOpKind::kAny || model.op_kind == ReduceOpKind::kAll;
}

bool IsProdReduce(const ReduceOpCostModel &model) {
  return model.op_kind == ReduceOpKind::kProd;
}

bool IsMinMaxReduce(const ReduceOpCostModel &model) {
  return model.op_kind == ReduceOpKind::kMin || model.op_kind == ReduceOpKind::kMax;
}

bool IsSupportedReduceDtype(const std::string &dtype, const ReduceOpCostModel &model) {
  switch (model.op_kind) {
    case ReduceOpKind::kMin:
    case ReduceOpKind::kMax:
      return IsReduceMinMaxDtype(dtype);
    case ReduceOpKind::kAny:
    case ReduceOpKind::kAll:
      return IsLogicalReduceDtype(dtype);
    case ReduceOpKind::kSum:
      return IsReduceSumDtype(dtype);
    case ReduceOpKind::kProd:
      return IsReduceProdDtype(dtype);
    case ReduceOpKind::kMean:
      return IsReduceMeanDtype(dtype);
    default:
      return false;
  }
}

af::Status WarnUnsupportedReduceDtype(const ReduceApiPerfContext &context, const ReduceOpCostModel &model) {
  GELOGW("[ATT Reduce] Reduce op[%s] dtype[%s] is unsupported, node[%s].", model.reduce_op.c_str(),
         context.node_detail.input_dtype[0].c_str(), context.node_detail.name.c_str());
  return af::FAILED;
}

af::Status ValidateReduceDtype(const ReduceApiPerfContext &context, const ReduceOpCostModel &model) {
  const std::string &dtype = context.node_detail.input_dtype[0];
  if (!IsSupportedReduceDtype(dtype, model)) {
    return WarnUnsupportedReduceDtype(context, model);
  }
  return af::SUCCESS;
}

bool IsByteSupportedDtype(const std::string &dtype) {
  return dtype == kInt8 || dtype == kUInt8;
}

Expr PipeBarrierVPerf() {
  return CreateExpr(0);
}

af::Status AddPipeCost(const PerfOutputInfo &src, PipeType pipe_type, Expr &dst) {
  const auto iter = src.pipe_res.find(pipe_type);
  GE_ASSERT_TRUE(iter != src.pipe_res.end());
  dst = dst + iter->second;
  return af::SUCCESS;
}

Expr GetRepeatEle(const std::string &dtype) {
  const auto iter = kRptEleMap.find(dtype);
  return iter == kRptEleMap.end() ? kRptSizeFloat : iter->second;
}

af::Status GetPositiveConstValue(const Expr &value, const std::string &name, int64_t &const_value) {
  GE_ASSERT_TRUE(value.IsConstExpr(), "%s must be const.", name.c_str());
  value.GetConstValue(const_value);
  GE_ASSERT_TRUE(const_value > 0, "%s must be positive, actual value:%ld.", name.c_str(), const_value);
  return af::SUCCESS;
}

Expr CeilDivByPositiveConst(const Expr &value, int64_t divisor) {
  return af::sym::Ceiling(value * af::sym::Rational(1, divisor));
}

af::Status CeilDivByPositiveConstExpr(const Expr &value, const Expr &divisor, const std::string &divisor_name,
                                      Expr &result) {
  int64_t divisor_value = 0;
  GE_ASSERT_SUCCESS(GetPositiveConstValue(divisor, divisor_name, divisor_value));
  result = CeilDivByPositiveConst(value, divisor_value);
  return af::SUCCESS;
}

af::Status AlignUp(const Expr &value, const Expr &alignment, Expr &result) {
  int64_t alignment_value = 0;
  GE_ASSERT_SUCCESS(GetPositiveConstValue(alignment, "alignment", alignment_value));
  result = CeilDivByPositiveConst(value, alignment_value) * CreateExpr(alignment_value);
  return af::SUCCESS;
}

Expr GetVectorBlockEle(const std::string &dtype) {
  return CreateExpr(kVectorBlockBytes) / kDataTypeSizeMap.at(dtype);
}

Expr MakeAlignKey(const Expr &last, const Expr &dtype_size, bool treat_one_as_aligned) {
  Expr mod_result = af::sym::Mod(last * dtype_size, CreateExpr(32));
  return treat_one_as_aligned ? mod_result * (last - CreateExpr(1)) : mod_result;
}

// 计算指定 VF 指令类型发射 instruct_count 次的开销（含 VFHead）
// 内部使用 AddVfInstructPerf 累加语义：latency 取 Max，throughput 累加
af::Status VfOpCost(const std::string &vf_instruct_type, const std::string &dtype, const Expr &count,
                    const Expr &repeat_ele, uint32_t instruct_count, Expr &cost) {
  if (instruct_count == 0U) {
    cost = CreateExpr(0);
    return af::SUCCESS;
  }
  Expr max_latency = CreateExpr(0);
  Expr throughput = CreateExpr(0);
  Expr repeat_time = CreateExpr(0);
  GE_ASSERT_SUCCESS(CeilDivByPositiveConstExpr(count, repeat_ele, "repeat_ele", repeat_time));
  for (uint32_t i = 0; i < instruct_count; i++) {
    GE_ASSERT_SUCCESS(VfPerfUtils::AddVfInstructPerf(vf_instruct_type, dtype, max_latency, throughput, repeat_time));
  }
  cost = VfPerfUtils::GetVFHeadCost() + max_latency + throughput;
  cost.Simplify();
  return af::SUCCESS;
}

struct VfCostAccumulator {
  Expr max_latency = CreateExpr(0);
  Expr throughput = CreateExpr(0);
};

af::Status AddVfInstructCost(const std::string &vf_instruct_type, const std::string &dtype, const Expr &repeat_time,
                             uint32_t instruct_count, VfCostAccumulator &acc) {
  for (uint32_t i = 0; i < instruct_count; ++i) {
    GE_ASSERT_SUCCESS(
        VfPerfUtils::AddVfInstructPerf(vf_instruct_type, dtype, acc.max_latency, acc.throughput, repeat_time));
  }
  return af::SUCCESS;
}

Expr GetVfGroupCost(const VfCostAccumulator &acc) {
  Expr cost = VfPerfUtils::GetVFHeadCost() + acc.max_latency + acc.throughput;
  cost.Simplify();
  return cost;
}

Expr GetVfGroupBodyCost(const VfCostAccumulator &acc) {
  Expr cost = acc.max_latency + acc.throughput;
  cost.Simplify();
  return cost;
}

Expr GetB64RepeatEle(const std::string &dtype) {
  (void)dtype;
  return CreateExpr(kB64RegTraitNumTwoElements);
}

af::Status GetRaB64VectorBlockCount(const Expr &dim_a, const std::string &dtype, Expr &count) {
  return AlignUp(dim_a, GetB64RepeatEle(dtype), count);
}

af::Status GetPerInnerAxisVfCount(const Expr &outer, const Expr &inner, const Expr &repeat_ele, Expr &count) {
  Expr aligned_inner = CreateExpr(0);
  GE_ASSERT_SUCCESS(AlignUp(inner, repeat_ele, aligned_inner));
  count = outer * aligned_inner;
  return af::SUCCESS;
}

af::Status BuildVfGroupCost(const ReduceOpCostModel &model, const std::string &dtype, const Expr &count,
                            const Expr &repeat_ele, uint32_t load_count, uint32_t binary_count, uint32_t store_count,
                            Expr &cost) {
  Expr load_cost = CreateExpr(0);
  Expr binary_cost = CreateExpr(0);
  Expr store_cost = CreateExpr(0);
  GE_ASSERT_SUCCESS(VfOpCost(kLoad, dtype, count, repeat_ele, load_count, load_cost));
  GE_ASSERT_SUCCESS(VfOpCost(model.binary_op, dtype, count, repeat_ele, binary_count, binary_cost));
  GE_ASSERT_SUCCESS(VfOpCost(kStore, dtype, count, repeat_ele, store_count, store_cost));
  cost = load_cost + binary_cost + store_cost;
  cost.Simplify();
  return af::SUCCESS;
}

af::Status BuildVfGroupBodyCost(const ReduceOpCostModel &model, const std::string &dtype, const Expr &count,
                                const Expr &repeat_ele, uint32_t load_count, uint32_t binary_count,
                                uint32_t store_count, Expr &cost) {
  Expr repeat_time = CreateExpr(0);
  GE_ASSERT_SUCCESS(CeilDivByPositiveConstExpr(count, repeat_ele, "repeat_ele", repeat_time));
  VfCostAccumulator acc;
  GE_ASSERT_SUCCESS(AddVfInstructCost(kLoad, dtype, repeat_time, load_count, acc));
  GE_ASSERT_SUCCESS(AddVfInstructCost(model.binary_op, dtype, repeat_time, binary_count, acc));
  GE_ASSERT_SUCCESS(AddVfInstructCost(kStore, dtype, repeat_time, store_count, acc));
  cost = acc.max_latency + acc.throughput;
  cost.Simplify();
  return af::SUCCESS;
}

af::Status AddB64BinaryFuncInstructCost(const ReduceOpCostModel &model, const Expr &repeat_time, uint32_t binary_count,
                                        VfCostAccumulator &acc) {
  for (uint32_t i = 0; i < binary_count; ++i) {
    if (model.op_kind == ReduceOpKind::kSum) {
      GE_ASSERT_SUCCESS(AddVfInstructCost(kVcadd, kUInt32, repeat_time, 3U, acc));
      GE_ASSERT_SUCCESS(AddVfInstructCost(kAdd, kUInt32, repeat_time, 2U, acc));
      GE_ASSERT_SUCCESS(AddVfInstructCost(kAnd, kUInt32, repeat_time, 3U, acc));
      GE_ASSERT_SUCCESS(AddVfInstructCost(kVshrs, kUInt32, repeat_time, 2U, acc));
      continue;
    }
    const std::string compare_op = (model.binary_op == kMin) ? kLt : kGt;
    GE_ASSERT_SUCCESS(AddVfInstructCost(kEq, kUInt32, repeat_time, 1U, acc));
    GE_ASSERT_SUCCESS(AddVfInstructCost(compare_op, kUInt32, repeat_time, 2U, acc));
    GE_ASSERT_SUCCESS(AddVfInstructCost(kSelect, kUInt32, repeat_time, 5U, acc));
    GE_ASSERT_SUCCESS(AddVfInstructCost(kDuplicate, kUInt32, repeat_time, 2U, acc));
  }
  return af::SUCCESS;
}

af::Status BuildB64BinaryFuncCost(const ReduceOpCostModel &model, const Expr &repeat_time, Expr &cost) {
  VfCostAccumulator acc;
  GE_ASSERT_SUCCESS(AddB64BinaryFuncInstructCost(model, repeat_time, 1U, acc));
  cost = GetVfGroupCost(acc);
  return af::SUCCESS;
}

af::Status BuildB64BinaryCost(const ReduceOpCostModel &model, const Expr &count, const Expr &repeat_ele,
                              uint32_t binary_count, Expr &cost) {
  if (binary_count == 0U) {
    cost = CreateExpr(0);
    return af::SUCCESS;
  }
  int64_t repeat_ele_value = 0;
  GE_ASSERT_SUCCESS(GetPositiveConstValue(repeat_ele, "repeat_ele", repeat_ele_value));
  Expr binary_func_cost = CreateExpr(0);
  GE_ASSERT_SUCCESS(BuildB64BinaryFuncCost(model, CeilDivByPositiveConst(count, repeat_ele_value), binary_func_cost));
  cost = binary_func_cost * CreateExpr(static_cast<int64_t>(binary_count));
  cost.Simplify();
  return af::SUCCESS;
}

af::Status BuildB64ReduceSumCost(const Expr &count, const Expr &repeat_ele, uint32_t reduce_count, Expr &cost) {
  return BuildB64BinaryCost({nullptr, kVcadd, kAdd, ReduceOpKind::kSum}, count, repeat_ele, reduce_count, cost);
}

af::Status BuildB64VfGroupCost(const ReduceOpCostModel &model, const std::string &dtype, const Expr &count,
                               uint32_t load_count, uint32_t binary_count, uint32_t store_count, Expr &cost) {
  if (!IsMinMaxReduce(model) && model.op_kind != ReduceOpKind::kSum) {
    const Expr repeat_ele = GetB64RepeatEle(dtype);
    return BuildVfGroupCost(model, dtype, count, repeat_ele, load_count, binary_count, store_count, cost);
  }
  const Expr repeat_ele = GetB64RepeatEle(dtype);
  Expr load_cost = CreateExpr(0);
  Expr binary_cost = CreateExpr(0);
  Expr store_cost = CreateExpr(0);
  GE_ASSERT_SUCCESS(VfOpCost(kLoad, dtype, count, repeat_ele, load_count, load_cost));
  if (model.op_kind == ReduceOpKind::kSum) {
    GE_ASSERT_SUCCESS(BuildB64ReduceSumCost(count, repeat_ele, binary_count, binary_cost));
  } else {
    GE_ASSERT_SUCCESS(BuildB64BinaryCost(model, count, repeat_ele, binary_count, binary_cost));
  }
  GE_ASSERT_SUCCESS(VfOpCost(kStore, dtype, count, repeat_ele, store_count, store_cost));
  cost = load_cost + binary_cost + store_cost;
  cost.Simplify();
  return af::SUCCESS;
}

af::Status BuildB64VfGroupBodyCost(const ReduceOpCostModel &model, const std::string &dtype, const Expr &count,
                                   uint32_t load_count, uint32_t binary_count, uint32_t store_count, Expr &cost) {
  const Expr repeat_ele = GetB64RepeatEle(dtype);
  Expr repeat_time = CreateExpr(0);
  GE_ASSERT_SUCCESS(CeilDivByPositiveConstExpr(count, repeat_ele, "repeat_ele", repeat_time));
  VfCostAccumulator acc;
  GE_ASSERT_SUCCESS(AddVfInstructCost(kLoad, dtype, repeat_time, load_count, acc));
  if (!IsMinMaxReduce(model) && model.op_kind != ReduceOpKind::kSum) {
    GE_ASSERT_SUCCESS(AddVfInstructCost(model.binary_op, dtype, repeat_time, binary_count, acc));
  } else {
    GE_ASSERT_SUCCESS(AddB64BinaryFuncInstructCost(model, repeat_time, binary_count, acc));
  }
  GE_ASSERT_SUCCESS(AddVfInstructCost(kStore, dtype, repeat_time, store_count, acc));
  cost = GetVfGroupBodyCost(acc);
  return af::SUCCESS;
}

af::Status BuildReduceVfGroupCost(const ReduceOpCostModel &model, const std::string &dtype, const Expr &count,
                                  const Expr &repeat_ele, const ReduceVfGroupCounts &counts, Expr &cost) {
  Expr load_cost = CreateExpr(0);
  Expr reduce_cost = CreateExpr(0);
  Expr store_cost = CreateExpr(0);
  GE_ASSERT_SUCCESS(VfOpCost(kLoad, dtype, count, repeat_ele, counts.load_count, load_cost));
  if (model.op_kind == ReduceOpKind::kSum && IsB64Dtype(dtype)) {
    GE_ASSERT_SUCCESS(BuildB64ReduceSumCost(count, repeat_ele, counts.reduce_count, reduce_cost));
  } else {
    GE_ASSERT_SUCCESS(VfOpCost(model.reduce_op, dtype, count, repeat_ele, counts.reduce_count, reduce_cost));
  }
  GE_ASSERT_SUCCESS(VfOpCost(kStore, dtype, count, repeat_ele, counts.store_count, store_cost));
  cost = load_cost + reduce_cost + store_cost;
  cost.Simplify();
  return af::SUCCESS;
}

af::Status BuildCopyOutCost(const ReduceOpCostModel &model, const std::string &dtype, const Expr &count,
                            const Expr &repeat_ele, bool b64, Expr &cost) {
  if (b64) {
    return BuildB64VfGroupCost(model, dtype, count, 1U, 0U, 1U, cost);
  }
  return BuildVfGroupCost(model, dtype, count, repeat_ele, 1U, 0U, 1U, cost);
}

af::Status BuildRepeatedVfGroupCost(const Expr &repeats, const std::function<af::Status(Expr &)> &body_builder,
                                    Expr &cost) {
  Expr body_cost = CreateExpr(0);
  GE_ASSERT_SUCCESS(body_builder(body_cost));
  cost = af::sym::Min(repeats, CreateExpr(1)) * CreateExpr(kReduceVfHeadCost) + repeats * body_cost;
  cost.Simplify();
  return af::SUCCESS;
}

bool IsPositiveConst(const Expr &value) {
  if (!value.IsConstExpr()) {
    return true;
  }
  int64_t const_value = 0;
  value.GetConstValue(const_value);
  return const_value > 0;
}

struct FoldFlags {
  Expr foldZero;
  Expr foldOne;
  Expr foldTwo;
  Expr foldThree;
};

struct TreeReduceParams {
  Expr mainR;      // largest power of 2 ≤ first
  Expr tailR;      // first - mainR
  Expr folds;      // log2(mainR)
  Expr mainTimes;  // folds / avgFolds
  Expr tailFolds;  // folds % avgFolds
  FoldFlags flags;
  Expr legacyFoldOne;
  Expr legacyFoldZero;
};

struct NormalSymbolicTreeSpec {
  std::string dtype;
  Expr dim_r;
  Expr count;
  Expr repeat_ele;
  uint32_t inplace_load_count;
  uint32_t main_load_count;
  uint32_t tail_load_count;
  uint32_t store_count;
  uint32_t inplace_store_count = 0U;
  uint32_t main_store_count = 0U;
  uint32_t tail_store_count = 0U;
};

struct RaNormalContext {
  Expr dim_r;
  Expr dim_a;
  std::string dtype;
  Expr repeat_ele;
};

using TreeCostBuilder = std::function<af::Status(uint32_t, const Expr &, Expr &)>;
using TailInplaceCostBuilder = std::function<af::Status(const TreeReduceParams &, Expr &)>;

Expr MakeBoolExpr(bool value) {
  return CreateExpr(value ? 1 : 0);
}

uint32_t CalculateMainRConst(uint32_t dim_r) {
  uint32_t main_r = 1U;
  while (main_r <= (dim_r / 2U)) {
    main_r <<= 1U;
  }
  return main_r;
}

uint32_t CalculateFoldsConst(uint32_t main_r) {
  uint32_t folds = 0U;
  while (main_r > 1U) {
    main_r >>= 1U;
    folds++;
  }
  return folds;
}

FoldFlags CalculateFoldFlagsConst(uint16_t tail_folds, bool supports_fold_three) {
  FoldFlags flags;
  flags.foldZero = MakeBoolExpr(tail_folds == 0U);
  flags.foldOne = MakeBoolExpr(tail_folds == 1U);
  flags.foldTwo = MakeBoolExpr(tail_folds == 2U);
  flags.foldThree = MakeBoolExpr(supports_fold_three && tail_folds == 3U);
  return flags;
}

void SetLegacyFoldFlags(uint32_t tail_folds, TreeReduceParams &params) {
  params.legacyFoldOne = MakeBoolExpr(tail_folds >= 1U);
  params.legacyFoldZero = MakeBoolExpr(tail_folds >= 2U);
}

af::Status MakeTreeReduceParams(uint32_t main_r, const Expr &tail_r, uint32_t avg_folds, bool supports_fold_three,
                                TreeReduceParams &params) {
  GE_ASSERT_TRUE(avg_folds > 0U, "avg_folds[%u] is invalid, must be positive.", avg_folds);
  const uint32_t folds = CalculateFoldsConst(main_r);
  const uint32_t tail_folds = folds % avg_folds;
  params.mainR = CreateExpr(static_cast<int64_t>(main_r));
  params.tailR = tail_r;
  params.folds = CreateExpr(static_cast<int64_t>(folds));
  params.mainTimes = CreateExpr(static_cast<int64_t>(folds / avg_folds));
  params.tailFolds = CreateExpr(static_cast<int64_t>(tail_folds));
  params.flags = CalculateFoldFlagsConst(static_cast<uint16_t>(tail_folds), supports_fold_three);
  SetLegacyFoldFlags(tail_folds, params);
  return af::SUCCESS;
}

std::shared_ptr<IfCase> MakeLeafCase(const Expr &expr) {
  return std::make_shared<IfCase>(expr);
}

af::Status BuildSymbolicTreeCurrentCase(const Expr &dim_r, const TreeCostBuilder &builder, uint32_t main_r,
                                        bool split_exact_power, std::shared_ptr<IfCase> &case_expr) {
  Expr normal_cost = CreateExpr(0);
  GE_ASSERT_SUCCESS(builder(main_r, dim_r - CreateExpr(static_cast<int64_t>(main_r)), normal_cost));
  if (!split_exact_power || main_r <= 1U) {
    case_expr = MakeLeafCase(normal_cost);
    return af::SUCCESS;
  }

  Expr exact_power_cost = CreateExpr(0);
  GE_ASSERT_SUCCESS(builder(main_r / 2U, CreateExpr(static_cast<int64_t>(main_r / 2U)), exact_power_cost));
  case_expr = std::make_shared<IfCase>(CondType::K_EQ, dim_r, CreateExpr(static_cast<int64_t>(main_r)),
                                       MakeLeafCase(exact_power_cost), MakeLeafCase(normal_cost));
  return af::SUCCESS;
}

af::Status BuildSymbolicTreeCostCase(const Expr &dim_r, const TreeCostBuilder &builder, uint32_t power,
                                     bool split_exact_power, std::shared_ptr<IfCase> &case_expr) {
  const uint32_t main_r = 1U << power;
  std::shared_ptr<IfCase> cur_case;
  GE_ASSERT_SUCCESS(BuildSymbolicTreeCurrentCase(dim_r, builder, main_r, split_exact_power, cur_case));
  if (power == kReduceMaxMainRPower) {
    case_expr = std::move(cur_case);
    return af::SUCCESS;
  }

  std::shared_ptr<IfCase> next_case;
  GE_ASSERT_SUCCESS(BuildSymbolicTreeCostCase(dim_r, builder, power + 1U, split_exact_power, next_case));
  case_expr = std::make_shared<IfCase>(CondType::K_LT, dim_r, CreateExpr(static_cast<int64_t>(1U << (power + 1U))),
                                       std::move(cur_case), std::move(next_case));
  return af::SUCCESS;
}

std::vector<Expr> GetFreeSymbols(const Expr &expr) {
  std::vector<Expr> symbols;
  for (const auto &arg : expr.FreeSymbols()) {
    symbols.emplace_back(arg);
  }
  return symbols;
}

af::Status SelectSymbolicTreeCost(const std::string &var_name, const Expr &dim_r, const TreeCostBuilder &builder,
                                  PerfOutputInfo &perf, Expr &cost, bool split_exact_power = false) {
  std::shared_ptr<IfCase> root_case;
  GE_ASSERT_SUCCESS(BuildSymbolicTreeCostCase(dim_r, builder, 0U, split_exact_power, root_case));
  cost = CreateExpr(var_name.c_str());
  TernaryOp ternary_op(cost, std::move(root_case), GetFreeSymbols(dim_r));
  perf.ternary_ops[cost] = ternary_op;
  return af::SUCCESS;
}

uint32_t GetInplaceStoreCount(const NormalSymbolicTreeSpec &spec) {
  return spec.inplace_store_count == 0U ? spec.store_count : spec.inplace_store_count;
}

uint32_t GetMainStoreCount(const NormalSymbolicTreeSpec &spec) {
  return spec.main_store_count == 0U ? spec.store_count : spec.main_store_count;
}

uint32_t GetTailStoreCount(const NormalSymbolicTreeSpec &spec) {
  return spec.tail_store_count == 0U ? spec.store_count : spec.tail_store_count;
}

af::Status CalculateTreeReduceParams(const Expr &dim_r, uint32_t avg_folds, bool supports_fold_three,
                                     TreeReduceParams &params) {
  GE_ASSERT_TRUE(avg_folds > 0U, "avg_folds[%u] is invalid, must be positive.", avg_folds);
  if (dim_r.IsConstExpr()) {
    int64_t dim_r_val = 0;
    dim_r.GetConstValue(dim_r_val);
    GE_ASSERT_TRUE(dim_r_val > 0, "Reduce RA dimR[%ld] is invalid, must be positive.", dim_r_val);
    const uint32_t main_r = CalculateMainRConst(static_cast<uint32_t>(dim_r_val));
    const uint32_t folds = CalculateFoldsConst(main_r);
    const uint32_t main_times = folds / avg_folds;
    const uint32_t tail_folds = folds % avg_folds;
    params.mainR = CreateExpr(static_cast<int64_t>(main_r));
    params.tailR = CreateExpr(dim_r_val - static_cast<int64_t>(main_r));
    params.folds = CreateExpr(static_cast<int64_t>(folds));
    params.mainTimes = CreateExpr(static_cast<int64_t>(main_times));
    params.tailFolds = CreateExpr(static_cast<int64_t>(tail_folds));
    params.flags = CalculateFoldFlagsConst(static_cast<uint16_t>(tail_folds), supports_fold_three);
    SetLegacyFoldFlags(tail_folds, params);
    return af::SUCCESS;
  }

  GELOGW("[ATT Reduce] dim_r is symbolic, using conservative tree reduce params.");
  params.mainR = dim_r;
  params.tailR = CreateExpr(0);
  params.folds = CreateExpr(0);
  params.mainTimes = CreateExpr(0);
  params.tailFolds = CreateExpr(0);
  params.flags = CalculateFoldFlagsConst(0U, supports_fold_three);
  SetLegacyFoldFlags(0U, params);
  return af::SUCCESS;
}

af::Status ApplyAlignedNonReuseAdjustment(bool is_reuse_source, uint32_t avg_folds, bool supports_fold_three,
                                          TreeReduceParams &params) {
  GE_ASSERT_TRUE(avg_folds > 0U, "avg_folds[%u] is invalid, must be positive.", avg_folds);
  if (is_reuse_source || !params.tailR.IsConstExpr() || !params.mainR.IsConstExpr()) {
    return af::SUCCESS;
  }

  int64_t tail_r = 0;
  int64_t main_r = 0;
  params.tailR.GetConstValue(tail_r);
  params.mainR.GetConstValue(main_r);
  if (tail_r != 0 || main_r <= 1) {
    return af::SUCCESS;
  }

  const uint32_t adjusted_main_r = static_cast<uint32_t>(main_r / 2);
  const uint32_t folds = CalculateFoldsConst(adjusted_main_r);
  const uint32_t main_times = folds / avg_folds;
  const uint32_t tail_folds = folds % avg_folds;
  params.mainR = CreateExpr(static_cast<int64_t>(adjusted_main_r));
  params.tailR = CreateExpr(static_cast<int64_t>(adjusted_main_r));
  params.folds = CreateExpr(static_cast<int64_t>(folds));
  params.mainTimes = CreateExpr(static_cast<int64_t>(main_times));
  params.tailFolds = CreateExpr(static_cast<int64_t>(tail_folds));
  params.flags = CalculateFoldFlagsConst(static_cast<uint16_t>(tail_folds), supports_fold_three);
  SetLegacyFoldFlags(tail_folds, params);
  return af::SUCCESS;
}

af::Status AddMainFoldCost(const ReduceOpCostModel &model, const std::string &dtype, const Expr &count,
                           const Expr &repeat_ele, const TreeReduceParams &params, uint32_t avg_folds, bool b64,
                           Expr &cost) {
  GE_ASSERT_TRUE(avg_folds > 0U, "avg_folds[%u] is invalid, must be positive.", avg_folds);
  const uint32_t load_count = 1U << avg_folds;
  const uint32_t binary_count = load_count - 1U;
  Expr per_group = CreateExpr(0);
  if (b64) {
    GE_ASSERT_SUCCESS(BuildB64VfGroupCost(model, dtype, count, load_count, binary_count, 1U, per_group));
  } else {
    GE_ASSERT_SUCCESS(BuildVfGroupCost(model, dtype, count, repeat_ele, load_count, binary_count, 1U, per_group));
  }

  if (!params.mainTimes.IsConstExpr()) {
    cost = cost + params.mainR * per_group;
    return af::SUCCESS;
  }

  int64_t main_times_val = 0;
  params.mainTimes.GetConstValue(main_times_val);
  Expr current_r = params.mainR;
  const int64_t group_width = static_cast<int64_t>(1ULL << avg_folds);
  const Expr group_width_rational = af::sym::Rational(1, group_width);
  for (int64_t i = 0; i < main_times_val; i++) {
    current_r = current_r * group_width_rational;
    cost = cost + current_r * per_group;
  }
  return af::SUCCESS;
}

af::Status AddTailFoldCaseCost(const ReduceOpCostModel &model, const std::string &dtype, const Expr &count,
                               const Expr &repeat_ele, bool b64, uint32_t load_count, uint32_t binary_count,
                               uint32_t store_count, Expr &cost) {
  Expr case_cost = CreateExpr(0);
  if (b64) {
    GE_ASSERT_SUCCESS(BuildB64VfGroupCost(model, dtype, count, load_count, binary_count, store_count, case_cost));
  } else {
    GE_ASSERT_SUCCESS(
        BuildVfGroupCost(model, dtype, count, repeat_ele, load_count, binary_count, store_count, case_cost));
  }
  cost = cost + case_cost;
  return af::SUCCESS;
}

af::Status AddTailFoldCost(const ReduceOpCostModel &model, const std::string &dtype, const Expr &count,
                           const Expr &repeat_ele, const FoldFlags &flags, bool b64, Expr &cost) {
  if (IsPositiveConst(flags.foldOne)) {
    GE_ASSERT_SUCCESS(AddTailFoldCaseCost(model, dtype, count, repeat_ele, b64, 2U, 1U, 1U, cost));
  }
  if (IsPositiveConst(flags.foldTwo)) {
    GE_ASSERT_SUCCESS(AddTailFoldCaseCost(model, dtype, count, repeat_ele, b64, 4U, 3U, 1U, cost));
  }
  if (!b64 && IsPositiveConst(flags.foldThree)) {
    GE_ASSERT_SUCCESS(AddTailFoldCaseCost(model, dtype, count, repeat_ele, b64, 8U, 7U, 1U, cost));
  }
  if (IsPositiveConst(flags.foldZero)) {
    GE_ASSERT_SUCCESS(AddTailFoldCaseCost(model, dtype, count, repeat_ele, b64, 1U, 0U, 1U, cost));
  }
  return af::SUCCESS;
}

af::Status SelectByAlignment(const std::string &var_name, const Expr &align_key, const Expr &aligned_cost,
                             const Expr &unaligned_cost, PerfOutputInfo &perf, Expr &cost) {
  auto aligned_case = std::make_shared<IfCase>(aligned_cost);
  GE_ASSERT_NOTNULL(aligned_case);
  auto unaligned_case = std::make_shared<IfCase>(unaligned_cost);
  GE_ASSERT_NOTNULL(unaligned_case);
  TernaryOp ternary_op =
      TernaryOp(CondType::K_EQ, align_key, CreateExpr(0), std::move(aligned_case), std::move(unaligned_case));
  cost = CreateExpr(var_name.c_str());
  ternary_op.SetVariable(cost);
  perf.ternary_ops[cost] = ternary_op;
  return af::SUCCESS;
}

af::Status SelectByLessEqual(const std::string &var_name, const Expr &left, const Expr &right, const Expr &true_cost,
                             const Expr &false_cost, PerfOutputInfo &perf, Expr &cost) {
  auto true_case = std::make_shared<IfCase>(true_cost);
  GE_ASSERT_NOTNULL(true_case);
  auto false_case = std::make_shared<IfCase>(false_cost);
  GE_ASSERT_NOTNULL(false_case);
  TernaryOp ternary_op = TernaryOp(CondType::K_LE, left, right, std::move(true_case), std::move(false_case));
  cost = CreateExpr(var_name.c_str());
  ternary_op.SetVariable(cost);
  perf.ternary_ops[cost] = ternary_op;
  return af::SUCCESS;
}

Expr GetNormalVlSize(const std::string &dtype);

Expr GetArSegmentFirst(const NodeDetail &node_detail) {
  return IsByteSupportedDtype(node_detail.input_dtype[0])
             ? af::sym::Min(node_detail.input_dims[0], CreateExpr(kOneStride))
             : node_detail.input_dims[0];
}

Expr GetArSegments(const NodeDetail &node_detail) {
  return IsByteSupportedDtype(node_detail.input_dtype[0])
             ? af::sym::Ceiling(node_detail.input_dims[0] / CreateExpr(kOneStride))
             : CreateExpr(1);
}

af::Status BuildArLessThanVlCost(const ReduceApiPerfContext &context, const ReduceOpCostModel &model, bool b64,
                                 bool unaligned, Expr &cost) {
  const auto &nd = context.node_detail;
  const std::string &dtype = nd.input_dtype[0];
  const Expr first = GetArSegmentFirst(nd);
  const Expr segments = GetArSegments(nd);
  const Expr repeat_ele = b64 ? GetB64RepeatEle(dtype) : GetNormalVlSize(dtype);
  Expr count = CreateExpr(0);
  GE_ASSERT_SUCCESS(GetPerInnerAxisVfCount(first, nd.input_dims[1], repeat_ele, count));
  const ReduceVfGroupCounts counts = unaligned ? ReduceVfGroupCounts{2U, 1U, 2U} : ReduceVfGroupCounts{1U, 1U, 1U};
  GE_ASSERT_SUCCESS(BuildReduceVfGroupCost(model, dtype, count, repeat_ele, counts, cost));
  cost = segments * cost;
  cost.Simplify();
  return af::SUCCESS;
}

af::Status AddArMainFoldCost(const ReduceOpCostModel &model, const std::string &dtype, const Expr &count,
                             const TreeReduceParams &params, uint32_t avg_folds, bool b64, Expr &cost) {
  TreeReduceParams base_params = params;
  const Expr repeat_ele = b64 ? GetB64RepeatEle(dtype) : GetNormalVlSize(dtype);
  GE_ASSERT_SUCCESS(AddMainFoldCost(model, dtype, count, repeat_ele, base_params, avg_folds, b64, cost));
  return af::SUCCESS;
}

af::Status BuildArTailInplaceCost(const ReduceApiPerfContext &context, const ReduceOpCostModel &model, bool b64,
                                  bool unaligned, const TreeReduceParams &params, Expr &cost) {
  cost = CreateExpr(0);
  if (!IsPositiveConst(params.tailR)) {
    return af::SUCCESS;
  }
  const auto &nd = context.node_detail;
  const std::string &dtype = nd.input_dtype[0];
  const Expr repeat_ele = b64 ? GetB64RepeatEle(dtype) : GetNormalVlSize(dtype);
  const Expr count = GetArSegmentFirst(nd) * repeat_ele;
  Expr repeats = CreateExpr(0);
  GE_ASSERT_SUCCESS(CeilDivByPositiveConstExpr(params.tailR, repeat_ele, "repeat_ele", repeats));
  const VfGroupCounts counts = unaligned ? VfGroupCounts{4U, 1U, 2U} : VfGroupCounts{2U, 1U, 1U};
  return BuildRepeatedVfGroupCost(
      repeats,
      [&model, dtype, count, repeat_ele, counts, b64](Expr &body_cost) -> af::Status {
        if (b64) {
          return BuildB64VfGroupBodyCost(model, dtype, count, counts.load_count, counts.binary_count,
                                         counts.store_count, body_cost);
        }
        return BuildVfGroupBodyCost(model, dtype, count, repeat_ele, counts.load_count, counts.binary_count,
                                    counts.store_count, body_cost);
      },
      cost);
}

af::Status CalculateArTreeReduceParams(const ReduceApiPerfContext &context, uint32_t avg_folds,
                                       bool supports_fold_three, Expr vl_size, bool apply_non_reuse,
                                       TreeReduceParams &params) {
  GE_ASSERT_TRUE(avg_folds > 0U, "avg_folds[%u] is invalid, must be positive.", avg_folds);
  const auto &nd = context.node_detail;
  if (!nd.input_dims[1].IsConstExpr() || !vl_size.IsConstExpr()) {
    params.mainR = nd.input_dims[1];
    params.tailR = CreateExpr(0);
    GE_ASSERT_SUCCESS(CeilDivByPositiveConstExpr(nd.input_dims[1], vl_size, "vl_size", params.folds));
    params.mainTimes = CeilDivByPositiveConst(params.folds, static_cast<int64_t>(avg_folds));
    params.tailFolds = CreateExpr(0);
    params.flags = CalculateFoldFlagsConst(0U, supports_fold_three);
    SetLegacyFoldFlags(0U, params);
    return af::SUCCESS;
  }
  int64_t dim_r = 0;
  int64_t vl_size_value = 0;
  nd.input_dims[1].GetConstValue(dim_r);
  vl_size.GetConstValue(vl_size_value);
  GE_ASSERT_TRUE(dim_r > 0 && vl_size_value > 0,
                 "Reduce AR dimR[%ld] or vlSize[%ld] is invalid, both must be positive.", dim_r, vl_size_value);
  uint32_t main_r = CalculateMainRConst(static_cast<uint32_t>(dim_r));
  uint32_t tail_r = static_cast<uint32_t>(dim_r) - main_r;
  if (apply_non_reuse && !context.is_reuse_source) {
    main_r = tail_r > 0U ? main_r : main_r / 2U;
    tail_r = tail_r > 0U ? tail_r : main_r;
  }
  const uint32_t base = std::max<uint32_t>(1U, main_r / static_cast<uint32_t>(vl_size_value));
  const uint32_t folds = CalculateFoldsConst(base);
  const uint32_t main_times = folds / avg_folds;
  const uint32_t tail_folds = folds % avg_folds;
  params.mainR = CreateExpr(static_cast<int64_t>(base));
  params.tailR = CreateExpr(static_cast<int64_t>(tail_r));
  params.folds = CreateExpr(static_cast<int64_t>(folds));
  params.mainTimes = CreateExpr(static_cast<int64_t>(main_times));
  params.tailFolds = CreateExpr(static_cast<int64_t>(tail_folds));
  params.flags = CalculateFoldFlagsConst(static_cast<uint16_t>(tail_folds), supports_fold_three);
  SetLegacyFoldFlags(tail_folds, params);
  return af::SUCCESS;
}

af::Status BuildArOverVlCost(const ReduceApiPerfContext &context, const ReduceOpCostModel &model, bool b64,
                             bool unaligned, PerfOutputInfo &perf, Expr &cost) {
  const auto &nd = context.node_detail;
  const std::string &dtype = nd.input_dtype[0];
  const Expr repeat_ele = b64 ? GetB64RepeatEle(dtype) : GetNormalVlSize(dtype);
  const uint32_t avg_folds = b64 ? 3U : 4U;
  const bool supports_fold_three = !b64;
  if (!nd.input_dims[1].IsConstExpr()) {
    int64_t vl_size = 0;
    repeat_ele.GetConstValue(vl_size);
    GE_ASSERT_TRUE(vl_size > 0, "Reduce AR vlSize[%ld] is invalid, must be positive.", vl_size);
    const Expr count = GetArSegmentFirst(nd) * repeat_ele;
    const std::string var_name =
        b64 ? (unaligned ? "reduce_ar_b64_unaligned_tree_cost" : "reduce_ar_b64_aligned_tree_cost")
            : (unaligned ? "reduce_ar_normal_unaligned_tree_cost" : "reduce_ar_normal_aligned_tree_cost");
    TreeCostBuilder builder = [&context, &model, dtype, count, repeat_ele, avg_folds, supports_fold_three, b64,
                               unaligned, vl_size](uint32_t main_r, const Expr &tail_r, Expr &case_cost) -> af::Status {
      const uint32_t base = std::max<uint32_t>(1U, main_r / static_cast<uint32_t>(vl_size));
      TreeReduceParams params;
      GE_ASSERT_SUCCESS(MakeTreeReduceParams(base, tail_r, avg_folds, supports_fold_three, params));
      Expr inplace_cost = CreateExpr(0);
      GE_ASSERT_SUCCESS(BuildArTailInplaceCost(context, model, b64, unaligned, params, inplace_cost));
      Expr main_fold = CreateExpr(0);
      GE_ASSERT_SUCCESS(AddArMainFoldCost(model, dtype, count, params, avg_folds, b64, main_fold));
      Expr tail_fold = CreateExpr(0);
      GE_ASSERT_SUCCESS(AddTailFoldCost(model, dtype, count, repeat_ele, params.flags, b64, tail_fold));
      case_cost = GetArSegments(context.node_detail) * (inplace_cost + main_fold + tail_fold);
      case_cost.Simplify();
      return af::SUCCESS;
    };
    return SelectSymbolicTreeCost(var_name, nd.input_dims[1], builder, perf, cost,
                                  !unaligned && !context.is_reuse_source);
  }
  TreeReduceParams params;
  GE_ASSERT_SUCCESS(
      CalculateArTreeReduceParams(context, avg_folds, supports_fold_three, repeat_ele, !unaligned, params));
  const Expr count = GetArSegmentFirst(nd) * repeat_ele;
  Expr inplace_cost = CreateExpr(0);
  GE_ASSERT_SUCCESS(BuildArTailInplaceCost(context, model, b64, unaligned, params, inplace_cost));
  Expr main_fold = CreateExpr(0);
  GE_ASSERT_SUCCESS(AddArMainFoldCost(model, dtype, count, params, avg_folds, b64, main_fold));
  Expr tail_fold = CreateExpr(0);
  GE_ASSERT_SUCCESS(AddTailFoldCost(model, dtype, count, repeat_ele, params.flags, b64, tail_fold));
  cost = GetArSegments(nd) * (inplace_cost + main_fold + tail_fold);
  cost.Simplify();
  return af::SUCCESS;
}

bool IsConstLe(const Expr &lhs, const Expr &rhs) {
  if (!lhs.IsConstExpr() || !rhs.IsConstExpr()) {
    return false;
  }
  int64_t lhs_value = 0;
  int64_t rhs_value = 0;
  lhs.GetConstValue(lhs_value);
  rhs.GetConstValue(rhs_value);
  return lhs_value <= rhs_value;
}

af::Status BuildArReduceCost(const ReduceApiPerfContext &context, const ReduceOpCostModel &model, bool b64,
                             bool unaligned, PerfOutputInfo &perf, Expr &cost) {
  const auto &nd = context.node_detail;
  const Expr vl_size = b64 ? GetB64RepeatEle(nd.input_dtype[0]) : GetNormalVlSize(nd.input_dtype[0]);
  if (!nd.input_dims[1].IsConstExpr()) {
    Expr less_cost = CreateExpr(0);
    Expr over_cost = CreateExpr(0);
    GE_ASSERT_SUCCESS(BuildArLessThanVlCost(context, model, b64, unaligned, less_cost));
    GE_ASSERT_SUCCESS(BuildArOverVlCost(context, model, b64, unaligned, perf, over_cost));
    const std::string var_name =
        b64 ? (unaligned ? "reduce_ar_b64_unaligned_vl_case" : "reduce_ar_b64_aligned_vl_case")
            : (unaligned ? "reduce_ar_normal_unaligned_vl_case" : "reduce_ar_normal_aligned_vl_case");
    return SelectByLessEqual(var_name, nd.input_dims[1], vl_size, less_cost, over_cost, perf, cost);
  }
  if (IsConstLe(nd.input_dims[1], vl_size)) {
    return BuildArLessThanVlCost(context, model, b64, unaligned, cost);
  }
  return BuildArOverVlCost(context, model, b64, unaligned, perf, cost);
}

af::Status BuildArB64AlignedCost(const ReduceApiPerfContext &context, const ReduceOpCostModel &model,
                                 PerfOutputInfo &perf, Expr &cost) {
  return BuildArReduceCost(context, model, true, false, perf, cost);
}

af::Status BuildArB64UnalignedCost(const ReduceApiPerfContext &context, const ReduceOpCostModel &model,
                                   PerfOutputInfo &perf, Expr &cost) {
  return BuildArReduceCost(context, model, true, true, perf, cost);
}

af::Status BuildArB64Cost(const ReduceApiPerfContext &context, const ReduceOpCostModel &model, PerfOutputInfo &perf,
                          Expr &cost) {
  const auto &nd = context.node_detail;
  const Expr last = nd.input_dims[1];
  const Expr dtype_size = kDataTypeSizeMap.at(nd.input_dtype[0]);
  Expr aligned_cost = CreateExpr(0);
  Expr unaligned_cost = CreateExpr(0);
  GE_ASSERT_SUCCESS(BuildArB64AlignedCost(context, model, perf, aligned_cost));
  GE_ASSERT_SUCCESS(BuildArB64UnalignedCost(context, model, perf, unaligned_cost));
  GE_ASSERT_SUCCESS(SelectByAlignment("reduce_ar_b64_align_case", MakeAlignKey(last, dtype_size, false), aligned_cost,
                                      unaligned_cost, perf, cost));
  AddReduceBreakdown(perf, "reduce_ar_b64_align_perf", cost, "AR B64 32B alignment selected branch", 1U);
  return af::SUCCESS;
}

af::Status BuildArNormalAlignedCost(const ReduceApiPerfContext &context, const ReduceOpCostModel &model,
                                    PerfOutputInfo &perf, Expr &cost) {
  return BuildArReduceCost(context, model, false, false, perf, cost);
}

af::Status BuildArNormalUnalignedCost(const ReduceApiPerfContext &context, const ReduceOpCostModel &model,
                                      PerfOutputInfo &perf, Expr &cost) {
  return BuildArReduceCost(context, model, false, true, perf, cost);
}

af::Status BuildArNormalCost(const ReduceApiPerfContext &context, const ReduceOpCostModel &model, PerfOutputInfo &perf,
                             Expr &cost) {
  const auto &nd = context.node_detail;
  const Expr last = nd.input_dims[1];
  const Expr dtype_size = kDataTypeSizeMap.at(nd.input_dtype[0]);
  Expr aligned_cost = CreateExpr(0);
  Expr unaligned_cost = CreateExpr(0);
  GE_ASSERT_SUCCESS(BuildArNormalAlignedCost(context, model, perf, aligned_cost));
  GE_ASSERT_SUCCESS(BuildArNormalUnalignedCost(context, model, perf, unaligned_cost));
  GE_ASSERT_SUCCESS(SelectByAlignment("reduce_ar_normal_align_case", MakeAlignKey(last, dtype_size, true), aligned_cost,
                                      unaligned_cost, perf, cost));
  AddReduceBreakdown(perf, "reduce_ar_normal_align_perf", cost, "AR normal 32B alignment selected branch", 1U);
  return af::SUCCESS;
}

af::Status AddSymbolicMainFoldCost(const ReduceOpCostModel &model, const NormalSymbolicTreeSpec &spec,
                                   const TreeReduceParams &params, uint32_t avg_folds, bool b64, Expr &cost) {
  GE_ASSERT_TRUE(avg_folds > 0U, "avg_folds[%u] is invalid, must be positive.", avg_folds);
  Expr per_group = CreateExpr(0);
  const uint32_t store_count = GetMainStoreCount(spec);
  if (b64) {
    GE_ASSERT_SUCCESS(BuildB64VfGroupCost(model, spec.dtype, spec.count, spec.main_load_count,
                                          spec.main_load_count - 1U, store_count, per_group));
  } else {
    GE_ASSERT_SUCCESS(BuildVfGroupCost(model, spec.dtype, spec.count, spec.repeat_ele, spec.main_load_count,
                                       spec.main_load_count - 1U, store_count, per_group));
  }

  Expr current_r = params.mainR;
  const int64_t group_width = static_cast<int64_t>(1ULL << avg_folds);
  const Expr group_width_rational = af::sym::Rational(1, group_width);
  int64_t main_times = 0;
  params.mainTimes.GetConstValue(main_times);
  cost = CreateExpr(0);
  for (int64_t i = 0; i < main_times; ++i) {
    current_r = current_r * group_width_rational;
    cost = cost + current_r * per_group;
  }
  return af::SUCCESS;
}

af::Status AddSymbolicTailFoldCost(const ReduceOpCostModel &model, const NormalSymbolicTreeSpec &spec,
                                   const FoldFlags &flags, bool b64, Expr &cost) {
  cost = CreateExpr(0);
  const uint32_t store_count = GetTailStoreCount(spec);
  if (IsPositiveConst(flags.foldOne)) {
    GE_ASSERT_SUCCESS(AddTailFoldCaseCost(model, spec.dtype, spec.count, spec.repeat_ele, b64, spec.tail_load_count, 1U,
                                          store_count, cost));
  }
  if (IsPositiveConst(flags.foldTwo)) {
    GE_ASSERT_SUCCESS(AddTailFoldCaseCost(model, spec.dtype, spec.count, spec.repeat_ele, b64,
                                          spec.tail_load_count * 2U, 3U, store_count, cost));
  }
  if (!b64 && IsPositiveConst(flags.foldThree)) {
    GE_ASSERT_SUCCESS(AddTailFoldCaseCost(model, spec.dtype, spec.count, spec.repeat_ele, false,
                                          spec.tail_load_count * 4U, 7U, store_count, cost));
  }
  if (IsPositiveConst(flags.foldZero)) {
    GE_ASSERT_SUCCESS(
        AddTailFoldCaseCost(model, spec.dtype, spec.count, spec.repeat_ele, b64, store_count, 0U, store_count, cost));
  }
  return af::SUCCESS;
}

af::Status BuildTreeBodyCost(const ReduceOpCostModel &model, const NormalSymbolicTreeSpec &spec, uint32_t avg_folds,
                             bool b64, const TreeReduceParams &params, const TailInplaceCostBuilder &tail_builder,
                             Expr &cost) {
  Expr inplace_one = CreateExpr(0);
  if (tail_builder != nullptr) {
    GE_ASSERT_SUCCESS(tail_builder(params, inplace_one));
  } else {
    const uint32_t inplace_store_count = GetInplaceStoreCount(spec);
    if (b64) {
      GE_ASSERT_SUCCESS(BuildRepeatedVfGroupCost(
          params.tailR,
          [&model, spec, inplace_store_count](Expr &body_cost) -> af::Status {
            return BuildB64VfGroupBodyCost(model, spec.dtype, spec.count, spec.inplace_load_count, 1U,
                                           inplace_store_count, body_cost);
          },
          inplace_one));
    } else {
      GE_ASSERT_SUCCESS(BuildRepeatedVfGroupCost(
          params.tailR,
          [&model, spec, inplace_store_count](Expr &body_cost) -> af::Status {
            return BuildVfGroupBodyCost(model, spec.dtype, spec.count, spec.repeat_ele, spec.inplace_load_count, 1U,
                                        inplace_store_count, body_cost);
          },
          inplace_one));
    }
  }

  Expr main_fold = CreateExpr(0);
  GE_ASSERT_SUCCESS(AddSymbolicMainFoldCost(model, spec, params, avg_folds, b64, main_fold));
  Expr tail_fold = CreateExpr(0);
  GE_ASSERT_SUCCESS(AddSymbolicTailFoldCost(model, spec, params.flags, b64, tail_fold));
  cost = inplace_one + main_fold + tail_fold;
  cost.Simplify();
  return af::SUCCESS;
}

TreeCostBuilder MakeTreeCostBuilder(const ReduceOpCostModel &model, const NormalSymbolicTreeSpec &spec,
                                    uint32_t avg_folds, bool supports_fold_three, bool b64,
                                    const TailInplaceCostBuilder &tail_builder = nullptr) {
  return [&model, spec, avg_folds, supports_fold_three, b64, tail_builder](uint32_t main_r, const Expr &tail_r,
                                                                           Expr &case_cost) -> af::Status {
    TreeReduceParams params;
    GE_ASSERT_SUCCESS(MakeTreeReduceParams(main_r, tail_r, avg_folds, supports_fold_three, params));
    return BuildTreeBodyCost(model, spec, avg_folds, b64, params, tail_builder, case_cost);
  };
}

TreeCostBuilder MakeAlignedTreeCostBuilder(const ReduceApiPerfContext &context, const ReduceOpCostModel &model,
                                           const NormalSymbolicTreeSpec &spec, uint32_t avg_folds,
                                           bool supports_fold_three, bool b64) {
  (void)context;
  return MakeTreeCostBuilder(model, spec, avg_folds, supports_fold_three, b64);
}

af::Status BuildRaB64ConstTreeCost(const ReduceOpCostModel &model, const std::string &dtype, const Expr &count,
                                   const Expr &repeat_ele, const TreeReduceParams &params, uint32_t inplace_load,
                                   uint32_t inplace_store, Expr &cost) {
  Expr inplace_add = CreateExpr(0);
  if (IsPositiveConst(params.tailR)) {
    GE_ASSERT_SUCCESS(BuildRepeatedVfGroupCost(
        params.tailR,
        [&model, dtype, count, inplace_load, inplace_store](Expr &body_cost) -> af::Status {
          return BuildB64VfGroupBodyCost(model, dtype, count, inplace_load, 1U, inplace_store, body_cost);
        },
        inplace_add));
  }
  Expr main_fold = CreateExpr(0);
  GE_ASSERT_SUCCESS(AddMainFoldCost(model, dtype, count, repeat_ele, params, 3U, true, main_fold));
  Expr tail_fold = CreateExpr(0);
  GE_ASSERT_SUCCESS(AddTailFoldCost(model, dtype, count, repeat_ele, params.flags, true, tail_fold));
  cost = inplace_add + main_fold + tail_fold;
  cost.Simplify();
  return af::SUCCESS;
}

af::Status BuildRaB64AlignedCost(const ReduceApiPerfContext &context, const ReduceOpCostModel &model,
                                 PerfOutputInfo &perf, Expr &cost) {
  const auto &nd = context.node_detail;
  const Expr dim_r = nd.input_dims[0];
  const Expr dim_a = nd.input_dims[1];
  const std::string &dtype = nd.input_dtype[0];
  const Expr repeat_ele = GetB64RepeatEle(dtype);
  Expr count = CreateExpr(0);
  GE_ASSERT_SUCCESS(GetRaB64VectorBlockCount(dim_a, dtype, count));
  if (!dim_r.IsConstExpr()) {
    NormalSymbolicTreeSpec spec{dtype, dim_r, count, repeat_ele, 2U, 8U, 2U, 1U};
    return SelectSymbolicTreeCost("reduce_ra_b64_aligned_tree_cost", dim_r,
                                  MakeAlignedTreeCostBuilder(context, model, spec, 3U, false, true), perf, cost,
                                  !context.is_reuse_source);
  }

  TreeReduceParams params;
  GE_ASSERT_SUCCESS(CalculateTreeReduceParams(dim_r, 3U, false, params));
  if (params.mainR.IsConstExpr()) {
    int64_t main_r = 0;
    params.mainR.GetConstValue(main_r);
    if (main_r == 1) {
      GE_ASSERT_SUCCESS(BuildCopyOutCost(model, dtype, dim_a, repeat_ele, true, cost));
      return af::SUCCESS;
    }
  }
  GE_ASSERT_SUCCESS(ApplyAlignedNonReuseAdjustment(context.is_reuse_source, 3U, false, params));

  GE_ASSERT_SUCCESS(BuildRaB64ConstTreeCost(model, dtype, count, repeat_ele, params, 2U, 1U, cost));
  return af::SUCCESS;
}

af::Status BuildRaB64UnalignedCost(const ReduceApiPerfContext &context, const ReduceOpCostModel &model,
                                   PerfOutputInfo &perf, Expr &cost) {
  const auto &nd = context.node_detail;
  const Expr dim_r = nd.input_dims[0];
  const Expr dim_a = nd.input_dims[1];
  const std::string &dtype = nd.input_dtype[0];
  const Expr repeat_ele = GetB64RepeatEle(dtype);
  Expr count = CreateExpr(0);
  GE_ASSERT_SUCCESS(GetRaB64VectorBlockCount(dim_a, dtype, count));
  if (!dim_r.IsConstExpr()) {
    NormalSymbolicTreeSpec spec{dtype, dim_r, count, repeat_ele, 4U, 8U, 2U, 1U, 2U, 1U, 1U};
    return SelectSymbolicTreeCost("reduce_ra_b64_unaligned_tree_cost", dim_r,
                                  MakeTreeCostBuilder(model, spec, 3U, false, true), perf, cost);
  }

  TreeReduceParams params;
  GE_ASSERT_SUCCESS(CalculateTreeReduceParams(dim_r, 3U, false, params));

  GE_ASSERT_SUCCESS(BuildRaB64ConstTreeCost(model, dtype, count, repeat_ele, params, 4U, 2U, cost));
  return af::SUCCESS;
}

af::Status BuildRaB64Cost(const ReduceApiPerfContext &context, const ReduceOpCostModel &model, PerfOutputInfo &perf,
                          Expr &cost) {
  const auto &nd = context.node_detail;
  const Expr last = nd.input_dims[1];
  const Expr dtype_size = kDataTypeSizeMap.at(nd.input_dtype[0]);
  Expr block_count = CreateExpr(0);
  GE_ASSERT_SUCCESS(GetRaB64VectorBlockCount(last, nd.input_dtype[0], block_count));
  GELOGD("[ATT Reduce] RA-B64 vector block: last[%s], vector_block_ele[%s], perf_count[%s], node[%s].",
         last.Str().get(), GetB64RepeatEle(nd.input_dtype[0]).Str().get(), block_count.Str().get(), nd.name.c_str());
  Expr aligned_cost = CreateExpr(0);
  Expr unaligned_cost = CreateExpr(0);
  GE_ASSERT_SUCCESS(BuildRaB64AlignedCost(context, model, perf, aligned_cost));
  GE_ASSERT_SUCCESS(BuildRaB64UnalignedCost(context, model, perf, unaligned_cost));
  GE_ASSERT_SUCCESS(SelectByAlignment("reduce_ra_b64_align_case", MakeAlignKey(last, dtype_size, false), aligned_cost,
                                      unaligned_cost, perf, cost));
  AddReduceBreakdown(perf, "reduce_ra_b64_align_perf", cost, "RA B64 32B alignment selected branch", 1U);
  return af::SUCCESS;
}

Expr GetNormalVlSize(const std::string &dtype) {
  return GetRepeatEle(dtype);
}

af::Status HandleDimREqualsOne(const ReduceApiPerfContext &context, const ReduceOpCostModel &model, Expr &cost,
                               bool &handled) {
  handled = false;
  const Expr dim_r = context.node_detail.input_dims[0];
  if (!dim_r.IsConstExpr()) {
    return af::SUCCESS;
  }
  int64_t dim_r_val = 0;
  dim_r.GetConstValue(dim_r_val);
  if (dim_r_val != 1) {
    return af::SUCCESS;
  }
  handled = true;
  const Expr dim_a = context.node_detail.input_dims[1];
  const std::string &dtype = context.node_detail.input_dtype[0];
  const Expr repeat_ele = GetNormalVlSize(dtype);
  return BuildCopyOutCost(model, dtype, dim_a, repeat_ele, false, cost);
}

af::Status PrepareRaNormalContext(const ReduceApiPerfContext &context, const ReduceOpCostModel &model, Expr &cost,
                                  RaNormalContext &ra_context, bool &handled) {
  GE_ASSERT_SUCCESS(HandleDimREqualsOne(context, model, cost, handled));
  if (handled) {
    return af::SUCCESS;
  }
  const auto &nd = context.node_detail;
  ra_context.dim_r = nd.input_dims[0];
  ra_context.dim_a = nd.input_dims[1];
  ra_context.dtype = nd.input_dtype[0];
  ra_context.repeat_ele = GetNormalVlSize(ra_context.dtype);
  return af::SUCCESS;
}

af::Status BuildRaNormalConcatAlignedSymbolicCost(const ReduceOpCostModel &model, const NormalSymbolicTreeSpec &spec,
                                                  Expr &cost) {
  GE_ASSERT_SUCCESS(BuildVfGroupCost(model, spec.dtype, spec.count, spec.repeat_ele, 2U, 1U, 1U, cost));
  cost.Simplify();
  return af::SUCCESS;
}

af::Status BuildRaNormalConcatAlignedCost(const ReduceApiPerfContext &context, const ReduceOpCostModel &model,
                                          Expr &cost) {
  RaNormalContext ra_ctx;
  bool handled = false;
  GE_ASSERT_SUCCESS(PrepareRaNormalContext(context, model, cost, ra_ctx, handled));
  if (handled) {
    return af::SUCCESS;
  }
  if (ra_ctx.dim_r.IsConstExpr()) {
    int64_t dim_r_val = 0;
    ra_ctx.dim_r.GetConstValue(dim_r_val);
    if (dim_r_val == 2) {
      return BuildVfGroupCost(model, ra_ctx.dtype, ra_ctx.dim_a, ra_ctx.repeat_ele, 2U, 1U, 1U, cost);
    }
  } else {
    NormalSymbolicTreeSpec spec{
        ra_ctx.dtype, ra_ctx.dim_r, ra_ctx.dim_r * ra_ctx.dim_a, ra_ctx.repeat_ele, 2U, 2U, 2U, 1U};
    return BuildRaNormalConcatAlignedSymbolicCost(model, spec, cost);
  }

  TreeReduceParams params;
  GE_ASSERT_SUCCESS(CalculateTreeReduceParams(ra_ctx.dim_r, 1U, false, params));
  GE_ASSERT_SUCCESS(ApplyAlignedNonReuseAdjustment(context.is_reuse_source, 1U, false, params));
  Expr inplace_add = CreateExpr(0);
  if (IsPositiveConst(params.tailR)) {
    Expr inplace_one = CreateExpr(0);
    GE_ASSERT_SUCCESS(
        BuildVfGroupCost(model, ra_ctx.dtype, params.tailR * ra_ctx.dim_a, ra_ctx.repeat_ele, 2U, 1U, 1U, inplace_one));
    inplace_add = inplace_one;
  }

  Expr fold_cost = CreateExpr(0);
  if (params.folds.IsConstExpr()) {
    int64_t fold_time = 0;
    params.folds.GetConstValue(fold_time);
    Expr current_r = params.mainR;
    for (int64_t idx = 1; idx < fold_time; idx++) {
      current_r = current_r / kSymTwo;
      Expr fold_one = CreateExpr(0);
      GE_ASSERT_SUCCESS(
          BuildVfGroupCost(model, ra_ctx.dtype, current_r * ra_ctx.dim_a, ra_ctx.repeat_ele, 2U, 1U, 1U, fold_one));
      fold_cost = fold_cost + fold_one;
    }
  }

  Expr final_fold = CreateExpr(0);
  GE_ASSERT_SUCCESS(BuildVfGroupCost(model, ra_ctx.dtype, ra_ctx.dim_a, ra_ctx.repeat_ele, 2U, 1U, 1U, final_fold));
  cost = inplace_add + fold_cost + final_fold;
  cost.Simplify();
  return af::SUCCESS;
}

af::Status BuildRaNormalLessThanVlAlignedCost(const ReduceApiPerfContext &context, const ReduceOpCostModel &model,
                                              PerfOutputInfo &perf, Expr &cost) {
  RaNormalContext ra_ctx;
  bool handled = false;
  GE_ASSERT_SUCCESS(PrepareRaNormalContext(context, model, cost, ra_ctx, handled));
  if (handled) {
    return af::SUCCESS;
  }
  if (!ra_ctx.dim_r.IsConstExpr()) {
    NormalSymbolicTreeSpec spec{ra_ctx.dtype, ra_ctx.dim_r, ra_ctx.dim_a, ra_ctx.repeat_ele, 2U, 16U, 2U, 1U};
    return SelectSymbolicTreeCost("reduce_ra_normal_less_tree_cost", ra_ctx.dim_r,
                                  MakeAlignedTreeCostBuilder(context, model, spec, 4U, true, false), perf, cost,
                                  !context.is_reuse_source);
  }

  TreeReduceParams params;
  GE_ASSERT_SUCCESS(CalculateTreeReduceParams(ra_ctx.dim_r, 4U, true, params));
  GE_ASSERT_SUCCESS(ApplyAlignedNonReuseAdjustment(context.is_reuse_source, 4U, true, params));
  Expr inplace_add = CreateExpr(0);
  if (IsPositiveConst(params.tailR)) {
    Expr inplace_one = CreateExpr(0);
    GE_ASSERT_SUCCESS(
        BuildVfGroupCost(model, ra_ctx.dtype, params.tailR * ra_ctx.dim_a, ra_ctx.repeat_ele, 2U, 1U, 1U, inplace_one));
    inplace_add = inplace_one;
  }

  Expr main_fold = CreateExpr(0);
  GE_ASSERT_SUCCESS(
      AddMainFoldCost(model, ra_ctx.dtype, ra_ctx.dim_a, ra_ctx.repeat_ele, params, 4U, false, main_fold));
  Expr tail_fold = CreateExpr(0);
  GE_ASSERT_SUCCESS(
      AddTailFoldCost(model, ra_ctx.dtype, ra_ctx.dim_a, ra_ctx.repeat_ele, params.flags, false, tail_fold));
  cost = inplace_add + main_fold + tail_fold;
  cost.Simplify();
  return af::SUCCESS;
}

af::Status BuildRaNormalOverVlAlignedCost(const ReduceApiPerfContext &context, const ReduceOpCostModel &model,
                                          PerfOutputInfo &perf, Expr &cost) {
  const auto &nd = context.node_detail;
  const Expr dim_r = nd.input_dims[0];
  const Expr dim_a = nd.input_dims[1];
  const std::string &dtype = nd.input_dtype[0];
  const Expr repeat_ele = GetNormalVlSize(dtype);
  Expr count = CreateExpr(0);
  GE_ASSERT_SUCCESS(AlignUp(dim_a, repeat_ele, count));
  if (!dim_r.IsConstExpr()) {
    NormalSymbolicTreeSpec spec{dtype, dim_r, count, repeat_ele, 2U, 16U, 2U, 1U};
    return SelectSymbolicTreeCost("reduce_ra_normal_over_tree_cost", dim_r,
                                  MakeAlignedTreeCostBuilder(context, model, spec, 4U, true, false), perf, cost,
                                  !context.is_reuse_source);
  }

  TreeReduceParams params;
  GE_ASSERT_SUCCESS(CalculateTreeReduceParams(dim_r, 4U, true, params));
  if (params.mainR.IsConstExpr()) {
    int64_t main_r = 0;
    params.mainR.GetConstValue(main_r);
    if (main_r == 1) {
      return BuildCopyOutCost(model, dtype, dim_a, repeat_ele, false, cost);
    }
  }
  GE_ASSERT_SUCCESS(ApplyAlignedNonReuseAdjustment(context.is_reuse_source, 4U, true, params));

  Expr inplace_add = CreateExpr(0);
  if (IsPositiveConst(params.tailR)) {
    GE_ASSERT_SUCCESS(BuildRepeatedVfGroupCost(
        params.tailR,
        [&model, dtype, count, repeat_ele](Expr &body_cost) -> af::Status {
          return BuildVfGroupBodyCost(model, dtype, count, repeat_ele, 2U, 1U, 1U, body_cost);
        },
        inplace_add));
  }

  Expr main_fold = CreateExpr(0);
  GE_ASSERT_SUCCESS(AddMainFoldCost(model, dtype, count, repeat_ele, params, 4U, false, main_fold));
  Expr tail_fold = CreateExpr(0);
  GE_ASSERT_SUCCESS(AddTailFoldCost(model, dtype, count, repeat_ele, params.flags, false, tail_fold));
  cost = inplace_add + main_fold + tail_fold;
  cost.Simplify();
  return af::SUCCESS;
}

af::Status BuildRaNormalAlignedCost(const ReduceApiPerfContext &context, const ReduceOpCostModel &model,
                                    PerfOutputInfo &perf, Expr &cost) {
  const auto &nd = context.node_detail;
  const Expr dim_a = nd.input_dims[1];
  const Expr vl_size = GetNormalVlSize(nd.input_dtype[0]);
  if (!dim_a.IsConstExpr()) {
    // ReduceRAImpl selects these branches from the actual inner width, including tiling symbols.
    Expr concat_cost = CreateExpr(0);
    Expr less_cost = CreateExpr(0);
    Expr over_cost = CreateExpr(0);
    GE_ASSERT_SUCCESS(BuildRaNormalConcatAlignedCost(context, model, concat_cost));
    GE_ASSERT_SUCCESS(BuildRaNormalLessThanVlAlignedCost(context, model, perf, less_cost));
    GE_ASSERT_SUCCESS(BuildRaNormalOverVlAlignedCost(context, model, perf, over_cost));
    Expr vector_cost = CreateExpr(0);
    GE_ASSERT_SUCCESS(SelectByLessEqual("reduce_ra_normal_vector_width_case", dim_a, vl_size, less_cost, over_cost,
                                        perf, vector_cost));
    Expr inner_cost = CreateExpr(0);
    GE_ASSERT_SUCCESS(SelectByLessEqual("reduce_ra_normal_small_inner_case", dim_a, vl_size / kSymTwo, concat_cost,
                                        vector_cost, perf, inner_cost));
    return SelectByLessEqual("reduce_ra_normal_inner_width_case", dim_a, CreateExpr(kMaxNormalRaStride), inner_cost,
                             concat_cost, perf, cost);
  }

  int64_t dim_a_val = 0;
  int64_t vl_size_val = 0;
  dim_a.GetConstValue(dim_a_val);
  vl_size.GetConstValue(vl_size_val);
  if (dim_a_val <= vl_size_val / 2 || dim_a_val > kMaxNormalRaStride) {
    return BuildRaNormalConcatAlignedCost(context, model, cost);
  }
  if (dim_a_val <= vl_size_val) {
    return BuildRaNormalLessThanVlAlignedCost(context, model, perf, cost);
  }
  return BuildRaNormalOverVlAlignedCost(context, model, perf, cost);
}

af::Status AddNormalUnalignedMainFoldCost(const ReduceOpCostModel &model, const std::string &dtype, const Expr &dim_a,
                                          const Expr &repeat_ele, const TreeReduceParams &params, Expr &cost) {
  Expr per_group = CreateExpr(0);
  GE_ASSERT_SUCCESS(BuildVfGroupCost(model, dtype, dim_a, repeat_ele, 32U, 15U, 2U, per_group));
  if (!params.mainTimes.IsConstExpr()) {
    cost = params.mainR * per_group;
    return af::SUCCESS;
  }

  cost = CreateExpr(0);
  int64_t main_times = 0;
  params.mainTimes.GetConstValue(main_times);
  Expr current_r = params.mainR;
  for (int64_t idx = 0; idx < main_times; idx++) {
    current_r = current_r / CreateExpr(16);
    cost = cost + current_r * per_group;
  }
  return af::SUCCESS;
}

af::Status AddNormalUnalignedTailFoldCost(const ReduceOpCostModel &model, const std::string &dtype, const Expr &dim_a,
                                          const Expr &repeat_ele, const FoldFlags &flags, Expr &cost) {
  cost = CreateExpr(0);
  if (IsPositiveConst(flags.foldOne)) {
    GE_ASSERT_SUCCESS(AddTailFoldCaseCost(model, dtype, dim_a, repeat_ele, false, 4U, 1U, 2U, cost));
  }
  if (IsPositiveConst(flags.foldTwo)) {
    GE_ASSERT_SUCCESS(AddTailFoldCaseCost(model, dtype, dim_a, repeat_ele, false, 8U, 3U, 2U, cost));
  }
  if (IsPositiveConst(flags.foldThree)) {
    GE_ASSERT_SUCCESS(AddTailFoldCaseCost(model, dtype, dim_a, repeat_ele, false, 16U, 7U, 2U, cost));
  }
  if (IsPositiveConst(flags.foldZero)) {
    GE_ASSERT_SUCCESS(AddTailFoldCaseCost(model, dtype, dim_a, repeat_ele, false, 2U, 0U, 2U, cost));
  }
  return af::SUCCESS;
}

af::Status BuildNormalUnalignedInplaceCost(const ReduceOpCostModel &model, const std::string &dtype, const Expr &dim_a,
                                           const Expr &repeat_ele, const Expr &tail_r, Expr &cost) {
  return BuildRepeatedVfGroupCost(
      tail_r,
      [&model, dtype, dim_a, repeat_ele](Expr &body_cost) -> af::Status {
        return BuildVfGroupBodyCost(model, dtype, dim_a, repeat_ele, 4U, 1U, 2U, body_cost);
      },
      cost);
}

af::Status BuildRaNormalUnalignedCost(const ReduceApiPerfContext &context, const ReduceOpCostModel &model,
                                      PerfOutputInfo &perf, Expr &cost) {
  const auto &nd = context.node_detail;
  const Expr dim_r = nd.input_dims[0];
  const Expr dim_a = nd.input_dims[1];
  const std::string &dtype = nd.input_dtype[0];
  const Expr repeat_ele = GetNormalVlSize(dtype);
  if (!dim_r.IsConstExpr()) {
    NormalSymbolicTreeSpec spec{dtype, dim_r, dim_a, repeat_ele, 4U, 32U, 4U, 2U};
    TailInplaceCostBuilder tail_builder = [&model, dtype, dim_a, repeat_ele](const TreeReduceParams &params,
                                                                             Expr &tail_cost) -> af::Status {
      return BuildNormalUnalignedInplaceCost(model, dtype, dim_a, repeat_ele, params.tailR, tail_cost);
    };
    return SelectSymbolicTreeCost("reduce_ra_normal_unaligned_tree_cost", dim_r,
                                  MakeTreeCostBuilder(model, spec, 4U, true, false, tail_builder), perf, cost);
  }

  TreeReduceParams params;
  GE_ASSERT_SUCCESS(CalculateTreeReduceParams(dim_r, 4U, true, params));
  Expr inplace_add = CreateExpr(0);
  if (IsPositiveConst(params.tailR)) {
    GE_ASSERT_SUCCESS(BuildNormalUnalignedInplaceCost(model, dtype, dim_a, repeat_ele, params.tailR, inplace_add));
  }

  Expr main_fold = CreateExpr(0);
  GE_ASSERT_SUCCESS(AddNormalUnalignedMainFoldCost(model, dtype, dim_a, repeat_ele, params, main_fold));
  Expr tail_fold = CreateExpr(0);
  GE_ASSERT_SUCCESS(AddNormalUnalignedTailFoldCost(model, dtype, dim_a, repeat_ele, params.flags, tail_fold));
  cost = inplace_add + main_fold + tail_fold;
  cost.Simplify();
  return af::SUCCESS;
}

af::Status BuildRaNormalCost(const ReduceApiPerfContext &context, const ReduceOpCostModel &model, PerfOutputInfo &perf,
                             Expr &cost) {
  const auto &nd = context.node_detail;
  const Expr last = nd.input_dims[1];
  const Expr dtype_size = kDataTypeSizeMap.at(nd.input_dtype[0]);
  Expr aligned_cost = CreateExpr(0);
  Expr unaligned_cost = CreateExpr(0);
  GE_ASSERT_SUCCESS(BuildRaNormalAlignedCost(context, model, perf, aligned_cost));
  GE_ASSERT_SUCCESS(BuildRaNormalUnalignedCost(context, model, perf, unaligned_cost));
  GE_ASSERT_SUCCESS(SelectByAlignment("reduce_ra_normal_align_case", MakeAlignKey(last, dtype_size, false),
                                      aligned_cost, unaligned_cost, perf, cost));
  AddReduceBreakdown(perf, "reduce_ra_normal_align_perf", cost, "RA normal 32B alignment selected branch", 1U);
  return af::SUCCESS;
}

af::Status GetMergeTempSize(const ReduceApiPerfContext &context, Expr &temp_size) {
  return AlignUp(context.merge_size, GetVectorBlockEle(context.node_detail.input_dtype[0]), temp_size);
}

af::Status BuildMergeCopyCost(const ReduceApiPerfContext &context, const Expr &temp_size, Expr &cost) {
  GE_ASSERT_SUCCESS(GetPerf(
      {kUb2ub, context.node_detail.input_dtype[0], context.node_detail.output_dtype[0], {temp_size}, CreateExpr(0)},
      cost));
  return af::SUCCESS;
}

af::Status BuildMergeElementwiseCost(const ReduceApiPerfContext &context, ElementwisePerf elementwise_perf,
                                     const Expr &temp_size, Expr &cost) {
  PerfOutputInfo elem_perf;
  NodeDetail elem_nd = context.node_detail;
  elem_nd.input_dims = {temp_size};
  elem_nd.output_dims = {temp_size};
  GE_ASSERT_SUCCESS(elementwise_perf(elem_nd, elem_perf));
  GE_ASSERT_SUCCESS(AddPipeCost(elem_perf, PipeType::AIV_VEC, cost));
  return af::SUCCESS;
}

af::Status BuildMultiReduceCopyMergeCost(const ReduceApiPerfContext &context, ElementwisePerf elementwise_perf,
                                         const Expr &temp_size, Expr &cost) {
  Expr copy_cost = CreateExpr(0);
  Expr elementwise_cost = CreateExpr(0);
  GE_ASSERT_SUCCESS(BuildMergeCopyCost(context, temp_size, copy_cost));
  GE_ASSERT_SUCCESS(BuildMergeElementwiseCost(context, elementwise_perf, temp_size, elementwise_cost));
  cost = (copy_cost + (context.merge_times - CreateExpr(1)) * elementwise_cost) / context.merge_times;
  cost.Simplify();
  return af::SUCCESS;
}

af::Status BuildMergeCost(const ReduceApiPerfContext &context, ElementwisePerf elementwise_perf, Expr &cost) {
  cost = PipeBarrierVPerf();
  Expr temp_size = CreateExpr(0);
  GE_ASSERT_SUCCESS(GetMergeTempSize(context, temp_size));
  GELOGD("[ATT Reduce] merge perf size: merge_size[%s], vector_block_ele[%s], temp_size[%s], node[%s].",
         context.merge_size.Str().get(), GetVectorBlockEle(context.node_detail.input_dtype[0]).Str().get(),
         temp_size.Str().get(), context.node_detail.name.c_str());
  if (context.merge_mode == ReduceMergeMode::kCopy) {
    Expr merge_cost = CreateExpr(0);
    GE_ASSERT_SUCCESS(BuildMultiReduceCopyMergeCost(context, elementwise_perf, temp_size, merge_cost));
    cost = cost + merge_cost;
  } else if (context.merge_mode == ReduceMergeMode::kMergeByElementwise) {
    GE_ASSERT_SUCCESS(BuildMergeElementwiseCost(context, elementwise_perf, temp_size, cost));
  }
  return af::SUCCESS;
}

af::Status BuildLogicalReduceCost(const ReduceApiPerfContext &context, const ReduceOpCostModel &model,
                                  PerfOutputInfo &perf, Expr &cost) {
  if (context.pattern == ReducePattern::kAR) {
    return BuildArNormalAlignedCost(context, model, perf, cost);
  }
  return BuildRaNormalAlignedCost(context, model, perf, cost);
}

af::Status BuildValidatedReduceExecContext(const ReduceApiPerfContext &context, const ReduceOpCostModel &model,
                                           ReduceApiPerfContext &exec_context) {
  GE_ASSERT_SUCCESS(ValidateReduceContext(context));
  exec_context = context;
  if (IsLogicalReduce(model) || IsProdReduce(model)) {
    const auto &input_exec_dtype = IsProdReduce(model) ? GetProdReduceExecDtype(context.node_detail.input_dtype[0])
                                                       : GetLogicalReduceExecDtype(context.node_detail.input_dtype[0]);
    const auto &output_exec_dtype = IsProdReduce(model)
                                        ? GetProdReduceExecDtype(context.node_detail.output_dtype[0])
                                        : GetLogicalReduceExecDtype(context.node_detail.output_dtype[0]);
    exec_context.node_detail.input_dtype[0] = input_exec_dtype;
    exec_context.node_detail.output_dtype[0] = output_exec_dtype;
  }
  const af::Status dtype_status = ValidateReduceDtype(exec_context, model);
  if (dtype_status != af::SUCCESS) {
    return dtype_status;
  }
  return af::SUCCESS;
}

af::Status BuildReduceApiBodyCost(const ReduceApiPerfContext &exec_context, const ReduceOpCostModel &model, bool is_b64,
                                  PerfOutputInfo &perf, Expr &cost) {
  const auto &nd = exec_context.node_detail;
  if (IsLogicalReduce(model) && !is_b64) {
    GELOGD("[ATT Reduce] ReduceApiPerf: branch logical-%s, node[%s].",
           exec_context.pattern == ReducePattern::kAR ? "AR" : "RA", nd.name.c_str());
    return BuildLogicalReduceCost(exec_context, model, perf, cost);
  }
  if (exec_context.pattern == ReducePattern::kAR && is_b64) {
    GELOGD("[ATT Reduce] ReduceApiPerf: branch AR-B64, node[%s].", nd.name.c_str());
    return BuildArB64Cost(exec_context, model, perf, cost);
  }
  if (exec_context.pattern == ReducePattern::kAR) {
    GELOGD("[ATT Reduce] ReduceApiPerf: branch AR-normal, node[%s].", nd.name.c_str());
    return BuildArNormalCost(exec_context, model, perf, cost);
  }
  if (is_b64) {
    GELOGD("[ATT Reduce] ReduceApiPerf: branch RA-B64, node[%s].", nd.name.c_str());
    return BuildRaB64Cost(exec_context, model, perf, cost);
  }
  GELOGD("[ATT Reduce] ReduceApiPerf: branch RA-normal, node[%s].", nd.name.c_str());
  return BuildRaNormalCost(exec_context, model, perf, cost);
}

af::Status AddReduceMergeCost(const ReduceApiPerfContext &exec_context, ElementwisePerf elementwise_perf,
                              PerfOutputInfo &perf, Expr &total) {
  if (exec_context.merge_mode == ReduceMergeMode::kNone) {
    AddReduceBreakdown(perf, "reduce_merge_perf", CreateExpr(0), "multi-reduce merge perf, disabled", 0U);
    return af::SUCCESS;
  }
  Expr merge_cost = CreateExpr(0);
  GE_ASSERT_SUCCESS(BuildMergeCost(exec_context, elementwise_perf, merge_cost));
  GELOGD("[ATT Reduce] ReduceApiPerf: merge_mode[%d], merge_size[%s], merge_times[%s], node[%s].",
         static_cast<int32_t>(exec_context.merge_mode), exec_context.merge_size.Str().get(),
         exec_context.merge_times.Str().get(), exec_context.node_detail.name.c_str());
  AddReduceBreakdown(perf, "reduce_merge_perf", merge_cost, "multi-reduce merge perf", 0U);
  total = total + merge_cost;
  return af::SUCCESS;
}
}  // namespace

af::Status ReduceApiPerf(const ReduceApiPerfContext &context, const ReduceOpCostModel &model, PerfOutputInfo &perf) {
  ReduceApiPerfContext exec_context = context;
  const af::Status exec_status = BuildValidatedReduceExecContext(context, model, exec_context);
  if (exec_status != af::SUCCESS) {
    return exec_status;
  }
  const auto &nd = exec_context.node_detail;
  const bool is_b64 = IsB64Dtype(nd.input_dtype[0]);
  GELOGD(
      "[ATT Reduce] ReduceApiPerf: dtype[%s], dims[%s], pattern[%s], b64[%d], reuse[%d], merge[%d], "
      "merge_times[%s], node[%s].",
      nd.input_dtype[0].c_str(), nd.ToString().c_str(), context.pattern == ReducePattern::kAR ? "AR" : "RA",
      static_cast<int32_t>(is_b64), static_cast<int32_t>(context.is_reuse_source),
      static_cast<int32_t>(context.merge_mode), context.merge_times.Str().get(), nd.name.c_str());
  Expr total = CreateExpr(0);
  GE_ASSERT_SUCCESS(BuildReduceApiBodyCost(exec_context, model, is_b64, perf, total));
  const Expr body_cost = total;
  AddReduceBreakdown(perf, "reduce_body_perf", body_cost, "single Reduce API body perf", 0U);
  GE_ASSERT_SUCCESS(AddReduceMergeCost(exec_context, model.elementwise_perf, perf, total));
  total.Simplify();
  AddReduceBreakdown(perf, "reduce_total_perf", total, "Reduce API total perf = body + merge", 0U);
  perf.pipe_res[PipeType::AIV_VEC] = total;
  GELOGD("[ATT Reduce] ReduceApiPerf: result[%s], node[%s].", total.Str().get(), nd.name.c_str());
  return af::SUCCESS;
}

af::Status ReduceMinPerf(const ReduceApiPerfContext &context, PerfOutputInfo &perf) {
  const ReduceOpCostModel model = {ascendcperf_v2::MinPerf, kReduceMin, kMin, ReduceOpKind::kMin};
  return ReduceApiPerf(context, model, perf);
}

af::Status ReduceMaxPerf(const ReduceApiPerfContext &context, PerfOutputInfo &perf) {
  const ReduceOpCostModel model = {ascendcperf_v2::MaxPerf, kReduceMax, kMax, ReduceOpKind::kMax};
  return ReduceApiPerf(context, model, perf);
}

af::Status ReduceAnyPerf(const ReduceApiPerfContext &context, PerfOutputInfo &perf) {
  const ReduceOpCostModel model = {ascendcperf_v2::MaxPerf, kReduceAny, kMax, ReduceOpKind::kAny};
  return ReduceApiPerf(context, model, perf);
}

af::Status ReduceAllPerf(const ReduceApiPerfContext &context, PerfOutputInfo &perf) {
  const ReduceOpCostModel model = {ascendcperf_v2::MinPerf, kReduceAll, kMin, ReduceOpKind::kAll};
  return ReduceApiPerf(context, model, perf);
}

af::Status ReduceSumPerf(const ReduceApiPerfContext &context, PerfOutputInfo &perf) {
  const ReduceOpCostModel model = {ascendcperf_v2::AddPerf, kVcadd, kAdd, ReduceOpKind::kSum};
  return ReduceApiPerf(context, model, perf);
}

af::Status ReduceProdPerf(const ReduceApiPerfContext &context, PerfOutputInfo &perf) {
  const ReduceOpCostModel model = {ascendcperf_v2::MulPerf, kMul, kMul, ReduceOpKind::kProd};
  return ReduceApiPerf(context, model, perf);
}

af::Status ReduceMeanPerf(const ReduceApiPerfContext &context, PerfOutputInfo &perf) {
  const ReduceOpCostModel model = {ascendcperf_v2::AddPerf, kVcadd, kAdd, ReduceOpKind::kMean};
  GE_ASSERT_SUCCESS(ValidateReduceContext(context));
  const af::Status dtype_status = ValidateReduceDtype(context, model);
  if (dtype_status != af::SUCCESS) {
    return dtype_status;
  }
  const af::Status sum_status = ReduceSumPerf(context, perf);
  if (sum_status != af::SUCCESS) {
    return sum_status;
  }
  PerfOutputInfo muls_perf;
  NodeDetail muls_nd = context.node_detail;
  muls_nd.input_dims = {context.pattern == ReducePattern::kAR ? context.node_detail.input_dims[0]
                                                              : context.node_detail.input_dims[1]};
  muls_nd.output_dims = muls_nd.input_dims;
  GE_ASSERT_SUCCESS(ascendcperf_v2::MeanPerf(muls_nd, muls_perf));
  Expr muls_cost = CreateExpr(0);
  GE_ASSERT_SUCCESS(AddPipeCost(muls_perf, PipeType::AIV_VEC, muls_cost));
  perf.pipe_res[PipeType::AIV_VEC] = perf.pipe_res[PipeType::AIV_VEC] + muls_cost;
  perf.pipe_res[PipeType::AIV_VEC].Simplify();
  AddReduceBreakdown(perf, "reduce_mean_muls_perf", muls_cost, "ReduceMean post ReduceSum Muls perf", 0U);
  return af::SUCCESS;
}
}  // namespace ascendcapi_v2
}  // namespace att
