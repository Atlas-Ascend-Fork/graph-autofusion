/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <string>
#include <numeric>
#include "common/checker.h"
#include "v35/att/api_perf_register/perf_param_v2.h"
#include "api_perf_register/ascendc_api_perf.h"
#include "api_perf_register/api_perf_factory.h"
#include "v35/att/api_perf_register/ascendc_api_perf/data_move_api_perf_v2.h"
#include "v35/att/api_perf_register/ascendc_regbase_perf.h"
namespace att {
namespace ascendcapi_v2 {
namespace {
constexpr size_t kCodegenDmaMaxLen = 2U;
constexpr size_t kCodegenLoopModeMaxLen = 2U;
}  // namespace
// 获取 CacheLine 大小，与 TilingScheduleConfigTable 保持一致
uint32_t GetCacheLineSize() {
  return TilingScheduleConfigTableV2().GetCacheLineSize();
}

Expr GetCodegenTailAxis(const NodeDetail &node_info, const std::vector<Expr> &dims) {
  if (!node_info.vectorized_shape.empty()) {
    return node_info.vectorized_shape.back();
  }
  return dims.back();
}

bool IsFlattenedCodegenShape(const NodeDetail &node_info, const std::vector<Expr> &dims) {
  return dims.size() == 1U && node_info.vectorized_shape.size() > 1U;
}

// 初始化block_len扩展所需的参数
af::Status InitBlockLenExpandParams(const NodeDetail &node_info, const std::vector<Expr> &dims, Expr &block_len,
                                    Expr &cache_line_ele_num) {
  block_len = GetCodegenTailAxis(node_info, dims);
  const int32_t kCacheLineSize = static_cast<int32_t>(GetCacheLineSize());
  const auto &data_type_size = kDataTypeSizeMap.find(node_info.input_dtype[0]);
  GE_ASSERT_TRUE(data_type_size != kDataTypeSizeMap.cend(), "Check data type size failed, node[%s]",
                 node_info.ToString().c_str());
  cache_line_ele_num = CreateExpr(kCacheLineSize) / data_type_size->second;
  return af::SUCCESS;
}

af::Status AppendCacheLineConfig(const NodeDetail &node_info, CacheLineDirection direction,
                                 vector<CacheLineConfig> *cache_line_config) {
  if (cache_line_config != nullptr) {
    GE_ASSERT_TRUE(!node_info.input_dims.empty(), "Check cache line dims failed, node[%s]",
                   node_info.ToString().c_str());
    auto data_type_size = kDataTypeSizeMap.find(node_info.input_dtype[0]);
    GE_ASSERT_TRUE(data_type_size != kDataTypeSizeMap.end(), "Check data type size failed, node[%s]",
                   node_info.ToString().c_str());
    const Expr &transfer_len = node_info.repeats.empty() ? node_info.input_dims.back() : node_info.repeats.back();
    Expr cache_line_expr = af::sym::Mul(transfer_len, data_type_size->second);
    CacheLineConfig cfg{node_info.name, cache_line_expr, GetCacheLineSize(), direction};
    Expr dim_product = accumulate(node_info.input_dims.begin(), node_info.input_dims.end(), CreateExpr(1),
                                  [](const Expr &a, const Expr &b) { return a * b; });
    Expr total_bytes = af::sym::Mul(dim_product, data_type_size->second);
    if (!total_bytes.IsConstExpr()) {
      cfg.solver_cache_line_expr = std::move(total_bytes);
    }
    cache_line_config->push_back(std::move(cfg));
  }
  return af::SUCCESS;
}

af::Status GetLoadCase(const NodeDetail &node_info, Expr &blk, int32_t &use_case) {
  size_t dim_size = node_info.input_dims.size();
  GE_ASSERT_TRUE(!node_info.input_dims.empty(), "Check node input dims failed, node[%s]", node_info.ToString().c_str());
  const auto blk_len = GetCodegenTailAxis(node_info, node_info.input_dims);
  if (blk_len.IsConstExpr()) {
    int32_t blk_len_val;
    constexpr int32_t blk_threshold = 256U;
    blk_len.GetConstValue(blk_len_val);
    if (blk_len_val > blk_threshold) {
      use_case = kCaseOne;
    } else {
      use_case = kCaseTwo;
    }
  } else {
    const auto blk_thr = CreateExpr(256);
    blk = blk_len - blk_thr;
    use_case = kCaseDefault;
  }
  GELOGD("input dtype is %s, dim_size[%zu], block_len[%s], use_case[%d], node[%s]", node_info.input_dtype[0].c_str(),
         dim_size, blk_len.Str().get(), use_case, node_info.ToString().c_str());
  return af::SUCCESS;
}

af::Status InitBlockLenExpand(const NodeDetail &node_info, std::vector<Expr> &dims, Expr &block_len,
                              Expr &cache_line_ele_num, Expr &block_len_ref, int32_t &kCacheLineSize,
                              int32_t &blk_len_val, int32_t &stride_val) {
  GE_ASSERT_SUCCESS(InitBlockLenExpandParams(node_info, dims, block_len, cache_line_ele_num));
  block_len_ref = GetCodegenTailAxis(node_info, dims);
  kCacheLineSize = static_cast<int32_t>(GetCacheLineSize());
  if (block_len_ref.IsConstExpr() && node_info.gm_stride.IsConstExpr()) {
    block_len_ref.GetConstValue(blk_len_val);
    node_info.gm_stride.GetConstValue(stride_val);
    return af::SUCCESS;
  }
  return af::FAILED;
}

// 静态分支初始化结果
struct StaticExpandResult {
  int32_t blk_len_val{1};
  int32_t stride_val{0};
  int32_t cache_line_size{0};
  int32_t data_type_size_val{0};
  Expr cache_line_ele_num;
};

// 尝试静态初始化block_len扩展参数，成功返回af::SUCCESS
af::Status TryInitStaticExpand(const NodeDetail &node_info, std::vector<Expr> &dims, StaticExpandResult &result) {
  Expr block_len;
  Expr block_len_ref;
  result.cache_line_size = static_cast<int32_t>(GetCacheLineSize());
  if (InitBlockLenExpand(node_info, dims, block_len, result.cache_line_ele_num, block_len_ref, result.cache_line_size,
                         result.blk_len_val, result.stride_val) != af::SUCCESS) {
    return af::FAILED;
  }
  const auto &data_type_size = kDataTypeSizeMap.find(node_info.input_dtype[0]);
  data_type_size->second.GetConstValue(result.data_type_size_val);
  return af::SUCCESS;
}

// Load场景的block_len扩展：非连续且block_len字节<cache_line时pad到cache_line_ele_num
af::Status ExpandLoadBlockLen(const NodeDetail &node_info, PerfOutputInfo &perf, std::vector<Expr> &dims) {
  StaticExpandResult static_res;
  const int32_t kCacheLineSize = static_cast<int32_t>(GetCacheLineSize());

  if (TryInitStaticExpand(node_info, dims, static_res) == af::SUCCESS) {
    int32_t block_len_bytes = static_res.blk_len_val * static_res.data_type_size_val;
    if ((block_len_bytes < static_res.cache_line_size) && (static_res.stride_val > 0)) {
      if (IsFlattenedCodegenShape(node_info, dims)) {
        dims.emplace_back(static_res.cache_line_ele_num);
      } else {
        dims[dims.size() - 1UL] = static_res.cache_line_ele_num;
      }
    }
  } else {
    auto block_len_ref_actual = GetCodegenTailAxis(node_info, dims);
    const auto &data_type_size = kDataTypeSizeMap.find(node_info.input_dtype[0]);
    Expr cache_line_ele_num_threshold = CreateExpr(kCacheLineSize) / data_type_size->second;
    auto need_expand_checker = af::sym::LogicalAnd({af::sym::Gt(node_info.gm_stride, CreateExpr(0)),
                                                    af::sym::Lt(block_len_ref_actual, cache_line_ele_num_threshold)});
    bool need_expand{false};
    if (need_expand_checker.IsConstExpr()) {
      need_expand_checker.GetConstValue(need_expand);
      block_len_ref_actual = need_expand ? std::move(static_res.cache_line_ele_num) : std::move(block_len_ref_actual);
    } else {
      Expr res = CreateExpr("load_block_len");
      auto normal_case = std::make_shared<IfCase>(block_len_ref_actual);
      GE_ASSERT_NOTNULL(normal_case);
      auto expand_case = std::make_shared<IfCase>(static_res.cache_line_ele_num);
      GE_ASSERT_NOTNULL(expand_case);
      TernaryOp ternary_op = TernaryOp(CondType::K_EQ, need_expand_checker, CreateExpr(false), std::move(normal_case),
                                       std::move(expand_case));
      ternary_op.SetVariable(res);
      perf.ternary_ops[res] = ternary_op;
      block_len_ref_actual = res;
    }
    if (IsFlattenedCodegenShape(node_info, dims)) {
      dims.emplace_back(block_len_ref_actual);
    } else {
      dims[dims.size() - 1UL] = block_len_ref_actual;
    }
    GELOGD("Load block len checker[%s], need_expand[%d], block_len[%s]", need_expand_checker.Serialize().get(),
           need_expand, block_len_ref_actual.Serialize().get());
  }
  return af::SUCCESS;
}

af::Status LoadPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res_normal;
  Expr res_small_blk;
  Expr blk;
  Expr res_stride;
  int32_t use_case;
  GELOGD("Dma with Load: %s", node_info.ToString().c_str());
  // 非连续且block_len小于cache line时，扩展block_len到cache line大小
  std::vector<Expr> expanded_dims{node_info.input_dims};
  GE_ASSERT_SUCCESS(ExpandLoadBlockLen(node_info, perf, expanded_dims), "Expand load block len failed, node[%s]",
                    node_info.ToString().c_str());
  GE_ASSERT_SUCCESS(GetPerf(
      {kMoveGmToUb, node_info.input_dtype[0], node_info.output_dtype[0], expanded_dims, CreateExpr(0)}, res_normal));
  GE_ASSERT_SUCCESS(GetPerf(
      {kMoveGmToUb + "SmallBlk", node_info.input_dtype[0], node_info.output_dtype[0], expanded_dims, CreateExpr(0)},
      res_small_blk));
  GE_ASSERT_SUCCESS(GetPerf({kMoveGmToUb + "Stride", node_info.input_dtype[0], node_info.output_dtype[0], expanded_dims,
                             node_info.gm_stride, node_info.block_count_idx},
                            res_stride));
  GE_ASSERT_SUCCESS(GetLoadCase(node_info, blk, use_case));
  GE_ASSERT_SUCCESS(AppendCacheLineConfig(node_info, CacheLineDirection::kGmToUb, perf.cache_line_config));
  if (use_case == kCaseOne) {
    perf.pipe_res[PipeType::AIV_MTE2] = res_normal + res_stride;
  } else if (use_case == kCaseTwo) {
    perf.pipe_res[PipeType::AIV_MTE2] = res_small_blk + res_stride;
  } else {
    Expr res = CreateExpr("load_node");
    std::shared_ptr<IfCase> branch_a = std::make_shared<IfCase>(res_normal + res_stride);
    GE_ASSERT_NOTNULL(branch_a);
    std::shared_ptr<IfCase> branch_b = std::make_shared<IfCase>(res_small_blk + res_stride);
    GE_ASSERT_NOTNULL(branch_b);
    // blocklen < 512B时走branch_b；否则走branch_a
    TernaryOp ternary_op = TernaryOp(CondType::K_LT, blk, CreateExpr(0), std::move(branch_b), std::move(branch_a));
    ternary_op.SetVariable(res);
    perf.ternary_ops[res] = ternary_op;
    perf.pipe_res[PipeType::AIV_MTE2] = res;
  }
  return af::SUCCESS;
}

af::Status ExpandMTE3BlockLen(const NodeDetail &node_info, PerfOutputInfo &perf, std::vector<Expr> &dims);

af::Status StorePerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res_normal;
  Expr res_stride;
  GELOGD("Dma with Store: %s", node_info.ToString().c_str());
  std::vector<Expr> padded_dims{node_info.input_dims};
  // 应用MTE3场景的block_len padding（cache line大小128字节）
  GE_ASSERT_SUCCESS(ExpandMTE3BlockLen(node_info, perf, padded_dims), "Expand MTE3 block len failed, node[%s]",
                    node_info.ToString().c_str());
  GE_ASSERT_SUCCESS(GetPerf(
      {kMoveUbToGm, node_info.input_dtype[0], node_info.output_dtype[0], padded_dims, CreateExpr(0)}, res_normal));
  GE_ASSERT_SUCCESS(
      GetPerf({kMoveUbToGm + "Stride", node_info.input_dtype[0], node_info.output_dtype[0], node_info.repeats,
               node_info.gm_stride, node_info.block_count_idx,
               static_cast<int32_t>(node_info.repeats.size() > kCodegenDmaMaxLen
                                        ? std::min(node_info.repeats.size() - kCodegenDmaMaxLen, kCodegenLoopModeMaxLen)
                                        : 0U)},
              res_stride));
  GE_ASSERT_SUCCESS(AppendCacheLineConfig(node_info, CacheLineDirection::kUbToGm, perf.cache_line_config));
  perf.pipe_res[PipeType::AIV_MTE3] = res_normal + res_stride;
  return af::SUCCESS;
}

af::Status ExpandBlockLen(const NodeDetail &node_info, PerfOutputInfo &perf, std::vector<Expr> &dims) {
  // 满足gm非连续且搬运小于cache line，扩展每次搬运数据量到cache line
  StaticExpandResult static_res;

  if (TryInitStaticExpand(node_info, dims, static_res) == af::SUCCESS) {
    // 存在非连续并且block_len较小，无法并包，考虑将block_len对齐到cache line大小
    // 原有行为：block_len_ref是局部拷贝，赋值不反映到dims
    const int32_t block_len_bytes = static_res.blk_len_val * static_res.data_type_size_val;
    if ((block_len_bytes < static_res.cache_line_size) && (static_res.stride_val > 0)) {
      if (IsFlattenedCodegenShape(node_info, dims)) {
        dims.emplace_back(static_res.cache_line_ele_num);
      } else {
        dims[dims.size() - 1UL] = static_res.cache_line_ele_num;
      }
      GELOGD("Nddma: block_len padded to codegen tail cache line (static)");
    }
  } else {
    auto block_len_ref_actual = GetCodegenTailAxis(node_info, dims);
    auto &cache_line_ele_num = static_res.cache_line_ele_num;
    auto is_small_block_len_checker = af::sym::LogicalAnd(
        {af::sym::Gt(node_info.gm_stride, CreateExpr(0)), af::sym::Lt(block_len_ref_actual, cache_line_ele_num)});
    bool is_small_block_len{false};
    if (is_small_block_len_checker.IsConstExpr()) {
      is_small_block_len_checker.GetConstValue(is_small_block_len);
      block_len_ref_actual = is_small_block_len ? std::move(cache_line_ele_num) : std::move(block_len_ref_actual);
    } else {
      Expr res = CreateExpr("block_len");
      auto normal_case = std::make_shared<IfCase>(block_len_ref_actual);
      GE_ASSERT_NOTNULL(normal_case);
      auto small_block_len_case = std::make_shared<IfCase>(cache_line_ele_num);
      GE_ASSERT_NOTNULL(small_block_len_case);
      TernaryOp ternary_op = TernaryOp(CondType::K_EQ, is_small_block_len_checker, CreateExpr(false),
                                       std::move(normal_case), std::move(small_block_len_case));
      ternary_op.SetVariable(res);
      perf.ternary_ops[res] = ternary_op;
      block_len_ref_actual = res;
    }
    if (IsFlattenedCodegenShape(node_info, dims)) {
      dims.emplace_back(block_len_ref_actual);
    } else {
      dims[dims.size() - 1UL] = block_len_ref_actual;
    }
    GELOGD("Block len checker[%s], is_small_block_len[%d], block_len[%s]", is_small_block_len_checker.Serialize().get(),
           is_small_block_len, block_len_ref_actual.Serialize().get());
  }
  return af::SUCCESS;
}

af::Status ExpandMTE3BlockLen(const NodeDetail &node_info, PerfOutputInfo &perf, std::vector<Expr> &dims) {
  // 满足GM非连续且搬运小于cache line，扩展每次搬运数据量到cache line大小
  StaticExpandResult static_res;

  if (TryInitStaticExpand(node_info, dims, static_res) == af::SUCCESS) {
    // MTE3静态分支：原有行为中block_len_ref是局部拷贝，pad未反映到dims
    int32_t block_len_bytes = static_res.blk_len_val * static_res.data_type_size_val;
    if ((block_len_bytes < static_res.cache_line_size) && (static_res.stride_val > 0)) {
      GELOGD("MTE3 Store: block_len needs padding (static)");
    }
  } else {
    // 动态shape处理 - 创建三元表达式
    auto &block_len_ref_actual = dims[dims.size() - 1UL];
    auto &cache_line_ele_num = static_res.cache_line_ele_num;
    const auto &data_type_size = kDataTypeSizeMap.find(node_info.input_dtype[0]);
    Expr block_len_bytes = af::sym::Mul(block_len_ref_actual, data_type_size->second);
    auto need_padding_checker =
        af::sym::LogicalAnd({af::sym::Gt(node_info.gm_stride, CreateExpr(0)),
                             af::sym::Lt(block_len_bytes, CreateExpr(static_res.cache_line_size))});
    bool need_padding{false};
    if (need_padding_checker.IsConstExpr()) {
      need_padding_checker.GetConstValue(need_padding);
      block_len_ref_actual = need_padding ? std::move(cache_line_ele_num) : std::move(block_len_ref_actual);
    } else {
      Expr res = CreateExpr("mte3_block_len");
      auto normal_case = std::make_shared<IfCase>(block_len_ref_actual);
      GE_ASSERT_NOTNULL(normal_case);
      auto padding_case = std::make_shared<IfCase>(cache_line_ele_num);
      GE_ASSERT_NOTNULL(padding_case);
      TernaryOp ternary_op = TernaryOp(CondType::K_EQ, need_padding_checker, CreateExpr(false), std::move(normal_case),
                                       std::move(padding_case));
      ternary_op.SetVariable(res);
      perf.ternary_ops[res] = ternary_op;
      block_len_ref_actual = res;
    }
    GELOGD("MTE3 Block len padding checker[%s], need_padding[%d], block_len[%s]",
           need_padding_checker.Serialize().get(), need_padding, block_len_ref_actual.Serialize().get());
  }
  return af::SUCCESS;
}

af::Status NddmaPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res_normal;
  Expr res_stride;
  GELOGD("Dma with nddma: %s", node_info.ToString().c_str());
  std::vector<Expr> dims{node_info.input_dims};
  // 将block_len先扩展到cache line再计算性能
  GE_ASSERT_TRUE(!dims.empty(), "Check node input dims failed, node[%s]", node_info.ToString().c_str());
  GE_ASSERT_SUCCESS(ExpandBlockLen(node_info, perf, dims), "Expand block len failed, node[%s]",
                    node_info.ToString().c_str());
  GE_ASSERT_SUCCESS(
      GetPerf({kNddma, node_info.input_dtype[0], node_info.output_dtype[0], dims, node_info.gm_stride}, res_normal));
  GE_ASSERT_SUCCESS(GetPerf({kNddma + "Stride", node_info.input_dtype[0], node_info.output_dtype[0], node_info.repeats,
                             node_info.gm_stride, node_info.block_count_idx},
                            res_stride));
  GE_ASSERT_SUCCESS(AppendCacheLineConfig(node_info, CacheLineDirection::kGmToUb, perf.cache_line_config));
  perf.pipe_res[PipeType::AIV_MTE2] = res_normal + res_stride;
  return af::SUCCESS;
}

}  // namespace ascendcapi_v2

namespace {
// Data movement APIs only need NodeDetail, so keep them registered for generic ApiPerfFactory lookup.
REGISTER_ASCENDC_EVAL_FUNC_TAG(kLoad, V2, ascendcapi_v2::LoadPerf);
REGISTER_ASCENDC_EVAL_FUNC_TAG(kStore, V2, ascendcapi_v2::StorePerf);
REGISTER_ASCENDC_EVAL_FUNC_TAG(kNddma, V2, ascendcapi_v2::NddmaPerf);
}  // namespace
}  // namespace att
