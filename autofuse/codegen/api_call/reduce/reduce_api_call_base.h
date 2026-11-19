/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __AUTOFUSE_REDUCE_API_CALL_BASE_H__
#define __AUTOFUSE_REDUCE_API_CALL_BASE_H__

#include <map>
#include <sstream>
#include <string>
#include "codegen_kernel.h"

namespace reduce_base {
using namespace codegen;

struct ReduceOpType {
  static constexpr int32_t kMin = 0;
  static constexpr int32_t kMax = 1;
  static constexpr int32_t kSum = 2;
  static constexpr int32_t kProd = 3;
  static constexpr int32_t kAny = 4;
  static constexpr int32_t kAll = 5;
  static constexpr int32_t kMean = 6;
  static constexpr int32_t kArgmax = 7;
  static constexpr int32_t kArgMaxMultiRPhase1 = 8;
  static constexpr int32_t kArgMaxMultiRPhase2 = 9;
};

struct ReduceCodegenShadowCheckInput {
  af::AscNodePtr node;
  std::string api_name;
  const TPipe *tpipe{nullptr};
  const Tensor *input{nullptr};
  const Tensor *output{nullptr};
  ascir::AxisId axis_id{-1};
  bool has_reuse{false};
  bool is_reuse_source{false};
};

static std::map<std::string, std::pair<int, std::string>> reduce_type_map = {
    {"ReduceMin", {ReduceOpType::kMin, "Min"}},
    {"ReduceMax", {ReduceOpType::kMax, "Max"}},
    {"Argmax", {ReduceOpType::kMax, "Max"}},
    {"ArgMaxMultiRPhase1", {ReduceOpType::kMax, "Max"}},
    {"ArgMaxMultiRPhase2", {ReduceOpType::kMax, "Max"}},
    {"ReduceAny", {ReduceOpType::kAny, "Max"}},
    {"ReduceAll", {ReduceOpType::kAll, "Min"}},
    {"ReduceSum", {ReduceOpType::kSum, "Add"}},
    {"ReduceProd", {ReduceOpType::kProd, "Mul"}},
    {"ReduceMean", {ReduceOpType::kMean, "Add"}},
};

void GetIsArAndPattern(const Tensor &x, const Tensor &y, bool &isAr, std::string &reduce_pattern);
void CheckReduceSpecificParamsForCodegen(const ReduceCodegenShadowCheckInput &input);
void ReduceMergedSizeCodeGen(const TPipe &tpipe, std::vector<std::string> &lines, const Tensor &src, const Tensor &dst,
                             bool is_tail = false);
bool IsNeedMultiReduce(const Tiler &tiler, const Tensor &input, const Tensor &output, ascir::AxisId axis_id);
void ReduceMeanCodeGen(std::string &dtype_name, const TPipe &tpipe, const Tensor &src, const Tensor &dst,
                       std::vector<std::string> &lines);
void ReduceInitCodeGen(const Tensor &x, const Tensor &y, const int &type_value, std::vector<std::string> &lines,
                       const TPipe &tpipe, const std::string &dtype_name,
                       const std::string &x_name_after_reinterpret = "");
void ReduceDimACodeGen(const Tensor &x, const std::string &apiName, std::vector<std::string> &lines);
Status GetDtypeNameForReduce(const std::string &api_name, const Tensor &x, const Tensor &y, std::string &dtype_name);
void GenAccumulatedOffsetDeclForArgmax(const std::string &api_name, const Tensor &x, const Tensor &y,
                                       const TPipe &tpipe, std::vector<std::string> &lines);

/**
 * @brief 生成获取最后两个R轴大小乘积的代码字符串
 * @param x 输入张量
 * @param y 输出张量
 * @param tpipe Tiler对象
 * @param lines 输出代码行集合，每条语句一个 emplace_back
 */
void GenLastTwoRAxisSizeProductCode(const Tensor &x, const Tensor &y, const TPipe &tpipe,
                                    std::vector<std::string> &lines);
}  // namespace reduce_base
#endif  // __AUTOFUSE_REDUCE_API_CALL_BASE_H__
