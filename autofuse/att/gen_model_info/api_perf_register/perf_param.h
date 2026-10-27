/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_PERF_PARAM_H
#define AUTOFUSE_PERF_PARAM_H

#include <string>
#include <map>
#include "base/arch_param.h"
#include "base/base_types.h"
#include "gen_model_info/parser/tuning_space.h"
#include "util/ternary_op.h"
namespace att {
struct NodePerfInfo {
  std::string optype;
  std::string input_dtype;
  std::string output_dtype;
  std::vector<Expr> dims;
  Expr gm_stride;
  int32_t block_count_idx{0};      // 用于 LoadStoreStrideV2Func，表示发生非连续的轴索引
  int32_t loop_mode_dim_count{0};  // Codegen 外抛到 LoopMode 的维度数
};
using PipeHeadPerfFunc = att::Expr (*)(const std::vector<att::NodeInfo> &,
                                       std::map<att::Expr, att::TernaryOp, att::ExprCmp> &);
class PerfParamTable {
 public:
  PerfParamTable() = default;
  virtual ~PerfParamTable() = default;
  [[nodiscard]] virtual const std::string *GetAscendCApiPerfTable() const = 0;
  [[nodiscard]] virtual PipeHeadPerfFunc GetPipeHeadPerfFunc(PipeType pipe_type) const = 0;
  // 获取MicroApi的latency/throughput等信息
  [[nodiscard]] virtual const std::vector<VfInstructPerf> &GetVfInstructPerfTable(
      [[maybe_unused]] const std::string &micro_api_type) const {
    static std::vector<VfInstructPerf> empty{};
    return empty;
  }
  // 获取同时依赖输入输出dtype的MicroApi性能信息
  [[nodiscard]] virtual const std::vector<VfInstructDtypeMappingPerf> &GetVfInstructDtypeMappingPerfTable(
      [[maybe_unused]] const std::string &micro_api_type) const {
    static std::vector<VfInstructDtypeMappingPerf> empty{};
    return empty;
  }
  // 获取Vector Function的头开销
  [[nodiscard]] virtual Expr GetVectorFunctionHeadCost() const {
    return CreateExpr(0);
  }
  // 获取每条MicroApi指令能处理的字节数
  [[nodiscard]] virtual uint32_t GetMicroApiLen() const {
    return arch_param::kDefaultVectorLenSize;
  }
  // 获取注册的关键字名
  [[nodiscard]] virtual std::string GetApiRegisterVerName() const {
    return "";
  }
  // 获取算子的头开销
  [[nodiscard]] virtual Expr GetOpHeadCost() const {
    return CreateExpr(0);
  }

 private:
  std::map<PipeType, PipeHeadPerfFunc> pipes_head_perf;
};
}  // namespace att

#endif  // AUTOFUSE_PERF_PARAM_H
