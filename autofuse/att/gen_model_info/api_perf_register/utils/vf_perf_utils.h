/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_VF_PERF_UTILS_H
#define AUTOFUSE_VF_PERF_UTILS_H

#include "base/att_const_values.h"
#include "ascir_node_param/ascir_node_param.h"
#include "gen_model_info/api_perf_register/perf_param.h"
#include "util/ternary_op.h"
namespace att {
class VfPerfUtils {
 public:
  // 根据MicroApiType和DataType获取MicroApi级别的性能
  // 场景1，没有直接可以映射的指令，需要基于原有指令进行拼接，可以使用该接口拼接
  // 场景2，有直接可以映射的指令，可以直接调用该接口获取
  static af::Status GetVfInstructPerf(const std::string &micro_api_type, const std::string &data_type, Expr &latency,
                                      Expr &throughput, bool use_calibrated_rsqrt = true);
  static af::Status AddVfInstructPerf(const std::string &vf_instruct_type, const std::string &data_type, Expr &latency,
                                      Expr &throughput, Expr repeat_time);
  static af::Status GetVfInstructDtypeMappingPerf(const std::string &vf_instruct_type, const std::string &input_dtype,
                                                  const std::string &output_dtype, Expr &latency, Expr &throughput);
  static af::Status AddVfInstructDtypeMappingPerf(const std::string &vf_instruct_type, const std::string &input_dtype,
                                                  const std::string &output_dtype, Expr &latency, Expr &throughput,
                                                  Expr repeat_time);
  // 获取vf头开销
  static Expr GetVFHeadCost();
  // 根据vf function子图解析的结果获取vf function的性能
  static af::Status GetVectorFunctionPerf(const std::vector<NodePerfInfo> &node_perf_infos,
                                          const ascir_param::VectorFuncNodeParams &vector_func_params,
                                          std::map<Expr, TernaryOp, ExprCmp> &ternary_ops, Expr &res);
};
}  // namespace att

#endif  // AUTOFUSE_VF_PERF_UTILS_H
