/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "vf_perf_utils.h"
#include <numeric>
#include "common_utils.h"
#include "api_perf_register/api_perf_factory.h"
namespace att {
namespace {
Expr GetDataTypeSize(const std::string &data_type) {
  constexpr int32_t kDefaultDataTypeSize = 4;
  const auto &iter = kDataTypeSizeMap.find(data_type);
  if (iter == kDataTypeSizeMap.end()) {
    GELOGW("data type %s not supported, use default %d byte", data_type.c_str(), kDefaultDataTypeSize);
    return CreateExpr(kDefaultDataTypeSize);
  }
  return iter->second;
}

const PerfParamTable *GetParamPerfTable() {
  const auto default_impl = ascgen_utils::GetAscIrAttImpl(kDefaultApi);
  GE_ASSERT_NOTNULL(default_impl);
  const auto api_name = ge::PtrToPtr<void, ge::char_t>(default_impl->GetApiPerf());
  GE_ASSERT_NOTNULL(api_name);
  auto api_perf = ApiPerfFactory::Instance().Create(api_name);
  GE_ASSERT_NOTNULL(api_perf);
  return api_perf->GetPerfParam();
}

bool ContainsDtype(const std::vector<std::string> &dtypes, const std::string &dtype) {
  return std::count(dtypes.begin(), dtypes.end(), dtype) > 0;
}

bool MatchDtypeMappingPerf(const VfInstructDtypeMappingPerf &api_perf, const std::string &input_dtype,
                           const std::string &output_dtype) {
  return ContainsDtype(api_perf.input_dtypes, input_dtype) && ContainsDtype(api_perf.output_dtypes, output_dtype);
}

bool FindDtypeMappingPerf(const std::vector<VfInstructDtypeMappingPerf> &api_perf_table, const std::string &input_dtype,
                          const std::string &output_dtype, Expr &latency, Expr &throughput) {
  for (const auto &api_perf : api_perf_table) {
    if (MatchDtypeMappingPerf(api_perf, input_dtype, output_dtype)) {
      latency = CreateExpr(api_perf.latency);
      throughput = CreateExpr(api_perf.throughput);
      return true;
    }
  }
  return false;
}

Expr GetVFApiCount(const NodePerfInfo &node_info, const uint32_t micro_api_len, bool strides_equal,
                   const ascir_param::VectorFuncNodeParams &vector_func_params) {
  const Expr data_type_size = GetDataTypeSize(node_info.input_dtype);
  if (vector_func_params.is_double_loop && strides_equal) {
    const Expr element_count = vector_func_params.all_strides[0] * vector_func_params.output_dims[0];
    return af::sym::Ceiling(element_count * data_type_size / CreateExpr(micro_api_len));
  }
  if (vector_func_params.is_double_loop && !strides_equal) {
    GE_ASSERT_TRUE(vector_func_params.output_dims.size() >= 2U,
                   "VectorFunc double loop with unequal strides requires output_dims size >= 2, but got %zu.",
                   vector_func_params.output_dims.size());
    return vector_func_params.output_dims[0] *
           af::sym::Ceiling(vector_func_params.output_dims[1] * data_type_size / CreateExpr(micro_api_len));
  }
  Expr dim_product = std::accumulate(node_info.dims.begin(), node_info.dims.end(), CreateExpr(1),
                                     [](const Expr &a, const Expr &b) { return a * b; });
  return af::sym::Ceiling(dim_product * data_type_size / CreateExpr(micro_api_len));
}

// 获取vf api的基础latency和throughput
af::Status GetVFNodePerf(const NodePerfInfo &node_info, const uint32_t micro_api_len, bool strides_equal,
                         const ascir_param::VectorFuncNodeParams &vector_func_params, Expr &latency, Expr &throughput) {
  GE_ASSERT_SUCCESS(VfPerfUtils::GetVfInstructPerf(node_info.optype, node_info.input_dtype, latency, throughput));
  // 简化计算，后续进一步考虑每个op的latency和throughput的掩盖
  Expr api_count = GetVFApiCount(node_info, micro_api_len, strides_equal, vector_func_params);
  api_count = api_count.Simplify();
  throughput = throughput * api_count;
  throughput = throughput.Simplify();
  GELOGD("Got node %s input %s reg base latency %s, throughput %s, api_count %s", node_info.optype.c_str(),
         node_info.input_dtype.c_str(), latency.Str().get(), throughput.Str().get(), api_count.Str().get());
  return af::SUCCESS;
}

af::Status GetVectorFunctionPerfByStrideStatus(const std::vector<NodePerfInfo> &node_perf_infos,
                                               const uint32_t micro_api_len, bool strides_equal,
                                               const ascir_param::VectorFuncNodeParams &vector_func_params, Expr &res) {
  Expr all_micro_api_cost = CreateExpr(0);
  Expr max_latency = CreateExpr(0);
  for (const auto &node_info : node_perf_infos) {
    Expr latency = CreateExpr(0);
    Expr throughput = CreateExpr(0);
    GE_ASSERT_SUCCESS(GetVFNodePerf(node_info, micro_api_len, strides_equal, vector_func_params, latency, throughput));
    all_micro_api_cost = af::sym::Add(all_micro_api_cost, throughput);
    max_latency = af::sym::Max(max_latency, latency);
  }
  max_latency = max_latency.Simplify();
  all_micro_api_cost = all_micro_api_cost.Simplify();
  res = af::sym::Add(all_micro_api_cost, max_latency);
  const auto vector_function_head_cost = GetParamPerfTable()->GetVectorFunctionHeadCost();
  res = af::sym::Add(res, vector_function_head_cost);
  res = res.Simplify();
  GELOGD("Got vector function perf %s, vector_function_head_cost %s, max_latency %s, all_micro_api_cost %s",
         res.Str().get(), vector_function_head_cost.Str().get(), max_latency.Str().get(),
         all_micro_api_cost.Str().get());
  return af::SUCCESS;
}

}  // namespace

af::Status VfPerfUtils::GetVfInstructPerf(const std::string &micro_api_type, const std::string &data_type,
                                          Expr &latency, Expr &throughput) {
  const auto param_table = GetParamPerfTable();
  GE_ASSERT_NOTNULL(param_table);
  const auto &api_perf_table = param_table->GetVfInstructPerfTable(micro_api_type);
  for (const auto &api_perf : api_perf_table) {
    if (std::count(api_perf.support_data_types.begin(), api_perf.support_data_types.end(), data_type) > 0) {
      latency = CreateExpr(api_perf.latency);
      throughput = CreateExpr(api_perf.throughput);
      break;
    }
  }
  return af::SUCCESS;
}

af::Status VfPerfUtils::AddVfInstructPerf(const std::string &vf_instruct_type, const std::string &data_type,
                                          Expr &latency, Expr &throughput, Expr repeat_time) {
  const auto param_table = GetParamPerfTable();
  GE_ASSERT_NOTNULL(param_table);
  const auto &api_perf_table = param_table->GetVfInstructPerfTable(vf_instruct_type);
  GELOGD("Begin to add perf of vf instruct [%s].", vf_instruct_type.c_str());
  for (const auto &api_perf : api_perf_table) {
    if (std::count(api_perf.support_data_types.begin(), api_perf.support_data_types.end(), data_type) > 0) {
      GELOGD("Found perf of vf instruct [%s]: latency is {%d} cycles, throughput is {%d} cycles, repeat_time is [%s].",
             vf_instruct_type.c_str(), api_perf.latency, api_perf.throughput,
             af::SymbolicUtils::ToString(repeat_time).c_str());
      latency = af::sym::Max(CreateExpr(api_perf.latency), latency);
      throughput = throughput + CreateExpr(api_perf.throughput) * repeat_time;
      break;
    }
  }
  return af::SUCCESS;
}

af::Status VfPerfUtils::GetVfInstructDtypeMappingPerf(const std::string &vf_instruct_type,
                                                      const std::string &input_dtype, const std::string &output_dtype,
                                                      Expr &latency, Expr &throughput) {
  latency = CreateExpr(0);
  throughput = CreateExpr(0);
  const auto param_table = GetParamPerfTable();
  GE_ASSERT_NOTNULL(param_table);
  const auto &api_perf_table = param_table->GetVfInstructDtypeMappingPerfTable(vf_instruct_type);
  if (FindDtypeMappingPerf(api_perf_table, input_dtype, output_dtype, latency, throughput) ||
      FindDtypeMappingPerf(api_perf_table, input_dtype, kDefault, latency, throughput) ||
      FindDtypeMappingPerf(api_perf_table, kDefault, output_dtype, latency, throughput) ||
      FindDtypeMappingPerf(api_perf_table, kDefault, kDefault, latency, throughput)) {
    GELOGD(
        "Found dtype mapping perf of vf instruct [%s]: input_dtype is [%s], output_dtype is [%s], latency is [%s], "
        "throughput is [%s].",
        vf_instruct_type.c_str(), input_dtype.c_str(), output_dtype.c_str(), latency.Str().get(),
        throughput.Str().get());
  }
  return af::SUCCESS;
}

af::Status VfPerfUtils::AddVfInstructDtypeMappingPerf(const std::string &vf_instruct_type,
                                                      const std::string &input_dtype, const std::string &output_dtype,
                                                      Expr &latency, Expr &throughput, Expr repeat_time) {
  Expr api_latency = CreateExpr(0);
  Expr api_throughput = CreateExpr(0);
  GE_ASSERT_SUCCESS(
      GetVfInstructDtypeMappingPerf(vf_instruct_type, input_dtype, output_dtype, api_latency, api_throughput));
  latency = af::sym::Max(api_latency, latency);
  throughput = throughput + api_throughput * repeat_time;
  return af::SUCCESS;
}

Expr VfPerfUtils::GetVFHeadCost() {
  const auto param_table = GetParamPerfTable();
  GE_ASSERT_NOTNULL(param_table);
  return param_table->GetVectorFunctionHeadCost();
}

// Vector Function建模影响因素：
// 1.Micro Api的latency, throughput;
// 2.调用Micro Api的次数
// 3.Vector Function的启动开销
// 4.调用Micro Api的循环轴每次的循环数对应的头开销
// 5.Micro Api的并发度
// 第一版建模简化模型，仅考虑1,2,3，每个op的latency求最大值，throughput求和
// 后续建模考虑4,5
af::Status VfPerfUtils::GetVectorFunctionPerf(const std::vector<NodePerfInfo> &node_perf_infos,
                                              const ascir_param::VectorFuncNodeParams &vector_func_params,
                                              std::map<Expr, TernaryOp, ExprCmp> &ternary_ops, Expr &res) {
  const auto param_table = GetParamPerfTable();
  GE_ASSERT_NOTNULL(param_table);
  const uint32_t micro_api_len = param_table->GetMicroApiLen();
  const auto &all_strides = vector_func_params.all_strides;
  if (!vector_func_params.is_double_loop) {
    return GetVectorFunctionPerfByStrideStatus(node_perf_infos, micro_api_len, false, vector_func_params, res);
  }
  if (all_strides.empty()) {
    return GetVectorFunctionPerfByStrideStatus(node_perf_infos, micro_api_len, false, vector_func_params, res);
  }
  if (all_strides.size() == 1U) {
    return GetVectorFunctionPerfByStrideStatus(node_perf_infos, micro_api_len, true, vector_func_params, res);
  }
  Expr strides_equal_perf;
  Expr strides_unequal_perf;
  GE_ASSERT_SUCCESS(GetVectorFunctionPerfByStrideStatus(node_perf_infos, micro_api_len, true, vector_func_params,
                                                        strides_equal_perf));
  GE_ASSERT_SUCCESS(GetVectorFunctionPerfByStrideStatus(node_perf_infos, micro_api_len, false, vector_func_params,
                                                        strides_unequal_perf));
  res = CreateExpr("vf_dynamic_perf");
  GetPerfVar("vf_dynamic_perf", res, ternary_ops);
  std::shared_ptr<IfCase> equal_case = std::make_shared<IfCase>(strides_equal_perf);
  for (size_t i = all_strides.size() - 1U; i > 0U; --i) {
    equal_case = std::make_shared<IfCase>(CondType::K_EQ, all_strides[0], all_strides[i], std::move(equal_case),
                                          std::make_shared<IfCase>(strides_unequal_perf));
  }
  std::vector<Expr> related_vars;
  equal_case->GetUsedArgs(related_vars);
  ternary_ops[res] = TernaryOp(res, std::move(equal_case), related_vars);
  ternary_ops[res].SetVariable(res);
  ternary_ops[res].SetDescription("vf_dynamic_perf");
  return af::SUCCESS;
}
}  // namespace att
