/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "perf_param_v2.h"
#include "nddma_model.h"
#include "v35/att/api_perf_register/ascir_reduce_api_perf_v2.h"
#include "v35/att/api_perf_register/ascendc_api_perf/broadcast_api_perf_v2.h"
#include "v35/att/api_perf_register/ascendc_regbase_perf.h"
#include "api_perf_register/api_perf_factory.h"
#include "api_perf_register/ascendc_api_perf.h"
namespace att {
namespace {
constexpr int32_t kMaxDmaLen = 4;
constexpr int32_t kMaxNddmaLen = 5;
PerfParamTableV2 perf_param_table_v2;
TilingScheduleConfigTableV2 tiling_schedule_config_table_v2;

std::vector<int64_t> GetNddmaVectorizedAxis(const af::AscNodePtr &node) {
  if (node == nullptr || node->outputs().empty()) {
    return {};
  }
  return node->outputs[0].attr.vectorized_axis;
}

af::Status TryNewNddmaModel(const TensorShapeInfo &shape_info, const NodeInfo &node, NodeDetail &node_detail,
                            PerfOutputInfo &perf_res, bool &selected) {
  selected = false;
  NddmaDescriptorInfo descriptor;
  NddmaModelResult result;
  GELOGD(
      "[ATT NDDMA] evaluate: node=%s, cv_ub_fusion=%d, raw_rank=%zu, repeats=[%s], gm_strides=[%s], "
      "ub_strides=[%s]",
      node_detail.name.c_str(), node.is_cv_ub_fusion, shape_info.repeats.size(),
      GetVecString(shape_info.repeats).c_str(), GetVecString(shape_info.gm_strides).c_str(),
      GetVecString(shape_info.strides).c_str());
  // kUBFuse Codegen 分支生成 {curAivM, curAlignN} 和固定 2D stride，与下方 raw 描述不等价；
  // 在专用 2D 模型接入前保守回退 legacy ATT 模型。
  if (node.is_cv_ub_fusion) {
    result.raw_rank = shape_info.repeats.size();
    result.fallback_reason = NddmaFallbackReason::kCodegenMismatch;
    LogNddmaFallback(node_detail.name, shape_info.data_type, nullptr, result);
    return af::SUCCESS;
  }
  uint64_t expected_dtype_size = 0U;
  const auto dtype_iter = kDataTypeSizeMap.find(shape_info.data_type);
  if (dtype_iter == kDataTypeSizeMap.end() || !dtype_iter->second.GetConstValue(expected_dtype_size) ||
      expected_dtype_size == 0U || expected_dtype_size != shape_info.data_type_size) {
    result.raw_rank = shape_info.repeats.size();
    result.fallback_reason = NddmaFallbackReason::kDtypeUnsupported;
    LogNddmaFallback(node_detail.name, shape_info.data_type, nullptr, result);
    return af::SUCCESS;
  }
  std::vector<bool> tile_inner;
  if (!node.outputs.empty() && node.outputs[0] != nullptr &&
      node.outputs[0]->dim_info.size() == shape_info.repeats.size()) {
    tile_inner.reserve(node.outputs[0]->dim_info.size());
    for (const auto *axis : node.outputs[0]->dim_info) {
      tile_inner.push_back(axis != nullptr && axis->axis_type == AxisPosition::INNER && !axis->is_bind_multi_core);
    }
  }
  const auto build_reason =
      BuildNddmaDescriptor(shape_info, GetNddmaVectorizedAxis(node.node_ptr), descriptor, tile_inner);
  if (build_reason != NddmaFallbackReason::kNone) {
    result.raw_rank = shape_info.repeats.size();
    result.fallback_reason = build_reason;
    LogNddmaFallback(node_detail.name, shape_info.data_type, nullptr, result);
    return af::SUCCESS;
  }
  node_detail.nddma_descriptor = descriptor;
  GE_ASSERT_SUCCESS(EvaluateNddmaModel(descriptor, shape_info.data_type, CreateExpr("block_dim"), result));
  result.raw_rank = shape_info.repeats.size();
  if (!result.selected) {
    LogNddmaFallback(node_detail.name, shape_info.data_type, &descriptor, result);
    return af::SUCCESS;
  }
  perf_res.pipe_res[PipeType::AIV_MTE2] = result.cycles;
  perf_res.ternary_ops.insert(result.ternary_ops.begin(), result.ternary_ops.end());
  selected = true;
  GELOGD("[ATT NDDMA] selected: node=%s, model=%s, raw_rank=%zu, effective_rank=%zu", node_detail.name.c_str(),
         result.model_name.c_str(), result.raw_rank, result.effective_rank);
  return af::SUCCESS;
}

ApiPerfRegister<ApiPerf> ApiPerfRegisterV2(const std::string &api_name, Perf perf_func, MicroPerfFunc micro_perf_func,
                                           const PerfParamTable *perf_param,
                                           const TilingScheduleConfigTable *tiling_schedule_config_table) {
  return ApiPerfRegister<ApiPerf>(api_name + "V2", perf_func, micro_perf_func, perf_param,
                                  tiling_schedule_config_table);
}

ApiPerfRegister<ApiPerf> ApiPerfRegisterV2(const std::string &api_name, const std::string &perf_func_name,
                                           MicroPerfFunc micro_perf_func, const PerfParamTable *perf_param,
                                           const TilingScheduleConfigTable *tiling_schedule_config_table) {
  return ApiPerfRegister<ApiPerf>(api_name + "V2", perf_func_name, micro_perf_func, perf_param,
                                  tiling_schedule_config_table);
}
namespace ascir_v2 {
/*
LoadApi(DataCopy from GM to UB)的性能公式：（其中a-b-c-d-e为待拟合参数）
  1. 单次MTE2 = S(数据量Byte)/T + h(指令头开销)，针对非连续搬运场景会增加stride建模值(0.043 * (stride % (256) *
block_count))
  2. 总MTE2 = 单次MTE2 * 调用次数 + H(pipe启动头开销)
  当Shape > 256B时：
  3. H = 1174.3
  4. h = 34
  5. T = 11.8292 + 6.6155 / blockdim
     (单核的峰值带宽，核数越多，带宽抢占越严重，直到收敛到稳定值)
  当前Shape <= 256B时：
  3. H = 775.0
  4. h = 15.01
  5. T = 13.1355 + 6.4088 / blockdim
  6. mte2 = S/T + h
  7. overall_mte2 = mte2 * mte2_count + H
  8. 外抛for循环：最外侧4个维度丢到循环次数里面去
*/
af::Status LoadApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  auto const &node_ptr = node.node_ptr;
  GE_ASSERT_TRUE(!input_shapes.empty());
  GE_ASSERT_TRUE(!output_shapes.empty());
  std::string node_name = node_ptr != nullptr ? node_ptr->GetName() : "LoadNode";
  auto merged_output_shapes = output_shapes[0];
  GE_ASSERT_SUCCESS(MergeTensorContinuousDims(node_ptr, GetNodeOutTensorName(node_ptr, 0), merged_output_shapes));
  NodeDetail dma_info;
  dma_info.name = node_name;
  dma_info.optype = node_ptr->GetType();
  dma_info.input_dtype = {merged_output_shapes.data_type};
  dma_info.output_dtype = {merged_output_shapes.data_type};
  GE_ASSERT_SUCCESS(SetDims(merged_output_shapes, dma_info));
  GE_ASSERT_SUCCESS(GetDmaPerf(merged_output_shapes, dma_info, perf_res, kMaxDmaLen));
  return af::SUCCESS;
}

/*
NddmaApi(MultiDataCopy from GM to UB)的性能公式：
  1. 单次Nddma = S(数据量Byte)/T + h(指令头开销)
  2. 总Nddma = 单次Nddma * 调用次数 + H(pipe启动头开销)
  3. 当Shape > 256B时：H = 1174.3，当Shape <= 256B时：H = 775.0
  4. h = 418.9789
  5. T = 7.61 + 6.39 / blockdim
     (单核的峰值带宽，核数越多，带宽抢占越严重，直到收敛到稳定值)
  6. nddma = S/T + h
  7. overall_nddma = nddma * nddma_count + H
  8. 外抛for循环：最外侧4个维度丢到循环次数里面去
*/
af::Status NddmaApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                    [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                    [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  auto const &node_ptr = node.node_ptr;
  GE_ASSERT_TRUE(!input_shapes.empty());
  GE_ASSERT_TRUE(!output_shapes.empty());
  std::string node_name = node_ptr != nullptr ? node_ptr->GetName() : "NddmaNode";
  NodeDetail dma_info;
  dma_info.name = node_name;
  dma_info.optype = node_ptr != nullptr ? node_ptr->GetType() : "Nddma";
  dma_info.input_dtype = {output_shapes[0].data_type};
  dma_info.output_dtype = {output_shapes[0].data_type};
  bool selected = false;
  GE_ASSERT_SUCCESS(TryNewNddmaModel(output_shapes[0], node, dma_info, perf_res, selected));
  if (selected) {
    return af::SUCCESS;
  }
  auto merged_output_shapes = output_shapes[0];
  GE_ASSERT_SUCCESS(MergeTensorContinuousDims(node_ptr, GetNodeOutTensorName(node_ptr, 0), merged_output_shapes));
  GE_ASSERT_SUCCESS(SetDims(merged_output_shapes, dma_info));
  GE_ASSERT_SUCCESS(GetDmaPerf(merged_output_shapes, dma_info, perf_res, kMaxNddmaLen, false));
  return af::SUCCESS;
}

/*
StoreApiV2(DataCopy from UB to GM)的性能公式：（其中a-b-c-d为待拟合参数）
  1. 单次MTE3 = S(数据量Byte)/T + h(指令头开销)，针对非连续搬运场景会增加stride建模值(k*(stride%(256)*block_count))
  2. 总MTE3 = 单次MTE3 * 调用次数 + H(pipe启动头开销)
  3. H = 571
  4. h = 160
  5. T = 11.774 + 10.265 / blockdim(单核的峰值带宽，核数越多，带宽抢占越严重，直到收敛到稳定值)
  6. mte3 = S/T + h
  7. overall_mte3 = mte3 * mte3_count + H
  8. 外抛for循环：最外侧4个维度丢到循环次数里面去
*/
af::Status StoreApiV2([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                      [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                      [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  auto const &store_node_ptr = node.node_ptr;
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());
  std::string store_node_name = store_node_ptr != nullptr ? store_node_ptr->GetName() : "StoreNode";
  auto merged_output_shapes = output_shapes[0];
  GE_ASSERT_SUCCESS(
      MergeTensorContinuousDims(store_node_ptr, GetNodeOutTensorName(store_node_ptr, 0), merged_output_shapes));
  NodeDetail dma_info;
  dma_info.name = store_node_name;
  dma_info.optype = store_node_ptr->GetType();
  dma_info.input_dtype = {merged_output_shapes.data_type};
  dma_info.output_dtype = {merged_output_shapes.data_type};
  GE_ASSERT_SUCCESS(SetDims(merged_output_shapes, dma_info));
  GE_ASSERT_SUCCESS(GetDmaPerf(merged_output_shapes, dma_info, perf_res, kMaxDmaLen));
  return af::SUCCESS;
}

inline af::Status CompareSpecificPerf(const std::string &mode, const NodeDetail &node_info, PerfOutputInfo &perf) {
  if (mode == kGe) {
    ascendcperf_v2::CompareGEPerf(node_info, perf);
  } else if (mode == kEq) {
    ascendcperf_v2::CompareEQPerf(node_info, perf);
  } else if (mode == kNe) {
    ascendcperf_v2::CompareNEPerf(node_info, perf);
  } else if (mode == kGt) {
    ascendcperf_v2::CompareGTPerf(node_info, perf);
  } else if (mode == kLe) {
    ascendcperf_v2::CompareLEPerf(node_info, perf);
  } else if (mode == kLt) {
    ascendcperf_v2::CompareLTPerf(node_info, perf);
  } else {
    GELOGW("compare mode %s is not registered", mode.c_str());
  }
  return af::SUCCESS;
}

af::Status CompareApiV2([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, const std::string &mode, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(input_shapes.size() >= 2U && !output_shapes.empty());
  NodeDetail node_info;
  Expr outer_repeat;
  vector<Expr> used_dims;
  GE_ASSERT_SUCCESS(GetOuterParams(output_shapes[0].dims, outer_repeat, used_dims));
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(SetDims(used_dims, node_info));
  node_info.compare_node_params = node.compare_node_params;
  GE_ASSERT_SUCCESS(CompareSpecificPerf(mode, node_info, perf_res));
  return af::SUCCESS;
}

af::Status CompareGeApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  return CompareApiV2(input_shapes, output_shapes, node, kGe, perf_res);
}

af::Status CompareEqApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  return CompareApiV2(input_shapes, output_shapes, node, kEq, perf_res);
}

af::Status CompareNeApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  return CompareApiV2(input_shapes, output_shapes, node, kNe, perf_res);
}

af::Status CompareGtApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  return CompareApiV2(input_shapes, output_shapes, node, kGt, perf_res);
}

af::Status CompareLeApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  return CompareApiV2(input_shapes, output_shapes, node, kLe, perf_res);
}

af::Status CompareLtApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  return CompareApiV2(input_shapes, output_shapes, node, kLt, perf_res);
}

af::Status AbsApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::AbsPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status ExpApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::ExpPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status LnApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                 [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                 [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::LnPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status SqrtApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::SqrtPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status RsqrtApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                    [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                    [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::RsqrtPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status DivApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::DivPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status ReciprocalApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                         [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                         [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::ReciprocalPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status ReluApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::ReluPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status NegApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::NegPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status AddApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::AddPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status SubApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::SubPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status MulApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::MulPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status LeakyReluApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::LeakyReluPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status CastApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  node_info.cast_node_params = node.cast_node_params;
  GE_ASSERT_SUCCESS(ascendcperf_v2::CastPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status Ceil2IntApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                       [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                       [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  node_info.cast_node_params = node.cast_node_params;
  GE_ASSERT_SUCCESS(ascendcperf_v2::CastPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status RemovePadApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::RemovePadPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status WhereApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                    [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                    [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  node_info.where_node_params = node.where_node_params;
  GE_ASSERT_SUCCESS(ascendcperf_v2::WherePerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status PowApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::PowPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status ErfApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::ErfPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status TanhApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::TanhPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status SigmoidApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                      [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                      [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::SigmoidPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status GeluApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::GeluPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status SignApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::SignPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status LogicalNotApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                         [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                         [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::LogicalNotPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status LogicalOrApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::LogicalOrPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status LogicalAndApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                         [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                         [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::LogicalAndPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status ClipByValueApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                          [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                          [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::ClipByValuePerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status BitwiseAndApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                         [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                         [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::BitwiseAndPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status FloorDivApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                       [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                       [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf_v2::FloorDivPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status IsNanApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                    [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                    PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  node_info.unary_bitwidth_change_node_params = node.unary_bitwidth_change_node_params;
  GE_ASSERT_SUCCESS(ascendcperf_v2::IsNanPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status IsFiniteApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                       [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                       PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  node_info.unary_bitwidth_change_node_params = node.unary_bitwidth_change_node_params;
  GE_ASSERT_SUCCESS(ascendcperf_v2::IsFinitePerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status IsInfApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                    [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                    PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  node_info.unary_bitwidth_change_node_params = node.unary_bitwidth_change_node_params;
  GE_ASSERT_SUCCESS(ascendcperf_v2::IsInfPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status TransposeApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                        PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  node_info.transpose_node_params = node.transpose_node_params;
  GE_ASSERT_SUCCESS(ascendcperf_v2::TransposePerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status ValidateBroadcastBasic(const std::vector<TensorShapeInfo> &input_shapes,
                                  const std::vector<TensorShapeInfo> &output_shapes) {
  GE_ASSERT_TRUE(input_shapes.size() == 1U && output_shapes.size() == 1U,
                 "Broadcast requires exactly one input and one output shape.");
  GE_ASSERT_TRUE(!input_shapes[0].data_type.empty() && !output_shapes[0].data_type.empty(),
                 "Broadcast input/output dtype is missing.");
  return af::SUCCESS;
}

af::Status ValidateBroadcastShapes(const std::vector<TensorShapeInfo> &input_shapes,
                                   const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node) {
  if (node.broadcast_node_params.is_scalar) {
    GE_ASSERT_TRUE(node.broadcast_node_params.duplicate_count.IsValid(), "Broadcast scalar actual size is missing.");
    return af::SUCCESS;
  }
  GE_ASSERT_TRUE(!output_shapes[0].dims.empty(), "Broadcast output shape is missing.");
  GE_ASSERT_TRUE(!input_shapes[0].dims.empty());
  GE_ASSERT_TRUE(input_shapes[0].dims.size() == output_shapes[0].dims.size(),
                 "Broadcast input/output shape ranks are different.");
  GE_ASSERT_TRUE(input_shapes[0].repeats.size() == input_shapes[0].dims.size(),
                 "Broadcast input repeats length is invalid.");
  return af::SUCCESS;
}

af::Status LegacyBroadcastPerf(const std::vector<TensorShapeInfo> &input_shapes,
                               const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                               PerfOutputInfo &perf_res) {
  const auto legacy_perf = GetPerfFunc(kBroadcast);
  GE_ASSERT_NOTNULL(legacy_perf, "Legacy Broadcast performance function is unavailable.");
  return legacy_perf(input_shapes, output_shapes, node, perf_res);
}

af::Status BroadcastApiV2([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                          [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                          PerfOutputInfo &perf_res) {
  GE_ASSERT_SUCCESS(ValidateBroadcastBasic(input_shapes, output_shapes));
  if (!node.broadcast_node_params.valid ||
      (!node.broadcast_node_params.is_scalar &&
       (node.broadcast_node_params.src_shape.empty() || node.broadcast_node_params.dst_shape.empty()))) {
    return LegacyBroadcastPerf(input_shapes, output_shapes, node, perf_res);
  }
  GE_ASSERT_SUCCESS(ValidateBroadcastShapes(input_shapes, output_shapes, node));
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  node_info.broadcast_node_params = node.broadcast_node_params;
  if (!input_shapes.empty()) {
    node_info.repeats = input_shapes[0].repeats;
  }
  const auto new_status = ascendcperf_v2::BroadcastPerf(node_info, perf_res);
  if (new_status == af::SUCCESS) {
    return af::SUCCESS;
  }
  if (!ascendcperf_v2::IsBroadcastFallback(node_info)) {
    return new_status;
  }
  // Only a classified unsupported implementation branch is delegated to the legacy model.
  return LegacyBroadcastPerf(input_shapes, output_shapes, node, perf_res);
}
}  // namespace ascir_v2

REGISTER_EVAL_FUNC_TAG(kStore, V2, ascir_v2::StoreApiV2);
REGISTER_EVAL_FUNC_TAG(kLoad, V2, ascir_v2::LoadApi);
REGISTER_EVAL_FUNC_TAG(kNddma, V2, ascir_v2::NddmaApi);
REGISTER_EVAL_FUNC_TAG(kGe, V2, ascir_v2::CompareGeApi);
REGISTER_EVAL_FUNC_TAG(kEq, V2, ascir_v2::CompareEqApi);
REGISTER_EVAL_FUNC_TAG(kNe, V2, ascir_v2::CompareNeApi);
REGISTER_EVAL_FUNC_TAG(kGt, V2, ascir_v2::CompareGtApi);
REGISTER_EVAL_FUNC_TAG(kLe, V2, ascir_v2::CompareLeApi);
REGISTER_EVAL_FUNC_TAG(kLt, V2, ascir_v2::CompareLtApi);
REGISTER_EVAL_FUNC_TAG(kAbs, V2, ascir_v2::AbsApi);
REGISTER_EVAL_FUNC_TAG(kExp, V2, ascir_v2::ExpApi);
REGISTER_EVAL_FUNC_TAG(kLn, V2, ascir_v2::LnApi);
REGISTER_EVAL_FUNC_TAG(kSqrt, V2, ascir_v2::SqrtApi);
REGISTER_EVAL_FUNC_TAG(kRsqrt, V2, ascir_v2::RsqrtApi);
REGISTER_EVAL_FUNC_TAG(kDiv, V2, ascir_v2::DivApi);
REGISTER_EVAL_FUNC_TAG(kReciprocal, V2, ascir_v2::ReciprocalApi);
REGISTER_EVAL_FUNC_TAG(kRelu, V2, ascir_v2::ReluApi);
REGISTER_EVAL_FUNC_TAG(kMax, V2, ascir_reduce_v2::MaxApi);
REGISTER_EVAL_FUNC_TAG(kMin, V2, ascir_reduce_v2::MinApi);
REGISTER_EVAL_FUNC_TAG(kMaximum, V2, ascir_reduce_v2::ElementwiseMaxApi);
REGISTER_EVAL_FUNC_TAG(kMinimum, V2, ascir_reduce_v2::ElementwiseMinApi);
REGISTER_EVAL_FUNC_TAG(kAny, V2, ascir_reduce_v2::AnyApi);
REGISTER_EVAL_FUNC_TAG(kAll, V2, ascir_reduce_v2::AllApi);
REGISTER_EVAL_FUNC_TAG(kReduceMax, V2, ascir_reduce_v2::ReduceMaxApi);
REGISTER_EVAL_FUNC_TAG(kReduceMin, V2, ascir_reduce_v2::ReduceMinApi);
REGISTER_EVAL_FUNC_TAG(kReduceAny, V2, ascir_reduce_v2::ReduceAnyApi);
REGISTER_EVAL_FUNC_TAG(kReduceAll, V2, ascir_reduce_v2::ReduceAllApi);
REGISTER_EVAL_FUNC_TAG(kReduceSum, V2, ascir_reduce_v2::ReduceSumApi);
REGISTER_EVAL_FUNC_TAG(kReduceMean, V2, ascir_reduce_v2::ReduceMeanApi);
REGISTER_EVAL_FUNC_TAG(kReduceProd, V2, ascir_reduce_v2::ReduceProdApi);
REGISTER_EVAL_FUNC_TAG(kNeg, V2, ascir_v2::NegApi);
REGISTER_EVAL_FUNC_TAG(kMean, V2, ascir_reduce_v2::MeanApi);
REGISTER_EVAL_FUNC_TAG(kAdd, V2, ascir_v2::AddApi);
REGISTER_EVAL_FUNC_TAG(kSub, V2, ascir_v2::SubApi);
REGISTER_EVAL_FUNC_TAG(kMul, V2, ascir_v2::MulApi);
REGISTER_EVAL_FUNC_TAG(kProd, V2, ascir_reduce_v2::ProdApi);
REGISTER_EVAL_FUNC_TAG(kLeakyRelu, V2, ascir_v2::LeakyReluApi);
REGISTER_EVAL_FUNC_TAG(kCast, V2, ascir_v2::CastApi);
REGISTER_EVAL_FUNC_TAG(kCeil2Int, V2, ascir_v2::Ceil2IntApi);
REGISTER_EVAL_FUNC_TAG(kSum, V2, ascir_reduce_v2::SumApi);
REGISTER_EVAL_FUNC_TAG(kRemovePad, V2, ascir_v2::RemovePadApi);
REGISTER_EVAL_FUNC_TAG(kWhere, V2, ascir_v2::WhereApi);
REGISTER_EVAL_FUNC_TAG(kBroadcast, V2, ascir_v2::BroadcastApiV2);
REGISTER_EVAL_FUNC_TAG(kPow, V2, ascir_v2::PowApi);
REGISTER_EVAL_FUNC_TAG(kErf, V2, ascir_v2::ErfApi);
REGISTER_EVAL_FUNC_TAG(kTanh, V2, ascir_v2::TanhApi);
REGISTER_EVAL_FUNC_TAG(kSigmoid, V2, ascir_v2::SigmoidApi);
REGISTER_EVAL_FUNC_TAG(kGelu, V2, ascir_v2::GeluApi);
REGISTER_EVAL_FUNC_TAG(kSign, V2, ascir_v2::SignApi);
REGISTER_EVAL_FUNC_TAG(kLogicalNot, V2, ascir_v2::LogicalNotApi);
REGISTER_EVAL_FUNC_TAG(kLogicalOr, V2, ascir_v2::LogicalOrApi);
REGISTER_EVAL_FUNC_TAG(kLogicalAnd, V2, ascir_v2::LogicalAndApi);
REGISTER_EVAL_FUNC_TAG(kClipByValue, V2, ascir_v2::ClipByValueApi);
REGISTER_EVAL_FUNC_TAG(kBitwiseAnd, V2, ascir_v2::BitwiseAndApi);
REGISTER_EVAL_FUNC_TAG(kFloorDiv, V2, ascir_v2::FloorDivApi);
REGISTER_EVAL_FUNC_TAG(kIsnan, V2, ascir_v2::IsNanApi);
REGISTER_EVAL_FUNC_TAG(kIsFinite, V2, ascir_v2::IsFiniteApi);
REGISTER_EVAL_FUNC_TAG(kIsInf, V2, ascir_v2::IsInfApi);
REGISTER_EVAL_FUNC_TAG(kTranspose, V2, ascir_v2::TransposeApi);
ApiPerfRegister<ApiPerf> add_api_perf_v2(ApiPerfRegisterV2(kAdd, kAdd + "V2", nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> gather_api_perf_v2(ApiPerfRegisterV2(kGather, kGather, nullptr, &perf_param_table_v2,
                                                              &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> indirect_load_api_perf_v2(ApiPerfRegisterV2(kIndirectLoad, kUnitVector, nullptr,
                                                                     &perf_param_table_v2,
                                                                     &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> abs_api_perf_v2(ApiPerfRegisterV2(kAbs, kAbs + "V2", nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> broadcast_api_perf_v2(ApiPerfRegisterV2(kBroadcast, GetPerfFunc(kBroadcast + "V2"), nullptr,
                                                                 &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> cast_api_perf_v2(ApiPerfRegisterV2(kCast, kCast + "V2", nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> div_api_perf_v2(ApiPerfRegisterV2(kDiv, kDiv + "V2", nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> erf_api_perf_v2(ApiPerfRegisterV2(kErf, kErf + "V2", nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> exp_api_perf_v2(ApiPerfRegisterV2(kExp, kExp + "V2", nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> exp2_api_perf_v2(ApiPerfRegisterV2(kExp2, kExp2 + "V2", nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> floor_api_perf_v2(ApiPerfRegisterV2(kFloor, kFloor + "V2", nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> fma_api_perf_v2(ApiPerfRegisterV2(kFma, kFma + "V2", nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> bitwise_not_api_perf_v2(ApiPerfRegisterV2(kBitwiseNot, kUnitVector, nullptr,
                                                                   &perf_param_table_v2,
                                                                   &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> bitwise_or_api_perf_v2(ApiPerfRegisterV2(kBitwiseOr, kUnitVector, nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> bitwise_xor_api_perf_v2(ApiPerfRegisterV2(kBitwiseXor, kUnitVector, nullptr,
                                                                   &perf_param_table_v2,
                                                                   &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> ceil_api_perf_v2(ApiPerfRegisterV2(kCeil, kUnitVector, nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> cos_api_perf_v2(ApiPerfRegisterV2(kCos, kUnitVector, nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> acos_api_perf_v2(ApiPerfRegisterV2(kAcos, kUnitVector, nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> cosh_api_perf_v2(ApiPerfRegisterV2(kCosh, kUnitVector, nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> atan2_api_perf_v2(ApiPerfRegisterV2(kAtan2, kUnitVector, nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> copysign_api_perf_v2(ApiPerfRegisterV2(kCopySign, kUnitVector, nullptr, &perf_param_table_v2,
                                                                &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> ceil2int_api_perf_v2(ApiPerfRegisterV2(kCeil2Int, kCeil2Int + "V2", nullptr,
                                                                &perf_param_table_v2,
                                                                &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> logical_and_api_perf_v2(ApiPerfRegisterV2(kLogicalAnd, kLogicalAnd + "V2", nullptr,
                                                                   &perf_param_table_v2,
                                                                   &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> logical_or_api_perf_v2(ApiPerfRegisterV2(kLogicalOr, kLogicalOr + "V2", nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> logical_not_api_perf_v2(ApiPerfRegisterV2(kLogicalNot, kLogicalNot + "V2", nullptr,
                                                                   &perf_param_table_v2,
                                                                   &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> maximum_api_perf_v2(ApiPerfRegisterV2(kMaximum, kMaximum + "V2", nullptr, &perf_param_table_v2,
                                                               &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> minimum_api_perf_v2(ApiPerfRegisterV2(kMinimum, kMinimum + "V2", nullptr, &perf_param_table_v2,
                                                               &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> max_api_perf_v2(ApiPerfRegisterV2(kMax, kMax + "V2", nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> reduce_max_api_perf_v2(ApiPerfRegisterV2(kReduceMax, kReduceMax + "V2", nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> min_api_perf_v2(ApiPerfRegisterV2(kMin, kMin + "V2", nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> reduce_min_api_perf_v2(ApiPerfRegisterV2(kReduceMin, kReduceMin + "V2", nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> mul_api_perf_v2(ApiPerfRegisterV2(kMul, kMul + "V2", nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> neg_api_perf_v2(ApiPerfRegisterV2(kNeg, kNeg + "V2", nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> reciprocal_api_perf_v2(ApiPerfRegisterV2(kReciprocal, kReciprocal + "V2", nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> relu_api_perf_v2(ApiPerfRegisterV2(kRelu, kRelu + "V2", nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> remove_pad_api_perf_v2(ApiPerfRegisterV2(kRemovePad, kRemovePad + "V2", nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> rsqrt_api_perf_v2(ApiPerfRegisterV2(kRsqrt, kRsqrt + "V2", nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> sign_api_perf_v2(ApiPerfRegisterV2(kSign, kSign + "V2", nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> sqrt_api_perf_v2(ApiPerfRegisterV2(kSqrt, kSqrt + "V2", nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> sub_api_perf_v2(ApiPerfRegisterV2(kSub, kSub + "V2", nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> tanh_api_perf_v2(ApiPerfRegisterV2(kTanh, kTanh + "V2", nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> sin_api_perf_v2(ApiPerfRegisterV2(kSin, kSin + "V2", nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> asin_api_perf_v2(ApiPerfRegisterV2(kAsin, kUnitVector, nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> asinh_api_perf_v2(ApiPerfRegisterV2(kAsinh, kUnitVector, nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> atan_api_perf_v2(ApiPerfRegisterV2(kAtan, kUnitVector, nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> atanh_api_perf_v2(ApiPerfRegisterV2(kAtanh, kUnitVector, nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> digamma_api_perf_v2(ApiPerfRegisterV2(kDigamma, kUnitVector, nullptr, &perf_param_table_v2,
                                                               &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> erfc_api_perf_v2(ApiPerfRegisterV2(kErfc, kUnitVector, nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> erfcx_api_perf_v2(ApiPerfRegisterV2(kErfcx, kUnitVector, nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> acosh_api_perf_v2(ApiPerfRegisterV2(kAcosh, kUnitVector, nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> rshift_api_perf_v2(ApiPerfRegisterV2(kRShift, kRShift + "V2", nullptr, &perf_param_table_v2,
                                                              &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> where_api_perf_v2(ApiPerfRegisterV2(kWhere, kWhere + "V2", nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> select_api_perf_v2(ApiPerfRegisterV2(kSelect, kWhere + "V2", nullptr, &perf_param_table_v2,
                                                              &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> ge_api_perf_v2(ApiPerfRegisterV2(kGe, kGe + "V2", nullptr, &perf_param_table_v2,
                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> eq_api_perf_v2(ApiPerfRegisterV2(kEq, kEq + "V2", nullptr, &perf_param_table_v2,
                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> ne_api_perf_v2(ApiPerfRegisterV2(kNe, kNe + "V2", nullptr, &perf_param_table_v2,
                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> gt_api_perf_v2(ApiPerfRegisterV2(kGt, kGt + "V2", nullptr, &perf_param_table_v2,
                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> le_api_perf_v2(ApiPerfRegisterV2(kLe, kLe + "V2", nullptr, &perf_param_table_v2,
                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> lt_api_perf_v2(ApiPerfRegisterV2(kLt, kLt + "V2", nullptr, &perf_param_table_v2,
                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> ub2ub_api_perf_v2(ApiPerfRegisterV2(kUb2ub, kUb2ub, nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> load_api_perf_v2(ApiPerfRegisterV2(kLoad, kLoad + "V2", nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> store_api_perf_v2(ApiPerfRegisterV2(kStore, kStore + "V2", nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> nddma_api_perf_v2(ApiPerfRegisterV2(kNddma, kNddma + "V2", nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> all_api_perf_v2(ApiPerfRegisterV2(kAll, kAll + "V2", nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> reduce_all_api_perf_v2(ApiPerfRegisterV2(kReduceAll, kReduceAll + "V2", nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> any_api_perf_v2(ApiPerfRegisterV2(kAny, kAny + "V2", nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> reduce_any_api_perf_v2(ApiPerfRegisterV2(kReduceAny, kReduceAny + "V2", nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> reduce_mean_api_perf_v2(ApiPerfRegisterV2(kReduceMean, kReduceMean + "V2", nullptr,
                                                                   &perf_param_table_v2,
                                                                   &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> reduce_prod_api_perf_v2(ApiPerfRegisterV2(kReduceProd, kReduceProd + "V2", nullptr,
                                                                   &perf_param_table_v2,
                                                                   &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> reduce_sum_api_perf_v2(ApiPerfRegisterV2(kReduceSum, kReduceSum + "V2", nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> mean_api_perf_v2(ApiPerfRegisterV2(kMean, kMean + "V2", nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> prod_api_perf_v2(ApiPerfRegisterV2(kProd, kProd + "V2", nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> sum_api_perf_v2(ApiPerfRegisterV2(kSum, kSum + "V2", nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
// 不需要建模的ASCIR
ApiPerfRegister<ApiPerf> data_api_perf_v2(ApiPerfRegisterV2(kData, DefaultGetPerf, nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> scalar_api_perf_v2(ApiPerfRegisterV2(kScalar, DefaultGetPerf, nullptr, &perf_param_table_v2,
                                                              &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> scalar_data_api_perf_v2(ApiPerfRegisterV2(kScalarData, DefaultGetPerf, nullptr,
                                                                   &perf_param_table_v2,
                                                                   &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> index_expr_api_perf_v2(ApiPerfRegisterV2(kIndexExpr, DefaultGetPerf, nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> output_api_perf_v2(ApiPerfRegisterV2(kOutput, DefaultGetPerf, nullptr, &perf_param_table_v2,
                                                              &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> workspace_api_perf_v2(ApiPerfRegisterV2(kWorkspace, DefaultGetPerf, nullptr,
                                                                 &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
// 目前无建模的ASCIR
ApiPerfRegister<ApiPerf> pad_api_perf_v2(ApiPerfRegisterV2(kPad, kUnitVector, nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> round_api_perf_v2(ApiPerfRegisterV2(kRound, kUnitVector, nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> nop_api_perf_v2(ApiPerfRegisterV2(kNop, kUnitVector, nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> ln_api_perf_v2(ApiPerfRegisterV2(kLn, kLn + "V2", nullptr, &perf_param_table_v2,
                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> floor_to_int_api_perf_v2(ApiPerfRegisterV2(kFloorToInt, kFloorToInt + "V2", nullptr,
                                                                    &perf_param_table_v2,
                                                                    &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> fmod_api_perf_v2(ApiPerfRegisterV2(kFmod, kFmod + "V2", nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> hypot_api_perf_v2(ApiPerfRegisterV2(kHypot, kHypot + "V2", nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> lgamma_api_perf_v2(ApiPerfRegisterV2(kLgamma, kLgamma + "V2", nullptr, &perf_param_table_v2,
                                                              &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> log10_api_perf_v2(ApiPerfRegisterV2(kLog10, kLog10 + "V2", nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> logical_xor_api_perf_v2(ApiPerfRegisterV2(kLogicalXor, kLogicalXor + "V2", nullptr,
                                                                   &perf_param_table_v2,
                                                                   &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> log1p_api_perf_v2(ApiPerfRegisterV2(kLog1p, kLog1p + "V2", nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> expm1_api_perf_v2(ApiPerfRegisterV2(kExpm1, kExpm1 + "V2", nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> log2_api_perf_v2(ApiPerfRegisterV2(kLog2, kLog2 + "V2", nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> lShift_api_perf_v2(ApiPerfRegisterV2(kLShift, kLShift + "V2", nullptr, &perf_param_table_v2,
                                                              &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> mod_api_perf_v2(ApiPerfRegisterV2(kMod, kMod + "V2", nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> isnan_api_perf_v2(ApiPerfRegisterV2(kIsnan, kIsnan + "V2", nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> isfinite_api_perf_v2(ApiPerfRegisterV2(kIsFinite, kIsFinite + "V2", nullptr,
                                                                &perf_param_table_v2,
                                                                &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> isinf_api_perf_v2(ApiPerfRegisterV2(kIsInf, kIsInf + "V2", nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> maskedfill_api_perf_v2(ApiPerfRegisterV2(kMaskedFill, kUnitVector, nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> sigmoid_api_perf_v2(ApiPerfRegisterV2(kSigmoid, kSigmoid + "V2", nullptr, &perf_param_table_v2,
                                                               &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> true_div_api_perf_v2(ApiPerfRegisterV2(kTrueDiv, kDiv + "V2", nullptr, &perf_param_table_v2,
                                                                &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> pow_api_perf_v2(ApiPerfRegisterV2(kPow, kPow + "V2", nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> clip_by_value_api_perf_v2(ApiPerfRegisterV2(kClipByValue, kClipByValue + "V2", nullptr,
                                                                     &perf_param_table_v2,
                                                                     &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> concat_api_perf_v2(ApiPerfRegisterV2(kConcat, kUnitVector, nullptr, &perf_param_table_v2,
                                                              &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> leaky_relu_api_perf_v2(ApiPerfRegisterV2(kLeakyRelu, kLeakyRelu + "V2", nullptr,
                                                                  &perf_param_table_v2,
                                                                  &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> bitwise_and_api_perf_v2(ApiPerfRegisterV2(kBitwiseAnd, kBitwiseAnd + "V2", nullptr,
                                                                   &perf_param_table_v2,
                                                                   &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> transpose_api_perf_v2(ApiPerfRegisterV2(kTranspose, kTranspose + "V2", nullptr,
                                                                 &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> floor_div_api_perf_v2(ApiPerfRegisterV2(kFloorDiv, kFloorDiv + "V2", nullptr,
                                                                 &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> gelu_api_perf_v2(ApiPerfRegisterV2(kGelu, kGelu + "V2", nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> trunc_api_perf_v2(ApiPerfRegisterV2(kTrunc, kTrunc + "V2", nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> tan_api_perf_v2(ApiPerfRegisterV2(kTan, kTan + "V2", nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> sinh_api_perf_v2(ApiPerfRegisterV2(kSinh, kSinh + "V2", nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> trunc_div_api_perf_v2(ApiPerfRegisterV2(kTruncDiv, kTruncDiv + "V2", nullptr,
                                                                 &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> trunc_to_int_api_perf_v2(ApiPerfRegisterV2(kTruncToInt, kTruncToInt + "V2", nullptr,
                                                                    &perf_param_table_v2,
                                                                    &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> round_to_int_api_perf_v2(ApiPerfRegisterV2(kRoundToInt, kRoundToInt + "V2", nullptr,
                                                                    &perf_param_table_v2,
                                                                    &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> xor_api_perf_v2(ApiPerfRegisterV2(kXor, kXor + "V2", nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> remainder_api_perf_v2(ApiPerfRegisterV2(kRemainder, kRemainder + "V2", nullptr,
                                                                 &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> modified_bessel_i0_api_perf_v2(ApiPerfRegisterV2(kModifiedBesselI0, kUnitVector, nullptr,
                                                                          &perf_param_table_v2,
                                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> modified_bessel_i1_api_perf_v2(ApiPerfRegisterV2(kModifiedBesselI1, kUnitVector, nullptr,
                                                                          &perf_param_table_v2,
                                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> modified_bessel_k0_api_perf_v2(ApiPerfRegisterV2(kModifiedBesselK0, kUnitVector, nullptr,
                                                                          &perf_param_table_v2,
                                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> modified_bessel_k1_api_perf_v2(ApiPerfRegisterV2(kModifiedBesselK1, kUnitVector, nullptr,
                                                                          &perf_param_table_v2,
                                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> i0_api_perf_v2(ApiPerfRegisterV2(kI0, kUnitVector, nullptr, &perf_param_table_v2,
                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> i0e_api_perf_v2(ApiPerfRegisterV2(kI0e, kUnitVector, nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> i1e_api_perf_v2(ApiPerfRegisterV2(kI1e, kUnitVector, nullptr, &perf_param_table_v2,
                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> i1_api_perf_v2(ApiPerfRegisterV2(kI1, kUnitVector, nullptr, &perf_param_table_v2,
                                                          &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> identity_api_perf_v2(ApiPerfRegisterV2(kIdentity, kUnitVector, nullptr, &perf_param_table_v2,
                                                                &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> device_assert_async_api_perf_v2(ApiPerfRegisterV2(kDeviceAssertAsync, kUnitVector, nullptr,
                                                                           &perf_param_table_v2,
                                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> inline_asm_elementwise_api_perf_v2(ApiPerfRegisterV2(kInlineAsmElementwise, kUnitVector,
                                                                              nullptr, &perf_param_table_v2,
                                                                              &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> masked_api_perf_v2(ApiPerfRegisterV2(kMasked, kUnitVector, nullptr, &perf_param_table_v2,
                                                              &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> randint64_api_perf_v2(ApiPerfRegisterV2(kRandint64, kUnitVector, nullptr, &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> xor_sum_api_perf_v2(ApiPerfRegisterV2(kXorSum, kUnitVector, nullptr, &perf_param_table_v2,
                                                               &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> welford_api_perf_v2(ApiPerfRegisterV2(kWelford, kUnitVector, nullptr, &perf_param_table_v2,
                                                               &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> scan_api_perf_v2(ApiPerfRegisterV2(kScan, kUnitVector, nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> store_reduction_api_perf_v2(ApiPerfRegisterV2(kStoreReduction, kUnitVector, nullptr,
                                                                       &perf_param_table_v2,
                                                                       &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> bessel_j0_api_perf_v2(ApiPerfRegisterV2(kBesselJ0, kUnitVector, nullptr, &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> bessel_j1_api_perf_v2(ApiPerfRegisterV2(kBesselJ1, kUnitVector, nullptr, &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> bessel_y0_api_perf_v2(ApiPerfRegisterV2(kBesselY0, kUnitVector, nullptr, &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> bessel_y1_api_perf_v2(ApiPerfRegisterV2(kBesselY1, kUnitVector, nullptr, &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> scaled_modified_bessel_k0_api_perf_v2(ApiPerfRegisterV2(kScaledModifiedBesselK0, kUnitVector,
                                                                                 nullptr, &perf_param_table_v2,
                                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> scaled_modified_bessel_k1_api_perf_v2(ApiPerfRegisterV2(kScaledModifiedBesselK1, kUnitVector,
                                                                                 nullptr, &perf_param_table_v2,
                                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> spherical_bessel_j0_api_perf_v2(ApiPerfRegisterV2(kSphericalBesselJ0, kUnitVector, nullptr,
                                                                           &perf_param_table_v2,
                                                                           &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> igamma_api_perf_v2(ApiPerfRegisterV2(kIgamma, kUnitVector, nullptr, &perf_param_table_v2,
                                                              &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> igammac_api_perf_v2(ApiPerfRegisterV2(kIgammac, kUnitVector, nullptr, &perf_param_table_v2,
                                                               &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> zeta_api_perf_v2(ApiPerfRegisterV2(kZeta, kUnitVector, nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> ndtr_api_perf_v2(ApiPerfRegisterV2(kNdtr, kUnitVector, nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> ndtri_api_perf_v2(ApiPerfRegisterV2(kNdtri, kUnitVector, nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> logndtr_api_perf_v2(ApiPerfRegisterV2(kLogNdtr, kUnitVector, nullptr, &perf_param_table_v2,
                                                               &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> nextafter_api_perf_v2(ApiPerfRegisterV2(kNextAfter, kUnitVector, nullptr, &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> bucketize_api_perf_v2(ApiPerfRegisterV2(kBucketize, kUnitVector, nullptr, &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> polygamma_api_perf_v2(ApiPerfRegisterV2(kPolyGamma, kUnitVector, nullptr, &perf_param_table_v2,
                                                                 &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> signbit_api_perf_v2(ApiPerfRegisterV2(kSignBit, kUnitVector, nullptr, &perf_param_table_v2,
                                                               &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> frexp_api_perf_v2(ApiPerfRegisterV2(kFrexp, kUnitVector, nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> shifted_chebyshev_polynomial_t_api_perf_v2(ApiPerfRegisterV2(
    kShiftedChebyshevPolynomialT, kUnitVector, nullptr, &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> shifted_chebyshev_polynomial_u_api_perf_v2(ApiPerfRegisterV2(
    kShiftedChebyshevPolynomialU, kUnitVector, nullptr, &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> shifted_chebyshev_polynomial_v_api_perf_v2(ApiPerfRegisterV2(
    kShiftedChebyshevPolynomialV, kUnitVector, nullptr, &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> shifted_chebyshev_polynomial_w_api_perf_v2(ApiPerfRegisterV2(
    kShiftedChebyshevPolynomialW, kUnitVector, nullptr, &perf_param_table_v2, &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> chebyshev_polynomial_t_api_perf_v2(ApiPerfRegisterV2(kChebyshevPolynomialT, kUnitVector,
                                                                              nullptr, &perf_param_table_v2,
                                                                              &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> chebyshev_polynomial_u_api_perf_v2(ApiPerfRegisterV2(kChebyshevPolynomialU, kUnitVector,
                                                                              nullptr, &perf_param_table_v2,
                                                                              &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> chebyshev_polynomial_v_api_perf_v2(ApiPerfRegisterV2(kChebyshevPolynomialV, kUnitVector,
                                                                              nullptr, &perf_param_table_v2,
                                                                              &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> chebyshev_polynomial_w_api_perf_v2(ApiPerfRegisterV2(kChebyshevPolynomialW, kUnitVector,
                                                                              nullptr, &perf_param_table_v2,
                                                                              &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> hermite_polynomial_h_api_perf_v2(ApiPerfRegisterV2(kHermitePolynomialH, kUnitVector, nullptr,
                                                                            &perf_param_table_v2,
                                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> hermite_polynomial_he_api_perf_v2(ApiPerfRegisterV2(kHermitePolynomialHe, kUnitVector, nullptr,
                                                                             &perf_param_table_v2,
                                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> laguerre_polynomial_l_api_perf_v2(ApiPerfRegisterV2(kLaguerrePolynomialL, kUnitVector, nullptr,
                                                                             &perf_param_table_v2,
                                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> legendre_polynomial_p_api_perf_v2(ApiPerfRegisterV2(kLegendrePolynomialP, kUnitVector, nullptr,
                                                                             &perf_param_table_v2,
                                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> airy_ai_api_perf_v2(ApiPerfRegisterV2(kAiryAi, kUnitVector, nullptr, &perf_param_table_v2,
                                                               &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> erfinv_api_perf_v2(ApiPerfRegisterV2(kErfinv, kUnitVector, nullptr, &perf_param_table_v2,
                                                              &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> rand_api_perf_v2(ApiPerfRegisterV2(kRand, kUnitVector, nullptr, &perf_param_table_v2,
                                                            &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> randn_api_perf_v2(ApiPerfRegisterV2(kRandn, kUnitVector, nullptr, &perf_param_table_v2,
                                                             &tiling_schedule_config_table_v2));
ApiPerfRegister<ApiPerf> square_api_perf_v2(ApiPerfRegisterV2(kSquare, kSquare + "V2", nullptr, &perf_param_table_v2,
                                                              &tiling_schedule_config_table_v2));

ApiPerfRegister<ApiPerf> vector_func_api_perf(kVectorFunc, DefaultGetPerf, nullptr, &perf_param_table_v2,
                                              &tiling_schedule_config_table_v2);
}  // namespace
}  // namespace att
