/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascir_api_perf_v1.h"
#include <string>
#include <algorithm>
#include <numeric>
#include "common/checker.h"
#include "base/att_const_values.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "api_perf_register/ascendc_api_perf.h"
#include "api_perf_register/api_perf_factory.h"
#include "perf_param_v1.h"

using namespace af::sym;
namespace att {
namespace {
constexpr int32_t kMaxDmaLen = 2;
// bandwidth
inline const std::map<std::string, Expr> kBandwidthMap = {
    {kMoveGmToL1, CreateExpr(32U)},   {kMoveL2ToL1, CreateExpr(110U)}, {kMoveL1ToL0a, CreateExpr(512U)},
    {kMoveL1ToL0b, CreateExpr(256U)}, {kMoveL0cToL2, CreateExpr(86U)}, {kMoveL0cToGm, CreateExpr(32U)},
    {kMoveGmToUb, CreateExpr(32U)},   {kMoveUbToGm, CreateExpr(32U)}};

af::Status GetBlkEleNum(const std::string &data_type, Expr &one_block_ele_num) {
  auto it = kBlkEleMap.find(data_type);
  GE_ASSERT_TRUE(it != kBlkEleMap.end(), "Data type [%s] unsatisfied.", data_type.c_str());
  one_block_ele_num = it->second;
  return af::SUCCESS;
}

af::Status CopyPerf(const std::string &op_type, uint32_t data_type_size, const std::vector<Expr> &dims,
                    const Expr &dim_product, Expr &res) {
  GE_ASSERT_TRUE(!dims.empty());
  Expr data_size = Mul(dim_product, CreateExpr(data_type_size));
  auto it = kBandwidthMap.find(op_type);
  GE_ASSERT_TRUE(it != kBandwidthMap.end(), "Op_type[%s] not found in kBandwidthMap", op_type.c_str());
  auto cycles = Div(data_size, it->second);
  Expr dens = Sub(Mul(dims.back(), kInitA), kInitB);
  if (dens == 0) {
    dens = CreateExpr(1);
  }
  auto weight = Div(kSymPowerofEight, dens);
  res = Mul(cycles, weight);
  return af::SUCCESS;
}
}  // namespace

namespace ascir_v1 {
af::Status MoveGmtoL1Api([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                         [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                         [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  const auto kNd2NzStartCost = CreateExpr(210U);
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());
  auto data_type_size = input_shapes[0].data_type_size;
  auto dims = output_shapes[0].dims;
  Expr dim_product = accumulate(dims.begin(), dims.end(), CreateExpr(1), [](Expr a, Expr b) { return Mul(a, b); });
  GE_ASSERT_TRUE(!dims.empty());
  Expr data_size = Mul(dim_product, CreateExpr(data_type_size));
  auto it = kBandwidthMap.find(kMoveGmToL1);
  GE_ASSERT_TRUE(it != kBandwidthMap.end(), "kMoveGmToL1 not found in kBandwidthMap");
  auto cycles = Div(data_size, it->second);
  cycles = Add(cycles, kNd2NzStartCost);
  Expr dens = Sub(Mul(dims.back(), kInitA), kInitB);
  if (dens == 0) {
    dens = CreateExpr(1);
  }
  auto weight = Div(kSymPowerofEight, dens);
  perf_res.pipe_res[PipeType::AICORE_MTE2] = Mul(cycles, weight);
  return af::SUCCESS;
}

af::Status MoveL2ToL1Api([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                         [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                         [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());
  auto dims = output_shapes[0].dims;
  Expr dim_product = accumulate(dims.begin(), dims.end(), CreateExpr(1), [](Expr a, Expr b) { return Mul(a, b); });
  Expr res;
  GE_ASSERT_SUCCESS(CopyPerf(kMoveL2ToL1, input_shapes[0].data_type_size, output_shapes[0].dims, dim_product, res));
  perf_res.pipe_res[PipeType::AICORE_MTE2] = res;
  return af::SUCCESS;
}

af::Status MovefromL1Perf(const std::string &op_type, uint32_t data_type_size, const std::vector<Expr> &dims,
                          Expr &res) {
  const auto kMte1StartCost = CreateExpr(26U);
  GE_ASSERT_TRUE(!dims.empty());
  Expr dim_product = accumulate(dims.begin(), dims.end(), CreateExpr(1), [](Expr a, Expr b) { return Mul(a, b); });
  Expr data_size = Mul(dim_product, CreateExpr(data_type_size));
  auto it = kBandwidthMap.find(op_type);
  GE_ASSERT_TRUE(it != kBandwidthMap.end(), "Op_type[%s] not found in kBandwidthMap", op_type.c_str());
  auto cycles = Div(data_size, it->second);
  cycles = Add(cycles, kMte1StartCost);
  Expr dens = Sub(Mul(dims.back(), kInitA), kInitB);
  if (dens == 0) {
    dens = CreateExpr(1);
  }
  auto weight = Div(kSymPowerofEight, dens);
  res = Mul(cycles, weight);
  return af::SUCCESS;
}

af::Status MoveL1toL0aApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                          [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                          [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());
  Expr res;
  GE_ASSERT_SUCCESS(MovefromL1Perf(kMoveL1ToL0a, input_shapes[0].data_type_size, output_shapes[0].dims, res));
  perf_res.pipe_res[PipeType::AICORE_MTE1] = res;
  return af::SUCCESS;
}

af::Status MoveL1toL0bApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                          [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                          [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());
  Expr res;
  GE_ASSERT_SUCCESS(MovefromL1Perf(kMoveL1ToL0b, input_shapes[0].data_type_size, output_shapes[0].dims, res));
  perf_res.pipe_res[PipeType::AICORE_MTE1] = res;
  return af::SUCCESS;
}

af::Status MoveL0cToL2Api([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                          [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                          [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());
  auto dims = output_shapes[0].dims;
  Expr dim_product = accumulate(dims.begin(), dims.end(), CreateExpr(1), [](Expr a, Expr b) { return Mul(a, b); });
  Expr res;
  GE_ASSERT_SUCCESS(CopyPerf(kMoveL0cToL2, input_shapes[0].data_type_size, dims, dim_product, res));
  perf_res.pipe_res[PipeType::AIC_FIXPIPE] = res;
  return af::SUCCESS;
}

af::Status MoveL0cToGmApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                          [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                          [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());
  auto dims = output_shapes[0].dims;
  auto input_dims = input_shapes[0].dims;
  Expr dim_product =
      accumulate(input_dims.begin(), input_dims.end(), CreateExpr(1), [](Expr a, Expr b) { return Mul(a, b); });
  Expr res;
  GE_ASSERT_SUCCESS(CopyPerf(kMoveL0cToGm, input_shapes[0].data_type_size, dims, dim_product, res));
  perf_res.pipe_res[PipeType::AIC_FIXPIPE] = res;
  return af::SUCCESS;
}

af::Status SoftmaxFlashV2([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                          [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                          [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  (void)output_shapes;
  Expr t = CreateExpr(0.89f);
  Expr a = CreateExpr(6.14f);
  Expr b = CreateExpr(-5.60f);
  Expr c = CreateExpr(0.06f);
  Expr h = CreateExpr(44.39f);
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());
  auto dims = input_shapes[0].dims;
  GE_ASSERT_TRUE(!dims.empty());
  Expr dim_product = accumulate(dims.begin(), dims.end(), CreateExpr(1), [](Expr a, Expr b) { return Mul(a, b); });
  auto cycles = Div(dim_product, t);
  auto weight = Add(Div(a, Add(dims.back(), b)), c);
  cycles = Mul(cycles, weight);
  perf_res.pipe_res[PipeType::AIV_VEC] = Add(cycles, h);
  return af::SUCCESS;
}

/*
Loadapi(DataCopy from GM to UB)的性能公式：
  1. 单次MTE2 = S(数据量Byte)/T + h(指令头开销)，针对非连续搬运场景会增加stride建模值(k*(stride%(256)*block_count))
  2. 总MTE2 = 单次MTE2 * 调用次数 + H(pipe启动头开销)
  3. H = 15.8854 * blockdim + 882.0878 (datasize < 25000B)
     H = 32.7221 * blockdim + 1575.0306 (datasize >= 25000B)
  4. h = 27.01
  5. T = 7.9052 + 7.3100/blockdim (datasize < 25000B)
     T = 9.9074 + 15.8960/blockdim (datasize >= 25000B)
     (单核的峰值带宽，核数越多，带宽抢占越严重，直到收敛到稳定值约为8.47)
  6. mte2 = S/T + h
  7. overall_mte2 = mte2 * mte2_count + H
  8. 外抛for循环：最外侧两个维度丢到循环次数里面去
*/

// DMA节点性能计算辅助函数（消除LoadApi和StoreApi重复代码）
af::Status InitDmaNodeAndGetPerf(const af::AscNodePtr &node_ptr, const std::string &node_name,
                                 TensorShapeInfo &merged_shapes, PerfOutputInfo &perf_res) {
  GE_ASSERT_SUCCESS(MergeTensorContinuousDims(node_ptr, GetNodeOutTensorName(node_ptr, 0), merged_shapes));

  NodeDetail dma_info;
  dma_info.name = node_ptr != nullptr ? node_ptr->GetName() : node_name;
  dma_info.optype = node_ptr->GetType();
  dma_info.input_dtype = {merged_shapes.data_type};
  dma_info.output_dtype = {merged_shapes.data_type};

  GE_ASSERT_SUCCESS(SetDims(merged_shapes, dma_info));
  GE_ASSERT_SUCCESS(GetDmaPerf(merged_shapes, dma_info, perf_res, kMaxDmaLen, true));
  return af::SUCCESS;
}

af::Status LoadApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  auto const &node_ptr = node.node_ptr;
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());

  auto merged_shapes = output_shapes[0];
  return InitDmaNodeAndGetPerf(node_ptr, "LoadNode", merged_shapes, perf_res);
}

/*
StoreApi(DataCopy from UB to GM)的性能公式：
  1. 单次MTE3 = S(数据量Byte)/T + h(指令头开销)，针对非连续搬运场景会增加stride建模值(k*(stride%(256)*block_count))
  2. 总MTE3 = 单次MTE3 * 调用次数 + H(pipe启动头开销)
  3. H = 497.36
  4. h = 12.09
  5. T = 9.96 + 3.79/blockdim(单核的峰值带宽，核数越多，带宽抢占越严重，直到收敛到稳定值约为9.96)
  6. mte3 = S/T + h
  7. overall_mte3 = mte3 * mte3_count + H
  8. 外抛for循环：最外侧两个维度丢到循环次数里面去
*/
af::Status StoreApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                    [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                    [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  auto const &node_ptr = node.node_ptr;
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());

  auto merged_shapes = output_shapes[0];
  return InitDmaNodeAndGetPerf(node_ptr, "StoreNode", merged_shapes, perf_res);
}

af::Status AbsApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::AbsPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status AddApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::AddPerf(node_info, perf_res), "Add perf failed, node name: %s, type: %s",
                    node_info.name.c_str(), node_info.optype.c_str());
  return af::SUCCESS;
}

af::Status CastApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  auto const &node_ptr = node.node_ptr;
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());
  NodeDetail node_info;
  Expr outer_repeat;
  vector<Expr> used_dims;
  auto merged_output_shapes = output_shapes[0];
  GE_ASSERT_SUCCESS(MergeTensorContinuousDims(node_ptr, GetNodeOutTensorName(node_ptr, 0), merged_output_shapes));
  GE_ASSERT_SUCCESS(GetOuterParams(merged_output_shapes.dims, outer_repeat, used_dims));
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(SetDims(used_dims, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::CastPerf(node_info, perf_res), "Cast perf failed, node name: %s, type: %s",
                    node_info.name.c_str(), node_info.optype.c_str());
  perf_res.pipe_res[PipeType::AIV_VEC] = outer_repeat * GetPipeCost(perf_res, PipeType::AIV_VEC);
  return af::SUCCESS;
}

af::Status CopyApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::CopyPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status DivApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::DivPerf(node_info, perf_res), "Div perf failed, node name: %s, type: %s",
                    node_info.name.c_str(), node_info.optype.c_str());
  return af::SUCCESS;
}

af::Status ErfApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::ErfPerf(node_info, perf_res), "Erf perf failed, node name: %s, type: %s",
                    node_info.name.c_str(), node_info.optype.c_str());
  return af::SUCCESS;
}

af::Status ExpApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::ExpPerf(node_info, perf_res), "Exp perf failed, node name: %s, type: %s",
                    node_info.name.c_str(), node_info.optype.c_str());
  return af::SUCCESS;
}

af::Status GatherPerf(const std::vector<TensorShapeInfo> &input_shapes,
                      const std::vector<TensorShapeInfo> &output_shapes, [[maybe_unused]] const NodeInfo &node,
                      PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::GatherPerf(node_info, perf_res), "Gather perf failed, node name: %s, type: %s",
                    node_info.name.c_str(), node_info.optype.c_str());
  return af::SUCCESS;
}

af::Status MaxApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::MaxPerf(node_info, perf_res), "Max perf failed, node name: %s, type: %s",
                    node_info.name.c_str(), node_info.optype.c_str());
  return af::SUCCESS;
}

af::Status MinApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::MinPerf(node_info, perf_res), "Min perf failed, node name: %s, type: %s",
                    node_info.name.c_str(), node_info.optype.c_str());
  return af::SUCCESS;
}

af::Status ReciprocalApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                         [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                         [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::ReciprocalPerf(node_info, perf_res), "Reciprocal perf failed, node name: %s, type: %s",
                    node_info.name.c_str(), node_info.optype.c_str());
  return af::SUCCESS;
}

af::Status ReluApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::ReluPerf(node_info, perf_res), "Relu perf failed, node name: %s, type: %s",
                    node_info.name.c_str(), node_info.optype.c_str());
  return af::SUCCESS;
}

af::Status RsqrtApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                    [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                    [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::RsqrtPerf(node_info, perf_res), "Rsqrt perf failed, node name: %s, type: %s",
                    node_info.name.c_str(), node_info.optype.c_str());
  return af::SUCCESS;
}

af::Status SignApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::SignPerf(node_info, perf_res), "Sign perf failed, node name: %s, type: %s",
                    node_info.name.c_str(), node_info.optype.c_str());
  return af::SUCCESS;
}

af::Status SqrtApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::SqrtPerf(node_info, perf_res), "Sqrt perf failed, node name: %s, type: %s",
                    node_info.name.c_str(), node_info.optype.c_str());
  return af::SUCCESS;
}

af::Status SubApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                  [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                  [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::SubPerf(node_info, perf_res), "Sub perf failed, node name: %s, type: %s",
                    node_info.name.c_str(), node_info.optype.c_str());
  return af::SUCCESS;
}

af::Status TanhApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                   [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                   [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::TanhPerf(node_info, perf_res), "Tanh perf failed, node name: %s, type: %s",
                    node_info.name.c_str(), node_info.optype.c_str());
  return af::SUCCESS;
}

af::Status BroadcastOuter(const std::string &data_type, const std::vector<Expr> &input_dims,
                          const std::vector<Expr> &output_dims, PerfOutputInfo &perf_res) {
  (void)input_dims;
  auto it = kRptEleMap.find(data_type);
  GE_ASSERT_TRUE(it != kRptEleMap.end(), "Data type [%s] unsatisfied.", data_type.c_str());
  Expr ele_per_rpt = it->second;
  Expr dst_m = output_dims[0];
  Expr max_rpt_cnt = dst_m / kMaxRepeatTime;
  Expr tail_rpt_times = dst_m - max_rpt_cnt * kMaxRepeatTime;
  Expr tail_pad = af::sym::Ceiling(tail_rpt_times / (tail_rpt_times + CreateExpr(1)));
  PerfOutputInfo rpt_perf;
  PerfOutputInfo tail_perf;
  NodeDetail copy_node1 = GenNodeDetail(data_type, data_type, {kMaxRepeatTime * ele_per_rpt});
  NodeDetail copy_node2 = GenNodeDetail(data_type, data_type, {tail_rpt_times * ele_per_rpt});
  GE_ASSERT_SUCCESS(ascendcperf::CopyPerf(copy_node1, rpt_perf), "Gen CopyApi perf failed, node name: %s, type: %s",
                    copy_node1.name.c_str(), copy_node1.optype.c_str());
  GE_ASSERT_SUCCESS(ascendcperf::CopyPerf(copy_node2, tail_perf), "Gen CopyApi perf failed, node name: %s, type: %s",
                    copy_node2.name.c_str(), copy_node2.optype.c_str());
  perf_res.pipe_res[PipeType::AIV_VEC] =
      max_rpt_cnt * GetPipeCost(rpt_perf, PipeType::AIV_VEC) + tail_pad * GetPipeCost(tail_perf, PipeType::AIV_VEC);
  return af::SUCCESS;
}

af::Status BrcbToOneBlockPerf(const std::string &data_type, Expr first_dim, Expr last_dim, PerfOutputInfo &perf_res) {
  (void)last_dim;
  auto iter1 = kBlkEleMap.find(data_type);
  auto iter2 = kBrcbRepeatMap.find(data_type);
  GE_ASSERT_TRUE((iter1 != kBlkEleMap.end() && iter2 != kBrcbRepeatMap.end()), "Data type [%s] unsatisfied.",
                 data_type.c_str());
  PerfOutputInfo brcb_repeat_perf;
  PerfOutputInfo brcb_tail_perf;
  const auto kBrcbSize = CreateExpr(8U);
  Expr brcb_rpt_times = af::sym::Ceiling(first_dim / kBrcbSize) * kBrcbSize;
  Expr ele_per_order = iter1->second;
  Expr brcb_max_rpt_times = iter2->second;
  Expr loop_cnt = brcb_rpt_times / brcb_max_rpt_times;
  Expr tail_rpt = brcb_rpt_times - loop_cnt * brcb_max_rpt_times;
  Expr brcb_pad = af::sym::Ceiling(tail_rpt / (tail_rpt + CreateExpr(1)));
  NodeDetail brcb_node1 = GenNodeDetail(data_type, data_type, {brcb_max_rpt_times * ele_per_order});
  NodeDetail brcb_node2 = GenNodeDetail(data_type, data_type, {tail_rpt * ele_per_order});
  GE_ASSERT_SUCCESS(ascendcperf::BrcbPerf(brcb_node1, brcb_repeat_perf),
                    "Gen BrcbApi perf failed, node name: %s, type: %s", brcb_node1.name.c_str(),
                    brcb_node1.optype.c_str());
  GE_ASSERT_SUCCESS(ascendcperf::BrcbPerf(brcb_node2, brcb_tail_perf),
                    "Gen BrcbApi perf failed, node name: %s, type: %s", brcb_node2.name.c_str(),
                    brcb_node2.optype.c_str());
  perf_res.pipe_res[PipeType::AIV_VEC] = loop_cnt * GetPipeCost(brcb_repeat_perf, PipeType::AIV_VEC) +
                                         brcb_pad * GetPipeCost(brcb_tail_perf, PipeType::AIV_VEC);
  return af::SUCCESS;
}

af::Status TwoDimBroadCastLastDimAlignPerf(const std::string &data_type, Expr first_dim, Expr last_dim,
                                           PerfOutputInfo &perf_res) {
  auto it = kRptEleMap.find(data_type);
  GE_ASSERT_TRUE(it != kRptEleMap.end(), "Data type [%s] unsatisfied.", data_type.c_str());
  PerfOutputInfo brcb_perf;
  PerfOutputInfo copy_repeat_perf;
  PerfOutputInfo copy_tail_perf;
  Expr ele_per_rpt = it->second;
  Expr copy_cnt = first_dim / kMaxRepeatTime;
  Expr copy_tail = first_dim - copy_cnt * kMaxRepeatTime;
  Expr copy_pad = af::sym::Ceiling(copy_tail / (copy_tail + CreateExpr(1)));
  GE_ASSERT_SUCCESS(BrcbToOneBlockPerf(data_type, first_dim, last_dim, brcb_perf), "Gen BrcbToOneBlock perf Failed.");
  NodeDetail copy_node1 = GenNodeDetail(data_type, data_type, {kMaxRepeatTime * ele_per_rpt});
  NodeDetail copy_node2 = GenNodeDetail(data_type, data_type, {copy_tail * ele_per_rpt});
  GE_ASSERT_SUCCESS(ascendcperf::CopyPerf(copy_node1, copy_repeat_perf),
                    "Gen Copy perf failed, node name: %s, type: %s", copy_node1.name.c_str(),
                    copy_node1.optype.c_str());
  GE_ASSERT_SUCCESS(ascendcperf::CopyPerf(copy_node2, copy_tail_perf), "Gen Copy perf failed, node name: %s, type: %s",
                    copy_node2.name.c_str(), copy_node2.optype.c_str());
  perf_res.pipe_res[PipeType::AIV_VEC] = GetPipeCost(brcb_perf, PipeType::AIV_VEC) +
                                         copy_cnt * GetPipeCost(copy_repeat_perf, PipeType::AIV_VEC) +
                                         GetPipeCost(copy_tail_perf, PipeType::AIV_VEC) * copy_pad;
  return af::SUCCESS;
}

af::Status TwoDimBroadcastLastDim(const std::string &data_type, const std::vector<Expr> &input_dims,
                                  const std::vector<Expr> &output_dims, PerfOutputInfo &perf_res, bool with_stride) {
  (void)input_dims;
  Expr one_block_ele_num;
  GE_ASSERT_SUCCESS(GetBlkEleNum(data_type, one_block_ele_num), "Data type [%s] unsatisfied.", data_type.c_str());
  PerfOutputInfo repeat_perf;
  PerfOutputInfo tail_perf;
  const auto &first_dim = output_dims[0];
  const auto &last_dim = output_dims[1];
  Expr stride = with_stride ? one_block_ele_num : CreateExpr(0);
  Expr min_tmp_buffer_size = one_block_ele_num * one_block_ele_num + stride;
  Expr one_repeat_size = kTempBufSize / min_tmp_buffer_size * one_block_ele_num;
  Expr range_m = first_dim / one_repeat_size;
  Expr tail_m = first_dim - one_repeat_size * range_m;
  Expr pad_m = af::sym::Ceiling(tail_m / (tail_m + CreateExpr(1)));
  GE_ASSERT_SUCCESS(TwoDimBroadCastLastDimAlignPerf(data_type, one_repeat_size, last_dim, repeat_perf),
                    "Gen TwoDimBroadCastLastDimAlignPerf perf Failed.");
  GE_ASSERT_SUCCESS(TwoDimBroadCastLastDimAlignPerf(data_type, tail_m, last_dim, tail_perf),
                    "Gen TwoDimBroadCastLastDimAlignPerf perf Failed.");
  perf_res.pipe_res[PipeType::AIV_VEC] =
      range_m * GetPipeCost(repeat_perf, PipeType::AIV_VEC) + pad_m * GetPipeCost(tail_perf, PipeType::AIV_VEC);
  return af::SUCCESS;
}

af::Status BroadcastMiddle(const std::string &data_type, const std::vector<Expr> &input_dims,
                           const std::vector<Expr> &output_dims, PerfOutputInfo &perf_res) {
  (void)input_dims;
  auto iter1 = kBlkEleMap.find(data_type);
  auto iter2 = kRptEleMap.find(data_type);
  GE_ASSERT_TRUE((iter1 != kBlkEleMap.end() && iter2 != kRptEleMap.end()), "Data type [%s] unsatisfied.",
                 data_type.c_str());
  PerfOutputInfo duplicate_perf;
  PerfOutputInfo or_perf;
  Expr dst_m = output_dims[0];
  Expr dst_k = output_dims[1];
  Expr dst_z = output_dims[2];
  Expr one_blk_num = iter1->second;
  Expr one_rpt_num = iter2->second;
  Expr outer_dim_max_loop = kMaxRepeatTime * one_blk_num;
  Expr loop_cnt = dst_k / kSymEight;
  Expr rpt = dst_z / one_blk_num;
  Expr loop_tail = dst_k - loop_cnt * kSymEight;
  Expr loop_pad = af::sym::Ceiling(loop_tail / (loop_tail + CreateExpr(1)));
  NodeDetail duplicate_node = GenNodeDetail("uint16", "uint16", {CreateExpr(16)});
  NodeDetail or_node = GenNodeDetail("uint16", "uint16", {rpt * one_rpt_num});
  GE_ASSERT_SUCCESS(ascendcperf::DuplicatePerf(duplicate_node, duplicate_perf),
                    "Gen Duplicate perf failed, node name: %s, type: %s", duplicate_node.name.c_str(),
                    duplicate_node.optype.c_str());
  GE_ASSERT_SUCCESS(ascendcperf::OrPerf(or_node, or_perf), "Gen Or perf failed, node name: %s, type: %s",
                    or_node.name.c_str(), or_node.optype.c_str());
  perf_res.pipe_res[PipeType::AIV_VEC] = dst_m * (GetPipeCost(duplicate_perf, PipeType::AIV_VEC) +
                                                  (loop_cnt + loop_pad) * GetPipeCost(or_perf, PipeType::AIV_VEC));
  return af::SUCCESS;
}

af::Status BroadcastBoth(const std::string &data_type, const std::vector<Expr> &input_dims,
                         const std::vector<Expr> &output_dims, PerfOutputInfo &perf_res) {
  (void)input_dims;
  PerfOutputInfo duplicate_perf;
  NodeDetail duplicate_node = GenNodeDetail(data_type, data_type, output_dims);
  GE_ASSERT_SUCCESS(ascendcperf::DuplicatePerf(duplicate_node, duplicate_perf), "Gen Duplicate perf Failed.");
  perf_res.pipe_res[PipeType::AIV_VEC] = GetPipeCost(duplicate_perf, PipeType::AIV_VEC);
  return af::SUCCESS;
}

af::Status BroadcastTwoDim(const std::string &data_type, const std::vector<Expr> &input_dims,
                           const std::vector<Expr> &output_dims, PerfOutputInfo &perf_res) {
  if (input_dims[0] == 1U && input_dims[1] == output_dims[1]) {
    GE_ASSERT_SUCCESS(BroadcastOuter(data_type, input_dims, output_dims, perf_res), "Gen BroadcastOuter perf Failed.");
  } else if (input_dims[0] == 1U && input_dims[1] == 1U) {
    GE_ASSERT_SUCCESS(BroadcastBoth(data_type, input_dims, output_dims, perf_res), "Gen BroadcastBoth perf Failed.");
  } else if (input_dims[0] == output_dims[0]) {
    if (input_dims[1] == 1U) {
      GE_ASSERT_SUCCESS(TwoDimBroadcastLastDim(data_type, input_dims, output_dims, perf_res, false),
                        "Gen BroadcastInner perf Failed.");
    } else {
      GE_ASSERT_SUCCESS(TwoDimBroadcastLastDim(data_type, input_dims, output_dims, perf_res, true),
                        "Gen BroadcastWithStride perf Failed.");
    }
  }
  return af::SUCCESS;
}

af::Status BroadcastThreeDim(const std::string &data_type, const std::vector<Expr> &input_dims,
                             const std::vector<Expr> &output_dims, PerfOutputInfo &perf_res) {
  if (input_dims[0] == 1U && input_dims[1] == output_dims[1]) {
    GE_ASSERT_SUCCESS(BroadcastOuter(data_type, input_dims, output_dims, perf_res), "Gen BroadcastOuter perf Failed.");
  } else if (input_dims[0] == output_dims[0] && input_dims[1] == 1U && input_dims[2] == output_dims[2]) {
    GE_ASSERT_SUCCESS(BroadcastMiddle(data_type, input_dims, output_dims, perf_res),
                      "Gen BroadcastMiddle perf Failed.");
  }
  return af::SUCCESS;
}

af::Status BroadcastFourDim(const std::string &data_type, const std::vector<Expr> &input_dims,
                            const std::vector<Expr> &output_dims, PerfOutputInfo &perf_res) {
  std::vector<Expr> cur_input_dims;
  std::vector<Expr> cur_output_dims;
  PerfOutputInfo broadcast_perf1;
  PerfOutputInfo broadcast_perf2;
  if (input_dims[kNumZero] == 1U && input_dims[kNumTwo] == 1U && input_dims[kNumOne] == output_dims[kNumOne] &&
      input_dims[kNumThree] == output_dims[kNumThree]) {
    cur_input_dims = {input_dims[kNumOne], input_dims[kNumTwo], input_dims[kNumThree]};
    cur_output_dims = {output_dims[kNumOne], output_dims[kNumTwo], output_dims[kNumThree]};
    GE_ASSERT_SUCCESS(BroadcastThreeDim(data_type, cur_input_dims, cur_output_dims, broadcast_perf1),
                      "Gen Broadcast perf Failed.");
    cur_input_dims = {input_dims[kNumZero], output_dims[kNumOne] * output_dims[kNumTwo] * output_dims[kNumThree]};
    cur_output_dims = {output_dims[kNumZero], output_dims[kNumOne] * output_dims[kNumTwo] * output_dims[kNumThree]};
    GE_ASSERT_SUCCESS(BroadcastTwoDim(data_type, cur_input_dims, cur_output_dims, broadcast_perf2),
                      "Gen Broadcast perf Failed.");
    perf_res.pipe_res[PipeType::AIV_VEC] =
        GetPipeCost(broadcast_perf1, PipeType::AIV_VEC) + GetPipeCost(broadcast_perf2, PipeType::AIV_VEC);
  } else if (input_dims[kNumZero] == output_dims[kNumZero] && input_dims[kNumOne] == 1U &&
             input_dims[kNumTwo] == output_dims[kNumTwo] && input_dims[kNumThree] == 1U) {
    cur_input_dims = {input_dims[kNumZero] * input_dims[kNumOne] * input_dims[kNumTwo], input_dims[kNumThree]};
    cur_output_dims = {input_dims[kNumZero] * input_dims[kNumOne] * input_dims[kNumTwo], output_dims[kNumThree]};
    GE_ASSERT_SUCCESS(BroadcastTwoDim(data_type, cur_input_dims, cur_output_dims, broadcast_perf1),
                      "Gen Broadcast perf Failed.");
    cur_input_dims = {input_dims[kNumZero], input_dims[kNumOne], input_dims[kNumTwo] * output_dims[kNumThree]};
    cur_output_dims = {output_dims[kNumZero], output_dims[kNumOne], output_dims[kNumTwo] * output_dims[kNumThree]};
    GE_ASSERT_SUCCESS(BroadcastThreeDim(data_type, cur_input_dims, cur_output_dims, broadcast_perf2),
                      "Gen Broadcast perf Failed.");
    perf_res.pipe_res[PipeType::AIV_VEC] =
        GetPipeCost(broadcast_perf1, PipeType::AIV_VEC) + GetPipeCost(broadcast_perf2, PipeType::AIV_VEC);
  }
  return af::SUCCESS;
}

af::Status ExpandBroadCastDims(std::vector<Expr> &input_dims, const std::vector<Expr> &output_dims) {
  size_t offset = 0;
  Expr insert_dim;
  std::string expand_log;
  std::vector<bool> valid_dims(output_dims.size(), false);
  for (size_t i = 0; i < input_dims.size(); ++i) {
    if (input_dims[i] != output_dims[i + offset]) {
      if (offset == 0 && input_dims[i] == output_dims[i + 1]) {
        valid_dims[i] = true;
        ++offset;
      } else {
        GELOGW("Two Dim broadcast.");
      }
    }
  }
  input_dims.clear();
  for (size_t i = 0; i < output_dims.size(); ++i) {
    if (i > 0) {
      expand_log += ", ";
    }
    insert_dim = valid_dims[i] ? af::sym::kSymbolOne : output_dims[i];
    input_dims.emplace_back(insert_dim);
    expand_log += Str(insert_dim);
  }
  GELOGD("Expand input shape: {%s}", expand_log.c_str());
  return af::SUCCESS;
}

/*
BroadCast合轴处理：
(1, A, B) -> (1, A * B)
(A, 1, B, C) -> (A, 1, B * C)
*/
af::Status MergeAxis(std::vector<Expr> &input_dims, std::vector<Expr> &output_dims) {
  Expr tmp = af::sym::kSymbolOne;
  bool status = true;
  for (const auto &expr : input_dims) {
    if (expr == 1U) {
      status = false;
      break;
    }
  }
  if (status) {
    return af::SUCCESS;
  }
  std::vector<Expr> tmp_input_dims = input_dims;
  std::vector<Expr> tmp_output_dims = output_dims;
  input_dims.clear();
  output_dims.clear();
  for (size_t i = 0; i < tmp_input_dims.size(); ++i) {
    if (tmp_input_dims[i] == 1 && tmp_output_dims[i] != 1) {
      if (tmp != 1) {
        input_dims.emplace_back(tmp);
        output_dims.emplace_back(tmp);
      }
      tmp = af::sym::kSymbolOne;
      input_dims.emplace_back(tmp_input_dims[i]);
      output_dims.emplace_back(tmp_output_dims[i]);
    } else {
      if (tmp_input_dims[i] != tmp_output_dims[i]) {
        GELOGW("Invalid merge.");
      }
      tmp = Mul(tmp, tmp_input_dims[i]);
    }
  }
  if (tmp != 1) {
    input_dims.emplace_back(tmp);
    output_dims.emplace_back(tmp);
  }
  return af::SUCCESS;
}

/*
Broadcastapi的性能公式：
1) input_shapes[0].dims[0] = 1 && input_shapes[0].dims[1] = output_shapes[0].dims[1]:
  (1, B) -> (A, B)
  (1, B, C) -> (A, B, C)
  调用BroadcastFirstDim函数
  A * CopyApi(B * C)
2) input_shapes[0].dims[0] = output_shapes[0].dims[0] && input_shapes[0].dims[1] = 1:
  2.1) input_shapes[0].dims.size() == 2
    (A, 1) -> (A, B)
    调用TwoDimBroadCastLastDim<2,1>的函数
    首先执行tempbuf / (A / (32B * 32B) * 32B)次Align220函数：
      执行Align((A / (32B * 32B) * 32B), 8) / 255次Brcb，处理数据量为254或25
      尾块执行一次Brcb
      执行(A / (32B * 32B) * 32B)/255次Copy，处理数据为255 * B
      首轴的尾块再执行一次Copy
    再根据tempbuf / (A / (32B * 32B) * 32B)的尾块执行一次Align220
  2.2) input_shapes[0].dims.size() != 2
    (A, 1, C) -> (A, B, C)
      Duplicate + Or
3) input_shapes[0].dims.size() == 2 && output_shapes[0].dims.size() == 2 &&
  input_shapes[0].dims[0] = 1 && input_shapes[0].dims[1] = 1
    (1, 1) -> (A, B)
    DuplicateApi(A * B)
4) input_dims[0] == 1U && input_dims[2] == 1U && input_dims[1] == output_dims[1] && input_dims[3] == output_dims[3]
  (1, B, 1, B) -> (A, B, A, B)
  Broadcast (B,1,B)->(B,A,B)
  Broadcast (1,BAB)->(A,BAB)
*/
af::Status BroadcastApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  auto const &node_ptr = node.node_ptr;
  GE_ASSERT_TRUE(!output_shapes.empty() && !input_shapes.empty());
  auto input_dims = input_shapes[0].dims;
  auto output_dims = output_shapes[0].dims;
  GE_ASSERT_TRUE(!input_dims.empty() && !output_dims.empty());
  if (input_dims.size() != output_dims.size()) {
    GE_ASSERT_SUCCESS(ExpandBroadCastDims(input_dims, output_dims), "Expand failed, input size {%s}, output size {%s}.",
                      input_shapes[0].GetDimExpr().c_str(), output_shapes[0].GetDimExpr().c_str());
  }
  GE_ASSERT_TRUE(input_dims.size() == output_dims.size(), "input_dims.size() != output_dims.size()");
  GE_ASSERT_SUCCESS(MergeAxis(input_dims, output_dims), "MergeAxis Failed!");
  if (input_dims.size() == 2U) {
    GE_ASSERT_SUCCESS(BroadcastTwoDim(input_shapes[0].data_type, input_dims, output_dims, perf_res),
                      "Gen BroadcastTwoDim perf Failed, input size {%s}, output size {%s}.",
                      input_shapes[0].GetDimExpr().c_str(), output_shapes[0].GetDimExpr().c_str());
  } else if (input_dims.size() == 3U) {
    GE_ASSERT_SUCCESS(BroadcastThreeDim(input_shapes[0].data_type, input_dims, output_dims, perf_res),
                      "Gen BroadcastThreeDim perf Failed, input size {%s}, output size {%s}.",
                      input_shapes[0].GetDimExpr().c_str(), output_shapes[0].GetDimExpr().c_str());
  } else if (input_dims.size() == 4U) {
    GE_ASSERT_SUCCESS(BroadcastFourDim(input_shapes[0].data_type, input_dims, output_dims, perf_res),
                      "Gen BroadcastFourDim perf Failed, input size {%s}, output size {%s}.",
                      input_shapes[0].GetDimExpr().c_str(), output_shapes[0].GetDimExpr().c_str());
  } else if (input_dims.size() == 1U) {
    GE_ASSERT_SUCCESS(ascendcperf::DuplicatePerf(
                          GenNodeDetail(input_shapes[0].data_type, output_shapes[0].data_type, output_dims), perf_res),
                      "Gen DuplicateApi perf failed, node name: %s, type: %s", node_ptr->GetNamePtr(),
                      node_ptr->GetTypePtr());
  } else {
    GELOGW("input_dims.size[%zu] unsupported, input size {%s}, output size {%s}.", input_dims.size(),
           input_shapes[0].GetDimExpr().c_str(), output_shapes[0].GetDimExpr().c_str());
    perf_res.pipe_res[PipeType::AIV_VEC] = af::sym::kSymbolZero;
  }
  return af::SUCCESS;
}

af::Status LogicalCommonApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                            [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                            [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res,
                            const std::function<af::Status(const NodeDetail &, PerfOutputInfo &)> &calc_func_perf) {
  auto const &node_ptr = node.node_ptr;
  GE_ASSERT_TRUE(!input_shapes.empty());
  Expr data_size;
  Expr max_repeat_size = af::sym::Min(kTempBufSize / kSymPowerofEight, kMaxRepeatTime) * kSymPowerofSeven;
  GE_ASSERT_SUCCESS(ascendcperf::GetDatasize(output_shapes[0], data_size));
  PerfOutputInfo cast_perf1;
  PerfOutputInfo cast_perf2;
  PerfOutputInfo cast_perf3;
  PerfOutputInfo cast_perf4;
  PerfOutputInfo mul_perf1;
  PerfOutputInfo mul_perf2;
  Expr cycle_num = data_size / max_repeat_size;
  GE_ASSERT_SUCCESS(
      ascendcperf::CastPerf(GenNodeDetail(output_shapes[0].data_type, "float16", {max_repeat_size}), cast_perf1),
      "CastPerf failed, node=[%s,%s]", node_ptr->GetNamePtr(), node_ptr->GetTypePtr());
  GE_ASSERT_SUCCESS(ascendcperf::CastPerf(GenNodeDetail("float16", "uint8", {max_repeat_size}), cast_perf2),
                    "CastPerf failed, node=[%s,%s]", node_ptr->GetNamePtr(), node_ptr->GetTypePtr());
  GE_ASSERT_SUCCESS(
      ascendcperf::CastPerf(
          GenNodeDetail(output_shapes[0].data_type, "float16", {data_size - cycle_num * max_repeat_size}), cast_perf3),
      "CastPerf failed, node=[%s,%s]", node_ptr->GetNamePtr(), node_ptr->GetTypePtr());
  GE_ASSERT_SUCCESS(
      ascendcperf::CastPerf(GenNodeDetail("float16", "uint8", {data_size - cycle_num * max_repeat_size}), cast_perf4),
      "CastPerf failed, node=[%s,%s]", node_ptr->GetNamePtr(), node_ptr->GetTypePtr());
  GE_ASSERT_SUCCESS(calc_func_perf(GenNodeDetail("float16", "float16", {max_repeat_size}), mul_perf1),
                    "Calc func perf failed, node=[%s,%s]", node_ptr->GetNamePtr(), node_ptr->GetTypePtr());
  GE_ASSERT_SUCCESS(
      calc_func_perf(GenNodeDetail("float16", "float16", {data_size - cycle_num * max_repeat_size}), mul_perf2),
      "Calc func perf failed, node=[%s,%s]", node_ptr->GetNamePtr(), node_ptr->GetTypePtr());
  perf_res.pipe_res[PipeType::AIV_VEC] =
      kSymTwo * cycle_num * GetPipeCost(cast_perf1, PipeType::AIV_VEC) +
      cycle_num * GetPipeCost(mul_perf1, PipeType::AIV_VEC) + cycle_num * GetPipeCost(cast_perf2, PipeType::AIV_VEC) +
      kSymTwo * GetPipeCost(cast_perf3, PipeType::AIV_VEC) + GetPipeCost(mul_perf2, PipeType::AIV_VEC) +
      GetPipeCost(cast_perf4, PipeType::AIV_VEC);
  return af::SUCCESS;
}

/*
LogicalAndapi的性能公式：
  float16：1个Cast(halftouint8_t)搭配1个Mul
  float32: 1个Cast(halftouint8_t)搭配1个Mul搭配2个Cast(fp32tofp16)
*/
af::Status LogicalAndApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                         [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                         [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  return LogicalCommonApi(input_shapes, output_shapes, node, perf_res, ascendcperf::MulPerf);
}

/*
LogicalOrapi的性能公式：
  float16：1个Cast(halftouint8_t)搭配1个Max
  float32: 1个Cast(halftouint8_t)搭配1个Max搭配2个Cast(fp32tofp16)
*/
af::Status LogicalOrApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  return LogicalCommonApi(input_shapes, output_shapes, node, perf_res, ascendcperf::MaxPerf);
}

/*
LogicalNotapi的性能公式：
  比较复杂，按顺序执行Abs、Min、Sub、Abs算子
*/
af::Status LogicalNotApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                         [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                         [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  auto const &node_ptr = node.node_ptr;
  GE_ASSERT_TRUE(!input_shapes.empty());
  Expr data_size;
  Expr max_repeat_size = af::sym::Min((CreateExpr(8160)) / kSymPowerofEight, kMaxRepeatTime) * kSymPowerofSeven;
  GE_ASSERT_SUCCESS(ascendcperf::GetDatasize(output_shapes[0], data_size));
  GE_ASSERT_TRUE(output_shapes[0].data_type != "float32");
  PerfOutputInfo duplicate_perf;
  PerfOutputInfo abs_perf1;
  PerfOutputInfo min_perf1;
  PerfOutputInfo sub_perf1;
  PerfOutputInfo abs_perf2;
  PerfOutputInfo min_perf2;
  PerfOutputInfo sub_perf2;
  Expr cycle_num = data_size / max_repeat_size;
  GE_ASSERT_SUCCESS(ascendcperf::DuplicatePerf(GenNodeDetail("float16", "float16", {kSymSixteen}), duplicate_perf));
  GE_ASSERT_SUCCESS(ascendcperf::AbsPerf(GenNodeDetail("float16", "float16", {max_repeat_size}), abs_perf1));
  GE_ASSERT_SUCCESS(ascendcperf::MinPerf(GenNodeDetail("float16", "float16", {max_repeat_size}), min_perf1));
  GE_ASSERT_SUCCESS(ascendcperf::SubPerf(GenNodeDetail("float16", "float16", {max_repeat_size}), sub_perf1));
  GE_ASSERT_SUCCESS(
      ascendcperf::AbsPerf(GenNodeDetail("float16", "float16", {data_size - cycle_num * max_repeat_size}), abs_perf2),
      "Abs perf failed, node=[%s,%s]", node_ptr->GetNamePtr(), node_ptr->GetTypePtr());
  GE_ASSERT_SUCCESS(
      ascendcperf::MinPerf(GenNodeDetail("float16", "float16", {data_size - cycle_num * max_repeat_size}), min_perf2),
      "Min perf failed, node=[%s,%s]", node_ptr->GetNamePtr(), node_ptr->GetTypePtr());
  GE_ASSERT_SUCCESS(
      ascendcperf::SubPerf(GenNodeDetail("float16", "float16", {data_size - cycle_num * max_repeat_size}), sub_perf2),
      "Sub perf failed, node=[%s,%s]", node_ptr->GetNamePtr(), node_ptr->GetTypePtr());
  perf_res.pipe_res[PipeType::AIV_VEC] =
      GetPipeCost(duplicate_perf, PipeType::AIV_VEC) + kSymTwo * cycle_num * GetPipeCost(abs_perf1, PipeType::AIV_VEC) +
      cycle_num * GetPipeCost(min_perf1, PipeType::AIV_VEC) + cycle_num * GetPipeCost(sub_perf1, PipeType::AIV_VEC) +
      kSymTwo * GetPipeCost(abs_perf2, PipeType::AIV_VEC) + GetPipeCost(min_perf2, PipeType::AIV_VEC) +
      GetPipeCost(sub_perf2, PipeType::AIV_VEC);
  return af::SUCCESS;
}

/*
ReduceMaxapi的性能公式：
  比较复杂，目前先实现AR形式且isReuseSource=false的场景
  建模时根据尾轴>32B且<256B的分支建模
*/
af::Status ReduceMaxPerf([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes, PerfOutputInfo &perf_res) {
  std::string data_type = input_shapes[0].data_type;
  auto dims = input_shapes[0].dims;
  GE_ASSERT_TRUE(dims.size() == 2U);
  auto it = kBlkEleMap.find(input_shapes[0].data_type);
  GE_ASSERT_TRUE(it != kBlkEleMap.end());
  PerfOutputInfo adds_perf;
  PerfOutputInfo max_perf1;
  PerfOutputInfo max_perf2;
  PerfOutputInfo max_perf3;
  PerfOutputInfo blockreducemax_perf;
  const auto &first = dims[0];
  const auto &last = dims[1];
  Expr first_repeat = first / kMaxRepeatTime;
  Expr repeat_tail = first - first_repeat * kMaxRepeatTime;
  Expr ele_per_blk = it->second;
  Expr blk_count = last / ele_per_blk;
  Expr blk_tail = last - blk_count * ele_per_blk;
  Expr blk_pad = af::sym::Ceiling(blk_tail / (blk_tail + CreateExpr(1)));
  GE_ASSERT_SUCCESS(ascendcperf::AddsPerf(GenNodeDetail(data_type, data_type, {first * ele_per_blk}), adds_perf));
  GE_ASSERT_SUCCESS(ascendcperf::MaxPerf(GenNodeDetail(data_type, data_type, {first * ele_per_blk}), max_perf1));
  GE_ASSERT_SUCCESS(ascendcperf::MaxPerf(GenNodeDetail(data_type, data_type, {kMaxRepeatTime}), max_perf2));
  GE_ASSERT_SUCCESS(ascendcperf::MaxPerf(GenNodeDetail(data_type, data_type, {repeat_tail}), max_perf3));
  GE_ASSERT_SUCCESS(
      ascendcperf::BlockReduceMaxPerf(GenNodeDetail(data_type, data_type, {first * ele_per_blk}), blockreducemax_perf));
  perf_res.pipe_res[PipeType::AIV_VEC] =
      GetPipeCost(adds_perf, PipeType::AIV_VEC) + blk_count * GetPipeCost(max_perf1, PipeType::AIV_VEC) +
      blk_pad * first_repeat * GetPipeCost(max_perf2, PipeType::AIV_VEC) +
      blk_pad * GetPipeCost(max_perf3, PipeType::AIV_VEC) + GetPipeCost(blockreducemax_perf, PipeType::AIV_VEC);
  return af::SUCCESS;
}

af::Status ReduceMaxApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(!output_shapes.empty() && !input_shapes.empty());
  return ReduceMaxPerf(input_shapes, perf_res);
}

/*
Argmaxapi的性能公式：
  Argmax的性能特征与Max类似，复用ReduceMaxPerf
*/
af::Status ArgmaxPerf([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes, PerfOutputInfo &perf_res) {
  return ReduceMaxPerf(input_shapes, perf_res);
}

af::Status ArgmaxApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                     [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                     [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(!output_shapes.empty() && !input_shapes.empty());
  return ArgmaxPerf(input_shapes, perf_res);
}

/*
ArgMaxMultiRPhase1和ArgMaxMultiRPhase2的性能公式：
  性能特征与Argmax类似，复用ArgmaxPerf
*/
af::Status ArgMaxMultiRPhase1Api([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                                 [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                                 [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(!output_shapes.empty() && !input_shapes.empty());
  return ArgmaxPerf(input_shapes, perf_res);
}

af::Status ArgMaxMultiRPhase2Api([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                                 [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                                 [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(!input_shapes.empty());
  return ArgmaxPerf(input_shapes, perf_res);
}

/*
ReduceMinapi的性能公式：
  比较复杂，目前先实现AR形式且isReuseSource=false的场景
  建模时根据尾轴>32B且<256B的分支建模
*/
af::Status ReduceMinPerf([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes, PerfOutputInfo &perf_res) {
  std::string data_type = input_shapes[0].data_type;
  auto dims = input_shapes[0].dims;
  GE_ASSERT_TRUE(dims.size() == 2U);
  auto it = kBlkEleMap.find(input_shapes[0].data_type);
  GE_ASSERT_TRUE(it != kBlkEleMap.end());
  PerfOutputInfo adds_perf;
  PerfOutputInfo min_perf1;
  PerfOutputInfo min_perf2;
  PerfOutputInfo min_perf3;
  PerfOutputInfo blockreducemin_perf;
  Expr first = dims[0];
  Expr last = dims[1];
  Expr first_repeat = first / kMaxRepeatTime;
  Expr repeat_tail = first - first_repeat * kMaxRepeatTime;
  Expr ele_per_blk = it->second;
  Expr blk_count = last / ele_per_blk;
  Expr blk_tail = last - blk_count * ele_per_blk;
  Expr blk_pad = af::sym::Ceiling(blk_tail / (blk_tail + CreateExpr(1)));
  GE_ASSERT_SUCCESS(ascendcperf::AddsPerf(GenNodeDetail(data_type, data_type, {first * ele_per_blk}), adds_perf));
  GE_ASSERT_SUCCESS(ascendcperf::MinPerf(GenNodeDetail(data_type, data_type, {first * ele_per_blk}), min_perf1));
  GE_ASSERT_SUCCESS(ascendcperf::MinPerf(GenNodeDetail(data_type, data_type, {kMaxRepeatTime}), min_perf2));
  GE_ASSERT_SUCCESS(ascendcperf::MinPerf(GenNodeDetail(data_type, data_type, {repeat_tail}), min_perf3));
  GE_ASSERT_SUCCESS(
      ascendcperf::BlockReduceMinPerf(GenNodeDetail(data_type, data_type, {first * ele_per_blk}), blockreducemin_perf));
  perf_res.pipe_res[PipeType::AIV_VEC] =
      GetPipeCost(adds_perf, PipeType::AIV_VEC) + blk_count * GetPipeCost(min_perf1, PipeType::AIV_VEC) +
      blk_pad * first_repeat * GetPipeCost(min_perf2, PipeType::AIV_VEC) +
      blk_pad * GetPipeCost(min_perf3, PipeType::AIV_VEC) + GetPipeCost(blockreducemin_perf, PipeType::AIV_VEC);
  return af::SUCCESS;
}

af::Status ReduceMinApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(!output_shapes.empty() && !input_shapes.empty());
  return ReduceMinPerf(input_shapes, perf_res);
}

/*
  ReduceAllapi的性能公式：
*/
af::Status ReduceAllApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(!output_shapes.empty() && !input_shapes.empty());
  GE_ASSERT_TRUE(input_shapes[0].data_type != "uint8");
  return ReduceMinPerf(input_shapes, perf_res);
}

/*
  ReduceAnyapi的性能公式：
*/
af::Status ReduceAnyApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(!output_shapes.empty() && !input_shapes.empty());
  GE_ASSERT_TRUE(input_shapes[0].data_type != "uint8");
  return ReduceMaxPerf(input_shapes, perf_res);
}

/*
ReduceSumapi的性能公式：
  比较复杂，目前先实现RA形式且isReuseSource=false的场景
  ReduceSum在计算时会触发log2(first)次Add操作，由于目前symengine不支持，因此不建模
  这会导致计算出的cycle数偏小(差不多少log2(first/2)*40cycle)
*/
af::Status ReduceSumPerf([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes, PerfOutputInfo &perf_res) {
  std::string data_type = input_shapes[0].data_type;
  auto dims = input_shapes[0].dims;
  GE_ASSERT_TRUE(dims.size() == 2U);
  auto it = kBlkEleMap.find(input_shapes[0].data_type);
  GE_ASSERT_TRUE(it != kBlkEleMap.end());
  PerfOutputInfo adds_perf1;
  PerfOutputInfo adds_perf2;
  PerfOutputInfo add_perf1;
  PerfOutputInfo add_perf2;
  Expr first = dims[0];
  Expr last = dims[1];
  Expr split_k = (first / kSymTwo) * kSymTwo;
  Expr tail = first - split_k;
  Expr ele_per_blk = it->second;
  Expr tail_pad = af::sym::Ceiling(tail / (tail + CreateExpr(1)));
  Expr pad_last = af::sym::Ceiling(last / ele_per_blk) * ele_per_blk;
  GE_ASSERT_SUCCESS(ascendcperf::AddsPerf(GenNodeDetail(data_type, data_type, {split_k * pad_last}), adds_perf1));
  GE_ASSERT_SUCCESS(ascendcperf::AddPerf(GenNodeDetail(data_type, data_type, {tail * pad_last}), add_perf1));
  GE_ASSERT_SUCCESS(
      ascendcperf::AddPerf(GenNodeDetail(data_type, data_type, {(kSymTwo * split_k - kSymTwo) * pad_last}), add_perf2));
  GE_ASSERT_SUCCESS(ascendcperf::AddsPerf(GenNodeDetail(data_type, data_type, {last}), adds_perf2));
  perf_res.pipe_res[PipeType::AIV_VEC] =
      tail_pad * GetPipeCost(adds_perf1, PipeType::AIV_VEC) + tail_pad * GetPipeCost(add_perf1, PipeType::AIV_VEC) +
      GetPipeCost(add_perf2, PipeType::AIV_VEC) + GetPipeCost(adds_perf2, PipeType::AIV_VEC);
  return af::SUCCESS;
}

af::Status ReduceSumApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(!output_shapes.empty() && !input_shapes.empty());
  return ReduceSumPerf(input_shapes, perf_res);
}

/*
ReduceProdapi的性能公式：
  比较复杂，目前先实现RA形式且isReuseSource=false的场景
  ReduceProd在计算时会触发log2(first)次Mul操作，由于目前symengine不支持，因此不建模
  这会导致计算出的cycle数偏小(差不多少log2(first/2)*40cycle)
*/
af::Status ReduceProdPerf([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes, PerfOutputInfo &perf_res) {
  std::string data_type = input_shapes[0].data_type;
  auto dims = input_shapes[0].dims;
  GE_ASSERT_TRUE(dims.size() == 2U);
  auto it = kBlkEleMap.find(input_shapes[0].data_type);
  GE_ASSERT_TRUE(it != kBlkEleMap.end());
  PerfOutputInfo adds_perf1;
  PerfOutputInfo adds_perf2;
  PerfOutputInfo mul_perf1;
  PerfOutputInfo mul_perf2;
  Expr first = dims[0];
  Expr last = dims[1];
  Expr split_k = (first / kSymTwo) * kSymTwo;
  Expr remain = first - split_k;
  Expr remain_pad = af::sym::Ceiling(remain / (remain + CreateExpr(1)));
  Expr ele_per_blk = it->second;
  Expr pad_last = af::sym::Ceiling(last / ele_per_blk) * ele_per_blk;
  GE_ASSERT_SUCCESS(ascendcperf::AddsPerf(GenNodeDetail(data_type, data_type, {pad_last * split_k}), adds_perf1));
  GE_ASSERT_SUCCESS(ascendcperf::MulPerf(GenNodeDetail(data_type, data_type, {pad_last * remain}), mul_perf1));
  GE_ASSERT_SUCCESS(
      ascendcperf::MulPerf(GenNodeDetail(data_type, data_type, {(kSymTwo * split_k - kSymTwo) * pad_last}), mul_perf2));
  GE_ASSERT_SUCCESS(ascendcperf::AddsPerf(GenNodeDetail(data_type, data_type, {last}), adds_perf2));
  perf_res.pipe_res[PipeType::AIV_VEC] =
      remain_pad * GetPipeCost(adds_perf1, PipeType::AIV_VEC) + remain_pad * GetPipeCost(mul_perf1, PipeType::AIV_VEC) +
      GetPipeCost(mul_perf2, PipeType::AIV_VEC) + GetPipeCost(adds_perf2, PipeType::AIV_VEC);
  return af::SUCCESS;
}

af::Status ReduceProdApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                         [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                         [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(!output_shapes.empty() && !input_shapes.empty());
  return ReduceProdPerf(input_shapes, perf_res);
}

af::Status ReduceMeanApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                         [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                         [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(!output_shapes.empty() && !input_shapes.empty());
  auto dims = input_shapes[0].dims;
  GE_ASSERT_TRUE(dims.size() == 2U);
  PerfOutputInfo reduce_perf;
  PerfOutputInfo muls_perf;
  GE_ASSERT_SUCCESS(ReduceSumPerf(input_shapes, reduce_perf));
  GE_ASSERT_SUCCESS(
      ascendcperf::MulsPerf(GenNodeDetail(input_shapes[0].data_type, input_shapes[0].data_type, {dims[1]}), muls_perf));
  perf_res.pipe_res[PipeType::AIV_VEC] =
      GetPipeCost(reduce_perf, PipeType::AIV_VEC) + GetPipeCost(muls_perf, PipeType::AIV_VEC);
  return af::SUCCESS;
}

/*
Gatherapi的性能公式：
  DataCopy一次：处理src1的shape(gm->ub)
  DataCopy一次：处理src0的shape(gm->ub)
  Cast一次：处理src1的shape(int64_t->int32_t)
  Muls一次：处理src1的shape(int32_t)
  Gather一次：处理src1的shape
*/
af::Status GatherApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                     [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                     [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());
  Expr data_size0;
  Expr data_size1;
  PerfOutputInfo load_perf0;
  PerfOutputInfo load_perf1;
  PerfOutputInfo cast_perf;
  PerfOutputInfo muls_perf;
  PerfOutputInfo gather_perf;
  GE_ASSERT_SUCCESS(ascendcperf::GetDatasize(input_shapes[0], data_size0));
  GE_ASSERT_SUCCESS(ascendcperf::GetDatasize(input_shapes[1], data_size1));
  GE_ASSERT_SUCCESS(ascendcperf::LoadPerf(
                        GenNodeDetail(input_shapes[0].data_type, input_shapes[0].data_type, {data_size0}), load_perf0),
                    "Gen LoadApi perf Failed.");
  GE_ASSERT_SUCCESS(ascendcperf::LoadPerf(
                        GenNodeDetail(input_shapes[1].data_type, input_shapes[1].data_type, {data_size1}), load_perf1),
                    "Gen LoadApi perf Failed.");
  GE_ASSERT_SUCCESS(ascendcperf::CastPerf(GenNodeDetail(input_shapes[1].data_type, "int32", {data_size0}), cast_perf),
                    "Gen Cast perf Failed.");
  GE_ASSERT_SUCCESS(ascendcperf::MulsPerf(GenNodeDetail("int32", "int32", {data_size1}), muls_perf),
                    "Gen Muls perf Failed.");
  GE_ASSERT_SUCCESS(ascendcperf::GatherPerf(
                        GenNodeDetail(input_shapes[0].data_type, input_shapes[0].data_type, {data_size1}), gather_perf),
                    "Gen Gather perf Failed.");
  GE_ASSERT_SUCCESS(UpdateTenary(load_perf0, perf_res));
  GE_ASSERT_SUCCESS(UpdateTenary(load_perf1, perf_res));
  perf_res.pipe_res[PipeType::AIV_MTE2] =
      GetPipeCost(load_perf0, PipeType::AIV_MTE2) + GetPipeCost(load_perf0, PipeType::AIV_MTE2);
  perf_res.pipe_res[PipeType::AIV_VEC] = GetPipeCost(cast_perf, PipeType::AIV_VEC) +
                                         GetPipeCost(muls_perf, PipeType::AIV_VEC) +
                                         GetPipeCost(gather_perf, PipeType::AIV_VEC);
  return af::SUCCESS;
}

af::Status SafeCastNormalPerf(const std::string &input_dtype, const std::string &output_dtype, const Expr &repeat,
                              PerfOutputInfo &perf) {
  PerfOutputInfo cast_perf;
  GE_ASSERT_SUCCESS(ascendcperf::CastPerf(GenNodeDetail(input_dtype, output_dtype, {kRptSizeFloat}), cast_perf),
                    "Gen Cast perf Failed.");
  perf.pipe_res[PipeType::AIV_VEC] = repeat * GetPipeCost(cast_perf, PipeType::AIV_VEC);
  return af::SUCCESS;
}

af::Status DoSelectNormalPerf(const std::string &output_dtype, const Expr &repeat, PerfOutputInfo &perf) {
  PerfOutputInfo safe_cast_perf1;
  PerfOutputInfo eq_perf;
  PerfOutputInfo select_perf;
  PerfOutputInfo safe_cast_perf2;
  GE_ASSERT_SUCCESS(SafeCastNormalPerf("uint8", "float16", repeat, safe_cast_perf1));
  Expr cmp_size = (kRptSizeFloat * repeat + kRptSizeHalf - af::sym::kSymbolOne) / (kRptSizeHalf * kRptSizeHalf);
  GE_ASSERT_SUCCESS(ascendcperf::CompareScalarEQPerf(GenNodeDetail("float16", "float16", {cmp_size}), eq_perf),
                    "Gen CompareScalarEQ perf Failed.");
  GE_ASSERT_SUCCESS(ascendcperf::SelectPerf(GenNodeDetail("float32", "float32", {repeat * kRptSizeFloat}), select_perf),
                    "Gen Select perf Failed.");
  GE_ASSERT_SUCCESS(SafeCastNormalPerf("float32", output_dtype, repeat, safe_cast_perf2));
  perf.pipe_res[PipeType::AIV_VEC] =
      GetPipeCost(safe_cast_perf1, PipeType::AIV_VEC) + GetPipeCost(eq_perf, PipeType::AIV_VEC) +
      GetPipeCost(select_perf, PipeType::AIV_VEC) + GetPipeCost(safe_cast_perf2, PipeType::AIV_VEC);
  return af::SUCCESS;
}

af::Status CastSelectNormalPerf(const std::string &src1_dtype, const std::string &src2_dtype,
                                const std::string &dst_dtype, const Expr &repeat, PerfOutputInfo &perf) {
  PerfOutputInfo safe_cast_perf1;
  PerfOutputInfo safe_cast_perf2;
  PerfOutputInfo select_perf;
  GE_ASSERT_SUCCESS(SafeCastNormalPerf(src1_dtype, "float32", repeat, safe_cast_perf1));
  GE_ASSERT_SUCCESS(SafeCastNormalPerf(src2_dtype, "float32", repeat, safe_cast_perf2));
  GE_ASSERT_SUCCESS(DoSelectNormalPerf(dst_dtype, repeat, select_perf));
  perf.pipe_res[PipeType::AIV_VEC] = GetPipeCost(safe_cast_perf1, PipeType::AIV_VEC) +
                                     GetPipeCost(safe_cast_perf2, PipeType::AIV_VEC) +
                                     GetPipeCost(select_perf, PipeType::AIV_VEC);
  return af::SUCCESS;
}

af::Status SafeCastPerf(const std::string &input_dtype, const std::string &output_dtype, const Expr &do_size,
                        PerfOutputInfo &perf) {
  PerfOutputInfo cast_perf;
  GE_ASSERT_SUCCESS(ascendcperf::CastPerf(GenNodeDetail(input_dtype, output_dtype, {do_size}), cast_perf),
                    "Gen Cast perf failed,"
                    " input_dtype is %s, output_dtype is %s, do_size is %s.",
                    input_dtype.c_str(), output_dtype.c_str(), af::SymbolicUtils::ToString(do_size).c_str());
  perf.pipe_res[PipeType::AIV_VEC] = GetPipeCost(cast_perf, PipeType::AIV_VEC);
  return af::SUCCESS;
}

af::Status DoSelectPerf(const std::string &output_dtype, const Expr &do_size, Expr &repeat_times,
                        PerfOutputInfo &perf) {
  PerfOutputInfo safe_cast_perf1;
  PerfOutputInfo eq_perf;
  PerfOutputInfo select_perf;
  PerfOutputInfo safe_cast_perf2;
  GE_ASSERT_SUCCESS(SafeCastPerf("uint8", "float16", do_size, safe_cast_perf1));
  Expr cmp_size = af::sym::Floor((do_size + kRptSizeHalf - af::sym::kSymbolOne) / (kRptSizeHalf * kRptSizeHalf));
  GE_ASSERT_SUCCESS(ascendcperf::CompareScalarEQPerf(GenNodeDetail("float16", "float16", {cmp_size}), eq_perf),
                    "Gen CompareScalarEQ perf Failed.");
  GE_ASSERT_SUCCESS(
      ascendcperf::SelectPerf(GenNodeDetail("float32", "float32", {repeat_times * kRptSizeFloat}), select_perf),
      "Gen Select perf Failed.");
  GE_ASSERT_SUCCESS(SafeCastNormalPerf("float32", output_dtype, do_size, safe_cast_perf2));
  perf.pipe_res[PipeType::AIV_VEC] =
      GetPipeCost(safe_cast_perf1, PipeType::AIV_VEC) + GetPipeCost(eq_perf, PipeType::AIV_VEC) +
      GetPipeCost(select_perf, PipeType::AIV_VEC) + GetPipeCost(safe_cast_perf2, PipeType::AIV_VEC);
  return af::SUCCESS;
}

Expr CastBeforeSelectPerf(const std::string &src1_dtype, const std::string &src2_dtype, const std::string &dst_dtype,
                          const Expr &do_size, Expr repeat_times) {
  PerfOutputInfo safe_cast_perf1;
  PerfOutputInfo safe_cast_perf2;
  PerfOutputInfo select_perf;
  GE_ASSERT_SUCCESS(SafeCastPerf(src1_dtype, "float32", do_size, safe_cast_perf1));
  GE_ASSERT_SUCCESS(SafeCastPerf(src2_dtype, "float32", do_size, safe_cast_perf2));
  GE_ASSERT_SUCCESS(DoSelectPerf(dst_dtype, do_size, repeat_times, select_perf));
  return GetPipeCost(safe_cast_perf1, PipeType::AIV_VEC) + GetPipeCost(safe_cast_perf2, PipeType::AIV_VEC) +
         GetPipeCost(select_perf, PipeType::AIV_VEC);
}

af::Status WhereBasePerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("WhereBasePerf: node info is %s.", node_info.ToString().c_str());
  Expr one_rpt_size = kRptSizeFloat;
  Expr size = node_info.input_dims[kNumZero];
  GELOGD("one_rpt_size: [%s], dim size is [%s]", af::SymbolicUtils::ToString(one_rpt_size).c_str(),
         af::SymbolicUtils::ToString(size).c_str());
  Expr max_do_size = kMaxRepeatTime * one_rpt_size;  // 16320
  GELOGD("max_do_size is [%s]", af::SymbolicUtils::ToString(max_do_size).c_str());
  Expr branch_max_repeat_cost = CastBeforeSelectPerf(node_info.input_dtype[kNumOne], node_info.input_dtype[kNumTwo],
                                                     node_info.output_dtype[kNumZero], max_do_size, kMaxRepeatTime);
  GELOGD("branch_max_repeat_cost is [%s]", af::SymbolicUtils::ToString(branch_max_repeat_cost).c_str());
  Expr left_repeat_times =
      (size - max_do_size) / kSymFour;  // 三元表达式的max_do_size <= size可以保证left_repeat_times >= 0
  Expr left_sign =
      af::sym::Ceiling(left_repeat_times / (left_repeat_times + af::sym::kSymbolOne));  // 判断left_repeat_times是否为0
  Expr left_do_size = left_repeat_times * kRptSizeFloat;
  GELOGD("left_sign is [%s], left_repeat_times is [%s], left_do_size is [%s]",
         af::SymbolicUtils::ToString(left_sign).c_str(), af::SymbolicUtils::ToString(left_repeat_times).c_str(),
         af::SymbolicUtils::ToString(left_do_size).c_str());
  Expr branch_left_repeat_cost =
      CastBeforeSelectPerf(node_info.input_dtype[kNumOne], node_info.input_dtype[kNumTwo],
                           node_info.output_dtype[kNumZero], left_do_size, left_repeat_times);
  GELOGD("branch_left_repeat_cost is [%s]", af::SymbolicUtils::ToString(branch_left_repeat_cost).c_str());
  Expr repeat_times = size / kSymFour;
  Expr do_size = repeat_times * kRptSizeFloat;
  GELOGD("repeat_times is [%s], do_size is [%s]", af::SymbolicUtils::ToString(repeat_times).c_str(),
         af::SymbolicUtils::ToString(do_size).c_str());
  Expr branch_small_repeat_cost = CastBeforeSelectPerf(node_info.input_dtype[kNumOne], node_info.input_dtype[kNumTwo],
                                                       node_info.output_dtype[kNumZero], do_size, repeat_times);
  GELOGD("branch_small_repeat_cost is [%s]", af::SymbolicUtils::ToString(branch_small_repeat_cost).c_str());
  Expr res = CreateExpr("where_base_node");
  TernaryOp ternary_op =
      TernaryOp(CondType::K_LE, max_do_size, size, branch_max_repeat_cost + left_sign * branch_left_repeat_cost,
                branch_small_repeat_cost);
  ternary_op.SetVariable(res);
  perf.ternary_ops[res] = ternary_op;
  perf.pipe_res[PipeType::AIV_VEC] = res;
  return af::SUCCESS;
}

af::Status WhereExtendPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  GELOGD("WhereExtendPerf: node info is %s.", node_info.ToString().c_str());
  Expr one_rpt_size = kRptSizeFloat;
  GELOGD("one_rpt_size: [%s]", af::SymbolicUtils::ToString(one_rpt_size).c_str());
  Expr first_axis = node_info.input_dims[kNumZero];
  Expr last_axis = node_info.input_dims[kNumOne];
  GELOGD("first_axis is [%s], last_axis is [%s]", af::SymbolicUtils::ToString(first_axis).c_str(),
         af::SymbolicUtils::ToString(last_axis).c_str());
  Expr element_extent = af::sym::Floor(last_axis / one_rpt_size);
  Expr max_buf_size = af::sym::Div(kTempBufSize, kSymEight);
  Expr max_buf_rpt_num = af::sym::Div(max_buf_size, one_rpt_size);
  Expr max_do_rpt_num = af::sym::Min(kMaxRepeatTime, max_buf_rpt_num);
  GELOGD("max_do_rpt_num is [%s]", af::SymbolicUtils::ToString(max_do_rpt_num).c_str());
  GE_ASSERT_TRUE(max_do_rpt_num != af::sym::kSymbolZero);
  Expr repeat_throw_for_extent = af::sym::Floor(first_axis / max_do_rpt_num);
  Expr repeat_reminder = af::sym::Mod(first_axis, max_do_rpt_num);
  Expr repeat_sign = af::sym::Ceiling(repeat_reminder / (repeat_reminder + af::sym::kSymbolOne));
  GELOGD("element_extent is [%s], repeat_sign is [%s], repeat_reminder is [%s], repeat_throw_for_extent is [%s]",
         af::SymbolicUtils::ToString(element_extent).c_str(), af::SymbolicUtils::ToString(repeat_sign).c_str(),
         af::SymbolicUtils::ToString(repeat_reminder).c_str(),
         af::SymbolicUtils::ToString(repeat_throw_for_extent).c_str());
  PerfOutputInfo cast_select_perf1;
  PerfOutputInfo cast_select_perf2;
  GE_ASSERT_SUCCESS(CastSelectNormalPerf(node_info.input_dtype[kNumOne], node_info.input_dtype[kNumTwo],
                                         node_info.output_dtype[kNumZero], max_do_rpt_num, cast_select_perf1));
  GELOGD("cast_select_perf1 is [%s]",
         af::SymbolicUtils::ToString(GetPipeCost(cast_select_perf1, PipeType::AIV_VEC)).c_str());
  GE_ASSERT_SUCCESS(CastSelectNormalPerf(node_info.input_dtype[kNumOne], node_info.input_dtype[kNumTwo],
                                         node_info.output_dtype[kNumZero], repeat_reminder, cast_select_perf2));
  GELOGD("cast_select_perf2 is [%s]",
         af::SymbolicUtils::ToString(GetPipeCost(cast_select_perf2, PipeType::AIV_VEC)).c_str());
  perf.pipe_res[PipeType::AIV_VEC] =
      element_extent * (repeat_throw_for_extent * GetPipeCost(cast_select_perf1, PipeType::AIV_VEC) +
                        repeat_sign * GetPipeCost(cast_select_perf2, PipeType::AIV_VEC));
  return af::SUCCESS;
}

af::Status WhereApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                    [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                    [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  auto const &node_ptr = node.node_ptr;
  GE_ASSERT_TRUE(input_shapes.size() >= 3U && !output_shapes.empty());
  auto merged_output_shapes = output_shapes[0];
  GE_ASSERT_SUCCESS(MergeTensorContinuousDims(node_ptr, GetNodeOutTensorName(node_ptr, 0), merged_output_shapes));
  NodeDetail node_info;
  Expr outer_repeat;
  vector<Expr> used_dims;
  GE_ASSERT_SUCCESS(GetOuterParams(merged_output_shapes.dims, outer_repeat, used_dims));
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(SetDims(used_dims, node_info));
  auto input_dims_size = node_info.input_dims.size();
  GELOGD("input_dims_size is {%u}", input_dims_size);
  if (input_dims_size == 1U) {
    GE_ASSERT_SUCCESS(WhereBasePerf(node_info, perf_res));
  } else {
    GE_ASSERT_SUCCESS(WhereExtendPerf(node_info, perf_res));
  }
  perf_res.pipe_res[PipeType::AIV_VEC] = outer_repeat * GetPipeCost(perf_res, PipeType::AIV_VEC);
  GELOGD("outer_repeat is [%s]", af::SymbolicUtils::ToString(outer_repeat).c_str());
  GELOGD("Result is [%s]", af::SymbolicUtils::ToString(GetPipeCost(perf_res, PipeType::AIV_VEC)).c_str());
  return af::SUCCESS;
}

inline af::Status CompareSpecificPerf(const NodeDetail &node_info, const std::string &mode, PerfOutputInfo &perf,
                                      TernaryOpMap &ternary_ops_map) {
  if (mode == kGe) {
    ascendcperf::CompareGEPerf(node_info, perf);
  } else if (mode == kEq) {
    ascendcperf::CompareEQPerf(node_info, perf);
  } else if (mode == kNe) {
    ascendcperf::CompareNEPerf(node_info, perf);
  } else if (mode == kGt) {
    ascendcperf::CompareGTPerf(node_info, perf);
  } else if (mode == kLe) {
    ascendcperf::CompareLEPerf(node_info, perf);
  } else if (mode == kLt) {
    ascendcperf::CompareLTPerf(node_info, perf);
  }
  ternary_ops_map.insert(perf.ternary_ops.begin(), perf.ternary_ops.end());
  return af::SUCCESS;
}

inline af::Status CompareScalarEqNePerf(const NodeDetail &node_info, const std::string &mode, PerfOutputInfo &perf) {
  GE_ASSERT_TRUE(mode == kEq || mode == kNe);
  if (mode == kEq) {
    ascendcperf::CompareScalarEQPerf(node_info, perf);
  } else {
    ascendcperf::CompareScalarNEPerf(node_info, perf);
  }
  return af::SUCCESS;
}

inline Expr CompareInt64EqNeBranchA(const std::string &mode, const Expr &repeat_times) {
  PerfOutputInfo sub_perf;
  ascendcperf::SubPerf(GenNodeDetail("float32", "float32", {repeat_times * kRptSizeFloat}), sub_perf);
  Expr sub_cost = GetPipeCost(sub_perf, PipeType::AIV_VEC);
  PerfOutputInfo cast_perf;
  ascendcperf::CastPerf(GenNodeDetail("int64", "int32", {repeat_times * kRptSizeInt64}), cast_perf);
  Expr cast_cost = GetPipeCost(cast_perf, PipeType::AIV_VEC);
  PerfOutputInfo compare_scalar_perf;
  CompareScalarEqNePerf(GenNodeDetail("float32", "float32", {repeat_times * kRptSizeFloat}), mode, compare_scalar_perf);
  Expr compare_scalar_cost = GetPipeCost(compare_scalar_perf, PipeType::AIV_VEC);
  PerfOutputInfo select_perf;
  ascendcperf::SelectPerf(GenNodeDetail("float16", "float16", {repeat_times * kRptSizeHalf}), select_perf);
  Expr select_cost = GetPipeCost(select_perf, PipeType::AIV_VEC);
  PerfOutputInfo cast_perf2;
  ascendcperf::CastPerf(GenNodeDetail("float16", "uint8", {repeat_times * kRptSizeHalf}), cast_perf2);
  Expr cast_cost2 = GetPipeCost(cast_perf2, PipeType::AIV_VEC);
  return sub_cost + cast_cost + compare_scalar_cost + select_cost + cast_cost2;
}

inline Expr CompareInt64EqNeBranchB1(const std::string &mode, const Expr &repeat_times, const Expr &element_extent) {
  PerfOutputInfo sub_perf;
  ascendcperf::SubPerf(GenNodeDetail("float32", "float32", {repeat_times * kRptSizeFloat}), sub_perf);
  Expr sub_cost = GetPipeCost(sub_perf, PipeType::AIV_VEC);
  PerfOutputInfo cast_perf;
  ascendcperf::CastPerf(GenNodeDetail("int64", "int32", {repeat_times * kRptSizeInt64}), cast_perf);
  Expr cast_cost = GetPipeCost(cast_perf, PipeType::AIV_VEC);
  PerfOutputInfo compare_scalar_perf;
  CompareScalarEqNePerf(GenNodeDetail("float32", "float32", {af::sym::Ceiling(repeat_times / kSymTwo) * kRptSizeFloat}),
                        mode, compare_scalar_perf);
  Expr compare_scalar_cost = GetPipeCost(compare_scalar_perf, PipeType::AIV_VEC);
  PerfOutputInfo select_perf;
  ascendcperf::SelectPerf(
      GenNodeDetail("float16", "float16", {af::sym::Ceiling(repeat_times / kSymFour) * kRptSizeHalf}), select_perf);
  Expr select_cost = GetPipeCost(select_perf, PipeType::AIV_VEC);
  PerfOutputInfo cast_perf2;
  ascendcperf::CastPerf(GenNodeDetail("float16", "uint8", {repeat_times * kRptSizeHalf}), cast_perf2);
  Expr cast_cost2 = GetPipeCost(cast_perf2, PipeType::AIV_VEC);
  return element_extent * (sub_cost + cast_cost + compare_scalar_cost + select_cost + cast_cost2);
}

inline Expr CompareInt64EqNeBranchB2(const std::string &mode, const Expr &repeat_times, const Expr &element_extent) {
  PerfOutputInfo sub_perf;
  ascendcperf::SubPerf(GenNodeDetail("float32", "float32", {element_extent * kRptSizeFloat}), sub_perf);
  Expr sub_cost = GetPipeCost(sub_perf, PipeType::AIV_VEC);
  PerfOutputInfo cast_perf;
  ascendcperf::CastPerf(GenNodeDetail("int64", "int32", {element_extent * kRptSizeInt64}), cast_perf);
  Expr cast_cost = GetPipeCost(cast_perf, PipeType::AIV_VEC);
  PerfOutputInfo compare_scalar_perf;
  CompareScalarEqNePerf(
      GenNodeDetail("float32", "float32", {af::sym::Ceiling(element_extent / kSymTwo) * kRptSizeFloat}), mode,
      compare_scalar_perf);
  Expr compare_scalar_cost = GetPipeCost(compare_scalar_perf, PipeType::AIV_VEC);
  PerfOutputInfo select_perf;
  ascendcperf::SelectPerf(
      GenNodeDetail("float16", "float16", {af::sym::Ceiling(element_extent / kSymFour) * kRptSizeHalf}), select_perf);
  Expr select_cost = GetPipeCost(select_perf, PipeType::AIV_VEC);
  PerfOutputInfo cast_perf2;
  ascendcperf::CastPerf(GenNodeDetail("float16", "uint8", {element_extent * kRptSizeHalf}), cast_perf2);
  Expr cast_cost2 = GetPipeCost(cast_perf2, PipeType::AIV_VEC);
  return repeat_times * (sub_cost + cast_cost + compare_scalar_cost + select_cost + cast_cost2);
}

inline af::Status CompareInt64EqNePerf(const std::string &mode, const Expr &repeat_times, const Expr &element_extent,
                                       const Expr &one_rpt_size, const Expr &last_axis, PerfOutputInfo &perf) {
  PerfOutputInfo duplicate_perf;
  ascendcperf::DuplicatePerf(GenNodeDetail("float16", "float16", {kRptSizeHalf}), duplicate_perf);
  Expr duplicate_cost = GetPipeCost(duplicate_perf, PipeType::AIV_VEC);
  Expr branch_a_pipe_cost = CompareInt64EqNeBranchA(mode, repeat_times) + duplicate_cost;
  GELOGD("CompareInt64EqNe branch_a: [%s]", af::SymbolicUtils::ToString(branch_a_pipe_cost).c_str());
  Expr branch_b1_pipe_cost = CompareInt64EqNeBranchB1(mode, repeat_times, element_extent) + duplicate_cost;
  GELOGD("CompareInt64EqNe branch_b1: [%s]", af::SymbolicUtils::ToString(branch_b1_pipe_cost).c_str());
  Expr branch_b2_pipe_cost = CompareInt64EqNeBranchB2(mode, repeat_times, element_extent) + duplicate_cost;
  GELOGD("CompareInt64EqNe branch_b2: [%s]", af::SymbolicUtils::ToString(branch_b2_pipe_cost).c_str());
  Expr res = CreateExpr("compare_node");
  std::shared_ptr<IfCase> branch_a = std::make_shared<IfCase>(branch_a_pipe_cost);
  GE_ASSERT_NOTNULL(branch_a);
  std::shared_ptr<IfCase> branch_b1 = std::make_shared<IfCase>(branch_b1_pipe_cost);
  std::shared_ptr<IfCase> branch_b2 = std::make_shared<IfCase>(branch_b2_pipe_cost);
  GE_ASSERT_NOTNULL(branch_b1);
  GE_ASSERT_NOTNULL(branch_b2);
  std::shared_ptr<IfCase> branch_b = std::make_shared<IfCase>(CondType::K_GT, element_extent, repeat_times,
                                                              std::move(branch_b2), std::move(branch_b1));
  GE_ASSERT_NOTNULL(branch_b);
  TernaryOp ternary_op = TernaryOp(CondType::K_LT, last_axis, one_rpt_size, std::move(branch_a), std::move(branch_b));
  ternary_op.SetVariable(res);
  perf.ternary_ops[res] = ternary_op;
  GELOGD("CompareInt64EqNe's adjustment factor is [%lf]", kCompareInt64EqNeAdjustmentFactor);
  perf.pipe_res[PipeType::AIV_VEC] = res * CreateExpr(kCompareInt64EqNeAdjustmentFactor);  // verify得到的修正系数
  return af::SUCCESS;
}

inline Expr GetSignBitTensorNormalCost(const Expr &double_cal_cnt, const Expr &repeat_times) {
  PerfOutputInfo duplicate_perf;
  ascendcperf::DuplicatePerf(GenNodeDetail("float32", "float32", {double_cal_cnt}), duplicate_perf);
  Expr duplicate_cost = GetPipeCost(duplicate_perf, PipeType::AIV_VEC) * kSymTwo;
  PerfOutputInfo and_perf;
  ascendcperf::AndPerf(GenNodeDetail("float16", "float16", {repeat_times * kRptSizeHalf}), and_perf);
  Expr and_cost = GetPipeCost(and_perf, PipeType::AIV_VEC) * kSymTwo;
  PerfOutputInfo maxs_perf;
  ascendcperf::MaxsPerf(GenNodeDetail("float32", "float32", {repeat_times * kRptSizeFloat}), maxs_perf);
  Expr maxs_cost = GetPipeCost(maxs_perf, PipeType::AIV_VEC);
  PerfOutputInfo duplicate_perf2;
  ascendcperf::DuplicatePerf(GenNodeDetail("float32", "float32", {repeat_times * kRptSizeFloat}), duplicate_perf2);
  Expr duplicate_cost2 = GetPipeCost(duplicate_perf2, PipeType::AIV_VEC);
  return duplicate_cost + and_cost + maxs_cost + duplicate_cost2;
}

inline Expr CastTensorToHalfNormalCost(const Expr &double_cal_cnt, const Expr &repeat_times) {
  PerfOutputInfo duplicate_perf;
  ascendcperf::DuplicatePerf(GenNodeDetail("float32", "float32", {double_cal_cnt}), duplicate_perf);
  Expr duplicate_cost = GetPipeCost(duplicate_perf, PipeType::AIV_VEC) * kSymThree;
  PerfOutputInfo and_perf;
  ascendcperf::AndPerf(GenNodeDetail("float16", "float16", {repeat_times * kRptSizeHalf}), and_perf);
  Expr and_cost = GetPipeCost(and_perf, PipeType::AIV_VEC) * kSymTwo;
  PerfOutputInfo sub_perf;
  ascendcperf::SubPerf(GenNodeDetail("float32", "float32", {repeat_times * kRptSizeFloat}), sub_perf);
  Expr sub_cost = GetPipeCost(sub_perf, PipeType::AIV_VEC);
  PerfOutputInfo or_perf;
  ascendcperf::OrPerf(GenNodeDetail("uint16", "uint16", {repeat_times * kRptSizeHalf}), or_perf);
  Expr or_cost = GetPipeCost(or_perf, PipeType::AIV_VEC);
  PerfOutputInfo pairreducesum_perf;
  ascendcperf::PairReduceSumPerf(GenNodeDetail("float16", "float16", {repeat_times * kRptSizeHalf}),
                                 pairreducesum_perf);
  Expr pairreducesum_cost = GetPipeCost(pairreducesum_perf, PipeType::AIV_VEC);
  return duplicate_cost + and_cost + sub_cost + or_cost + pairreducesum_cost;
}

inline Expr CalcWeightedTensorNormalCost(const Expr &double_cal_cnt, const Expr &repeat_times) {
  PerfOutputInfo duplicate_perf;
  ascendcperf::DuplicatePerf(GenNodeDetail("float16", "float16", {double_cal_cnt * kSymTwo}), duplicate_perf);
  Expr duplicate_cost = GetPipeCost(duplicate_perf, PipeType::AIV_VEC);
  PerfOutputInfo duplicate_perf2;
  ascendcperf::DuplicatePerf(GenNodeDetail("float16", "float16", {repeat_times * kRptSizeHalf}), duplicate_perf2);
  Expr duplicate_cost2 = GetPipeCost(duplicate_perf2, PipeType::AIV_VEC);
  PerfOutputInfo mul_perf;
  ascendcperf::MulPerf(GenNodeDetail("float16", "float16", {repeat_times * kRptSizeHalf}), mul_perf);
  Expr mul_cost = GetPipeCost(mul_perf, PipeType::AIV_VEC);
  PerfOutputInfo pairreducesum_perf;
  ascendcperf::PairReduceSumPerf(GenNodeDetail("float16", "float16", {repeat_times * kRptSizeHalf}),
                                 pairreducesum_perf);
  Expr pairreducesum_cost = GetPipeCost(pairreducesum_perf, PipeType::AIV_VEC);
  return duplicate_cost + duplicate_cost2 + mul_cost + pairreducesum_cost;
}

inline af::Status CompareInt64GtGeLePerf(const std::string &mode, const Expr &repeat_times, const Expr &element_extent,
                                         const Expr &one_rpt_size, const Expr &last_axis, PerfOutputInfo &perf) {
  (void)mode;
  Expr double_cal_cnt = kSymTwo * repeat_times * af::sym::Max(last_axis, one_rpt_size);
  GELOGD("CompareInt64GtGeLe double_cal_cnt: [%s]", af::SymbolicUtils::ToString(double_cal_cnt).c_str());
  Expr get_sign_bit_tensor_normal_cost = GetSignBitTensorNormalCost(double_cal_cnt, repeat_times) * kSymTwo;
  PerfOutputInfo sub_perf;
  ascendcperf::SubPerf(GenNodeDetail("float32", "float32", {repeat_times * kRptSizeFloat}), sub_perf);
  Expr sub_cost = GetPipeCost(sub_perf, PipeType::AIV_VEC) * kSymTwo;
  Expr cast_tensor_to_half_normal_cost = CastTensorToHalfNormalCost(double_cal_cnt, repeat_times) * kSymTwo;
  Expr calc_weighted_tensor_normal_cost = CalcWeightedTensorNormalCost(double_cal_cnt, repeat_times) * kSymTwo;
  PerfOutputInfo duplicate_perf;
  ascendcperf::DuplicatePerf(GenNodeDetail("float32", "float32", {double_cal_cnt}), duplicate_perf);
  Expr duplicate_cost = GetPipeCost(duplicate_perf, PipeType::AIV_VEC);
  PerfOutputInfo and_perf;
  ascendcperf::AndPerf(GenNodeDetail("float16", "float16", {repeat_times * kRptSizeHalf}), and_perf);
  Expr and_cost = GetPipeCost(and_perf, PipeType::AIV_VEC) * kSymTwo;
  PerfOutputInfo maxs_perf;
  ascendcperf::MaxsPerf(GenNodeDetail("float32", "float32", {repeat_times * kRptSizeFloat}), maxs_perf);
  Expr maxs_cost = GetPipeCost(maxs_perf, PipeType::AIV_VEC);
  PerfOutputInfo mins_perf;
  ascendcperf::MinsPerf(GenNodeDetail("float32", "float32", {repeat_times * kRptSizeFloat}), mins_perf);
  Expr mins_cost = GetPipeCost(mins_perf, PipeType::AIV_VEC);
  PerfOutputInfo add_perf;
  ascendcperf::AddPerf(GenNodeDetail("float16", "float16", {repeat_times * kRptSizeHalf}), add_perf);
  Expr add_cost = GetPipeCost(add_perf, PipeType::AIV_VEC);
  PerfOutputInfo maxs_perf2;
  ascendcperf::MaxsPerf(GenNodeDetail("float16", "float16", {repeat_times * kRptSizeHalf}), maxs_perf2);
  Expr maxs_cost2 = GetPipeCost(maxs_perf2, PipeType::AIV_VEC);
  PerfOutputInfo mins_perf2;
  ascendcperf::MinsPerf(GenNodeDetail("float16", "float16", {repeat_times * kRptSizeHalf}), mins_perf2);
  Expr mins_cost2 = GetPipeCost(mins_perf2, PipeType::AIV_VEC);
  PerfOutputInfo cast_perf;
  ascendcperf::CastPerf(GenNodeDetail("float16", "uint8", {repeat_times * kRptSizeHalf}), cast_perf);
  Expr cast_cost = GetPipeCost(cast_perf, PipeType::AIV_VEC);
  Expr one_rpt_cost = get_sign_bit_tensor_normal_cost + sub_cost + cast_tensor_to_half_normal_cost +
                      calc_weighted_tensor_normal_cost + duplicate_cost + and_cost + maxs_cost + mins_cost + add_cost +
                      maxs_cost2 + mins_cost2 + cast_cost;
  GELOGD("CompareInt64GtGeLe one_rpt_cost: [%s], adjustment factor is [%lf]",
         af::SymbolicUtils::ToString(one_rpt_cost).c_str(), kCompareInt64GtGeLeAdjustmentFactor);
  perf.pipe_res[PipeType::AIV_VEC] =
      element_extent * one_rpt_cost * CreateExpr(kCompareInt64GtGeLeAdjustmentFactor);  // verify得到的修正系数
  return af::SUCCESS;
}

inline Expr CompareBranchInputRepeatTimeCost(const NodeDetail &node_info, const std::string &mode,
                                             const Expr &repeat_times, const Expr &one_rpt_size,
                                             const Expr &element_extent, TernaryOpMap &ternary_ops_map) {
  PerfOutputInfo compare_perf;
  CompareSpecificPerf(GenNodeDetail(node_info.input_dtype[0], node_info.input_dtype[0], {repeat_times * one_rpt_size}),
                      mode, compare_perf, ternary_ops_map);
  Expr compare_cost = GetPipeCost(compare_perf, PipeType::AIV_VEC);
  PerfOutputInfo duplicate_perf;
  ascendcperf::DuplicatePerf(
      GenNodeDetail(node_info.input_dtype[0], node_info.input_dtype[0], {repeat_times * one_rpt_size}), duplicate_perf);
  Expr duplicate_cost = GetPipeCost(duplicate_perf, PipeType::AIV_VEC);
  PerfOutputInfo select_perf;
  ascendcperf::SelectPerf(
      GenNodeDetail(node_info.input_dtype[0], node_info.input_dtype[0], {repeat_times * one_rpt_size}), select_perf);
  Expr select_cost = GetPipeCost(select_perf, PipeType::AIV_VEC);
  PerfOutputInfo cast_perf;
  ascendcperf::CastPerf(
      GenNodeDetail(node_info.input_dtype[0], node_info.output_dtype[0], {repeat_times * one_rpt_size}), cast_perf);
  Expr cast_cost = GetPipeCost(cast_perf, PipeType::AIV_VEC);
  Expr one_cycle_total = compare_cost + duplicate_cost + select_cost + cast_cost;
  return element_extent * one_cycle_total;
}

inline Expr CompareBranchInputLastAxisCost(const NodeDetail &node_info, const std::string &mode,
                                           const Expr &repeat_times, const Expr &last_axis,
                                           TernaryOpMap &ternary_ops_map) {
  PerfOutputInfo compare_perf;
  CompareSpecificPerf(GenNodeDetail(node_info.input_dtype[0], node_info.input_dtype[0], {last_axis}), mode,
                      compare_perf, ternary_ops_map);
  Expr compare_cost = GetPipeCost(compare_perf, PipeType::AIV_VEC);
  PerfOutputInfo duplicate_perf;
  ascendcperf::DuplicatePerf(GenNodeDetail(node_info.input_dtype[0], node_info.input_dtype[0], {last_axis}),
                             duplicate_perf);
  Expr duplicate_cost = GetPipeCost(duplicate_perf, PipeType::AIV_VEC);
  PerfOutputInfo select_perf;
  ascendcperf::SelectPerf(GenNodeDetail(node_info.input_dtype[0], node_info.input_dtype[0], {last_axis}), select_perf);
  Expr select_cost = GetPipeCost(select_perf, PipeType::AIV_VEC);
  PerfOutputInfo cast_perf;
  ascendcperf::CastPerf(GenNodeDetail(node_info.input_dtype[0], node_info.output_dtype[0], {last_axis}), cast_perf);
  Expr cast_cost = GetPipeCost(cast_perf, PipeType::AIV_VEC);
  return repeat_times * (compare_cost + duplicate_cost + select_cost + cast_cost);
}

inline af::Status CompareNormalPerf(const NodeDetail &node_info, const std::string &mode, const Expr &repeat_times,
                                    const Expr &element_extent, const Expr &one_rpt_size, const Expr &last_axis,
                                    PerfOutputInfo &perf) {
  Expr branch_input_repeat_time_cost =
      CompareBranchInputRepeatTimeCost(node_info, mode, repeat_times, one_rpt_size, element_extent, perf.ternary_ops);
  GELOGD("CompareNormal branch_1: [%s]", af::SymbolicUtils::ToString(branch_input_repeat_time_cost).c_str());
  Expr branch_input_last_axis_cost =
      CompareBranchInputLastAxisCost(node_info, mode, repeat_times, last_axis, perf.ternary_ops);
  GELOGD("CompareNormal branch_2: [%s]", af::SymbolicUtils::ToString(branch_input_last_axis_cost).c_str());
  Expr res = CreateExpr("compare_node");
  TernaryOp ternary_op = TernaryOp(CondType::K_GT, element_extent, repeat_times, branch_input_last_axis_cost,
                                   branch_input_repeat_time_cost);
  ternary_op.SetVariable(res);
  perf.ternary_ops[res] = ternary_op;
  GELOGD("CompareNormal's adjustment factor is [%lf]", kCompareNormalAdjustmentFactor);
  perf.pipe_res[PipeType::AIV_VEC] = res * CreateExpr(kCompareNormalAdjustmentFactor);  // verify得到的修正系数
  return af::SUCCESS;
}

af::Status CompareExtendPerf(const NodeDetail &node_info, const std::string &mode, PerfOutputInfo &perf) {
  auto input_dims_size = node_info.input_dims.size();
  GELOGD("CompareExtend input_dims_size is {%u}.", input_dims_size);
  Expr first_axis = input_dims_size == 1U ? af::sym::kSymbolOne : node_info.input_dims[kNumZero];
  Expr last_axis = input_dims_size == 1U ? node_info.input_dims[kNumZero] : node_info.input_dims[kNumOne];
  Expr one_rpt_size = kRptSizeFloat;
  auto it = kRptEleMap.find(node_info.input_dtype[0]);
  if (it != kRptEleMap.end()) {
    one_rpt_size = it->second;
  }
  GE_ASSERT_TRUE(one_rpt_size != af::sym::kSymbolZero);
  Expr element_extent = af::sym::Ceiling(last_axis / one_rpt_size);
  const Expr &repeat_times = first_axis;
  GELOGD("CompareExtend input_dtype is {%s}, compare mode is {%s}, element_extent is [%s], repeat_times is [%s].",
         node_info.input_dtype[0].c_str(), mode.c_str(), af::SymbolicUtils::ToString(element_extent).c_str(),
         af::SymbolicUtils::ToString(repeat_times).c_str());
  if (node_info.input_dtype[0] == "int64") {
    if (mode == kEq || mode == kNe) {
      return CompareInt64EqNePerf(mode, repeat_times, element_extent, one_rpt_size, last_axis, perf);
    } else {
      return CompareInt64GtGeLePerf(mode, repeat_times, element_extent, one_rpt_size, last_axis, perf);
    }
  } else {
    return CompareNormalPerf(node_info, mode, repeat_times, element_extent, one_rpt_size, last_axis, perf);
  }
  return af::SUCCESS;
}

af::Status CompareApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                      [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                      [[maybe_unused]] const NodeInfo &node, const std::string &mode, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(input_shapes.size() >= 2U && !output_shapes.empty());
  NodeDetail node_info;
  Expr outer_repeat;
  vector<Expr> used_dims;
  GE_ASSERT_SUCCESS(GetOuterParams(output_shapes[0].dims, outer_repeat, used_dims));
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(SetDims(used_dims, node_info));
  GE_ASSERT_SUCCESS(CompareExtendPerf(node_info, mode, perf_res));
  perf_res.pipe_res[PipeType::AIV_VEC] = outer_repeat * GetPipeCost(perf_res, PipeType::AIV_VEC);
  return af::SUCCESS;
}

af::Status CompareGeApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  return CompareApi(input_shapes, output_shapes, node, kGe, perf_res);
}

af::Status CompareEqApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  return CompareApi(input_shapes, output_shapes, node, kEq, perf_res);
}

af::Status CompareNeApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  return CompareApi(input_shapes, output_shapes, node, kNe, perf_res);
}

af::Status CompareGtApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  return CompareApi(input_shapes, output_shapes, node, kGt, perf_res);
}

af::Status CompareLeApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  return CompareApi(input_shapes, output_shapes, node, kLe, perf_res);
}

af::Status CompareLtApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  return CompareApi(input_shapes, output_shapes, node, kLt, perf_res);
}

/*
RemovePad的性能公式：
  执行gathermask操作，重复mergeaxis次
*/
af::Status RemovePadApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());
  PerfOutputInfo gather_perf;
  GE_ASSERT_SUCCESS(ascendcperf::GatherMaskPerf(
      GenNodeDetail(input_shapes[0].data_type, output_shapes[0].data_type, output_shapes[0].dims), gather_perf));
  perf_res.pipe_res[PipeType::AIV_VEC] = GetPipeCost(gather_perf, PipeType::AIV_VEC);
  return af::SUCCESS;
}

af::Status CopyUbtoUbApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                         [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                         [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::CopyUbtoUbPerf(node_info, perf_res));
  return af::SUCCESS;
}

af::Status ZerosLikeApi([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                        [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                        [[maybe_unused]] const NodeInfo &node, PerfOutputInfo &perf_res) {
  NodeDetail node_info;
  GE_ASSERT_SUCCESS(SetNodeDetail(input_shapes, output_shapes, node_info));
  GE_ASSERT_SUCCESS(ascendcperf::DuplicatePerf(node_info, perf_res));
  return af::SUCCESS;
}
}  // namespace ascir_v1
REGISTER_EVAL_FUNC(kMoveGmToL1, ascir_v1::MoveGmtoL1Api);
REGISTER_EVAL_FUNC(kMoveL2ToL1, ascir_v1::MoveL2ToL1Api);
REGISTER_EVAL_FUNC(kMoveL1ToL0a, ascir_v1::MoveL1toL0aApi);
REGISTER_EVAL_FUNC(kMoveL1ToL0b, ascir_v1::MoveL1toL0bApi);
REGISTER_EVAL_FUNC(kMoveL0cToL2, ascir_v1::MoveL0cToL2Api);
REGISTER_EVAL_FUNC(kMoveL0cToGm, ascir_v1::MoveL0cToGmApi);
REGISTER_EVAL_FUNC(kStore, ascir_v1::StoreApi);
REGISTER_EVAL_FUNC(kLoad, ascir_v1::LoadApi);
REGISTER_EVAL_FUNC(kAbs, ascir_v1::AbsApi);
REGISTER_EVAL_FUNC(kAdd, ascir_v1::AddApi);
REGISTER_EVAL_FUNC(kBroadcast, ascir_v1::BroadcastApi);
REGISTER_EVAL_FUNC(kCast, ascir_v1::CastApi);
REGISTER_EVAL_FUNC(kUb2ub, ascir_v1::CopyUbtoUbApi);
REGISTER_EVAL_FUNC(kDiv, ascir_v1::DivApi);
REGISTER_EVAL_FUNC(kErf, ascir_v1::ErfApi);
REGISTER_EVAL_FUNC(kExp, ascir_v1::ExpApi);
REGISTER_EVAL_FUNC(kGather, ascir_v1::GatherApi);
REGISTER_EVAL_FUNC(kLogicalAnd, ascir_v1::LogicalAndApi);
REGISTER_EVAL_FUNC(kLogicalOr, ascir_v1::LogicalOrApi);
REGISTER_EVAL_FUNC(kLogicalNot, ascir_v1::LogicalNotApi);
REGISTER_EVAL_FUNC(kMaximum, ascir_v1::MaxApi);
REGISTER_EVAL_FUNC(kMinimum, ascir_v1::MinApi);
REGISTER_EVAL_FUNC(kReciprocal, ascir_v1::ReciprocalApi);
REGISTER_EVAL_FUNC(kRelu, ascir_v1::ReluApi);
REGISTER_EVAL_FUNC(kReduceAll, ascir_v1::ReduceAllApi);
REGISTER_EVAL_FUNC(kReduceAny, ascir_v1::ReduceAnyApi);
REGISTER_EVAL_FUNC(kReduceMax, ascir_v1::ReduceMaxApi);
REGISTER_EVAL_FUNC(kArgmax, ascir_v1::ArgmaxApi);
REGISTER_EVAL_FUNC(kArgMaxMultiRPhase1, ascir_v1::ArgMaxMultiRPhase1Api);
REGISTER_EVAL_FUNC(kArgMaxMultiRPhase2, ascir_v1::ArgMaxMultiRPhase2Api);
REGISTER_EVAL_FUNC(kReduceMean, ascir_v1::ReduceMeanApi);
REGISTER_EVAL_FUNC(kReduceMin, ascir_v1::ReduceMinApi);
REGISTER_EVAL_FUNC(kReduceSum, ascir_v1::ReduceSumApi);
REGISTER_EVAL_FUNC(kReduceProd, ascir_v1::ReduceProdApi);
REGISTER_EVAL_FUNC(kRemovePad, ascir_v1::RemovePadApi);
REGISTER_EVAL_FUNC(kRsqrt, ascir_v1::RsqrtApi);
REGISTER_EVAL_FUNC(kSign, ascir_v1::SignApi);
REGISTER_EVAL_FUNC(kSqrt, ascir_v1::SqrtApi);
REGISTER_EVAL_FUNC(kSub, ascir_v1::SubApi);
REGISTER_EVAL_FUNC(kTanh, ascir_v1::TanhApi);
REGISTER_EVAL_FUNC(kSelect, ascir_v1::WhereApi);
REGISTER_EVAL_FUNC(kWhere, ascir_v1::WhereApi);
REGISTER_EVAL_FUNC(kGe, ascir_v1::CompareGeApi);
REGISTER_EVAL_FUNC(kEq, ascir_v1::CompareEqApi);
REGISTER_EVAL_FUNC(kNe, ascir_v1::CompareNeApi);
REGISTER_EVAL_FUNC(kGt, ascir_v1::CompareGtApi);
REGISTER_EVAL_FUNC(kLe, ascir_v1::CompareLeApi);
REGISTER_EVAL_FUNC(kLt, ascir_v1::CompareLtApi);
REGISTER_EVAL_FUNC(kZerosLike, ascir_v1::ZerosLikeApi);
// Cube
REGISTER_EVAL_FUNC(kFlashSoftmax, ascir_v1::SoftmaxFlashV2);
namespace {
PerfParamTableV1 perf_param_table_v1;
TilingScheduleConfigTableV1 tiling_schedule_config_table_v1;
TilingScheduleConfigTableV1HeavyOp tiling_schedule_config_table_v1_heavy_op;
ApiPerfRegister<ApiPerf> add_api_perf(kAdd, kAdd, nullptr, &perf_param_table_v1, &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> gather_api_perf(kGather, kGather, nullptr, &perf_param_table_v1,
                                         &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> abs_api_perf(kAbs, kAbs, nullptr, &perf_param_table_v1, &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> broadcast_api_perf(kBroadcast, kBroadcast, nullptr, &perf_param_table_v1,
                                            &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> cast_api_perf(kCast, kCast, nullptr, &perf_param_table_v1, &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> div_api_perf(kDiv, kDiv, nullptr, &perf_param_table_v1, &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> erf_api_perf(kErf, kErf, nullptr, &perf_param_table_v1, &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> exp_api_perf(kExp, kExp, nullptr, &perf_param_table_v1, &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> logical_and_api_perf(kLogicalAnd, kLogicalAnd, nullptr, &perf_param_table_v1,
                                              &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> logical_or_api_perf(kLogicalOr, kLogicalOr, nullptr, &perf_param_table_v1,
                                             &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> logical_not_api_perf(kLogicalNot, kLogicalNot, nullptr, &perf_param_table_v1,
                                              &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> maximum_api_perf(kMaximum, kMaximum, nullptr, &perf_param_table_v1,
                                          &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> minimum_api_perf(kMinimum, kMinimum, nullptr, &perf_param_table_v1,
                                          &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> min_api_perf(kMin, kMin, nullptr, &perf_param_table_v1,
                                      &tiling_schedule_config_table_v1_heavy_op);
ApiPerfRegister<ApiPerf> mul_api_perf(kMul, kMul, nullptr, &perf_param_table_v1, &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> neg_api_perf(kNeg, kNeg, nullptr, &perf_param_table_v1, &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> reciprocal_api_perf(kReciprocal, kReciprocal, nullptr, &perf_param_table_v1,
                                             &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> relu_api_perf(kRelu, kRelu, nullptr, &perf_param_table_v1, &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> remove_pad_api_perf(kRemovePad, kRemovePad, nullptr, &perf_param_table_v1,
                                             &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> rsqrt_api_perf(kRsqrt, kRsqrt, nullptr, &perf_param_table_v1,
                                        &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> sign_api_perf(kSign, kSign, nullptr, &perf_param_table_v1, &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> sqrt_api_perf(kSqrt, kSqrt, nullptr, &perf_param_table_v1, &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> sub_api_perf(kSub, kSub, nullptr, &perf_param_table_v1, &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> tanh_api_perf(kTanh, kTanh, nullptr, &perf_param_table_v1, &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> where_api_perf(kWhere, kWhere, nullptr, &perf_param_table_v1,
                                        &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> select_api_perf(kSelect, kWhere, nullptr, &perf_param_table_v1,
                                         &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> ge_api_perf(kGe, kGe, nullptr, &perf_param_table_v1, &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> eq_api_perf(kEq, kEq, nullptr, &perf_param_table_v1, &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> ne_api_perf(kNe, kNe, nullptr, &perf_param_table_v1, &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> gt_api_perf(kGt, kGt, nullptr, &perf_param_table_v1, &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> le_api_perf(kLe, kLe, nullptr, &perf_param_table_v1, &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> lt_api_perf(kLt, kLt, nullptr, &perf_param_table_v1, &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> ub2ub_api_perf(kUb2ub, kUb2ub, nullptr, &perf_param_table_v1,
                                        &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> load_api_perf(kLoad, kLoad, nullptr, &perf_param_table_v1, &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> store_api_perf(kStore, kStore, nullptr, &perf_param_table_v1,
                                        &tiling_schedule_config_table_v1);

ApiPerfRegister<ApiPerf> reduce_all_api_perf(kAll, kUnitVector, nullptr, &perf_param_table_v1,
                                             &tiling_schedule_config_table_v1_heavy_op);
ApiPerfRegister<ApiPerf> reduce_any_api_perf(kAny, kUnitVector, nullptr, &perf_param_table_v1,
                                             &tiling_schedule_config_table_v1_heavy_op);
ApiPerfRegister<ApiPerf> reduce_max_api_perf(kMax, kMax, nullptr, &perf_param_table_v1,
                                             &tiling_schedule_config_table_v1_heavy_op);
ApiPerfRegister<ApiPerf> reduce_mean_api_perf(kMean, kUnitVector, nullptr, &perf_param_table_v1,
                                              &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> reduce_min_api_perf(kMin, kMin, nullptr, &perf_param_table_v1,
                                             &tiling_schedule_config_table_v1_heavy_op);
ApiPerfRegister<ApiPerf> reduce_prod_api_perf(kProd, kUnitVector, nullptr, &perf_param_table_v1,
                                              &tiling_schedule_config_table_v1_heavy_op);
ApiPerfRegister<ApiPerf> reduce_sum_api_perf(kSum, kSum, nullptr, &perf_param_table_v1,
                                             &tiling_schedule_config_table_v1_heavy_op);
// 不需要建模的ASCIR
ApiPerfRegister<ApiPerf> data_api_perf(kData, DefaultGetPerf, nullptr, &perf_param_table_v1,
                                       &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> scalar_api_perf(kScalar, DefaultGetPerf, nullptr, &perf_param_table_v1,
                                         &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> index_expr_api_perf(kIndexExpr, DefaultGetPerf, nullptr, &perf_param_table_v1,
                                             &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> output_api_perf(kOutput, DefaultGetPerf, nullptr, &perf_param_table_v1,
                                         &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> workspace_api_perf(kWorkspace, DefaultGetPerf, nullptr, &perf_param_table_v1,
                                            &tiling_schedule_config_table_v1);
// 目前无建模的ASCIR
ApiPerfRegister<ApiPerf> pad_api_perf(kPad, kUnitVector, nullptr, &perf_param_table_v1,
                                      &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> nop_api_perf(kNop, kUnitVector, nullptr, &perf_param_table_v1,
                                      &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> ln_api_perf(kLn, kUnitVector, nullptr, &perf_param_table_v1, &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> isnan_api_perf(kIsnan, kUnitVector, nullptr, &perf_param_table_v1,
                                        &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> isfinite_api_perf(kIsFinite, kUnitVector, nullptr, &perf_param_table_v1,
                                           &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> max_api_perf(kMax, kUnitVector, nullptr, &perf_param_table_v1,
                                      &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> mean_api_perf(kMean, kUnitVector, nullptr, &perf_param_table_v1,
                                       &tiling_schedule_config_table_v1_heavy_op);
ApiPerfRegister<ApiPerf> prod_api_perf(kProd, kUnitVector, nullptr, &perf_param_table_v1,
                                       &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> any_api_perf(kAny, kUnitVector, nullptr, &perf_param_table_v1,
                                      &tiling_schedule_config_table_v1_heavy_op);
ApiPerfRegister<ApiPerf> all_api_perf(kAll, kUnitVector, nullptr, &perf_param_table_v1,
                                      &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> sigmoid_api_perf(kSigmoid, kUnitVector, nullptr, &perf_param_table_v1,
                                          &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> true_div_api_perf(kTrueDiv, kUnitVector, nullptr, &perf_param_table_v1,
                                           &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> pow_api_perf(kPow, kUnitVector, nullptr, &perf_param_table_v1,
                                      &tiling_schedule_config_table_v1_heavy_op);
ApiPerfRegister<ApiPerf> clip_by_value_api_perf(kClipByValue, kUnitVector, nullptr, &perf_param_table_v1,
                                                &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> concat_api_perf(kConcat, kUnitVector, nullptr, &perf_param_table_v1,
                                         &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> leaky_relu_api_perf(kLeakyRelu, kUnitVector, nullptr, &perf_param_table_v1,
                                             &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> bitwise_and_api_perf(kBitwiseAnd, kUnitVector, nullptr, &perf_param_table_v1,
                                              &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> transpose_api_perf(kTranspose, kUnitVector, nullptr, &perf_param_table_v1,
                                            &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> floor_div_api_perf(kFloorDiv, kUnitVector, nullptr, &perf_param_table_v1,
                                            &tiling_schedule_config_table_v1);
ApiPerfRegister<ApiPerf> gelu_api_perf(kGelu, kUnitVector, nullptr, &perf_param_table_v1,
                                       &tiling_schedule_config_table_v1);
}  // namespace
}  // namespace att
