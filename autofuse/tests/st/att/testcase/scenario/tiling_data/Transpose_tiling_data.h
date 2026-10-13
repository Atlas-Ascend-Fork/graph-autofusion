/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_TILING_DATA_TRANSPOSE_H_
#define ATT_TILING_DATA_TRANSPOSE_H_
#include <stdint.h>
#include <vector>
#include <array>
#include "register/tilingdata_base.h"
#include "kernel_tiling/kernel_tiling.h"
namespace optiling {

BEGIN_TILING_DATA_DEF(graph_normalTilingData)
// definitions of BaseParams
TILING_DATA_FIELD_DEF(uint32_t, z0Tb_size)
TILING_DATA_FIELD_DEF(uint32_t, z0t_size)

// definitions of HardWareParams
TILING_DATA_FIELD_DEF(uint32_t, block_dim)
TILING_DATA_FIELD_DEF(uint32_t, ub_size)
TILING_DATA_FIELD_DEF(uint32_t, workspaceSize)

// definitions of InputParams
TILING_DATA_FIELD_DEF(uint32_t, s0)
TILING_DATA_FIELD_DEF(uint32_t, s1)
TILING_DATA_FIELD_DEF(uint32_t, s2)
TILING_DATA_FIELD_DEF(uint32_t, s3)

// definitions of CoreParams
// 参数：{轴}_tail_size
// 含义：本轴的最后一次循环，需要循环多少次最内轴元素
// 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
// 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
TILING_DATA_FIELD_DEF(uint32_t, z0t_tail_size)
// 参数：{轴}_loop_num
// 含义：本轴的父轴需要循环多少次本轴
// 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
// 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
//                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
TILING_DATA_FIELD_DEF(uint32_t, z0t_loop_num)
// 参数：{轴}_tail_size
// 含义：本轴的最后一次循环，需要循环多少次最内轴元素
// 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成tail_size)
// 计算公式：{轴}_tail_size = ({轴的父轴}_size % {轴}_size) == 0 ? {轴}_size : ({轴的父轴}_size % {轴}_size)
TILING_DATA_FIELD_DEF(uint32_t, z0Tb_tail_size)
// 参数：{轴}_loop_num
// 含义：本轴的父轴需要循环多少次本轴
// 约束：仅切分后的内轴会生成该参数(所以最外轴不会生成loop_num)
// 计算公式：{轴}_loop_num = Ceil({轴的父轴}_size / {轴}_size)
//                        = (({轴的父轴}_size + {轴}_size) - 1) / {轴}_size
TILING_DATA_FIELD_DEF(uint32_t, z0Tb_loop_num)

// definitions of MemoryParams
// 含义：单核输出的gm大小
TILING_DATA_FIELD_DEF(uint32_t, output0_single_core_size)
// 含义：输出的gm大小
TILING_DATA_FIELD_DEF(uint32_t, output0_total_size)
// 含义：Tbuf或者Queue使用的内存大小
// 约束：定义了Tbuf或者Queue则会生成该参数
// 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
TILING_DATA_FIELD_DEF(uint32_t, q3_size)
// 含义：Tbuf或者Queue使用的内存大小
// 约束：定义了Tbuf或者Queue则会生成该参数
// 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
TILING_DATA_FIELD_DEF(uint32_t, q1_size)
// 含义：GM的总大小
TILING_DATA_FIELD_DEF(uint32_t, gm_size)
// 含义：Tbuf或者Queue使用的内存大小
// 约束：定义了Tbuf或者Queue则会生成该参数
// 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
TILING_DATA_FIELD_DEF(uint32_t, q0_size)
// 含义：Tbuf或者Queue使用的内存大小
// 约束：定义了Tbuf或者Queue则会生成该参数
// 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
TILING_DATA_FIELD_DEF(uint32_t, KERNEL_INIT_BUFFER)
// 含义：Tbuf或者Queue使用的内存大小
// 约束：定义了Tbuf或者Queue则会生成该参数
// 计算公式：若该内存被多个算子使用，不考虑内存复用，则大小为各个算子的总和，考虑内存复用则为峰值时刻内存
TILING_DATA_FIELD_DEF(uint32_t, b2_size)

// definitions of TilingKeyParms
TILING_DATA_FIELD_DEF(uint32_t, tiling_key)

// Transpose API Tiling结构体
TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, transpose_tilingData_0)

END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(Transpose, graph_normalTilingData)
using AutofuseTilingData = graph_normalTilingData;

struct AutofuseTilingDataPerf {
  AutofuseTilingData tiling_data;
  double best_perf;
};
typedef long int (*ProfilingCallback)(void *stream, uint64_t workspaceSize, AutofuseTilingData *tiling_data,
                                      double *cost_time);
typedef long int (*ProfilingBatchCallback)(void *stream, uint64_t workspaceSize,
                                           std::vector<AutofuseTilingDataPerf> *profiles);
class PgoConfig {
 public:
  static PgoConfig &Instance() {
    static PgoConfig instance;
    return instance;
  }
  ProfilingCallback single_callback;
  ProfilingBatchCallback batch_callback;
  int32_t pgo_algorithm = 1;
  bool need_change_solver_run = false;
  size_t pgo_threshold_index = 0;
  constexpr static size_t pgo_threshold_list_size = 5;
  std::array<double, pgo_threshold_list_size> pgo_ub_threshold_list{0.2, 0.1, 0, 0.05, 0.1};
  std::array<double, pgo_threshold_list_size> pgo_corenum_threshold_list{0.4, 0.4, 1, 1, 0.8};

 private:
  PgoConfig() = default;
  ~PgoConfig() = default;
  PgoConfig(const PgoConfig &) = delete;
  PgoConfig &operator=(const PgoConfig &) = delete;
};
bool GetTiling(graph_normalTilingData &tiling_data, int32_t tilingCaseId = -1);
}  // namespace optiling
using optiling::AutofuseTilingData;
static uint64_t GetWorkspaceSize(const AutofuseTilingData &tiling_data) {
  return 0;
}
#endif
