/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef API_PERF_REGISTER_API_PERF_H_
#define API_PERF_REGISTER_API_PERF_H_

#include <functional>
#include <optional>
#include <utility>
#include "ascir_node_param/ascir_node_param.h"
#include "base/perf_breakdown.h"
#include "perf_param.h"

namespace att {
// Codegen DataCopyNddma 的 effective view。向量已按 vectorized_axis 排列，并应与
// CalculateDmaParams 合轴、忽略双零 stride 轴及 tail actual size 后的参数一致；dim 和 stride 的单位
// 分别为元素个数和元素，input_strides 对应 GM stride，output_strides 对应 UB vectorized stride。
struct NddmaDescriptorInfo {
  std::vector<Expr> output_dims;
  std::vector<Expr> input_strides;
  std::vector<Expr> output_strides;
  std::vector<int64_t> vectorized_axis;
};

struct NodeDetail {
  std::string name;
  std::string optype;
  std::vector<std::string> input_dtype{};
  std::vector<std::string> output_dtype{};
  std::vector<Expr> repeats{};           // 合轴前
  std::vector<Expr> origin_repeats{};    // 原始图 shape，未经过 tail 切分，用于整张量规模判断
  std::vector<Expr> vectorized_shape{};  // codegen实际使用的vectorized shape
  std::vector<Expr> input_dims{};        // 合轴后
  std::vector<Expr> output_dims{};       // 合轴后
  Expr gm_stride{CreateExpr(0)};
  Expr ub_stride{CreateExpr(0)};
  ascir_param::CastNodeParams cast_node_params;
  ascir_param::BroadcastNodeParams broadcast_node_params;
  ascir_param::CompareNodeParams compare_node_params;
  ascir_param::WhereNodeParams where_node_params;
  ascir_param::UnaryBitWidthChangeNodeParams unary_bitwidth_change_node_params;
  ascir_param::TransposeNodeParams transpose_node_params;
  int32_t block_count_idx{0};                      // 用于 LoadStoreStrideV2Func，表示发生非连续的轴索引
  std::map<Expr, TernaryOp, ExprCmp> ternary_ops;  // 动态shape表达式
  std::optional<NddmaDescriptorInfo> nddma_descriptor;

  std::string ToString() const {
    std::string res = name + "[" + optype + "]";
    res += ", input_dtype[";
    if (!input_dtype.empty()) {
      res += input_dtype[0];
    }
    res += "], output_dtype[";
    if (!output_dtype.empty()) {
      res += output_dtype[0];
    }
    res += "], dim_info[";
    res += GetVecString(input_dims);
    res += "], gm_stride[" + std::string(gm_stride.Str().get());
    res += "], ub_stride[" + std::string(ub_stride.Str().get()) + "]";
    return res;
  }
};
using TernaryOpMap = std::map<Expr, TernaryOp, ExprCmp>;
struct PerfOutputInfo {
  std::map<PipeType, Expr> pipe_res;
  TernaryOpMap ternary_ops;
  std::vector<PerfBreakdownGroup> perf_breakdowns;
  vector<CacheLineConfig> *cache_line_config{nullptr};
  std::string ToString() {
    std::string res;
    auto replace_vars = ConcursiveReplaceVars(ternary_ops);
    for (const auto &pair : pipe_res) {
      res += PipeType2Str.at(pair.first) + ":" + Str(pair.second.Replace(replace_vars)) + ",";
    }
    return res;
  }
};
using MicroPerfFunc = af::Status (*)(const std::vector<NodePerfInfo> &node_perf_infos, Expr &res);
using Perf = af::Status (*)(const std::vector<TensorShapeInfo> &input_shapes,
                            const std::vector<TensorShapeInfo> &output_shapes, const NodeInfo &node,
                            PerfOutputInfo &res);
using AscendCPerf = af::Status (*)(const NodeDetail &node_info, PerfOutputInfo &perf);
class ApiPerf {
 public:
  ApiPerf(const std::string &api_name, Perf perf_func, MicroPerfFunc micro_perf_func, const PerfParamTable *perf_param,
          const TilingScheduleConfigTable *tiling_schedule_config_table)
      : api_perf_func_(perf_func),
        micro_perf_func_(micro_perf_func),
        perf_param_(perf_param),
        tiling_schedule_config_table_(tiling_schedule_config_table),
        api_name_(api_name) {}
  virtual ~ApiPerf() = default;
  const std::string &GetApiName() const {
    return api_name_;
  }
  Perf GetPerfFunc() const {
    return api_perf_func_;
  }
  MicroPerfFunc GetMicroPerfFunc() const {
    return micro_perf_func_;
  }
  const PerfParamTable *GetPerfParam() const {
    return perf_param_;
  }
  const TilingScheduleConfigTable *GetTilingScheduleConfigTable() const {
    return tiling_schedule_config_table_;
  }

 private:
  Perf api_perf_func_;
  MicroPerfFunc micro_perf_func_;
  const PerfParamTable *perf_param_;
  const TilingScheduleConfigTable *tiling_schedule_config_table_{nullptr};
  std::string api_name_;
};

inline af::Status DefaultGetPerf([[maybe_unused]] const std::vector<TensorShapeInfo> &input_shapes,
                                 [[maybe_unused]] const std::vector<TensorShapeInfo> &output_shapes,
                                 [[maybe_unused]] const NodeInfo &node, [[maybe_unused]] PerfOutputInfo &res) {
  (void)res;
  return af::SUCCESS;
}
}  // namespace att
#endif
