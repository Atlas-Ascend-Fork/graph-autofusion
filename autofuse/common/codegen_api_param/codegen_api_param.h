/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __CODEGEN_API_PARAM_H__
#define __CODEGEN_API_PARAM_H__

#include <cstddef>
#include <cstdint>
#include <variant>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include "ge_common_af/ge_api_types_af.h"
#include "ascir.h"
#include "expression_struct.h"

namespace codegen {

class Tiler;

const char *const kCodegenApiParam = "CodegenApiParam";

struct MergeAxesInfo {
  // 有效性标志：表示是否有merge axes信息
  bool valid = false;

  // 这些是原始 SizeExpr，统一用 ActualSize 转换
  std::vector<ge::Expression> repeats;
  std::vector<ge::Expression> gm_strides;
  std::vector<ge::Expression> ub_strides;
};

struct DataCopyBaseParams {
  // 有效性标志：表示是否有本层的结果
  bool valid = false;

  // block_count/block_len: 简单表达式，用 ActualSize
  // src_stride/dst_stride: 可能是组合表达式（Size - ActualSize）
  CombinedExpression block_count;
  CombinedExpression block_len;
  CombinedExpression src_stride;
  CombinedExpression dst_stride;
};

struct DataCopyLoopModeParams {
  // 有效性标志：表示是否有loop mode参数
  bool valid = false;

  // loop_sizes: 用 ActualSize + cast
  // loop_strides: 用 Size * dtype_size + cast
  std::vector<CombinedExpression> loop_sizes;
  std::vector<CombinedExpression> loop_src_strides;
  std::vector<CombinedExpression> loop_dst_strides;
};

struct DmaSpecificParams {
  MergeAxesInfo merge_axes_info;
  DataCopyBaseParams data_copy_params;
  DataCopyLoopModeParams loop_mode_params;
};
enum class ReducePattern {
  kUnknown,
  kAR,
  kRA,
};

enum class ReduceMergeMode {
  kUnknown,
  kNone,
  kCopy,
  kMergeByElementwise,
};

struct ReduceMergedDims {
  bool valid{false};
  ge::Expression first{ge::Symbol(1U)};
  ge::Expression last{ge::Symbol(1U)};
};

struct ReduceReuseInfo {
  bool valid{false};
  bool is_reuse_source{false};
};

struct ReduceSpecificParams {
  bool valid{false};
  std::string reduce_type;
  ReducePattern pattern{ReducePattern::kUnknown};
  ReduceMergedDims merged_dims;
  ReduceMergeMode merge_mode{ReduceMergeMode::kUnknown};
  ge::Expression merge_size{ge::Symbol(0U)};
  ge::Expression merge_times{ge::Symbol(1U)};
  ReduceReuseInfo reuse;
};

struct ReduceSpecificParamBuildInput {
  std::string node_name;
  std::string reduce_type;
  std::vector<ge::Expression> input_repeats;
  std::vector<ge::Expression> input_strides;
  std::vector<ge::Expression> output_dims;
  std::vector<ge::Expression> output_strides;
  uint32_t dtype_size{0U};
  ReducePattern pattern{ReducePattern::kUnknown};
  bool need_multi_reduce{false};
  ge::Expression merge_times{ge::Symbol(1U)};
  ReduceReuseInfo reuse;
};
struct BroadcastSpecificParams {
  std::string broadcast_type;
};
struct TransposeSpecificParams {
  // Transpose api内部处理的vectorized output dims（用 ActualSize）
  std::vector<CombinedExpression> output_dims;
  // Transpose api内部处理的vectorized input strides（用 Size）
  std::vector<CombinedExpression> input_strides;
  // Transpose api内部处理的vectorized output strides（用 Size）
  std::vector<CombinedExpression> output_strides;
};

struct ReduceMergedAxisPlan {
  bool valid{false};
  bool is_all_axis_reduce{false};
  bool align_last_axis{false};
  bool use_last_non_zero_stride{false};
  // 兼容旧ReduceMergedSizeCodeGen：切到last阶段但还没记录到非零src stride时，使用Size(0)。
  bool use_zero_stride{false};
  size_t aligned_axis_index{0U};
  size_t last_non_zero_stride_index{0U};
  std::vector<size_t> first_axis_indices;
  std::vector<size_t> last_axis_indices;
};

struct ReduceMergedShape {
  bool valid{false};
  ge::Expression first{ge::Symbol(1U)};
  ge::Expression last{ge::Symbol(1U)};
};

enum class ReduceMergedAxisAction {
  kSkip,
  kFirstAxis,
  kLastAxis,
  kAlignFirst,
  kAlignLast,
  kLastStride,
};

struct ReduceMergedAxisState {
  bool is_first{true};
  bool is_all_axis_reduce{true};
  bool use_zero_stride{false};
  size_t last_not_one_axis_index{static_cast<size_t>(-1)};
  size_t last_non_zero_stride_index{static_cast<size_t>(-1)};
};

bool IsReduceMergedZeroTail(bool src_stride_zero, bool dst_stride_zero);
bool IsReduceMergedSameSide(bool current_dst_stride_zero, bool last_dst_stride_zero);
ReduceMergedAxisAction UpdateReduceMergedAxisState(bool src_stride_zero, bool dst_stride_zero,
                                                   bool last_not_one_dst_stride_zero, bool is_last_axis,
                                                   size_t axis_index, ReduceMergedAxisState &state);
ReduceMergedAxisPlan BuildReduceMergedAxisPlan(const std::vector<bool> &src_stride_zero,
                                               const std::vector<bool> &dst_stride_zero);
ReduceMergedShape BuildReduceMergedShape(const std::vector<ge::Expression> &src_repeats,
                                         const std::vector<ge::Expression> &src_strides,
                                         const std::vector<ge::Expression> &dst_strides, uint32_t dtype_size);
af::Status BuildReduceSpecificParams(const ReduceSpecificParamBuildInput &input, ReduceSpecificParams &param);

struct CodegenApiParam;
using CodegenApiParamPtr = std::shared_ptr<CodegenApiParam>;
struct CodegenApiParam {
  static af::Status Register(af::AscNodePtr node, CodegenApiParamPtr api_param);
  static CodegenApiParamPtr TryGetNodeApiParam(const af::AscNodePtr &node);
  static CodegenApiParamPtr GetNodeApiParam(af::AscNodePtr node);
  static af::Status DumpGraphApiParams(const ascir::ImplGraph &graph, const Tiler &tiler, const std::string &prefix);
  std::string api_name;
  std::vector<std::string> template_params;
  std::string semantic_feature;

  struct TensorParam {
    TensorParam() = default;
    TensorParam(const std::string &name, bool is_tensor, const CombinedExpression &offset)
        : name(name), is_tensor(is_tensor), offset(offset) {}
    std::string name;
    bool is_tensor = true;
    CombinedExpression offset;
  };

  // outer_loop_axes: 简单表达式，统一用 ActualSize 转换
  std::vector<CombinedExpression> outer_loop_axes;
  std::vector<std::string> api_pre_process;
  std::vector<std::string> api_post_process;

  std::vector<TensorParam> input_params;
  std::vector<TensorParam> output_params;

  std::string tmp_buf_name;
  CombinedExpression cal_count;

  using AnySpecificParams = std::variant<std::monostate, DmaSpecificParams, ReduceSpecificParams,
                                         BroadcastSpecificParams, TransposeSpecificParams>;
  AnySpecificParams specific_params;
};
}  // namespace codegen
#endif
