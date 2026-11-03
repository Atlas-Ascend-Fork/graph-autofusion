/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCIR_NODE_PARAM_ASCIR_NODE_PARAM_H_
#define ASCIR_NODE_PARAM_ASCIR_NODE_PARAM_H_

#include <cstdint>
#include <memory>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "codegen_api_param/codegen_api_param.h"

namespace ascir_param {
enum class CacheGuardKind { kNone, kOriginBroadcast, kFusedBroadcast };

struct CacheGuardInfo {
  CacheGuardKind kind{CacheGuardKind::kNone};
  ge::Expression guard_axis;
  ge::Expression period;
  ge::Expression block_inner_extent;
  ge::Expression loop_extent;
  bool address_invariant{false};
  // Only Codegen/scheduler expressions with a proven positive range may set this.
  bool positive_range_proven{false};
  bool clip_to_period{true};
};

enum class CacheExecutionMode { kUnknown, kNoGuard, kOriginBroadcast, kFusedBroadcast, kReduceDoubleTile };
enum class ReduceCacheCondition { kA, kR, kControl };

struct CacheLoopInfo {
  int64_t axis_id{-1};
  ge::Expression axis;
  ge::Expression extent;
};

struct CacheExecutionParams {
  CacheExecutionMode mode{CacheExecutionMode::kUnknown};
  int64_t call_loop_axis_id{-1};
  CacheGuardInfo guard;
  // One record per emitted call. Repeated generation replaces the same stage.
  uint32_t stage{0U};
  std::vector<CacheLoopInfo> loops;
  std::map<ReduceCacheCondition, CacheGuardInfo> reduce_guards;
};

enum class ParamBuildStatus {
  // ASCIR 参数构建前的默认状态。
  kUnknown,
  // API 已支持且 specific_params 中包含有效载荷。
  kBuilt,
  // API 暂不通过 ASCIR 参数建模。
  kSkipped,
  // API 已支持，但生成的载荷不可用。
  kInvalid,
};

enum class ParamExprRole {
  // 直接按语义表达式使用。
  kSemantic,
  // 预留给生成代码侧按 tpipe.tiler.Size(expr) 解释。
  kSize,
  // 预留给生成代码侧按 tpipe.tiler.ActualSize(expr) 解释。
  kActualSize,
};

struct ParamExprLeaf {
  // API 参数表达式配方中的一个表达式因子。
  ge::Expression expr;
  // 使用方对表达式因子的解释方式。
  ParamExprRole role{ParamExprRole::kSemantic};
};

struct ParamExprProduct {
  // false 表示该语义配方不可用。
  bool valid{false};
  // 乘法因子列表。当前 Reduce 参数只需要乘法表达式。
  std::vector<ParamExprLeaf> factors;
};

struct ReduceParamExprs {
  // Reduce merge_size 的语义表达式，必须与 canonical_params.merge_size 描述同一参数。
  ParamExprProduct merge_size;
  // Reduce merge_times 的语义表达式，必须与 canonical_params.merge_times 描述同一参数。
  ParamExprProduct merge_times;
};

struct ReduceNodeParams {
  // ASCIR 使用方共用的生成代码兼容规范参数，不表示该字段只归生成代码使用。
  codegen::ReduceSpecificParams canonical_params;
  // 规范参数丢失表达式语义时使用的语义配方。
  ReduceParamExprs exprs;
};

struct VectorFuncNodeParams {
  bool is_double_loop{false};
  std::vector<ge::Expression> all_strides;
  std::vector<ge::Expression> output_dims;
};

struct CastNodeParams {
  bool valid{false};
  std::vector<ge::Expression> output_dims;
  std::vector<ge::Expression> output_strides;
  std::vector<ge::Expression> input_strides;
};

struct BroadcastNodeParams {
  bool valid{false};
  bool is_scalar{false};
  int32_t const_rank{-1};
  ge::Expression duplicate_count{ge::Symbol(1U)};
  ParamExprRole duplicate_count_role{ParamExprRole::kSemantic};
  std::vector<ParamExprLeaf> dst_shape;
  std::vector<ParamExprLeaf> src_shape;
};

struct CompareNodeParams {
  bool valid{false};
  bool is_scalar{false};
  ge::Expression outer_call_count{ge::Symbol(1U)};
  std::vector<ge::Expression> output_dims;
  std::vector<ge::Expression> output_strides;
  std::vector<ge::Expression> input_strides;
};

struct WhereNodeParams {
  bool valid{false};
  bool is_bcast_src0{false};
  bool is_bcast_src1{false};
  ge::Expression outer_call_count{ge::Symbol(1U)};
  std::vector<ge::Expression> output_dims;
  std::vector<ge::Expression> output_strides;
  std::vector<ge::Expression> mask_strides;
  std::vector<ge::Expression> input_strides;
};

struct UnaryBitWidthChangeNodeParams {
  bool valid{false};
  ge::Expression cal_count{ge::Symbol(1U)};
  std::vector<ge::Expression> outer_repeats;
  std::vector<ge::Expression> output_strides;
  std::vector<ge::Expression> input_strides;
};

struct TransposeNodeParams {
  bool valid{false};
  uint32_t inner_dim{0U};
  uint32_t total_dim{0U};
  std::vector<ge::Expression> outer_loop_axes;
  std::vector<ge::Expression> output_dims;
  std::vector<ge::Expression> input_strides;
  std::vector<ge::Expression> output_strides;
};

using AnySpecificParams =
    std::variant<std::monostate, ReduceNodeParams, VectorFuncNodeParams, CastNodeParams, BroadcastNodeParams,
                 CompareNodeParams, WhereNodeParams, UnaryBitWidthChangeNodeParams, TransposeNodeParams>;

struct AscirNodeParams {
  // 扩展属性载荷版本，用于后续兼容。
  uint32_t version{1U};
  // 生成该节点的 ASCIR API 名称。
  std::string api_name;
  // 构建结果。使用方仅应在 kBuilt 时读取 specific_params。
  ParamBuildStatus status{ParamBuildStatus::kUnknown};
  // API 专用参数，跳过构建或载荷为空时使用 std::monostate。
  AnySpecificParams specific_params;
  // Final call-site decision; independent of API-specific build status.
  std::vector<CacheExecutionParams> cache_executions;
};

using AscirNodeParamsPtr = std::shared_ptr<AscirNodeParams>;

AscirNodeParamsPtr GetAscirNodeParams(af::AscNodePtr node);
AscirNodeParamsPtr GetOrCreateAscirNodeParams(af::AscNodePtr node);
void RecordCacheExecution(AscirNodeParams &params, const CacheExecutionParams &execution);
const codegen::ReduceSpecificParams &GetCanonicalReduceParams(const ReduceNodeParams &params);
ge::Expression ResolveForAtt(const ParamExprProduct &expr);
af::Status ValidateReduceNodeParams(const ReduceNodeParams &params);

template <typename T>
const T *GetSpecificParams(const AscirNodeParams &params) {
  return std::get_if<T>(&params.specific_params);
}
}  // namespace ascir_param

#endif  // ASCIR_NODE_PARAM_ASCIR_NODE_PARAM_H_
