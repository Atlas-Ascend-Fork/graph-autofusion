/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AUTOFUSE_REG_ARG_REDUCE_API_CALL_H__
#define __AUTOFUSE_REG_ARG_REDUCE_API_CALL_H__
#include "codegen_kernel.h"

namespace codegen {
// Argmin/Argmax share geometry and result materialization, independently of the comparison direction.
// They cannot reuse RegReduceApiCall:
// 1) RegReduceApiCall 的模板参数为 <dtype(y), pattern, is_reuse_source>，
//    Argmax 系需 <value_dtype(x), pattern>，内部 index 固定为 int32，出口按需转换；
// 2) RegReduceApiCall 的 multi_reduce 分支（tmp_reduce + instr_type 后处理）与
//    ArgMaxExtend 的单调用归约语义冲突。
// 本文件的 ApiCall 按普通/WithValue 输出签名以及 Phase1/Phase2 职责划分。
class RegArgReduceApiCall : public ApiCall {
 public:
  using ApiCall::Generate;
  explicit RegArgReduceApiCall(const std::string &api_name) : ApiCall(api_name) {}
  ~RegArgReduceApiCall() final = default;
  Status BuildApiParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                       const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                       const std::vector<std::reference_wrapper<const Tensor>> &outputs) const override;
  Status GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler, std::stringstream &ss) const override;

 private:
  Status BuildSingleReduceParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis, const Tensor &y,
                                bool need_cast, CodegenApiParam &api_param) const;
  Status BuildMultiReduceParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis, const Tensor &x,
                               const Tensor &y, const std::string &value_dtype, bool need_cast,
                               CodegenApiParam &api_param) const;
  Status BuildMultiReducePreProcess(const TPipe &tpipe, const std::string &value_dtype, const std::string &loop_axis,
                                    CodegenApiParam &api_param) const;
  // 实际有效 R 长度表达式（last_actual / first_actual），用于屏蔽 tail padding lane。
  mutable std::string valid_r_expr_{"0"};
  mutable bool has_valid_a_{false};
};

class RegArgReducePhase1ApiCall : public ApiCall {
 public:
  using ApiCall::Generate;
  explicit RegArgReducePhase1ApiCall(const std::string &api_name) : ApiCall(api_name) {}
  ~RegArgReducePhase1ApiCall() final = default;
  Status BuildApiParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                       const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                       const std::vector<std::reference_wrapper<const Tensor>> &outputs) const override;
  Status GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler, std::stringstream &ss) const override;

 private:
  Status BuildTileState(const TPipe &tpipe, const std::string &value_dtype, const std::string &index_base,
                        const std::string &loop_axis, CodegenApiParam &api_param) const;
  // 实际有效 R 长度表达式（last_actual / first_actual），用于屏蔽 tail padding lane。
  mutable std::string valid_r_expr_{"0"};
  mutable bool has_valid_a_{false};
};

class RegArgReducePhase2ApiCall : public ApiCall {
 public:
  using ApiCall::Generate;
  explicit RegArgReducePhase2ApiCall(const std::string &api_name) : ApiCall(api_name) {}
  ~RegArgReducePhase2ApiCall() final = default;
  Status BuildApiParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                       const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                       const std::vector<std::reference_wrapper<const Tensor>> &outputs) const override;
  Status GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler, std::stringstream &ss) const override;

 private:
  // A 轴实际有效长度表达式，用于 Phase2 只处理真实列、不碰 padding。
  mutable std::string valid_a_expr_{"0"};
};

// 前端可调用的 ArgmaxWithValue：x(T) -> value(T) + index(U)，1 入 2 出。
// 与 RegArgMaxPhase1ApiCall 签名相同，但作为独立前端算子存在（ArgMaxMultiRPhase1 仅
// 供 schedule R 分核内部使用，不开放给前端）；Phase1 使用带历史累加的 Reg 实现。
class RegArgReduceWithValueApiCall : public ApiCall {
 public:
  using ApiCall::Generate;
  explicit RegArgReduceWithValueApiCall(const std::string &api_name) : ApiCall(api_name) {}
  ~RegArgReduceWithValueApiCall() final = default;
  Status BuildApiParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                       const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                       const std::vector<std::reference_wrapper<const Tensor>> &outputs) const override;
  Status GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler, std::stringstream &ss) const override;

 private:
  Status BuildSingleReduceParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                const Tensor &y_value, const Tensor &y_index, bool need_cast,
                                CodegenApiParam &api_param) const;
  Status BuildMultiReduceParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis, const Tensor &x,
                               const Tensor &y_value, const Tensor &y_index, const std::string &value_dtype,
                               bool need_cast, CodegenApiParam &api_param) const;
  Status BuildMultiReducePreProcess(const TPipe &tpipe, const std::string &value_dtype, const std::string &loop_axis,
                                    CodegenApiParam &api_param) const;
  // BuildApiParam 中判定是否需要 R 轴多 tile 合并，供 GenDimensionParam 传递 accumulated_offset。
  mutable bool multi_reduce_{false};
  // 实际有效 R 长度表达式（last_actual / first_actual），用于屏蔽 tail padding lane。
  mutable std::string valid_r_expr_{"0"};
  mutable bool has_valid_a_{false};
};

// ArgmaxWithValue R 分核 Phase2（WithValue 变体）：value(T)+index(U) -> value(T)+index(U)，
// 2 入 2 出。一次归约（复用 ArgMaxMultiRPhase2VectorImpl 向量路径）同时输出全局 value 与全局 index，
// 无二次归约。与 RegArgMaxPhase2ApiCall 的差异仅在输出个数（2 出）与 regbase 函数名。
class RegArgReducePhase2WithValueApiCall : public ApiCall {
 public:
  using ApiCall::Generate;
  explicit RegArgReducePhase2WithValueApiCall(const std::string &api_name) : ApiCall(api_name) {}
  ~RegArgReducePhase2WithValueApiCall() final = default;
  Status BuildApiParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                       const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                       const std::vector<std::reference_wrapper<const Tensor>> &outputs) const override;
  Status GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler, std::stringstream &ss) const override;

 private:
  // A 轴实际有效长度表达式，用于 Phase2 只处理真实列、不碰 padding。
  mutable std::string valid_a_expr_{"0"};
};
// Factory identities stay distinct; implementation is shared without changing existing include/type names.
using RegArgmaxApiCall = RegArgReduceApiCall;
using RegArgminApiCall = RegArgReduceApiCall;
using RegArgMaxPhase1ApiCall = RegArgReducePhase1ApiCall;
using RegArgMinPhase1ApiCall = RegArgReducePhase1ApiCall;
using RegArgMaxPhase2ApiCall = RegArgReducePhase2ApiCall;
using RegArgMinPhase2ApiCall = RegArgReducePhase2ApiCall;
using RegArgmaxWithValueApiCall = RegArgReduceWithValueApiCall;
using RegArgminWithValueApiCall = RegArgReduceWithValueApiCall;
using RegArgMaxPhase2WithValueApiCall = RegArgReducePhase2WithValueApiCall;
using RegArgMinPhase2WithValueApiCall = RegArgReducePhase2WithValueApiCall;
}  // namespace codegen
#endif  // __AUTOFUSE_REG_ARG_REDUCE_API_CALL_H__
