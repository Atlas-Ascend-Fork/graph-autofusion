/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AUTOFUSE_REG_ARGMAX_API_CALL_H__
#define __AUTOFUSE_REG_ARGMAX_API_CALL_H__
#include "codegen_kernel.h"

namespace codegen {
// Argmax 系列不能复用 RegReduceApiCall：
// 1) RegReduceApiCall 的模板参数为 <dtype(y), pattern, is_reuse_source>，
//    Argmax 系需 <index_dtype, value_dtype, pattern>（两个 dtype，来源 x/y 不同）；
// 2) RegReduceApiCall 的 multi_reduce 分支（tmp_reduce + instr_type 后处理）与
//    ArgMaxExtend 的单调用归约语义冲突。
// 本文件三个 ApiCall 按输入输出签名划分职责：单算子 1入1出、Phase1 1入2出、Phase2 2入1出。
class RegArgmaxApiCall : public ApiCall {
 public:
  using ApiCall::Generate;
  explicit RegArgmaxApiCall(const std::string &api_name) : ApiCall(api_name) {}
  ~RegArgmaxApiCall() final = default;
  Status BuildApiParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                       const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                       const std::vector<std::reference_wrapper<const Tensor>> &outputs) const override;
  Status GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler, std::stringstream &ss) const override;

 private:
  // 实际有效 R 长度表达式（last_actual / first_actual），用于屏蔽 tail padding lane。
  mutable std::string valid_r_expr_{"0"};
};

class RegArgMaxPhase1ApiCall : public ApiCall {
 public:
  using ApiCall::Generate;
  explicit RegArgMaxPhase1ApiCall(const std::string &api_name) : ApiCall(api_name) {}
  ~RegArgMaxPhase1ApiCall() final = default;
  Status BuildApiParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                       const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                       const std::vector<std::reference_wrapper<const Tensor>> &outputs) const override;
  Status GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler, std::stringstream &ss) const override;

 private:
  // 实际有效 R 长度表达式（last_actual / first_actual），用于屏蔽 tail padding lane。
  mutable std::string valid_r_expr_{"0"};
};

class RegArgMaxPhase2ApiCall : public ApiCall {
 public:
  using ApiCall::Generate;
  explicit RegArgMaxPhase2ApiCall(const std::string &api_name) : ApiCall(api_name) {}
  ~RegArgMaxPhase2ApiCall() final = default;
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
// 供 schedule R 分核内部使用，不开放给前端）；两者均调用 ArgMaxWithValueExtend Reg 实现。
class RegArgmaxWithValueApiCall : public ApiCall {
 public:
  using ApiCall::Generate;
  explicit RegArgmaxWithValueApiCall(const std::string &api_name) : ApiCall(api_name) {}
  ~RegArgmaxWithValueApiCall() final = default;
  Status BuildApiParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                       const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                       const std::vector<std::reference_wrapper<const Tensor>> &outputs) const override;
  Status GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler, std::stringstream &ss) const override;

 private:
  // BuildApiParam 中判定是否需要 R 轴多 tile 合并，供 GenDimensionParam 传递 accumulated_offset。
  mutable bool multi_reduce_{false};
  // 实际有效 R 长度表达式（last_actual / first_actual），用于屏蔽 tail padding lane。
  mutable std::string valid_r_expr_{"0"};
};

// ArgmaxWithValue R 分核 Phase2（WithValue 变体）：value(T)+index(U) -> value(T)+index(U)，
// 2 入 2 出。一次归约（复用 ArgMaxMultiRPhase2VectorImpl 向量路径）同时输出全局 value 与全局 index，
// 无二次归约。与 RegArgMaxPhase2ApiCall 的差异仅在输出个数（2 出）与 regbase 函数名。
class RegArgMaxPhase2WithValueApiCall : public ApiCall {
 public:
  using ApiCall::Generate;
  explicit RegArgMaxPhase2WithValueApiCall(const std::string &api_name) : ApiCall(api_name) {}
  ~RegArgMaxPhase2WithValueApiCall() final = default;
  Status BuildApiParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                       const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                       const std::vector<std::reference_wrapper<const Tensor>> &outputs) const override;
  Status GenDimensionParam(const CodegenApiParam &api_param, const Tiler &tiler, std::stringstream &ss) const override;

 private:
  // A 轴实际有效长度表达式，用于 Phase2 只处理真实列、不碰 padding。
  mutable std::string valid_a_expr_{"0"};
};
}  // namespace codegen
#endif  // __AUTOFUSE_REG_ARGMAX_API_CALL_H__
