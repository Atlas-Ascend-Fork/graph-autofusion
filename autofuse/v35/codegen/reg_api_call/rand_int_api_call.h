/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __AUTOFUSE_RAND_INT_API_CALL_H__
#define __AUTOFUSE_RAND_INT_API_CALL_H__
#include "codegen_kernel.h"

namespace codegen {
// RandInt 为四输入算子：x1 seed/x2 counter/x3 low/x4 high，其中 x2 恒为逐元素位置流（tensor），
// 其余输入可能为编译期常量或 ub_scalar 标量场景，按输入独立选择重载
class RandIntApiCall final : public ApiCall {
 public:
  using ApiCall::Generate;
  explicit RandIntApiCall(const std::string &api_name) : ApiCall(api_name) {}
  ~RandIntApiCall() final = default;
  Status Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                  const std::vector<std::reference_wrapper<const Tensor>> &outputs, std::string &result) const override;
};
}  // namespace codegen
#endif  // __AUTOFUSE_RAND_INT_API_CALL_H__
