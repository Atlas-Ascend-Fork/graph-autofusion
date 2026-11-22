/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_CODEGEN_REG_API_CALL_CONCAT_REG_API_CALL_H_
#define ASCGEN_DEV_CODEGEN_REG_API_CALL_CONCAT_REG_API_CALL_H_

#include "codegen_kernel.h"
#include "api_call/concat/concat_api_call.h"

namespace codegen {
class ConcatRegApiCall : public ConcatApiCall {
 public:
  using ApiCall::Generate;
  explicit ConcatRegApiCall(const std::string &api_name) : ConcatApiCall(api_name) {}
  ~ConcatRegApiCall() override = default;
  Status Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                  const std::vector<std::reference_wrapper<const Tensor>> &outputs, std::string &result) const override;
  bool AreContiguousBufsPreferred() const override;

 protected:
  Status ParseAttr(const ascir::NodeView &node) override;
  static af::Status GenerateDefault(const vector<std::reference_wrapper<const Tensor>> &inputs, const Tensor &y,
                                    const ConcatApiCall::ConcatTiling &tiling, const TPipe &t_pipe,
                                    std::stringstream &ss, const int64_t tmp_buf_id);

 private:
  static std::string GetTilingDataType(const ConcatTiling &tiling);
  static Status GenerateForOneAxis(const vector<std::reference_wrapper<const Tensor>> &inputs, const Tensor &y,
                                   std::stringstream &ss);
  static bool CanConcatOneAxis(const std::vector<std::reference_wrapper<const Tensor>> &inputs, const Tensor &y);
  static std::string GetSemanticFeature(const ConcatTiling &tiling);
  static af::Status GenerateForGather(const vector<std::reference_wrapper<const Tensor>> &inputs, const Tensor &y,
                                      const ConcatApiCall::ConcatTiling &tiling, const TPipe &t_pipe,
                                      std::stringstream &ss, const int64_t tmp_buf_id);
  static void DefineConcatTiling(const ConcatTiling &tiling, const Tiler &tiler, std::stringstream &ss);
  static void DefineConcatTilingGather(const ConcatTiling &tiling, const Tiler &tiler, std::stringstream &ss);
  static void GenSrcAddrs(const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                          const std::string &dtype_name, std::stringstream &ss);
  static bool NeedB8ToB16(const ConcatTiling &tiling);
  bool IsShareInputs() const;
  static ConcatTiling B64ToB32(const ConcatTiling &tiling);
  static ConcatTiling B8ToB16(const ConcatTiling &tiling);
  af::Status CanUseGather(ConcatTiling &tiling) const;
  bool IsTile() const;
  static void NormalizeDtype(std::string &dtype_name);

  ascir::NodeView node_ = nullptr;
};
}  // namespace codegen

#endif  // ASCGEN_DEV_CODEGEN_REG_API_CALL_CONCAT_REG_API_CALL_H_
