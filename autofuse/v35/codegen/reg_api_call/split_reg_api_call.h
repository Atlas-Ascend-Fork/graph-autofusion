/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_CODEGEN_REG_API_CALL_SPLIT_REG_API_CALL_H_
#define ASCGEN_DEV_CODEGEN_REG_API_CALL_SPLIT_REG_API_CALL_H_

#include "codegen_kernel.h"

namespace codegen {
class SplitRegApiCall : public ApiCall {
 public:
  using ApiCall::Generate;
  explicit SplitRegApiCall(const std::string &api_name) : ApiCall(api_name) {}
  ~SplitRegApiCall() override = default;
  Status Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                  const std::vector<std::reference_wrapper<const Tensor>> &outputs, std::string &result) const override;

 protected:
  struct SplitTiling {
    uint32_t gcd = 1U;
    uint32_t tmp_buf_size = 0U;
    uint32_t dst_col_size = 0U;
    uint32_t dst_row_num_unit = 0U;
    uint32_t max_repeat_times = 0U;
    uint32_t max_element_num = 0U;
    uint32_t max_orig_row_num = 0U;
    uint32_t per_repeat_size = 0U;
    uint32_t first_copy_repeat_times = 0U;  // for diff dim
    uint32_t last_trans_repeat_times = 0U;  // for diff dim
    bool any_padded = false;
    bool all_static = true;
    af::Expression src_col_size_expr;
    af::Expression src_col_actual_size_expr;
    std::vector<af::Expression> dst_col_size_exprs;
    std::vector<af::Expression> dst_col_actual_size_exprs;
    std::vector<int64_t> src_col_sizes;
    std::vector<af::Expression> src_offsets;
    std::vector<uint32_t> src_strides;
    std::vector<uint32_t> src_buffer_offsets;
    // std::vector<bool> is_padded;
    // std::vector<uint32_t> second_last_dim_strides;
    // std::vector<uint32_t> gather_mask_dim_sizes;
    af::Expression total_rows_expr;
    uint32_t data_type_size = 0;
  };
  Status ParseAttr(const ascir::NodeView &node) override;

 private:
  static Status ParseSplitDim(const Tensor &x, const Tensor &y0, size_t &split_dim);
  static Status InitializeTiling(size_t split_dim, const vector<std::reference_wrapper<const Tensor>> &outputs,
                                 const Tensor &x, SplitTiling &tiling);
  static bool IsAllAligned(SplitTiling &tiling);
  static std::string GetSemanticFeature(const SplitTiling &tiling);
  static void GenSplitTilingForAllAligned(SplitTiling &tiling, const Tiler &tiler, std::stringstream &ss);
  static void GenSrcTensors(const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                            const std::string &dtype_name, std::stringstream &ss);
  static af::Status GenerateForAllAligned(const vector<std::reference_wrapper<const Tensor>> &outputs, const Tensor &x,
                                          SplitTiling &tiling, const Tiler &tiler, std::stringstream &ss);
  static af::Status GenerateDefault(const vector<std::reference_wrapper<const Tensor>> &outputs, const Tensor &x,
                                    SplitTiling &tiling, const TPipe &t_pipe, std::stringstream &ss,
                                    const int64_t tmp_buf_id);
  static void DefineSplitTiling(SplitTiling &tiling, const Tiler &tiler, std::stringstream &ss);
  static void GenDstAddrs(const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                          const std::string &dtype_name, std::stringstream &ss);
  static bool NeedB8ToB16(SplitTiling &tiling);
  ascir::NodeView node_ = nullptr;
};
}  // namespace codegen

#endif  // ASCGEN_DEV_CODEGEN_REG_API_CALL_SPLIT_REG_API_CALL_H_
