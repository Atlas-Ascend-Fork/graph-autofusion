/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "remove_pad_api_call.h"

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"
#include "codegen/expression_convert_struct.h"

namespace codegen {
using namespace std;
using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;

static void GetOneAxisSize(const TPipe &tpipe, const Tensor &tensor, const uint32_t idx, std::stringstream &ss) {
  auto axis_pos = tensor.vectorized_axis_pos[idx];
  ascir::AxisId axis_id = tensor.axis[axis_pos];
  if (tpipe.tiler.GetAxis(axis_id).type != ascir::Axis::Type::kAxisTypeTileInner ||
      tensor.vectorized_axis[0] == axis_id) {
    ss << tpipe.tiler.ActualSize(tensor.axis_size[axis_pos]);
    return;
  }
  ss << tpipe.tiler.Size(tensor.axis_size[axis_pos]);
}

static std::string GetLastAxisSize(const TPipe &tpipe, const Tensor &tensor, bool is_last_axis_align) {
  if (is_last_axis_align) {
    const size_t vectorized_axis_size = tensor.vectorized_axis.size();
    return tpipe.tiler.Size(tensor.vectorized_strides[vectorized_axis_size - 1]);
  }
  auto axis_pos = tensor.vectorized_axis_pos[tensor.vectorized_axis_pos.size() - 1];
  ascir::AxisId axis_id = tensor.vectorized_axis[tensor.vectorized_axis.size() - 1];
  if (tpipe.tiler.GetAxis(axis_id).type != ascir::Axis::Type::kAxisTypeTileInner ||
      tensor.vectorized_axis[0] == axis_id) {
    return tpipe.tiler.ActualSize(tensor.axis_size[axis_pos]);
  }
  return tpipe.tiler.Size(tensor.axis_size[axis_pos]);
}

static std::string GetNonLastAxisMergeSize(const TPipe &tpipe, const Tensor &tensor, bool is_last_axis_align) {
  if (tensor.vectorized_axis.size() == 1UL && !is_last_axis_align) {
    return "1";
  }

  std::stringstream ss;
  const size_t non_last_axis_index =
      is_last_axis_align ? tensor.vectorized_axis.size() : tensor.vectorized_axis.size() - 1;
  for (size_t i = 0; i < non_last_axis_index; ++i) {
    GetOneAxisSize(tpipe, tensor, i, ss);
    if (i != non_last_axis_index - 1) {
      ss << " * ";
    }
  }

  return ss.str();
}

Status RemovePadApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                  const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                  std::string &result) const {
  auto x = inputs[0].get();
  auto y = outputs[0].get();
  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs);
  stringstream ss;
  std::string dtype_name;
  Tensor::DtypeName(y.dtype, dtype_name);
  const auto tail_axis_stride = x.vectorized_strides.back();
  bool is_last_axis_align = tail_axis_stride != af::sym::kSymbolOne && tail_axis_stride != af::sym::kSymbolZero;
  std::string x_last_axis_size = GetLastAxisSize(tpipe, x, is_last_axis_align);
  std::string y_last_axis_size = GetLastAxisSize(tpipe, y, is_last_axis_align);
  std::string non_last_axis_size = GetNonLastAxisMergeSize(tpipe, x, is_last_axis_align);

  std::string src_row_pitch = KernelUtils::SizeAlign() + "(" + x_last_axis_size + ", 32 / sizeof(" + dtype_name + "))";
  const size_t axis_num = x.vectorized_axis.size();
  if (!is_last_axis_align && axis_num > 1UL && x.vectorized_strides.size() >= axis_num &&
      x.vectorized_strides[axis_num - 2UL] != af::sym::kSymbolZero) {
    src_row_pitch = tpipe.tiler.Size(x.vectorized_strides[axis_num - 2UL]);
  }

  ss << "RemovePadExtend(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x << "["
     << tpipe.tiler.TensorVectorizedOffset(current_axis, x) << "], " << y_last_axis_size << ", " << src_row_pitch
     << ", " << non_last_axis_size << ");" << std::endl;

  result = ss.str();
  return af::SUCCESS;
}

static ApiCallRegister<RemovePadApiCall> register_remove_pad_api_call("RemovePadApiCall");
}  // namespace codegen
