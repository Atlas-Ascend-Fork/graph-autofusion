/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "binary_api_call.h"

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "common/checker.h"
#include "api_call/utils/api_call_factory.h"
#include "api_call/utils/api_call_utils.h"
#include "codegen/expression_convert_struct.h"

namespace codegen {
using namespace std;
using namespace af::ops;
using namespace af::ascir_op;
using namespace ascgen_utils;

Status BinaryApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                               const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                               const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                               std::string &result) const {
  if (generalized_brc_inline_scene) {
    return BrcInlineGenerate(tpipe, current_axis, inputs, outputs, result);
  }
  size_t x1_idx = 0;
  size_t x2_idx = 1;
  bool switch_scalar = false;
  // 对于input[0]为Data节点，input[1]为Scalar节点，scheduler无法调换顺序，需要codegen调换顺序
  // input[0] input[1] 都为Scalar输入的情况不进行调换
  if (inputs[0].get().IsAnyScalar() && !(inputs[1].get().IsAnyScalar())) {
    x1_idx = 1;
    x2_idx = 0;
    switch_scalar = true;
  }
  const auto &x1 = inputs[x1_idx].get();
  const auto &x2 = inputs[x2_idx].get();

  GELOGD("x2, is_constant:%d, is_ub_scalar:%d, need_gen_get_value_of_ub_scalar:%d",
         static_cast<int32_t>(x2.is_constant), static_cast<int32_t>(x2.is_ub_scalar),
         static_cast<int32_t>(x2.need_gen_get_value_of_ub_scalar));

  const auto &y = outputs[0].get();

  (void)RegisterBasicDumpParam(this->api_name_, inputs, outputs, CombinedExprFactory::SymbolVar(x1.actual_size.Str()));

  stringstream ss;
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(x1.dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(x1.dtype));
  const std::string is_scalar_latter = switch_scalar ? "false" : "true";
  // 获取tmp_buf复用TBuf的id
  int64_t life_time_axis_id = -1L;
  int64_t id = -1L;
  auto it = this->tmp_buf_id.find(life_time_axis_id);
  if (it != this->tmp_buf_id.end()) {
    id = it->second;
  }

  if (x1.IsAnyScalar() && x2.IsAnyScalar()) {  // 两个输入都是Scalar
    ss << this->api_name_ << "s(";
    ss << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], ";
    ss << "(" << dtype_name << ")" << x1.GetScalarValue() << ", ";
    ss << "(" << dtype_name << ")" << x2.GetScalarValue();
    ss << ");" << std::endl;
  } else if (x1.IsAnyScalar() || x2.IsAnyScalar()) {  // 只有1个输入是Scalar
    if (this->api_name_ == "Div" || this->api_name_ == "Sub") {
      GE_ASSERT_TRUE(id != -1L, "BinaryApiCall cannot find tmp buffer id to use.");
      ss << this->api_name_ << "s<" << dtype_name << ", " << is_scalar_latter << ">" << "(" << y << "["
         << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x1 << "["
         << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], "
         << "(" << dtype_name << ")" << x2.GetScalarValue() << ", " << tpipe.tmp_buf << "_" << std::to_string(id)
         << ", " << x1.actual_size << ");" << std::endl;
    } else if (this->api_name_ == "DivExtend" || this->api_name_ == "SubExtend") {
      ss << this->api_name_ << "s<" << dtype_name << ", " << is_scalar_latter << ">" << "(" << y << "["
         << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x1 << "["
         << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], "
         << "(" << dtype_name << ")" << x2.GetScalarValue() << ", " << x1.actual_size << ");" << std::endl;
    } else {
      ss << this->api_name_ << "s(";
      ss << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], ";
      ss << x1 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], ";
      ss << "(" << dtype_name << ")" << x2.GetScalarValue() << ", ";
      ss << x1.actual_size;
      ss << ");" << std::endl;
    }
  } else {  // 两个输入都不是Scalar
    if (IsCVFusionStage(this->api_call_context) || tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse) {
      // CV UBFuse 下 flat 1D 调用的 count 取自 stage 尺寸, 无法覆盖按输出 dtype 对齐的二维布局
      // (如 bool 输入 N=4: 行距32, 总量 M*32, stage 仅 M*8), 尾部行读到未初始化数据。
      // 按 per-dtype 物理行距逐行调用, 与 UnaryApiCall/位宽类 API 的 CV 修复方式一致。
      const std::string output_stride = GenBlockAlignNExpr(y, "curAivN");
      const auto gen_input_stride = [&](const Tensor &t) {
        return (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse && t.id == tpipe.cube_output_tensor_id)
                   ? std::string("curAlignN")
                   : GenBlockAlignNExpr(t, "curAivN");
      };
      ss << y.actual_size << " = curAivM * " << output_stride << ";" << std::endl;
      ss << "for (uint32_t cv_row = 0; cv_row < ConvertToUint32(curAivM); cv_row++) {" << std::endl;
      ss << "  " << this->api_name_ << "(" << y << "[cv_row * ConvertToUint32(" << output_stride << ")], " << x1
         << "[cv_row * ConvertToUint32(" << gen_input_stride(x1) << ")], " << x2 << "[cv_row * ConvertToUint32("
         << gen_input_stride(x2) << ")], ConvertToUint32(curAivN));" << std::endl;
      ss << "}" << std::endl;
      result = ss.str();
      return af::SUCCESS;
    }
    ss << this->api_name_ << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x1
       << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], " << x2 << "["
       << tpipe.tiler.TensorVectorizedOffset(current_axis, x2) << "], " << x1.actual_size << ");" << std::endl;
  }
  result = ss.str();
  return af::SUCCESS;
}

std::string BinaryApiCall::GetAscendApiName(const std::string &api_name) {
  const std::string prefix = "AscendC::";
  return api_name.find(prefix) != std::string::npos ? api_name : prefix + api_name;
}

Status BinaryApiCall::BrcInlineGenerate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                        const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                        const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                        std::string &result) const {
  const auto &x1 = inputs[0].get();
  const auto &x2 = inputs[1].get();

  const auto &y = outputs[0].get();

  std::vector<af::Expression> i0_v_repeates;
  std::vector<af::Expression> i1_v_repeates;

  for (size_t i = 0UL; i < x1.vectorized_axis.size(); ++i) {
    auto pos1 = x1.vectorized_axis_pos[i];
    auto pos2 = x2.vectorized_axis_pos[i];
    i0_v_repeates.push_back(x1.axis_size[pos1]);
    i1_v_repeates.push_back(x2.axis_size[pos2]);
  }

  GELOGD("input0_v_stride: %s, input1_v_stride: %s", VectorToStr(x1.vectorized_strides).c_str(),
         VectorToStr(x2.vectorized_strides).c_str());

  std::vector<af::Expression> i0_meger_repeates;
  std::vector<af::Expression> i1_meger_repeates;

  if (input_idx_2_brc_inline[0] != 0) {
    MergeBrcAxisRepeats(i0_v_repeates, i1_v_repeates, x2.vectorized_strides, i0_meger_repeates, i1_meger_repeates);
  } else {
    MergeBrcAxisRepeats(i1_v_repeates, i0_v_repeates, x1.vectorized_strides, i1_meger_repeates, i0_meger_repeates);
  }

  auto &meger_shape = (input_idx_2_brc_inline[0] != 0) ? i1_meger_repeates : i0_meger_repeates;

  std::string shape =
      "{" + tpipe.tiler.ActualSize(meger_shape[0]) + ", " + af::SymbolicUtils::ToString(meger_shape[1]) + "}";

  af::Expression v_strides = af::sym::kSymbolZero;
  auto &x_in = (input_idx_2_brc_inline[0] != 0) ? x1 : x2;
  for (size_t i = 0UL; i < x_in.vectorized_axis_pos.size(); ++i) {
    const uint32_t axis_ids = x_in.vectorized_axis_pos[i];
    af::Expression cur_axis_strides = y.vectorized_strides[i];
    af::Expression cur_axis_repeats = x_in.axis_size[axis_ids];
    if (af::SymbolicUtils::StaticCheckEq(cur_axis_repeats, af::sym::kSymbolOne) != af::TriBool::kTrue) {
      break;
    }
    v_strides = ((af::SymbolicUtils::StaticCheckEq(cur_axis_strides, af::sym::kSymbolZero) != af::TriBool::kTrue)
                     ? cur_axis_strides
                     : v_strides);
  }

  int64_t type_size = GetSizeByDataType(y.dtype);
  stringstream ss;
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(x1.dtype, dtype_name), "Codegen get data type:%d failed",
                    static_cast<int32_t>(x1.dtype));

  ss << "BinaryBrcInlineApiWithTwoVectorizedAxis<" << dtype_name << ">" << "(" << y << "["
     << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], " << x1 << "["
     << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], " << x2 << "["
     << tpipe.tiler.TensorVectorizedOffset(current_axis, x2) << "], " << tpipe.tiler.ActualSize(meger_shape[0]) << ", "
     << tpipe.tiler.ActualSize(meger_shape[1]) << ", " << static_cast<int>(input_idx_2_brc_inline[0] == 0) << ", "
     << static_cast<int>(input_idx_2_brc_inline[1] == 0) << ", " << tpipe.tiler.ActualSize(v_strides) << ", "
     << static_cast<int>(type_size) << ", &" << GetAscendApiName(this->api_name_) << ", &"
     << GetAscendApiName(this->api_name_) << ");" << std::endl;
  result = ss.str();
  return af::SUCCESS;
}

Status BinaryApiCall::Init(const ascir::NodeView &node) {
  GE_CHK_STATUS_RET(ApiCall::Init(node));
  generalized_brc_inline_scene = IsGeneralizeBrcInlineScene(node, input_idx_2_brc_inline);
  return af::SUCCESS;
}

static ApiCallRegister<BinaryApiCall> register_binary_api_call("BinaryApiCall");
}  // namespace codegen
